import argparse
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers


def _load_config(config_path: str, task_name: str | None):
    cfg = OmegaConf.load(config_path)
    if task_name is not None:
        task_cfg = OmegaConf.load(Path("configs") / "task" / f"{task_name}.yaml")
        cfg = OmegaConf.merge(cfg, task_cfg)
    return cfg


def _latent_features(inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    latents = inputs["input_latents"]
    first = inputs["first_frame_latents"]
    if first is None:
        first = latents[:, :, 0:1]
    if latents.shape[2] <= 1:
        raise ValueError(f"Need at least one future latent step, got latent shape {tuple(latents.shape)}")

    state_feat = first.float().mean(dim=(2, 3, 4))
    future_feat = latents[:, :, 1:].float().mean(dim=(3, 4)).flatten(1)
    return state_feat, future_feat


def main():
    parser = argparse.ArgumentParser(description="Precompute pooled FastWAM VAE latent features for CRF probes.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--task", default="libero_uncond_2cam224_1e-4")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-path", default="runs/crf_features_libero.pt")
    parser.add_argument("--shard-dir", default=None, help="Optional directory for intermediate feature shards.")
    parser.add_argument("--save-every", type=int, default=0, help="Save one shard every N rows when shard-dir is set.")
    args = parser.parse_args()

    register_default_resolvers()
    cfg = _load_config(args.config, args.task)
    misc.register_work_dir(cfg.output_dir)

    dataset = instantiate(cfg.data.train)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    mixed_precision = _normalize_mixed_precision(str(cfg.mixed_precision))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    model.eval()
    model.requires_grad_(False)

    state_chunks = []
    action_chunks = []
    future_chunks = []
    idx_chunks = []
    prompt_chunks = []
    seen = 0
    shard_id = 0

    shard_dir = Path(args.shard_dir) if args.shard_dir else None
    if shard_dir is not None:
        shard_dir.mkdir(parents=True, exist_ok=True)

    def flush_shard(force: bool = False):
        nonlocal shard_id, state_chunks, action_chunks, future_chunks, idx_chunks, prompt_chunks
        if shard_dir is None or not state_chunks:
            return
        rows = sum(chunk.shape[0] for chunk in state_chunks)
        if not force and (args.save_every <= 0 or rows < args.save_every):
            return
        payload = {
            "state_feat": torch.cat(state_chunks, dim=0),
            "action_feat": torch.cat(action_chunks, dim=0),
            "future_feat": torch.cat(future_chunks, dim=0),
            "task": args.task,
            "config": args.config,
            "shard_id": shard_id,
        }
        if idx_chunks:
            payload["idx"] = torch.cat(idx_chunks, dim=0)
        if prompt_chunks:
            payload["prompt"] = prompt_chunks
        shard_path = shard_dir / f"shard_{shard_id:05d}.pt"
        torch.save(payload, shard_path)
        print(f"Saved shard {shard_id} with {payload['state_feat'].shape[0]} rows to {shard_path}")
        shard_id += 1
        state_chunks = []
        action_chunks = []
        future_chunks = []
        idx_chunks = []
        prompt_chunks = []

    progress_total = min(len(dataset), args.max_samples) if args.max_samples > 0 else len(dataset)
    pbar = tqdm(total=progress_total, desc="Precomputing CRF features")
    for batch in loader:
        with torch.no_grad():
            inputs = model.build_inputs(batch, tiled=False)
            state_feat, future_feat = _latent_features(inputs)
            action_feat = inputs["action"].float().flatten(1)

        remaining = progress_total - seen
        if remaining <= 0:
            break
        take = min(state_feat.shape[0], remaining)
        state_chunks.append(state_feat[:take].cpu())
        action_chunks.append(action_feat[:take].cpu())
        future_chunks.append(future_feat[:take].cpu())

        if "idx" in batch and isinstance(batch["idx"], torch.Tensor):
            idx_chunks.append(batch["idx"][:take].cpu())
        if "prompt" in batch:
            prompt_chunks.extend(list(batch["prompt"][:take]))

        seen += take
        pbar.update(take)
        flush_shard(force=False)
        if seen >= progress_total:
            break
    pbar.close()
    flush_shard(force=True)

    if shard_dir is not None:
        shard_paths = sorted(shard_dir.glob("shard_*.pt"))
        if not shard_paths:
            raise RuntimeError(f"No shards were written to {shard_dir}")
        shards = [torch.load(path, map_location="cpu") for path in shard_paths]
        payload = {
            "state_feat": torch.cat([s["state_feat"] for s in shards], dim=0),
            "action_feat": torch.cat([s["action_feat"] for s in shards], dim=0),
            "future_feat": torch.cat([s["future_feat"] for s in shards], dim=0),
            "task": args.task,
            "config": args.config,
            "shard_dir": str(shard_dir),
        }
        if all("idx" in s for s in shards):
            payload["idx"] = torch.cat([s["idx"] for s in shards], dim=0)
        if all("prompt" in s for s in shards):
            payload["prompt"] = [prompt for s in shards for prompt in s["prompt"]]
    else:
        payload = {
            "state_feat": torch.cat(state_chunks, dim=0),
            "action_feat": torch.cat(action_chunks, dim=0),
            "future_feat": torch.cat(future_chunks, dim=0),
            "task": args.task,
            "config": args.config,
        }
        if idx_chunks:
            payload["idx"] = torch.cat(idx_chunks, dim=0)
        if prompt_chunks:
            payload["prompt"] = prompt_chunks

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, save_path)
    print(
        f"Saved {payload['state_feat'].shape[0]} CRF feature rows to {save_path}\n"
        f"  state_feat={tuple(payload['state_feat'].shape)}\n"
        f"  action_feat={tuple(payload['action_feat'].shape)}\n"
        f"  future_feat={tuple(payload['future_feat'].shape)}"
    )


if __name__ == "__main__":
    main()
