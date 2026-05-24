import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
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


def _pool_spatial(x: torch.Tensor, pool: int) -> torch.Tensor:
    if pool <= 1:
        return x
    shape = x.shape
    flat = x.reshape(-1, shape[-3], shape[-2], shape[-1])
    pooled = F.avg_pool2d(flat.float(), kernel_size=pool, stride=pool)
    return pooled.reshape(*shape[:-2], pooled.shape[-2], pooled.shape[-1])


def _token_features(inputs: dict, spatial_pool: int):
    latents = inputs["input_latents"]
    first = inputs["first_frame_latents"]
    if first is None:
        first = latents[:, :, 0:1]
    if latents.shape[2] <= 1:
        raise ValueError(f"Need future latent steps, got {tuple(latents.shape)}")

    first = _pool_spatial(first.float(), spatial_pool)
    future = _pool_spatial(latents[:, :, 1:].float(), spatial_pool)

    # state_tokens: [B, S, C]
    state_tokens = first[:, :, 0].permute(0, 2, 3, 1).contiguous()
    state_tokens = state_tokens.reshape(state_tokens.shape[0], -1, state_tokens.shape[-1])

    # future_tokens: [B, T_future * S, C]
    future_tokens = future.permute(0, 2, 3, 4, 1).contiguous()
    future_tokens = future_tokens.reshape(future_tokens.shape[0], -1, future_tokens.shape[-1])
    return state_tokens, future_tokens


def _episode_phase_from_indices(dataset, indices: torch.Tensor):
    base = getattr(dataset, "lerobot_dataset", None)
    if base is None or not hasattr(base, "episode_data_index"):
        return None, None
    starts = base.episode_data_index["from"].cpu().long()
    ends = base.episode_data_index["to"].cpu().long()
    idx = indices.cpu().long()
    episode = torch.bucketize(idx, ends, right=False)
    episode = episode.clamp(max=starts.numel() - 1)
    start = starts[episode]
    end = ends[episode]
    denom = (end - start - 1).clamp(min=1)
    phase = (idx - start).float() / denom.float()
    return episode.long(), phase.float()


def main():
    parser = argparse.ArgumentParser(description="Precompute spatial/token FastWAM VAE latent features for CRF.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--task", default="libero_uncond_2cam224_1e-4")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--spatial-pool", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-path", default="runs/crf_token_features_libero_1k.pt")
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
    episode_chunks = []
    phase_chunks = []
    prompt_rows = []
    seen = 0
    total = min(len(dataset), args.max_samples) if args.max_samples > 0 else len(dataset)

    pbar = tqdm(total=total, desc="Precomputing CRF token features")
    for batch in loader:
        with torch.no_grad():
            inputs = model.build_inputs(batch, tiled=False)
            state_tokens, future_tokens = _token_features(inputs, spatial_pool=args.spatial_pool)
            action_feat = inputs["action"].float().flatten(1)

        remaining = total - seen
        if remaining <= 0:
            break
        take = min(state_tokens.shape[0], remaining)
        state_chunks.append(state_tokens[:take].cpu())
        action_chunks.append(action_feat[:take].cpu())
        future_chunks.append(future_tokens[:take].cpu())
        if "idx" in batch:
            idx = batch["idx"]
            if not isinstance(idx, torch.Tensor):
                idx = torch.as_tensor(idx)
            idx = idx[:take].cpu().long()
            idx_chunks.append(idx)
            episode, phase = _episode_phase_from_indices(dataset, idx)
            if episode is not None:
                episode_chunks.append(episode)
                phase_chunks.append(phase)
        if "prompt" in batch:
            prompt_rows.extend(list(batch["prompt"])[:take])
        seen += take
        pbar.update(take)
        if seen >= total:
            break
    pbar.close()

    payload = {
        "state_tokens": torch.cat(state_chunks, dim=0),
        "action_feat": torch.cat(action_chunks, dim=0),
        "future_tokens": torch.cat(future_chunks, dim=0),
        "spatial_pool": int(args.spatial_pool),
        "task": args.task,
        "config": args.config,
    }
    if idx_chunks:
        payload["idx"] = torch.cat(idx_chunks, dim=0)
    if episode_chunks:
        payload["episode_index"] = torch.cat(episode_chunks, dim=0)
        payload["episode_phase"] = torch.cat(phase_chunks, dim=0)
    if prompt_rows:
        payload["prompt"] = prompt_rows
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, save_path)
    print(
        f"Saved {payload['state_tokens'].shape[0]} token rows to {save_path}\n"
        f"  state_tokens={tuple(payload['state_tokens'].shape)}\n"
        f"  action_feat={tuple(payload['action_feat'].shape)}\n"
        f"  future_tokens={tuple(payload['future_tokens'].shape)}"
    )
    if "prompt" in payload:
        print(f"  prompts={len(set(payload['prompt']))} unique")
    if "episode_phase" in payload:
        print(
            f"  episode_phase=({payload['episode_phase'].min().item():.4f}, "
            f"{payload['episode_phase'].max().item():.4f})"
        )


if __name__ == "__main__":
    main()
