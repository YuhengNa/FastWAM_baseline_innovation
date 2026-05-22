import argparse
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers


def _describe_value(name, value):
    if isinstance(value, torch.Tensor):
        pad_info = ""
        if value.dtype == torch.bool:
            pad_info = f", true={int(value.sum().item())}"
        print(f"{name}: tensor shape={tuple(value.shape)} dtype={value.dtype}{pad_info}")
        return
    print(f"{name}: {type(value).__name__} = {value}")


def _describe_batch(batch):
    print("Batch keys:")
    for key in sorted(batch.keys()):
        _describe_value(f"  {key}", batch[key])

    video = batch.get("video")
    action = batch.get("action")
    proprio = batch.get("proprio")
    if isinstance(video, torch.Tensor):
        print(f"\nvideo convention: [B, C, T_video, H, W] = {tuple(video.shape)}")
    if isinstance(action, torch.Tensor):
        print(f"action convention: [B, T_action, D_action] = {tuple(action.shape)}")
    if isinstance(proprio, torch.Tensor):
        print(f"proprio convention: [B, T_proprio, D_proprio] = {tuple(proprio.shape)}")
    if isinstance(video, torch.Tensor) and isinstance(action, torch.Tensor):
        transitions = max(int(video.shape[2]) - 1, 1)
        print(f"actions per video transition: {int(action.shape[1])} / {transitions} = {action.shape[1] / transitions:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect FastWAM batches before adding Action Response Field experiments."
    )
    parser.add_argument("--config", default="configs/train.yaml", help="Path to the Hydra train config.")
    parser.add_argument("--task", default=None, help="Optional task config override, e.g. libero_uncond_2cam224_1e-4.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--encode-latents", action="store_true", help="Also instantiate the model and print VAE latent shape.")
    parser.add_argument("--device", default=None, help="Device for optional latent encoding. Defaults to cuda if available.")
    args = parser.parse_args()

    register_default_resolvers()
    cfg = OmegaConf.load(args.config)
    if args.task is not None:
        task_cfg = OmegaConf.load(Path("configs") / "task" / f"{args.task}.yaml")
        cfg = OmegaConf.merge(cfg, task_cfg)

    misc.register_work_dir(cfg.output_dir)
    dataset = instantiate(cfg.data.train)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    batch = next(iter(loader))
    _describe_batch(batch)

    if not args.encode_latents:
        return

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    mixed_precision = _normalize_mixed_precision(str(cfg.mixed_precision))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    model.eval()
    with torch.no_grad():
        inputs = model.build_inputs(batch, tiled=False)
    latents = inputs["input_latents"]
    print(f"\nVAE latent shape: {tuple(latents.shape)} dtype={latents.dtype}")
    if inputs["first_frame_latents"] is not None:
        print(f"first-frame latent shape: {tuple(inputs['first_frame_latents'].shape)}")


if __name__ == "__main__":
    main()
