import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers


class LowRankTokenResponseHead(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, future_dim: int, hidden_dim: int = 512, rank: int = 8):
        super().__init__()
        self.action_dim = int(action_dim)
        self.future_dim = int(future_dim)
        self.rank = int(rank)
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.u_head = nn.Linear(hidden_dim, future_dim * rank)
        self.v_head = nn.Linear(hidden_dim, action_dim * rank)

    def forward(self, state_feat, action_feat, delta_action):
        h = self.trunk(torch.cat([state_feat, action_feat], dim=-1))
        u = self.u_head(h).view(-1, self.future_dim, self.rank)
        v = self.v_head(h).view(-1, self.action_dim, self.rank)
        coeff = torch.einsum("ba,bar->br", delta_action, v)
        return torch.einsum("bfr,br->bf", u, coeff)


def _load_config(config_path: str, task_name: str | None):
    cfg = OmegaConf.load(config_path)
    if task_name is not None:
        task_cfg = OmegaConf.load(Path("configs") / "task" / f"{task_name}.yaml")
        cfg = OmegaConf.merge(cfg, task_cfg)
    return cfg


def _nearest_pairs(state_feat):
    state_float = F.normalize(state_feat.float(), dim=-1)
    dist = torch.cdist(state_float, state_float, p=2)
    dist.fill_diagonal_(float("inf"))
    pair_idx = dist.argmin(dim=1)
    anchor_idx = torch.arange(state_feat.shape[0], device=state_feat.device)
    return anchor_idx, pair_idx


def _make_response_tensors(batch, device):
    state_tokens, action_feat, future_tokens = [x.to(device, dtype=torch.float32) for x in batch]
    state_global = state_tokens.mean(dim=1)
    anchor_idx, pair_idx = _nearest_pairs(state_global)
    state_anchor = state_global[anchor_idx]
    action_anchor = action_feat[anchor_idx]
    delta_action = action_feat[pair_idx] - action_feat[anchor_idx]
    target_delta_future = future_tokens[pair_idx] - future_tokens[anchor_idx]
    return state_anchor, action_anchor, delta_action, target_delta_future


def _split_features(payload, train_ratio, seed):
    state = payload["state_tokens"].float()
    action = payload["action_feat"].float()
    future = payload["future_tokens"].float()
    n = state.shape[0]
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    train_n = max(1, min(n - 1, int(n * train_ratio)))
    return (
        TensorDataset(state[perm[:train_n]], action[perm[:train_n]], future[perm[:train_n]]),
        TensorDataset(state[perm[train_n:]], action[perm[train_n:]], future[perm[train_n:]]),
        perm[train_n:],
    )


def _train_head(head, train_loader, device, steps, lr):
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    data_iter = iter(train_loader)
    head.train()
    for step in range(1, steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        state, action, delta_action, target = _make_response_tensors(batch, device)
        pred = head(state, action, delta_action).reshape(target.shape)
        loss = F.mse_loss(pred.float(), target.float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 500 == 0:
            cos = F.cosine_similarity(pred.flatten(1).float(), target.flatten(1).float(), dim=-1).mean()
            print(f"train_step={step:05d} loss={loss.item():.6f} cos={cos.item():.4f}")


def _normalize(x):
    x = x.float()
    x = x - x.min()
    return x / x.max().clamp(min=1e-8)


def _heatmap_to_image(energy, height, width):
    heat = _normalize(energy)[None, None]
    up = F.interpolate(heat, size=(height, width), mode="bilinear", align_corners=False)
    return up[0, 0].cpu().numpy()


def _first_frame_rgb(sample):
    video = sample["video"]
    if video.ndim != 4:
        raise ValueError(f"Expected sample video [C,T,H,W], got {tuple(video.shape)}")
    frame = video[:, 0].float()
    frame = ((frame + 1.0) * 0.5).clamp(0, 1)
    return frame.permute(1, 2, 0).cpu().numpy()


def _save_overlay(rgb, gt_map, pred_map, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
    panels = [("Input frame", None), ("GT response overlay", gt_map), ("Pred response overlay", pred_map)]
    for ax, (title, heat) in zip(axes[0], panels):
        ax.imshow(rgb)
        if heat is not None:
            ax.imshow(heat, cmap="magma", alpha=0.55, interpolation="bilinear")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Overlay token-level CRF response heatmaps on RGB input frames.")
    parser.add_argument("--features", default="runs/crf_token_features_libero_5k.pt")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--task", default="libero_uncond_2cam224_1e-4")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--grid-h", type=int, default=7)
    parser.add_argument("--grid-w", type=int, default=14)
    parser.add_argument("--time-index", type=int, default=0, help="Future latent step to visualize, 0 for t+1.")
    parser.add_argument("--out-dir", default="runs/crf_rgb_overlay")
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.features, map_location="cpu")
    train_ds, eval_ds, eval_indices = _split_features(payload, args.train_ratio, args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    state_dim = payload["state_tokens"].shape[-1]
    action_dim = payload["action_feat"].shape[-1]
    future_dim = payload["future_tokens"].shape[1] * payload["future_tokens"].shape[2]
    head = LowRankTokenResponseHead(state_dim, action_dim, future_dim, hidden_dim=args.hidden_dim, rank=args.rank).to(device)

    print(f"Training token CRF for RGB overlay: features={args.features}")
    _train_head(head, train_loader, device=device, steps=args.train_steps, lr=args.lr)

    register_default_resolvers()
    cfg = _load_config(args.config, args.task)
    misc.register_work_dir(cfg.output_dir)
    raw_dataset = instantiate(cfg.data.train)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    head.eval()

    saved = 0
    offset = 0
    with torch.no_grad():
        for batch in eval_loader:
            state, action, delta_action, target = _make_response_tensors(batch, device)
            pred = head(state, action, delta_action).reshape(target.shape)
            bsz = pred.shape[0]
            for i in range(bsz):
                token_count_per_t = args.grid_h * args.grid_w
                start = args.time_index * token_count_per_t
                end = start + token_count_per_t
                gt_energy = target[i, start:end].pow(2).sum(dim=-1).reshape(args.grid_h, args.grid_w)
                pred_energy = pred[i, start:end].pow(2).sum(dim=-1).reshape(args.grid_h, args.grid_w)

                raw_idx = int(eval_indices[offset + i].item())
                sample = raw_dataset[raw_idx]
                rgb = _first_frame_rgb(sample)
                height, width = rgb.shape[:2]
                gt_map = _heatmap_to_image(gt_energy, height=height, width=width)
                pred_map = _heatmap_to_image(pred_energy, height=height, width=width)
                _save_overlay(rgb, gt_map, pred_map, out_dir / f"overlay_{saved:03d}.png")

                saved += 1
                if saved >= args.num_samples:
                    print(f"Saved {saved} RGB overlays to {out_dir}")
                    return
            offset += bsz
    print(f"Saved {saved} RGB overlays to {out_dir}")


if __name__ == "__main__":
    main()
