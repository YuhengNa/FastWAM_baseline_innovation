import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


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
    target_flat = target_delta_future.flatten(1)
    return state_anchor, action_anchor, delta_action, target_flat, target_delta_future


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
        state, action, delta_action, target_flat, _ = _make_response_tensors(batch, device)
        pred = head(state, action, delta_action)
        loss = F.mse_loss(pred.float(), target_flat.float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 500 == 0:
            cos = F.cosine_similarity(pred.float(), target_flat.float(), dim=-1).mean()
            print(f"train_step={step:05d} loss={loss.item():.6f} cos={cos.item():.4f}")


def _normalize_map(x):
    x = x.float()
    x = x - x.min()
    denom = x.max().clamp(min=1e-8)
    return x / denom


def _save_heatmap_grid(pred_tokens, target_tokens, out_path: Path, grid_h: int, grid_w: int):
    pred_energy = pred_tokens.float().pow(2).sum(dim=-1)
    target_energy = target_tokens.float().pow(2).sum(dim=-1)
    if pred_energy.shape[0] % (grid_h * grid_w) != 0:
        raise ValueError(f"Token count {pred_energy.shape[0]} is not divisible by grid {grid_h}x{grid_w}")
    time_steps = pred_energy.shape[0] // (grid_h * grid_w)
    pred_maps = pred_energy.reshape(time_steps, grid_h, grid_w)
    target_maps = target_energy.reshape(time_steps, grid_h, grid_w)

    fig, axes = plt.subplots(time_steps, 3, figsize=(9, 3 * time_steps), squeeze=False)
    for t in range(time_steps):
        gt = _normalize_map(target_maps[t]).cpu().numpy()
        pred = _normalize_map(pred_maps[t]).cpu().numpy()
        diff = abs(gt - pred)
        panels = [
            ("GT response", gt),
            ("Pred response", pred),
            ("Abs diff", diff),
        ]
        for col, (title, image) in enumerate(panels):
            ax = axes[t][col]
            ax.imshow(image, cmap="magma", interpolation="nearest")
            ax.set_title(f"t+{t + 1} {title}")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize token-level CRF response heatmaps.")
    parser.add_argument("--features", default="runs/crf_token_features_libero_5k.pt")
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
    parser.add_argument("--out-dir", default="runs/crf_token_vis")
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.features, map_location="cpu")
    train_ds, eval_ds = _split_features(payload, args.train_ratio, args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    state_dim = payload["state_tokens"].shape[-1]
    action_dim = payload["action_feat"].shape[-1]
    future_dim = payload["future_tokens"].shape[1] * payload["future_tokens"].shape[2]
    head = LowRankTokenResponseHead(
        state_dim=state_dim,
        action_dim=action_dim,
        future_dim=future_dim,
        hidden_dim=args.hidden_dim,
        rank=args.rank,
    ).to(device)

    print(
        f"features={args.features} train={len(train_ds)} eval={len(eval_ds)} "
        f"future_tokens={tuple(payload['future_tokens'].shape[1:])}"
    )
    _train_head(head, train_loader, device=device, steps=args.train_steps, lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    head.eval()
    saved = 0
    with torch.no_grad():
        for batch in eval_loader:
            state, action, delta_action, target_flat, target_tokens = _make_response_tensors(batch, device)
            pred_flat = head(state, action, delta_action)
            pred_tokens = pred_flat.reshape(target_tokens.shape)
            for i in range(pred_tokens.shape[0]):
                out_path = out_dir / f"sample_{saved:03d}.png"
                _save_heatmap_grid(
                    pred_tokens=pred_tokens[i],
                    target_tokens=target_tokens[i],
                    out_path=out_path,
                    grid_h=args.grid_h,
                    grid_w=args.grid_w,
                )
                saved += 1
                if saved >= args.num_samples:
                    print(f"Saved {saved} visualizations to {out_dir}")
                    return
    print(f"Saved {saved} visualizations to {out_dir}")


if __name__ == "__main__":
    main()
