import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class LowRankActionResponseHead(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, future_dim: int, hidden_dim: int = 512, rank: int = 16):
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

    def forward(self, state_feat: torch.Tensor, action_feat: torch.Tensor, delta_action: torch.Tensor) -> torch.Tensor:
        base = torch.cat([state_feat, action_feat], dim=-1)
        h = self.trunk(base)
        u = self.u_head(h).view(-1, self.future_dim, self.rank)
        v = self.v_head(h).view(-1, self.action_dim, self.rank)
        coeff = torch.einsum("ba,bar->br", delta_action, v)
        return torch.einsum("bfr,br->bf", u, coeff)


def _nearest_pairs(state_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state_float = F.normalize(state_feat.float(), dim=-1)
    dist = torch.cdist(state_float, state_float, p=2)
    dist.fill_diagonal_(float("inf"))
    nn_idx = dist.argmin(dim=1)
    anchor_idx = torch.arange(state_feat.shape[0], device=state_feat.device)
    pair_dist = dist[anchor_idx, nn_idx]
    return anchor_idx, nn_idx, pair_dist


def _response_batch(batch, device: str):
    state_feat, action_feat, future_feat = [x.to(device, dtype=torch.float32) for x in batch]
    anchor_idx, nn_idx, pair_dist = _nearest_pairs(state_feat)
    return {
        "state": state_feat[anchor_idx],
        "action": action_feat[anchor_idx],
        "delta_action": action_feat[nn_idx] - action_feat[anchor_idx],
        "target_delta_future": future_feat[nn_idx] - future_feat[anchor_idx],
        "pair_dist": pair_dist,
    }


def _metrics(pred_delta_future: torch.Tensor, target_delta_future: torch.Tensor) -> dict[str, float]:
    pred = pred_delta_future.float()
    target = target_delta_future.float()
    mse = F.mse_loss(pred, target).detach()
    zero_mse = target.pow(2).mean().detach().clamp(min=1e-12)
    cosine = F.cosine_similarity(pred, target, dim=-1).mean().detach()
    return {
        "mse": float(mse.item()),
        "rel_mse": float((mse / zero_mse).item()),
        "cos": float(cosine.item()),
        "pred_norm": float(pred.norm(dim=-1).mean().detach().item()),
        "target_norm": float(target.norm(dim=-1).mean().detach().item()),
    }


@torch.no_grad()
def _evaluate(head, loader, device: str, num_batches: int):
    head.eval()
    metrics = []
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        rb = _response_batch(batch, device=device)
        pred = head(rb["state"], rb["action"], rb["delta_action"])
        metric = _metrics(pred, rb["target_delta_future"])
        metric["pair_dist"] = float(rb["pair_dist"].mean().item())
        metrics.append(metric)
    head.train()
    return {key: sum(m[key] for m in metrics) / len(metrics) for key in metrics[0]}


def _split_features(payload: dict, train_ratio: float, seed: int):
    state = payload["state_feat"].float()
    action = payload["action_feat"].float()
    future = payload["future_feat"].float()
    n = state.shape[0]
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    train_n = max(1, min(n - 1, int(n * train_ratio)))
    train_idx = perm[:train_n]
    eval_idx = perm[train_n:]
    return (
        TensorDataset(state[train_idx], action[train_idx], future[train_idx]),
        TensorDataset(state[eval_idx], action[eval_idx], future[eval_idx]),
    )


def main():
    parser = argparse.ArgumentParser(description="Train a CRF probe from cached FastWAM latent features.")
    parser.add_argument("--features", default="runs/crf_features_libero.pt")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--save-path", default="runs/crf_probe_cached.pt")
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.features, map_location="cpu")
    train_ds, eval_ds = _split_features(payload, train_ratio=args.train_ratio, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    state_dim = payload["state_feat"].shape[-1]
    action_dim = payload["action_feat"].shape[-1]
    future_dim = payload["future_feat"].shape[-1]
    head = LowRankActionResponseHead(
        state_dim=state_dim,
        action_dim=action_dim,
        future_dim=future_dim,
        hidden_dim=args.hidden_dim,
        rank=args.rank,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    print(
        f"Loaded features: n={payload['state_feat'].shape[0]} train={len(train_ds)} eval={len(eval_ds)} "
        f"state_dim={state_dim} action_dim={action_dim} future_dim={future_dim}"
    )

    data_iter = iter(train_loader)
    running = []
    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        rb = _response_batch(batch, device=device)
        pred = head(rb["state"], rb["action"], rb["delta_action"])
        loss = F.mse_loss(pred.float(), rb["target_delta_future"].float())
        metric = _metrics(pred, rb["target_delta_future"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running.append((metric["mse"], metric["rel_mse"], metric["cos"], metric["pred_norm"], metric["target_norm"], float(rb["pair_dist"].mean().item())))
        if step == 1 or step % args.log_every == 0:
            denom = min(len(running), args.log_every)
            recent = running[-args.log_every:]
            print(
                f"step={step:05d} train_mse={sum(x[0] for x in recent)/denom:.6f} "
                f"train_rel_mse={sum(x[1] for x in recent)/denom:.4f} "
                f"train_cos={sum(x[2] for x in recent)/denom:.4f} "
                f"pred_norm={sum(x[3] for x in recent)/denom:.4f} "
                f"target_norm={sum(x[4] for x in recent)/denom:.4f} "
                f"pair_dist={sum(x[5] for x in recent)/denom:.4f}"
            )
        if step == 1 or (args.eval_every > 0 and step % args.eval_every == 0):
            eval_metric = _evaluate(head, eval_loader, device=device, num_batches=args.eval_batches)
            print(
                f"eval@{step:05d} mse={eval_metric['mse']:.6f} rel_mse={eval_metric['rel_mse']:.4f} "
                f"cos={eval_metric['cos']:.4f} pred_norm={eval_metric['pred_norm']:.4f} "
                f"target_norm={eval_metric['target_norm']:.4f} pair_dist={eval_metric['pair_dist']:.4f}"
            )

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head": head.state_dict(),
            "args": vars(args),
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
            "future_dim": int(future_dim),
            "rank": int(args.rank),
        },
        save_path,
    )
    print(f"Saved cached CRF probe checkpoint to {save_path}")


if __name__ == "__main__":
    main()
