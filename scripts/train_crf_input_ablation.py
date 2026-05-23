import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class MLPDeltaPredictor(nn.Module):
    def __init__(self, input_dim: int, future_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, future_dim),
        )

    def forward(self, x):
        return self.net(x)


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

    def forward(self, state_feat, action_feat, delta_action):
        base = torch.cat([state_feat, action_feat], dim=-1)
        h = self.trunk(base)
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
    state, action, future = [x.to(device, dtype=torch.float32) for x in batch]
    anchor_idx, pair_idx = _nearest_pairs(state)
    state_anchor = state[anchor_idx]
    action_anchor = action[anchor_idx]
    delta_action = action[pair_idx] - action[anchor_idx]
    target_delta_future = future[pair_idx] - future[anchor_idx]
    return state_anchor, action_anchor, delta_action, target_delta_future


def _make_input(mode, state, action, delta_action):
    if mode == "delta_only":
        return delta_action
    if mode == "state_delta":
        return torch.cat([state, delta_action], dim=-1)
    if mode == "action_delta":
        return torch.cat([action, delta_action], dim=-1)
    if mode == "full_mlp":
        return torch.cat([state, action, delta_action], dim=-1)
    raise ValueError(f"Unsupported MLP mode: {mode}")


def _metrics(pred, target):
    pred = pred.float()
    target = target.float()
    mse = F.mse_loss(pred, target).detach()
    zero_mse = target.pow(2).mean().detach().clamp(min=1e-12)
    cos = F.cosine_similarity(pred, target, dim=-1).mean().detach()
    return {
        "mse": float(mse.item()),
        "rel_mse": float((mse / zero_mse).item()),
        "cos": float(cos.item()),
    }


@torch.no_grad()
def _evaluate(model, mode, loader, device, eval_batches):
    model.eval()
    metrics = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= eval_batches:
            break
        state, action, delta_action, target = _make_response_tensors(batch, device)
        if mode == "lowrank":
            pred = model(state, action, delta_action)
        else:
            pred = model(_make_input(mode, state, action, delta_action))
        metrics.append(_metrics(pred, target))
    model.train()
    return {key: sum(m[key] for m in metrics) / len(metrics) for key in metrics[0]}


def _split_features(payload, train_ratio, seed):
    state = payload["state_feat"].float()
    action = payload["action_feat"].float()
    future = payload["future_feat"].float()
    n = state.shape[0]
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    train_n = max(1, min(n - 1, int(n * train_ratio)))
    return (
        TensorDataset(state[perm[:train_n]], action[perm[:train_n]], future[perm[:train_n]]),
        TensorDataset(state[perm[train_n:]], action[perm[train_n:]], future[perm[train_n:]]),
    )


def main():
    parser = argparse.ArgumentParser(description="Input ablations for CRF delta-future prediction.")
    parser.add_argument("--features", default="runs/crf_features_libero_10k.pt")
    parser.add_argument(
        "--mode",
        choices=["lowrank", "delta_only", "state_delta", "action_delta", "full_mlp"],
        default="lowrank",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--eval-batches", type=int, default=4)
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.features, map_location="cpu")
    train_ds, eval_ds = _split_features(payload, args.train_ratio, args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    state_dim = payload["state_feat"].shape[-1]
    action_dim = payload["action_feat"].shape[-1]
    future_dim = payload["future_feat"].shape[-1]
    if args.mode == "lowrank":
        model = LowRankActionResponseHead(state_dim, action_dim, future_dim, hidden_dim=args.hidden_dim, rank=args.rank)
    else:
        input_dim = {
            "delta_only": action_dim,
            "state_delta": state_dim + action_dim,
            "action_delta": action_dim + action_dim,
            "full_mlp": state_dim + action_dim + action_dim,
        }[args.mode]
        model = MLPDeltaPredictor(input_dim, future_dim, hidden_dim=args.hidden_dim)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(
        f"mode={args.mode} n={payload['state_feat'].shape[0]} train={len(train_ds)} eval={len(eval_ds)} "
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

        state, action, delta_action, target = _make_response_tensors(batch, device)
        if args.mode == "lowrank":
            pred = model(state, action, delta_action)
        else:
            pred = model(_make_input(args.mode, state, action, delta_action))
        loss = F.mse_loss(pred.float(), target.float())
        metric = _metrics(pred, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running.append(metric)
        if step == 1 or step % args.log_every == 0:
            recent = running[-args.log_every:]
            print(
                f"step={step:05d} train_rel_mse={sum(x['rel_mse'] for x in recent)/len(recent):.4f} "
                f"train_cos={sum(x['cos'] for x in recent)/len(recent):.4f}"
            )
        if step == 1 or (args.eval_every > 0 and step % args.eval_every == 0):
            eval_metric = _evaluate(model, args.mode, eval_loader, device, args.eval_batches)
            print(
                f"eval@{step:05d} rel_mse={eval_metric['rel_mse']:.4f} "
                f"cos={eval_metric['cos']:.4f} mse={eval_metric['mse']:.6f}"
            )


if __name__ == "__main__":
    main()
