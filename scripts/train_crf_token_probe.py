import argparse

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


def _metrics(pred_flat, target_flat):
    pred = pred_flat.float()
    target = target_flat.float()
    mse = F.mse_loss(pred, target).detach()
    zero_mse = target.pow(2).mean().detach().clamp(min=1e-12)
    cos = F.cosine_similarity(pred, target, dim=-1).mean().detach()
    return {
        "mse": float(mse.item()),
        "rel_mse": float((mse / zero_mse).item()),
        "cos": float(cos.item()),
    }


def _token_energy_overlap(pred_flat, target_tokens, top_frac=0.1):
    pred_tokens = pred_flat.reshape(target_tokens.shape)
    pred_energy = pred_tokens.float().pow(2).sum(dim=-1)
    target_energy = target_tokens.float().pow(2).sum(dim=-1)
    k = max(1, int(target_energy.shape[1] * top_frac))
    pred_top = pred_energy.topk(k, dim=1).indices
    target_top = target_energy.topk(k, dim=1).indices
    pred_mask = torch.zeros_like(pred_energy, dtype=torch.bool)
    target_mask = torch.zeros_like(target_energy, dtype=torch.bool)
    pred_mask.scatter_(1, pred_top, True)
    target_mask.scatter_(1, target_top, True)
    inter = (pred_mask & target_mask).float().sum(dim=1)
    union = (pred_mask | target_mask).float().sum(dim=1).clamp(min=1.0)
    return float((inter / union).mean().item())


@torch.no_grad()
def _evaluate(model, loader, device, eval_batches):
    model.eval()
    metrics = []
    overlaps = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= eval_batches:
            break
        state, action, delta_action, target_flat, target_tokens = _make_response_tensors(batch, device)
        pred = model(state, action, delta_action)
        metrics.append(_metrics(pred, target_flat))
        overlaps.append(_token_energy_overlap(pred, target_tokens))
    model.train()
    out = {key: sum(m[key] for m in metrics) / len(metrics) for key in metrics[0]}
    out["top10_energy_iou"] = sum(overlaps) / len(overlaps)
    return out


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


def main():
    parser = argparse.ArgumentParser(description="Train a token-level Action Response Field probe.")
    parser.add_argument("--features", default="runs/crf_token_features_libero_1k.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--eval-batches", type=int, default=4)
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.features, map_location="cpu")
    train_ds, eval_ds = _split_features(payload, args.train_ratio, args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    state_dim = payload["state_tokens"].shape[-1]
    action_dim = payload["action_feat"].shape[-1]
    future_dim = payload["future_tokens"].shape[1] * payload["future_tokens"].shape[2]
    model = LowRankTokenResponseHead(
        state_dim=state_dim,
        action_dim=action_dim,
        future_dim=future_dim,
        hidden_dim=args.hidden_dim,
        rank=args.rank,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(
        f"Loaded token features: n={payload['state_tokens'].shape[0]} train={len(train_ds)} eval={len(eval_ds)} "
        f"state_tokens={tuple(payload['state_tokens'].shape[1:])} "
        f"future_tokens={tuple(payload['future_tokens'].shape[1:])} future_dim={future_dim}"
    )

    data_iter = iter(train_loader)
    running = []
    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        state, action, delta_action, target_flat, target_tokens = _make_response_tensors(batch, device)
        pred = model(state, action, delta_action)
        loss = F.mse_loss(pred.float(), target_flat.float())
        metric = _metrics(pred, target_flat)

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
            eval_metric = _evaluate(model, eval_loader, device, args.eval_batches)
            print(
                f"eval@{step:05d} rel_mse={eval_metric['rel_mse']:.4f} "
                f"cos={eval_metric['cos']:.4f} top10_energy_iou={eval_metric['top10_energy_iou']:.4f} "
                f"mse={eval_metric['mse']:.6f}"
            )


if __name__ == "__main__":
    main()
