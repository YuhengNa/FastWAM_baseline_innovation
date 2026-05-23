import argparse

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
    true_delta_action = action[pair_idx] - action[anchor_idx]
    target_delta_future = future[pair_idx] - future[anchor_idx]
    return state_anchor, action_anchor, true_delta_action, target_delta_future


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
        state, action, delta_action, target_future = _make_response_tensors(batch, device)
        pred_future = head(state, action, delta_action)
        loss = F.mse_loss(pred_future.float(), target_future.float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 500 == 0:
            cos = F.cosine_similarity(pred_future.float(), target_future.float(), dim=-1).mean()
            print(f"train_step={step:05d} loss={loss.item():.6f} cos={cos.item():.4f}")


def _optimize_delta_action(head, state, action, target_future, steps, lr, l2_weight):
    delta = torch.zeros_like(action, requires_grad=True)
    optimizer = torch.optim.AdamW([delta], lr=lr)
    for _ in range(steps):
        pred_future = head(state, action, delta)
        loss = F.mse_loss(pred_future.float(), target_future.float())
        if l2_weight > 0:
            loss = loss + l2_weight * delta.float().pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return delta.detach()


@torch.no_grad()
def _candidate_rerank(head, state, action, true_delta_action, target_future, num_candidates, noise_scale):
    batch_size = state.shape[0]
    candidates = []
    candidates.append(torch.zeros_like(true_delta_action))
    candidates.append(true_delta_action)
    for _ in range(max(num_candidates - 2, 0)):
        candidates.append(torch.randn_like(true_delta_action) * noise_scale)
    cand = torch.stack(candidates, dim=1)
    flat_state = state[:, None, :].expand(-1, cand.shape[1], -1).reshape(batch_size * cand.shape[1], -1)
    flat_action = action[:, None, :].expand(-1, cand.shape[1], -1).reshape(batch_size * cand.shape[1], -1)
    flat_delta = cand.reshape(batch_size * cand.shape[1], -1)
    pred = head(flat_state, flat_action, flat_delta).reshape(batch_size, cand.shape[1], -1)
    err = (pred.float() - target_future[:, None, :].float()).pow(2).mean(dim=-1)
    best_idx = err.argmin(dim=1)
    chosen = cand[torch.arange(batch_size, device=state.device), best_idx]
    oracle_rank = (err.argsort(dim=1) == 1).nonzero()
    oracle_top1 = float((best_idx == 1).float().mean().item())
    oracle_top5 = 0.0
    sorted_idx = err.argsort(dim=1)[:, : min(5, cand.shape[1])]
    oracle_top5 = float((sorted_idx == 1).any(dim=1).float().mean().item())
    return chosen, oracle_top1, oracle_top5


def _action_metrics(pred_delta, true_delta):
    pred = pred_delta.float()
    target = true_delta.float()
    cos = F.cosine_similarity(pred, target, dim=-1)
    mse = F.mse_loss(pred, target).detach()
    zero_mse = target.pow(2).mean().detach().clamp(min=1e-12)
    return {
        "action_cos": float(cos.mean().item()),
        "action_pos_cos_rate": float((cos > 0).float().mean().item()),
        "action_rel_mse": float((mse / zero_mse).item()),
        "pred_norm": float(pred.norm(dim=-1).mean().item()),
        "true_norm": float(target.norm(dim=-1).mean().item()),
    }


def main():
    parser = argparse.ArgumentParser(description="Offline toy eval: recover action correction from target future delta.")
    parser.add_argument("--features", default="runs/crf_features_libero_5k.pt")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--opt-steps", type=int, default=80)
    parser.add_argument("--opt-lr", type=float, default=0.05)
    parser.add_argument("--l2-weight", type=float, default=1e-4)
    parser.add_argument("--rerank-candidates", type=int, default=64)
    parser.add_argument("--rerank-noise-scale", type=float, default=1.0)
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.features, map_location="cpu")
    train_ds, eval_ds = _split_features(payload, args.train_ratio, args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

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

    print(
        f"features={args.features} train={len(train_ds)} eval={len(eval_ds)} "
        f"state_dim={state_dim} action_dim={action_dim} future_dim={future_dim}"
    )
    _train_head(head, train_loader, device=device, steps=args.train_steps, lr=args.lr)

    head.eval()
    opt_metrics = []
    rerank_metrics = []
    oracle_top1 = []
    oracle_top5 = []
    for batch in eval_loader:
        state, action, true_delta_action, target_future = _make_response_tensors(batch, device)
        pred_delta = _optimize_delta_action(
            head=head,
            state=state,
            action=action,
            target_future=target_future,
            steps=args.opt_steps,
            lr=args.opt_lr,
            l2_weight=args.l2_weight,
        )
        opt_metrics.append(_action_metrics(pred_delta, true_delta_action))
        chosen_delta, top1, top5 = _candidate_rerank(
            head=head,
            state=state,
            action=action,
            true_delta_action=true_delta_action,
            target_future=target_future,
            num_candidates=args.rerank_candidates,
            noise_scale=args.rerank_noise_scale,
        )
        rerank_metrics.append(_action_metrics(chosen_delta, true_delta_action))
        oracle_top1.append(top1)
        oracle_top5.append(top5)

    def summarize(name, metrics):
        keys = metrics[0].keys()
        print(name)
        for key in keys:
            print(f"  {key}: {sum(m[key] for m in metrics) / len(metrics):.6f}")

    summarize("optimized_delta_action", opt_metrics)
    summarize("rerank_chosen_delta_action", rerank_metrics)
    print(f"rerank_oracle_delta_top1: {sum(oracle_top1) / len(oracle_top1):.6f}")
    print(f"rerank_oracle_delta_top5: {sum(oracle_top5) / len(oracle_top5):.6f}")


if __name__ == "__main__":
    main()
