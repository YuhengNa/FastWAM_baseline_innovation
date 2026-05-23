import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class MLPTokenResponseHead(nn.Module):
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
    return state_anchor, action_anchor, delta_action, target_delta_future.flatten(1), target_delta_future


def _make_input(mode, state, action, delta_action):
    if mode == "delta_only":
        return delta_action
    if mode == "action_delta":
        return torch.cat([action, delta_action], dim=-1)
    if mode == "full_mlp":
        return torch.cat([state, action, delta_action], dim=-1)
    raise ValueError(f"Unsupported MLP mode: {mode}")


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


def _rank_1d(x):
    order = x.argsort(dim=-1)
    ranks = torch.empty_like(order, dtype=torch.float32)
    values = torch.arange(x.shape[-1], device=x.device, dtype=torch.float32).expand_as(ranks)
    ranks.scatter_(dim=-1, index=order, src=values)
    return ranks


def _energy_metrics(pred_flat, target_flat, target_tokens, top_fracs):
    pred = pred_flat.float()
    target = target_flat.float()
    pred_tokens = pred.reshape(target_tokens.shape)
    pred_energy = pred_tokens.pow(2).sum(dim=-1)
    target_energy = target_tokens.float().pow(2).sum(dim=-1)

    mse = F.mse_loss(pred, target)
    zero_mse = target.pow(2).mean().clamp(min=1e-12)
    cos = F.cosine_similarity(pred, target, dim=-1).mean()

    pred_centered = pred_energy - pred_energy.mean(dim=1, keepdim=True)
    target_centered = target_energy - target_energy.mean(dim=1, keepdim=True)
    pearson = (
        pred_centered.mul(target_centered).sum(dim=1)
        / (
            pred_centered.pow(2).sum(dim=1).sqrt()
            * target_centered.pow(2).sum(dim=1).sqrt()
        ).clamp(min=1e-12)
    ).mean()

    pred_rank = _rank_1d(pred_energy)
    target_rank = _rank_1d(target_energy)
    rank_pred_centered = pred_rank - pred_rank.mean(dim=1, keepdim=True)
    rank_target_centered = target_rank - target_rank.mean(dim=1, keepdim=True)
    spearman = (
        rank_pred_centered.mul(rank_target_centered).sum(dim=1)
        / (
            rank_pred_centered.pow(2).sum(dim=1).sqrt()
            * rank_target_centered.pow(2).sum(dim=1).sqrt()
        ).clamp(min=1e-12)
    ).mean()

    out = {
        "rel_mse": float((mse / zero_mse).item()),
        "cos": float(cos.item()),
        "energy_pearson": float(pearson.item()),
        "energy_spearman": float(spearman.item()),
    }
    for frac in top_fracs:
        k = max(1, int(target_energy.shape[1] * frac))
        pred_top = pred_energy.topk(k, dim=1).indices
        target_top = target_energy.topk(k, dim=1).indices
        pred_mask = torch.zeros_like(pred_energy, dtype=torch.bool)
        target_mask = torch.zeros_like(target_energy, dtype=torch.bool)
        pred_mask.scatter_(1, pred_top, True)
        target_mask.scatter_(1, target_top, True)
        inter = (pred_mask & target_mask).float().sum(dim=1)
        union = (pred_mask | target_mask).float().sum(dim=1).clamp(min=1.0)
        pred_top1 = pred_energy.argmax(dim=1, keepdim=True)
        pointing = target_mask.gather(1, pred_top1).float().mean()
        mass = pred_energy.masked_fill(~target_mask, 0.0).sum(dim=1) / pred_energy.sum(dim=1).clamp(min=1e-12)
        pct = int(round(frac * 100))
        out[f"top{pct}_iou"] = float((inter / union).mean().item())
        out[f"pointing@{pct}"] = float(pointing.item())
        out[f"mass_in_top{pct}"] = float(mass.mean().item())
    return out


def _new_model(mode, state_dim, action_dim, future_dim, hidden_dim, rank):
    if mode == "lowrank":
        return LowRankTokenResponseHead(state_dim, action_dim, future_dim, hidden_dim=hidden_dim, rank=rank)
    input_dim = {
        "delta_only": action_dim,
        "action_delta": action_dim + action_dim,
        "full_mlp": state_dim + action_dim + action_dim,
    }[mode]
    return MLPTokenResponseHead(input_dim=input_dim, future_dim=future_dim, hidden_dim=hidden_dim)


def _predict(model, mode, state, action, delta_action):
    if mode == "lowrank":
        return model(state, action, delta_action)
    return model(_make_input(mode, state, action, delta_action))


def _average_dicts(rows):
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


@torch.no_grad()
def _evaluate_model(name, model, mode, loader, device, eval_batches, top_fracs, random_seed):
    if model is not None:
        model.eval()
    rows = []
    generator = torch.Generator(device=device).manual_seed(random_seed)
    for batch_idx, batch in enumerate(loader):
        if eval_batches > 0 and batch_idx >= eval_batches:
            break
        state, action, delta_action, target_flat, target_tokens = _make_response_tensors(batch, device)
        if name == "zero":
            pred = torch.zeros_like(target_flat)
        elif name == "random_map":
            pred = torch.randn(target_flat.shape, device=device, generator=generator)
        elif name == "shuffle_target":
            perm = torch.randperm(target_flat.shape[0], device=device, generator=generator)
            pred = target_flat[perm]
        else:
            pred = _predict(model, mode, state, action, delta_action)
        rows.append(_energy_metrics(pred, target_flat, target_tokens, top_fracs))
    if model is not None:
        model.train()
    return _average_dicts(rows)


def _train_model(mode, train_loader, device, args, state_dim, action_dim, future_dim):
    model = _new_model(mode, state_dim, action_dim, future_dim, args.hidden_dim, args.rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    data_iter = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        state, action, delta_action, target_flat, _ = _make_response_tensors(batch, device)
        pred = _predict(model, mode, state, action, delta_action)
        loss = F.mse_loss(pred.float(), target_flat.float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or (args.log_every > 0 and step % args.log_every == 0):
            cos = F.cosine_similarity(pred.detach().float(), target_flat.float(), dim=-1).mean()
            print(f"{mode} step={step:05d} loss={loss.item():.6f} cos={cos.item():.4f}")
    return model


def _print_table(results, top_frac):
    pct = int(round(top_frac * 100))
    columns = [
        ("model", "model"),
        ("rel_mse", "rel_mse"),
        ("cos", "cos"),
        ("energy_pearson", "e_pearson"),
        ("energy_spearman", "e_spearman"),
        (f"top{pct}_iou", f"top{pct}_iou"),
        (f"pointing@{pct}", f"point@{pct}"),
        (f"mass_in_top{pct}", f"mass@{pct}"),
    ]
    widths = [max(len(title), 12) for _, title in columns]
    print(" ".join(title.ljust(width) for width, (_, title) in zip(widths, columns)))
    for name, metrics in results:
        values = []
        for key, _ in columns:
            if key == "model":
                values.append(name)
            else:
                values.append(f"{metrics[key]:.4f}")
        print(" ".join(value.ljust(width) for width, value in zip(widths, values)))


def main():
    parser = argparse.ArgumentParser(description="Quantitatively evaluate spatial token CRF response maps.")
    parser.add_argument("--features", default="runs/crf_token_features_libero_5k.pt")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["lowrank", "delta_only", "action_delta", "full_mlp"],
        default=["lowrank", "action_delta", "delta_only"],
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=0)
    parser.add_argument("--top-fracs", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.features, map_location="cpu")
    train_ds, eval_ds = _split_features(payload, args.train_ratio, args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    state_dim = payload["state_tokens"].shape[-1]
    action_dim = payload["action_feat"].shape[-1]
    future_dim = payload["future_tokens"].shape[1] * payload["future_tokens"].shape[2]
    print(
        f"features={args.features} n={payload['state_tokens'].shape[0]} train={len(train_ds)} eval={len(eval_ds)} "
        f"state_dim={state_dim} action_dim={action_dim} future_dim={future_dim} device={device}"
    )

    results = []
    for baseline in ["zero", "random_map", "shuffle_target"]:
        metrics = _evaluate_model(
            baseline, None, "", eval_loader, device, args.eval_batches, args.top_fracs, args.seed + 100
        )
        results.append((baseline, metrics))

    for mode in args.modes:
        model = _train_model(mode, train_loader, device, args, state_dim, action_dim, future_dim)
        metrics = _evaluate_model(mode, model, mode, eval_loader, device, args.eval_batches, args.top_fracs, args.seed)
        results.append((mode, metrics))

    print("\nMain table:")
    _print_table(results, top_frac=0.10)
    for frac in args.top_fracs:
        pct = int(round(frac * 100))
        print(f"\nTop-{pct}% spatial overlap:")
        for name, metrics in results:
            print(
                f"{name}: iou={metrics[f'top{pct}_iou']:.4f} "
                f"pointing={metrics[f'pointing@{pct}']:.4f} "
                f"mass={metrics[f'mass_in_top{pct}']:.4f}"
            )


if __name__ == "__main__":
    main()
