import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from fastwam.models.wan22.response_field import TokenResponseFieldHead, response_metrics


def _split_pairs(payload, train_ratio, seed):
    state = payload["state_tokens"].float()
    action = payload["action_feat"].float()
    delta_action = payload["delta_action"].float()
    target = payload["target_response"].float()
    n = state.shape[0]
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    train_n = max(1, min(n - 1, int(n * train_ratio)))
    train_idx = perm[:train_n]
    eval_idx = perm[train_n:]
    train_ds = TensorDataset(state[train_idx], action[train_idx], delta_action[train_idx], target[train_idx])
    eval_ds = TensorDataset(state[eval_idx], action[eval_idx], delta_action[eval_idx], target[eval_idx])
    return train_ds, eval_ds


def _batch_to_device(batch, device):
    return [x.to(device, dtype=torch.float32) for x in batch]


def _loss_and_metrics(model, batch, device):
    state, action, delta_action, target = _batch_to_device(batch, device)
    pred = model(
        state_tokens=state,
        action_flat=action,
        delta_action_flat=delta_action,
        target_token_count=target.shape[1],
    )
    loss = F.mse_loss(pred.float(), target.float())
    metrics = response_metrics(pred, target)
    return loss, metrics


@torch.no_grad()
def _evaluate(model, loader, device, max_batches):
    model.eval()
    rows = []
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        _, metrics = _loss_and_metrics(model, batch, device)
        rows.append(metrics)
    model.train()
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def main():
    parser = argparse.ArgumentParser(description="Train the RF-WAM token response head on a fixed CRF pair dataset.")
    parser.add_argument("--pairs", default="runs/crf_pairs_libero_5k.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=0)
    parser.add_argument("--save-path", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.pairs, map_location="cpu")
    train_ds, eval_ds = _split_pairs(payload, args.train_ratio, args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    state_dim = payload["state_tokens"].shape[-1]
    action_dim = payload["action_feat"].shape[-1]
    target_tokens = payload["target_response"].shape[1]
    model = TokenResponseFieldHead(
        latent_dim=state_dim,
        action_flat_dim=action_dim,
        hidden_dim=args.hidden_dim,
        rank=args.rank,
        spatial_pool=int(payload.get("spatial_pool") or 2),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"pairs={args.pairs} n={payload['state_tokens'].shape[0]} train={len(train_ds)} eval={len(eval_ds)} "
        f"state_tokens={tuple(payload['state_tokens'].shape[1:])} "
        f"action_dim={action_dim} target_response={tuple(payload['target_response'].shape[1:])} "
        f"device={device}"
    )
    if "pair_dist" in payload:
        print(
            "pair stats: dist_mean={:.6f} action_delta_mean={:.6f} future_delta_mean={:.6f}".format(
                payload["pair_dist"].float().mean().item(),
                payload["action_delta_norm"].float().mean().item(),
                payload["future_delta_norm"].float().mean().item(),
            )
        )

    data_iter = iter(train_loader)
    running = []
    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        loss, metrics = _loss_and_metrics(model, batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running.append(metrics)

        if step == 1 or step % args.log_every == 0:
            recent = running[-args.log_every:]
            mean = {key: sum(row[key] for row in recent) / len(recent) for key in recent[0]}
            print(
                f"step={step:05d} "
                f"train_rel_mse={mean['loss_response_rel_mse']:.4f} "
                f"train_cos={mean['response_cos']:.4f} "
                f"train_top10_iou={mean['response_top10_iou']:.4f} "
                f"train_pointing10={mean['response_pointing10']:.4f}"
            )

        if step == 1 or (args.eval_every > 0 and step % args.eval_every == 0):
            eval_metrics = _evaluate(model, eval_loader, device, args.eval_batches)
            print(
                f"eval@{step:05d} "
                f"rel_mse={eval_metrics['loss_response_rel_mse']:.4f} "
                f"cos={eval_metrics['response_cos']:.4f} "
                f"top10_iou={eval_metrics['response_top10_iou']:.4f} "
                f"pointing10={eval_metrics['response_pointing10']:.4f}"
            )

    if args.save_path:
        torch.save(
            {
                "model": model.state_dict(),
                "config": {
                    "latent_dim": state_dim,
                    "action_flat_dim": action_dim,
                    "hidden_dim": args.hidden_dim,
                    "rank": args.rank,
                    "target_tokens": target_tokens,
                    "source_pairs": args.pairs,
                },
            },
            args.save_path,
        )
        print(f"Saved response head checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
