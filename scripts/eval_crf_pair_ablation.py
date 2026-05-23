import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def _nearest_pairs(state_feat: torch.Tensor):
    state_float = F.normalize(state_feat.float(), dim=-1)
    dist = torch.cdist(state_float, state_float, p=2)
    dist.fill_diagonal_(float("inf"))
    nn_idx = dist.argmin(dim=1)
    anchor_idx = torch.arange(state_feat.shape[0], device=state_feat.device)
    return anchor_idx, nn_idx, dist[anchor_idx, nn_idx]


def _random_pairs(state_feat: torch.Tensor):
    n = state_feat.shape[0]
    anchor_idx = torch.arange(n, device=state_feat.device)
    nn_idx = torch.randperm(n, device=state_feat.device)
    same = nn_idx == anchor_idx
    if bool(same.any()):
        nn_idx[same] = (nn_idx[same] + 1) % n
    dist = torch.cdist(F.normalize(state_feat.float(), dim=-1), F.normalize(state_feat.float(), dim=-1), p=2)
    return anchor_idx, nn_idx, dist[anchor_idx, nn_idx]


def _metrics(delta_action, delta_future, pair_dist):
    action_norm = delta_action.norm(dim=-1)
    future_norm = delta_future.norm(dim=-1)
    action_future_norm_corr = torch.corrcoef(torch.stack([action_norm, future_norm]))[0, 1]
    return {
        "pair_dist": float(pair_dist.mean().item()),
        "action_delta_norm": float(action_norm.mean().item()),
        "future_delta_norm": float(future_norm.mean().item()),
        "action_future_norm_corr": float(action_future_norm_corr.nan_to_num(0.0).item()),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare nearest-neighbor and random-pair CRF supervision statistics.")
    parser.add_argument("--features", default="runs/crf_features_libero_10k.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.features, map_location="cpu")
    dataset = TensorDataset(
        payload["state_feat"].float(),
        payload["action_feat"].float(),
        payload["future_feat"].float(),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    nn_stats = []
    random_stats = []
    for batch_id, batch in enumerate(loader):
        if batch_id >= args.batches:
            break
        state, action, future = [x.to(device) for x in batch]

        for mode, pair_fn, sink in (
            ("nearest", _nearest_pairs, nn_stats),
            ("random", _random_pairs, random_stats),
        ):
            anchor_idx, pair_idx, pair_dist = pair_fn(state)
            delta_action = action[pair_idx] - action[anchor_idx]
            delta_future = future[pair_idx] - future[anchor_idx]
            sink.append(_metrics(delta_action, delta_future, pair_dist))

    def summarize(name, stats):
        keys = stats[0].keys()
        print(f"{name}:")
        for key in keys:
            value = sum(s[key] for s in stats) / len(stats)
            print(f"  {key}: {value:.6f}")

    summarize("nearest_pairs", nn_stats)
    summarize("random_pairs", random_stats)


if __name__ == "__main__":
    main()
