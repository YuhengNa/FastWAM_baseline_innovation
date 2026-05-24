import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm


def _first_valid_candidate(
    candidate_indices: torch.Tensor,
    candidate_dist: torch.Tensor,
    anchor_idx: int,
    action: torch.Tensor,
    future: torch.Tensor,
    min_action_delta: float,
    min_future_delta: float,
):
    anchor_action = action[anchor_idx]
    anchor_future = future[anchor_idx]
    for rank in range(candidate_indices.numel()):
        pair_idx = int(candidate_indices[rank].item())
        action_delta_norm = float((action[pair_idx] - anchor_action).norm().item())
        if action_delta_norm < min_action_delta:
            continue
        future_delta_norm = float((future[pair_idx] - anchor_future).flatten().norm().item())
        if future_delta_norm < min_future_delta:
            continue
        return pair_idx, float(candidate_dist[rank].item()), action_delta_norm, future_delta_norm
    return None


def main():
    parser = argparse.ArgumentParser(description="Build fixed nearest-neighbor response pairs from CRF token features.")
    parser.add_argument("--features", default="runs/crf_token_features_libero_5k.pt")
    parser.add_argument("--save-path", default="runs/crf_pairs_libero_5k.pt")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--neighbor-topk", type=int, default=32)
    parser.add_argument("--min-action-delta", type=float, default=0.0)
    parser.add_argument("--min-future-delta", type=float, default=0.0)
    parser.add_argument("--max-pair-dist", type=float, default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.features, map_location="cpu")
    state = payload["state_tokens"].float()
    action = payload["action_feat"].float()
    future = payload["future_tokens"].float()
    n = state.shape[0]
    if n < 2:
        raise ValueError(f"Need at least 2 rows to build pairs, got {n}")

    state_global = F.normalize(state.mean(dim=1), dim=-1)
    state_global_device = state_global.to(device)
    action_device = action.to(device)
    future_device = future.to(device)

    pair_indices = torch.empty(n, dtype=torch.long)
    pair_dist = torch.empty(n, dtype=torch.float32)
    action_delta_norm = torch.empty(n, dtype=torch.float32)
    future_delta_norm = torch.empty(n, dtype=torch.float32)
    keep = torch.zeros(n, dtype=torch.bool)

    topk = min(max(2, int(args.neighbor_topk)), n)
    pbar = tqdm(range(0, n, args.chunk_size), desc="Building CRF response pairs")
    for start in pbar:
        end = min(start + args.chunk_size, n)
        query = state_global_device[start:end]
        dist = torch.cdist(query, state_global_device, p=2)
        row = torch.arange(end - start, device=device)
        col = torch.arange(start, end, device=device)
        dist[row, col] = float("inf")
        values, indices = dist.topk(k=topk, largest=False, dim=1)

        for local_idx in range(end - start):
            anchor_idx = start + local_idx
            selected = _first_valid_candidate(
                candidate_indices=indices[local_idx],
                candidate_dist=values[local_idx],
                anchor_idx=anchor_idx,
                action=action_device,
                future=future_device,
                min_action_delta=float(args.min_action_delta),
                min_future_delta=float(args.min_future_delta),
            )
            if selected is None:
                continue
            pair_idx, dist_value, action_norm, future_norm = selected
            if args.max_pair_dist is not None and dist_value > float(args.max_pair_dist):
                continue
            pair_indices[anchor_idx] = pair_idx
            pair_dist[anchor_idx] = dist_value
            action_delta_norm[anchor_idx] = action_norm
            future_delta_norm[anchor_idx] = future_norm
            keep[anchor_idx] = True

    kept = int(keep.sum().item())
    if kept == 0:
        raise ValueError("No valid response pairs were found. Relax filtering thresholds.")

    anchor_indices = torch.arange(n, dtype=torch.long)[keep]
    pair_indices = pair_indices[keep]
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    state_anchor = state[anchor_indices].to(dtype=dtype)
    action_anchor = action[anchor_indices].to(dtype=dtype)
    delta_action = (action[pair_indices] - action[anchor_indices]).to(dtype=dtype)
    target_response = (future[pair_indices] - future[anchor_indices]).to(dtype=dtype)

    out = {
        "state_tokens": state_anchor,
        "action_feat": action_anchor,
        "delta_action": delta_action,
        "target_response": target_response,
        "anchor_indices": anchor_indices,
        "pair_indices": pair_indices.cpu(),
        "pair_dist": pair_dist[keep].cpu(),
        "action_delta_norm": action_delta_norm[keep].cpu(),
        "future_delta_norm": future_delta_norm[keep].cpu(),
        "source_features": str(args.features),
        "spatial_pool": payload.get("spatial_pool", None),
        "task": payload.get("task", None),
        "filters": {
            "neighbor_topk": int(args.neighbor_topk),
            "min_action_delta": float(args.min_action_delta),
            "min_future_delta": float(args.min_future_delta),
            "max_pair_dist": None if args.max_pair_dist is None else float(args.max_pair_dist),
        },
    }
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, save_path)

    print(f"Saved {kept}/{n} response pairs to {save_path}")
    print(f"  state_tokens={tuple(out['state_tokens'].shape)} dtype={out['state_tokens'].dtype}")
    print(f"  action_feat={tuple(out['action_feat'].shape)} dtype={out['action_feat'].dtype}")
    print(f"  delta_action={tuple(out['delta_action'].shape)} dtype={out['delta_action'].dtype}")
    print(f"  target_response={tuple(out['target_response'].shape)} dtype={out['target_response'].dtype}")
    print(
        "  pair_dist mean={:.6f} median={:.6f} max={:.6f}".format(
            out["pair_dist"].mean().item(),
            out["pair_dist"].median().item(),
            out["pair_dist"].max().item(),
        )
    )
    print(
        "  action_delta_norm mean={:.6f} median={:.6f}".format(
            out["action_delta_norm"].mean().item(),
            out["action_delta_norm"].median().item(),
        )
    )
    print(
        "  future_delta_norm mean={:.6f} median={:.6f}".format(
            out["future_delta_norm"].mean().item(),
            out["future_delta_norm"].median().item(),
        )
    )


if __name__ == "__main__":
    main()
