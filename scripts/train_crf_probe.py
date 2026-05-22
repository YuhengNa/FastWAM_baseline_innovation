import argparse
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers


class LowRankActionResponseHead(nn.Module):
    """Predicts a local low-rank response operator from state/action context.

    Given a base state/action context and an action perturbation da, the head
    predicts df ~= R(s, a) da without materializing a full dense matrix.
    """

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


def _load_config(config_path: str, task_name: str | None):
    cfg = OmegaConf.load(config_path)
    if task_name is not None:
        task_cfg = OmegaConf.load(Path("configs") / "task" / f"{task_name}.yaml")
        cfg = OmegaConf.merge(cfg, task_cfg)
    return cfg


def _nearest_pairs(state_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state_float = F.normalize(state_feat.float(), dim=-1)
    dist = torch.cdist(state_float, state_float, p=2)
    dist.fill_diagonal_(float("inf"))
    nn_idx = dist.argmin(dim=1)
    anchor_idx = torch.arange(state_feat.shape[0], device=state_feat.device)
    pair_dist = dist[anchor_idx, nn_idx]
    return anchor_idx, nn_idx, pair_dist


def _latent_features(inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    latents = inputs["input_latents"]
    first = inputs["first_frame_latents"]
    if first is None:
        first = latents[:, :, 0:1]
    if latents.shape[2] <= 1:
        raise ValueError(f"Need at least one future latent step, got latent shape {tuple(latents.shape)}")

    state_feat = first.float().mean(dim=(2, 3, 4))
    future_feat = latents[:, :, 1:].float().mean(dim=(3, 4)).flatten(1)
    return state_feat, future_feat


def main():
    parser = argparse.ArgumentParser(description="Train a lightweight Action Response Field probe on FastWAM latents.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--task", default="libero_uncond_2cam224_1e-4")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-path", default="runs/crf_probe.pt")
    args = parser.parse_args()

    register_default_resolvers()
    cfg = _load_config(args.config, args.task)
    misc.register_work_dir(cfg.output_dir)

    dataset = instantiate(cfg.data.train)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    mixed_precision = _normalize_mixed_precision(str(cfg.mixed_precision))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    model.eval()
    model.requires_grad_(False)

    head = None
    optimizer = None
    running = []
    data_iter = iter(loader)

    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        with torch.no_grad():
            inputs = model.build_inputs(batch, tiled=False)
            state_feat, future_feat = _latent_features(inputs)
            action_feat = inputs["action"].float().flatten(1)

        anchor_idx, nn_idx, pair_dist = _nearest_pairs(state_feat)
        delta_action = action_feat[nn_idx] - action_feat[anchor_idx]
        target_delta_future = future_feat[nn_idx] - future_feat[anchor_idx]

        if head is None:
            head = LowRankActionResponseHead(
                state_dim=state_feat.shape[-1],
                action_dim=action_feat.shape[-1],
                future_dim=future_feat.shape[-1],
                hidden_dim=args.hidden_dim,
                rank=args.rank,
            ).to(device=device)
            optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
            print(
                "Initialized CRF probe: "
                f"state_dim={state_feat.shape[-1]} action_dim={action_feat.shape[-1]} "
                f"future_dim={future_feat.shape[-1]} rank={args.rank}"
            )

        pred_delta_future = head(
            state_feat[anchor_idx].to(device),
            action_feat[anchor_idx].to(device),
            delta_action.to(device),
        )
        target_delta_future = target_delta_future.to(device)
        loss = F.mse_loss(pred_delta_future.float(), target_delta_future.float())
        cosine = F.cosine_similarity(pred_delta_future.float(), target_delta_future.float(), dim=-1).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running.append((float(loss.detach().item()), float(cosine.detach().item()), float(pair_dist.mean().item())))
        if step == 1 or step % args.log_every == 0:
            mean_loss = sum(x[0] for x in running[-args.log_every:]) / min(len(running), args.log_every)
            mean_cos = sum(x[1] for x in running[-args.log_every:]) / min(len(running), args.log_every)
            mean_pair_dist = sum(x[2] for x in running[-args.log_every:]) / min(len(running), args.log_every)
            print(
                f"step={step:05d} loss={mean_loss:.6f} "
                f"cos={mean_cos:.4f} pair_dist={mean_pair_dist:.4f}"
            )

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head": head.state_dict(),
            "args": vars(args),
            "state_dim": int(head.trunk[0].in_features - head.action_dim),
            "action_dim": int(head.action_dim),
            "future_dim": int(head.future_dim),
            "rank": int(head.rank),
        },
        save_path,
    )
    print(f"Saved CRF probe checkpoint to {save_path}")


if __name__ == "__main__":
    main()
