from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def latent_tokens(latents: torch.Tensor, spatial_pool: int = 2) -> torch.Tensor:
    """Convert [B,C,T,H,W] VAE latents into [B,T*H'*W',C] tokens."""
    if latents.ndim != 5:
        raise ValueError(f"`latents` must be [B,C,T,H,W], got {tuple(latents.shape)}")
    if spatial_pool < 1:
        raise ValueError(f"`spatial_pool` must be >= 1, got {spatial_pool}")

    bsz, channels, steps, height, width = latents.shape
    x = latents.float().permute(0, 2, 1, 3, 4).reshape(bsz * steps, channels, height, width)
    if spatial_pool > 1:
        x = F.avg_pool2d(x, kernel_size=spatial_pool, stride=spatial_pool)
    pooled_h, pooled_w = x.shape[-2:]
    return x.reshape(bsz, steps, channels, pooled_h, pooled_w).permute(0, 1, 3, 4, 2).reshape(
        bsz, steps * pooled_h * pooled_w, channels
    )


class TokenResponseFieldHead(nn.Module):
    """Predict token-level future response caused by local action perturbations."""

    def __init__(
        self,
        latent_dim: int,
        action_flat_dim: int,
        hidden_dim: int = 512,
        rank: int = 8,
        spatial_pool: int = 2,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_flat_dim = int(action_flat_dim)
        self.hidden_dim = int(hidden_dim)
        self.rank = int(rank)
        self.spatial_pool = int(spatial_pool)

        self.context_trunk = nn.Sequential(
            nn.Linear(self.latent_dim + self.action_flat_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.delta_head = nn.Linear(self.hidden_dim, self.action_flat_dim * self.rank)
        self.token_basis = nn.Sequential(
            nn.Linear(self.latent_dim + self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.latent_dim * self.rank),
        )

    def _repeat_state_tokens(self, state_tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
        if state_tokens.shape[1] == target_tokens:
            return state_tokens
        repeat = (target_tokens + state_tokens.shape[1] - 1) // state_tokens.shape[1]
        return state_tokens.repeat(1, repeat, 1)[:, :target_tokens]

    def forward(
        self,
        state_tokens: torch.Tensor,
        action_flat: torch.Tensor,
        delta_action_flat: torch.Tensor,
        target_token_count: int,
    ) -> torch.Tensor:
        if action_flat.shape[-1] != self.action_flat_dim:
            raise ValueError(
                f"`action_flat` dim mismatch: expected {self.action_flat_dim}, got {action_flat.shape[-1]}"
            )
        if delta_action_flat.shape[-1] != self.action_flat_dim:
            raise ValueError(
                f"`delta_action_flat` dim mismatch: expected {self.action_flat_dim}, got {delta_action_flat.shape[-1]}"
            )
        if state_tokens.shape[-1] != self.latent_dim:
            raise ValueError(
                f"`state_tokens` channel mismatch: expected {self.latent_dim}, got {state_tokens.shape[-1]}"
            )

        dtype = self.context_trunk[0].weight.dtype
        state_tokens = state_tokens.to(dtype=dtype)
        action_flat = action_flat.to(dtype=dtype)
        delta_action_flat = delta_action_flat.to(dtype=dtype)

        state_global = state_tokens.mean(dim=1)
        context = self.context_trunk(torch.cat([state_global, action_flat], dim=-1))
        v = self.delta_head(context).view(-1, self.action_flat_dim, self.rank)
        coeff = torch.einsum("ba,bar->br", delta_action_flat, v)

        query_tokens = self._repeat_state_tokens(state_tokens, int(target_token_count))
        context_tokens = context.unsqueeze(1).expand(-1, query_tokens.shape[1], -1)
        basis = self.token_basis(torch.cat([query_tokens, context_tokens], dim=-1))
        basis = basis.view(query_tokens.shape[0], query_tokens.shape[1], self.latent_dim, self.rank)
        return torch.einsum("bncr,br->bnc", basis, coeff)


def nearest_response_batch(
    state_tokens: torch.Tensor,
    action: torch.Tensor,
    future_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if state_tokens.shape[0] < 2:
        return None
    state_global = F.normalize(state_tokens.float().mean(dim=1), dim=-1)
    dist = torch.cdist(state_global, state_global, p=2)
    dist.fill_diagonal_(float("inf"))
    pair_idx = dist.argmin(dim=1)
    anchor_idx = torch.arange(state_tokens.shape[0], device=state_tokens.device)

    action_flat = action.float().flatten(1)
    delta_action = action_flat[pair_idx] - action_flat[anchor_idx]
    target_response = future_tokens[pair_idx] - future_tokens[anchor_idx]
    return state_tokens[anchor_idx], action_flat[anchor_idx], delta_action, target_response


def response_metrics(pred_tokens: torch.Tensor, target_tokens: torch.Tensor, top_frac: float = 0.1) -> dict[str, float]:
    pred = pred_tokens.float()
    target = target_tokens.float()
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    mse = F.mse_loss(pred_flat, target_flat)
    zero_mse = target_flat.pow(2).mean().clamp(min=1e-12)
    cos = F.cosine_similarity(pred_flat, target_flat, dim=-1).mean()

    pred_energy = pred.pow(2).sum(dim=-1)
    target_energy = target.pow(2).sum(dim=-1)
    k = max(1, int(target_energy.shape[1] * float(top_frac)))
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

    return {
        "loss_response_rel_mse": float((mse / zero_mse).detach().item()),
        "response_cos": float(cos.detach().item()),
        "response_top10_iou": float((inter / union).mean().detach().item()),
        "response_pointing10": float(pointing.detach().item()),
    }
