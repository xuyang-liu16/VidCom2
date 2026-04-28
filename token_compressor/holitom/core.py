from __future__ import annotations

import os
import torch

from .holitom import holitom_compression


def compute_keep_indices(
    flat_features: torch.Tensor,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    base_scale: float,
) -> torch.Tensor:
    t, h, w = [int(x) for x in grid_thw.tolist()]
    frame_tokens = (h * w) // (spatial_merge_size ** 2)
    total = int(flat_features.shape[0])
    if t <= 0 or frame_tokens <= 0 or total == 0 or total != t * frame_tokens:
        return torch.arange(total, device=flat_features.device, dtype=torch.long)

    tau = float(os.getenv("HOLITOM_T", "0.8"))
    beta = float(os.getenv("HOLITOM_BETA", "0.6"))
    d_ratio = float(os.getenv("HOLITOM_D", "0.0"))
    k_neighbors = int(os.getenv("HOLITOM_K", "7"))
    max_window_size = int(os.getenv("HOLITOM_MAX_WINDOW_SIZE", "1024"))
    retention_ratio = float(max(0.0, min(1.0, base_scale)))

    frames = flat_features.view(t, frame_tokens, -1)
    attn = frames.norm(dim=-1)
    _, keep = holitom_compression(
        video_feat=frames,
        attn_weights=attn,
        retain_ratio=retention_ratio,
        tau=tau,
        beta=beta,
        D=d_ratio,
        K=max(1, k_neighbors),
        max_window_size=max(1, max_window_size),
    )

    keep = keep.to(dtype=torch.long, device=flat_features.device)
    keep = keep[(keep >= 0) & (keep < total)]
    if keep.numel() == 0:
        keep = torch.tensor([0], device=flat_features.device, dtype=torch.long)
    keep = torch.unique(keep, sorted=True)
    return keep

