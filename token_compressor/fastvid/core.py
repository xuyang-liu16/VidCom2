from __future__ import annotations

import os
import torch

from .fastvid_algo import fastvid_compression


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

    importance_scores = flat_features.float().norm(dim=-1)
    cfg = {
        "retention_ratio": float(os.getenv("fastvid_retention_ratio", str(base_scale))),
        "dyseg_c": int(os.getenv("fastvid_DySeg_c", "8")),
        "dyseg_tau": float(os.getenv("fastvid_DySeg_tau", "0.84")),
        "dtm_p": int(os.getenv("fastvid_DTM_p", "4")),
    }
    grid_batch = grid_thw.view(1, 3).to(device=flat_features.device)
    keep = fastvid_compression(
        video_embeds=flat_features,
        importance_scores=importance_scores,
        grid_thw=grid_batch,
        config_args=cfg,
    )
    keep = keep.to(dtype=torch.long, device=flat_features.device)
    keep = keep[(keep >= 0) & (keep < total)]
    if keep.numel() == 0:
        keep = torch.tensor([0], device=flat_features.device, dtype=torch.long)
    keep = torch.unique(keep, sorted=True)
    return keep

