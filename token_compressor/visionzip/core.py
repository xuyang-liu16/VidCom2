from __future__ import annotations

import os
import torch

from .visionzip import visionzip_compression


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

    dominant_ratio = float(os.getenv("visionzip_dominant_ratio", "0.6"))
    k_neighbors = int(os.getenv("visionzip_k_neighbors", "5"))
    retention_ratio = float(max(0.0, min(1.0, base_scale)))

    frames = flat_features.view(t, frame_tokens, -1)
    attention_scores = frames.norm(dim=-1)
    _, local_keep = visionzip_compression(
        features=frames,
        attention_scores=attention_scores,
        retention_ratio=retention_ratio,
        dominant_ratio=dominant_ratio,
        cls_token=False,
        k_neighbors=max(1, k_neighbors),
    )

    keep_list = []
    for fi in range(t):
        idx = local_keep[fi].to(dtype=torch.long, device=flat_features.device)
        idx = idx[(idx >= 0) & (idx < frame_tokens)]
        if idx.numel() == 0:
            idx = torch.tensor([0], device=flat_features.device, dtype=torch.long)
        idx = torch.unique(idx, sorted=True)
        keep_list.append(idx + fi * frame_tokens)

    keep = torch.cat(keep_list, dim=0).to(dtype=torch.long)
    keep = torch.unique(keep, sorted=True)
    return keep

