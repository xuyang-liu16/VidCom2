from __future__ import annotations

import os
import torch

from .temporal_token_merging import dycoke_ttm_with_indices


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

    # Keep `R_RATIO` semantics aligned with other compressors: it denotes retention ratio.
    retention_ratio = float(max(0.0, min(1.0, base_scale)))
    default_merging_ratio = 1.0 - retention_ratio
    merging_ratio = float(os.getenv("DYCOKE_K", str(default_merging_ratio)))
    merging_ratio = float(max(0.0, min(1.0, merging_ratio)))
    num_tokens_per_frame = int(os.getenv("DYCOKE_NUM_TOKENS_PER_FRAME", "-1"))
    chosen_tpf = num_tokens_per_frame if num_tokens_per_frame > 0 else frame_tokens

    if chosen_tpf <= 0 or total != t * chosen_tpf:
        chosen_tpf = frame_tokens
    if total != t * chosen_tpf:
        return torch.arange(total, device=flat_features.device, dtype=torch.long)

    feat_3d = flat_features.view(t, chosen_tpf, -1)
    _, keep = dycoke_ttm_with_indices(
        image_feature=feat_3d,
        num_tokens_per_frame=chosen_tpf,
        merging_ratio=merging_ratio,
    )
    keep = keep.to(dtype=torch.long, device=flat_features.device)
    keep = keep[(keep >= 0) & (keep < total)]
    if keep.numel() == 0:
        keep = torch.tensor([0], device=flat_features.device, dtype=torch.long)
    keep = torch.unique(keep, sorted=True)
    return keep
