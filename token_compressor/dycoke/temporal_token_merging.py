"""
DyCoke temporal token merging.

Adapted from:
https://github.com/KD-TAO/DyCoke
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _lowest_similarity_indices(similarity: torch.Tensor, num_keep: int) -> torch.Tensor:
    if num_keep <= 0:
        return torch.empty((0,), device=similarity.device, dtype=torch.long)
    num_keep = min(int(num_keep), int(similarity.numel()))
    if num_keep <= 0:
        return torch.empty((0,), device=similarity.device, dtype=torch.long)
    return similarity.topk(num_keep, largest=False).indices


def dycoke_ttm_with_indices(
    image_feature: torch.Tensor,
    num_tokens_per_frame: int = 256,
    merging_ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Temporal token merging (LLaVA-style) with returned kept indices in flattened order.

    Notes:
    - Logic follows DyCoke-main `llava_arch.py::dycole_ttm` closely.
    - The odd-frame fallback used in our previous adaptation is intentionally removed
      to match LLaVA behavior.
    """
    if image_feature.ndim != 3:
        raise ValueError(f"Expected image_feature to be 3D, got shape={tuple(image_feature.shape)}")

    num_frames = int(image_feature.shape[0])
    if num_frames <= 0:
        empty = image_feature.new_zeros((0, image_feature.shape[-1]))
        return empty, torch.empty((0,), dtype=torch.long, device=image_feature.device)

    if num_tokens_per_frame <= 0:
        num_tokens_per_frame = int(image_feature.shape[1])
    if image_feature.shape[1] != num_tokens_per_frame:
        raise ValueError(
            f"num_tokens_per_frame mismatch: expected {num_tokens_per_frame}, got {image_feature.shape[1]}"
        )

    keep_ratio = 1.0 - float(merging_ratio)
    keep_ratio = min(max(keep_ratio, 0.0), 1.0)

    flat = image_feature.reshape(num_frames * num_tokens_per_frame, -1)

    if num_frames == 1:
        indices = torch.arange(flat.shape[0], device=flat.device, dtype=torch.long)
        return flat, indices

    # Calculate similarities between adjacent even frames.
    similarities = []
    for i in range(0, num_frames - 1, 2):
        frame1_tokens = flat[i * num_tokens_per_frame : (i + 1) * num_tokens_per_frame]
        frame2_tokens = flat[(i + 1) * num_tokens_per_frame : (i + 2) * num_tokens_per_frame]
        frame1_norm = F.normalize(frame1_tokens, p=2, dim=1)
        frame2_norm = F.normalize(frame2_tokens, p=2, dim=1)
        similarity = F.cosine_similarity(frame1_norm, frame2_norm, dim=1)
        similarities.append(similarity)

    if similarities:
        similarities = torch.stack(similarities)
    else:
        similarities = flat.new_zeros((0, num_tokens_per_frame))

    # Process even frames.
    modified_tokens: list[torch.Tensor] = []
    modified_indices: list[torch.Tensor] = []
    for i in range(0, num_frames - 1, 2):
        frame1_tokens = flat[i * num_tokens_per_frame : (i + 1) * num_tokens_per_frame]
        frame2_tokens = flat[(i + 1) * num_tokens_per_frame : (i + 2) * num_tokens_per_frame]

        num_keep = int(keep_ratio * num_tokens_per_frame)
        tokens_to_keep = _lowest_similarity_indices(similarities[i // 2], num_keep)

        frame1_idx = torch.arange(
            i * num_tokens_per_frame,
            (i + 1) * num_tokens_per_frame,
            device=flat.device,
            dtype=torch.long,
        )
        frame2_idx = tokens_to_keep + (i + 1) * num_tokens_per_frame

        modified_tokens.append(frame1_tokens)
        modified_indices.append(frame1_idx)
        modified_tokens.append(frame2_tokens[tokens_to_keep])
        modified_indices.append(frame2_idx)

    # Process odd frames (same indexing pattern as LLaVA dycole_ttm).
    odd_similarities = []
    for i in range(0, num_frames - 4, 4):
        frame1_tokens = flat[i * num_tokens_per_frame : (i + 1) * num_tokens_per_frame]
        frame2_tokens = flat[(i + 2) * num_tokens_per_frame : (i + 3) * num_tokens_per_frame]
        similarity = F.cosine_similarity(frame1_tokens, frame2_tokens, dim=1)
        odd_similarities.append(similarity)

    if odd_similarities:
        odd_similarities = torch.stack(odd_similarities)
    else:
        odd_similarities = flat.new_zeros((0, num_tokens_per_frame))

    for i in range(0, num_frames - 4, 4):
        frame1_tokens = flat[i * num_tokens_per_frame : (i + 1) * num_tokens_per_frame]
        frame2_tokens = flat[(i + 2) * num_tokens_per_frame : (i + 3) * num_tokens_per_frame]

        num_keep = int(keep_ratio * num_tokens_per_frame)
        tokens_to_keep = _lowest_similarity_indices(odd_similarities[i // 4], num_keep)

        frame1_idx = torch.arange(
            i * num_tokens_per_frame,
            (i + 1) * num_tokens_per_frame,
            device=flat.device,
            dtype=torch.long,
        )
        frame2_idx = tokens_to_keep + (i + 2) * num_tokens_per_frame

        # Keep list index updates aligned with dycole_ttm behavior.
        if i < len(modified_tokens):
            modified_tokens[i] = frame1_tokens
            modified_indices[i] = frame1_idx
        if i + 2 < len(modified_tokens):
            modified_tokens[i + 2] = frame2_tokens[tokens_to_keep]
            modified_indices[i + 2] = frame2_idx

    if not modified_tokens:
        indices = torch.arange(flat.shape[0], device=flat.device, dtype=torch.long)
        return flat, indices

    combined_tokens = torch.cat(modified_tokens, dim=0)
    combined_indices = torch.cat(modified_indices, dim=0).to(torch.long)
    return combined_tokens, combined_indices


def dycoke_ttm(
    image_feature: torch.Tensor,
    num_tokens_per_frame: int = 256,
    merging_ratio: float = 0.5,
) -> torch.Tensor:
    combined_tokens, _ = dycoke_ttm_with_indices(
        image_feature=image_feature,
        num_tokens_per_frame=num_tokens_per_frame,
        merging_ratio=merging_ratio,
    )
    return combined_tokens
