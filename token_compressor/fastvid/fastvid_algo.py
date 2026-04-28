from __future__ import annotations

import torch
import torch.nn.functional as F


def fastvid_compression(
    video_embeds: torch.Tensor,
    importance_scores: torch.Tensor,
    grid_thw: torch.Tensor,
    config_args: dict,
) -> torch.Tensor:
    """
    Lightweight FastVID-style frame-segment budget allocation.
    Returns kept global token indices.
    """
    device = video_embeds.device
    if grid_thw is None or grid_thw.numel() == 0:
        return torch.arange(video_embeds.shape[0], device=device, dtype=torch.long)
    if grid_thw.ndim != 2 or grid_thw.shape[0] != 1:
        return torch.arange(video_embeds.shape[0], device=device, dtype=torch.long)

    t = int(grid_thw[0, 0].item())
    total_tokens, dim = video_embeds.shape
    if t <= 0 or total_tokens == 0 or total_tokens % t != 0:
        return torch.arange(total_tokens, device=device, dtype=torch.long)

    retention_ratio = float(config_args.get("retention_ratio", 0.27))
    dyseg_c = int(config_args.get("dyseg_c", 8))
    dyseg_tau = float(config_args.get("dyseg_tau", 0.84))
    dtm_p = int(config_args.get("dtm_p", 4))

    retention_ratio = max(0.0, min(1.0, retention_ratio))
    dtm_p = max(1, dtm_p)

    tokens_per_frame = total_tokens // t
    frames = video_embeds.view(t, tokens_per_frame, dim)
    score_frames = importance_scores.view(t, tokens_per_frame)

    if t == 1:
        k = max(1, min(tokens_per_frame, int(round(tokens_per_frame * retention_ratio))))
        idx = torch.topk(score_frames[0], k=k, largest=True, sorted=False).indices
        return torch.sort(idx)[0].to(dtype=torch.long)

    # Dynamic segmentation by frame-level similarity.
    frame_global = F.normalize(frames.mean(dim=1), dim=-1)
    sim = (frame_global[:-1] * frame_global[1:]).sum(dim=1)
    k_val = min(max(0, dyseg_c - 1), max(0, t - 2))
    if k_val > 0:
        low_sim = torch.topk(sim, k_val, largest=False).indices
        tau_cuts = torch.nonzero(sim < dyseg_tau, as_tuple=False).squeeze(-1)
        cut_indices = torch.unique(torch.cat([low_sim, tau_cuts], dim=0)).sort().values
        if cut_indices.numel() > 0:
            segment_sizes = [int(cut_indices[0].item()) + 1]
            for i in range(1, int(cut_indices.numel())):
                segment_sizes.append(int(cut_indices[i].item() - cut_indices[i - 1].item()))
            segment_sizes.append(int(t - cut_indices[-1].item() - 1))
        else:
            segment_sizes = [t]
    else:
        segment_sizes = [t]

    keep_indices = []
    frame_offset = 0
    for seg_size in segment_sizes:
        seg_scores = score_frames[frame_offset : frame_offset + seg_size]
        # Segment budget.
        seg_total = seg_size * tokens_per_frame
        seg_keep_total = max(1, min(seg_total, int(round(seg_total * retention_ratio))))

        # Per-frame budget allocation by frame salience.
        frame_salience = seg_scores.mean(dim=1)
        frame_weights = torch.softmax(frame_salience, dim=0)
        frame_keep = torch.clamp((frame_weights * seg_keep_total).round().to(torch.long), min=1)
        # Fix budget drift.
        diff = int(seg_keep_total - int(frame_keep.sum().item()))
        if diff != 0:
            order = torch.argsort(frame_salience, descending=(diff > 0))
            i = 0
            while diff != 0 and i < order.numel() * 3:
                fi = int(order[i % order.numel()].item())
                if diff > 0:
                    frame_keep[fi] += 1
                    diff -= 1
                else:
                    if frame_keep[fi] > 1:
                        frame_keep[fi] -= 1
                        diff += 1
                i += 1

        for local_f in range(seg_size):
            global_f = frame_offset + local_f
            k = int(max(1, min(tokens_per_frame, int(frame_keep[local_f].item()))))
            if dtm_p > 1 and (local_f % dtm_p) != 0:
                # non-key frame: keep at least one but slightly fewer tokens
                k = max(1, int(round(k * 0.7)))
            top = torch.topk(seg_scores[local_f], k=k, largest=True, sorted=False).indices
            keep_indices.append(top + global_f * tokens_per_frame)

        frame_offset += seg_size

    if not keep_indices:
        return torch.arange(total_tokens, device=device, dtype=torch.long)
    keep = torch.cat(keep_indices, dim=0).to(dtype=torch.long)
    keep = torch.unique(keep, sorted=True)
    return keep

