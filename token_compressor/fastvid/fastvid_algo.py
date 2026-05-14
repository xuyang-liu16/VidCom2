from __future__ import annotations

import torch
import torch.nn.functional as F


def _segment_sizes_from_cuts(num_frames: int, cut_indices: torch.Tensor) -> list[int]:
    cut_indices = cut_indices.to(dtype=torch.long)
    cut_indices = cut_indices[(cut_indices >= 0) & (cut_indices < num_frames - 1)]
    cut_indices = torch.unique(cut_indices, sorted=True)
    if cut_indices.numel() == 0:
        return [num_frames]

    sizes = [int(cut_indices[0].item()) + 1]
    for i in range(1, int(cut_indices.numel())):
        sizes.append(int(cut_indices[i].item() - cut_indices[i - 1].item()))
    sizes.append(int(num_frames - cut_indices[-1].item() - 1))
    return [size for size in sizes if size > 0]


def _dpc_knn_centers(x: torch.Tensor, cluster_num: int, k: int = 4) -> torch.Tensor:
    if cluster_num <= 0 or x.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=x.device)
    cluster_num = min(int(cluster_num), int(x.shape[0]))
    k = max(1, min(int(k), int(x.shape[0])))

    x_batched = x.unsqueeze(0)
    dist_matrix = torch.cdist(x_batched.float(), x_batched.float()) / (x.shape[-1] ** 0.5)
    dist_nearest, _ = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
    density = (-(dist_nearest**2).mean(dim=-1)).exp()
    density = density + torch.rand(density.shape, device=x.device, dtype=density.dtype) * 1e-6

    mask = (density[:, None, :] > density[:, :, None]).type(x_batched.dtype)
    dist_max = dist_matrix.flatten(1).max(dim=-1).values[:, None, None]
    dist, _ = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)
    score = dist * density
    centers = torch.topk(score, k=cluster_num, dim=-1).indices[0]
    return centers.sort().values


def _allocate_inner_budgets(
    segment_sizes: list[int],
    frame_tokens: int,
    seg_salient_num: int,
    seg_context_num: int,
    dtm_p: int,
    stprune_d: float,
) -> tuple[list[int], list[int]]:
    valid_seg_len = len(segment_sizes)
    if valid_seg_len == 0:
        return [], []

    salient_base = seg_salient_num // valid_seg_len
    salient_rem = seg_salient_num % valid_seg_len
    inner_salient = [salient_base + (1 if i < salient_rem else 0) for i in range(valid_seg_len)]

    temp_num = (valid_seg_len + dtm_p - 1) // dtm_p
    context_base = seg_context_num // max(1, temp_num)
    context_rem = seg_context_num % max(1, temp_num)
    temp_context = [context_base + (1 if i < context_rem else 0) for i in range(max(1, temp_num))]

    context_by_inner = []
    tmp_ctx_idx = 0
    for inner_idx in range(valid_seg_len):
        if inner_idx % dtm_p == 0 and tmp_ctx_idx < len(temp_context):
            ctx = min(temp_context[tmp_ctx_idx], frame_tokens // 2)
            context_by_inner.append(ctx)
            tmp_ctx_idx += 1
        else:
            context_by_inner.append(0)
    context_by_inner.reverse()

    frame_salient: list[int] = []
    frame_context: list[int] = []
    for inner_idx, inner_size in enumerate(segment_sizes):
        frame_salient.extend([0] * (inner_size - 1))
        frame_context.extend([0] * (inner_size - 1))

        ctx = context_by_inner[inner_idx]
        if ctx == 0:
            frame_salient.append(min(inner_salient[inner_idx], frame_tokens))
            frame_context.append(0)
        elif ctx > 0:
            ctx = min(ctx, frame_tokens // 2)
            frame_context.append(ctx)
            frame_salient.append(min(inner_salient[inner_idx], frame_tokens - ctx))
        else:
            frame_salient.append(frame_tokens - int(frame_tokens * stprune_d))
            frame_context.append(int(frame_tokens * stprune_d))

    return frame_salient, frame_context


def fastvid_compression(
    video_embeds: torch.Tensor,
    importance_scores: torch.Tensor,
    grid_thw: torch.Tensor,
    config_args: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FastVID-style DySeg + STPrune + DTM compression.

    The official FastVID Qwen2.5-VL path uses attention weights from the visual
    encoder. This implementation keeps the same segmentation, salient/context budget,
    DPC center selection, and contextual token merge steps, while accepting a
    generic per-token importance score so it can plug into this repository's
    Qwen-VL patching style.
    """
    device = video_embeds.device
    if grid_thw is None or grid_thw.numel() == 0:
        keep = torch.arange(video_embeds.shape[0], device=device, dtype=torch.long)
        return video_embeds, keep
    if grid_thw.ndim != 2 or grid_thw.shape[0] != 1:
        keep = torch.arange(video_embeds.shape[0], device=device, dtype=torch.long)
        return video_embeds, keep

    t = int(grid_thw[0, 0].item())
    total_tokens, dim = video_embeds.shape
    if t <= 0 or total_tokens == 0 or total_tokens % t != 0:
        keep = torch.arange(total_tokens, device=device, dtype=torch.long)
        return video_embeds, keep

    retention_ratio = max(0.0, min(1.0, float(config_args.get("retention_ratio", 0.27))))
    dyseg_c = max(1, int(config_args.get("dyseg_c", 8)))
    dyseg_tau = float(config_args.get("dyseg_tau", 0.84))
    dyseg_ignore = float(config_args.get("dyseg_ignore", 0.99))
    stprune_d = max(0.0, min(1.0, float(config_args.get("stprune_d", 0.5))))
    dtm_p = max(1, int(config_args.get("dtm_p", 4)))
    dtm_beta = max(0.0, min(1.0, float(config_args.get("dtm_beta", 0.5))))

    frame_tokens = total_tokens // t
    frames = video_embeds.view(t, frame_tokens, dim)
    scores = importance_scores.view(t, frame_tokens)

    if retention_ratio >= 1.0:
        keep = torch.arange(total_tokens, device=device, dtype=torch.long)
        return video_embeds, keep

    frame_global = F.normalize(frames.mean(dim=1).float(), dim=-1)
    sim = (frame_global[:-1] * frame_global[1:]).sum(dim=1)
    k_val = min(dyseg_c - 1, max(0, t - 1))
    low_sim = torch.topk(sim, k=k_val, largest=False).indices if k_val > 0 else sim.new_empty(0, dtype=torch.long)
    tau_cuts = torch.nonzero(sim < dyseg_tau, as_tuple=False).squeeze(-1)
    segment_sizes = _segment_sizes_from_cuts(t, torch.cat([low_sim, tau_cuts], dim=0))

    frame_retain_num = max(1, min(frame_tokens, int(frame_tokens * retention_ratio)))
    compressed_segments: list[torch.Tensor] = []
    kept_segments: list[torch.Tensor] = []
    frame_offset = 0

    for seg_size in segment_sizes:
        seg_frames = frames[frame_offset : frame_offset + seg_size]
        seg_scores = scores[frame_offset : frame_offset + seg_size]
        seg_global = frame_global[frame_offset : frame_offset + seg_size]
        seg_len = int(seg_frames.shape[0])

        seg_retain_num = max(1, min(seg_len * frame_tokens, frame_retain_num * seg_len))
        seg_context_num = max(1, int(seg_retain_num * stprune_d)) if seg_retain_num > 1 else 0
        seg_context_num = min(seg_context_num, seg_retain_num)
        seg_salient_num = seg_retain_num - seg_context_num

        if seg_len == 1:
            frame_salient_num = [seg_salient_num]
            frame_context_num = [seg_context_num]
        else:
            inner_sim = (seg_global[:-1] * seg_global[1:]).sum(dim=1)
            inner_cuts = torch.nonzero(inner_sim < dyseg_ignore, as_tuple=False).squeeze(-1)
            inner_segment_sizes = _segment_sizes_from_cuts(seg_len, inner_cuts)
            frame_salient_num, frame_context_num = _allocate_inner_budgets(
                inner_segment_sizes,
                frame_tokens,
                seg_salient_num,
                seg_context_num,
                dtm_p,
                stprune_d,
            )

        salient_indices: list[torch.Tensor] = []
        context_indices: list[torch.Tensor] = []
        for frame_i in range(seg_len):
            sal_num = min(max(0, int(frame_salient_num[frame_i])), frame_tokens)
            ctx_num = min(max(0, int(frame_context_num[frame_i])), frame_tokens - sal_num)

            top_indices = None
            if sal_num > 0:
                top_indices = torch.topk(seg_scores[frame_i], k=sal_num, largest=True, sorted=False).indices
                salient_indices.append(top_indices + frame_i * frame_tokens)

            if ctx_num > 0:
                all_frame_indices = torch.arange(frame_tokens, device=device)
                if top_indices is not None:
                    remaining = all_frame_indices[~torch.isin(all_frame_indices, top_indices)]
                else:
                    remaining = all_frame_indices
                if remaining.numel() > 0:
                    centers = _dpc_knn_centers(seg_frames[frame_i, remaining], ctx_num, k=4)
                    context_indices.append(remaining[centers] + frame_i * frame_tokens)

        if not salient_indices and not context_indices:
            fallback = torch.topk(seg_scores.flatten(), k=seg_retain_num, largest=True, sorted=False).indices
            cur_indices = fallback.sort().values
            compressed_segments.append(seg_frames.reshape(seg_len * frame_tokens, dim)[cur_indices])
            kept_segments.append(cur_indices + frame_offset * frame_tokens)
            frame_offset += seg_size
            continue

        cur_flat = seg_frames.reshape(seg_len * frame_tokens, dim)
        cur_all_indices = torch.arange(seg_len * frame_tokens, device=device)
        cur_salient = torch.cat(salient_indices) if salient_indices else cur_all_indices.new_empty(0)
        cur_context = torch.cat(context_indices) if context_indices else cur_all_indices.new_empty(0)
        cur_salient_hidden = cur_flat[cur_salient] if cur_salient.numel() > 0 else cur_flat.new_empty((0, dim))

        cur_context_hidden_states: list[torch.Tensor] = []
        for tmp_context in context_indices:
            retained = torch.cat([cur_salient, tmp_context])
            merge_indices = cur_all_indices[~torch.isin(cur_all_indices, retained)]
            if merge_indices.numel() == 0 or tmp_context.numel() == 0:
                cur_context_hidden_states.append(cur_flat[tmp_context])
                continue

            norm_tokens = F.normalize(cur_flat.float(), dim=-1)
            target_tokens = norm_tokens[tmp_context]
            merge_tokens = norm_tokens[merge_indices]
            similarity = torch.mm(merge_tokens, target_tokens.T)
            assign = torch.zeros(merge_tokens.shape[0], tmp_context.shape[0], dtype=cur_flat.dtype, device=device)
            assign.scatter_(1, similarity.argmax(dim=1).unsqueeze(-1), 1)

            counts = assign.sum(dim=0).clamp(min=1).unsqueeze(-1)
            avg_weights = (1 / (assign.sum(dim=0).unsqueeze(-1) + 1)).clamp(min=dtm_beta)
            aggregated = torch.mm(assign.T, cur_flat[merge_indices]) / counts
            targets = cur_flat[tmp_context]
            cur_context_hidden_states.append(avg_weights * targets + (1 - avg_weights) * aggregated)

        cur_context_hidden = (
            torch.cat(cur_context_hidden_states, dim=0) if cur_context_hidden_states else cur_flat.new_empty((0, dim))
        )
        cur_index_combined = torch.cat([cur_salient, cur_context])
        cur_hidden_combined = torch.cat([cur_salient_hidden, cur_context_hidden], dim=0)

        sorted_indices = torch.argsort(cur_index_combined)
        cur_index_combined = cur_index_combined[sorted_indices]
        cur_hidden_combined = cur_hidden_combined[sorted_indices]
        compressed_segments.append(cur_hidden_combined)
        kept_segments.append(cur_index_combined + frame_offset * frame_tokens)
        frame_offset += seg_size

    if not compressed_segments:
        keep = torch.arange(total_tokens, device=device, dtype=torch.long)
        return video_embeds, keep

    compressed = torch.cat(compressed_segments, dim=0)
    keep = torch.cat(kept_segments, dim=0).to(dtype=torch.long)
    order = torch.argsort(keep)
    return compressed[order], keep[order]
