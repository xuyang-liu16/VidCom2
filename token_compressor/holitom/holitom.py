"""
HoliTom: Holistic Token Merging for Fast Video Large Language Models

This implementation is based on the paper:
"HoliTom: Holistic Token Merging for Fast Video Large Language Models"
NeurIPS 2025

Key idea: Holistic token merging via temporal segmentation and DPC-KNN clustering.
- Outer-LLM pruning: Global redundancy-aware temporal segmentation + spatial-temporal merging
- Inner-LLM pruning (optional): Token similarity-based merging at layer K

Reference: https://github.com/cokeshao/HoliTom
"""

import math
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


def cluster_dpc_knn(
    x: torch.Tensor,
    cluster_num: int,
    k: int = 7
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Density Peak Clustering with KNN.
    
    Args:
        x: Token features [batch_size, seq_len, embed_dim]
        cluster_num: Number of cluster centers to select
        k: Number of nearest neighbors for density estimation
        
    Returns:
        index_center: Indices of cluster centers [batch_size, cluster_num]
        dist_matrix: Pairwise distance matrix [batch_size, seq_len, seq_len]
    """
    with torch.no_grad():
        batch_size, seq_len, embed_dim = x.shape
        
        # Compute pairwise distance matrix (normalized by sqrt(embed_dim))
        dist_matrix = torch.cdist(x.float(), x.float()) / (embed_dim ** 0.5)
        
        # Get local density using k-nearest neighbors
        dist_nearest, _ = torch.topk(dist_matrix, k, dim=-1, largest=False)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()  # [batch_size, seq_len]
        
        # Add small noise to ensure unique densities
        density = density + torch.rand(
            density.shape, device=density.device, dtype=density.dtype
        ) * 1e-6
        
        # Get distance indicator: for each token, find distance to nearest higher-density token
        mask = (density[:, None, :] > density[:, :, None]).type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1).values[:, None, None]
        dist, _ = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)
        
        # Select cluster centers based on score = distance * density
        score = dist * density
        _, index_center = score.topk(cluster_num, dim=-1)
        
    return index_center, dist_matrix


def select_static_windows(
    feature_sim: torch.Tensor,
    batch_size: int,
    tau: float,
    max_window_size: int
) -> Tuple[List[Tuple[int, int]], float]:
    """
    Select static windows based on frame similarity using dynamic programming.
    
    Args:
        feature_sim: Frame-to-frame similarity [num_frames-1, seq_len]
        batch_size: Number of frames
        tau: Similarity threshold for considering frames as static
        max_window_size: Maximum window size for segmentation
        
    Returns:
        selected_frames: List of (start, end) tuples for each window
        total_reduced: Total number of redundant features that can be pruned
    """
    def get_pruned_static_count_vectorized(feature_sim, batch_size, tau):
        """Compute pruned static count matrix."""
        similarity_matrix = torch.ones(
            (batch_size, batch_size, feature_sim.shape[1]), 
            device=feature_sim.device
        )
        for start in range(batch_size - 1):
            cum_similarity = torch.cumprod(feature_sim[start:] > tau, dim=0)
            end_idx = start + 1 + len(cum_similarity)
            similarity_matrix[start, start+1:end_idx] = cum_similarity
        
        window_lengths = torch.arange(batch_size, device=feature_sim.device).unsqueeze(0) - \
                        torch.arange(batch_size, device=feature_sim.device).unsqueeze(1)
        window_lengths = window_lengths.clamp(min=0)
        pruned_static_count = (similarity_matrix.sum(dim=-1) * window_lengths).float()
        return pruned_static_count
    
    pruned_static_count = get_pruned_static_count_vectorized(feature_sim, batch_size, tau)
    
    # Dynamic programming to find optimal segmentation
    dp = torch.zeros(batch_size, device=pruned_static_count.device)
    prev = torch.zeros(batch_size, dtype=torch.long, device=pruned_static_count.device)
    
    for i in range(batch_size):
        max_val = dp[i-1] if i > 0 else 0
        best_j = i
        
        for window_size in range(2, min(i + 1, max_window_size) + 1):
            j = i - window_size
            current_val = (dp[j] if j >= 0 else 0) + pruned_static_count[j+1, i]
            if current_val > max_val:
                max_val = current_val
                best_j = j + 1
        
        dp[i] = max_val
        prev[i] = best_j
    
    # Backtrack to find selected frames
    selected_frames = []
    i = batch_size - 1
    while i >= 0:
        selected_frames.append((prev[i].item(), i))
        i = prev[i].item() - 1
    selected_frames = selected_frames[::-1]
    
    total_reduced = dp[-1].item()
    return selected_frames, total_reduced


def get_static_dynamic_features(
    image_feat: torch.Tensor,
    attn_weights: torch.Tensor,
    selected_frames: List[Tuple[int, int]],
    feature_sim: torch.Tensor,
    tau: float
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Separate features into static and dynamic components based on temporal windows.
    
    Args:
        image_feat: Image features [num_frames, seq_len, embed_dim]
        attn_weights: Attention weights [num_frames, seq_len]
        selected_frames: List of (start, end) tuples for windows
        feature_sim: Frame similarity matrix [num_frames-1, seq_len]
        tau: Similarity threshold
        
    Returns:
        static_feat_list: Static features for each window
        dynamic_feat_list: Dynamic features for each window
        dynamic_attn_list: Attention weights for dynamic features
        static_pos_list: Position indices for static features
        dynamic_pos_list: Position indices for dynamic features
    """
    batch_size, seq_len, embed_dim = image_feat.shape
    
    static_feat_list = []
    dynamic_feat_list = []
    dynamic_attn_list = []
    static_pos_list = []
    dynamic_pos_list = []
    
    for start, end in selected_frames:
        all_indices = torch.arange(seq_len, device=image_feat.device).unsqueeze(0)
        
        if start == end:
            # Single frame window - all tokens are dynamic
            static_feat_list.append(torch.empty((0, embed_dim), device=image_feat.device))
            dynamic_feat_list.append(image_feat[start:end+1])
            dynamic_attn_list.append(attn_weights[start:end+1])
            static_pos_list.append(torch.empty((0, seq_len), device=image_feat.device))
            dynamic_pos_list.append(all_indices)
        else:
            # Multi-frame window - separate static and dynamic
            window_size = end - start + 1
            
            # Tokens are static if similarity > tau across all frames in window
            mask = torch.all(feature_sim[start:end, :] > tau, dim=0)
            
            static_feat = image_feat[start:end+1, mask]
            dynamic_feat = image_feat[start:end+1, ~mask]
            dynamic_attn = attn_weights[start:end+1, ~mask]
            
            # Static features are averaged across frames
            static_feat_list.append(static_feat.mean(dim=0))
            dynamic_feat_list.append(dynamic_feat)
            dynamic_attn_list.append(dynamic_attn)
            static_pos_list.append(all_indices[:, mask].expand(1, -1))
            dynamic_pos_list.append(all_indices[:, ~mask].expand(window_size, -1))
    
    return static_feat_list, dynamic_feat_list, dynamic_attn_list, static_pos_list, dynamic_pos_list


def merge_tokens_by_clustering(
    feat: torch.Tensor,
    target_indices: torch.Tensor,
    dist_matrix: torch.Tensor,
    cluster_num: int,
    beta: float
) -> torch.Tensor:
    """
    Merge tokens by assigning non-center tokens to nearest cluster center.
    
    Args:
        feat: Token features [batch_size, seq_len, embed_dim]
        target_indices: Indices of cluster centers [batch_size, cluster_num]
        dist_matrix: Distance matrix [batch_size, seq_len, seq_len]
        cluster_num: Number of clusters
        beta: Weight for center token in merging (beta * center + (1-beta) * mean)
        
    Returns:
        cluster_tokens: Merged cluster tokens [batch_size, cluster_num, embed_dim]
    """
    batch_size, seq_len, embed_dim = feat.shape
    
    all_indices = torch.arange(seq_len, device=feat.device)
    all_indices = all_indices.unsqueeze(0).expand(batch_size, -1)
    
    # Get non-center token indices
    non_target_indices = torch.zeros(
        (batch_size, seq_len - cluster_num), 
        dtype=torch.long, 
        device=feat.device
    )
    for b in range(batch_size):
        non_target_mask = ~torch.isin(all_indices[b], target_indices[b])
        non_target_indices[b] = all_indices[b][non_target_mask]
    
    # Get non-center features
    non_target_feat = torch.gather(
        feat, dim=1,
        index=non_target_indices.unsqueeze(-1).expand(-1, -1, feat.size(-1))
    )
    
    # Get distance from non-center to center tokens
    dist_to_centers = torch.gather(
        dist_matrix, dim=1,
        index=non_target_indices.unsqueeze(-1).expand(-1, -1, dist_matrix.size(-1))
    )
    dist_to_centers = torch.gather(
        dist_to_centers, dim=2,
        index=target_indices.unsqueeze(1).expand(-1, dist_to_centers.size(1), -1)
    )
    
    # Assign each non-center token to nearest center
    idx_cluster = torch.argmin(dist_to_centers, dim=-1)
    
    # Merge tokens
    cluster_tokens = []
    for b in range(batch_size):
        batch_tokens = []
        for i in range(cluster_num):
            mask = (idx_cluster[b] == i)
            if mask.any():
                cluster_features = non_target_feat[b][mask]
                cluster_mean = cluster_features.mean(dim=0)
                # Weighted combination of center and cluster mean
                merged = beta * feat[b][target_indices[b][i]] + (1 - beta) * cluster_mean
                batch_tokens.append(merged)
            else:
                batch_tokens.append(feat[b][target_indices[b][i]])
        cluster_tokens.append(torch.stack(batch_tokens))
    
    cluster_tokens = torch.stack(cluster_tokens)
    return cluster_tokens


def merge_tokens_by_attention_density(
    feat: torch.Tensor,
    attn: torch.Tensor,
    pos: torch.Tensor,
    retain_ratio: float,
    D: float,
    beta: float,
    K: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merge tokens using attention-based selection and density clustering.
    
    Args:
        feat: Token features [batch_size, seq_len, embed_dim]
        attn: Attention weights [batch_size, seq_len]
        pos: Position indices [batch_size, seq_len]
        retain_ratio: Ratio of tokens to retain
        D: Ratio of dominant tokens (selected purely by attention)
        beta: Weight for merging
        K: Number of neighbors for clustering
        
    Returns:
        image_feat: Merged features [batch_size, num_keep, embed_dim]
        image_pos: Position indices for kept tokens [batch_size, num_keep]
    """
    batch_size, seq_len, embed_dim = feat.shape
    
    dominant_num = round(math.ceil(seq_len * retain_ratio) * (1 - D))
    contextual_num = math.ceil(seq_len * retain_ratio) - dominant_num
    
    # Select dominant tokens (high attention)
    if dominant_num > 0:
        all_indices = attn.topk(dominant_num, dim=1).indices
        mask = torch.ones_like(feat[:, :, 0], dtype=torch.bool, device=feat.device)
        mask.scatter_(1, all_indices, False)
        
        dominant_tokens = feat.masked_select(~mask.unsqueeze(-1)).view(batch_size, dominant_num, embed_dim)
        dominant_pos = pos.masked_select(~mask).view(batch_size, dominant_num)
    else:
        mask = torch.ones_like(feat[:, :, 0], dtype=torch.bool, device=feat.device)
        dominant_tokens = torch.empty((batch_size, 0, embed_dim), device=feat.device)
        dominant_pos = torch.empty((batch_size, 0), device=feat.device)
    
    # Create contextual tokens via clustering
    if contextual_num > 0:
        feat_filtered = feat.masked_select(mask.unsqueeze(-1)).view(
            batch_size, seq_len - dominant_num, embed_dim
        )
        contextual_pos_raw = pos.masked_select(mask).view(batch_size, seq_len - dominant_num)
        
        target_indices, dist_matrix = cluster_dpc_knn(
            feat_filtered, contextual_num, k=min(K, contextual_num)
        )
        target_indices = torch.sort(target_indices, dim=-1)[0]
        
        contextual_pos = torch.stack([
            contextual_pos_raw[b][target_indices[b]] for b in range(batch_size)
        ])
        
        contextual_tokens = merge_tokens_by_clustering(
            feat_filtered, target_indices, dist_matrix, contextual_num, beta
        )
    else:
        contextual_tokens = torch.empty((batch_size, 0, embed_dim), device=feat.device)
        contextual_pos = torch.empty((batch_size, 0), device=feat.device)
    
    # Combine dominant and contextual tokens
    image_feat = []
    image_pos = []
    for b in range(batch_size):
        batch_tokens = torch.cat([dominant_tokens[b], contextual_tokens[b]], dim=0)
        batch_pos = torch.cat([dominant_pos[b], contextual_pos[b]], dim=0)
        image_feat.append(batch_tokens)
        image_pos.append(batch_pos)
    
    image_feat = torch.stack(image_feat)
    image_pos = torch.stack(image_pos)
    
    return image_feat, image_pos


def merge_tokens_by_density(
    feat: torch.Tensor,
    pos: torch.Tensor,
    retain_ratio: float,
    beta: float,
    K: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merge tokens using pure density clustering (no attention).
    
    Args:
        feat: Token features [batch_size, seq_len, embed_dim]
        pos: Position indices [batch_size, seq_len]
        retain_ratio: Ratio of tokens to retain
        beta: Weight for merging
        K: Number of neighbors for clustering
        
    Returns:
        image_feat: Merged features [batch_size, cluster_num, embed_dim]
        image_pos: Position indices [batch_size, cluster_num]
    """
    batch_size, seq_len, embed_dim = feat.shape
    cluster_num = round(seq_len * retain_ratio)
    
    if cluster_num > 0:
        target_indices, dist_matrix = cluster_dpc_knn(feat, cluster_num, k=min(K, cluster_num))
        target_indices = torch.sort(target_indices, dim=-1)[0]
        
        image_pos = torch.stack([pos[b][target_indices[b]] for b in range(batch_size)])
        cluster_tokens = merge_tokens_by_clustering(
            feat, target_indices, dist_matrix, cluster_num, beta
        )
        image_feat = cluster_tokens
    else:
        image_feat = torch.empty((batch_size, 0, embed_dim), device=feat.device)
        image_pos = torch.empty((batch_size, 0), device=feat.device)
    
    return image_feat, image_pos


def holitom_segment_compression(
    static_feat: torch.Tensor,
    dynamic_feat: torch.Tensor,
    dynamic_attn: torch.Tensor,
    static_pos: torch.Tensor,
    dynamic_pos: torch.Tensor,
    window_size: int,
    retain_ratio: float,
    D: float,
    beta: float,
    K: int,
    target_dtype: torch.dtype
) -> torch.Tensor:
    """
    Apply HoliTom compression to a single temporal segment.
    
    Args:
        static_feat: Static features [num_static_tokens, embed_dim] or empty
        dynamic_feat: Dynamic features [window_size, seq_len, embed_dim]
        dynamic_attn: Attention weights [window_size, seq_len]
        static_pos: Static position indices
        dynamic_pos: Dynamic position indices
        window_size: Size of the temporal window
        retain_ratio: Ratio of tokens to retain
        D: Dominant token ratio
        beta: Merging weight
        K: KNN neighbors
        target_dtype: Output dtype
        
    Returns:
        feat: Compressed features for this segment [num_tokens, embed_dim]
    """
    if window_size == 1:
        # Single frame - only apply attention-based merging
        dynamic_feat, dynamic_pos = merge_tokens_by_attention_density(
            dynamic_feat, dynamic_attn, dynamic_pos, retain_ratio, D, beta, K
        )
        feat = dynamic_feat.flatten(0, 1)
    else:
        # Multi-frame - merge both static and dynamic
        dynamic_feat, dynamic_pos = merge_tokens_by_attention_density(
            dynamic_feat, dynamic_attn, dynamic_pos, retain_ratio, D, beta, K
        )
        
        if static_feat.numel() > 0:
            static_feat, static_pos = merge_tokens_by_density(
                static_feat.unsqueeze(0), static_pos, retain_ratio, beta, K
            )
            feat = torch.cat([static_feat.flatten(0, 1), dynamic_feat.flatten(0, 1)])
        else:
            feat = dynamic_feat.flatten(0, 1)
    
    return feat.to(target_dtype)


def holitom_compression(
    video_feat: torch.Tensor,
    attn_weights: Optional[torch.Tensor] = None,
    retain_ratio: float = 0.15,
    tau: float = 0.8,
    beta: float = 0.6,
    D: float = 0.0,
    K: int = 7,
    max_window_size: int = 1024
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Main HoliTom compression function.
    
    Args:
        video_feat: Video features [num_frames, seq_len, embed_dim]
        attn_weights: Optional attention weights [num_frames, seq_len]
                     If None, uses self-similarity based scores
        retain_ratio: Base ratio of tokens to retain (default: 0.15)
        tau: Similarity threshold for static detection (default: 0.8)
        beta: Clustering merge weight (default: 0.6)
        D: Dominant token ratio (default: 0.0)
        K: KNN neighbors (default: 7)
        max_window_size: Maximum temporal window size (default: 1024)
        
    Returns:
        compressed_feat: Compressed video features [num_tokens, embed_dim]
        keep_indices: Indices of kept tokens (approximation for compatibility)
    """
    num_frames, seq_len, embed_dim = video_feat.shape
    target_dtype = video_feat.dtype
    
    # Compute frame-to-frame similarity
    video_feat_norm = F.normalize(video_feat.float(), p=2, dim=-1)
    feature_sim = F.cosine_similarity(
        video_feat_norm[:-1], video_feat_norm[1:], dim=-1
    )  # [num_frames-1, seq_len]
    
    # Select static windows via DP
    selected_frames, total_reduced = select_static_windows(
        feature_sim, num_frames, tau, max_window_size
    )
    
    # Adjust retain ratio based on static reduction
    total_tokens = num_frames * seq_len
    if total_tokens > total_reduced:
        adjusted_ratio = min(retain_ratio / ((total_tokens - total_reduced) / total_tokens), 1.0)
    else:
        adjusted_ratio = retain_ratio
    
    # Separate static and dynamic features
    static_feats, dynamic_feats, dynamic_attns, static_poses, dynamic_poses = \
        get_static_dynamic_features(
            video_feat, attn_weights, selected_frames, feature_sim, tau
        )
    
    # Compress each segment
    segment_features = []
    for idx, (start, end) in enumerate(selected_frames):
        window_size = end - start + 1
        segment_feat = holitom_segment_compression(
            static_feats[idx],
            dynamic_feats[idx],
            dynamic_attns[idx],
            static_poses[idx],
            dynamic_poses[idx],
            window_size,
            adjusted_ratio,
            D,
            beta,
            K,
            target_dtype
        )
        segment_features.append(segment_feat)
    
    compressed_feat = torch.cat(segment_features, dim=0)
    
    # Generate approximate keep indices (for compatibility with framework)
    num_keep = compressed_feat.shape[0]
    keep_indices = torch.linspace(0, num_frames * seq_len - 1, num_keep, dtype=torch.long, device=video_feat.device)
    
    return compressed_feat, keep_indices



