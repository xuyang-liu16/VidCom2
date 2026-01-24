"""
ToMe: Token Merging for Fast and Accurate Vision Transformer

This implementation is based on the paper:
"Token Merging: Your ViT But Faster"
ICLR 2023

Reference: https://github.com/facebookresearch/ToMe

Key idea: Use bipartite soft matching on keys to efficiently merge similar tokens.
The algorithm alternates tokens into source and destination sets, then matches
the most similar pairs using cosine similarity on key vectors.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Callable
import math


def do_nothing(x: torch.Tensor, mode: str = None) -> torch.Tensor:
    """Identity function for when no merging is needed."""
    return x


def bipartite_soft_matching_tome(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False
) -> Tuple[Callable, Callable]:
    """
    ToMe's bipartite soft matching algorithm.
    
    Applies a soft matching between tokens to merge the r most similar pairs.
    Uses alternating assignment to split tokens into src/dst sets.
    
    Args:
        metric: Metric tensor used for matching [batch, tokens, channels]
                Typically the key vectors from self-attention
        r: Number of tokens to remove (merge)
        class_token: Whether the first token is a class token to be protected
        distill_token: Whether the second token is a distillation token
        
    Returns:
        merge: Function to merge tokens
        unmerge: Function to unmerge tokens (for reconstruction)
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1
    
    # Protect special tokens by moving them to the front
    # We won't include them in the matching
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)
    
    if r <= 0:
        return do_nothing, do_nothing
    
    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        
        # Handle protected tokens
        if protected:
            a = a[..., protected:, :]
        
        scores = a @ b.transpose(-1, -2)
        
        # Find the most similar pairs
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        
        unm_idx = edge_idx[..., r:, :]  # Unmerged
        src_idx = edge_idx[..., :r, :]  # Merged (source)
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        
        if protected:
            # Adjust indices for protected tokens
            unm_idx = unm_idx + protected
    
    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        """Merge tokens according to the matching."""
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        
        if protected:
            protected_tokens = src[..., :protected, :]
            src = src[..., protected:, :]
        
        unm = src.gather(dim=-2, index=unm_idx.expand(n, -1, c))
        src_merged = src.gather(dim=-2, index=src_idx.expand(n, -1, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, -1, c), src_merged, reduce=mode)
        
        if protected:
            return torch.cat([protected_tokens, unm, dst], dim=1)
        return torch.cat([unm, dst], dim=1)
    
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        """Unmerge tokens (approximate reconstruction)."""
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape
        
        src = dst.gather(dim=-2, index=dst_idx.expand(n, -1, c))
        
        # Interleave back
        out = torch.zeros(n, (unm_len + r) * 2 + dst.shape[1], c, device=x.device, dtype=x.dtype)
        out[..., ::2, :] = torch.cat([unm, src], dim=1)
        out[..., 1::2, :] = dst
        
        return out
    
    return merge, unmerge


def merge_wavg(
    merge: Callable,
    x: torch.Tensor,
    size: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merge tokens using weighted averaging based on token size.
    
    Args:
        merge: Merge function from bipartite_soft_matching_tome
        x: Token features [batch, tokens, channels]
        size: Optional token sizes for weighted averaging [batch, tokens, 1]
        
    Returns:
        x: Merged tokens
        size: Updated sizes after merging
    """
    if size is None:
        size = torch.ones_like(x[..., :1])
    
    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size
    
    return x, size


def tome_compression(
    features: torch.Tensor,
    attention_scores: Optional[torch.Tensor] = None,
    retention_ratio: float = 0.25,
    r_per_step: Optional[int] = None,
    use_keys: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ToMe compression: Token merging using bipartite soft matching.
    
    Args:
        features: Visual token features [batch_size, seq_len, hidden_dim]
        attention_scores: Optional attention scores (can be used as metric)
        retention_ratio: Ratio of tokens to keep (default: 0.25 for 25%)
        r_per_step: Number of tokens to merge per step (if None, computed from ratio)
        use_keys: Whether to use features directly as keys for matching
        
    Returns:
        compressed_features: Compressed features [batch_size, num_keep, hidden_dim]
        keep_indices: Approximate indices of kept tokens
    """
    batch_size, seq_len, hidden_dim = features.shape
    
    # Calculate target length and number of tokens to remove
    target_length = max(1, int(seq_len * retention_ratio))
    total_r = seq_len - target_length
    
    if total_r <= 0:
        indices = torch.arange(seq_len, device=features.device).unsqueeze(0).expand(batch_size, -1)
        return features, indices
    
    # Use features as metric (keys)
    if use_keys:
        metric = features
    else:
        # Could use attention scores reshaped, but features work well
        metric = features
    
    # Calculate r per step for iterative merging
    if r_per_step is None:
        # Single step: merge all at once
        r_per_step = total_r
    
    current_features = features
    current_length = seq_len
    size = None
    
    while current_length > target_length:
        r = min(r_per_step, current_length - target_length)
        if r <= 0:
            break
        
        # Get metric for current features
        current_metric = current_features
        
        # Compute merge function
        merge_fn, _ = bipartite_soft_matching_tome(
            current_metric,
            r=r,
            class_token=False,  # Qwen3-VL doesn't use CLS token
            distill_token=False
        )
        
        # Apply merge with weighted averaging
        current_features, size = merge_wavg(merge_fn, current_features, size)
        current_length = current_features.shape[1]
    
    # Generate approximate indices (for compatibility)
    num_keep = current_features.shape[1]
    keep_indices = torch.linspace(
        0, seq_len - 1, num_keep,
        dtype=torch.long, device=features.device
    ).unsqueeze(0).expand(batch_size, -1)
    
    return current_features, keep_indices


def tome_compression_simple(
    features: torch.Tensor,
    retention_ratio: float = 0.25
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simplified ToMe compression that merges tokens in a single pass.
    
    This is a more straightforward implementation that:
    1. Computes pairwise cosine similarity
    2. Greedily merges the most similar pairs
    3. Repeats until target length is reached
    
    Args:
        features: Visual token features [batch_size, seq_len, hidden_dim]
        retention_ratio: Ratio of tokens to keep
        
    Returns:
        compressed_features: Compressed features
        keep_indices: Approximate indices
    """
    batch_size, seq_len, hidden_dim = features.shape
    target_length = max(1, int(seq_len * retention_ratio))
    
    if target_length >= seq_len:
        indices = torch.arange(seq_len, device=features.device).unsqueeze(0).expand(batch_size, -1)
        return features, indices
    
    # Normalize for cosine similarity
    features_norm = F.normalize(features, dim=-1)
    
    compressed_list = []
    indices_list = []
    
    for b in range(batch_size):
        current = features[b]  # [seq_len, hidden]
        current_norm = features_norm[b]
        current_indices = torch.arange(seq_len, device=features.device)
        
        while len(current) > target_length:
            n = len(current)
            
            # Compute similarity matrix
            sim = current_norm @ current_norm.T  # [n, n]
            
            # Mask self-similarity
            sim.fill_diagonal_(-float('inf'))
            
            # Find most similar pair
            flat_idx = sim.argmax()
            i, j = flat_idx // n, flat_idx % n
            
            # Ensure i < j for consistent ordering
            if i > j:
                i, j = j, i
            
            # Merge: average the two tokens
            merged = (current[i] + current[j]) / 2
            merged_norm = F.normalize(merged.unsqueeze(0), dim=-1).squeeze(0)
            
            # Create new tensor without merged tokens
            mask = torch.ones(n, dtype=torch.bool, device=features.device)
            mask[i] = False
            mask[j] = False
            
            remaining = current[mask]
            remaining_norm = current_norm[mask]
            remaining_indices = current_indices[mask]
            
            # Add merged token
            current = torch.cat([remaining, merged.unsqueeze(0)], dim=0)
            current_norm = torch.cat([remaining_norm, merged_norm.unsqueeze(0)], dim=0)
            current_indices = torch.cat([remaining_indices, current_indices[i:i+1]], dim=0)
        
        compressed_list.append(current)
        indices_list.append(current_indices)
    
    compressed_features = torch.stack(compressed_list, dim=0)
    keep_indices = torch.stack(indices_list, dim=0)
    
    return compressed_features, keep_indices

