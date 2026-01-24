"""
iLLaVA: Token Merging for Vision-Language Models

This implementation is based on the paper and code:
"iLLaVA: An Image is Worth Fewer Tokens in Large Vision-Language Models"

Reference: https://github.com/hulianyuyy/iLLaVA

Key idea: Use bipartite soft matching (same as ToMe) to merge similar visual tokens,
reducing computational cost while preserving important visual information.

Following official ToMe/iLLaVA bipartite matching:
- Split tokens into src/dst using alternating indices (even=a, odd=b)
- Find most similar pairs and merge top-r pairs
- Use scatter_reduce for efficient merging
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Callable


def do_nothing(x: torch.Tensor, mode: str = None) -> torch.Tensor:
    """Identity function for when no merging is needed."""
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
) -> Tuple[Callable, Callable]:
    """
    Bipartite soft matching algorithm (following official ToMe/iLLaVA).
    
    Uses alternating assignment to split tokens into src/dst sets,
    then merges the r most similar pairs.
    
    Args:
        metric: Metric tensor for matching [batch, tokens, channels]
        r: Number of tokens to remove (merge)
        
    Returns:
        merge: Function to merge tokens
        unmerge: Function to unmerge tokens (approximate reconstruction)
    """
    # Can only reduce by max 50% tokens
    t = metric.shape[1]
    r = min(r, t // 2)
    
    if r <= 0:
        return do_nothing, do_nothing
    
    with torch.no_grad():
        # Normalize for cosine similarity
        metric = metric / metric.norm(dim=-1, keepdim=True)
        
        # Alternating split: even indices = a (src candidates), odd indices = b (dst)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        
        # Compute similarity scores between a and b
        scores = a @ b.transpose(-1, -2)  # [batch, len_a, len_b]
        
        # For each token in a, find most similar token in b
        node_max, node_idx = scores.max(dim=-1)  # [batch, len_a]
        
        # Sort a tokens by their max similarity (descending)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]  # [batch, len_a, 1]
        
        # Split: top r go to src (to be merged), rest go to unm (unmerged)
        unm_idx = edge_idx[..., r:, :]  # Unmerged tokens from a
        src_idx = edge_idx[..., :r, :]  # Tokens to merge
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)  # Their destinations in b
    
    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        """Merge tokens according to the matching."""
        # Split by alternating indices
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        
        # Gather unmerged and to-be-merged tokens from src (a)
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src_merged = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        
        # Merge src tokens into dst using scatter_reduce
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src_merged, reduce=mode)
        
        # Concatenate: unmerged from a + merged b
        return torch.cat([unm, dst], dim=1)
    
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        """Unmerge tokens (approximate reconstruction)."""
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape
        
        # Reconstruct src from dst
        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))
        
        # Interleave back to original order
        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)
        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)
        
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
        merge: Merge function from bipartite_soft_matching
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


def iterative_token_merging(
    features: torch.Tensor,
    target_length: int,
    r_per_step: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Iteratively merge tokens until reaching target length.
    
    Args:
        features: Token features [batch_size, seq_len, hidden_dim]
        target_length: Target number of tokens after merging
        r_per_step: Tokens to merge per step (if None, merge all at once)
        
    Returns:
        merged_features: Merged features [batch_size, target_length, hidden_dim]
        final_indices: Approximate indices of final tokens (for compatibility)
    """
    batch_size, seq_len, hidden_dim = features.shape
    
    if target_length >= seq_len:
        indices = torch.arange(seq_len, device=features.device).unsqueeze(0).expand(batch_size, -1)
        return features, indices
    
    current_features = features
    current_length = seq_len
    size = None
    
    while current_length > target_length:
        r = min(current_length - target_length, current_length // 2)
        if r <= 0:
            break
        
        if r_per_step is not None:
            r = min(r, r_per_step)
        
        # Get merge function
        merge_fn, _ = bipartite_soft_matching(current_features, r=r)
        
        # Apply merge with weighted averaging
        current_features, size = merge_wavg(merge_fn, current_features, size)
        current_length = current_features.shape[1]
    
    # Generate approximate indices for compatibility
    num_keep = current_features.shape[1]
    keep_indices = torch.linspace(
        0, seq_len - 1, num_keep,
        dtype=torch.long, device=features.device
    ).unsqueeze(0).expand(batch_size, -1)
    
    return current_features, keep_indices


def illava_compression(
    features: torch.Tensor,
    attention_scores: Optional[torch.Tensor] = None,
    retention_ratio: float = 0.25,
    merge_ratio: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    iLLaVA compression: Token merging using bipartite soft matching.
    
    Following official ToMe/iLLaVA implementation with alternating split.
    
    Args:
        features: Visual token features [batch_size, seq_len, hidden_dim]
        attention_scores: Optional attention scores (not used, for API compatibility)
        retention_ratio: Ratio of tokens to keep (default: 0.25 for 25%)
        merge_ratio: Ratio of tokens to merge per iteration (default: 0.5)
        
    Returns:
        compressed_features: Compressed features [batch_size, num_keep, hidden_dim]
        keep_indices: Approximate indices of kept tokens [batch_size, num_keep]
    """
    batch_size, seq_len, hidden_dim = features.shape
    
    # Calculate target length
    target_length = max(1, int(seq_len * retention_ratio))
    
    # Calculate r per step based on merge_ratio
    r_per_step = max(1, int(seq_len * merge_ratio / 2))  # /2 because max merge is 50%
    
    # Apply iterative token merging
    compressed_features, keep_indices = iterative_token_merging(
        features,
        target_length,
        r_per_step=r_per_step
    )
    
    return compressed_features, keep_indices
