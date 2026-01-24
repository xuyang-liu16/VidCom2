"""
Pooling: Spatial Pooling for Token Compression

A simple but effective baseline that reduces token count through spatial pooling.
This method preserves spatial structure while reducing the number of tokens.

Supported pooling types:
- Average pooling: Computes mean of pooled regions
- Max pooling: Takes maximum value from pooled regions
- Adaptive pooling: Uses PyTorch's adaptive pooling for exact output size
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import math


def spatial_average_pooling(
    features: torch.Tensor,
    pool_size: int = 2
) -> torch.Tensor:
    """
    Apply spatial average pooling to reduce token count.
    
    Assumes tokens are arranged in a 2D grid (flattened).
    
    Args:
        features: Token features [batch_size, seq_len, hidden_dim]
        pool_size: Pooling window size (default: 2 for 2x2 pooling)
        
    Returns:
        pooled_features: Pooled features [batch_size, seq_len // pool_size^2, hidden_dim]
    """
    batch_size, seq_len, hidden_dim = features.shape
    
    # Estimate spatial dimensions (assume square grid)
    h = w = int(math.sqrt(seq_len))
    
    if h * w != seq_len:
        # Not a perfect square, try to find closest factors
        for i in range(int(math.sqrt(seq_len)), 0, -1):
            if seq_len % i == 0:
                h = i
                w = seq_len // i
                break
    
    # Reshape to 2D spatial layout: [batch, h, w, hidden]
    features_2d = features.view(batch_size, h, w, hidden_dim)
    
    # Permute to [batch, hidden, h, w] for pooling
    features_2d = features_2d.permute(0, 3, 1, 2)
    
    # Apply average pooling
    pooled = F.avg_pool2d(features_2d, kernel_size=pool_size, stride=pool_size)
    
    # Permute back and flatten: [batch, new_h * new_w, hidden]
    pooled = pooled.permute(0, 2, 3, 1)
    pooled = pooled.reshape(batch_size, -1, hidden_dim)
    
    return pooled


def spatial_max_pooling(
    features: torch.Tensor,
    pool_size: int = 2
) -> torch.Tensor:
    """
    Apply spatial max pooling to reduce token count.
    
    Args:
        features: Token features [batch_size, seq_len, hidden_dim]
        pool_size: Pooling window size
        
    Returns:
        pooled_features: Pooled features
    """
    batch_size, seq_len, hidden_dim = features.shape
    
    h = w = int(math.sqrt(seq_len))
    if h * w != seq_len:
        for i in range(int(math.sqrt(seq_len)), 0, -1):
            if seq_len % i == 0:
                h = i
                w = seq_len // i
                break
    
    features_2d = features.view(batch_size, h, w, hidden_dim)
    features_2d = features_2d.permute(0, 3, 1, 2)
    
    pooled = F.max_pool2d(features_2d, kernel_size=pool_size, stride=pool_size)
    
    pooled = pooled.permute(0, 2, 3, 1)
    pooled = pooled.reshape(batch_size, -1, hidden_dim)
    
    return pooled


def adaptive_pooling(
    features: torch.Tensor,
    target_length: int,
    pool_type: str = "avg"
) -> torch.Tensor:
    """
    Apply adaptive pooling to achieve exact target token count.
    
    Args:
        features: Token features [batch_size, seq_len, hidden_dim]
        target_length: Target number of tokens
        pool_type: Type of pooling ("avg" or "max")
        
    Returns:
        pooled_features: Pooled features [batch_size, target_length, hidden_dim]
    """
    batch_size, seq_len, hidden_dim = features.shape
    
    if target_length >= seq_len:
        return features
    
    # Estimate output spatial dimensions (prefer square)
    target_h = target_w = int(math.sqrt(target_length))
    while target_h * target_w < target_length:
        target_w += 1
    
    # Adjust to match exact target (may not be exact for non-square)
    target_h = int(math.sqrt(target_length))
    target_w = (target_length + target_h - 1) // target_h
    
    # Get input spatial dimensions
    h = w = int(math.sqrt(seq_len))
    if h * w != seq_len:
        for i in range(int(math.sqrt(seq_len)), 0, -1):
            if seq_len % i == 0:
                h = i
                w = seq_len // i
                break
    
    # Reshape to 2D
    features_2d = features.view(batch_size, h, w, hidden_dim)
    features_2d = features_2d.permute(0, 3, 1, 2)  # [batch, hidden, h, w]
    
    # Apply adaptive pooling
    if pool_type == "avg":
        pooled = F.adaptive_avg_pool2d(features_2d, (target_h, target_w))
    else:
        pooled = F.adaptive_max_pool2d(features_2d, (target_h, target_w))
    
    # Flatten
    pooled = pooled.permute(0, 2, 3, 1)
    pooled = pooled.reshape(batch_size, -1, hidden_dim)
    
    # Truncate to exact target length if needed
    if pooled.shape[1] > target_length:
        pooled = pooled[:, :target_length, :]
    
    return pooled


def stride_pooling(
    features: torch.Tensor,
    stride: int = 2
) -> torch.Tensor:
    """
    Simple strided sampling to reduce token count.
    
    Takes every stride-th token in a regular pattern.
    
    Args:
        features: Token features [batch_size, seq_len, hidden_dim]
        stride: Sampling stride
        
    Returns:
        sampled_features: Strided features
    """
    return features[:, ::stride, :]


def pooling_compression(
    features: torch.Tensor,
    attention_scores: Optional[torch.Tensor] = None,
    retention_ratio: float = 0.25,
    pool_type: str = "avg"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pooling compression: Reduce tokens via spatial pooling.
    
    Args:
        features: Visual token features [batch_size, seq_len, hidden_dim]
        attention_scores: Optional attention scores (not used, for API compatibility)
        retention_ratio: Ratio of tokens to keep (default: 0.25 for 25%)
        pool_type: Type of pooling ("avg", "max", "stride")
        
    Returns:
        compressed_features: Compressed features [batch_size, num_keep, hidden_dim]
        keep_indices: Approximate indices of kept tokens
    """
    batch_size, seq_len, hidden_dim = features.shape
    
    # Calculate target length
    target_length = max(1, int(seq_len * retention_ratio))
    
    if target_length >= seq_len:
        indices = torch.arange(seq_len, device=features.device).unsqueeze(0).expand(batch_size, -1)
        return features, indices
    
    # Apply pooling based on type
    if pool_type == "avg":
        compressed_features = adaptive_pooling(features, target_length, "avg")
    elif pool_type == "max":
        compressed_features = adaptive_pooling(features, target_length, "max")
    elif pool_type == "stride":
        stride = max(1, seq_len // target_length)
        compressed_features = stride_pooling(features, stride)
        # Truncate to target if needed
        if compressed_features.shape[1] > target_length:
            compressed_features = compressed_features[:, :target_length, :]
    else:
        # Default to adaptive average pooling
        compressed_features = adaptive_pooling(features, target_length, "avg")
    
    # Generate approximate indices (for compatibility)
    num_keep = compressed_features.shape[1]
    keep_indices = torch.linspace(
        0, seq_len - 1, num_keep,
        dtype=torch.long, device=features.device
    ).unsqueeze(0).expand(batch_size, -1)
    
    return compressed_features, keep_indices

