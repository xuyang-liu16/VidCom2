"""
FastV: An Image is Worth 1/2 Tokens After Layer 2
Plug-and-Play Inference Acceleration for Large Vision-Language Models

This implementation is based on the paper:
"An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models"
ECCV 2024 Oral

Key idea: Prune redundant visual tokens at early LLM layers based on attention scores.
"""

import torch
from typing import Optional, Tuple


def select_important_tokens(
    attention_scores: torch.Tensor,
    num_keep: int,
    cls_token: bool = True
) -> torch.Tensor:
    """
    Select important tokens based on attention scores.
    
    Args:
        attention_scores: Attention scores of shape [batch_size, num_tokens] or [num_tokens]
        num_keep: Number of tokens to keep
        cls_token: Whether to always keep the first token (CLS token)
        
    Returns:
        indices: Indices of selected tokens [num_keep] or [batch_size, num_keep]
    """
    if attention_scores.dim() == 1:
        attention_scores = attention_scores.unsqueeze(0)
    
    batch_size, num_tokens = attention_scores.shape
    
    if cls_token and num_tokens > 0:
        # Always keep CLS token (first token)
        # Select top k-1 from remaining tokens
        remaining_scores = attention_scores[:, 1:]
        _, topk_indices = torch.topk(remaining_scores, k=min(num_keep - 1, num_tokens - 1), dim=1)
        # Add 1 to indices to account for CLS token
        topk_indices = topk_indices + 1
        # Concatenate CLS token index (0) with selected indices
        cls_indices = torch.zeros((batch_size, 1), dtype=torch.long, device=attention_scores.device)
        selected_indices = torch.cat([cls_indices, topk_indices], dim=1)
    else:
        # Select top k tokens directly
        _, selected_indices = torch.topk(attention_scores, k=min(num_keep, num_tokens), dim=1)
    
    # Sort indices to maintain relative order
    selected_indices, _ = torch.sort(selected_indices, dim=1)
    
    if selected_indices.shape[0] == 1:
        return selected_indices.squeeze(0)
    return selected_indices


def fastv_compression(
    hidden_states: torch.Tensor,
    attention_scores: torch.Tensor,
    retention_ratio: float = 0.5,
    cls_token: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FastV token compression: prune visual tokens based on attention scores.
    
    Args:
        hidden_states: Visual token features [batch_size, seq_len, hidden_dim]
        attention_scores: Attention scores received by each token [batch_size, seq_len]
                         This should be the average attention score across all heads
        retention_ratio: Ratio of tokens to keep (default: 0.5 for 50%)
        cls_token: Whether to always keep the first token (CLS token)
        
    Returns:
        compressed_states: Compressed token features [batch_size, num_keep, hidden_dim]
        keep_indices: Indices of kept tokens [batch_size, num_keep]
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape
    num_keep = max(1, int(seq_len * retention_ratio))
    
    # Select important tokens based on attention scores
    keep_indices = select_important_tokens(
        attention_scores, 
        num_keep=num_keep,
        cls_token=cls_token
    )
    
    # Gather selected tokens
    if keep_indices.dim() == 1:
        keep_indices = keep_indices.unsqueeze(0).expand(batch_size, -1)
    
    # Expand indices for gathering hidden states
    keep_indices_expanded = keep_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
    compressed_states = torch.gather(hidden_states, dim=1, index=keep_indices_expanded)
    
    return compressed_states, keep_indices


def compute_token_attention_scores(
    attention_weights: torch.Tensor,
    aggregate_method: str = "mean"
) -> torch.Tensor:
    """
    Compute per-token attention scores from attention weights.
    
    Args:
        attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
                          or [batch_size, seq_len, seq_len]
        aggregate_method: Method to aggregate attention ("mean" or "max")
        
    Returns:
        token_scores: Aggregated attention scores per token [batch_size, seq_len]
    """
    if attention_weights.dim() == 4:
        # Average across heads: [batch_size, num_heads, seq_len, seq_len] -> [batch_size, seq_len, seq_len]
        attention_weights = attention_weights.mean(dim=1)
    
    # Aggregate attention received by each token (column-wise)
    if aggregate_method == "mean":
        token_scores = attention_weights.mean(dim=1)  # [batch_size, seq_len]
    elif aggregate_method == "max":
        token_scores = attention_weights.max(dim=1)[0]  # [batch_size, seq_len]
    else:
        raise ValueError(f"Unknown aggregate method: {aggregate_method}")
    
    return token_scores

