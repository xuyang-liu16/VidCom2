"""
VisionZip: Longer is Better but Not Necessary in Vision Language Models

This implementation is based on the paper:
"VisionZip: Longer is Better but Not Necessary in Vision Language Models"
CVPR 2025

Key idea: Select dominant tokens with high attention and merge remaining tokens based on density.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def select_dominant_tokens(
    features: torch.Tensor,
    attention_scores: torch.Tensor,
    num_dominant: int,
    cls_token: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select dominant tokens that receive high attention scores.
    
    Args:
        features: Token features [batch_size, seq_len, hidden_dim]
        attention_scores: Attention scores for each token [batch_size, seq_len]
        num_dominant: Number of dominant tokens to select
        cls_token: Whether to always keep the first token (CLS token)
        
    Returns:
        dominant_features: Features of dominant tokens [batch_size, num_dominant, hidden_dim]
        dominant_indices: Indices of dominant tokens [batch_size, num_dominant]
    """
    batch_size, seq_len, hidden_dim = features.shape
    
    if cls_token and seq_len > 0:
        # Always keep CLS token, select from remaining
        remaining_scores = attention_scores[:, 1:]
        _, topk_indices = torch.topk(remaining_scores, k=min(num_dominant - 1, seq_len - 1), dim=1)
        topk_indices = topk_indices + 1  # Adjust for CLS token
        cls_indices = torch.zeros((batch_size, 1), dtype=torch.long, device=features.device)
        dominant_indices = torch.cat([cls_indices, topk_indices], dim=1)
    else:
        _, dominant_indices = torch.topk(attention_scores, k=min(num_dominant, seq_len), dim=1)
    
    # Sort to maintain order
    dominant_indices, _ = torch.sort(dominant_indices, dim=1)
    
    # Gather dominant features
    dominant_indices_expanded = dominant_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
    dominant_features = torch.gather(features, dim=1, index=dominant_indices_expanded)
    
    return dominant_features, dominant_indices


def compute_density_scores(
    features: torch.Tensor,
    k_neighbors: int = 5
) -> torch.Tensor:
    """
    Compute density scores for each token based on k-nearest neighbors.
    
    Args:
        features: Token features [batch_size, seq_len, hidden_dim]
        k_neighbors: Number of nearest neighbors to consider
        
    Returns:
        density_scores: Density score for each token [batch_size, seq_len]
    """
    batch_size, seq_len, hidden_dim = features.shape
    
    # Normalize features
    features_norm = F.normalize(features, dim=-1)
    
    # Compute pairwise similarity: [batch_size, seq_len, seq_len]
    similarity_matrix = torch.bmm(features_norm, features_norm.transpose(1, 2))
    
    # For each token, compute average similarity to k nearest neighbors
    k = min(k_neighbors + 1, seq_len)  # +1 because token is similar to itself
    topk_similarities, _ = torch.topk(similarity_matrix, k=k, dim=2)
    
    # Exclude self-similarity (first column after topk)
    neighbor_similarities = topk_similarities[:, :, 1:]  # [batch_size, seq_len, k_neighbors]
    
    # Density is average similarity to k nearest neighbors
    density_scores = neighbor_similarities.mean(dim=2)  # [batch_size, seq_len]
    
    return density_scores


def density_based_merging(
    features: torch.Tensor,
    dominant_indices: torch.Tensor,
    num_contextual: int,
    k_neighbors: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merge remaining tokens based on density to create contextual tokens.
    
    Args:
        features: All token features [batch_size, seq_len, hidden_dim]
        dominant_indices: Indices of dominant tokens [batch_size, num_dominant]
        num_contextual: Number of contextual tokens to create
        k_neighbors: Number of neighbors for density computation
        
    Returns:
        contextual_features: Merged contextual features [batch_size, num_contextual, hidden_dim]
        contextual_indices: Representative indices for contextual tokens [batch_size, num_contextual]
    """
    batch_size, seq_len, hidden_dim = features.shape
    
    # Create mask for non-dominant tokens
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=features.device)
    mask.scatter_(1, dominant_indices, False)
    
    # Get remaining tokens
    remaining_features = []
    remaining_indices = []
    for b in range(batch_size):
        remaining_features.append(features[b, mask[b]])
        remaining_indices.append(torch.where(mask[b])[0])
    
    # Handle edge case: if no remaining tokens or num_contextual is 0
    if len(remaining_features[0]) == 0 or num_contextual == 0:
        # Return empty contextual tokens
        empty_features = torch.zeros(batch_size, 0, hidden_dim, device=features.device)
        empty_indices = torch.zeros(batch_size, 0, dtype=torch.long, device=features.device)
        return empty_features, empty_indices
    
    # Stack remaining features
    max_remaining = max(len(rf) for rf in remaining_features)
    
    contextual_features_list = []
    contextual_indices_list = []
    
    for b in range(batch_size):
        rem_feat = remaining_features[b]  # [num_remaining, hidden_dim]
        rem_idx = remaining_indices[b]    # [num_remaining]
        
        if len(rem_feat) == 0:
            # No remaining tokens, create zero features
            contextual_features_list.append(
                torch.zeros(num_contextual, hidden_dim, device=features.device)
            )
            contextual_indices_list.append(
                torch.zeros(num_contextual, dtype=torch.long, device=features.device)
            )
            continue
        
        # Compute density scores for remaining tokens
        rem_feat_batch = rem_feat.unsqueeze(0)  # [1, num_remaining, hidden_dim]
        density = compute_density_scores(rem_feat_batch, k_neighbors=k_neighbors)[0]  # [num_remaining]
        
        # Select high-density tokens as anchors for merging
        num_merge = min(num_contextual, len(rem_feat))
        _, anchor_idx = torch.topk(density, k=num_merge, dim=0)
        anchor_idx, _ = torch.sort(anchor_idx)
        
        anchor_features = rem_feat[anchor_idx]  # [num_merge, hidden_dim]
        anchor_indices = rem_idx[anchor_idx]    # [num_merge]
        
        # Assign remaining tokens to nearest anchor and merge
        if len(anchor_features) < len(rem_feat):
            # Compute similarity to anchors
            rem_feat_norm = F.normalize(rem_feat, dim=-1)
            anchor_norm = F.normalize(anchor_features, dim=-1)
            similarity = torch.mm(rem_feat_norm, anchor_norm.t())  # [num_remaining, num_merge]
            
            # Assign each token to nearest anchor
            assignments = similarity.argmax(dim=1)  # [num_remaining]
            
            # Merge tokens assigned to each anchor
            merged_features = []
            for i in range(len(anchor_features)):
                assigned_mask = (assignments == i)
                if assigned_mask.any():
                    assigned_features = rem_feat[assigned_mask]
                    # Average assigned features
                    merged = assigned_features.mean(dim=0)
                else:
                    # No tokens assigned, use anchor itself
                    merged = anchor_features[i]
                merged_features.append(merged)
            
            merged_features = torch.stack(merged_features)  # [num_merge, hidden_dim]
        else:
            merged_features = anchor_features
        
        # Pad if necessary
        if len(merged_features) < num_contextual:
            padding = torch.zeros(
                num_contextual - len(merged_features),
                hidden_dim,
                device=features.device
            )
            merged_features = torch.cat([merged_features, padding], dim=0)
            
            padding_idx = torch.zeros(
                num_contextual - len(anchor_indices),
                dtype=torch.long,
                device=features.device
            )
            anchor_indices = torch.cat([anchor_indices, padding_idx], dim=0)
        
        contextual_features_list.append(merged_features)
        contextual_indices_list.append(anchor_indices)
    
    contextual_features = torch.stack(contextual_features_list)  # [batch_size, num_contextual, hidden_dim]
    contextual_indices = torch.stack(contextual_indices_list)    # [batch_size, num_contextual]
    
    return contextual_features, contextual_indices


def visionzip_compression(
    features: torch.Tensor,
    attention_scores: torch.Tensor,
    retention_ratio: float = 0.2,
    dominant_ratio: float = 0.5,
    cls_token: bool = True,
    k_neighbors: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    VisionZip compression: select dominant tokens and merge remaining via density.
    
    Args:
        features: Visual token features [batch_size, seq_len, hidden_dim]
        attention_scores: Attention scores for each token [batch_size, seq_len]
        retention_ratio: Overall ratio of tokens to keep (default: 0.2 for 20%)
        dominant_ratio: Ratio of retained tokens that are dominant (default: 0.5)
        cls_token: Whether to always keep the first token (CLS token)
        k_neighbors: Number of neighbors for density computation
        
    Returns:
        compressed_features: Compressed features [batch_size, num_keep, hidden_dim]
        keep_indices: Indices of kept tokens [batch_size, num_keep]
    """
    batch_size, seq_len, hidden_dim = features.shape
    
    num_keep = max(1, int(seq_len * retention_ratio))
    num_dominant = max(1, int(num_keep * dominant_ratio))
    num_contextual = num_keep - num_dominant
    
    # Step 1: Select dominant tokens
    dominant_features, dominant_indices = select_dominant_tokens(
        features,
        attention_scores,
        num_dominant=num_dominant,
        cls_token=cls_token
    )
    
    # Step 2: Merge remaining tokens based on density
    if num_contextual > 0:
        contextual_features, contextual_indices = density_based_merging(
            features,
            dominant_indices,
            num_contextual=num_contextual,
            k_neighbors=k_neighbors
        )
        
        # Combine dominant and contextual tokens
        compressed_features = torch.cat([dominant_features, contextual_features], dim=1)
        keep_indices = torch.cat([dominant_indices, contextual_indices], dim=1)
    else:
        compressed_features = dominant_features
        keep_indices = dominant_indices
    
    return compressed_features, keep_indices

