"""
IPCV: Information-Preserving Compression for MLLM Visual Encoders

This implementation is based on the paper and official code:
"IPCV: Information-Preserving Compression for MLLM Visual Encoders"

Reference: https://github.com/Perkzi/IPCV

Key mechanism (following official implementation):
1. Prune at layer K based on diff between layer K and K-1 (L2 norm)
2. Compute Top-K nearest neighbors for each removed token in kept tokens
3. Use Accumulated Similarity (AS) to restore pruned tokens at subsequent layers:
   - Restore: removed_states + avg_delta_from_neighbors
   - Where delta = new_kept[neighbor] - orig_kept[neighbor]
4. Multiple layers of AS restoration before final output
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List


def compute_pruning_scores(
    hidden_states: torch.Tensor,
    hidden_states_prev: torch.Tensor,
) -> torch.Tensor:
    """
    Compute pruning scores based on the difference between two layers.
    
    Official IPCV uses L2 norm of diff: tokens with larger changes are more important.
    
    Args:
        hidden_states: Current layer features [seq_len, hidden_dim]
        hidden_states_prev: Previous layer features [seq_len, hidden_dim]
        
    Returns:
        scores: Importance scores [seq_len]
    """
    diff = hidden_states - hidden_states_prev
    scores = torch.norm(diff, dim=-1)  # L2 norm
    return scores


def select_tokens_by_score(
    scores: torch.Tensor,
    retention_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select tokens to keep based on scores.
    
    Args:
        scores: Importance scores [seq_len]
        retention_ratio: Ratio of tokens to keep
        
    Returns:
        keep_indices: Sorted indices of tokens to keep [num_keep]
        remove_indices: Sorted indices of tokens to remove [num_remove]
    """
    seq_len = scores.shape[0]
    num_keep = max(1, int(seq_len * retention_ratio))
    
    # Select top tokens by score (higher score = more important = keep)
    _, sorted_indices = torch.sort(scores, descending=True)
    keep_indices = sorted_indices[:num_keep].sort()[0]
    remove_indices = sorted_indices[num_keep:].sort()[0]
    
    return keep_indices, remove_indices


def compute_topk_neighbors(
    kept_states: torch.Tensor,
    removed_states: torch.Tensor,
    top_k: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Top-K nearest neighbors for each removed token among kept tokens.
    Following official IPCV: use L2 distance (cdist).
    
    Args:
        kept_states: Features of kept tokens [num_kept, hidden_dim]
        removed_states: Features of removed tokens [num_removed, hidden_dim]
        top_k: Number of nearest neighbors
        
    Returns:
        rem_to_kept_idx: Indices of top-k nearest kept tokens [num_removed, top_k]
        unique_idx: Unique indices in kept tokens that are referenced
        inv_idx: Inverse indices to map back from unique to full [num_removed * top_k]
    """
    with torch.no_grad():
        # Use L2 distance (official implementation)
        dists = torch.cdist(
            removed_states.float(),
            kept_states.float(),
            p=2.0
        )
        
        # Get top-k minimum distance indices
        top_k = min(top_k, kept_states.shape[0])
        _, rem_to_kept_idx = dists.topk(top_k, largest=False, dim=1)  # [R, top_k]
        
        # Compute unique indices and inverse mapping
        flat_idx = rem_to_kept_idx.view(-1)  # [R * top_k]
        unique_idx, inv_idx = torch.unique(flat_idx, return_inverse=True)
        
    return rem_to_kept_idx, unique_idx, inv_idx


class IPCVState:
    """
    State holder for IPCV compression across multiple ViT layers.
    
    Following official IPCV implementation exactly:
    - Store original kept states and removed states at pruning layer
    - For AS restoration: compute delta from current vs original kept states
    - Apply average delta to removed tokens
    """
    
    def __init__(
        self,
        retention_ratio: float = 0.25,
        top_k: int = 10,
        as_layers: int = 4,
    ):
        """
        Initialize IPCV state.
        
        Args:
            retention_ratio: Ratio of tokens to keep (default: 0.25 = 25%)
            top_k: Number of nearest neighbors (official default: 10)
            as_layers: Number of layers to apply AS restoration
        """
        self.retention_ratio = retention_ratio
        self.top_k = top_k
        self.as_layers = as_layers
        
        # State variables (populated during pruning)
        self.keep_indices: Optional[torch.Tensor] = None
        self.remove_indices: Optional[torch.Tensor] = None
        
        # For AS restoration (following official)
        self.orig_seq_len: int = 0
        self.orig_kept_states: Optional[torch.Tensor] = None
        self.removed_states: Optional[torch.Tensor] = None
        self.rem_to_kept_idx: Optional[torch.Tensor] = None
        self.unique_idx: Optional[torch.Tensor] = None
        self.inv_idx: Optional[torch.Tensor] = None
        
        # Track pruning state
        self.is_pruned = False
        self.layers_after_prune = 0
        
    def prune(
        self,
        hidden_states: torch.Tensor,
        hidden_states_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform initial pruning at layer K (following official implementation).
        
        Args:
            hidden_states: Current layer features [seq_len, hidden_dim]
            hidden_states_prev: Previous layer features [seq_len, hidden_dim]
            
        Returns:
            pruned_states: Pruned features [num_keep, hidden_dim]
        """
        self.orig_seq_len = hidden_states.shape[0]
        
        # Compute scores and select tokens
        scores = compute_pruning_scores(hidden_states, hidden_states_prev)
        self.keep_indices, self.remove_indices = select_tokens_by_score(
            scores, self.retention_ratio
        )
        
        # Get kept and removed states
        kept_states = hidden_states[self.keep_indices]
        removed_states = hidden_states[self.remove_indices]
        
        # Store for AS restoration (official: save orig_kept_states and removed_states)
        self.orig_kept_states = kept_states.clone()
        self.removed_states = removed_states.clone()
        
        if removed_states.shape[0] > 0:
            # Compute Top-K nearest neighbors
            self.rem_to_kept_idx, self.unique_idx, self.inv_idx = compute_topk_neighbors(
                kept_states, removed_states, self.top_k
            )
        else:
            self.rem_to_kept_idx = None
            self.unique_idx = None
            self.inv_idx = None
        
        self.is_pruned = True
        self.layers_after_prune = 0
        
        return kept_states
    
    def restore_for_as(
        self,
        kept_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Restore full sequence for AS layer processing (following official).
        
        Official implementation:
        1. Compute delta_unique = new_kept[unique_idx] - orig_kept[unique_idx]
        2. avg_delta_removed = delta_unique[inv_idx].mean(dim=1)
        3. restored = removed_states + avg_delta_removed
        
        Args:
            kept_states: Current kept states [num_keep, hidden_dim]
            
        Returns:
            full_states: Restored full sequence [seq_len, hidden_dim]
        """
        if not self.is_pruned or self.removed_states is None:
            return kept_states
            
        if self.rem_to_kept_idx is None or self.removed_states.shape[0] == 0:
            return kept_states
        
        device = kept_states.device
        dtype = kept_states.dtype
        hidden_dim = kept_states.shape[-1]
        
        # 1. Compute delta for unique indices (official: delta = new - orig)
        delta_unique = kept_states[self.unique_idx] - self.orig_kept_states[self.unique_idx]
        
        # 2. Reshape inv_idx and compute average delta for each removed token
        inv_idx_reshaped = self.inv_idx.view(self.rem_to_kept_idx.shape)  # [R, top_k]
        avg_delta_removed = delta_unique[inv_idx_reshaped].mean(dim=1)  # [R, D]
        
        # 3. Restore removed tokens: removed_states + avg_delta
        restored_removed = self.removed_states + avg_delta_removed
        
        # 4. Reconstruct full sequence
        full_states = torch.zeros(
            self.orig_seq_len, hidden_dim,
            device=device, dtype=dtype
        )
        full_states[self.keep_indices] = kept_states
        full_states[self.remove_indices] = restored_removed
        
        return full_states
    
    def re_prune_after_block(
        self,
        full_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Re-prune after processing a block in AS mode.
        
        Following official: after attention computation with full sequence,
        extract only kept tokens to continue.
        
        Args:
            full_states: Full sequence after block processing [seq_len, hidden_dim]
            
        Returns:
            pruned_states: Pruned features [num_keep, hidden_dim]
        """
        if not self.is_pruned:
            return full_states
        
        self.layers_after_prune += 1
        return full_states[self.keep_indices]
    
    def final_restore(
        self,
        kept_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Final restoration at the last layer (following official).
        
        Official: at final layer, restore full sequence using AS mechanism.
        
        Args:
            kept_states: Final kept states [num_keep, hidden_dim]
            
        Returns:
            final_states: Full restored sequence [seq_len, hidden_dim]
        """
        if not self.is_pruned or self.removed_states is None:
            return kept_states
            
        if self.rem_to_kept_idx is None or self.removed_states.shape[0] == 0:
            return kept_states
        
        # Use AS restoration for final output
        return self.restore_for_as(kept_states)
    
    def should_apply_as(self) -> bool:
        """Check if we should apply AS restoration at current layer."""
        return self.is_pruned and self.layers_after_prune < self.as_layers


# Simple interface for basic compression (backward compatibility)
def ipcv_compression(
    features: torch.Tensor,
    attention_scores: Optional[torch.Tensor] = None,
    retention_ratio: float = 0.25,
    merge_method: str = "weighted_avg"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simple IPCV compression (backward compatible interface).
    
    Note: For full IPCV with multi-layer AS, use IPCVState class directly
    within the ViT forward pass.
    
    Args:
        features: Visual token features [batch_size, seq_len, hidden_dim]
        attention_scores: Optional pre-computed attention scores [batch_size, seq_len]
        retention_ratio: Ratio of tokens to keep (default: 0.25 for 25%)
        merge_method: Method to merge remaining tokens (not used, for API compatibility)
        
    Returns:
        compressed_features: Compressed features [batch_size, num_keep, hidden_dim]
        keep_indices: Indices of selected tokens [batch_size, num_keep]
    """
    batch_size, seq_len, hidden_dim = features.shape
    num_keep = max(1, int(seq_len * retention_ratio))
    
    compressed_list = []
    indices_list = []
    
    for b in range(batch_size):
        feat = features[b]  # [seq_len, hidden_dim]
        
        if attention_scores is not None:
            scores = attention_scores[b]
        else:
            # Use feature norm as simple importance score
            scores = torch.norm(feat, dim=-1)
        
        # Select top tokens
        _, sorted_idx = torch.sort(scores, descending=True)
        keep_idx = sorted_idx[:num_keep].sort()[0]
        
        compressed = feat[keep_idx]
        compressed_list.append(compressed)
        indices_list.append(keep_idx)
    
    compressed_features = torch.stack(compressed_list, dim=0)
    keep_indices = torch.stack(indices_list, dim=0)
    
    return compressed_features, keep_indices
