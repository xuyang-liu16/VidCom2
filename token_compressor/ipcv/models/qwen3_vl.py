"""
IPCV adaptation for Qwen3-VL model.

IPCV (Information-Preserving Compression for MLLM Visual Encoders) applies
token compression INSIDE the Vision Transformer with multi-layer AS restoration.

Key mechanism (following the official implementation exactly):
1. Forward through layers 0 to K-2 normally
2. At layer K-1, save hidden_states_prev
3. At layer K, prune based on L2 norm of diff, save orig_kept and removed states
4. For layers K+1 to K+AS_layers: 
   - Restore full sequence using AS (removed + avg_delta_from_neighbors)
   - Process attention with full sequence
   - Re-prune to keep only kept tokens
5. Continue with pruned tokens for remaining layers
6. Final layer: restore full sequence using AS for output

Reference: https://github.com/Perkzi/IPCV
"""

from typing import Optional, Union, List, Tuple
import os
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers.cache_utils import Cache
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModelOutputWithPast,
    is_torchdynamo_compiling,
)

from token_compressor.ipcv.ipcv import IPCVState


def _forward_block_with_packed_states(
    block,
    hidden_states: Tensor,
    cu_seqlens: Tensor,
    position_embeddings: Tuple[Tensor, Tensor],
) -> Tensor:
    """Forward through a single vision block."""
    return block(
        hidden_states,
        cu_seqlens=cu_seqlens,
        position_embeddings=position_embeddings,
    )


def _update_cu_seqlens_after_pruning(
    cu_seqlens: Tensor,
    keep_indices: Tensor,
) -> Tensor:
    """
    Compute new cu_seqlens after pruning based on keep_indices.
    
    Args:
        cu_seqlens: Original cumulative sequence lengths [num_segments + 1]
        keep_indices: Sorted indices of kept tokens [num_keep]
        
    Returns:
        new_cu_seqlens: Updated cumulative sequence lengths
    """
    device = cu_seqlens.device
    num_frames = len(cu_seqlens) - 1
    
    # Count how many kept tokens fall in each frame
    frame_counts = torch.zeros(num_frames, dtype=torch.int32, device=device)
    frame_indices = torch.searchsorted(cu_seqlens[1:], keep_indices, right=False)
    frame_indices = frame_indices.clamp(min=0, max=num_frames - 1)
    
    for idx in frame_indices:
        frame_counts[idx] += 1
    
    # Build new cu_seqlens
    new_cu_seqlens = torch.zeros(num_frames + 1, dtype=torch.int32, device=device)
    new_cu_seqlens[1:] = frame_counts.cumsum(dim=0)
    
    return new_cu_seqlens


def _update_position_embeddings_after_pruning(
    position_embeddings: Tuple[Tensor, Tensor],
    keep_indices: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Select position embeddings for kept tokens."""
    cos_emb, sin_emb = position_embeddings
    return cos_emb[keep_indices], sin_emb[keep_indices]


def _get_video_features_with_ipcv_compression(
    visual_model,
    pixel_values: Tensor,
    grid_thw: Tensor,
    prune_layer: int,
    retention_ratio: float,
    as_layers: int,
    top_k: int,
) -> Tuple[Tensor, List[Tensor]]:
    """
    Extract video features with IPCV compression applied INSIDE the ViT.
    
    Following official IPCV implementation exactly.
    
    Args:
        visual_model: Qwen3VLVisionModel instance
        pixel_values: Video pixel values
        grid_thw: Grid dimensions [num_videos, 3] (time, height, width)
        prune_layer: Layer index at which to start pruning (K)
        retention_ratio: Ratio of tokens to keep
        as_layers: Number of layers to apply AS restoration
        top_k: Number of nearest neighbors for delta computation (official default: 10)
        
    Returns:
        hidden_states: Compressed video features after all processing
        deepstack_features: List of deepstack features (always full-size for compatibility)
    """
    # Patch embedding
    hidden_states = visual_model.patch_embed(pixel_values)
    
    # Position embedding interpolation
    pos_embeds = visual_model.fast_pos_embed_interpolate(grid_thw)
    hidden_states = hidden_states + pos_embeds
    
    # Rotary position embedding
    rotary_pos_emb = visual_model.rot_pos_emb(grid_thw)
    
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())
    
    # Compute cu_seqlens for variable length attention
    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    
    # Store original position info for AS restoration
    orig_cu_seqlens = cu_seqlens.clone()
    orig_position_embeddings = (position_embeddings[0].clone(), position_embeddings[1].clone())
    
    # Initialize IPCV state (official parameters)
    ipcv_state = IPCVState(
        retention_ratio=retention_ratio,
        top_k=top_k,
        as_layers=as_layers,
    )
    
    # Track states for pruning
    hidden_states_prev = None
    
    # Current pruned state tracking
    pruned_cu_seqlens = None
    pruned_position_embeddings = None
    
    # Process through vision blocks with IPCV compression
    deepstack_feature_lists = []
    num_blocks = len(visual_model.blocks)
    
    for layer_num, blk in enumerate(visual_model.blocks):
        # Save hidden states at layer K-1
        if layer_num == prune_layer - 1:
            hidden_states_prev = hidden_states.clone()
        
        # Determine which cu_seqlens and position_embeddings to use
        if ipcv_state.is_pruned and ipcv_state.should_apply_as():
            # AS mode: restore full sequence for attention
            hidden_states = ipcv_state.restore_for_as(hidden_states)
            
            # Process with full sequence (original cu_seqlens and positions)
            hidden_states = _forward_block_with_packed_states(
                blk,
                hidden_states,
                orig_cu_seqlens,
                orig_position_embeddings,
            )
            
            # Re-prune after block
            hidden_states = ipcv_state.re_prune_after_block(hidden_states)
            
        elif ipcv_state.is_pruned:
            # Past AS layers: use pruned sequence
            hidden_states = _forward_block_with_packed_states(
                blk,
                hidden_states,
                pruned_cu_seqlens,
                pruned_position_embeddings,
            )
        else:
            # Before pruning: normal forward
            hidden_states = _forward_block_with_packed_states(
                blk,
                hidden_states,
                cu_seqlens,
                position_embeddings,
            )
        
        # Apply IPCV pruning at layer K
        if layer_num == prune_layer and hidden_states_prev is not None:
            # Perform initial pruning (official: based on diff L2 norm)
            hidden_states = ipcv_state.prune(hidden_states, hidden_states_prev)
            
            # Update cu_seqlens and position_embeddings for pruned sequence
            pruned_cu_seqlens = _update_cu_seqlens_after_pruning(
                cu_seqlens, ipcv_state.keep_indices
            )
            pruned_position_embeddings = _update_position_embeddings_after_pruning(
                position_embeddings, ipcv_state.keep_indices
            )
        
        # Collect deepstack features
        # IMPORTANT: For IPCV, we must ensure deepstack features are full-size
        # to match visual_pos_masks in the language model
        if hasattr(visual_model, 'deepstack_visual_indexes') and \
           layer_num in visual_model.deepstack_visual_indexes:
            # If pruned, restore to full sequence before collecting deepstack features
            if ipcv_state.is_pruned:
                full_hidden_states = ipcv_state.restore_for_as(hidden_states)
                deepstack_feature = visual_model.deepstack_merger_list[
                    visual_model.deepstack_visual_indexes.index(layer_num)
                ](full_hidden_states)
            else:
                deepstack_feature = visual_model.deepstack_merger_list[
                    visual_model.deepstack_visual_indexes.index(layer_num)
                ](hidden_states)
            deepstack_feature_lists.append(deepstack_feature)
    
    # Final layer: restore full sequence using AS (following official)
    if ipcv_state.is_pruned:
        hidden_states = ipcv_state.final_restore(hidden_states)
    
    # Final merger for hidden states
    hidden_states = visual_model.merger(hidden_states)
    
    return hidden_states, deepstack_feature_lists


def Qwen3VLModel_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[tuple, Qwen3VLModelOutputWithPast]:
    """
    Patched forward with IPCV compression INSIDE the ViT.
    
    IPCV Configuration via environment variables:
    - R_RATIO: Token retention ratio (default: 0.25)
    - IPCV_LAYER: Layer at which to start pruning (default: num_blocks // 2)
    - IPCV_AS_LAYERS: Number of AS restoration layers (default: 4)
    - IPCV_TOP_K: Number of nearest neighbors for delta (official default: 10)
    """

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_mask = None
    video_mask = None
    deepstack_image_embeds = None
    deepstack_video_embeds = None

    if pixel_values is not None:
        image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    # Check if IPCV compression should be applied
    compression_on = (
        pixel_values_videos is not None
        and video_grid_thw is not None
        and (past_key_values is None or past_key_values.get_seq_length() == 0)
    )

    if compression_on:
        batch_size = inputs_embeds.shape[0]
        if batch_size != 1:
            compression_on = False

    if pixel_values_videos is not None:
        if compression_on:
            # IPCV configuration (following official defaults)
            retention_ratio = float(os.getenv("R_RATIO", "0.25"))
            num_blocks = len(self.visual.blocks)
            default_layer = num_blocks // 2
            prune_layer = int(os.getenv("IPCV_LAYER", str(default_layer)))
            as_layers = int(os.getenv("IPCV_AS_LAYERS", "4"))
            top_k = int(os.getenv("IPCV_TOP_K", "10"))  # Official default: 10
            
            # Apply IPCV compression INSIDE the ViT
            pixel_values_videos_typed = pixel_values_videos.type(self.visual.dtype)
            video_embeds_raw, deepstack_video_embeds = _get_video_features_with_ipcv_compression(
                self.visual,
                pixel_values_videos_typed,
                video_grid_thw,
                prune_layer,
                retention_ratio,
                as_layers,
                top_k,
            )
            
            video_embeds = video_embeds_raw.to(inputs_embeds.device, inputs_embeds.dtype)
        else:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    # Pre-compute position ids
    if position_ids is None:
        attention_mask_tensor = attention_mask if not isinstance(attention_mask, dict) else attention_mask.get(
            "full_attention", None
        )
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )

        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask_tensor,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    if compression_on:
        # Prune sequence to match compressed video tokens
        video_token_positions = video_mask[..., 0][0].nonzero(as_tuple=False).squeeze(-1)
        num_compressed_tokens = len(video_embeds)
        
        all_positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        non_video_mask = ~video_mask[..., 0][0]
        non_video_positions = all_positions[non_video_mask]
        
        kept_video_positions = video_token_positions[:num_compressed_tokens]
        keep_token_indices = torch.cat((non_video_positions, kept_video_positions)).sort().values

        def _prune_attention(attn: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if attn is None:
                return None
            if attn.dim() == 2:
                return attn[:, keep_token_indices]
            if attn.dim() == 4:
                return attn[:, :, keep_token_indices, :][:, :, :, keep_token_indices]
            return attn

        inputs_embeds = inputs_embeds[:, keep_token_indices, :]
        if input_ids is not None:
            input_ids = input_ids[:, keep_token_indices]
        attention_mask = (
            {k: _prune_attention(v) for k, v in attention_mask.items()}
            if isinstance(attention_mask, dict)
            else _prune_attention(attention_mask)
        )
        position_ids = position_ids[:, :, keep_token_indices]

        if image_mask is not None:
            image_mask = image_mask[:, keep_token_indices, :]
        if video_mask is not None:
            video_mask = video_mask[:, keep_token_indices, :]

        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds.to(inputs_embeds.dtype))

    # Build visual_pos_masks and deepstack_visual_embeds
    visual_pos_masks = None
    deepstack_visual_embeds = None
    if image_mask is not None and video_mask is not None:
        image_mask_compact = image_mask[..., 0]
        video_mask_compact = video_mask[..., 0]
        visual_pos_masks = image_mask_compact | video_mask_compact
        deepstack_visual_embeds = []
        image_mask_joint = image_mask_compact[visual_pos_masks]
        video_mask_joint = video_mask_compact[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        image_mask_compact = image_mask[..., 0]
        visual_pos_masks = image_mask_compact
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        video_mask_compact = video_mask[..., 0]
        visual_pos_masks = video_mask_compact
        deepstack_visual_embeds = deepstack_video_embeds

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        **kwargs,
    )

    return Qwen3VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        rope_deltas=self.rope_deltas,
    )
