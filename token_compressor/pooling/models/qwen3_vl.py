"""
Pooling adaptation for Qwen3-VL model.

Pooling applies spatial pooling INSIDE the Vision Transformer at a specific layer,
reducing the spatial resolution of feature maps before continuing with remaining layers.

Key difference from post-ViT compression:
- Post-ViT: Pool after all ViT blocks are processed
- This impl: Pool INSIDE ViT at layer K, then continue with remaining layers
"""

from typing import Optional, Union, List, Tuple
import os
import math
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers.cache_utils import Cache
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModelOutputWithPast,
    is_torchdynamo_compiling,
)

from token_compressor.pooling import pooling_compression


def _get_video_features_with_pooling_compression(
    visual_model,
    pixel_values: Tensor,
    grid_thw: Tensor,
    compression_layer: int,
    retention_ratio: float,
    pool_type: str,
) -> Tuple[Tensor, List[Tensor]]:
    """
    Extract video features with Pooling applied INSIDE the ViT.
    
    Args:
        visual_model: Qwen3VLVisionModel instance
        pixel_values: Video pixel values
        grid_thw: Grid dimensions [num_videos, 3] (time, height, width)
        compression_layer: Layer index after which to apply pooling
        retention_ratio: Ratio of tokens to keep
        pool_type: Type of pooling ("avg", "max", "stride")
        
    Returns:
        hidden_states: Compressed video features after all processing
        deepstack_features: List of deepstack features
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
    
    # Track compression state
    compression_applied = False
    
    # Process through vision blocks with pooling in the middle
    deepstack_feature_lists = []
    num_blocks = len(visual_model.blocks)
    
    for layer_num, blk in enumerate(visual_model.blocks):
        # Normal forward through the block
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        
        # Apply pooling after the specified layer
        if layer_num == compression_layer and not compression_applied:
            hidden_states, cu_seqlens, position_embeddings = \
                _apply_pooling_in_vit(
                    hidden_states,
                    cu_seqlens,
                    position_embeddings,
                    grid_thw,
                    retention_ratio,
                    pool_type
                )
            compression_applied = True
        
        # Collect deepstack features (after potential compression)
        if layer_num in visual_model.deepstack_visual_indexes:
            deepstack_feature = visual_model.deepstack_merger_list[
                visual_model.deepstack_visual_indexes.index(layer_num)
            ](hidden_states)
            deepstack_feature_lists.append(deepstack_feature)
    
    # Final merger for hidden states
    hidden_states = visual_model.merger(hidden_states)
    
    return hidden_states, deepstack_feature_lists


def _apply_pooling_in_vit(
    hidden_states: Tensor,
    cu_seqlens: Tensor,
    position_embeddings: Tuple[Tensor, Tensor],
    grid_thw: Tensor,
    retention_ratio: float,
    pool_type: str,
) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]:
    """
    Apply spatial pooling inside the ViT.
    
    This pools each video segment spatially while preserving temporal structure.
    
    Args:
        hidden_states: Current hidden states [total_tokens, hidden_dim]
        cu_seqlens: Cumulative sequence lengths
        position_embeddings: (cos, sin) rotary embeddings
        grid_thw: Grid dimensions for computing spatial layout
        retention_ratio: Ratio of tokens to keep
        pool_type: Type of pooling
        
    Returns:
        pooled_hidden: Pooled hidden states
        new_cu_seqlens: Updated cumulative sequence lengths
        new_position_embeddings: Updated position embeddings
    """
    total_tokens, hidden_dim = hidden_states.shape
    cos_emb, sin_emb = position_embeddings
    
    # Split by video segments
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    
    pooled_segments = []
    new_lengths = []
    new_cos_list = []
    new_sin_list = []
    
    offset = 0
    for seg_idx, seg_len in enumerate(lengths):
        # Extract segment
        segment = hidden_states[offset:offset + seg_len]
        seg_cos = cos_emb[offset:offset + seg_len]
        seg_sin = sin_emb[offset:offset + seg_len]
        
        # Calculate target length
        target_len = max(1, int(seg_len * retention_ratio))
        
        # Apply pooling
        segment_batch = segment.unsqueeze(0)
        pooled, _ = pooling_compression(
            segment_batch,
            retention_ratio=retention_ratio,
            pool_type=pool_type
        )
        
        pooled_segment = pooled[0]
        new_len = pooled_segment.shape[0]
        
        pooled_segments.append(pooled_segment)
        new_lengths.append(new_len)
        
        # Update position embeddings by uniform sampling
        if new_len < seg_len:
            pos_indices = torch.linspace(0, seg_len - 1, new_len, dtype=torch.long, device=hidden_states.device)
        else:
            pos_indices = torch.arange(new_len, device=hidden_states.device)
        
        new_cos_list.append(seg_cos[pos_indices])
        new_sin_list.append(seg_sin[pos_indices])
        
        offset += seg_len
    
    # Concatenate all pooled segments
    pooled_hidden = torch.cat(pooled_segments, dim=0)
    
    # Update cu_seqlens
    new_cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(new_lengths), dim=0).tolist()),
        dtype=torch.int32,
        device=cu_seqlens.device
    )
    
    # Update position embeddings
    new_cos = torch.cat(new_cos_list, dim=0)
    new_sin = torch.cat(new_sin_list, dim=0)
    new_position_embeddings = (new_cos, new_sin)
    
    return pooled_hidden, new_cu_seqlens, new_position_embeddings


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
    """Patched forward with Pooling compression INSIDE the ViT."""

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
            # Apply Pooling INSIDE the ViT
            retention_ratio = float(os.getenv("R_RATIO", "0.25"))
            pool_type = os.getenv("POOLING_TYPE", "avg")
            
            # POOLING_LAYER: which layer to apply pooling after (default: middle layer)
            num_blocks = len(self.visual.blocks)
            default_layer = num_blocks // 2
            compression_layer = int(os.getenv("POOLING_LAYER", str(default_layer)))
            
            pixel_values_videos_typed = pixel_values_videos.type(self.visual.dtype)
            video_embeds_raw, deepstack_video_embeds = _get_video_features_with_pooling_compression(
                self.visual,
                pixel_values_videos_typed,
                video_grid_thw,
                compression_layer,
                retention_ratio,
                pool_type
            )
            
            video_embeds = video_embeds_raw.to(inputs_embeds.device, inputs_embeds.dtype)
        else:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

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
