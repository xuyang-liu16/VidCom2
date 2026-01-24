"""
ToMe adaptation for Qwen3-VL model.

ToMe (Token Merging) applies bipartite soft matching INSIDE the Vision Transformer,
typically after every block or every few blocks.

Key difference from post-ViT compression:
- Post-ViT: Compress after all ViT blocks are processed
- ToMe: Merge r tokens after EVERY block (or specified blocks) inside ViT

Reference: https://github.com/facebookresearch/ToMe
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

from token_compressor.tome import tome_compression


def _downsample_features_to_target(
    features: Tensor,
    target_len: int,
) -> Tensor:
    """
    Downsample features to target length using uniform sampling.
    
    Args:
        features: Features to downsample [seq_len, hidden_dim]
        target_len: Target sequence length
        
    Returns:
        downsampled: Downsampled features [target_len, hidden_dim]
    """
    seq_len = features.shape[0]
    if seq_len == target_len:
        return features
    if seq_len < target_len:
        # If features are smaller than target, pad with last token
        padding = features[-1:].expand(target_len - seq_len, -1)
        return torch.cat([features, padding], dim=0)
    
    # Uniform sampling to downsample
    indices = torch.linspace(0, seq_len - 1, target_len, dtype=torch.long, device=features.device)
    return features[indices]


def _get_video_features_with_tome_compression(
    visual_model,
    pixel_values: Tensor,
    grid_thw: Tensor,
    r_per_layer: int,
    apply_every_n_layers: int = 1,
) -> Tuple[Tensor, List[Tensor]]:
    """
    Extract video features with ToMe merging applied INSIDE the ViT.
    
    Token merging is applied after every `apply_every_n_layers` blocks,
    merging `r_per_layer` tokens each time.
    
    Args:
        visual_model: Qwen3VLVisionModel instance
        pixel_values: Video pixel values
        grid_thw: Grid dimensions [num_videos, 3] (time, height, width)
        r_per_layer: Number of tokens to merge per application
        apply_every_n_layers: Apply ToMe every N layers (default: 1 = every layer)
        
    Returns:
        hidden_states: Compressed video features after all processing
        deepstack_features: List of deepstack features (all downsampled to final size)
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
    
    # Process through vision blocks with ToMe merging
    deepstack_feature_lists = []
    num_blocks = len(visual_model.blocks)
    
    for layer_num, blk in enumerate(visual_model.blocks):
        # Normal forward through the block
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        
        # Apply ToMe merging after this block
        if (layer_num + 1) % apply_every_n_layers == 0 and layer_num < num_blocks - 1:
            hidden_states, cu_seqlens, position_embeddings = \
                _apply_tome_merging_in_vit(
                    hidden_states,
                    cu_seqlens,
                    position_embeddings,
                    r_per_layer
                )
        
        # Collect deepstack features
        if layer_num in visual_model.deepstack_visual_indexes:
            deepstack_feature = visual_model.deepstack_merger_list[
                visual_model.deepstack_visual_indexes.index(layer_num)
            ](hidden_states)
            deepstack_feature_lists.append(deepstack_feature)
    
    # Final merger for hidden states
    hidden_states = visual_model.merger(hidden_states)
    
    # Downsample all deepstack features to match final token count
    # This ensures deepstack features match the compressed sequence length
    if deepstack_feature_lists:
        final_merged_len = hidden_states.shape[0]
        downsampled_deepstack = []
        for ds_feat in deepstack_feature_lists:
            downsampled_deepstack.append(
                _downsample_features_to_target(ds_feat, final_merged_len)
            )
        deepstack_feature_lists = downsampled_deepstack
    
    return hidden_states, deepstack_feature_lists


def _apply_tome_merging_in_vit(
    hidden_states: Tensor,
    cu_seqlens: Tensor,
    position_embeddings: Tuple[Tensor, Tensor],
    r: int,
) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]:
    """
    Apply ToMe bipartite soft matching inside the ViT.
    
    This is the core ToMe algorithm: split tokens into source and destination,
    find most similar pairs, and merge them.
    
    Args:
        hidden_states: Current hidden states [total_tokens, hidden_dim]
        cu_seqlens: Cumulative sequence lengths
        position_embeddings: (cos, sin) rotary embeddings
        r: Number of tokens to merge
        
    Returns:
        merged_hidden: Merged hidden states
        new_cu_seqlens: Updated cumulative sequence lengths
        new_position_embeddings: Updated position embeddings
    """
    total_tokens, hidden_dim = hidden_states.shape
    cos_emb, sin_emb = position_embeddings
    
    # Split by video segments
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    
    merged_segments = []
    new_lengths = []
    new_cos_list = []
    new_sin_list = []
    
    offset = 0
    for seg_len in lengths:
        # Extract segment
        segment = hidden_states[offset:offset + seg_len]
        seg_cos = cos_emb[offset:offset + seg_len]
        seg_sin = sin_emb[offset:offset + seg_len]
        
        # Calculate how many tokens to merge for this segment (proportional)
        seg_r = max(0, min(r, seg_len // 2))  # Can't merge more than half
        
        if seg_r > 0 and seg_len > 2:
            # Apply ToMe bipartite matching
            merged_segment, kept_indices = _bipartite_soft_matching_tome(
                segment, seg_r
            )
            new_len = merged_segment.shape[0]
            
            merged_segments.append(merged_segment)
            new_lengths.append(new_len)
            
            # Update position embeddings using kept indices
            new_cos_list.append(seg_cos[kept_indices])
            new_sin_list.append(seg_sin[kept_indices])
        else:
            # No merging for this segment
            merged_segments.append(segment)
            new_lengths.append(seg_len)
            new_cos_list.append(seg_cos)
            new_sin_list.append(seg_sin)
        
        offset += seg_len
    
    # Concatenate all merged segments
    merged_hidden = torch.cat(merged_segments, dim=0)
    
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
    
    return merged_hidden, new_cu_seqlens, new_position_embeddings


def _bipartite_soft_matching_tome(
    tokens: Tensor,
    r: int,
) -> Tuple[Tensor, Tensor]:
    """
    Apply ToMe's bipartite soft matching to merge r tokens.
    
    Args:
        tokens: Token features [seq_len, hidden_dim]
        r: Number of tokens to remove via merging
        
    Returns:
        merged_tokens: Merged tokens [seq_len - r, hidden_dim]
        kept_indices: Indices of tokens that are kept (for position tracking)
    """
    seq_len, hidden_dim = tokens.shape
    
    if r <= 0 or r >= seq_len // 2:
        return tokens, torch.arange(seq_len, device=tokens.device)
    
    # Normalize for cosine similarity
    tokens_norm = F.normalize(tokens, dim=-1)
    
    # Split into source (odd indices) and destination (even indices)
    # This alternating split ensures good spatial coverage
    dst_idx = torch.arange(0, seq_len, 2, device=tokens.device)  # Even indices
    src_idx = torch.arange(1, seq_len, 2, device=tokens.device)  # Odd indices
    
    num_dst = len(dst_idx)
    num_src = len(src_idx)
    
    if num_src == 0 or num_dst == 0:
        return tokens, torch.arange(seq_len, device=tokens.device)
    
    dst_tokens = tokens_norm[dst_idx]  # [num_dst, hidden]
    src_tokens = tokens_norm[src_idx]  # [num_src, hidden]
    
    # Compute similarity between source and destination
    similarity = src_tokens @ dst_tokens.T  # [num_src, num_dst]
    
    # For each source, find the most similar destination
    max_sim, max_dst_for_src = similarity.max(dim=-1)  # [num_src]
    
    # Sort source tokens by similarity (most similar first - these get merged)
    sorted_sim, sorted_src_order = max_sim.sort(descending=True)
    
    # Tokens to merge: top r source tokens
    merge_src_local = sorted_src_order[:r]  # local indices in src_idx
    merge_src_global = src_idx[merge_src_local]  # global indices
    merge_dst_local = max_dst_for_src[merge_src_local]  # which dst to merge into
    merge_dst_global = dst_idx[merge_dst_local]  # global indices
    
    # Tokens to keep from source (not merged)
    keep_src_local = sorted_src_order[r:]
    keep_src_global = src_idx[keep_src_local]
    
    # Perform merging: average merged source into destination
    merged_dst = tokens[dst_idx].clone()
    
    # Count merges per destination for weighted averaging
    for i in range(r):
        dst_local = merge_dst_local[i]
        src_global = merge_src_global[i]
        # Simple average: (dst + src) / 2
        merged_dst[dst_local] = (merged_dst[dst_local] + tokens[src_global]) / 2
    
    # Combine: merged destinations + kept sources
    kept_src_tokens = tokens[keep_src_global]
    
    # All kept indices (for position tracking)
    all_kept_indices = torch.cat([dst_idx, keep_src_global])
    all_kept_tokens = torch.cat([merged_dst, kept_src_tokens], dim=0)
    
    # Sort by original position to maintain spatial order
    sorted_positions, sort_order = all_kept_indices.sort()
    sorted_tokens = all_kept_tokens[sort_order]
    
    return sorted_tokens, sorted_positions


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
    """Patched forward with ToMe token merging INSIDE the ViT."""

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
            # Apply ToMe merging INSIDE the ViT
            retention_ratio = float(os.getenv("R_RATIO", "0.25"))
            
            # Calculate r (tokens to merge per layer) to achieve target retention
            # With N layers and r merges per layer, final tokens = original - N*r
            num_blocks = len(self.visual.blocks)
            merge_size = self.visual.spatial_merge_size
            original_tokens_per_video = (video_grid_thw.prod(-1) // merge_size**2).sum().item()
            target_tokens = int(original_tokens_per_video * retention_ratio)
            total_to_remove = original_tokens_per_video - target_tokens
            
            # TOME_R: tokens to merge per layer (overrides auto calculation)
            tome_r_env = os.getenv("TOME_R")
            if tome_r_env:
                r_per_layer = int(tome_r_env)
            else:
                # Auto calculate: distribute merges across layers
                # Apply every 2 layers to avoid too aggressive merging
                apply_every = int(os.getenv("TOME_APPLY_EVERY", "2"))
                num_merge_points = num_blocks // apply_every
                r_per_layer = max(1, total_to_remove // max(1, num_merge_points))
            
            apply_every_n_layers = int(os.getenv("TOME_APPLY_EVERY", "2"))
            
            pixel_values_videos_typed = pixel_values_videos.type(self.visual.dtype)
            video_embeds_raw, deepstack_video_embeds = _get_video_features_with_tome_compression(
                self.visual,
                pixel_values_videos_typed,
                video_grid_thw,
                r_per_layer,
                apply_every_n_layers
            )
            
            video_embeds = video_embeds_raw.to(inputs_embeds.device, inputs_embeds.dtype)
            
            # For ToMe compression, we cannot use get_placeholder_mask because
            # the number of video_embeds (compressed) doesn't match video tokens in input_ids.
            # We compute video_mask manually based on original video token positions.
            video_token_id = self.config.video_token_id
            video_mask = (input_ids == video_token_id)
            video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            # Note: inputs_embeds will be scattered later after sequence pruning
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
