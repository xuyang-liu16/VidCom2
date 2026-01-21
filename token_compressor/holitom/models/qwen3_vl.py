"""
HoliTom adaptation for Qwen3-VL model.

HoliTom applies holistic token merging with:
1. Temporal segmentation based on frame similarity
2. Static/dynamic feature separation
3. DPC-KNN clustering for token merging

This module patches the Qwen3VLModel forward method to apply HoliTom compression
after visual encoding, before feeding to the LLM.

Attention scores are extracted from the vision encoder's last layer for
more accurate token importance estimation.
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

from token_compressor.holitom import (
    holitom_compression
)


def _get_video_features_with_attention(
    visual_model,
    pixel_values: Tensor,
    grid_thw: Tensor,
) -> Tuple[Tensor, List[Tensor], Tensor]:
    """
    Extract video features along with attention scores from the last vision encoder layer.
    
    Args:
        visual_model: Qwen3VLVisionModel instance
        pixel_values: Video pixel values
        grid_thw: Grid dimensions [num_videos, 3] (time, height, width)
        
    Returns:
        hidden_states: Encoded video features [merged_seq_len, hidden_dim]
        deepstack_features: List of deepstack features from intermediate layers
        attention_scores: Attention scores from last layer [merged_seq_len] (after spatial merge)
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
    
    # Process through vision blocks
    deepstack_feature_lists = []
    attention_weights = None
    num_blocks = len(visual_model.blocks)
    
    for layer_num, blk in enumerate(visual_model.blocks):
        # For the last layer, we need to capture attention weights
        if layer_num == num_blocks - 1:
            # Manually execute the last block with attention output
            attention_weights = _forward_vision_block_with_attention(
                blk, hidden_states, cu_seqlens, position_embeddings
            )
            # Also run the normal forward to get hidden states
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
        else:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
        
        # Collect deepstack features
        if layer_num in visual_model.deepstack_visual_indexes:
            deepstack_feature = visual_model.deepstack_merger_list[
                visual_model.deepstack_visual_indexes.index(layer_num)
            ](hidden_states)
            deepstack_feature_lists.append(deepstack_feature)
    
    # Final merger for hidden states
    hidden_states = visual_model.merger(hidden_states)
    
    # Process attention weights and apply spatial merge to match merged hidden states
    attention_scores = _compute_attention_scores(attention_weights)
    
    # Apply spatial merge to attention scores (same as hidden states merger)
    spatial_merge_size = visual_model.spatial_merge_size
    attention_scores = _spatial_merge_attention_scores(
        attention_scores, grid_thw, spatial_merge_size
    )
    
    return hidden_states, deepstack_feature_lists, attention_scores


def _forward_vision_block_with_attention(
    block,
    hidden_states: Tensor,
    cu_seqlens: Tensor,
    position_embeddings: Tuple[Tensor, Tensor],
) -> Tensor:
    """
    Execute a vision block and capture attention weights.
    
    Args:
        block: Qwen3VLVisionBlock instance
        hidden_states: Input hidden states [seq_len, hidden_dim]
        cu_seqlens: Cumulative sequence lengths
        position_embeddings: Rotary position embeddings (cos, sin)
        
    Returns:
        attention_weights: Attention weights from this block [seq_len]
    """
    attn = block.attn
    seq_length = hidden_states.shape[0]
    
    # Apply input norm
    normed_states = block.norm1(hidden_states)
    
    # Project to Q, K, V
    query_states, key_states, value_states = (
        attn.qkv(normed_states)
        .reshape(seq_length, 3, attn.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    
    # Apply rotary position embeddings
    cos, sin = position_embeddings
    query_states, key_states = _apply_rotary_pos_emb_vision(
        query_states, key_states, cos, sin
    )
    
    # Reshape for attention computation
    query_states = query_states.transpose(0, 1).unsqueeze(0)  # [1, num_heads, seq_len, head_dim]
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    
    # Compute attention weights manually using eager attention
    scaling = attn.head_dim ** -0.5
    
    # Process each chunk separately based on cu_seqlens
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    
    all_attn_weights = []
    offset = 0
    for length in lengths:
        q_chunk = query_states[:, :, offset:offset+length, :]
        k_chunk = key_states[:, :, offset:offset+length, :]
        
        # Compute attention: [1, num_heads, length, length]
        attn_weights = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * scaling
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        
        # Average over heads: [length, length]
        attn_weights = attn_weights[0].mean(dim=0)
        
        # Sum attention received by each token (column sum): [length]
        token_attn = attn_weights.sum(dim=0)
        all_attn_weights.append(token_attn)
        
        offset += length
    
    # Concatenate all chunks
    attention_weights = torch.cat(all_attn_weights, dim=0)  # [seq_len]
    
    return attention_weights


def _apply_rotary_pos_emb_vision(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
) -> Tuple[Tensor, Tensor]:
    """Apply rotary position embedding to query and key tensors."""
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    
    # Rotate half
    q1, q2 = q[..., :q.shape[-1]//2], q[..., q.shape[-1]//2:]
    k1, k2 = k[..., :k.shape[-1]//2], k[..., k.shape[-1]//2:]
    q_rotated = torch.cat((-q2, q1), dim=-1)
    k_rotated = torch.cat((-k2, k1), dim=-1)
    
    q_embed = (q * cos) + (q_rotated * sin)
    k_embed = (k * cos) + (k_rotated * sin)
    
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


def _compute_attention_scores(attention_weights: Tensor) -> Tensor:
    """
    Compute per-token attention scores from attention weights.
    
    Args:
        attention_weights: Raw attention weights [seq_len]
        
    Returns:
        attention_scores: Normalized attention scores [seq_len]
    """
    min_val = attention_weights.min()
    max_val = attention_weights.max()
    if max_val - min_val > 1e-8:
        attention_scores = (attention_weights - min_val) / (max_val - min_val)
    else:
        attention_scores = torch.ones_like(attention_weights)
    
    return attention_scores


def _spatial_merge_attention_scores(
    attention_scores: Tensor,
    grid_thw: Tensor,
    spatial_merge_size: int,
) -> Tensor:
    """
    Apply spatial merge to attention scores to match the merged hidden states.
    
    Args:
        attention_scores: Pre-merge attention scores [total_pre_merge_tokens]
        grid_thw: Grid dimensions [num_videos, 3] (time, height, width)
        spatial_merge_size: Size of spatial merge (typically 2)
        
    Returns:
        merged_scores: Post-merge attention scores [total_post_merge_tokens]
    """
    merge_unit = spatial_merge_size ** 2
    merged_scores_list = []
    
    offset = 0
    for t, h, w in grid_thw.tolist():
        num_frames = t
        tokens_per_frame = h * w
        merged_h = h // spatial_merge_size
        merged_w = w // spatial_merge_size
        merged_tokens_per_frame = merged_h * merged_w
        
        for frame_idx in range(num_frames):
            frame_start = offset + frame_idx * tokens_per_frame
            frame_scores = attention_scores[frame_start:frame_start + tokens_per_frame]
            
            # Reshape to [h, w]
            frame_scores_2d = frame_scores.view(h, w)
            
            # Reshape to merge blocks: [merged_h, merge_size, merged_w, merge_size]
            frame_scores_blocks = frame_scores_2d.view(
                merged_h, spatial_merge_size,
                merged_w, spatial_merge_size
            )
            
            # Permute to [merged_h, merged_w, merge_size, merge_size]
            frame_scores_blocks = frame_scores_blocks.permute(0, 2, 1, 3)
            
            # Reshape to [merged_h * merged_w, merge_size * merge_size]
            frame_scores_blocks = frame_scores_blocks.reshape(merged_tokens_per_frame, merge_unit)
            
            # Average over the merge unit
            merged_frame_scores = frame_scores_blocks.mean(dim=-1)
            
            merged_scores_list.append(merged_frame_scores)
        
        offset += num_frames * tokens_per_frame
    
    merged_scores = torch.cat(merged_scores_list, dim=0)
    return merged_scores


def _compress_video_tokens(
    video_embeds: Tensor,
    attention_scores: Tensor,
    grid_thw: Tensor,
    spatial_merge_size: int,
    retain_ratio: float,
    tau: float,
    beta: float,
    D: float,
    K: int,
    max_window_size: int,
) -> Tuple[Tensor, Tensor]:
    """
    Apply HoliTom compression to video tokens using real attention scores.
    
    Args:
        video_embeds: Video token embeddings [total_tokens, hidden_dim]
        attention_scores: Attention scores from vision encoder [total_tokens]
        grid_thw: Grid dimensions [num_videos, 3] (time, height, width)
        spatial_merge_size: Spatial merge size from visual encoder
        retain_ratio: Ratio of tokens to retain
        tau: Similarity threshold for static detection
        beta: Clustering merge weight
        D: Dominant token ratio
        K: KNN neighbors
        max_window_size: Maximum temporal window size
        
    Returns:
        compressed_embeds: Compressed video tokens
        keep_indices: Indices of kept tokens (approximate)
    """
    # Split video embeddings and attention scores by video
    split_sizes = (grid_thw.prod(-1) // spatial_merge_size**2).tolist()
    video_splits = torch.split(video_embeds, split_sizes)
    attn_splits = torch.split(attention_scores, split_sizes)
    
    compressed_chunks = []
    kept_indices_list = []
    offset = 0
    
    for vid_idx, (video_feat, video_attn, grid) in enumerate(zip(video_splits, attn_splits, grid_thw)):
        t, h, w = grid.tolist()
        frame_tokens = (h * w) // (spatial_merge_size ** 2)
        num_frames = t
        
        if frame_tokens <= 0 or video_feat.numel() == 0:
            compressed_chunks.append(video_feat)
            kept_indices_list.append(torch.arange(len(video_feat), device=video_feat.device) + offset)
            offset += len(video_feat)
            continue
        
        # Reshape to [num_frames, frame_tokens, hidden_dim]
        video_feat_3d = video_feat.view(num_frames, frame_tokens, -1)
        
        # Reshape attention scores to [num_frames, frame_tokens]
        attn_weights_3d = video_attn.view(num_frames, frame_tokens)
        
        # Apply HoliTom compression with real attention scores
        compressed, local_indices = holitom_compression(
            video_feat_3d,
            attn_weights=attn_weights_3d,  # Pass real attention scores
            retain_ratio=retain_ratio,
            tau=tau,
            beta=beta,
            D=D,
            K=K,
            max_window_size=max_window_size
        )
        
        compressed_chunks.append(compressed)
        kept_indices_list.append(local_indices + offset)
        offset += len(video_feat)
    
    # Concatenate all compressed chunks
    compressed_embeds = torch.cat(compressed_chunks, dim=0)
    keep_indices = torch.cat(kept_indices_list, dim=0)
    
    return compressed_embeds, keep_indices


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
    """Patched forward that enables HoliTom token compression for Qwen3-VL.
    
    HoliTom compresses video tokens by:
    1. Extracting attention scores from the vision encoder's last layer
    2. Temporal segmentation based on frame similarity
    3. Static/dynamic feature separation
    4. DPC-KNN clustering for token merging
    """

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_mask = None
    video_mask = None
    deepstack_image_embeds = None
    deepstack_video_embeds = None
    video_attention_scores = None

    if pixel_values is not None:
        image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    # Check if HoliTom compression should be applied
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
            # Use custom feature extraction to get attention scores
            pixel_values_videos_typed = pixel_values_videos.type(self.visual.dtype)
            video_embeds_raw, deepstack_video_embeds, video_attention_scores = \
                _get_video_features_with_attention(
                    self.visual,
                    pixel_values_videos_typed,
                    video_grid_thw
                )
            
            # Split embeddings like the original implementation
            split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
            video_embeds_list = torch.split(video_embeds_raw, split_sizes)
            video_embeds = torch.cat(video_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        else:
            # Normal path without compression
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        
        # Get video_mask BEFORE compression, using original video_embeds
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    # Pre-compute position ids before pruning so we can safely slice them later.
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

    if compression_on and video_attention_scores is not None:
        # Get HoliTom hyperparameters from environment
        retain_ratio = float(os.getenv("R_RATIO", "0.15"))
        tau = float(os.getenv("HOLITOM_T", "0.8"))
        beta = float(os.getenv("HOLITOM_BETA", "0.6"))
        D = float(os.getenv("HOLITOM_D", "0.0"))
        K = int(os.getenv("HOLITOM_K", "7"))
        max_window_size = int(os.getenv("HOLITOM_MAX_WINDOW_SIZE", "1024"))
        merge_size = self.visual.spatial_merge_size
        
        # Apply HoliTom compression with real attention scores
        compressed_video, keep_indices = _compress_video_tokens(
            video_embeds,
            video_attention_scores.to(video_embeds.device),
            video_grid_thw,
            merge_size,
            retain_ratio,
            tau,
            beta,
            D,
            K,
            max_window_size
        )
        
        # Also compress deepstack embeddings if present
        if deepstack_video_embeds is not None:
            split_sizes = (video_grid_thw.prod(-1) // merge_size**2).tolist()
            
            # For each layer in deepstack, we need to compress similarly
            # Since HoliTom changes the structure, we approximate by keeping proportional tokens
            compressed_deepstack = []
            total_original = sum(split_sizes)
            total_compressed = len(compressed_video)
            
            for layer_embeds in deepstack_video_embeds:
                # Simple approximation: uniformly sample to match compressed size
                if total_original > 0 and total_compressed > 0:
                    indices = torch.linspace(
                        0, len(layer_embeds) - 1, total_compressed, 
                        dtype=torch.long, device=layer_embeds.device
                    )
                    compressed_deepstack.append(layer_embeds[indices])
                else:
                    compressed_deepstack.append(layer_embeds)
            deepstack_video_embeds = compressed_deepstack
        
        video_embeds = compressed_video
        
        # Now prune inputs_embeds, attention_mask, position_ids, video_mask
        video_token_positions = video_mask[..., 0][0].nonzero(as_tuple=False).squeeze(-1)
        num_original_video_tokens = len(video_token_positions)
        num_compressed_tokens = len(video_embeds)
        
        # Create new indices: keep non-video tokens + first num_compressed_tokens video positions
        all_positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        non_video_mask = ~video_mask[..., 0][0]
        non_video_positions = all_positions[non_video_mask]
        
        # Keep first num_compressed_tokens video positions
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

        # Re-scatter compressed video embeddings
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
