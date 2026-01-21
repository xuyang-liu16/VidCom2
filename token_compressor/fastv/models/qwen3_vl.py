"""
FastV adaptation for Qwen3-VL model.

FastV compresses video tokens AFTER LLM layer K (default K=2) based on attention scores.
This is different from VidCom2/VisionZip which compress before entering the LLM.

Implementation Strategy:
1. Replace Qwen3VLTextModel.forward to enable layer-by-layer processing
2. At layer K, directly call self_attn to get attention weights (since decoder layer discards them)
3. Prune video tokens based on attention scores
4. Continue forward pass with remaining layers using pruned sequence

IMPORTANT: Must use attn_implementation=eager to get attention weights.
"""

from typing import Optional, Union, List
import os
import torch
from torch import Tensor
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModelOutputWithPast,
    is_torchdynamo_compiling,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.masking_utils import create_causal_mask


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
    """Patched forward that enables FastV token compression for Qwen3-VL.
    
    FastV prunes video tokens after LLM layer K based on attention scores.
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

    if pixel_values_videos is not None:
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

    # Check if FastV compression should be applied
    compression_on = (
        pixel_values_videos is not None
        and video_grid_thw is not None
        and (past_key_values is None or past_key_values.get_seq_length() == 0)
    )

    if compression_on:
        batch_size = inputs_embeds.shape[0]
        if batch_size != 1:
            compression_on = False

    # Build deepstack visual masks before potential pruning
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

    if compression_on:
        # FastV: Run LLM with custom layer-by-layer execution
        layer_k = int(os.getenv("FASTV_K", "2"))
        retention_ratio = float(os.getenv("R_RATIO", "0.5"))
        
        # Get video token mask
        video_token_mask = video_mask[0, :, 0].bool()  # [seq_len]
        
        outputs = _fastv_text_model_forward(
            self.language_model,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            video_token_mask=video_token_mask,
            layer_k=layer_k,
            retention_ratio=retention_ratio,
            **kwargs,
        )
    else:
        # No compression, run normally
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


def _fastv_text_model_forward(
    text_model,
    inputs_embeds: Tensor,
    position_ids: Tensor,
    attention_mask: Optional[Tensor],
    past_key_values: Optional[Cache],
    cache_position: Optional[Tensor],
    visual_pos_masks: Optional[Tensor],
    deepstack_visual_embeds: Optional[List[Tensor]],
    video_token_mask: Tensor,
    layer_k: int,
    retention_ratio: float,
    **kwargs,
) -> BaseModelOutputWithPast:
    """
    Custom forward for Qwen3VLTextModel that applies FastV compression after layer K.
    
    Key insight: Qwen3VLTextDecoderLayer.forward() discards attention weights.
    We need to manually call self_attn to get attention weights at layer K.
    
    Args:
        text_model: The Qwen3VLTextModel instance
        inputs_embeds: Input embeddings [batch, seq_len, hidden_dim]
        position_ids: Position IDs [3, batch, seq_len] for Qwen3-VL MRoPE
        attention_mask: Attention mask (2D or 4D)
        past_key_values: KV cache
        cache_position: Cache position indices
        visual_pos_masks: Mask indicating visual token positions [batch, seq_len]
        deepstack_visual_embeds: List of visual embeddings for deepstack
        video_token_mask: Boolean mask for video tokens [seq_len]
        layer_k: Layer after which to apply pruning
        retention_ratio: Ratio of video tokens to keep
    """
    lm = text_model
    use_cache = kwargs.get("use_cache", True)
    
    # Initialize cache
    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()
    
    # Initialize cache_position if needed
    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    
    # Get text position ids (first dimension of position_ids for Qwen3-VL MRoPE)
    # position_ids shape: [3, batch, seq_len] for temporal, height, width
    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        mrope_position_ids = position_ids[1:]
    elif position_ids.ndim == 3:
        text_position_ids = position_ids[0]
        mrope_position_ids = position_ids
    else:
        text_position_ids = position_ids
        mrope_position_ids = position_ids
    
    # Create causal mask
    causal_mask = create_causal_mask(
        config=lm.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )
    
    hidden_states = inputs_embeds
    
    # Compute position embeddings (rotary embeddings)
    position_embeddings = lm.rotary_emb(hidden_states, mrope_position_ids)
    
    # Track compression state
    compression_applied = False
    keep_indices = None
    
    # Process layers one by one
    for layer_idx, decoder_layer in enumerate(lm.layers):
        # Apply deepstack visual embeddings if available
        if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
            if visual_pos_masks is not None:
                visual_embeds = deepstack_visual_embeds[layer_idx]
                hidden_states = _deepstack_process(hidden_states, visual_pos_masks, visual_embeds)
        
        # At layer K, we need to manually call self_attn to get attention weights
        if layer_idx == layer_k and not compression_applied and video_token_mask.sum() > 0:
            # Manually execute decoder layer logic to get attention weights
            residual = hidden_states
            hidden_states = decoder_layer.input_layernorm(hidden_states)
            
            # Directly call self_attn to get attention weights
            # This is the key difference - decoder_layer.forward() discards attn_weights
            attn_output, attn_weights = decoder_layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            
            hidden_states = residual + attn_output
            
            # MLP part
            residual = hidden_states
            hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
            hidden_states = decoder_layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
            
            # Now apply FastV pruning based on attention weights
            if attn_weights is not None:
                # Compute per-token attention scores
                # attn_weights shape: [batch, num_heads, seq_len, seq_len]
                # Average across heads and aggregate attention received by each token
                token_scores = attn_weights.mean(dim=1).mean(dim=1)  # [batch, seq_len]
                
                # Get video token indices and their scores
                video_indices = video_token_mask.nonzero(as_tuple=False).squeeze(-1)
                video_scores = token_scores[0, video_indices]  # [num_video_tokens]
                
                # Determine how many video tokens to keep
                num_video_tokens = video_indices.shape[0]
                num_keep = max(1, int(num_video_tokens * retention_ratio))
                
                # Select top-k video tokens by attention score
                _, topk_local_indices = torch.topk(video_scores, k=num_keep, largest=True)
                topk_local_indices = torch.sort(topk_local_indices).values
                kept_video_indices = video_indices[topk_local_indices]
                
                # Build final keep indices (all non-video + selected video tokens)
                non_video_indices = (~video_token_mask).nonzero(as_tuple=False).squeeze(-1)
                keep_indices = torch.cat([non_video_indices, kept_video_indices]).sort().values
                
                # Prune hidden states
                hidden_states = hidden_states[:, keep_indices, :]
                
                # Prune text_position_ids
                text_position_ids = text_position_ids[:, keep_indices]
                
                # Prune mrope_position_ids
                mrope_position_ids = mrope_position_ids[:, :, keep_indices]
                
                # Update cache_position
                if cache_position is not None:
                    cache_position = cache_position[keep_indices]
                
                # Recreate causal mask for pruned sequence
                causal_mask = create_causal_mask(
                    config=lm.config,
                    input_embeds=hidden_states,
                    attention_mask=None,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=text_position_ids,
                )
                
                # Recompute position embeddings for pruned sequence
                position_embeddings = lm.rotary_emb(hidden_states, mrope_position_ids)
                
                # Prune KV cache for layers 0 to K
                if past_key_values is not None:
                    _prune_kv_cache(past_key_values, keep_indices, layer_k + 1)
                
                # Update video_token_mask for subsequent processing
                new_video_mask = torch.zeros(len(keep_indices), dtype=torch.bool, device=video_token_mask.device)
                kept_video_new_positions = torch.searchsorted(keep_indices, kept_video_indices)
                new_video_mask[kept_video_new_positions] = True
                video_token_mask = new_video_mask
                
                # Update visual_pos_masks and deepstack_visual_embeds
                if visual_pos_masks is not None:
                    old_visual_positions = visual_pos_masks[0].nonzero(as_tuple=False).squeeze(-1)
                    kept_visual_mask = torch.isin(old_visual_positions, keep_indices)
                    visual_pos_masks = visual_pos_masks[:, keep_indices]
                    if deepstack_visual_embeds is not None:
                        deepstack_visual_embeds = [
                            embeds[kept_visual_mask] for embeds in deepstack_visual_embeds
                        ]
                
                compression_applied = True
        else:
            # Normal layer forward - use the standard decoder layer
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
    
    # Final layer norm
    hidden_states = lm.norm(hidden_states)
    
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


def _deepstack_process(
    hidden_states: Tensor, 
    visual_pos_masks: Tensor, 
    visual_embeds: Tensor
) -> Tensor:
    """Apply deepstack visual embeddings to hidden states."""
    if visual_pos_masks is None or visual_embeds is None:
        return hidden_states
    visual_pos_masks = visual_pos_masks.to(hidden_states.device)
    visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
    # hidden_states shape: [batch, seq, hidden]
    # visual_pos_masks shape: [batch, seq]
    hidden_states = hidden_states.clone()
    local_hidden = hidden_states[visual_pos_masks, :].clone() + visual_embeds
    hidden_states[visual_pos_masks, :] = local_hidden
    return hidden_states


def _prune_kv_cache(cache: Cache, keep_indices: Tensor, num_layers: int):
    """Prune KV cache for the first num_layers layers."""
    if not hasattr(cache, 'key_cache') or not hasattr(cache, 'value_cache'):
        return
    
    for layer_idx in range(min(num_layers, len(cache.key_cache))):
        if cache.key_cache[layer_idx] is not None:
            # KV cache shape: [batch, num_heads, seq_len, head_dim]
            cache.key_cache[layer_idx] = cache.key_cache[layer_idx][:, :, keep_indices, :]
            cache.value_cache[layer_idx] = cache.value_cache[layer_idx][:, :, keep_indices, :]
