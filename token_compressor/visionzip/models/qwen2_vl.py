from __future__ import annotations

import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast

from token_compressor.visionzip import compress_features




def _prefill_stage(past_key_values: Optional[Cache], cache_position: Optional[Tensor]) -> bool:
    if cache_position is not None and cache_position.numel() > 0 and int(cache_position[0]) != 0:
        return False
    if past_key_values is None:
        return True
    get_seq_length = getattr(past_key_values, "get_seq_length", None)
    return get_seq_length is None or get_seq_length() == 0


def _prune_attention(attn: Optional[Union[Tensor, dict]], keep_token_indices: Tensor) -> Optional[Union[Tensor, dict]]:
    if attn is None:
        return None
    if isinstance(attn, dict):
        return {k: _prune_attention(v, keep_token_indices) for k, v in attn.items()}
    if attn.dim() == 2:
        return attn[:, keep_token_indices]
    if attn.dim() == 4:
        return attn[:, :, keep_token_indices, :][:, :, :, keep_token_indices]
    return attn


def _split_sizes(video_grid_thw: Tensor, spatial_merge_size: int) -> List[int]:
    return (video_grid_thw.prod(-1) // (spatial_merge_size**2)).tolist()


def _compute_kept_video(
    video_embeds: Tensor,
    video_grid_thw: Tensor,
    spatial_merge_size: int,
    base_scale: float,
) -> Tuple[Tensor, Tensor, List[Tensor], List[Tensor]]:
    splits = torch.split(video_embeds, _split_sizes(video_grid_thw, spatial_merge_size))
    kept_indices: List[Tensor] = []
    kept_chunks: List[Tensor] = []
    local_keeps: List[Tensor] = []
    offset = 0

    for grid, feat in zip(video_grid_thw, splits):
        compressed_feat, keep_local = compress_features(feat, grid, spatial_merge_size, base_scale)
        keep_local = keep_local.to(dtype=torch.long, device=feat.device)
        compressed_feat = compressed_feat.to(device=feat.device, dtype=feat.dtype)
        if keep_local.numel() == 0 or compressed_feat.shape[0] != keep_local.numel():
            keep_local = torch.arange(feat.shape[0], device=feat.device, dtype=torch.long)
            compressed_feat = feat
        local_keeps.append(keep_local)
        kept_indices.append(keep_local + offset)
        kept_chunks.append(compressed_feat)
        offset += feat.shape[0]

    return torch.cat(kept_chunks, dim=0), torch.cat(kept_indices).sort().values, local_keeps, list(splits)


def _keep_token_indices_from_video_mask(video_mask: Tensor, kept_video_indices: Tensor, seq_len: int) -> Tensor:
    video_positions = video_mask[..., 0][0].nonzero(as_tuple=False).squeeze(-1)
    kept_video_positions = video_positions[kept_video_indices]
    all_positions = torch.arange(seq_len, device=video_mask.device)
    non_video_positions = all_positions[~video_mask[..., 0][0]]
    return torch.cat((non_video_positions, kept_video_positions)).sort().values


def Qwen2VL_ViT_forward(self, hidden_states: Tensor, grid_thw: Tensor) -> Tuple[Tensor, Tensor]:
    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    for blk in self.blocks:
        if self.gradient_checkpointing and self.training:
            hidden_states = self._gradient_checkpointing_func(
                blk.__call__, hidden_states, cu_seqlens, None, position_embeddings
            )
        else:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)

    return self.merger(hidden_states)


def Qwen2VLGeneration_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[Tensor]] = None,
    inputs_embeds: Optional[Tensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[Tensor] = None,
    pixel_values_videos: Optional[Tensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    keep_video_indices = None
    video_mask = None
    compression_on = (
        os.getenv("COMPRESSOR") == "visionzip"
        and pixel_values_videos is not None
        and video_grid_thw is not None
        and _prefill_stage(past_key_values, cache_position)
    )

    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)

        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.get_dtype())
            image_outputs = self.visual(pixel_values, grid_thw=image_grid_thw)
            image_embeds = image_outputs[0] if isinstance(image_outputs, tuple) else image_outputs
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds.to(inputs_embeds.device, inputs_embeds.dtype))

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
            visual_outputs = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            video_embeds = visual_outputs[0] if isinstance(visual_outputs, tuple) else visual_outputs
            video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            defer_video_scatter = compression_on and inputs_embeds.shape[0] == 1
            if defer_video_scatter:
                merge_size = int(getattr(self.visual, "spatial_merge_size", 2))
                base_scale = float(os.getenv("R_RATIO", "0.25"))
                video_embeds, keep_video_indices, _, _ = _compute_kept_video(video_embeds, video_grid_thw, merge_size, base_scale)
            else:
                keep_video_indices = torch.arange(video_embeds.shape[0], device=video_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds.to(inputs_embeds.device, inputs_embeds.dtype))

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        if _prefill_stage(past_key_values, cache_position) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0).to(position_ids.device)
            position_ids = position_ids.add(delta).unsqueeze(0).expand(3, -1, -1)

    if compression_on and keep_video_indices is not None and video_mask is not None and inputs_embeds.shape[0] == 1:
        keep_token_indices = _keep_token_indices_from_video_mask(video_mask, keep_video_indices, inputs_embeds.shape[1])
        inputs_embeds = inputs_embeds[:, keep_token_indices, :]
        input_ids = input_ids[:, keep_token_indices] if input_ids is not None else None
        labels = labels[:, keep_token_indices] if labels is not None else None
        attention_mask = _prune_attention(attention_mask, keep_token_indices)
        position_ids = position_ids[:, :, keep_token_indices] if position_ids is not None else None
        if cache_position is not None and cache_position.ndim == 1 and cache_position.numel() > int(keep_token_indices.max()):
            cache_position = cache_position[keep_token_indices]
        video_mask = video_mask[:, keep_token_indices, :]
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds.to(inputs_embeds.device, inputs_embeds.dtype))

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        logits = logits.float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
