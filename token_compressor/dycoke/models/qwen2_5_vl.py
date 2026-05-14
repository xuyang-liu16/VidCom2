from __future__ import annotations

import os
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from transformers.cache_utils import Cache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModelOutputWithPast,
    is_torchdynamo_compiling,
)

from token_compressor.dycoke import compress_features




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


def Qwen2_5_VLModel_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[Tensor] = None,
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
    second_per_grid_ts: Optional[Tensor] = None,
    **kwargs,
) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_mask = None
    video_mask = None

    if pixel_values is not None:
        image_embeds = torch.cat(self.get_image_features(pixel_values, image_grid_thw), dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds = torch.cat(self.get_video_features(pixel_values_videos, video_grid_thw), dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if position_ids is None:
        prefill_compiled_stage = is_torchdynamo_compiling() and ((input_ids is not None and input_ids.shape[1] != 1) or (inputs_embeds is not None and inputs_embeds.shape[1] != 1))
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and _prefill_stage(past_key_values, cache_position)
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids = position_ids + delta.to(position_ids.device)

    compression_on = (
        os.getenv("COMPRESSOR") == "dycoke"
        and pixel_values_videos is not None
        and video_grid_thw is not None
        and _prefill_stage(past_key_values, cache_position)
        and inputs_embeds.shape[0] == 1
    )

    if compression_on:
        merge_size = self.visual.spatial_merge_size
        base_scale = float(os.getenv("R_RATIO", "0.25"))
        video_embeds, kept_indices, _, _ = _compute_kept_video(video_embeds, video_grid_thw, merge_size, base_scale)
        keep_token_indices = _keep_token_indices_from_video_mask(video_mask, kept_indices, inputs_embeds.shape[1])

        inputs_embeds = inputs_embeds[:, keep_token_indices, :]
        input_ids = input_ids[:, keep_token_indices] if input_ids is not None else None
        attention_mask = _prune_attention(attention_mask, keep_token_indices)
        position_ids = position_ids[:, :, keep_token_indices]
        if cache_position is not None and cache_position.ndim == 1 and cache_position.numel() > int(keep_token_indices.max()):
            cache_position = cache_position[keep_token_indices]
        image_mask = image_mask[:, keep_token_indices, :] if image_mask is not None else None
        video_mask = video_mask[:, keep_token_indices, :]
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds.to(inputs_embeds.dtype))

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    return Qwen2_5_VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        rope_deltas=self.rope_deltas,
    )
