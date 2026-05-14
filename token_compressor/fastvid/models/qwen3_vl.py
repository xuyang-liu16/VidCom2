from __future__ import annotations

import os
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from transformers.cache_utils import Cache
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModelOutputWithPast,
    is_torchdynamo_compiling,
)

from token_compressor.fastvid import compress_features




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


def Qwen3VLModel_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[Tensor] = None,
    pixel_values: Optional[Tensor] = None,
    pixel_values_videos: Optional[Tensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[tuple, Qwen3VLModelOutputWithPast]:
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
        image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if position_ids is None:
        attention_mask_tensor = attention_mask if not isinstance(attention_mask, dict) else attention_mask.get("full_attention", None)
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        prefill_compiled_stage = is_torchdynamo_compiling() and ((input_ids is not None and input_ids.shape[1] != 1) or (inputs_embeds is not None and inputs_embeds.shape[1] != 1))
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and _prefill_stage(past_key_values, cache_position)

        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask=attention_mask_tensor)
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta).unsqueeze(0).expand(3, -1, -1)

    compression_on = (
        os.getenv("COMPRESSOR") == "fastvid"
        and pixel_values_videos is not None
        and video_grid_thw is not None
        and _prefill_stage(past_key_values, cache_position)
        and inputs_embeds.shape[0] == 1
    )

    if compression_on:
        merge_size = self.visual.spatial_merge_size
        base_scale = float(os.getenv("R_RATIO", "0.25"))
        video_embeds, kept_indices, local_keeps, video_splits = _compute_kept_video(video_embeds, video_grid_thw, merge_size, base_scale)

        if deepstack_video_embeds is not None:
            split_sizes = [split.shape[0] for split in video_splits]
            compressed_deepstack = []
            for layer_embeds in deepstack_video_embeds:
                layer_splits = torch.split(layer_embeds, split_sizes)
                compressed_deepstack.append(torch.cat([split[keep] for split, keep in zip(layer_splits, local_keeps)], dim=0))
            deepstack_video_embeds = compressed_deepstack

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
        visual_pos_masks = image_mask[..., 0]
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        visual_pos_masks = video_mask[..., 0]
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
