from typing import Optional, Union, List
import os
import torch
from torch import Tensor
from transformers.cache_utils import Cache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModelOutputWithPast,
    is_torchdynamo_compiling,
)

from token_compressor.vidcom2 import (
    select_low_var_channels,
    compute_gaussian_scores,
    compute_scales,
    select_outlier_indices,
    _map_linear_offset,
)


def _compute_keep_indices(
    flat_features: Tensor, grid_thw: Tensor, spatial_merge_size: int, base_scale: float
) -> Tensor:
    """Runs VidCom2 scoring to obtain kept token indices for a single video."""
    t, h, w = grid_thw.tolist()
    frame_tokens = (h * w) // (spatial_merge_size**2)
    if frame_tokens <= 0 or flat_features.numel() == 0:
        return torch.arange(flat_features.shape[0], device=flat_features.device)

    sel_feat = select_low_var_channels(flat_features)
    vid_score, frame_score = compute_gaussian_scores(sel_feat, frame_tokens)
    scales = compute_scales(-vid_score.mean(dim=-1), base_scale)
    indices = select_outlier_indices(vid_score + frame_score, scales, frame_tokens)
    return _map_linear_offset(indices, frame_tokens)


def Qwen2_5_VLModel_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
    """Patched forward that enables VidCom2 token compression for Qwen2.5-VL."""

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_mask = None
    video_mask = None

    if pixel_values is not None:
        image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    # Pre-compute position ids before pruning so we can safely slice them later.
    if position_ids is None:
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
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids = position_ids + delta.to(position_ids.device)

    compression_on = (
        os.getenv("COMPRESSOR") == "vidcom2"
        and pixel_values_videos is not None
        and video_grid_thw is not None
        and (past_key_values is None or past_key_values.get_seq_length() == 0)
    )

    if compression_on:
        batch_size = inputs_embeds.shape[0]
        if batch_size != 1:
            compression_on = False

    if compression_on:
        merge_size = self.visual.spatial_merge_size
        base_scale = float(os.getenv("R_RATIO", "0.25"))
        split_sizes = (video_grid_thw.prod(-1) // merge_size**2).tolist()

        video_splits = torch.split(video_embeds, split_sizes)
        kept_indices: List[Tensor] = []
        kept_video_chunks: List[Tensor] = []
        offset = 0

        for grid, feat in zip(video_grid_thw, video_splits):
            keep_local = _compute_keep_indices(
                flat_features=feat,
                grid_thw=grid,
                spatial_merge_size=merge_size,
                base_scale=base_scale,
            )
            kept_indices.append(keep_local + offset)
            kept_video_chunks.append(feat[keep_local])
            offset += feat.shape[0]

        kept_indices = torch.sort(torch.cat(kept_indices)).values
        video_embeds = torch.cat(kept_video_chunks, dim=0)

        video_token_positions = video_mask[..., 0][0].nonzero(as_tuple=False).squeeze(-1)
        kept_video_positions = video_token_positions[kept_indices]
        all_positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        non_video_positions = all_positions[~video_mask[..., 0][0]]
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
        attention_mask = _prune_attention(attention_mask)
        position_ids = position_ids[:, :, keep_token_indices]

        if image_mask is not None:
            image_mask = image_mask[:, keep_token_indices, :]
        if video_mask is not None:
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
