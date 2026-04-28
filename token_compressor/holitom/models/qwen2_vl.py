from typing import Optional, List, Tuple, Union
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast

from token_compressor.vidcom2 import *
import os

def Qwen2VL_ViT_forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
    hidden_states = self.patch_embed(hidden_states)#([106652, 1176])->([106652, 1280]) #106652=t*h*w
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
            #############################
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
            ##########################

    merged_hidden_states=self.merger(hidden_states)#([106652, 1280])->([26663, 3584])
    t=grid_thw[:, 0]
    h=grid_thw[:, 1]
    w=grid_thw[:, 2]
    resize_h=h//2
    resize_w=w//2
    token_per_frame=resize_h*resize_w
    model="qwen2_vl"
    base_scale = float(os.getenv('R_RATIO', '0.25'))
    
    frame_token_lenth=token_per_frame
    keep_index=vidcom2_compression(flattened_feat=merged_hidden_states,model=model,base_scale=base_scale,frame_token_len=frame_token_lenth)
    return merged_hidden_states,keep_index

def Qwen2VLGeneration_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
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
) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.get_dtype())
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            

        if pixel_values_videos is not None:
            ########################### visual 包含 vit和projector
            pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())

            #############################################################
            video_embeds,keep_index = self.visual(pixel_values_videos, grid_thw=video_grid_thw)# video_embeds.shape torch.Size([26663, 3584])
            # video_grid_thw[:, 1:] = video_grid_thw[:, 1:] / 2
            system_token_length=15
            image_token_lenth=video_embeds.shape[0]
            device = self.device
            # video_embeds=merged_video_embeds[keep_index,:]
            selected_image_index=keep_index+system_token_length
            retention_video_indice_in_all_tokens = torch.cat( (torch.arange(system_token_length,device=device), selected_image_index, torch.arange(image_token_lenth+system_token_length,inputs_embeds.shape[1],device=device)))
            # inputs_embeds=inputs_embeds[:,retention_video_indice_in_all_tokens,:]
            # input_ids=input_ids[:,retention_video_indice_in_all_tokens]

            video_mask = (
                (input_ids == self.config.video_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)# 26663,3584
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)# 26688,3584

        if attention_mask is not None:

            attention_mask = attention_mask.to(inputs_embeds.device)



    # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # calculate RoPE index once per generation in the pre-fill stage only
        if (
            (cache_position is not None and cache_position[0] == 0)
            or self.rope_deltas is None
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        ):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, attention_mask
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                delta = delta.to(position_ids.device)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    if pixel_values_videos is not None:
        print("before_prune_input",inputs_embeds.shape)
        inputs_embeds=inputs_embeds[:,retention_video_indice_in_all_tokens,:]
        print("after_prune_input",inputs_embeds.shape)

        input_ids= input_ids[:,retention_video_indice_in_all_tokens]
        position_ids=position_ids[:,:,retention_video_indice_in_all_tokens]
        attention_mask=attention_mask[:,retention_video_indice_in_all_tokens]

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
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
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