import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
import os
from typing import List
import random
import numpy as np
import json
import warnings
from typing import List, Optional, Tuple
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from transformers.cache_utils import Cache

def load_head_score(model_type):
    if 'llava' in model_type:
        if 'mistral' not in model_type:
            head_score_path = './visual_head/head_score/llava-v1.6.json'
        else:
            head_score_path = './visual_head/head_score/llava-mistral-v1.6.json'
    elif 'qwen' in model_type:
        head_score_path = './visual_head/head_score/qwen.json'
    else:
        raise NotImplementedError
    with open(head_score_path, 'r') as f:
        head_score = json.load(f)
    return head_score


def load_similarity_score():
    similarity_score_path = "./visual_head/head_score/qwen_kv_similarity.json"
    with open(similarity_score_path, 'r') as f:
        similarity_score = json.load(f)

    key_scores = {}
    value_scores = {}

    max_layer = 0
    max_head = 0

    for full_key, values in similarity_score.items():
        layer_head, score_type = full_key.rsplit('-', 1)
        layer, head = map(int, layer_head.split('-'))
        avg_score = sum(values) / len(values)

        if score_type == "key":
            key_scores[(layer, head)] = avg_score
        elif score_type == "value":
            value_scores[(layer, head)] = avg_score

        max_layer = max(max_layer, layer)
        max_head = max(max_head, head)

    key_tensor = torch.zeros((max_layer + 1, max_head + 1))
    value_tensor = torch.zeros((max_layer + 1, max_head + 1))

    for (layer, head), score in key_scores.items():
        key_tensor[layer][head] = score

    for (layer, head), score in value_scores.items():
        value_tensor[layer][head] = score
    return key_tensor

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def merge_kv(key_states, value_states, indices, window_size, merge):
    # merge methods in LOOK-M

    bsz, num_heads, k_len, head_dim = key_states.shape

    # kv-selected
    selected_keys = key_states.gather(dim=2, index=indices)  # [bsz, num_heads, topk_len, head_dim]
    selected_values = value_states.gather(dim=2, index=indices)  # [bsz, num_heads, topk_len, head_dim]

    # kv-drop
    all_indices = torch.arange(k_len, device=key_states.device).unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, k_len)
    all_indices_flattened = all_indices.flatten()  # [bsz * num_heads * (k_len-window_size)]
    selected_indices_flattened = indices.flatten()  # [bsz * num_heads * topk_len]
    is_selected = torch.isin(all_indices_flattened, selected_indices_flattened)
    drop_indices_flattened = all_indices_flattened[~is_selected]
    drop_len = drop_indices_flattened.shape[0] // (all_indices.shape[0] * all_indices.shape[1])
    drop_indices = drop_indices_flattened.reshape(all_indices.shape[0], all_indices.shape[1], drop_len) # [bsz * num_heads * (k_len-window_size-topk_len)]
    drop_indices = drop_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # [bsz, num_heads, (k_len-window_size-topk_len), head_dim]
    drop_keys = key_states.gather(dim=2, index=drop_indices)
    drop_values = value_states.gather(dim=2, index=drop_indices)

    # kv-recent
    recent_keys = key_states[:, :, -window_size:, :]

    ##### apply merge #####
    # prepare for merge
    k_hh_pruned = drop_keys  # [bsz, num_heads, k_len-topk_len-window_size, head_dim]
    k_hh_recent = torch.cat([recent_keys, selected_keys], dim=2)  # [bsz, num_heads, topk_len+window_size, head_dim]
    v_hh_pruned = drop_values  # [bsz, num_heads, k_len-topk_len-window_size, head_dim]
    v_hh_recent = torch.cat([selected_values, value_states[:, :, -window_size:, :]], dim=2)  # [bsz, num_heads, topk_len+window_size, head_dim]
    # similarity matrix
    similarity = (k_hh_pruned / torch.norm(k_hh_pruned, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128)) @ ((k_hh_recent / (torch.norm(k_hh_recent, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128))).transpose(-1, -2)) # cosin
    max_values, max_indices = similarity.max(dim=-1)

    # pivot merge
    if merge=="pivot":
        print("Pivot merge")
        merged_indices = max_indices.unsqueeze(-1).repeat(1, 1, 1, 128)
        k_hh_selected = torch.gather(input=k_hh_recent, dim=2, index=merged_indices)
        k_hh_merged = (k_hh_pruned + k_hh_selected)/2
        k_hh_recent = torch.scatter_reduce(input=k_hh_recent, dim=2, index=merged_indices, src=k_hh_merged, reduce='mean', include_self=True) # include_self=True seems decrease the performance
        v_hh_selected = torch.gather(input=v_hh_recent, dim=2, index=merged_indices)
        v_hh_merged = (v_hh_pruned + v_hh_selected)/2
        v_hh_recent = torch.scatter_reduce(input=v_hh_recent, dim=2, index=merged_indices, src=v_hh_merged, reduce='mean', include_self=True)
    else:
        raise ValueError('Merge method not supported')

    # TODO: other merge strategies
    # average merge
    # weight merge

    return k_hh_recent, v_hh_recent


 
class DynamicCacheSplitHeadFlatten(Cache):
    '''
    adapt from https://github.com/FFY0/AdaKV.
    '''
    def __init__(self) ->None:
        # Token wise List[]  Head wise KV List[torch.Tensor]
        super().__init__()
        self.key_cache: List[List[torch.Tensor]] = []
        self.value_cache: List[List[torch.Tensor]] = []
        self._seen_tokens = 0

    def __len__(self):
        return len(self.key_cache)

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))

    def __getitem__(self, layer_idx: int) -> Tuple[Tuple[torch.Tensor],Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if len(self.key_cache) <= layer_idx: #prefilling
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:#decoding
            assert self.key_cache[layer_idx].dim() == 2

            bs, head, seqlen, dim = key_states.shape
            assert bs == 1 and seqlen == 1
            head_lens = cache_kwargs["head_lens"]
            cu_klen = cache_kwargs["cu_klen"]

            # import nvtx
            # copy_old_rng = nvtx.start_range("copy old")
            from tiny_api_cuda import update_flatten_view
            new_key_cache = update_flatten_view(self.key_cache[layer_idx].view(-1,dim), key_states.view(-1, dim), head_lens, cu_klen)
            new_value_cache = update_flatten_view(self.value_cache[layer_idx].view(-1,dim), value_states.view(-1, dim), head_lens, cu_klen)

            # nvtx.end_range(copy_old_rng)

            self.key_cache[layer_idx] = new_key_cache
            self.value_cache[layer_idx] = new_value_cache


        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        # TODO: return 1 to means has content for now
        return 1
        # return max(map(lambda states: states.shape[-2], self.key_cache[layer_idx]))

    def get_max_length(self) -> Optional[int]:
        return None

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. DynamicCache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCacheEachHead":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

class SnapKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', layer_idx = None, num_hidden_layers = None, 
                 pyram_mode = False, pyram_beta = 20,num_key_value_groups = 1, gqa_func='mean',select_method=None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

        self.pyram_init = False
        self.pyram_mode = pyram_mode
        self.pyram_beta = pyram_beta
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        self.num_key_value_groups = num_key_value_groups
        self.gqa_func = gqa_func
        self.method=select_method
        self.key_score=load_similarity_score()

    @torch.compile
    def calcul_similarity_score(self, key_states):
        device = key_states.device
        bsz, num_heads, seq_len, head_dim = key_states.shape
        
        # Normalize vectors
        key_norm = F.normalize(key_states, dim=-1)
        # Only consider tokens before the last `window_size`
        valid_len = seq_len - self.window_size
        key_valid = key_norm[:, :, :valid_len, :]  # (B, H, valid_len, D)

        # Compute mean of valid tokens
        key_mean = key_norm.sum(dim=2, keepdim=True) / seq_len  # (B, H, 1, D)
        # Compute similarity of all tokens to the valid mean
        
        key_sim = torch.matmul(key_valid, key_mean.transpose(-2, -1)).squeeze(-1)  # (B, H, N-W)
        key_mean_norm_sq = torch.sum(key_mean ** 2, dim=-1)
        avg_similarity = (seq_len * key_mean_norm_sq - 1) / (seq_len - 1)
        return key_sim,avg_similarity
    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, origin_key_states, query_states, origin_value_states):
        B,H,N,D=origin_key_states.shape
        # support gqa
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
        value_states = repeat_kv(origin_value_states, self.num_key_value_groups)
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]

        bsz, num_heads, q_len, head_dim = query_states.shape
        origin_heads_key_states = torch.split(origin_key_states, 1, dim=1)
        origin_heads_value_states = torch.split(origin_value_states, 1, dim=1)
        # compute pyramidal capacity
        if self.pyram_mode and not self.pyram_init:
            # NOTE: (max_num + min_num) / 2 == base_capacity to restrict the total capacity
            base_capacity = self.max_capacity_prompt - self.window_size
            min_num = base_capacity // self.pyram_beta
            max_num = base_capacity * 2 - min_num
                
            # if the max_num is larger than the query length, we need to adjust the max_num
            if max_num >= q_len - self.window_size:
                max_num = q_len - self.window_size
                min_num = base_capacity * 2 - max_num
        
            # NOTE: compute interval
            steps = (max_num - min_num) // (self.num_hidden_layers - 1)

            self.max_capacity_prompt = max_num - self.layer_idx * steps + self.window_size
            self.pyram_init = True
            print(f"Pyram mode adaptive capacity, layer: {self.layer_idx}, max_capacity_prompt: {self.max_capacity_prompt}, base_capacity: {self.max_capacity_prompt - self.window_size}", flush=True)

        if q_len < self.max_capacity_prompt or q_len<self.window_size:
            return origin_key_states, origin_value_states
        else:
            
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim = -2)
            
            attn_weights_mean = attn_weights_mean.view(attn_weights_mean.shape[0], -1, self.num_key_value_groups, attn_weights_mean.shape[-1])
            if self.gqa_func == 'max':
                attn_weights_mean = attn_weights_mean.max(dim=-2).values
            elif self.gqa_func == 'mean':
                attn_weights_mean = attn_weights_mean.mean(dim=-2)
            else:
                raise ValueError('gqa_func not supported')
                
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_mean, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_mean, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            capacity=self.max_capacity_prompt - self.window_size
            attn_score=attn_cache
            similarity_score, online_score = self.calcul_similarity_score(origin_key_states)
            neg_sim = -similarity_score
            sim_min = neg_sim.amin(dim=-1, keepdim=True)
            sim_max = neg_sim.amax(dim=-1, keepdim=True)
            sim_norm = (neg_sim - sim_min) / (sim_max - sim_min + 1e-8)
            sim_mean = sim_norm.mean(dim=-1, keepdim=True)        # (..., 1)
            att_mean = attn_score.mean(dim=-1, keepdim=True)      # (..., 1)
            scale = att_mean / (sim_mean + 1e-8)
            sim_scaled = sim_norm * scale
            _device = key_states.device
            valid_len=N-self.window_size
            if self.method=='vnorm'or 'headwisemixkv':
              origin_value_states_valid=origin_value_states[:, :, :valid_len, :]
              vnorm_score = torch.norm(origin_value_states_valid, p=2, dim=-1)  # (bsz, num_heads, q_len)
              vnorm_min=vnorm_score.amin(dim=-1,keepdim=True)
              vnorm_max=vnorm_score.amax(dim=-1,keepdim=True)
              vnorm_score=(vnorm_score-vnorm_min)/(vnorm_max-vnorm_min)
              vnorm_mean=vnorm_score.mean(dim=-1, keepdim=True)
              scale = att_mean / (vnorm_mean + 1e-8)
              vnorm_score=vnorm_score*scale
              vnorm_score_head=vnorm_score.split(1,dim=1)
            elif self.method=='knorm':
              origin_key_states_valid=origin_key_states[:, :, :valid_len, :]
              knorm_score=-torch.norm(origin_key_states_valid, p=2, dim=-1)
              knorm_min=knorm_score.amin(dim=-1,keepdim=True)
              knorm_max=knorm_score.amax(dim=-1,keepdim=True)
              knorm_score=(knorm_score-knorm_min)/(knorm_max-knorm_min)
              knorm_mean=knorm_score.mean(dim=-1, keepdim=True)
              scale = att_mean / (knorm_mean + 1e-8)
              knorm_score=knorm_score*scale
              knorm_score_head=knorm_score.split(1,dim=1)

            head_attn_score=torch.split(attn_score,1,dim=1)
            head_sim_score=torch.split(sim_scaled,1,dim=1)
           
            heads_key_states = []
            heads_value_states = []
            for head_idx in range(num_heads//self.num_key_value_groups):
                #cache_index=indices[head_idx].squeeze()[...,:capacity]
                #head_score=online_score[:,head_idx,]
                head_score=self.key_score[self.layer_idx][head_idx]
                attn_score=head_attn_score[head_idx]
                sim_score=head_sim_score[head_idx]
                
                
                
                if self.method=='attn':
                    combined_score_head=attn_score
                elif self.method=='keydiff':
                    combined_score_head=sim_score
                elif self.method=='mixkv':
                    vnorm_score=vnorm_score_head[head_idx]
                    combined_score_head=0.33*sim_score+0.33*attn_score+0.33*vnorm_score
                elif self.method=='headwisemixkv':
                    vnorm_score=vnorm_score_head[head_idx]
                    importance_score=attn_score+vnorm_score
                    combined_score_head=head_score*sim_score+(1-head_score)*importance_score
                elif self.method=='knorm':
                    knorm_score=knorm_score_head[head_idx]
                    combined_score_head=attn_score+knorm_score
                elif self.method=='vnorm':
                    vnorm_score=vnorm_score_head[head_idx]
                    combined_score_head=attn_score+vnorm_score
                  
                    
                _,indices_head=combined_score_head.sort(dim=-1,descending=True)
                cache_index=indices_head.squeeze()[...,:capacity]
                cache_index = cache_index.view(1, 1, -1, 1).expand(-1, -1, -1, head_dim)
                k_past_compress = origin_heads_key_states[head_idx][:, :, :-self.window_size, :].gather(dim = 2, index = cache_index)
                v_past_compress = origin_heads_value_states[head_idx][:, :, :-self.window_size, :].gather(dim = 2, index = cache_index)
                k_cur = origin_heads_key_states[head_idx][:, :, -self.window_size:, :]
                v_cur = origin_heads_value_states[head_idx][:, :, -self.window_size:, :]
                key_states_head = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states_head = torch.cat([v_past_compress, v_cur], dim = 2)
                heads_key_states.append(key_states_head)
                heads_value_states.append(value_states_head)

            heads_key_states = torch.cat(heads_key_states, dim=1)
            heads_value_states = torch.cat(heads_value_states, dim=1)

            return heads_key_states, heads_value_states
class AdaKVCluster():
    def __init__(self, window_size = 32, kernel_size = 7, pooling = 'maxpool',base_capacity=None,floor_alpha = None,skip = None,normalize=None, 
                 layer_idx = None, num_hidden_layers = None, pyram_mode = False, pyram_beta = 20, num_key_value_groups=1, gqa_func='mean',select_method=None):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.base_capacity = base_capacity - window_size
        self.floor_ratio = floor_alpha
        self.floor_capacity = int(self.base_capacity * self.floor_ratio)
        self.adaptive_capacity = self.base_capacity - self.floor_capacity
        self.skip = skip

        self.normalize = normalize
        self.pyram_init = False
        self.pyram_mode = pyram_mode
        self.pyram_beta = pyram_beta
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        # NOTE: layer-wise meta-data
        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None

        self.num_key_value_groups = num_key_value_groups
        self.gqa_func = gqa_func
        self.method=select_method
        self.key_score=load_similarity_score()

    @torch.compile
    def calcul_similarity_score(self, key_states):
        device = key_states.device
        bsz, num_heads, seq_len, head_dim = key_states.shape
        
        # Normalize vectors
        key_norm = F.normalize(key_states, dim=-1)
        # Only consider tokens before the last `window_size`
        valid_len = seq_len - self.window_size
        key_valid = key_norm[:, :, :valid_len, :]  # (B, H, valid_len, D)

        # Compute mean of valid tokens
        key_mean = key_norm.sum(dim=2, keepdim=True) / seq_len  # (B, H, 1, D)
        # Compute similarity of all tokens to the valid mean
        
        key_sim = torch.matmul(key_valid, key_mean.transpose(-2, -1)).squeeze(-1)  # (B, H, N-W)
        key_mean_norm_sq = torch.sum(key_mean ** 2, dim=-1)
        avg_similarity = (seq_len * key_mean_norm_sq - 1) / (seq_len - 1)
        return key_sim,avg_similarity
    def calcul_attn_sore(self, key_states, query_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(
            head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim=-2)

        attn_weights_mean = attn_weights_mean.view(attn_weights_mean.shape[0],num_heads//self.num_key_value_groups,self.num_key_value_groups,-1)
        if self.gqa_func == 'max':
            attn_weights_mean = attn_weights_mean.max(dim=-2).values
        elif self.gqa_func == 'mean':
            attn_weights_mean = attn_weights_mean.mean(dim=-2)
        else:
            raise ValueError('gqa_func not supported')

        if self.pooling == 'avgpool':
            attn_weights_mean_pooling = F.avg_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        elif self.pooling == 'maxpool':
            attn_weights_mean_pooling = F.max_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        else:
            raise ValueError('Pooling method not supported')
        return attn_weights_mean_pooling


    def update_kv(self, origin_key_states, query_states, origin_value_states):
        B,H,N,D=origin_key_states.shape
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
        # value_states = repeat_kv(origin_value_states, self.num_key_value_groups)

        # check if prefix phase        assert key_states.shape[-2] == query_states.shape[-2]
        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        # import pdb; pdb.set_trace()
        origin_heads_key_states = torch.split(origin_key_states, 1, dim=1)
        origin_heads_value_states = torch.split(origin_value_states, 1, dim=1)

        # compute pyramidal capacity
        if self.pyram_mode and not self.pyram_init:
            # NOTE: (max_num + min_num) / 2 == base_capacity to restrict the total capacity
            min_num = self.base_capacity // self.pyram_beta
            max_num = self.base_capacity * 2 - min_num
                
            # if the max_num is larger than the query length, we need to adjust the max_num
            if max_num >= q_len - self.window_size:
                max_num = q_len - self.window_size
                min_num = self.base_capacity * 2 - max_num
        
            # NOTE: compute interval
            steps = (max_num - min_num) // (self.num_hidden_layers - 1)

            # renew adaptive capacity
            self.base_capacity = max_num - self.layer_idx * steps
            self.floor_capacity = int(self.base_capacity * self.floor_ratio)
            self.adaptive_capacity = self.base_capacity - self.floor_capacity
            self.pyram_init = True
            print(f"Pyram mode adaptive capacity, layer: {self.layer_idx}, acap: {self.adaptive_capacity}, bcap: {self.base_capacity}, fcap: {self.floor_capacity}",  flush=True)
        attn_score = self.calcul_attn_sore(key_states, query_states)
        def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
            # init metadata
            self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=_device)
            self.klen_sum = klen_sum
            self.max_seqlen_k = max_seqlen_k
            self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
            # init varlen flash attention metadata
            self.cu_klen = self.cu_headlens - self.head_lens
            self.cu_klen = torch.cat(
                [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=_device)], dim=0)
            # check bug
            self.layer_qlens = torch.ones(num_heads//self.num_key_value_groups, dtype=torch.int32,device=_device)
            self.qlen_sum = num_heads//self.num_key_value_groups
            self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
            self.cu_qlen = torch.cat(
                [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=_device)], dim=0)
            
            
            self.cu_offset = torch.arange(0, num_heads//self.num_key_value_groups + 1, dtype=torch.int32, device=_device)
            self.cu_head_offset = torch.arange(1, num_heads//self.num_key_value_groups +1, dtype=torch.int32, device=_device)

        if self.base_capacity > attn_score.size(-1) or q_len<self.window_size:
            init_metadata(num_heads, [q_len] * (num_heads//self.num_key_value_groups), q_len * (num_heads//self.num_key_value_groups), q_len)
            # not compress
            return origin_key_states.reshape(-1, head_dim), origin_value_states.reshape(-1, head_dim)

        similarity_score, online_score = self.calcul_similarity_score(origin_key_states)
        neg_sim = -similarity_score
        
        sim_min = neg_sim.amin(dim=-1, keepdim=True)
        sim_max = neg_sim.amax(dim=-1, keepdim=True)
        sim_norm = (neg_sim - sim_min) / (sim_max - sim_min + 1e-8)
        sim_mean = sim_norm.mean(dim=-1, keepdim=True)        # (..., 1)
        att_mean = attn_score.mean(dim=-1, keepdim=True)      # (..., 1)
        scale = att_mean / (sim_mean + 1e-8)
        sim_scaled = sim_norm * scale
        #alpha=0.5
        _device = key_states.device
        valid_len=N-self.window_size
        
        if self.method=='vnorm' or 'headwisemixkv':
              origin_value_states_valid=origin_value_states[:, :, :valid_len, :]
              vnorm_score = torch.norm(origin_value_states_valid, p=2, dim=-1)  # (bsz, num_heads, q_len)
              vnorm_min=vnorm_score.amin(dim=-1,keepdim=True)
              vnorm_max=vnorm_score.amax(dim=-1,keepdim=True)
              vnorm_score=(vnorm_score-vnorm_min)/(vnorm_max-vnorm_min)
              vnorm_mean=vnorm_score.mean(dim=-1, keepdim=True)
              scale = att_mean / (vnorm_mean + 1e-8)
              vnorm_score=vnorm_score*scale
              vnorm_score_head=vnorm_score.split(1,dim=1)
        elif self.method=='knorm':
          origin_key_states_valid=origin_key_states[:, :, :valid_len, :]
          knorm_score=-torch.norm(origin_key_states_valid, p=2, dim=-1)
          knorm_min=knorm_score.amin(dim=-1,keepdim=True)
          knorm_max=knorm_score.amax(dim=-1,keepdim=True)
          knorm_score=(knorm_score-knorm_min)/(knorm_max-knorm_min)
          knorm_mean=knorm_score.mean(dim=-1, keepdim=True)
          scale = att_mean / (knorm_mean + 1e-8)
          knorm_score=knorm_score*scale
          knorm_score_head=knorm_score.split(1,dim=1)

        
        head_attn_score=torch.split(attn_score,1,dim=1)
        head_sim_score=torch.split(sim_scaled,1,dim=1)
        sorted_attn_score,sorted_attn_score_indices = attn_score.sort(dim=-1,descending=True)
        

        if self.layer_idx >= self.skip:
            adaptive_attn_score = sorted_attn_score
            length = adaptive_attn_score.size(dim=-1)
            if self.normalize:
                ratio_weight = sorted_attn_score[...,:self.base_capacity].sum(dim=-1,keepdim=True)/sorted_attn_score.sum(dim=-1,keepdim=True)
                adaptive_attn_score = adaptive_attn_score*ratio_weight
            adaptive_attn_score = adaptive_attn_score.reshape(bsz,length*num_heads//self.num_key_value_groups)
            sorted_indices = torch.topk(adaptive_attn_score,k=num_heads*self.base_capacity//self.num_key_value_groups,dim=-1).indices
            sorted_indices = sorted_indices//length

            # floor_alpha capacity set
            head_adaptive_capacity = torch.zeros((bsz,num_heads//self.num_key_value_groups),device=_device,dtype = sorted_indices.dtype)
            head_adaptive_capacity.scatter_add_(-1,sorted_indices,torch.ones_like(sorted_indices,dtype=head_adaptive_capacity.dtype),)
            assert head_adaptive_capacity.sum().item() == num_heads*self.base_capacity//self.num_key_value_groups
            head_adaptive_capacity = torch.round(head_adaptive_capacity * (1-self.floor_ratio) + self.floor_capacity).int()
        else:
            head_adaptive_capacity = torch.ones((bsz,num_heads),device=_device,dtype = sorted_attn_score_indices.dtype) * self.base_capacity
        sorted_attn_score_indices = sorted_attn_score_indices.split(1,dim=1)
        
        heads_key_states = []
        heads_value_states = []
        assert bsz == 1
        # per head

        # reinit varlen metadata
        k_lens = []
        klen_sum = 0
        max_seqlen_k = 0
        self.cu_klen = 0


        for head_idx in range(num_heads//self.num_key_value_groups):
            capacity=head_adaptive_capacity[0][head_idx]
            #head_score=online_score[:,head_idx,]
            head_score=self.key_score[self.layer_idx][head_idx]
            attn_score=head_attn_score[head_idx]
            sim_score=head_sim_score[head_idx]
            
            
            
            if self.method=='attn':
                combined_score_head=attn_score
            elif self.method=='keydiff':
                combined_score_head=sim_score
            elif self.method=='mixkv':
                vnorm_score=vnorm_score_head[head_idx]
                combined_score_head=0.33*sim_score+0.33*attn_score+0.33*vnorm_score
            elif self.method=='headwisemixkv':
                vnorm_score=vnorm_score_head[head_idx]
                importance_score=attn_score+vnorm_score
                combined_score_head=head_score*sim_score+(1-head_score)*importance_score
            elif self.method=='knorm':
                knorm_score=knorm_score_head[head_idx]
                combined_score_head=attn_score+knorm_score
            elif self.method=='vnorm':
                vnorm_score=vnorm_score_head[head_idx]
                combined_score_head=attn_score+vnorm_score
            
            
            _,indices_head=combined_score_head.sort(dim=-1,descending=True)
            cache_index=indices_head.squeeze()[...,:capacity]
            #cache_index=indices[head_idx].squeeze()[...,:capacity]
            l = cache_index.shape[-1] + self.window_size
            k_lens.append(l)
            max_seqlen_k = max(max_seqlen_k, l)
            klen_sum += l

            cache_index = cache_index.view(1, 1, -1, 1).expand(-1, -1, -1, head_dim)
            top_Kcache = origin_heads_key_states[head_idx].gather(dim=2,index=cache_index)
            top_Vcache = origin_heads_value_states[head_idx].gather(dim=2,index=cache_index)
            selected_k = torch.cat([top_Kcache,origin_heads_key_states[head_idx][:, :, -self.window_size:, :]],dim=2)
            selected_v = torch.cat([top_Vcache,origin_heads_value_states[head_idx][:, :, -self.window_size:, :]],dim=2)

            # NOTE: flatten view
            heads_key_states.append(selected_k.view(-1, head_dim))
            heads_value_states.append(selected_v.view(-1, head_dim))

        init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k)

        # NOTE: compose as flatten view
        heads_key_states = torch.cat(heads_key_states, dim=0)
        heads_value_states = torch.cat(heads_value_states, dim=0)

        return heads_key_states, heads_value_states

class SparseMM():
    def __init__(self, window_size = 32, kernel_size = 7, pooling = 'maxpool', base_capacity=None, ratio=None, normalize=None, 
                 layer_idx = None, num_hidden_layers = None, head_score=None, num_attention_heads=32, num_key_value_groups=1, gqa_func='mean', model_type=None):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.base_capacity = base_capacity - window_size
        self.ratio = ratio

        self.normalize = normalize
        self.layer_idx = layer_idx
        self.num_attention_heads = num_attention_heads  
        self.num_hidden_layers = num_hidden_layers

        # NOTE: layer-wise meta-data
        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None

        self.num_key_value_groups = num_key_value_groups
        self.gqa_func = gqa_func

        if head_score == 'random':
            head_score_list = np.array([random.random() for _ in range(self.num_hidden_layers * self.num_attention_heads)])
            head_score_list = [np.mean(l[1]) for l in head_score.items()]
            head_score_list = torch.tensor(head_score_list / sum(head_score_list))
            self.score = head_score_list.view(self.num_hidden_layers, self.num_attention_heads//self.num_key_value_groups, self.num_key_value_groups)
            self.score = self.score.sum(dim=-1)
            min_cache = int(self.base_capacity * self.ratio)
            remain_capacity = (self.base_capacity - min_cache) * self.num_hidden_layers * self.num_attention_heads // self.num_key_value_groups
            self.head_adaptive_capacity = torch.round(self.score * remain_capacity + min_cache).int()
        elif head_score == 'visual':
            head_score = load_head_score(model_type)
            head_score_list = [np.mean(l[1]) for l in head_score.items()]
            head_score_list = torch.tensor(head_score_list / sum(head_score_list))
            self.score = head_score_list.view(self.num_hidden_layers, self.num_attention_heads//self.num_key_value_groups, self.num_key_value_groups)
            self.score = self.score.sum(dim=-1)
            min_cache = int(self.base_capacity * self.ratio)
            remain_capacity = (self.base_capacity - min_cache) * self.num_hidden_layers * self.num_attention_heads // self.num_key_value_groups
            self.head_adaptive_capacity = torch.round(self.score * remain_capacity + min_cache).int()
        

    def calcul_attn_sore(self, key_states, query_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(
            head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim=-2)

        attn_weights_mean = attn_weights_mean.view(attn_weights_mean.shape[0],num_heads//self.num_key_value_groups,self.num_key_value_groups,-1)
        if self.gqa_func == 'max':
            attn_weights_mean = attn_weights_mean.max(dim=-2).values
        elif self.gqa_func == 'mean':
            attn_weights_mean = attn_weights_mean.mean(dim=-2)
        else:
            raise ValueError('gqa_func not supported')

        if self.pooling == 'avgpool':
            attn_weights_mean_pooling = F.avg_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        elif self.pooling == 'maxpool':
            attn_weights_mean_pooling = F.max_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        else:
            raise ValueError('Pooling method not supported')
        return attn_weights_mean_pooling


    def update_kv(self, origin_key_states, query_states, origin_value_states):
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
        # value_states = repeat_kv(origin_value_states, self.num_key_value_groups)
        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_score= self.calcul_attn_sore(key_states,query_states)
        # import pdb; pdb.set_trace()
        origin_heads_key_states = torch.split(origin_key_states, 1, dim=1)
        origin_heads_value_states = torch.split(origin_value_states, 1, dim=1)

        def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
            # init metadata
            self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=_device)
            self.klen_sum = klen_sum
            self.max_seqlen_k = max_seqlen_k
            self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
            # init varlen flash attention metadata
            self.cu_klen = self.cu_headlens - self.head_lens
            self.cu_klen = torch.cat(
                [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=_device)], dim=0)
            # check bug
            self.layer_qlens = torch.ones(num_heads//self.num_key_value_groups, dtype=torch.int32,device=_device)
            self.qlen_sum = num_heads//self.num_key_value_groups
            self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
            self.cu_qlen = torch.cat(
                [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=_device)], dim=0)
            
            
            self.cu_offset = torch.arange(0, num_heads//self.num_key_value_groups + 1, dtype=torch.int32, device=_device)
            self.cu_head_offset = torch.arange(1, num_heads//self.num_key_value_groups +1, dtype=torch.int32, device=_device)

        if self.base_capacity > attn_score.size(-1):
            init_metadata(num_heads, [q_len] * (num_heads//self.num_key_value_groups), q_len * (num_heads//self.num_key_value_groups), q_len)
            # not compress
            return origin_key_states.reshape(-1, head_dim), origin_value_states.reshape(-1, head_dim)

        _,indices = attn_score.sort(dim=-1,descending=True)

        indices = indices.split(1,dim=1)

        heads_key_states = []
        heads_value_states = []
        assert bsz == 1
        # per head

        # reinit varlen metadata
        k_lens = []
        klen_sum = 0
        max_seqlen_k = 0
        self.cu_klen = 0


        for head_idx in range(num_heads//self.num_key_value_groups):
            cache_index = indices[head_idx][...,:self.head_adaptive_capacity[self.layer_idx][head_idx]]

            l = cache_index.shape[-1] + self.window_size
            k_lens.append(l)
            max_seqlen_k = max(max_seqlen_k, l)
            klen_sum += l

            cache_index = cache_index.view(1, 1, -1, 1).expand(-1, -1, -1, head_dim)
            top_Kcache = origin_heads_key_states[head_idx].gather(dim=2,index=cache_index)
            top_Vcache = origin_heads_value_states[head_idx].gather(dim=2,index=cache_index)
            selected_k = torch.cat([top_Kcache,origin_heads_key_states[head_idx][:, :, -self.window_size:, :]],dim=2)
            selected_v = torch.cat([top_Vcache,origin_heads_value_states[head_idx][:, :, -self.window_size:, :]],dim=2)

            # NOTE: flatten view
            heads_key_states.append(selected_k.view(-1, head_dim))
            heads_value_states.append(selected_v.view(-1, head_dim))

        init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k)

        # NOTE: compose as flatten view
        heads_key_states = torch.cat(heads_key_states, dim=0)
        heads_value_states = torch.cat(heads_value_states, dim=0)

        return heads_key_states, heads_value_states
class MixedSparseMM():
    def __init__(self, window_size = 32, kernel_size = 7, pooling = 'maxpool', base_capacity=None, ratio=None, normalize=None, 
                 layer_idx = None, num_hidden_layers = None, head_score=None, num_attention_heads=32, num_key_value_groups=1, gqa_func='mean', model_type=None
                 ,select_method=None):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.base_capacity = base_capacity - window_size
        self.ratio = ratio

        self.normalize = normalize
        self.layer_idx = layer_idx
        self.num_attention_heads = num_attention_heads  
        self.num_hidden_layers = num_hidden_layers

        # NOTE: layer-wise meta-data
        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None

        self.num_key_value_groups = num_key_value_groups
        self.gqa_func = gqa_func
        self.method=select_method
        self.key_score=load_similarity_score()
   
        
        if head_score == 'random':
            head_score_list = np.array([random.random() for _ in range(self.num_hidden_layers * self.num_attention_heads)])
            head_score_list = [np.mean(l[1]) for l in head_score.items()]
            head_score_list = torch.tensor(head_score_list / sum(head_score_list))
            self.score = head_score_list.view(self.num_hidden_layers, self.num_attention_heads//self.num_key_value_groups, self.num_key_value_groups)
            self.score = self.score.sum(dim=-1)
            min_cache = int(self.base_capacity * self.ratio)
            remain_capacity = (self.base_capacity - min_cache) * self.num_hidden_layers * self.num_attention_heads // self.num_key_value_groups
            self.head_adaptive_capacity = torch.round(self.score * remain_capacity + min_cache).int()
        elif head_score == 'visual' :
            head_score = load_head_score(model_type)
            head_score_list = [np.mean(l[1]) for l in head_score.items()]
            head_score_list = torch.tensor(head_score_list / sum(head_score_list))
            self.score = head_score_list.view(self.num_hidden_layers, self.num_attention_heads//self.num_key_value_groups, self.num_key_value_groups)
            self.score = self.score.sum(dim=-1)
            min_cache = int(self.base_capacity * self.ratio)
            remain_capacity = (self.base_capacity - min_cache) * self.num_hidden_layers * self.num_attention_heads // self.num_key_value_groups
            self.head_adaptive_capacity = torch.round(self.score * remain_capacity + min_cache).int()
        
        self.key_score,self.value_score=load_similarity_score()
        
        

            
    def merge_tokens(self, tokens, target_len):
        """
        Merge tokens by grouping similar tokens together (based on cosine similarity),
        then averaging each group to form a merged token. This function dynamically selects
        tokens for merging based on their similarity and reduces the computational cost.

        Args:
            tokens (Tensor): shape (B, H, S, D) - Input tokens.
            target_len (int): number of output merged tokens.

        Returns:
            merged (Tensor): shape (B, H, target_len, D) - Merged tokens.
        """
        b, h, s, d = tokens.shape
        
        assert s >= target_len, "Cannot merge to more tokens than original"

        device=tokens.device
        tokens = tokens.contiguous().view(b * h, s, d)
        tokens = tokens.to(device)

        
        tokens_norm = F.normalize(tokens, dim=-1)  # Normalize 

        
        group_a, group_b = tokens_norm[..., ::2, :], tokens_norm[..., 1::2, :]

        
        sim = torch.matmul(group_a, group_b.transpose(-1, -2))  # (B*H, group_a_len, group_b_len)
        
        assignment = F.softmax(sim, dim=-1)  # (B*H, group_a_len, group_b_len)

       
        merged_a = torch.einsum("bsd,bsk->bkd", group_a, assignment)  # (B*H, group_a_len, D)
        merged_b = torch.einsum("bsd,bsk->bkd", group_b, assignment.transpose(-1, -2))  # (B*H, group_b_len, D)

        # Step 5: Combine the merged tokens from both groups
        merged = torch.cat([merged_a, merged_b], dim=-2)  # (B*H, merged_len, D)

        
        if merged.shape[1] > target_len:
            # Select the top tokens (based on the softmax scores) to keep the target_len
            merged = merged[:, :target_len, :]

        # Reshape back to (B, H, target_len, D)
        merged = merged.view(b, h, target_len, d)
        merged
        return merged
    
    def calcul_attn_sore(self, key_states, query_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(
            head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim=-2)

        attn_weights_mean = attn_weights_mean.view(attn_weights_mean.shape[0],num_heads//self.num_key_value_groups,self.num_key_value_groups,-1)
        if self.gqa_func == 'max':
            attn_weights_mean = attn_weights_mean.max(dim=-2).values
        elif self.gqa_func == 'mean':
            attn_weights_mean = attn_weights_mean.mean(dim=-2)
        else:
            raise ValueError('gqa_func not supported')

        if self.pooling == 'avgpool':
            attn_weights_mean_pooling = F.avg_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        elif self.pooling == 'maxpool':
            attn_weights_mean_pooling = F.max_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        else:
            raise ValueError('Pooling method not supported')
        return attn_weights_mean_pooling
   
    @torch.compile
    def calcul_similarity_score(self, key_states):
        device = key_states.device
        bsz, num_heads, seq_len, head_dim = key_states.shape
        
        # Normalize vectors
        key_norm = F.normalize(key_states, dim=-1)
        # Only consider tokens before the last `window_size`
        valid_len = seq_len - self.window_size
        key_valid = key_norm[:, :, :valid_len, :]  # (B, H, valid_len, D)

        # Compute mean of valid tokens
        key_mean = key_norm.sum(dim=2, keepdim=True) / seq_len  # (B, H, 1, D)
        # Compute similarity of all tokens to the valid mean
        
        key_sim = torch.matmul(key_valid, key_mean.transpose(-2, -1)).squeeze(-1)  # (B, H, N-W)
        key_mean_norm_sq = torch.sum(key_mean ** 2, dim=-1)
        avg_similarity = (seq_len * key_mean_norm_sq - 1) / (seq_len - 1)
        return key_sim,avg_similarity

    
    def update_kv(self, origin_key_states, query_states, origin_value_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        B,H,N,D=origin_key_states.shape
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
        #value_states = repeat_kv(origin_value_states, self.num_key_value_groups)
        _device = key_states.device
        valid_len=N-self.window_size
        
       
        
        def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
            # init metadata
            self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=_device)
            self.klen_sum = klen_sum
            self.max_seqlen_k = max_seqlen_k
            self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
            # init varlen flash attention metadata
            self.cu_klen = self.cu_headlens - self.head_lens
            self.cu_klen = torch.cat(
                [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=_device)], dim=0)
            # check bug
            self.layer_qlens = torch.ones(num_heads//self.num_key_value_groups, dtype=torch.int32,device=_device)
            self.qlen_sum = num_heads//self.num_key_value_groups
            self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
            self.cu_qlen = torch.cat(
                [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.cu_offset = torch.arange(0, num_heads//self.num_key_value_groups + 1, dtype=torch.int32, device=_device)
            self.cu_head_offset = torch.arange(1, num_heads//self.num_key_value_groups +1, dtype=torch.int32, device=_device)
        
          
        similarity_score, online_score = self.calcul_similarity_score(origin_key_states)
        neg_sim = -similarity_score
        attn_score = self.calcul_attn_sore(key_states, query_states)
        sim_min = neg_sim.amin(dim=-1, keepdim=True)
        sim_max = neg_sim.amax(dim=-1, keepdim=True)
        sim_norm = (neg_sim - sim_min) / (sim_max - sim_min + 1e-8)
        sim_mean = sim_norm.mean(dim=-1, keepdim=True)        # (..., 1)
        att_mean = attn_score.mean(dim=-1, keepdim=True)      # (..., 1)
        scale = att_mean / (sim_mean + 1e-8)
        sim_scaled = sim_norm * scale
        if self.method=='vnorm'or 'headwisemixkv':
              origin_value_states_valid=origin_value_states[:, :, :valid_len, :]
              vnorm_score = torch.norm(origin_value_states_valid, p=2, dim=-1)  # (bsz, num_heads, q_len)
              vnorm_min=vnorm_score.amin(dim=-1,keepdim=True)
              vnorm_max=vnorm_score.amax(dim=-1,keepdim=True)
              vnorm_score=(vnorm_score-vnorm_min)/(vnorm_max-vnorm_min)
              vnorm_mean=vnorm_score.mean(dim=-1, keepdim=True)
              scale = att_mean / (vnorm_mean + 1e-8)
              vnorm_score=vnorm_score*scale
              vnorm_score_head=vnorm_score.split(1,dim=1)
        elif self.method=='knorm':
          origin_key_states_valid=origin_key_states[:, :, :valid_len, :]
          knorm_score=-torch.norm(origin_key_states_valid, p=2, dim=-1)
          knorm_min=knorm_score.amin(dim=-1,keepdim=True)
          knorm_max=knorm_score.amax(dim=-1,keepdim=True)
          knorm_score=(knorm_score-knorm_min)/(knorm_max-knorm_min)
          knorm_mean=knorm_score.mean(dim=-1, keepdim=True)
          scale = att_mean / (knorm_mean + 1e-8)
          knorm_score=knorm_score*scale
          knorm_score_head=knorm_score.split(1,dim=1)
        
        head_attn_score=torch.split(attn_score,1,dim=1)
        head_sim_score=torch.split(sim_scaled,1,dim=1)
        if (self.base_capacity > attn_score.shape[-1]) or q_len<self.window_size :
            init_metadata(num_heads, [q_len] * (num_heads//self.num_key_value_groups), q_len * (num_heads//self.num_key_value_groups), q_len)
            return origin_key_states.reshape(-1, head_dim), origin_value_states.reshape(-1, head_dim)

        #_,indices_attn = attn_score.sort(dim=-1,descending=True)
        #_,indices_sim=similarity_score.sort(dim=-1,descending=False)
        #indices_attn = indices_attn.split(1,dim=1)
        #indices_sim=indices_sim.split(1,dim=1)
        #_,indices=combined_score.sort(dim=-1,descending=True)
        #indices=indices.split(1,dim=1)
        bsz, num_heads, q_len, head_dim = query_states.shape
        origin_heads_key_states = torch.split(origin_key_states, 1, dim=1)
        origin_heads_value_states = torch.split(origin_value_states, 1, dim=1)
        heads_key_states = []
        heads_value_states = []
        k_lens = []
        klen_sum = 0
        max_seqlen_k = 0
        self.cu_klen = 0
        for head_idx in range(num_heads//self.num_key_value_groups):
            capacity = self.head_adaptive_capacity[self.layer_idx][head_idx]
            #head_score=online_score[:,head_idx,:]
            head_score=self.key_score[self.layer_idx][head_idx]
            attn_score=head_attn_score[head_idx]
            sim_score=head_sim_score[head_idx]
           
            if self.method=='attn':
                    combined_score_head=attn_score
            elif self.method=='keydiff':
                combined_score_head=sim_score
            elif self.method=='mixkv':
                vnorm_score=vnorm_score_head[head_idx]
                combined_score_head=0.33*sim_score+0.33*attn_score+0.33*vnorm_score
            elif self.method=='headwisemixkv':
                vnorm_score=vnorm_score_head[head_idx]
                importance_score=vnorm_score+attn_score
                combined_score_head=head_score*sim_score+(1-head_score)*importance_score
            elif self.method=='knorm':
                knorm_score=knorm_score_head[head_idx]
                combined_score_head=attn_score+knorm_score
            elif self.method=='vnorm':
                vnorm_score=vnorm_score_head[head_idx]
                combined_score_head=attn_score+vnorm_score
            

            _,indices_head=combined_score_head.sort(dim=-1,descending=True)
            cache_index=indices_head.squeeze()[...,:capacity]
            l = cache_index.shape[-1] + self.window_size
            k_lens.append(l)
            max_seqlen_k = max(max_seqlen_k, l)
            klen_sum += l
            cache_index = cache_index.view(1, 1, -1, 1).expand(-1, -1, -1, head_dim)
            top_Kcache = origin_heads_key_states[head_idx].gather(dim=2,index=cache_index)
            top_Vcache = origin_heads_value_states[head_idx].gather(dim=2,index=cache_index)
            selected_k = torch.cat([top_Kcache,origin_heads_key_states[head_idx][:, :, -self.window_size:, :]],dim=2)
            selected_v = torch.cat([top_Vcache,origin_heads_value_states[head_idx][:, :, -self.window_size:, :]],dim=2)

            # NOTE: flatten view
            heads_key_states.append(selected_k.view(-1, head_dim))
            heads_value_states.append(selected_v.view(-1, head_dim))

        init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k)
        heads_key_states = torch.cat(heads_key_states, dim=0)
        heads_value_states = torch.cat(heads_value_states, dim=0)

       
        return heads_key_states, heads_value_states


    

def init_pyramidkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = int(os.getenv('BUDGET'))
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config,'num_hidden_layers'):
            raise ValueError('num_hidden_layers should be set')
        if not hasattr(self.config,'gqa_func'):
            if 'llama' in self.config.model_type or 'mistral' in self.config.model_type or \
                'llava' in self.config.model_type or 'qwen' in self.config.model_type:
                self.config.gqa_func = 'mean'
        if not hasattr(self.config, 'select_method'):
            self.config.select_method = os.getenv('SELECT_METHOD')
            
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = SnapKVCluster(
            window_size = self.config.window_size,
            max_capacity_prompt = self.config.max_capacity_prompt,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            layer_idx = self.layer_idx,
            num_hidden_layers = self.config.num_hidden_layers,
            pyram_mode=True,
            pyram_beta=20,
            num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
            gqa_func=self.config.gqa_func,
            select_method=self.config.select_method

        )

def init_snapkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = int(os.getenv('BUDGET'))
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config,'num_hidden_layers'):
            raise ValueError('num_hidden_layers should be set')
        if not hasattr(self.config,'gqa_func'):
            if 'llama' in self.config.model_type or 'mistral' in self.config.model_type or \
                'llava' in self.config.model_type or 'qwen' in self.config.model_type:
                self.config.gqa_func = 'mean'
        if not hasattr(self.config, 'select_method'):
            self.config.select_method = os.getenv('SELECT_METHOD')
            
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = SnapKVCluster(
            window_size = self.config.window_size,
            max_capacity_prompt = self.config.max_capacity_prompt,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
            gqa_func=self.config.gqa_func,
            select_method=self.config.select_method
        )

def init_adakv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = int(os.getenv('BUDGET'))
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'floor_ratio'):
            self.config.floor_ratio = 0.2
        if not hasattr(self.config, 'normalize'):
            self.config.normalize = True
        if not hasattr(self.config, 'num_hidden_layers'):
            raise ValueError('num_hidden_layers should be set')
        if not hasattr(self.config, 'skip'):
            self.config.skip = 0
        if not hasattr(self.config,'gqa_func'):
            if 'llama' in self.config.model_type or 'mistral' in self.config.model_type or \
                'llava' in self.config.model_type or 'qwen' in self.config.model_type:
                self.config.gqa_func = 'mean'
        if not hasattr(self.config, 'select_method'):
            self.config.select_method = os.getenv('SELECT_METHOD')
            
    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = AdaKVCluster(
            window_size = self.config.window_size,
            base_capacity=self.config.max_capacity_prompt,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            floor_alpha= self.config.floor_ratio,
            skip = self.config.skip,
            layer_idx = self.layer_idx,
            normalize = self.config.normalize,
            num_hidden_layers = self.config.num_hidden_layers,
            num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
            gqa_func = self.config.gqa_func,
            select_method=self.config.select_method
        )

def init_sparsemm(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = int(os.getenv('BUDGET'))
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'head_score'):
            method = os.getenv('METHOD', None)
            if method == 'sparsemm':
                self.config.head_score = 'visual' 
            elif method == 'random':
                self.config.head_score = 'random'
            else:
                raise ValueError('head_score should be set')
        if not hasattr(self.config, 'ratio'):
            self.config.ratio = float(os.getenv('RATIO'))
        if not hasattr(self.config,'gqa_func'):
            if 'llama' in self.config.model_type or 'mistral' in self.config.model_type or \
                'llava' in self.config.model_type or 'qwen' in self.config.model_type:
                self.config.gqa_func = 'mean'
        

    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = SparseMM(
            window_size = self.config.window_size,
            base_capacity=self.config.max_capacity_prompt,
            head_score=self.config.head_score,
            ratio=self.config.ratio,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            layer_idx = self.layer_idx,
            num_hidden_layers = self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
            gqa_func = self.config.gqa_func,
            model_type=self.config.model_type
        )

def init_mask(self):
    if not hasattr(self, "head_list"):
        method = os.getenv('METHOD', None)

        head_score = load_head_score(self.config.model_type)
        head_list = [(l[0], np.mean(l[1])) for l in head_score.items()]
        head_list = sorted(head_list, key=lambda x: x[1], reverse=True) 

        if method == 'mask':
            ratio = float(os.getenv('MASK_RATIO'))
            num = int(ratio * len(head_list))
            print(f"mask ratio: {ratio}, num: {num}")
            head_list = [[int(ll) for ll in l[0].split("-")] for l in head_list][:num]
            self.head_list = head_list
        else:
            ratio = float(os.getenv('MASK_RATIO'))
            layer_num = 32 if 'llava' in self.config.model_type else 28
            head_num = 32 if 'llava' in self.config.model_type else 32
            num = int(ratio * layer_num * head_num)
            print(f"mask random ratio: {ratio}, num: {num}")
            head_list = [[int(ll) for ll in l[0].split("-")] for l in head_list][:num]
            self.head_list = []
            seed_list = [i  for i in range(layer_num)]
            random.shuffle(seed_list)
            while len(self.head_list) < num:
                l, h = random.choices(seed_list, k=2)
                if (l, h) in self.head_list or (h, l) in head_list:
                    continue
                else:
                    self.head_list.append((l, h))




def init_mixsparsemm(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = int(os.getenv('BUDGET'))
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'head_score'):
            method = os.getenv('METHOD', None)
            if method == 'mixsparsemm':
                self.config.head_score = 'visual' 
            else:
                raise ValueError('head_score should be set')
        if not hasattr(self.config, 'ratio'):
            self.config.ratio = float(os.getenv('RATIO'))
        if not hasattr(self.config,'gqa_func'):
            if 'llama' in self.config.model_type or 'mistral' in self.config.model_type or \
                'llava' in self.config.model_type or 'qwen' in self.config.model_type:
                self.config.gqa_func = 'mean'
        if not hasattr(self.config, 'select_method'):
            self.config.select_method = os.getenv('SELECT_METHOD')
            
        

    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = MixedSparseMM(
            window_size = self.config.window_size,
            base_capacity=self.config.max_capacity_prompt,
            head_score=self.config.head_score,
            ratio=self.config.ratio,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            layer_idx = self.layer_idx,
            num_hidden_layers = self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
            gqa_func = self.config.gqa_func,
            model_type=self.config.model_type,
            select_method=self.config.select_method
            
        )
