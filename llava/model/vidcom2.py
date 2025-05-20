import torch
from typing import Any, Dict, List, Optional, Tuple, Union

def vidcom2_compression(flattened_image_feature,base_scale=0.25,token_score_mode="negtive_video_mean_and_global_mean",attn_score=None,frame_score_mode="negtive_video_mean", model="llava_ov",  image_feature=None):
    if model == "llava_vid" and image_feature is None:
        raise ValueError("image_feature must be provided when using 'llava_vid' model")
    if model == "llava_ov":
        token_per_frame = 196
    elif model == "llava_vid":
        token_per_frame = 169
    else:
        raise ValueError("Invalid model name. Choose from 'llava_ov', 'llava_vid'.")
    selected_tensor = select_feature_channel(flattened_image_feature)
    frame_mean_score = compute_frame_mean_score_multi_gaussian_matrix(selected_tensor, token_per_frame)
    video_mean_score = compute_video_mean_score_multi_gaussian_matrix(selected_tensor, token_per_frame)
    if frame_score_mode == "equal":
        num_frames = flattened_image_feature.shape[0] // token_per_frame
        scales = torch.full((num_frames,), base_scale)
    elif frame_score_mode == "negtive_video_mean" : 
        frame_score = -video_mean_score.mean(-1) 
        scales = generate_scales_from_frame_video_score(frame_score, base_scale)
    elif frame_score_mode == "positive_video_mean":
        frame_score = video_mean_score.mean(-1) 
        scales = generate_scales_from_frame_video_score(frame_score, base_scale)
    elif frame_score_mode == "positive_video_score_max":
        frame_score = video_mean_score.max(dim=1).values
        scales = generate_scales_from_frame_video_score(frame_score, base_scale)
    elif frame_score_mode == "negtive_video_score_max":
        frame_score = -video_mean_score.max(dim=1).values
        scales = generate_scales_from_frame_video_score(frame_score, base_scale)
    elif frame_score_mode == "positive_video_score_min":
        frame_score = video_mean_score.min(dim=1).values
        scales = generate_scales_from_frame_video_score(frame_score, base_scale)
    elif frame_score_mode == "negtive_video_score_min":
        frame_score = -video_mean_score.min(dim=1).values
        scales = generate_scales_from_frame_video_score(frame_score, base_scale)
    else:
        raise ValueError("Invalid frame_score_mode. Choose from 'equal', 'negtive', or 'positive'.")
    
    if token_score_mode == "positive_frame_mean":
        token_score=-frame_mean_score
        sorted_indices=get_indeice_of_select_token_with_single_specific_retention(flattened_image_feature, scales,token_score,token_per_frame)
    elif token_score_mode == "negtive_frame_mean":
        token_score=frame_mean_score
        sorted_indices=get_indeice_of_select_token_with_single_specific_retention(flattened_image_feature, scales,token_score,token_per_frame)
    elif token_score_mode == "positive_patch_attn":
        token_score=-attn_score
        sorted_indices=get_indeice_of_select_token_with_single_specific_retention(flattened_image_feature, scales,token_score,token_per_frame)
    elif token_score_mode == "negtive_patch_attn":
        token_score=attn_score
        sorted_indices=get_indeice_of_select_token_with_single_specific_retention(flattened_image_feature, scales,token_score,token_per_frame)
    elif token_score_mode == "positive_video_mean":
        token_score=-video_mean_score
        sorted_indices=get_indeice_of_select_token_with_single_specific_retention(flattened_image_feature, scales,token_score,token_per_frame)
    elif token_score_mode == "negtive_video_mean":
        token_score=video_mean_score
        sorted_indices=get_indeice_of_select_token_with_single_specific_retention(flattened_image_feature, scales,token_score,token_per_frame)
    elif token_score_mode == "negtive_video_mean_and_global_mean":
        sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(
        flattened_image_feature, scales, video_mean_score, frame_mean_score, token_per_frame
    )
    else:
        raise ValueError("Invalid token_score_mode. Choose from 'video_mean', 'negtive_video_mean', 'negtive_video_mean_and_global_mean'.")

    if model == "llava_ov":
        converted_indice = get_keep_indices_of_llava_ov(sorted_indices)
        converted_image_feature = flattened_image_feature[converted_indice]
    elif model == "llava_vid":
        converted_indice = get_keep_indices_of_llava_vid(sorted_indices, resize_h=13)
        converted_image_feature = image_feature[converted_indice]

    return converted_image_feature


def select_feature_channel(input_tensor, k=0.5):
    channel_var = input_tensor.var(dim=0, unbiased=False) 
    k = channel_var.shape[0] // 2
    _, topk_indices = torch.topk(channel_var, k=k,largest=False)
    selected_tensor = input_tensor[:, topk_indices]  
    return selected_tensor
def compute_frame_mean_score_multi_gaussian_matrix(selected_tensor, token_per_frame=169, alphas=None):
    if alphas is None:
        alphas = [2**k for k in range(-3, 2)]  
    num_frame = selected_tensor.shape[0] // token_per_frame
    frames = selected_tensor.view(num_frame, token_per_frame, -1) 
    frames = torch.nn.functional.normalize(frames, dim=-1) 
    avg_token = frames.mean(dim=1, keepdim=True) 
    expanded_avg = avg_token.expand_as(frames)
    l2_distance_square = torch.sum((frames - expanded_avg) ** 2, dim=2)  
    k_xy = sum([torch.exp(-l2_distance_square / (2 * alpha)) for alpha in alphas])
    return k_xy  


def compute_video_mean_score_multi_gaussian_matrix(selected_tensor, token_per_frame=169, alphas=None):
    if alphas is None:
        alphas = [2**k for k in range(-3, 2)]  
    num_frame = selected_tensor.shape[0] // token_per_frame
    frames = selected_tensor.view(num_frame, token_per_frame, -1)  
    frames = torch.nn.functional.normalize(frames, dim=-1)  
    avg_token = frames.mean(dim=(0,1), keepdim=True)  
    expanded_avg = avg_token.expand_as(frames)
    l2_distance_square = torch.sum((frames - expanded_avg) ** 2, dim=2)  
    k_xy = sum([torch.exp(-l2_distance_square / (2 * alpha)) for alpha in alphas])
    return k_xy  


def generate_scales_from_frame_video_score(frame_scores, base_scale=0.1, temperature=0.01):
    shifted_scores = (frame_scores - torch.max(frame_scores)) / temperature
    exp_scores = torch.exp(shifted_scores)
    softmax_scores = exp_scores / (torch.sum(exp_scores) + 1e-8)
    scales = base_scale * (1 + softmax_scores - torch.mean(softmax_scores))
    scales = torch.clip(scales, None, 1.0)
    return scales
def get_indeice_of_select_token_with_conbine_specific_retention(input_tensor, scales,video_mean_score,frame_mean_score,token_per_frame=169):

    num_frame = input_tensor.shape[0] // token_per_frame
    frames = input_tensor.view(num_frame, token_per_frame, -1)
    total_frames, tokens_per_frame, dim = frames.shape
    assert len(scales) == num_frame, "scales must equal to frame_num"
    scales = torch.as_tensor(scales, dtype=torch.float32, device=frames.device)
    k_list = []
    for scale in scales:
        scaled_value = scale.item() * tokens_per_frame  
        k = max(1, int(round(scaled_value)))
        k_list.append(k)
    combined_similarities = frame_mean_score + video_mean_score
    pruned_indices = []
    for i in range(num_frame):
        frame = frames[i].unsqueeze(0)
        sim = combined_similarities[i].unsqueeze(0)
        k = k_list[i]
        _, indices = torch.topk(sim, k=k, dim=1, largest=False, sorted=False)
        sorted_indices, _ = torch.sort(indices, dim=1)
        pruned_indices.append(sorted_indices[0])
    return pruned_indices


def get_indeice_of_select_token_with_single_specific_retention(input_tensor, scales,video_mean_score,token_per_frame=169):

    num_frame = input_tensor.shape[0] // token_per_frame
    frames = input_tensor.view(num_frame, token_per_frame, -1)
    total_frames, tokens_per_frame, dim = frames.shape

    assert len(scales) == num_frame, "scales must equal to frame_num"
    scales = torch.as_tensor(scales, dtype=torch.float32, device=frames.device)
    k_list = []
    for scale in scales:
        scaled_value = scale.item() * tokens_per_frame  
        k = max(1, int(round(scaled_value)))
        k_list.append(k)

    combined_similarities = video_mean_score

    pruned_indices = []
    for i in range(num_frame):
        frame = frames[i].unsqueeze(0)
        sim = combined_similarities[i].unsqueeze(0)
        k = k_list[i]

        _, indices = torch.topk(sim, k=k, dim=1, largest=False, sorted=False)
        sorted_indices, _ = torch.sort(indices, dim=1)
        pruned_indices.append(sorted_indices[0])
    return pruned_indices


def get_keep_indices_of_llava_ov(indices_list, token_per_frame=196):

    device = indices_list[0].device
    num_frames = len(indices_list)
    frame_indices = torch.arange(num_frames, device=device).unsqueeze(1)
    frame_offsets = frame_indices * token_per_frame
    global_indices = torch.cat([indices + offset for indices, offset in zip(indices_list, frame_offsets)])
    return global_indices



def get_keep_indices_of_llava_vid(keep_indices, resize_h=13):

    H = resize_h
    W = resize_h
    W_new = W + 1  
    num_frames=len(keep_indices)
    all_indices = []
    
    for frame_idx in range(num_frames):
        frame_keep = keep_indices[frame_idx]  
        rows = frame_keep // W  
        cols = frame_keep % W    
        frame_local_indices = rows * W_new + cols  
        frame_start = frame_idx * (H * W_new)
        global_indices = frame_start + frame_local_indices
        all_indices.append(global_indices)
        row_indices = torch.arange(H, dtype=torch.long, device=frame_keep.device)  
        new_token_frame_local = row_indices * W_new + W  
        new_token_global = frame_start + new_token_frame_local  
        all_indices.append(new_token_global)
    final_indices = torch.cat(all_indices, dim=0)
    return final_indices   

def get_keep_indices_of_qwen2_vl(keep_indices: List[torch.Tensor], tokens_per_frame: int) -> torch.Tensor:

    all_indices = []
    for frame_idx, frame_keep in enumerate(keep_indices):
        frame_start = frame_idx * tokens_per_frame
        global_indices = frame_start + frame_keep  
        all_indices.append(global_indices)
    
    return torch.cat(all_indices, dim=0)

def dycoke_ttm_retention_llava_video(flaten_image_feature,image_feature,num_tokens_per_frame=169, retention_ratio=0.07):
    num_frames = flaten_image_feature.shape[0] // num_tokens_per_frame
    device = flaten_image_feature.device
    similarities = []
    for i in range(num_frames - 1):
        frame1 = flaten_image_feature[i*num_tokens_per_frame : (i+1)*num_tokens_per_frame]
        frame2 = flaten_image_feature[(i+1)*num_tokens_per_frame : (i+2)*num_tokens_per_frame]
        frame1_norm = torch.nn.functional.normalize(frame1, p=2, dim=1)
        frame2_norm = torch.nn.functional.normalize(frame2, p=2, dim=1)
        similarity = torch.nn.functional.cosine_similarity(frame1_norm, frame2_norm, dim=1)
        similarities.append(similarity)
    similarities = torch.stack(similarities) 

    retained_indices = [torch.arange(num_tokens_per_frame, device=device) for _ in range(num_frames)]

    for i in range(0, num_frames-1, 2):
        avg_similarity = similarities[i]  
        num_keep = int(retention_ratio * num_tokens_per_frame)
        keep_indices = avg_similarity.topk(num_keep, largest=False).indices
        retained_indices[i+1] = keep_indices  

    odd_similarities = []
    for i in range(0, num_frames-4, 4):
        frame1 = flaten_image_feature[i*num_tokens_per_frame : (i+1)*num_tokens_per_frame]
        frame2 = flaten_image_feature[(i+2)*num_tokens_per_frame : (i+3)*num_tokens_per_frame]
        similarity = torch.nn.functional.cosine_similarity(frame1, frame2, dim=1)
        odd_similarities.append(similarity)
    if odd_similarities:
        odd_similarities = torch.stack(odd_similarities)

    for idx, i in enumerate(range(0, num_frames-4, 4)):
        avg_similarity = odd_similarities[idx]
        num_keep = int(retention_ratio * num_tokens_per_frame)
        keep_indices = avg_similarity.topk(num_keep, largest=False).indices
        retained_indices[i+2] = keep_indices  
    converted_indice = get_keep_indices_of_llava_vid(retained_indices, resize_h=13)
    converted_image_feature = image_feature[converted_indice]
    return converted_image_feature


def full_tokens(image_feature):
    return image_feature