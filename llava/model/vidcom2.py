import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union

# Model configuration constants
LLAVA_OV_SPEC = {"tpf": 196, "grid": 14}
LLAVA_VID_SPEC = {"tpf": 169, "grid": 13}

def vidcom2_compression(flattened_feat: torch.Tensor,  
                        model="llava_ov", img_feat: Optional[torch.Tensor] = None,base_scale=0.25,) -> torch.Tensor:
    """Main compression pipeline with hardcoded outlier-retention strategy."""
    if model == "llava_vid" and img_feat is None:
        raise ValueError("img_feat is required for llava_vid model.")
    
    spec = LLAVA_VID_SPEC if model == "llava_vid" else LLAVA_OV_SPEC
    tpf = spec["tpf"]

    # 1. Feature Analysis
    sel_feat = select_low_var_channels(flattened_feat)
    vid_score, frame_score = compute_gaussian_scores(sel_feat, tpf)

    # 2. Hardcoded Logic: Frame score = -VideoMean; Token score = Video + Frame
    # Logic: Alloc more tokens to outlier frames; Keep outlier tokens (lowest similarity)
    scales = compute_scales(-vid_score.mean(dim=-1), base_scale)
    indices = select_outlier_indices(vid_score + frame_score, scales, tpf)

    # 3. Index Mapping & Retrieval
    return map_features(indices, flattened_feat, img_feat, model, spec["grid"])

def select_low_var_channels(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    """Selects channel dimensions with the least variance."""
    # Heuristic: Lower variance channels might contain more stable background info
    variances = x.var(dim=0, unbiased=False)
    k = int(x.shape[-1] * ratio)
    _, topk_idx = torch.topk(variances, k=k, largest=False)
    return x[:, topk_idx]

def compute_gaussian_scores(x: torch.Tensor, tpf: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes similarity to Video Center and Frame Center."""
    frames = x.view(-1, tpf, x.shape[-1])
    frames = F.normalize(frames, dim=-1)
    
    # Pre-calculate centers to leverage broadcasting
    vid_center = frames.mean(dim=(0, 1), keepdim=True) # Global temporal mean
    frame_center = frames.mean(dim=1, keepdim=True)    # Local spatial mean
    
    alphas = [2**k for k in range(-3, 2)]
    v_score = _multi_scale_gaussian(frames, vid_center, alphas)
    f_score = _multi_scale_gaussian(frames, frame_center, alphas)
    return v_score, f_score

def _multi_scale_gaussian(x: torch.Tensor, center: torch.Tensor, alphas: List[float]) -> torch.Tensor:
    """Helper: vectorized multi-scale Gaussian kernel."""
    # (B, T, C) - (1, 1, C) -> (B, T)
    dist_sq = ((x - center) ** 2).sum(dim=-1)
    return sum(torch.exp(-dist_sq / (2 * a)) for a in alphas)

def compute_scales(scores: torch.Tensor, base: float, temp: float = 0.01) -> torch.Tensor:
    """Generates per-frame retention ratios using Softmax."""
    # Normalize scores to probability distribution
    probs = F.softmax((scores - scores.max()) / temp, dim=0)
    # Scale modulation: higher score -> higher scale
    scales = base * (1 + probs - probs.mean())
    return scales.clamp(max=1.0)

def select_outlier_indices(scores: torch.Tensor, scales: torch.Tensor, tpf: int) -> List[torch.Tensor]:
    """Selects indices with the LOWEST similarity scores (Outliers)."""
    ks = (scales * tpf).round().long().clamp(min=1).tolist()
    batch_indices = []
    
    for i, k in enumerate(ks):
        # largest=False means we keep tokens LEAST similar to the means (details)
        _, idx = torch.topk(scores[i], k=k, largest=False, sorted=False)
        batch_indices.append(idx.sort().values) # Sort for sequential memory access
        
    return batch_indices

def map_features(indices: List[torch.Tensor], flat: torch.Tensor, img: torch.Tensor, 
                 model: str, grid: int) -> torch.Tensor:
    """Dispatches to the correct index mapper."""
    if model == "llava_ov":
        global_idx = _map_llava_ov(indices, LLAVA_OV_SPEC["tpf"])
        return flat[global_idx]
    elif model == "llava_vid":
        global_idx = _map_llava_vid(indices, grid)
        return img[global_idx]
    return flat

def _map_llava_ov(indices: List[torch.Tensor], tpf: int) -> torch.Tensor:
    """Simple offset mapping for flattened features."""
    device = indices[0].device
    offsets = torch.arange(len(indices), device=device) * tpf
    # Vectorized offset addition
    return torch.cat([idx + off for idx, off in zip(indices, offsets)])

def _map_llava_vid(indices: List[torch.Tensor], h: int) -> torch.Tensor:
    """Complex mapping handling 2D spatial grids + separators."""
    w_new = h + 1 # Width + newline token
    frame_stride = h * w_new
    global_idx = []
    
    for i, idx in enumerate(indices):
        # Convert local flat index -> 2D grid -> new global layout
        local_mapped = (idx // h) * w_new + (idx % h)
        start = i * frame_stride
        global_idx.append(start + local_mapped)
        # Add the newline tokens for this frame (essential for Vid structure)
        global_idx.append(start + (torch.arange(h, device=idx.device) * w_new + h))
        
    return torch.cat(global_idx)

