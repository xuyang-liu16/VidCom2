import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any

# Configuration: 'mapper' determines the index mapping strategy
# 'tpf' is default token_per_frame, can be overridden for qwen2_vl
MODEL_SPECS = {
    "llava_ov": {"tpf": 196, "mapper": "linear"},
    "llava_vid": {"tpf": 169, "mapper": "grid_vid", "grid": 13},
    "qwen2_vl": {"tpf": None, "mapper": "linear"}, # tpf provided dynamically
}

def vidcom2_compression(flattened_feat: torch.Tensor, model: str = "llava_ov", 
                        base_scale: float = 0.25, frame_token_len: Optional[int] = None,
                        img_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compression pipeline supporting llava_ov, llava_vid, and qwen2_vl."""
    if model not in MODEL_SPECS: raise ValueError(f"Unknown model: {model}")
    
    spec = MODEL_SPECS[model]
    # Use dynamic tpf for qwen2_vl, else use constant from spec
    tpf = frame_token_len if model == "qwen2_vl" else spec["tpf"]
    if tpf is None: raise ValueError(f"frame_token_len required for {model}")

    # 1. Feature Analysis (Vectorized Gaussian Scores)
    sel_feat = select_low_var_channels(flattened_feat)
    vid_score, frame_score = compute_gaussian_scores(sel_feat, tpf)

    # 2. Score Fusion & Selection (Hardcoded: Outlier Retention)
    # Strategy: Keep tokens different from both Global Video Mean and Local Frame Mean
    scales = compute_scales(-vid_score.mean(dim=-1), base_scale)
    indices = select_outlier_indices(vid_score + frame_score, scales, tpf)

    # 3. Index Mapping (Routes to linear or grid mapper)
    return map_features(indices, flattened_feat, img_feat, spec)

def select_low_var_channels(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    """Selects least informative channels (lowest variance)."""
    variances = x.var(dim=0, unbiased=False)
    k = int(x.shape[-1] * ratio)
    _, topk_idx = torch.topk(variances, k=k, largest=False)
    return x[:, topk_idx]

def compute_gaussian_scores(x: torch.Tensor, tpf: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes Gaussian similarity to Video and Frame centers."""
    frames = x.view(-1, tpf, x.shape[-1])
    frames = F.normalize(frames, dim=-1)
    
    # Broadcastable centers: (1, 1, C) and (B, 1, C)
    vid_center = frames.mean(dim=(0, 1), keepdim=True) 
    frame_center = frames.mean(dim=1, keepdim=True)
    
    alphas = [2**k for k in range(-3, 2)]
    v_score = _multi_scale_gaussian(frames, vid_center, alphas)
    f_score = _multi_scale_gaussian(frames, frame_center, alphas)
    return v_score, f_score

def _multi_scale_gaussian(x: torch.Tensor, center: torch.Tensor, alphas: List[float]) -> torch.Tensor:
    """Helper: vectorized multi-scale Gaussian kernel."""
    dist_sq = ((x - center) ** 2).sum(dim=-1)
    return sum(torch.exp(-dist_sq / (2 * a)) for a in alphas)

def compute_scales(scores: torch.Tensor, base: float, temp: float = 0.01) -> torch.Tensor:
    """Generates dynamic retention rates based on frame importance."""
    probs = F.softmax((scores - scores.max()) / temp, dim=0)
    scales = base * (1 + probs - probs.mean())
    return scales.clamp(max=1.0)

def select_outlier_indices(scores: torch.Tensor, scales: torch.Tensor, tpf: int) -> List[torch.Tensor]:
    """Selects top-k indices with lowest similarity (largest outliers)."""
    ks = (scales * tpf).round().long().clamp(min=1).tolist()
    batch_indices = []
    for i, k in enumerate(ks):
        # largest=False -> retain most distinct tokens
        _, idx = torch.topk(scores[i], k=k, largest=False, sorted=False)
        batch_indices.append(idx.sort().values) 
    return batch_indices

def map_features(indices: List[torch.Tensor], flat: torch.Tensor, 
                 img: Optional[torch.Tensor], spec: Dict[str, Any]) -> torch.Tensor:
    """Dispatches to the correct mapper based on model spec."""
    if spec["mapper"] == "linear":
        # Shared logic for llava_ov and qwen2_vl
        tpf = indices[0].shape[0] if not spec["tpf"] else spec["tpf"] # Infer or use const
        # Note: We must use the tpf used during scoring/splitting
        # But indices logic is per-frame, so we reconstruct global offsets
        # For variable K per frame, we need the ORIGINAL tpf stride
        stride = flat.shape[0] // len(indices)
        global_idx = _map_linear_offset(indices, stride)
        return flat[global_idx]
    
    elif spec["mapper"] == "grid_vid":
        if img is None: raise ValueError("img_feat required for grid mapping")
        global_idx = _map_grid_vid(indices, spec["grid"])
        return img[global_idx]
    return flat

def _map_linear_offset(indices: List[torch.Tensor], tpf: int) -> torch.Tensor:
    """Standard mapping: adds frame_offset to local indices (Llava-OV / Qwen2-VL)."""
    device = indices[0].device
    offsets = torch.arange(len(indices), device=device) * tpf
    return torch.cat([idx + off for idx, off in zip(indices, offsets)])

def _map_grid_vid(indices: List[torch.Tensor], h: int) -> torch.Tensor:
    """Mapping for Llava-Vid: handles 2D grid structure + newline tokens."""
    w_new = h + 1
    stride = h * w_new
    global_idx = []
    for i, idx in enumerate(indices):
        local_mapped = (idx // h) * w_new + (idx % h)
        start = i * stride
        global_idx.append(start + local_mapped)
        global_idx.append(start + (torch.arange(h, device=idx.device) * w_new + h))
    return torch.cat(global_idx)