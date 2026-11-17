"""
Fast Numba-accelerated Pretrainer for 2-Fold Models (K=4 and K=8)
Compiles the pixel accumulation loop with @njit so pretraining completes in < 3 seconds
instead of 7 minutes, saving models to results/phase1/pretrained_models.npz.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import json
import numpy as np
from collections import defaultdict
from numba import njit

from carle.rigorous_codecs import precompute_bmap_fast, get_spatial_context_fast
from experiments.exp_12_pidnet_sequence_benchmark import get_spatial_context_k8
from experiments.exp_13_debug_temporal import get_compact_temporal_context_k4, get_compact_temporal_context_k8

CACHE_DIR = Path("results/phase1/cache_pidnet")

FOLD1_SCENES = {"scene-0061", "scene-0103", "scene-0553", "scene-0655", "scene-0757"}
FOLD2_SCENES = {"scene-0796", "scene-0916", "scene-1077", "scene-1094", "scene-1100"}
FOLD1_CLIPS = {"india_bolpur", "malaysia", "dhaka_day"}
FOLD2_CLIPS = {"sikkim", "dhaka_night"}

def get_stream_fold(stream_name: str) -> int:
    for sc in FOLD1_SCENES:
        if sc in stream_name: return 1
    for sc in FOLD2_SCENES:
        if sc in stream_name: return 2
    for cl in FOLD1_CLIPS:
        if cl in stream_name: return 1
    for cl in FOLD2_CLIPS:
        if cl in stream_name: return 2
    return 1

@njit(fastmath=True)
def accumulate_counts_k4(curr: np.ndarray, prev: np.ndarray, b_prev: np.ndarray,
                         c_intra: np.ndarray, c_temp: np.ndarray):
    H, W = curr.shape
    for r in range(0, H, 2):
        for c in range(0, W, 2):
            ctx_i = get_spatial_context_fast(curr, r, c)
            ctx_t = get_compact_temporal_context_k4(curr, prev, b_prev, r, c)
            sym = int(curr[r, c] - 1)
            c_intra[ctx_i, sym] += 1
            c_temp[ctx_t, sym] += 1

@njit(fastmath=True)
def accumulate_counts_k8(curr: np.ndarray, prev: np.ndarray, b_prev: np.ndarray,
                         c_intra: np.ndarray, c_temp: np.ndarray):
    H, W = curr.shape
    for r in range(0, H, 2):
        for c in range(0, W, 2):
            ctx_i = get_spatial_context_k8(curr, r, c)
            ctx_t = get_compact_temporal_context_k8(curr, prev, b_prev, r, c)
            sym = int(curr[r, c] - 1)
            c_intra[ctx_i, sym] += 1
            c_temp[ctx_t, sym] += 1

def compute_fast_models(k_val: int):
    t0 = time.perf_counter()
    k_tag = f"k{k_val}"
    num_classes = k_val
    num_ctx_intra = 256 if k_val == 4 else 512
    num_ctx_temp = 2048 if k_val == 4 else 4096
    
    counts_intra = {1: np.ones((num_ctx_intra, num_classes), dtype=np.int64) * 8,
                    2: np.ones((num_ctx_intra, num_classes), dtype=np.int64) * 8}
    counts_temp = {1: np.ones((num_ctx_temp, num_classes), dtype=np.int64) * 8,
                   2: np.ones((num_ctx_temp, num_classes), dtype=np.int64) * 8}
                   
    kf_p = CACHE_DIR / f"nuscenes_keyframes_2hz_{k_tag}.npz"
    meta_p = CACHE_DIR / "nuscenes_keyframes_2hz_meta.json"
    with open(meta_p) as f: meta = json.load(f)
    kf_masks = np.load(kf_p, mmap_mode="r")["masks"]
    
    by_cam = defaultdict(list)
    for i, m in enumerate(meta):
        by_cam[(m["scene"], m["camera"])].append((i, m["timestamp"]))
        
    for s_name, items in by_cam.items():
        stream_id = f"{s_name[0]}_{s_name[1]}"
        s_fold = get_stream_fold(stream_id)
        target_fold = 2 if s_fold == 1 else 1
        items.sort(key=lambda x: x[1])
        indices = [x[0] for x in items[:15]]
        
        for f_i in range(1, len(indices)):
            curr = kf_masks[indices[f_i]]
            prev = kf_masks[indices[f_i - 1]]
            b_prev = precompute_bmap_fast(prev)
            
            if k_val == 4:
                accumulate_counts_k4(curr, prev, b_prev, counts_intra[target_fold], counts_temp[target_fold])
            else:
                accumulate_counts_k8(curr, prev, b_prev, counts_intra[target_fold], counts_temp[target_fold])
                
    # Scale to w=16384
    prior_w = 16384
    scaled_models = {}
    for f in [1, 2]:
        tot_i = counts_intra[f].sum(axis=1, keepdims=True)
        sc_i = np.maximum(np.round((counts_intra[f] / tot_i) * (prior_w / num_classes * 4)).astype(np.int64), 1)
        tot_t = counts_temp[f].sum(axis=1, keepdims=True)
        sc_t = np.maximum(np.round((counts_temp[f] / tot_t) * (prior_w / num_classes * 4)).astype(np.int64), 1)
        sc_t[:num_ctx_intra, :] = sc_i
        scaled_models[f] = {"intra": sc_i, "temp": sc_t}
        
    print(f"Computed 2-Fold models for K={k_val} in {time.perf_counter() - t0:.2f} s")
    return scaled_models

if __name__ == "__main__":
    m4 = compute_fast_models(4)
    m8 = compute_fast_models(8)
    
    out_path = Path("results/phase1/pretrained_models.npz")
    np.savez_compressed(
        out_path,
        k4_f1_intra=m4[1]["intra"],
        k4_f1_temp=m4[1]["temp"],
        k4_f2_intra=m4[2]["intra"],
        k4_f2_temp=m4[2]["temp"],
        k8_f1_intra=m8[1]["intra"],
        k8_f1_temp=m8[1]["temp"],
        k8_f2_intra=m8[2]["intra"],
        k8_f2_temp=m8[2]["temp"]
    )
    print(f"Saved pretrained models to {out_path} ({out_path.stat().st_size} bytes)")
