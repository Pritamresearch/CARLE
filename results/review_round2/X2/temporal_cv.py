"""
X2: Temporal Ratios with Leave-One-Scene-Out Training (10-Fold Cross-Validation)
================================================================================
Evaluates SemCM intra (SemCM-a), persistent-counts, and compact-temporal variants
at half resolution (450x800) across all 60 nuScenes sweep streams (10 scenes x 6 cameras x 100 frames),
for K=4 and K=8.

Priors are retrained in each fold from the other 9 scenes (all 6 cameras) using the
repository's fast accumulation and scaling routine. Frame 0 of each stream is coded intra.
Decoders verify bit-exact lossless reproduction for every frame.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

# Enforce single-thread CPU protocol
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path.cwd()))

from carle.rigorous_codecs import (
    get_spatial_context_fast, precompute_bmap_fast,
    cm_prior_adaptive_encode
)
from experiments.exp_12_pidnet_sequence_benchmark import (
    get_spatial_context_k8, cm_k8_prior_adaptive_encode
)
from experiments.exp_13_debug_temporal import (
    get_compact_temporal_context_k4, get_compact_temporal_context_k8,
    cm_compact_temp_adaptive_encode
)

RC_TOP = np.int64(1 << 24)
RC_BOT = np.int64(1 << 16)
RC_TOTAL_FREQ = np.int64(16384)
MAX_TOTAL_FREQ = 16384

# -----------------------------------------------------------------------------
# Bit-Exact Numba Decoders
# -----------------------------------------------------------------------------

@njit(fastmath=True)
def cm_prior_adaptive_decode_k4(payload: np.ndarray, H: int, W: int,
                                counts_state: np.ndarray, totals_state: np.ndarray,
                                max_total: int = 16384) -> np.ndarray:
    out = np.zeros((H, W), dtype=np.uint8)
    in_ptr = 0
    code = np.int64(0)
    for _ in range(4):
        code = (code << 8) | (np.int64(payload[in_ptr]) if in_ptr < len(payload) else 0)
        in_ptr += 1
    low = np.int64(0)
    range_val = np.int64(0xFFFFFFFF)
    
    for r in range(H):
        for c in range(W):
            ctx = get_spatial_context_fast(out, r, c)
            tot = int(totals_state[ctx])
            r_step = range_val // RC_TOTAL_FREQ
            val = (code - low) // r_step
            if val < 0: val = 0
            if val >= RC_TOTAL_FREQ: val = RC_TOTAL_FREQ - 1
            
            cum = 0
            sym = 3
            c_low = 0
            c_high = 16384
            for s in range(4):
                c_low_s = (cum * 16384) // tot
                c_high_s = ((cum + int(counts_state[ctx, s])) * 16384) // tot
                if val >= c_low_s and val < c_high_s:
                    sym = s
                    c_low = c_low_s
                    c_high = c_high_s
                    break
                cum += int(counts_state[ctx, s])
                
            low = low + c_low * r_step
            range_val = (c_high - c_low) * r_step
            out[r, c] = np.uint8(sym + 1)
            
            counts_state[ctx, sym] += 1
            totals_state[ctx] += 1
            if totals_state[ctx] >= max_total:
                new_tot = 0
                for s in range(4):
                    counts_state[ctx, s] = (counts_state[ctx, s] + 1) >> 1
                    new_tot += counts_state[ctx, s]
                totals_state[ctx] = new_tot
                
            while True:
                if (low ^ (low + range_val)) < RC_TOP:
                    byte_val = payload[in_ptr] if in_ptr < len(payload) else 0
                    code = ((code << 8) | np.int64(byte_val)) & np.int64(0xFFFFFFFF)
                    in_ptr += 1
                    low = (low << 8) & np.int64(0xFFFFFFFF)
                    range_val = (range_val << 8) & np.int64(0xFFFFFFFF)
                elif range_val < RC_BOT:
                    byte_val = payload[in_ptr] if in_ptr < len(payload) else 0
                    code = ((code << 8) | np.int64(byte_val)) & np.int64(0xFFFFFFFF)
                    in_ptr += 1
                    range_val = ((-low) & (RC_BOT - 1)) << 8
                    low = (low << 8) & np.int64(0xFFFFFFFF)
                else:
                    break
    return out

@njit(fastmath=True)
def cm_prior_adaptive_decode_k8(payload: np.ndarray, H: int, W: int,
                                counts_state: np.ndarray, totals_state: np.ndarray,
                                max_total: int = 16384) -> np.ndarray:
    out = np.zeros((H, W), dtype=np.uint8)
    in_ptr = 0
    code = np.int64(0)
    for _ in range(4):
        code = (code << 8) | (np.int64(payload[in_ptr]) if in_ptr < len(payload) else 0)
        in_ptr += 1
    low = np.int64(0)
    range_val = np.int64(0xFFFFFFFF)
    
    for r in range(H):
        for c in range(W):
            ctx = get_spatial_context_k8(out, r, c)
            tot = int(totals_state[ctx])
            r_step = range_val // RC_TOTAL_FREQ
            val = (code - low) // r_step
            if val < 0: val = 0
            if val >= RC_TOTAL_FREQ: val = RC_TOTAL_FREQ - 1
            
            cum = 0
            sym = 7
            c_low = 0
            c_high = 16384
            for s in range(8):
                c_low_s = (cum * 16384) // tot
                c_high_s = ((cum + int(counts_state[ctx, s])) * 16384) // tot
                if val >= c_low_s and val < c_high_s:
                    sym = s
                    c_low = c_low_s
                    c_high = c_high_s
                    break
                cum += int(counts_state[ctx, s])
                
            low = low + c_low * r_step
            range_val = (c_high - c_low) * r_step
            out[r, c] = np.uint8(sym + 1)
            
            counts_state[ctx, sym] += 1
            totals_state[ctx] += 1
            if totals_state[ctx] >= max_total:
                new_tot = 0
                for s in range(8):
                    counts_state[ctx, s] = (counts_state[ctx, s] + 1) >> 1
                    new_tot += counts_state[ctx, s]
                totals_state[ctx] = new_tot
                
            while True:
                if (low ^ (low + range_val)) < RC_TOP:
                    byte_val = payload[in_ptr] if in_ptr < len(payload) else 0
                    code = ((code << 8) | np.int64(byte_val)) & np.int64(0xFFFFFFFF)
                    in_ptr += 1
                    low = (low << 8) & np.int64(0xFFFFFFFF)
                    range_val = (range_val << 8) & np.int64(0xFFFFFFFF)
                elif range_val < RC_BOT:
                    byte_val = payload[in_ptr] if in_ptr < len(payload) else 0
                    code = ((code << 8) | np.int64(byte_val)) & np.int64(0xFFFFFFFF)
                    in_ptr += 1
                    range_val = ((-low) & (RC_BOT - 1)) << 8
                    low = (low << 8) & np.int64(0xFFFFFFFF)
                else:
                    break
    return out

@njit(fastmath=True)
def cm_compact_temp_adaptive_decode_k4(payload: np.ndarray, prev: np.ndarray, H: int, W: int,
                                       counts_state: np.ndarray, totals_state: np.ndarray,
                                       max_total: int = 16384) -> np.ndarray:
    out = np.zeros((H, W), dtype=np.uint8)
    b_prev = precompute_bmap_fast(prev)
    in_ptr = 0
    code = np.int64(0)
    for _ in range(4):
        code = (code << 8) | (np.int64(payload[in_ptr]) if in_ptr < len(payload) else 0)
        in_ptr += 1
    low = np.int64(0)
    range_val = np.int64(0xFFFFFFFF)
    
    for r in range(H):
        for c in range(W):
            ctx = get_compact_temporal_context_k4(out, prev, b_prev, r, c)
            tot = int(totals_state[ctx])
            r_step = range_val // RC_TOTAL_FREQ
            val = (code - low) // r_step
            if val < 0: val = 0
            if val >= RC_TOTAL_FREQ: val = RC_TOTAL_FREQ - 1
            
            cum = 0
            sym = 3
            c_low = 0
            c_high = 16384
            for s in range(4):
                c_low_s = (cum * 16384) // tot
                c_high_s = ((cum + int(counts_state[ctx, s])) * 16384) // tot
                if val >= c_low_s and val < c_high_s:
                    sym = s
                    c_low = c_low_s
                    c_high = c_high_s
                    break
                cum += int(counts_state[ctx, s])
                
            low = low + c_low * r_step
            range_val = (c_high - c_low) * r_step
            out[r, c] = np.uint8(sym + 1)
            
            counts_state[ctx, sym] += 1
            totals_state[ctx] += 1
            if totals_state[ctx] >= max_total:
                new_tot = 0
                for s in range(4):
                    counts_state[ctx, s] = (counts_state[ctx, s] + 1) >> 1
                    new_tot += counts_state[ctx, s]
                totals_state[ctx] = new_tot
                
            while True:
                if (low ^ (low + range_val)) < RC_TOP:
                    byte_val = payload[in_ptr] if in_ptr < len(payload) else 0
                    code = ((code << 8) | np.int64(byte_val)) & np.int64(0xFFFFFFFF)
                    in_ptr += 1
                    low = (low << 8) & np.int64(0xFFFFFFFF)
                    range_val = (range_val << 8) & np.int64(0xFFFFFFFF)
                elif range_val < RC_BOT:
                    byte_val = payload[in_ptr] if in_ptr < len(payload) else 0
                    code = ((code << 8) | np.int64(byte_val)) & np.int64(0xFFFFFFFF)
                    in_ptr += 1
                    range_val = ((-low) & (RC_BOT - 1)) << 8
                    low = (low << 8) & np.int64(0xFFFFFFFF)
                else:
                    break
    return out

@njit(fastmath=True)
def cm_compact_temp_adaptive_decode_k8(payload: np.ndarray, prev: np.ndarray, H: int, W: int,
                                       counts_state: np.ndarray, totals_state: np.ndarray,
                                       max_total: int = 16384) -> np.ndarray:
    out = np.zeros((H, W), dtype=np.uint8)
    b_prev = precompute_bmap_fast(prev)
    in_ptr = 0
    code = np.int64(0)
    for _ in range(4):
        code = (code << 8) | (np.int64(payload[in_ptr]) if in_ptr < len(payload) else 0)
        in_ptr += 1
    low = np.int64(0)
    range_val = np.int64(0xFFFFFFFF)
    
    for r in range(H):
        for c in range(W):
            ctx = get_compact_temporal_context_k8(out, prev, b_prev, r, c)
            tot = int(totals_state[ctx])
            r_step = range_val // RC_TOTAL_FREQ
            val = (code - low) // r_step
            if val < 0: val = 0
            if val >= RC_TOTAL_FREQ: val = RC_TOTAL_FREQ - 1
            
            cum = 0
            sym = 7
            c_low = 0
            c_high = 16384
            for s in range(8):
                c_low_s = (cum * 16384) // tot
                c_high_s = ((cum + int(counts_state[ctx, s])) * 16384) // tot
                if val >= c_low_s and val < c_high_s:
                    sym = s
                    c_low = c_low_s
                    c_high = c_high_s
                    break
                cum += int(counts_state[ctx, s])
                
            low = low + c_low * r_step
            range_val = (c_high - c_low) * r_step
            out[r, c] = np.uint8(sym + 1)
            
            counts_state[ctx, sym] += 1
            totals_state[ctx] += 1
            if totals_state[ctx] >= max_total:
                new_tot = 0
                for s in range(8):
                    counts_state[ctx, s] = (counts_state[ctx, s] + 1) >> 1
                    new_tot += counts_state[ctx, s]
                totals_state[ctx] = new_tot
                
            while True:
                if (low ^ (low + range_val)) < RC_TOP:
                    byte_val = payload[in_ptr] if in_ptr < len(payload) else 0
                    code = ((code << 8) | np.int64(byte_val)) & np.int64(0xFFFFFFFF)
                    in_ptr += 1
                    low = (low << 8) & np.int64(0xFFFFFFFF)
                    range_val = (range_val << 8) & np.int64(0xFFFFFFFF)
                elif range_val < RC_BOT:
                    byte_val = payload[in_ptr] if in_ptr < len(payload) else 0
                    code = ((code << 8) | np.int64(byte_val)) & np.int64(0xFFFFFFFF)
                    in_ptr += 1
                    range_val = ((-low) & (RC_BOT - 1)) << 8
                    low = (low << 8) & np.int64(0xFFFFFFFF)
                else:
                    break
    return out

# -----------------------------------------------------------------------------
# Training Routine (Leave-One-Scene-Out)
# -----------------------------------------------------------------------------

@njit(fastmath=True)
def accumulate_scene_k4(masks: np.ndarray, c_intra: np.ndarray, c_temp: np.ndarray):
    # 6 cameras, each 100 frames
    for cam_idx in range(6):
        start = cam_idx * 100
        for f_i in range(1, 100):
            curr = masks[start + f_i]
            prev = masks[start + f_i - 1]
            b_prev = precompute_bmap_fast(prev)
            H, W = curr.shape
            for r in range(0, H, 2):
                for c in range(0, W, 2):
                    ctx_i = get_spatial_context_fast(curr, r, c)
                    ctx_t = get_compact_temporal_context_k4(curr, prev, b_prev, r, c)
                    sym = int(curr[r, c] - 1)
                    c_intra[ctx_i, sym] += 1
                    c_temp[ctx_t, sym] += 1

@njit(fastmath=True)
def accumulate_scene_k8(masks: np.ndarray, c_intra: np.ndarray, c_temp: np.ndarray):
    for cam_idx in range(6):
        start = cam_idx * 100
        for f_i in range(1, 100):
            curr = masks[start + f_i]
            prev = masks[start + f_i - 1]
            b_prev = precompute_bmap_fast(prev)
            H, W = curr.shape
            for r in range(0, H, 2):
                for c in range(0, W, 2):
                    ctx_i = get_spatial_context_k8(curr, r, c)
                    ctx_t = get_compact_temporal_context_k8(curr, prev, b_prev, r, c)
                    sym = int(curr[r, c] - 1)
                    c_intra[ctx_i, sym] += 1
                    c_temp[ctx_t, sym] += 1

def train_fold_prior(train_scenes: list, all_masks: dict, k_val: int):
    num_classes = k_val
    num_ctx_intra = 256 if k_val == 4 else 512
    num_ctx_temp = 2048 if k_val == 4 else 4096
    
    c_intra = np.ones((num_ctx_intra, num_classes), dtype=np.int64) * 8
    c_temp = np.ones((num_ctx_temp, num_classes), dtype=np.int64) * 8
    
    for sc in train_scenes:
        if k_val == 4:
            accumulate_scene_k4(all_masks[sc], c_intra, c_temp)
        else:
            accumulate_scene_k8(all_masks[sc], c_intra, c_temp)
            
    prior_w = 16384
    tot_i = c_intra.sum(axis=1, keepdims=True)
    sc_i = np.maximum(np.round((c_intra / tot_i) * (prior_w / num_classes * 4)).astype(np.int64), 1)
    
    tot_t = c_temp.sum(axis=1, keepdims=True)
    sc_t = np.maximum(np.round((c_temp / tot_t) * (prior_w / num_classes * 4)).astype(np.int64), 1)
    sc_t[:num_ctx_intra, :] = sc_i
    
    return sc_i, sc_t

# -----------------------------------------------------------------------------
# Main Evaluation Function
# -----------------------------------------------------------------------------

CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

def run_x2(smoke: bool = False):
    out_dir = Path("results/review_round2/X2")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sweeps_dir = Path("results/phase1b/cache_pidnet_half/sweeps")
    assert sweeps_dir.exists(), f"Sweeps dir missing: {sweeps_dir}"
    
    all_scenes = [
        "scene-0061", "scene-0103", "scene-0553", "scene-0655", "scene-0757",
        "scene-0796", "scene-0916", "scene-1077", "scene-1094", "scene-1100"
    ]
    
    test_scenes = ["scene-0061"] if smoke else all_scenes
    
    records = []
    t0_start = time.perf_counter()
    
    for k_val in [4, 8]:
        print(f"\n{'='*80}")
        print(f"RUNNING X2: 10-FOLD CV FOR K={k_val} (smoke={smoke})")
        print(f"{'='*80}")
        
        num_classes = k_val
        num_ctx_intra = 256 if k_val == 4 else 512
        num_ctx_temp = 2048 if k_val == 4 else 4096
        
        # Load all 10 scenes and pre-accumulate counts per scene
        all_masks = {}
        sc_c_intra = {}
        sc_c_temp = {}
        
        print("Pre-accumulating counts for all 10 scenes...")
        t_pre = time.perf_counter()
        tot_c_intra = np.zeros((num_ctx_intra, num_classes), dtype=np.int64)
        tot_c_temp = np.zeros((num_ctx_temp, num_classes), dtype=np.int64)
        
        for sc in all_scenes:
            npz_p = sweeps_dir / f"sweeps_{sc}_k{k_val}.npz"
            all_masks[sc] = np.load(npz_p)["masks"]
            c_i = np.zeros((num_ctx_intra, num_classes), dtype=np.int64)
            c_t = np.zeros((num_ctx_temp, num_classes), dtype=np.int64)
            if k_val == 4:
                accumulate_scene_k4(all_masks[sc], c_i, c_t)
            else:
                accumulate_scene_k8(all_masks[sc], c_i, c_t)
            sc_c_intra[sc] = c_i
            sc_c_temp[sc] = c_t
            tot_c_intra += c_i
            tot_c_temp += c_t
            
        print(f"Pre-accumulation complete in {time.perf_counter() - t_pre:.2f}s")
        
        prior_w = 16384
        for fold_idx, test_sc in enumerate(test_scenes):
            # Leave-one-scene-out counts: total minus test scene
            fold_c_i = (tot_c_intra - sc_c_intra[test_sc]) + 8
            fold_c_t = (tot_c_temp - sc_c_temp[test_sc]) + 8
            
            tot_i = fold_c_i.sum(axis=1, keepdims=True)
            prior_intra = np.maximum(np.round((fold_c_i / tot_i) * (prior_w / num_classes * 4)).astype(np.int64), 1)
            
            tot_t = fold_c_t.sum(axis=1, keepdims=True)
            prior_temp = np.maximum(np.round((fold_c_t / tot_t) * (prior_w / num_classes * 4)).astype(np.int64), 1)
            prior_temp[:num_ctx_intra, :] = prior_intra
            
            print(f"Fold {fold_idx+1}/{len(test_scenes)}: Test={test_sc}")
            
            test_masks = all_masks[test_sc]
            H, W = test_masks[0].shape
            
            for cam_i, cam_name in enumerate(CAMERAS):
                stream_name = f"{test_sc}_{cam_name}"
                cam_offset = cam_i * 100
                
                # Persistent state for stream
                pers_c = prior_intra.copy()
                pers_t = pers_c.sum(axis=1)
                
                # Persistent decode state
                pers_c_dec = prior_intra.copy()
                pers_t_dec = pers_c_dec.sum(axis=1)
                
                # Compact temp state for stream
                temp_c = prior_temp.copy()
                temp_t = temp_c.sum(axis=1)
                
                # Compact temp decode state
                temp_c_dec = prior_temp.copy()
                temp_t_dec = temp_c_dec.sum(axis=1)
                
                for f_idx in range(100):
                    curr = test_masks[cam_offset + f_idx]
                    prev = test_masks[cam_offset + f_idx - 1] if f_idx > 0 else curr
                    is_identical = np.array_equal(curr, prev) and (f_idx > 0)
                    
                    # ---------------------------------------------------------
                    # 1. SemCM Intra (SemCM-a)
                    # ---------------------------------------------------------
                    c_intra = prior_intra.copy()
                    t_intra = c_intra.sum(axis=1)
                    if k_val == 4:
                        p_intra = cm_prior_adaptive_encode(curr, prev, False, c_intra, t_intra, max_total=MAX_TOTAL_FREQ)
                        c_dec = prior_intra.copy()
                        t_dec = c_dec.sum(axis=1)
                        rec_intra = cm_prior_adaptive_decode_k4(p_intra, H, W, c_dec, t_dec, max_total=MAX_TOTAL_FREQ)
                    else:
                        p_intra = cm_k8_prior_adaptive_encode(curr, prev, False, c_intra, t_intra, max_total=MAX_TOTAL_FREQ)
                        c_dec = prior_intra.copy()
                        t_dec = c_dec.sum(axis=1)
                        rec_intra = cm_prior_adaptive_decode_k8(p_intra, H, W, c_dec, t_dec, max_total=MAX_TOTAL_FREQ)
                        
                    ok_intra = 1 if np.array_equal(curr, rec_intra) else 0
                    assert ok_intra == 1, f"Intra decode mismatch on {stream_name} frame {f_idx}"
                    b_intra = 8 if is_identical else (len(p_intra) * 8 + 1)
                    
                    # ---------------------------------------------------------
                    # 2. Persistent-Counts SemCM
                    # ---------------------------------------------------------
                    if k_val == 4:
                        p_pers = cm_prior_adaptive_encode(curr, prev, False, pers_c, pers_t, max_total=MAX_TOTAL_FREQ)
                        rec_pers = cm_prior_adaptive_decode_k4(p_pers, H, W, pers_c_dec, pers_t_dec, max_total=MAX_TOTAL_FREQ)
                    else:
                        p_pers = cm_k8_prior_adaptive_encode(curr, prev, False, pers_c, pers_t, max_total=MAX_TOTAL_FREQ)
                        rec_pers = cm_prior_adaptive_decode_k8(p_pers, H, W, pers_c_dec, pers_t_dec, max_total=MAX_TOTAL_FREQ)
                        
                    ok_pers = 1 if np.array_equal(curr, rec_pers) else 0
                    assert ok_pers == 1, f"Persistent decode mismatch on {stream_name} frame {f_idx}"
                    b_pers = 8 if is_identical else (len(p_pers) * 8 + 1)
                    
                    # ---------------------------------------------------------
                    # 3. Compact Temporal SemCM
                    # ---------------------------------------------------------
                    if f_idx == 0:
                        # Frame 0 coded intra
                        b_temp = b_intra
                        ok_temp = ok_intra
                    else:
                        p_temp = cm_compact_temp_adaptive_encode(curr, prev, k_val, temp_c, temp_t, max_total=MAX_TOTAL_FREQ)
                        if k_val == 4:
                            rec_temp = cm_compact_temp_adaptive_decode_k4(p_temp, prev, H, W, temp_c_dec, temp_t_dec, max_total=MAX_TOTAL_FREQ)
                        else:
                            rec_temp = cm_compact_temp_adaptive_decode_k8(p_temp, prev, H, W, temp_c_dec, temp_t_dec, max_total=MAX_TOTAL_FREQ)
                        ok_temp = 1 if np.array_equal(curr, rec_temp) else 0
                        assert ok_temp == 1, f"Compact temp decode mismatch on {stream_name} frame {f_idx}"
                        b_temp = 8 if is_identical else (len(p_temp) * 8 + 1)
                        
                    records.append({
                        "K": k_val,
                        "fold_test_scene": test_sc,
                        "stream": stream_name,
                        "frame_idx": f_idx,
                        "intra_bits": int(b_intra),
                        "persistent_bits": int(b_pers),
                        "compact_temp_bits": int(b_temp),
                        "decode_ok_intra": ok_intra,
                        "decode_ok_persistent": ok_pers,
                        "decode_ok_temp": ok_temp
                    })
                    
        del all_masks
        
    df_out = pd.DataFrame(records)
    out_csv = out_dir / "per_frame.csv"
    df_out.to_csv(out_csv, index=False)
    
    elapsed = time.perf_counter() - t0_start
    print(f"\nSaved {len(df_out)} rows to {out_csv} in {elapsed:.1f} s")
    
    # Sanity checks
    df_k4 = df_out[df_out["K"] == 4]
    mean_intra_k4 = float(df_k4["intra_bits"].mean())
    p1b_ref = 4554.1
    diff_pct = ((mean_intra_k4 - p1b_ref) / p1b_ref) * 100.0
    
    sanity_lines = [
        f"X2 SANITY CHECKS (smoke={smoke})",
        "="*50,
        f"Total frames evaluated: {len(df_out)} (K=4: {len(df_k4)}, K=8: {len(df_out)-len(df_k4)})",
        f"Mean intra_bits at K=4: {mean_intra_k4:.2f} bpf",
        f"P1b reference value   : {p1b_ref:.2f} bpf",
        f"Difference vs P1b     : {diff_pct:+.2f}%",
        f"Decode ok intra       : {int(df_out['decode_ok_intra'].sum())}/{len(df_out)} bit-exact",
        f"Decode ok persistent  : {int(df_out['decode_ok_persistent'].sum())}/{len(df_out)} bit-exact",
        f"Decode ok temp        : {int(df_out['decode_ok_temp'].sum())}/{len(df_out)} bit-exact",
        f"Mean persistent_bits K=4: {df_k4['persistent_bits'].mean():.2f} bpf",
        f"Mean compact_temp_bits K=4: {df_k4['compact_temp_bits'].mean():.2f} bpf",
        f"Temporal ratio K=4 (temp/intra): {df_k4['compact_temp_bits'].mean() / mean_intra_k4:.4f}",
        "="*50
    ]
    sanity_text = "\n".join(sanity_lines)
    print("\n" + sanity_text)
    
    sanity_p = out_dir / "sanity.txt"
    sanity_p.write_text(sanity_text + "\n", encoding="utf-8")
    print(f"Wrote sanity output to {sanity_p}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X2: Temporal ratios with 10-fold CV")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test on 1 scene")
    args = parser.parse_args()
    run_x2(smoke=args.smoke)
