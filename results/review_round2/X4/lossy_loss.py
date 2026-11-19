"""
X4: Downsampled Maps Under Packet Loss
======================================
Evaluates rate-accuracy tradeoffs under 10% Bernoulli packet loss on PIDNet-S
maps (6,000 frames across 10 scenes, 6 cameras, 100 frames) at K=4 and K=8.

Configurations:
- lossless (full resolution 450x800)
- down2 (nearest-neighbour downsampling by 2 -> 225x400, upsampled to 450x800)
- down4 (nearest-neighbour downsampling by 4 -> 112x200, upsampled to 450x800)

Simulator:
- Channel: Bernoulli with average packet loss p = 0.10.
- Design: chain_S1 (single packet per frame).
- Stream start: Frame 0 delivered intact. Loss applies from frame 1..99.
- Receiver displays last received frame upon packet loss.
- AoI: (t - t_src) * Delta_t + T_proc, where Delta_t = 1000/12 ms, T_proc = 103.4 ms.
- Pixel accuracy: share of pixels matching sender's full-res map at frame t.
- Class IoU: raw intersection and union counts per (K, config, seed, scene, class_index).
"""

import os
import sys
import time
import uuid
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import pandas as pd
from numba import njit

# Enforce single-thread CPU protocol for math libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

FPS = 12.0
DT_MS = 1000.0 / FPS  # 83.3333... ms
T_PROC = 103.4
CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

YANG_DIR = Path("baselines/yang2026").resolve()
YANG_ENC = str(YANG_DIR / "encoder.exe")

CLASS_NAMES_K4 = {
    1: "Road",
    2: "Vehicle",
    3: "VRU",
    4: "Other"
}

CLASS_NAMES_K8 = {
    1: "Road",
    2: "Sidewalk",
    3: "Static_structure",
    4: "Vegetation_terrain",
    5: "Sky",
    6: "Vehicle",
    7: "Two_wheeler",
    8: "Pedestrian"
}

def encode_single_frame(arr: np.ndarray, temp_id: str) -> int:
    H, W = arr.shape
    t_in = YANG_DIR / f"x4_{temp_id}_in.yuv"
    t_out = YANG_DIR / f"x4_{temp_id}_out.bin"
    try:
        arr.tofile(t_in)
        cmd = [
            YANG_ENC,
            "-i", str(t_in),
            "-o", str(t_out),
            "-r", str(H),
            "-c", str(W),
            "-f", "1",
            "-s", "0",
            "-t", "400"
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode == 0 and t_out.exists():
            return t_out.stat().st_size * 8
        else:
            return 48
    except Exception:
        return 48
    finally:
        if t_in.exists():
            try: t_in.unlink()
            except: pass
        if t_out.exists():
            try: t_out.unlink()
            except: pass

def compute_downsampled_bits_for_stream(masks: np.ndarray, factor: int, workers: int = 8) -> np.ndarray:
    """Encodes all 100 frames downsampled by factor."""
    H, W = masks.shape[1], masks.shape[2]
    target_w = W // factor
    target_h = H // factor
    
    down_masks = [cv2.resize(masks[i], (target_w, target_h), interpolation=cv2.INTER_NEAREST) for i in range(len(masks))]
    
    def worker_func(item):
        idx, arr = item
        tid = f"f{factor}_{uuid.uuid4().hex}"
        return encode_single_frame(arr, tid)
        
    with ThreadPoolExecutor(max_workers=workers) as ex:
        bits_list = list(ex.map(worker_func, enumerate(down_masks)))
        
    return np.array(bits_list, dtype=np.float64)

@njit
def compute_pair_confusion(pred_map: np.ndarray, gt_map: np.ndarray, K: int):
    """
    Computes exact matches, per-class intersection and union between candidate pred_map and ground truth gt_map.
    """
    H, W = gt_map.shape
    matches = 0
    inter = np.zeros(K + 1, dtype=np.int64)
    uni = np.zeros(K + 1, dtype=np.int64)
    
    for r in range(H):
        for c in range(W):
            p = pred_map[r, c]
            g = gt_map[r, c]
            if p == g:
                matches += 1
            for k in range(1, K + 1):
                p_k = (p == k)
                g_k = (g == k)
                if p_k and g_k:
                    inter[k] += 1
                if p_k or g_k:
                    uni[k] += 1
                    
    return matches, inter, uni

def run_x4(smoke: bool = False):
    out_dir = Path("results/review_round2/X4")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sweeps_dir = Path("results/phase1b/cache_pidnet_half/sweeps")
    assert sweeps_dir.exists(), f"Sweeps dir missing: {sweeps_dir}"
    
    p1b_k4 = pd.read_csv("results/phase1/p1b_all_frames_raw_k4.csv")
    p1b_k8 = pd.read_csv("results/phase1/p1b_all_frames_raw_k8.csv")
    p1b_k4_sw = p1b_k4[p1b_k4["dataset"] == "nuScenes_Sweeps_12Hz"]
    p1b_k8_sw = p1b_k8[p1b_k8["dataset"] == "nuScenes_Sweeps_12Hz"]
    
    all_scenes = [
        "scene-0061", "scene-0103", "scene-0553", "scene-0655", "scene-0757",
        "scene-0796", "scene-0916", "scene-1077", "scene-1094", "scene-1100"
    ]
    eval_scenes = ["scene-0061"] if smoke else all_scenes
    seeds = [0] if smoke else [0, 1, 2, 3, 4]
    configs = ["lossless", "down2", "down4"]
    
    per_stream_rows = []
    
    # Structure for class_iou: (K, config, seed, scene, class_index) -> [intersection_sum, union_sum]
    class_iou_totals = {}
    
    t0_start = time.perf_counter()
    
    for K in [4, 8]:
        p1b_sw = p1b_k4_sw if K == 4 else p1b_k8_sw
        class_names = CLASS_NAMES_K4 if K == 4 else CLASS_NAMES_K8
        
        print(f"\n{'='*80}", flush=True)
        print(f"RUNNING X4: DOWNSAMPLED MAPS BENCHMARK FOR K={K} (smoke={smoke})", flush=True)
        print(f"{'='*80}", flush=True)
        
        for sc_idx, sc in enumerate(eval_scenes):
            t_sc = time.perf_counter()
            npz_p = sweeps_dir / f"sweeps_{sc}_k{K}.npz"
            masks = np.load(npz_p)["masks"]  # (600, 450, 800)
            
            for cam_i, cam_name in enumerate(CAMERAS):
                stream_name = f"{sc}_{cam_name}"
                stream_masks = masks[cam_i * 100 : (cam_i + 1) * 100]  # (100, 450, 800)
                
                # Precompute candidate displayed frames for each config:
                # For lossless: U[t] = stream_masks[t]
                # For down2: downsample by 2 then upsample to 450x800
                # For down4: downsample by 4 then upsample to 450x800
                upsampled_cache = {}
                upsampled_cache["lossless"] = stream_masks
                
                down2_ups = np.zeros_like(stream_masks)
                down4_ups = np.zeros_like(stream_masks)
                for f in range(100):
                    orig = stream_masks[f]
                    # down2:
                    d2 = cv2.resize(orig, (400, 225), interpolation=cv2.INTER_NEAREST)
                    down2_ups[f] = cv2.resize(d2, (800, 450), interpolation=cv2.INTER_NEAREST)
                    # down4:
                    d4 = cv2.resize(orig, (200, 112), interpolation=cv2.INTER_NEAREST)
                    down4_ups[f] = cv2.resize(d4, (800, 450), interpolation=cv2.INTER_NEAREST)
                    
                upsampled_cache["down2"] = down2_ups
                upsampled_cache["down4"] = down4_ups
                
                # Bits per frame for each config:
                p1b_sub = p1b_sw[p1b_sw["stream"] == stream_name].sort_values("frame_idx")
                lossless_bits = p1b_sub["yang_bits"].values if len(p1b_sub) == 100 else np.zeros(100, dtype=np.float64)
                
                bits_cache = {}
                bits_cache["lossless"] = lossless_bits
                bits_cache["down2"] = compute_downsampled_bits_for_stream(stream_masks, factor=2)
                bits_cache["down4"] = compute_downsampled_bits_for_stream(stream_masks, factor=4)
                
                # Pair cache for on-demand (s, t) confusion computation
                # pair_cache[cfg][(s, t)] -> (matches, inter, uni)
                pair_cache = {cfg: {} for cfg in configs}
                
                for cfg in configs:
                    mean_bits_val = float(np.mean(bits_cache[cfg]))
                    U_arr = upsampled_cache[cfg]
                    
                    for seed in seeds:
                        rng = np.random.default_rng(seed * 10000 + cam_i * 100 + K * 1000)
                        
                        trial_accs = []
                        trial_aois = []
                        
                        # Generate 100 trials
                        for tr in range(100):
                            # Bernoulli 10% packet loss
                            loss = rng.random(100) < 0.10
                            loss[0] = False  # Frame 0 delivered intact
                            
                            last_rcv = 0
                            tr_matches = 0
                            tr_aoi_sum = 0.0
                            
                            for t in range(100):
                                if not loss[t]:
                                    last_rcv = t
                                s = last_rcv
                                
                                aoi_frame = (t - s) * DT_MS + T_PROC
                                tr_aoi_sum += aoi_frame
                                
                                # Query or compute pair confusion
                                if (s, t) not in pair_cache[cfg]:
                                    pair_cache[cfg][(s, t)] = compute_pair_confusion(U_arr[s], stream_masks[t], K)
                                    
                                m_cnt, inter_arr, uni_arr = pair_cache[cfg][(s, t)]
                                tr_matches += m_cnt
                                
                                # Accumulate to global class IoU for this scene and seed
                                for k_cls in range(1, K + 1):
                                    key = (K, cfg, seed, sc, k_cls)
                                    if key not in class_iou_totals:
                                        class_iou_totals[key] = [0, 0]
                                    class_iou_totals[key][0] += inter_arr[k_cls]
                                    class_iou_totals[key][1] += uni_arr[k_cls]
                                    
                            trial_accs.append(tr_matches / (100.0 * 450.0 * 800.0))
                            trial_aois.append(tr_aoi_sum / 100.0)
                            
                        per_stream_rows.append({
                            "K": K,
                            "config": cfg,
                            "seed": seed,
                            "scene": sc,
                            "stream": stream_name,
                            "n_trials": 100,
                            "mean_bits": round(mean_bits_val, 2),
                            "mean_pixel_acc": round(float(np.mean(trial_accs)), 6),
                            "mean_frame_aoi_ms": round(float(np.mean(trial_aois)), 3)
                        })
                        
            print(f"K={K}: Completed scene {sc} ({sc_idx+1}/{len(eval_scenes)}) in {time.perf_counter() - t_sc:.2f} s", flush=True)
            
    # Save per_stream.csv
    df_stream = pd.DataFrame(per_stream_rows)
    p_stream_csv = out_dir / "per_stream.csv"
    df_stream.to_csv(p_stream_csv, index=False)
    print(f"\nSaved {len(df_stream)} rows to {p_stream_csv}", flush=True)
    
    # Save class_iou.csv
    iou_rows = []
    for (k_val, cfg, seed, sc, k_cls), (inter_cnt, uni_cnt) in sorted(class_iou_totals.items()):
        c_name = CLASS_NAMES_K4[k_cls] if k_val == 4 else CLASS_NAMES_K8[k_cls]
        iou_rows.append({
            "K": k_val,
            "config": cfg,
            "seed": seed,
            "scene": sc,
            "class_index": k_cls,
            "class_name": c_name,
            "intersection": int(inter_cnt),
            "union": int(uni_cnt)
        })
        
    df_iou = pd.DataFrame(iou_rows)
    p_iou_csv = out_dir / "class_iou.csv"
    df_iou.to_csv(p_iou_csv, index=False)
    print(f"Saved {len(df_iou)} rows to {p_iou_csv}", flush=True)
    
    # Sanity checks
    sanity_lines = [
        f"X4 SANITY CHECKS (smoke={smoke})",
        "="*60,
        f"Total stream rows: {len(df_stream)}",
        f"Total class IoU rows: {len(df_iou)}"
    ]
    for K in [4, 8]:
        sub_k = df_stream[df_stream["K"] == K]
        sanity_lines.append(f"\n--- K={K} Aggregates ---")
        for cfg in configs:
            sub_cfg = sub_k[sub_k["config"] == cfg]
            m_bits = sub_cfg["mean_bits"].mean()
            m_acc = sub_cfg["mean_pixel_acc"].mean() * 100.0
            m_aoi = sub_cfg["mean_frame_aoi_ms"].mean()
            sanity_lines.append(f"{cfg:10s}: bits={m_bits:7.1f} bpf, pixel_acc={m_acc:6.2f}%, AoI={m_aoi:6.2f} ms")
            
    sanity_lines.append("\n" + "="*60)
    sanity_text = "\n".join(sanity_lines)
    print("\n" + sanity_text, flush=True)
    
    sanity_p = out_dir / "sanity.txt"
    sanity_p.write_text(sanity_text + "\n", encoding="utf-8")
    print(f"Wrote sanity output to {sanity_p}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X4: Downsampled maps under packet loss")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test on 1 scene, 1 seed")
    args = parser.parse_args()
    run_x4(smoke=args.smoke)
