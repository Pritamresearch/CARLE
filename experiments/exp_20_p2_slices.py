"""
Phase 2: Slice-Based Coding with Yang et al. 2026 at Half Resolution (450x800)
Evaluates independent horizontal slicing:
  S in {1, 2, 4, 8}
Across nuScenes sweeps (12 Hz) and field native at half resolution (450x800), K = 4 and K = 8.
Outputs:
- Bit overhead vs 1 slice: (bits(S) - bits(1)) / bits(1) * 100%
- Per-slice byte size distribution against a 1,200-byte payload limit
  (Mean, Median, 95th, 99th, Max, and % of slices <= 1200 bytes)
Saves results to:
- results/phase2/p2_slice_benchmark.csv
- results/phase2/p2_slice_distribution.json
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import json
import subprocess
import numpy as np
import pandas as pd
from collections import defaultdict

# Enforce single-thread CPU protocol
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

YANG_DIR = Path("baselines/yang2026").resolve()
YANG_ENC = str(YANG_DIR / "encoder.exe")

CACHE_DIR_HALF = Path("results/phase1b/cache_pidnet_half")
SWEEPS_DIR_HALF = CACHE_DIR_HALF / "sweeps"
OUT_DIR = Path("results/phase2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def encode_yang_slice(slice_arr: np.ndarray, worker_id: int = 0) -> int:
    H, W = slice_arr.shape
    pid = os.getpid()
    t_in = YANG_DIR / f"p2_{pid}_{worker_id}_in.yuv"
    t_out = YANG_DIR / f"p2_{pid}_{worker_id}_out.bin"
    slice_arr.tofile(t_in)
    cmd = [YANG_ENC, "-i", str(t_in), "-o", str(t_out), "-r", str(H), "-c", str(W), "-f", "1", "-s", "0", "-t", "400"]
    subprocess.run(cmd, capture_output=True, cwd=str(YANG_DIR), stdin=subprocess.DEVNULL)
    bytes_val = t_out.stat().st_size if t_out.exists() else 0
    for p in [t_in, t_out]:
        if p.exists():
            try: p.unlink()
            except: pass
    return bytes_val

def get_slice_splits(H: int, S: int):
    # Splits H into S horizontal slices
    # e.g. H=450, S=4 -> heights 112, 113, 112, 113
    base = H // S
    rem = H % S
    heights = [base + (1 if i < rem else 0) for i in range(S)]
    splits = []
    curr = 0
    for h in heights:
        splits.append((curr, curr + h))
        curr += h
    return splits

def load_representative_sequences(k_tag: str):
    # Load 12 Hz nuScenes sweeps (all 10 scenes, CAM_FRONT) and all 5 unflagged field native clips
    sequences = []
    
    # 1. nuScenes sweeps (10 scenes x 100 frames = 1,000 frames)
    for sf in sorted(list(SWEEPS_DIR_HALF.glob(f"sweeps_*_{k_tag}.npz"))):
        sc = sf.stem.replace(f"_{k_tag}", "").replace("sweeps_", "")
        meta_p = SWEEPS_DIR_HALF / f"sweeps_{sc}_meta.json"
        if sf.exists() and meta_p.exists():
            with open(meta_p) as f: meta = json.load(f)
            masks = np.load(sf, mmap_mode="r")["masks"]
            cam_front_idxs = [i for i, m in enumerate(meta) if m["camera"] == "CAM_FRONT"]
            sequences.append((f"nuScenes_Sweeps_{sc}_CAM_FRONT", [masks[i] for i in cam_front_idxs]))
            
    # 2. Field native (all 5 unflagged clips)
    for c in ["india_bolpur", "malaysia", "sikkim", "dhaka_day", "dhaka_night"]:
        p = CACHE_DIR_HALF / f"field_{c}_native_{k_tag}.npz"
        if p.exists():
            masks = np.load(p, mmap_mode="r")["masks"]
            sequences.append((f"Field_Native_{c}", [masks[i] for i in range(len(masks))]))
            
    return sequences

def run_p2_slices():
    print("\n" + "="*80)
    print("STARTING PHASE 2: SLICE-BASED CODING BENCHMARK (450x800, S in {1, 2, 4, 8})")
    print("="*80)
    
    all_summary = []
    distribution_data = {}
    
    for k_val in [4, 8]:
        k_tag = f"k{k_val}"
        print(f"\n--- Evaluating Slice Coding for K={k_val} ---")
        seqs = load_representative_sequences(k_tag)
        
        slice_counts = [1, 2, 4, 8]
        raw_rows = []
        slice_bytes_by_S = {S: [] for S in slice_counts}
        
        for seq_name, frames in seqs:
            print(f"Processing {seq_name} ({len(frames)} frames)...", flush=True)
            H, W = frames[0].shape
            
            for f_idx, frame in enumerate(frames):
                frame_bytes_by_S = {}
                
                for S in slice_counts:
                    splits = get_slice_splits(H, S)
                    s_bytes_list = []
                    for s_idx, (r_start, r_end) in enumerate(splits):
                        slice_img = np.ascontiguousarray(frame[r_start:r_end, :])
                        sb = encode_yang_slice(slice_img)
                        s_bytes_list.append(sb)
                        slice_bytes_by_S[S].append(sb)
                        
                    total_f_bytes = sum(s_bytes_list)
                    frame_bytes_by_S[S] = total_f_bytes
                    
                raw_rows.append({
                    "K": k_val,
                    "sequence": seq_name,
                    "frame_idx": f_idx,
                    "bytes_S1": frame_bytes_by_S[1],
                    "bytes_S2": frame_bytes_by_S[2],
                    "bytes_S4": frame_bytes_by_S[4],
                    "bytes_S8": frame_bytes_by_S[8]
                })
                
        df_raw = pd.DataFrame(raw_rows)
        
        # Compute summary statistics
        s1_mean_bytes = df_raw["bytes_S1"].mean()
        
        for S in slice_counts:
            col = f"bytes_S{S}"
            mean_bytes = df_raw[col].mean()
            mean_bits = mean_bytes * 8.0
            overhead_pct = ((mean_bytes - s1_mean_bytes) / s1_mean_bytes) * 100.0
            
            # Slice size distribution against 1200 byte payload
            s_arr = np.array(slice_bytes_by_S[S])
            pct_under_1200 = float(np.mean(s_arr <= 1200)) * 100.0
            pct_under_600 = float(np.mean(s_arr <= 600)) * 100.0
            
            summary_row = {
                "K": k_val,
                "Slices": S,
                "Mean_Frame_Bits": round(mean_bits, 1),
                "Mean_Frame_Bytes": round(mean_bytes, 1),
                "Overhead_vs_S1_Pct": round(overhead_pct, 2),
                "Mean_Slice_Bytes": round(float(np.mean(s_arr)), 1),
                "Median_Slice_Bytes": round(float(np.median(s_arr)), 1),
                "P95_Slice_Bytes": round(float(np.percentile(s_arr, 95)), 1),
                "P99_Slice_Bytes": round(float(np.percentile(s_arr, 99)), 1),
                "Max_Slice_Bytes": int(np.max(s_arr)),
                "Pct_Slices_LE_1200B": round(pct_under_1200, 2),
                "Pct_Slices_LE_600B": round(pct_under_600, 2)
            }
            all_summary.append(summary_row)
            
        distribution_data[f"k{k_val}"] = {
            f"S{S}": {
                "mean": round(float(np.mean(slice_bytes_by_S[S])), 1),
                "median": round(float(np.median(slice_bytes_by_S[S])), 1),
                "p95": round(float(np.percentile(slice_bytes_by_S[S], 95)), 1),
                "p99": round(float(np.percentile(slice_bytes_by_S[S], 99)), 1),
                "max": int(np.max(slice_bytes_by_S[S])),
                "pct_le_1200": round(float(np.mean(np.array(slice_bytes_by_S[S]) <= 1200)) * 100.0, 2)
            } for S in slice_counts
        }
        
    df_summary = pd.DataFrame(all_summary)
    out_csv = OUT_DIR / "p2_slice_benchmark.csv"
    out_json = OUT_DIR / "p2_slice_distribution.json"
    df_summary.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(distribution_data, f, indent=2)
        
    print(f"\n--- Phase 2 Slice-Based Coding Summary ---")
    print(df_summary.to_string(index=False))
    return df_summary

if __name__ == "__main__":
    run_p2_slices()
