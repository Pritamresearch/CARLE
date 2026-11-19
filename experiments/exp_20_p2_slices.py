"""
Phase 2 (Full Dataset): Slice-Based Coding Benchmark Across All 60 nuScenes Sweep Streams
========================================================================================
Evaluates horizontal slicing:
  S in {1, 2, 4, 8}
at half resolution (450x800), K = 4 and K = 8 across all 60 nuScenes sweep streams (6,000 frames total).

Outputs:
1. results/phase2/p2_per_frame_per_slice_bytes.csv
   Columns: stream, frame_idx, K, S, slice_idx, bytes
2. results/phase2/p2_slice_benchmark.csv
   Dataset-wide summary table (overhead %, quantiles, % <= 1200 B)
"""

import sys
import os
import time
import json
import uuid
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

YANG_DIR = Path("baselines/yang2026").resolve()
YANG_ENC = str(YANG_DIR / "encoder.exe")

CACHE_DIR_HALF = Path("results/phase1b/cache_pidnet_half/sweeps")
OUT_DIR = Path("results/phase2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_slice_splits(H: int, S: int):
    base = H // S
    rem = H % S
    heights = [base + (1 if i < rem else 0) for i in range(S)]
    splits = []
    curr = 0
    for h in heights:
        splits.append((curr, curr + h))
        curr += h
    return splits

def encode_yang_slice(slice_arr: np.ndarray) -> int:
    H, W = slice_arr.shape
    u = uuid.uuid4().hex[:8]
    t_in = YANG_DIR / f"p2_{u}_in.yuv"
    t_out = YANG_DIR / f"p2_{u}_out.bin"
    slice_arr.tofile(t_in)
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
    try:
        subprocess.run(cmd, capture_output=True, cwd=str(YANG_DIR), stdin=subprocess.DEVNULL, timeout=20)
        bytes_val = t_out.stat().st_size if t_out.exists() else 0
    except Exception:
        bytes_val = 0
    finally:
        for p in [t_in, t_out]:
            if p.exists():
                try: p.unlink()
                except: pass
    return bytes_val

def encode_task(args):
    stream, frame_idx, k_val, s_val, slice_idx, slice_arr = args
    sz = encode_yang_slice(slice_arr)
    return stream, frame_idx, k_val, s_val, slice_idx, sz

def run_p2():
    print("="*80, flush=True)
    print("STARTING FULL P2 SLICE BENCHMARK ON ALL 60 NUSCENES STREAMS (6,000 FRAMES)", flush=True)
    print("="*80, flush=True)
    
    meta_files = sorted(list(CACHE_DIR_HALF.glob("sweeps_scene-*_meta.json")))
    print(f"Loaded {len(meta_files)} scenes.", flush=True)
    
    raw_csv = OUT_DIR / "p2_per_frame_per_slice_bytes.csv"
    completed_pairs = set()
    if raw_csv.exists():
        try:
            df_exist = pd.read_csv(raw_csv)
            if not df_exist.empty and "stream" in df_exist.columns:
                df_exist["scene"] = df_exist["stream"].apply(lambda x: x.split("_")[0])
                counts = df_exist.groupby(["scene", "K"]).size()
                for (sc, k), cnt in counts.items():
                    if cnt == 9000:
                        completed_pairs.add((sc, int(k)))
                        print(f"Found existing complete run: {sc} (K={k}) with {cnt} records.", flush=True)
        except Exception as e:
            print(f"Error reading existing CSV: {e}", flush=True)
            
    if not raw_csv.exists() or len(completed_pairs) == 0:
        with open(raw_csv, "w") as f:
            f.write("stream,frame_idx,K,S,slice_idx,bytes\n")
            
    # We will iterate scene by scene
    for k_val in [4, 8]:
        print(f"\n---> Processing K={k_val}...", flush=True)
        t0_k = time.time()
        
        for mf in meta_files:
            scene_name = mf.stem.replace("sweeps_", "").replace("_meta", "")
            if (scene_name, k_val) in completed_pairs:
                print(f"  [K={k_val}] Skipping already completed scene {scene_name}", flush=True)
                continue
                
            npz_p = CACHE_DIR_HALF / f"sweeps_{scene_name}_k{k_val}.npz"
            if not npz_p.exists():
                continue
                
            with open(mf, "r") as f:
                meta = json.load(f)
            masks = np.load(npz_p, mmap_mode="r")["masks"]
            
            # Map each of the 600 masks to (stream, frame_idx)
            cam_counts = {}
            tasks = []
            
            for idx in range(len(meta)):
                r = meta[idx]
                cam = r["camera"]
                if cam not in cam_counts:
                    cam_counts[cam] = 0
                f_idx = cam_counts[cam]
                cam_counts[cam] += 1
                
                stream = f"{r['scene']}_{cam}"
                mask = masks[idx]
                
                # S in {1, 2, 4, 8}
                for S in [1, 2, 4, 8]:
                    splits = get_slice_splits(450, S)
                    for s_idx, (r0, r1) in enumerate(splits):
                        slice_arr = mask[r0:r1, :]
                        tasks.append((stream, f_idx, k_val, S, s_idx, slice_arr))
                        
            # Execute tasks with 8 worker threads
            t0_scene = time.time()
            with ThreadPoolExecutor(max_workers=8) as ex:
                results = list(ex.map(encode_task, tasks))
                
            # Write results incrementally
            with open(raw_csv, "a") as f:
                for stream, frame_idx, k_v, s_v, s_idx, sz in results:
                    f.write(f"{stream},{frame_idx},{k_v},{s_v},{s_idx},{sz}\n")
                    
            dt_scene = time.time() - t0_scene
            print(f"  [K={k_val}] Completed scene {scene_name} (9,000 slices) in {dt_scene:.1f}s ({9000/dt_scene:.1f} slices/s)", flush=True)
            
        print(f"Finished K={k_val} loop in {(time.time() - t0_k)/60:.1f} minutes.", flush=True)
        
    df_raw = pd.read_csv(raw_csv)
    print(f"\nTotal slice records logged: {len(df_raw)}", flush=True)
    
    # Generate p2_slice_benchmark.csv
    benchmark_rows = []
    distribution_dict = {}
    
    for k_val in [4, 8]:
        df_k = df_raw[df_raw["K"] == k_val]
        
        # S=1 baseline
        s1_frames = df_k[df_k["S"] == 1].groupby(["stream", "frame_idx"])["bytes"].sum() * 8
        s1_mean_bits = float(s1_frames.mean())
        
        for S in [1, 2, 4, 8]:
            df_ks = df_k[df_k["S"] == S]
            frame_bits = df_ks.groupby(["stream", "frame_idx"])["bytes"].sum() * 8
            slice_bytes = df_ks["bytes"].values
            
            mean_fb = float(frame_bits.mean())
            mean_fB = mean_fb / 8.0
            overhead_pct = ((mean_fb - s1_mean_bits) / s1_mean_bits) * 100.0 if S > 1 else 0.0
            
            mean_sb = float(np.mean(slice_bytes))
            med_sb = float(np.median(slice_bytes))
            p95_sb = float(np.percentile(slice_bytes, 95))
            p99_sb = float(np.percentile(slice_bytes, 99))
            max_sb = int(np.max(slice_bytes))
            
            pct_le_1200 = float(np.mean(slice_bytes <= 1200) * 100.0)
            pct_le_600 = float(np.mean(slice_bytes <= 600) * 100.0)
            
            row = {
                "K": k_val,
                "Slices": S,
                "Mean_Frame_Bits": round(mean_fb, 1),
                "Mean_Frame_Bytes": round(mean_fB, 1),
                "Overhead_vs_S1_Pct": round(overhead_pct, 2),
                "Mean_Slice_Bytes": round(mean_sb, 1),
                "Median_Slice_Bytes": round(med_sb, 1),
                "P95_Slice_Bytes": round(p95_sb, 1),
                "P99_Slice_Bytes": round(p99_sb, 1),
                "Max_Slice_Bytes": max_sb,
                "Pct_Slices_LE_1200B": round(pct_le_1200, 2),
                "Pct_Slices_LE_600B": round(pct_le_600, 2)
            }
            benchmark_rows.append(row)
            distribution_dict[f"k{k_val}_s{S}"] = {
                "mean_slice_bytes": mean_sb,
                "p95_slice_bytes": p95_sb,
                "p99_slice_bytes": p99_sb,
                "max_slice_bytes": max_sb,
                "pct_le_1200": pct_le_1200,
                "pct_le_600": pct_le_600
            }
            
    df_bm = pd.DataFrame(benchmark_rows)
    df_bm.to_csv(OUT_DIR / "p2_slice_benchmark.csv", index=False)
    with open(OUT_DIR / "p2_slice_distribution.json", "w") as f:
        json.dump(distribution_dict, f, indent=2)
        
    print("\nSaved p2_slice_benchmark.csv and p2_slice_distribution.json:")
    print(df_bm.to_string(index=False))

if __name__ == "__main__":
    run_p2()
