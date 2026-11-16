"""
Phase 1: Build Final Master Tables for V2X Semantic-Map Sharing Study
Calculates PNG (level 9) across all cached frames, merges with Yang, SemCM (a), JPEG-XL e9, WebP,
and generates the final master tables with scene/clip-level 95% bootstrap CIs.
Hardware strictly loaded from results/phase0/env.json:
- CPU: Intel Core i7-13620H
- GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
"""

import sys
import os
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict

# Enforce single-thread CPU protocol
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

CACHE_DIR = Path("results/phase1/cache_pidnet")
SWEEPS_DIR = CACHE_DIR / "sweeps"

def load_env():
    with open("results/phase0/env.json") as f:
        meta = json.load(f)
    print(f"Hardware: CPU = {meta['cpu']['model']} | GPU = {meta['gpu']['name']}")
    return meta

def compute_png9_bits(mask):
    _, buf = cv2.imencode('.png', mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return len(buf) * 8

def compute_png9_for_dataset(df_raw, k_val):
    k_tag = f"k{k_val}"
    print(f"\nComputing PNG level 9 for K={k_val} ({len(df_raw)} frames)...")
    
    png_bits_list = []
    
    # Cache open npz
    current_path = None
    current_masks = None
    
    # Map dataset/stream to file path
    for idx, row in df_raw.iterrows():
        ds = row["dataset"]
        stream = row["stream"]
        f_idx = int(row["frame_idx"])
        
        if ds == "nuScenes_Keyframes_2Hz":
            file_path = CACHE_DIR / f"nuscenes_keyframes_2hz_{k_tag}.npz"
        elif ds == "nuScenes_Sweeps_12Hz":
            file_path = SWEEPS_DIR / f"sweeps_{stream}_{k_tag}.npz"
        else: # Field
            fps_tag = "10hz" if "10Hz" in ds else "native"
            file_path = CACHE_DIR / f"field_{stream}_{fps_tag}_{k_tag}.npz"
            
        if file_path != current_path:
            current_masks = np.load(file_path)["masks"]
            current_path = file_path
            
        mask = current_masks[f_idx]
        png_bits = compute_png9_bits(mask)
        png_bits_list.append(png_bits)
        
        if (len(png_bits_list)) % 2500 == 0:
            print(f"  Processed {len(png_bits_list)}/{len(df_raw)} frames...")
            
    df_raw["png9_bits"] = png_bits_list
    return df_raw

def bootstrap_ci(df_ds, col_name, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    streams = df_ds["stream"].unique()
    n_streams = len(streams)
    
    if n_streams <= 1:
        # If single stream, bootstrap over frames
        vals = df_ds[col_name].values
        boot_means = [np.mean(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n_boot)]
    else:
        # Clustered bootstrap over streams/scenes
        stream_data = {s: df_ds[df_ds["stream"] == s][col_name].values for s in streams}
        boot_means = []
        for _ in range(n_boot):
            sampled_streams = rng.choice(streams, size=n_streams, replace=True)
            sampled_vals = np.concatenate([stream_data[s] for s in sampled_streams])
            boot_means.append(np.mean(sampled_vals))
            
    ci_low = np.percentile(boot_means, 2.5)
    ci_high = np.percentile(boot_means, 97.5)
    return float(ci_low), float(ci_high)

def generate_master_summary(k_val):
    raw_path = Path(f"results/phase1/benchmark_raw_k{k_val}_unified.csv")
    df = pd.read_csv(raw_path)
    
    # Compute PNG9 if not present
    if "png9_bits" not in df.columns:
        df = compute_png9_for_dataset(df, k_val)
        df.to_csv(raw_path, index=False)
        
    datasets = ["nuScenes_Keyframes_2Hz", "nuScenes_Sweeps_12Hz", "Field_Native", "Field_10Hz"]
    codec_cols = [
        ("PNG (OpenCV level 9)", "png9_bits"),
        ("WebP Lossless", "webp_bits"),
        ("JPEG-XL (Effort 9)", "jxl9_bits"),
        ("SemCM (a) [Frame-Indep] (Frozen Ref)", "semcm_a_bits"),
        ("Yang et al. 2026", "yang_bits")
    ]
    
    rows = []
    
    for ds in datasets:
        df_ds = df[df["dataset"] == ds]
        ref_mean = df_ds["semcm_a_bits"].mean()
        
        for codec_name, col in codec_cols:
            mean_bpf = float(df_ds[col].mean())
            ci_low, ci_high = bootstrap_ci(df_ds, col)
            
            # 10 Hz single camera and 6-camera bitrates
            kbps_1cam = (mean_bpf * 10.0) / 1000.0
            kbps_6cam = kbps_1cam * 6.0
            
            # Gain vs SemCM (a): positive means fewer bits (compression savings)
            gain_pct = ((ref_mean - mean_bpf) / ref_mean) * 100.0 if ref_mean > 0 else 0.0
            
            # Relative CI for gain
            # Bootstrap gain directly
            rng = np.random.default_rng(42)
            streams = df_ds["stream"].unique()
            if len(streams) <= 1:
                vals_c = df_ds[col].values
                vals_r = df_ds["semcm_a_bits"].values
                boot_gains = []
                for _ in range(1000):
                    idx = rng.choice(len(vals_c), size=len(vals_c), replace=True)
                    m_c = np.mean(vals_c[idx])
                    m_r = np.mean(vals_r[idx])
                    boot_gains.append(((m_r - m_c) / m_r) * 100.0)
            else:
                s_c = {s: df_ds[df_ds["stream"] == s][col].values for s in streams}
                s_r = {s: df_ds[df_ds["stream"] == s]["semcm_a_bits"].values for s in streams}
                boot_gains = []
                for _ in range(1000):
                    sampled_s = rng.choice(streams, size=len(streams), replace=True)
                    m_c = np.mean(np.concatenate([s_c[s] for s in sampled_s]))
                    m_r = np.mean(np.concatenate([s_r[s] for s in sampled_s]))
                    boot_gains.append(((m_r - m_c) / m_r) * 100.0)
                    
            gain_ci_low = float(np.percentile(boot_gains, 2.5))
            gain_ci_high = float(np.percentile(boot_gains, 97.5))
            
            rows.append({
                "K": k_val,
                "Dataset": ds,
                "N_Frames": len(df_ds),
                "Codec": codec_name,
                "Bits/Frame": round(mean_bpf, 1),
                "95% CI Lower": round(ci_low, 1),
                "95% CI Upper": round(ci_high, 1),
                "1-Cam 10Hz (kb/s)": round(kbps_1cam, 2),
                "6-Cam 10Hz (kb/s)": round(kbps_6cam, 2),
                "Gain vs SemCM(a) (%)": round(gain_pct, 2),
                "Gain CI Lower": round(gain_ci_low, 2),
                "Gain CI Upper": round(gain_ci_high, 2)
            })
            
    df_summary = pd.DataFrame(rows)
    out_csv = Path(f"results/phase1/p1_master_table_k{k_val}.csv")
    df_summary.to_csv(out_csv, index=False)
    print(f"Saved master summary to {out_csv}")
    return df_summary

def generate_flagged_summary(k_val):
    flagged_p = Path(f"results/phase1/flagged_bhubaneswar_raw_k{k_val}.csv")
    df = pd.read_csv(flagged_p)
    
    k_tag = f"k{k_val}"
    # Compute PNG9 for flagged clip if needed
    if "png9_bits" not in df.columns:
        png_bits_list = []
        for idx, row in df.iterrows():
            ds = row["dataset"]
            f_idx = int(row["frame_idx"])
            fps_tag = "10hz" if "10Hz" in ds else "native"
            file_path = CACHE_DIR / f"field_bhubaneswar_night_{fps_tag}_{k_tag}.npz"
            mask = np.load(file_path)["masks"][f_idx]
            png_bits_list.append(compute_png9_bits(mask))
        df["png9_bits"] = png_bits_list
        df.to_csv(flagged_p, index=False)
        
    rows = []
    codec_cols = [
        ("PNG (OpenCV level 9)", "png9_bits"),
        ("WebP Lossless", "webp_bits"),
        ("JPEG-XL (Effort 9)", "jxl9_bits"),
        ("SemCM (a) [Frame-Indep] (Frozen Ref)", "semcm_a_bits"),
        ("Yang et al. 2026", "yang_bits")
    ]
    
    for ds in ["Field_Bhubaneswar_Night_Native", "Field_Bhubaneswar_Night_10Hz"]:
        df_ds = df[df["dataset"] == ds]
        ref_mean = df_ds["semcm_a_bits"].mean()
        for codec_name, col in codec_cols:
            mean_bpf = float(df_ds[col].mean())
            ci_low, ci_high = bootstrap_ci(df_ds, col)
            kbps_1cam = (mean_bpf * 10.0) / 1000.0
            kbps_6cam = kbps_1cam * 6.0
            gain_pct = ((ref_mean - mean_bpf) / ref_mean) * 100.0
            
            rows.append({
                "K": k_val,
                "Dataset": ds,
                "N_Frames": len(df_ds),
                "Codec": codec_name,
                "Bits/Frame": round(mean_bpf, 1),
                "95% CI Lower": round(ci_low, 1),
                "95% CI Upper": round(ci_high, 1),
                "1-Cam 10Hz (kb/s)": round(kbps_1cam, 2),
                "6-Cam 10Hz (kb/s)": round(kbps_6cam, 2),
                "Gain vs SemCM(a) (%)": round(gain_pct, 2)
            })
            
    df_flagged = pd.DataFrame(rows)
    out_csv = Path(f"results/phase1/p1_flagged_bhubaneswar_table_k{k_val}.csv")
    df_flagged.to_csv(out_csv, index=False)
    print(f"Saved flagged summary to {out_csv}")
    return df_flagged

if __name__ == "__main__":
    load_env()
    for k in [4, 8]:
        generate_master_summary(k)
        generate_flagged_summary(k)
    print("\nAll Phase 1 Master Tables successfully generated!")
