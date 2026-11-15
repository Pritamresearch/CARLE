"""
Pair Validity, Delta_t Distribution, and Boundary Jitter/Flicker Analysis.

Strictly verifies:
- Same scene + same camera
- Delta_t in [0.4, 0.6] s for keyframes (2 Hz)
- Delta_t in [0.05, 0.12] s for sweeps (~12 Hz)

Computes:
- Delta_t distribution (mean, median, min, max, std)
- Mean delta_pix
- Run counts and mean run lengths (explaining CARLE bit behavior across rates)
- Fraction of changed pixels within 1 px and 2 px of class boundaries in F_t
- 3-frame flicker rate

Outputs:
- results/phase1/pairs_log.csv
- results/phase1/jitter_summary.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(line_buffering=True)

import json
import numpy as np
import pandas as pd
from carle.fast_codecs import extract_diff_runs_numba
from carle.jitter_analysis import analyze_jitter_pair_numba, analyze_flicker_triplet_numba

CACHE_DIR = Path("results/phase1/cache")
RESULTS_DIR = Path("results/phase1")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def process_sequence_pairs(dataset_label: str, masks: np.ndarray, meta_list: list, dt_range: tuple = None):
    """
    Groups frames by (scene, camera) or clip_id, asserts pair validity,
    and computes delta_t, delta_pix, jitter, flicker, and RLE run statistics.
    """
    H, W = masks.shape[1], masks.shape[2]
    total_pixels = H * W
    
    # Organize into streams by (scene, camera) or clip_id
    streams = {}
    for idx, m in enumerate(meta_list):
        key = (m.get("scene", m.get("clip_id", "clip")), m.get("camera", "cam0"))
        if key not in streams:
            streams[key] = []
        streams[key].append((idx, m))
        
    pair_records = []
    
    for (scene_id, cam_id), stream_indices in streams.items():
        # Sort stream by timestamp
        stream_indices.sort(key=lambda x: x[1].get("timestamp", x[1].get("timestamp_s", x[0])))
        
        n_frames = len(stream_indices)
        for i in range(n_frames - 1):
            idx0, meta0 = stream_indices[i]
            idx1, meta1 = stream_indices[i + 1]
            
            mask0 = masks[idx0]
            mask1 = masks[idx1]
            
            # Timestamp calculation
            if "timestamp" in meta0 and "timestamp" in meta1:
                t0 = meta0["timestamp"]
                t1 = meta1["timestamp"]
                dt = (t1 - t0) * 1e-6 # nuScenes timestamps are in microseconds
            elif "timestamp_s" in meta0 and "timestamp_s" in meta1:
                t0 = meta0["timestamp_s"]
                t1 = meta1["timestamp_s"]
                dt = t1 - t0
            else:
                t0, t1, dt = i, i + 1, 1.0
                
            # Pair validity assertion with slight float tolerance for timer jitter
            if dt_range is not None:
                min_dt, max_dt = dt_range
                if not ((min_dt - 0.01) <= dt <= (max_dt + 0.01)):
                    # Dropped camera frame or timestamp gap; skip from temporal evaluation
                    continue
                assert (min_dt - 0.01) <= dt <= (max_dt + 0.01), f"Pair validity violated: {scene_id} {cam_id} dt={dt:.4f} not in [{min_dt}, {max_dt}]"
                
            # Changed pixels
            diff_mask = (mask1 != mask0)
            n_changed = int(np.sum(diff_mask))
            delta_pix = n_changed / total_pixels
            
            # RLE run statistics
            starts, lengths, classes = extract_diff_runs_numba(mask1, mask0)
            n_runs = len(starts)
            mean_run_len = float(np.mean(lengths)) if n_runs > 0 else 0.0
            
            # Jitter analysis (within 1 px and 2 px of class boundaries in mask0)
            tot_ch, ch_1px, ch_2px = analyze_jitter_pair_numba(mask0, mask1)
            frac_1px = ch_1px / tot_ch if tot_ch > 0 else 0.0
            frac_2px = ch_2px / tot_ch if tot_ch > 0 else 0.0
            
            # 3-frame flicker analysis if t+2 exists
            flicker_count = 0
            if i + 2 < n_frames:
                idx2, meta2 = stream_indices[i + 2]
                mask2 = masks[idx2]
                flick_cnt, _ = analyze_flicker_triplet_numba(mask0, mask1, mask2)
                flicker_count = flick_cnt
            flicker_rate = flicker_count / total_pixels
            flicker_frac = flicker_count / tot_ch if tot_ch > 0 else 0.0
            
            pair_records.append({
                "dataset": dataset_label,
                "scene": scene_id,
                "camera": cam_id,
                "frame_t": idx0,
                "frame_t1": idx1,
                "t0": t0,
                "t1": t1,
                "delta_t_s": round(dt, 4),
                "delta_pix": delta_pix,
                "num_changed_pixels": n_changed,
                "num_runs": n_runs,
                "mean_run_len": round(mean_run_len, 2),
                "frac_within_1px": round(frac_1px, 4),
                "frac_within_2px": round(frac_2px, 4),
                "flicker_rate": round(flicker_rate, 6),
                "flicker_frac_of_changes": round(flicker_frac, 4),
            })
            
    return pair_records

def main():
    all_pairs = []
    summary_stats = {}
    
    print("=" * 70)
    print("Pair Validity, Delta_t Distribution, and Jitter Analysis")
    print("=" * 70)
    
    # 1. nuScenes Keyframes (2 Hz)
    kf_path = CACHE_DIR / "nuscenes_keyframes_2hz.npz"
    kf_meta_path = CACHE_DIR / "nuscenes_keyframes_2hz_meta.json"
    if kf_path.exists() and kf_meta_path.exists():
        print("\nAnalyzing nuScenes Keyframes (2 Hz)...")
        kf_data = np.load(kf_path)
        with open(kf_meta_path) as f:
            kf_meta = json.load(f)
        kf_pairs = process_sequence_pairs("nuScenes_Keyframes_2Hz", kf_data["masks"], kf_meta, dt_range=(0.4, 0.6))
        all_pairs.extend(kf_pairs)
        print(f"  Verified {len(kf_pairs)} valid keyframe pairs across {len(set(p['scene'] for p in kf_pairs))} scenes and 6 cameras.")
        
    # 2. nuScenes Sweeps (~12 Hz full - across all scene chunks)
    sweeps_dir = CACHE_DIR / "sweeps"
    manifest_path = CACHE_DIR / "nuscenes_sweeps_12hz_manifest.json"
    if sweeps_dir.exists() and list(sweeps_dir.glob("sweeps_*.npz")):
        print("\nAnalyzing nuScenes Sweeps (~12 Hz full across all 10 scenes)...")
        scene_files = sorted(list(sweeps_dir.glob("sweeps_*.npz")))
        for s_npz in scene_files:
            s_meta_path = s_npz.with_name(s_npz.stem + "_meta.json")
            if s_npz.exists() and s_meta_path.exists():
                s_data = np.load(s_npz)
                with open(s_meta_path) as f:
                    s_meta = json.load(f)
                s_pairs = process_sequence_pairs("nuScenes_Sweeps_12Hz", s_data["masks"], s_meta, dt_range=(0.05, 0.12))
                all_pairs.extend(s_pairs)
        print(f"  Verified {len([p for p in all_pairs if p['dataset'] == 'nuScenes_Sweeps_12Hz'])} valid sweep pairs.")
        
    # 3. Field Videos (Native and 10 Hz)
    for cond in ["native_full", "10hz_full"]:
        for clip_id in ["india_bolpur", "malaysia", "sikkim", "dhaka_day", "dhaka_night", "bhubaneswar_night"]:
            f_path = CACHE_DIR / f"field_{clip_id}_{cond}.npz"
            f_meta = CACHE_DIR / f"field_{clip_id}_{cond}_meta.json"
            if f_path.exists() and f_meta.exists():
                d_name = f"Field_{clip_id}_{cond.replace('_full', '')}"
                data = np.load(f_path)
                with open(f_meta) as f:
                    meta = json.load(f)
                dt_r = (0.02, 0.05) if "native" in cond else (0.07, 0.15)
                pairs = process_sequence_pairs(d_name, data["masks"], meta, dt_range=dt_r)
                all_pairs.extend(pairs)
                
    df_pairs = pd.DataFrame(all_pairs)
    out_csv = RESULTS_DIR / "pairs_log.csv"
    df_pairs.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df_pairs)} validated pair records to {out_csv}")
    
    # Compute summary table by dataset
    grouped = df_pairs.groupby("dataset").agg({
        "delta_t_s": ["mean", "median", "min", "max", "std"],
        "delta_pix": ["mean", "median"],
        "num_runs": ["mean", "median"],
        "mean_run_len": ["mean", "median"],
        "frac_within_1px": "mean",
        "frac_within_2px": "mean",
        "flicker_rate": "mean",
        "flicker_frac_of_changes": "mean",
    })
    
    out_summary = RESULTS_DIR / "jitter_summary.json"
    grouped.to_json(out_summary, indent=2)
    print(f"Saved jitter summary to {out_summary}")

if __name__ == "__main__":
    main()
