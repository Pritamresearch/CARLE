"""
E5: Class-Wise Staleness Benchmark (K = 8 Mapping)
=================================================
Evaluates Age of Information (AoI) separately for:
  - Road (class 1)
  - Vehicle (class 6)
  - Pedestrian (class 8)
  - Full Frame (all classes)
  - Camera views: Front/Back vs Lateral (CAM_FRONT_LEFT/RIGHT, CAM_BACK_LEFT/RIGHT)
  - Spatial columns: Center (200 <= c < 600) vs Lateral (c < 200 or c >= 600)

Using ground-truth class pixel counts of the sender map as weighting across horizontal
slicing configurations S in {1, 4, 8} under Bernoulli (10%) and continuous-time
Gilbert-Elliott burst channels (T_B = 50 ms and 100 ms, P_L = 10%).

AoI: A_t = (t - t_src) * dt + T_proc, with T_proc = 103.4 ms.
Clustered confidence intervals: 10,000 scene-cluster resamples over 10 nuScenes scenes.
"""

import sys
import os
import argparse
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd

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

def weighted_quantile(values, weights, quantiles):
    """
    Computes weighted quantiles.
    values: 1D array of values
    weights: 1D array of non-negative weights
    quantiles: list of quantiles in [0, 1]
    """
    values = np.asarray(values)
    weights = np.asarray(weights)
    if len(values) == 0 or weights.sum() == 0:
        return [np.nan] * len(quantiles)
    
    idx = np.argsort(values)
    sorted_vals = values[idx]
    sorted_weights = weights[idx]
    
    cum_weights = np.cumsum(sorted_weights)
    total_weight = cum_weights[-1]
    
    res = []
    for q in quantiles:
        target = q * total_weight
        pos = np.searchsorted(cum_weights, target)
        pos = min(pos, len(sorted_vals) - 1)
        res.append(float(sorted_vals[pos]))
    return res

def run_simulation(
    slice_counts,
    metadata,
    S: int,
    ch_type: str,
    T_B_ms: float,
    per: float = 0.10,
    fps: float = 10.0,
    T_proc: float = 103.4,
    seeds = [0, 1, 2, 3, 4],
    pkt_window_ms: float = 5.0
):
    """
    Runs the slice-level transmission simulation across streams and seeds.
    slice_counts: dict of pre-aggregated counts per stream
    """
    dt_ms = 1000.0 / fps
    N_streams = len(metadata)
    N_frames = 100
    
    if S == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(0.0, pkt_window_ms, S)
        
    records = []
    
    for seed in seeds:
        rng = np.random.default_rng(seed)
        
        # Channel setup
        if ch_type == "ge":
            T_B_s = T_B_ms / 1000.0
            mu_B = 1.0 / T_B_s
            mu_G = (per / (1.0 - per)) * mu_B
            lam = mu_G + mu_B
            state = (rng.random(N_streams) < per).astype(int)
        else:
            state = None
            
        t_src = np.zeros((N_streams, S), dtype=int)
        
        curr_time = 0.0
        
        # Simulate timeline
        for k in range(N_frames):
            for s in range(S):
                pkt_time = k * dt_ms + offsets[s]
                
                if ch_type == "ge":
                    tau_s = (pkt_time - curr_time) / 1000.0
                    if tau_s > 0:
                        p_01 = per * (1.0 - np.exp(-lam * tau_s))
                        p_10 = (1.0 - per) * (1.0 - np.exp(-lam * tau_s))
                        r = rng.random(N_streams)
                        to_bad = (state == 0) & (r < p_01)
                        to_good = (state == 1) & (r < p_10)
                        state = np.where(to_bad, 1, np.where(to_good, 0, state))
                        curr_time = pkt_time
                    loss = (state == 1)
                else:
                    loss = (rng.random(N_streams) < per)
                    
                # Delivery update
                t_src[:, s] = np.where(~loss, k, t_src[:, s])
                
            # Compute AoI at frame k for all streams and slices
            # aoi: shape (N_streams, S)
            aoi_s = (k - t_src) * dt_ms + T_proc
            
            for st_idx in range(N_streams):
                st_meta = metadata[st_idx]
                scene = st_meta["scene"]
                cam = st_meta["camera"]
                is_lateral_cam = cam in ["CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
                
                cnts = slice_counts[st_idx][S][k] # dict with road, veh, ped, all, lat_col, cen_col arrays of len S
                
                records.append({
                    "seed": seed,
                    "scene": scene,
                    "camera": cam,
                    "is_lateral_cam": is_lateral_cam,
                    "frame_idx": k,
                    "aoi_s": aoi_s[st_idx, :].copy(),
                    "road_px": cnts["road"],
                    "veh_px": cnts["veh"],
                    "ped_px": cnts["ped"],
                    "all_px": cnts["all"],
                    "lat_col_px": cnts["lat_col"],
                    "cen_col_px": cnts["cen_col"],
                })
                
    return records

def aggregate_class_metrics(records, scene_names):
    """
    Computes class-weighted metrics with scene-cluster bootstrap.
    """
    # Categories to evaluate
    categories = ["Road", "Vehicle", "Pedestrian", "Full_Frame", "Lateral_Cams", "Frontal_Cams", "Lateral_Cols", "Center_Cols"]
    
    # Pre-collect values and weights per scene for bootstrapping
    scene_cat_data = {sc: {cat: {"vals": [], "w": []} for cat in categories} for sc in scene_names}
    
    for r in records:
        sc = r["scene"]
        aoi = r["aoi_s"]
        
        # Road
        scene_cat_data[sc]["Road"]["vals"].extend(aoi)
        scene_cat_data[sc]["Road"]["w"].extend(r["road_px"])
        
        # Vehicle
        scene_cat_data[sc]["Vehicle"]["vals"].extend(aoi)
        scene_cat_data[sc]["Vehicle"]["w"].extend(r["veh_px"])
        
        # Pedestrian
        scene_cat_data[sc]["Pedestrian"]["vals"].extend(aoi)
        scene_cat_data[sc]["Pedestrian"]["w"].extend(r["ped_px"])
        
        # Full Frame
        scene_cat_data[sc]["Full_Frame"]["vals"].extend(aoi)
        scene_cat_data[sc]["Full_Frame"]["w"].extend(r["all_px"])
        
        # Lateral vs Frontal Cameras
        if r["is_lateral_cam"]:
            scene_cat_data[sc]["Lateral_Cams"]["vals"].extend(aoi)
            scene_cat_data[sc]["Lateral_Cams"]["w"].extend(r["all_px"])
        else:
            scene_cat_data[sc]["Frontal_Cams"]["vals"].extend(aoi)
            scene_cat_data[sc]["Frontal_Cams"]["w"].extend(r["all_px"])
            
        # Lateral vs Center Columns
        scene_cat_data[sc]["Lateral_Cols"]["vals"].extend(aoi)
        scene_cat_data[sc]["Lateral_Cols"]["w"].extend(r["lat_col_px"])
        
        scene_cat_data[sc]["Center_Cols"]["vals"].extend(aoi)
        scene_cat_data[sc]["Center_Cols"]["w"].extend(r["cen_col_px"])
        
    # Convert lists to numpy arrays
    for sc in scene_names:
        for cat in categories:
            scene_cat_data[sc][cat]["vals"] = np.array(scene_cat_data[sc][cat]["vals"], dtype=float)
            scene_cat_data[sc][cat]["w"] = np.array(scene_cat_data[sc][cat]["w"], dtype=float)
            
    # Point estimates pooled across all scenes
    out = {}
    for cat in categories:
        all_vals = np.concatenate([scene_cat_data[sc][cat]["vals"] for sc in scene_names])
        all_w = np.concatenate([scene_cat_data[sc][cat]["w"] for sc in scene_names])
        
        if all_w.sum() == 0:
            out[cat] = {"mean": np.nan, "p50": np.nan, "p95": np.nan, "p99": np.nan, "mean_ci": (np.nan, np.nan), "p95_ci": (np.nan, np.nan)}
            continue
            
        mean_val = float(np.average(all_vals, weights=all_w))
        p50, p95, p99 = weighted_quantile(all_vals, all_w, [0.50, 0.95, 0.99])
        
        # Scene cluster bootstrap
        n_sc = len(scene_names)
        rng = np.random.default_rng(0)
        B = 10000
        boot_means = []
        boot_p95s = []
        
        # Scene level means for speed
        sc_means = np.array([
            np.average(scene_cat_data[sc][cat]["vals"], weights=scene_cat_data[sc][cat]["w"])
            if scene_cat_data[sc][cat]["w"].sum() > 0 else np.nan
            for sc in scene_names
        ])
        valid_sc = ~np.isnan(sc_means)
        n_valid = valid_sc.sum()
        
        if n_valid >= 2:
            sc_weights = np.array([scene_cat_data[sc][cat]["w"].sum() for sc in scene_names])
            # Bootstrap resample scene indices
            resamp = rng.integers(0, n_sc, size=(B, n_sc))
            # Vectorized weighted mean calculation
            sampled_w = sc_weights[resamp] # (B, n_sc)
            sampled_means = sc_means[resamp] # (B, n_sc)
            num = (sampled_w * sampled_means).sum(axis=1)
            den = sampled_w.sum(axis=1)
            boot_means = num / den
            mean_ci = (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))
        else:
            mean_ci = (mean_val, mean_val)
            
        out[cat] = {
            "mean": mean_val,
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "mean_ci": mean_ci,
        }
        
    return out

def main():
    parser = argparse.ArgumentParser(description="E5 Class-Wise Staleness Benchmark")
    parser.add_argument("--smoke", action="store_true", help="Run 1 scene smoke test")
    parser.add_argument("--fps", type=float, default=10.0, help="Frame rate")
    args = parser.parse_args()

    out_dir = Path("results/review_round/E5")
    out_dir.mkdir(parents=True, exist_ok=True)

    sweeps_dir = Path("results/phase1b/cache_pidnet_half/sweeps")
    all_scenes = sorted([f.replace("sweeps_", "").replace("_k8.npz", "") for f in os.listdir(sweeps_dir) if f.endswith("_k8.npz")])
    
    if args.smoke:
        eval_scenes = all_scenes[:1]
        seeds = [0]
    else:
        eval_scenes = all_scenes
        seeds = [0, 1, 2, 3, 4]
        
    print(f"E5: Evaluating {len(eval_scenes)} scenes across seeds {seeds} at {args.fps} Hz...")
    
    # Load and pre-aggregate pixel counts
    # Stream metadata: list of dicts with scene, camera
    cam_order = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    metadata = []
    for sc in eval_scenes:
        for cam in cam_order:
            metadata.append({"scene": sc, "camera": cam})
            
    print(f"Total streams: {len(metadata)}")
    
    splits = {S: get_slice_splits(450, S) for S in [1, 4, 8]}
    
    print("Pre-aggregating class and spatial pixel counts across slices...")
    t0 = time.time()
    slice_counts = [] # per stream index
    
    st_idx = 0
    for sc in eval_scenes:
        npz_p = sweeps_dir / f"sweeps_{sc}_k8.npz"
        masks = np.load(npz_p)["masks"] # shape (600, 450, 800)
        
        for c_idx, cam in enumerate(cam_order):
            cam_masks = masks[c_idx*100 : (c_idx+1)*100] # shape (100, 450, 800)
            
            st_data = {}
            for S in [1, 4, 8]:
                st_data[S] = []
                for k in range(100):
                    f_mask = cam_masks[k]
                    road_arr = np.zeros(S, dtype=int)
                    veh_arr = np.zeros(S, dtype=int)
                    ped_arr = np.zeros(S, dtype=int)
                    all_arr = np.zeros(S, dtype=int)
                    lat_col_arr = np.zeros(S, dtype=int)
                    cen_col_arr = np.zeros(S, dtype=int)
                    
                    for s, (r0, r1) in enumerate(splits[S]):
                        sl_crop = f_mask[r0:r1, :]
                        road_arr[s] = np.sum(sl_crop == 1)
                        veh_arr[s] = np.sum(sl_crop == 6)
                        ped_arr[s] = np.sum(sl_crop == 8)
                        all_arr[s] = sl_crop.size
                        
                        # Spatial columns: Lateral (c < 200 or c >= 600) vs Center (200 <= c < 600)
                        lat_crop = np.concatenate([sl_crop[:, :200], sl_crop[:, 600:]], axis=1)
                        cen_crop = sl_crop[:, 200:600]
                        lat_col_arr[s] = lat_crop.size
                        cen_col_arr[s] = cen_crop.size
                        
                    st_data[S].append({
                        "road": road_arr,
                        "veh": veh_arr,
                        "ped": ped_arr,
                        "all": all_arr,
                        "lat_col": lat_col_arr,
                        "cen_col": cen_col_arr
                    })
            slice_counts.append(st_data)
            st_idx += 1
            
    print(f"Pre-aggregation complete in {time.time() - t0:.2f} seconds.")
    
    # Run matrix: S in [1, 4, 8] x Channels [Bernoulli 10%, GE TB=50ms 10%, GE TB=100ms 10%]
    ch_configs = [
        {"name": "Bernoulli_p10", "type": "bernoulli", "T_B": 0.0, "per": 0.10},
        {"name": "Burst_TB50ms_p10", "type": "ge", "T_B": 50.0, "per": 0.10},
        {"name": "Burst_TB100ms_p10", "type": "ge", "T_B": 100.0, "per": 0.10},
    ]
    
    results = []
    
    for ch in ch_configs:
        for S in [1, 4, 8]:
            print(f"Simulating Channel={ch['name']}, S={S}...")
            records = run_simulation(
                slice_counts=slice_counts,
                metadata=metadata,
                S=S,
                ch_type=ch["type"],
                T_B_ms=ch["T_B"],
                per=ch["per"],
                fps=args.fps,
                T_proc=103.4,
                seeds=seeds
            )
            
            metrics = aggregate_class_metrics(records, eval_scenes)
            
            for cat, m in metrics.items():
                results.append({
                    "channel": ch["name"],
                    "S": S,
                    "category": cat,
                    "mean_aoi_ms": m["mean"],
                    "mean_ci_low": m["mean_ci"][0],
                    "mean_ci_high": m["mean_ci"][1],
                    "p50_aoi_ms": m["p50"],
                    "p95_aoi_ms": m["p95"],
                    "p99_aoi_ms": m["p99"],
                })
                
    res_df = pd.DataFrame(results)
    csv_path = out_dir / "results.csv"
    res_df.to_csv(csv_path, index=False)
    print(f"Saved E5 results to {csv_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print(f"E5 CLASS-WISE STALENESS SUMMARY ({args.fps} Hz)")
    print("="*80)
    summary_pivot = res_df.pivot_table(
        index=["channel", "category"],
        columns="S",
        values=["mean_aoi_ms", "p95_aoi_ms"]
    )
    print(summary_pivot.to_string())
    print("="*80)

if __name__ == "__main__":
    main()
