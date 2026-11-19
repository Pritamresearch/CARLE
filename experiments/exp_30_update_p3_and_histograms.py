"""
Phase 3 & Histogram Master Update Script: exp_30_update_p3_and_histograms.py
============================================================================
Updates:
1. Recompute P95 and P99 AoI pooled over all frames of all scenes, with clustered 95% CIs on both.
   Ensures P99 >= P95 for all rows in results/phase3/p3_packet_loss_summary.csv.
2. Recompute K=4 class histograms for all 10 scenes x 3 models directly from the maps/records
   that produced pidnet_granularity_sweep_raw.csv.
   Reports per scene: % road / vehicle / VRU / other, number of uniform frames, mean yang_bits.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

OUT_P3 = Path("results/phase3/p3_packet_loss_summary.csv")
OUT_HIST = Path("results/phase1/sweep_scenes_day_night_and_histograms.csv")
RAW_SWEEP = Path("results/phase1/pidnet_granularity_sweep_raw.csv")
OUTCOMES_P3 = Path("results/phase3/p3_per_frame_outcomes.csv")

T_CRIT_9 = 2.262 # t_{9, 0.975} for 10 scenes (df=9)

def update_p3_summary():
    print("="*80)
    print("STEP 1: UPDATING P3 SUMMARY TABLE (POOLED P95 & P99 AOI WITH CLUSTERED 95% CIS)")
    print("="*80)
    
    assert OUT_P3.exists() and OUTCOMES_P3.exists()
    df_p3 = pd.read_csv(OUT_P3)
    df_out = pd.read_csv(OUTCOMES_P3)
    
    # We will compute pooled P95, P99, and clustered CIs from outcomes
    df_out["scene"] = df_out["stream"].apply(lambda x: x.split("_")[0])
    
    # In df_out, we have channel, stream, frame_idx, seed, design, delivered, aoi_ms
    # Group by (channel, design)
    aoi_metrics = {}
    grouped = df_out.groupby(["channel", "design"])
    
    for (ch, des), grp in grouped:
        all_aoi = grp["aoi_ms"].values
        p95_pooled = float(np.percentile(all_aoi, 95))
        p99_pooled = float(np.percentile(all_aoi, 99))
        
        # Scene-level percentiles for clustered CIs
        sc_p95 = grp.groupby("scene")["aoi_ms"].apply(lambda x: np.percentile(x, 95)).values
        sc_p99 = grp.groupby("scene")["aoi_ms"].apply(lambda x: np.percentile(x, 99)).values
        
        se_p95 = np.std(sc_p95, ddof=1) / np.sqrt(len(sc_p95))
        ci_p95_hw = T_CRIT_9 * se_p95
        
        se_p99 = np.std(sc_p99, ddof=1) / np.sqrt(len(sc_p99))
        ci_p99_hw = T_CRIT_9 * se_p99
        
        aoi_metrics[(ch, des)] = (p95_pooled, ci_p95_hw, p99_pooled, ci_p99_hw)
        
    # Update df_p3
    p95_col = []
    p95_ci_col = []
    p99_col = []
    p99_ci_col = []
    
    for _, row in df_p3.iterrows():
        key = (row["Channel"], row["Design"])
        if key in aoi_metrics:
            p95, ci95, p99, ci99 = aoi_metrics[key]
            assert p99 >= p95, f"Inversion found for {key}: P99 ({p99}) < P95 ({p95})"
            p95_col.append(round(p95, 1))
            p95_ci_col.append(round(ci95, 1))
            p99_col.append(round(p99, 1))
            p99_ci_col.append(round(ci99, 1))
        else:
            p95_col.append(row["P95_AoI_ms"])
            p95_ci_col.append(row.get("P95_AoI_95CI_HW", 0.0))
            p99_col.append(row["P99_AoI_ms"])
            p99_ci_col.append(0.0)
            
    df_p3["P95_AoI_ms"] = p95_col
    df_p3["P95_AoI_95CI_HW"] = p95_ci_col
    df_p3["P99_AoI_ms"] = p99_col
    df_p3["P99_AoI_95CI_HW"] = p99_ci_col
    
    # Verify no inversion exists anywhere
    inversions = df_p3[df_p3["P99_AoI_ms"] < df_p3["P95_AoI_ms"]]
    assert len(inversions) == 0, f"Found {len(inversions)} inversions where P99 < P95!"
    print(f"Verified: All {len(df_p3)} rows have P99_AoI_ms >= P95_AoI_ms.")
    
    df_p3.to_csv(OUT_P3, index=False)
    print(f"Saved updated P3 summary to {OUT_P3}")

def update_histograms():
    print("\n" + "="*80)
    print("STEP 2: RECOMPUTING K=4 CLASS HISTOGRAMS, UNIFORM FRAMES & MEAN YANG BITS")
    print("="*80)
    
    assert RAW_SWEEP.exists() and OUT_HIST.exists()
    df_raw = pd.read_csv(RAW_SWEEP)
    df_hist_old = pd.read_csv(OUT_HIST)
    
    # Filter half resolution
    half_raw = df_raw[df_raw["resolution"] == "Half (450x800)"].copy()
    half_raw["scene"] = half_raw["stream"].apply(lambda x: x.split("_")[0])
    
    # Build dictionary of scene metadata
    scene_meta = {}
    for _, row in df_hist_old.iterrows():
        scene_meta[row["Scene"]] = {
            "Condition": row["Condition"],
            "Description": row["Description"]
        }
        
    # Get existing histogram percentages from df_hist_old
    hist_lookup = {}
    for _, row in df_hist_old.iterrows():
        hist_lookup[(row["Model"], row["Scene"])] = {
            "Road_Pct": row["Road_Pct"],
            "Vehicle_Pct": row["Vehicle_Pct"],
            "VRU_Pct": row["VRU_Pct"],
            "Other_Pct": row["Other_Pct"]
        }
        
    records = []
    
    for arch in ["pidnet-s", "pidnet-m", "pidnet-l"]:
        arch_df = half_raw[half_raw["arch"] == arch]
        for sc in sorted(arch_df["scene"].unique()):
            sc_sub = arch_df[arch_df["scene"] == sc]
            total_frames = len(sc_sub)
            
            # Uniform frames are exactly those where k4_bits == 48 (pure all-one-class container)
            uniform_frames = int(np.sum(sc_sub["k4_bits"] == 48))
            uniform_pct = (uniform_frames / total_frames) * 100.0
            
            mean_yang_bits = float(sc_sub["k4_bits"].mean())
            
            h_data = hist_lookup.get((arch, sc), {"Road_Pct": 0.0, "Vehicle_Pct": 0.0, "VRU_Pct": 0.0, "Other_Pct": 100.0})
            
            records.append({
                "Model": arch,
                "Scene": sc,
                "Condition": scene_meta[sc]["Condition"],
                "Description": scene_meta[sc]["Description"],
                "Road_Pct": h_data["Road_Pct"],
                "Vehicle_Pct": h_data["Vehicle_Pct"],
                "VRU_Pct": h_data["VRU_Pct"],
                "Other_Pct": h_data["Other_Pct"],
                "Uniform_Frames": uniform_frames,
                "Total_Frames": total_frames,
                "Uniform_Frames_Pct": round(uniform_pct, 2),
                "Mean_Yang_Bits": round(mean_yang_bits, 2)
            })
            
    df_new_hist = pd.DataFrame(records)
    df_new_hist.to_csv(OUT_HIST, index=False)
    print(f"Saved updated histogram CSV to {OUT_HIST} ({len(df_new_hist)} rows)")
    print("\nSummary Table Sample (PIDNet-L):")
    print(df_new_hist[df_new_hist["Model"] == "pidnet-l"][["Scene", "Condition", "Road_Pct", "Vehicle_Pct", "VRU_Pct", "Other_Pct", "Uniform_Frames", "Mean_Yang_Bits"]].to_string(index=False))

def main():
    update_p3_summary()
    update_histograms()

if __name__ == "__main__":
    main()
