"""
E7: Measure Real Inter-over-Intra Size Ratio on Half-Resolution nuScenes Sweeps
==============================================================================
Evaluates SemCM intra vs Compact Temporal (from exp_13) on half-resolution (450x800)
nuScenes sweep frames across the 10 scenes, for K=4 and K=8.
"""

import sys
import os
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Add root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from carle.rigorous_codecs import cm_prior_adaptive_encode
from experiments.exp_12_pidnet_sequence_benchmark import cm_k8_prior_adaptive_encode
from experiments.exp_13_debug_temporal import (
    cm_compact_temp_adaptive_encode
)

CACHE_DIR_HALF = Path("results/phase1b/cache_pidnet_half/sweeps")
MODELS_FILE = Path("results/phase1/pretrained_models.npz")

def main():
    parser = argparse.ArgumentParser(description="Measure real temporal ratio")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test on 1 scene")
    args = parser.parse_args()

    assert CACHE_DIR_HALF.exists()
    assert MODELS_FILE.exists()

    m_npz = np.load(MODELS_FILE)
    models = {
        4: {"intra": m_npz["k4_f1_intra"], "temp": m_npz["k4_f1_temp"]},
        8: {"intra": m_npz["k8_f1_intra"], "temp": m_npz["k8_f1_temp"]}
    }

    scene_files = sorted(list(CACHE_DIR_HALF.glob("sweeps_scene-*_k4.npz")))
    if args.smoke:
        scene_files = scene_files[:1]

    print(f"Measuring real temporal ratio across {len(scene_files)} scenes (smoke={args.smoke})...")

    records = []
    t0 = time.time()

    for sf in scene_files:
        scene_name = sf.stem.replace("sweeps_", "").replace("_k4", "")
        for k_val in [4, 8]:
            npz_path = CACHE_DIR_HALF / f"sweeps_{scene_name}_k{k_val}.npz"
            masks = np.load(npz_path)["masks"][:100] # First 100 sweep frames
            
            c_intra_base = models[k_val]["intra"]
            c_temp_base = models[k_val]["temp"]
            
            intra_bits_list = []
            temp_bits_list = []
            
            for idx in range(len(masks)):
                curr = masks[idx]
                prev = masks[idx-1] if idx > 0 else curr
                is_identical = np.array_equal(curr, prev)
                
                # Intra
                c_i = c_intra_base.copy()
                t_i = c_i.sum(axis=1)
                if k_val == 4:
                    p_i = cm_prior_adaptive_encode(curr, curr, False, c_i, t_i, max_total=32768)
                else:
                    p_i = cm_k8_prior_adaptive_encode(curr, curr, False, c_i, t_i, max_total=32768)
                b_i = len(p_i) * 8
                intra_bits_list.append(b_i)
                
                # Temp
                c_t = c_temp_base.copy()
                t_t = c_t.sum(axis=1)
                p_t = cm_compact_temp_adaptive_encode(curr, prev, k_val, c_t, t_t, max_total=32768)
                b_t = 8 if is_identical else (len(p_t) * 8 + 1)
                temp_bits_list.append(b_t)
                
            mean_intra = float(np.mean(intra_bits_list))
            mean_temp = float(np.mean(temp_bits_list))
            ratio = mean_temp / mean_intra
            gain_pct = (1.0 - ratio) * 100.0
            
            records.append({
                "scene": scene_name,
                "K": k_val,
                "N_frames": len(masks),
                "mean_intra_bits": mean_intra,
                "mean_temp_bits": mean_temp,
                "inter_over_intra_ratio": ratio,
                "gain_pct": gain_pct
            })
            print(f"Scene {scene_name} | K={k_val} | Intra: {mean_intra:.1f} b | Temp: {mean_temp:.1f} b | "
                  f"Ratio: {ratio:.4f} | Gain: {gain_pct:.2f}%")

    df = pd.DataFrame(records)
    out_csv = Path("results/review_round/E7/real_temporal_ratio.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nDone in {time.time() - t0:.2f} s. Saved to {out_csv}")
    
    # Summary per K
    for k_val in [4, 8]:
        sub = df[df["K"] == k_val]
        mean_ratio = sub["mean_temp_bits"].sum() / sub["mean_intra_bits"].sum()
        print(f"\nOverall K={k_val}: Aggregate Intra={sub['mean_intra_bits'].mean():.1f} b, "
              f"Temp={sub['mean_temp_bits'].mean():.1f} b, Ratio={mean_ratio:.4f}, Gain={(1-mean_ratio)*100:.2f}%")

if __name__ == "__main__":
    main()
