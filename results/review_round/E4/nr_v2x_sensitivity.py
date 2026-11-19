"""
E4: 3GPP NR-V2X Capacity Sensitivity Analysis (Full Grid & OAT Variations)
==========================================================================
Recomputes N_max for frame-independent chain coder (Yang S=1) and SemCM (a) at K=4
across:
- rho in {0.3, 0.45, 0.6, 0.75, 0.9} (Baseline: 0.6)
- MCS in {10, 13, 16, 19, 20} (TBS: 157, 233, 309, 372, 405 B) (Baseline: 16 -> 309 B)
- header in {12, 24, 40} bytes (Baseline: 12 B)
- frame rate f in {5, 10} Hz (Baseline: 10 Hz)

Includes 10,000 scene-cluster bootstrap 95% intervals over the 10 nuScenes scenes.
Verifies that the baseline reproduces N_max = 367 for the chain coder.
Outputs:
- results/review_round/E4/capacity_grid.csv
- results/review_round/E4/capacity_oat_summary.csv
- results/review_round/E4/report.md
"""

import sys
import os
import argparse
import itertools
from pathlib import Path
import numpy as np
import pandas as pd

P1B_K4_CSV = Path("results/phase1/p1b_all_frames_raw_k4.csv")
OUT_DIR = Path("results/review_round/E4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 3GPP TS 38.214 Table 5.1.3.1-1 and Table 5.1.3.2-1 derived TBS values (10 PRB, 96 REs/PRB)
MCS_TBS_MAP = {
    10: (1256 // 8, "MCS 10: QPSK (R=658/1024)"),   # 157 B
    13: (1864 // 8, "MCS 13: 16QAM (R=490/1024)"),  # 233 B
    16: (2472 // 8, "MCS 16: 16QAM (R=658/1024)"),  # 309 B (Baseline)
    19: (2976 // 8, "MCS 19: 64QAM (R=517/1024)"),  # 372 B
    20: (3240 // 8, "MCS 20: 64QAM (R=567/1024)")   # 405 B
}

# Baseline Parameters
BASE_RHO = 0.60
BASE_MCS = 16
BASE_HEADER = 12
BASE_FPS = 10.0

N_SUB_PER_SLOT = 5
R_SLOT = 2000 # slots/sec
TOTAL_SUBCH_PER_SEC = N_SUB_PER_SLOT * R_SLOT # 10,000 subchannels/sec

def compute_capacity_stream(
    df: pd.DataFrame,
    col_name: str,
    rho: float,
    mcs: int,
    header_bytes: int,
    fps: float,
    rng: np.random.Generator = None
) -> dict:
    tbs_bytes = MCS_TBS_MAP[mcs][0]
    usable_subch_per_sec = rho * TOTAL_SUBCH_PER_SEC
    
    # Payload bytes
    bits = df[col_name].values
    B_i = np.ceil(bits / 8.0) + header_bytes
    subch_per_frame = np.ceil(B_i / tbs_bytes)
    
    e_subch = float(np.mean(subch_per_frame))
    denom = fps * e_subch
    n_max = int(np.floor(usable_subch_per_sec / denom))
    
    # Scene-cluster bootstrap
    scenes = sorted(df["scene"].unique())
    n_sc = len(scenes)
    scene_means_arr = df.assign(subch=subch_per_frame).groupby("scene")["subch"].mean().values
    
    if rng is not None:
        resamp_idx = rng.integers(0, n_sc, size=(10000, n_sc))
        boot_e_subch = scene_means_arr[resamp_idx].mean(axis=1)
        boot_n_max = np.floor(usable_subch_per_sec / (fps * boot_e_subch)).astype(int)
        ci_low, ci_high = np.percentile(boot_n_max, [2.5, 97.5])
    else:
        ci_low, ci_high = n_max, n_max
        
    return {
        "rho": rho,
        "MCS": mcs,
        "TBS_Bytes": tbs_bytes,
        "MCS_Label": MCS_TBS_MAP[mcs][1],
        "Header_Bytes": header_bytes,
        "FPS": fps,
        "E_Subch_Per_Frame": e_subch,
        "N_max": n_max,
        "N_max_95CI_Low": int(ci_low),
        "N_max_95CI_High": int(ci_high)
    }

def run_e4(smoke: bool = False):
    print("="*80)
    print(f"RUNNING E4: NR-V2X CAPACITY SENSITIVITY BENCHMARK (smoke={smoke})")
    print("="*80)
    
    assert P1B_K4_CSV.exists()
    df_raw = pd.read_csv(P1B_K4_CSV)
    df_sw = df_raw[df_raw["dataset"] == "nuScenes_Sweeps_12Hz"].copy()
    df_sw["scene"] = df_sw["stream"].apply(lambda s: s.split("_")[0])
    
    if smoke:
        # Use only 2 scenes for quick smoke run
        sc_subset = sorted(df_sw["scene"].unique())[:2]
        df_sw = df_sw[df_sw["scene"].isin(sc_subset)].copy()
        
    print(f"Loaded {len(df_sw)} sweep frames across {df_sw['scene'].nunique()} scenes.")
    
    rng = np.random.default_rng(0)
    
    # Parameter values
    rhos = [0.3, 0.45, 0.6, 0.75, 0.9]
    mcss = [10, 13, 16, 19, 20]
    headers = [12, 24, 40]
    fps_vals = [5.0, 10.0]
    
    codecs = [
        ("Yang_ECC", "yang_bits"),
        ("SemCM_a", "semcm_a_bits")
    ]
    
    # -------------------------------------------------------------------------
    # 1. Full Grid Evaluation
    # -------------------------------------------------------------------------
    print("\n--- Running Full Parameter Grid (5 x 5 x 3 x 2 = 150 points per codec) ---")
    grid_records = []
    
    for c_name, c_col in codecs:
        for rho, mcs, h, fps in itertools.product(rhos, mcss, headers, fps_vals):
            res = compute_capacity_stream(df_sw, c_col, rho, mcs, h, fps, rng=rng)
            res["Codec"] = c_name
            grid_records.append(res)
            
    df_grid = pd.DataFrame(grid_records)
    grid_csv = OUT_DIR / "capacity_grid.csv"
    df_grid.to_csv(grid_csv, index=False)
    print(f"Full grid saved to {grid_csv} ({len(df_grid)} configurations).")
    
    # Check baseline cell
    base_yang = df_grid[(df_grid["Codec"] == "Yang_ECC") &
                        (df_grid["rho"] == BASE_RHO) &
                        (df_grid["MCS"] == BASE_MCS) &
                        (df_grid["Header_Bytes"] == BASE_HEADER) &
                        (df_grid["FPS"] == BASE_FPS)].iloc[0]
    print(f"\n>>> Baseline Yang ECC (rho=0.6, MCS=16, Header=12B, FPS=10Hz):")
    print(f"    N_max = {base_yang['N_max']} streams [95% CI: {base_yang['N_max_95CI_Low']}, {base_yang['N_max_95CI_High']}]")
    if not smoke:
        assert base_yang["N_max"] == 367, f"Expected 367 but got {base_yang['N_max']}"
        print("    [PASS] Exact reproduction of 367 streams verified!")
        
    base_semcm = df_grid[(df_grid["Codec"] == "SemCM_a") &
                         (df_grid["rho"] == BASE_RHO) &
                         (df_grid["MCS"] == BASE_MCS) &
                         (df_grid["Header_Bytes"] == BASE_HEADER) &
                         (df_grid["FPS"] == BASE_FPS)].iloc[0]
    print(f"\n>>> Baseline SemCM (a) (rho=0.6, MCS=16, Header=12B, FPS=10Hz):")
    print(f"    N_max = {base_semcm['N_max']} streams [95% CI: {base_semcm['N_max_95CI_Low']}, {base_semcm['N_max_95CI_High']}]")

    # -------------------------------------------------------------------------
    # 2. One-At-A-Time (OAT) Sensitivity Summary
    # -------------------------------------------------------------------------
    print("\n--- One-At-A-Time Sensitivity Summary Around Baseline ---")
    oat_records = []
    
    factors = [
        ("rho (CBR Limit)", "rho", rhos, BASE_RHO),
        ("MCS (Code Rate)", "MCS", mcss, BASE_MCS),
        ("Header (Bytes)", "Header_Bytes", headers, BASE_HEADER),
        ("Frame Rate (Hz)", "FPS", fps_vals, BASE_FPS)
    ]
    
    for c_name in ["Yang_ECC", "SemCM_a"]:
        sub_c = df_grid[df_grid["Codec"] == c_name]
        for factor_name, param_key, param_vals, base_val in factors:
            for val in param_vals:
                # Filter to baseline for all other params
                cond = (sub_c["rho"] == (val if param_key == "rho" else BASE_RHO)) & \
                       (sub_c["MCS"] == (val if param_key == "MCS" else BASE_MCS)) & \
                       (sub_c["Header_Bytes"] == (val if param_key == "Header_Bytes" else BASE_HEADER)) & \
                       (sub_c["FPS"] == (val if param_key == "FPS" else BASE_FPS))
                row = sub_c[cond].iloc[0]
                oat_records.append({
                    "Codec": c_name,
                    "Factor": factor_name,
                    "Parameter_Value": val,
                    "Is_Baseline": (val == base_val),
                    "N_max": row["N_max"],
                    "N_max_95CI_Low": row["N_max_95CI_Low"],
                    "N_max_95CI_High": row["N_max_95CI_High"],
                    "E_Subch_Per_Frame": row["E_Subch_Per_Frame"],
                    "Delta_vs_Baseline_Pct": ((row["N_max"] - (base_yang['N_max'] if c_name == 'Yang_ECC' else base_semcm['N_max'])) /
                                             (base_yang['N_max'] if c_name == 'Yang_ECC' else base_semcm['N_max'])) * 100.0
                })
                
    df_oat = pd.DataFrame(oat_records)
    oat_csv = OUT_DIR / "capacity_oat_summary.csv"
    df_oat.to_csv(oat_csv, index=False)
    print(f"OAT sensitivity summary saved to {oat_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_e4(smoke=args.smoke)
