"""
Phase 5: 3GPP NR-V2X Sidelink Analytical Capacity Benchmark (v2)
================================================================
Methodological Derivation:
- Numerology: mu = 1 (30 kHz SCS, 0.5 ms slot, 2000 slots/s).
- Channel Bandwidth: 20 MHz (51 PRBs per TS 38.101-1 Table 5.3.2-1).
- Subchannels: 5 subchannels of 10 PRBs each (50 PRBs data, 1 guard PRB). Total: 10,000 subchannels/s.
- MCS & TBS Derivation:
  MCS 16 (TS 38.214 Table 5.1.3.1-1, 16QAM, q=4, Target Code Rate R = 658 / 1024).
  Resource Elements: 10 PRBs x 96 REs = 960 REs.
  N_info = 960 x 4 x (658 / 1024) = 2,467.5 bits.
  Closest valid 3GPP Transport Block Size (TBS) = 2,472 bits = 309 Bytes.
- Congestion Control:
  sl-CR-Limit configured by RRC (TS 38.331), nominal CR_max = 0.60 -> 6,000 usable subchannels/s.
- Payload Evaluation:
  ALL codecs evaluated on nuScenes_Sweeps_12Hz rows ONLY from p1b_all_frames_raw_k4/k8.csv.
  Per-frame payload: B_i = ceil(bits / 8) + 12 Bytes.
  Resource demand per frame: E[ceil(B_i / 309)].
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT_DIR = Path("results/phase5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

P1B_K4_CSV = Path("results/phase1/p1b_all_frames_raw_k4.csv")
P1B_K8_CSV = Path("results/phase1/p1b_all_frames_raw_k8.csv")
P2_SLICES_CSV = Path("results/phase2/p2_per_frame_per_slice_bytes.csv")

TBS_BYTES = 309
TOTAL_SUBCH_PER_SEC = 10000

def run_p5_capacity():
    print("="*80)
    print("STARTING P5 CAPACITY EVALUATION (3GPP TS 38.214 / TS 38.331)")
    print("="*80)
    
    assert P1B_K4_CSV.exists() and P1B_K8_CSV.exists() and P2_SLICES_CSV.exists()
    
    df_p2 = pd.read_csv(P2_SLICES_CSV)
    
    summary_records = []
    sensitivity_records = []
    
    for k_val in [4, 8]:
        p1b_path = P1B_K4_CSV if k_val == 4 else P1B_K8_CSV
        df_p1b = pd.read_csv(p1b_path)
        sw = df_p1b[df_p1b["dataset"] == "nuScenes_Sweeps_12Hz"].copy()
        
        # Whole-frame codecs from p1b nuScenes sweeps
        codecs = [
            ("Yang", "Whole (S=1)", "yang_bits"),
            ("SemCM (a)", "Whole (S=1)", "semcm_a_bits"),
            ("JPEG-XL e9", "Whole (S=1)", "jxl9_bits"),
            ("WebP Lossless", "Whole (S=1)", "webp_bits")
        ]
        
        for name, mode, col in codecs:
            bits = sw[col].values
            B_i = np.ceil(bits / 8.0) + 12.0 # 12 B packet header
            subch_arr = np.ceil(B_i / TBS_BYTES)
            
            mean_bytes = float(np.mean(B_i))
            p50_bytes = float(np.percentile(B_i, 50))
            p95_bytes = float(np.percentile(B_i, 95))
            p99_bytes = float(np.percentile(B_i, 99))
            
            mean_subch = float(np.mean(subch_arr))
            p95_subch = float(np.percentile(subch_arr, 95))
            p99_subch = float(np.percentile(subch_arr, 99))
            
            # Sizing at nominal CR_max = 0.60 (6,000 usable subchannels/s)
            cams_10hz = int(np.floor(6000.0 / (mean_subch * 10.0)))
            vehs_10hz = cams_10hz // 6
            cams_12hz = int(np.floor(6000.0 / (mean_subch * 12.0)))
            vehs_12hz = cams_12hz // 6
            
            summary_records.append({
                "K": k_val,
                "Codec": name,
                "Mode": mode,
                "Source_Column": col,
                "Mean_Payload_Bytes": round(mean_bytes, 1),
                "P50_Payload_Bytes": round(p50_bytes, 1),
                "P95_Payload_Bytes": round(p95_bytes, 1),
                "P99_Payload_Bytes": round(p99_bytes, 1),
                "Mean_Subch_Per_Frame": round(mean_subch, 3),
                "P95_Subch_Per_Frame": round(p95_subch, 1),
                "P99_Subch_Per_Frame": round(p99_subch, 1),
                "Usable_Subch_Per_Sec": 6000,
                "Cameras_10Hz": cams_10hz,
                "Vehicles_10Hz": vehs_10hz,
                "Cameras_12Hz": cams_12hz,
                "Vehicles_12Hz": vehs_12hz
            })
            
            # Sensitivity across CR_max in {0.40, 0.60, 0.80}
            for cr in [0.40, 0.60, 0.80]:
                usable = int(TOTAL_SUBCH_PER_SEC * cr)
                c_10 = int(np.floor(usable / (mean_subch * 10.0)))
                v_10 = c_10 // 6
                c_12 = int(np.floor(usable / (mean_subch * 12.0)))
                v_12 = c_12 // 6
                sensitivity_records.append({
                    "K": k_val,
                    "Codec": name,
                    "Mode": mode,
                    "CR_max": cr,
                    "Usable_Subch_Per_Sec": usable,
                    "Cameras_10Hz": c_10,
                    "Vehicles_10Hz": v_10,
                    "Cameras_12Hz": c_12,
                    "Vehicles_12Hz": v_12
                })
                
        # Yang S=4 from P2
        p2_s4 = df_p2[(df_p2["K"] == k_val) & (df_p2["S"] == 4)].copy()
        p2_s4["subch"] = np.ceil((p2_s4["bytes"] + 12.0) / TBS_BYTES)
        fr_subch = p2_s4.groupby(["stream", "frame_idx"])["subch"].sum().values
        fr_bytes = p2_s4.groupby(["stream", "frame_idx"])["bytes"].sum().values + 48.0 # 4 slices x 12B
        
        mean_bytes_s4 = float(np.mean(fr_bytes))
        p50_bytes_s4 = float(np.percentile(fr_bytes, 50))
        p95_bytes_s4 = float(np.percentile(fr_bytes, 95))
        p99_bytes_s4 = float(np.percentile(fr_bytes, 99))
        
        mean_subch_s4 = float(np.mean(fr_subch))
        p95_subch_s4 = float(np.percentile(fr_subch, 95))
        p99_subch_s4 = float(np.percentile(fr_subch, 99))
        
        cams_10hz_s4 = int(np.floor(6000.0 / (mean_subch_s4 * 10.0)))
        vehs_10hz_s4 = cams_10hz_s4 // 6
        cams_12hz_s4 = int(np.floor(6000.0 / (mean_subch_s4 * 12.0)))
        vehs_12hz_s4 = cams_12hz_s4 // 6
        
        summary_records.append({
            "K": k_val,
            "Codec": "Yang",
            "Mode": "Slices (S=4)",
            "Source_Column": "p2_per_frame_per_slice_bytes.csv (S=4)",
            "Mean_Payload_Bytes": round(mean_bytes_s4, 1),
            "P50_Payload_Bytes": round(p50_bytes_s4, 1),
            "P95_Payload_Bytes": round(p95_bytes_s4, 1),
            "P99_Payload_Bytes": round(p99_bytes_s4, 1),
            "Mean_Subch_Per_Frame": round(mean_subch_s4, 3),
            "P95_Subch_Per_Frame": round(p95_subch_s4, 1),
            "P99_Subch_Per_Frame": round(p99_subch_s4, 1),
            "Usable_Subch_Per_Sec": 6000,
            "Cameras_10Hz": cams_10hz_s4,
            "Vehicles_10Hz": vehs_10hz_s4,
            "Cameras_12Hz": cams_12hz_s4,
            "Vehicles_12Hz": vehs_12hz_s4
        })
        
        for cr in [0.40, 0.60, 0.80]:
            usable = int(TOTAL_SUBCH_PER_SEC * cr)
            c_10 = int(np.floor(usable / (mean_subch_s4 * 10.0)))
            v_10 = c_10 // 6
            c_12 = int(np.floor(usable / (mean_subch_s4 * 12.0)))
            v_12 = c_12 // 6
            sensitivity_records.append({
                "K": k_val,
                "Codec": "Yang",
                "Mode": "Slices (S=4)",
                "CR_max": cr,
                "Usable_Subch_Per_Sec": usable,
                "Cameras_10Hz": c_10,
                "Vehicles_10Hz": v_10,
                "Cameras_12Hz": c_12,
                "Vehicles_12Hz": v_12
            })
            
    df_sum = pd.DataFrame(summary_records)
    sum_csv = OUT_DIR / "p5_nr_v2x_capacity_summary.csv"
    df_sum.to_csv(sum_csv, index=False)
    print(f"Saved capacity summary to {sum_csv}")
    print(df_sum[["K", "Codec", "Mode", "Source_Column", "Mean_Payload_Bytes", "Mean_Subch_Per_Frame", "Cameras_10Hz", "Vehicles_10Hz", "Cameras_12Hz", "Vehicles_12Hz"]].to_string(index=False))
    
    df_sens = pd.DataFrame(sensitivity_records)
    sens_csv = OUT_DIR / "p5_nr_v2x_sensitivity.csv"
    df_sens.to_csv(sens_csv, index=False)
    print(f"Saved sensitivity summary to {sens_csv}")

if __name__ == "__main__":
    run_p5_capacity()
