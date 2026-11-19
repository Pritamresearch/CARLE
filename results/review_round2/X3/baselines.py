"""
X3: Additional Lossless Baselines (JPEG-LS)
===========================================
Evaluates lossless JPEG-LS (CharLS implementation via imagecodecs) directly
on uint8 label maps across all 60 nuScenes sweep streams (6,000 frames) at K=4 and K=8.
Decodes every frame and verifies bit-exact reconstruction.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import imagecodecs

# Enforce single-thread CPU protocol
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

def run_x3(smoke: bool = False):
    out_dir = Path("results/review_round2/X3")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sweeps_dir = Path("results/phase1b/cache_pidnet_half/sweeps")
    assert sweeps_dir.exists(), f"Sweeps dir missing: {sweeps_dir}"
    
    all_scenes = [
        "scene-0061", "scene-0103", "scene-0553", "scene-0655", "scene-0757",
        "scene-0796", "scene-0916", "scene-1077", "scene-1094", "scene-1100"
    ]
    test_scenes = ["scene-0061"] if smoke else all_scenes
    
    records = []
    t0_start = time.perf_counter()
    codec_name = "JPEG-LS"
    codec_ver = imagecodecs.jpegls_version()
    
    for k_val in [4, 8]:
        print(f"\n{'='*80}")
        print(f"RUNNING X3: JPEG-LS BENCHMARK FOR K={k_val} (smoke={smoke})")
        print(f"{'='*80}")
        
        for sc in test_scenes:
            npz_p = sweeps_dir / f"sweeps_{sc}_k{k_val}.npz"
            masks = np.load(npz_p)["masks"]
            
            for cam_i, cam_name in enumerate(CAMERAS):
                stream_name = f"{sc}_{cam_name}"
                offset = cam_i * 100
                
                for f_idx in range(100):
                    curr = masks[offset + f_idx]
                    
                    t0 = time.perf_counter_ns()
                    payload = imagecodecs.jpegls_encode(curr)
                    t_enc_ms = (time.perf_counter_ns() - t0) / 1e6
                    
                    t0 = time.perf_counter_ns()
                    rec = imagecodecs.jpegls_decode(payload)
                    t_dec_ms = (time.perf_counter_ns() - t0) / 1e6
                    
                    ok = 1 if np.array_equal(curr, rec) else 0
                    assert ok == 1, f"JPEG-LS decode mismatch on {stream_name} frame {f_idx}"
                    
                    bits = len(payload) * 8
                    records.append({
                        "K": k_val,
                        "stream": stream_name,
                        "frame_idx": f_idx,
                        "codec": codec_name,
                        "codec_version": codec_ver,
                        "bits": bits,
                        "decode_ok": ok,
                        "encode_ms": round(t_enc_ms, 3),
                        "decode_ms": round(t_dec_ms, 3)
                    })
                    
    df_out = pd.DataFrame(records)
    out_csv = out_dir / "per_frame.csv"
    df_out.to_csv(out_csv, index=False)
    
    elapsed = time.perf_counter() - t0_start
    print(f"\nSaved {len(df_out)} rows to {out_csv} in {elapsed:.1f} s")
    
    # Sanity checks
    sanity_lines = [
        f"X3 SANITY CHECKS (smoke={smoke})",
        "="*50,
        f"Total frames evaluated: {len(df_out)}",
        f"Codec: {codec_name} ({codec_ver})",
        f"Number with decode_ok = 1: {int(df_out['decode_ok'].sum())}",
        f"Number with decode_ok = 0: {len(df_out) - int(df_out['decode_ok'].sum())}",
        f"Mean bits K=4: {df_out[df_out['K'] == 4]['bits'].mean():.1f} bpf",
        f"Mean bits K=8: {df_out[df_out['K'] == 8]['bits'].mean():.1f} bpf",
        "="*50
    ]
    sanity_text = "\n".join(sanity_lines)
    print("\n" + sanity_text)
    
    sanity_p = out_dir / "sanity.txt"
    sanity_p.write_text(sanity_text + "\n", encoding="utf-8")
    print(f"Wrote sanity output to {sanity_p}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X3: Lossless baselines (JPEG-LS)")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test on 1 scene")
    args = parser.parse_args()
    run_x3(smoke=args.smoke)
