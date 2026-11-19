"""
V6.5 Rerun: SemCM N=10 and N=30 under Continuous-Time Simulator at 12 Hz
======================================================================
Reruns SemCM N=10 and N=30 using:
1. Inter-frame size ratio 0.9697 (K=4 measured ratio)
2. Inter-frame size ratio 1.0 (no saving baseline)
Across:
- Bernoulli (10%)
- Burst TB = 50 ms (10%)
- Burst TB = 100 ms (10%)
Evaluates 60 streams x 100 frames x 5 seeds (0..4) at 12 Hz (dt = 83.333 ms, T_proc = 103.4 ms).
Reports: Mean AoI, 95% CI of mean, p95 AoI, p99 AoI, Max AoI, Pixel Accuracy, Sent kbps, Recv kbps.
"""

import sys
import os
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd

CACHE_DIR_HALF = Path("results/phase1b/cache_pidnet_half/sweeps")
P1B_K4_CSV = Path("results/phase1/p1b_all_frames_raw_k4.csv")

def run_v6_recheck():
    t0 = time.time()
    print("Loading P1b K=4 CSV...", flush=True)
    df_p1b = pd.read_csv(P1B_K4_CSV)
    sw = df_p1b[df_p1b["dataset"] == "nuScenes_Sweeps_12Hz"].copy()
    
    intra_map = {}
    for _, row in sw.iterrows():
        st = row["stream"]
        fidx = int(row["frame_idx"])
        intra_map[(st, fidx)] = int(np.ceil(row["semcm_a_bits"] / 8.0))
        
    meta_files = sorted(list(CACHE_DIR_HALF.glob("sweeps_scene-*_meta.json")))
    streams_meta = {}
    for mf in meta_files:
        with open(mf, "r") as f:
            records = json.load(f)
        for r in records:
            s_key = f"{r['scene']}_{r['camera']}"
            if s_key not in streams_meta:
                streams_meta[s_key] = []
            if len(streams_meta[s_key]) < 100:
                streams_meta[s_key].append(r)
                
    stream_keys = sorted(list(streams_meta.keys()))
    scenes = sorted(list(set([k.split("_")[0] for k in stream_keys])))
    N_streams = len(stream_keys)
    N_frames = 100
    fps = 12.0
    dt_ms = 1000.0 / fps # 83.333 ms
    T_proc = 103.4
    
    print(f"Loading cached masks for {len(scenes)} scenes...", flush=True)
    scene_masks = {}
    for sc in scenes:
        npz_p = CACHE_DIR_HALF / f"sweeps_{sc}_k4.npz"
        scene_masks[sc] = np.load(npz_p)["masks"] # shape (600, 450, 800)
        
    cam_order = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    stream_cam_idx = [cam_order.index(s.split("_", 1)[1]) for s in stream_keys]
    stream_sc_name = [s.split("_", 1)[0] for s in stream_keys]
    
    channels = [
        ("Bernoulli_10%", "bernoulli", 0.0, 0.10),
        ("Burst_TB50ms", "ge", 50.0, 0.10),
        ("Burst_TB100ms", "ge", 100.0, 0.10),
    ]
    
    ratios = [
        ("K4_Ratio_0.9697", 0.9697),
        ("NoSaving_Ratio_1.0", 1.0)
    ]
    
    results = []
    
    for r_name, ratio in ratios:
        for ch_name, ch_type, tb, per in channels:
            for N in [10, 30]:
                t_cfg = time.time()
                all_stream_mean_aois = {sc: [] for sc in scenes}
                all_aois = []
                all_max_aois = []
                total_acc = 0.0
                total_frames_count = 0
                total_sent_bytes = 0
                total_recv_bytes = 0
                
                for seed in range(5):
                    rng = np.random.default_rng(seed)
                    
                    if ch_type == "ge":
                        T_B_s = tb / 1000.0
                        mu_B = 1.0 / T_B_s
                        mu_G = (per / (1.0 - per)) * mu_B
                        lam = mu_G + mu_B
                        state = (rng.random(N_streams) < per).astype(int)
                    else:
                        state = None
                        
                    curr_time = 0.0
                    sync_state = np.ones(N_streams, dtype=bool)
                    last_valid = np.zeros(N_streams, dtype=int)
                    stream_max_stale = np.zeros(N_streams, dtype=float)
                    
                    for k in range(N_frames):
                        pkt_time = k * dt_ms
                        
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
                            
                        is_intra = (k % N == 0)
                        
                        for st_idx in range(N_streams):
                            s_key = stream_keys[st_idx]
                            b_intra = intra_map.get((s_key, k), 569)
                            b_delta = int(np.round(b_intra * ratio))
                            frame_bytes = (b_intra if is_intra else b_delta) + 12
                            
                            total_sent_bytes += frame_bytes
                            
                            if not loss[st_idx]:
                                total_recv_bytes += frame_bytes
                                if is_intra or sync_state[st_idx]:
                                    sync_state[st_idx] = True
                                    last_valid[st_idx] = k
                            else:
                                sync_state[st_idx] = False
                                
                            aoi_k = (k - last_valid[st_idx]) * dt_ms + T_proc
                            all_aois.append(aoi_k)
                            all_stream_mean_aois[stream_sc_name[st_idx]].append(aoi_k)
                            if aoi_k > stream_max_stale[st_idx]:
                                stream_max_stale[st_idx] = aoi_k
                                
                            # Fast pixel accuracy: 1.0 if refreshed, else compare
                            if last_valid[st_idx] == k:
                                total_acc += 1.0
                            else:
                                sc_n = stream_sc_name[st_idx]
                                c_i = stream_cam_idx[st_idx]
                                m_curr = scene_masks[sc_n][c_i * 100 + k]
                                m_disp = scene_masks[sc_n][c_i * 100 + last_valid[st_idx]]
                                total_acc += float(np.mean(m_curr == m_disp))
                            total_frames_count += 1
                            
                    all_max_aois.extend(stream_max_stale.tolist())
                    
                mean_aoi = float(np.mean(all_aois))
                p95_aoi = float(np.percentile(all_aois, 95))
                p99_aoi = float(np.percentile(all_aois, 99))
                max_aoi = float(np.max(all_max_aois))
                mean_acc = (total_acc / total_frames_count) * 100.0
                
                sc_means = np.array([np.mean(all_stream_mean_aois[sc]) for sc in scenes])
                rng_boot = np.random.default_rng(0)
                resamp = rng_boot.integers(0, len(scenes), size=(10000, len(scenes)))
                boot_means = sc_means[resamp].mean(axis=1)
                ci_l, ci_h = np.percentile(boot_means, [2.5, 97.5])
                
                sent_kbps = (total_sent_bytes * 8.0) / (5 * N_streams * N_frames * (dt_ms / 1000.0) * 1000.0)
                recv_kbps = (total_recv_bytes * 8.0) / (5 * N_streams * N_frames * (dt_ms / 1000.0) * 1000.0)
                
                rec = {
                    "Ratio_Config": r_name,
                    "Ratio_Value": ratio,
                    "Channel": ch_name,
                    "Design": f"SemCM_N{N}",
                    "N": N,
                    "Mean_AoI_ms": mean_aoi,
                    "CI_95_Low": float(ci_l),
                    "CI_95_High": float(ci_h),
                    "p95_AoI_ms": p95_aoi,
                    "p99_AoI_ms": p99_aoi,
                    "Max_AoI_ms": max_aoi,
                    "Pixel_Accuracy_pct": mean_acc,
                    "Sent_kbps": sent_kbps,
                    "Recv_kbps": recv_kbps
                }
                results.append(rec)
                print(f"[{r_name}] {ch_name} {rec['Design']} in {time.time()-t_cfg:.2f}s | "
                      f"Mean AoI: {mean_aoi:.1f} [{ci_l:.1f}, {ci_h:.1f}] | p95: {p95_aoi:.1f} | "
                      f"p99: {p99_aoi:.1f} | Sent: {sent_kbps:.2f} kb/s", flush=True)
                
    res_df = pd.DataFrame(results)
    out_csv = Path("results/review_round/E7/v6_semcm_recheck_results.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nAll SemCM recheck runs complete in {time.time()-t0:.2f}s! Saved to {out_csv}", flush=True)

if __name__ == "__main__":
    run_v6_recheck()
