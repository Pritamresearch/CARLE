"""
E7: Continuous-time Gilbert-Elliott and Bernoulli Channel Simulator
===================================================================
Simulates packet transmission at 10 Hz (and 12 Hz), slicing each frame into S in {1, 4, 8}
packets transmitted within a 5 ms window, under continuous-time Gilbert-Elliott burst channels
(T_B = 50 ms and 100 ms, P_L = 10%) and Bernoulli packet loss (P_L = 10%).

AoI: A_t = (t - t_src) * dt + T_proc, with T_proc = 103.4 ms.
Evaluates 6,000 streams x 100 frames across seeds 0..4.
"""

import sys
import os
import argparse
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd

def simulate_channel(
    fps: float = 10.0,
    ch_type: str = "ge",
    T_B_ms: float = 50.0,
    per: float = 0.10,
    S: int = 1,
    N_streams: int = 6000,
    N_frames: int = 100,
    seed: int = 0,
    pkt_window_ms: float = 5.0
):
    dt_ms = 1000.0 / fps
    T_proc = 103.4
    rng = np.random.default_rng(seed)
    
    # Packet transmission offsets within the frame window
    if S == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(0.0, pkt_window_ms, S)
        
    # State tracking: 0 = Good (delivered), 1 = Bad (lost)
    if ch_type == "ge":
        T_B_s = T_B_ms / 1000.0
        mu_B = 1.0 / T_B_s
        mu_G = (per / (1.0 - per)) * mu_B
        lam = mu_G + mu_B
        # Initial stationary distribution
        state = (rng.random(N_streams) < per).astype(int)
    else:
        # Bernoulli
        state = None
        
    t_src = np.zeros((N_streams, S), dtype=int)
    frame_aoi = np.zeros((N_streams, N_frames), dtype=float)
    slice_aoi = np.zeros((N_streams, N_frames, S), dtype=float)
    
    # Realised loss tracking
    total_pkts = N_streams * N_frames * S
    dropped_pkts = 0
    
    curr_time = 0.0
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
                
            dropped_pkts += int(np.sum(loss))
            delivered = ~loss
            t_src[:, s] = np.where(delivered, k, t_src[:, s])
            
        s_aoi = (k - t_src) * dt_ms + T_proc
        slice_aoi[:, k, :] = s_aoi
        frame_aoi[:, k] = np.mean(s_aoi, axis=1)
        
    realised_loss = (dropped_pkts / total_pkts) * 100.0
    
    # Percentiles over pooled frames and pooled pixels
    # In equal slice sizing, per-pixel AoI is equivalent to pooled slice AoI
    return {
        "fps": fps,
        "channel": f"Burst_TB{int(T_B_ms)}ms" if ch_type == "ge" else f"Bernoulli_{int(per*100)}%",
        "T_B_ms": T_B_ms if ch_type == "ge" else 0.0,
        "S": S,
        "seed": seed,
        "N_streams": N_streams,
        "N_frames": N_frames,
        "realised_loss_pct": realised_loss,
        "mean_frame_aoi": float(np.mean(frame_aoi)),
        "p95_frame_aoi": float(np.percentile(frame_aoi, 95)),
        "p99_frame_aoi": float(np.percentile(frame_aoi, 99)),
        "mean_pixel_aoi": float(np.mean(slice_aoi)),
        "p95_pixel_aoi": float(np.percentile(slice_aoi, 95)),
        "p99_pixel_aoi": float(np.percentile(slice_aoi, 99)),
    }

def main():
    parser = argparse.ArgumentParser(description="E7 Channel and AoI Simulator")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test with 100 streams")
    parser.add_argument("--fps", type=float, default=10.0, help="Frame rate in Hz (default: 10.0)")
    parser.add_argument("--out_dir", type=str, default="results/review_round/E7", help="Output directory")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    N_streams = 100 if args.smoke else 6000
    N_frames = 100
    seeds = [0] if args.smoke else [0, 1, 2, 3, 4]
    
    print(f"Starting E7 simulation (smoke={args.smoke}, fps={args.fps}, N_streams={N_streams}, seeds={seeds})...")
    
    channels = [
        ("bernoulli", 0.0, 0.10),
        ("ge", 50.0, 0.10),
        ("ge", 100.0, 0.10)
    ]
    
    records = []
    t0 = time.time()
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        for ch_type, tb, per in channels:
            for S in [1, 4, 8]:
                res = simulate_channel(
                    fps=args.fps,
                    ch_type=ch_type,
                    T_B_ms=tb,
                    per=per,
                    S=S,
                    N_streams=N_streams,
                    N_frames=N_frames,
                    seed=seed
                )
                records.append(res)
                ch_lbl = f"Burst {int(tb)}ms" if ch_type == "ge" else "Bernoulli"
                print(f"Seed {seed} | {ch_lbl:<12} | S={S} | Realised Loss: {res['realised_loss_pct']:.2f}% | "
                      f"Frame AoI: Mean={res['mean_frame_aoi']:.2f}, p95={res['p95_frame_aoi']:.2f}, p99={res['p99_frame_aoi']:.2f} | "
                      f"Pixel AoI: p95={res['p95_pixel_aoi']:.2f}, p99={res['p99_pixel_aoi']:.2f}")
                
    df = pd.DataFrame(records)
    csv_path = out_dir / f"sim_results_fps{int(args.fps)}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSimulation complete in {time.time() - t0:.2f} s. Saved to {csv_path}")

if __name__ == "__main__":
    main()
