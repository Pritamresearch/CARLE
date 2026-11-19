"""
X1: Unified Continuous-Time Packet Loss Simulator
=================================================
Evaluates all transmission designs on real nuScenes sweep label maps (K=4):
- chain_S1, chain_S4, chain_S8 (horizontal slicing with 5 ms packet spreading)
- semcm_N10, semcm_N30 (inter-frame coding with periodic intra resets)
- semcm_N10_nack_fb0, semcm_N10_nack_fb10, semcm_N10_nack_fb30 (NACK feedback with 20 ms RTT)

Under three channel models at 10% packet loss:
- Bernoulli
- Continuous-time Gilbert-Elliott (T_B = 50 ms)
- Continuous-time Gilbert-Elliott (T_B = 100 ms)

Runs 100 trials per stream across seeds 0..4 (60 streams x 100 trials = 6,000 trials per seed).
Calculates exact pixel accuracy and AoI using the real label maps.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from numba import njit

# Enforce single-thread CPU protocol
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path.cwd()))

FPS = 12.0
DT_MS = 1000.0 / FPS  # 83.3333... ms
T_PROC = 103.4
CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

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

SPLITS_BY_S = {
    1: get_slice_splits(450, 1),
    4: get_slice_splits(450, 4),
    8: get_slice_splits(450, 8)
}

HEIGHTS_BY_S = {
    S: np.array([r1 - r0 for (r0, r1) in SPLITS_BY_S[S]], dtype=np.int64)
    for S in [1, 4, 8]
}

# Precompute AoI values for delta_k in 0..99
AOI_VALS = np.array([round(k * DT_MS + T_PROC, 1) for k in range(100)])

# -----------------------------------------------------------------------------
# On-Demand Memoized Band Pixel Agreements
# -----------------------------------------------------------------------------
class StreamBandAgreements:
    def __init__(self, stream_masks: np.ndarray):
        # stream_masks: shape (100, 450, 800)
        self.masks = stream_masks
        self.H, self.W = stream_masks.shape[1], stream_masks.shape[2]
        self.cache = {} # (S, s, t_src, k) -> int
        
    def get_match(self, S: int, s: int, t_src: int, k: int) -> int:
        if t_src == k:
            return int(HEIGHTS_BY_S[S][s] * self.W)
        if t_src > k:
            t_src, k = k, t_src
        key = (S, s, t_src, k)
        if key in self.cache:
            return self.cache[key]
        r0, r1 = SPLITS_BY_S[S][s]
        m1 = self.masks[t_src, r0:r1, :]
        m2 = self.masks[k, r0:r1, :]
        cnt = int(np.sum(m1 == m2))
        self.cache[key] = cnt
        return cnt

# -----------------------------------------------------------------------------
# Simulation Engine
# -----------------------------------------------------------------------------
def simulate_stream_trials(
    stream_masks: np.ndarray,
    stream_agreements: StreamBandAgreements,
    ch_type: str,
    T_B_ms: float,
    design: str,
    S: int,
    N: int,
    fb_loss: float,
    rng: np.random.Generator,
    n_trials: int = 100,
    p1b_yang_bits: np.ndarray = None,
    p1b_semcm_bits: np.ndarray = None,
    x2_temp_bits: np.ndarray = None
):
    N_frames = 100
    total_pix = 450 * 800
    
    # Slice offsets within 5 ms window
    offsets = np.array([0.0]) if S == 1 else np.linspace(0.0, 5.0, S)
    heights = HEIGHTS_BY_S[S]
    slice_weights = heights / np.sum(heights)
    
    # GE channel parameters
    per = 0.10
    if ch_type == "ge":
        T_B_s = T_B_ms / 1000.0
        mu_B = 1.0 / T_B_s
        mu_G = (per / (1.0 - per)) * mu_B
        lam = mu_G + mu_B
        
    # Trackers across all trials
    total_pkts = 0
    dropped_pkts = 0
    total_frame_instances = 0
    dropped_frame_instances = 0
    
    tot_pixel_acc = 0.0
    tot_frame_aoi = 0.0
    max_stale_val = 0.0
    tot_sent_bits = 0.0
    
    frame_aoi_hist_arr = defaultdict(int)
    pixel_aoi_hist_arr = defaultdict(int)
    
    for tr in range(n_trials):
        if ch_type == "ge":
            state = (rng.random() < per) # stationary start
            curr_time = 0.0
        else:
            state = False
            
        t_src = np.zeros(S, dtype=int) # frame 0 intact
        last_shown_k = 0
        nack_pending_for_next = False
        
        stream_sent_bits_tr = 0.0
        
        for k in range(N_frames):
            # -----------------------------------------------------------------
            # 1. Determine packet transmissions for frame k
            # -----------------------------------------------------------------
            if design.startswith("chain"):
                # Frame-independent chain coding: S slices
                frame_lost = False
                for s in range(S):
                    pkt_time = k * DT_MS + offsets[s]
                    
                    if k == 0:
                        loss = False # frame 0 intact
                    else:
                        if ch_type == "ge":
                            tau_s = (pkt_time - curr_time) / 1000.0
                            if tau_s > 0:
                                p_01 = per * (1.0 - np.exp(-lam * tau_s))
                                p_10 = (1.0 - per) * (1.0 - np.exp(-lam * tau_s))
                                r = rng.random()
                                if not state and r < p_01:
                                    state = True
                                elif state and r < p_10:
                                    state = False
                                curr_time = pkt_time
                            loss = state
                        else:
                            loss = (rng.random() < per)
                            
                    total_pkts += 1
                    if loss:
                        dropped_pkts += 1
                        frame_lost = True
                    else:
                        t_src[s] = k
                        
                total_frame_instances += 1
                if frame_lost:
                    dropped_frame_instances += 1
                    
                # Accuracy & AoI for chain
                match_sum = 0
                frame_aoi = 0.0
                for s in range(S):
                    src_f = t_src[s]
                    match_sum += stream_agreements.get_match(S, s, src_f, k)
                    delta_s = k - src_f
                    s_aoi = delta_s * DT_MS + T_PROC
                    frame_aoi += slice_weights[s] * s_aoi
                    
                    # Pixel AoI histogram
                    p_rounded = round(s_aoi, 1)
                    pixel_aoi_hist_arr[p_rounded] += heights[s] * 800
                    if delta_s * DT_MS > max_stale_val:
                        max_stale_val = delta_s * DT_MS
                    
                acc = match_sum / total_pix
                tot_pixel_acc += acc
                tot_frame_aoi += frame_aoi
                f_rounded = round(frame_aoi, 1)
                frame_aoi_hist_arr[f_rounded] += 1
                
                # Bits sent
                b_sent = p1b_yang_bits[k] if p1b_yang_bits is not None else 0.0
                stream_sent_bits_tr += b_sent
                
            else:
                # Inter-frame SemCM (S=1)
                pkt_time = k * DT_MS
                is_intra = (k == 0) or (k % N == 0) or nack_pending_for_next
                nack_pending_for_next = False
                
                if k == 0:
                    loss = False # frame 0 intact
                else:
                    if ch_type == "ge":
                        tau_s = (pkt_time - curr_time) / 1000.0
                        if tau_s > 0:
                            p_01 = per * (1.0 - np.exp(-lam * tau_s))
                            p_10 = (1.0 - per) * (1.0 - np.exp(-lam * tau_s))
                            r = rng.random()
                            if not state and r < p_01:
                                state = True
                            elif state and r < p_10:
                                state = False
                            curr_time = pkt_time
                        loss = state
                    else:
                        loss = (rng.random() < per)
                        
                total_pkts += 1
                total_frame_instances += 1
                
                # Check decodability
                if loss:
                    dropped_pkts += 1
                    dropped_frame_instances += 1
                    
                    # NACK handling if design has feedback
                    if "nack" in design:
                        fb_dropped = (rng.random() < fb_loss)
                        if not fb_dropped:
                            # NACK received in 20 ms < 83.33 ms, so next frame is forced intra
                            nack_pending_for_next = True
                else:
                    # Packet arrived
                    if is_intra:
                        last_shown_k = k
                    else:
                        # Can decode only if previous frame was shown
                        if last_shown_k == k - 1:
                            last_shown_k = k
                        else:
                            # Loss propagation: missing intermediate frame prevents decode
                            dropped_frame_instances += 1
                            if "nack" in design:
                                fb_dropped = (rng.random() < fb_loss)
                                if not fb_dropped:
                                    nack_pending_for_next = True
                                    
                # Accuracy & AoI for SemCM (displayed is last_shown_k)
                match_cnt = stream_agreements.get_match(1, 0, last_shown_k, k)
                acc = match_cnt / total_pix
                delta_k = k - last_shown_k
                frame_aoi = delta_k * DT_MS + T_PROC
                if delta_k * DT_MS > max_stale_val:
                    max_stale_val = delta_k * DT_MS
                
                tot_pixel_acc += acc
                tot_frame_aoi += frame_aoi
                f_rounded = round(frame_aoi, 1)
                frame_aoi_hist_arr[f_rounded] += 1
                pixel_aoi_hist_arr[f_rounded] += total_pix
                
                # Bits sent
                if is_intra:
                    b_sent = p1b_semcm_bits[k] if p1b_semcm_bits is not None else 0.0
                else:
                    if x2_temp_bits is not None:
                        b_sent = x2_temp_bits[k]
                    else:
                        b_sent = p1b_semcm_bits[k]
                stream_sent_bits_tr += b_sent
                
        tot_sent_bits += (stream_sent_bits_tr / N_frames)
        
    realised_p_loss = (dropped_pkts / total_pkts) if total_pkts > 0 else 0.0
    realised_f_loss = (dropped_frame_instances / total_frame_instances) if total_frame_instances > 0 else 0.0
    total_eval_frames = n_trials * N_frames
    
    return {
        "realised_packet_loss": realised_p_loss,
        "realised_frame_loss": realised_f_loss,
        "mean_pixel_acc": float(tot_pixel_acc / total_eval_frames),
        "mean_frame_aoi_ms": float(tot_frame_aoi / total_eval_frames),
        "max_stale_ms": float(max_stale_val),
        "sent_bits_per_frame": float(tot_sent_bits / n_trials),
        "aoi_hist": frame_aoi_hist_arr,
        "pixel_aoi_hist": pixel_aoi_hist_arr
    }

# -----------------------------------------------------------------------------
# Main Runner for X1
# -----------------------------------------------------------------------------
def run_x1(smoke: bool = False):
    out_dir = Path("results/review_round2/X1")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sweeps_dir = Path("results/phase1b/cache_pidnet_half/sweeps")
    p1b_k4_p = Path("results/phase1/p1b_all_frames_raw_k4.csv")
    x2_csv_p = Path("results/review_round2/X2/per_frame.csv")
    
    assert sweeps_dir.exists()
    assert p1b_k4_p.exists()
    
    p1b_df = pd.read_csv(p1b_k4_p)
    p1b_sw = p1b_df[p1b_df["dataset"] == "nuScenes_Sweeps_12Hz"]
    
    x2_df = None
    inter_ratio_used = "1.0"
    if x2_csv_p.exists():
        x2_raw = pd.read_csv(x2_csv_p)
        x2_df = x2_raw[x2_raw["K"] == 4]
        inter_ratio_used = "measured"
        print(f"X1: Using measured X2 compact-temporal sizes ({len(x2_df)} rows)")
    else:
        print("X1: X2 per_frame.csv not found; using inter_ratio_used = 1.0")
        
    all_scenes = [
        "scene-0061", "scene-0103", "scene-0553", "scene-0655", "scene-0757",
        "scene-0796", "scene-0916", "scene-1077", "scene-1094", "scene-1100"
    ]
    eval_scenes = ["scene-0061"] if smoke else all_scenes
    seeds = [0] if smoke else [0, 1, 2, 3, 4]
    
    channels = [
        ("Bernoulli", 0.0),
        ("Burst_TB50ms", 50.0),
        ("Burst_TB100ms", 100.0)
    ]
    
    designs = [
        ("chain_S1", 1, None, None),
        ("chain_S4", 4, None, None),
        ("chain_S8", 8, None, None),
        ("semcm_N10", 1, 10, None),
        ("semcm_N30", 1, 30, None),
        ("semcm_N10_nack_fb0", 1, 10, 0.0),
        ("semcm_N10_nack_fb10", 1, 10, 0.10),
        ("semcm_N10_nack_fb30", 1, 10, 0.30)
    ]
    
    print(f"\n{'='*80}")
    print(f"RUNNING X1: SIMULATOR BENCHMARK (smoke={smoke}, scenes={len(eval_scenes)}, seeds={seeds})")
    print(f"{'='*80}")
    
    t0_start = time.perf_counter()
    per_stream_rows = []
    
    # Global histograms: (channel, T_B_ms, design, seed) -> dict(aoi -> count)
    global_aoi_hist = defaultdict(lambda: defaultdict(int))
    global_pixel_aoi_hist = defaultdict(lambda: defaultdict(int))
    
    for sc in eval_scenes:
        t_sc = time.perf_counter()
        npz_p = sweeps_dir / f"sweeps_{sc}_k4.npz"
        masks = np.load(npz_p)["masks"]
        
        for cam_i, cam_name in enumerate(CAMERAS):
            stream_name = f"{sc}_{cam_name}"
            stream_masks = masks[cam_i*100 : (cam_i+1)*100] # shape (100, 450, 800)
            
            # On-demand memoized band agreements for stream
            stream_agreements = StreamBandAgreements(stream_masks)
                
            # Extract bits for this stream
            p1b_sub = p1b_sw[p1b_sw["stream"] == stream_name].sort_values("frame_idx")
            p1b_yang = p1b_sub["yang_bits"].values if len(p1b_sub) == 100 else None
            p1b_semcm = p1b_sub["semcm_a_bits"].values if len(p1b_sub) == 100 else None
            
            x2_temp = None
            if x2_df is not None:
                x2_sub = x2_df[x2_df["stream"] == stream_name].sort_values("frame_idx")
                if len(x2_sub) == 100:
                    x2_temp = x2_sub["compact_temp_bits"].values
                    
            for ch_name, tb_ms in channels:
                ch_type = "bernoulli" if ch_name == "Bernoulli" else "ge"
                
                for des_name, s_val, n_val, fb_val in designs:
                    for seed in seeds:
                        rng = np.random.default_rng(seed * 10000 + cam_i * 100 + int(tb_ms))
                        
                        res = simulate_stream_trials(
                            stream_masks=stream_masks,
                            stream_agreements=stream_agreements,
                            ch_type=ch_type,
                            T_B_ms=tb_ms,
                            design=des_name,
                            S=s_val,
                            N=n_val,
                            fb_loss=fb_val if fb_val is not None else 0.0,
                            rng=rng,
                            n_trials=100,
                            p1b_yang_bits=p1b_yang,
                            p1b_semcm_bits=p1b_semcm,
                            x2_temp_bits=x2_temp
                        )
                        
                        per_stream_rows.append({
                            "channel": ch_name,
                            "T_B_ms": tb_ms,
                            "design": des_name,
                            "S": s_val,
                            "N": n_val if n_val is not None else "",
                            "fb_loss": fb_val if fb_val is not None else "",
                            "seed": seed,
                            "scene": sc,
                            "stream": stream_name,
                            "n_trials": 100,
                            "realised_packet_loss": round(res["realised_packet_loss"], 6),
                            "realised_frame_loss": round(res["realised_frame_loss"], 6),
                            "mean_pixel_acc": round(res["mean_pixel_acc"], 6),
                            "mean_frame_aoi_ms": round(res["mean_frame_aoi_ms"], 3),
                            "max_stale_ms": round(res["max_stale_ms"], 3),
                            "sent_bits_per_frame": round(res["sent_bits_per_frame"], 2),
                            "inter_ratio_used": inter_ratio_used
                        })
                        
                        # Merge histograms
                        hist_key = (ch_name, tb_ms, des_name, seed)
                        for aoi_val, cnt in res["aoi_hist"].items():
                            global_aoi_hist[hist_key][aoi_val] += cnt
                        for p_aoi_val, cnt in res["pixel_aoi_hist"].items():
                            global_pixel_aoi_hist[hist_key][p_aoi_val] += cnt
                            
        print(f"Completed scene {sc} in {time.perf_counter() - t_sc:.2f} s")
        
    # Save per_stream.csv
    df_stream = pd.DataFrame(per_stream_rows)
    p_stream_csv = out_dir / "per_stream.csv"
    df_stream.to_csv(p_stream_csv, index=False)
    print(f"\nSaved {len(df_stream)} rows to {p_stream_csv}")
    
    # Save aoi_hist.csv
    aoi_hist_rows = []
    for (ch_name, tb_ms, des_name, seed), h_dict in global_aoi_hist.items():
        for aoi_val, cnt in sorted(h_dict.items()):
            aoi_hist_rows.append({
                "channel": ch_name,
                "T_B_ms": tb_ms,
                "design": des_name,
                "seed": seed,
                "aoi_ms_rounded_0p1": aoi_val,
                "count": cnt
            })
    df_aoi_hist = pd.DataFrame(aoi_hist_rows)
    p_aoi_hist_csv = out_dir / "aoi_hist.csv"
    df_aoi_hist.to_csv(p_aoi_hist_csv, index=False)
    print(f"Saved {len(df_aoi_hist)} rows to {p_aoi_hist_csv}")
    
    # Save pixel_aoi_hist.csv
    pix_hist_rows = []
    for (ch_name, tb_ms, des_name, seed), h_dict in global_pixel_aoi_hist.items():
        for aoi_val, cnt in sorted(h_dict.items()):
            pix_hist_rows.append({
                "channel": ch_name,
                "T_B_ms": tb_ms,
                "design": des_name,
                "seed": seed,
                "aoi_ms_rounded_0p1": aoi_val,
                "count": cnt
            })
    df_pix_hist = pd.DataFrame(pix_hist_rows)
    p_pix_hist_csv = out_dir / "pixel_aoi_hist.csv"
    df_pix_hist.to_csv(p_pix_hist_csv, index=False)
    print(f"Saved {len(df_pix_hist)} rows to {p_pix_hist_csv}")
    
    # Sanity checks
    sanity_lines = [
        f"X1 SANITY CHECKS (smoke={smoke})",
        "="*60
    ]
    for ch_name, _ in channels:
        sub_ch = df_stream[df_stream["channel"] == ch_name]
        mean_p_loss = sub_ch["realised_packet_loss"].mean() * 100.0
        sanity_lines.append(f"Realised packet loss for {ch_name:15s}: {mean_p_loss:.2f}% (target: 10.00%)")
        
    # Bernoulli chain_S1 check
    b_s1 = df_stream[(df_stream["channel"] == "Bernoulli") & (df_stream["design"] == "chain_S1")]
    sim_mean_aoi = b_s1["mean_frame_aoi_ms"].mean()
    p_real = b_s1["realised_packet_loss"].mean()
    theory_aoi = T_PROC + DT_MS * (p_real / (1.0 - p_real))
    sanity_lines.append(f"Bernoulli chain_S1 mean AoI: {sim_mean_aoi:.2f} ms")
    sanity_lines.append(f"Theoretical AoI (103.4 + dt*p/(1-p)): {theory_aoi:.2f} ms (p={p_real:.4f})")
    sanity_lines.append(f"Difference: {sim_mean_aoi - theory_aoi:+.2f} ms")
    
    # Determinism / Equality check
    b_s1_seed0 = b_s1[b_s1["seed"] == seeds[0]]
    sanity_lines.append(f"Verified chain_S1 trials count: {len(b_s1_seed0)} streams x 100 trials")
    sanity_lines.append("="*60)
    
    sanity_text = "\n".join(sanity_lines)
    print("\n" + sanity_text)
    
    sanity_p = out_dir / "sanity.txt"
    sanity_p.write_text(sanity_text + "\n", encoding="utf-8")
    print(f"Wrote sanity output to {sanity_p}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X1: Simulator for all loss results")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test on 1 scene, 1 seed")
    args = parser.parse_args()
    run_x1(smoke=args.smoke)
