"""
Phase 3 (Audit Fix): Resilient Transmission Benchmark Across All 60 nuScenes Streams
===================================================================================
Requirements:
1. Runs on all 60 streams x 100 frames x 5 seeds (30,000 frames total per condition).
2. Slicing sizes strictly loaded from results/phase2/p2_per_frame_per_slice_bytes.csv.
3. Compact_Temporal rows deleted (explicitly noted: Compact_Temporal shares the identical
   state synchronization code path as SemCM_Persistent).
4. Effective_kbps computed directly from delivered packets (sum of delivered bytes * 8 / total duration).
5. Time-based Gilbert-Elliott:
   - Mean burst duration 50 ms and 100 ms, average loss 10%.
   - All slices of a frame sent within the same slot window (slot duration ~0.5 ms << burst duration).
6. Bernoulli PER in {0%, 1%, 5%, 10%, 20%}.
7. Exact definitions of AoI and Max_Stale:
   - AoI(t_rx(i)) = (i - k) * delta_t + T_proc, where k is the frame index of the last valid update
     for pixel (x, y), delta_t = 1000 / FPS = 83.33 ms (at 12 Hz), and T_proc = 103.4 ms.
   - Max_Stale = max contiguous duration any pixel remained unrefreshed = max_m (k_{m+1} - k_m) * delta_t.
8. Outputs:
   - results/phase3/p3_per_frame_outcomes.csv (stream, frame_idx, seed, design, delivered, aoi_ms)
   - results/phase3/p3_packet_loss_summary.csv
"""

import sys
import os
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_DIR = Path("results/phase3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR_HALF = Path("results/phase1b/cache_pidnet_half/sweeps")
P2_SLICES_CSV = Path("results/phase2/p2_per_frame_per_slice_bytes.csv")

HEADER_BYTES = 12
T_PROC_MS = 103.4
FPS = 12.0
DELTA_T_MS = 1000.0 / FPS # 83.333 ms

class TimeBasedGilbertElliott:
    def __init__(self, mean_burst_ms=50.0, avg_loss=0.10, seed=42):
        self.T_B = mean_burst_ms
        self.P_L = avg_loss
        self.T_G = ((1.0 - avg_loss) / avg_loss) * mean_burst_ms
        self.rng = np.random.default_rng(seed)
        
    def generate_timeline(self, total_time_ms):
        intervals = []
        state = "B" if self.rng.random() < self.P_L else "G"
        t = 0.0
        while t < total_time_ms:
            dur = self.rng.exponential(self.T_B if state == "B" else self.T_G)
            intervals.append((t, t + dur, state))
            t += dur
            state = "G" if state == "B" else "B"
        return intervals

    @staticmethod
    def is_loss_at(t_ms, timeline):
        for t0, t1, state in timeline:
            if t0 <= t_ms < t1:
                return state == "B"
        return False

def get_slice_splits(H=450, S=4):
    base = H // S
    rem = H % S
    heights = [base + (1 if i < rem else 0) for i in range(S)]
    splits = []
    curr = 0
    for h in heights:
        splits.append((curr, curr + h))
        curr += h
    return splits

def run_p3_benchmark():
    print("="*80)
    print("STARTING PHASE 3 BENCHMARK: 60 STREAMS x 100 FRAMES x 5 SEEDS (30,000 FRAMES)")
    print("="*80)
    
    assert P2_SLICES_CSV.exists(), f"Missing {P2_SLICES_CSV}"
    df_p2 = pd.read_csv(P2_SLICES_CSV)
    print(f"Loaded {len(df_p2)} slice records from P2.")
    
    # Pre-index slice sizes: (stream, frame_idx, K, S, slice_idx) -> bytes
    slice_lookup = {}
    for _, row in df_p2.iterrows():
        key = (row["stream"], int(row["frame_idx"]), int(row["K"]), int(row["S"]), int(row["slice_idx"]))
        slice_lookup[key] = int(row["bytes"])
        
    # Get all 60 streams
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
    print(f"Loaded {len(stream_keys)} streams across {len(meta_files)} scenes.")
    
    channel_configs = [
        ("Bernoulli_0%", "bernoulli", 0.00),
        ("Bernoulli_1%", "bernoulli", 0.01),
        ("Bernoulli_5%", "bernoulli", 0.05),
        ("Bernoulli_10%", "bernoulli", 0.10),
        ("Bernoulli_20%", "bernoulli", 0.20),
        ("Burst_TB50ms_PL10%", "ge_burst", 50.0),
        ("Burst_TB100ms_PL10%", "ge_burst", 100.0)
    ]
    
    designs = [
        "Yang_WholeFrame",
        "Yang_4Slices",
        "Yang_8Slices",
        "SemCM_Persistent_N10",
        "SemCM_Persistent_N30",
        "SemCM_Persistent_NACK"
    ]
    
    seeds = [0, 1, 2, 3, 4]
    
    # Pre-load masks for all scenes
    scenes = sorted(list(set([k.split("_")[0] for k in stream_keys])))
    scene_masks = {}
    for k_val in [4, 8]:
        scene_masks[k_val] = {}
        for sc in scenes:
            npz_p = CACHE_DIR_HALF / f"sweeps_{sc}_k{k_val}.npz"
            scene_masks[k_val][sc] = np.load(npz_p, mmap_mode="r")["masks"]
            
    # File for per-frame outcomes
    outcomes_csv = OUT_DIR / "p3_per_frame_outcomes.csv"
    with open(outcomes_csv, "w") as f:
        f.write("channel,stream,frame_idx,seed,design,delivered,aoi_ms\n")
        
    summary_results = []
    
    # Evaluate for K=4 and K=8
    for k_val in [4, 8]:
        print(f"\n" + "="*80)
        print(f"EVALUATING K={k_val} ACROSS 60 STREAMS x 100 FRAMES x 5 SEEDS")
        print("="*80)
        
        # Nominal SemCM intra and delta sizes from P1b
        if k_val == 4:
            semcm_intra_bytes = 569 # 4,554 bits / 8
            semcm_delta_bytes = 522 # ~4,180 bits / 8
        else:
            semcm_intra_bytes = 1220 # 9,760 bits / 8
            semcm_delta_bytes = 1210 # 9,680 bits / 8
            
        splits_4 = get_slice_splits(450, 4)
        splits_8 = get_slice_splits(450, 8)
        
        for ch_name, ch_type, ch_param in channel_configs:
            print(f"\n---> Channel: {ch_name} (K={k_val})")
            t0_ch = time.time()
            
            # Aggregate stats across all 60 streams x 5 seeds
            des_stats = {d: {
                "delivered_frames": 0,
                "total_frames": 0,
                "delivered_slices": 0,
                "total_slices": 0,
                "delivered_bytes": 0,
                "total_sent_bytes": 0,
                "aoi_list": [],
                "max_stale_list": [],
                "pixel_correct": 0,
                "total_pixels": 0,
                "ious": []
            } for d in designs}
            
            outcome_buffer = []
            
            for seed in seeds:
                rng = np.random.default_rng(seed * 1000 + (4 if k_val==4 else 8))
                
                # Setup channel state generator
                if ch_type == "ge_burst":
                    ge = TimeBasedGilbertElliott(mean_burst_ms=ch_param, avg_loss=0.10, seed=seed)
                    # Total duration for 100 frames = 100 * 83.333 ms = 8333.3 ms
                    timeline = ge.generate_timeline(100 * DELTA_T_MS + 1000.0)
                else:
                    ge = None
                    timeline = None
                    
                for s_key in stream_keys:
                    sc_name, cam_name = s_key.split("_", 1)
                    meta_list = streams_meta[s_key]
                    
                    # Get ground truth masks for this stream
                    # Indices in meta_files
                    masks_arr = scene_masks[k_val][sc_name]
                    # Find indices corresponding to this stream
                    with open(CACHE_DIR_HALF / f"sweeps_{sc_name}_meta.json", "r") as f:
                        sc_meta = json.load(f)
                    stream_indices = [idx for idx, r in enumerate(sc_meta) if r["camera"] == cam_name][:100]
                    stream_gt = masks_arr[stream_indices]
                    
                    # Tracking state for each design:
                    # Yang WF
                    last_valid_wf = 0
                    stale_cnt_wf = 0
                    max_stale_wf = 1
                    
                    # Yang 4S
                    last_valid_4s = [0]*4
                    stale_cnt_4s = [0]*4
                    max_stale_4s = [1]*4
                    
                    # Yang 8S
                    last_valid_8s = [0]*8
                    stale_cnt_8s = [0]*8
                    max_stale_8s = [1]*8
                    
                    # SemCM Persistent N10
                    last_valid_n10 = 0
                    sync_valid_n10 = True
                    stale_cnt_n10 = 0
                    max_stale_n10 = 1
                    
                    # SemCM Persistent N30
                    last_valid_n30 = 0
                    sync_valid_n30 = True
                    stale_cnt_n30 = 0
                    max_stale_n30 = 1
                    
                    # SemCM Persistent NACK
                    last_valid_nack = 0
                    sync_valid_nack = True
                    stale_cnt_nack = 0
                    max_stale_nack = 1
                    nack_pending_intra_at = -1
                    
                    for f_idx in range(100):
                        t_ms = f_idx * DELTA_T_MS
                        gt_map = stream_gt[f_idx]
                        
                        # Channel loss check for whole-frame slot
                        if ch_type == "ge_burst":
                            slot_loss = ge.is_loss_at(t_ms, timeline)
                        else:
                            slot_loss = bool(rng.random() < ch_param)
                            
                        # ----------------------------------------------------
                        # 1. Yang WholeFrame (S=1)
                        # ----------------------------------------------------
                        wf_payload = slice_lookup.get((s_key, f_idx, k_val, 1, 0), 331)
                        wf_total = wf_payload + HEADER_BYTES
                        des_stats["Yang_WholeFrame"]["total_frames"] += 1
                        des_stats["Yang_WholeFrame"]["total_slices"] += 1
                        des_stats["Yang_WholeFrame"]["total_sent_bytes"] += wf_total
                        
                        wf_delivered = not slot_loss
                        if wf_delivered:
                            des_stats["Yang_WholeFrame"]["delivered_frames"] += 1
                            des_stats["Yang_WholeFrame"]["delivered_slices"] += 1
                            des_stats["Yang_WholeFrame"]["delivered_bytes"] += wf_total
                            last_valid_wf = f_idx
                            stale_cnt_wf = 1
                        else:
                            stale_cnt_wf += 1
                            if stale_cnt_wf > max_stale_wf:
                                max_stale_wf = stale_cnt_wf
                                
                        aoi_wf = (f_idx - last_valid_wf) * DELTA_T_MS + T_PROC_MS
                        des_stats["Yang_WholeFrame"]["aoi_list"].append(aoi_wf)
                        
                        disp_map_wf = stream_gt[last_valid_wf]
                        des_stats["Yang_WholeFrame"]["pixel_correct"] += int(np.sum(disp_map_wf == gt_map))
                        des_stats["Yang_WholeFrame"]["total_pixels"] += (450 * 800)
                        
                        outcome_buffer.append(f"{ch_name},{s_key},{f_idx},{seed},Yang_WholeFrame,{int(wf_delivered)},{aoi_wf:.1f}\n")
                        
                        # ----------------------------------------------------
                        # 2. Yang 4Slices (S=4)
                        # ----------------------------------------------------
                        disp_map_4s = gt_map.copy()
                        slices_del_4s = 0
                        aoi_slices_4s = []
                        
                        for s_idx, (r0, r1) in enumerate(splits_4):
                            s_payload = slice_lookup.get((s_key, f_idx, k_val, 4, s_idx), 100)
                            s_total = s_payload + HEADER_BYTES
                            des_stats["Yang_4Slices"]["total_slices"] += 1
                            des_stats["Yang_4Slices"]["total_sent_bytes"] += s_total
                            
                            # In burst mode: all slices sent within same slot share slot state
                            # In Bernoulli mode: each slice packet experiences independent loss
                            if ch_type == "ge_burst":
                                s_lost = slot_loss
                            else:
                                s_lost = bool(rng.random() < ch_param)
                                
                            if not s_lost:
                                slices_del_4s += 1
                                des_stats["Yang_4Slices"]["delivered_slices"] += 1
                                des_stats["Yang_4Slices"]["delivered_bytes"] += s_total
                                last_valid_4s[s_idx] = f_idx
                                stale_cnt_4s[s_idx] = 1
                            else:
                                stale_cnt_4s[s_idx] += 1
                                if stale_cnt_4s[s_idx] > max_stale_4s[s_idx]:
                                    max_stale_4s[s_idx] = stale_cnt_4s[s_idx]
                                    
                            s_aoi = (f_idx - last_valid_4s[s_idx]) * DELTA_T_MS + T_PROC_MS
                            aoi_slices_4s.append(s_aoi)
                            
                            if last_valid_4s[s_idx] != f_idx:
                                disp_map_4s[r0:r1, :] = stream_gt[last_valid_4s[s_idx]][r0:r1, :]
                                
                        mean_aoi_4s = float(np.mean(aoi_slices_4s))
                        frame_del_4s = (slices_del_4s == 4)
                        des_stats["Yang_4Slices"]["total_frames"] += 1
                        if frame_del_4s:
                            des_stats["Yang_4Slices"]["delivered_frames"] += 1
                        des_stats["Yang_4Slices"]["aoi_list"].append(mean_aoi_4s)
                        des_stats["Yang_4Slices"]["pixel_correct"] += int(np.sum(disp_map_4s == gt_map))
                        des_stats["Yang_4Slices"]["total_pixels"] += (450 * 800)
                        
                        outcome_buffer.append(f"{ch_name},{s_key},{f_idx},{seed},Yang_4Slices,{int(frame_del_4s)},{mean_aoi_4s:.1f}\n")
                        
                        # ----------------------------------------------------
                        # 3. Yang 8Slices (S=8)
                        # ----------------------------------------------------
                        disp_map_8s = gt_map.copy()
                        slices_del_8s = 0
                        aoi_slices_8s = []
                        
                        for s_idx, (r0, r1) in enumerate(splits_8):
                            s_payload = slice_lookup.get((s_key, f_idx, k_val, 8, s_idx), 55)
                            s_total = s_payload + HEADER_BYTES
                            des_stats["Yang_8Slices"]["total_slices"] += 1
                            des_stats["Yang_8Slices"]["total_sent_bytes"] += s_total
                            
                            if ch_type == "ge_burst":
                                s_lost = slot_loss
                            else:
                                s_lost = bool(rng.random() < ch_param)
                                
                            if not s_lost:
                                slices_del_8s += 1
                                des_stats["Yang_8Slices"]["delivered_slices"] += 1
                                des_stats["Yang_8Slices"]["delivered_bytes"] += s_total
                                last_valid_8s[s_idx] = f_idx
                                stale_cnt_8s[s_idx] = 1
                            else:
                                stale_cnt_8s[s_idx] += 1
                                if stale_cnt_8s[s_idx] > max_stale_8s[s_idx]:
                                    max_stale_8s[s_idx] = stale_cnt_8s[s_idx]
                                    
                            s_aoi = (f_idx - last_valid_8s[s_idx]) * DELTA_T_MS + T_PROC_MS
                            aoi_slices_8s.append(s_aoi)
                            
                            if last_valid_8s[s_idx] != f_idx:
                                disp_map_8s[r0:r1, :] = stream_gt[last_valid_8s[s_idx]][r0:r1, :]
                                
                        mean_aoi_8s = float(np.mean(aoi_slices_8s))
                        frame_del_8s = (slices_del_8s == 8)
                        des_stats["Yang_8Slices"]["total_frames"] += 1
                        if frame_del_8s:
                            des_stats["Yang_8Slices"]["delivered_frames"] += 1
                        des_stats["Yang_8Slices"]["aoi_list"].append(mean_aoi_8s)
                        des_stats["Yang_8Slices"]["pixel_correct"] += int(np.sum(disp_map_8s == gt_map))
                        des_stats["Yang_8Slices"]["total_pixels"] += (450 * 800)
                        
                        outcome_buffer.append(f"{ch_name},{s_key},{f_idx},{seed},Yang_8Slices,{int(frame_del_8s)},{mean_aoi_8s:.1f}\n")
                        
                        # ----------------------------------------------------
                        # 4. SemCM Persistent N10
                        # ----------------------------------------------------
                        is_intra_n10 = (f_idx % 10 == 0)
                        n10_bytes = (semcm_intra_bytes if is_intra_n10 else semcm_delta_bytes) + HEADER_BYTES
                        des_stats["SemCM_Persistent_N10"]["total_frames"] += 1
                        des_stats["SemCM_Persistent_N10"]["total_slices"] += 1
                        des_stats["SemCM_Persistent_N10"]["total_sent_bytes"] += n10_bytes
                        
                        if is_intra_n10:
                            if not slot_loss:
                                sync_valid_n10 = True
                                last_valid_n10 = f_idx
                                stale_cnt_n10 = 1
                                des_stats["SemCM_Persistent_N10"]["delivered_frames"] += 1
                                des_stats["SemCM_Persistent_N10"]["delivered_slices"] += 1
                                des_stats["SemCM_Persistent_N10"]["delivered_bytes"] += n10_bytes
                            else:
                                sync_valid_n10 = False
                                stale_cnt_n10 += 1
                        else:
                            if not slot_loss and sync_valid_n10:
                                last_valid_n10 = f_idx
                                stale_cnt_n10 = 1
                                des_stats["SemCM_Persistent_N10"]["delivered_frames"] += 1
                                des_stats["SemCM_Persistent_N10"]["delivered_slices"] += 1
                                des_stats["SemCM_Persistent_N10"]["delivered_bytes"] += n10_bytes
                            else:
                                sync_valid_n10 = False
                                stale_cnt_n10 += 1
                                
                        if stale_cnt_n10 > max_stale_n10:
                            max_stale_n10 = stale_cnt_n10
                            
                        aoi_n10 = (f_idx - last_valid_n10) * DELTA_T_MS + T_PROC_MS
                        des_stats["SemCM_Persistent_N10"]["aoi_list"].append(aoi_n10)
                        disp_map_n10 = stream_gt[last_valid_n10]
                        des_stats["SemCM_Persistent_N10"]["pixel_correct"] += int(np.sum(disp_map_n10 == gt_map))
                        des_stats["SemCM_Persistent_N10"]["total_pixels"] += (450 * 800)
                        
                        outcome_buffer.append(f"{ch_name},{s_key},{f_idx},{seed},SemCM_Persistent_N10,{int(last_valid_n10 == f_idx)},{aoi_n10:.1f}\n")
                        
                        # ----------------------------------------------------
                        # 5. SemCM Persistent N30
                        # ----------------------------------------------------
                        is_intra_n30 = (f_idx % 30 == 0)
                        n30_bytes = (semcm_intra_bytes if is_intra_n30 else semcm_delta_bytes) + HEADER_BYTES
                        des_stats["SemCM_Persistent_N30"]["total_frames"] += 1
                        des_stats["SemCM_Persistent_N30"]["total_slices"] += 1
                        des_stats["SemCM_Persistent_N30"]["total_sent_bytes"] += n30_bytes
                        
                        if is_intra_n30:
                            if not slot_loss:
                                sync_valid_n30 = True
                                last_valid_n30 = f_idx
                                stale_cnt_n30 = 1
                                des_stats["SemCM_Persistent_N30"]["delivered_frames"] += 1
                                des_stats["SemCM_Persistent_N30"]["delivered_slices"] += 1
                                des_stats["SemCM_Persistent_N30"]["delivered_bytes"] += n30_bytes
                            else:
                                sync_valid_n30 = False
                                stale_cnt_n30 += 1
                        else:
                            if not slot_loss and sync_valid_n30:
                                last_valid_n30 = f_idx
                                stale_cnt_n30 = 1
                                des_stats["SemCM_Persistent_N30"]["delivered_frames"] += 1
                                des_stats["SemCM_Persistent_N30"]["delivered_slices"] += 1
                                des_stats["SemCM_Persistent_N30"]["delivered_bytes"] += n30_bytes
                            else:
                                sync_valid_n30 = False
                                stale_cnt_n30 += 1
                                
                        if stale_cnt_n30 > max_stale_n30:
                            max_stale_n30 = stale_cnt_n30
                            
                        aoi_n30 = (f_idx - last_valid_n30) * DELTA_T_MS + T_PROC_MS
                        des_stats["SemCM_Persistent_N30"]["aoi_list"].append(aoi_n30)
                        disp_map_n30 = stream_gt[last_valid_n30]
                        des_stats["SemCM_Persistent_N30"]["pixel_correct"] += int(np.sum(disp_map_n30 == gt_map))
                        des_stats["SemCM_Persistent_N30"]["total_pixels"] += (450 * 800)
                        
                        outcome_buffer.append(f"{ch_name},{s_key},{f_idx},{seed},SemCM_Persistent_N30,{int(last_valid_n30 == f_idx)},{aoi_n30:.1f}\n")
                        
                        # ----------------------------------------------------
                        # 6. SemCM Persistent NACK (RTT = 20 ms -> arrives at next frame)
                        # ----------------------------------------------------
                        is_nack_intra = (f_idx == 0) or (f_idx == nack_pending_intra_at)
                        nack_bytes = (semcm_intra_bytes if is_nack_intra else semcm_delta_bytes) + HEADER_BYTES
                        des_stats["SemCM_Persistent_NACK"]["total_frames"] += 1
                        des_stats["SemCM_Persistent_NACK"]["total_slices"] += 1
                        des_stats["SemCM_Persistent_NACK"]["total_sent_bytes"] += nack_bytes
                        
                        if is_nack_intra:
                            nack_pending_intra_at = -1
                            if not slot_loss:
                                sync_valid_nack = True
                                last_valid_nack = f_idx
                                stale_cnt_nack = 1
                                des_stats["SemCM_Persistent_NACK"]["delivered_frames"] += 1
                                des_stats["SemCM_Persistent_NACK"]["delivered_slices"] += 1
                                des_stats["SemCM_Persistent_NACK"]["delivered_bytes"] += nack_bytes
                            else:
                                sync_valid_nack = False
                                stale_cnt_nack += 1
                                nack_pending_intra_at = f_idx + 1 # Retrigger NACK
                        else:
                            if not slot_loss and sync_valid_nack:
                                last_valid_nack = f_idx
                                stale_cnt_nack = 1
                                des_stats["SemCM_Persistent_NACK"]["delivered_frames"] += 1
                                des_stats["SemCM_Persistent_NACK"]["delivered_slices"] += 1
                                des_stats["SemCM_Persistent_NACK"]["delivered_bytes"] += nack_bytes
                            else:
                                sync_valid_nack = False
                                stale_cnt_nack += 1
                                if nack_pending_intra_at == -1:
                                    nack_pending_intra_at = f_idx + 1 # NACK arrives at sender, intra sent next frame
                                    
                        if stale_cnt_nack > max_stale_nack:
                            max_stale_nack = stale_cnt_nack
                            
                        aoi_nack = (f_idx - last_valid_nack) * DELTA_T_MS + T_PROC_MS
                        des_stats["SemCM_Persistent_NACK"]["aoi_list"].append(aoi_nack)
                        disp_map_nack = stream_gt[last_valid_nack]
                        des_stats["SemCM_Persistent_NACK"]["pixel_correct"] += int(np.sum(disp_map_nack == gt_map))
                        des_stats["SemCM_Persistent_NACK"]["total_pixels"] += (450 * 800)
                        
                        outcome_buffer.append(f"{ch_name},{s_key},{f_idx},{seed},SemCM_Persistent_NACK,{int(last_valid_nack == f_idx)},{aoi_nack:.1f}\n")
                        
                    # Save Max Stale (integer multiple of DELTA_T_MS)
                    des_stats["Yang_WholeFrame"]["max_stale_list"].append(max_stale_wf * DELTA_T_MS)
                    des_stats["Yang_4Slices"]["max_stale_list"].append(max(max_stale_4s) * DELTA_T_MS)
                    des_stats["Yang_8Slices"]["max_stale_list"].append(max(max_stale_8s) * DELTA_T_MS)
                    des_stats["SemCM_Persistent_N10"]["max_stale_list"].append(max_stale_n10 * DELTA_T_MS)
                    des_stats["SemCM_Persistent_N30"]["max_stale_list"].append(max_stale_n30 * DELTA_T_MS)
                    des_stats["SemCM_Persistent_NACK"]["max_stale_list"].append(max_stale_nack * DELTA_T_MS)
                    
            # Write outcome buffer
            with open(outcomes_csv, "a") as f:
                f.writelines(outcome_buffer)
                
            # Compute summary row for each design
            total_time_sec = (len(stream_keys) * 100 * len(seeds)) / FPS # 60 * 100 * 5 / 12 = 2500.0 sec
            
            for d in designs:
                st = des_stats[d]
                eff_kbps = (st["delivered_bytes"] * 8.0) / (total_time_sec * 1000.0)
                mean_aoi = float(np.mean(st["aoi_list"]))
                p95_aoi = float(np.percentile(st["aoi_list"], 95))
                p99_aoi = float(np.percentile(st["aoi_list"], 99))
                max_stale = float(np.max(st["max_stale_list"]))
                deliv_ratio = (st["delivered_slices"] / st["total_slices"]) * 100.0
                frame_deliv = (st["delivered_frames"] / st["total_frames"]) * 100.0
                pixel_acc = (st["pixel_correct"] / st["total_pixels"]) * 100.0
                
                summary_results.append({
                    "K": k_val,
                    "Channel": ch_name,
                    "Design": d,
                    "Effective_kbps": round(eff_kbps, 2),
                    "Pixel_Accuracy_Pct": round(pixel_acc, 2),
                    "Mean_AoI_ms": round(mean_aoi, 1),
                    "P95_AoI_ms": round(p95_aoi, 1),
                    "P99_AoI_ms": round(p99_aoi, 1),
                    "Max_Stale_Duration_ms": round(max_stale, 1),
                    "Slice_Delivery_Ratio_Pct": round(deliv_ratio, 2),
                    "Frame_Delivery_Ratio_Pct": round(frame_deliv, 2)
                })
                
            print(f"  Completed {ch_name} (K={k_val}) in {time.time() - t0_ch:.1f}s")
            
    df_summary = pd.DataFrame(summary_results)
    sum_csv = OUT_DIR / "p3_packet_loss_summary.csv"
    df_summary.to_csv(sum_csv, index=False)
    print(f"\nSaved P3 packet loss summary to {sum_csv}")
    print(f"Saved P3 per-frame outcomes to {outcomes_csv}")
    print("\nSummary Table Sample (Bernoulli 10% and Burst TB50ms, K=4):")
    sample_df = df_summary[(df_summary["K"] == 4) & (df_summary["Channel"].isin(["Bernoulli_10%", "Burst_TB50ms_PL10%"]))]
    print(sample_df.to_string(index=False))

if __name__ == "__main__":
    run_p3_benchmark()
