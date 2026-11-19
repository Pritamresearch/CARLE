"""
Phase 3 (Corrected & Reconciled): Packet Loss and Resilient Transmission Benchmark
Evaluates V2X transmission over 1,200-byte MTU packetized channels:
- 12-byte header: (frame_id: uint32, slice_id: uint16, pkt_idx: uint16, pkt_count: uint16, crc32: uint32)
- Channels:
  * Bernoulli PER in {0, 1, 5, 10, 20}%
  * Gilbert-Elliott Burst 1 (LB=3, PL=5%)
  * Gilbert-Elliott Burst 2 (LB=5, PL=10%)
  * 5 random seeds (0..4) per condition

Evaluated Transmission Schemes:
1. Yang_WholeFrame (loss of any packet drops entire frame)
2. Yang_4Slices (independent horizontal slices S=4; losses isolated to 1/4 frame)
3. Yang_8Slices (independent horizontal slices S=8; losses isolated to 1/8 frame)
4. SemCM_Persistent_N10 (stream-persistent state with periodic intra refresh every N=10 frames)
5. SemCM_Persistent_N30 (stream-persistent state with periodic intra refresh every N=30 frames)
6. Compact_Temporal_N10 (compact temporal context with periodic intra refresh every N=10 frames)
7. Compact_Temporal_N30 (compact temporal context with periodic intra refresh every N=30 frames)
8. Compact_Temporal_NACK_Refresh (receiver NACK upon packet loss triggers intra refresh after RTT=20 ms)

Metrics (Strict per-pixel tracking):
- Pixel Accuracy (%)
- Mean IoU (mIoU %)
- Mean AoI (ms): time since displayed label's source frame was captured, including processing latency, averaged over pixels and frames
- p95 AoI (ms): 95th percentile over all pixels and frames
- p99 AoI (ms): 99th percentile over all pixels and frames
- Max Consecutive Stale Duration (ms): max contiguous duration any pixel region remained unrefreshed
- Frame / Slice Delivery Ratio (%)
- Effective Bitrate (kb/s): reconciled with P1b dataset-wide bits/frame x 12 Hz + 12-byte headers + intra refreshes

Evaluated on:
- nuScenes Sweeps 12 Hz (half resolution 450x800, K=4 and K=8)
- Field Native 25 Hz (half resolution 450x800, K=4 and K=8)

Saves results to:
- results/phase3/p3_packet_loss_summary.csv
- results/phase3/p3_packet_loss_detailed.json
"""

import sys
import os
import time
import json
import zlib
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_DIR = Path("results/phase3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR_HALF = Path("results/phase1b/cache_pidnet_half")
SWEEPS_DIR_HALF = CACHE_DIR_HALF / "sweeps"

HEADER_BYTES = 12
MAX_PAYLOAD_BYTES = 1200

class PacketChannel:
    def __init__(self, mode="bernoulli", per=0.0, seed=42):
        self.mode = mode
        self.per = per
        self.rng = np.random.default_rng(seed)
        self.state = "G"
        
        if mode == "burst1": # LB=3, PL=5%
            self.p_BG = 1.0 / 3.0
            self.p_GB = (0.05 * self.p_BG) / (1.0 - 0.05)
        elif mode == "burst2": # LB=5, PL=10%
            self.p_BG = 1.0 / 5.0
            self.p_GB = (0.10 * self.p_BG) / (1.0 - 0.10)
            
    def transmit(self) -> bool:
        if self.mode == "bernoulli":
            return bool(self.rng.random() >= self.per)
        else: # Gilbert-Elliott
            if self.state == "G":
                if self.rng.random() < self.p_GB:
                    self.state = "B"
            else:
                if self.rng.random() < self.p_BG:
                    self.state = "G"
            return bool(self.state == "G")

def compute_miou(pred: np.ndarray, truth: np.ndarray, num_classes: int) -> float:
    ious = []
    for c in range(1, num_classes + 1):
        p_c = (pred == c)
        t_c = (truth == c)
        intersection = np.sum(p_c & t_c)
        union = np.sum(p_c | t_c)
        if union > 0:
            ious.append(intersection / union)
    return float(np.mean(ious)) if len(ious) > 0 else 1.0

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

def run_phase3_simulation():
    print("\n" + "="*80)
    print("STARTING PHASE 3 (CORRECTED): RESILIENT TRANSMISSION & PER-PIXEL AOI BENCHMARK")
    print("="*80)
    
    test_datasets = [
        ("nuScenes_Sweeps_12Hz", SWEEPS_DIR_HALF / "sweeps_scene-0061_k4.npz", SWEEPS_DIR_HALF / "sweeps_scene-0061_k8.npz", 12.0),
        ("Field_Native_25Hz", CACHE_DIR_HALF / "field_india_bolpur_native_k4.npz", CACHE_DIR_HALF / "field_india_bolpur_native_k8.npz", 25.0)
    ]
    
    channel_configs = [
        ("Bernoulli_0%", "bernoulli", 0.00),
        ("Bernoulli_1%", "bernoulli", 0.01),
        ("Bernoulli_5%", "bernoulli", 0.05),
        ("Bernoulli_10%", "bernoulli", 0.10),
        ("Bernoulli_20%", "bernoulli", 0.20),
        ("Burst1_PL5%_LB3", "burst1", 0.05),
        ("Burst2_PL10%_LB5", "burst2", 0.10)
    ]
    
    designs = [
        "Yang_WholeFrame",
        "Yang_4Slices",
        "Yang_8Slices",
        "SemCM_Persistent_N10",
        "SemCM_Persistent_N30",
        "Compact_Temporal_N10",
        "Compact_Temporal_N30",
        "Compact_Temporal_NACK_Refresh"
    ]
    
    # Reconciled baseline bitrates from P1b (dataset-wide bits/frame x FPS + headers)
    # nuScenes Sweeps 12 Hz:
    # Yang WF: 2648.7 bpf = 331.1 B. Header: 12 B -> (331.1+12)*8*12/1000 = 32.94 kb/s
    # Yang 4S: 3172.6 bpf = 396.6 B. Header: 4*12=48 B -> (396.6+48)*8*12/1000 = 42.68 kb/s
    # Yang 8S: 3479.9 bpf = 435.0 B. Header: 8*12=96 B -> (435.0+96)*8*12/1000 = 50.98 kb/s
    # SemCM Intra: 4554.1 bpf = 569.3 B -> 581.3 B = 55.80 kb/s
    # Compact Temp delta: 4140 bpf = 517.5 B -> 529.5 B = 50.83 kb/s
    # SemCM Persist delta: ~4180 bpf -> 51.22 kb/s
    bitrate_specs = {
        ("nuScenes_Sweeps_12Hz", 4): {
            "Yang_WholeFrame": 32.94,
            "Yang_4Slices": 42.68,
            "Yang_8Slices": 50.98,
            "SemCM_Persistent_N10": 51.33,
            "SemCM_Persistent_N30": 50.99,
            "Compact_Temporal_N10": 51.33,
            "Compact_Temporal_N30": 50.99,
            "Compact_Temporal_NACK_base": 50.83,
            "Compact_Temporal_NACK_extra_per_loss": 4.97 # (Intra - Delta) * 12 / 1000
        },
        ("nuScenes_Sweeps_12Hz", 8): {
            "Yang_WholeFrame": 66.44,
            "Yang_4Slices": 71.86,
            "Yang_8Slices": 81.73,
            "SemCM_Persistent_N10": 121.89,
            "SemCM_Persistent_N30": 121.29,
            "Compact_Temporal_N10": 121.89,
            "Compact_Temporal_N30": 121.29,
            "Compact_Temporal_NACK_base": 120.99,
            "Compact_Temporal_NACK_extra_per_loss": 9.00
        },
        ("Field_Native_25Hz", 4): {
            "Yang_WholeFrame": 75.10,
            "Yang_4Slices": 97.28,
            "Yang_8Slices": 116.20,
            "SemCM_Persistent_N10": 117.00,
            "SemCM_Persistent_N30": 116.22,
            "Compact_Temporal_N10": 117.00,
            "Compact_Temporal_N30": 116.22,
            "Compact_Temporal_NACK_base": 115.85,
            "Compact_Temporal_NACK_extra_per_loss": 11.35
        },
        ("Field_Native_25Hz", 8): {
            "Yang_WholeFrame": 125.52,
            "Yang_4Slices": 135.78,
            "Yang_8Slices": 154.43,
            "SemCM_Persistent_N10": 230.12,
            "SemCM_Persistent_N30": 228.98,
            "Compact_Temporal_N10": 230.12,
            "Compact_Temporal_N30": 228.98,
            "Compact_Temporal_NACK_base": 228.42,
            "Compact_Temporal_NACK_extra_per_loss": 17.00
        }
    }
    
    # End-to-end processing latency T_proc = 103.4 ms (operating point latency)
    t_proc_specs = {4: 103.4, 8: 103.4}
    
    all_summary = []
    
    for seq_name, p_k4, p_k8, fps in test_datasets:
        delta_t_ms = 1000.0 / fps
        print(f"\n---> Evaluating Dataset: {seq_name} (FPS={fps:.1f}, delta_t={delta_t_ms:.2f} ms)")
        
        for k_val, p_data in [(4, p_k4), (8, p_k8)]:
            frames = np.load(p_data, mmap_mode="r")["masks"][:100]
            N_frames = len(frames)
            H, W = frames[0].shape
            splits4 = get_slice_splits(H, 4)
            splits8 = get_slice_splits(H, 8)
            T_proc = t_proc_specs[k_val]
            rates = bitrate_specs[(seq_name, k_val)]
            
            # Packets per frame/slice (all <= 1200 bytes fit in 1 packet at S=4 and S=8)
            # At K=4 whole-frame fits in 1 packet. At K=8 whole-frame requires 1 packet (P95=1179 B).
            for ch_label, ch_mode, per in channel_configs:
                for des in designs:
                    seed_accs = []
                    seed_mious = []
                    seed_mean_aois = []
                    seed_p95_aois = []
                    seed_p99_aois = []
                    seed_max_stales = []
                    seed_delivery = []
                    seed_bitrates = []
                    
                    for seed in range(5):
                        channel = PacketChannel(mode=ch_mode, per=per, seed=seed)
                        
                        # Display buffer: (H, W) array of labels
                        displayed = frames[0].copy()
                        
                        # Per-pixel source frame index: tracks which source frame t is currently displayed at (r, c)
                        src_frame_idx = np.zeros((H, W), dtype=np.int32)
                        
                        # Max consecutive stale duration per pixel
                        max_stale_per_pixel = np.zeros((H, W), dtype=np.float32)
                        
                        # State variables for temporal codecs
                        semcm_sync = True
                        temp_sync = True
                        nack_sent = False
                        
                        total_transmissions = 0
                        delivered_transmissions = 0
                        nack_intra_count = 0
                        
                        pixel_acc_list = []
                        miou_list = []
                        aoi_pixel_samples = []
                        
                        for t in range(N_frames):
                            curr = frames[t]
                            
                            # 1. Yang Whole Frame (Independent, 1 packet)
                            if des == "Yang_WholeFrame":
                                total_transmissions += 1
                                pkt_ok = channel.transmit()
                                if pkt_ok:
                                    displayed = curr.copy()
                                    src_frame_idx[:, :] = t
                                    delivered_transmissions += 1
                                    
                            # 2. Yang 4 Slices (Independent, 4 packets)
                            elif des == "Yang_4Slices":
                                for s_idx, (r0, r1) in enumerate(splits4):
                                    total_transmissions += 1
                                    pkt_ok = channel.transmit()
                                    if pkt_ok:
                                        displayed[r0:r1, :] = curr[r0:r1, :]
                                        src_frame_idx[r0:r1, :] = t
                                        delivered_transmissions += 1
                                        
                            # 3. Yang 8 Slices (Independent, 8 packets)
                            elif des == "Yang_8Slices":
                                for s_idx, (r0, r1) in enumerate(splits8):
                                    total_transmissions += 1
                                    pkt_ok = channel.transmit()
                                    if pkt_ok:
                                        displayed[r0:r1, :] = curr[r0:r1, :]
                                        src_frame_idx[r0:r1, :] = t
                                        delivered_transmissions += 1
                                        
                            # 4. SemCM Persistent N=10 (Intra keyframe every 10 frames)
                            elif des == "SemCM_Persistent_N10":
                                is_keyframe = (t % 10 == 0)
                                if is_keyframe:
                                    semcm_sync = True
                                total_transmissions += 1
                                pkt_ok = channel.transmit()
                                if pkt_ok and semcm_sync:
                                    displayed = curr.copy()
                                    src_frame_idx[:, :] = t
                                    delivered_transmissions += 1
                                else:
                                    semcm_sync = False
                                    
                            # 5. SemCM Persistent N=30 (Intra keyframe every 30 frames)
                            elif des == "SemCM_Persistent_N30":
                                is_keyframe = (t % 30 == 0)
                                if is_keyframe:
                                    semcm_sync = True
                                total_transmissions += 1
                                pkt_ok = channel.transmit()
                                if pkt_ok and semcm_sync:
                                    displayed = curr.copy()
                                    src_frame_idx[:, :] = t
                                    delivered_transmissions += 1
                                else:
                                    semcm_sync = False
                                    
                            # 6. Compact Temporal N=10 (Intra keyframe every 10 frames)
                            elif des == "Compact_Temporal_N10":
                                is_keyframe = (t % 10 == 0)
                                if is_keyframe:
                                    temp_sync = True
                                total_transmissions += 1
                                pkt_ok = channel.transmit()
                                if pkt_ok and temp_sync:
                                    displayed = curr.copy()
                                    src_frame_idx[:, :] = t
                                    delivered_transmissions += 1
                                else:
                                    temp_sync = False
                                    
                            # 7. Compact Temporal N=30 (Intra keyframe every 30 frames)
                            elif des == "Compact_Temporal_N30":
                                is_keyframe = (t % 30 == 0)
                                if is_keyframe:
                                    temp_sync = True
                                total_transmissions += 1
                                pkt_ok = channel.transmit()
                                if pkt_ok and temp_sync:
                                    displayed = curr.copy()
                                    src_frame_idx[:, :] = t
                                    delivered_transmissions += 1
                                else:
                                    temp_sync = False
                                    
                            # 8. Compact Temporal + NACK Refresh (RTT = 20 ms)
                            elif des == "Compact_Temporal_NACK_Refresh":
                                is_intra = (t == 0) or nack_sent
                                if is_intra:
                                    nack_intra_count += 1
                                total_transmissions += 1
                                pkt_ok = channel.transmit()
                                if pkt_ok:
                                    displayed = curr.copy()
                                    src_frame_idx[:, :] = t
                                    delivered_transmissions += 1
                                    nack_sent = False # recovered sync
                                else:
                                    # Loss triggers NACK; sender receives after RTT=20 ms (< delta_t) and sends intra at t+1
                                    nack_sent = True
                                    
                            # Per-pixel AoI calculation at instant t
                            # AoI(r, c) = (t - src_frame_idx(r, c)) * delta_t + T_proc
                            stale_steps = t - src_frame_idx
                            stale_duration_ms = stale_steps * delta_t_ms
                            max_stale_per_pixel = np.maximum(max_stale_per_pixel, stale_duration_ms)
                            
                            aoi_frame = stale_duration_ms + T_proc
                            
                            # Metrics evaluation
                            acc = float(np.mean(displayed == curr)) * 100.0
                            miou = compute_miou(displayed, curr, k_val) * 100.0
                            pixel_acc_list.append(acc)
                            miou_list.append(miou)
                            
                            # Subsample AoI pixels for precise quantile calculation
                            aoi_pixel_samples.append(aoi_frame[::10, ::10].ravel())
                            
                        # End of sequence for this seed
                        all_aoi_vals = np.concatenate(aoi_pixel_samples)
                        mean_aoi = float(np.mean(all_aoi_vals))
                        p95_aoi = float(np.percentile(all_aoi_vals, 95))
                        p99_aoi = float(np.percentile(all_aoi_vals, 99))
                        max_stale = float(np.max(max_stale_per_pixel))
                        deliv_pct = (delivered_transmissions / total_transmissions) * 100.0
                        
                        # Effective bitrate calculation
                        if des == "Compact_Temporal_NACK_Refresh":
                            extra_rate = (nack_intra_count / N_frames) * rates["Compact_Temporal_NACK_extra_per_loss"]
                            eff_kbps = rates["Compact_Temporal_NACK_base"] + extra_rate
                        else:
                            eff_kbps = rates[des]
                            
                        seed_accs.append(np.mean(pixel_acc_list))
                        seed_mious.append(np.mean(miou_list))
                        seed_mean_aois.append(mean_aoi)
                        seed_p95_aois.append(p95_aoi)
                        seed_p99_aois.append(p99_aoi)
                        seed_max_stales.append(max_stale)
                        seed_delivery.append(deliv_pct)
                        seed_bitrates.append(eff_kbps)
                        
                    res_row = {
                        "Dataset": seq_name,
                        "K": k_val,
                        "Channel": ch_label,
                        "Design": des,
                        "Pixel_Accuracy_Pct": round(float(np.mean(seed_accs)), 2),
                        "mIoU_Pct": round(float(np.mean(seed_mious)), 2),
                        "Mean_AoI_ms": round(float(np.mean(seed_mean_aois)), 1),
                        "p95_AoI_ms": round(float(np.mean(seed_p95_aois)), 1),
                        "p99_AoI_ms": round(float(np.mean(seed_p99_aois)), 1),
                        "Max_Stale_Duration_ms": round(float(np.mean(seed_max_stales)), 1),
                        "Delivery_Ratio_Pct": round(float(np.mean(seed_delivery)), 2),
                        "Effective_kbps": round(float(np.mean(seed_bitrates)), 2)
                    }
                    all_summary.append(res_row)
                    print(f"[{des[:20]:20s}] {ch_label:16s} -> Acc: {res_row['Pixel_Accuracy_Pct']:5.2f}% | mIoU: {res_row['mIoU_Pct']:5.2f}% | AoI Mean: {res_row['Mean_AoI_ms']:5.1f}ms, p95: {res_row['p95_AoI_ms']:5.1f}ms | StaleMax: {res_row['Max_Stale_Duration_ms']:5.1f}ms | Rate: {res_row['Effective_kbps']:5.1f} kb/s", flush=True)

    df_res = pd.DataFrame(all_summary)
    out_csv = OUT_DIR / "p3_packet_loss_summary.csv"
    out_json = OUT_DIR / "p3_packet_loss_detailed.json"
    df_res.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(all_summary, f, indent=2)
        
    print(f"\nSaved Corrected Phase 3 Results to {out_csv} and {out_json}")
    return df_res

if __name__ == "__main__":
    run_phase3_simulation()
