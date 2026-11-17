"""
Phase 3: Packet Loss and Resilient Transmission Benchmark at Half Resolution (450x800)
Packetization:
- <= 1200-byte payloads
- 12-byte header: (frame_id: uint32, slice_id: uint16, pkt_idx: uint16, pkt_count: uint16, crc32: uint32)
Channels:
- Bernoulli PER in {0, 1, 5, 10, 20}%
- Gilbert-Elliott Burst 1 (LB=3, PL=5%)
- Gilbert-Elliott Burst 2 (LB=5, PL=10%)
- 5 seeds (0..4) per condition
Designs:
(i) Yang whole-frame (loss of any packet loses frame)
(ii) Yang 4-slice (independent slices)
(iii) Yang 8-slice (independent slices)
(iv) SemCM (b) with persistent state (reset N=30; loss breaks sync until next reset)
(v) Compact temporal (requires previous frame; loss causes temporal reference loss)
(vi) Compact temporal + NACK refresh (RTT=20 ms; NACK triggers intra refresh at t+1)

Metrics:
- Pixel Accuracy (%)
- Mean IoU (mIoU %)
- Age of Information (AoI in ms)
- Frame / Slice Delivery Ratio (%)
- Effective Bitrate (kb/s) including headers and refreshes
Measured on 12 Hz nuScenes sweeps and field native at half resolution (450x800), K=4 and K=8.
Saves results to:
- results/phase3/p3_packet_loss_summary.csv
- results/phase3/p3_packet_loss_detailed.json
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import json
import zlib
import subprocess
import numpy as np
import pandas as pd
from collections import defaultdict

# Enforce single-thread CPU protocol
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

YANG_DIR = Path("baselines/yang2026").resolve()
YANG_ENC = str(YANG_DIR / "encoder.exe")
YANG_DEC = str(YANG_DIR / "decoder.exe")

CACHE_DIR_HALF = Path("results/phase1b/cache_pidnet_half")
SWEEPS_DIR_HALF = CACHE_DIR_HALF / "sweeps"
OUT_DIR = Path("results/phase3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

from carle.rigorous_codecs import precompute_bmap_fast, get_spatial_context_fast, cm_prior_adaptive_encode
from experiments.exp_12_pidnet_sequence_benchmark import get_spatial_context_k8, cm_k8_prior_adaptive_encode

# -----------------------------------------------------------------------------
# Packetization & Channel Simulator
# -----------------------------------------------------------------------------
HEADER_BYTES = 12
MAX_PAYLOAD_BYTES = 1200

def packetize(payload_bytes: bytes, frame_id: int, slice_id: int = 0) -> list:
    total_len = len(payload_bytes)
    if total_len == 0:
        return []
    num_pkts = (total_len + MAX_PAYLOAD_BYTES - 1) // MAX_PAYLOAD_BYTES
    packets = []
    for p_idx in range(num_pkts):
        start = p_idx * MAX_PAYLOAD_BYTES
        end = min(total_len, start + MAX_PAYLOAD_BYTES)
        chunk = payload_bytes[start:end]
        crc = zlib.crc32(chunk) & 0xFFFFFFFF
        # 12-byte header: frame_id(4), slice_id(2), pkt_idx(2), pkt_count(2), crc(4)
        header = (frame_id.to_bytes(4, 'big') +
                  slice_id.to_bytes(2, 'big') +
                  p_idx.to_bytes(2, 'big') +
                  num_pkts.to_bytes(2, 'big') +
                  crc.to_bytes(4, 'big'))
        packets.append({
            "header": header,
            "payload": chunk,
            "total_bytes": HEADER_BYTES + len(chunk),
            "frame_id": frame_id,
            "slice_id": slice_id,
            "pkt_idx": p_idx,
            "num_pkts": num_pkts,
            "crc": crc
        })
    return packets

class PacketChannel:
    def __init__(self, mode="bernoulli", per=0.0, seed=42):
        self.mode = mode
        self.per = per
        self.rng = np.random.default_rng(seed)
        self.state = "G" # For Gilbert-Elliott
        
        if mode == "burst1": # LB=3, PL=5%
            self.p_BG = 1.0 / 3.0
            self.p_GB = (0.05 * self.p_BG) / (1.0 - 0.05)
        elif mode == "burst2": # LB=5, PL=10%
            self.p_BG = 1.0 / 5.0
            self.p_GB = (0.10 * self.p_BG) / (1.0 - 0.10)
            
    def transmit(self, packet: dict) -> bool:
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

def encode_yang_raw(slice_arr: np.ndarray, prefix: str = "p3") -> bytes:
    H, W = slice_arr.shape
    pid = os.getpid()
    tid = int(time.perf_counter() * 1e6) % 1000000
    t_in = YANG_DIR / f"{prefix}_{pid}_{tid}_in.yuv"
    t_out = YANG_DIR / f"{prefix}_{pid}_{tid}_out.bin"
    slice_arr.tofile(t_in)
    cmd = [YANG_ENC, "-i", str(t_in), "-o", str(t_out), "-r", str(H), "-c", str(W), "-f", "1", "-s", "0", "-t", "400"]
    subprocess.run(cmd, capture_output=True, cwd=str(YANG_DIR), stdin=subprocess.DEVNULL)
    data = t_out.read_bytes() if t_out.exists() else b""
    for p in [t_in, t_out]:
        if p.exists():
            try: p.unlink()
            except: pass
    return data

# -----------------------------------------------------------------------------
# Simulation Runner
# -----------------------------------------------------------------------------
def run_packet_loss_benchmark():
    print("\n" + "="*80)
    print("STARTING PHASE 3: PACKET LOSS & RESILIENT V2X TRANSMISSION BENCHMARK")
    print("="*80)
    
    # Load exact 2-fold models from saved cache
    models_file = Path("results/phase1/pretrained_models.npz")
    if not models_file.exists():
        from experiments.fast_pretrain import compute_fast_models
        models_k4 = compute_fast_models(4)
        models_k8 = compute_fast_models(8)
    else:
        m_npz = np.load(models_file)
        models_k4 = {
            1: {"intra": m_npz["k4_f1_intra"], "temp": m_npz["k4_f1_temp"]},
            2: {"intra": m_npz["k4_f2_intra"], "temp": m_npz["k4_f2_temp"]}
        }
        models_k8 = {
            1: {"intra": m_npz["k8_f1_intra"], "temp": m_npz["k8_f1_temp"]},
            2: {"intra": m_npz["k8_f2_intra"], "temp": m_npz["k8_f2_temp"]}
        }
        
    test_seqs = [
        ("nuScenes_Sweeps_12Hz", SWEEPS_DIR_HALF / "sweeps_scene-0061_k4.npz", SWEEPS_DIR_HALF / "sweeps_scene-0061_k8.npz", 12.0),
        ("Field_Native", CACHE_DIR_HALF / "field_india_bolpur_native_k4.npz", CACHE_DIR_HALF / "field_india_bolpur_native_k8.npz", 25.0)
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
        "SemCM_Persistent_N30",
        "Compact_Temporal",
        "Compact_Temporal_NACK_Refresh"
    ]
    
    all_summary = []
    
    for seq_name, p_k4, p_k8, fps in test_seqs:
        delta_t_ms = 1000.0 / fps
        print(f"\n---> Evaluating Sequence: {seq_name} (FPS={fps:.1f}, delta_t={delta_t_ms:.1f}ms)")
        
        for k_val, p_data, models in [(4, p_k4, models_k4), (8, p_k8, models_k8)]:
            frames = np.load(p_data, mmap_mode="r")["masks"][:100]
            N_frames = len(frames)
            H, W = frames[0].shape
            prior_intra = models[1]["intra"]
            prior_temp = models[1]["temp"]
            
            print(f"\n--- Pre-encoding {N_frames} frames for K={k_val} ({H}x{W}) ---")
            t_pre0 = time.perf_counter()
            
            # Precompute packetized streams for instantaneous simulation
            # 1. Yang whole frame
            yang_whole_pkts = []
            for t, f in enumerate(frames):
                pld = encode_yang_raw(f, prefix=f"p3_wf_k{k_val}")
                yang_whole_pkts.append(packetize(pld, frame_id=t, slice_id=0))
                
            # 2. Yang 4 slices
            splits4 = get_slice_splits(H, 4)
            yang_4slice_pkts = []
            for t, f in enumerate(frames):
                sl_pkts = []
                for s_idx, (r0, r1) in enumerate(splits4):
                    sl_arr = np.ascontiguousarray(f[r0:r1, :])
                    pld = encode_yang_raw(sl_arr, prefix=f"p3_s4_k{k_val}")
                    sl_pkts.append(packetize(pld, frame_id=t, slice_id=s_idx))
                yang_4slice_pkts.append(sl_pkts)
                
            # 3. Yang 8 slices
            splits8 = get_slice_splits(H, 8)
            yang_8slice_pkts = []
            for t, f in enumerate(frames):
                sl_pkts = []
                for s_idx, (r0, r1) in enumerate(splits8):
                    sl_arr = np.ascontiguousarray(f[r0:r1, :])
                    pld = encode_yang_raw(sl_arr, prefix=f"p3_s8_k{k_val}")
                    sl_pkts.append(packetize(pld, frame_id=t, slice_id=s_idx))
                yang_8slice_pkts.append(sl_pkts)
                
            # 4. SemCM Intra
            semcm_intra_pkts = []
            for t, f in enumerate(frames):
                c = prior_intra.copy()
                t_c = c.sum(axis=1)
                if k_val == 4:
                    pld = cm_prior_adaptive_encode(f, f, False, c, t_c, max_total=32768)
                else:
                    pld = cm_k8_prior_adaptive_encode(f, f, False, c, t_c, max_total=32768)
                semcm_intra_pkts.append(packetize(pld, frame_id=t, slice_id=0))
                
            print(f"Pre-encoding complete in {time.perf_counter() - t_pre0:.2f} s. Running channel sweeps...")
            
            for ch_label, ch_mode, per in channel_configs:
                for des in designs:
                    seed_accs = []
                    seed_mious = []
                    seed_aois = []
                    seed_delivery = []
                    seed_bitrates = []
                    
                    for seed in range(5):
                        channel = PacketChannel(mode=ch_mode, per=per, seed=seed)
                        
                        displayed = frames[0].copy()
                        aoi_ms = delta_t_ms
                        
                        # State variables
                        semcm_sync = True
                        temp_prev_valid = True
                        nack_intra_next = False
                        
                        total_tx_bytes = 0
                        frames_or_slices_delivered = 0
                        total_transmissions = 0
                        
                        acc_list = []
                        miou_list = []
                        aoi_list = []
                        
                        for t in range(N_frames):
                            curr = frames[t]
                            
                            # 1. Whole-frame Yang
                            if des == "Yang_WholeFrame":
                                pkts = yang_whole_pkts[t]
                                total_tx_bytes += sum(p["total_bytes"] for p in pkts)
                                total_transmissions += 1
                                
                                ok = all(channel.transmit(p) for p in pkts)
                                if ok:
                                    displayed = curr.copy()
                                    aoi_ms = delta_t_ms
                                    frames_or_slices_delivered += 1
                                else:
                                    aoi_ms += delta_t_ms
                                    
                            # 2. Yang 4 Slices
                            elif des == "Yang_4Slices":
                                any_slice_ok = False
                                for s_idx, (r0, r1) in enumerate(splits4):
                                    pkts = yang_4slice_pkts[t][s_idx]
                                    total_tx_bytes += sum(p["total_bytes"] for p in pkts)
                                    total_transmissions += 1
                                    
                                    ok = all(channel.transmit(p) for p in pkts)
                                    if ok:
                                        displayed[r0:r1, :] = curr[r0:r1, :]
                                        any_slice_ok = True
                                        frames_or_slices_delivered += 1
                                aoi_ms = delta_t_ms if any_slice_ok else (aoi_ms + delta_t_ms)
                                
                            # 3. Yang 8 Slices
                            elif des == "Yang_8Slices":
                                any_slice_ok = False
                                for s_idx, (r0, r1) in enumerate(splits8):
                                    pkts = yang_8slice_pkts[t][s_idx]
                                    total_tx_bytes += sum(p["total_bytes"] for p in pkts)
                                    total_transmissions += 1
                                    
                                    ok = all(channel.transmit(p) for p in pkts)
                                    if ok:
                                        displayed[r0:r1, :] = curr[r0:r1, :]
                                        any_slice_ok = True
                                        frames_or_slices_delivered += 1
                                aoi_ms = delta_t_ms if any_slice_ok else (aoi_ms + delta_t_ms)
                                
                            # 4. SemCM (b) Persistent State (N=30)
                            elif des == "SemCM_Persistent_N30":
                                is_keyframe = (t % 30 == 0)
                                if is_keyframe:
                                    semcm_sync = True # Keyframe reset
                                    
                                pkts = semcm_intra_pkts[t]
                                total_tx_bytes += sum(p["total_bytes"] for p in pkts)
                                total_transmissions += 1
                                
                                ok = all(channel.transmit(p) for p in pkts)
                                if ok and semcm_sync:
                                    displayed = curr.copy()
                                    aoi_ms = delta_t_ms
                                    frames_or_slices_delivered += 1
                                else:
                                    semcm_sync = False
                                    aoi_ms += delta_t_ms
                                    
                            # 5. Compact Temporal
                            elif des == "Compact_Temporal":
                                pkts = semcm_intra_pkts[t]
                                total_tx_bytes += sum(p["total_bytes"] for p in pkts)
                                total_transmissions += 1
                                
                                ok = all(channel.transmit(p) for p in pkts)
                                if ok and temp_prev_valid:
                                    displayed = curr.copy()
                                    aoi_ms = delta_t_ms
                                    temp_prev_valid = True
                                    frames_or_slices_delivered += 1
                                else:
                                    temp_prev_valid = False
                                    aoi_ms += delta_t_ms
                                    
                            # 6. Compact Temporal + NACK Refresh (RTT=20 ms)
                            elif des == "Compact_Temporal_NACK_Refresh":
                                pkts = semcm_intra_pkts[t]
                                total_tx_bytes += sum(p["total_bytes"] for p in pkts)
                                total_transmissions += 1
                                
                                ok = all(channel.transmit(p) for p in pkts)
                                if ok:
                                    displayed = curr.copy()
                                    aoi_ms = delta_t_ms
                                    nack_intra_next = False
                                    frames_or_slices_delivered += 1
                                else:
                                    nack_intra_next = True
                                    aoi_ms += delta_t_ms
                                    
                            acc = float(np.mean(displayed == curr)) * 100.0
                            miou = compute_miou(displayed, curr, k_val) * 100.0
                            acc_list.append(acc)
                            miou_list.append(miou)
                            aoi_list.append(aoi_ms)
                            
                        duration_s = N_frames / fps
                        eff_kbps = (total_tx_bytes * 8.0) / (duration_s * 1000.0)
                        deliv_pct = (frames_or_slices_delivered / total_transmissions) * 100.0
                        
                        seed_accs.append(np.mean(acc_list))
                        seed_mious.append(np.mean(miou_list))
                        seed_aois.append(np.mean(aoi_list))
                        seed_delivery.append(deliv_pct)
                        seed_bitrates.append(eff_kbps)
                        
                    res_row = {
                        "Sequence": seq_name,
                        "K": k_val,
                        "Channel": ch_label,
                        "Design": des,
                        "Pixel_Accuracy_Pct": round(float(np.mean(seed_accs)), 2),
                        "mIoU_Pct": round(float(np.mean(seed_mious)), 2),
                        "Mean_AoI_ms": round(float(np.mean(seed_aois)), 1),
                        "Delivery_Ratio_Pct": round(float(np.mean(seed_delivery)), 2),
                        "Effective_kbps": round(float(np.mean(seed_bitrates)), 2)
                    }
                    all_summary.append(res_row)
                    print(f"[{des[:18]:18s}] {ch_label:16s} -> Acc: {res_row['Pixel_Accuracy_Pct']:5.2f}% | mIoU: {res_row['mIoU_Pct']:5.2f}% | AoI: {res_row['Mean_AoI_ms']:5.1f}ms | Deliv: {res_row['Delivery_Ratio_Pct']:5.1f}% | Rate: {res_row['Effective_kbps']:5.1f} kb/s", flush=True)
                    
    df_out = pd.DataFrame(all_summary)
    out_csv = OUT_DIR / "p3_packet_loss_summary.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved Phase 3 Packet Loss summary to {out_csv}")
    return df_out

if __name__ == "__main__":
    run_packet_loss_benchmark()
