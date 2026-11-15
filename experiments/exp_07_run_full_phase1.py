"""
Full Phase 1 Benchmark Runner.

Runs all codecs across:
- All 2,424 nuScenes keyframes (2 Hz)
- All 14,008 nuScenes sweeps (~12 Hz, 10 scenes, 6 cameras)
- All 5,484 frames of all 6 field clips (native fps and 10 Hz subsampled)

Codecs:
- FullMap_PNG9, FullMap_Zstd3, FullMap_Zstd19
- DiffMap_PNG9, DiffMap_Zstd3, DiffMap_Zstd19
- CARLE_Canonical_Numba (7-byte binary packets)
- CARLE_v2_rANS_Numba (gap-coding + rANS)
- CARLE_v2_IP (refresh N in {10, 30, inf}) with intra/inter tracking
- CM_intra and CM_temporal (Numba context-model coders)
- XOR_Legacy

Key features:
- Bootstrap CIs clustered by scene/clip (not by frame)
- Tracking of CARLE-v2-IP intra vs inter decision percentages and intra coder identity
- Strictly CPU single-threaded execution for timing parity
- Bit-exact lossless verification on every frame
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(line_buffering=True)

import time
import json
import numpy as np
import pandas as pd
import torch

# Pin CPU to single thread
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from carle.codecs import FullMapPNG, FullMapZstd, DiffMapPNG, DiffMapZstd, XORLegacy
from carle.fast_codecs import (
    CARLECanonicalNumba,
    CARLEv2Numba,
    ContextModelCoder,
    extract_diff_runs_numba,
    apply_diff_runs_numba,
)

CACHE_DIR = Path("results/phase1/cache")
RESULTS_DIR = Path("results/phase1")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class CARLEv2IPNumba:
    """CARLE-v2 with dynamic Intra/Inter switching and periodic keyframe refresh."""
    def __init__(self, refresh_N: int = 30, intra_level: int = 3):
        self.refresh_N = refresh_N
        self.intra_coder = FullMapZstd(level=intra_level)
        self.intra_codec_name = f"FullMapZstd(level={intra_level})"
        self.v2_coder = CARLEv2Numba()
        self.frame_idx = 0
        self.prev_mask = None
        self.name = f"CARLE_v2_IP_{refresh_N if refresh_N != float('inf') else 'inf'}"
        
        # Statistics
        self.total_frames = 0
        self.intra_count = 0
        self.inter_count = 0
        
    def reset(self):
        self.frame_idx = 0
        self.prev_mask = None
        self.v2_coder.reset()
        
    def encode(self, mask: np.ndarray) -> bytes:
        self.total_frames += 1
        H, W = mask.shape
        force_intra = (self.frame_idx % self.refresh_N == 0) or (self.prev_mask is None)
        
        # Always try intra
        intra_bytes = self.intra_coder.encode(mask)
        
        if force_intra:
            self.prev_mask = mask.copy()
            self.v2_coder.prev_mask = mask.copy()
            self.frame_idx += 1
            self.intra_count += 1
            return b"I" + intra_bytes
            
        # Try inter
        inter_bytes = self.v2_coder.encode(mask)
        
        # Choose smaller
        if len(inter_bytes) < len(intra_bytes):
            self.frame_idx += 1
            self.prev_mask = mask.copy()
            self.inter_count += 1
            return b"P" + inter_bytes
        else:
            self.frame_idx += 1
            self.prev_mask = mask.copy()
            self.v2_coder.prev_mask = mask.copy()
            self.intra_count += 1
            return b"I" + intra_bytes
            
    def decode(self, payload: bytes) -> np.ndarray:
        flag = payload[:1]
        data = payload[1:]
        if flag == b"I":
            rec = self.intra_coder.decode(data)
            self.prev_mask = rec.copy()
            self.v2_coder.prev_mask = rec.copy()
            return rec
        else:
            rec = self.v2_coder.decode(data)
            self.prev_mask = rec.copy()
            return rec

def bootstrap_ci_by_cluster(df_records: pd.DataFrame, cluster_col: str, val_col: str, n_boot: int = 1000, ci: float = 95.0):
    """Computes cluster-bootstrap confidence interval grouped by scene/clip."""
    clusters = df_records[cluster_col].unique()
    if len(clusters) <= 1:
        # Fallback to standard bootstrap if single cluster
        vals = df_records[val_col].values
        boot_means = [np.mean(np.random.choice(vals, size=len(vals), replace=True)) for _ in range(n_boot)]
        return float(np.percentile(boot_means, (100 - ci) / 2.0)), float(np.percentile(boot_means, 100 - (100 - ci) / 2.0))
        
    rng = np.random.default_rng(42)
    boot_means = []
    
    # Pre-group frames by cluster
    clust_groups = {c: df_records[df_records[cluster_col] == c][val_col].values for c in clusters}
    
    for _ in range(n_boot):
        chosen_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        # Pool all frames from chosen clusters
        sample_vals = np.concatenate([clust_groups[c] for c in chosen_clusters])
        boot_means.append(np.mean(sample_vals))
        
    low = np.percentile(boot_means, (100 - ci) / 2.0)
    high = np.percentile(boot_means, 100 - (100 - ci) / 2.0)
    return float(low), float(high)

def run_codec_benchmark(dataset_name: str, masks: np.ndarray, meta_list: list, cluster_col: str = "scene", max_frames: int = None):
    N, H, W = masks.shape
    if max_frames is not None and N > max_frames:
        masks = masks[:max_frames]
        meta_list = meta_list[:max_frames]
        N = max_frames
        
    print(f"\nEvaluating {dataset_name}: {N} frames of {H}x{W} (Clusters: {len(set(m.get(cluster_col, m.get('clip_id')) for m in meta_list))})...", flush=True)
    raw_2bit_bytes_per_frame = H * W * 2 / 8.0
    total_raw_2bit = raw_2bit_bytes_per_frame * N
    
    # Stream grouping for sequential temporal differencing
    streams = {}
    for i, m in enumerate(meta_list):
        key = (m.get("scene", m.get("clip_id", "clip")), m.get("camera", "cam0"))
        if key not in streams:
            streams[key] = []
        streams[key].append(i)
        
    codecs = {
        "FullMap_PNG9": FullMapPNG(level=9),
        "FullMap_Zstd3": FullMapZstd(level=3),
        "FullMap_Zstd19": FullMapZstd(level=19),
        "DiffMap_PNG9": DiffMapPNG(level=9),
        "DiffMap_Zstd3": DiffMapZstd(level=3),
        "DiffMap_Zstd19": DiffMapZstd(level=19),
        "CARLE_Canonical": CARLECanonicalNumba(),
        "CARLE_v2_rANS": CARLEv2Numba(),
        "CARLE_v2_IP_10": CARLEv2IPNumba(refresh_N=10),
        "CARLE_v2_IP_30": CARLEv2IPNumba(refresh_N=30),
        "CARLE_v2_IP_inf": CARLEv2IPNumba(refresh_N=float("inf")),
        "CM_intra": ContextModelCoder(mode="intra"),
        "CM_temporal": ContextModelCoder(mode="temporal"),
        "XOR_Legacy": XORLegacy(),
    }
    
    # Pre-compute PNG baseline
    png_codec = codecs["FullMap_PNG9"]
    png_bytes_list = [len(png_codec.encode(masks[i])) for i in range(N)]
    total_png_bytes = sum(png_bytes_list)
    
    benchmark_records = []
    
    for codec_name, codec in codecs.items():
        # Reset codec
        if hasattr(codec, "reset"):
            codec.reset()
            
        frame_bytes = []
        enc_times = []
        dec_times = []
        lossless_passed = True
        cluster_labels = []
        
        # Process streams sequentially
        for stream_key, frame_indices in streams.items():
            if hasattr(codec, "reset"):
                codec.reset()
            prev_m = None
            
            for fi in frame_indices:
                m = masks[fi]
                clust_val = meta_list[fi].get(cluster_col, meta_list[fi].get("clip_id", "clip"))
                cluster_labels.append(clust_val)
                
                # Encode
                t0 = time.perf_counter()
                payload = codec.encode(m)
                t_enc = (time.perf_counter() - t0) * 1000.0
                enc_times.append(t_enc)
                frame_bytes.append(len(payload))
                
                # Decode
                t0 = time.perf_counter()
                rec = codec.decode(payload)
                t_dec = (time.perf_counter() - t0) * 1000.0
                dec_times.append(t_dec)
                
                # Lossless check
                if codec_name != "XOR_Legacy":
                    if not np.array_equal(rec, m):
                        lossless_passed = False
                prev_m = m
                
        # Metrics
        bpf_list = [b * 8 / 1024.0 for b in frame_bytes]
        mean_bpf = float(np.mean(bpf_list))
        median_bpf = float(np.median(bpf_list))
        p5_bpf = float(np.percentile(bpf_list, 5))
        p95_bpf = float(np.percentile(bpf_list, 95))
        
        # Cluster Bootstrap CI by scene/clip
        df_cluster = pd.DataFrame({"cluster": cluster_labels, "bpf": bpf_list})
        ci_low, ci_high = bootstrap_ci_by_cluster(df_cluster, "cluster", "bpf")
        
        tot_bytes = sum(frame_bytes)
        cr_raw2 = total_raw_2bit / tot_bytes if tot_bytes > 0 else 0.0
        cr_png = total_png_bytes / tot_bytes if tot_bytes > 0 else 0.0
        
        mean_enc = float(np.mean(enc_times))
        p95_enc = float(np.percentile(enc_times, 95))
        mean_dec = float(np.mean(dec_times))
        p95_dec = float(np.percentile(dec_times, 95))
        
        # Mode decision info for IP
        intra_pct = 0.0
        inter_pct = 0.0
        intra_coder_name = "N/A"
        if isinstance(codec, CARLEv2IPNumba):
            tot_c = codec.total_frames if codec.total_frames > 0 else 1
            intra_pct = round(codec.intra_count / tot_c * 100.0, 2)
            inter_pct = round(codec.inter_count / tot_c * 100.0, 2)
            intra_coder_name = codec.intra_codec_name
            
        record = {
            "dataset": dataset_name,
            "codec": codec_name,
            "frames": N,
            "lossless": lossless_passed if codec_name != "XOR_Legacy" else "KNOWN_CORRUPT",
            "bpf_mean_Kb": round(mean_bpf, 2),
            "bpf_median_Kb": round(median_bpf, 2),
            "bpf_p5_Kb": round(p5_bpf, 2),
            "bpf_p95_Kb": round(p95_bpf, 2),
            "bpf_ci95_low": round(ci_low, 2),
            "bpf_ci95_high": round(ci_high, 2),
            "CR_vs_raw2": round(cr_raw2, 2),
            "CR_vs_PNG": round(cr_png, 2),
            "enc_ms_mean": round(mean_enc, 3),
            "enc_ms_p95": round(p95_enc, 3),
            "dec_ms_mean": round(mean_dec, 3),
            "dec_ms_p95": round(p95_dec, 3),
            "intra_percent": intra_pct,
            "inter_percent": inter_pct,
            "intra_coder_used": intra_coder_name
        }
        benchmark_records.append(record)
        print(f"  {codec_name:<18}: bpf={mean_bpf:6.2f} Kb (CI [{ci_low:5.2f}, {ci_high:5.2f}]) | CR_raw2={cr_raw2:5.1f}x | enc={mean_enc:5.2f}ms | dec={mean_dec:5.2f}ms | Lossless: {lossless_passed}", flush=True)
        
    return benchmark_records

def main():
    all_results = []
    
    # 1. Full nuScenes 2 Hz Keyframes (all 2,424 keyframes)
    kf_path = CACHE_DIR / "nuscenes_keyframes_2hz.npz"
    if kf_path.exists():
        kf_data = np.load(kf_path)["masks"]
        with open(CACHE_DIR / "nuscenes_keyframes_2hz_meta.json") as f:
            kf_meta = json.load(f)
        res_kf = run_codec_benchmark("nuScenes_Keyframes_2Hz", kf_data, kf_meta, cluster_col="scene")
        all_results.extend(res_kf)
        
    # 2. Full nuScenes ~12 Hz Sweeps (all 10 scenes, all 6 cameras, scene-by-scene)
    sweeps_dir = CACHE_DIR / "sweeps"
    if sweeps_dir.exists() and list(sweeps_dir.glob("sweeps_*.npz")):
        print(f"\nEvaluating nuScenes_Sweeps_12Hz across all 10 scenes (all 6 cameras)...", flush=True)
        scene_files = sorted(list(sweeps_dir.glob("sweeps_*.npz")))
        
        # We evaluate each codec across all scenes
        codecs = {
            "FullMap_PNG9": FullMapPNG(level=9),
            "FullMap_Zstd3": FullMapZstd(level=3),
            "FullMap_Zstd19": FullMapZstd(level=19),
            "DiffMap_PNG9": DiffMapPNG(level=9),
            "DiffMap_Zstd3": DiffMapZstd(level=3),
            "DiffMap_Zstd19": DiffMapZstd(level=19),
            "CARLE_Canonical": CARLECanonicalNumba(),
            "CARLE_v2_rANS": CARLEv2Numba(),
            "CARLE_v2_IP_10": CARLEv2IPNumba(refresh_N=10),
            "CARLE_v2_IP_30": CARLEv2IPNumba(refresh_N=30),
            "CARLE_v2_IP_inf": CARLEv2IPNumba(refresh_N=float("inf")),
            "CM_intra": ContextModelCoder(mode="intra"),
            "CM_temporal": ContextModelCoder(mode="temporal"),
            "XOR_Legacy": XORLegacy(),
        }
        
        total_sw_frames = 0
        total_raw_bytes = 0
        total_png_bytes = 0
        
        # Container per codec
        codec_results = {c: {"bytes": [], "enc_ms": [], "dec_ms": [], "clusters": [], "lossless": True} for c in codecs}
        
        # Iterate scene by scene to maintain tiny memory footprint
        for s_idx, s_npz in enumerate(scene_files):
            s_name = s_npz.stem.replace("sweeps_", "")
            s_meta_path = s_npz.with_name(s_npz.stem + "_meta.json")
            with open(s_meta_path) as f:
                s_meta = json.load(f)
            s_masks = np.load(s_npz)["masks"]
            N_s, H_s, W_s = s_masks.shape
            total_sw_frames += N_s
            total_raw_bytes += N_s * (H_s * W_s * 2 / 8.0)
            
            # Pre-compute PNG baseline
            png_codec = FullMapPNG(level=9)
            s_png_bytes = sum(len(png_codec.encode(s_masks[i])) for i in range(N_s))
            total_png_bytes += s_png_bytes
            
            # Group into streams per camera within this scene
            s_streams = {}
            for i, m in enumerate(s_meta):
                cam = m.get("camera", "cam0")
                if cam not in s_streams:
                    s_streams[cam] = []
                s_streams[cam].append(i)
                
            for c_name, codec in codecs.items():
                if hasattr(codec, "reset"):
                    codec.reset()
                for cam, indices in s_streams.items():
                    if hasattr(codec, "reset"):
                        codec.reset()
                    for fi in indices:
                        m = s_masks[fi]
                        t0 = time.perf_counter()
                        payload = codec.encode(m)
                        t_enc = (time.perf_counter() - t0) * 1000.0
                        
                        t0 = time.perf_counter()
                        rec = codec.decode(payload)
                        t_dec = (time.perf_counter() - t0) * 1000.0
                        
                        if c_name != "XOR_Legacy" and not np.array_equal(rec, m):
                            codec_results[c_name]["lossless"] = False
                            
                        codec_results[c_name]["bytes"].append(len(payload))
                        codec_results[c_name]["enc_ms"].append(t_enc)
                        codec_results[c_name]["dec_ms"].append(t_dec)
                        codec_results[c_name]["clusters"].append(s_name)
                        
            del s_masks; import gc; gc.collect()
            print(f"  Processed scene {s_idx+1}/10 ({s_name}, {N_s} frames)...", flush=True)
            
        # Consolidate sweeps benchmark records
        for c_name, c_data in codec_results.items():
            bpf_list = [b * 8 / 1024.0 for b in c_data["bytes"]]
            mean_bpf = float(np.mean(bpf_list))
            median_bpf = float(np.median(bpf_list))
            p5_bpf = float(np.percentile(bpf_list, 5))
            p95_bpf = float(np.percentile(bpf_list, 95))
            
            df_clust = pd.DataFrame({"cluster": c_data["clusters"], "bpf": bpf_list})
            ci_low, ci_high = bootstrap_ci_by_cluster(df_clust, "cluster", "bpf")
            
            tot_b = sum(c_data["bytes"])
            cr_raw2 = total_raw_bytes / tot_b if tot_b > 0 else 0.0
            cr_png = total_png_bytes / tot_b if tot_b > 0 else 0.0
            
            mean_enc = float(np.mean(c_data["enc_ms"]))
            p95_enc = float(np.percentile(c_data["enc_ms"], 95))
            mean_dec = float(np.mean(c_data["dec_ms"]))
            p95_dec = float(np.percentile(c_data["dec_ms"], 95))
            
            codec_inst = codecs[c_name]
            intra_pct = 0.0
            inter_pct = 0.0
            intra_coder_name = "N/A"
            if isinstance(codec_inst, CARLEv2IPNumba):
                tot_c = codec_inst.total_frames if codec_inst.total_frames > 0 else 1
                intra_pct = round(codec_inst.intra_count / tot_c * 100.0, 2)
                inter_pct = round(codec_inst.inter_count / tot_c * 100.0, 2)
                intra_coder_name = codec_inst.intra_codec_name
                
            rec = {
                "dataset": "nuScenes_Sweeps_12Hz",
                "codec": c_name,
                "frames": total_sw_frames,
                "lossless": c_data["lossless"] if c_name != "XOR_Legacy" else "KNOWN_CORRUPT",
                "bpf_mean_Kb": round(mean_bpf, 2),
                "bpf_median_Kb": round(median_bpf, 2),
                "bpf_p5_Kb": round(p5_bpf, 2),
                "bpf_p95_Kb": round(p95_bpf, 2),
                "bpf_ci95_low": round(ci_low, 2),
                "bpf_ci95_high": round(ci_high, 2),
                "CR_vs_raw2": round(cr_raw2, 2),
                "CR_vs_PNG": round(cr_png, 2),
                "enc_ms_mean": round(mean_enc, 3),
                "enc_ms_p95": round(p95_enc, 3),
                "dec_ms_mean": round(mean_dec, 3),
                "dec_ms_p95": round(p95_dec, 3),
                "intra_percent": intra_pct,
                "inter_percent": inter_pct,
                "intra_coder_used": intra_coder_name
            }
            all_results.append(rec)
            print(f"  {c_name:<18}: bpf={mean_bpf:6.2f} Kb (CI [{ci_low:5.2f}, {ci_high:5.2f}]) | CR_raw2={cr_raw2:5.1f}x | enc={mean_enc:5.2f}ms | dec={mean_dec:5.2f}ms", flush=True)
        
    # 3. Full Field Test Videos
    for cond in ["native_full", "10hz_full"]:
        for clip_id in ["india_bolpur", "malaysia", "sikkim", "dhaka_day", "dhaka_night", "bhubaneswar_night"]:
            f_path = CACHE_DIR / f"field_{clip_id}_{cond}.npz"
            f_meta_path = CACHE_DIR / f"field_{clip_id}_{cond}_meta.json"
            if f_path.exists() and f_meta_path.exists():
                f_data = np.load(f_path)["masks"]
                with open(f_meta_path) as f:
                    f_meta = json.load(f)
                d_name = f"Field_{clip_id}_{cond.replace('_full', '')}"
                res_f = run_codec_benchmark(d_name, f_data, f_meta, cluster_col="clip_id")
                all_results.extend(res_f)
                
    df_out = pd.DataFrame(all_results)
    out_csv = RESULTS_DIR / "results_full_phase1.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved complete benchmark results ({len(df_out)} rows) to {out_csv}!")

if __name__ == "__main__":
    main()
