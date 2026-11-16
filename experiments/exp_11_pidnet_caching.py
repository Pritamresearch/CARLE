"""
Phase 1b - Experiment 11: Build Grounded Semantic Label-Map Caches using PIDNet-S (Cityscapes).
Projects predictions to:
  K=4: 1 Road, 2 Vehicle, 3 Vulnerable Road User, 4 Other Static
  K=8: 1 Road, 2 Sidewalk, 3 Static Structure, 4 Vegetation/Terrain, 5 Sky, 6 Vehicle, 7 Two-Wheeler, 8 Person/Rider
Saves:
  - Compressed npz caches for K=4 and K=8: nuScenes Keyframes, Sweeps (6 cameras), Field Native, Field 10 Hz
  - 10 Visual overlay images per dataset in results/phase1/overlays/
  - GPU segmentation time logging
  - Dataset statistics (class shares, % single-class frames, mean delta_pix)
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import json
import cv2
import numpy as np
import torch
import yaml
from collections import defaultdict

PIDNET_DIR = Path("legacy_local/CARLE CODE/PIDNet").resolve()
sys.path.insert(0, str(PIDNET_DIR))
from models.pidnet import get_pred_model

CACHE_DIR = Path("results/phase1/cache_pidnet")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SWEEPS_DIR = CACHE_DIR / "sweeps"
SWEEPS_DIR.mkdir(parents=True, exist_ok=True)

OVERLAY_DIR = Path("results/phase1/overlays")
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

# Cityscapes 19 trainIds:
# 0: road, 1: sidewalk, 2: building, 3: wall, 4: fence, 5: pole,
# 6: traffic light, 7: traffic sign, 8: vegetation, 9: terrain, 10: sky,
# 11: person, 12: rider, 13: car, 14: truck, 15: bus, 16: train,
# 17: motorcycle, 18: bicycle

CITYSCAPES_TO_K4 = np.zeros(19, dtype=np.uint8)
CITYSCAPES_TO_K4[0] = 1                         # Road {0}
CITYSCAPES_TO_K4[[13, 14, 15, 16]] = 2         # Vehicle {13,14,15,16}
CITYSCAPES_TO_K4[[11, 12, 17, 18]] = 3         # Vulnerable Road User {11,12,17,18}
CITYSCAPES_TO_K4[list(range(1, 11))] = 4       # Other {1..10}

CITYSCAPES_TO_K8 = np.zeros(19, dtype=np.uint8)
CITYSCAPES_TO_K8[0] = 1                         # Road {0}
CITYSCAPES_TO_K8[1] = 2                         # Sidewalk {1}
CITYSCAPES_TO_K8[[2, 3, 4, 5, 6, 7]] = 3       # Static structure {2,3,4,5,6,7}
CITYSCAPES_TO_K8[[8, 9]] = 4                   # Vegetation / terrain {8,9}
CITYSCAPES_TO_K8[10] = 5                        # Sky {10}
CITYSCAPES_TO_K8[[13, 14, 15, 16]] = 6         # Vehicle {13,14,15,16}
CITYSCAPES_TO_K8[[17, 18]] = 7                 # Two-wheeler {17,18}
CITYSCAPES_TO_K8[[11, 12]] = 8                 # Person / rider {11,12}

PALETTE_K4 = np.array([
    [0, 0, 0],          # 0: Void
    [128, 64, 128],     # 1: Road (Purple)
    [142, 0, 0],        # 2: Vehicle (Blue)
    [60, 20, 220],      # 3: Vulnerable Road User (Red)
    [70, 70, 70]        # 4: Other (Gray)
], dtype=np.uint8)

PALETTE_K8 = np.array([
    [0, 0, 0],          # 0: Void
    [128, 64, 128],     # 1: Road (Purple)
    [232, 35, 244],     # 2: Sidewalk (Pink)
    [70, 70, 70],       # 3: Static structure (Gray)
    [35, 142, 107],     # 4: Vegetation/terrain (Green)
    [180, 130, 70],     # 5: Sky (Sky blue)
    [142, 0, 0],        # 6: Vehicle (Blue)
    [0, 128, 255],      # 7: Two-wheeler (Orange)
    [60, 20, 220]       # 8: Person/rider (Red)
], dtype=np.uint8)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_pidnet_model():
    print("Loading PIDNet-S Cityscapes model...")
    model = get_pred_model("pidnet-s", 19)
    weight_path = PIDNET_DIR / "pretrained_models/Cityscapes/PIDNet_S_Cityscapes_val.pt"
    ckpt = torch.load(weight_path, map_location="cpu")
    sd = {k[6:]: v for k, v in ckpt.items() if k.startswith("model.")}
    model.load_state_dict(sd, strict=False)
    model.eval().cuda().half() # FP16 for speed
    print("PIDNet-S loaded successfully in FP16 on GPU.")
    return model


def segment_batch_pidnet(model, img_bgr_list, device="cuda"):
    """
    Batched inference at native resolution with zero distortion via multiple-of-32 padding.
    """
    B = len(img_bgr_list)
    H, W = img_bgr_list[0].shape[:2]
    pad_H = ((H + 31) // 32) * 32
    pad_W = ((W + 31) // 32) * 32
    
    t0 = time.perf_counter()
    batch_tensor = torch.empty((B, 3, pad_H, pad_W), dtype=torch.float16, device=device)
    for b in range(B):
        img_bgr = img_bgr_list[b]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rgb_norm = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        padded = np.zeros((pad_H, pad_W, 3), dtype=np.float32)
        padded[:H, :W] = rgb_norm
        batch_tensor[b] = torch.from_numpy(padded).permute(2, 0, 1).half()
        
    with torch.inference_mode():
        pred = model(batch_tensor)
        if isinstance(pred, (list, tuple)): pred = pred[0]
        pred = torch.nn.functional.interpolate(pred, size=(pad_H, pad_W), mode="bilinear", align_corners=True)
        if pad_H != H or pad_W != W:
            pred = pred[:, :, :H, :W]
        masks = torch.argmax(pred, dim=1).byte().cpu().numpy()
        
    torch.cuda.synchronize()
    gpu_ms = (time.perf_counter() - t0) * 1000.0 / B
    
    # Project to K=4 and K=8
    masks_k4 = CITYSCAPES_TO_K4[masks]
    masks_k8 = CITYSCAPES_TO_K8[masks]
    
    return masks_k4, masks_k8, gpu_ms


def save_overlays(prefix, img_bgr, m_k4, m_k8, idx):
    H, W = img_bgr.shape[:2]
    c_k4 = PALETTE_K4[m_k4]
    c_k8 = PALETTE_K8[m_k8]
    ov_k4 = cv2.addWeighted(img_bgr, 0.5, c_k4, 0.5, 0)
    ov_k8 = cv2.addWeighted(img_bgr, 0.5, c_k8, 0.5, 0)
    
    # Horizontal side-by-side: Original | K=4 Overlay | K=8 Overlay
    # Downsample for compact preview
    disp_w = 480
    disp_h = int(H * (disp_w / W))
    p_orig = cv2.resize(img_bgr, (disp_w, disp_h))
    p_k4 = cv2.resize(ov_k4, (disp_w, disp_h))
    p_k8 = cv2.resize(ov_k8, (disp_w, disp_h))
    
    cv2.putText(p_orig, "Original RGB", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(p_k4, "PIDNet K=4 Overlay", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(p_k8, "PIDNet K=8 Overlay", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    combined = np.hstack([p_orig, p_k4, p_k8])
    out_path = OVERLAY_DIR / f"{prefix}_overlay_{idx:02d}.jpg"
    cv2.imwrite(str(out_path), combined)


# =============================================================================
# 1. nuScenes Keyframes (2 Hz) Caching
# =============================================================================
def cache_nuscenes_keyframes(model):
    print("\n--- 1. Caching nuScenes Keyframes (2 Hz) ---")
    meta_p = Path("results/phase1/cache/nuscenes_keyframes_2hz_meta.json")
    with open(meta_p, "r") as f:
        meta_list = json.load(f)
        
    dataroot = Path("data/nuscenes")
    N = len(meta_list)
    print(f"Total keyframes to segment: {N}")
    
    batch_size = 4
    masks_k4_all = []
    masks_k8_all = []
    gpu_times = []
    
    overlay_indices = set(np.linspace(0, N - 1, 10, dtype=int))
    
    for i in range(0, N, batch_size):
        batch_meta = meta_list[i : i + batch_size]
        batch_imgs = [cv2.imread(str(dataroot / m["filename"])) for m in batch_meta]
        
        m_k4, m_k8, gpu_ms = segment_batch_pidnet(model, batch_imgs)
        for b in range(len(batch_meta)):
            masks_k4_all.append(m_k4[b])
            masks_k8_all.append(m_k8[b])
            gpu_times.append(gpu_ms)
            global_idx = i + b
            if global_idx in overlay_indices:
                save_overlays("nuscenes_keyframes", batch_imgs[b], m_k4[b], m_k8[b], global_idx)
                
        if (i // batch_size) % 50 == 0 or i + batch_size >= N:
            print(f"Keyframes: {min(i + batch_size, N)}/{N} segmented ({gpu_ms:.2f} ms/frame)")
            
    masks_k4_arr = np.array(masks_k4_all, dtype=np.uint8)
    masks_k8_arr = np.array(masks_k8_all, dtype=np.uint8)
    
    np.savez_compressed(CACHE_DIR / "nuscenes_keyframes_2hz_k4.npz", masks=masks_k4_arr)
    np.savez_compressed(CACHE_DIR / "nuscenes_keyframes_2hz_k8.npz", masks=masks_k8_arr)
    
    # Save metadata with PIDNet GPU times
    for idx, m in enumerate(meta_list):
        m["gpu_ms_pidnet"] = round(gpu_times[idx], 2)
    with open(CACHE_DIR / "nuscenes_keyframes_2hz_meta.json", "w") as f:
        json.dump(meta_list, f, indent=2)
        
    print(f"Saved nuScenes keyframes: K4 shape={masks_k4_arr.shape}, K8 shape={masks_k8_arr.shape}, mean GPU time={np.mean(gpu_times):.2f} ms")


# =============================================================================
# 2. nuScenes Sweeps (~12 Hz) Caching
# =============================================================================
def cache_nuscenes_sweeps(model):
    print("\n--- 2. Caching nuScenes Sweeps (~12 Hz, All 6 Cameras, 100 Frames/Stream) ---")
    cache_yolo_dir = Path("results/phase1/cache/sweeps")
    sweep_files = sorted(list(cache_yolo_dir.glob("sweeps_scene-*.npz")))
    dataroot = Path("data/nuscenes")
    
    batch_size = 4
    total_sweeps_cached = 0
    overlay_count = 0
    
    for sf in sweep_files:
        sc_name = sf.stem.replace("sweeps_", "")
        meta_p = cache_yolo_dir / f"sweeps_{sc_name}_meta.json"
        with open(meta_p, "r") as f:
            meta = json.load(f)
            
        # Group by camera and sort by timestamp
        by_cam = defaultdict(list)
        for i, m in enumerate(meta):
            by_cam[m["camera"]].append(m)
            
        scene_k4 = []
        scene_k8 = []
        scene_meta_out = []
        
        # Take 100 contiguous frames per camera stream (all 6 cameras = 600 frames/scene)
        for cam in ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]:
            c_meta = by_cam[cam]
            c_meta.sort(key=lambda x: x["timestamp"])
            selected = c_meta[:100]
            
            for i in range(0, len(selected), batch_size):
                b_meta = selected[i : i + batch_size]
                b_imgs = [cv2.imread(str(dataroot / m["filename"])) for m in b_meta]
                m_k4, m_k8, gpu_ms = segment_batch_pidnet(model, b_imgs)
                
                for b in range(len(b_meta)):
                    scene_k4.append(m_k4[b])
                    scene_k8.append(m_k8[b])
                    m_dict = dict(b_meta[b])
                    m_dict["gpu_ms_pidnet"] = round(gpu_ms, 2)
                    scene_meta_out.append(m_dict)
                    
                    if overlay_count < 10 and cam == "CAM_FRONT" and b == 0:
                        save_overlays("nuscenes_sweeps", b_imgs[b], m_k4[b], m_k8[b], overlay_count)
                        overlay_count += 1
                        
        sc_k4_arr = np.array(scene_k4, dtype=np.uint8)
        sc_k8_arr = np.array(scene_k8, dtype=np.uint8)
        
        np.savez_compressed(SWEEPS_DIR / f"sweeps_{sc_name}_k4.npz", masks=sc_k4_arr)
        np.savez_compressed(SWEEPS_DIR / f"sweeps_{sc_name}_k8.npz", masks=sc_k8_arr)
        with open(SWEEPS_DIR / f"sweeps_{sc_name}_meta.json", "w") as f:
            json.dump(scene_meta_out, f, indent=2)
            
        total_sweeps_cached += len(scene_k4)
        print(f"Scene {sc_name}: cached {len(scene_k4)} frames across 6 cameras")
        
    print(f"Total nuScenes sweeps cached: {total_sweeps_cached} frames across 10 scenes")


# =============================================================================
# 3. Field Videos Caching (Native and 10 Hz)
# =============================================================================
def cache_field_videos(model):
    print("\n--- 3. Caching Field Videos (Native and 10 Hz) ---")
    with open("configs/fieldtest.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    clips = cfg["clips"]
    batch_size = 4
    
    for clip in clips:
        cid = clip["clip_id"]
        vpath = Path("data/field_videos") / clip["filename"]
        if not vpath.exists():
            print(f"Video {vpath} not found!")
            continue
            
        cap = cv2.VideoCapture(str(vpath))
        fps = cap.get(cv2.CAP_PROP_FPS)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frames_all = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frames_all.append(frame)
        cap.release()
        
        N_tot = len(frames_all)
        print(f"\nProcessing {cid}: {N_tot} frames, {W}x{H} at {fps:.2f} fps")
        
        # Segment all frames in batches
        k4_all = []
        k8_all = []
        gpu_times = []
        
        overlay_saved = 0
        for i in range(0, N_tot, batch_size):
            b_imgs = frames_all[i : i + batch_size]
            m_k4, m_k8, gpu_ms = segment_batch_pidnet(model, b_imgs)
            for b in range(len(b_imgs)):
                k4_all.append(m_k4[b])
                k8_all.append(m_k8[b])
                gpu_times.append(gpu_ms)
                if overlay_saved < 10 and (i + b) % max(1, N_tot // 10) == 0:
                    save_overlays(f"field_{cid}", b_imgs[b], m_k4[b], m_k8[b], overlay_saved)
                    overlay_saved += 1
                    
        k4_arr = np.array(k4_all, dtype=np.uint8)
        k8_arr = np.array(k8_all, dtype=np.uint8)
        
        # Save Native
        np.savez_compressed(CACHE_DIR / f"field_{cid}_native_k4.npz", masks=k4_arr)
        np.savez_compressed(CACHE_DIR / f"field_{cid}_native_k8.npz", masks=k8_arr)
        nat_meta = [{"frame_idx": idx, "fps": fps, "gpu_ms_pidnet": round(gpu_times[idx], 2)} for idx in range(N_tot)]
        with open(CACHE_DIR / f"field_{cid}_native_meta.json", "w") as f:
            json.dump(nat_meta, f, indent=2)
            
        # 10 Hz subsampled (stride = round(fps / 10))
        stride = max(1, int(round(fps / 10.0)))
        k4_10hz = k4_arr[::stride]
        k8_10hz = k8_arr[::stride]
        np.savez_compressed(CACHE_DIR / f"field_{cid}_10hz_k4.npz", masks=k4_10hz)
        np.savez_compressed(CACHE_DIR / f"field_{cid}_10hz_k8.npz", masks=k8_10hz)
        meta_10hz = [nat_meta[idx] for idx in range(0, N_tot, stride)]
        with open(CACHE_DIR / f"field_{cid}_10hz_meta.json", "w") as f:
            json.dump(meta_10hz, f, indent=2)
            
        print(f"{cid}: Saved Native ({len(k4_arr)} frames) & 10 Hz ({len(k4_10hz)} frames). Mean GPU time={np.mean(gpu_times):.2f} ms")


# =============================================================================
# 4. Dataset Audit: Class Shares, Single-Class %, Mean Delta Pix
# =============================================================================
def audit_pidnet_caches():
    print("\n================================================================================")
    print("AUDITING PIDNET GROUNDED SEMANTIC CACHES (K=4 and K=8)")
    print("================================================================================")
    
    datasets_k4 = {
        "nuScenes_Keyframes_2Hz": CACHE_DIR / "nuscenes_keyframes_2hz_k4.npz",
        "Field_Bhubaneswar_Night_Native": CACHE_DIR / "field_bhubaneswar_night_native_k4.npz",
        "Field_Dhaka_Day_Native": CACHE_DIR / "field_dhaka_day_native_k4.npz",
        "Field_Dhaka_Night_Native": CACHE_DIR / "field_dhaka_night_native_k4.npz",
        "Field_Bolpur_Native": CACHE_DIR / "field_india_bolpur_native_k4.npz",
        "Field_Malaysia_Native": CACHE_DIR / "field_malaysia_native_k4.npz",
        "Field_Sikkim_Native": CACHE_DIR / "field_sikkim_native_k4.npz",
    }
    
    stats_k4 = []
    for name, p in datasets_k4.items():
        if not p.exists(): continue
        m = np.load(p)["masks"]
        N, H, W = m.shape
        tot = N * H * W
        cnts = np.bincount(m.ravel(), minlength=5)
        
        n_single = sum(1 for i in range(N) if np.max(np.bincount(m[i].ravel(), minlength=5)[1:]) == H * W)
        has_road = sum(1 for i in range(N) if (m[i] == 1).any())
        
        # Mean delta_pix
        deltas = [np.mean(m[i] != m[i-1]) for i in range(1, N)]
        mean_dp = np.mean(deltas) if len(deltas) > 0 else 0.0
        
        stats_k4.append({
            "dataset": name,
            "N_frames": N,
            "Road_c1_pct": round(cnts[1] / tot * 100, 2),
            "Vehicle_c2_pct": round(cnts[2] / tot * 100, 2),
            "VRU_c3_pct": round(cnts[3] / tot * 100, 2),
            "Other_c4_pct": round(cnts[4] / tot * 100, 2),
            "has_road_pct": round(has_road / N * 100, 2),
            "single_class_pct": round(n_single / N * 100, 2),
            "mean_delta_pix": round(mean_dp, 4)
        })
        
    # nuScenes Sweeps K=4
    sw_files = sorted(list(SWEEPS_DIR.glob("sweeps_*_k4.npz")))
    sw_N = 0
    sw_tot = 0
    sw_cnts = np.zeros(5, dtype=np.int64)
    sw_single = 0
    sw_has_road = 0
    sw_deltas = []
    
    for sf in sw_files:
        m = np.load(sf)["masks"]
        N, H, W = m.shape
        sw_N += N
        sw_tot += N * H * W
        for i in range(N):
            fc = np.bincount(m[i].ravel(), minlength=5)
            sw_cnts += fc
            if fc[1] > 0: sw_has_road += 1
            if np.max(fc[1:]) == H * W: sw_single += 1
            if i > 0: sw_deltas.append(np.mean(m[i] != m[i-1]))
            
    stats_k4.append({
        "dataset": "nuScenes_Sweeps_12Hz_All6Cams",
        "N_frames": sw_N,
        "Road_c1_pct": round(sw_cnts[1] / sw_tot * 100, 2),
        "Vehicle_c2_pct": round(sw_cnts[2] / sw_tot * 100, 2),
        "VRU_c3_pct": round(sw_cnts[3] / sw_tot * 100, 2),
        "Other_c4_pct": round(sw_cnts[4] / sw_tot * 100, 2),
        "has_road_pct": round(sw_has_road / sw_N * 100, 2),
        "single_class_pct": round(sw_single / sw_N * 100, 2),
        "mean_delta_pix": round(np.mean(sw_deltas), 4)
    })
    
    df_k4 = pd.DataFrame(stats_k4)
    print("\n--- PIDNet K=4 Class Distribution & Single-Class Audit ---")
    print(df_k4.to_string(index=False))
    df_k4.to_csv("results/phase1/pidnet_k4_class_distribution_audit.csv", index=False)
    
    # K=8 audit
    datasets_k8 = {
        "nuScenes_Keyframes_2Hz": CACHE_DIR / "nuscenes_keyframes_2hz_k8.npz",
        "Field_Bhubaneswar_Night_Native": CACHE_DIR / "field_bhubaneswar_night_native_k8.npz",
        "Field_Dhaka_Day_Native": CACHE_DIR / "field_dhaka_day_native_k8.npz",
        "Field_Dhaka_Night_Native": CACHE_DIR / "field_dhaka_night_native_k8.npz",
        "Field_Bolpur_Native": CACHE_DIR / "field_india_bolpur_native_k8.npz",
        "Field_Malaysia_Native": CACHE_DIR / "field_malaysia_native_k8.npz",
        "Field_Sikkim_Native": CACHE_DIR / "field_sikkim_native_k8.npz",
    }
    stats_k8 = []
    for name, p in datasets_k8.items():
        if not p.exists(): continue
        m = np.load(p)["masks"]
        N, H, W = m.shape
        tot = N * H * W
        cnts = np.bincount(m.ravel(), minlength=9)
        n_single = sum(1 for i in range(N) if np.max(np.bincount(m[i].ravel(), minlength=9)[1:]) == H * W)
        deltas = [np.mean(m[i] != m[i-1]) for i in range(1, N)]
        mean_dp = np.mean(deltas) if len(deltas) > 0 else 0.0
        row = {
            "dataset": name,
            "N_frames": N,
            "Road_c1": round(cnts[1] / tot * 100, 2),
            "Sidewalk_c2": round(cnts[2] / tot * 100, 2),
            "Static_c3": round(cnts[3] / tot * 100, 2),
            "Veg_c4": round(cnts[4] / tot * 100, 2),
            "Sky_c5": round(cnts[5] / tot * 100, 2),
            "Vehicle_c6": round(cnts[6] / tot * 100, 2),
            "TwoWheel_c7": round(cnts[7] / tot * 100, 2),
            "Person_c8": round(cnts[8] / tot * 100, 2),
            "single_class_pct": round(n_single / N * 100, 2),
            "mean_delta_pix": round(mean_dp, 4)
        }
        stats_k8.append(row)
        
    df_k8 = pd.DataFrame(stats_k8)
    print("\n--- PIDNet K=8 Class Distribution & Single-Class Audit ---")
    print(df_k8.to_string(index=False))
    df_k8.to_csv("results/phase1/pidnet_k8_class_distribution_audit.csv", index=False)


def main():
    print("================================================================================")
    print("STARTING PIDNET-S GROUNDED SEMANTIC CACHING PIPELINE (K=4 and K=8)")
    print("================================================================================")
    
    model = load_pidnet_model()
    
    # 1. Keyframes
    cache_nuscenes_keyframes(model)
    
    # 2. Sweeps
    cache_nuscenes_sweeps(model)
    
    # 3. Field Videos
    cache_field_videos(model)
    
    # 4. Audit
    audit_pidnet_caches()
    
    print("\n================================================================================")
    print("PIDNET-S CACHING COMPLETED SUCCESSFULLY!")
    print("================================================================================")


if __name__ == "__main__":
    main()
