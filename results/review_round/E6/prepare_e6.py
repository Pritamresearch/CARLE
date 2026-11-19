"""
E6: Precision and recall of the single-class (48-bit) flag.
Prepares:
1. all_48bit_frames.csv: full list of all 48-bit frames across datasets with metadata.
2. Sample of up to 100 flagged and 100 unflagged frames (night and day) saved side-by-side in E6/sample/.
3. E6/labels.csv: template with empty human_label column.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

def get_palette_k4():
    # K=4 palette: 1: Road (purple), 2: Vehicle (blue), 3: VRU/Pedestrian (red), 4: Background/Other (gray)
    return {
        1: (128, 64, 128),   # Road / Drivable
        2: (70, 70, 70),     # Vehicle (or standard Cityscapes blue/dark) -> (0, 0, 142)
        3: (220, 20, 60),    # VRU / Pedestrian / Cyclist
        4: (30, 30, 30),     # Background / Other / Void
    }

def colorize_mask(mask, palette):
    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_val, color in palette.items():
        rgb[mask == cls_val] = color
    return rgb

def main():
    parser = argparse.ArgumentParser(description="E6 Flagged Frames Extraction and Sampling")
    parser.add_argument("--smoke", action="store_true", help="Smoke run with 5 frames")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    out_dir = "results/review_round/E6"
    sample_dir = os.path.join(out_dir, "sample")
    os.makedirs(sample_dir, exist_ok=True)

    print("Step 1: Loading metadata for nuScenes and Field datasets...")
    # Load sweeps metadata
    sweeps_meta = {}
    sweeps_dir = "results/phase1b/cache_pidnet_half/sweeps"
    for fname in os.listdir(sweeps_dir):
        if fname.startswith("sweeps_") and fname.endswith("_meta.json"):
            scene = fname.replace("sweeps_", "").replace("_meta.json", "")
            with open(os.path.join(sweeps_dir, fname), "r") as f:
                sweeps_meta[scene] = json.load(f)

    # Load keyframes metadata
    with open("results/phase1b/cache_pidnet_half/nuscenes_keyframes_2hz_meta.json", "r") as f:
        kf_meta = json.load(f)

    # Load field metadata
    field_meta = {}
    cache_dir = "results/phase1b/cache_pidnet_half"
    for fname in os.listdir(cache_dir):
        if fname.startswith("field_") and fname.endswith("_meta.json"):
            field_name = fname.replace("_meta.json", "")
            with open(os.path.join(cache_dir, fname), "r") as f:
                field_meta[field_name] = json.load(f)

    print("Step 2: Parsing p1b_all_frames_raw_k4.csv and k8.csv...")
    df_k4 = pd.read_csv("results/phase1/p1b_all_frames_raw_k4.csv")
    df_k8 = pd.read_csv("results/phase1/p1b_all_frames_raw_k8.csv")

    # Night scene classification
    night_scenes = {"scene-1077", "scene-1094", "scene-1100"}

    records = []
    # Index sweeps by (scene, camera, frame_idx)
    # In sweeps_meta[scene], entries are ordered by camera (100 per camera)
    cam_order = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    sweeps_lookup = {}
    for scene, items in sweeps_meta.items():
        for i, item in enumerate(items):
            cam = item["camera"]
            fidx = i % 100
            sweeps_lookup[(scene, cam, fidx)] = item

    # Index keyframes by row index
    kf_lookup = {i: item for i, item in enumerate(kf_meta)}

    for df, K in [(df_k4, 4), (df_k8, 8)]:
        for idx, row in df.iterrows():
            dset = row["dataset"]
            stream = row["stream"]
            fidx = int(row["frame_idx"])
            y_bits = int(row["yang_bits"])
            flagged = (y_bits == 48)

            scene = None
            cam = None
            ts = None
            fn = None
            is_night = False

            if dset == "nuScenes_Sweeps_12Hz":
                # stream format: scene-0061_CAM_FRONT
                parts = stream.rsplit("_", 1) if "_" in stream else (stream, "")
                if len(parts) == 2 and parts[0] in sweeps_meta:
                    scene = parts[0]
                    cam = parts[1]
                else:
                    # check CAM_FRONT_LEFT etc
                    for c in cam_order:
                        if stream.endswith(c):
                            cam = c
                            scene = stream[:-(len(c)+1)]
                            break
                item = sweeps_lookup.get((scene, cam, fidx))
                if item:
                    ts = item.get("timestamp")
                    fn = item.get("filename")
                is_night = (scene in night_scenes)

            elif dset == "nuScenes_Keyframes_2Hz":
                item = kf_lookup.get(idx)
                if item:
                    scene = item.get("scene")
                    cam = item.get("camera")
                    ts = item.get("timestamp")
                    fn = item.get("filename")
                    is_night = (scene in night_scenes)

            elif "Field" in dset:
                is_night = ("night" in stream.lower())
                scene = stream
                cam = "single"
                fn = None

            records.append({
                "row_id": idx,
                "K": K,
                "dataset": dset,
                "stream": stream,
                "scene": scene,
                "camera": cam,
                "frame_idx": fidx,
                "timestamp": ts,
                "filename": fn,
                "is_night": is_night,
                "yang_bits": y_bits,
                "flagged_48bit": flagged
            })

    full_df = pd.DataFrame(records)

    # Filter all 48-bit flagged frames
    flagged_df = full_df[full_df["flagged_48bit"]].copy()
    print(f"Total flagged 48-bit frames found: {len(flagged_df)} (K=4: {(flagged_df['K']==4).sum()}, K=8: {(flagged_df['K']==8).sum()})")

    flagged_csv = os.path.join(out_dir, "all_48bit_frames.csv")
    flagged_df.to_csv(flagged_csv, index=False)
    print(f"Saved all 48-bit frames to {flagged_csv}")

    # Step 3: Draw random sample of up to 100 flagged and 100 unflagged frames
    # Prompt: "Draw a random sample of up to 100 flagged frames and 100 unflagged frames from the night and day clips (seed 0). Write them with the source image side by side into E6/sample/."
    # We focus on K=4 where source RGB image is available in data/nuscenes
    k4_df = full_df[(full_df["K"] == 4) & (full_df["filename"].notna())].copy()

    # Verify files exist on disk
    k4_df["exists"] = k4_df["filename"].apply(lambda f: os.path.exists(os.path.join("data/nuscenes", str(f))))
    k4_df = k4_df[k4_df["exists"]].copy()

    flagged_pool = k4_df[k4_df["flagged_48bit"]]
    unflagged_pool = k4_df[~k4_df["flagged_48bit"]]

    print(f"Available pool with existing RGB images: flagged={len(flagged_pool)}, unflagged={len(unflagged_pool)}")

    rng = np.random.default_rng(args.seed)

    n_sample = 5 if args.smoke else 100
    n_flagged = min(n_sample, len(flagged_pool))
    n_unflagged = min(n_sample, len(unflagged_pool))

    # Sample balanced across night and day where possible
    flagged_night = flagged_pool[flagged_pool["is_night"]]
    flagged_day = flagged_pool[~flagged_pool["is_night"]]
    unflagged_night = unflagged_pool[unflagged_pool["is_night"]]
    unflagged_day = unflagged_pool[~unflagged_pool["is_night"]]

    print(f"Flagged pool: night={len(flagged_night)}, day={len(flagged_day)}")
    print(f"Unflagged pool: night={len(unflagged_night)}, day={len(unflagged_day)}")

    # Draw samples
    sampled_flagged_indices = rng.choice(flagged_pool.index, size=n_flagged, replace=False)
    sampled_unflagged_indices = rng.choice(unflagged_pool.index, size=n_unflagged, replace=False)

    sample_df = pd.concat([k4_df.loc[sampled_flagged_indices], k4_df.loc[sampled_unflagged_indices]]).copy()
    # Shuffle order so annotator is blinded to whether it's flagged or not
    sample_df = sample_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    sample_df["frame_id"] = [f"sample_{i:04d}" for i in range(len(sample_df))]

    # Load cached masks
    # Cache loaders for sweeps and keyframes
    sweeps_masks = {}
    for scene in sweeps_meta:
        npz_p = os.path.join(sweeps_dir, f"sweeps_{scene}_k4.npz")
        if os.path.exists(npz_p):
            sweeps_masks[scene] = np.load(npz_p)["masks"]

    kf_npz = np.load("results/phase1b/cache_pidnet_half/nuscenes_keyframes_2hz_k4.npz")
    kf_masks = kf_npz["masks"]

    palette = get_palette_k4()

    print(f"Step 4: Generating {len(sample_df)} side-by-side visualization images...")
    label_records = []
    for _, row in sample_df.iterrows():
        fid = row["frame_id"]
        fn = row["filename"]
        dset = row["dataset"]
        flagged = int(row["flagged_48bit"])
        scene = row["scene"]
        cam = row["camera"]
        fidx = row["frame_idx"]

        img_path = os.path.join("data/nuscenes", fn)
        img = Image.open(img_path).convert("RGB")
        # Half resolution is 450 x 800
        img_resized = img.resize((800, 450), Image.Resampling.BILINEAR)

        # Get mask
        mask = None
        if dset == "nuScenes_Sweeps_12Hz":
            cam_idx = cam_order.index(cam) if cam in cam_order else 0
            abs_idx = cam_idx * 100 + fidx
            mask = sweeps_masks[scene][abs_idx]
        elif dset == "nuScenes_Keyframes_2Hz":
            mask = kf_masks[row["row_id"]]

        mask_rgb = colorize_mask(mask, palette)
        mask_img = Image.fromarray(mask_rgb)

        # Create side-by-side composite
        comp = Image.new("RGB", (1600, 450))
        comp.paste(img_resized, (0, 0))
        comp.paste(mask_img, (800, 0))

        # Add text header
        draw = ImageDraw.Draw(comp)
        draw.text((20, 20), f"Frame ID: {fid} | {scene} {cam} frame {fidx}", fill=(255, 255, 255))
        draw.text((820, 20), f"PIDNet K=4 Segmentation (Unique classes: {len(np.unique(mask))})", fill=(255, 255, 255))

        out_img_path = os.path.join(sample_dir, f"{fid}.png")
        comp.save(out_img_path)

        label_records.append({
            "frame_id": fid,
            "flagged": flagged,
            "human_label": ""  # Left empty for human annotator
        })

    # Save labels.csv
    labels_df = pd.DataFrame(label_records)
    labels_csv = os.path.join(out_dir, "labels.csv")
    labels_df.to_csv(labels_csv, index=False)
    print(f"Saved {len(labels_df)} sample rows to {labels_csv}")
    print(f"Sample images written to {sample_dir}")

if __name__ == "__main__":
    main()
