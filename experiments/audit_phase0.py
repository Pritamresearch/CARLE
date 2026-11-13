"""
Phase 0 Comprehensive Audit Script:
1. Cityscapes pairing check (20 pairs, analysis of 0.007% delta)
2. NuScenes frame-rate check (timestamps, delta-t)
3. Field videos matching (which 4 of 9 clips, frame counts, sampling rate)
4. Method mismatch: 3x3 toy proof, class-0 collision
5. Size accounting and Table reproduction
"""

import os
import re
import ast
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = Path("configs/paths.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

PHASE0_DIR = Path("results/phase0")
PHASE0_DIR.mkdir(parents=True, exist_ok=True)
PAIRS_DIR = PHASE0_DIR / "cityscapes_pairs"
PAIRS_DIR.mkdir(parents=True, exist_ok=True)

def audit_cityscapes_pairs():
    print("=" * 60)
    print("0.2 Cityscapes Pairing Check")
    print("=" * 60)
    
    csv_path = Path("legacy/results/compression_output.csv")
    df = pd.read_csv(csv_path)
    print(f"Total rows in compression_output.csv: {len(df)}")
    
    pairs_info = []
    for i in range(20):
        r = df.iloc[i]
        im1 = r["Image1"]
        im2 = r["Image2"]
        m1 = re.match(r"([a-z]+)_(\d+)_(\d+)", im1)
        m2 = re.match(r"([a-z]+)_(\d+)_(\d+)", im2)
        c1, s1, f1 = m1.groups() if m1 else ("?", "?", "?")
        c2, s2, f2 = m2.groups() if m2 else ("?", "?", "?")
        comp_b = r["CompressedSize(Bytes)"]
        rle = ast.literal_eval(r["RLE_Code"]) if isinstance(r["RLE_Code"], str) else r["RLE_Code"]
        
        # Check changed pixels:
        # In xor_rle, each entry is [start, length, value].
        # If value != 0, it's changed pixels.
        changed_px = sum(l for s, l, v in rle if v != 0)
        total_px = 1024 * 2048 # 2097152
        delta_pct = (changed_px / total_px) * 100
        
        pairs_info.append({
            "idx": i + 1,
            "im1": im1,
            "im2": im2,
            "city": c1,
            "seq1": s1,
            "seq2": s2,
            "frame1": f1,
            "frame2": f2,
            "comp_bytes": comp_b,
            "changed_px": changed_px,
            "delta_pct": delta_pct,
            "rle": rle
        })
        print(f"Pair {i+1:2d}: {im1} <-> {im2} | City: {c1} | Seq: {s1} vs {s2} | Frame: {f1} vs {f2} | Changed: {changed_px} px ({delta_pct:.4f}%) | CompBytes: {comp_b}")

    # Inspect the naming convention:
    # Notice: aachen_000000_000019_gtFinePanopticParts.tif vs aachen_000001_000019_gtFinePanopticParts.tif
    # In Cityscapes:
    # <city>_<sequenceId>_<frameId>_gtFine_...
    # Here: sequenceId is 000000 and 000001! Frame ID is 000019 for BOTH!
    # Frame 19 is the 20th frame of sequence 000000 and sequence 000001.
    # ARE THEY TEMPORAL? NO! They are two completely different driving sequences from Aachen!
    # But wait, why are only 0.007% pixels different between two different scenes?
    # Let's inspect the RLE code of the pairs!
    print("\nDetailed breakdown of first 5 pairs RLE codes:")
    for p in pairs_info[:5]:
        print(f"Pair {p['idx']}: {p['rle']}")

    return pairs_info

def generate_cityscapes_overlays():
    print("\nGenerating visual overlays for 5 Cityscapes pairs...")
    # Let's check if the corresponding Cityscapes images exist in data/cityscapes
    cs_gt = Path(config["datasets"]["cityscapes"]["gt_fine"])
    cs_img = Path(config["datasets"]["cityscapes"]["left_img8bit"])
    
    # Let's find aachen images
    aachen_imgs = sorted(list(cs_img.glob("train/aachen/*_leftImg8bit.png")))
    aachen_gts = sorted(list(cs_gt.glob("train/aachen/*_gtFine_color.png")))
    print(f"Found {len(aachen_imgs)} Aachen images in Cityscapes train split.")
    
    for i in range(min(5, len(aachen_imgs) - 1)):
        p1 = aachen_imgs[i]
        p2 = aachen_imgs[i + 1]
        im1 = cv2.imread(str(p1))
        im2 = cv2.imread(str(p2))
        
        # Real Cityscapes consecutive annotated images:
        diff = cv2.absdiff(im1, im2)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        changed = np.count_nonzero(diff_gray > 20)
        pct = (changed / diff_gray.size) * 100
        
        # Create side-by-side comparison
        vis = np.hstack([im1, im2])
        vis = cv2.resize(vis, (1600, 400))
        cv2.putText(vis, f"Image 1: {p1.name}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Image 2: {p2.name} (Diff: {pct:.1f}% px)", (820, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        out_path = PAIRS_DIR / f"cityscapes_pair_{i+1}.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"Saved overlay {out_path.name}: {p1.name} vs {p2.name} (real visual diff: {pct:.1f}%)")

if __name__ == "__main__":
    audit_cityscapes_pairs()
    generate_cityscapes_overlays()
