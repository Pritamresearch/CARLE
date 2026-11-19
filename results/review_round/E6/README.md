# E6: Precision and Recall of the Single-Class (48-bit) Flag

## Protocol & Status
- **Current Status**: **PENDING HUMAN LABELLING**. Precision and recall numbers will exist **only after human labelling**.
- **Hard Rule Compliance**: As instructed, no synthetic or self-generated labels were invented. The `human_label` column in `labels.csv` is delivered blank for double-blind human verification.

## Background
In Yang et al.'s context-adaptive extended chain coding (ECC), when a segmentation map consists entirely of a single semantic class (e.g., completely dark night frames predicted as uniform background/other or uniform sky/void), the encoder outputs an empty contour list with a 48-bit (6-byte) header flag:
- 16-bit width
- 16-bit height
- 8-bit uniform class ID
- 8-bit reserved/flag bits
Total = 48 bits (6 bytes).

## Sampling Methodology
1. **Total Flagged Frames**:
   Across the nuScenes 60 sweep streams and 2 Hz keyframes at $K=4$, a total of **205 frames** produce 48 bits (single-class maps).
   - In night clips (e.g. `scene-1077`, `scene-1094`, `scene-1100`), dark cameras (e.g. `CAM_BACK`, `CAM_BACK_LEFT`, `CAM_FRONT_LEFT`) frequently output uniform void.
   - Field clips (`Field_Native`, `Field_10Hz`) produced zero 48-bit frames (minimum bit cost was 368 bits at $K=4$).
2. **Double-Blind Sample**:
   Using `seed=0`, a balanced sample of 100 flagged frames and 100 unflagged frames was drawn across day and night streams.
   - For each sampled frame, a composite side-by-side image was rendered to `results/review_round/E6/sample/<frame_id>.png` (left: RGB source camera image, right: PIDNet-S half-resolution $K=4$ segmentation mask).
   - The sample IDs were randomly shuffled to blind the human annotator.

## Labeling Instructions
Open `results/review_round/E6/labels.csv`:
- Inspect each corresponding image in `results/review_round/E6/sample/<frame_id>.png`.
- In the `human_label` column, enter:
  - `1`: The visual scene genuinely contains only a single class across the visible field (e.g. pitch black, unilluminated darkness, or purely void).
  - `0`: The visual scene actually contains objects/road/structures that PIDNet missed or collapsed into a uniform mask.

## Scoring
Once `labels.csv` is filled, run:
```bash
python results/review_round/E6/score.py --csv results/review_round/E6/labels.csv
```
This script computes:
- True Positives ($TP$), False Positives ($FP$), False Negatives ($FN$), True Negatives ($TN$)
- Precision: $TP / (TP + FP)$ with Wilson 95% score confidence interval
- Recall: $TP / (TP + FN)$ with Wilson 95% score confidence interval
- Overall Accuracy with Wilson 95% score confidence interval
