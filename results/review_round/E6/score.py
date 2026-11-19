"""
E6 Scoring Script: Precision, Recall and Wilson 95% Confidence Intervals
Usage:
    python score.py [--csv labels.csv]
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd

def wilson_score_interval(successes, total, conf=0.95):
    """
    Computes Wilson score confidence interval for a binomial proportion.
    """
    if total == 0:
        return 0.0, 0.0, 0.0
    z = 1.959963984540054  # 95% confidence
    p = successes / total
    denominator = 1 + (z**2) / total
    center = (p + (z**2) / (2 * total)) / denominator
    spread = (z / denominator) * np.sqrt((p * (1 - p) / total) + (z**2) / (4 * (total**2)))
    low = max(0.0, center - spread)
    high = min(1.0, center + spread)
    return p, low, high

def main():
    parser = argparse.ArgumentParser(description="Compute precision and recall with Wilson score CIs")
    parser.add_argument("--csv", type=str, default="results/review_round/E6/labels.csv", help="Path to labelled CSV")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        sys.exit(1)

    df = pd.read_csv(args.csv)

    # Check columns
    required_cols = {"frame_id", "flagged", "human_label"}
    if not required_cols.issubset(df.columns):
        print(f"Error: CSV missing required columns. Found: {df.columns.tolist()}, Required: {required_cols}")
        sys.exit(1)

    # Check if human_label is populated
    valid_mask = df["human_label"].notna() & (df["human_label"].astype(str).str.strip() != "")
    if valid_mask.sum() == 0:
        print("="*60)
        print("E6 EVALUATION NOTICE:")
        print("The human_label column is currently empty.")
        print("Per protocol, precision and recall numbers will exist ONLY after human labelling.")
        print(f"Total unlabelled rows: {len(df)}")
        print("="*60)
        return

    df_eval = df[valid_mask].copy()
    # Accept 1/0, true/false, uniform/non-uniform
    def parse_label(val):
        s = str(val).strip().lower()
        if s in ["1", "true", "yes", "uniform", "single_class", "positive"]:
            return 1
        elif s in ["0", "false", "no", "non_uniform", "multi_class", "negative"]:
            return 0
        else:
            try:
                return int(float(s))
            except:
                return np.nan

    df_eval["y_true"] = df_eval["human_label"].apply(parse_label)
    df_eval["y_pred"] = df_eval["flagged"].astype(int)

    # Drop unparseable
    df_eval = df_eval.dropna(subset=["y_true"])
    n_eval = len(df_eval)

    if n_eval == 0:
        print("No valid numeric labels found in human_label.")
        return

    y_true = df_eval["y_true"].astype(int).values
    y_pred = df_eval["y_pred"].values

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    prec, prec_low, prec_high = wilson_score_interval(tp, tp + fp)
    rec, rec_low, rec_high = wilson_score_interval(tp, tp + fn)
    acc, acc_low, acc_high = wilson_score_interval(tp + tn, n_eval)

    print("="*60)
    print(f"E6 RESULTS: Single-Class (48-bit) Flag Verification (N={n_eval})")
    print("="*60)
    print(f"Confusion Matrix:")
    print(f"  TP: {tp:<6} FP: {fp}")
    print(f"  FN: {fn:<6} TN: {tn}")
    print("-" * 60)
    print(f"Precision: {prec*100:6.2f}%  [95% Wilson CI: {prec_low*100:5.2f}%, {prec_high*100:5.2f}%] (TP / (TP + FP))")
    print(f"Recall:    {rec*100:6.2f}%  [95% Wilson CI: {rec_low*100:5.2f}%, {rec_high*100:5.2f}%] (TP / (TP + FN))")
    print(f"Accuracy:  {acc*100:6.2f}%  [95% Wilson CI: {acc_low*100:5.2f}%, {acc_high*100:5.2f}%]")
    print("="*60)

if __name__ == "__main__":
    main()
