#!/usr/bin/env python3
"""
Evaluates a saved Enformer chromatin-state classifier on a validation dataset.

Computes per-class accuracy and balanced accuracy (average of per-class
accuracies, which gives equal weight to rare and common classes).
Prints intermediate results every --print_every steps so you can monitor
how performance evolves across different genomic regions.

Usage:
    python playground/evaluate_enformer.py \
        --model_path checkpoints/enformer_step_100.pt \
        --data_dir ../sample/binned_dataframe/val_shards \
        --batch_size 1
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Must import the model and dataset classes from the training script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune_enformer import EnformerForSequenceClassification, ShardedChromatinDataset


# Human-readable state names (1-indexed ChromHMM)
STATE_NAMES = [
    "1_TssA",
    "2_TssFlnk",
    "3_TssFlnkU",
    "4_TssFlnkD",
    "5_Tx",
    "6_TxWk",
    "7_EnhG1",
    "8_EnhG2",
    "9_EnhA1",
    "10_EnhA2",
    "11_EnhWk",
    "12_ZNF/Rpts",
    "13_Het",
    "14_TssBiv",
    "15_EnhBiv",
    "16_ReprPC",
    "17_ReprPCWk",
    "18_Quies",
]


def compute_metrics(correct, total):
    """
    Computes per-class accuracy, balanced accuracy, and overall accuracy.
    
    Args:
        correct: 1D array of per-class correct counts
        total: 1D array of per-class total counts
    
    Returns:
        Dict with per_class_acc, balanced_acc, overall_acc
    """
    # Avoid division by zero for missing classes
    per_class_acc = np.zeros_like(correct, dtype=np.float64)
    mask = total > 0
    per_class_acc[mask] = correct[mask] / total[mask]
    
    balanced_acc = per_class_acc[mask].mean() if mask.any() else 0.0
    overall_acc = correct.sum() / total.sum() if total.sum() > 0 else 0.0
    
    return {
        "per_class_acc": per_class_acc,
        "balanced_acc": balanced_acc,
        "overall_acc": overall_acc,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Enformer chromatin-state classifier")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved .pt model checkpoint")
    parser.add_argument("--data_dir", type=str, default="../sample/binned_dataframe/val_shards",
                        help="Directory containing validation parquet shards")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--print_every", type=int, default=50,
                        help="Print intermediate per-class accuracy every N steps")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run evaluation on (cuda or cpu)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load model ---
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Loading model from {model_path}...")
    model = EnformerForSequenceClassification(num_labels=18)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # --- Load validation dataset ---
    data_dir_path = Path(__file__).resolve().parent / args.data_dir
    print(f"Loading validation data from {data_dir_path.resolve()}...")
    dataset = ShardedChromatinDataset(data_dir_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    print(f"Found {len(dataset.files)} parquet file(s) in validation directory.")

    # --- Accumulators ---
    num_classes = 18
    correct = np.zeros(num_classes, dtype=np.int64)
    total = np.zeros(num_classes, dtype=np.int64)
    
    step = 0
    total_bins_processed = 0

    print("\n=== Starting evaluation ===\n")
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)       # [B, 196608]
            labels = labels.to(device)       # [B, 896] - 0-indexed

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(inputs)       # [B, 896, 18]
            
            predictions = logits.argmax(dim=-1)  # [B, 896]

            # Flatten and move to CPU for counting
            preds_flat = predictions.cpu().numpy().ravel()
            labels_flat = labels.cpu().numpy().ravel()

            # Accumulate per-class counts
            for c in range(num_classes):
                mask = labels_flat == c
                correct[c] += (preds_flat[mask] == c).sum()
                total[c] += mask.sum()

            total_bins_processed += len(preds_flat)
            step += 1

            # --- Print intermediate results ---
            if step % args.print_every == 0:
                metrics = compute_metrics(correct, total)
                print(f"--- Step {step} ({total_bins_processed:,} bins processed) ---")
                print(f"  Overall Accuracy:  {metrics['overall_acc']:.4f}")
                print(f"  Balanced Accuracy: {metrics['balanced_acc']:.4f}")
                print(f"  Per-class accuracy:")
                for c in range(num_classes):
                    if total[c] > 0:
                        print(f"    {STATE_NAMES[c]:>14s}: {metrics['per_class_acc'][c]:.4f}  "
                              f"(n={total[c]:,})")
                    else:
                        print(f"    {STATE_NAMES[c]:>14s}:  N/A  (no samples)")
                print()

    # --- Final results ---
    print("=" * 60)
    print("FINAL EVALUATION RESULTS")
    print("=" * 60)
    metrics = compute_metrics(correct, total)
    print(f"Total bins evaluated: {total_bins_processed:,}")
    print(f"Overall Accuracy:     {metrics['overall_acc']:.4f}")
    print(f"Balanced Accuracy:    {metrics['balanced_acc']:.4f}")
    print()
    print("Per-class breakdown (sorted by accuracy):")
    
    # Sort by accuracy descending
    class_results = []
    for c in range(num_classes):
        if total[c] > 0:
            class_results.append((c, metrics['per_class_acc'][c], total[c]))
    class_results.sort(key=lambda x: x[1], reverse=True)
    
    for c, acc, n in class_results:
        print(f"  {STATE_NAMES[c]:>14s}: {acc:.4f}  (n={n:,})")


if __name__ == "__main__":
    main()
