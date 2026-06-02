#!/usr/bin/env python3
"""Run genome-wide chromatin-state inference with a finetuned Enformer model.

Slides the Enformer model across every human chromosome, predicts 18-class
ChromHMM chromatin states for each 128 bp bin, merges adjacent same-state
bins into BED intervals, and writes the result as a gzipped BED file.

Usage:

    python playground/enformer_inference.py \
        --model_path enformer_finetuned.pt \
        --genome_dir sample/human_genome \
        --output_file sample/predictions.bed.gz \
        --device cuda

The output BED file follows the same format as the ChromHMM annotation files
(``chrom``, ``start``, ``end``, ``state_name``).
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import pandas as pd
import torch

# Ensure the project root is on sys.path so we can import from templates
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from templates.data import (
    CHROMOSOMES,
    DOWNLOAD_PATH,
    STATE_NAMES,
    decompress_chromosome,
    merge_bed_intervals,
)
from templates.enformer_trainer import EnformerForSequenceClassification


def build_model(model_path: str | Path, device: torch.device) -> torch.nn.Module:
    """Load the finetuned Enformer classifier from a checkpoint.

    Args:
        model_path: Path to the ``.pt`` state-dict checkpoint.
        device: Torch device to load the model onto.

    Returns:
        The model in evaluation mode.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print(f"Loading model from {model_path} ...")
    model = EnformerForSequenceClassification(num_labels=18)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("  Model loaded successfully.")
    return model

def predict_on_chromosome(
    model,
    chrom: str,
    seq: str,
    device,
    window_size: int = 196608,
    bin_size: int = 128,
    mixed_precision: bool = True,
):
    """Slide Enformer windows across a chromosome and yield per-bin predictions.

    Enformer takes 196,608 bp of context and predicts 896 output bins (each
    covering ``bin_size`` bp) for the central 114,688 bp.  Windows are advanced
    by ``896 * bin_size`` (the effective prediction span) to produce contiguous
    non-overlapping predictions across the chromosome.

    Predictions for the first ~40 kbp and last ~40 kbp of the chromosome are
    omitted because they fall outside the central prediction window.

    Args:
        model: An ``EnformerForSequenceClassification`` instance (or
            ``nn.Module``) that accepts ``[B, window_size]`` integer-encoded
            DNA and returns ``[B, 896, num_classes]`` logits.
        chrom: Chromosome name (for logging / coordinate tracking).
        seq: Uppercase genomic sequence string for the chromosome.
        device: Torch device to run inference on.
        window_size: Enformer input width in bp.  Default 196,608.
        bin_size: Output bin width in bp.  Default 128.
        mixed_precision: If True, use ``torch.amp.autocast`` with bfloat16.

    Yields:
        Tuples of ``(chrom, abs_start, abs_end, state_1indexed)`` where
        ``state_1indexed`` is an integer in ``[1, 18]``.
    """

    n_bins = 896  # Enformer output bins
    stride = n_bins * bin_size  # 114,688 bp – the effective prediction span
    flank = (window_size - stride) // 2  # 40,960 bp of context on each side

    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}

    seq_len = len(seq)

    for w_start in range(0, seq_len - window_size + 1, stride):
        w_end = w_start + window_size
        chunk = seq[w_start:w_end]

        # Integer-encode
        ids = [mapping.get(base, 4) for base in chunk]
        tensor = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, 196608]

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=mixed_precision):
                logits = model(tensor)  # [1, 896, 18]
            preds = logits.argmax(dim=-1).squeeze(0)  # [896]

        # Map each output bin to absolute genomic coordinates
        pred_start = w_start + flank
        for i, cls_idx in enumerate(preds.cpu().numpy()):
            abs_start = pred_start + i * bin_size
            abs_end = abs_start + bin_size
            yield chrom, abs_start, abs_end, int(cls_idx) + 1  # 1-indexed



def run_genome_inference(
    model: torch.nn.Module,
    chromosomes: list[str],
    genome_dir: Path,
    device: torch.device,
) -> pd.DataFrame:
    """Run inference on all chromosomes and return merged BED annotations.

    Args:
        model: The finetuned Enformer classifier.
        chromosomes: List of chromosome names (e.g. ``["chr1", "chr2", ...]``).
        genome_dir: Directory containing ``chrN.fa.gz`` FASTA files.
        device: Torch device.

    Returns:
        DataFrame with columns ``chrom``, ``start``, ``end``, ``state``,
        ``name`` containing merged BED-format annotations for all
        chromosomes.
    """
    all_dfs: list[pd.DataFrame] = []

    for chrom in chromosomes:
        print(f"\n{'='*60}")
        print(f"Processing {chrom} ...")

        # Load genomic sequence
        gz_path = genome_dir / f"{chrom}.fa.gz"
        if not gz_path.exists():
            print(f"  Skipping {chrom}: FASTA not found at {gz_path}")
            continue

        fasta_str = decompress_chromosome(chrom)
        seq = "".join(fasta_str.split("\n")[1:]).upper()
        print(f"  Sequence length: {len(seq):,} bp")

        # Run sliding-window inference
        n_bins = 0
        n_windows = 0
        BINS_PER_WINDOW = 896
        preds_for_chrom: list[tuple] = []

        for pred in predict_on_chromosome(model, chrom, seq, device):
            preds_for_chrom.append(pred)
            n_bins += 1
            if n_bins % BINS_PER_WINDOW == 0:
                n_windows += 1

            # Progress every 50 windows (~5.7 Mbp)
            if n_bins % (50 * BINS_PER_WINDOW) == 0:
                bp_done = n_windows * BINS_PER_WINDOW * 128
                pct = min(100.0, bp_done / len(seq) * 100)
                print(f"  ... {n_windows} windows processed "
                      f"({bp_done / 1e6:.1f} Mbp, {pct:.1f}%)")

        n_windows = n_bins // BINS_PER_WINDOW
        print(f"  {chrom}: {n_windows} windows → "
              f"{n_bins:,} raw bins")

        if not preds_for_chrom:
            print(f"  No predictions for {chrom}, skipping.")
            continue

        # Merge adjacent bins into compact BED intervals
        chrom_df = merge_bed_intervals(preds_for_chrom, state_names=STATE_NAMES)
        print(f"  After merging: {len(chrom_df):,} intervals")
        all_dfs.append(chrom_df)

    if not all_dfs:
        raise RuntimeError("No predictions were generated for any chromosome.")

    result = pd.concat(all_dfs, ignore_index=True)
    return result


def write_bed_output(df: pd.DataFrame, output_path: Path) -> None:
    """Write merged predictions as a gzipped BED file.

    The output format is: ``chrom<TAB>start<TAB>end<TAB>state_name`` matching
    the ChromHMM annotation BED files.

    Args:
        df: DataFrame with columns ``chrom``, ``start``, ``end``, ``name``.
        output_path: Destination path (``.bed.gz`` suffix recommended).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write using the same format as the ChromHMM BED files
    bed_df = df[["chrom", "start", "end", "name"]].copy()
    bed_df["start"] = bed_df["start"].astype(int)
    bed_df["end"] = bed_df["end"].astype(int)

    print(f"\nWriting {len(bed_df):,} intervals to {output_path} ...")

    with gzip.open(output_path, "wt") as f:
        bed_df.to_csv(f, sep="\t", header=False, index=False)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Genome-wide chromatin-state inference with finetuned Enformer"
    )
    parser.add_argument(
        "--model_path", type=str,
        default="enformer_finetuned.pt",
        help="Path to the finetuned Enformer .pt checkpoint",
    )
    parser.add_argument(
        "--genome_dir", type=str,
        default=None,
        help="Directory containing chrN.fa.gz files (default: sample/human_genome)",
    )
    parser.add_argument(
        "--output_file", type=str,
        default="predictions.bed.gz",
        help="Path for the output gzipped BED file",
    )
    parser.add_argument(
        "--chromosomes", type=str, nargs="*", default=None,
        help="Chromosomes to process (default: chr1-22, chrX)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for inference (cuda or cpu)",
    )
    args = parser.parse_args()

    # Resolve paths relative to the script's directory
    base = Path(__file__).resolve().parent
    model_path = base / args.model_path if not Path(args.model_path).is_absolute() else Path(args.model_path)
    genome_dir = (
        Path(args.genome_dir) if args.genome_dir
        else DOWNLOAD_PATH
    )
    output_path = (
        base / args.output_file if not Path(args.output_file).is_absolute()
        else Path(args.output_file)
    )

    chromosomes = args.chromosomes if args.chromosomes else CHROMOSOMES
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Chromosomes to process: {chromosomes}")
    print(f"Genome directory: {genome_dir}")

    # Build model
    model = build_model(model_path, device)

    # Run inference
    result_df = run_genome_inference(model, chromosomes, genome_dir, device)

    # Write output
    write_bed_output(result_df, output_path)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total intervals: {len(result_df):,}")
    print(f"Output: {output_path.resolve()}")


if __name__ == "__main__":
    main()
