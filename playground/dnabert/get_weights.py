import h5py
import numpy as np
import glob
import os

# Updated for Scratch
DATA_DIR = "/u/scratch/a/aparikh/dnabert_results"

h5_files = sorted(glob.glob(os.path.join(DATA_DIR, "processed_part_*.h5")))

label_counts = {}

print("Counting labels...")

for file in h5_files:
    print("Processing:", file)

    with h5py.File(file, "r") as f:
        labels = f["labels"]

        for i in range(0, len(labels), 100_000):
            chunk = labels[i:i+100_000]

            # 🚀 FAST VERSION (NumPy)
            vals, counts = np.unique(chunk, return_counts=True)

            for v, c in zip(vals, counts):
                label_counts[v] = label_counts.get(v, 0) + int(c)

# compute weights
unique_labels = np.array(sorted(label_counts.keys()))
counts = np.array([label_counts[l] for l in unique_labels])

weights = len(counts) / (len(unique_labels) * counts)

print("\n=== COPY THESE INTO YOUR ML SCRIPT ===")
print("labels:", unique_labels.tolist())
print("weights:", weights.tolist())