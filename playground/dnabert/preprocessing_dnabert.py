import os
import pandas as pd
import numpy as np
import glob, zipfile
from transformers import AutoTokenizer
import h5py

# ==============================
# SETTINGS
# ==============================

DEBUG = False

MODEL_NAME = "zhihan1996/DNA_bert_6"
MAX_LENGTH = 200
CHUNK_SIZE = 200
CONTEXT_SIZE = 50
STEP_SIZE = 50

CHUNK_SAVE_SIZE = 100_000
FILES_PER_PART = 20  # number of chunks per file

# ==============================
# PATHS
# ==============================

user_home = os.path.expanduser("~")

FASTA_PATH = "/u/home/a/aparikh/dnabert/chr4.fa"
BED_DIR = "/u/project/ernst/ernst/IHEC/FOURCOLS_NOHEADER_BROWSERFILES_ANNOTATIONS_MERGEDBINARY_BYCELLWITHIMPUTED_EPIATLAS_INCLUDEONLY"
TARGET_LIST_PATH = "/u/home/a/aparikh/dnabert/fully_observed_samples.txt"

SAVE_PATH = os.path.join(user_home, "dnabert_results")
os.makedirs(SAVE_PATH, exist_ok=True)

# ==============================
# LOAD CHROMOSOME
# ==============================

print("Loading chromosome...")

dna_chunks = []
with open(FASTA_PATH, "r") as f:
    f.readline()
    for line in f:
        dna_chunks.append(line.strip().upper())
dna_sequence = "".join(dna_chunks)

print("Chromosome length:", len(dna_sequence))

# ==============================
# LOAD BED (FILTER EARLY)
# ==============================

print("Loading BED files...")

with open(TARGET_LIST_PATH, "r") as f:
    target_ids = {line.strip() for line in f}

selected_paths = []
for t in target_ids:
    selected_paths.extend(glob.glob(f"{BED_DIR}/**/{t}*.bed*", recursive=True))

frames = []

for path in selected_paths:
    try:
        if path.endswith(".gz"):
            df = pd.read_csv(path, sep="\t", header=None,
                             names=["chr","start","end","state"],
                             compression="gzip")
        else:
            df = pd.read_csv(path, sep="\t", header=None,
                             names=["chr","start","end","state"])
        df = df[df["chr"] == "chr4"]
        if not df.empty:
            frames.append(df)
    except:
        continue

bed_df = pd.concat(frames, ignore_index=True)

# labels
bed_df["state_num"] = bed_df["state"].str.split("_", n=1).str[0].astype(int)
state_to_label = {s:i for i,s in enumerate(sorted(bed_df["state_num"].unique()))}
bed_df["label"] = bed_df["state_num"].map(state_to_label)

print("Total regions:", len(bed_df))

# ==============================
# HELPERS
# ==============================

def reverse_complement(seq):
    return seq.translate(str.maketrans("ACGTN","TGCAN"))[::-1]

def kmerize(seq, k=6):
    return " ".join([seq[i:i+k] for i in range(len(seq)-k+1)])

# ==============================
# RESUME LOGIC
# ==============================

progress_file = os.path.join(SAVE_PATH, "progress.txt")

if os.path.exists(progress_file):
    with open(progress_file) as f:
        start_idx = int(f.read().strip())
else:
    start_idx = 0

print("Resuming from row:", start_idx)

# ==============================
# PROCESSING
# ==============================

starts = bed_df["start"].values
ends = bed_df["end"].values
labs = bed_df["label"].values

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

chunk_sequences = []
chunk_labels = []

part_idx = start_idx // (CHUNK_SAVE_SIZE * FILES_PER_PART)
chunk_counter = 0

def save_part(seqs, labels, part_idx):
    print(f"Saving part {part_idx} with {len(seqs)} samples")

    enc = tokenizer(
        seqs,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="np"
    )

    path = os.path.join(SAVE_PATH, f"processed_part_{part_idx}.h5")

    with h5py.File(path, "w") as f:
        f.create_dataset("input_ids", data=enc["input_ids"], compression="gzip")
        f.create_dataset("attention_mask", data=enc["attention_mask"], compression="gzip")
        f.create_dataset("labels", data=np.array(labels), compression="gzip")

# ==============================
# MAIN LOOP
# ==============================

for i in range(start_idx, len(bed_df)):

    if i % 10000 == 0:
        print(f"{i}/{len(bed_df)}")

    start, end, label = int(starts[i]), int(ends[i]), int(labs[i])

    region_start = max(0, start - CONTEXT_SIZE)
    region_end = min(len(dna_sequence), end + CONTEXT_SIZE)

    for chunk_start in range(region_start, region_end - CHUNK_SIZE + 1, STEP_SIZE):

        seq = dna_sequence[chunk_start:chunk_start + CHUNK_SIZE]

        if "N" in seq:
            continue

        chunk_sequences.append(kmerize(seq))
        chunk_labels.append(label)

        chunk_sequences.append(kmerize(reverse_complement(seq)))
        chunk_labels.append(label)

        if len(chunk_sequences) >= CHUNK_SAVE_SIZE:

            save_part(chunk_sequences, chunk_labels, part_idx)

            part_idx += 1
            chunk_counter += 1

            chunk_sequences = []
            chunk_labels = []

            # save progress
            with open(progress_file, "w") as f:
                f.write(str(i))

# ==============================
# FINAL SAVE
# ==============================

if chunk_sequences:
    save_part(chunk_sequences, chunk_labels, part_idx)

print("Done.")