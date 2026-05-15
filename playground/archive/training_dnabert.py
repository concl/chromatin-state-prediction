import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import autocast, GradScaler

# ==============================
# SETTINGS
# ==============================

MODEL_NAME = "zhihan1996/DNA_bert_6"
MAX_LENGTH = 200  # 6-mer model; we will use 200bp input windows
CHUNK_SIZE = 200
CONTEXT_SIZE = 50  # context window around BED region
STEP_SIZE = 50  # overlapping windows for full data coverage
BATCH_SIZE = 64
EPOCHS = 10
NUM_CLASSES = 18  # maximum expected labels (will auto-adjust to actual classes)

SAMPLE_FRACTION = 1.0  # use all BED regions for full run
MAX_EXAMPLES = None  # no cap; process entire dataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", DEVICE)

# ==============================
# FILE PATHS (UPDATED FOR HOFFMAN2)
# ==============================

user_home = os.path.expanduser("~")
FASTA_PATH = "/u/home/a/aparikh/dnabert/chr4.fa"  # validate version with lab
BED_DIR = "/u/project/ernst/ernst/IHEC/FOURCOLS_NOHEADER_BROWSERFILES_ANNOTATIONS_MERGEDBINARY_BYCELLWITHIMPUTED_EPIATLAS_INCLUDEONLY"
TARGET_LIST_PATH = "/u/home/a/aparikh/dnabert/fully_observed_samples.txt"
SAVE_PATH = os.path.join(user_home, "dnabert_results")
os.makedirs(SAVE_PATH, exist_ok=True)

# ==============================
# LOAD CHROMOSOME (FASTA)
# ==============================

print("Loading chromosome...")

dna_chunks = []
with open(FASTA_PATH, "r") as f:
    for line in f:
        if line.startswith(">"):
            continue
        dna_chunks.append(line.strip().upper())

dna_sequence = "".join(dna_chunks)

print("Chromosome length:", len(dna_sequence))

# ==============================
# LOAD BED ANNOTATIONS
# ==============================

print("Loading target list and BED annotations...")

# read required sample IDs
with open(TARGET_LIST_PATH, "r") as f:
    target_ids = {line.strip() for line in f if line.strip()}

print(f"Loaded {len(target_ids)} target IDs")

# find matching BED files in BED_DIR (can be .bed, .bed.gz, .zip)
import glob, zipfile

selected_paths = []
for t in target_ids:
    patterns = [f"{BED_DIR}/**/{t}*.bed", f"{BED_DIR}/**/{t}*.bed.gz", f"{BED_DIR}/**/{t}*.zip"]
    for p in patterns:
        selected_paths.extend(glob.glob(p, recursive=True))

selected_paths = sorted(set(selected_paths))
if len(selected_paths) == 0:
    raise FileNotFoundError(f"No BED/ZIP files found for target IDs in {BED_DIR}")

print(f"Found {len(selected_paths)} files for target IDs")

# read and concatenate selected BED data
frames = []
for path in selected_paths:
    if path.endswith(".zip"):
        with zipfile.ZipFile(path, "r") as zf:
            for member in zf.namelist():
                if member.endswith(".bed") or member.endswith(".bed.gz"):
                    with zf.open(member, "r") as f:
                        frames.append(pd.read_csv(f, sep="\t", header=None, names=["chr", "start", "end", "state"]))
    elif path.endswith(".gz"):
        frames.append(pd.read_csv(path, sep="\t", header=None, names=["chr", "start", "end", "state"], compression="gzip"))
    else:
        frames.append(pd.read_csv(path, sep="\t", header=None, names=["chr", "start", "end", "state"]))

if len(frames) == 0:
    raise ValueError("No usable BED rows were loaded from selected files")

bed_df = pd.concat(frames, ignore_index=True)

# filter to only chromosome 4 regions (since FASTA is chr4.fa)
bed_df = bed_df[bed_df["chr"] == "chr4"].reset_index(drop=True)

if SAMPLE_FRACTION < 1.0:
    bed_df = bed_df.sample(frac=SAMPLE_FRACTION, random_state=42).reset_index(drop=True)
    print(f"Using sample fraction {SAMPLE_FRACTION:.0%}, selected {len(bed_df)} regions.")

# extract numeric state and map to compact label range
bed_df["state_num"] = bed_df["state"].str.split("_", n=1).str[0].astype(int)
unique_state_nums = sorted(bed_df["state_num"].unique())
state_to_label = {s: i for i, s in enumerate(unique_state_nums)}
bed_df["label"] = bed_df["state_num"].map(state_to_label)

print("Total BED regions:", len(bed_df))
print("Unique chromatin states in data:", unique_state_nums)

# ==============================
# BUILD DATASET FROM GENOME
# ==============================

def reverse_complement(seq):
    translation = str.maketrans("ACGTN", "TGCAN")
    return seq.translate(translation)[::-1]


def kmerize(seq, k=6, stride=1):
    return " ".join([seq[i:i+k] for i in range(0, len(seq) - k + 1, stride)])

sequences = []
labels = []

print("Extracting sequences from genome...")

stop_early = False
for _, row in bed_df.iterrows():

    start = int(row["start"])
    end = int(row["end"])
    label = int(row["label"])

    region_start = max(0, start - CONTEXT_SIZE)
    region_end = min(len(dna_sequence), end + CONTEXT_SIZE)

    if region_end - region_start < CHUNK_SIZE:
        # enforce minimal window length by center padding if needed
        center = (start + end) // 2
        region_start = max(0, center - CHUNK_SIZE // 2)
        region_end = min(len(dna_sequence), region_start + CHUNK_SIZE)

    for chunk_start in range(region_start, region_end - CHUNK_SIZE + 1, STEP_SIZE):
        seq = dna_sequence[chunk_start:chunk_start + CHUNK_SIZE].upper()

        # skip sequences with unknown bases
        if "N" in seq or len(seq) < CHUNK_SIZE:
            continue

        # raw 6-merization for DNA_BERT_6 (it treats tokens as 6-mers)
        seq_kmers = kmerize(seq, k=6, stride=1)
        sequences.append(seq_kmers)
        labels.append(label)

        # reverse complement augmentation
        seq_rc = reverse_complement(seq)
        seq_rc_kmers = kmerize(seq_rc, k=6, stride=1)
        sequences.append(seq_rc_kmers)
        labels.append(label)

        if MAX_EXAMPLES is not None and len(sequences) >= MAX_EXAMPLES:
            stop_early = True
            break

    if stop_early:
        break

sequences = np.array(sequences)
labels = np.array(labels)

print("Final dataset size:", len(sequences))

# ==============================
# CLASS WEIGHTS
# ==============================

unique_labels = np.unique(labels)
if len(unique_labels) != NUM_CLASSES:
    print(f"Warning: NUM_CLASSES = {NUM_CLASSES} but only {len(unique_labels)} classes in this sample ({unique_labels}).")

active_num_classes = len(unique_labels)
NUM_CLASSES = active_num_classes

weights = compute_class_weight(
    class_weight="balanced",
    classes=unique_labels,
    y=labels
)

class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

print("Active class labels:", unique_labels)
print("Class weights:", class_weights)

# ==============================
# TOKENIZER
# ==============================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ==============================
# DATASET
# ==============================

class DNADataset(Dataset):

    def __init__(self, sequences, labels, tokenizer, max_length):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.sequences[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

dataset = DNADataset(sequences, labels, tokenizer, MAX_LENGTH)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0  # Windows safe
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ==============================
# MODEL
# ==============================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES
)

model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

scaler = GradScaler()

# ==============================
# TRAINING LOOP
# ==============================

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for batch in train_loader:

        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels_batch = batch["labels"].to(DEVICE)

        with autocast():

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits

            loss = F.cross_entropy(
                logits,
                labels_batch,
                weight=class_weights
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} Loss:", avg_loss)

# ==============================
# EVALUATION
# ==============================

model.eval()

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():

    for batch in val_loader:

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels_batch = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        probs = F.softmax(logits, dim=1)

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# ==============================
# EVALUATION METRICS
# ==============================

mcc = matthews_corrcoef(all_labels, all_preds)
f1_macro = f1_score(all_labels, all_preds, average="macro")

print(f"Matthews Correlation Coefficient: {mcc:.4f}")
print(f"Macro F1 Score: {f1_macro:.4f}")

# save metrics to file
with open(os.path.join(SAVE_PATH, "evaluation_metrics.txt"), "w") as f:
    f.write(f"Matthews Correlation Coefficient: {mcc:.4f}\n")
    f.write(f"Macro F1 Score: {f1_macro:.4f}\n")
    f.write("Confidence stats:\n")
    f.write(f"Average: {confidences.mean()}\n")
    f.write(f"Min: {confidences.min()}\n")
    f.write(f"Max: {confidences.max()}\n")

# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Chromatin State Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(SAVE_PATH, "confusion_matrix.png"))
plt.show()

# ==============================
# CONFIDENCE ANALYSIS
# ==============================

all_probs = np.array(all_probs)
confidences = np.max(all_probs, axis=1)

print("Confidence stats:")
print("Average:", confidences.mean())
print("Min:", confidences.min())
print("Max:", confidences.max())

plt.hist(confidences, bins=40)
plt.title("Prediction Confidence Distribution")
plt.xlabel("Softmax Probability")
plt.ylabel("Frequency")
plt.savefig(os.path.join(SAVE_PATH, "confidence_histogram.png"))
plt.show()

# ==============================
# SAVE MODEL
# ==============================

torch.save(model.state_dict(), os.path.join(SAVE_PATH, "dnabert_chromatin_model.pt"))

print("Training complete.")