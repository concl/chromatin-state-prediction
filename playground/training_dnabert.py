
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.amp import autocast, GradScaler

# Config
SEQ_FILE   = "C:/Users/aravp/Internships/Ernst/chr4_sampled_20Mb_block.csv"
LABEL_FILE = "C:/Users/aravp/Internships/Ernst/chr4_sampled_20Mb_chromatin_states.csv"

MODEL_NAME = "zhihan1996/DNA_bert_6"

NUM_CLASSES = 19
BATCH_SIZE = 8
EPOCHS = 10
LR = 2e-5
MAX_LEN = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

scaler = GradScaler("cpu")  # CPU-friendly

CHECKPOINT_DIR = "./dnabert_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
LOG_EVERY_N_BATCHES = 4  # prints progress every 10 batches

# Reverse Complement
def reverse_complement(seq):
    complement = {"A":"T", "T":"A", "C":"G", "G":"C"}
    return "".join(complement.get(base, base) for base in reversed(seq))

# Load Data 
seq_df = pd.read_csv(SEQ_FILE, header=None, names=["sequence"])
lab_df = pd.read_csv(LABEL_FILE, header=None, names=["label"])

df = pd.concat([seq_df, lab_df], axis=1)
df = df.dropna()
df["sequence"] = df["sequence"].astype(str)
df["label"] = df["label"] - 1  # convert labels to 0-index

# Data Augmentation (Reverse Complement)
print("Original dataset size:", len(df))
rev_df = df.copy()
rev_df["sequence"] = rev_df["sequence"].apply(reverse_complement)
df = pd.concat([df, rev_df])
print("After reverse complement augmentation:", len(df))


train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# Oversampling Rare Classes
label_counts = train_df["label"].value_counts()
weights = 1 / label_counts
sample_weights = train_df["label"].map(weights).values
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Dataset
class DNADataset(Dataset):
    def __init__(self, df):
        self.seqs = df["sequence"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        label = self.labels[idx]
        encoding = tokenizer(
            seq,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(label).long()
        return item

train_dataset = DNADataset(train_df)
val_dataset   = DNADataset(val_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE
)

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES
)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# LR Scheduler
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

criterion = nn.CrossEntropyLoss()

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

    for batch_idx, batch in enumerate(train_loader, 1):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        with autocast(device_type="cpu"):  # CPU-friendly autocast
            outputs = model(**batch)
            loss = criterion(outputs.logits, batch["labels"])

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

        if batch_idx % LOG_EVERY_N_BATCHES == 0:
            avg_batch_loss = total_loss / batch_idx
            print(f"Batch {batch_idx}/{len(train_loader)} | Avg Loss: {avg_batch_loss:.4f}")

    avg_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    print(f"\nEpoch {epoch+1}/{EPOCHS} Summary")
    print(f"Train Loss: {avg_loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")

    # Save checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"dnabert_epoch{epoch+1}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    ###