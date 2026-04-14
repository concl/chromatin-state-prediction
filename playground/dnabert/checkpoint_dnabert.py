import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import autocast, GradScaler
import h5py
import glob

# ==============================
# SETTINGS
# ==============================

DEBUG = True  # Set to False for full training

MODEL_NAME = "zhihan1996/DNA_bert_6"
BATCH_SIZE = 16 if DEBUG else 64
EPOCHS = 1 if DEBUG else 10
CHECKPOINT_EVERY = 500  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", DEVICE)

# ==============================
# PATHS
# ==============================

user_home = os.path.expanduser("~")

DATA_DIR = "/u/scratch/a/aparikh/dnabert_results"
SAVE_PATH = os.path.join(user_home, "dnabert_results")
os.makedirs(SAVE_PATH, exist_ok=True)

CHECKPOINT_PATH = os.path.join(SAVE_PATH, "dnabert_checkpoint.pt")

# ==============================
# FIND ALL H5 FILES
# ==============================

all_h5_files = sorted(glob.glob(os.path.join(DATA_DIR, "processed_part_*.h5")))

h5_files = all_h5_files[::10]  # 10% subset

if len(h5_files) == 0:
    raise FileNotFoundError("No HDF5 data files found")

print(f"Total files available: {len(all_h5_files)}")
print(f"Subsampling active: Using {len(h5_files)} files (10% subset)")

# ==============================
# HARD-CODED CLASS WEIGHTS
# ==============================

unique_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                          10, 11, 12, 13, 14, 15, 16, 17])

weights = np.array([
    2.4049, 4.8066, 3.6084, 4.4506, 0.2170,
    0.0680, 1.8119, 9.6210, 1.2433, 1.3915,
    0.3274, 0.4415, 0.0604, 7.3748, 6.8031,
    0.5283, 0.0622, 0.0100
])

NUM_CLASSES = len(unique_labels)
class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

print("Using precomputed class weights")

# ==============================
# DATASET
# ==============================

class HDF5Dataset(Dataset):
    def __init__(self, h5_files):
        self.file_handles = [h5py.File(f, "r") for f in h5_files]
        self.lengths = [len(f["labels"]) for f in self.file_handles]
        self.cumulative = np.cumsum(self.lengths)

    def __len__(self):
        return int(self.cumulative[-1])

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cumulative, idx, side="right")
        prev = 0 if file_idx == 0 else self.cumulative[file_idx - 1]
        local_idx = idx - prev

        f = self.file_handles[file_idx]

        return {
            "input_ids": torch.tensor(f["input_ids"][local_idx], dtype=torch.long),
            "attention_mask": torch.tensor(f["attention_mask"][local_idx], dtype=torch.long),
            "labels": torch.tensor(f["labels"][local_idx], dtype=torch.long)
        }

dataset = HDF5Dataset(h5_files)

print("Total dataset size:", len(dataset))

# ==============================
# TRAIN / VAL SPLIT
# ==============================

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
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
# LOAD CHECKPOINT
# ==============================

start_epoch = 0
start_batch = 0

if os.path.exists(CHECKPOINT_PATH):
    print("Resuming from checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]
    start_batch = checkpoint.get("batch_idx", 0)

    print(f"Resuming at epoch {start_epoch}, batch {start_batch}")

# ==============================
# TRAINING LOOP
# ==============================

for epoch in range(start_epoch, EPOCHS):

    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):

        # 🔥 skip already processed batches
        if epoch == start_epoch and batch_idx < start_batch:
            continue

        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels_batch = batch["labels"].to(DEVICE, non_blocking=True)

        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = F.cross_entropy(logits, labels_batch, weight=class_weights)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # 🔥 checkpoint every N batches
        if batch_idx % CHECKPOINT_EVERY == 0:
            torch.save({
                "epoch": epoch,
                "batch_idx": batch_idx,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, CHECKPOINT_PATH)

            print(f"Checkpoint saved at epoch {epoch}, batch {batch_idx}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} Loss:", avg_loss)

    # reset batch counter
    start_batch = 0

    # save end-of-epoch checkpoint
    torch.save({
        "epoch": epoch,
        "batch_idx": 0,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }, CHECKPOINT_PATH)

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

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        probs = F.softmax(logits, dim=1)

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# ==============================
# METRICS
# ==============================

all_probs = np.array(all_probs)
confidences = np.max(all_probs, axis=1)

mcc = matthews_corrcoef(all_labels, all_preds)
f1_macro = f1_score(all_labels, all_preds, average="macro")

print(f"MCC: {mcc:.4f}")
print(f"F1 Macro: {f1_macro:.4f}")

# ==============================
# SAVE RESULTS
# ==============================

with open(os.path.join(SAVE_PATH, "evaluation_metrics.txt"), "w") as f:
    f.write(f"MCC: {mcc:.4f}\n")
    f.write(f"F1 Macro: {f1_macro:.4f}\n")
    f.write(f"Confidence Avg: {confidences.mean()}\n")

# ==============================
# SAVE MODEL
# ==============================

torch.save(model.state_dict(), os.path.join(SAVE_PATH, "dnabert_final.pt"))

print("Training complete.")