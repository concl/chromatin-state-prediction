import os
import re
import gc
import glob
import h5py
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# =====================
# CONFIG
# =====================
DATA_DIR = "/u/scratch/a/aparikh/dnabert_embeddings"
SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 8192
EPOCHS = 20                # CHANGED: 5 -> 20, minority classes need more time
LR = 3e-4
NUM_CLASSES = 18
INPUT_DIM = 768
FOCAL_GAMMA = 2.0          # CHANGED: focal loss gamma, 2.0 is standard starting point

FILES_PER_CHUNK = 8
VAL_SPLIT = 0.10
NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("DEVICE:", DEVICE)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("=" * 60)

# ==========================================================
# NUMERIC SORTING
# ==========================================================
def extract_num(path):
    name = os.path.basename(path)
    match = re.search(r"embedding_part_(\d+)\.h5", name)
    return int(match.group(1))

all_files = sorted(
    glob.glob(os.path.join(DATA_DIR, "embedding_part_*.h5")),
    key=extract_num
)

if len(all_files) == 0:
    raise FileNotFoundError("No embedding files found.")

print(f"Found {len(all_files)} files")

np.random.seed(42)
np.random.shuffle(all_files)

split_idx = int(len(all_files) * (1 - VAL_SPLIT))
train_files = all_files[:split_idx]
val_files = all_files[split_idx:]

print(f"Train files: {len(train_files)}")
print(f"Val files:   {len(val_files)}")

# ==========================================================
# LABEL SCAN FOR CLASS WEIGHTS
# ==========================================================
print("\nScanning labels for class weights...")
class_counts = np.zeros(NUM_CLASSES, dtype=np.float64)

for fp in train_files:
    try:
        with h5py.File(fp, "r") as h5:
            if "labels" not in h5:
                continue
            labels = h5["labels"][:]
            class_counts += np.bincount(labels, minlength=NUM_CLASSES)
    except Exception as e:
        print(f"Skipping {fp}: {e}")

print("Class counts:")
for i, c in enumerate(class_counts):
    if c > 0:
        print(f"  Class {i:2d}: {int(c):>10,} ({100*c/class_counts.sum():.1f}%)")

# Sqrt weighting — gentle, avoids gradient explosion on tiny classes
class_weights = 1.0 / np.sqrt(np.where(class_counts == 0, 1.0, class_counts))
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
class_weights_tensor = torch.from_numpy(class_weights).float().to(DEVICE)

# ==========================================================
# CHANGED: Focal Loss
# -- Downweights easy/confident predictions (class 17 mostly)
# -- Forces model to spend gradient budget on hard minority cases
# -- gamma=2.0 is standard; higher = more focus on hard examples
# ==========================================================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Per-sample CE loss (unreduced)
        ce_loss = nn.CrossEntropyLoss(weight=self.weight, reduction='none')(inputs, targets)
        # Probability of the correct class
        pt = torch.exp(-ce_loss)
        # Focal modulation: downweight easy examples
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

# ==========================================================
# CHANGED: Wider + deeper architecture
# -- 768 -> 1024 -> 512 -> 256 -> 18
# -- More capacity to separate overlapping minority embeddings
# ==========================================================
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_CLASSES)
        )
    def forward(self, x): return self.net(x)

model = SimpleClassifier().to(DEVICE)

criterion = FocalLoss(weight=class_weights_tensor, gamma=FOCAL_GAMMA)
optimizer = optim.Adam(model.parameters(), lr=LR)

# CHANGED: LR scheduler — halves LR when macro F1 stops improving
# patience=2 means it waits 2 epochs before reducing
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2, verbose=True
)

# ==========================================================
# HELPERS
# ==========================================================
def load_chunk(file_list):
    xs, ys = [], []
    for fp in file_list:
        try:
            with h5py.File(fp, "r") as h5:
                xs.append(h5["embeddings"][:])
                ys.append(h5["labels"][:])
        except:
            continue
    if not xs:
        return None, None
    return np.vstack(xs).astype(np.float32), np.concatenate(ys).astype(np.int64)

def make_loader(X, Y, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True)

def run_validation():
    model.eval()
    total_loss, total_seen, val_batches = 0, 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, len(val_files), FILES_PER_CHUNK):
            X, Y = load_chunk(val_files[i:i + FILES_PER_CHUNK])
            if X is None:
                continue
            loader = make_loader(X, Y, shuffle=False)
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                total_loss += criterion(logits, yb).item()
                val_batches += 1
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(yb.cpu().numpy())
                total_seen += yb.size(0)
            del X, Y, loader
            gc.collect()

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    avg_loss     = total_loss / max(1, val_batches)
    accuracy     = 100 * (all_preds == all_labels).sum() / max(1, total_seen)
    macro_f1     = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    per_class_recall = {}
    for c in range(NUM_CLASSES):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class_recall[c] = 100 * (all_preds[mask] == c).sum() / mask.sum()

    return avg_loss, accuracy, macro_f1, balanced_acc, per_class_recall

# ==========================================================
# MAIN LOOP
# ==========================================================
best_macro_f1 = -1.0

for epoch in range(1, EPOCHS + 1):
    print(f"\n{'='*60}\nEPOCH {epoch}/{EPOCHS}\n{'='*60}")
    model.train()
    total_train_loss, batches_seen = 0, 0
    np.random.shuffle(train_files)

    for i in range(0, len(train_files), FILES_PER_CHUNK):
        X, Y = load_chunk(train_files[i:i + FILES_PER_CHUNK])
        if X is None:
            continue
        perm = np.random.permutation(len(Y))
        loader = make_loader(X[perm], Y[perm], shuffle=False)
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            batches_seen += 1
        del X, Y, loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    val_loss, accuracy, macro_f1, balanced_acc, per_class_recall = run_validation()

    print(f"Train Loss:    {total_train_loss/max(1, batches_seen):.4f}")
    print(f"Val Loss:      {val_loss:.4f}")
    print(f"Accuracy:      {accuracy:.2f}%")
    print(f"Macro F1:      {macro_f1:.4f}")
    print(f"Balanced Acc:  {balanced_acc:.4f}")
    print(f"Current LR:    {optimizer.param_groups[0]['lr']:.2e}")
    print("Per-class recall:")
    for c, r in per_class_recall.items():
        print(f"  Class {c:2d}: {r:.1f}%")

    epoch_save_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch}.pt")
    torch.save(model.state_dict(), epoch_save_path)
    print(f"  -> Saved epoch {epoch} checkpoint.")

    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pt"))
        print(f"  -> New best model saved (Macro F1: {best_macro_f1:.4f})")

    # CHANGED: step scheduler on macro F1 — reduces LR when improvement stalls
    scheduler.step(macro_f1)

print("\nTraining Complete.")