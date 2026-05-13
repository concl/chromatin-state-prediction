import os
import re
import gc
import glob
import h5py
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score
from sklearn.calibration import calibration_curve

# =====================
# CONFIG
# =====================
DATA_DIR = "/u/scratch/a/aparikh/dnabert_embeddings"
CHECKPOINT_PATH = "./checkpoints/best_model.pt" # Weighted checkpoint

BATCH_SIZE = 8192
NUM_CLASSES = 18
INPUT_DIM = 768
FILES_PER_CHUNK = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_CLASS = 17   # state 18

# =====================
# MODEL (Weighted Architecture)
# =====================
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

    def forward(self, x):
        return self.net(x)

# =====================
# HELPERS
# =====================
def extract_num(path):
    m = re.search(r"embedding_part_(\d+)\.h5", os.path.basename(path))
    return int(m.group(1))

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

def make_loader(X, Y):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

def evaluate_predictions(name, preds, labels):
    bal_acc = balanced_accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    s18_recall = (((preds == 17) & (labels == 17)).sum() / max(1, (labels == 17).sum()))
    print(f"{name:35s} | BalAcc={bal_acc:.4f} | MacroF1={macro_f1:.4f} | S18Recall={s18_recall:.4f}")

# =====================
# LOAD MODEL & RUN INFERENCE
# =====================
model = SimpleClassifier().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

all_files = sorted(glob.glob(os.path.join(DATA_DIR, "embedding_part_*.h5")), key=extract_num)
np.random.seed(42)
np.random.shuffle(all_files)
val_files = all_files[int(len(all_files)*0.9):]

all_probs, all_labels = [], []
with torch.no_grad():
    for i in range(0, len(val_files), FILES_PER_CHUNK):
        X, Y = load_chunk(val_files[i:i+FILES_PER_CHUNK])
        if X is None: continue
        loader = make_loader(X, Y)
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            probs = torch.softmax(model(xb), dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(yb.numpy())
        del X, Y, loader
        gc.collect()

all_probs = np.vstack(all_probs)
all_labels = np.concatenate(all_labels)
baseline_preds = np.argmax(all_probs, axis=1)

# =====================
# CALIBRATION
# =====================
y_true_binary = (all_labels == 17).astype(int)
prob_true, prob_pred = calibration_curve(y_true_binary, all_probs[:, 17], n_bins=15, strategy='quantile')
plt.figure(figsize=(6,6))
plt.plot(prob_pred, prob_true, marker='o', label='Weighted Model')
plt.plot([0,1], [0,1], linestyle='--')
plt.title("Calibration curve for State 18 (Weighted)")
plt.savefig("weighted_class17_calibration.png")

# =====================
# ALTERNATIVE DECODING
# =====================
print("\n" + "="*70)
print("DECODING ANALYSIS: WEIGHTED MODEL")
print("="*70)

evaluate_predictions("Baseline argmax", baseline_preds, all_labels)

# 1. State18 Thresholding
print("\nMETHOD 1: State18 Thresholding")
best_non18 = np.argmax(np.delete(all_probs, 17, axis=1), axis=1)
for thresh in [0.20, 0.40, 0.60]:
    preds = baseline_preds.copy()
    preds[all_probs[:, 17] < thresh] = best_non18[all_probs[:, 17] < thresh]
    evaluate_predictions(f"Threshold {thresh:.2f}", preds, all_labels)

# 2. Margin Decoding
print("\nMETHOD 2: Margin Decoding")
for m in [0.05, 0.10, 0.15]:
    preds = []
    for probs in all_probs:
        top2_idx = np.argsort(probs)[-2:]
        second, best = top2_idx[0], top2_idx[1]
        if best == 17 and (probs[17] - probs[second] < m):
            preds.append(second)
        else:
            preds.append(best)
    evaluate_predictions(f"Margin {m:.2f}", np.array(preds), all_labels)

# 3. Temperature Sharpening (T < 1 sharpens, T > 1 flattens)
print("\nMETHOD 3: Temperature Sharpening")
for T in [0.7, 1.5]:
    sharpened = np.exp(np.log(all_probs + 1e-12) / T)
    sharpened /= sharpened.sum(axis=1, keepdims=True)
    evaluate_predictions(f"Temp {T:.2f}", np.argmax(sharpened, axis=1), all_labels)

# 4. State18 Penalty (Useful if weighted model is over-predicting S18)
print("\nMETHOD 4: Penalize State 18")
for penalty in [0.85, 0.95]:
    adj = all_probs.copy()
    adj[:, 17] *= penalty
    evaluate_predictions(f"Penalty {penalty:.2f}", np.argmax(adj, axis=1), all_labels)

# 5. Entropy-aware Rejection
print("\nMETHOD 5: Entropy Rejection")
ent = -np.sum(all_probs * np.log(all_probs + 1e-12), axis=1)
for e_thresh in [2.2, 2.5]:
    preds = baseline_preds.copy()
    mask = (ent > e_thresh) & (preds == 17)
    preds[mask] = best_non18[mask]
    evaluate_predictions(f"Entropy > {e_thresh}", preds, all_labels)