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
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix

# =====================
# CONFIG
# =====================
DATA_DIR = "/u/scratch/a/aparikh/dnabert_embeddings"
# Make sure this points to the specific .pt file from your UNWEIGHTED run
CHECKPOINT_PATH = "./uw_checkpoints/checkpoint_epoch_3.pt" 

BATCH_SIZE = 8192
NUM_CLASSES = 18
INPUT_DIM = 768
FILES_PER_CHUNK = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_CLASS = 17   # state 18

# =====================
# MODEL (Updated to match Unweighted Architecture)
# =====================
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 256),
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

# =====================
# LOAD MODEL
# =====================
print(f"Loading unweighted model from: {CHECKPOINT_PATH}")
model = SimpleClassifier().to(DEVICE)
# This will now load without the "Missing Keys" error
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# =====================
# GET VALIDATION FILES
# =====================
all_files = sorted(
    glob.glob(os.path.join(DATA_DIR, "embedding_part_*.h5")),
    key=extract_num
)

# Using the same 10% split logic as your training script
split_idx = int(len(all_files) * 0.9)
val_files = all_files[split_idx:]

print(f"Running inference on {len(val_files)} validation files...")

# =====================
# INFERENCE
# =====================
all_probs = []
all_labels = []

with torch.no_grad():
    for i in range(0, len(val_files), FILES_PER_CHUNK):
        X, Y = load_chunk(val_files[i:i+FILES_PER_CHUNK])
        if X is None:
            continue

        loader = make_loader(X, Y)

        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(yb.numpy())

        del X, Y, loader
        gc.collect()

all_probs = np.vstack(all_probs)
all_labels = np.concatenate(all_labels)
preds = all_probs.argmax(axis=1)

# =====================
# STATS & PLOTS
# =====================
max_prob = all_probs.max(axis=1)
print(f"Mean max softmax (Unweighted): {max_prob.mean():.4f}")

p17 = all_probs[:, TARGET_CLASS]
true17 = (all_labels == TARGET_CLASS).astype(int)

# Histogram
plt.figure(figsize=(10,6))
plt.hist(p17[true17==1], bins=50, alpha=0.6, label="True State 18", color='blue')
plt.hist(p17[true17==0], bins=50, alpha=0.6, label="Other States", color='orange')
plt.xlabel("Softmax probability of State 18")
plt.ylabel("Count")
plt.title("Distribution of P(State 18) - Unweighted Model")
plt.legend()
plt.tight_layout()
plt.savefig("uw_class17_probability_hist.png")

# Threshold Sweep
rows = []
for t in np.arange(0.05, 0.96, 0.05):
    assign17 = p17 >= t
    tp = np.sum(assign17 & (true17==1))
    fp = np.sum(assign17 & (true17==0))
    fn = np.sum((~assign17) & (true17==1))
    precision = tp / max(tp+fp, 1)
    recall = tp / max(tp+fn, 1)
    rows.append([t, tp, fp, precision, recall])

df = pd.DataFrame(rows, columns=["threshold","tp","fp","precision","recall"])
df.to_csv("uw_class17_threshold_table.csv", index=False)

# Confusion Matrix
cm = confusion_matrix(all_labels, preds, labels=range(NUM_CLASSES))
np.savetxt("uw_confusion_matrix.csv", cm, delimiter=",", fmt="%d")

print("Done. Saved 'uw_' prefixed files.")

# =====================
# Calibration Curve for class17
# =====================

from sklearn.calibration import calibration_curve

# Binary target:
# true if label == 17
y_true_binary = (all_labels == 17).astype(int)

# predicted probability for class17
y_prob = all_probs[:, 17]

prob_true, prob_pred = calibration_curve(
    y_true_binary,
    y_prob,
    n_bins=15,
    strategy='quantile'
)

plt.figure(figsize=(6,6))
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("Predicted P(class17)")
plt.ylabel("Observed frequency")
plt.title("Calibration curve for class17")

plt.savefig("class17_calibration.png")


from sklearn.metrics import balanced_accuracy_score, f1_score

thresholds = np.arange(0.30, 0.76, 0.02)

results = []

print("\nThreshold sweep for state18 decoding\n")

for t in thresholds:

    preds = []

    for p in all_probs:

        p18 = p[17]

        # assign state18 only if confident enough
        if p18 >= t:
            pred = 17
        else:
            # choose best NON-state18 class
            non18_pred = np.argmax(p[:17])
            pred = non18_pred

        preds.append(pred)

    preds = np.array(preds)

    bal_acc = balanced_accuracy_score(all_labels, preds)

    macro_f1 = f1_score(
        all_labels,
        preds,
        average="macro",
        zero_division=0
    )

    state18_recall = (
        ((preds == 17) & (all_labels == 17)).sum()
        / (all_labels == 17).sum()
    )

    results.append([
        t,
        bal_acc,
        macro_f1,
        state18_recall
    ])

    print(
        f"Threshold={t:.2f} | "
        f"BalancedAcc={bal_acc:.4f} | "
        f"MacroF1={macro_f1:.4f} | "
        f"State18Recall={state18_recall:.4f}"
    )

from sklearn.metrics import balanced_accuracy_score, f1_score

print("\n" + "="*70)
print("ALTERNATIVE DECODING METHODS")
print("="*70)

# ==========================================================
# Helper
# ==========================================================
def evaluate_predictions(name, preds):
    bal_acc = balanced_accuracy_score(all_labels, preds)
    macro_f1 = f1_score(all_labels, preds, average="macro", zero_division=0)

    state18_recall = (
        ((preds == 17) & (all_labels == 17)).sum()
        / max(1, (all_labels == 17).sum())
    )

    print(
        f"{name:35s} | "
        f"BalancedAcc={bal_acc:.4f} | "
        f"MacroF1={macro_f1:.4f} | "
        f"State18Recall={state18_recall:.4f}"
    )

# ==========================================================
# BASELINE ARGMAX
# ==========================================================
baseline_preds = np.argmax(all_probs, axis=1)
evaluate_predictions("Baseline argmax", baseline_preds)

# ==========================================================
# METHOD 1:
# State18 thresholding
# ==========================================================
print("\nMETHOD 1: State18 thresholding")

for thresh in [0.40, 0.50, 0.60, 0.70]:
    preds = baseline_preds.copy()

    p18 = all_probs[:, 17]

    # Best non-18 class
    non18_probs = all_probs.copy()
    non18_probs[:, 17] = -1
    best_non18 = np.argmax(non18_probs, axis=1)

    preds[p18 < thresh] = best_non18[p18 < thresh]

    evaluate_predictions(f"Threshold {thresh:.2f}", preds)

# ==========================================================
# METHOD 2:
# Margin decoding
# Assign state18 ONLY if it beats runner-up by margin
# ==========================================================
print("\nMETHOD 2: Margin decoding")

for margin in [0.00, 0.02, 0.05, 0.10, 0.15]:

    preds = []

    for probs in all_probs:

        top2 = np.argsort(probs)[-2:]
        second = top2[0]
        best = top2[1]

        if best == 17:
            if probs[17] - probs[second] < margin:
                preds.append(second)
            else:
                preds.append(17)
        else:
            preds.append(best)

    preds = np.array(preds)

    evaluate_predictions(f"Margin {margin:.2f}", preds)

# ==========================================================
# METHOD 3:
# Temperature sharpening
# Makes confident predictions stronger
# ==========================================================
print("\nMETHOD 3: Temperature sharpening")

for T in [0.5, 0.7, 1.5, 2.0]:

    logits_like = np.log(all_probs + 1e-12) / T
    sharpened = np.exp(logits_like)
    sharpened /= sharpened.sum(axis=1, keepdims=True)

    preds = np.argmax(sharpened, axis=1)

    evaluate_predictions(f"Temperature {T:.2f}", preds)

# ==========================================================
# METHOD 4:
# Penalize state18 probability
# ==========================================================
print("\nMETHOD 4: Penalize state18")

for penalty in [0.80, 0.85, 0.90, 0.95]:

    adjusted = all_probs.copy()
    adjusted[:, 17] *= penalty

    preds = np.argmax(adjusted, axis=1)

    evaluate_predictions(f"Penalty {penalty:.2f}", preds)

# ==========================================================
# METHOD 5:
# Entropy rejection
# If prediction uncertain, avoid state18
# ==========================================================
print("\nMETHOD 5: Entropy-aware decoding")

entropy = -np.sum(all_probs * np.log(all_probs + 1e-12), axis=1)

for ent_thresh in [2.0, 2.2, 2.4, 2.6]:

    preds = baseline_preds.copy()

    non18_probs = all_probs.copy()
    non18_probs[:, 17] = -1
    best_non18 = np.argmax(non18_probs, axis=1)

    uncertain = entropy > ent_thresh
    state18_pred = preds == 17

    mask = uncertain & state18_pred

    preds[mask] = best_non18[mask]

    evaluate_predictions(f"Entropy>{ent_thresh:.1f}", preds)
