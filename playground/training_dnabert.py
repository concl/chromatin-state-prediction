import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import pairwise2
from torch.cuda.amp import autocast, GradScaler

# ==============================
# SETTINGS
# ==============================

MODEL_NAME = "zhihan1996/DNA_bert_6"
MAX_LENGTH = 600
BATCH_SIZE = 64
EPOCHS = 10
NUM_CLASSES = 19

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", DEVICE)

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv("training_data.csv")

sequences = df["sequence"].astype(str).values
labels = df["label"].values

# ==============================
# BUILD 600bp CONTEXT WINDOWS
# ==============================

context_sequences = []
context_labels = []

for i in range(1, len(sequences)-1):

    combined_seq = sequences[i-1] + sequences[i] + sequences[i+1]

    context_sequences.append(combined_seq)
    context_labels.append(labels[i])  # predict middle state

sequences = np.array(context_sequences)
labels = np.array(context_labels)

print("Total samples after context windows:", len(sequences))

# ==============================
# CLASS WEIGHTS
# ==============================

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

print("Class weights:", class_weights)

# ==============================
# TOKENIZER
# ==============================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizing dataset...")

encodings = tokenizer(
    list(sequences),
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="pt"
)

# ==============================
# DATASET
# ==============================

class DNADataset(Dataset):

    def __init__(self, encodings, labels, sequences):

        self.encodings = encodings
        self.labels = labels
        self.sequences = sequences

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        item["sequence"] = self.sequences[idx]

        return item


dataset = DNADataset(encodings, labels, sequences)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=6,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=6,
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
# TRAINING LOOP
# ==============================

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for batch in train_loader:

        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        with autocast():

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits

            loss = F.cross_entropy(
                logits,
                labels,
                weight=class_weights
            )

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

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
all_sequences = []

with torch.no_grad():

    for batch in val_loader:

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        probs = F.softmax(logits, dim=1)

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_sequences.extend(batch["sequence"])

# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Chromatin State Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# MOST CONFUSED STATES
# ==============================

confusions = []

for i in range(len(cm)):
    for j in range(len(cm)):
        if i != j and cm[i][j] > 0:
            confusions.append((i, j, cm[i][j]))

confusions = sorted(confusions, key=lambda x: x[2], reverse=True)

print("\nMost common misclassifications:")

for a,p,c in confusions[:10]:
    print(f"Actual {a} -> Predicted {p} : {c}")

# ==============================
# CONFIDENCE ANALYSIS
# ==============================

all_probs = np.array(all_probs)
confidences = np.max(all_probs, axis=1)

print("\nConfidence Stats")
print("Average:", confidences.mean())
print("Min:", confidences.min())
print("Max:", confidences.max())

plt.hist(confidences, bins=50)
plt.title("Model Confidence Distribution")
plt.xlabel("Softmax Probability")
plt.ylabel("Frequency")
plt.show()

# ==============================
# QUIESCENT STATE CONFIDENCE
# ==============================

QUIESCENT_LABEL = 18

quiescent_conf = []

for i,label in enumerate(all_labels):
    if label == QUIESCENT_LABEL:
        quiescent_conf.append(confidences[i])

if len(quiescent_conf) > 0:
    print("Average quiescent confidence:", np.mean(quiescent_conf))

# ==============================
# DISTANCE MATRIX ANALYSIS
# ==============================

try:

    dist_df = pd.read_csv("chromatin_state_distances.csv", index_col=0)
    distance_matrix = dist_df.values

    distances = []

    for true,pred in zip(all_labels, all_preds):

        if true != pred:
            distances.append(distance_matrix[true][pred])

    if len(distances) > 0:
        print("Average distance of misclassifications:", np.mean(distances))

except:
    print("Distance matrix not found, skipping.")

# ==============================
# GENOME TRACK VISUALIZATION
# ==============================

plt.figure(figsize=(14,4))

plt.plot(all_labels[:1000], label="Actual")
plt.plot(all_preds[:1000], label="Predicted")

plt.title("Chromatin State Along DNA")
plt.xlabel("Sequence Index")
plt.ylabel("State")

plt.legend()
plt.show()

# ==============================
# MISCLASSIFIED SEQUENCES
# ==============================

misclassified = []

for seq,true,pred in zip(all_sequences, all_labels, all_preds):

    if true != pred:
        misclassified.append((seq,true,pred))

print("Total misclassified:", len(misclassified))

# ==============================
# SEQUENCE ALIGNMENT EXAMPLE
# ==============================

if len(misclassified) >= 2:

    seq1 = misclassified[0][0]
    seq2 = misclassified[1][0]

    alignment = pairwise2.align.globalxx(seq1, seq2)[0]

    print("\nExample sequence alignment:")
    print(pairwise2.format_alignment(*alignment))

# ==============================
# SAVE MODEL
# ==============================

torch.save(model.state_dict(), "dnabert_chromatin_model.pt")

print("Training complete.")