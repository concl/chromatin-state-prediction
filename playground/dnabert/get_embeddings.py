import os
import torch
import h5py
import glob
import numpy as np
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# =======================
# SETTINGS
# =======================
DATA_DIR = "/u/scratch/a/aparikh/dnabert_results"
OUT_DIR = "/u/scratch/a/aparikh/dnabert_embeddings"
os.makedirs(OUT_DIR, exist_ok=True)

# Detect hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

BATCH_SIZE = 32
POOLING = "mean"   # options: "mean" or "cls"

# =======================
# LOAD MODEL (DNABERT-1 6-mer)
# =======================
print("Loading DNABERT-1 (6-mer)...")
tokenizer = BertTokenizer.from_pretrained("zhihan1996/DNA_bert_6")
model = BertModel.from_pretrained("zhihan1996/DNA_bert_6").to(DEVICE)

model.eval()

pad_id = tokenizer.pad_token_id
hidden_size = model.config.hidden_size

# =======================
# FILES
# =======================
h5_files = sorted(glob.glob(os.path.join(DATA_DIR, "processed_part_*.h5")))
print(f"Found {len(h5_files)} files to process.")

# =======================
# MAIN LOOP
# =======================
with torch.no_grad():
    for f_path in h5_files:
        out_f_path = os.path.join(
            OUT_DIR,
            os.path.basename(f_path).replace("processed", "embedding")
        )

        # Skip if already processed (Check-pointing)
        if os.path.exists(out_f_path):
            print(f"Skipping {f_path} (already exists).")
            continue

        print(f"Processing: {f_path}")

        with h5py.File(f_path, "r") as f_in, h5py.File(out_f_path, "w") as f_out:
            input_ids_np = f_in["input_ids"][:]
            labels_np = f_in["labels"][:]

            n_samples = len(input_ids_np)
            all_embeddings = np.zeros((n_samples, hidden_size), dtype=np.float32)

            for i in tqdm(range(0, n_samples, BATCH_SIZE), desc="Batches"):
                end_idx = min(i + BATCH_SIZE, n_samples)
                
                # FIXED: Convert numpy slice to torch tensor without direct pinning
                batch_ids = torch.from_numpy(input_ids_np[i:end_idx]).long().to(DEVICE)

                # Mixed precision for speed on GPUs
                with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                    outputs = model(batch_ids)
                    hidden_states = outputs.last_hidden_state  # Shape: [Batch, Seq_Len, Hidden]

                if POOLING == "cls":
                    # Use the [CLS] token (first token)
                    emb = hidden_states[:, 0, :]

                elif POOLING == "mean":
                    # Masked mean pooling to ignore padding tokens
                    attention_mask = (batch_ids != pad_id).float()
                    masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
                    sum_hidden = masked_hidden.sum(dim=1)
                    lengths = attention_mask.sum(dim=1, keepdim=True)
                    
                    # Avoid division by zero if a sequence is somehow empty
                    emb = sum_hidden / torch.clamp(lengths, min=1)

                else:
                    raise ValueError("POOLING must be 'mean' or 'cls'")

                # Move back to CPU and store in our numpy array
                all_embeddings[i:end_idx] = emb.cpu().numpy()

            # Save results back to a new HDF5 file
            f_out.create_dataset("embeddings", data=all_embeddings, compression="gzip")
            f_out.create_dataset("labels", data=labels_np, compression="gzip")

        print(f"Successfully saved: {out_f_path}")

print("--- All files processed! ---")