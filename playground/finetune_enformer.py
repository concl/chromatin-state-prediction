import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from enformer_pytorch import Enformer
from huggingface_hub import login
from dotenv import load_dotenv
import os

class EnformerForSequenceClassification(nn.Module):
    def __init__(self, num_labels=18):
        super().__init__()
        # Patch Enformer to bypass transformers version conflict
        if not hasattr(Enformer, "all_tied_weights_keys"):
            Enformer.all_tied_weights_keys = property(lambda self: {})
        
        # Load from enformer-pytorch
        self.enformer = Enformer.from_pretrained('EleutherAI/enformer-official-rough')
        self.classifier = nn.Linear(3072, num_labels)

    def forward(self, input_ids):
        outputs = self.enformer(input_ids, return_only_embeddings=True)
        # outputs shape: [Batch, 896, 3072]
        # Classifier transforms local embeddings into class logits
        # Final shape: [Batch, 896, num_labels]
        return self.classifier(outputs)

class ShardedChromatinDataset(IterableDataset):
    """
    Streams data from a directory of sharded Parquet files to save memory.
    Yields sequence tensors and labels directly.
    """
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.files = sorted(list(self.data_dir.glob("*.parquet")))
        if not self.files and self.data_dir.parent.exists():
            print(f"Warning: No parquet files found in {self.data_dir}, using parent {self.data_dir.parent}")
            self.files = sorted(list(self.data_dir.parent.glob("*.parquet")))
        if not self.files:
            print(f"Warning: No parquet files found in {self.data_dir}")
        self._length = None

    def __len__(self):
        if self._length is None:
            total = 0
            for file_path in self.files:
                df = pd.read_parquet(file_path)
                if 'sequence' in df.columns:
                    df = df.rename(columns={"sequence": "seq"})
                if 'labels' not in df.columns and 'label' in df.columns:
                    df = df.rename(columns={"label": "labels"})
                df = df.dropna(subset=['seq', 'labels'])
                total += len(df)
            self._length = total
        return self._length

    def __iter__(self):
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        target_len = 196608
        target_bins = 896  # Enformer crops 320 from each end of 1536 bins
        
        for file_path in self.files:
            df = pd.read_parquet(file_path)
            
            if 'sequence' in df.columns:
                df = df.rename(columns={"sequence": "seq"})
            
            if 'labels' not in df.columns and 'label' in df.columns:
                df = df.rename(columns={"label": "labels"})
                
            df = df.dropna(subset=['seq', 'labels'])
            
            for _, row in df.iterrows():
                seq = str(row['seq']).upper()
                labels = row['labels'] 
                
                # Labels are stored as numpy arrays (from data.py extract_long_sequences)
                # with length 1536 and values 1-18 (1-indexed ChromHMM states).
                # Enformer crops 320 bins from start and 320 from end → 896 bins.
                # PyTorch expects labels in range [0, num_classes-1], so subtract 1.
                if hasattr(labels, '__len__'):
                    if len(labels) == 1536:
                        labels = labels[320:-320]  # Crop to 896 bins
                    label_tensors = torch.tensor(labels, dtype=torch.long) - 1
                else:  # Fallback: single label
                    label_tensors = torch.tensor(int(labels) - 1, dtype=torch.long)

                mapped_seq = [mapping.get(nuc, 4) for nuc in seq]
                tensor_seq = torch.tensor(mapped_seq, dtype=torch.long)

                if len(tensor_seq) < target_len:
                    pad_size = target_len - len(tensor_seq)
                    tensor_seq = torch.nn.functional.pad(tensor_seq, (0, pad_size), value=4)
                else:
                    tensor_seq = tensor_seq[:target_len]

                yield tensor_seq, label_tensors

def compute_class_weights(data_dir: Path, num_labels: int = 18):
    """
    Scans the dataset to compute class counts, and returns normalized weights
    inversely proportional to class frequencies to handle extreme imbalances.
    """
    print("Computing class weights from dataset...")
    files = list(data_dir.glob("*.parquet"))
    
    if not files and data_dir.parent.exists():
        files = list(data_dir.parent.glob("*.parquet"))
        
    counts = torch.zeros(num_labels, dtype=torch.float64)
    
    for file_path in files:
        df = pd.read_parquet(file_path)
        col_name = "labels" if "labels" in df.columns else ("state" if "state" in df.columns else "label")
        
        if 'labels' in df.columns:
            import numpy as np
            # Flatten lists of labels
            all_labels = np.concatenate(df['labels'].values)
            values = pd.Series(all_labels)
        elif not pd.api.types.is_numeric_dtype(df[col_name]):
            # Fallback legacy extraction
            values = df[col_name].astype(str).str.extract(r'(\d+)', expand=False).fillna(0).astype(int)
        else:
            values = df[col_name]
            
        freqs = values.value_counts()
        for idx, count in freqs.items():
            # ChromHMM states are 1-indexed (1-18), but arrays are 0-indexed (0-17)
            if 1 <= idx <= num_labels:
                counts[idx - 1] += count
                
    # Calculate inverse frequencies
    # Add a small epsilon to avoid division by zero for completely missing classes
    total = counts.sum()
    if total == 0:
        return torch.ones(num_labels)
        
    weights = total / (num_labels * (counts + 1e-5))
    
    # Normalize weights so they don't artificially blow up the learning rate
    weights = weights / weights.sum() * num_labels
    return weights


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Enformer on multi-GPU")
    parser.add_argument("--data_dir", type=str, default="../sample/binned_dataframe/train_shards", help="Directory containing train sharded parquets")
    parser.add_argument("--val_data_dir", type=str, default=None, help="Directory containing validation sharded parquets")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="enformer_finetuned.pt", help="Path to save model")
    args = parser.parse_args()
    
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    if hf_token:
        login(token=hf_token)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])
    
    data_dir_path = Path(__file__).resolve().parent / args.data_dir
    
    if accelerator.is_main_process:
        print(f"Starting distributed training using '{data_dir_path.resolve()}'...")

    model = EnformerForSequenceClassification(num_labels=18)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    dataset = ShardedChromatinDataset(data_dir_path)
    
    # Standard DataLoader is completely compatible with IterableDatasets on accelerate
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    
    val_dataloader = None
    if args.val_data_dir:
        val_data_dir_path = Path(__file__).resolve().parent / args.val_data_dir
        val_dataset = ShardedChromatinDataset(val_data_dir_path)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Wrap to multi-GPU automatically
    if val_dataloader:
        model, optimizer, dataloader, val_dataloader = accelerator.prepare(model, optimizer, dataloader, val_dataloader)
    else:
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
        
    # Class Weights for Imbalanced Dataset
    # Compute optimal weights dynamically from the exact training shards available
    device = accelerator.device
    raw_weights = compute_class_weights(data_dir_path, num_labels=18)
    weights = raw_weights.to(device, dtype=torch.float32)
    
    if accelerator.is_main_process:
        print(f"Computed class weights: {weights}")
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    model.train()
    step = 0
    for epoch in range(args.epochs):
        accumulated_loss = 0.0
        
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs) # Shape: [Batch, 896, 18]
            
            # Flatten predictions and target arrays to compute Cross Entropy locally across bins
            # outputs views into [Batch * 896, 18], labels into [Batch * 896]
            # Cast to float32 because CrossEntropyLoss with class weights requires float32
            loss = criterion(outputs.float().view(-1, outputs.size(-1)), labels.view(-1))
            
            accelerator.backward(loss)
            
            # Clip gradients to prevent optimizer divergence causing NaNs
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            accumulated_loss += loss.item()
            
            if step % 10 == 0 and accelerator.is_main_process:
                print(f"Epoch {epoch} | Step {step} | Loss: {accumulated_loss/10:.4f}")
                accumulated_loss = 0.0
            
            step += 1
            
        # Validation Loop
        if val_dataloader:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for val_inputs, val_labels in val_dataloader:
                    val_outputs = model(val_inputs)
                    loss = criterion(val_outputs.float().view(-1, val_outputs.size(-1)), val_labels.view(-1))
                    # Gather losses across all GPUs
                    loss_gathered = accelerator.gather(loss)
                    val_loss += loss_gathered.mean().item()
                    val_steps += 1
                    
            if accelerator.is_main_process and val_steps > 0:
                print(f"--- Epoch {epoch} Validation Loss: {val_loss / val_steps:.4f} ---")
            model.train()

    if accelerator.is_main_process:
        save_path = Path(__file__).resolve().parent / args.output_dir
        print(f"Saving model to {save_path}...")
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), save_path)
        print("Training complete.")

if __name__ == "__main__":
    main()
