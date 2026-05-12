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
        pooled = outputs.mean(dim=1)
        return self.classifier(pooled)

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

    def __iter__(self):
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        target_len = 196608
        
        for file_path in self.files:
            df = pd.read_parquet(file_path)
            
            if 'sequence' in df.columns:
                df = df.rename(columns={"sequence": "seq", "state": "label"})
                
            if not pd.api.types.is_numeric_dtype(df['label']):
                # Extract int state if formatted as string ('1_TssA')
                df['label'] = df['label'].astype(str).str.extract(r'(\d+)', expand=False).dropna().astype(int)
            
            df = df.dropna(subset=['seq', 'label'])
            
            for _, row in df.iterrows():
                seq = str(row['seq']).upper()
                # PyTorch expects labels in range [0, num_classes-1]
                label = int(row['label']) - 1

                mapped_seq = [mapping.get(nuc, 4) for nuc in seq]
                tensor_seq = torch.tensor(mapped_seq, dtype=torch.long)

                if len(tensor_seq) < target_len:
                    pad_size = target_len - len(tensor_seq)
                    tensor_seq = torch.nn.functional.pad(tensor_seq, (0, pad_size), value=4)
                else:
                    tensor_seq = tensor_seq[:target_len]

                yield tensor_seq, torch.tensor(label, dtype=torch.long)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Enformer on multi-GPU")
    parser.add_argument("--data_dir", type=str, default="../sample/binned_dataframe/enformer_shards", help="Directory containing train sharded parquets")
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
        
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    step = 0
    for epoch in range(args.epochs):
        accumulated_loss = 0.0
        
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
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
                    loss = criterion(val_outputs, val_labels)
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
