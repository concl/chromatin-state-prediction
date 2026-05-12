
from torch.utils.data import IterableDataset
import pandas as pd
from pathlib import Path
import torch


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
    
    @staticmethod
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
    
    def get_class_weights(self, num_labels: int = 18):
        if self._length is None:
            self.__len__()  # Trigger length calculation and caching
        return self.compute_class_weights(self.data_dir, num_labels=num_labels)