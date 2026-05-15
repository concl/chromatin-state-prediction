"""
Fine-tune Enformer for chromatin-state prediction (18-class classification).

Thin CLI entry point — all heavy lifting is delegated to:

    templates.enformer_dataset   → ShardedChromatinDataset, class weights
    templates.enformer_trainer   → EnformerForSequenceClassification, EnformerTrainer

Usage:
    accelerate launch \
        --multi_gpu \
        --num_processes 3 \
        --gpu_ids 1,2,3 \
        --mixed_precision bf16 \
        playground/finetune_enformer.py \
        --data_dir ../sample/binned_dataframe/train_shards \
        --val_data_dir ../sample/binned_dataframe/val_shards \
        --batch_size 2 \
        --epochs 1 \
        --lr 5e-5 \
        --output_dir enformer_finetuned.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader

from templates.enformer_dataset import ShardedChromatinDataset
from templates.enformer_trainer import EnformerForSequenceClassification, EnformerTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Enformer on multi-GPU")
    parser.add_argument(
        "--data_dir", type=str, default="../sample/binned_dataframe/train_shards",
        help="Directory containing train sharded parquets",
    )
    parser.add_argument(
        "--val_data_dir", type=str, default=None,
        help="Directory containing validation sharded parquets",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output_dir", type=str, default="enformer_finetuned.pt")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--num_checkpoints_to_keep", type=int, default=3)
    args = parser.parse_args()
    
    # ----- Auth ------
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    if hf_token:
        login(token=hf_token)

    # ----- Accelerator -----
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])

    base = Path(__file__).resolve().parent
    train_path = base / args.data_dir

    if accelerator.is_main_process:
        print(f"Starting distributed training using '{train_path.resolve()}'...")

    # ----- Model, optimizer, dataloaders -----
    model = EnformerForSequenceClassification(num_labels=18)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_dataset = ShardedChromatinDataset(train_path)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    val_dataloader = None
    if args.val_data_dir:
        val_path = base / args.val_data_dir
        val_dataset = ShardedChromatinDataset(val_path)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Accelerator prepare
    if val_dataloader:
        model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader
        )
    else:
        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )

    # ----- Class weights (computed from the actual training data on disk) -----
    raw_weights = ShardedChromatinDataset.compute_class_weights(train_path, num_labels=18)
    weights = raw_weights.to(accelerator.device, dtype=torch.float32)

    if accelerator.is_main_process:
        print(f"Computed class weights: {weights}")

    criterion = nn.CrossEntropyLoss(weight=weights)

    # ----- Train -----
    trainer = EnformerTrainer(
        model,
        optimizer,
        criterion,
        accelerator,
        checkpoint_dir=base / args.checkpoint_dir,
        save_every=args.save_every,
        num_checkpoints_to_keep=args.num_checkpoints_to_keep,
    )

    trainer.train(train_dataloader, val_dataloader, epochs=args.epochs)
    trainer.save_model(base / args.output_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()

