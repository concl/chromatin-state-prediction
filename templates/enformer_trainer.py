"""
Enformer fine-tuning trainer with Accelerate-backed multi-GPU support.

Usage:
    from templates.enformer_trainer import EnformerForSequenceClassification, EnformerTrainer

    model = EnformerForSequenceClassification(num_labels=18)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    trainer = EnformerTrainer(model, optimizer, criterion, accelerator, checkpoint_dir=...)
    trainer.train(train_dataloader, val_dataloader, epochs=3)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from accelerate import Accelerator

logger = logging.getLogger(__name__)


class EnformerForSequenceClassification(nn.Module):
    """Wraps a pretrained Enformer trunk with a linear classification head.

    Enformer produces embeddings of shape [B, 896, 3072].  This module
    projects each of the 896 genomic bins to ``num_labels`` class logits,
    yielding output of shape [B, 896, num_labels].
    """

    def __init__(self, num_labels: int = 18):
        super().__init__()
        # Patch Enformer to bypass transformers version conflict
        from enformer_pytorch import Enformer

        if not hasattr(Enformer, "all_tied_weights_keys"):
            Enformer.all_tied_weights_keys = property(lambda self: {})

        self.enformer = Enformer.from_pretrained("EleutherAI/enformer-official-rough")
        self.classifier = nn.Linear(3072, num_labels)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.enformer(input_ids, return_only_embeddings=True)  # [B, 896, 3072]
        return self.classifier(embeddings)  # [B, 896, num_labels]


class EnformerTrainer:
    """Encapsulates the Enformer fine-tuning loop with checkpointing and validation.

    Parameters
    ----------
    model : nn.Module
        An ``EnformerForSequenceClassification`` (or Accelerator-wrapped equivalent).
    optimizer : torch.optim.Optimizer
    criterion : nn.Module
        Loss function (e.g. ``CrossEntropyLoss`` with class weights).
    accelerator : Accelerator
        HuggingFace Accelerate instance managing device placement, mixed precision,
        and gradient accumulation.
    checkpoint_dir : Path
        Directory where intermediate checkpoints are saved.
    save_every : int
        Save a checkpoint every *N* training steps.
    num_checkpoints_to_keep : int
        Only keep this many of the most recent checkpoints (old ones are deleted).
    log_every : int
        Log average loss every *N* steps (only on the main process).
    grad_clip_norm : float | None
        Max gradient norm for clipping; ``None`` disables.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        accelerator: Accelerator,
        *,
        checkpoint_dir: Path,
        save_every: int = 1000,
        num_checkpoints_to_keep: int = 3,
        log_every: int = 10,
        grad_clip_norm: float | None = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.accelerator = accelerator

        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_every = save_every
        self.num_checkpoints_to_keep = num_checkpoints_to_keep
        self.log_every = log_every
        self.grad_clip_norm = grad_clip_norm

        self._saved_checkpoints: list[Path] = []
        self._global_step: int = 0
    
    def train(
        self,
        train_dataloader,
        val_dataloader=None,
        epochs: int = 3,
    ) -> None:
        """Run the full multi-epoch training loop."""
        self._maybe_create_checkpoint_dir()

        for epoch in range(epochs):
            self.model.train()
            self._train_one_epoch(train_dataloader, epoch)

            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                if self.accelerator.is_main_process:
                    print(f"--- Epoch {epoch} Validation Loss: {val_loss:.4f} ---")

    def validate(self, dataloader) -> float:
        """Run one validation pass; returns mean loss across all processes."""
        self.model.eval()
        total_loss = 0.0
        steps = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = self.model(inputs)
                loss = self._compute_loss(outputs, labels)

                # Gather losses from all accelerators
                gathered = self.accelerator.gather(loss)
                total_loss += gathered.mean().item()
                steps += 1

        return total_loss / max(steps, 1)

    def save_model(self, path: Path | str) -> None:
        """Save the unwrapped model state dict."""
        if self.accelerator.is_main_process:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            unwrapped = self.accelerator.unwrap_model(self.model)
            torch.save(unwrapped.state_dict(), path)
            print(f"Model saved to {path}")

    # ----- Internal helpers -----

    def _train_one_epoch(self, dataloader, epoch: int) -> None:
        accumulated_loss = 0.0

        for inputs, labels in dataloader:
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self._compute_loss(outputs, labels)

            self.accelerator.backward(loss)

            if self.grad_clip_norm is not None:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip_norm
                )

            self.optimizer.step()
            accumulated_loss += loss.item()

            if self._should_log():
                avg = accumulated_loss / self.log_every
                print(
                    f"Epoch {epoch} | Step {self._global_step} | Loss: {avg:.4f}"
                )
                accumulated_loss = 0.0

            if self._should_save_checkpoint():
                self._save_checkpoint()

            self._global_step += 1

    def _compute_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Flatten predictions and targets for per-bin cross-entropy."""
        # outputs: [B, 896, C] → [B*896, C]  (cast to float32 for weighted CE)
        # labels:  [B, 896]    → [B*896]
        return self.criterion(
            outputs.float().view(-1, outputs.size(-1)),
            labels.view(-1),
        )

    def _should_log(self) -> bool:
        return (
            self._global_step % self.log_every == 0
            and self.accelerator.is_main_process
        )

    def _should_save_checkpoint(self) -> bool:
        return (
            self._global_step % self.save_every == 0
            and self._global_step > 0
            and self.accelerator.is_main_process
        )

    def _maybe_create_checkpoint_dir(self) -> None:
        if self.accelerator.is_main_process:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(self) -> None:
        ckpt_path = self.checkpoint_dir / f"enformer_step_{self._global_step}.pt"
        print(f"Saving checkpoint to {ckpt_path}...")
        unwrapped = self.accelerator.unwrap_model(self.model)
        torch.save(unwrapped.state_dict(), ckpt_path)

        self._saved_checkpoints.append(ckpt_path)
        self._rotate_checkpoints()

    def _rotate_checkpoints(self) -> None:
        while len(self._saved_checkpoints) > self.num_checkpoints_to_keep:
            old = self._saved_checkpoints.pop(0)
            old.unlink()
            print(f"Deleted old checkpoint: {old.name}")

