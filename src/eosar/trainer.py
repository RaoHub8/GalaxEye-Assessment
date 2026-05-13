"""Production training loop for EO-SAR binary segmentation."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from eosar.config import Config
from eosar.metrics import evaluate_loader
from eosar.model import load_model_state_flexible
from eosar.utils import CSVLogger, cleanup_memory, is_directml_device, supports_amp

logger = logging.getLogger("eosar")


class Trainer:
    """Train, validate, checkpoint, and resume a segmentation model."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        cfg: Config,
        device: torch.device,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.device = device
        self.start_epoch = 1
        self.best_f1 = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.history: list[dict[str, Any]] = []

        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.to_json(self.cfg.config_path)

        self.use_amp = bool(cfg.mixed_precision and supports_amp(device))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.csv_logger = CSVLogger(cfg.csv_path) if cfg.csv_log else None
        self.writer = self._build_writer()

        if cfg.mixed_precision and not self.use_amp:
            logger.info("Mixed precision requested but disabled; AMP is only used on CUDA.")
        if is_directml_device(device):
            logger.info("DirectML selected: using conservative memory settings and FP32.")

    def _build_writer(self):
        if not self.cfg.tensorboard:
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter

            return SummaryWriter(log_dir=str(self.cfg.output_dir / "tensorboard"))
        except Exception as exc:
            logger.warning("TensorBoard disabled: %s", exc)
            return None

    def maybe_resume(self) -> None:
        """Resume from configured checkpoint if present."""
        resume_path = self.cfg.resume_path or self.cfg.last_checkpoint_path
        if not self.cfg.resume or not resume_path.exists():
            return
        logger.info("Resuming from checkpoint: %s", resume_path)
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        load_model_state_flexible(self.model, checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_f1 = float(checkpoint.get("best_f1", 0.0))
        self.best_epoch = int(checkpoint.get("best_epoch", 0))
        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
        self.history = list(checkpoint.get("history", []))
        logger.info("Resume complete. Next epoch: %d", self.start_epoch)

    def train_epoch(self, train_loader, epoch: int) -> float:
        """Run one training epoch."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        steps = 0

        progress = tqdm(train_loader, desc=f"Train {epoch}", leave=False)
        for batch_idx, batch_data in enumerate(progress, start=1):
            # Handle both 2-tuple and 3-tuple batches
            if len(batch_data) == 3:
                imgs, masks, valids = batch_data
                valids = valids.to(self.device, non_blocking=False)
            else:
                imgs, masks = batch_data
                valids = None
            
            imgs = imgs.to(self.device, non_blocking=False)
            masks = masks.to(self.device, non_blocking=False)

            try:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    logits = self.model(imgs)
                    loss = self.criterion(logits, masks, valids)
                    scaled_loss = loss / self.cfg.accumulation_steps

                self.scaler.scale(scaled_loss).backward()

                should_step = (
                    batch_idx % self.cfg.accumulation_steps == 0
                    or batch_idx == len(train_loader)
                )
                if should_step:
                    if self.cfg.gradient_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    steps += 1

                total_loss += float(loss.detach().cpu())
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                )
            except RuntimeError as exc:
                cleanup_memory()
                message = str(exc).lower()
                if "out of memory" in message or "allocate" in message:
                    raise RuntimeError(
                        "Training ran out of memory. Try --preset low, "
                        "--batch-size 1, lower --base-channels, or increase "
                        "--accumulation-steps."
                    ) from exc
                if is_directml_device(self.device):
                    raise RuntimeError(
                        "A DirectML operation failed. Try --device cpu to confirm "
                        "the pipeline, or reduce patch/model size for Vega 10."
                    ) from exc
                raise

            if batch_idx % 20 == 0:
                cleanup_memory()

        return total_loss / max(len(train_loader), 1)

    def validate(self, val_loader) -> dict[str, float]:
        """Run validation with patch-based inference."""
        return evaluate_loader(
            self.model,
            val_loader,
            self.cfg,
            self.device,
            use_tta=False,
            threshold=self.cfg.threshold,
            desc="Validate",
        )

    def train(self, train_loader, val_loader) -> list[dict[str, Any]]:
        """Run full training."""
        self.maybe_resume()
        logger.info("Starting training for %d epochs", self.cfg.epochs)
        logger.info(
            "Device=%s batch_size=%d accumulation=%d patch=%d amp=%s",
            self.device,
            self.cfg.batch_size,
            self.cfg.accumulation_steps,
            self.cfg.patch_size,
            self.use_amp,
        )

        for epoch in range(self.start_epoch, self.cfg.epochs + 1):
            epoch_start = time.time()
            train_loss = self.train_epoch(train_loader, epoch)
            metrics: dict[str, float] = {}
            is_best = False
            should_stop = False

            if epoch % self.cfg.val_interval == 0:
                metrics = self.validate(val_loader)
                is_best, should_stop = self._update_early_stopping(epoch, metrics["f1"])

            if self.scheduler is not None:
                self.scheduler.step()

            row: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "seconds": float(time.time() - epoch_start),
                **metrics,
            }
            self.history.append(row)
            self._log_epoch(row)
            self._save_checkpoint(epoch, train_loss, is_best)
            self._save_history()
            self._save_training_curves()
            cleanup_memory()

            if should_stop:
                logger.info(
                    "Early stopping at epoch %d. Best F1 %.4f at epoch %d.",
                    epoch,
                    self.best_f1,
                    self.best_epoch,
                )
                break

        if self.writer is not None:
            self.writer.close()
        logger.info("Training complete. Best F1 %.4f at epoch %d.", self.best_f1, self.best_epoch)
        return self.history

    def _update_early_stopping(self, epoch: int, f1: float) -> tuple[bool, bool]:
        """Update best score and patience state."""
        if f1 > self.best_f1 + self.cfg.early_stopping_min_delta:
            self.best_f1 = float(f1)
            self.best_epoch = epoch
            self.patience_counter = 0
            return True, False
        self.patience_counter += 1
        return False, self.patience_counter >= self.cfg.early_stopping_patience

    def _log_epoch(self, row: dict[str, Any]) -> None:
        parts = [
            f"Epoch {row['epoch']:3d}/{self.cfg.epochs}",
            f"Train Loss: {row['train_loss']:.4f}",
        ]
        for key in ("f1", "iou", "dice", "precision", "recall"):
            if key in row:
                parts.append(f"Val {key.upper()}: {row[key]:.4f}")
        logger.info(" | ".join(parts))

        if self.csv_logger is not None:
            self.csv_logger.log(row)
        if self.writer is not None:
            for key, value in row.items():
                if key != "epoch" and isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, int(row["epoch"]))

    def _save_checkpoint(self, epoch: int, train_loss: float, is_best: bool) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "config": self.cfg.to_dict(),
            "train_loss": float(train_loss),
            "best_f1": self.best_f1,
            "best_epoch": self.best_epoch,
            "history": self.history,
        }
        torch.save(checkpoint, self.cfg.last_checkpoint_path)
        if is_best:
            torch.save(checkpoint, self.cfg.checkpoint_path)
            logger.info("Saved best checkpoint: %s", self.cfg.checkpoint_path)

    def _save_history(self) -> None:
        with open(self.cfg.history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def _save_training_curves(self) -> None:
        if not self.history:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return

        epochs = [row["epoch"] for row in self.history]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, [row.get("train_loss", 0.0) for row in self.history], marker="o")
        axes[0].set_title("Training loss")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].grid(alpha=0.3)

        if any("f1" in row for row in self.history):
            axes[1].plot(epochs, [row.get("f1", 0.0) for row in self.history], marker="o", label="F1")
            axes[1].plot(epochs, [row.get("iou", 0.0) for row in self.history], marker=".", label="IoU")
            axes[1].plot(epochs, [row.get("precision", 0.0) for row in self.history], marker=".", label="precision")
            axes[1].plot(epochs, [row.get("recall", 0.0) for row in self.history], marker=".", label="recall")
            axes[1].legend()
        axes[1].set_title("Validation metrics")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("score")
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.cfg.output_dir / "training_curves.png", dpi=150)
        plt.close(fig)


def create_optimizer(model: nn.Module, cfg: Config) -> Optimizer:
    """Create the configured optimizer."""
    if cfg.optimizer.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            momentum=0.9,
        )
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def create_scheduler(optimizer: Optimizer, cfg: Config) -> LRScheduler | None:
    """Create the configured learning-rate scheduler."""
    name = cfg.scheduler.lower()
    if name == "constant":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1))
    if name == "linear":
        return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=max(cfg.epochs, 1))
    raise ValueError(f"Unknown scheduler: {cfg.scheduler}")
