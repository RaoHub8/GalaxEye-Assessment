"""Train the EO-SAR binary change detection model."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eosar.config import Config
from eosar.data import build_dataloaders, compute_modality_stats
from eosar.eda import run_eda
from eosar.losses import FocalDiceLoss
from eosar.model import EOSARChangeDetector
from eosar.trainer import Trainer, create_optimizer, create_scheduler
from eosar.utils import get_device, get_device_info, make_run_dir, set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train EO-SAR change detection")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--preset", choices=["low", "medium", "high"], default=None)
    parser.add_argument("--device", choices=["auto", "directml", "cuda", "cpu"], default="auto")

    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--patch-size", "--crop-size", dest="patch_size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--gradient-clip", type=float, default=None)
    parser.add_argument("--accumulation-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)

    parser.add_argument("--model", dest="model_name", default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--encoder", dest="encoder_name", default=None)
    parser.add_argument("--no-pretrained", action="store_true")

    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default=None)
    parser.add_argument("--scheduler", choices=["cosine", "linear", "constant"], default=None)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")

    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--val-interval", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--eda-train-limit", type=int, default=80)
    parser.add_argument("--eda-eval-limit", type=int, default=60)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Config:
    """Load base config and apply explicit CLI overrides."""
    if args.config is not None:
        cfg = Config.from_yaml(args.config)
    elif args.preset == "low":
        cfg = Config.preset_low_memory()
    elif args.preset == "medium":
        cfg = Config.preset_medium_memory()
    elif args.preset == "high":
        cfg = Config.preset_high_memory()
    else:
        cfg = Config()

    overrides = {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "crop_size": args.patch_size,
        "stride": args.stride,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "accumulation_steps": args.accumulation_steps,
        "num_workers": args.num_workers,
        "model_name": args.model_name,
        "base_channels": args.base_channels,
        "dropout": args.dropout,
        "encoder_name": args.encoder_name,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "early_stopping_patience": args.early_stopping_patience,
        "val_interval": args.val_interval,
        "resume_path": args.resume,
        "seed": args.seed,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)

    if args.no_pretrained:
        cfg.encoder_weights = None
    if args.mixed_precision:
        cfg.mixed_precision = True
    if args.no_mixed_precision:
        cfg.mixed_precision = False
    if args.no_resume:
        cfg.resume = False
    cfg.output_dir = make_run_dir(cfg.output_dir, cfg.run_name)
    cfg.__post_init__()
    return cfg


def main() -> None:
    """Train from CLI."""
    args = parse_args()
    cfg = load_config(args)
    logger = setup_logging(cfg.output_dir, level=logging.INFO)
    set_seed(cfg.seed, deterministic=cfg.deterministic)

    device, device_name = get_device(args.device)
    logger.info("Device: %s", device_name)
    logger.info("Device info: %s", get_device_info(args.device))
    logger.info("Config: %s", cfg)

    if args.skip_eda:
        logger.info("Computing modality statistics from training split...")
        compute_modality_stats(cfg, split="train", limit=args.eda_train_limit)
    else:
        logger.info("Running notebook EDA before training...")
        run_eda(
            cfg,
            train_limit=args.eda_train_limit,
            eval_limit=args.eda_eval_limit,
            examples_split="train",
            examples=4,
        )
    
    logger.info("Building dataloaders...")
    train_loader, val_loader, _ = build_dataloaders(cfg)
    logger.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))

    logger.info("Creating model: %s", cfg.model_name)
    model = EOSARChangeDetector(
        encoder_name=cfg.encoder_name,
        encoder_weights=cfg.encoder_weights,
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        base_channels=cfg.base_channels,
        channel_multipliers=cfg.channel_multipliers,
        dropout=cfg.dropout,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Trainable parameters: %s", f"{total_params:,}")

    criterion = FocalDiceLoss(cfg).to(device)
    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(optimizer, cfg)
    trainer = Trainer(model, criterion, optimizer, scheduler, cfg, device)

    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
