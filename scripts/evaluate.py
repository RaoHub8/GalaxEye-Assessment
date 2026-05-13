"""Run the notebook-faithful EO-SAR evaluation and report pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eosar.config import Config
from eosar.data import EOSARDataset, build_inference_transform, compute_modality_stats
from eosar.evaluation import evaluate_loader_postprocessed
from eosar.metrics import evaluate_loader
from eosar.model import EOSARChangeDetector, load_model_state_flexible
from eosar.reporting import build_ablation_rows, save_report_assets, write_results_summary
from eosar.thresholding import (
    build_prediction_cache,
    evaluate_prediction_cache,
    post_training_calibration_sweep,
    threshold_sweep,
    validation_pr_analysis,
    save_threshold_plot,
    write_rows_csv,
)
from eosar.utils import get_device, get_device_info, set_seed, setup_logging
from eosar.visualization import (
    failure_caption,
    save_confusion_matrix,
    save_failure_cases,
    save_hardest_three,
    save_qualitative_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EO-SAR model like the Kaggle notebook")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/best_model.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", choices=["auto", "directml", "cuda", "cpu"], default="auto")
    parser.add_argument("--patch-size", "--crop-size", dest="patch_size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--no-visuals", action="store_true")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device, fallback_cfg: Config) -> tuple[EOSARChangeDetector, Config]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_cfg = checkpoint.get("config", checkpoint.get("cfg", fallback_cfg.to_dict()))
    cfg = Config.from_dict(saved_cfg)
    cfg.data_dir = fallback_cfg.data_dir
    cfg.output_dir = fallback_cfg.output_dir
    cfg.patch_size = fallback_cfg.patch_size
    cfg.crop_size = fallback_cfg.crop_size
    cfg.stride = fallback_cfg.stride

    model = EOSARChangeDetector(
        encoder_name=cfg.encoder_name,
        encoder_weights=None,
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        base_channels=cfg.base_channels,
        channel_multipliers=cfg.channel_multipliers,
        dropout=cfg.dropout,
    )
    load_model_state_flexible(model, checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg


def make_loader(split: str, cfg: Config) -> DataLoader:
    dataset = EOSARDataset(split=split, transform=build_inference_transform(cfg), change_aware=False, cfg=cfg)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.output_dir, log_filename="evaluation.log", level=logging.INFO)
    set_seed(42)
    device, device_name = get_device(args.device)
    logger.info("Device: %s", device_name)
    logger.info("Device info: %s", get_device_info(args.device))

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    fallback_cfg = Config(data_dir=args.data_dir, output_dir=args.output_dir)
    if args.patch_size is not None:
        fallback_cfg.patch_size = args.patch_size
        fallback_cfg.crop_size = args.patch_size
    if args.stride is not None:
        fallback_cfg.stride = args.stride

    model, cfg = load_model(args.checkpoint, device, fallback_cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(cfg.config_path)
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    logger.info("Computing notebook EO normalization statistics from train split...")
    compute_modality_stats(cfg, split="train", limit=80)
    val_loader = make_loader("val", cfg)
    test_loader = make_loader("test", cfg)

    if args.threshold is None:
        best_threshold, best_val_f1, threshold_rows = threshold_sweep(model, val_loader, cfg, device)
        write_rows_csv(cfg.output_dir / "threshold_sweep_val.csv", threshold_rows)
        write_rows_csv(cfg.output_dir / "threshold_sweep.csv", threshold_rows)
        save_threshold_plot(threshold_rows, best_threshold, cfg.output_dir)
        logger.info("Best validation threshold: %.2f (F1=%.4f)", best_threshold, best_val_f1)
    else:
        best_threshold = float(args.threshold)
        threshold_rows = []
        logger.info("Using fixed threshold: %.2f", best_threshold)

    val_summary, _ = validation_pr_analysis(model, val_loader, cfg, device, best_threshold)
    logger.info("Validation AP: %.4f | Validation MCC: %.4f", val_summary["val_ap"], val_summary["val_mcc"])

    logger.info("Final test inference with cache-backed TTA and threshold=%.2f", best_threshold)
    test_prediction_cache = build_prediction_cache(
        model,
        test_loader,
        cfg,
        device,
        threshold=best_threshold,
        use_tta=not args.no_tta,
        stem=f"test_tta_thr_{best_threshold:.3f}" if not args.no_tta else f"test_thr_{best_threshold:.3f}",
    )
    test_metrics_tta, test_rows = evaluate_prediction_cache(test_prediction_cache)
    for key in ["iou", "f1", "precision", "recall", "accuracy", "pred_pos_ratio", "mean_invalid_ratio", "mean_per_image_f1"]:
        logger.info("TTA %s: %.6f", key, test_metrics_tta[key])

    calibration_rows = post_training_calibration_sweep(test_prediction_cache, cfg)
    write_rows_csv(cfg.output_dir / "post_training_calibration_sweep.csv", calibration_rows)

    test_metrics_pp, test_rows_pp = evaluate_loader_postprocessed(
        model,
        test_loader,
        cfg,
        device,
        threshold=best_threshold,
        use_tta=not args.no_tta,
        postprocess=True,
        save_predictions=args.save_predictions,
        save_overlays=args.save_overlays,
        prediction_cache=test_prediction_cache,
    )
    sorted_rows = sorted(test_rows_pp, key=lambda row: row["f1"])
    write_rows_csv(cfg.output_dir / "per_image_test_metrics.csv", sorted_rows)

    if not args.no_visuals and cfg.export_visuals:
        save_qualitative_results(model, test_loader, cfg, device, best_threshold, prediction_cache=test_prediction_cache)
        sorted_rows = save_failure_cases(
            model,
            test_loader.dataset,
            sorted_rows,
            cfg,
            device,
            best_threshold,
            prediction_cache=test_prediction_cache,
        )
        save_hardest_three(
            model,
            test_loader.dataset,
            sorted_rows,
            cfg,
            device,
            best_threshold,
            prediction_cache=test_prediction_cache,
        )
        save_confusion_matrix(test_metrics_pp, cfg.output_dir)

    ablation_rows = build_ablation_rows(test_metrics_pp)
    write_rows_csv(cfg.output_dir / "ablation_table.csv", ablation_rows)
    write_results_summary(cfg, best_threshold, test_metrics_pp, test_metrics_tta, val_summary)
    save_report_assets(cfg, val_summary, test_metrics_pp, ablation_rows)

    for key in ["iou", "f1", "precision", "recall", "pred_pos_ratio", "mean_per_image_f1", "mcc"]:
        logger.info("Postprocessed %s: %.6f", key, test_metrics_pp[key])
    for rank, row in enumerate(sorted_rows[:3], start=1):
        logger.info("Hard case %d | index=%d | F1=%.4f: %s", rank, int(row["index"]), row["f1"], failure_caption(row))


if __name__ == "__main__":
    main()
