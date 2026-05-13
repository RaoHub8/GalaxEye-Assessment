"""Threshold sweep and validation PR analysis from the notebook."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, matthews_corrcoef, precision_recall_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from eosar.config import Config
from eosar.inference import load_prediction_cache, save_prediction_cache, sliding_window_inference, tta_inference
from eosar.metrics import evaluate_loader
from eosar.metrics import confusion_counts, metrics_from_counts
from eosar.postprocessing import postprocess_prob_map
from eosar.utils import cleanup_memory


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def threshold_sweep(
    model: torch.nn.Module,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> tuple[float, float, list[dict[str, float]]]:
    """Sweep validation thresholds from cache-backed valid-pixel probabilities."""
    val_probs, val_targets = collect_valid_prob_targets(model, val_loader, cfg, device, use_tta=False)
    thresholds = np.round(np.arange(cfg.threshold_min, cfg.threshold_max + 1e-9, cfg.threshold_step), 3)
    rows = []
    best_threshold, best_f1 = cfg.threshold, -1.0
    for threshold in thresholds:
        pred = (val_probs >= float(threshold)).astype(np.uint8)
        metrics = metrics_from_counts(confusion_counts(pred, val_targets))
        row = {"threshold": float(threshold), **metrics}
        rows.append(row)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
    return best_threshold, best_f1, rows


def save_threshold_plot(rows: list[dict[str, float]], best_threshold: float, output_dir: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot([r["threshold"] for r in rows], [r["f1"] for r in rows], marker="o", label="F1")
    plt.plot([r["threshold"] for r in rows], [r["precision"] for r in rows], marker=".", label="precision")
    plt.plot([r["threshold"] for r in rows], [r["recall"] for r in rows], marker=".", label="recall")
    plt.axvline(best_threshold, color="red", linestyle="--", label=f"best={best_threshold:.2f}")
    plt.xlabel("threshold")
    plt.ylabel("score")
    plt.title("Validation threshold sweep")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_vs_f1.png", dpi=150)
    plt.savefig(output_dir / "threshold_plot.png", dpi=150)
    plt.close()


def collect_valid_prob_targets(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
    use_tta: bool = False,
    max_images: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect valid-pixel probabilities/targets for validation PR analysis."""
    cache_path = cfg.cache_path("validation_valid_probs")
    if cfg.reuse_cache and cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return data["prob"].astype(np.float32), data["target"].astype(np.uint8)

    infer_fn = tta_inference if use_tta else sliding_window_inference
    prob_parts, target_parts = [], []
    for i, (imgs, masks, valids) in enumerate(tqdm(loader, desc="Collect PR pixels", leave=False)):
        if max_images is not None and i >= max_images:
            break
        _, prob = infer_fn(
            model,
            imgs,
            device=device,
            crop=cfg.crop_size,
            stride=cfg.stride,
            threshold=0.0,
            use_amp=cfg.mixed_precision and device.type == "cuda",
        )
        gt = masks.squeeze().numpy().astype(np.uint8)
        valid = valids.squeeze().numpy().astype(bool)
        if valid.any():
            prob_parts.append(prob[valid].reshape(-1).astype(np.float32))
            target_parts.append(gt[valid].reshape(-1).astype(np.uint8))
        cleanup_memory()
    probs = np.concatenate(prob_parts) if prob_parts else np.array([], dtype=np.float32)
    targets = np.concatenate(target_parts) if target_parts else np.array([], dtype=np.uint8)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, prob=probs.astype(np.float32), target=targets.astype(np.uint8))
    return probs, targets


def build_prediction_cache(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
    threshold: float,
    use_tta: bool = True,
    stem: str | None = None,
) -> list[dict]:
    """Build or load notebook-style full-image prediction cache."""
    cache_path = cfg.cache_path(stem or f"test_tta_thr_{threshold:.3f}")
    if cfg.reuse_cache and cache_path.exists():
        return load_prediction_cache(cache_path)

    infer_fn = tta_inference if use_tta else sliding_window_inference
    cache = []
    for i, (imgs, masks, valids) in enumerate(tqdm(loader, desc="Cache predictions", leave=False)):
        pred, prob = infer_fn(
            model,
            imgs,
            device=device,
            crop=cfg.crop_size,
            stride=cfg.stride,
            threshold=threshold,
            use_amp=cfg.mixed_precision and device.type == "cuda",
        )
        cache.append(
            {
                "index": i,
                "prob": prob.astype(np.float32),
                "pred": pred.astype(np.uint8),
                "gt": masks.squeeze().numpy().astype(np.uint8),
                "valid": valids.squeeze().numpy().astype(np.uint8),
            }
        )
        cleanup_memory()

    save_prediction_cache(cache, cache_path)
    return cache


def evaluate_postprocessed_cache(
    cache: list[dict],
    threshold: float,
    min_component_size: int,
    morph_kernel_size: int = 0,
) -> dict[str, float]:
    """Evaluate cached probabilities after notebook postprocessing."""
    total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for item in cache:
        pred = postprocess_prob_map(
            prob=item["prob"],
            threshold=threshold,
            valid=item["valid"],
            min_component_size=min_component_size,
            morph_kernel_size=morph_kernel_size,
        )
        counts = confusion_counts(pred, item["gt"], item["valid"])
        for key in total:
            total[key] += counts[key]
    return metrics_from_counts(total)


def evaluate_prediction_cache(cache: list[dict]) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Evaluate cached thresholded predictions without extra postprocessing."""
    total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    rows: list[dict[str, float]] = []
    for item in cache:
        pred = (item["pred"] * item["valid"]).astype(np.uint8)
        counts = confusion_counts(pred, item["gt"], item["valid"])
        for key in total:
            total[key] += counts[key]
        metrics = metrics_from_counts(counts)
        rows.append(
            {
                "index": int(item["index"]),
                "f1": metrics["f1"],
                "pred_pos_ratio": metrics["pred_pos_ratio"],
                "invalid_ratio": float(1.0 - item["valid"].mean()),
                **counts,
            }
        )
    metrics = metrics_from_counts(total)
    if rows:
        metrics["mean_per_image_f1"] = float(np.mean([row["f1"] for row in rows]))
        metrics["mean_invalid_ratio"] = float(np.mean([row["invalid_ratio"] for row in rows]))
    return metrics, rows


def post_training_calibration_sweep(cache: list[dict], cfg: Config) -> list[dict[str, float]]:
    """Run the notebook post-training threshold/component/opening calibration grid."""
    rows: list[dict[str, float]] = []
    for threshold in cfg.extended_thresholds:
        for min_size in cfg.min_component_sizes:
            for kernel in (0, cfg.morph_kernel_size) if cfg.morph_kernel_size > 1 else (0,):
                metrics = evaluate_postprocessed_cache(
                    cache,
                    threshold=float(threshold),
                    min_component_size=int(min_size),
                    morph_kernel_size=int(kernel),
                )
                rows.append(
                    {
                        "threshold": float(threshold),
                        "min_component_size": float(min_size),
                        "morph_kernel_size": float(kernel),
                        **metrics,
                    }
                )
    rows.sort(key=lambda row: row["f1"], reverse=True)
    return rows


def validation_pr_analysis(
    model: torch.nn.Module,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    threshold: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Save validation PR curve CSV/PNG and return AP/MCC/point metrics."""
    val_probs, val_targets = collect_valid_prob_targets(model, val_loader, cfg, device, use_tta=False)
    pr_precision, pr_recall, _ = precision_recall_curve(val_targets, val_probs)
    val_ap = average_precision_score(val_targets, val_probs)

    val_pred_at_best = (val_probs >= threshold).astype(np.uint8)
    val_mcc = matthews_corrcoef(val_targets, val_pred_at_best)
    val_point_metrics = metrics_from_counts(confusion_counts(val_pred_at_best, val_targets))

    plt.figure(figsize=(7, 5))
    plt.plot(pr_recall, pr_precision, color="navy", lw=2.0, label=f"Validation PR (AP={val_ap:.4f})")
    plt.scatter(
        val_point_metrics["recall"],
        val_point_metrics["precision"],
        color="crimson",
        s=70,
        zorder=3,
        label=f"chosen threshold={threshold:.2f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Validation Precision-Recall Curve")
    plt.grid(alpha=0.3)
    plt.xlim(0, 1.01)
    plt.ylim(0, 1.01)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(cfg.output_dir / "validation_precision_recall_curve.png", dpi=300)
    plt.close()

    pr_rows = [{"recall": float(r), "precision": float(p)} for r, p in zip(pr_recall, pr_precision)]
    write_rows_csv(cfg.output_dir / "validation_precision_recall_curve.csv", pr_rows)
    summary = {"val_ap": float(val_ap), "val_mcc": float(val_mcc), **val_point_metrics}
    return summary, pr_rows
