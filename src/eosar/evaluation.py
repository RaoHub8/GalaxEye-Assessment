"""Evaluation loops matching the Kaggle notebook."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader
from tqdm import tqdm

from eosar.config import Config
from eosar.inference import save_mask, save_overlay
from eosar.metrics import confusion_counts, metrics_from_counts
from eosar.postprocessing import postprocess_prob_map
from eosar.thresholding import build_prediction_cache
from eosar.utils import cleanup_memory


def evaluate_loader_postprocessed(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
    threshold: float,
    use_tta: bool = True,
    postprocess: bool = True,
    desc: str = "Postprocess eval",
    save_predictions: bool = False,
    save_overlays: bool = False,
    prediction_cache: list[dict] | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Evaluate with optional TTA and small-component filtering."""
    total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    rows: list[dict[str, float]] = []
    y_true_parts, y_pred_parts = [], []
    pred_dir = cfg.output_dir / "predictions"
    overlay_dir = cfg.output_dir / "overlays"
    dataset = loader.dataset
    cache = prediction_cache or build_prediction_cache(
        model,
        loader,
        cfg,
        device,
        threshold=threshold,
        use_tta=use_tta,
        stem=f"{desc.lower().replace(' ', '_')}_tta_thr_{threshold:.3f}" if use_tta else f"{desc.lower().replace(' ', '_')}_thr_{threshold:.3f}",
    )

    for i, item in enumerate(tqdm(cache, desc=desc, leave=False)):
        prob = item["prob"]
        gt = item["gt"]
        valid = item["valid"]
        if postprocess:
            pred = postprocess_prob_map(
                prob,
                threshold=threshold,
                valid=valid,
                min_component_size=cfg.min_blob_size,
                morph_kernel_size=cfg.morph_kernel_size,
            )
        else:
            pred = (item["pred"] * valid).astype(np.uint8)
        counts = confusion_counts(pred, gt, valid)
        for key in total:
            total[key] += counts[key]
        m = metrics_from_counts(counts)
        rows.append(
            {
                "index": i,
                "f1": m["f1"],
                "pred_pos_ratio": m["pred_pos_ratio"],
                "invalid_ratio": float(1 - valid.mean()),
                **counts,
            }
        )
        if valid.astype(bool).any():
            y_true_parts.append(gt[valid.astype(bool)].reshape(-1).astype(np.uint8))
            y_pred_parts.append(pred[valid.astype(bool)].reshape(-1).astype(np.uint8))

        if save_predictions or save_overlays:
            pre_path, _, mask_path = dataset.samples[int(item["index"])]
            if save_predictions:
                save_mask(pred, mask_path, pred_dir / Path(mask_path).name)
            if save_overlays:
                save_overlay(prob, overlay_dir / f"{Path(pre_path).stem}.png")
        cleanup_memory()

    metrics = metrics_from_counts(total)
    metrics["mean_per_image_f1"] = float(np.mean([r["f1"] for r in rows])) if rows else 0.0
    metrics["mean_invalid_ratio"] = float(np.mean([r["invalid_ratio"] for r in rows])) if rows else 0.0
    metrics["mcc"] = (
        float(matthews_corrcoef(np.concatenate(y_true_parts), np.concatenate(y_pred_parts)))
        if y_true_parts
        else 0.0
    )
    return metrics, rows
