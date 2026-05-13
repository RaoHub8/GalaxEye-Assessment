"""Segmentation metrics for binary change detection."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from eosar.config import Config
from eosar.inference import sliding_window_inference, tta_inference


def confusion_counts(
    preds: np.ndarray,
    targets: np.ndarray,
    valid: np.ndarray | None = None,
) -> dict[str, int]:
    """Return TP/FP/FN/TN counts for binary arrays, optionally masked by validity."""
    p = preds.astype(bool).reshape(-1)
    t = targets.astype(bool).reshape(-1)
    
    if valid is not None:
        v = valid.astype(bool).reshape(-1)
        p, t = p[v], t[v]
    
    return {
        "tp": int(np.logical_and(p, t).sum()),
        "fp": int(np.logical_and(p, ~t).sum()),
        "fn": int(np.logical_and(~p, t).sum()),
        "tn": int(np.logical_and(~p, ~t).sum()),
    }


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    """Compute notebook binary metrics plus MCC from confusion counts."""
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    pred_pos_ratio = (tp + fp) / (tp + fp + fn + tn + eps)
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + eps
    mcc = ((tp * tn) - (fp * fn)) / mcc_denom
    
    return {
        "iou": float(iou),
        "dice": float(f1),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "pred_pos_ratio": float(pred_pos_ratio),
        "mcc": float(mcc),
        **{k: float(v) for k, v in counts.items()},
    }


def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    valid: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute binary segmentation metrics from prediction and target arrays."""
    return metrics_from_counts(confusion_counts(preds, targets, valid))


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
    use_tta: bool = False,
    threshold: float = 0.5,
    desc: str = "Validation",
    return_per_image: bool = False,
) -> dict[str, float] | tuple[dict[str, float], list[dict]]:
    """Evaluate a model with sliding-window inference over a dataloader.
    
    Returns:
        If return_per_image=False: dict of aggregate metrics
        If return_per_image=True: (aggregate_metrics, per_image_metrics_list)
    """
    model.eval()
    infer_fn = tta_inference if use_tta else sliding_window_inference
    total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    rows = []
    
    for i, batch_data in enumerate(tqdm(loader, desc=desc, leave=False)):
        # Handle both 2-tuple (imgs, masks) and 3-tuple (imgs, masks, valids)
        if len(batch_data) == 3:
            imgs, masks, valids = batch_data
            valid = valids.squeeze().numpy().astype(np.uint8)
        else:
            imgs, masks = batch_data
            valid = None
        
        pred_mask, prob = infer_fn(
            model,
            imgs,
            device=device,
            crop=cfg.crop_size,
            stride=cfg.stride,
            threshold=threshold,
            use_amp=cfg.mixed_precision and device.type == "cuda",
        )
        
        gt = masks.squeeze().numpy().astype(np.uint8)
        counts = confusion_counts(pred_mask, gt, valid)
        
        for key in total:
            total[key] += counts[key]
        
        m = metrics_from_counts(counts)
        
        invalid_ratio = float(1.0 - valid.mean()) if valid is not None else 0.0
        rows.append({
            "index": i,
            "f1": m["f1"],
            "pred_pos_ratio": m["pred_pos_ratio"],
            "invalid_ratio": invalid_ratio,
            **counts,
        })
    
    aggregate_metrics = metrics_from_counts(total)
    
    if rows:
        aggregate_metrics["mean_per_image_f1"] = float(np.mean([r["f1"] for r in rows]))
        aggregate_metrics["mean_invalid_ratio"] = float(np.mean([r["invalid_ratio"] for r in rows]))
    
    return (aggregate_metrics, rows) if return_per_image else aggregate_metrics
