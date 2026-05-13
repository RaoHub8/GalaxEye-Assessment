"""Qualitative and error-analysis visualizations from the notebook."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader

import eosar.data as data_mod
from eosar.config import Config
from eosar.inference import tta_inference
from eosar.postprocessing import postprocess_prob_map


def denorm_eo_for_display(img_tensor: torch.Tensor) -> np.ndarray:
    eo = img_tensor[:3].detach().cpu().numpy()
    eo = eo * data_mod.MODALITY_STATS["eo_std"][:, None, None] + data_mod.MODALITY_STATS["eo_mean"][:, None, None]
    return np.transpose(np.clip(eo, 0, 1), (1, 2, 0))


def error_rgb(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    if valid is None:
        valid = np.ones_like(gt, dtype=np.uint8)
    rgb = np.zeros((*gt.shape, 3), dtype=np.float32)
    v = valid.astype(bool)
    rgb[(gt == 1) & (pred == 1) & v] = [0, 1, 0]
    rgb[(gt == 0) & (pred == 1) & v] = [1, 0, 0]
    rgb[(gt == 1) & (pred == 0) & v] = [0, 0.25, 1]
    rgb[~v] = [0.6, 0.0, 0.8]
    return rgb


def save_qualitative_results(
    model: torch.nn.Module,
    test_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    threshold: float,
    prediction_cache: list[dict] | None = None,
) -> None:
    samples_info = []
    for i, (imgs, masks, valids) in enumerate(test_loader):
        valid = valids.squeeze().numpy().astype(bool)
        gt = masks.squeeze().numpy()
        samples_info.append((i, float(gt[valid].mean() if valid.any() else gt.mean()), imgs, masks, valids))
    if not samples_info:
        return
    samples_info.sort(key=lambda x: x[1])
    picks = [
        samples_info[max(0, len(samples_info) // 10)],
        samples_info[len(samples_info) // 2],
        samples_info[min(len(samples_info) - 1, 9 * len(samples_info) // 10)],
    ]

    fig, axes = plt.subplots(len(picks), 6, figsize=(21, 4 * len(picks)))
    if len(picks) == 1:
        axes = axes[None, :]
    headers = ["EO", "SAR", "Ground truth", "Probability", "Postprocessed", "Error + invalid"]
    for col, title in enumerate(headers):
        axes[0, col].set_title(title, fontweight="bold")

    cache_by_index = {int(item["index"]): item for item in prediction_cache or []}
    for row, (idx, change_pct, imgs, masks, valids) in enumerate(picks):
        if idx in cache_by_index:
            prob = cache_by_index[idx]["prob"]
        else:
            _, prob = tta_inference(
                model,
                imgs,
                device=device,
                crop=cfg.crop_size,
                stride=cfg.stride,
                threshold=threshold,
                use_amp=cfg.mixed_precision and device.type == "cuda",
            )
        valid = valids.squeeze().numpy().astype(np.uint8)
        gt = masks.squeeze().numpy().astype(np.uint8)
        pred_pp = postprocess_prob_map(
            prob,
            threshold=threshold,
            valid=valid,
            min_component_size=cfg.min_blob_size,
            morph_kernel_size=cfg.morph_kernel_size,
        )
        sar = imgs[0, 3].numpy()
        sar_disp = (sar - sar.min()) / (sar.max() - sar.min() + 1e-8)
        axes[row, 0].imshow(denorm_eo_for_display(imgs[0]))
        axes[row, 1].imshow(sar_disp, cmap="gray")
        axes[row, 2].imshow(gt, cmap="hot")
        axes[row, 3].imshow(prob, cmap="hot", vmin=0, vmax=1)
        axes[row, 4].imshow(pred_pp, cmap="hot")
        axes[row, 5].imshow(error_rgb(pred_pp, gt, valid))
        axes[row, 0].set_ylabel(f"idx={idx}, change={100 * change_pct:.2f}%", fontsize=8)
        for ax in axes[row]:
            ax.axis("off")

    legend = [
        mpatches.Patch(color="green", label="TP"),
        mpatches.Patch(color="red", label="FP"),
        mpatches.Patch(color=(0, 0.25, 1), label="FN"),
        mpatches.Patch(color=(0.6, 0, 0.8), label="Invalid"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()
    plt.savefig(cfg.output_dir / "qualitative_results_with_validity.png", dpi=150, bbox_inches="tight")
    plt.savefig(cfg.output_dir / "qualitative_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def failure_caption(row: dict) -> str:
    fp, fn = row.get("fp", 0), row.get("fn", 0)
    invalid_ratio = row.get("invalid_ratio", 0.0)
    pred_ratio = row.get("pred_pos_ratio", 0.0)
    if invalid_ratio > 0.02 and fp >= fn:
        return "Likely border-driven false positives near invalid co-registration regions."
    if fp > 2 * max(fn, 1):
        return "Likely SAR texture-induced pseudo changes or bright backscatter confusion."
    if fn > fp:
        return "Likely missed sparse change under train-test prevalence or appearance shift."
    if pred_ratio > 0.05:
        return "Likely noisy edge activations producing overly large change blobs."
    return "Mixed EO-SAR ambiguity with both false alarms and missed small structures."


def save_failure_cases(
    model: torch.nn.Module,
    test_dataset,
    rows: list[dict],
    cfg: Config,
    device: torch.device,
    threshold: float,
    max_cases: int = 8,
    prediction_cache: list[dict] | None = None,
) -> list[dict]:
    failure_dir = cfg.output_dir / "failure_cases"
    failure_dir.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda r: r["f1"])
    cache_by_index = {int(item["index"]): item for item in prediction_cache or []}
    for rank, row in enumerate(sorted_rows[:max_cases]):
        idx = int(row["index"])
        imgs, masks, valids = test_dataset[idx]
        if idx in cache_by_index:
            prob = cache_by_index[idx]["prob"]
        else:
            _, prob = tta_inference(
                model,
                imgs.unsqueeze(0),
                device=device,
                crop=cfg.crop_size,
                stride=cfg.stride,
                threshold=threshold,
                use_amp=cfg.mixed_precision and device.type == "cuda",
            )
        gt = masks.squeeze().numpy().astype(np.uint8)
        valid = valids.squeeze().numpy().astype(np.uint8)
        pred_pp = postprocess_prob_map(
            prob,
            threshold=threshold,
            valid=valid,
            min_component_size=cfg.min_blob_size,
            morph_kernel_size=cfg.morph_kernel_size,
        )
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        axes[0].imshow(denorm_eo_for_display(imgs))
        axes[0].set_title("EO")
        axes[1].imshow(prob, cmap="hot", vmin=0, vmax=1)
        axes[1].set_title("Probability")
        axes[2].imshow(pred_pp, cmap="hot")
        axes[2].set_title("Prediction")
        axes[3].imshow(error_rgb(pred_pp, gt, valid))
        axes[3].set_title(f"Error map F1={row['f1']:.3f}")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(failure_dir / f"hard_case_{rank:02d}_idx_{idx}.png", dpi=150)
        plt.close(fig)
    return sorted_rows


def save_hardest_three(
    model: torch.nn.Module,
    test_dataset,
    sorted_rows: list[dict],
    cfg: Config,
    device: torch.device,
    threshold: float,
    prediction_cache: list[dict] | None = None,
) -> None:
    hardest_three = sorted_rows[:3]
    if not hardest_three:
        return
    fig, axes = plt.subplots(len(hardest_three), 4, figsize=(15, 4 * len(hardest_three)))
    if len(hardest_three) == 1:
        axes = axes[None, :]
    cache_by_index = {int(item["index"]): item for item in prediction_cache or []}
    for rank, row in enumerate(hardest_three):
        idx = int(row["index"])
        imgs, masks, valids = test_dataset[idx]
        if idx in cache_by_index:
            prob = cache_by_index[idx]["prob"]
        else:
            _, prob = tta_inference(
                model,
                imgs.unsqueeze(0),
                device=device,
                crop=cfg.crop_size,
                stride=cfg.stride,
                threshold=threshold,
                use_amp=cfg.mixed_precision and device.type == "cuda",
            )
        gt = masks.squeeze().numpy().astype(np.uint8)
        valid = valids.squeeze().numpy().astype(np.uint8)
        pred_pp = postprocess_prob_map(
            prob,
            threshold=threshold,
            valid=valid,
            min_component_size=cfg.min_blob_size,
            morph_kernel_size=cfg.morph_kernel_size,
        )
        axes[rank, 0].imshow(denorm_eo_for_display(imgs))
        axes[rank, 0].set_title(f"EO sample {idx}")
        axes[rank, 1].imshow(prob, cmap="magma", vmin=0, vmax=1)
        axes[rank, 1].set_title("Predicted probability")
        axes[rank, 2].imshow(gt, cmap="gray")
        axes[rank, 2].set_title("Ground truth")
        axes[rank, 3].imshow(error_rgb(pred_pp, gt, valid))
        axes[rank, 3].set_title(f"Error map F1={row['f1']:.3f}")
        axes[rank, 0].set_ylabel(failure_caption(row), fontsize=9)
        for ax in axes[rank]:
            ax.axis("off")
    plt.suptitle("Three Hardest Failure Cases with Likely Causes", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(cfg.output_dir / "hardest_three_failure_cases.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(metrics: dict[str, float], output_dir: Path) -> None:
    cm = np.array([[int(metrics["tn"]), int(metrics["fp"])], [int(metrics["fn"]), int(metrics["tp"])]])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No-change", "Change"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Masked confusion matrix - test set")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_masked_postprocessed.png", dpi=150)
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
