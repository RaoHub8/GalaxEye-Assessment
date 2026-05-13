"""Notebook-faithful EDA utilities for EO-SAR data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from tqdm import tqdm

from eosar.config import Config
from eosar.data import list_samples, raw_validity_mask, read_mask, sample_indices, set_modality_stats


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    return value


def compute_split_eda(cfg: Config, split: str, limit: int = 80) -> dict[str, Any]:
    """Compute the same split statistics as the Kaggle notebook EDA cell."""
    root = cfg.split_root(split)
    samples = list_samples(root)
    pos_ratios, invalid_ratios, zero_patch_flags = [], [], []
    eo_sum = np.zeros(3, dtype=np.float64)
    eo_sumsq = np.zeros(3, dtype=np.float64)
    eo_count = 0
    sar_values = []

    for i in tqdm(sample_indices(len(samples), limit, cfg.seed), desc=f"EDA {split}", leave=False):
        pre_path, post_path, mask_path = samples[int(i)]
        with rasterio.open(pre_path) as src:
            eo_raw = src.read().astype(np.float32)
        with rasterio.open(post_path) as src:
            sar_raw = src.read().astype(np.float32)

        mask = read_mask(mask_path)
        valid = raw_validity_mask(eo_raw, sar_raw, cfg.invalid_zero_threshold)
        valid_bool = valid > 0
        denom = max(int(valid_bool.sum()), 1)
        pos_ratios.append(float(mask[valid_bool].sum() / denom))
        invalid_ratios.append(float(1.0 - valid.mean()))

        eo = eo_raw[:3].astype(np.float32)
        if eo.shape[0] == 1:
            eo = np.repeat(eo, 3, axis=0)
        if eo.max(initial=0) > 1.5:
            eo = eo / 255.0
        eo_valid = eo[:, valid_bool]
        eo_sum += eo_valid.sum(axis=1)
        eo_sumsq += (eo_valid ** 2).sum(axis=1)
        eo_count += eo_valid.shape[1]

        sar = sar_raw[:1].astype(np.float32)
        sar_valid = sar[:, valid_bool].reshape(-1)
        if sar_valid.size:
            sar_values.append(np.percentile(sar_valid, np.linspace(1, 99, 400)))

        h, w = mask.shape
        rng = np.random.default_rng(cfg.seed + int(i))
        for _ in range(8):
            y = int(rng.integers(0, max(h - cfg.crop_size + 1, 1)))
            x = int(rng.integers(0, max(w - cfg.crop_size + 1, 1)))
            patch = mask[y : y + cfg.crop_size, x : x + cfg.crop_size]
            zero_patch_flags.append(float(patch.sum() == 0))

    pos = np.array(pos_ratios)
    invalid = np.array(invalid_ratios)
    eo_mean = eo_sum / max(eo_count, 1)
    eo_std = np.sqrt(np.maximum(eo_sumsq / max(eo_count, 1) - eo_mean**2, 1e-8))
    sar_profile = np.mean(np.stack(sar_values), axis=0) if sar_values else np.zeros(400)
    return {
        "split": split,
        "n_images": len(samples),
        "sampled": len(pos),
        "pos_ratios": pos,
        "invalid_ratios": invalid,
        "zero_patch_pct": float(np.mean(zero_patch_flags) * 100) if zero_patch_flags else 0.0,
        "eo_mean": eo_mean.astype(np.float32),
        "eo_std": eo_std.astype(np.float32),
        "sar_profile": sar_profile.astype(np.float32),
    }


def compute_eda(
    cfg: Config,
    train_limit: int = 80,
    eval_limit: int = 60,
    save: bool = True,
) -> dict[str, dict[str, Any]]:
    """Compute train/val/test EDA and set global EO normalization stats."""
    eda = {
        split: compute_split_eda(cfg, split, limit=train_limit if split == "train" else eval_limit)
        for split in ["train", "val", "test"]
    }
    set_modality_stats(eda["train"]["eo_mean"], np.maximum(eda["train"]["eo_std"], 1e-6))
    if save:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg.output_dir / "eda_summary.json", "w", encoding="utf-8") as f:
            json.dump(_json_ready(eda), f, indent=2)
    return eda


def print_eda_summary(eda: dict[str, dict[str, Any]]) -> None:
    """Print the same concise split summary as the notebook."""
    for split, stats in eda.items():
        pos_pct = stats["pos_ratios"] * 100
        inv_pct = stats["invalid_ratios"] * 100
        print(f"\n{split.upper()} ({stats['sampled']}/{stats['n_images']} sampled)")
        print(
            "  change pixels %: "
            f"mean={pos_pct.mean():.3f}, median={np.median(pos_pct):.3f}, "
            f"min={pos_pct.min():.3f}, max={pos_pct.max():.3f}"
        )
        print(
            "  invalid pixels %: "
            f"mean={inv_pct.mean():.3f}, median={np.median(inv_pct):.3f}, "
            f"min={inv_pct.min():.3f}, max={inv_pct.max():.3f}"
        )
        print(f"  random patches with zero positives: {stats['zero_patch_pct']:.1f}%")
    print("\nEO train mean/std:", eda["train"]["eo_mean"], np.maximum(eda["train"]["eo_std"], 1e-6))


def save_eda_plots(cfg: Config, eda: dict[str, dict[str, Any]]) -> None:
    """Save notebook EDA histograms and SAR percentile profile plots."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, split in enumerate(["train", "val", "test"]):
        axes[0, col].hist(eda[split]["pos_ratios"] * 100, bins=30, color="steelblue", alpha=0.85)
        axes[0, col].set_title(f"{split}: positive pixel %")
        axes[0, col].set_xlabel("change pixels (%)")
        axes[0, col].set_ylabel("images")
        axes[1, col].hist(eda[split]["invalid_ratios"] * 100, bins=30, color="dimgray", alpha=0.85)
        axes[1, col].set_title(f"{split}: invalid pixel %")
        axes[1, col].set_xlabel("invalid pixels (%)")
        axes[1, col].set_ylabel("images")
    plt.tight_layout()
    plt.savefig(cfg.output_dir / "eda_imbalance_invalid_histograms.png", dpi=150)
    plt.close(fig)

    plt.figure(figsize=(7, 4))
    for split in ["train", "val", "test"]:
        plt.plot(np.linspace(1, 99, 400), eda[split]["sar_profile"], label=split)
    plt.title("SAR intensity percentile profiles")
    plt.xlabel("percentile")
    plt.ylabel("raw SAR intensity")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.output_dir / "eda_sar_percentiles.png", dpi=150)
    plt.close()


def save_eda_examples(cfg: Config, split: str = "train", n: int = 4) -> None:
    """Save the notebook's EO/SAR/mask/invalid-overlay example panel."""
    samples = list_samples(cfg.split_root(split))
    if not samples:
        return
    rng = np.random.default_rng(cfg.seed)
    picks = rng.choice(len(samples), size=min(n, len(samples)), replace=False)
    fig, axes = plt.subplots(len(picks), 4, figsize=(14, 3.4 * len(picks)))
    if len(picks) == 1:
        axes = axes[None, :]

    for row, idx in enumerate(picks):
        pre_path, post_path, mask_path = samples[int(idx)]
        with rasterio.open(pre_path) as src:
            eo_raw = src.read().astype(np.float32)
        with rasterio.open(post_path) as src:
            sar_raw = src.read().astype(np.float32)
        mask = read_mask(mask_path)
        valid = raw_validity_mask(eo_raw, sar_raw, cfg.invalid_zero_threshold)

        eo = eo_raw[:3]
        if eo.shape[0] == 1:
            eo = np.repeat(eo, 3, axis=0)
        if eo.max(initial=0) > 1.5:
            eo = eo / 255.0
        eo = np.transpose(np.clip(eo, 0, 1), (1, 2, 0))

        sar = sar_raw[0]
        sar_disp = np.clip(sar, np.percentile(sar, 1), np.percentile(sar, 99))
        sar_disp = (sar_disp - sar_disp.min()) / (sar_disp.max() - sar_disp.min() + 1e-8)

        axes[row, 0].imshow(eo)
        axes[row, 1].imshow(sar_disp, cmap="gray")
        axes[row, 2].imshow(mask, cmap="hot")
        axes[row, 3].imshow(eo)
        axes[row, 3].imshow(1 - valid, cmap="cool", alpha=0.45)
        axes[row, 0].set_ylabel(pre_path.stem[:18], fontsize=8)
        for ax, title in zip(axes[row], ["EO", "SAR", "Change mask", "Invalid overlay"]):
            ax.set_title(title)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(cfg.output_dir / f"eda_{split}_scene_examples.png", dpi=150)
    plt.close(fig)


def run_eda(
    cfg: Config,
    train_limit: int = 80,
    eval_limit: int = 60,
    examples_split: str = "train",
    examples: int = 4,
) -> dict[str, dict[str, Any]]:
    """Run the notebook EDA cells end to end."""
    eda = compute_eda(cfg, train_limit=train_limit, eval_limit=eval_limit, save=True)
    print_eda_summary(eda)
    save_eda_plots(cfg, eda)
    save_eda_examples(cfg, split=examples_split, n=examples)
    return eda
