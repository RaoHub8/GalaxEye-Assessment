"""Notebook report asset and summary exports."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eosar.config import Config


def build_ablation_rows(test_metrics: dict[str, float]) -> list[dict[str, float | str]]:
    return [
        {"Variant": "Baseline model", "IoU": np.nan, "F1": np.nan, "Precision": np.nan, "Recall": np.nan},
        {"Variant": "+ validity masking", "IoU": np.nan, "F1": np.nan, "Precision": np.nan, "Recall": np.nan},
        {"Variant": "+ change-aware sampling", "IoU": np.nan, "F1": np.nan, "Precision": np.nan, "Recall": np.nan},
        {
            "Variant": "+ postprocessing (current final)",
            "IoU": test_metrics["iou"],
            "F1": test_metrics["f1"],
            "Precision": test_metrics["precision"],
            "Recall": test_metrics["recall"],
        },
    ]


def write_results_summary(
    cfg: Config,
    threshold: float,
    test_metrics_pp: dict[str, float],
    test_metrics_tta: dict[str, float],
    val_summary: dict[str, float],
) -> None:
    cm_counts = {k: int(test_metrics_pp[k]) for k in ["tp", "fp", "fn", "tn"]}
    summary = f"""
GalaxEye EO-SAR Change Detection - Robustness Refinement Summary
================================================================

Model        : Lightweight early-fusion U-Net
Input        : EO RGB + SAR, modality-normalized separately
Loss         : Masked focal + Dice
Sampling     : Change-aware training patches ({cfg.positive_patch_prob:.0%} positive-biased)
Inference    : Gaussian-weighted sliding window + geometric TTA
Postprocess  : Threshold, optional opening ({cfg.morph_kernel_size}), remove components smaller than {cfg.min_blob_size}, suppress invalid pixels
Threshold    : {threshold:.2f} chosen on validation F1

Test metrics, invalid pixels excluded:
  IoU              : {test_metrics_pp['iou']:.4f}
  F1 / Dice        : {test_metrics_pp['f1']:.4f}
  Precision        : {test_metrics_pp['precision']:.4f}
  Recall           : {test_metrics_pp['recall']:.4f}
  Accuracy         : {test_metrics_pp['accuracy']:.4f}
  Pred positive %  : {100 * test_metrics_pp['pred_pos_ratio']:.4f}
  Mean image F1    : {test_metrics_pp['mean_per_image_f1']:.4f}
  Mean invalid %   : {100 * test_metrics_tta.get('mean_invalid_ratio', test_metrics_pp.get('mean_invalid_ratio', 0.0)):.4f}
  Test MCC         : {test_metrics_pp['mcc']:.4f}
  Validation AP    : {val_summary.get('val_ap', 0.0):.4f}
  Validation MCC   : {val_summary.get('val_mcc', 0.0):.4f}

Error profile:
  TP: {cm_counts['tp']:,}
  FP: {cm_counts['fp']:,}
  FN: {cm_counts['fn']:,}
  TN: {cm_counts['tn']:,}

Conclusion:
  The refinement keeps the original lightweight architecture and evaluation flow, while adding per-image SAR
  p2/p98 normalization, Gaussian-weighted sliding-window blending, threshold selection, connected-component
  filtering, and final invalid-region suppression. These changes target SAR intensity shift, tile-boundary
  artifacts, tiny edge activations, and invalid-border false positives.
"""
    (cfg.output_dir / "results_summary.txt").write_text(summary, encoding="utf-8")


def save_report_assets(
    cfg: Config,
    val_summary: dict[str, float],
    test_metrics_pp: dict[str, float],
    ablation_rows: list[dict],
) -> None:
    report_dir = cfg.output_dir / "report_assets"
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_summary_df = pd.DataFrame(
        [
            {
                "split": "validation",
                "stage": "threshold_selected",
                "IoU": val_summary["iou"],
                "F1": val_summary["f1"],
                "Precision": val_summary["precision"],
                "Recall": val_summary["recall"],
                "AP": val_summary.get("val_ap", np.nan),
                "MCC": val_summary.get("val_mcc", np.nan),
            },
            {
                "split": "test",
                "stage": "tta_postprocessed",
                "IoU": test_metrics_pp["iou"],
                "F1": test_metrics_pp["f1"],
                "Precision": test_metrics_pp["precision"],
                "Recall": test_metrics_pp["recall"],
                "AP": np.nan,
                "MCC": test_metrics_pp["mcc"],
            },
        ]
    )
    metrics_summary_df.to_csv(report_dir / "metrics_summary.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(report_dir / "ablation_table.csv", index=False)

    for name in [
        "validation_precision_recall_curve.csv",
        "threshold_sweep_val.csv",
        "threshold_sweep.csv",
        "validation_precision_recall_curve.png",
        "threshold_vs_f1.png",
        "threshold_plot.png",
        "qualitative_results_with_validity.png",
        "hardest_three_failure_cases.png",
        "confusion_matrix_masked_postprocessed.png",
        "eda_imbalance_invalid_histograms.png",
        "eda_sar_percentiles.png",
    ]:
        src = cfg.output_dir / name
        if src.exists():
            shutil.copy2(src, report_dir / name)

    plt.figure(figsize=(7, 4.5))
    plot_df = metrics_summary_df.set_index(["split", "stage"])[["IoU", "F1", "Precision", "Recall", "MCC"]]
    plot_df.T.plot(kind="bar", ax=plt.gca(), width=0.8)
    plt.title("Final Validation/Test Metrics Summary")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="split / stage", fontsize=8)
    plt.tight_layout()
    plt.savefig(report_dir / "final_metrics_summary_barplot.png", dpi=300, bbox_inches="tight")
    plt.close()
