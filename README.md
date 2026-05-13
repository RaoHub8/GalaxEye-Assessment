# GalaxEye EO-SAR Binary Change Detection

Clean repository refactor of the Kaggle notebook for the GalaxEye EO-SAR Binary Change Detection assignment.

The notebook is the source of truth. This codebase preserves its lightweight early-fusion U-Net pipeline, preprocessing, masked focal + Dice loss, change-aware crop sampling, sliding-window inference, geometric TTA, validation threshold selection, and connected-component postprocessing.

## Task

The model receives co-registered Earth Observation optical RGB imagery and SAR backscatter imagery and predicts a binary change mask. The repository expects the same split layout as the notebook:

```text
dataset/
  train/train/{pre-event,post-event,target}/*.tif
  val/val/{pre-event,post-event,target}/*.tif
  test/test/{pre-event,post-event,target}/*.tif
```

## Repository Structure

```text
EO-SAR/
  dataset/                  # local data, not intended for source control
  outputs/                  # checkpoints, metrics, plots, summaries
  outputs_*/                # local/generated artifacts, ignored by Git
  scripts/
    eda.py
    prepare_data.py
    verify_data.py
    train.py
    evaluate.py
  notebooks/
    galaxeye-assessment.ipynb
  src/eosar/
    config.py               # notebook hyperparameters
    data.py                 # EO/SAR loading, normalization, masks, crops, augmentation
    model.py                # lightweight early-fusion U-Net
    losses.py               # masked focal + Dice loss
    metrics.py              # masked IoU/F1/precision/recall/MCC
    inference.py            # Gaussian sliding-window inference, TTA, caches
    trainer.py              # AMP training, resume, checkpointing, logging
    utils.py
    eda.py                  # notebook EDA/statistics cells
    postprocessing.py       # threshold/opening/components/invalid suppression
    thresholding.py         # threshold sweep, PR analysis, prediction caches
    visualization.py        # qualitative grids and failure cases
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Kaggle/CUDA, use CUDA PyTorch. On this Windows machine, CPU execution works but full-image evaluation is slow.

## Data Preparation

If the split zip files already exist in `dataset/`, extract them with:

```bash
python scripts/prepare_data.py --data-dir dataset
```

Verify file counts and sampled label ranges:

```bash
python scripts/verify_data.py --data-dir dataset
```

Expected counts are 2781 train, 334 val, and 77 test triplets.

## EDA

The Kaggle notebook runs EDA before training. The same cells are preserved here:

```bash
python scripts/eda.py --data-dir dataset --output-dir outputs
```

This writes:

```text
outputs/
  eda_summary.json
  eda_imbalance_invalid_histograms.png
  eda_sar_percentiles.png
  eda_train_scene_examples.png
```

`scripts/train.py` runs this EDA/statistics step by default before training so the EO normalization stats match the notebook. Use `--skip-eda` only when resuming quickly and you are sure the preprocessing statistics are already correct for the dataset.

Generated outputs, caches, checkpoints, local datasets, and notebooks are intentionally ignored by Git. For submission, commit the source files, scripts, README, requirements, pyproject, license, and small documentation files only.

## Training

Notebook-faithful defaults are in `src/eosar/config.py`: 20 epochs, batch size 8, 256 crops, stride 128, base channels 32, AMP enabled on CUDA, positive crop bias, validity masking, and masked focal + Dice loss.

```bash
python scripts/train.py --data-dir dataset --output-dir outputs --device cuda
```

For the assignment evaluators with the remaining dataset and GPU, the intended clean run is:

```bash
python scripts/verify_data.py --data-dir dataset
python scripts/eda.py --data-dir dataset --output-dir outputs
python scripts/train.py --data-dir dataset --output-dir outputs --device cuda
python scripts/evaluate.py --data-dir dataset --checkpoint outputs/best_model.pth --output-dir outputs --device cuda
```

Resume is automatic from `outputs/last_model.pth` unless disabled:

```bash
python scripts/train.py --no-resume
```

Low-memory machines can run a smaller execution preset, but that is not the Kaggle notebook training configuration:

```bash
python scripts/train.py --preset low --device cpu
```

## Evaluation

Notebook-style evaluation loads `outputs/best_model.pth`, recomputes the train EO normalization statistics, sweeps validation thresholds from 0.20 to 0.80, then runs test inference with TTA and small-blob filtering:

```bash
python scripts/evaluate.py --data-dir dataset --checkpoint outputs/best_model.pth --output-dir outputs --device cuda
```

For a fixed threshold:

```bash
python scripts/evaluate.py --threshold 0.5
```

Expected evaluation artifacts:

```text
outputs/
  best_model.pth
  last_model.pth
  metrics.csv
  history.json
  training_curves.png
  eda_summary.json
  eda_imbalance_invalid_histograms.png
  eda_sar_percentiles.png
  eda_train_scene_examples.png
  threshold_sweep_val.csv
  threshold_sweep.csv
  threshold_vs_f1.png
  threshold_plot.png
  validation_precision_recall_curve.csv
  validation_precision_recall_curve.png
  per_image_test_metrics.csv
  ablation_table.csv
  qualitative_results.png
  qualitative_results_with_validity.png
  confusion_matrix.png
  confusion_matrix_masked_postprocessed.png
  results_summary.txt
  hardest_three_failure_cases.png
  failure_cases/
  report_assets/
```

## Notebook-Faithful Details

- EO preprocessing: RGB channels are scaled to `[0, 1]`, clipped, then z-scored with train split statistics.
- SAR preprocessing: single SAR channel, optional `log1p` disabled by default, per-image p2/p98 clipping, then min-max normalization to `[0, 1]`.
- Validity mask: pixels where both EO and SAR are near zero are excluded from loss and metrics.
- Training crops: 256 by 256 patches with notebook change-aware sampling behavior.
- Augmentation: flips, 90-degree rotations, EO-only color jitter/blur, and SAR-only speckle/scale perturbation.
- Model: lightweight four-level U-Net with GroupNorm, ReLU, bilinear upsampling, and 4-channel early fusion.
- Inference: overlapping sliding-window probabilities are blended with a Gaussian weight map, then thresholded.
- TTA: original, horizontal flip, vertical flip, and both-flip predictions are averaged. The notebook does not use transpose TTA.
- Validation analysis: threshold sweep, best-threshold selection, precision-recall curve, AP, and MCC.
- Final notebook test flow: cache-backed TTA probabilities, valid-pixel masking, thresholding, optional opening, connected-component filtering, invalid-region suppression, per-image metrics, qualitative panels, failure-case plots, confusion matrix, ablation table, report assets, and `results_summary.txt`.

## Expected Metrics

The successful Kaggle notebook run reported approximately:

| Split | Metric | Value |
| --- | --- | ---: |
| Validation | F1 | 0.6186 |
| Test | F1 | 0.3161 |
| Test | IoU | 0.1877 |

Current local checkpoints may differ if they were produced by partial smoke runs or low-memory settings. To compare fairly, use a checkpoint trained with the notebook configuration.

## Reproducibility Notes

- Seed defaults to `42`.
- DataLoader workers default to `0`, matching the notebook’s deterministic single-process loading.
- CUDA AMP is enabled only on CUDA.
- `last_model.pth` stores optimizer, scheduler, best score, epoch, config, and history for automatic resume.
- The repository includes flexible checkpoint loading for both notebook-style state dicts and the structured repo wrapper state dicts.

## Limitations

This repository is a faithful engineering refactor, not a new research pipeline. It intentionally does not introduce new architectures, pretrained encoders, alternative losses, or extra postprocessing beyond what appears in the notebook.
