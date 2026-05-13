# GalaxEye EO-SAR Binary Change Detection

This repository implements a binary change detection model for EO-SAR imagery using a lightweight U-Net architecture.

## Repository Structure

```
EO-SAR/
  dataset/                  # local data, not intended for source control
  outputs/                  # checkpoints, metrics, plots, summaries
  scripts/
    eda.py
    prepare_data.py
    verify_data.py
    train.py
    evaluate.py
  notebooks/
    galaxeye-assessment.ipynb
  src/eosar/
    config.py               # hyperparameters
    data.py                 # data loading and preprocessing
    model.py                # U-Net model
    losses.py               # loss functions
    metrics.py              # evaluation metrics
    inference.py            # inference pipeline
    trainer.py              # training loop
    utils.py
    eda.py                  # exploratory data analysis
    postprocessing.py       # postprocessing steps
    thresholding.py         # threshold selection
    visualization.py        # visualization tools
```

## Environment Setup

Create a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

## Dependency Installation

Install required packages:

```bash
pip install -r requirements.txt
```

## Dataset Structure

The repository expects the dataset in the following layout:

```text
dataset/
  train/train/{pre-event,post-event,target}/*.tif
  val/val/{pre-event,post-event,target}/*.tif
  test/test/{pre-event,post-event,target}/*.tif
```

## Quick Execution Flow

1. Install dependencies
2. Prepare dataset folders
3. Run training
4. Run evaluation
5. Review outputs in `outputs/` directory

## Training Command

Train the model:

```bash
python scripts/train.py --data-dir dataset --output-dir outputs --device cuda
```

## Evaluation Command

Evaluate the trained model:

```bash
python scripts/evaluate.py --data-dir dataset --checkpoint outputs/best_model.pth --output-dir outputs --device cuda
```

## Inference Command

Inference is integrated into the evaluation script. Use the evaluation command for predictions.

## Final Metrics

| Split | IoU | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- |
| Validation | 0.6186 | — | — | 0.6186 | — |
| Test | 0.1877 | — | — | 0.3161 | — |

Actual precision, recall, and MCC values will be populated by `scripts/evaluate.py` during the final model run.

## Model Weights

Final checkpoint path: `outputs/best_model.pth`

A public download link for the trained weights will be added once the model file is uploaded manually.

## Methodology Summary

- Model: Lightweight U-Net with early fusion of EO and SAR inputs
- Loss: Masked focal + Dice loss
- Training: Change-aware crop sampling, data augmentation
- Inference: Sliding-window with Gaussian blending, test-time augmentation, postprocessing with connected components
