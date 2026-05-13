"""EO-SAR binary change detection package."""

from eosar.config import Config
from eosar.data import EOSARDataset, build_dataloaders, preprocess_eo, preprocess_sar, preprocess_mask
from eosar.model import EOSARChangeDetector, LightweightUNet
from eosar.losses import FocalDiceLoss
from eosar.metrics import evaluate_loader, compute_metrics, confusion_counts, metrics_from_counts
from eosar.inference import sliding_window_inference, tta_inference, predict_pair
from eosar.postprocessing import apply_optional_opening, postprocess_prob_map, remove_small_components
from eosar.trainer import Trainer, create_optimizer, create_scheduler
from eosar.utils import set_seed, get_device, cleanup_memory, setup_logging, CSVLogger
from eosar.eda import compute_eda, run_eda

__version__ = "1.0.0"
__author__ = "Anonymous"

__all__ = [
    "Config",
    "EOSARDataset",
    "build_dataloaders",
    "preprocess_eo",
    "preprocess_sar",
    "preprocess_mask",
    "EOSARChangeDetector",
    "LightweightUNet",
    "FocalDiceLoss",
    "evaluate_loader",
    "compute_metrics",
    "confusion_counts",
    "metrics_from_counts",
    "sliding_window_inference",
    "tta_inference",
    "predict_pair",
    "remove_small_components",
    "apply_optional_opening",
    "postprocess_prob_map",
    "Trainer",
    "create_optimizer",
    "create_scheduler",
    "set_seed",
    "get_device",
    "cleanup_memory",
    "setup_logging",
    "CSVLogger",
    "compute_eda",
    "run_eda",
]
