"""Runtime utilities for device selection, reproducibility, and logging."""

from __future__ import annotations

import csv
import gc
import logging
import platform
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def make_run_dir(base_dir: Path, run_name: str | None = None) -> Path:
    """Create a timestamped experiment directory when requested."""
    if run_name is None:
        return Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base_dir) / f"{timestamp}_{run_name}"


def setup_logging(
    output_dir: Path,
    log_filename: str = "training.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure a project logger with console and file handlers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eosar")
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(output_dir / log_filename, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def get_device(prefer: str = "auto") -> tuple[torch.device, str]:
    """Select DirectML, CUDA, or CPU with graceful fallback."""
    prefer = prefer.lower()
    if prefer in {"auto", "directml", "dml"}:
        try:
            import torch_directml

            if torch_directml.is_available():
                return torch_directml.device(), "DirectML"
        except Exception as exc:
            if prefer in {"directml", "dml"}:
                logging.getLogger("eosar").warning("DirectML unavailable: %s", exc)

    if prefer in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda"), f"CUDA ({torch.cuda.get_device_name(0)})"

    return torch.device("cpu"), "CPU"


def is_directml_device(device: torch.device) -> bool:
    """Return True for torch-directml privateuseone devices."""
    return str(device).startswith("privateuseone")


def supports_amp(device: torch.device) -> bool:
    """AMP is only enabled for CUDA in this project."""
    return device.type == "cuda"


def get_device_info(prefer: str = "auto") -> dict[str, Any]:
    """Collect device and package information for logs."""
    device, name = get_device(prefer)
    info: dict[str, Any] = {
        "device": str(device),
        "device_name": name,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch_directml

        info["directml_installed"] = True
        info["directml_available"] = bool(torch_directml.is_available())
    except Exception:
        info["directml_installed"] = False
        info["directml_available"] = False

    if device.type == "cuda":
        info["cuda_available"] = True
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    elif is_directml_device(device):
        info["directml_selected"] = True
    else:
        info["cpu_only"] = True
    return info


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def cleanup_memory() -> None:
    """Release Python and CUDA caches when available."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class CSVLogger:
    """Append epoch metrics to a CSV file."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None

    def log(self, row: dict[str, Any]) -> None:
        """Append one row, creating the header on first write."""
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            write_header = not self.path.exists() or self.path.stat().st_size == 0
        else:
            write_header = False

        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in self._fieldnames})
