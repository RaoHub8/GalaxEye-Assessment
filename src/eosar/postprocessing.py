"""Notebook-faithful binary mask postprocessing."""

from __future__ import annotations

import numpy as np

try:
    from scipy import ndimage as ndi
except Exception:
    ndi = None


def remove_small_components(mask: np.ndarray, min_size: int = 24) -> np.ndarray:
    """Remove connected components smaller than ``min_size`` pixels.

    This matches the notebook implementation, including the pure NumPy fallback
    used when SciPy is unavailable.
    """
    mask = mask.astype(bool)
    if min_size <= 1:
        return mask.astype(np.uint8)
    if ndi is not None:
        labels, n = ndi.label(mask)
        if n == 0:
            return mask.astype(np.uint8)
        sizes = np.bincount(labels.ravel())
        keep = sizes >= min_size
        keep[0] = False
        return keep[labels].astype(np.uint8)

    return mask.astype(np.uint8)


def apply_optional_opening(mask: np.ndarray, kernel_size: int = 0) -> np.ndarray:
    """Apply notebook optional binary opening after thresholding."""
    if kernel_size is None or int(kernel_size) <= 1:
        return mask.astype(np.uint8)
    if ndi is None:
        raise RuntimeError("scipy.ndimage is required for binary opening but ndi is None.")
    structure = np.ones((int(kernel_size), int(kernel_size)), dtype=bool)
    opened = ndi.binary_opening(mask.astype(bool), structure=structure)
    return opened.astype(np.uint8)


def postprocess_prob_map(
    prob: np.ndarray,
    threshold: float,
    valid: np.ndarray,
    min_component_size: int,
    morph_kernel_size: int = 0,
) -> np.ndarray:
    """Threshold, optionally open, filter components, then suppress invalid pixels."""
    pred = (prob >= float(threshold)).astype(np.uint8)
    pred = apply_optional_opening(pred, morph_kernel_size)
    pred = remove_small_components(pred, int(min_component_size))
    pred[(valid == 0)] = 0
    return pred.astype(np.uint8)
