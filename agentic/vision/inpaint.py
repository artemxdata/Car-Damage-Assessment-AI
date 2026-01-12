from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def _as_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _pick(d: Any, key: str, default=None):
    if d is None:
        return default
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


@dataclass
class InpaintResult:
    after: np.ndarray
    mask: np.ndarray
    diff: np.ndarray
    meta: Dict[str, Any]


def bbox_to_mask(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    pad_px: int = 14,
    feather: int = 9,
) -> np.ndarray:
    """
    Creates a soft mask from bbox for inpainting.
    Returns uint8 mask [H,W] with 0..255.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad_px)
    y1 = max(0, y1 - pad_px)
    x2 = min(w - 1, x2 + pad_px)
    y2 = min(h - 1, y2 + pad_px)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

    if feather and feather > 0:
        k = feather if feather % 2 == 1 else feather + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    return mask


def _diff_map(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """
    Returns a grayscale diff heatmap-like image (uint8).
    """
    b = before.astype(np.int16)
    a = after.astype(np.int16)
    d = np.abs(a - b).mean(axis=2)  # H,W
    d = np.clip(d * 4.0, 0, 255).astype(np.uint8)  # amplify
    d = cv2.GaussianBlur(d, (7, 7), 0)
    return d


def smart_cv_inpaint(
    bgr: np.ndarray,
    mask: np.ndarray,
    intensity: float = 0.7,
) -> np.ndarray:
    """
    A more 'visible' inpaint than plain cv2.inpaint:
    - does inpaint
    - then mild local smoothing / tone match inside mask
    """
    intensity = float(np.clip(intensity, 0.0, 1.0))

    # binary mask for cv2.inpaint
    hard = (mask > 16).astype(np.uint8) * 255

    # inpaint
    # TELEA tends to look more natural for scratches; NS can be better for larger holes.
    inpainted = cv2.inpaint(bgr, hard, 3, cv2.INPAINT_TELEA)

    # extra blending to make "after" noticeable but still plausible
    # create a softened alpha from mask
    alpha = (mask.astype(np.float32) / 255.0) * intensity
    alpha = cv2.GaussianBlur(alpha, (0, 0), 3)
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha3 = np.repeat(alpha[..., None], 3, axis=2)

    # local smoothing on inpainted (only affects masked zone after blend)
    smooth = cv2.bilateralFilter(inpainted, d=7, sigmaColor=35, sigmaSpace=35)

    out = (bgr.astype(np.float32) * (1 - alpha3) + smooth.astype(np.float32) * alpha3).astype(np.uint8)
    return out


def make_after_preview(
    original_rgb: np.ndarray,
    primary: Optional[Dict[str, Any]] = None,
    intensity: float = 0.7,
    pad_px: int = 18,
) -> InpaintResult:
    """
    original_rgb: RGB uint8 image (H,W,3)
    primary: detection dict containing bbox
    """
    if original_rgb is None or original_rgb.size == 0:
        raise ValueError("original_rgb is empty")

    rgb = original_rgb
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    bbox = _pick(primary, "bbox", None)
    if bbox is None:
        # no bbox -> return identity with empty diff
        diff = np.zeros(rgb.shape[:2], dtype=np.uint8)
        mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        return InpaintResult(after=rgb, mask=mask, diff=diff, meta={"mode": "noop"})

    x1, y1, x2, y2 = map(_as_int, bbox)
    mask = bbox_to_mask(bgr, (x1, y1, x2, y2), pad_px=pad_px, feather=11)

    after_bgr = smart_cv_inpaint(bgr, mask, intensity=float(intensity))
    after_rgb = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2RGB)

    diff = _diff_map(rgb, after_rgb)

    return InpaintResult(
        after=after_rgb,
        mask=mask,
        diff=diff,
        meta={"mode": "cv_inpaint", "bbox": [x1, y1, x2, y2], "intensity": float(intensity), "pad_px": int(pad_px)},
    )
