from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np


def cv_inpaint_bgr(image_bgr: np.ndarray, mask_01: np.ndarray, radius: int = 5) -> Tuple[np.ndarray, str]:
    """
    OpenCV Telea inpaint: быстрый fallback.
    mask_01: 1 = inpaint region, 0 = keep
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        return image_bgr, "CV inpaint: invalid image shape."
    h, w = image_bgr.shape[:2]
    if mask_01.shape[:2] != (h, w):
        return image_bgr, "CV inpaint: mask size mismatch."

    mask = (mask_01 > 0).astype("uint8") * 255
    try:
        out = cv2.inpaint(image_bgr, mask, radius, cv2.INPAINT_TELEA)
        return out, "CV inpaint OK."
    except Exception as e:
        return image_bgr, f"CV inpaint failed: {e}"
