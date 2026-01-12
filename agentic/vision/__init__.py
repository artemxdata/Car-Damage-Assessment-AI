from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import cv2


def make_repaired_preview(
    original_rgb: np.ndarray,
    detections: List[Dict[str, Any]],
    alpha: float = 0.35,
) -> np.ndarray:
    """
    'Before/After Vision' (UX magic without real generative).
    We gently "blend/blur" inside detected boxes to simulate repaired appearance.
    Expects original_rgb as RGB uint8.
    Returns RGB uint8.
    """
    if original_rgb is None or len(original_rgb.shape) != 3:
        return original_rgb

    img = original_rgb.copy()
    h, w = img.shape[:2]

    # operate in BGR for OpenCV, return to RGB
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for d in detections or []:
        bbox = d.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(x1), w - 1))
        x2 = max(0, min(int(x2), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        y2 = max(0, min(int(y2), h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        roi = bgr[y1:y2, x1:x2]

        # smooth (like polished)
        smoothed = cv2.bilateralFilter(roi, d=9, sigmaColor=55, sigmaSpace=55)

        # slight inpaint-ish effect by blending to average color
        avg = np.full_like(smoothed, np.mean(smoothed.reshape(-1, 3), axis=0, dtype=np.float32), dtype=np.uint8)
        repaired = cv2.addWeighted(smoothed, 0.75, avg, 0.25, 0)

        # blend back
        bgr[y1:y2, x1:x2] = cv2.addWeighted(repaired, 1.0 - alpha, roi, alpha, 0)

        # subtle white outline to show "repaired zone"
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (230, 230, 230), 2)

    out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return out
