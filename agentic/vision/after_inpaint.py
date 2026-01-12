from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np
import cv2

from agentic.vision.inpaint_cv import cv_inpaint_bgr
from agentic.vision.inpaint_sd import load_sd_config_from_env, sd_inpaint_bgr


@dataclass
class AfterPreviewResult:
    before_bgr: np.ndarray
    after_bgr: np.ndarray
    diff_bgr: np.ndarray
    status: str
    used_method: str


def _pick(d: Any, key: str, default=None):
    if d is None:
        return default
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


def _make_soft_mask_from_bbox(
    h: int,
    w: int,
    bbox: Tuple[int, int, int, int],
    pad: float = 0.12,
    feather: int = 21,
) -> np.ndarray:
    """
    bbox: x1,y1,x2,y2 in pixels
    Returns mask_01 float32 in [0..1]
    """
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    px = int(bw * pad)
    py = int(bh * pad)

    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w - 1, x2 + px)
    y2 = min(h - 1, y2 + py)

    mask = np.zeros((h, w), dtype=np.float32)
    mask[y1 : y2 + 1, x1 : x2 + 1] = 1.0

    # Feather edges (gaussian blur)
    feather = int(feather)
    if feather % 2 == 0:
        feather += 1
    feather = max(3, feather)
    mask = cv2.GaussianBlur(mask, (feather, feather), 0)
    mask = np.clip(mask, 0.0, 1.0)
    return mask


def _bbox_from_detection(det: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    bbox = det.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return (x1, y1, x2, y2)


def make_repaired_after_preview(
    original_bgr: np.ndarray,
    primary_detection: Optional[Dict[str, Any]],
    intensity: float = 0.65,
    method: str = "auto",  # "auto" | "sd" | "cv"
) -> AfterPreviewResult:
    """
    Uses inpainting on the primary bbox region to simulate 'repaired' look.
    intensity controls how wide/soft the mask is (bigger => stronger change).
    """
    before = original_bgr.copy()
    h, w = before.shape[:2]

    if not primary_detection:
        return AfterPreviewResult(before, before, np.zeros_like(before), "No primary detection.", "none")

    bbox = _bbox_from_detection(primary_detection)
    if bbox is None:
        return AfterPreviewResult(before, before, np.zeros_like(before), "Primary detection has no bbox.", "none")

    # Mask params: higher intensity => more padding + more feather (more natural blend)
    pad = 0.06 + 0.22 * float(np.clip(intensity, 0.0, 1.0))
    feather = int(15 + 35 * float(np.clip(intensity, 0.0, 1.0)))
    mask_f = _make_soft_mask_from_bbox(h, w, bbox, pad=pad, feather=feather)
    mask_01 = (mask_f > 0.15).astype(np.uint8)  # hard mask for inpainting

    dmg_type = str(primary_detection.get("type", "damage"))
    sev = str(primary_detection.get("severity", "unknown")).lower()

    prompt = (
        f"photo of a car, {dmg_type} repaired, clean smooth car body panel, "
        f"no {dmg_type}, no dents, no scratches, realistic texture, natural lighting"
    )
    if sev in ("severe", "moderate"):
        prompt += ", professional repair finish"

    used = "cv"
    status_parts = []

    # Decide backend
    if method == "sd" or method == "auto":
        sd_cfg = load_sd_config_from_env()
        if sd_cfg.enabled:
            after_sd, msg = sd_inpaint_bgr(before, mask_01, prompt=prompt, cfg=sd_cfg)
            status_parts.append(msg)
            if "OK" in msg:
                after = after_sd
                used = "sd"
            else:
                after = before
        else:
            status_parts.append("SD disabled -> fallback to CV.")
            after = before
    else:
        after = before

    if used != "sd":
        # Fallback OpenCV inpaint
        after_cv, msg = cv_inpaint_bgr(before, mask_01, radius=max(3, int(4 + 10 * intensity)))
        status_parts.append(msg)
        after = after_cv
        used = "cv"

    # Diff visualization (what changed)
    diff = cv2.absdiff(before, after)

    return AfterPreviewResult(
        before_bgr=before,
        after_bgr=after,
        diff_bgr=diff,
        status=" | ".join(status_parts),
        used_method=used,
    )
