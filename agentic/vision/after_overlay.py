from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .inpaint import make_after_preview


def make_repaired_preview(
    original_rgb: np.ndarray,
    primary: Optional[Dict[str, Any]] = None,
    intensity: float = 0.7,
) -> Dict[str, Any]:
    """
    Returns dict with:
      - after: RGB image
      - mask: mask uint8
      - diff: diff map uint8
      - meta: debug metadata
    """
    res = make_after_preview(original_rgb=original_rgb, primary=primary, intensity=float(intensity))
    return {"after": res.after, "mask": res.mask, "diff": res.diff, "meta": res.meta}
