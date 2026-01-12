from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import os
import numpy as np

# Optional dependency (only if SD enabled)
_SD_AVAILABLE = True
try:
    import torch
    from diffusers import AutoPipelineForInpainting
except Exception:
    _SD_AVAILABLE = False


@dataclass
class SDInpaintConfig:
    enabled: bool
    model_id: str = "runwayml/stable-diffusion-inpainting"
    device: str = "cuda"  # "cuda" or "cpu"
    dtype: str = "fp16"   # "fp16" or "fp32"
    num_steps: int = 25
    guidance_scale: float = 6.5
    strength: float = 0.98
    negative_prompt: str = (
        "damage, dent, scratch, crack, broken, rust, text, watermark, label, box, annotation"
    )


_PIPE = None
_PIPE_KEY = None


def _get_dtype(dtype: str):
    if not _SD_AVAILABLE:
        return None
    if dtype.lower() in ("fp16", "float16", "16"):
        return torch.float16
    return torch.float32


def load_sd_config_from_env() -> SDInpaintConfig:
    enabled = os.getenv("INPAINT_SD_ENABLED", "0").strip() == "1"
    return SDInpaintConfig(
        enabled=enabled,
        model_id=os.getenv("INPAINT_SD_MODEL", "runwayml/stable-diffusion-inpainting").strip(),
        device=os.getenv("INPAINT_SD_DEVICE", "cuda").strip(),
        dtype=os.getenv("INPAINT_SD_DTYPE", "fp16").strip(),
        num_steps=int(os.getenv("INPAINT_SD_STEPS", "25")),
        guidance_scale=float(os.getenv("INPAINT_SD_GUIDANCE", "6.5")),
        strength=float(os.getenv("INPAINT_SD_STRENGTH", "0.98")),
        negative_prompt=os.getenv(
            "INPAINT_SD_NEGATIVE",
            "damage, dent, scratch, crack, broken, rust, text, watermark, label, box, annotation",
        ),
    )


def sd_inpaint_bgr(
    image_bgr: np.ndarray,
    mask_01: np.ndarray,
    prompt: str,
    cfg: Optional[SDInpaintConfig] = None,
) -> Tuple[np.ndarray, str]:
    """
    Returns: (out_bgr, status_message)
    mask_01: uint8 or float array, 1 = inpaint region, 0 = keep
    """
    if not _SD_AVAILABLE:
        return image_bgr, "SD inpaint unavailable: diffusers/torch not installed."

    cfg = cfg or load_sd_config_from_env()
    if not cfg.enabled:
        return image_bgr, "SD inpaint disabled by env (INPAINT_SD_ENABLED=0)."

    global _PIPE, _PIPE_KEY

    # Normalize inputs
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        return image_bgr, "SD inpaint: invalid image shape."
    h, w = image_bgr.shape[:2]
    m = mask_01
    if m.shape[:2] != (h, w):
        return image_bgr, "SD inpaint: mask size mismatch."

    m = (m > 0).astype(np.uint8) * 255

    # Lazy-load pipeline (cached)
    key = f"{cfg.model_id}|{cfg.device}|{cfg.dtype}"
    if _PIPE is None or _PIPE_KEY != key:
        dtype = _get_dtype(cfg.dtype)
        _PIPE = AutoPipelineForInpainting.from_pretrained(
            cfg.model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        if cfg.device == "cuda" and torch.cuda.is_available():
            _PIPE = _PIPE.to("cuda")
        else:
            _PIPE = _PIPE.to("cpu")
        _PIPE_KEY = key

    # Convert BGR->RGB PIL-like (diffusers accepts PIL or numpy RGB)
    image_rgb = image_bgr[:, :, ::-1]
    mask_rgb = np.stack([m, m, m], axis=-1)

    neg = cfg.negative_prompt
    try:
        out = _PIPE(
            prompt=prompt,
            negative_prompt=neg,
            image=image_rgb,
            mask_image=mask_rgb,
            num_inference_steps=cfg.num_steps,
            guidance_scale=cfg.guidance_scale,
            strength=cfg.strength,
        ).images[0]
        out_rgb = np.array(out)
        out_bgr = out_rgb[:, :, ::-1].copy()
        return out_bgr, "SD inpaint OK."
    except Exception as e:
        return image_bgr, f"SD inpaint failed: {e}"
