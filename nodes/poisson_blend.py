# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
"""
Eric Poisson Blend node.

Wraps OpenCV's seamlessClone (Pérez et al. 2003 Poisson image editing) so any
ComfyUI workflow can paste a source image into a destination through a mask
without a visible seam. The pasted region's overall colour / brightness drift
toward the destination's local tone, while keeping the source's gradient
field (shape, texture, internal structure) intact.

Three modes:

- normal     (cv2.NORMAL_CLONE)        Standard Poisson editing. Best for
                                       opaque solid objects; full colour /
                                       luminance match to destination.
- mixed      (cv2.MIXED_CLONE)         At each pixel the stronger gradient
                                       (source vs dest) wins. Best for
                                       transparent / wispy / textured things
                                       (foliage, smoke, hair) where some
                                       destination structure should show
                                       through.
- monochrome (cv2.MONOCHROME_TRANSFER) Source contributes only luminance;
                                       hue/saturation come from the dest.

Inputs/outputs follow ComfyUI conventions (IMAGE = (B,H,W,3) float [0,1],
MASK = (B,H,W) float [0,1] or (H,W)).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_to_np_uint8(img: torch.Tensor) -> np.ndarray:
    """ComfyUI IMAGE -> HxWx3 uint8."""
    if img.dim() == 4:
        img = img[0]
    a = img.detach().cpu().numpy()
    a = np.clip(a * 255.0, 0, 255).astype(np.uint8)
    return a


def _mask_to_np_uint8(mask: torch.Tensor) -> np.ndarray:
    """ComfyUI MASK -> HxW uint8 (0 or 255)."""
    if mask is None:
        return None
    m = mask
    if m.dim() == 3:
        m = m[0]
    a = m.detach().cpu().numpy().astype(np.float32)
    return (a > 0.5).astype(np.uint8) * 255


def _np_to_image_tensor(a: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 -> (1,H,W,3) float tensor."""
    arr = a.astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


def _np_to_mask_tensor(a: np.ndarray) -> torch.Tensor:
    """HxW uint8 -> (1,H,W) float tensor."""
    arr = a.astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


def _shrink_mask(m: np.ndarray, px: int) -> np.ndarray:
    import cv2
    if px <= 0 or m is None:
        return m
    k = int(px)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    return cv2.erode(m, kernel, iterations=1)


def _feather_mask(m: np.ndarray, px: int) -> np.ndarray:
    """Soft alpha for the post-blend re-mix step (NOT passed to seamlessClone,
    which requires a hard binary mask)."""
    import cv2
    if px <= 0 or m is None:
        return (m.astype(np.float32) / 255.0)
    sigma = max(0.5, float(px))
    ksz = int(2 * round(sigma * 2) + 1)
    f = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (ksz, ksz), sigma)
    return np.clip(f, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class EricPoissonBlend:
    """Seamlessly paste a source image into a background through a mask."""

    CATEGORY = "Eric_Composer_Studio/compose"
    FUNCTION = "blend"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "used_mask")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background": ("IMAGE", {
                    "tooltip": "Destination canvas. The output is the same size as this."
                }),
                "source": ("IMAGE", {
                    "tooltip": "Image containing the object to clone in. Must be the SAME SIZE as background "
                               "(crop/pad the source to background size first if needed; centre + offset_x/y "
                               "are then applied to the mask only). For a different-sized source, see the "
                               "auto_fit toggle below."
                }),
                "source_mask": ("MASK", {
                    "tooltip": "Hard mask over source: which pixels to clone in. Anything > 0.5 is IN. "
                               "Soft masks are binarised \u2014 OpenCV requires a binary mask."
                }),
                "mode": (["normal", "mixed", "monochrome"], {
                    "default": "normal",
                    "tooltip": "normal (NORMAL_CLONE): standard Poisson, full colour/luminance match. "
                               "Best for opaque objects.\n"
                               "mixed (MIXED_CLONE): per-pixel stronger gradient wins. Best for transparent / "
                               "wispy / textured things (foliage, smoke, hair).\n"
                               "monochrome (MONOCHROME_TRANSFER): source contributes only luminance; "
                               "hue/saturation come from the destination."
                }),
            },
            "optional": {
                "auto_fit": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If source is a different size than background, automatically resize source "
                               "and source_mask to background size BEFORE blending. With auto_fit OFF a "
                               "size mismatch is an error."
                }),
                "center_x": ("INT", {
                    "default": -1, "min": -1, "max": 16384, "step": 1,
                    "tooltip": "Destination x-coordinate where the mask centroid lands (in pixels). "
                               "-1 = use the source mask's own centroid (no movement)."
                }),
                "center_y": ("INT", {
                    "default": -1, "min": -1, "max": 16384, "step": 1,
                    "tooltip": "Destination y-coordinate where the mask centroid lands. -1 = auto."
                }),
                "offset_x": ("INT", {
                    "default": 0, "min": -16384, "max": 16384, "step": 1,
                    "tooltip": "Additional x-shift applied AFTER center_x (lets you nudge without "
                               "overriding the centroid)."
                }),
                "offset_y": ("INT", {
                    "default": 0, "min": -16384, "max": 16384, "step": 1,
                    "tooltip": "Additional y-shift applied AFTER center_y."
                }),
                "mask_shrink_px": ("INT", {
                    "default": 1, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Erode mask by N pixels before cloning. Recommended >= 1 because OpenCV's "
                               "seamlessClone fails / produces black borders if the mask touches the image "
                               "border. Also reduces fringe artefacts at sharp matte edges."
                }),
                "post_feather_px": ("INT", {
                    "default": 0, "min": 0, "max": 128, "step": 1,
                    "tooltip": "After Poisson blending, re-mix the result over the original background "
                               "using a feathered version of the mask (Gaussian blur). 0 disables. "
                               "Use this to soften the very edge of the cloned region without breaking "
                               "the gradient match (the Poisson result is already seamless; this only "
                               "helps if your mask edge is jagged)."
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Linearly blend (1-s)*background + s*poisson_result inside the mask. "
                               "1.0 = full Poisson. <1.0 lets some of the original (unmodified) source "
                               "show through (only meaningful if source was already pre-pasted at the "
                               "same coords as the mask)."
                }),
            },
        }

    # ------------------------------------------------------------------
    def blend(
        self,
        background: torch.Tensor,
        source: torch.Tensor,
        source_mask: torch.Tensor,
        mode: str,
        auto_fit: bool = True,
        center_x: int = -1,
        center_y: int = -1,
        offset_x: int = 0,
        offset_y: int = 0,
        mask_shrink_px: int = 1,
        post_feather_px: int = 0,
        strength: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        import cv2

        # ---- Normalise inputs ----
        bg = _image_to_np_uint8(background)
        src = _image_to_np_uint8(source)
        msk = _mask_to_np_uint8(source_mask)

        bg_h, bg_w = bg.shape[:2]
        src_h, src_w = src.shape[:2]
        msk_h, msk_w = msk.shape[:2]

        # Mask must match source size
        if (msk_h, msk_w) != (src_h, src_w):
            msk = cv2.resize(msk, (src_w, src_h), interpolation=cv2.INTER_NEAREST)

        # Source must match background size (or auto-fit)
        if (src_h, src_w) != (bg_h, bg_w):
            if not auto_fit:
                raise ValueError(
                    f"[EricPoissonBlend] source size {(src_w, src_h)} != background "
                    f"{(bg_w, bg_h)} and auto_fit is OFF."
                )
            src = cv2.resize(src, (bg_w, bg_h), interpolation=cv2.INTER_LANCZOS4)
            msk = cv2.resize(msk, (bg_w, bg_h), interpolation=cv2.INTER_NEAREST)
            src_h, src_w = bg_h, bg_w

        # ---- Mask sanity ----
        msk = _shrink_mask(msk, mask_shrink_px)
        if int(msk.sum()) == 0:
            print("[EricPoissonBlend] mask is empty after shrink \u2014 returning background unchanged.")
            return (_np_to_image_tensor(bg), _np_to_mask_tensor(msk))

        # Avoid mask touching image borders (cv2 limitation)
        if (msk[0, :].sum() or msk[-1, :].sum() or
                msk[:, 0].sum() or msk[:, -1].sum()):
            border = 2
            padded_src = cv2.copyMakeBorder(src, border, border, border, border, cv2.BORDER_REPLICATE)
            padded_bg = cv2.copyMakeBorder(bg, border, border, border, border, cv2.BORDER_REPLICATE)
            padded_msk = cv2.copyMakeBorder(msk, border, border, border, border, cv2.BORDER_CONSTANT, value=0)
            do_pad = True
        else:
            padded_src, padded_bg, padded_msk = src, bg, msk
            do_pad = False

        # ---- Compute centre ----
        ys, xs = np.where(padded_msk > 0)
        if xs.size == 0:
            return (_np_to_image_tensor(bg), _np_to_mask_tensor(msk))
        cx_auto = int(round((xs.min() + xs.max()) / 2.0))
        cy_auto = int(round((ys.min() + ys.max()) / 2.0))

        cx = cx_auto if center_x < 0 else int(center_x) + (2 if do_pad else 0)
        cy = cy_auto if center_y < 0 else int(center_y) + (2 if do_pad else 0)
        cx += int(offset_x)
        cy += int(offset_y)

        # Clamp so the mask's bounding box fits in the dest
        bbox_w = int(xs.max() - xs.min()) + 1
        bbox_h = int(ys.max() - ys.min()) + 1
        H, W = padded_bg.shape[:2]
        cx = int(np.clip(cx, bbox_w // 2 + 1, W - bbox_w // 2 - 2))
        cy = int(np.clip(cy, bbox_h // 2 + 1, H - bbox_h // 2 - 2))

        flag = {
            "normal": cv2.NORMAL_CLONE,
            "mixed": cv2.MIXED_CLONE,
            "monochrome": cv2.MONOCHROME_TRANSFER,
        }.get(mode, cv2.NORMAL_CLONE)

        # ---- Run Poisson ----
        try:
            blended = cv2.seamlessClone(
                padded_src, padded_bg, padded_msk, (cx, cy), flag
            )
        except cv2.error as e:
            print(f"[EricPoissonBlend] cv2.seamlessClone failed: {e}")
            return (_np_to_image_tensor(bg), _np_to_mask_tensor(msk))

        # Strip border padding
        if do_pad:
            blended = blended[2:-2, 2:-2]
            msk = padded_msk[2:-2, 2:-2]

        # ---- Optional strength + feather re-mix ----
        if strength < 1.0 or post_feather_px > 0:
            soft = _feather_mask(msk, post_feather_px) if post_feather_px > 0 \
                else (msk.astype(np.float32) / 255.0)
            soft = soft * float(strength)
            soft3 = soft[..., None]
            mix = (
                blended.astype(np.float32) * soft3
                + bg.astype(np.float32) * (1.0 - soft3)
            )
            blended = np.clip(mix, 0, 255).astype(np.uint8)

        return (_np_to_image_tensor(blended), _np_to_mask_tensor(msk))
