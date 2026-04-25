"""
DepthComposer node for Eric_Composer_Studio.

Composites up to 4 depth map patches onto a configurable canvas.
Each slot accepts a depth image (grayscale, bright=near) and an optional mask.
If a slot has no mask and its BG toggle is set, it becomes the background image
(cover-fit to canvas). If it has no mask and no BG flag, depth is auto-thresholded.

Compositing order: back-to-front by z-order. Front element mask pixels fully
replace back element pixels — no cross-element blending. Within-element feathering
comes naturally from rembg/SAM soft mask edges.

Depth shift per element:
    output = clamp(depth_placement + (depth − mean(masked_region)) × gradient_scale, 0, 1)

JSON composition_data structure:
{
  "canvas_w": 1024,
  "canvas_h": 1024,
  "background_depth": 0.10,
  "slots": [
    {
      "id": "s0",
      "label": "Slot 1",
      "is_bg": false,
      "depth_placement": 0.7,
      "gradient_scale": 1.0,
      "visible": true,
      "z_order": 0,        // 0 = backmost among elements; re-sorted on compose
      "x": 0.5,            // centroid as fraction of canvas (0-1)
      "y": 0.5,
      "scale": 1.0,        // uniform scale factor
      "scale_x": 1.0,      // additional X stretch
      "scale_y": 1.0,      // additional Y stretch
      "rotation": 0.0,     // degrees
      "src_w": 512,        // source image width (filled by Python on input)
      "src_h": 512,
    }, ...
  ]
}
"""

from __future__ import annotations
import json
import math
import os
import hashlib
import tempfile
import time

import numpy as np

MAX_SLOTS = 4

ELEMENT_COLORS = [
    (74,  158, 255),
    (255, 107, 53 ),
    (76,  175, 80 ),
    (224, 64,  251),
]

_SLOT_KEYS = (
    "id", "label", "is_bg", "depth_placement", "gradient_scale",
    "visible", "z_order", "x", "y", "scale", "scale_x", "scale_y",
    "rotation", "src_w", "src_h",
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _tensor_to_np_gray(t) -> np.ndarray:
    """ComfyUI IMAGE tensor [B,H,W,C] or [B,H,W,1] → (H,W) float32 [0,1]."""
    arr = t[0].cpu().numpy().astype(np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)   # RGB→gray (already gray in practice)
    return arr


def _tensor_to_np_mask(t) -> np.ndarray:
    """ComfyUI MASK tensor [B,H,W] → (H,W) float32 [0,1]."""
    return t[0].cpu().numpy().astype(np.float32)


def _np_gray_to_tensor(arr: np.ndarray):
    """(H,W) float32 → ComfyUI IMAGE tensor [1,H,W,3]."""
    import torch
    rgb = np.stack([arr, arr, arr], axis=-1)
    return torch.from_numpy(rgb[None]).float()


def _apply_false_color(arr: np.ndarray):
    """Apply a simple magma-like false-color LUT to a (H,W) float32 [0,1] array."""
    import torch
    # Simple warm-cool gradient: black→blue→cyan→green→yellow→red→white
    lut = np.array([
        [0.0,  0.0,  0.0 ],
        [0.1,  0.0,  0.3 ],
        [0.3,  0.0,  0.6 ],
        [0.6,  0.0,  0.5 ],
        [0.9,  0.3,  0.0 ],
        [1.0,  0.75, 0.0 ],
        [1.0,  1.0,  0.9 ],
    ], dtype=np.float32)
    n = len(lut) - 1
    idx_f = np.clip(arr * n, 0, n)
    idx_lo = idx_f.astype(np.int32)
    idx_hi = np.clip(idx_lo + 1, 0, n)
    frac = (idx_f - idx_lo)[..., None]
    rgb = lut[idx_lo] * (1 - frac) + lut[idx_hi] * frac
    return torch.from_numpy(rgb[None]).float()


def _cover_fit(src_w, src_h, dst_w, dst_h):
    """Return scale such that the source covers the destination (cover fit)."""
    return max(dst_w / src_w, dst_h / src_h)


def _contain_fit(src_w, src_h, dst_w, dst_h):
    return min(dst_w / src_w, dst_h / src_h)


def _auto_threshold_mask(depth: np.ndarray, bg_depth: float) -> np.ndarray:
    """Simple threshold: pixels above bg_depth + 0.1 are considered the element."""
    return (depth > bg_depth + 0.1).astype(np.float32)


def _shift_depth(depth: np.ndarray, mask: np.ndarray,
                 depth_placement: float, gradient_scale: float) -> np.ndarray:
    """
    Shift the depth patch so its masked mean lands at depth_placement,
    with the internal gradient compressed by gradient_scale.
    """
    where = mask > 0.5
    if not where.any():
        return depth
    mean_val = float(depth[where].mean())
    shifted = depth_placement + (depth - mean_val) * gradient_scale
    return np.clip(shifted, 0.0, 1.0)


def _affine_transform_patch(
    depth: np.ndarray, mask: np.ndarray,
    canvas_w: int, canvas_h: int,
    cx: float, cy: float,         # centroid as fraction of canvas
    scale: float,
    scale_x: float, scale_y: float,
    rotation_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Warp depth+mask into a canvas-sized array using the given transform.
    Returns (depth_on_canvas, mask_on_canvas) both (H,W) float32.
    """
    import cv2

    src_h, src_w = depth.shape[:2]
    if src_w == 0 or src_h == 0:
        z = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        return z, z

    # Build composite transform:
    #  1. Translate src centroid to origin
    #  2. Scale (uniform × per-axis)
    #  3. Rotate
    #  4. Translate to canvas position
    src_cx = src_w / 2.0
    src_cy = src_h / 2.0
    dst_cx = cx * canvas_w
    dst_cy = cy * canvas_h

    total_sx = scale * scale_x
    total_sy = scale * scale_y

    # OpenCV affine: we specify destination coords of 3 source points
    rad = math.radians(rotation_deg)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)

    def transform_pt(px, py):
        # Shift to origin
        lx = (px - src_cx) * total_sx
        ly = (py - src_cy) * total_sy
        # Rotate
        rx = lx * cos_r - ly * sin_r
        ry = lx * sin_r + ly * cos_r
        # Shift to canvas
        return rx + dst_cx, ry + dst_cy

    p0 = transform_pt(0, 0)
    p1 = transform_pt(src_w, 0)
    p2 = transform_pt(0, src_h)

    src_pts = np.float32([[0, 0], [src_w, 0], [0, src_h]])
    dst_pts = np.float32([p0, p1, p2])

    M = cv2.getAffineTransform(src_pts, dst_pts)

    depth_canvas = cv2.warpAffine(
        depth, M, (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    mask_canvas = cv2.warpAffine(
        mask, M, (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    return depth_canvas.astype(np.float32), np.clip(mask_canvas, 0.0, 1.0).astype(np.float32)


def _composite(slots_ordered: list, canvas_w: int, canvas_h: int, bg_depth: float) -> np.ndarray:
    """
    Composite depth patches back-to-front.
    Each entry in slots_ordered: (depth_np, mask_np, slot_dict)
    Returns (H, W) float32 canvas.
    """
    canvas = np.full((canvas_h, canvas_w), bg_depth, dtype=np.float32)

    for depth_patch, mask_patch, slot in slots_ordered:
        if not slot.get("visible", True):
            continue

        # Shift depth to target placement
        shifted = _shift_depth(
            depth_patch, mask_patch,
            slot["depth_placement"],
            slot["gradient_scale"],
        )

        # Composite: where mask > 0 blend onto canvas
        # Hard edge for mask > 0.5, feathered blending via soft mask
        alpha = mask_patch  # use soft mask for natural feathering
        canvas = canvas * (1.0 - alpha) + shifted * alpha

    return canvas


# ── temp PNG serving ──────────────────────────────────────────────────────────

try:
    import folder_paths as _fp
    def _tmp_dir() -> str:
        return os.path.join(_fp.get_output_directory(), "eric_depth_tmp")
except ImportError:
    def _tmp_dir() -> str:
        return os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "output", "eric_depth_tmp")


def _save_slot_thumb(node_id: str, slot_idx: int, depth: np.ndarray, mask: np.ndarray | None):
    """Save a depth patch as PNG for JS preview. Returns path."""
    import cv2
    folder = os.path.join(_tmp_dir(), str(node_id))
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"slot_{slot_idx}.png")

    # Blend depth over a mid-gray background using mask for preview
    if mask is not None:
        preview = depth * mask + 0.2 * (1.0 - mask)
    else:
        preview = depth
    img_u8 = (np.clip(preview, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(path, img_u8)
    return path


# ── REST endpoint ─────────────────────────────────────────────────────────────

try:
    from server import PromptServer
    from aiohttp import web as _aio_web

    @PromptServer.instance.routes.get("/eric_composer_studio/depth_slot/{node_id}/{slot}")
    async def _eric_depth_slot(request):
        node_id  = request.match_info["node_id"]
        slot_idx = request.match_info["slot"]
        path = os.path.join(_tmp_dir(), str(node_id), f"slot_{slot_idx}.png")
        if not os.path.isfile(path):
            raise _aio_web.HTTPNotFound()
        return _aio_web.FileResponse(path)

except Exception as _e:
    print(f"[Eric_Composer_Studio] DepthComposer: could not register REST endpoint: {_e}")


# ── default composition data ──────────────────────────────────────────────────

def _default_composition_data(canvas_w=1024, canvas_h=1024) -> dict:
    slots = []
    for i in range(MAX_SLOTS):
        slots.append({
            "id": f"s{i}",
            "label": f"Slot {i+1}",
            "is_bg": False,
            "depth_placement": 0.70,
            "gradient_scale": 1.0,
            "visible": True,
            "z_order": i,
            "x": 0.5,
            "y": 0.5,
            "scale": 1.0,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "rotation": 0.0,
            "src_w": 0,
            "src_h": 0,
        })
    return {
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "background_depth": 0.10,
        "slots": slots,
    }


def _merge_composition_data(existing_str: str, depths, masks, node_id: str) -> dict:
    """
    Parse existing composition_data, update src dimensions from live inputs,
    save slot thumbs, and return updated dict.
    """
    try:
        cd = json.loads(existing_str) if existing_str.strip() else {}
    except Exception:
        cd = {}

    if "slots" not in cd or len(cd["slots"]) != MAX_SLOTS:
        cd = _default_composition_data(
            cd.get("canvas_w", 1024), cd.get("canvas_h", 1024)
        )

    for i in range(MAX_SLOTS):
        d = depths[i]
        m = masks[i]
        slot = cd["slots"][i]
        if d is not None:
            depth_np = _tensor_to_np_gray(d)
            mask_np  = _tensor_to_np_mask(m) if m is not None else None
            slot["src_w"] = depth_np.shape[1]
            slot["src_h"] = depth_np.shape[0]
            _save_slot_thumb(node_id, i, depth_np, mask_np)
        else:
            slot["src_w"] = 0
            slot["src_h"] = 0

    return cd


# ── main node class ───────────────────────────────────────────────────────────

class DepthComposer:
    """
    Composites up to 4 depth map patches onto a configurable canvas.

    Typical upstream workflow:
      [Image] → [DepthPro Estimate] → [Depth Metric to Relative (invert=true)]
                                            │
      [Image] → [SAM2 / rembg] → mask ─────┘
                                            │
                              [Depth Composer]
    """

    CATEGORY     = "Eric_Composer_Studio"
    FUNCTION     = "compose"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("depth_image", "preview")
    OUTPUT_NODE  = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "composition_data": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": False,
                    "tooltip": "Internal composition state — managed by the canvas UI. Do not edit manually.",
                }),
            },
            "optional": {
                "depth_1": ("IMAGE",),
                "mask_1":  ("MASK",),
                "depth_2": ("IMAGE",),
                "mask_2":  ("MASK",),
                "depth_3": ("IMAGE",),
                "mask_3":  ("MASK",),
                "depth_4": ("IMAGE",),
                "mask_4":  ("MASK",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def IS_CHANGED(cls, composition_data="", unique_id="0",
                   depth_1=None, mask_1=None,
                   depth_2=None, mask_2=None,
                   depth_3=None, mask_3=None,
                   depth_4=None, mask_4=None):
        # Re-run whenever composition_data or any input changes.
        parts = [composition_data or ""]
        for d in [depth_1, depth_2, depth_3, depth_4]:
            if d is not None:
                parts.append(f"{d.shape}")
        return hashlib.md5("|".join(parts).encode()).hexdigest()

    def compose(self, composition_data="", unique_id="0",
                depth_1=None, mask_1=None,
                depth_2=None, mask_2=None,
                depth_3=None, mask_3=None,
                depth_4=None, mask_4=None):

        depths = [depth_1, depth_2, depth_3, depth_4]
        masks  = [mask_1,  mask_2,  mask_3,  mask_4 ]

        # ── Parse / update composition data ───────────────────────────────────
        cd = _merge_composition_data(composition_data, depths, masks, unique_id)
        canvas_w   = int(cd.get("canvas_w",   1024))
        canvas_h   = int(cd.get("canvas_h",   1024))
        bg_depth   = float(cd.get("background_depth", 0.10))

        # ── Build slot list sorted by z_order (back-to-front) ─────────────────
        # Find which slot is BG (if any).
        bg_slot_idx = None
        for i, slot in enumerate(cd["slots"]):
            if slot.get("is_bg", False) and depths[i] is not None:
                bg_slot_idx = i
                break

        slots_for_composite = []

        # Process each active slot
        for i, slot in enumerate(cd["slots"]):
            if depths[i] is None:
                continue
            if not slot.get("visible", True):
                continue

            depth_np = _tensor_to_np_gray(depths[i])
            src_h, src_w = depth_np.shape[:2]
            mask_np = _tensor_to_np_mask(masks[i]) if masks[i] is not None else None

            slot_cx = slot.get("x", 0.5)
            slot_cy = slot.get("y", 0.5)

            if i == bg_slot_idx:
                # BG flag means no mask cutout — use full white mask.
                # Depth ordering is still governed by depth_placement like every other slot.
                eff_mask = np.ones_like(depth_np)
            else:
                if mask_np is not None:
                    eff_mask = mask_np
                else:
                    # No mask, no BG flag — auto-threshold
                    eff_mask = _auto_threshold_mask(depth_np, bg_depth)
                    print(
                        f"[Eric_Composer_Studio] DepthComposer: slot {i+1} has no mask and "
                        f"no BG flag — auto-thresholding. Connect a mask for best results."
                    )
            # Sort key: (depth_placement, z_order) — brighter=higher dp=closer=drawn last.
            # Same rule applies to all slots including BG.
            z = (float(slot.get("depth_placement", 0.7)), int(slot.get("z_order", i)))

            # slot.scale is canvas-relative: max(src_w, src_h) * scale → canvas_w pixels.
            # This matches the JS preview which uses the same max_dim / canvas_size convention.
            max_src_dim = max(src_w, src_h) if max(src_w, src_h) > 0 else 1
            canvas_base = canvas_w / max_src_dim
            eff_scale   = float(slot.get("scale",   0.5)) * canvas_base
            eff_scale_x = float(slot.get("scale_x", 1.0))
            eff_scale_y = float(slot.get("scale_y", 1.0))
            eff_rot     = float(slot.get("rotation", 0.0))

            depth_c, mask_c = _affine_transform_patch(
                depth_np, eff_mask, canvas_w, canvas_h,
                slot_cx, slot_cy,
                eff_scale, eff_scale_x, eff_scale_y, eff_rot,
            )

            slots_for_composite.append((
                depth_c, mask_c, slot, z
            ))

        # Sort back-to-front: ascending (depth_placement, z_order).
        # Lower depth_placement = darker = further away = drawn first (behind).
        # BG slot has key (-1.0, -1) so it always sits behind all elements.
        slots_for_composite.sort(key=lambda x: x[3])

        # ── Composite ─────────────────────────────────────────────────────────
        canvas = np.full((canvas_h, canvas_w), bg_depth, dtype=np.float32)

        for depth_patch, mask_patch, slot, _z in slots_for_composite:
            shifted = _shift_depth(
                depth_patch, mask_patch,
                float(slot.get("depth_placement", 0.7)),
                float(slot.get("gradient_scale",  1.0)),
            )
            alpha  = mask_patch
            canvas = canvas * (1.0 - alpha) + shifted * alpha

        canvas = np.clip(canvas, 0.0, 1.0)

        depth_tensor   = _np_gray_to_tensor(canvas)
        preview_tensor = _apply_false_color(canvas)

        print(
            f"[Eric_Composer_Studio] DepthComposer: composed {len(slots_for_composite)} element(s) "
            f"onto {canvas_w}×{canvas_h} canvas (bg={bg_depth:.2f})"
        )

        return (depth_tensor, preview_tensor)


# ── REST endpoint: return current composition_data including src dims ─────────
try:
    from server import PromptServer as _PS
    from aiohttp import web as _aio_web2

    @_PS.instance.routes.get("/eric_composer_studio/depth_composition/{node_id}")
    async def _eric_depth_composition(request):
        """
        JS polls this to check if slot thumbs are ready.
        Returns the list of available slot thumb paths.
        """
        node_id = request.match_info["node_id"]
        folder  = os.path.join(_tmp_dir(), str(node_id))
        available = []
        for i in range(MAX_SLOTS):
            path = os.path.join(folder, f"slot_{i}.png")
            available.append(os.path.isfile(path))
        return _aio_web2.json_response({"slots_available": available})

except Exception:
    pass
