"""
ImageComposer node for Eric_Composer_Studio.

Composites up to 6 image layers onto a configurable canvas with per-layer
transform, blend mode, opacity, drop shadow, colour adjustments, horizontal /
vertical flipping, and an optional procedural background.

Each slot accepts an IMAGE and an optional MASK. If no mask is connected, the
image's own alpha channel is used; if there is no alpha either, the whole
rectangle is treated as opaque. One slot can be marked as the background (BG)
in which case its mask is ignored and it is cover-fit to the canvas.

JSON composition_data structure (all floats 0..1 unless noted):
{
  "canvas_w": 1024, "canvas_h": 1024,
  "display_w": 640, "display_h": 640,
  "megapixels": 1.0,
  "background": {
      "type": "solid",            # solid|gradient_lin|gradient_rad|paper|canvas|checker|dots|perlin
      "color1": [40, 40, 48],
      "color2": [20, 20, 28],
      "angle":  0.0,              # gradient angle in degrees
      "scale":  1.0,              # pattern scale
  },
  "slots": [
    {
      "id": "s0", "label": "Layer 1",
      "is_bg": false, "visible": true,
      "z_order": 0,
      "x": 0.5, "y": 0.5,
      "scale": 0.5, "scale_x": 1.0, "scale_y": 1.0,
      "rotation": 0.0,
      "flip_h": false, "flip_v": false,
      "opacity": 0.5,
      "blend_mode": "normal",
      "brightness": 0.0,          # -1..+1
      "contrast":   0.0,          # -1..+1
      "saturation": 0.0,          # -1..+1
      "shadow": {
          "enabled": false,
          "angle":   135.0,       # degrees (Photoshop-style: light from upper-left)
          "distance": 8.0,        # pixels in display space
          "blur":    12.0,        # gaussian sigma in display space
          "opacity": 0.6,
          "color":   [0, 0, 0],
      },
      "src_w": 0, "src_h": 0,
    }, ...
  ]
}
"""

from __future__ import annotations
import json
import math
import os
import hashlib
import numpy as np

MAX_SLOTS = 6

ELEMENT_COLORS = [
    (74, 158, 255), (255, 107, 53), (76, 175, 80),
    (224, 64, 251), (255, 215, 64), (38, 198, 218),
]

BLEND_MODES = [
    "normal", "multiply", "screen", "overlay", "soft_light", "hard_light",
    "add", "subtract", "difference", "darken", "lighten",
    "color_dodge", "color_burn", "hue", "saturation", "color", "luminosity",
]


# ── tensor helpers ────────────────────────────────────────────────────────────

def _tensor_to_rgba(t) -> np.ndarray:
    """IMAGE tensor [B,H,W,C] → (H,W,4) float32 RGBA in [0,1]."""
    arr = t[0].cpu().numpy().astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 3:
        alpha = np.ones(arr.shape[:2], dtype=np.float32)
        arr = np.concatenate([arr, alpha[..., None]], axis=-1)
    return arr


def _tensor_to_mask(t) -> np.ndarray:
    """MASK tensor [B,H,W] → (H,W) float32 [0,1]."""
    return t[0].cpu().numpy().astype(np.float32)


def _rgb_to_tensor(arr: np.ndarray):
    """(H,W,3) float32 → IMAGE tensor [1,H,W,3]."""
    import torch
    return torch.from_numpy(arr[None].astype(np.float32)).contiguous()


def _mask_to_tensor(arr: np.ndarray):
    """(H,W) float32 → MASK tensor [1,H,W]."""
    import torch
    return torch.from_numpy(arr[None].astype(np.float32)).contiguous()


# ── colour adjustments ───────────────────────────────────────────────────────

def _apply_bcs(rgb: np.ndarray, brightness: float, contrast: float, saturation: float) -> np.ndarray:
    """Apply brightness/contrast/saturation in [-1..+1] range. rgb is (H,W,3) float32."""
    out = rgb
    if brightness != 0.0:
        out = out + brightness
    if contrast != 0.0:
        f = 1.0 + contrast
        out = (out - 0.5) * f + 0.5
    if saturation != 0.0:
        # ITU-R BT.601 luma
        luma = (0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2])[..., None]
        s = 1.0 + saturation
        out = luma + (out - luma) * s
    return np.clip(out, 0.0, 1.0)


# ── blend modes ───────────────────────────────────────────────────────────────
# All operate on (H,W,3) float32 RGB in [0,1].
# base = underlying pixels, blend = incoming pixels. Return same shape.

def _bm_normal(b, s):      return s
def _bm_multiply(b, s):    return b * s
def _bm_screen(b, s):      return 1.0 - (1.0 - b) * (1.0 - s)
def _bm_darken(b, s):      return np.minimum(b, s)
def _bm_lighten(b, s):     return np.maximum(b, s)
def _bm_add(b, s):         return np.clip(b + s, 0.0, 1.0)
def _bm_subtract(b, s):    return np.clip(b - s, 0.0, 1.0)
def _bm_difference(b, s):  return np.abs(b - s)

def _bm_overlay(b, s):
    return np.where(b < 0.5, 2.0 * b * s, 1.0 - 2.0 * (1.0 - b) * (1.0 - s))

def _bm_hard_light(b, s):
    return np.where(s < 0.5, 2.0 * b * s, 1.0 - 2.0 * (1.0 - b) * (1.0 - s))

def _bm_soft_light(b, s):
    # W3C / SVG spec version
    d = np.where(b <= 0.25,
                 ((16.0 * b - 12.0) * b + 4.0) * b,
                 np.sqrt(np.clip(b, 0, 1)))
    return np.where(s <= 0.5,
                    b - (1.0 - 2.0 * s) * b * (1.0 - b),
                    b + (2.0 * s - 1.0) * (d - b))

def _bm_color_dodge(b, s):
    out = np.where(s >= 1.0, 1.0, np.clip(b / np.maximum(1.0 - s, 1e-6), 0, 1))
    return np.where(b <= 0.0, 0.0, out)

def _bm_color_burn(b, s):
    out = np.where(s <= 0.0, 0.0, 1.0 - np.clip((1.0 - b) / np.maximum(s, 1e-6), 0, 1))
    return np.where(b >= 1.0, 1.0, out)

# HSL-based modes use luma/sat from the "other" channel set per SVG spec.

def _lum(c):
    # (H,W,3) → (H,W)
    return 0.3 * c[..., 0] + 0.59 * c[..., 1] + 0.11 * c[..., 2]

def _clip_color(c):
    l = _lum(c)[..., None]
    n = np.min(c, axis=-1, keepdims=True)
    x = np.max(c, axis=-1, keepdims=True)
    c = np.where(n < 0, l + (c - l) * l / np.maximum(l - n, 1e-6), c)
    c = np.where(x > 1, l + (c - l) * (1.0 - l) / np.maximum(x - l, 1e-6), c)
    return c

def _set_lum(c, l):
    d = l - _lum(c)
    return _clip_color(c + d[..., None])

def _sat(c):
    return np.max(c, axis=-1) - np.min(c, axis=-1)

def _set_sat(c, s):
    # Distribute sat s across the three channels of c
    out = np.zeros_like(c)
    mn = np.min(c, axis=-1, keepdims=True)
    mx = np.max(c, axis=-1, keepdims=True)
    md_mask = (c > mn) & (c < mx)
    range_c = mx - mn
    # Mid channel: ((mid - min) * s) / (max - min)
    mid_val = np.where(range_c > 0, (c - mn) * s[..., None] / np.maximum(range_c, 1e-6), 0.0)
    out = np.where(md_mask, mid_val, out)
    out = np.where(c == mx, s[..., None], out)
    # min channel stays 0
    # If all three equal (range_c == 0): flat gray, leave zeros.
    return out

def _bm_hue(b, s):         return _set_lum(_set_sat(s, _sat(b)), _lum(b))
def _bm_saturation(b, s):  return _set_lum(_set_sat(b, _sat(s)), _lum(b))
def _bm_color(b, s):       return _set_lum(s, _lum(b))
def _bm_luminosity(b, s):  return _set_lum(b, _lum(s))

_BLEND_FN = {
    "normal": _bm_normal, "multiply": _bm_multiply, "screen": _bm_screen,
    "overlay": _bm_overlay, "soft_light": _bm_soft_light, "hard_light": _bm_hard_light,
    "add": _bm_add, "subtract": _bm_subtract, "difference": _bm_difference,
    "darken": _bm_darken, "lighten": _bm_lighten,
    "color_dodge": _bm_color_dodge, "color_burn": _bm_color_burn,
    "hue": _bm_hue, "saturation": _bm_saturation,
    "color": _bm_color, "luminosity": _bm_luminosity,
}


def _composite_layer(base_rgb: np.ndarray, layer_rgb: np.ndarray,
                     layer_alpha: np.ndarray, blend_mode: str, opacity: float) -> np.ndarray:
    """Composite a single layer onto base_rgb. Base is RGB (H,W,3). Returns new RGB."""
    fn = _BLEND_FN.get(blend_mode, _bm_normal)
    blended = fn(base_rgb, layer_rgb)
    blended = np.clip(blended, 0.0, 1.0)
    a = (layer_alpha * opacity)[..., None]
    return base_rgb * (1.0 - a) + blended * a


# ── procedural backgrounds ────────────────────────────────────────────────────

def _bg_solid(w, h, c1, _c2, _angle, _scale):
    return np.tile(np.array(c1, dtype=np.float32), (h, w, 1))

def _bg_gradient_lin(w, h, c1, c2, angle, _scale):
    rad = math.radians(angle)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Normalized projection along gradient direction
    nx = (xx / max(w - 1, 1)) - 0.5
    ny = (yy / max(h - 1, 1)) - 0.5
    t  = nx * math.cos(rad) + ny * math.sin(rad)
    t  = (t - t.min()) / max(t.max() - t.min(), 1e-6)
    c1a = np.array(c1, dtype=np.float32)
    c2a = np.array(c2, dtype=np.float32)
    return c1a * (1.0 - t[..., None]) + c2a * t[..., None]

def _bg_gradient_rad(w, h, c1, c2, _angle, scale):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r = r / (max(w, h) * 0.5 * max(scale, 0.1))
    r = np.clip(r, 0.0, 1.0)
    c1a = np.array(c1, dtype=np.float32)
    c2a = np.array(c2, dtype=np.float32)
    return c1a * (1.0 - r[..., None]) + c2a * r[..., None]

def _value_noise(h, w, cell, seed=1234):
    """Simple bilinearly-interpolated value noise."""
    rng = np.random.default_rng(seed)
    gh = max(int(h / cell) + 2, 2)
    gw = max(int(w / cell) + 2, 2)
    grid = rng.random((gh, gw), dtype=np.float32)
    # Upscale via bilinear
    yy = np.linspace(0, gh - 1, h, dtype=np.float32)
    xx = np.linspace(0, gw - 1, w, dtype=np.float32)
    y0 = yy.astype(np.int32)
    y1 = np.clip(y0 + 1, 0, gh - 1)
    x0 = xx.astype(np.int32)
    x1 = np.clip(x0 + 1, 0, gw - 1)
    ty = (yy - y0)[:, None]
    tx = (xx - x0)[None, :]
    a = grid[np.ix_(y0, x0)]
    b = grid[np.ix_(y0, x1)]
    c = grid[np.ix_(y1, x0)]
    d = grid[np.ix_(y1, x1)]
    top = a * (1 - tx) + b * tx
    bot = c * (1 - tx) + d * tx
    return top * (1 - ty) + bot * ty

def _perlin_like(h, w, scale, seed=4321):
    """Sum of octaves of value noise → fBm (Perlin-ish without gradients)."""
    cell = max(8, int(24 * scale))
    n = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    total = 0.0
    for i in range(4):
        n += amp * _value_noise(h, w, max(cell >> i, 2), seed + i)
        total += amp
        amp *= 0.5
    n = n / total
    return n

def _bg_paper(w, h, c1, c2, _angle, scale):
    c1a = np.array(c1, dtype=np.float32)
    c2a = np.array(c2, dtype=np.float32)
    base = c1a[None, None, :].repeat(h, 0).repeat(w, 1)
    # Fine high-freq grain + subtle low-freq blotches
    grain = _value_noise(h, w, max(2, int(2 * scale)), seed=11)
    blotch = _perlin_like(h, w, scale * 2.5, seed=17)
    t = np.clip(0.15 * (grain - 0.5) + 0.35 * (blotch - 0.5) + 0.5, 0.0, 1.0)
    return base * (1.0 - t[..., None]) + c2a * t[..., None] * 0.5 + base * 0.5 * t[..., None]

def _bg_canvas(w, h, c1, c2, _angle, scale):
    c1a = np.array(c1, dtype=np.float32)
    c2a = np.array(c2, dtype=np.float32)
    period = max(2, int(4 * scale))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    weave_x = 0.5 + 0.5 * np.sin(xx * (2 * math.pi / period))
    weave_y = 0.5 + 0.5 * np.sin(yy * (2 * math.pi / period))
    weave = 0.5 * (weave_x + weave_y)
    grain = _value_noise(h, w, max(2, int(2 * scale)), seed=23)
    t = np.clip(0.4 * weave + 0.2 * (grain - 0.5) + 0.3, 0.0, 1.0)
    return c1a * (1.0 - t[..., None]) + c2a * t[..., None]

def _bg_checker(w, h, c1, c2, _angle, scale):
    c1a = np.array(c1, dtype=np.float32)
    c2a = np.array(c2, dtype=np.float32)
    sz = max(2, int(32 * scale))
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((xx // sz + yy // sz) % 2).astype(np.float32)
    return c1a * (1.0 - mask[..., None]) + c2a * mask[..., None]

def _bg_dots(w, h, c1, c2, _angle, scale):
    c1a = np.array(c1, dtype=np.float32)
    c2a = np.array(c2, dtype=np.float32)
    period = max(6, int(24 * scale))
    radius = max(2.0, period * 0.25)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    fx = (xx % period) - period * 0.5
    fy = (yy % period) - period * 0.5
    d  = np.sqrt(fx * fx + fy * fy)
    t  = np.clip(1.0 - (d - (radius - 1.0)), 0.0, 1.0)
    base = c1a[None, None, :].repeat(h, 0).repeat(w, 1)
    return base * (1.0 - t[..., None]) + c2a * t[..., None]

def _bg_perlin(w, h, c1, c2, _angle, scale):
    n = _perlin_like(h, w, max(scale, 0.2))
    c1a = np.array(c1, dtype=np.float32)
    c2a = np.array(c2, dtype=np.float32)
    return c1a * (1.0 - n[..., None]) + c2a * n[..., None]

_BG_FN = {
    "solid":        _bg_solid,
    "gradient_lin": _bg_gradient_lin,
    "gradient_rad": _bg_gradient_rad,
    "paper":        _bg_paper,
    "canvas":       _bg_canvas,
    "checker":      _bg_checker,
    "dots":         _bg_dots,
    "perlin":       _bg_perlin,
}


def _render_background(canvas_w: int, canvas_h: int, bg: dict) -> np.ndarray:
    fn = _BG_FN.get(bg.get("type", "solid"), _bg_solid)
    c1 = [v / 255.0 for v in bg.get("color1", [40, 40, 48])]
    c2 = [v / 255.0 for v in bg.get("color2", [20, 20, 28])]
    angle = float(bg.get("angle", 0.0))
    scale = float(bg.get("scale", 1.0))
    return np.clip(fn(canvas_w, canvas_h, c1, c2, angle, scale), 0.0, 1.0)


# ── affine warp ───────────────────────────────────────────────────────────────

def _affine_transform(
    rgba: np.ndarray, mask: np.ndarray,
    canvas_w: int, canvas_h: int,
    cx: float, cy: float,
    scale: float, scale_x: float, scale_y: float,
    rotation_deg: float, flip_h: bool, flip_v: bool,
):
    """Warp (H,W,4) rgba + (H,W) mask to canvas. Returns (canvas_h, canvas_w, 4), (canvas_h, canvas_w)."""
    import cv2
    src_h, src_w = rgba.shape[:2]
    if src_w == 0 or src_h == 0:
        z4 = np.zeros((canvas_h, canvas_w, 4), dtype=np.float32)
        z1 = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        return z4, z1

    src_cx = src_w / 2.0
    src_cy = src_h / 2.0
    dst_cx = cx * canvas_w
    dst_cy = cy * canvas_h

    sx = scale * scale_x * (-1.0 if flip_h else 1.0)
    sy = scale * scale_y * (-1.0 if flip_v else 1.0)

    rad = math.radians(rotation_deg)
    cos_r, sin_r = math.cos(rad), math.sin(rad)

    def tp(px, py):
        lx = (px - src_cx) * sx
        ly = (py - src_cy) * sy
        rx = lx * cos_r - ly * sin_r
        ry = lx * sin_r + ly * cos_r
        return rx + dst_cx, ry + dst_cy

    src_pts = np.float32([[0, 0], [src_w, 0], [0, src_h]])
    dst_pts = np.float32([tp(0, 0), tp(src_w, 0), tp(0, src_h)])
    M = cv2.getAffineTransform(src_pts, dst_pts)

    rgba_c = cv2.warpAffine(rgba, M, (canvas_w, canvas_h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    mask_c = cv2.warpAffine(mask, M, (canvas_w, canvas_h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return rgba_c.astype(np.float32), np.clip(mask_c, 0.0, 1.0).astype(np.float32)


def _cover_fit_transform(src_w: int, src_h: int, dst_w: int, dst_h: int):
    """Return (scale, scale_x, scale_y) such that the src covers dst."""
    # Using the convention: base_px = scale * max(src_w, src_h) / canvas_w... etc.
    # Simpler: compute a direct scale factor that maps src → dst with cover-fit,
    # then derive the "scale" value equivalent to the UI convention.
    # In UI: final_px_per_src_x = (scale * canvas_w / max_dim) * scale_x
    # For cover fit we want final_px_per_src_x = dst_w / src_w (or bigger axis).
    # We use scale_x=scale_y=1.0 and pick scale such that result covers dst.
    max_dim = max(src_w, src_h) if max(src_w, src_h) > 0 else 1
    cover_px_per_src = max(dst_w / src_w, dst_h / src_h)
    scale = cover_px_per_src * max_dim / dst_w
    return scale


# ── drop shadow ───────────────────────────────────────────────────────────────

def _gaussian_blur_alpha(alpha: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.1:
        return alpha
    import cv2
    k = int(max(3, round(sigma * 6)))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(alpha, (k, k), sigma)


def _apply_shadow(base_rgb: np.ndarray, alpha: np.ndarray, shadow: dict,
                  display_to_canvas: float) -> np.ndarray:
    """Composite a drop shadow under the current layer. alpha is the layer's final alpha."""
    if not shadow or not shadow.get("enabled", False):
        return base_rgb
    import cv2
    angle_rad = math.radians(float(shadow.get("angle", 135.0)))
    dist      = float(shadow.get("distance", 8.0)) * display_to_canvas
    blur      = float(shadow.get("blur", 12.0)) * display_to_canvas
    op        = float(shadow.get("opacity", 0.6))
    col       = [c / 255.0 for c in shadow.get("color", [0, 0, 0])]

    # Shadow direction: Photoshop "angle" is light-from direction, so shadow falls opposite.
    dx =  math.cos(angle_rad) * dist
    dy = -math.sin(angle_rad) * dist   # screen Y down
    h, w = alpha.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(alpha, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    blurred = _gaussian_blur_alpha(shifted, blur)
    a = (blurred * op)[..., None]
    col_arr = np.array(col, dtype=np.float32)[None, None, :]
    return base_rgb * (1.0 - a) + col_arr * a


# ── temp PNG serving ──────────────────────────────────────────────────────────

try:
    import folder_paths as _fp
    def _tmp_dir() -> str:
        return os.path.join(_fp.get_output_directory(), "eric_imgcomp_tmp")
except ImportError:
    def _tmp_dir() -> str:
        return os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "output", "eric_imgcomp_tmp")


def _save_slot_thumb(node_id: str, slot_idx: int, rgba: np.ndarray, suffix: str = ""):
    """Save an RGBA layer as PNG for JS preview. Returns path.

    suffix="" → mask-applied thumb (slot_N.png)
    suffix="_raw" → mask-ignored thumb (slot_N_raw.png) used for BG-mode preview.
    """
    import cv2
    folder = os.path.join(_tmp_dir(), str(node_id))
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"slot_{slot_idx}{suffix}.png")
    bgra = rgba[..., [2, 1, 0, 3]]
    img_u8 = (np.clip(bgra, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(path, img_u8)
    return path


# ── REST endpoints ────────────────────────────────────────────────────────────

try:
    from server import PromptServer
    from aiohttp import web as _aio_web

    @PromptServer.instance.routes.get("/eric_composer_studio/imgcomp_slot/{node_id}/{slot}")
    async def _eric_imgcomp_slot(request):
        node_id  = request.match_info["node_id"]
        slot_idx = request.match_info["slot"]
        path = os.path.join(_tmp_dir(), str(node_id), f"slot_{slot_idx}.png")
        if not os.path.isfile(path):
            raise _aio_web.HTTPNotFound()
        return _aio_web.FileResponse(path)

    @PromptServer.instance.routes.get("/eric_composer_studio/imgcomp_slot_raw/{node_id}/{slot}")
    async def _eric_imgcomp_slot_raw(request):
        node_id  = request.match_info["node_id"]
        slot_idx = request.match_info["slot"]
        path = os.path.join(_tmp_dir(), str(node_id), f"slot_{slot_idx}_raw.png")
        if not os.path.isfile(path):
            raise _aio_web.HTTPNotFound()
        return _aio_web.FileResponse(path)

    @PromptServer.instance.routes.get("/eric_composer_studio/imgcomp_composition/{node_id}")
    async def _eric_imgcomp_composition(request):
        node_id = request.match_info["node_id"]
        folder  = os.path.join(_tmp_dir(), str(node_id))
        available = []
        for i in range(MAX_SLOTS):
            path = os.path.join(folder, f"slot_{i}.png")
            available.append(os.path.isfile(path))
        return _aio_web.json_response({"slots_available": available})

except Exception as _e:
    print(f"[Eric_Composer_Studio] ImageComposer: could not register REST endpoint: {_e}")


# ── default composition data ──────────────────────────────────────────────────

def _default_composition_data(canvas_w: int = 1024, canvas_h: int = 1024) -> dict:
    slots = []
    for i in range(MAX_SLOTS):
        slots.append({
            "id": f"s{i}", "label": f"Layer {i+1}",
            "is_bg": False, "visible": True,
            "z_order": i,
            "x": 0.5, "y": 0.5,
            "scale": 0.5, "scale_x": 1.0, "scale_y": 1.0,
            "rotation": 0.0,
            "flip_h": False, "flip_v": False,
            "opacity": 0.5, "blend_mode": "normal",
            "brightness": 0.0, "contrast": 0.0, "saturation": 0.0,
            "shadow": {
                "enabled": False, "angle": 135.0, "distance": 8.0,
                "blur": 12.0, "opacity": 0.6, "color": [0, 0, 0],
            },
            "src_w": 0, "src_h": 0,
        })
    return {
        "canvas_w": canvas_w, "canvas_h": canvas_h,
        "display_w": 640, "display_h": 640,
        "megapixels": 1.0,
        "background": {
            "type":   "solid",
            "color1": [40, 40, 48],
            "color2": [20, 20, 28],
            "angle":  0.0,
            "scale":  1.0,
        },
        "slots": slots,
    }


def _merge_composition_data(existing_str: str, images, masks, node_id: str) -> dict:
    try:
        cd = json.loads(existing_str) if existing_str.strip() else {}
    except Exception:
        cd = {}

    if "slots" not in cd or len(cd["slots"]) != MAX_SLOTS:
        cd = _default_composition_data(
            cd.get("canvas_w", 1024), cd.get("canvas_h", 1024)
        )

    # ensure background object exists with all keys
    default_bg = _default_composition_data()["background"]
    bg = cd.get("background", {}) or {}
    for k, v in default_bg.items():
        bg.setdefault(k, v)
    cd["background"] = bg

    # ensure each slot has all default keys (forward-compatible for saved nodes)
    for i, slot in enumerate(cd["slots"]):
        defaults = _default_composition_data()["slots"][i]
        for k, v in defaults.items():
            if k == "shadow":
                sh = slot.get("shadow", {}) or {}
                for sk, sv in v.items():
                    sh.setdefault(sk, sv)
                slot["shadow"] = sh
            else:
                slot.setdefault(k, v)

    # update src dims + write thumbs
    for i in range(MAX_SLOTS):
        img = images[i]
        msk = masks[i]
        slot = cd["slots"][i]
        if img is not None:
            rgba = _tensor_to_rgba(img)
            # Save the raw (mask-ignored) thumb first so BG-mode preview can show
            # the full input frame regardless of any attached mask.
            raw_rgba = rgba.copy()
            raw_rgba[..., 3] = 1.0
            _save_slot_thumb(node_id, i, raw_rgba, suffix="_raw")
            if msk is not None:
                m_np = _tensor_to_mask(msk)
                if m_np.shape[:2] == rgba.shape[:2]:
                    rgba = rgba.copy()
                    rgba[..., 3] = m_np
            slot["src_w"] = rgba.shape[1]
            slot["src_h"] = rgba.shape[0]
            _save_slot_thumb(node_id, i, rgba)
        else:
            slot["src_w"] = 0
            slot["src_h"] = 0

    return cd


# ── main node class ───────────────────────────────────────────────────────────

class ImageComposer:
    """Composite up to 6 images with blend modes, drop shadows and procedural BG."""

    CATEGORY     = "Eric_Composer_Studio"
    FUNCTION     = "compose"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    OUTPUT_NODE  = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "composition_data": ("STRING", {
                    "default": "", "multiline": True,
                    "dynamicPrompts": False,
                    "tooltip": "Internal composition state — managed by the canvas UI. Do not edit manually.",
                }),
            },
            "optional": {
                "image_1": ("IMAGE",), "mask_1": ("MASK",),
                "image_2": ("IMAGE",), "mask_2": ("MASK",),
                "image_3": ("IMAGE",), "mask_3": ("MASK",),
                "image_4": ("IMAGE",), "mask_4": ("MASK",),
                "image_5": ("IMAGE",), "mask_5": ("MASK",),
                "image_6": ("IMAGE",), "mask_6": ("MASK",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(cls, composition_data="", unique_id="0", **kwargs):
        parts = [composition_data or ""]
        for key in sorted(kwargs.keys()):
            v = kwargs[key]
            if v is not None and hasattr(v, "shape"):
                parts.append(f"{key}:{tuple(v.shape)}")
        return hashlib.md5("|".join(parts).encode()).hexdigest()

    def compose(self, composition_data="", unique_id="0",
                image_1=None, mask_1=None, image_2=None, mask_2=None,
                image_3=None, mask_3=None, image_4=None, mask_4=None,
                image_5=None, mask_5=None, image_6=None, mask_6=None):

        images = [image_1, image_2, image_3, image_4, image_5, image_6]
        masks  = [mask_1,  mask_2,  mask_3,  mask_4,  mask_5,  mask_6]

        cd = _merge_composition_data(composition_data, images, masks, unique_id)

        canvas_w = int(cd.get("canvas_w", 1024))
        canvas_h = int(cd.get("canvas_h", 1024))
        display_w = int(cd.get("display_w", 640)) or 640
        display_to_canvas = canvas_w / display_w

        # Render BG (procedural or solid)
        canvas_rgb = _render_background(canvas_w, canvas_h, cd["background"]).astype(np.float32)
        canvas_alpha = np.zeros((canvas_h, canvas_w), dtype=np.float32)  # accumulates non-BG mask union

        # Find BG slot (image-as-background)
        bg_slot_idx = None
        for i, slot in enumerate(cd["slots"]):
            if slot.get("is_bg", False) and images[i] is not None:
                bg_slot_idx = i
                break

        # Composite all slots. BG slot is forced to be drawn first (behind
        # everything) and its mask is ignored (full image rectangle), but it
        # otherwise honours the user's transform/opacity/blend settings.
        # Remaining slots draw in z_order ascending (back → front).
        non_bg = [(i, s) for i, s in enumerate(cd["slots"]) if i != bg_slot_idx]
        non_bg.sort(key=lambda t: int(t[1].get("z_order", t[0])))
        if bg_slot_idx is not None:
            ordered = [(bg_slot_idx, cd["slots"][bg_slot_idx])] + non_bg
        else:
            ordered = non_bg

        for i, slot in ordered:
            if images[i] is None or not slot.get("visible", True):
                continue
            rgba = _tensor_to_rgba(images[i])
            is_bg_layer = (i == bg_slot_idx)
            if is_bg_layer:
                # Ignore mask, full image rectangle.
                rgba = rgba.copy()
                rgba[..., 3] = 1.0
            elif masks[i] is not None:
                m_np = _tensor_to_mask(masks[i])
                if m_np.shape[:2] == rgba.shape[:2]:
                    rgba = rgba.copy()
                    rgba[..., 3] = m_np
            src_mask = rgba[..., 3].copy()

            # Convert the UI "scale" (canvas-relative, max_dim * scale → canvas_w)
            # into a per-pixel factor for _affine_transform.
            max_dim = max(rgba.shape[0], rgba.shape[1]) or 1
            canvas_base = canvas_w / max_dim
            eff_scale = float(slot.get("scale", 0.5)) * canvas_base
            eff_sx    = float(slot.get("scale_x", 1.0))
            eff_sy    = float(slot.get("scale_y", 1.0))
            eff_rot   = float(slot.get("rotation", 0.0))
            flip_h    = bool(slot.get("flip_h", False))
            flip_v    = bool(slot.get("flip_v", False))

            rgba_c, _ = _affine_transform(
                rgba, src_mask, canvas_w, canvas_h,
                float(slot.get("x", 0.5)), float(slot.get("y", 0.5)),
                eff_scale, eff_sx, eff_sy, eff_rot, flip_h, flip_v,
            )
            layer_rgb   = rgba_c[..., :3]
            layer_alpha = rgba_c[..., 3]
            layer_rgb = _apply_bcs(layer_rgb,
                                   slot.get("brightness", 0.0),
                                   slot.get("contrast", 0.0),
                                   slot.get("saturation", 0.0))

            opacity    = float(slot.get("opacity", 1.0))
            blend_mode = slot.get("blend_mode", "normal")

            # Drop shadow (painted before the layer itself, using final alpha)
            canvas_rgb = _apply_shadow(
                canvas_rgb, layer_alpha * opacity,
                slot.get("shadow", {}), display_to_canvas,
            )

            canvas_rgb = _composite_layer(
                canvas_rgb, layer_rgb, layer_alpha, blend_mode, opacity,
            )
            if not is_bg_layer:
                canvas_alpha = np.maximum(canvas_alpha, layer_alpha * opacity)

        canvas_rgb = np.clip(canvas_rgb, 0.0, 1.0)
        canvas_alpha = np.clip(canvas_alpha, 0.0, 1.0)

        print(
            f"[Eric_Composer_Studio] ImageComposer: composed {sum(1 for img in images if img is not None)} "
            f"layer(s) onto {canvas_w}×{canvas_h} (bg={cd['background'].get('type')})"
        )

        return (_rgb_to_tensor(canvas_rgb), _mask_to_tensor(canvas_alpha))
