"""
EricTextLayer — text-as-layer node tailored for `EricImageComposer`.

Outputs an (IMAGE, MASK) pair so it plugs directly into one of the composer's
image_N / mask_N socket pairs:
    • IMAGE = the fill (solid colour, linear gradient, or full-frame copy of an
      optional `fill_image` input)
    • MASK  = anti-aliased text alpha (white where text is, black elsewhere)

The composer then handles transform, blending, opacity and drop-shadow, so this
node intentionally stays focused on producing a clean text mask + fill.
"""
from __future__ import annotations

import os
import re
import sys
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont


# ── font discovery (cached at module load) ────────────────────────────────────

def _system_fonts_dir() -> List[str]:
    if sys.platform == "win32":
        return [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")]
    if sys.platform == "darwin":
        return ["/System/Library/Fonts", "/Library/Fonts",
                os.path.expanduser("~/Library/Fonts")]
    return ["/usr/share/fonts", "/usr/local/share/fonts",
            os.path.expanduser("~/.fonts"),
            os.path.expanduser("~/.local/share/fonts")]


def _discover_fonts() -> dict[str, str]:
    fonts: dict[str, str] = {}
    for d in _system_fonts_dir():
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith((".ttf", ".otf", ".ttc")):
                    name = re.sub(r"[_-]", " ", os.path.splitext(f)[0])
                    fonts.setdefault(name, os.path.join(root, f))
    if not fonts:
        # PIL default fallback so the dropdown is never empty.
        fonts["(default)"] = ""
    return dict(sorted(fonts.items(), key=lambda kv: kv[0].lower()))


_SYSTEM_FONTS = _discover_fonts()
_FONT_CACHE: dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}


def _load_font(name: str, size: int) -> ImageFont.ImageFont:
    key = (name, size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    path = _SYSTEM_FONTS.get(name, "")
    try:
        font = ImageFont.truetype(path, size) if path else ImageFont.truetype(name, size)
    except Exception:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_color(s: str, default=(255, 255, 255)) -> Tuple[int, int, int]:
    if not s:
        return default
    try:
        rgb = ImageColor.getrgb(s)
        return rgb[:3]
    except Exception:
        return default


def _wrap_to_width(text: str, font: ImageFont.ImageFont, max_w: int) -> List[str]:
    """Pixel-width-based word wrap that preserves explicit \\n line breaks."""
    if max_w <= 0:
        return text.splitlines() or [""]
    out: List[str] = []
    for raw in text.splitlines() or [""]:
        if not raw.strip():
            out.append("")
            continue
        words = raw.split()
        line = ""
        for w in words:
            trial = (line + " " + w).strip()
            if font.getlength(trial) <= max_w or not line:
                line = trial
            else:
                out.append(line)
                line = w
        if line:
            out.append(line)
    return out


def _measure_block(lines: List[str], font: ImageFont.ImageFont,
                   line_spacing: float) -> Tuple[int, int]:
    if not lines:
        return 0, 0
    # Use a representative ascent/descent for line height stability.
    ascent, descent = font.getmetrics()
    line_h = int((ascent + descent) * line_spacing)
    w = max((int(font.getlength(ln)) for ln in lines), default=0)
    h = line_h * len(lines)
    return w, h


def _build_linear_gradient(w: int, h: int, c1: Tuple[int, int, int],
                            c2: Tuple[int, int, int], angle_deg: float) -> np.ndarray:
    """Vectorized linear gradient → (h, w, 3) uint8."""
    if w <= 0 or h <= 0:
        return np.zeros((max(1, h), max(1, w), 3), dtype=np.uint8)
    rad = np.deg2rad(angle_deg)
    dx, dy = np.cos(rad), np.sin(rad)
    xs = np.linspace(-0.5, 0.5, w, dtype=np.float32)
    ys = np.linspace(-0.5, 0.5, h, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    proj = X * dx + Y * dy
    # Normalize to [0, 1] using the gradient's projected range so the gradient
    # always spans the full frame regardless of angle.
    span = abs(dx) + abs(dy)
    t = np.clip((proj + 0.5 * span) / max(span, 1e-6), 0.0, 1.0)
    c1 = np.array(c1, dtype=np.float32)
    c2 = np.array(c2, dtype=np.float32)
    grad = c1[None, None, :] * (1.0 - t[..., None]) + c2[None, None, :] * t[..., None]
    return np.clip(grad, 0, 255).astype(np.uint8)


# ── node ──────────────────────────────────────────────────────────────────────

FILL_MODES = ["solid", "linear_gradient", "fill_image"]
ALIGN_H = ["left", "center", "right"]
ALIGN_V = ["top", "middle", "bottom"]


class TextLayer:
    """Render text into an IMAGE + MASK pair sized for use as a composer layer."""

    CATEGORY = "Eric_Composer_Studio"
    FUNCTION = "render"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")

    @classmethod
    def INPUT_TYPES(cls):
        font_names = list(_SYSTEM_FONTS.keys())
        default_font = next(
            (n for n in ("Arial", "Segoe UI", "Helvetica", "DejaVu Sans") if n in _SYSTEM_FONTS),
            font_names[0] if font_names else "(default)",
        )
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Sample text",
                }),
                "font": (font_names, {"default": default_font}),
                "font_size": ("INT", {"default": 96, "min": 8, "max": 1024, "step": 1}),
                "align_h": (ALIGN_H, {"default": "center"}),
                "align_v": (ALIGN_V, {"default": "middle"}),
                "line_spacing": ("FLOAT", {"default": 1.15, "min": 0.5, "max": 3.0, "step": 0.05}),
                "padding": ("INT", {"default": 24, "min": 0, "max": 1024, "step": 1}),

                # canvas: when auto_fit is on, ignore canvas_w/h and size to text bbox + padding.
                "auto_fit": ("BOOLEAN", {"default": True,
                    "tooltip": "Size the output canvas to the text bbox + padding. Off → fixed canvas_w/h."}),
                "canvas_w": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 1}),
                "canvas_h": ("INT", {"default": 256,  "min": 16, "max": 8192, "step": 1}),

                # fill
                "fill_mode": (FILL_MODES, {"default": "solid"}),
                "color": ("STRING", {"default": "#FFFFFF",
                    "placeholder": "name or #RRGGBB"}),
                "gradient_color": ("STRING", {"default": "#3399FF"}),
                "gradient_angle": ("FLOAT", {"default": 90.0, "min": -360.0, "max": 360.0, "step": 1.0,
                    "tooltip": "0 = left→right, 90 = top→bottom"}),

                # stroke / outline
                "stroke_width": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1}),
                "stroke_color": ("STRING", {"default": "#000000"}),

                # glow (operates on the mask, expands soft halo around text)
                "glow_radius": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Adds a soft halo to the mask. Useful for neon-style overlays."}),
                "glow_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                    "tooltip": "Multiplier on the blurred halo intensity."}),
            },
            "optional": {
                # When fill_mode = fill_image, the RGB fill is taken from this image
                # (resized to the canvas). The mask still defines the text shape.
                "fill_image": ("IMAGE",),
            },
        }

    # ── core rendering ────────────────────────────────────────────────────────
    def _render_mask(self, text: str, font_name: str, font_size: int,
                     align_h: str, align_v: str, line_spacing: float, padding: int,
                     auto_fit: bool, canvas_w: int, canvas_h: int,
                     stroke_width: int) -> Tuple[Image.Image, int, int]:
        """Return an L-mode mask image of the text alpha and its (w, h)."""
        font = _load_font(font_name, font_size)

        # Decide canvas size first. For auto_fit, measure once at a generous wrap.
        if auto_fit:
            lines = _wrap_to_width(text, font, max_w=10_000)
            tw, th = _measure_block(lines, font, line_spacing)
            # account for stroke spilling outside glyph bbox
            sw = stroke_width * 2
            cw = max(16, tw + padding * 2 + sw)
            ch = max(16, th + padding * 2 + sw)
        else:
            cw, ch = max(16, canvas_w), max(16, canvas_h)
            lines = _wrap_to_width(text, font, max_w=cw - padding * 2)

        mask = Image.new("L", (cw, ch), 0)
        draw = ImageDraw.Draw(mask)

        ascent, descent = font.getmetrics()
        line_h = int((ascent + descent) * line_spacing)
        block_h = line_h * len(lines)

        if align_v == "top":
            y = padding
        elif align_v == "bottom":
            y = ch - block_h - padding
        else:
            y = (ch - block_h) // 2

        for ln in lines:
            lw = int(font.getlength(ln))
            if align_h == "left":
                x = padding
            elif align_h == "right":
                x = cw - lw - padding
            else:
                x = (cw - lw) // 2
            # Pillow ≥ 8.0 supports stroke_width on draw.text; fill=255 paints into L mask.
            draw.text(
                (x, y), ln, fill=255, font=font,
                stroke_width=stroke_width if stroke_width > 0 else 0,
                stroke_fill=255,
            )
            y += line_h

        return mask, cw, ch

    def _render_stroke_only(self, text: str, font_name: str, font_size: int,
                            align_h: str, align_v: str, line_spacing: float, padding: int,
                            cw: int, ch: int, stroke_width: int) -> Image.Image:
        """L-mode mask of the stroke ring only (stroke fill minus glyph fill)."""
        if stroke_width <= 0:
            return Image.new("L", (cw, ch), 0)
        font = _load_font(font_name, font_size)
        full = Image.new("L", (cw, ch), 0)
        glyph = Image.new("L", (cw, ch), 0)
        d_full = ImageDraw.Draw(full)
        d_glyph = ImageDraw.Draw(glyph)

        lines = _wrap_to_width(text, font, max_w=cw - padding * 2 if cw > padding * 2 else 10_000)
        ascent, descent = font.getmetrics()
        line_h = int((ascent + descent) * line_spacing)
        block_h = line_h * len(lines)
        if align_v == "top":
            y = padding
        elif align_v == "bottom":
            y = ch - block_h - padding
        else:
            y = (ch - block_h) // 2
        for ln in lines:
            lw = int(font.getlength(ln))
            if align_h == "left":
                x = padding
            elif align_h == "right":
                x = cw - lw - padding
            else:
                x = (cw - lw) // 2
            d_full.text((x, y), ln, fill=255, font=font,
                        stroke_width=stroke_width, stroke_fill=255)
            d_glyph.text((x, y), ln, fill=255, font=font)
            y += line_h
        # ring = full - glyph
        ring = np.maximum(np.array(full, dtype=np.int16) - np.array(glyph, dtype=np.int16), 0)
        return Image.fromarray(ring.astype(np.uint8), mode="L")

    # ── ComfyUI entry point ───────────────────────────────────────────────────
    def render(self, text, font, font_size, align_h, align_v, line_spacing, padding,
               auto_fit, canvas_w, canvas_h, fill_mode, color, gradient_color,
               gradient_angle, stroke_width, stroke_color, glow_radius, glow_strength,
               fill_image=None):
        text = text if text and text.strip() else " "
        font_size = int(font_size)
        padding = int(padding)
        stroke_width = int(stroke_width)

        # 1) build the glyph mask (fill+stroke together) and the canvas size.
        mask_img, cw, ch = self._render_mask(
            text, font, font_size, align_h, align_v, line_spacing, padding,
            auto_fit, canvas_w, canvas_h, stroke_width,
        )
        glyph_mask = np.array(mask_img, dtype=np.float32) / 255.0  # (h, w)

        # 2) optional glow expands the mask outward.
        if glow_radius > 0 and glow_strength > 0:
            glow = mask_img.filter(ImageFilter.GaussianBlur(radius=glow_radius))
            glow_arr = np.clip(np.array(glow, dtype=np.float32) / 255.0 * glow_strength, 0, 1)
            glyph_mask = np.maximum(glyph_mask, glow_arr)

        # 3) build the RGB fill the size of the canvas.
        col = _parse_color(color, (255, 255, 255))
        if fill_mode == "solid":
            rgb = np.empty((ch, cw, 3), dtype=np.uint8)
            rgb[..., 0], rgb[..., 1], rgb[..., 2] = col
        elif fill_mode == "linear_gradient":
            c2 = _parse_color(gradient_color, (51, 153, 255))
            rgb = _build_linear_gradient(cw, ch, col, c2, gradient_angle)
        elif fill_mode == "fill_image" and fill_image is not None:
            # Take the first batch and resize to canvas.
            ten = fill_image[0] if fill_image.ndim == 4 else fill_image
            arr = ten.detach().cpu().numpy()
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            pil = Image.fromarray(arr).resize((cw, ch), Image.LANCZOS)
            rgb = np.array(pil, dtype=np.uint8)
        else:
            rgb = np.empty((ch, cw, 3), dtype=np.uint8)
            rgb[..., 0], rgb[..., 1], rgb[..., 2] = col

        # 4) overlay the stroke colour onto the RGB fill (where stroke ring is solid).
        if stroke_width > 0:
            ring = self._render_stroke_only(
                text, font, font_size, align_h, align_v, line_spacing, padding,
                cw, ch, stroke_width,
            )
            ring_arr = np.array(ring, dtype=np.float32) / 255.0  # (h, w)
            sc = np.array(_parse_color(stroke_color, (0, 0, 0)), dtype=np.float32)
            rgb_f = rgb.astype(np.float32)
            a = ring_arr[..., None]
            rgb_f = rgb_f * (1.0 - a) + sc[None, None, :] * a
            rgb = np.clip(rgb_f, 0, 255).astype(np.uint8)

        # 5) build IMAGE + MASK tensors (ComfyUI conventions).
        img_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)  # [1,H,W,3]
        mask_t = torch.from_numpy(glyph_mask.astype(np.float32)).unsqueeze(0)  # [1,H,W]
        return (img_t, mask_t)


NODE_CLASS_MAPPINGS = {"EricTextLayer": TextLayer}
NODE_DISPLAY_NAME_MAPPINGS = {"EricTextLayer": "Text Layer (for Image Composer)"}
