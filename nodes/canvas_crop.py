"""
PoseCanvasCrop node for Eric_Composer_Studio.

Changes the visible canvas region WITHOUT distorting the skeleton.
Think of it like cropping or padding a photo - the person doesn't change
shape, you're just changing what area of space is visible.

Two primary modes:

  CROP:   Cut a sub-region of the source canvas. Skeleton coordinates
          shift by (-crop_x, -crop_y). Anything outside the crop rect
          will have keypoints outside the canvas bounds (hidden by renderer).

  PAD:    Expand the canvas with black margins. Skeleton shifts
          by (+pad_left, +pad_top). Useful for adding letterbox/pillarbox.

  AUTO FIT: Given a target aspect ratio, automatically letterbox or
          pillarbox the source canvas to match - skeleton centered,
          no stretch.

Typical workflow:
  1. Detect pose from a 4256x2832 photo
  2. Use PoseCanvasCrop to reframe to your generation aspect (e.g. 1:1 portrait crop)
  3. Use PoseTransform to fine-tune position / scale of the person
  4. Feed to PoseRenderer at your generation resolution
"""

from __future__ import annotations
import copy
import math


def _shift_flat(flat: list, dx: float, dy: float) -> list:
    """Translate all visible keypoints by (dx, dy)."""
    out = []
    for i in range(0, len(flat), 3):
        x, y, c = flat[i], flat[i + 1], flat[i + 2]
        if c <= 0:
            out.extend([0.0, 0.0, 0.0])
        else:
            out.extend([x + dx, y + dy, c])
    return out


def _scale_flat(flat: list, sx: float, sy: float) -> list:
    """Scale all visible keypoints (for canvas resize after crop/pad)."""
    out = []
    for i in range(0, len(flat), 3):
        x, y, c = flat[i], flat[i + 1], flat[i + 2]
        if c <= 0:
            out.extend([0.0, 0.0, 0.0])
        else:
            out.extend([x * sx, y * sy, c])
    return out


_KP_KEYS = [
    "pose_keypoints_2d",
    "face_keypoints_2d",
    "hand_left_keypoints_2d",
    "hand_right_keypoints_2d",
    "foot_keypoints_2d",
]


class PoseCanvasCrop:
    """
    Reframe the pose canvas by cropping or padding - no skeleton distortion.

    mode = 'crop':
        Defines a rectangular region of the source canvas to keep.
        crop_x, crop_y: top-left corner of the crop in source pixels.
        crop_w, crop_h: size of the crop region.
        All keypoints shift by (-crop_x, -crop_y).
        Output canvas = crop_w x crop_h (optionally rescaled to output_w x output_h).

    mode = 'pad':
        Adds black margins around the source canvas.
        pad_left, pad_top, pad_right, pad_bottom: margins in pixels.
        All keypoints shift by (+pad_left, +pad_top).
        Output canvas = (src_w + pad_left + pad_right) x (src_h + pad_top + pad_bottom).

    mode = 'fit_to_ratio':
        Automatically letterbox or pillarbox the source canvas to match
        the target aspect ratio (target_width / target_height).
        Skeleton is centered. No cropping - only padding.
        Output canvas = target_width x target_height.

    mode = 'crop_to_ratio':
        Automatically crop the source canvas to match the target aspect
        ratio, centering the crop. Skeleton coordinates shift accordingly.
        Output canvas = target_width x target_height (after rescale).

    In all modes, if output_width / output_height differ from the
    intermediate canvas size, a proportional rescale is applied to
    map coordinates to the final output dimensions.
    """

    CATEGORY      = "Eric_Composer_Studio"
    FUNCTION      = "crop"
    RETURN_TYPES  = ("POSE_KEYPOINT",)
    RETURN_NAMES  = ("pose_keypoint",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT",),
                "mode": ([
                    "crop",
                    "pad",
                    "fit_to_ratio",
                    "crop_to_ratio",
                ], {"default": "fit_to_ratio"}),
                "output_width":  ("INT", {
                    "default": 1024, "min": 64, "max": 8192, "step": 8
                }),
                "output_height": ("INT", {
                    "default": 1024, "min": 64, "max": 8192, "step": 8
                }),
            },
            "optional": {
                # For 'crop' mode
                "crop_x": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "crop_y": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "crop_w": ("INT", {"default": 0, "min": 0, "max": 8192,
                                    "tooltip": "0 = use source width - crop_x"}),
                "crop_h": ("INT", {"default": 0, "min": 0, "max": 8192,
                                    "tooltip": "0 = use source height - crop_y"}),
                # For 'pad' mode
                "pad_left":   ("INT", {"default": 0, "min": 0, "max": 4096}),
                "pad_top":    ("INT", {"default": 0, "min": 0, "max": 4096}),
                "pad_right":  ("INT", {"default": 0, "min": 0, "max": 4096}),
                "pad_bottom": ("INT", {"default": 0, "min": 0, "max": 4096}),
            }
        }

    def crop(
        self,
        pose_keypoint,
        mode:         str = "fit_to_ratio",
        output_width:  int = 1024,
        output_height: int = 1024,
        crop_x:  int = 0,
        crop_y:  int = 0,
        crop_w:  int = 0,
        crop_h:  int = 0,
        pad_left:   int = 0,
        pad_top:    int = 0,
        pad_right:  int = 0,
        pad_bottom: int = 0,
    ):
        if not isinstance(pose_keypoint, list):
            pose_keypoint = [pose_keypoint]

        result = []

        for entry in pose_keypoint:
            src_w = entry.get("canvas_width",  output_width)
            src_h = entry.get("canvas_height", output_height)

            # --- Compute intermediate canvas and coordinate offset ---
            if mode == "crop":
                eff_crop_w = crop_w if crop_w > 0 else (src_w - crop_x)
                eff_crop_h = crop_h if crop_h > 0 else (src_h - crop_y)
                inter_w = eff_crop_w
                inter_h = eff_crop_h
                dx = -float(crop_x)
                dy = -float(crop_y)

            elif mode == "pad":
                inter_w = src_w + pad_left + pad_right
                inter_h = src_h + pad_top  + pad_bottom
                dx = float(pad_left)
                dy = float(pad_top)

            elif mode == "fit_to_ratio":
                # Letterbox/pillarbox to match output aspect ratio
                target_aspect = output_width / max(output_height, 1)
                src_aspect    = src_w / max(src_h, 1)

                if src_aspect > target_aspect:
                    # Source is wider - add top/bottom padding
                    new_h = int(src_w / target_aspect)
                    inter_w = src_w
                    inter_h = new_h
                    dx = 0.0
                    dy = float((new_h - src_h) / 2)
                else:
                    # Source is taller - add left/right padding
                    new_w = int(src_h * target_aspect)
                    inter_w = new_w
                    inter_h = src_h
                    dx = float((new_w - src_w) / 2)
                    dy = 0.0

            elif mode == "crop_to_ratio":
                # Centre-crop to match output aspect ratio
                target_aspect = output_width / max(output_height, 1)
                src_aspect    = src_w / max(src_h, 1)

                if src_aspect > target_aspect:
                    # Source is wider - crop left/right
                    new_w  = int(src_h * target_aspect)
                    inter_w = new_w
                    inter_h = src_h
                    dx = -float((src_w - new_w) / 2)
                    dy = 0.0
                else:
                    # Source is taller - crop top/bottom
                    new_h  = int(src_w / target_aspect)
                    inter_w = src_w
                    inter_h = new_h
                    dx = 0.0
                    dy = -float((src_h - new_h) / 2)

            else:
                inter_w = src_w
                inter_h = src_h
                dx = 0.0
                dy = 0.0

            # --- Final rescale from intermediate to output dimensions ---
            # This is always proportional since we ensured same aspect above
            sx = output_width  / max(inter_w, 1)
            sy = output_height / max(inter_h, 1)

            new_entry = copy.deepcopy(entry)
            new_entry["canvas_width"]  = output_width
            new_entry["canvas_height"] = output_height

            for person in new_entry.get("people", []):
                for key in _KP_KEYS:
                    flat = person.get(key, [])
                    # 1. Shift (crop offset or pad offset)
                    flat = _shift_flat(flat, dx, dy)
                    # 2. Scale to output dimensions
                    flat = _scale_flat(flat, sx, sy)
                    person[key] = flat

            result.append(new_entry)

        return (result,)
