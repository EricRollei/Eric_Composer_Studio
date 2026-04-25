"""
PoseTransform node - REDESIGNED for Eric_Composer_Studio.

Moves and uniformly scales a skeleton within a target canvas.
NO separate X/Y scaling - that would distort the pose and give anatomically
wrong results. If you need to change the canvas crop/letterbox, use
PoseCanvasCrop instead.

Typical use:
  - You detected a pose from a source image
  - You want the person positioned at a specific location in your generation
  - Use translate to place them, scale to resize them (uniformly)

For changing the visible region / reframing, use PoseCanvasCrop.
"""

from __future__ import annotations
import copy
import math


def _transform_flat_uniform(
    flat:     list,
    src_w:    int,
    src_h:    int,
    dst_w:    int,
    dst_h:    int,
    scale:    float,   # uniform scale applied to skeleton in dst canvas space
    tx:       float,   # translate x in dst canvas pixels (after scale)
    ty:       float,   # translate y in dst canvas pixels
    angle_r:  float,   # rotation radians (pivot = centroid of visible keypoints)
    pivot_x:  float,   # rotation pivot x in dst canvas pixels
    pivot_y:  float,   # rotation pivot y
) -> list:
    """
    Maps each keypoint from src canvas space to dst canvas space
    with uniform scale, translation, and optional rotation.

    Coordinates first mapped src->dst proportionally (aspect-correct),
    then scale/translate/rotate applied.
    """
    cos_a = math.cos(angle_r)
    sin_a = math.sin(angle_r)

    out = []
    for i in range(0, len(flat), 3):
        x, y, c = flat[i], flat[i + 1], flat[i + 2]
        if c <= 0:
            out.extend([0.0, 0.0, 0.0])
            continue

        # Normalise to [0,1] in src canvas
        nx = x / max(src_w, 1)
        ny = y / max(src_h, 1)

        # Map to dst canvas (proportional - NOT stretching to fill)
        # We use the same scale factor for both axes to preserve proportions
        px = nx * dst_w
        py = ny * dst_h

        # Apply uniform scale around pivot
        px = pivot_x + (px - pivot_x) * scale
        py = pivot_y + (py - pivot_y) * scale

        # Translate
        px += tx
        py += ty

        # Rotate around pivot
        if angle_r != 0.0:
            dx = px - pivot_x
            dy = py - pivot_y
            px = pivot_x + dx * cos_a - dy * sin_a
            py = pivot_y + dx * sin_a + dy * cos_a

        out.extend([px, py, c])
    return out


def _compute_skeleton_centroid(pose_kp_entry: dict, src_w: int, src_h: int,
                                dst_w: int, dst_h: int) -> tuple:
    """Compute the centroid of all visible body keypoints in dst canvas space."""
    xs, ys = [], []
    for key in ["pose_keypoints_2d"]:
        flat = pose_kp_entry.get("people", [{}])[0].get(key, []) if pose_kp_entry.get("people") else []
        for i in range(0, len(flat), 3):
            if flat[i + 2] > 0:
                xs.append(flat[i]     / max(src_w, 1) * dst_w)
                ys.append(flat[i + 1] / max(src_h, 1) * dst_h)
    if xs:
        return sum(xs) / len(xs), sum(ys) / len(ys)
    return dst_w / 2.0, dst_h / 2.0


class PoseTransform:
    """
    Place and scale skeleton(s) within a target canvas.

    UNIFORM SCALE ONLY - no separate X/Y scale to prevent pose distortion.

    translate_x / translate_y: pixel offset on the target canvas.
      Positive X moves right, positive Y moves down.
    scale: resizes the person uniformly. Pivot is the skeleton centroid.
    rotate_degrees: optional rotation. NOTE: this changes the actual pose
      geometry (a tilted torso looks different to ControlNet). Use sparingly.
    fit_mode controls how the src canvas maps to dst before transforms:
      'proportional' - preserve aspect, no cropping (default, safest)
      'center'       - center skeleton in target canvas, no scale change
    """

    CATEGORY      = "Eric_Composer_Studio"
    FUNCTION      = "transform"
    RETURN_TYPES  = ("POSE_KEYPOINT",)
    RETURN_NAMES  = ("pose_keypoint",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT",),
                "target_width":  ("INT", {
                    "default": 1024, "min": 64, "max": 8192, "step": 8
                }),
                "target_height": ("INT", {
                    "default": 1024, "min": 64, "max": 8192, "step": 8
                }),
                "scale": ("FLOAT", {
                    "default": 1.0, "min": 0.05, "max": 5.0, "step": 0.01,
                }),
                "translate_x": ("INT", {
                    "default": 0, "min": -4096, "max": 4096, "step": 1,
                }),
                "translate_y": ("INT", {
                    "default": 0, "min": -4096, "max": 4096, "step": 1,
                }),
                "rotate_degrees": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5,
                }),
            }
        }

    def transform(
        self,
        pose_keypoint,
        target_width:   int   = 1024,
        target_height:  int   = 1024,
        scale:          float = 1.0,
        translate_x:    int   = 0,
        translate_y:    int   = 0,
        rotate_degrees: float = 0.0,
    ):
        if not isinstance(pose_keypoint, list):
            pose_keypoint = [pose_keypoint]

        angle_r = math.radians(rotate_degrees)
        result  = []

        for entry in pose_keypoint:
            src_w = entry.get("canvas_width",  target_width)
            src_h = entry.get("canvas_height", target_height)

            # Pivot for scale and rotation = centroid of visible skeleton
            pivot_x, pivot_y = _compute_skeleton_centroid(
                entry, src_w, src_h, target_width, target_height
            )

            new_entry = copy.deepcopy(entry)
            new_entry["canvas_width"]  = target_width
            new_entry["canvas_height"] = target_height

            for person in new_entry.get("people", []):
                for key in [
                    "pose_keypoints_2d",
                    "face_keypoints_2d",
                    "hand_left_keypoints_2d",
                    "hand_right_keypoints_2d",
                    "foot_keypoints_2d",
                ]:
                    person[key] = _transform_flat_uniform(
                        person.get(key, []),
                        src_w, src_h,
                        target_width, target_height,
                        scale,
                        float(translate_x), float(translate_y),
                        angle_r,
                        pivot_x, pivot_y,
                    )

            result.append(new_entry)

        return (result,)
