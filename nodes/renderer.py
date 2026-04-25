"""
PoseRenderer node for Eric_Composer_Studio.

keep_input_size=True  → render at POSE_KEYPOINT canvas size (recommended for ControlNet).
color_mode='dwpose'   → Official DWPose colors: ellipse limbs, rainbow hands.
color_mode='enhanced' → L/R warm/cool colors for visual analysis.
xinsr_stick_scaling   → Thicker sticks for xinsir/controlnet-openpose-sdxl-1.0.
"""

from __future__ import annotations
import torch
from ..core.draw       import draw_pose, numpy_to_tensor
from ..core.pose_types import PoseKeypoint


class PoseRenderer:
    CATEGORY      = "Eric_Composer_Studio"
    FUNCTION      = "render"
    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("skeleton_image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoint":    ("POSE_KEYPOINT",),
                "keep_input_size":  ("BOOLEAN", {
                    "default": True,
                    "tooltip": "True: render at POSE_KEYPOINT canvas size (recommended for ControlNet). False: use explicit dimensions below.",
                }),
                "canvas_width":     ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "canvas_height":    ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "color_mode":       (["dwpose", "enhanced"], {"default": "dwpose"}),
                "line_width":       ("INT", {"default": 4,  "min": 1, "max": 16, "step": 1,
                    "tooltip": "Enhanced mode only. DWPose mode always uses stickwidth=4."}),
                "joint_radius":     ("INT", {"default": 4,  "min": 1, "max": 20, "step": 1,
                    "tooltip": "Enhanced mode only. DWPose mode always uses radius=4."}),
                "face_dot_radius":  ("INT", {"default": 2,  "min": 1, "max": 8,  "step": 1,
                    "tooltip": "Face keypoint dot size. Default 2 for DWPose, scales with joint_radius in Enhanced."}),
                "draw_face":        ("BOOLEAN", {"default": True}),
                "draw_hands":       ("BOOLEAN", {"default": True}),
                "draw_feet":        ("BOOLEAN", {"default": True}),
                "body_only":        ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Disable face, hands and feet in one click. Overrides the individual toggles above.",
                }),
                "xinsr_stick_scaling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Scale stick thickness for xinsir/controlnet-openpose-sdxl-1.0. At 1024px: stickwidth=12.",
                }),
            }
        }

    def render(
        self,
        pose_keypoint,
        keep_input_size:     bool = True,
        canvas_width:        int  = 1024,
        canvas_height:       int  = 1024,
        color_mode:          str  = "dwpose",
        line_width:          int  = 4,
        joint_radius:        int  = 4,
        face_dot_radius:     int  = 2,
        draw_face:           bool = True,
        draw_hands:          bool = True,
        draw_feet:           bool = True,
        body_only:           bool = False,
        xinsr_stick_scaling: bool = False,
    ):
        if not isinstance(pose_keypoint, list):
            pose_keypoint = [pose_keypoint]

        if body_only:
            draw_face = draw_hands = draw_feet = False

        previews = []
        for entry in pose_keypoint:
            single = [entry]
            w = entry.get("canvas_width",  canvas_width)  if keep_input_size else canvas_width
            h = entry.get("canvas_height", canvas_height) if keep_input_size else canvas_height
            skeleton_bgr = draw_pose(
                single, canvas_width=w, canvas_height=h,
                line_width=line_width, joint_radius=joint_radius,
                face_dot_radius=face_dot_radius,
                color_mode=color_mode,
                draw_face=draw_face, draw_hands=draw_hands, draw_feet=draw_feet,
                xinsr_stick_scaling=xinsr_stick_scaling,
            )
            previews.append(numpy_to_tensor(skeleton_bgr))

        return (torch.cat(previews, dim=0),)
