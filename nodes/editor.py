"""
PoseEditor node for Eric_Composer_Studio.

Python backend for the interactive pose editor.
The full canvas editor UI is in web/pose_editor.js (phase 2).
For now this node acts as a passthrough with JSON serialisation,
compatible with huchenlei's POSE_KEYPOINT pipeline.
"""

from __future__ import annotations
import json
import copy
from ..core.pose_types import PoseKeypoint, empty_pose_keypoint


class PoseEditor:
    """
    Interactive pose editor node.

    Accepts POSE_KEYPOINT from a detector (or transform) node.
    Right-click -> "Open Pose Editor" to launch the editor canvas.
    Send edited pose back to continue the workflow.

    Also accepts raw JSON string for compatibility with huchenlei's
    Load Openpose JSON output.
    """

    CATEGORY      = "Eric_Composer_Studio"
    FUNCTION      = "edit"
    RETURN_TYPES  = ("POSE_KEYPOINT",)
    RETURN_NAMES  = ("pose_keypoint",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "canvas_width":  ("INT", {
                    "default": 1024, "min": 64, "max": 8192, "step": 8
                }),
                "canvas_height": ("INT", {
                    "default": 1024, "min": 64, "max": 8192, "step": 8
                }),
            },
            "optional": {
                # From PoseDetector / PoseTransform
                "pose_keypoint": ("POSE_KEYPOINT",),
                # From huchenlei's Load Openpose JSON or westNeighbor estimator
                "pose_json_str": ("STRING", {
                    "default": "", "multiline": True
                }),
                # Edited keypoints returned from the JS editor
                # (populated by the editor UI via widget)
                "edited_pose_json": ("STRING", {
                    "default": "", "multiline": True,
                    "dynamicPrompts": False,
                }),
            }
        }

    def edit(
        self,
        canvas_width:     int   = 1024,
        canvas_height:    int   = 1024,
        pose_keypoint     = None,
        pose_json_str:    str   = "",
        edited_pose_json: str   = "",
    ):
        # Priority: edited_pose_json > pose_keypoint > pose_json_str

        if edited_pose_json and edited_pose_json.strip():
            try:
                data = json.loads(edited_pose_json)
                if not isinstance(data, list):
                    data = [data]
                return (data,)
            except json.JSONDecodeError as e:
                print(f"[Eric_Composer_Studio] PoseEditor: invalid edited_pose_json: {e}")

        if pose_keypoint is not None:
            if isinstance(pose_keypoint, list):
                return (pose_keypoint,)
            return ([pose_keypoint],)

        if pose_json_str and pose_json_str.strip():
            try:
                data = json.loads(pose_json_str)
                # Handle both single dict and list-of-dicts
                if isinstance(data, dict):
                    data = [data]
                # Ensure canvas size is set
                for entry in data:
                    if "canvas_width"  not in entry:
                        entry["canvas_width"]  = canvas_width
                    if "canvas_height" not in entry:
                        entry["canvas_height"] = canvas_height
                return (data,)
            except json.JSONDecodeError as e:
                print(f"[Eric_Composer_Studio] PoseEditor: invalid pose_json_str: {e}")

        # Nothing connected - return empty pose
        return (empty_pose_keypoint(canvas_width, canvas_height),)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-evaluate when edited_pose_json changes
        return float("nan")
