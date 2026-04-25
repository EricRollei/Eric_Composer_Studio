"""
SavePoseKeypoint node for Eric_Composer_Studio.

Saves a POSE_KEYPOINT list to a timestamped JSON file.
Files are named:  {output_folder}/{YYYYMMDD_HHMMSS}_{name_slug}.json

The JSON includes an 'eric_pose_studio_meta' header (name, person count,
canvas size, timestamp) followed by the full pose_keypoints list.
LoadPoseKeypoint reads this format and also plain lists saved by other
nodes (controlnet_aux SavePoseKpsAsJsonFile, etc).
"""

from __future__ import annotations
import json
import os
import re
from datetime import datetime

try:
    import folder_paths as _fp
    def _output_dir() -> str:
        return _fp.get_output_directory()
except ImportError:
    def _output_dir() -> str:
        return os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "output")


def _resolve_folder(path: str) -> str:
    """Absolute path → as-is.  Relative path → joined with ComfyUI output dir."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(_output_dir(), path))


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text.strip("_") or "pose"


class SavePoseKeypoint:
    """
    Save a POSE_KEYPOINT to a timestamped JSON file.

    output_folder accepts relative paths (resolved from ComfyUI's output
    directory) or absolute paths, so you can organise poses into any
    folder hierarchy you like.

    name is free-form text used as both the filename slug and stored in
    the metadata header — e.g. 'woman standing arms folded'.
    """

    CATEGORY     = "Eric_Composer_Studio"
    FUNCTION     = "save"
    RETURN_TYPES = ()
    OUTPUT_NODE  = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT",),
                "name": ("STRING", {
                    "default": "my_pose",
                    "multiline": False,
                    "tooltip": (
                        "Descriptive name for this pose, e.g. 'woman standing arms folded'. "
                        "Used in the filename and stored in the file metadata."
                    ),
                }),
                "output_folder": ("STRING", {
                    "default": "poses",
                    "multiline": False,
                    "tooltip": (
                        "Folder to save into. Relative paths are resolved from ComfyUI's "
                        "output directory. Use an absolute path to save anywhere on disk."
                    ),
                }),
            }
        }

    def save(
        self,
        pose_keypoint,
        name:          str = "my_pose",
        output_folder: str = "poses",
    ):
        if not isinstance(pose_keypoint, list):
            pose_keypoint = [pose_keypoint]

        folder = _resolve_folder(output_folder)
        os.makedirs(folder, exist_ok=True)

        slug      = _slugify(name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"{timestamp}_{slug}.json"
        filepath  = os.path.join(folder, filename)

        first        = pose_keypoint[0] if pose_keypoint else {}
        person_count = sum(len(e.get("people", [])) for e in pose_keypoint)

        payload = {
            "eric_pose_studio_meta": {
                "name":         name,
                "person_count": person_count,
                "canvas_width":  first.get("canvas_width",  0),
                "canvas_height": first.get("canvas_height", 0),
                "saved":        datetime.now().isoformat(timespec="seconds"),
            },
            "pose_keypoints": pose_keypoint,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"[Eric_Composer_Studio] SavePoseKeypoint → {filepath}")
        return {}
