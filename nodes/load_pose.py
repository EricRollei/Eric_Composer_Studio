"""
LoadPoseKeypoint node for Eric_Composer_Studio.

Loads a POSE_KEYPOINT from a JSON file.  Handles all common formats:
  - Eric_Composer_Studio (legacy key) wrapped:  {"eric_pose_studio_meta": {...}, "pose_keypoints": [...]}
  - controlnet_aux bare list:  [{...pose dict...}, ...]
  - Single pose dict:          {...}  (auto-wrapped in a list)

folder_path accepts relative paths (resolved from ComfyUI's output directory)
or absolute paths, so you can load files saved by any other node.
"""

from __future__ import annotations
import json
import os

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


def _normalize_pose_json(raw) -> list:
    """
    Normalise all known pose JSON variants to a POSE_KEYPOINT list.

    dict with 'eric_pose_studio_meta'  →  return the inner 'pose_keypoints' list
    list                               →  return as-is  (controlnet_aux / huchenlei)
    any other dict                     →  wrap in a list  (single pose entry)
    """
    if isinstance(raw, dict):
        if "pose_keypoints" in raw:
            data = raw["pose_keypoints"]
            return data if isinstance(data, list) else [data]
        return [raw]
    if isinstance(raw, list):
        return raw
    raise ValueError(
        f"[Eric_Composer_Studio] LoadPoseKeypoint: unrecognised JSON format ({type(raw).__name__})"
    )


class LoadPoseKeypoint:
    """
    Load a POSE_KEYPOINT from a JSON file.

    folder_path points to the directory that contains your pose files.
    Relative paths are resolved from ComfyUI's output directory; absolute
    paths are used as-is, so you can point to any folder on disk including
    where controlnet_aux or other nodes saved their files.

    filename is the .json file to load within that folder.
    The node re-executes automatically whenever the file is modified on disk.
    """

    CATEGORY     = "Eric_Composer_Studio"
    FUNCTION     = "load"
    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoint",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "poses",
                    "multiline": False,
                    "tooltip": (
                        "Directory containing pose JSON files. "
                        "Relative paths resolved from ComfyUI output dir; absolute paths used as-is."
                    ),
                }),
                "filename": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Filename (with .json extension) to load from folder_path.",
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, folder_path: str = "poses", filename: str = ""):
        """Re-execute whenever the file's modification time changes."""
        if not filename or not filename.strip():
            return float("nan")
        path = os.path.join(_resolve_folder(folder_path), filename)
        try:
            return os.path.getmtime(path)
        except OSError:
            return float("nan")

    def load(self, folder_path: str = "poses", filename: str = ""):
        if not filename or not filename.strip():
            # No file selected – return a valid empty pose so downstream nodes degrade gracefully.
            return ([{"version": 1.3, "people": [], "canvas_width": 1024, "canvas_height": 1024}],)

        folder   = _resolve_folder(folder_path)
        filepath = os.path.join(folder, filename)

        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"[Eric_Composer_Studio] LoadPoseKeypoint: file not found:\n  {filepath}"
            )

        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)

        pose_kp = _normalize_pose_json(raw)
        print(f"[Eric_Composer_Studio] LoadPoseKeypoint ← {filepath} ({len(pose_kp)} frame(s))")
        return (pose_kp,)


# ── REST endpoint: list pose files for the gallery widget ─────────────────
try:
    from server import PromptServer
    from aiohttp import web as _aio_web

    @PromptServer.instance.routes.get("/eric_composer_studio/list_poses")
    async def _eric_list_poses(request):
        """
        Return subdirectory names and pose file data for the gallery widget.
        Response: {"subdirs": [...], "files": [{filename, name, canvas_width, ...}, ...]}
        Only pose_keypoints_2d (body) is returned per person to keep payload small.
        """
        folder_param = request.rel_url.query.get("folder", "poses")
        folder = _resolve_folder(folder_param)
        subdirs: list = []
        files: list = []
        if os.path.isdir(folder):
            for entry in sorted(os.scandir(folder), key=lambda e: e.name.lower()):
                if entry.is_dir():
                    subdirs.append(entry.name)
                    continue
                fname = entry.name
                if not fname.lower().endswith(".json"):
                    continue
                fpath = entry.path
                try:
                    with open(fpath, "r", encoding="utf-8") as _fh:
                        raw = json.load(_fh)
                    kp_list = _normalize_pose_json(raw)
                    meta    = raw.get("eric_pose_studio_meta", {}) if isinstance(raw, dict) else {}
                    first   = kp_list[0] if kp_list else {}
                    cw = first.get("canvas_width",  meta.get("canvas_width",  1024))
                    ch = first.get("canvas_height", meta.get("canvas_height", 1024))
                    people: list = []
                    for frame in kp_list:
                        for p in (frame.get("people") or []):
                            people.append({
                                "pose_keypoints_2d": p.get("pose_keypoints_2d", []),
                            })
                    files.append({
                        "filename":     fname,
                        "name":         meta.get("name", ""),
                        "canvas_width": cw,
                        "canvas_height": ch,
                        "person_count": len(people),
                        "people":       people,
                    })
                except Exception:
                    continue
        return _aio_web.json_response({"subdirs": subdirs, "files": files})

except Exception:
    pass
