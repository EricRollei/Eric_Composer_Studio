"""
Model manager for Eric_Composer_Studio.
Handles path resolution and auto-download of RTMW and DWPose ONNX models.
"""

from __future__ import annotations
import os
import urllib.request
import zipfile
import tempfile
from pathlib import Path
from typing import Optional
import folder_paths

MODEL_URLS = {
    "rtmw-x": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.zip"
    ),
    "rtmw-l": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmw-l_simcc-cocktail14_pt-ucoco_270e-256x192-2a88801a_20231122.zip"
    ),
    "rtmw-m": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmw-m_simcc-cocktail14_pt-ucoco_270e-256x192-e4cd8978_20231122.zip"
    ),
    "dwpose": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "dw-ll_ucoco_384.zip"
    ),
    # YOLOX detector - rtmlib uses the 'm' variant, not 'l'
    "yolox": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
    ),
}

# rtmlib mode strings - used only as fallback when explicit paths not available.
# NOTE: there is NO rtmlib mode that loads actual DWPose - it requires explicit paths.
RTMLIB_MODEL_MODES = {
    "rtmw-x": "performance",
    "rtmw-l": "balanced",
    "rtmw-m": "lightweight",
    "dwpose": None,   # No rtmlib mode for DWPose - must use explicit path
}

# ONNX filenames expected after zip extraction
ONNX_FILENAMES = {
    "rtmw-x": "rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.onnx",
    "rtmw-l": "rtmw-l_simcc-cocktail14_pt-ucoco_270e-256x192-2a88801a_20231122.onnx",
    "rtmw-m": "rtmw-m_simcc-cocktail14_pt-ucoco_270e-256x192-e4cd8978_20231122.onnx",
    "dwpose": "dw-ll_ucoco_384.onnx",
    # yolox: rtmlib caches 'm' variant; we search for both l and m
    "yolox":  "yolox_m_8xb8-300e_humanart-c2c7a14a.onnx",
}

# Alternative filenames to check (rtmlib may have downloaded either variant)
ONNX_ALTERNATES = {
    "yolox": [
        "yolox_m_8xb8-300e_humanart-c2c7a14a.onnx",
        "yolox_l_8xb8-300e_humanart-a39d44ed.onnx",
    ],
}


def get_pose_model_dir() -> Path:
    try:
        base = Path(folder_paths.models_dir)
    except AttributeError:
        base = Path(__file__).parent.parent.parent.parent / "models"
    pose_dir = base / "pose"
    pose_dir.mkdir(parents=True, exist_ok=True)
    return pose_dir


def _search_locations(filename: str) -> Optional[Path]:
    """Search all known locations where the ONNX file might be."""
    candidates = [
        get_pose_model_dir() / filename,
        Path.home() / ".rtmlib" / "hub" / "checkpoints" / filename,
        Path.home() / ".rtmlib" / "hub" / filename,
        Path.home() / ".cache" / "rtmlib" / "hub" / "checkpoints" / filename,
        Path.home() / ".cache" / "rtmlib" / "hub" / filename,
        Path.home() / ".cache" / "rtmlib" / filename,
        Path(os.environ.get("APPDATA", "")) / "rtmlib" / filename,
        Path(os.environ.get("LOCALAPPDATA", "")) / "rtmlib" / filename,
    ]
    for p in candidates:
        try:
            if p.exists():
                return p
        except Exception:
            pass
    return None


def get_model_path(model_key: str) -> Optional[Path]:
    """Return local ONNX path if found, searching all known locations."""
    # Check primary filename
    primary = ONNX_FILENAMES.get(model_key)
    if primary:
        found = _search_locations(primary)
        if found:
            return found

    # Check alternate filenames (e.g. yolox l vs m)
    for alt in ONNX_ALTERNATES.get(model_key, []):
        found = _search_locations(alt)
        if found:
            return found

    return None


def get_detector_path() -> Optional[Path]:
    return get_model_path("yolox")


def ensure_model_available(model_key: str) -> Optional[str]:
    path = get_model_path(model_key)
    return str(path) if path else None


def ensure_detector_available() -> Optional[str]:
    path = get_detector_path()
    return str(path) if path else None


def download_model(model_key: str) -> Optional[str]:
    """Download and extract model ONNX to models/pose/ if not already present."""
    existing = ensure_model_available(model_key)
    if existing:
        return existing

    url = MODEL_URLS.get(model_key)
    if not url:
        print(f"[Eric_Composer_Studio] No download URL for model: {model_key}")
        return None

    dest_dir  = get_pose_model_dir()
    filename  = ONNX_FILENAMES[model_key]
    dest_path = dest_dir / filename

    print(f"[Eric_Composer_Studio] Downloading {model_key} ...")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "model.zip"
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    if member.endswith(".onnx"):
                        zf.extract(member, tmp)
                        extracted = next(Path(tmp).rglob("*.onnx"))
                        extracted.rename(dest_path)
                        break
        if dest_path.exists():
            print(f"[Eric_Composer_Studio] Downloaded {model_key} → {dest_path}")
            return str(dest_path)
        return None
    except Exception as e:
        print(f"[Eric_Composer_Studio] Download failed for {model_key}: {e}")
        return None


def list_available_models() -> dict:
    result = {}
    for key in ["rtmw-x", "rtmw-l", "rtmw-m", "dwpose", "yolox"]:
        path = get_model_path(key)
        result[key] = {"found": path is not None, "path": str(path) if path else None}
    return result


def print_model_status():
    print("[Eric_Composer_Studio] Model status:")
    for key, info in list_available_models().items():
        status = f"  {info['path']}" if info["found"] else "  (not found)"
        print(f"  {key:12s}: {'✓' if info['found'] else '○'} {status}")
