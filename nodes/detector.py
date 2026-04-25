"""
PoseDetector node for Eric_Composer_Studio.

Runs RTMW-x/l/m or DWPose via rtmlib ONNX backend.
Outputs POSE_KEYPOINT data compatible with comfyui_controlnet_aux + preview image.

apply_leg_cleanup is passed to coco133_to_pose_keypoint:
  True  for RTMW (removes edge-clamped extrapolation artifacts)
  False for DWPose (conservative placement, no artifacts to remove)
"""

from __future__ import annotations
import numpy as np
import torch
import cv2

from ..core.pose_types  import coco133_to_pose_keypoint, empty_pose_keypoint
from ..core.draw        import draw_pose, numpy_to_tensor, tensor_to_numpy_bgr
from ..core.model_manager import (
    ensure_model_available, ensure_detector_available,
    RTMLIB_MODEL_MODES, print_model_status,
)

_wholebody_cache: dict = {}


def _load_wholebody(model_key: str, device: str, det_score_thresh: float = 0.5):
    cache_key = f"{model_key}_{device}_{det_score_thresh:.2f}"
    if cache_key in _wholebody_cache:
        return _wholebody_cache[cache_key]

    try:
        from rtmlib import Wholebody
    except ImportError:
        raise RuntimeError("[Eric_Composer_Studio] rtmlib is not installed. Please run: pip install rtmlib")

    backend    = "onnxruntime"
    ort_device = "cuda" if device == "cuda" else "cpu"
    det_path   = ensure_detector_available()
    pose_path  = ensure_model_available(model_key)
    mode       = RTMLIB_MODEL_MODES.get(model_key, "balanced")

    print(f"[Eric_Composer_Studio] Loading {model_key} "
          f"(det={'local' if det_path else 'auto'}, "
          f"pose={'local' if pose_path else 'auto'}, "
          f"det_score_thresh={det_score_thresh})...")

    try:
        kwargs = dict(to_openpose=False, backend=backend, device=ort_device)
        if det_path and pose_path:
            kwargs["det"]  = det_path
            kwargs["pose"] = pose_path
        else:
            kwargs["mode"] = mode
        wb = Wholebody(**kwargs)
        try:
            if hasattr(wb, "det") and hasattr(wb.det, "det_score_thr"):
                wb.det.det_score_thr = det_score_thresh
                print(f"[Eric_Composer_Studio] Set det_score_thr={det_score_thresh}")
            elif hasattr(wb, "body") and hasattr(wb.body, "det_score_thr"):
                wb.body.det_score_thr = det_score_thresh
        except Exception as e:
            print(f"[Eric_Composer_Studio] Could not set det_score_thr: {e}")
    except Exception as e:
        raise RuntimeError(f"[Eric_Composer_Studio] Failed to load {model_key}: {e}")

    _wholebody_cache[cache_key] = wb
    print(f"[Eric_Composer_Studio] {model_key} loaded successfully")
    return wb


class PoseDetector:
    CATEGORY      = "Eric_Composer_Studio"
    FUNCTION      = "detect"
    RETURN_TYPES  = ("POSE_KEYPOINT", "IMAGE")
    RETURN_NAMES  = ("pose_keypoint",  "skeleton_preview")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":  ("IMAGE",),
                "model":  (["rtmw-x", "rtmw-l", "rtmw-m", "dwpose"], {"default": "rtmw-x"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "score_threshold": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Keypoint confidence cutoff. Lower (0.15-0.20) recovers arms/hands at partial occlusion.",
                }),
                "det_score_thresh": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "YOLO person detector confidence. Only lower if entire people are being missed.",
                }),
                "preview_color_mode": (["dwpose", "enhanced"], {"default": "enhanced"}),
                "preview_line_width":   ("INT", {"default": 4, "min": 1, "max": 12,  "step": 1}),
                "preview_joint_radius": ("INT", {"default": 5, "min": 1, "max": 16, "step": 1}),
                "draw_face":  ("BOOLEAN", {"default": True}),
                "draw_hands": ("BOOLEAN", {"default": True}),
                "draw_feet":  ("BOOLEAN", {"default": True}),
            }
        }

    def detect(self, image, model="rtmw-x", device="cuda",
               score_threshold=0.3, det_score_thresh=0.5,
               preview_color_mode="enhanced", preview_line_width=4,
               preview_joint_radius=5, draw_face=True, draw_hands=True, draw_feet=True):

        wb = _load_wholebody(model, device, det_score_thresh)
        # DWPose doesn't extrapolate legs to image edges - skip RTMW artifact filters
        apply_cleanup = (model != "dwpose")

        all_pose_kps = []
        all_previews = []

        for b in range(image.shape[0]):
            img_rgb = (image[b].cpu().numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            h, w    = img_bgr.shape[:2]

            try:
                keypoints, scores = wb(img_bgr)
            except Exception as e:
                print(f"[Eric_Composer_Studio] Inference error on batch {b}: {e}")
                keypoints = np.zeros((0, 133, 2), dtype=np.float32)
                scores    = np.zeros((0, 133),    dtype=np.float32)

            pose_kp = coco133_to_pose_keypoint(
                keypoints, scores, w, h, score_threshold,
                apply_leg_cleanup=apply_cleanup,
            )
            all_pose_kps.append(pose_kp[0])

            skeleton_bgr = draw_pose(
                pose_kp, canvas_width=w, canvas_height=h,
                line_width=preview_line_width, joint_radius=preview_joint_radius,
                color_mode=preview_color_mode,
                draw_face=draw_face, draw_hands=draw_hands, draw_feet=draw_feet,
            )
            all_previews.append(numpy_to_tensor(skeleton_bgr))

        return (all_pose_kps, torch.cat(all_previews, dim=0))
