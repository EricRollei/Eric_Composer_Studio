"""
PoseComposer node for Eric_Composer_Studio. (v13 - DWPose skips RTMW artifact filters)

Per-photo detection model stored in composition_data["photo_models"]:
  "0": "rtmw"   → RTMW-x (performance) - leg artifact cleanup applied
  "1": "dwpose" → dw-ll_ucoco_384.onnx - NO leg cleanup (DWPose doesn't need it)

xinsr_stick_scaling stored in composition_data["xinsr_stick_scaling"] (bool, default False).
Toggled from JS ratio bar.

Key design: re-detection preserves all existing person transforms by matching
person IDs. Only genuinely new persons get default transforms.
"""

from __future__ import annotations
import json
import copy
import math
import hashlib
import numpy as np

from ..core.pose_types    import coco133_to_pose_keypoint
from ..core.draw          import draw_pose, numpy_to_tensor
from ..core.model_manager import (
    ensure_model_available, ensure_detector_available,
    download_model, RTMLIB_MODEL_MODES,
)

MAX_PERSONS = 6
DEFAULT_CW  = 1024
DEFAULT_CH  = 1024

PERSON_COLORS = [
    (74,  158, 255),
    (255, 107, 53 ),
    (76,  175, 80 ),
    (224, 64,  251),
    (255, 215, 64 ),
    (38,  198, 218),
]

_DET_MODEL_KEYS = {
    "rtmw":   "rtmw-x",
    "dwpose": "dwpose",
}

_TRANSFORM_KEYS = ("x", "y", "scale", "scale_x", "scale_y", "rotation", "visible", "coord_v")

_MODEL_CACHE: dict = {}


def _content_hash(photos, poses, score_thr, det_thr, photo_models):
    parts = [f"s{score_thr:.2f}_d{det_thr:.2f}"]
    for i, p in enumerate(photos):
        m = photo_models.get(str(i), "rtmw")
        parts.append(f"photo_{p.shape[1]}x{p.shape[2]}_{m}" if p is not None else "none")
    for pk in poses:
        if pk is not None and isinstance(pk, list) and pk:
            n  = len(pk[0].get("people", []))
            cw = pk[0].get("canvas_width",  0)
            ch = pk[0].get("canvas_height", 0)
            parts.append(f"pose_{n}_{cw}x{ch}")
        else:
            parts.append("none")
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


def _load_wb(det_model: str, det_score_thresh: float):
    model_key = _DET_MODEL_KEYS.get(det_model, "rtmw-x")
    cache_key = f"{model_key}_{det_score_thresh:.2f}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    from rtmlib import Wholebody
    det_path  = ensure_detector_available()
    pose_path = ensure_model_available(model_key)

    if pose_path is None and model_key == "dwpose":
        print("[PoseComposer] dw-ll_ucoco_384.onnx not found - downloading (~50MB)...")
        pose_path = download_model("dwpose")
        if pose_path is None:
            raise RuntimeError(
                "[PoseComposer] DWPose ONNX could not be downloaded.\n"
                "Please manually place dw-ll_ucoco_384.onnx in ComfyUI/models/pose/\n"
                "Download: https://download.openmmlab.com/mmpose/v1/projects/"
                "rtmposev1/onnx_sdk/dw-ll_ucoco_384.zip"
            )
    if det_path is None:
        print("[PoseComposer] YOLOX detector not found - attempting download...")
        det_path = download_model("yolox")

    print(f"[PoseComposer] Loading {model_key} "
          f"(det={'local' if det_path else 'rtmlib-auto'}, "
          f"pose={'local' if pose_path else 'rtmlib-auto'})")

    kwargs = dict(to_openpose=False, backend="onnxruntime", device="cuda")
    if det_path and pose_path:
        kwargs["det"]  = det_path
        kwargs["pose"] = pose_path
    elif pose_path:
        kwargs["pose"] = pose_path
        kwargs["mode"] = "balanced"
    else:
        mode = RTMLIB_MODEL_MODES.get(model_key)
        if mode:
            kwargs["mode"] = mode
        else:
            raise RuntimeError(f"[PoseComposer] Could not find model: {model_key}")

    wb = Wholebody(**kwargs)
    try:
        if hasattr(wb, "det") and hasattr(wb.det, "det_score_thr"):
            wb.det.det_score_thr = det_score_thresh
        elif hasattr(wb, "body") and hasattr(wb.body, "det_score_thr"):
            wb.body.det_score_thr = det_score_thresh
    except Exception:
        pass

    _MODEL_CACHE[cache_key] = wb
    return wb


def _detect_from_photo(photo_tensor, score_threshold=0.3,
                       det_score_thresh=0.5, det_model="rtmw"):
    import cv2
    img_rgb = (photo_tensor[0].cpu().numpy() * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w    = img_bgr.shape[:2]
    wb = _load_wb(det_model, det_score_thresh)
    try:
        keypoints, scores = wb(img_bgr)
    except Exception as e:
        print(f"[PoseComposer] Detection error ({det_model}): {e}")
        keypoints = np.zeros((0, 133, 2), dtype=np.float32)
        scores    = np.zeros((0, 133),    dtype=np.float32)
    apply_cleanup = (det_model != "dwpose")
    pose_kp = coco133_to_pose_keypoint(
        keypoints, scores, w, h, score_threshold,
        apply_leg_cleanup=apply_cleanup,
    )
    return pose_kp[0].get("people", []), w, h


def _collect_persons(photos, poses, score_threshold=0.3,
                     det_score_thresh=0.5, photo_models=None):
    if photo_models is None:
        photo_models = {}
    persons = []
    color_idx = 0
    for label, photo_idx in [("Photo 1", 0), ("Photo 2", 1), ("Photo 3", 2)]:
        photo = photos[photo_idx]
        if photo is None or len(persons) >= MAX_PERSONS: continue
        det_model = photo_models.get(str(photo_idx), "rtmw")
        people, src_w, src_h = _detect_from_photo(
            photo, score_threshold, det_score_thresh, det_model
        )
        for pidx, person in enumerate(people):
            if len(persons) >= MAX_PERSONS: break
            persons.append({
                "id": f"ph{len(persons)}", "label": f"{label} - Person {pidx+1}",
                "color": list(PERSON_COLORS[color_idx % len(PERSON_COLORS)]),
                "src_w": src_w, "src_h": src_h, "person": person,
                "photo_idx": photo_idx, "det_model": det_model,
            })
            color_idx += 1
    for label, pk in [("Pose 1", poses[0]), ("Pose 2", poses[1]), ("Pose 3", poses[2])]:
        if pk is None or len(persons) >= MAX_PERSONS: continue
        if not isinstance(pk, list): pk = [pk]
        if not pk: continue
        entry = pk[0]
        src_w = entry.get("canvas_width",1024)
        src_h = entry.get("canvas_height",1024)
        for pidx, person in enumerate(entry.get("people", [])):
            if len(persons) >= MAX_PERSONS: break
            persons.append({
                "id": f"pk{len(persons)}", "label": f"{label} - Person {pidx+1}",
                "color": list(PERSON_COLORS[color_idx % len(PERSON_COLORS)]),
                "src_w": src_w, "src_h": src_h, "person": person,
                "photo_idx": None, "det_model": "external",
            })
            color_idx += 1
    return persons


def _persons_from_dpj(dpj_str):
    try:
        dpj = json.loads(dpj_str)
        return [{
            "id": p["id"], "label": p["label"], "color": p["color"],
            "src_w": p["src_w"], "src_h": p["src_h"], "person": p["person"],
            "photo_idx": p.get("photo_idx", None),
            "det_model": p.get("det_model", "rtmw"),
        } for p in dpj.get("persons", [])]
    except Exception:
        return []


def _centroid_norm(person, src_w, src_h):
    flat = person.get("pose_keypoints_2d", [])
    xs, ys = [], []
    for i in range(0, min(len(flat), 54), 3):
        if flat[i+2] > 0:
            xs.append(flat[i]   / max(src_w, 1))
            ys.append(flat[i+1] / max(src_h, 1))
    return (sum(xs)/len(xs), sum(ys)/len(ys)) if xs else (0.5, 0.5)


def _apply_transform(person, src_w, src_h, tgt_w, tgt_h,
                     cx_n, cy_n, user_x_frac, user_y_frac,
                     scale, scale_x=1.0, scale_y=1.0, rotation=0.0):
    cos_a = math.cos(rotation)
    sin_a = math.sin(rotation)
    result = copy.deepcopy(person)
    for key in ["pose_keypoints_2d","face_keypoints_2d",
                "hand_left_keypoints_2d","hand_right_keypoints_2d","foot_keypoints_2d"]:
        flat = person.get(key, [])
        new_flat = []
        for i in range(0, len(flat), 3):
            x, y, c = flat[i], flat[i+1], flat[i+2]
            if c <= 0:
                new_flat.extend([0.0, 0.0, 0.0])
            else:
                kp_nx = x / max(src_w, 1)
                kp_ny = y / max(src_h, 1)
                dx = (kp_nx - cx_n) * scale * scale_x * tgt_w
                dy = (kp_ny - cy_n) * scale * scale_y * tgt_w
                new_flat.extend([
                    user_x_frac * tgt_w + dx * cos_a - dy * sin_a,
                    user_y_frac * tgt_h + dx * sin_a + dy * cos_a, c,
                ])
        result[key] = new_flat
    return result


def _build_composition(persons_info, canvas_w, canvas_h, display_w, display_h,
                       color_mode, photo_models, xinsr_stick_scaling=False, old_comp=None):
    old_persons = {}
    if old_comp and "persons" in old_comp:
        old_persons = {p["id"]: p for p in old_comp["persons"]}
    persons_out = []
    for info in persons_info:
        old = old_persons.get(info["id"])
        if old:
            entry = {k: old[k] for k in _TRANSFORM_KEYS if k in old}
            entry["id"] = info["id"]
        else:
            cx_n, cy_n = _centroid_norm(info["person"], info["src_w"], info["src_h"])
            entry = {
                "id": info["id"], "x": round(cx_n, 4), "y": round(cy_n, 4),
                "scale": 1.0, "scale_x": 1.0, "scale_y": 1.0,
                "rotation": 0.0, "visible": True, "coord_v": 2,
            }
        persons_out.append(entry)
    return {
        "canvas_width": canvas_w, "canvas_height": canvas_h,
        "display_w": display_w, "display_h": display_h,
        "color_mode": color_mode, "coord_v": 2,
        "persons": persons_out, "photo_models": photo_models,
        "xinsr_stick_scaling": xinsr_stick_scaling,
    }


class PoseComposer:
    CATEGORY     = "Eric_Composer_Studio"
    FUNCTION     = "compose"
    RETURN_TYPES = ("POSE_KEYPOINT", "IMAGE")
    RETURN_NAMES = ("pose_keypoint",  "skeleton_image")
    OUTPUT_NODE  = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "photo_1": ("IMAGE",), "photo_2": ("IMAGE",), "photo_3": ("IMAGE",),
                "pose_1":  ("POSE_KEYPOINT",), "pose_2": ("POSE_KEYPOINT",), "pose_3": ("POSE_KEYPOINT",),
                "score_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Keypoint confidence. Lower to 0.15-0.20 for occluded arms/hands."}),
                "det_score_thresh": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "Person detector confidence. Lower only if whole people are missed."}),
                "content_hash":        ("STRING", {"default": ""}),
                "detected_poses_json": ("STRING", {"default": ""}),
                "composition_data":    ("STRING", {"default": ""}),
            },
            "hidden": {"node_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("nan")

    def compose(self, photo_1=None, photo_2=None, photo_3=None,
                pose_1=None, pose_2=None, pose_3=None,
                score_threshold=0.3, det_score_thresh=0.5,
                content_hash="", detected_poses_json="", composition_data="",
                node_id=None):

        photos = [photo_1, photo_2, photo_3]
        poses  = [pose_1,  pose_2,  pose_3]

        try:
            comp_stored = json.loads(composition_data) if composition_data.strip() else {}
        except Exception:
            comp_stored = {}

        canvas_w            = comp_stored.get("canvas_width",  DEFAULT_CW)
        canvas_h            = comp_stored.get("canvas_height", DEFAULT_CH)
        display_w           = comp_stored.get("display_w",  440)
        display_h           = comp_stored.get("display_h",  440)
        color_mode          = comp_stored.get("color_mode", "dwpose")
        xinsr_stick_scaling = comp_stored.get("xinsr_stick_scaling", False)
        is_v2               = (comp_stored.get("coord_v", 1) == 2)
        photo_models        = comp_stored.get("photo_models", {"0":"rtmw","1":"rtmw","2":"rtmw"})

        cur_hash = _content_hash(photos, poses, score_threshold, det_score_thresh, photo_models)
        changed  = (cur_hash != content_hash)

        if changed or not detected_poses_json.strip():
            print(f"[PoseComposer] Detecting - models: {photo_models} "
                  f"(score={score_threshold}, det={det_score_thresh})")
            persons_info = _collect_persons(
                photos, poses, score_threshold, det_score_thresh, photo_models
            )
            dpj_new  = json.dumps({
                "canvas_width": canvas_w, "canvas_height": canvas_h,
                "persons": [{k: v for k, v in p.items()} for p in persons_info],
            })
            comp_new = json.dumps(_build_composition(
                persons_info, canvas_w, canvas_h, display_w, display_h,
                color_mode, photo_models, xinsr_stick_scaling,
                old_comp=comp_stored if comp_stored.get("persons") else None,
            ))
            is_v2 = True
        else:
            persons_info = _persons_from_dpj(detected_poses_json)
            dpj_new      = detected_poses_json
            comp_new     = composition_data

        try:
            comp   = json.loads(comp_new)
            xforms = {p["id"]: p for p in comp.get("persons", [])}
            is_v2  = (comp.get("coord_v", 1) == 2)
        except Exception:
            xforms = {}
            is_v2 = False

        merged = []
        for info in persons_info:
            xf = xforms.get(info["id"], {})
            if not xf.get("visible", True): continue
            cx_n, cy_n = _centroid_norm(info["person"], info["src_w"], info["src_h"])
            if is_v2:
                user_x, user_y = xf.get("x", cx_n), xf.get("y", cy_n)
            else:
                user_x = xf.get("x", cx_n * canvas_w) / max(canvas_w, 1)
                user_y = xf.get("y", cy_n * canvas_h) / max(canvas_h, 1)
            merged.append(_apply_transform(
                info["person"], info["src_w"], info["src_h"],
                canvas_w, canvas_h, cx_n, cy_n, user_x, user_y,
                xf.get("scale", 1.0), xf.get("scale_x", 1.0),
                xf.get("scale_y", 1.0), xf.get("rotation", 0.0),
            ))

        output_kp   = [{"version": 1.3, "people": merged,
                        "canvas_width": canvas_w, "canvas_height": canvas_h}]
        skel_bgr    = draw_pose(output_kp, canvas_w, canvas_h,
                                line_width=4, joint_radius=4,
                                color_mode=color_mode,
                                xinsr_stick_scaling=xinsr_stick_scaling)
        skel_tensor = numpy_to_tensor(skel_bgr)

        return {
            "ui": {"pose_composer_data": [{"content_hash": cur_hash,
                                           "detected_poses_json": dpj_new,
                                           "composition_data": comp_new}]},
            "result": (output_kp, skel_tensor),
        }
