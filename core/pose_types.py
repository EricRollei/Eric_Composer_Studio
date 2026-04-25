"""
POSE_KEYPOINT format utilities for Eric_Composer_Studio.

Cleanup pipeline - RTMW only (apply_leg_cleanup=True):
  1. Confidence threshold  [always applied]
  2. Bounds check          [always applied]
  3. Bottom-edge clamp for knees/ankles/feet  [RTMW only]
  4. Collinearity check for leg triplets      [RTMW only]

DWPose (apply_leg_cleanup=False): only confidence + bounds.
DWPose places keypoints conservatively and never extrapolates to image edges,
so the RTMW artifact filters must not be applied to it.
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Dict, Any

from .skeleton_def import (
    OPENPOSE_18_FROM_COCO,
    OPENPOSE_FOOT_FROM_COCO,
    NUM_HAND_KPS, NUM_FACE_KPS,
    LHAND_START, RHAND_START, FACE_START, FOOT_START,
)

PoseKeypoint = List[Dict[str, Any]]

_BOUNDS_MARGIN     = 4
_BOTTOM_FRAC       = 0.97
_FOOT_BOTTOM_FRAC  = 0.96
_LEG_LOWER_INDICES = {9, 10, 12, 13}

_COLLINEAR_BOTH_FRAC = 0.75
_COLLINEAR_FRAC      = 0.05
_COLLINEAR_MIN_PX    = 12.0


def empty_pose_keypoint(width, height):
    return [{"version": 1.3, "people": [], "canvas_width": width, "canvas_height": height}]

def empty_person_dict():
    return {
        "person_id": [-1],
        "pose_keypoints_2d":       [0.0] * 54,
        "face_keypoints_2d":       [0.0] * 210,
        "hand_left_keypoints_2d":  [0.0] * 63,
        "hand_right_keypoints_2d": [0.0] * 63,
        "foot_keypoints_2d":       [0.0] * 18,
        "pose_keypoints_3d":       [],
        "face_keypoints_3d":       [],
        "hand_left_keypoints_3d":  [],
        "hand_right_keypoints_3d": [],
    }

def _in_bounds(x, y, width, height):
    m = _BOUNDS_MARGIN
    return (-m <= x <= width + m) and (-m <= y <= height + m)

def _knee_ankle_ok(y, height): return y < height * _BOTTOM_FRAC
def _foot_ok(y, height):       return y < height * _FOOT_BOTTOM_FRAC


def _collinear_leg_zero(body_kps: np.ndarray, height: int) -> np.ndarray:
    """
    RTMW-only: zero knee+ankle when collinear with hip and both are
    in the bottom 25% of the image (RTMW edge-clamped extrapolation artifact).
    """
    result = body_kps.copy()
    threshold = height * _COLLINEAR_BOTH_FRAC
    for hip_i, knee_i, ankle_i in [(8, 9, 10), (11, 12, 13)]:
        h, k, a = result[hip_i], result[knee_i], result[ankle_i]
        if h[2] <= 0 or k[2] <= 0 or a[2] <= 0: continue
        if k[1] < threshold or a[1] < threshold: continue
        ax, ay = h[0], h[1]
        bx, by = k[0], k[1]
        cx, cy = a[0], a[1]
        cross  = abs((cx-ax)*(by-ay) - (cy-ay)*(bx-ax))
        ac_len = math.hypot(cx-ax, cy-ay)
        if ac_len < 1.0: continue
        if cross/ac_len < max(_COLLINEAR_MIN_PX, ac_len * _COLLINEAR_FRAC):
            result[knee_i]  = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            result[ankle_i] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return result


def coco133_to_pose_keypoint(
    keypoints, scores, width, height,
    score_threshold: float = 0.3,
    apply_leg_cleanup: bool = True,
) -> PoseKeypoint:
    """
    Convert rtmlib Wholebody output to POSE_KEYPOINT format.

    apply_leg_cleanup=True  (default, RTMW):
        Applies bottom-edge clamp and collinearity check to remove
        extrapolated leg artifacts specific to RTMW.

    apply_leg_cleanup=False (DWPose):
        Skips all leg-specific filters - only confidence + bounds check.
        DWPose doesn't extrapolate legs to image edges so the filters
        would incorrectly remove valid leg keypoints.
    """
    people = []
    for person_idx in range(len(keypoints)):
        kps = keypoints[person_idx]
        scores_ = scores[person_idx]
        person = empty_person_dict()

        # ── Body ──────────────────────────────────────────────────────────
        raw_body = np.zeros((18, 3), dtype=np.float32)
        for op_idx in range(18):
            coco_ref = OPENPOSE_18_FROM_COCO[op_idx]
            if isinstance(coco_ref, tuple):
                i,j=coco_ref
                x = float((kps[i,0]+kps[j,0])/2)
                y = float((kps[i,1]+kps[j,1])/2)
                c=float((scores_[i]+scores_[j])/2)
            else:
                x = float(kps[coco_ref,0])
                y = float(kps[coco_ref,1])
                c = float(scores_[coco_ref])
            ok = c >= score_threshold and _in_bounds(x, y, width, height)
            # RTMW only: reject knees/ankles clamped to image bottom edge
            if ok and apply_leg_cleanup and op_idx in _LEG_LOWER_INDICES:
                ok = _knee_ankle_ok(y, height)
            if not ok: x,y,c = 0.,0.,0.
            raw_body[op_idx] = [x, y, c]

        # RTMW only: remove collinear leg triplets
        if apply_leg_cleanup:
            raw_body = _collinear_leg_zero(raw_body, height)

        person["pose_keypoints_2d"] = raw_body.flatten().tolist()

        # ── Feet ──────────────────────────────────────────────────────────
        foot_flat = []
        for fi in range(6):
            ci=OPENPOSE_FOOT_FROM_COCO[fi]
            x = float(kps[ci,0])
            y = float(kps[ci,1])
            c = float(scores_[ci])
            ok = c >= score_threshold and _in_bounds(x, y, width, height)
            # RTMW only: reject feet clamped to image bottom
            if ok and apply_leg_cleanup:
                ok = _foot_ok(y, height)
            if not ok: x,y,c = 0.,0.,0.
            foot_flat.extend([x, y, c])
        person["foot_keypoints_2d"] = foot_flat

        # ── Hands ─────────────────────────────────────────────────────────
        for attr, start in [("hand_left_keypoints_2d", LHAND_START),
                             ("hand_right_keypoints_2d", RHAND_START)]:
            flat = []
            for i in range(NUM_HAND_KPS):
                x = float(kps[start+i,0])
                y = float(kps[start+i,1])
                c = float(scores_[start+i])
                if c < score_threshold or not _in_bounds(x, y, width, height): x,y,c=0.,0.,0.
                flat.extend([x, y, c])
            person[attr] = flat

        # ── Face ──────────────────────────────────────────────────────────
        face_flat = []
        for i in range(NUM_FACE_KPS):
            x = float(kps[FACE_START+i,0])
            y = float(kps[FACE_START+i,1])
            c = float(scores_[FACE_START+i])
            if c < score_threshold or not _in_bounds(x, y, width, height): x,y,c=0.,0.,0.
            face_flat.extend([x, y, c])
        face_flat.extend([0.,0.,0.,0.,0.,0.])
        person["face_keypoints_2d"] = face_flat

        people.append(person)

    return [{"version":1.3,"people":people,"canvas_width":width,"canvas_height":height}]


def pose_keypoint_to_arrays(pose_kp, image_idx=0):
    if not pose_kp or image_idx >= len(pose_kp): return [],512,512
    entry = pose_kp[image_idx]
    w = entry.get("canvas_width",512)
    h = entry.get("canvas_height",512)
    result=[]
    for person in entry.get("people",[]):
        p={}
        def _parse(key,n):
            flat = person.get(key,[])
            arr = np.array(flat,dtype=np.float32)
            return arr[:n*3].reshape(n,3) if len(arr)>=n*3 else np.zeros((n,3),dtype=np.float32)
        p["body"] = _parse("pose_keypoints_2d",18)
        p["face"] = _parse("face_keypoints_2d",70)
        p["hand_left"] = _parse("hand_left_keypoints_2d",21)
        p["hand_right"] = _parse("hand_right_keypoints_2d",21)
        p["foot"]=_parse("foot_keypoints_2d",6)
        result.append(p)
    return result,w,h


def scale_pose_keypoint(pose_kp,src_w,src_h,dst_w,dst_h):
    if not pose_kp: return empty_pose_keypoint(dst_w,dst_h)
    sx = dst_w/max(src_w,1)
    sy = dst_h/max(src_h,1)
    import copy
    result = copy.deepcopy(pose_kp)
    for entry in result:
        entry["canvas_width"] = dst_w
        entry["canvas_height"] = dst_h
        for person in entry.get("people",[]):
            for key in ["pose_keypoints_2d","face_keypoints_2d",
                        "hand_left_keypoints_2d","hand_right_keypoints_2d","foot_keypoints_2d"]:
                flat = person.get(key,[])
                new_flat = []
                for i in range(0,len(flat),3):
                    new_flat.extend([flat[i]*sx if flat[i+2]>0 else 0.,
                                     flat[i+1]*sy if flat[i+2]>0 else 0.,flat[i+2]])
                person[key]=new_flat
    return result
