"""
Drawing utilities for rendering pose skeletons to numpy images.

DWPose mode matches controlnet_aux util.py exactly:
  - Body limbs: filled ellipses (fillConvexPoly), color*0.6, stickwidth=4 (× stick_scale if xinsr)
  - Body joints: cv2.circle radius 4 (× stick_scale if xinsr), full bright color
  - Hands: 20 edges only (official spec), HSV rainbow per edge, thickness=2, no joint dots
  - Face: white dots, radius controlled by face_dot_radius (default 2)
  - Pelvis link (idx 17): Enhanced mode only — omitted in DWPose mode

Enhanced mode: warm/cool L/R colors, cv2.line, larger dots.

xinsr_stick_scaling: matches xinsir/controlnet-openpose-sdxl-1.0 training data.
"""

from __future__ import annotations
import math
import numpy as np
import cv2

from .skeleton_def import (
    BODY_LIMBS, FOOT_LIMBS, HAND_LIMBS,
    DWPOSE_LIMB_COLORS, ENHANCED_LIMB_COLORS,
    DWPOSE_KP_COLORS,  ENHANCED_KP_COLORS,
    LEFT_HAND_COLOR, RIGHT_HAND_COLOR,
    FOOT_COLOR_LEFT, FOOT_COLOR_RIGHT,
    FACE_COLOR,
)
from .pose_types import pose_keypoint_to_arrays, PoseKeypoint

MAX_LIMB_RATIO       = 5.0
MAX_LIMB_FALLBACK    = 0.65
COLLINEAR_BOTH_FRAC  = 0.75
COLLINEAR_FRAC       = 0.05
COLLINEAR_MIN_PX     = 10.0

_HAND_EDGES_OFFICIAL = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]


def _build_hand_colors():
    try:
        import matplotlib.colors as mc
        return [
            tuple(int(c) for c in (np.array(mc.hsv_to_rgb([ie / 20.0, 1.0, 1.0])) * 255).astype(int))
            for ie in range(20)
        ]
    except ImportError:
        colors = []
        for ie in range(20):
            h = ie / 20.0
            i = int(h * 6)
            f = h * 6 - i
            q = 1 - f
            rgb = [(1,f,0),(q,1,0),(0,1,f),(0,q,1),(f,0,1),(1,0,q)][i % 6]
            colors.append(tuple(int(c * 255) for c in rgb))
        return colors


_HAND_COLORS = None
def _get_hand_colors():
    global _HAND_COLORS
    if _HAND_COLORS is None:
        _HAND_COLORS = _build_hand_colors()
    return _HAND_COLORS


def _xinsr_stick_scale(canvas_width: int, canvas_height: int) -> int:
    max_side = max(canvas_width, canvas_height)
    if max_side < 500: return 1
    return min(2 + (max_side // 1000), 7)


def _draw_limb_ellipse(canvas, p1, p2, color_rgb, stickwidth):
    x1, y1 = p1
    x2, y2 = p2
    mX = int((x1+x2)/2)
    mY = int((y1+y2)/2)
    length=int(math.hypot(x2-x1,y2-y1)/2)
    if length<1: return
    angle=math.degrees(math.atan2(y2-y1,x2-x1))
    try:
        polygon=cv2.ellipse2Poly((mX,mY),(length,max(1,stickwidth)),int(angle),0,360,1)
    except Exception:
        cv2.line(canvas,p1,p2,[int(float(c)*0.6) for c in color_rgb][::-1],
                 max(1,stickwidth*2),lineType=cv2.LINE_AA)
        return
    dark=[int(float(c)*0.6) for c in color_rgb]
    cv2.fillConvexPoly(canvas, polygon, dark[::-1])


def _collinear_leg_skip(body, pt_fn, canvas_height):
    skip=set()
    threshold=canvas_height*COLLINEAR_BOTH_FRAC
    for hip_i,knee_i,ankle_i in [(8,9,10),(11,12,13)]:
        ph = pt_fn(body[hip_i])
        pk = pt_fn(body[knee_i])
        pa = pt_fn(body[ankle_i])
        if not(ph and pk and pa): continue
        if pk[1]<threshold or pa[1]<threshold: continue
        ax, ay = ph
        bx, by = pk
        cx, cy = pa
        cross=abs((cx-ax)*(by-ay)-(cy-ay)*(bx-ax))
        ac_len=math.hypot(cx-ax,cy-ay)
        if ac_len<1.: continue
        if cross/ac_len < max(COLLINEAR_MIN_PX, ac_len*COLLINEAR_FRAC):
            skip.add(knee_i)
            skip.add(ankle_i)
    return skip


def _max_limb_px(body, pt_fn, canvas_width, canvas_height):
    lengths=[]
    for i,j in BODY_LIMBS:
        p1 = pt_fn(body[i])
        p2 = pt_fn(body[j])
        if p1 and p2:
            d=math.hypot(p2[0]-p1[0],p2[1]-p1[1])
            if d>0: lengths.append(d)
    if len(lengths)>=4:
        lengths.sort()
        p75=lengths[int(len(lengths)*0.75)]
        return max(p75*MAX_LIMB_RATIO, 40.)
    return max(canvas_width,canvas_height)*MAX_LIMB_FALLBACK


def draw_pose(
    pose_kp, canvas_width, canvas_height,
    line_width=4, joint_radius=4, face_dot_radius=None,
    color_mode="dwpose",
    draw_face=True, draw_hands=True, draw_feet=True,
    bg_color=(0,0,0), image_idx=0,
    xinsr_stick_scaling=False,
):
    """
    face_dot_radius: explicit face dot size override.
    If None, defaults to:
      DWPose mode:  2  (official spec is 3, but smaller looks cleaner)
      Enhanced mode: max(1, joint_radius - 2)
    """
    canvas=np.full((canvas_height,canvas_width,3),bg_color,dtype=np.uint8)
    people,pw,ph=pose_keypoint_to_arrays(pose_kp,image_idx)
    if not people: return canvas

    sx = canvas_width/max(pw,1)
    sy = canvas_height/max(ph,1)

    if color_mode=="dwpose":
        stick_scale = _xinsr_stick_scale(canvas_width, canvas_height) if xinsr_stick_scaling else 1
        stickwidth  = 4 * stick_scale
        jr          = 4 * stick_scale
        fdr         = face_dot_radius if face_dot_radius is not None else 2
    else:
        stickwidth = max(1, line_width)
        jr         = max(1, joint_radius)
        fdr        = face_dot_radius if face_dot_radius is not None else max(1, jr - 2)

    limb_colors = ENHANCED_LIMB_COLORS if color_mode=="enhanced" else DWPOSE_LIMB_COLORS
    kp_colors   = ENHANCED_KP_COLORS   if color_mode=="enhanced" else DWPOSE_KP_COLORS

    for person in people:
        body = person["body"]
        face = person["face"]
        hand_left = person["hand_left"]
        hand_right = person["hand_right"]
        foot = person["foot"]

        def pt(kp_row):
            if kp_row[2]<=0: return None
            return (int(kp_row[0]*sx),int(kp_row[1]*sy))

        skip=_collinear_leg_skip(body,pt,canvas_height)
        max_px=_max_limb_px(body,pt,canvas_width,canvas_height)
        def ok(p1,p2): return math.hypot(p2[0]-p1[0],p2[1]-p1[1])<=max_px

        for idx,(i,j) in enumerate(BODY_LIMBS):
            if color_mode=="dwpose" and idx==17: continue  # pelvis: Enhanced mode only
            if i in skip or j in skip: continue
            p1 = pt(body[i])
            p2 = pt(body[j])
            if not(p1 and p2) or not ok(p1,p2): continue
            c=limb_colors[idx]
            if color_mode=="dwpose": _draw_limb_ellipse(canvas,p1,p2,c,stickwidth)
            else: cv2.line(canvas,p1,p2,c[::-1],stickwidth,lineType=cv2.LINE_AA)

        for ki,kr in enumerate(body):
            if ki in skip: continue
            p=pt(kr)
            if p:
                c=kp_colors[ki] if ki<len(kp_colors) else (255,255,255)
                cv2.circle(canvas,p,jr,c[::-1],-1,lineType=cv2.LINE_AA)

        if draw_feet:
            _draw_part_limbs(canvas,foot,FOOT_LIMBS,sx,sy,
                             FOOT_COLOR_LEFT,FOOT_COLOR_RIGHT,3,stickwidth,jr,color_mode,max_px)

        if draw_hands:
            if color_mode=="dwpose":
                _draw_hand_dwpose(canvas,hand_left, sx,sy,max_px)
                _draw_hand_dwpose(canvas,hand_right,sx,sy,max_px)
            else:
                _draw_hand_enhanced(canvas,hand_left, LEFT_HAND_COLOR, sx,sy,max(1,stickwidth-1),jr,max_px)
                _draw_hand_enhanced(canvas,hand_right,RIGHT_HAND_COLOR,sx,sy,max(1,stickwidth-1),jr,max_px)

        if draw_face:
            _draw_face(canvas,face,sx,sy,fdr)

    return canvas


def _draw_part_limbs(canvas,kps,limbs,sx,sy,lc,rc,si,sw,jr,cm,max_px):
    for i,j in limbs:
        if i>=len(kps) or j>=len(kps): continue
        if kps[i,2]<=0 or kps[j,2]<=0: continue
        p1=(int(kps[i,0]*sx),int(kps[i,1]*sy))
        p2=(int(kps[j,0]*sx),int(kps[j,1]*sy))
        if math.hypot(p2[0]-p1[0],p2[1]-p1[1])>max_px: continue
        c=lc if i<si else rc
        if cm=="dwpose": _draw_limb_ellipse(canvas,p1,p2,c,sw)
        else: cv2.line(canvas,p1,p2,c[::-1],sw,lineType=cv2.LINE_AA)
    for idx,kp in enumerate(kps):
        if kp[2]<=0: continue
        cv2.circle(canvas,(int(kp[0]*sx),int(kp[1]*sy)),jr,(lc if idx<si else rc)[::-1],-1,lineType=cv2.LINE_AA)


def _draw_hand_dwpose(canvas, hand_kps, sx, sy, max_px):
    if hand_kps is None or len(hand_kps)<21: return
    hand_colors=_get_hand_colors()
    hand_max = max_px*0.25
    eps = 0.01
    for ie,(i,j) in enumerate(_HAND_EDGES_OFFICIAL):
        if hand_kps[i,2]<=0 or hand_kps[j,2]<=0: continue
        x1 = int(hand_kps[i,0]*sx)
        y1 = int(hand_kps[i,1]*sy)
        x2 = int(hand_kps[j,0]*sx)
        y2 = int(hand_kps[j,1]*sy)
        if x1>eps and y1>eps and x2>eps and y2>eps:
            if math.hypot(x2-x1,y2-y1)<=hand_max:
                rgb=hand_colors[ie]
                cv2.line(canvas,(x1,y1),(x2,y2),(rgb[2],rgb[1],rgb[0]),thickness=2)


def _draw_hand_enhanced(canvas,hkps,color,sx,sy,sw,jr,max_px):
    if hkps is None or len(hkps)<21: return
    for i,j in HAND_LIMBS:
        if hkps[i,2]<=0 or hkps[j,2]<=0: continue
        p1=(int(hkps[i,0]*sx),int(hkps[i,1]*sy))
        p2=(int(hkps[j,0]*sx),int(hkps[j,1]*sy))
        if math.hypot(p2[0]-p1[0],p2[1]-p1[1])>max_px: continue
        cv2.line(canvas,p1,p2,color[::-1],sw,lineType=cv2.LINE_AA)
    for kp in hkps:
        if kp[2]<=0: continue
        cv2.circle(canvas,(int(kp[0]*sx),int(kp[1]*sy)),jr,color[::-1],-1,lineType=cv2.LINE_AA)


def _draw_face(canvas,fkps,sx,sy,r):
    if fkps is None: return
    c=FACE_COLOR[::-1]
    for kp in fkps:
        if kp[2]<=0: continue
        cv2.circle(canvas,(int(kp[0]*sx),int(kp[1]*sy)),max(1,r),c,-1,lineType=cv2.LINE_AA)


def numpy_to_tensor(img_bgr):
    import torch
    return torch.from_numpy(cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB).astype(np.float32)/255.).unsqueeze(0)

def tensor_to_numpy_bgr(tensor):
    img=(tensor.squeeze(0).cpu().numpy()*255).clip(0,255).astype(np.uint8)
    return cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
