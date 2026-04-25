"""
Skeleton topology, keypoint definitions, and color schemes for Eric_Composer_Studio.

COCO Wholebody 133-keypoint format (used internally by RTMW and DWPose):
  0-16:   Body (17 pts, COCO format)
  17-22:  Feet (6 pts)
  23-90:  Face (68 pts, 300W convention)
  91-111: Left hand (21 pts)
  112-132: Right hand (21 pts)

OpenPose 18-point body adds a synthetic 'neck' keypoint (index 1) computed
as the midpoint of left_shoulder (COCO 5) and right_shoulder (COCO 6).
"""

# ---------------------------------------------------------------------------
# COCO Wholebody 133 keypoint names
# ---------------------------------------------------------------------------

COCO_BODY_KPS = [
    "nose",          # 0
    "left_eye",      # 1
    "right_eye",     # 2
    "left_ear",      # 3
    "right_ear",     # 4
    "left_shoulder", # 5
    "right_shoulder",# 6
    "left_elbow",    # 7
    "right_elbow",   # 8
    "left_wrist",    # 9
    "right_wrist",   # 10
    "left_hip",      # 11
    "right_hip",     # 12
    "left_knee",     # 13
    "right_knee",    # 14
    "left_ankle",    # 15
    "right_ankle",   # 16
]

COCO_FOOT_KPS = [
    "left_big_toe",    # 17
    "left_small_toe",  # 18
    "left_heel",       # 19
    "right_big_toe",   # 20
    "right_small_toe", # 21
    "right_heel",      # 22
]

COCO_FACE_KPS = [f"face_{i}" for i in range(68)]  # indices 23-90

_FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
_JOINT_NAMES  = ["mcp", "pip", "dip", "tip"]

COCO_LEFT_HAND_KPS = ["left_wrist_hand"] + [
    f"left_{f}_{j}" for f in _FINGER_NAMES for j in _JOINT_NAMES
]

COCO_RIGHT_HAND_KPS = ["right_wrist_hand"] + [
    f"right_{f}_{j}" for f in _FINGER_NAMES for j in _JOINT_NAMES
]

NUM_BODY_KPS      = 17
NUM_FOOT_KPS      = 6
NUM_FACE_KPS      = 68
NUM_HAND_KPS      = 21
NUM_WHOLEBODY_KPS = 133

FOOT_START  = 17
FACE_START  = 23
LHAND_START = 91
RHAND_START = 112

# ---------------------------------------------------------------------------
# OpenPose 18-point body mapping from COCO 17-point
# ---------------------------------------------------------------------------

OPENPOSE_18_FROM_COCO = {
    0:  0,          # nose
    1:  (5, 6),     # neck = midpoint(left_shoulder, right_shoulder)
    2:  6,          # right_shoulder
    3:  8,          # right_elbow
    4:  10,         # right_wrist
    5:  5,          # left_shoulder
    6:  7,          # left_elbow
    7:  9,          # left_wrist
    8:  12,         # right_hip
    9:  14,         # right_knee
    10: 16,         # right_ankle
    11: 11,         # left_hip
    12: 13,         # left_knee
    13: 15,         # left_ankle
    14: 2,          # right_eye
    15: 1,          # left_eye
    16: 4,          # right_ear
    17: 3,          # left_ear
}

OPENPOSE_18_NAMES = [
    "nose",           # 0
    "neck",           # 1
    "right_shoulder", # 2
    "right_elbow",    # 3
    "right_wrist",    # 4
    "left_shoulder",  # 5
    "left_elbow",     # 6
    "left_wrist",     # 7
    "right_hip",      # 8
    "right_knee",     # 9
    "right_ankle",    # 10
    "left_hip",       # 11
    "left_knee",      # 12
    "left_ankle",     # 13
    "right_eye",      # 14
    "left_eye",       # 15
    "right_ear",      # 16
    "left_ear",       # 17
]

OPENPOSE_FOOT_FROM_COCO = {
    0: 17,  # left_big_toe
    1: 18,  # left_small_toe
    2: 19,  # left_heel
    3: 20,  # right_big_toe
    4: 21,  # right_small_toe
    5: 22,  # right_heel
}

# ---------------------------------------------------------------------------
# Limb connections - OpenPose 18-point indices
# ---------------------------------------------------------------------------
#
# BODY_LIMBS order matches controlnet_aux util.py limbSeq EXACTLY.
# The official sequence (1-indexed in source, converted to 0-indexed here):
#   [2,3],[2,6],[3,4],[4,5],[6,7],[7,8],[2,9],[9,10],[10,11],
#   [2,12],[12,13],[13,14],[2,1],[1,15],[15,17],[1,16],[16,18]
# Plus our extra pelvis (8,11) at the end using the 18th official color.
#
# This ordering is critical - DWPOSE_LIMB_COLORS is indexed in lockstep
# with BODY_LIMBS, and the official colors were assigned in this exact order.
# Inserting nose-neck at index 0 shifts all colors by one and causes mismatches.

BODY_LIMBS = [
    (1,  2),   # idx 0:  neck -> right_shoulder   [255,   0,   0] red
    (1,  5),   # idx 1:  neck -> left_shoulder    [255,  85,   0] orange
    (2,  3),   # idx 2:  right_shoulder -> right_elbow  [255, 170,   0]
    (3,  4),   # idx 3:  right_elbow -> right_wrist     [255, 255,   0] yellow
    (5,  6),   # idx 4:  left_shoulder -> left_elbow    [170, 255,   0]
    (6,  7),   # idx 5:  left_elbow -> left_wrist       [ 85, 255,   0]
    (1,  8),   # idx 6:  neck -> right_hip         [  0, 255,   0] green
    (8,  9),   # idx 7:  right_hip -> right_knee   [  0, 255,  85]
    (9,  10),  # idx 8:  right_knee -> right_ankle [  0, 255, 170]
    (1,  11),  # idx 9:  neck -> left_hip          [  0, 255, 255] cyan
    (11, 12),  # idx 10: left_hip -> left_knee     [  0, 170, 255]
    (12, 13),  # idx 11: left_knee -> left_ankle   [  0,  85, 255]
    (1,  0),   # idx 12: neck -> nose              [  0,   0, 255] blue
    (0,  14),  # idx 13: nose -> right_eye         [ 85,   0, 255]
    (14, 16),  # idx 14: right_eye -> right_ear    [170,   0, 255]
    (0,  15),  # idx 15: nose -> left_eye          [255,   0, 255]
    (15, 17),  # idx 16: left_eye -> left_ear      [255,   0, 170]
    (8,  11),  # idx 17: pelvis (our addition)     [255,   0,  85]
]

# Foot limbs (foot-local indices 0-5)
FOOT_LIMBS = [
    (2, 0),  # left_heel -> left_big_toe
    (2, 1),  # left_heel -> left_small_toe
    (5, 3),  # right_heel -> right_big_toe
    (5, 4),  # right_heel -> right_small_toe
]

# Hand limbs (21-pt hand-local indices)
# First 20 entries match official DWPose draw_handpose edges exactly.
# Knuckle bar (5,9),(9,13),(13,17) used in Enhanced mode only.
HAND_LIMBS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Knuckle bar (Enhanced mode only, not in official DWPose spec)
    (5, 9), (9, 13), (13, 17),
]

# ---------------------------------------------------------------------------
# Color schemes
# ---------------------------------------------------------------------------
#
# DWPOSE_LIMB_COLORS - indexed in lockstep with BODY_LIMBS above.
# Matches controlnet_aux util.py colors array exactly for indices 0-16.
# Index 17 (pelvis) uses the 18th official color [255,0,85].
#
# Official colors array from controlnet_aux:
#   [255,0,0],[255,85,0],[255,170,0],[255,255,0],[170,255,0],[85,255,0],
#   [0,255,0],[0,255,85],[0,255,170],[0,255,255],[0,170,255],[0,85,255],
#   [0,0,255],[85,0,255],[170,0,255],[255,0,255],[255,0,170],[255,0,85]

DWPOSE_LIMB_COLORS = [
    (255,   0,   0),  # 0  neck→rsh
    (255,  85,   0),  # 1  neck→lsh
    (255, 170,   0),  # 2  rsh→relbow
    (255, 255,   0),  # 3  relbow→rwrist
    (170, 255,   0),  # 4  lsh→lelbow
    ( 85, 255,   0),  # 5  lelbow→lwrist
    (  0, 255,   0),  # 6  neck→rhip
    (  0, 255,  85),  # 7  rhip→rknee
    (  0, 255, 170),  # 8  rknee→rankle
    (  0, 255, 255),  # 9  neck→lhip
    (  0, 170, 255),  # 10 lhip→lknee
    (  0,  85, 255),  # 11 lknee→lankle
    (  0,   0, 255),  # 12 neck→nose
    ( 85,   0, 255),  # 13 nose→reye
    (170,   0, 255),  # 14 reye→rear
    (255,   0, 255),  # 15 nose→leye
    (255,   0, 170),  # 16 leye→lear
    (255,   0,  85),  # 17 pelvis (our extra, 18th official color)
]

# Enhanced L/R warm/cool scheme - reordered to match new BODY_LIMBS order
ENHANCED_LIMB_COLORS = [
    (255, 100,  50),  # 0  neck→rsh          (right, warm)
    ( 50, 150, 255),  # 1  neck→lsh          (left, cool)
    (255,  80,   0),  # 2  rsh→relbow        (right)
    (255,  60,   0),  # 3  relbow→rwrist     (right)
    (  0, 120, 255),  # 4  lsh→lelbow        (left)
    (  0,  80, 255),  # 5  lelbow→lwrist     (left)
    (255, 200,   0),  # 6  neck→rhip         (right, warm)
    (255, 160,   0),  # 7  rhip→rknee        (right)
    (255, 120,   0),  # 8  rknee→rankle      (right)
    (  0, 220, 200),  # 9  neck→lhip         (left, cool)
    (  0, 180, 220),  # 10 lhip→lknee        (left)
    (  0, 140, 255),  # 11 lknee→lankle      (left)
    (255, 255, 255),  # 12 neck→nose         (center)
    (255, 100, 100),  # 13 nose→reye         (right, soft)
    (255,  80,  80),  # 14 reye→rear         (right)
    (100, 180, 255),  # 15 nose→leye         (left, soft)
    ( 80, 160, 255),  # 16 leye→lear         (left)
    (200, 200, 200),  # 17 pelvis            (center)
]

# Keypoint dot colors - one per OpenPose 18-point body keypoint index.
# Matches controlnet_aux colors array (same 18-color gradient, by keypoint index).
DWPOSE_KP_COLORS = [
    (255,   0,   0),  # 0  nose
    (255,  85,   0),  # 1  neck
    (255, 170,   0),  # 2  right_shoulder
    (255, 255,   0),  # 3  right_elbow
    (170, 255,   0),  # 4  right_wrist
    ( 85, 255,   0),  # 5  left_shoulder
    (  0, 255,   0),  # 6  left_elbow
    (  0, 255,  85),  # 7  left_wrist
    (  0, 255, 170),  # 8  right_hip
    (  0, 255, 255),  # 9  right_knee
    (  0, 170, 255),  # 10 right_ankle
    (  0,  85, 255),  # 11 left_hip
    (  0,   0, 255),  # 12 left_knee
    ( 85,   0, 255),  # 13 left_ankle
    (170,   0, 255),  # 14 right_eye
    (255,   0, 255),  # 15 left_eye
    (255,   0, 170),  # 16 right_ear
    (255,   0,  85),  # 17 left_ear
]

ENHANCED_KP_COLORS = [
    (255, 255, 255),  # 0  nose
    (255, 255, 255),  # 1  neck
    (255, 100,  50),  # 2  right_shoulder
    (255,  80,   0),  # 3  right_elbow
    (255,  60,   0),  # 4  right_wrist
    ( 50, 150, 255),  # 5  left_shoulder
    (  0, 120, 255),  # 6  left_elbow
    (  0,  80, 255),  # 7  left_wrist
    (255, 160,   0),  # 8  right_hip
    (255, 140,   0),  # 9  right_knee
    (255, 120,   0),  # 10 right_ankle
    (  0, 200, 220),  # 11 left_hip
    (  0, 180, 220),  # 12 left_knee
    (  0, 150, 255),  # 13 left_ankle
    (255, 100, 100),  # 14 right_eye
    (100, 180, 255),  # 15 left_eye
    (255,  80,  80),  # 16 right_ear
    ( 80, 160, 255),  # 17 left_ear
]

# Hand / foot / face colors
LEFT_HAND_COLOR  = (  0, 150, 255)
RIGHT_HAND_COLOR = (255, 100,   0)
FOOT_COLOR_LEFT  = (  0, 200, 200)
FOOT_COLOR_RIGHT = (255, 180,  50)
FACE_COLOR       = (200, 200, 200)
