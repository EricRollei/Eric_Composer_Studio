"""
Eric_Composer_Studio - ComfyUI pose detection plus image, depth, pose, and
text-layer compositing nodes.
"""

from .nodes.detector      import PoseDetector
from .nodes.renderer      import PoseRenderer
from .nodes.transform     import PoseTransform
from .nodes.canvas_crop   import PoseCanvasCrop
from .nodes.editor        import PoseEditor
from .nodes.pose_composer import PoseComposer
from .nodes.save_pose     import SavePoseKeypoint
from .nodes.load_pose       import LoadPoseKeypoint
from .nodes.depth_composer import DepthComposer
from .nodes.image_composer import ImageComposer
from .nodes.text_layer     import TextLayer
from .nodes.poisson_blend  import EricPoissonBlend
from .core.model_manager    import print_model_status

NODE_CLASS_MAPPINGS = {
    "EricPoseDetector":    PoseDetector,
    "EricPoseRenderer":    PoseRenderer,
    "EricPoseTransform":   PoseTransform,
    "EricPoseCanvasCrop":  PoseCanvasCrop,
    "EricPoseEditor":      PoseEditor,
    "EricPoseComposer":    PoseComposer,
    "EricSavePoseKeypoint": SavePoseKeypoint,
    "EricLoadPoseKeypoint": LoadPoseKeypoint,
    "EricDepthComposer":    DepthComposer,
    "EricImageComposer":    ImageComposer,
    "EricTextLayer":        TextLayer,
    "EricPoissonBlend":     EricPoissonBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EricPoseDetector":    "Pose Detector (RTMW/DWPose)",
    "EricPoseRenderer":    "Pose Renderer",
    "EricPoseTransform":   "Pose Transform (uniform)",
    "EricPoseCanvasCrop":  "Pose Canvas Crop/Fit",
    "EricPoseEditor":      "Pose Editor",
    "EricPoseComposer":    "Pose Composer",
    "EricSavePoseKeypoint": "Save Pose Keypoint",
    "EricLoadPoseKeypoint": "Load Pose Keypoint",
    "EricDepthComposer":    "Depth Composer",
    "EricImageComposer":    "Image Composer",
    "EricTextLayer":        "Text Layer (for Image Composer)",
    "EricPoissonBlend":     "Poisson Blend (Seamless Clone)",
}

WEB_DIRECTORY = "./web"

try:
    print_model_status()
except Exception:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
