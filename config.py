"""
config.py
─────────
Global constants, colour tables, and small pure-utility functions
shared across all modules.

Imported by: main.py, camera_widget.py, video_widget.py, recorder.py
"""

import os
import ctypes
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  TOP-LEVEL DEFAULTS  (overridden by CLI args)
# ─────────────────────────────────────────────────────────────────────────────
DETECTION_ON          = "Video"       # "Camera"  |  "Video"
MODEL_PATH            = "models/yolov8n_full_integer_quant.tflite"
VIDEO_PATH            =  r"C:\Users\india\Desktop\wifi-camera\videos\cam.mp4"#"videos/person_walking.mp4" # "http://192.168.1.52:4747/video"
FRAMERATE             = 30

RECORDING             = True
RECORDING_SAVE_DIR    = "recordings"
RECORDING_FPS         = 15            # FPS of the saved .avi file
RECORDING_QUALITY     = 75

FRAME_WIDTH           = 640
FRAME_HEIGHT          = 480

CONFIDENCE_TH         = 0.35
IOU_TH                = 0.45

# Change 1 — run inference once every INFER_EVERY_N_FRAMES frames
INFER_EVERY_N_FRAMES = 3

# ─────────────────────────────────────────────────────────────────────────────
#  ENCRYPTION
#  Key is shared across all recordings (AES-256, 32 bytes).
#  Generate once:
#      python -c "import os; os.makedirs('keys', exist_ok=True); \
#                 open('keys/master.key', 'wb').write(os.urandom(32))"
# ─────────────────────────────────────────────────────────────────────────────
ENCRYPTION_KEY_PATH  = "master.key"
ENCRYPTION           = False

# ─────────────────────────────────────────────────────────────────────────────
#  Per-class colours
#  Cairo (Camera mode): normalised float RGB
#  OpenCV (Video mode): uint8 BGR
# ─────────────────────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "Person":  (0.10, 0.95, 0.20),   # bright green
    "Vehicle": (0.10, 0.75, 1.00),   # cyan
}
CLASS_COLORS_BGR = {
    "Person":  (51,  242,  26),
    "Vehicle": (255, 191,  26),
}
_DEFAULT_COLOR     = (1.0, 1.0, 1.0)
_DEFAULT_COLOR_BGR = (255, 255, 255)

# Recording indicator
_REC_COLOR     = (1.0, 0.10, 0.10)   # Cairo
_REC_COLOR_BGR = (51,   26, 255)      # OpenCV BGR

# Recording indicator — size (shared by all 3 draw locations)
REC_DOT_RADIUS        = 10     # px   — circle radius (Cairo + OpenCV)
REC_TEXT_SCALE_CV     = 0.65   # OpenCV font scale  (cv2.FONT_HERSHEY_SIMPLEX)
REC_TEXT_THICKNESS_CV = 2      # OpenCV text stroke thickness
REC_TEXT_SIZE_CAIRO   = 18     # Cairo font size (pt)
REC_DOT_W_POS         = 25     # Width position of REC Dot
REC_DOT_H_POS         = 20     # Height position of REC Dot
REC_TEXT_W_POS        = 75     # Width position of REC Text
REC_TEXT_H_POS        = 27     # Height position of REC Text

# ─────────────────────────────────────────────────────────────────────────────
#  Shared cache for cv2.getTextSize — keyed by label string only.
#  Populated lazily at runtime; shared across Camera and Video modes.
# ─────────────────────────────────────────────────────────────────────────────
_LABEL_SIZE_CACHE: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def _model_label(model_path: str) -> str:
    """
    Derive a short display name from the model filename.
    e.g. 'models/yolov9t_full_integer_quant.nb' → 'YOLOv9t'
         'models/yolov8n_full_integer_quant.tflite' → 'YOLOv8n'
    """
    stem = Path(model_path).stem       # 'yolov9t_full_integer_quant'
    part = stem.split("_")[0]         # 'yolov9t'
    # Capitalise the version letter: yolov9t → YOLOv9t
    for i, c in enumerate(part):
        if c.isdigit():
            return part[:i].upper() + part[i:]
    return part.upper()


def _pin_thread(core: int):
    """Pin the calling thread to a CPU core (Linux / AArch64)."""
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        tid  = libc.syscall(178)       # SYS_gettid on AArch64
        os.sched_setaffinity(tid, {core})
        print(f"[CPU] Thread {tid} → core {core}")
    except Exception as exc:
        print(f"[CPU] Pin to core {core} failed: {exc}")
