import os

import numpy as np
import torch

# PATHS & GLOBAL CONSTANTS (RUNPOD-FRIENDLY)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")

MODELS_ROOT = "/workspace/models"

BALL_MODEL_PATH = os.path.join(MODELS_ROOT, "cricket_ball_detector.pt")
BAT_MODEL_PATH = os.path.join(MODELS_ROOT, "bestBat.pt")
VITPOSE_CKPT_PATH = os.path.join(MODELS_ROOT, "vitpose-b-multi-coco.pth")
LSTM_MODEL_PATH = os.path.join(MODELS_ROOT, "thirdlstm_shot_classifierupdated.keras")
REF_CSV_PATH = os.path.join(MODELS_ROOT, "1.csv")
LLM_MODEL_DIR = os.path.join(MODELS_ROOT, "cricket_t5_final_clean/cricket_t5_final_clean")
PERSON_MODEL_PATH = os.path.join(MODELS_ROOT, "yolov8n.pt")

PROCESSED_FOLDER = "/tmp/processed"
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Detection pipeline config
CROP_SIZE = 640
CONF_THRESH = 0.27
IOU = 0.5
CONTACT_RADIUS = 8
CONTACT_MIN_GAP = 24
PRE_FRAMES = 24
POST_FRAMES = 25
BALL_SEEN_FRAMES = 3
BALL_MISS_FRAMES = 2
LINGER_FRAMES = 5
START_FRAME = 2

# Classification constants
MERGE_PRE_FRAMES = 24
MERGE_POST_FRAMES = 25
MERGE_FRAMES_TOTAL = MERGE_PRE_FRAMES + 1 + MERGE_POST_FRAMES  # 50

THRESHOLD = 0.5
LABEL_CLASSES = np.array(
    [
        "cov",
        "covleft",
        "cut",
        "cutleft",
        "flick",
        "flickleft",
        "pull",
        "pullleft",
        "sweep",
        "sweepleft",
    ]
)
WINDOW_SIZE = 35
RIGHTY_CLASSES = {0, 2, 4, 6, 8}
LEFTY_CLASSES = {1, 3, 5, 7, 9}

BALL_RADIUS = 9
HIT_BOX_LENGTH_RATIO = 1.25
HIT_BOX_WIDTH_RATIO = 0.65
HIT_MIN_RATIO = 0.70

KEY_ORDER = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

ANGLES_ORDER = [
    "left_elbow",
    "right_elbow",
    "left_shoulder",
    "right_shoulder",
    "left_knee",
    "right_knee",
    "left_hip",
    "right_hip",
]

LSTM_KEY_ORDER = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
LSTM_EXPECTED_FEATURES = 24 + 8  # 12 joints*2 + 8 angles = 32


# =========================================================
# GLOBAL MODEL HANDLES
# =========================================================

ball_model = None
bat_model = None
yolo_model = None
vitpose = None
lstm_model = None
tokenizer = None
t5_model = None
