# helper.py
import os
import cv2
import numpy as np
import tensorflow as tf
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from ultralytics import YOLO

import constants

from vitpose.configs.ViTPose_base_coco_256x192 import model as model_cfg
from vitpose.models.model import ViTPose


def _require_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found at: {path}")


def load_models():
    """
    Load ALL models once at worker startup.
    """
    global constants.ball_model, constants.bat_model, constants.yolo_model
    global constants.vitpose, constants.lstm_model, constants.tokenizer, constants.t5_model

    print("🔄 Loading models (fail-fast)...")

    _require_file(constants.BALL_MODEL_PATH, "BALL_MODEL (.pt)")
    _require_file(constants.BAT_MODEL_PATH, "BAT_MODEL (.pt)")
    _require_file(constants.PERSON_MODEL_PATH, "PERSON_MODEL yolov8n (.pt)")
    _require_file(constants.VITPOSE_CKPT_PATH, "ViTPose checkpoint (.pth)")
    _require_file(constants.LSTM_MODEL_PATH, "LSTM model (.keras)")
    _require_file(constants.REF_CSV_PATH, "Reference CSV (1.csv)")
    _require_file(
        os.path.join(constants.LLM_MODEL_DIR, "config.json"), "T5 model dir (config.json)"
    )

    # YOLO models
    constants.ball_model = YOLO(constants.BALL_MODEL_PATH)
    constants.bat_model = YOLO(constants.BAT_MODEL_PATH)
    constants.yolo_model = YOLO(constants.PERSON_MODEL_PATH)
    print("✅ YOLO models loaded")

    # ViTPose
    ckpt = torch.load(constants.VITPOSE_CKPT_PATH, map_location=constants.DEVICE)
    state = ckpt.get("state_dict", ckpt)
    vp = ViTPose(model_cfg).to(constants.DEVICE).eval()
    vp.load_state_dict(state, strict=False)
    constants.vitpose = vp
    print("✅ ViTPose loaded")

    # LSTM
    constants.lstm_model = tf.keras.models.load_model(constants.LSTM_MODEL_PATH)
    print("✅ LSTM loaded")

    # T5 (offline)
    constants.tokenizer = AutoTokenizer.from_pretrained(constants.LLM_MODEL_DIR, local_files_only=True)
    constants.t5_model = (
        AutoModelForSeq2SeqLM.from_pretrained(constants.LLM_MODEL_DIR, local_files_only=True)
        .to(constants.DEVICE)
        .eval()
    )
    print("✅ T5 loaded")

    print("✅ Model loading done.")


def adaptive_square_crop(frame, target_size=None):
    """Crop frame to square and resize to target_size."""
    if target_size is None:
        target_size = constants.CROP_SIZE
    h, w = frame.shape[:2]
    size = min(h, w)
    x1 = (w - size) // 2
    y1 = (h - size) // 2
    cropped = frame[y1: y1 + size, x1: x1 + size]
    return cv2.resize(cropped, (target_size, target_size))


def polygon_centroid(pts):
    a = np.array(pts, dtype=float)
    return float(a[:, 0].mean()), float(a[:, 1].mean())


def translate_polygon(pts, dx, dy):
    return [[int(round(x + dx)), int(round(y + dy))] for x, y in pts]
