import os
import io
import time
import json
import glob
import zipfile
import shutil
import math
import subprocess
import threading
from itertools import combinations
from collections import deque

import cv2
import numpy as np
import torch
import tensorflow as tf
import pandas as pd
from PIL import Image

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from torchvision.transforms import transforms
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import sys

# ========================= ViTPose imports =========================
# Make sure ViTPose_pytorch is cloned to /workspace/ViTPose_pytorch in Docker
sys.path.append("/workspace/ViTPose_pytorch")
from models.model import ViTPose
from utils.top_down_eval import keypoints_from_heatmaps
from configs.ViTPose_base_coco_256x192 import model as model_cfg, data_cfg


# =========================================================
# PATHS & GLOBAL CONSTANTS (RUNPOD-FRIENDLY)
# =========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")

MODELS_ROOT = "/workspace/models"

BALL_MODEL_PATH = os.path.join(MODELS_ROOT, "cricket_ball_detector.pt")
BAT_MODEL_PATH = os.path.join(MODELS_ROOT, "bestBat.pt")
VITPOSE_CKPT_PATH = os.path.join(MODELS_ROOT, "vitpose-b-multi-coco.pth")
LSTM_MODEL_PATH = os.path.join(MODELS_ROOT, "thirdlstm_shot_classifierupdated.keras")
REF_CSV_PATH = os.path.join(MODELS_ROOT, "1.csv")
LLM_MODEL_DIR = os.path.join(MODELS_ROOT, "cricket_t5_final_clean")

PROCESSED_FOLDER = "/workspace/processed"
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
LABEL_CLASSES = np.array(["cov","covleft","cut","cutleft","flick","flickleft","pull","pullleft","sweep","sweepleft"])
WINDOW_SIZE = 35
RIGHTY_CLASSES = {0, 2, 4, 6, 8}
LEFTY_CLASSES  = {1, 3, 5, 7, 9}

BALL_RADIUS = 9
HIT_BOX_LENGTH_RATIO = 1.25
HIT_BOX_WIDTH_RATIO  = 0.65
HIT_MIN_RATIO = 0.70

KEY_ORDER = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder",
    "left_elbow","right_elbow",
    "left_wrist","right_wrist",
    "left_hip","right_hip",
    "left_knee","right_knee",
    "left_ankle","right_ankle"
]

ANGLES_ORDER = [
    "left_elbow","right_elbow","left_shoulder","right_shoulder",
    "left_knee","right_knee","left_hip","right_hip"
]

LSTM_KEY_ORDER = [
    "left_shoulder","right_shoulder",
    "left_elbow","right_elbow",
    "left_wrist","right_wrist",
    "left_hip","right_hip",
    "left_knee","right_knee",
    "left_ankle","right_ankle",
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


def load_models():
    """
    Load all models once per worker at startup.
    """
    global ball_model, bat_model, yolo_model, vitpose, lstm_model, tokenizer, t5_model

    print("🔄 Loading models...")

    # YOLO ball model
    try:
        if not os.path.exists(BALL_MODEL_PATH):
            raise FileNotFoundError(f"{BALL_MODEL_PATH} not found")
        ball_model = YOLO(BALL_MODEL_PATH)
        print("✅ BALL_MODEL loaded")
    except Exception as e:
        print("❌ BALL_MODEL failed:", e)

    # YOLO bat model
    try:
        if not os.path.exists(BAT_MODEL_PATH):
            raise FileNotFoundError(f"{BAT_MODEL_PATH} not found")
        bat_model = YOLO(BAT_MODEL_PATH)
        print("✅ BAT_MODEL loaded")
    except Exception as e:
        print("❌ BAT_MODEL failed:", e)

    # Person detection YOLO
    try:
        yolo_model = YOLO("yolov8n.pt")
        print("✅ YOLOv8n loaded")
    except Exception as e:
        print("❌ YOLOv8n load failed:", e)

    # ViTPose
    try:
        ckpt = torch.load(VITPOSE_CKPT_PATH, map_location=DEVICE)
        state = ckpt.get("state_dict", ckpt)
        vp = ViTPose(model_cfg).to(DEVICE).eval()
        vp.load_state_dict(state, strict=False)
        vitpose = vp
        print("✅ ViTPose loaded")
    except Exception as e:
        print("❌ ViTPose load failed:", e)

    # LSTM
    try:
        lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
        print("✅ LSTM model loaded")
    except Exception as e:
        print("❌ LSTM model load failed:", e)

    # T5 LLM
    try:
        tokenizer_local = AutoTokenizer.from_pretrained(LLM_MODEL_DIR)
        t5_local = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_DIR).to(DEVICE).eval()
        tokenizer, t5_model = tokenizer_local, t5_local
        print("✅ T5 tokenizer & model loaded")
    except Exception as e:
        print("❌ T5 model load failed:", e)

    print("✅ Model loading done.")


# =========================================================
# FLASK APP INIT
# =========================================================

app = Flask(__name__)
CORS(app)


# =========================================================
# SMALL HELPERS
# =========================================================

def adaptive_square_crop(frame, target_size=CROP_SIZE):
    """Crop frame to square and resize to target_size."""
    h, w = frame.shape[:2]
    size = min(h, w)
    x1 = (w - size) // 2
    y1 = (h - size) // 2
    cropped = frame[y1:y1+size, x1:x1+size]
    return cv2.resize(cropped, (target_size, target_size))


def polygon_centroid(pts):
    a = np.array(pts, dtype=float)
    return float(a[:, 0].mean()), float(a[:, 1].mean())


def translate_polygon(pts, dx, dy):
    return [[int(round(x + dx)), int(round(y + dy))] for x, y in pts]


def schedule_folder_cleanup(folder_path, delay_seconds=300):
    """
    Delete a folder after delay_seconds in a background thread.
    Called after response is fully sent.
    """
    def _cleanup():
        try:
            time.sleep(delay_seconds)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path, ignore_errors=True)
                print(f"🧹 Deleted folder: {folder_path}")
        except Exception as e:
            print(f"⚠ Cleanup error on {folder_path}: {e}")
    threading.Thread(target=_cleanup, daemon=True).start()


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models": {
            "ball_model": ball_model is not None,
            "bat_model": bat_model is not None,
            "yolo_model": yolo_model is not None,
            "vitpose": vitpose is not None,
            "lstm_model": lstm_model is not None,
            "t5_model": t5_model is not None,
        }
    })


# =========================================================
# /process ROUTE (VIDEO PROCESSING + HIGHLIGHT ZIP)
# =========================================================

@app.route("/process", methods=["POST"])
def process_video():
    # ---- Input checks ----
    video_path = None

    if "video_zip" in request.files:
        zf = request.files["video_zip"]
        filename = zf.filename or f"video_{int(time.time())}.zip"
        temp_zip = os.path.join("/tmp", filename)
        zf.save(temp_zip)

        with zipfile.ZipFile(temp_zip, "r") as zip_ref:
            zip_ref.extractall("/tmp")
            inner_files = zip_ref.namelist()
            if not inner_files:
                return jsonify({"error": "empty_zip"}), 400
            inner_name = inner_files[0]
            video_path = os.path.join("/tmp", inner_name)
            print(f"✅ Unzipped video from ZIP: {inner_name}")

    elif "video" in request.files:
        video = request.files["video"]
        filename = video.filename or f"upload_{int(time.time())}.mp4"
        video_path = os.path.join("/tmp", filename)
        video.save(video_path)
    else:
        return jsonify({"error": "no_video_uploaded"}), 400

    if ball_model is None:
        return jsonify({"error": "ball_model_not_loaded"}), 500
    if bat_model is None:
        return jsonify({"error": "bat_model_not_loaded"}), 500

    # ---- Paths / job setup ----
    filename_noext = os.path.splitext(os.path.basename(video_path))[0]
    job_id = f"{filename_noext}_{int(time.time())}"
    job_folder = os.path.join(PROCESSED_FOLDER, job_id)
    os.makedirs(job_folder, exist_ok=True)

    final_video_path = os.path.join(job_folder, os.path.basename(video_path))
    os.rename(video_path, final_video_path)

    print(f"🚀 Processing video: {filename_noext} -> Job ID: {job_id}")

    CONTACT_FRAMES_ROOT = os.path.join(job_folder, "frames")
    HIGHLIGHT_OUT_PATH = os.path.join(job_folder, "highlight.mp4")
    os.makedirs(CONTACT_FRAMES_ROOT, exist_ok=True)

    cap = cv2.VideoCapture(final_video_path)
    if not cap.isOpened():
        return jsonify({"error": "could_not_open_video"}), 500

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    crop_size_px = min(orig_h, orig_w)
    crop_x1 = (orig_w - crop_size_px) // 2
    crop_y1 = (orig_h - crop_size_px) // 2
    scale_from_640_to_crop = crop_size_px / float(CROP_SIZE)

    last_ball = deque(maxlen=2)
    last_bat  = deque(maxlen=2)
    contacts = []
    last_contact_frame = -9999
    skip_until = -1

    ball_visible_frames = 0
    ball_missing_frames = 0
    linger_counter = 0
    ball_active = False

    frame_buffer = deque(maxlen=PRE_FRAMES)
    post_frames_left = 0
    highlight_writer = None
    last_written_idx = -1
    written_frames = 0

    # OpenCV VideoWriter
    for codec in ['avc1', 'mp4v', 'XVID', 'H264']:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer_try = cv2.VideoWriter(HIGHLIGHT_OUT_PATH, fourcc, fps, (orig_w, orig_h))
        if writer_try.isOpened():
            highlight_writer = writer_try
            print(f"[INFO] Highlight writer opened with codec '{codec}'")
            break
        writer_try.release()
    if highlight_writer is None:
        print("[WARN] Could not open any VideoWriter codec. Highlight skipped.")

    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_buffer.append((frame_idx, frame.copy()))

            if frame_idx > last_contact_frame and frame_idx <= skip_until:
                if post_frames_left > 0 and highlight_writer:
                    highlight_writer.write(frame)
                    post_frames_left -= 1
                    last_written_idx = frame_idx
                    written_frames += 1
                frame_idx += 1
                continue

            cropped = adaptive_square_crop(frame)
            balls_current, bats_current = [], []

            # BALL DETECTION
            try:
                ball_results = ball_model(cropped, conf=CONF_THRESH, iou=IOU, classes=[0])
                if ball_results and len(ball_results) > 0:
                    r0 = ball_results[0]
                    if hasattr(r0, 'boxes') and r0.boxes is not None and len(r0.boxes) > 0:
                        boxes_xyxy = r0.boxes.xyxy.cpu().numpy()
                        confs = r0.boxes.conf.cpu().numpy()
                        for (x1, y1, x2, y2), conf in zip(boxes_xyxy, confs):
                            cx = int(round((x1 + x2) / 2.0))
                            cy = int(round((y1 + y2) / 2.0))
                            balls_current.append((cx, cy, float(conf)))
            except Exception as e:
                print(f"[WARN] Ball detection failed at frame {frame_idx}: {e}")

            # UPDATE BALL STATE
            if balls_current:
                ball_visible_frames += 1
                ball_missing_frames = 0
            else:
                ball_missing_frames += 1
                ball_visible_frames = max(0, ball_visible_frames - 1)

            if ball_visible_frames >= BALL_SEEN_FRAMES:
                ball_active = True
                linger_counter = LINGER_FRAMES
            elif ball_missing_frames >= BALL_MISS_FRAMES:
                if linger_counter > 0:
                    linger_counter -= 1
                    ball_active = True
                else:
                    ball_active = False

            # BAT DETECTION
            if ball_active:
                try:
                    bat_results = bat_model.predict(source=cropped, imgsz=CROP_SIZE, conf=CONF_THRESH, verbose=False)
                    if bat_results:
                        for r in bat_results:
                            obb_attr = getattr(r, 'obb', None)
                            if obb_attr is not None and getattr(obb_attr, 'xyxyxyxy', None) is not None:
                                obb_boxes = obb_attr.xyxyxyxy.cpu().numpy()
                                obb_confs = obb_attr.conf.cpu().numpy()
                                for box_flat, conf in zip(obb_boxes, obb_confs):
                                    pts = box_flat.reshape(4, 2).astype(int).tolist()
                                    bats_current.append((pts, float(conf)))
                            elif hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                                boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                                confs = r.boxes.conf.cpu().numpy()
                                for (x1, y1, x2, y2), conf in zip(boxes_xyxy, confs):
                                    pts = [[int(x1), int(y1)], [int(x2), int(y1)],
                                           [int(x2), int(y2)], [int(x1), int(y2)]]
                                    bats_current.append((pts, float(conf)))
                except Exception as e:
                    print(f"[WARN] Bat detection failed at frame {frame_idx}: {e}")

            # PREDICTIVE BALL
            if not balls_current:
                if len(last_ball) >= 2:
                    (x1, y1, c1, f1), (x2, y2, c2, f2) = last_ball[0], last_ball[1]
                    dx, dy = x2 - x1, y2 - y1
                    pred_x, pred_y = int(round(x2 + dx)), int(round(y2 + dy))
                    pred_conf = float(c2) * 0.8
                    if pred_conf >= CONF_THRESH:
                        balls_current.append((pred_x, pred_y, pred_conf))
                elif len(last_ball) == 1:
                    (x, y, c, f) = last_ball[-1]
                    pred_conf = float(c) * 0.9
                    if pred_conf >= CONF_THRESH:
                        balls_current.append((int(x), int(y), pred_conf))

            # PREDICTIVE BAT
            if not bats_current and len(last_bat) > 0:
                if len(last_bat) >= 2:
                    (pts1, conf1, f1), (pts2, conf2, f2) = last_bat[0], last_bat[1]
                    cx1, cy1 = polygon_centroid(pts1)
                    cx2, cy2 = polygon_centroid(pts2)
                    dx, dy = cx2 - cx1, cy2 - cy1
                    pred_pts = translate_polygon(pts2, dx, dy)
                    pred_conf = float(conf2) * 0.8
                    if pred_conf >= CONF_THRESH:
                        bats_current.append((pred_pts, pred_conf))
                else:
                    pts, conf, f = last_bat[-1]
                    pred_conf = float(conf) * 0.9
                    if pred_conf >= CONF_THRESH:
                        bats_current.append((pts, pred_conf))

            # CONTACT DETECTION
            contact_found = False
            contact_ball = None
            contact_bat = None
            if balls_current and bats_current:
                for (cx, cy, bconf) in balls_current:
                    ball_area = Point(cx, cy).buffer(CONTACT_RADIUS)
                    for (pts, bat_conf) in bats_current:
                        poly = Polygon(pts)
                        if poly.is_valid and ball_area.intersects(poly):
                            contact_found = True
                            contact_ball, contact_bat = (cx, cy, float(bconf)), (pts, float(bat_conf))
                            break
                    if contact_found:
                        break

            # HANDLE CONTACT
            if contact_found and frame_idx > last_contact_frame + CONTACT_MIN_GAP:
                ann = cropped.copy()
                if contact_bat:
                    pts, conf = contact_bat
                    cv2.polylines(ann, [np.array(pts, np.int32)], True, (0, 255, 0), 2)
                    cv2.putText(ann, f"{conf:.2f}", (pts[0][0], max(0, pts[0][1] - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                if contact_ball:
                    cx, cy, bconf = contact_ball
                    cv2.circle(ann, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(ann, f"{bconf:.2f}", (cx + 6, cy - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

                fname = os.path.join(CONTACT_FRAMES_ROOT, f"contact_{frame_idx:06d}.jpg")
                cv2.imwrite(fname, ann)

                def map_to_original(x640, y640):
                    x_in_crop = int(round(x640 * scale_from_640_to_crop))
                    y_in_crop = int(round(y640 * scale_from_640_to_crop))
                    return int(crop_x1 + x_in_crop), int(crop_y1 + y_in_crop)

                mapped_ball = None
                if contact_ball:
                    bx, by, bconf = contact_ball
                    bx_o, by_o = map_to_original(bx, by)
                    mapped_ball = {"x_640": bx, "y_640": by, "conf": bconf,
                                   "x_orig": bx_o, "y_orig": by_o}

                mapped_bat = None
                if contact_bat:
                    pts640, batconf = contact_bat
                    pts_orig = [[map_to_original(px, py)[0], map_to_original(px, py)[1]] for (px, py) in pts640]
                    mapped_bat = {"pts_640": pts640, "conf": batconf, "pts_orig": pts_orig}

                frame_highlight = (len(contacts)) * (PRE_FRAMES + POST_FRAMES + 1) + PRE_FRAMES if highlight_writer else None

                contacts.append({
                    "frame_idx": frame_idx,
                    "frame_highlight": frame_highlight,
                    "ball": mapped_ball,
                    "bat": mapped_bat,
                    "video_dimensions": {"width": orig_w, "height": orig_h}
                })

                if highlight_writer:
                    for idx, buf_frame in list(frame_buffer):
                        if idx <= last_written_idx:
                            continue
                        highlight_writer.write(buf_frame)
                        last_written_idx = idx
                        written_frames += 1
                    highlight_writer.write(frame)
                    last_written_idx = frame_idx
                    written_frames += 1
                    post_frames_left = POST_FRAMES

                last_contact_frame = frame_idx
                skip_until = frame_idx + CONTACT_MIN_GAP

            if post_frames_left > 0 and highlight_writer:
                highlight_writer.write(frame)
                last_written_idx = frame_idx
                written_frames += 1
                post_frames_left -= 1

            if balls_current:
                bx, by, bconf = balls_current[0]
                last_ball.append((bx, by, bconf, frame_idx))
            if bats_current:
                pts, bconf = bats_current[0]
                last_bat.append((pts, bconf, frame_idx))

            frame_idx += 1

    finally:
        cap.release()
        if highlight_writer:
            highlight_writer.release()

    # SAVE CONTACT INFO
    json_path = os.path.join(CONTACT_FRAMES_ROOT, "contact_info.json")
    with open(json_path, "w") as jf:
        json.dump(contacts, jf, indent=2)

    # CONVERT HIGHLIGHT
    highlight_fixed_path = os.path.join(job_folder, "highlight_web.mp4")
    if os.path.exists(HIGHLIGHT_OUT_PATH):
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", HIGHLIGHT_OUT_PATH,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-movflags", "+faststart",
            highlight_fixed_path
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True)
        except Exception as e:
            print(f"[WARN] ffmpeg re-encode failed: {e}")
            highlight_fixed_path = None
    else:
        highlight_fixed_path = None

    # CREATE ZIP
    zip_path = os.path.join(job_folder, "process.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        if highlight_fixed_path and os.path.exists(highlight_fixed_path):
            z.write(highlight_fixed_path, "highlight.mp4")
        elif os.path.exists(HIGHLIGHT_OUT_PATH):
            z.write(HIGHLIGHT_OUT_PATH, "highlight.mp4")
        if os.path.exists(json_path):
            z.write(json_path, "contact_info.json")

    print(f"✅ process.zip ready: {zip_path}")

    # Build response and cleanup AFTER response close (with delay)
    response = send_file(zip_path, as_attachment=True, mimetype="application/zip")

    @response.call_on_close
    def _cleanup():
        # delete job folder 5 minutes after response closed
        schedule_folder_cleanup(job_folder, delay_seconds=300)

    return response


# =========================================================
# CLASSIFICATION UTILITIES (same logic as your cell)
# =========================================================

def format_input(
    shot_type,
    confidence,
    bat_angle,
    hit_quality,
    face_stability,
    stance_bend,
    front_foot_stance_score,
    shot_accuracy,
):
    return (
        f"shot_type={shot_type} "
        f"confidence={confidence} "
        f"bat_angle={bat_angle} "
        f"hit_quality={hit_quality} "
        f"face_stability={face_stability} "
        f"stance_bend={stance_bend} "
        f"front_foot_stance_score={front_foot_stance_score} "
        f"shot_accuracy={shot_accuracy}"
    )


def clean_feedback(text: str) -> str:
    sentences = []
    for s in text.split("."):
        s = s.strip()
        if s and s not in sentences:
            sentences.append(s)
    return ". ".join(sentences) + "."


def generate_feedback(**features):
    if tokenizer is None or t5_model is None:
        return None
    text = format_input(**features)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = t5_model.generate(
            **inputs,
            max_new_tokens=160,
            num_beams=8,
            no_repeat_ngram_size=4,
            repetition_penalty=1.2,
            length_penalty=1.0,
            early_stopping=True,
            do_sample=False,
        )
    raw = tokenizer.decode(output[0], skip_special_tokens=True)
    return clean_feedback(raw)


def angle_360(a, b, c):
    v1 = np.array([a[0]-b[0], a[1]-b[1]])
    v2 = np.array([c[0]-b[0], c[1]-b[1]])
    if np.linalg.norm(v1)==0 or np.linalg.norm(v2)==0:
        return 0.0
    dot = np.dot(v1, v2)
    det = v1[0]*v2[1] - v1[1]*v2[0]
    ang = np.degrees(np.arctan2(det, dot))
    if ang < 0: ang += 360
    return ang


def euclid(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def safe_acos_angle(v1, v2):
    na = math.hypot(v1[0], v1[1]); nb = math.hypot(v2[0], v2[1])
    if na == 0 or nb == 0: return None
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    cosv = dot / (na*nb); cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(np.arccos(cosv))


def normalize_by_video_dims(pt, video_w, video_h):
    return [pt[0]/float(video_w), pt[1]/float(video_h)]


def compute_imaginary_box(bat_pts, wrist_pt, length_ratio=1.0, width_ratio=0.5):
    dists = [(pt, euclid(pt, wrist_pt)) for pt in bat_pts]
    dists.sort(key=lambda x: x[1], reverse=True)
    base_pts = [np.array(dists[0][0]), np.array(dists[1][0])]
    mid = (base_pts[0] + base_pts[1]) / 2.0
    axis_vec = base_pts[1] - base_pts[0]
    axis_len = np.linalg.norm(axis_vec) or 1.0
    axis_unit = axis_vec / axis_len
    perp_unit = np.array([-axis_unit[1], axis_unit[0]])
    half_width = axis_len * width_ratio
    p1 = mid + perp_unit*half_width
    p2 = mid - perp_unit*half_width
    p3 = p2 + axis_vec*length_ratio
    p4 = p1 + axis_vec*length_ratio
    return [p1.tolist(), p2.tolist(), p3.tolist(), p4.tolist()]


def compute_hit_quality(ball_center, ball_radius, box_pts):
    ball = Point(ball_center[0], ball_center[1]).buffer(ball_radius)
    box_poly = Polygon(box_pts)
    if not box_poly.is_valid: box_poly = box_poly.buffer(0)
    intersection_area = ball.intersection(box_poly).area
    ball_area = ball.area
    ratio = intersection_area / ball_area if ball_area > 0 else 0
    if ratio > 0:
        ratio = max(ratio, HIT_MIN_RATIO)
    return min(ratio, 1.0)


def _valid_point(p):
    return isinstance(p, (list, tuple)) and len(p) == 2 and (p[0] != 0 or p[1] != 0)


def _collect_prev_valid_frames(pose_cache, ref_frame, n=10):
    frames = sorted([f for f in pose_cache.keys() if f <= ref_frame])[::-1]
    out = []
    for f in frames:
        kp = pose_cache.get(f)
        if kp and any(_valid_point(k) for k in kp):
            out.append((f, kp))
        if len(out) >= n:
            break
    return out[::-1]


def face_stability_from_prev(pose_cache, ref_frame):
    idx_left_eye = KEY_ORDER.index("left_eye")
    idx_right_eye = KEY_ORDER.index("right_eye")
    collected = _collect_prev_valid_frames(pose_cache, ref_frame, n=10)
    vals = []
    for _, kp in collected:
        if idx_left_eye < len(kp) and idx_right_eye < len(kp):
            le = kp[idx_left_eye]; re = kp[idx_right_eye]
            if _valid_point(le) and _valid_point(re):
                dx = re[0]-le[0]; dy = re[1]-le[1]
                angle = math.degrees(math.atan2(dy, dx))
                stability = 100.0 if abs(angle) < 15 else (math.cos(math.radians(angle))**0.3) * 100
                vals.append(stability)
    return round(float(np.mean(vals)), 1) if vals else 0.0


def knee_bend_percentage(hip, knee, ankle):
    v1 = [hip[0]-knee[0], hip[1]-knee[1]]
    v2 = [ankle[0]-knee[0], ankle[1]-knee[1]]
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.hypot(v1[0], v1[1]); mag2 = math.hypot(v2[0], v2[1])
    if mag1 == 0 or mag2 == 0: return 0.0
    cos_angle = max(-1.0, min(1.0, dot/(mag1*mag2)))
    angle_deg = math.degrees(math.acos(cos_angle))
    optimal = 120; deviation = abs(angle_deg - optimal)
    score = max(0, 100 - deviation*1.2)
    return round(score, 1)


def stance_bend_from_prev(pose_cache, ref_frame, player_type):
    idx = {k: KEY_ORDER.index(k) for k in ["left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]}
    collected = _collect_prev_valid_frames(pose_cache, ref_frame, n=10)
    vals = []
    for _, kp in collected:
        if player_type == "righty":
            hip, knee, ankle = kp[idx["right_hip"]], kp[idx["right_knee"]], kp[idx["right_ankle"]]
        else:
            hip, knee, ankle = kp[idx["left_hip"]], kp[idx["left_knee"]], kp[idx["left_ankle"]]
        if _valid_point(hip) and _valid_point(knee) and _valid_point(ankle):
            vals.append(knee_bend_percentage(hip, knee, ankle))
    return round(float(np.mean(vals)), 1) if vals else 0.0


def _front_foot_score_from_kp(kp):
    try:
        idx_la = KEY_ORDER.index("left_ankle")
        idx_ra = KEY_ORDER.index("right_ankle")
    except ValueError:
        return None
    if not kp or idx_la >= len(kp) or idx_ra >= len(kp):
        return None
    la, ra = kp[idx_la], kp[idx_ra]
    if not (_valid_point(la) and _valid_point(ra)):
        return None
    dist = abs(float(la[0]) - float(ra[0]))
    if dist >= 0.05:
        score = 0.0
    elif dist >= 0.0269:
        score = round(60 - (dist - 0.0269) / (0.05 - 0.0269) * 60, 1)
    else:
        score = round(100 - dist / 0.0269 * 40, 1)
    return score


def front_foot_stance_from_prev10(pose_cache, ref_frame, search_back=10, radius=3):
    if ref_frame is None:
        return 0.0
    target = max(0, ref_frame - search_back)
    start = max(0, target - radius)
    end   = max(0, target + radius)

    candidates = []
    for f in range(start, min(end + 1, ref_frame)):
        kp = pose_cache.get(f)
        s = _front_foot_score_from_kp(kp)
        if s is not None:
            candidates.append(s)
    if candidates:
        return max(candidates)

    fallback = []
    for f in range(0, max(0, ref_frame)):
        kp = pose_cache.get(f)
        s = _front_foot_score_from_kp(kp)
        if s is not None:
            fallback.append(s)
    if fallback:
        return max(fallback)

    return 0.0


_LSTM_JOINT_IDX = {j: i for i, j in enumerate(LSTM_KEY_ORDER)}

def _xy_indices_for_joint(joint_name):
    j = _LSTM_JOINT_IDX.get(joint_name, None)
    if j is None: return None, None
    base = 2 * j
    return base, base + 1


_vel_re = re.compile(r"^(?P<joint>.+)_(?P<ax>x|y)_vel$")
_acc_re = re.compile(r"^(?P<joint>.+)_(?P<ax>x|y)_acc$")
_speed_re = re.compile(r"^(?P<joint>.+)_speed$")
_ang_re = re.compile(r"^ang_(?P<aname>.+)$")
_angvel_re = re.compile(r"^ang_(?P<aname>.+)_vel$")
_stat_re = re.compile(r"^(?P<base>.+)_(?P<stat>mean|std|last)$")


def _finite_diff(arr):
    if arr.size < 2: return np.array([0.0], dtype=float)
    return np.diff(arr, axis=0).astype(float)


def _stat(series, stat):
    if stat == "std": return float(np.nanstd(series)) if series.size else 0.0
    if stat == "last": return float(series[-1]) if series.size else 0.0
    return float(np.nanmean(series)) if series.size else 0.0


def _joint_series(window, joint):
    xi, yi = _xy_indices_for_joint(joint)
    if xi is None: return np.zeros((window.shape[0],)), np.zeros((window.shape[0],))
    return window[:, xi], window[:, yi]


def _angle_series(window, aname):
    try: ai = 24 + ANGLES_ORDER.index(aname)
    except ValueError: ai = None
    if ai is None: return np.zeros((window.shape[0],))
    return window[:, ai]


def _build_user_row_from_window(window, ref_cols):
    out = {}
    joint_xy_cache = {j: _joint_series(window, j) for j in LSTM_KEY_ORDER}
    angle_cache = {a: _angle_series(window, a) for a in ANGLES_ORDER}
    for col in ref_cols:
        stat = None; base_name = col
        mstat = _stat_re.match(col)
        if mstat: base_name = mstat.group("base"); stat = mstat.group("stat")
        m = _vel_re.match(base_name)
        if m:
            j, ax = m.group("joint"), m.group("ax")
            x, y = joint_xy_cache.get(j, (np.zeros(1), np.zeros(1)))
            series = x if ax == "x" else y
            out[col] = _stat(_finite_diff(series), stat); continue
        m = _acc_re.match(base_name)
        if m:
            j, ax = m.group("joint"), m.group("ax")
            x, y = joint_xy_cache.get(j, (np.zeros(1), np.zeros(1)))
            v = _finite_diff(x if ax == "x" else y); a = _finite_diff(v)
            out[col] = _stat(a, stat); continue
        m = _speed_re.match(base_name)
        if m:
            j = m.group("joint"); x, y = joint_xy_cache.get(j, (np.zeros(1), np.zeros(1)))
            sp = np.sqrt(_finite_diff(x)**2 + _finite_diff(y)**2)
            out[col] = _stat(sp, stat); continue
        if _ang_re.match(base_name) and not _angvel_re.match(base_name):
            an = base_name[4:]; a = angle_cache.get(an, np.zeros(1))
            out[col] = _stat(a, stat); continue
        m = _angvel_re.match(base_name)
        if m:
            an = m.group("aname"); a = angle_cache.get(an, np.zeros(1))
            out[col] = _stat(_finite_diff(a), stat); continue
        if base_name.endswith("_x") or base_name.endswith("_y"):
            j, ax = base_name[:-2], base_name[-1]
            x, y = joint_xy_cache.get(j, (np.zeros(1), np.zeros(1)))
            series = x if ax == "x" else y
            out[col] = _stat(series, stat); continue
        if base_name.startswith("ang_"):
            an = base_name[4:]; a = angle_cache.get(an, np.zeros(1))
            out[col] = _stat(a, stat); continue
        out[col] = 0.0
    return pd.Series(out, index=ref_cols, dtype=float)


def compute_similarity_for_window(user_window, ref_df_label):
    if ref_df_label.empty: return 0.0
    ref_feat_cols = [c for c in ref_df_label.columns if c != "label"]
    user_row = _build_user_row_from_window(user_window, ref_feat_cols).to_numpy().reshape(1, -1)
    ref_features = ref_df_label[ref_feat_cols].to_numpy()
    if ref_features.size == 0: return 0.0
    scaler = StandardScaler()
    scaler.fit(np.vstack([ref_features, user_row]))
    ref_scaled = scaler.transform(ref_features)
    user_scaled = scaler.transform(user_row)
    cos_sims = (cosine_similarity(user_scaled, ref_scaled)[0] + 1) / 2
    eucl_dists = np.linalg.norm(ref_scaled - user_scaled, axis=1)
    eucl_sims = np.exp(-0.04 * eucl_dists)
    combined = np.clip((0.75 * cos_sims + 0.25 * eucl_sims) * 1.22, 0, 1)
    return float(np.max(combined))


def _base_shot_name(shot_name: str):
    if not shot_name:
        return None
    s = str(shot_name)
    if s.endswith("left"):
        s = s[:-4]
    return s


def bat_angle_score(shot_name: str, angle_deg: float) -> float:
    if angle_deg is None:
        return 0.0
    base = _base_shot_name(shot_name)
    a = float(angle_deg)

    if base in ("cut", "pull", "sweep"):
        eff = min(a, 180.0 - a)
        score = max(0.0, 100.0 - (eff / 90.0) * 100.0)
    elif base == "cov":
        if 30.0 <= a <= 135.0:
            score = 100.0
        else:
            if a < 30.0:
                diff = 30.0 - a
            else:
                diff = a - 135.0
            score = max(0.0, 100.0 - (diff / 30.0) * 100.0)
    elif base == "flick":
        diff = abs(a - 60.0)
        score = max(0.0, 100.0 - (diff / 90.0) * 100.0)
    else:
        diff = abs(a - 90.0)
        score = max(0.0, 100.0 - (diff / 90.0) * 100.0)
    return round(score, 1)


def draw_metrics_overlay(frame, metrics, frame_w, frame_h):
    overlay = frame.copy()
    alpha = 0.78
    panel_h = int(frame_h*0.26)
    panel_w = int(frame_w*0.38)
    padding = 16
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_font_scale = 0.75
    val_font_scale = 0.9
    title_color = (225,225,225); val_color = (255,255,255); bg_color = (24,24,24)
    lx = padding; ly = frame_h - panel_h - padding
    cv2.rectangle(overlay, (lx,ly), (lx+panel_w,ly+panel_h), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

    def fmt(label, value):
        if label == "Bat angle":
            return f"{value:.1f}"+ chr(176)
        else:
            return f"{value:.1f}%"

    items = [
        ("Bat angle", metrics.get("bat_angle",0.0)),
        ("Hit quality", metrics.get("hit_quality",0.0)),
        ("Face stability", metrics.get("face_stability",0.0)),
        ("Stance bend", metrics.get("stance_bend",0.0)),
        ("Front foot", metrics.get("front_foot_stance_score",0.0)),
        ("Shot accuracy", metrics.get("shot_accuracy",0.0))
    ]

    for i,(label,value) in enumerate(items):
        base_y = ly + 32 + i*36
        cv2.putText(frame, label, (lx+10, base_y), font, title_font_scale, title_color, 1, cv2.LINE_AA)
        val_text = fmt(label, float(value))
        (tw, th), _ = cv2.getTextSize(val_text, font, val_font_scale, 2)
        cv2.putText(frame, val_text, (lx + panel_w - 6 - tw, base_y+6), font, val_font_scale, val_color, 2, cv2.LINE_AA)

    cv2.putText(frame, "Metrics", (lx+10, ly-10), font, 0.8, (245,245,245), 2, cv2.LINE_AA)
    return frame


# ========================= MAIN CLASSIFICATION FUNCTION =========================

def process_cricket_shot_classification(video_path, json_path, player_type, output_folder="classify"):
    if player_type not in ("righty","lefty"):
        raise ValueError("player_type must be 'righty' or 'lefty'")
    os.makedirs(output_folder, exist_ok=True)

    with open(json_path, "r") as f:
        contact_data = json.load(f)

    cf_to_idx = {}
    for idx, item in enumerate(contact_data):
        if "frame_highlight" in item:
            try:
                cf_to_idx[int(item["frame_highlight"])] = idx
            except:
                pass

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width_o = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_o = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_skip = 1
    all_data = []
    pose_cache = {}
    pose_frame_to_idx = {}

    key_map = {
        "left_shoulder":5, "right_shoulder":6,
        "left_elbow":7, "right_elbow":8,
        "left_wrist":9, "right_wrist":10,
        "left_hip":11, "right_hip":12,
        "left_knee":13, "right_knee":14,
        "left_ankle":15, "right_ankle":16
    }

    processed_count = 0
    real_frame = 0
    out_w, out_h = data_cfg["image_size"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if real_frame % frame_skip != 0:
            real_frame += 1
            continue

        results = yolo_model(frame, verbose=False)
        if len(results)==0 or results[0].boxes is None or len(results[0].boxes)==0:
            real_frame += 1
            continue

        try:
            bboxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
        except Exception:
            real_frame += 1
            continue

        person_indices = [i for i, cls in enumerate(class_ids) if int(cls)==0]
        if not person_indices:
            real_frame += 1
            continue

        areas = [(bboxes[i][2]-bboxes[i][0])*(bboxes[i][3]-bboxes[i][1]) for i in person_indices]
        largest_idx = person_indices[np.argmax(areas)]
        x1,y1,x2,y2 = map(int, bboxes[largest_idx])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            real_frame += 1
            continue

        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize((out_w,out_h))
        img_tensor = transforms.ToTensor()(pil_crop).unsqueeze(0).to(DEVICE)
        heatmaps = vitpose(img_tensor).detach().cpu().numpy()
        points, _ = keypoints_from_heatmaps(
            heatmaps,
            center=np.array([[out_w//2, out_h//2]]),
            scale=np.array([[out_w, out_h]]),
            unbiased=True, use_udp=True
        )
        kp = points[0][:, ::-1]

        full_kp_norm = np.zeros_like(kp, dtype=np.float32)
        full_kp_norm[:,0] = kp[:,0]/out_w
        full_kp_norm[:,1] = kp[:,1]/out_h
        pose_cache[real_frame] = full_kp_norm.tolist()

        coords_all = {}
        for idx_name in KEY_ORDER:
            if idx_name in key_map:
                i = key_map[idx_name]
                coords_all[idx_name] = (kp[i][0]/out_w, kp[i][1]/out_h) if i < kp.shape[0] else (0.0,0.0)
            else:
                head_guess = {"nose":0, "left_eye":1, "right_eye":2, "left_ear":3, "right_ear":4}.get(idx_name, None)
                if head_guess is not None and head_guess < kp.shape[0]:
                    coords_all[idx_name] = (kp[head_guess][0]/out_w, kp[head_guess][1]/out_h)
                else:
                    coords_all[idx_name] = (0.0,0.0)

        angles_vals = []
        for a in ANGLES_ORDER:
            if a == "left_elbow":
                ang = angle_360(coords_all.get("left_shoulder",(0,0)), coords_all.get("left_elbow",(0,0)), coords_all.get("left_wrist",(0,0)))
            elif a == "right_elbow":
                ang = angle_360(coords_all.get("right_shoulder",(0,0)), coords_all.get("right_elbow",(0,0)), coords_all.get("right_wrist",(0,0)))
            elif a == "left_shoulder":
                ang = angle_360(coords_all.get("left_hip",(0,0)), coords_all.get("left_shoulder",(0,0)), coords_all.get("left_elbow",(0,0)))
            elif a == "right_shoulder":
                ang = angle_360(coords_all.get("right_hip",(0,0)), coords_all.get("right_shoulder",(0,0)), coords_all.get("right_elbow",(0,0)))
            elif a == "left_knee":
                ang = angle_360(coords_all.get("left_hip",(0,0)), coords_all.get("left_knee",(0,0)), coords_all.get("left_ankle",(0,0)))
            elif a == "right_knee":
                ang = angle_360(coords_all.get("right_hip",(0,0)), coords_all.get("right_knee",(0,0)), coords_all.get("right_ankle",(0,0)))
            elif a == "left_hip":
                ang = angle_360(coords_all.get("left_shoulder",(0,0)), coords_all.get("left_hip",(0,0)), coords_all.get("left_knee",(0,0)))
            elif a == "right_hip":
                ang = angle_360(coords_all.get("right_shoulder",(0,0)), coords_all.get("right_hip",(0,0)), coords_all.get("right_knee",(0,0)))
            else:
                ang = 0.0
            angles_vals.append(ang/360.0)

        lstm_coords_flat = []
        for kname in LSTM_KEY_ORDER:
            xy = coords_all.get(kname, (0.0, 0.0))
            lstm_coords_flat.extend([float(xy[0]), float(xy[1])])
        lstm_feature_vector = lstm_coords_flat + [float(a) for a in angles_vals]

        if len(lstm_feature_vector) != LSTM_EXPECTED_FEATURES:
            if len(lstm_feature_vector) < LSTM_EXPECTED_FEATURES:
                lstm_feature_vector.extend([0.0] * (LSTM_EXPECTED_FEATURES - len(lstm_feature_vector)))
            else:
                lstm_feature_vector = lstm_feature_vector[:LSTM_EXPECTED_FEATURES]

        all_data.append(np.nan_to_num(np.array(lstm_feature_vector, dtype=np.float32)))
        pose_frame_to_idx[real_frame] = len(all_data) - 1

        real_frame += 1
        processed_count += 1

    cap.release()
    all_data = np.array(all_data, dtype=np.float32)

    X = []
    half_window = WINDOW_SIZE//2
    available_pose_frames = sorted(pose_frame_to_idx.keys())

    def nearest_pose_index_for_video_frame(video_cf):
        candidates = [pf for pf in available_pose_frames if pf <= video_cf]
        if candidates: pf = candidates[-1]
        else: pf = available_pose_frames[0] if available_pose_frames else None
        return pose_frame_to_idx[pf] if pf is not None else None

    contact_frames = [int(item["frame_highlight"]) for item in contact_data if "frame_highlight" in item]
    total_pose_frames = len(all_data)

    valid_windows = 0
    for cf in contact_frames:
        pose_idx = nearest_pose_index_for_video_frame(cf)
        if pose_idx is None:
            continue
        start = max(0, pose_idx - half_window)
        end = min(total_pose_frames, pose_idx + half_window)
        window = all_data[start:end]
        if len(window) < WINDOW_SIZE:
            pad = WINDOW_SIZE - len(window)
            window = np.pad(window, ((0,pad),(0,0)), mode='edge')
        X.append(window)
        valid_windows += 1

    if len(X) == 0:
        return output_folder

    X = np.stack(X).astype(np.float32)
    preds = lstm_model.predict(X, verbose=0)

    results = []
    predicted_shots = []
    used_contact_video_frames = []

    for cf in contact_frames:
        pose_idx = nearest_pose_index_for_video_frame(cf)
        if pose_idx is None: continue
        used_contact_video_frames.append(cf)

    for i, p in enumerate(preds):
        conf = float(np.max(p))
        cls_idx = int(np.argmax(p))
        shot_name = str(LABEL_CLASSES[cls_idx])
        if conf >= THRESHOLD:
            if player_type == "righty" and cls_idx not in RIGHTY_CLASSES:
                continue
            if player_type == "lefty" and cls_idx not in LEFTY_CLASSES:
                continue
            results.append((int(used_contact_video_frames[i]), cls_idx, conf))
            if shot_name not in predicted_shots:
                predicted_shots.append(shot_name)

    segments_info = []
    overall_path = os.path.join(output_folder, "merged_overall_highlights.mp4")

    if len(results) > 0:
        cap_o = cv2.VideoCapture(video_path)
        fps_o = cap_o.get(cv2.CAP_PROP_FPS) or fps
        overall_out = cv2.VideoWriter(
            overall_path,
            cv2.VideoWriter_fourcc(*'mp4v'), fps_o, (width_o, height_o)
        )
        overall_frame_counter = 0
        sorted_results = sorted(results, key=lambda x: int(x[0]))

        ref_df_all = pd.read_csv(REF_CSV_PATH) if os.path.exists(REF_CSV_PATH) else pd.DataFrame()

        for result_idx, (cframe, cls_idx, conf) in enumerate(sorted_results):
            if player_type == "righty" and cls_idx not in RIGHTY_CLASSES: continue
            if player_type == "lefty" and cls_idx not in LEFTY_CLASSES: continue

            pose_idx = nearest_pose_index_for_video_frame(cframe)
            if pose_idx is None: continue

            start = max(0, int(pose_idx)-half_window)
            end = min(total_pose_frames, int(pose_idx)+half_window)

            item_idx = cf_to_idx.get(int(cframe))
            temp_item = contact_data[item_idx].copy() if item_idx is not None else {}

            if "video_dimensions" not in temp_item or not temp_item.get("video_dimensions"):
                temp_item["video_dimensions"] = {"width": width_o, "height": height_o}
            video_dims = temp_item.get("video_dimensions", {})
            video_w = video_dims.get("width", width_o)
            video_h = video_dims.get("height", height_o)

            if temp_item.get("wrists") is None:
                if 0 <= int(pose_idx) < total_pose_frames:
                    feat = all_data[int(pose_idx)]
                    lw_base = 2 * LSTM_KEY_ORDER.index("left_wrist")
                    rw_base = 2 * LSTM_KEY_ORDER.index("right_wrist")
                    left_w = [float(feat[lw_base]), float(feat[lw_base+1])]
                    right_w = [float(feat[rw_base]), float(feat[rw_base+1])]
                    temp_item["wrists"] = {"left_wrist": left_w, "right_wrist": right_w}
                else:
                    temp_item["wrists"] = {"left_wrist":[0,0], "right_wrist":[0,0]}

            temp_item["bat_angle"] = None
            if temp_item.get("bat", {}) and temp_item["bat"].get("pts_orig") and video_w and video_h:
                pts_orig = temp_item["bat"]["pts_orig"]
                if pts_orig and len(pts_orig) >= 2:
                    longest_pair = None
                    max_len = -1.0
                    for p1, p2 in combinations(pts_orig, 2):
                        d = euclid(p1,p2)
                        if d > max_len:
                            max_len = d
                            longest_pair = (p1,p2)
                    if longest_pair:
                        pA_norm = normalize_by_video_dims(longest_pair[0], video_w, video_h)
                        pB_norm = normalize_by_video_dims(longest_pair[1], video_w, video_h)
                        wrists = temp_item.get("wrists", {})
                        lw = [float(wrists.get("left_wrist",[0,0])[0]), float(wrists.get("left_wrist",[0,0])[1])]
                        rw = [float(wrists.get("right_wrist",[0,0])[0]), float(wrists.get("right_wrist",[0,0])[1])]
                        if (lw != [0.0,0.0]) or (rw != [0.0,0.0]):
                            center = [(pA_norm[0]+pB_norm[0])/2.0, (pA_norm[1]+pB_norm[1])/2.0]
                            d_l = euclid(lw, center)
                            d_r = euclid(rw, center)
                            chosen_wrist = lw if d_l <= d_r else rw
                            dA = euclid(chosen_wrist, pA_norm)
                            dB = euclid(chosen_wrist, pB_norm)
                            far = pA_norm if dA >= dB else pB_norm
                            bat_vec = [far[0]-chosen_wrist[0], far[1]-chosen_wrist[1]]
                            horiz = [1.0, 0.0]
                            ang = safe_acos_angle(bat_vec, horiz)
                            if ang is not None:
                                temp_item["bat_angle"] = round(float(ang), 2)

            ball = temp_item.get("ball")
            bat = temp_item.get("bat")
            wrists = temp_item.get("wrists", {})
            if ball and bat and bat.get("pts_orig") and video_w and video_h:
                left_wrist = wrists.get("left_wrist")
                right_wrist = wrists.get("right_wrist")
                chosen_wrist = right_wrist if right_wrist not in ([0,0],[0.0,0.0],None) else left_wrist
                if chosen_wrist:
                    x_coords = [pt[0] for pt in bat["pts_orig"]]
                    y_coords = [pt[1] for pt in bat["pts_orig"]]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    crop_w = (x_max-x_min) or 1.0
                    crop_h = (y_max-y_min) or 1.0
                    wrist_pt = [x_min + chosen_wrist[0]*crop_w, y_min + chosen_wrist[1]*crop_h]
                    box_pts = compute_imaginary_box(
                        bat["pts_orig"], wrist_pt,
                        length_ratio=HIT_BOX_LENGTH_RATIO,
                        width_ratio=HIT_BOX_WIDTH_RATIO
                    )
                    ball_center = [ball.get("x_orig"), ball.get("y_orig")]
                    quality = compute_hit_quality(ball_center, BALL_RADIUS, box_pts)
                    temp_item["hit_quality"] = round(max(float(quality)*100, 40.0), 1)
                else:
                    temp_item["hit_quality"] = 40.0
            else:
                temp_item["hit_quality"] = 40.0

            inverse_map = {v:k for k,v in pose_frame_to_idx.items()}
            rf = inverse_map.get(int(pose_idx), int(cframe))
            temp_item["face_stability"] = face_stability_from_prev(pose_cache, rf)
            temp_item["stance_bend"] = stance_bend_from_prev(pose_cache, rf, player_type)
            temp_item["front_foot_stance_score"] = front_foot_stance_from_prev10(
                pose_cache, rf, search_back=10, radius=3
            )

            window = all_data[start:end]
            if window.shape[0] < WINDOW_SIZE:
                pad = WINDOW_SIZE - window.shape[0]
                window = np.pad(window, ((0,pad),(0,0)), mode='edge')
            ref_df_label = ref_df_all[ref_df_all["label"].str.lower() == str(LABEL_CLASSES[cls_idx]).lower()] if not ref_df_all.empty else pd.DataFrame()
            sim = compute_similarity_for_window(window, ref_df_label)
            temp_item["shot_accuracy"] = int(round(sim * 100))

            shot_name = str(LABEL_CLASSES[int(cls_idx)])
            if temp_item.get("bat_angle") is not None:
                temp_item["bat_angle_percent"] = bat_angle_score(shot_name, temp_item["bat_angle"])
            else:
                temp_item["bat_angle_percent"] = 0.0

            if item_idx is not None:
                contact_data[item_idx].update({
                    "bat_angle": temp_item.get("bat_angle"),
                    "bat_angle_percent": temp_item.get("bat_angle_percent"),
                    "hit_quality": temp_item.get("hit_quality"),
                    "face_stability": temp_item.get("face_stability"),
                    "stance_bend": temp_item.get("stance_bend"),
                    "front_foot_stance_score": temp_item.get("front_foot_stance_score"),
                    "shot_accuracy": temp_item.get("shot_accuracy")
                })

            video_start = max(0, int(cframe) - MERGE_PRE_FRAMES)
            video_end   = min(total_video_frames, int(cframe) + MERGE_POST_FRAMES + 1)

            cap_o.set(cv2.CAP_PROP_POS_FRAMES, video_start)
            frames_written = 0
            last_frame = None

            while frames_written < MERGE_FRAMES_TOTAL and cap_o.get(cv2.CAP_PROP_POS_FRAMES) < video_end:
                ret_f, frame_f = cap_o.read()
                if not ret_f:
                    break
                overall_out.write(frame_f)
                last_frame = frame_f
                frames_written += 1
                overall_frame_counter += 1

            if frames_written < MERGE_FRAMES_TOTAL:
                if last_frame is None:
                    last_frame = np.zeros((height_o, width_o, 3), dtype=np.uint8)
                for _ in range(frames_written, MERGE_FRAMES_TOTAL):
                    overall_out.write(last_frame)
                    overall_frame_counter += 1

            segments_info.append({
                "cframe": int(cframe),
                "cls_idx": int(cls_idx),
                "conf": float(conf),
                "merged_start_frame": overall_frame_counter - MERGE_FRAMES_TOTAL,
                "merged_frame_count": MERGE_FRAMES_TOTAL,
                "shot_name": shot_name,
                "start_frame": int(video_start),
                "metrics": {
                    "bat_angle": temp_item.get("bat_angle") or 0.0,
                    "hit_quality": temp_item.get("hit_quality") or 0.0,
                    "face_stability": temp_item.get("face_stability") or 0.0,
                    "stance_bend": temp_item.get("stance_bend") or 0.0,
                    "front_foot_stance_score": temp_item.get("front_foot_stance_score") or 0.0,
                    "shot_accuracy": temp_item.get("shot_accuracy") or 0.0
                }
            })

        overall_out.release()
        cap_o.release()

    if os.path.exists(overall_path) and len(segments_info) > 0:
        cap_s = cv2.VideoCapture(overall_path)
        fps_s = cap_s.get(cv2.CAP_PROP_FPS) or fps
        w_s = int(cap_s.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_s = int(cap_s.get(cv2.CAP_PROP_FRAME_HEIGHT))
        shot_writers = {}

        for seg in segments_info:
            shot = seg["shot_name"]
            start_frame = int(seg["merged_start_frame"])
            count = int(seg["merged_frame_count"])
            if player_type == "righty" and int(seg["cls_idx"]) not in RIGHTY_CLASSES: continue
            if player_type == "lefty" and int(seg["cls_idx"]) not in LEFTY_CLASSES: continue

            if shot not in shot_writers:
                out_path = os.path.join(output_folder, f"{shot}.mp4")
                shot_writers[shot] = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_s, (w_s, h_s))

            cap_s.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            metrics_panel = seg["metrics"]
            for _ in range(count):
                ret_f, frame_f = cap_s.read()
                if not ret_f: break
                annotated = draw_metrics_overlay(frame_f, metrics_panel, w_s, h_s)
                shot_writers[shot].write(annotated)

        cap_s.release()
        for w in shot_writers.values():
            w.release()

    compact_contacts = []
    results_map = {int(r[0]): (int(r[1]), float(r[2])) for r in results} if results else {}

    for item in contact_data:
        cf = int(item.get("frame_highlight", -1))
        start_frame = max(0, cf - half_window) if cf >= 0 else -1
        shot_type = None
        conf_01 = 0.0
        if cf in results_map:
            cls_idx, conf_01 = results_map[cf]
            shot_type = str(LABEL_CLASSES[cls_idx])

        bat_angle_deg = item.get("bat_angle", None)
        bat_angle_percent = item.get("bat_angle_percent", 0.0)

        compact = {
            "start_frame": int(start_frame),
            "frame_highlight": int(cf),
            "video_dimensions": {"width": int(width_o), "height": int(height_o)},
            "shot_type": shot_type,
            "bat_angle": bat_angle_percent,
            "bat_angle_deg": bat_angle_deg,
            "hit_quality": item.get("hit_quality", 0.0),
            "face_stability": item.get("face_stability", 0.0),
            "stance_bend": item.get("stance_bend", 0.0),
            "front_foot_stance_score": item.get("front_foot_stance_score", 0.0),
            "shot_accuracy": item.get("shot_accuracy", 0)
        }

        fb_text = None
        if shot_type is not None:
            base_shot = _base_shot_name(shot_type) or shot_type
            conf_pct = round(conf_01 * 100.0, 1)
            fb_text = generate_feedback(
                shot_type=base_shot,
                confidence=conf_pct,
                bat_angle=bat_angle_percent,
                hit_quality=compact["hit_quality"],
                face_stability=compact["face_stability"],
                stance_bend=compact["stance_bend"],
                front_foot_stance_score=compact["front_foot_stance_score"],
                shot_accuracy=compact["shot_accuracy"],
            )

        compact["feedback_paragraph"] = fb_text
        compact_contacts.append(compact)

    contacts_path = os.path.join(output_folder, "contacts.json")
    with open(contacts_path, "w") as f:
        json.dump(compact_contacts, f, indent=2)

    predicted_counts = {}
    for _, cls_idx, _ in results:
        shot_name = str(LABEL_CLASSES[int(cls_idx)])
        predicted_counts[shot_name] = predicted_counts.get(shot_name, 0) + 1

    predicted_path = os.path.join(output_folder, "predicted.json")
    with open(predicted_path, "w") as f:
        json.dump(predicted_counts, f, indent=2)

    return output_folder


# =========================================================
# /classify ROUTE (USES ALREADY-LOADED MODELS)
# =========================================================

@app.route("/classify", methods=["POST"])
def classify_route():
    try:
        t0 = time.time()

        if "video" not in request.files or "json_file" not in request.files:
            return jsonify({"error": "Both 'video' and 'json_file' are required"}), 400

        job_id = request.form.get("job_id", f"job_{int(time.time())}")
        player_type = request.form.get("player_type", "righty")

        if player_type not in ("righty", "lefty"):
            return jsonify({"error": "player_type must be 'righty' or 'lefty'"}), 400

        job_root = os.path.join(PROCESSED_FOLDER, job_id)
        work_root = os.path.join(job_root, "classify_work")
        out_dir = os.path.join(job_root, "classify")
        os.makedirs(work_root, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        video_path = os.path.join(work_root, "highlight.mp4")
        json_path = os.path.join(work_root, "contact_info.json")
        request.files["video"].save(video_path)
        request.files["json_file"].save(json_path)

        output_dir = process_cricket_shot_classification(
            video_path=video_path,
            json_path=json_path,
            player_type=player_type,
            output_folder=out_dir
        )

        # CONVERT VIDEOS
        for root, dirs, files in os.walk(out_dir):
            for file in files:
                if file.endswith(".mp4"):
                    video_file_path = os.path.join(root, file)
                    converted_video_path = os.path.join(root, f"converted_{file}")
                    ffmpeg_cmd = [
                        "ffmpeg", "-y", "-i", video_file_path,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                        "-movflags", "+faststart", converted_video_path
                    ]
                    subprocess.run(ffmpeg_cmd, check=True)
                    os.remove(video_file_path)
                    os.rename(converted_video_path, video_file_path)

        zip_path = os.path.join(out_dir, "classify_results.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(output_dir):
                for f in files:
                    if f.endswith('.zip'):
                        continue
                    absf = os.path.join(root, f)
                    relf = os.path.relpath(absf, output_dir)
                    z.write(absf, relf)

        elapsed = time.time() - t0
        print(f"✅ Job {job_id} completed in {elapsed:.1f}s")

        response = send_file(
            zip_path,
            as_attachment=True,
            download_name=f"classify_results_{job_id}.zip",
            mimetype="application/zip"
        )

        @response.call_on_close
        def _cleanup():
            # delete whole job_root 5 minutes after response closed
            schedule_folder_cleanup(job_root, delay_seconds=300)

        return response

    except Exception as e:
        print(f"🔥 CLASSIFY ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# =========================================================
# GLOBAL ERROR HANDLER
# =========================================================

@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return e
    print("❌ Unhandled server error:", repr(e))
    return jsonify({"error": "internal_server_error", "detail": str(e)}), 500


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    print("🚀 Starting Flask app (debug mode)...")
    load_models()
    app.run(host="0.0.0.0", port=8000, debug=True)
else:
    print("🚀 WSGI import: loading models once for this worker...")
    load_models()

