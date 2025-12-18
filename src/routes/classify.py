# src/routes/classify.py
# RunPod Serverless - classify (Base64 ZIP in, Base64 ZIP out)
# Expects the SAME “serverless flow” + ZIP structure style as process.py

import base64
import json
import math
import os
import re
import shutil
import subprocess
import time
import zipfile
from itertools import combinations

import cv2
import numpy as np
import pandas as pd

from PIL import Image
from torchvision import transforms

from shapely.geometry import Point, Polygon

import torch
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

import helper  # kept (you may use later); models must come from constants
import constants

# ViTPose utilities (must exist in your vitpose package)
from vitpose.utils.top_down_eval import keypoints_from_heatmaps
from vitpose.configs.ViTPose_base_coco_256x192 import data_cfg


# ----------------------------
# Small utilities
# ----------------------------
def _require_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found at: {path}")


def _b64_to_file(b64str: str, out_path: str) -> None:
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(b64str))


def _file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _cleanup_paths(*paths: str) -> None:
    for p in paths:
        try:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            elif os.path.exists(p):
                os.remove(p)
        except Exception as _e:
            print(f"⚠ Cleanup warning for {p}: {_e}")


# ----------------------------
# Feedback (T5) - uses already loaded constants.tokenizer/constants.t5_model
# ----------------------------
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
    if not sentences:
        return ""
    return ". ".join(sentences) + "."


def generate_feedback(**features) -> str:
    # tokenizer + t5_model are loaded in helper.load_models() and stored in constants.*
    text = format_input(**features)
    inputs = constants.tokenizer(text, return_tensors="pt").to(constants.DEVICE)

    with torch.no_grad():
        output = constants.t5_model.generate(
            **inputs,
            max_new_tokens=160,
            num_beams=8,
            no_repeat_ngram_size=4,
            repetition_penalty=1.2,
            length_penalty=1.0,
            early_stopping=True,
            do_sample=False,
        )

    raw = constants.tokenizer.decode(output[0], skip_special_tokens=True)
    return clean_feedback(raw)


# ----------------------------
# Geometry / Metrics
# ----------------------------
def angle_360(a, b, c):
    v1 = np.array([a[0] - b[0], a[1] - b[1]])
    v2 = np.array([c[0] - b[0], c[1] - b[1]])
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    dot = np.dot(v1, v2)
    det = v1[0] * v2[1] - v1[1] * v2[0]
    ang = np.degrees(np.arctan2(det, dot))
    if ang < 0:
        ang += 360
    return ang


def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def safe_acos_angle(v1, v2):
    na = math.hypot(v1[0], v1[1])
    nb = math.hypot(v2[0], v2[1])
    if na == 0 or nb == 0:
        return None
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cosv = dot / (na * nb)
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(np.arccos(cosv))


def normalize_by_video_dims(pt, video_w, video_h):
    return [pt[0] / float(video_w), pt[1] / float(video_h)]


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
    p1 = mid + perp_unit * half_width
    p2 = mid - perp_unit * half_width
    p3 = p2 + axis_vec * length_ratio
    p4 = p1 + axis_vec * length_ratio
    return [p1.tolist(), p2.tolist(), p3.tolist(), p4.tolist()]


def compute_hit_quality(ball_center, ball_radius, box_pts):
    ball = Point(ball_center[0], ball_center[1]).buffer(ball_radius)
    box_poly = Polygon(box_pts)
    if not box_poly.is_valid:
        box_poly = box_poly.buffer(0)
    intersection_area = ball.intersection(box_poly).area
    ball_area = ball.area
    ratio = intersection_area / ball_area if ball_area > 0 else 0
    if ratio > 0:
        ratio = max(ratio, constants.HIT_MIN_RATIO)
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
    idx_left_eye = constants.KEY_ORDER.index("left_eye")
    idx_right_eye = constants.KEY_ORDER.index("right_eye")
    collected = _collect_prev_valid_frames(pose_cache, ref_frame, n=10)
    vals = []
    for _, kp in collected:
        if idx_left_eye < len(kp) and idx_right_eye < len(kp):
            le = kp[idx_left_eye]
            re_ = kp[idx_right_eye]
            if _valid_point(le) and _valid_point(re_):
                dx = re_[0] - le[0]
                dy = re_[1] - le[1]
                angle = math.degrees(math.atan2(dy, dx))
                stability = 100.0 if abs(angle) < 15 else (math.cos(math.radians(angle)) ** 0.3) * 100
                vals.append(stability)
    return round(float(np.mean(vals)), 1) if vals else 0.0


def knee_bend_percentage(hip, knee, ankle):
    v1 = [hip[0] - knee[0], hip[1] - knee[1]]
    v2 = [ankle[0] - knee[0], ankle[1] - knee[1]]
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    angle_deg = math.degrees(math.acos(cos_angle))
    optimal = 120
    deviation = abs(angle_deg - optimal)
    score = max(0, 100 - deviation * 1.2)
    return round(score, 1)


def stance_bend_from_prev(pose_cache, ref_frame, player_type):
    idx = {
        k: constants.KEY_ORDER.index(k)
        for k in ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
    }
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
    idx_la = constants.KEY_ORDER.index("left_ankle")
    idx_ra = constants.KEY_ORDER.index("right_ankle")
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
    end = max(0, target + radius)

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


# ----------------------------
# Similarity computation (exactly like your colab cell)
# ----------------------------
_LSTM_JOINT_IDX = {j: i for i, j in enumerate(constants.LSTM_KEY_ORDER)}

_vel_re = re.compile(r"^(?P<joint>.+)_(?P<ax>x|y)_vel$")
_acc_re = re.compile(r"^(?P<joint>.+)_(?P<ax>x|y)_acc$")
_speed_re = re.compile(r"^(?P<joint>.+)_speed$")
_ang_re = re.compile(r"^ang_(?P<aname>.+)$")
_angvel_re = re.compile(r"^ang_(?P<aname>.+)_vel$")
_stat_re = re.compile(r"^(?P<base>.+)_(?P<stat>mean|std|last)$")


def _xy_indices_for_joint(joint_name):
    j = _LSTM_JOINT_IDX.get(joint_name, None)
    if j is None:
        return None, None
    base = 2 * j
    return base, base + 1


def _finite_diff(arr):
    if arr.size < 2:
        return np.array([0.0], dtype=float)
    return np.diff(arr, axis=0).astype(float)


def _stat(series, stat):
    if stat == "std":
        return float(np.nanstd(series)) if series.size else 0.0
    if stat == "last":
        return float(series[-1]) if series.size else 0.0
    return float(np.nanmean(series)) if series.size else 0.0


def _joint_series(window, joint):
    xi, yi = _xy_indices_for_joint(joint)
    if xi is None:
        return np.zeros((window.shape[0],)), np.zeros((window.shape[0],))
    return window[:, xi], window[:, yi]


def _angle_series(window, aname):
    try:
        ai = 24 + constants.ANGLES_ORDER.index(aname)
    except ValueError:
        ai = None
    if ai is None:
        return np.zeros((window.shape[0],))
    return window[:, ai]


def _build_user_row_from_window(window, ref_cols):
    out = {}
    joint_xy_cache = {j: _joint_series(window, j) for j in constants.LSTM_KEY_ORDER}
    angle_cache = {a: _angle_series(window, a) for a in constants.ANGLES_ORDER}

    for col in ref_cols:
        stat = None
        base_name = col
        mstat = _stat_re.match(col)
        if mstat:
            base_name = mstat.group("base")
            stat = mstat.group("stat")

        m = _vel_re.match(base_name)
        if m:
            j, ax = m.group("joint"), m.group("ax")
            x, y = joint_xy_cache.get(j, (np.zeros(1), np.zeros(1)))
            series = x if ax == "x" else y
            out[col] = _stat(_finite_diff(series), stat)
            continue

        m = _acc_re.match(base_name)
        if m:
            j, ax = m.group("joint"), m.group("ax")
            x, y = joint_xy_cache.get(j, (np.zeros(1), np.zeros(1)))
            v = _finite_diff(x if ax == "x" else y)
            a = _finite_diff(v)
            out[col] = _stat(a, stat)
            continue

        m = _speed_re.match(base_name)
        if m:
            j = m.group("joint")
            x, y = joint_xy_cache.get(j, (np.zeros(1), np.zeros(1)))
            sp = np.sqrt(_finite_diff(x) ** 2 + _finite_diff(y) ** 2)
            out[col] = _stat(sp, stat)
            continue

        if _ang_re.match(base_name) and not _angvel_re.match(base_name):
            an = base_name[4:]
            a = angle_cache.get(an, np.zeros(1))
            out[col] = _stat(a, stat)
            continue

        m = _angvel_re.match(base_name)
        if m:
            an = m.group("aname")
            a = angle_cache.get(an, np.zeros(1))
            out[col] = _stat(_finite_diff(a), stat)
            continue

        if base_name.endswith("_x") or base_name.endswith("_y"):
            j, ax = base_name[:-2], base_name[-1]
            x, y = joint_xy_cache.get(j, (np.zeros(1), np.zeros(1)))
            series = x if ax == "x" else y
            out[col] = _stat(series, stat)
            continue

        if base_name.startswith("ang_"):
            an = base_name[4:]
            a = angle_cache.get(an, np.zeros(1))
            out[col] = _stat(a, stat)
            continue

        out[col] = 0.0

    return pd.Series(out, index=ref_cols, dtype=float)


def compute_similarity_for_window(user_window, ref_df_label):
    if ref_df_label.empty:
        return 0.0
    ref_feat_cols = [c for c in ref_df_label.columns if c != "label"]
    user_row = _build_user_row_from_window(user_window, ref_feat_cols).to_numpy().reshape(1, -1)
    ref_features = ref_df_label[ref_feat_cols].to_numpy()
    if ref_features.size == 0:
        return 0.0

    scaler = StandardScaler()
    scaler.fit(np.vstack([ref_features, user_row]))
    ref_scaled = scaler.transform(ref_features)
    user_scaled = scaler.transform(user_row)

    cos_sims = (cosine_similarity(user_scaled, ref_scaled)[0] + 1) / 2
    eucl_dists = np.linalg.norm(ref_scaled - user_scaled, axis=1)
    eucl_sims = np.exp(-0.04 * eucl_dists)
    combined = np.clip((0.75 * cos_sims + 0.25 * eucl_sims) * 1.22, 0, 1)
    return float(np.max(combined))


# ----------------------------
# Bat angle percent score (shot-aware)
# ----------------------------
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
            diff = (30.0 - a) if a < 30.0 else (a - 135.0)
            score = max(0.0, 100.0 - (diff / 30.0) * 100.0)
    elif base == "flick":
        diff = abs(a - 60.0)
        score = max(0.0, 100.0 - (diff / 90.0) * 100.0)
    else:
        diff = abs(a - 90.0)
        score = max(0.0, 100.0 - (diff / 90.0) * 100.0)

    return round(score, 1)


# ----------------------------
# Overlay (per-shot clips)
# ----------------------------
def draw_metrics_overlay(frame, metrics, frame_w, frame_h):
    overlay = frame.copy()
    alpha = 0.78
    panel_h = int(frame_h * 0.26)
    panel_w = int(frame_w * 0.38)
    padding = 16
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_font_scale = 0.75
    val_font_scale = 0.9

    title_color = (225, 225, 225)
    val_color = (255, 255, 255)
    bg_color = (24, 24, 24)

    lx = padding
    ly = frame_h - panel_h - padding
    cv2.rectangle(overlay, (lx, ly), (lx + panel_w, ly + panel_h), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def fmt(label, value):
        if label == "Bat angle":
            return f"{value:.1f}" + chr(176)
        return f"{value:.1f}%"

    items = [
        ("Bat angle", metrics.get("bat_angle", 0.0)),
        ("Hit quality", metrics.get("hit_quality", 0.0)),
        ("Face stability", metrics.get("face_stability", 0.0)),
        ("Stance bend", metrics.get("stance_bend", 0.0)),
        ("Front foot", metrics.get("front_foot_stance_score", 0.0)),
        ("Shot accuracy", metrics.get("shot_accuracy", 0.0)),
    ]

    for i, (label, value) in enumerate(items):
        base_y = ly + 32 + i * 36
        cv2.putText(frame, label, (lx + 10, base_y), font, title_font_scale, title_color, 1, cv2.LINE_AA)
        val_text = fmt(label, float(value))
        (tw, _), _b = cv2.getTextSize(val_text, font, val_font_scale, 2)
        cv2.putText(
            frame,
            val_text,
            (lx + panel_w - 6 - tw, base_y + 6),
            font,
            val_font_scale,
            val_color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(frame, "Metrics", (lx + 10, ly - 10), font, 0.8, (245, 245, 245), 2, cv2.LINE_AA)
    return frame


# ----------------------------
# Core pipeline (serverless version)
# ----------------------------
def process_cricket_shot_classification(video_path, json_path, player_type, output_folder):
    print("=" * 60)
    print("CRICKET SHOT CLASSIFICATION (RUNPOD SERVERLESS)")
    print("=" * 60)

    if player_type not in ("righty", "lefty"):
        raise ValueError("player_type must be 'righty' or 'lefty'")

    _safe_mkdir(output_folder)

    # Load contact frames json (from process.zip)
    with open(json_path, "r") as f:
        contact_data = json.load(f)
    print(f"[INFO] Loaded {len(contact_data)} contacts")

    cf_to_idx = {}
    for idx, item in enumerate(contact_data):
        if "frame_highlight" in item:
            try:
                cf_to_idx[int(item["frame_highlight"])] = idx
            except Exception:
                pass

    # Video capture (highlight video)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width_o = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_o = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video: {fps}fps, {total_video_frames} frames, {width_o}x{height_o}")

    # ViTPose output image size
    out_w, out_h = data_cfg["image_size"]

    # Pose extraction caches
    all_data = []
    pose_cache = {}
    pose_frame_to_idx = {}

    key_map = {
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }

    processed_count = 0
    real_frame = 0
    frame_skip = 1

    print("[INFO] Extracting poses...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if real_frame % frame_skip != 0:
            real_frame += 1
            continue

        # Person detection
        results = constants.yolo_model(frame, verbose=False)
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            real_frame += 1
            continue

        try:
            bboxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
        except Exception:
            real_frame += 1
            continue

        person_indices = [i for i, cls in enumerate(class_ids) if int(cls) == 0]
        if not person_indices:
            real_frame += 1
            continue

        areas = [(bboxes[i][2] - bboxes[i][0]) * (bboxes[i][3] - bboxes[i][1]) for i in person_indices]
        largest_idx = person_indices[int(np.argmax(areas))]
        x1, y1, x2, y2 = map(int, bboxes[largest_idx])

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            real_frame += 1
            continue

        # ViTPose inference
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize((out_w, out_h))
        img_tensor = transforms.ToTensor()(pil_crop).unsqueeze(0).to(constants.DEVICE)

        with torch.no_grad():
            heatmaps = constants.vitpose(img_tensor).detach().cpu().numpy()

        points, _ = keypoints_from_heatmaps(
            heatmaps,
            center=np.array([[out_w // 2, out_h // 2]]),
            scale=np.array([[out_w, out_h]]),
            unbiased=True,
            use_udp=True,
        )
        kp = points[0][:, ::-1]  # (y,x) -> (x,y)

        # Normalize keypoints for pose_cache (0..1)
        full_kp_norm = np.zeros_like(kp, dtype=np.float32)
        full_kp_norm[:, 0] = kp[:, 0] / out_w
        full_kp_norm[:, 1] = kp[:, 1] / out_h
        pose_cache[real_frame] = full_kp_norm.tolist()

        # Build coordinates dictionary in KEY_ORDER
        coords_all = {}
        for idx_name in constants.KEY_ORDER:
            if idx_name in key_map:
                i = key_map[idx_name]
                coords_all[idx_name] = (
                    (kp[i][0] / out_w, kp[i][1] / out_h) if i < kp.shape[0] else (0.0, 0.0)
                )
            else:
                head_guess = {"nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4}.get(
                    idx_name, None
                )
                if head_guess is not None and head_guess < kp.shape[0]:
                    coords_all[idx_name] = (kp[head_guess][0] / out_w, kp[head_guess][1] / out_h)
                else:
                    coords_all[idx_name] = (0.0, 0.0)

        # Compute angles (normalized /360)
        angles_vals = []
        for a in constants.ANGLES_ORDER:
            if a == "left_elbow":
                ang = angle_360(
                    coords_all.get("left_shoulder", (0, 0)),
                    coords_all.get("left_elbow", (0, 0)),
                    coords_all.get("left_wrist", (0, 0)),
                )
            elif a == "right_elbow":
                ang = angle_360(
                    coords_all.get("right_shoulder", (0, 0)),
                    coords_all.get("right_elbow", (0, 0)),
                    coords_all.get("right_wrist", (0, 0)),
                )
            elif a == "left_shoulder":
                ang = angle_360(
                    coords_all.get("left_hip", (0, 0)),
                    coords_all.get("left_shoulder", (0, 0)),
                    coords_all.get("left_elbow", (0, 0)),
                )
            elif a == "right_shoulder":
                ang = angle_360(
                    coords_all.get("right_hip", (0, 0)),
                    coords_all.get("right_shoulder", (0, 0)),
                    coords_all.get("right_elbow", (0, 0)),
                )
            elif a == "left_knee":
                ang = angle_360(
                    coords_all.get("left_hip", (0, 0)),
                    coords_all.get("left_knee", (0, 0)),
                    coords_all.get("left_ankle", (0, 0)),
                )
            elif a == "right_knee":
                ang = angle_360(
                    coords_all.get("right_hip", (0, 0)),
                    coords_all.get("right_knee", (0, 0)),
                    coords_all.get("right_ankle", (0, 0)),
                )
            elif a == "left_hip":
                ang = angle_360(
                    coords_all.get("left_shoulder", (0, 0)),
                    coords_all.get("left_hip", (0, 0)),
                    coords_all.get("left_knee", (0, 0)),
                )
            elif a == "right_hip":
                ang = angle_360(
                    coords_all.get("right_shoulder", (0, 0)),
                    coords_all.get("right_hip", (0, 0)),
                    coords_all.get("right_knee", (0, 0)),
                )
            else:
                ang = 0.0
            angles_vals.append(ang / 360.0)

        # Build LSTM feature vector (32 features)
        lstm_coords_flat = []
        for kname in constants.LSTM_KEY_ORDER:
            xy = coords_all.get(kname, (0.0, 0.0))
            lstm_coords_flat.extend([float(xy[0]), float(xy[1])])

        lstm_feature_vector = lstm_coords_flat + [float(a) for a in angles_vals]

        if len(lstm_feature_vector) != constants.LSTM_EXPECTED_FEATURES:
            if len(lstm_feature_vector) < constants.LSTM_EXPECTED_FEATURES:
                lstm_feature_vector.extend([0.0] * (constants.LSTM_EXPECTED_FEATURES - len(lstm_feature_vector)))
            else:
                lstm_feature_vector = lstm_feature_vector[: constants.LSTM_EXPECTED_FEATURES]

        all_data.append(np.nan_to_num(np.array(lstm_feature_vector, dtype=np.float32)))
        pose_frame_to_idx[real_frame] = len(all_data) - 1

        real_frame += 1
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"  Processed {processed_count} frames...")

    cap.release()
    all_data = np.array(all_data, dtype=np.float32)
    print(f"[INFO] Extracted {processed_count} frames, feature shape: {all_data.shape}")

    # Prepare LSTM windows (COLAB-SAFE mapping)
    half_window = constants.WINDOW_SIZE // 2
    available_pose_frames = sorted(pose_frame_to_idx.keys())

    def nearest_pose_index_for_video_frame(video_cf):
        candidates = [pf for pf in available_pose_frames if pf <= video_cf]
        if candidates:
            pf = candidates[-1]
        else:
            pf = available_pose_frames[0] if available_pose_frames else None
        return pose_frame_to_idx[pf] if pf is not None else None

    contact_frames = [int(item["frame_highlight"]) for item in contact_data if "frame_highlight" in item]
    print(f"[INFO] Processing {len(contact_frames)} contact frames")

    total_pose_frames = len(all_data)

    pairs = []  # list of (contact_frame, window)
    for cf in contact_frames:
        pose_idx = nearest_pose_index_for_video_frame(cf)
        if pose_idx is None:
            continue

        start = max(0, pose_idx - half_window)
        end = min(total_pose_frames, pose_idx + half_window)
        window = all_data[start:end]
        if window.shape[0] < constants.WINDOW_SIZE:
            pad = constants.WINDOW_SIZE - window.shape[0]
            window = np.pad(window, ((0, pad), (0, 0)), mode="edge")

        pairs.append((int(cf), window))

    if not pairs:
        print("[ERROR] No valid windows created")
        return output_folder

    X = np.stack([w for _, w in pairs]).astype(np.float32)
    used_contact_video_frames = [cf for cf, _ in pairs]  # explicit mapping
    print(f"[INFO] LSTM input shape: {X.shape}")

    # LSTM inference
    print("[INFO] Running LSTM predictions...")
    preds = constants.lstm_model.predict(X, verbose=0)

    results = []
    predicted_shots = []

    for i, p in enumerate(preds):
        cf = int(used_contact_video_frames[i])
        conf = float(np.max(p))
        cls_idx = int(np.argmax(p))
        shot_name = str(constants.LABEL_CLASSES[cls_idx])

        if conf >= constants.THRESHOLD:
            # side filter (same as colab)
            if player_type == "righty" and cls_idx not in constants.RIGHTY_CLASSES:
                continue
            if player_type == "lefty" and cls_idx not in constants.LEFTY_CLASSES:
                continue

            results.append((cf, cls_idx, conf))
            if shot_name not in predicted_shots:
                predicted_shots.append(shot_name)

    print(f"[INFO] Predictions above threshold (after side filter): {len(results)}")

    # Video generation - merged overall + per-shot videos
    segments_info = []
    ref_df_all = pd.read_csv(constants.REF_CSV_PATH) if os.path.exists(constants.REF_CSV_PATH) else pd.DataFrame()

    if results:
        print("[INFO] Creating merged overall highlights video...")
        cap_o = cv2.VideoCapture(video_path)
        fps_o = cap_o.get(cv2.CAP_PROP_FPS) or fps

        overall_path = os.path.join(output_folder, "merged_overall_highlights.mp4")
        overall_out = cv2.VideoWriter(
            overall_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_o,
            (width_o, height_o),
        )

        overall_frame_counter = 0
        sorted_results = sorted(results, key=lambda x: int(x[0]))

        for result_idx, (cframe, cls_idx, conf) in enumerate(sorted_results):
            print(f"  Processing shot {result_idx+1}/{len(sorted_results)}")

            if player_type == "righty" and cls_idx not in constants.RIGHTY_CLASSES:
                continue
            if player_type == "lefty" and cls_idx not in constants.LEFTY_CLASSES:
                continue

            pose_idx = nearest_pose_index_for_video_frame(int(cframe))
            if pose_idx is None:
                continue

            start = max(0, int(pose_idx) - half_window)
            end = min(total_pose_frames, int(pose_idx) + half_window)

            item_idx = cf_to_idx.get(int(cframe))
            temp_item = contact_data[item_idx].copy() if item_idx is not None else {}

            if "video_dimensions" not in temp_item or not temp_item.get("video_dimensions"):
                temp_item["video_dimensions"] = {"width": width_o, "height": height_o}

            video_dims = temp_item.get("video_dimensions", {})
            video_w = video_dims.get("width", width_o)
            video_h = video_dims.get("height", height_o)

            # Ensure wrists exist (from LSTM features at pose_idx)
            if temp_item.get("wrists") is None:
                if 0 <= int(pose_idx) < total_pose_frames:
                    feat = all_data[int(pose_idx)]
                    lw_base = 2 * constants.LSTM_KEY_ORDER.index("left_wrist")
                    rw_base = 2 * constants.LSTM_KEY_ORDER.index("right_wrist")
                    left_w = [float(feat[lw_base]), float(feat[lw_base + 1])]
                    right_w = [float(feat[rw_base]), float(feat[rw_base + 1])]
                    temp_item["wrists"] = {"left_wrist": left_w, "right_wrist": right_w}
                else:
                    temp_item["wrists"] = {"left_wrist": [0, 0], "right_wrist": [0, 0]}

            # 1) Bat Angle (degrees)
            temp_item["bat_angle"] = None
            if temp_item.get("bat", {}) and temp_item["bat"].get("pts_orig") and video_w and video_h:
                pts_orig = temp_item["bat"]["pts_orig"]
                if pts_orig and len(pts_orig) >= 2:
                    longest_pair = None
                    max_len = -1.0
                    for p1, p2 in combinations(pts_orig, 2):
                        d = euclid(p1, p2)
                        if d > max_len:
                            max_len = d
                            longest_pair = (p1, p2)
                    if longest_pair:
                        pA_norm = normalize_by_video_dims(longest_pair[0], video_w, video_h)
                        pB_norm = normalize_by_video_dims(longest_pair[1], video_w, video_h)

                        wrists = temp_item.get("wrists", {})
                        lw = [
                            float(wrists.get("left_wrist", [0, 0])[0]),
                            float(wrists.get("left_wrist", [0, 0])[1]),
                        ]
                        rw = [
                            float(wrists.get("right_wrist", [0, 0])[0]),
                            float(wrists.get("right_wrist", [0, 0])[1]),
                        ]
                        if (lw != [0.0, 0.0]) or (rw != [0.0, 0.0]):
                            center = [(pA_norm[0] + pB_norm[0]) / 2.0, (pA_norm[1] + pB_norm[1]) / 2.0]
                            d_l = euclid(lw, center)
                            d_r = euclid(rw, center)
                            chosen_wrist = lw if d_l <= d_r else rw
                            dA = euclid(chosen_wrist, pA_norm)
                            dB = euclid(chosen_wrist, pB_norm)
                            far = pA_norm if dA >= dB else pB_norm
                            bat_vec = [far[0] - chosen_wrist[0], far[1] - chosen_wrist[1]]
                            horiz = [1.0, 0.0]
                            ang = safe_acos_angle(bat_vec, horiz)
                            if ang is not None:
                                temp_item["bat_angle"] = round(float(ang), 2)

            # 2) Hit Quality
            ball = temp_item.get("ball")
            bat = temp_item.get("bat")
            wrists = temp_item.get("wrists", {})
            if ball and bat and bat.get("pts_orig") and video_w and video_h:
                left_wrist = wrists.get("left_wrist")
                right_wrist = wrists.get("right_wrist")
                chosen_wrist = right_wrist if right_wrist not in ([0, 0], [0.0, 0.0], None) else left_wrist
                if chosen_wrist:
                    x_coords = [pt[0] for pt in bat["pts_orig"]]
                    y_coords = [pt[1] for pt in bat["pts_orig"]]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    crop_w = (x_max - x_min) or 1.0
                    crop_h = (y_max - y_min) or 1.0
                    wrist_pt = [x_min + chosen_wrist[0] * crop_w, y_min + chosen_wrist[1] * crop_h]
                    box_pts = compute_imaginary_box(
                        bat["pts_orig"],
                        wrist_pt,
                        length_ratio=constants.HIT_BOX_LENGTH_RATIO,
                        width_ratio=constants.HIT_BOX_WIDTH_RATIO,
                    )
                    ball_center = [ball.get("x_orig"), ball.get("y_orig")]
                    quality = compute_hit_quality(ball_center, constants.BALL_RADIUS, box_pts)
                    temp_item["hit_quality"] = round(max(float(quality) * 100, 40.0), 1)
                else:
                    temp_item["hit_quality"] = 40.0
            else:
                temp_item["hit_quality"] = 40.0

            # 3) Face Stability
            inverse_map = {v: k for k, v in pose_frame_to_idx.items()}
            rf = inverse_map.get(int(pose_idx), int(cframe))
            temp_item["face_stability"] = face_stability_from_prev(pose_cache, rf)

            # 4) Stance Bend
            temp_item["stance_bend"] = stance_bend_from_prev(pose_cache, rf, player_type)

            # 5) Front Foot Stance
            temp_item["front_foot_stance_score"] = front_foot_stance_from_prev10(
                pose_cache, rf, search_back=10, radius=3
            )

            # 6) Shot Accuracy (similarity)
            window = all_data[start:end]
            if window.shape[0] < constants.WINDOW_SIZE:
                pad = constants.WINDOW_SIZE - window.shape[0]
                window = np.pad(window, ((0, pad), (0, 0)), mode="edge")

            if not ref_df_all.empty and "label" in ref_df_all.columns:
                ref_df_label = ref_df_all[
                    ref_df_all["label"].str.lower() == str(constants.LABEL_CLASSES[cls_idx]).lower()
                ]
            else:
                ref_df_label = pd.DataFrame()

            sim = compute_similarity_for_window(window, ref_df_label)
            temp_item["shot_accuracy"] = int(round(sim * 100))

            # 7) Bat angle percent (shot aware)
            shot_name = str(constants.LABEL_CLASSES[int(cls_idx)])
            if temp_item.get("bat_angle") is not None:
                temp_item["bat_angle_percent"] = bat_angle_score(shot_name, temp_item["bat_angle"])
            else:
                temp_item["bat_angle_percent"] = 0.0

            # Save metrics back to contact_data (same as colab)
            if item_idx is not None:
                contact_data[item_idx].update(
                    {
                        "bat_angle": temp_item.get("bat_angle"),
                        "bat_angle_percent": temp_item.get("bat_angle_percent"),
                        "hit_quality": temp_item.get("hit_quality"),
                        "face_stability": temp_item.get("face_stability"),
                        "stance_bend": temp_item.get("stance_bend"),
                        "front_foot_stance_score": temp_item.get("front_foot_stance_score"),
                        "shot_accuracy": temp_item.get("shot_accuracy"),
                    }
                )

            # --- Decoupled VIDEO window: 50 frames around contact frame (MERGE_*) ---
            video_start = max(0, int(cframe) - constants.MERGE_PRE_FRAMES)
            video_end = min(total_video_frames, int(cframe) + constants.MERGE_POST_FRAMES + 1)

            cap_o.set(cv2.CAP_PROP_POS_FRAMES, video_start)
            frames_written = 0
            last_frame = None

            while frames_written < constants.MERGE_FRAMES_TOTAL and cap_o.get(cv2.CAP_PROP_POS_FRAMES) < video_end:
                ret_f, frame_f = cap_o.read()
                if not ret_f:
                    break
                overall_out.write(frame_f)
                last_frame = frame_f
                frames_written += 1
                overall_frame_counter += 1

            # pad if short
            if frames_written < constants.MERGE_FRAMES_TOTAL:
                if last_frame is None:
                    last_frame = np.zeros((height_o, width_o, 3), dtype=np.uint8)
                for _ in range(frames_written, constants.MERGE_FRAMES_TOTAL):
                    overall_out.write(last_frame)
                    overall_frame_counter += 1

            segments_info.append(
                {
                    "cframe": int(cframe),
                    "cls_idx": int(cls_idx),
                    "conf": float(conf),
                    "merged_start_frame": overall_frame_counter - constants.MERGE_FRAMES_TOTAL,
                    "merged_frame_count": constants.MERGE_FRAMES_TOTAL,
                    "shot_name": shot_name,
                    "start_frame": int(video_start),
                    "metrics": {
                        "bat_angle": temp_item.get("bat_angle") or 0.0,  # degrees for overlay
                        "hit_quality": temp_item.get("hit_quality") or 0.0,
                        "face_stability": temp_item.get("face_stability") or 0.0,
                        "stance_bend": temp_item.get("stance_bend") or 0.0,
                        "front_foot_stance_score": temp_item.get("front_foot_stance_score") or 0.0,
                        "shot_accuracy": temp_item.get("shot_accuracy") or 0.0,
                    },
                }
            )

        overall_out.release()
        cap_o.release()
        print("[INFO] Created merged overall highlights video")

    # Per-shot videos (overlay metrics)
    print("[INFO] Creating per-shot videos...")
    overall_path = os.path.join(output_folder, "merged_overall_highlights.mp4")
    if os.path.exists(overall_path) and segments_info:
        cap_s = cv2.VideoCapture(overall_path)
        fps_s = cap_s.get(cv2.CAP_PROP_FPS) or fps
        w_s = int(cap_s.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_s = int(cap_s.get(cv2.CAP_PROP_FRAME_HEIGHT))
        shot_writers = {}

        for seg in segments_info:
            shot = seg["shot_name"]
            start_frame = int(seg["merged_start_frame"])
            count = int(seg["merged_frame_count"])

            if player_type == "righty" and int(seg["cls_idx"]) not in constants.RIGHTY_CLASSES:
                continue
            if player_type == "lefty" and int(seg["cls_idx"]) not in constants.LEFTY_CLASSES:
                continue

            if shot not in shot_writers:
                out_path = os.path.join(output_folder, f"{shot}.mp4")
                shot_writers[shot] = cv2.VideoWriter(
                    out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_s, (w_s, h_s)
                )

            cap_s.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            metrics_panel = seg["metrics"]
            for _i in range(count):
                ret_f, frame_f = cap_s.read()
                if not ret_f:
                    break
                annotated = draw_metrics_overlay(frame_f, metrics_panel, w_s, h_s)
                shot_writers[shot].write(annotated)

        cap_s.release()
        for w in shot_writers.values():
            w.release()
        print(f"[INFO] Created {len(shot_writers)} per-shot videos")

    # Output JSONs (contacts.json + predicted.json) with LLM feedback
    print("[INFO] Creating output JSON files...")
    compact_contacts = []
    results_map = {int(r[0]): (int(r[1]), float(r[2])) for r in results} if results else {}

    for item in contact_data:
        cf = int(item.get("frame_highlight", -1))
        start_frame = max(0, cf - half_window) if cf >= 0 else -1

        shot_type = None
        conf_01 = 0.0
        if cf in results_map:
            cls_idx, conf_01 = results_map[cf]
            shot_type = str(constants.LABEL_CLASSES[cls_idx])

        bat_angle_deg = item.get("bat_angle", None)
        bat_angle_percent = item.get("bat_angle_percent", 0.0)

        compact = {
            "start_frame": int(start_frame),
            "frame_highlight": int(cf),
            "video_dimensions": {"width": int(width_o), "height": int(height_o)},
            "shot_type": shot_type,
            # JSON: main bat_angle is percent, degrees kept separately
            "bat_angle": bat_angle_percent,
            "bat_angle_deg": bat_angle_deg,
            "hit_quality": item.get("hit_quality", 0.0),
            "face_stability": item.get("face_stability", 0.0),
            "stance_bend": item.get("stance_bend", 0.0),
            "front_foot_stance_score": item.get("front_foot_stance_score", 0.0),
            "shot_accuracy": item.get("shot_accuracy", 0),
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
    print(f"[INFO] Saved contacts.json with {len(compact_contacts)} entries")

    predicted_counts = {}
    for _, cls_idx, _conf in results:
        shot_name = str(constants.LABEL_CLASSES[int(cls_idx)])
        predicted_counts[shot_name] = predicted_counts.get(shot_name, 0) + 1

    predicted_path = os.path.join(output_folder, "predicted.json")
    with open(predicted_path, "w") as f:
        json.dump(predicted_counts, f, indent=2)
    print(f"[INFO] Saved predicted.json with {len(predicted_counts)} unique shots (with counts)")

    return output_folder


# ----------------------------
# Serverless entry (same structure as process.py)
# ----------------------------
def classify_video(payload: dict):
    """
    RunPod Serverless classify:
    - Input: Base64 ZIP (must contain highlight.mp4 + contact_info.json)
    - Output: Base64 ZIP (classify results) with same "walk & zip" style
    """

    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")

    # Accept multiple possible keys (keep Flutter compatible)
    input_zip_b64 = (
        payload.get("input_zip")
        or payload.get("input_zip_b64")
        or payload.get("input_zip_base64")
        or payload.get("process_zip_b64")
        or payload.get("process_zip_base64")
        or payload.get("output_zip_b64")
        or payload.get("output_zip_base64")
    )
    if not input_zip_b64:
        raise ValueError("Missing required field: input_zip (base64)")

    job_id = payload.get("job_id") or f"job_{int(time.time())}"
    player_type = payload.get("player_type") or payload.get("handedness") or "righty"
    if player_type not in ("righty", "lefty"):
        raise ValueError("player_type must be 'righty' or 'lefty'")

    tmp_root = os.path.join("/tmp", job_id, "classify_tmp")
    _safe_mkdir(tmp_root)

    # Work dirs similar pattern to flask, but serverless-safe
    job_folder = os.path.join(constants.PROCESSED_FOLDER, job_id)
    work_root = os.path.join(job_folder, "classify_work")
    out_dir = os.path.join(job_folder, "classify")
    _safe_mkdir(work_root)
    _safe_mkdir(out_dir)

    # Write input zip and unzip
    temp_zip = os.path.join(tmp_root, "input.zip")
    _b64_to_file(input_zip_b64, temp_zip)

    with zipfile.ZipFile(temp_zip, "r") as z:
        z.extractall(work_root)

    # Find files inside extracted
    video_path = None
    json_path = None

    candidates_video = ["highlight.mp4", "highlight_web.mp4"]

    for root, _dirs, files in os.walk(work_root):
        for f in files:
            low = f.lower()
            if low.endswith(".mp4") and (f in candidates_video or "highlight" in low):
                if video_path is None or os.path.basename(video_path) != "highlight.mp4":
                    video_path = os.path.join(root, f)
            if low == "contact_info.json":
                json_path = os.path.join(root, f)

    if video_path is None:
        raise FileNotFoundError("highlight.mp4 not found in input zip")
    if json_path is None:
        raise FileNotFoundError("contact_info.json not found in input zip")

    print(f"🚀 Classify job: {job_id} | player_type={player_type}")
    print(f"   video_path={video_path}")
    print(f"   json_path={json_path}")

    # Ensure models are loaded (handler.py should do helper.load_models(); models live in constants.*)
    if (
        constants.yolo_model is None
        or constants.vitpose is None
        or constants.lstm_model is None
        or constants.tokenizer is None
        or constants.t5_model is None
    ):
        raise RuntimeError("models_not_loaded (check helper.load_models() in handler startup)")

    # Run pipeline
    t0 = time.time()
    process_cricket_shot_classification(
        video_path=video_path,
        json_path=json_path,
        player_type=player_type,
        output_folder=out_dir,
    )

    # Convert all mp4s before zipping (same as Flask logic)
    for root, _dirs, files in os.walk(out_dir):
        for f in files:
            if f.endswith(".mp4"):
                src = os.path.join(root, f)
                tmp_conv = os.path.join(root, f"converted_{f}")
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    src,
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-movflags",
                    "+faststart",
                    tmp_conv,
                ]
                try:
                    subprocess.run(ffmpeg_cmd, check=True)
                    os.remove(src)
                    os.rename(tmp_conv, src)
                except Exception as e:
                    print(f"[WARN] ffmpeg convert failed for {src}: {e}")
                    _cleanup_paths(tmp_conv)

    # Zip results (relative to out_dir)
    zip_path = os.path.join(out_dir, "classify_results.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _dirs, files in os.walk(out_dir):
            for f in files:
                if f.endswith(".zip"):
                    continue
                absf = os.path.join(root, f)
                relf = os.path.relpath(absf, out_dir)
                z.write(absf, relf)

    elapsed = time.time() - t0
    print(f"✅ classify_results.zip ready: {zip_path} ({elapsed:.1f}s)")

    # Compute count BEFORE cleanup (critical fix)
    try:
        predictions_count = len([f for f in os.listdir(out_dir)]) if os.path.exists(out_dir) else 0
    except Exception:
        predictions_count = 0

    # Read zip b64
    zip_b64 = _file_to_b64(zip_path)

    # Cleanup at the end of request
    _cleanup_paths(job_folder, tmp_root)

    return {
        "job_id": job_id,
        "player_type": player_type,
        "output_zip_base64": zip_b64,
        "output_zip_b64": zip_b64,
        "classify_zip_b64": zip_b64,
        "predictions_count": int(predictions_count),
    }
