# src/routes/process.py
import base64
import json
import os
import shutil
import subprocess
import time
import zipfile
from collections import deque

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

import constants
import helper


def process_video(payload: dict):
    """
    RunPod Serverless process:
    - Input: Base64 ZIP (contains an .mp4)
    - Output: Base64 ZIP with:
        - highlight.mp4
        - contact_info.json
    Cleans up temp files immediately after base64 is computed.
    """

    # ---- Input checks ----
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")

    # Accept multiple client key names (keep Flutter code compatible)
    video_zip_b64 = (
        payload.get("video_zip")
        or payload.get("video_zip_b64")
        or payload.get("video_zip_base64")
    )
    if not video_zip_b64:
        raise ValueError("Missing required field: video_zip (base64)")

    job_id = payload.get("job_id") or f"job_{int(time.time())}"

    # Serverless temp roots
    tmp_root = os.path.join("/tmp", job_id, "process_tmp")
    os.makedirs(tmp_root, exist_ok=True)

    # Persistent-ish job folder under /tmp (your constants points there)
    job_folder = os.path.join(constants.PROCESSED_FOLDER, job_id)
    os.makedirs(job_folder, exist_ok=True)

    # Ensure models loaded (handler.py should do this at startup)
    if constants.ball_model is None:
        raise RuntimeError("ball_model_not_loaded")
    if constants.bat_model is None:
        raise RuntimeError("bat_model_not_loaded")

    # ---- Decode + unzip input ----
    temp_zip = os.path.join(tmp_root, "input.zip")
    with open(temp_zip, "wb") as f:
        f.write(base64.b64decode(video_zip_b64))

    with zipfile.ZipFile(temp_zip, "r") as zip_ref:
        zip_ref.extractall(tmp_root)
        inner_files = [n for n in zip_ref.namelist() if not n.endswith("/")]

    if not inner_files:
        _cleanup_safely(job_folder, tmp_root)
        raise ValueError("empty_zip")

    # Pick first mp4 if present
    inner_name = None
    for n in inner_files:
        if n.lower().endswith(".mp4"):
            inner_name = n
            break
    if inner_name is None:
        inner_name = inner_files[0]

    video_path = os.path.join(tmp_root, inner_name)
    if not os.path.exists(video_path):
        video_path = os.path.join(tmp_root, os.path.basename(inner_name))
    if not os.path.exists(video_path):
        _cleanup_safely(job_folder, tmp_root)
        raise FileNotFoundError("Video file not found after unzip")

    # Move input video into job folder (so cv2 reads from stable path)
    filename_noext = os.path.splitext(os.path.basename(video_path))[0]
    final_video_path = os.path.join(job_folder, os.path.basename(video_path))
    os.rename(video_path, final_video_path)

    print(f"🚀 Processing video: {filename_noext} -> Job ID: {job_id}")

    # ---- Output paths ----
    contact_frames_root = os.path.join(job_folder, "frames")
    highlight_out_path = os.path.join(job_folder, "highlight.mp4")
    os.makedirs(contact_frames_root, exist_ok=True)

    cap = cv2.VideoCapture(final_video_path)
    if not cap.isOpened():
        _cleanup_safely(job_folder, tmp_root)
        raise RuntimeError("could_not_open_video")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    crop_size_px = min(orig_h, orig_w)
    crop_x1 = (orig_w - crop_size_px) // 2
    crop_y1 = (orig_h - crop_size_px) // 2
    scale_from_640_to_crop = crop_size_px / float(constants.CROP_SIZE)

    last_ball = deque(maxlen=2)
    last_bat = deque(maxlen=2)
    contacts = []
    last_contact_frame = -9999
    skip_until = -1

    ball_visible_frames = 0
    ball_missing_frames = 0
    linger_counter = 0
    ball_active = False

    frame_buffer = deque(maxlen=constants.PRE_FRAMES)
    post_frames_left = 0
    highlight_writer = None
    last_written_idx = -1

    # ---- VideoWriter (try multiple codecs) ----
    for codec in ["avc1", "mp4v", "XVID", "H264"]:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer_try = cv2.VideoWriter(highlight_out_path, fourcc, fps, (orig_w, orig_h))
        if writer_try.isOpened():
            highlight_writer = writer_try
            print(f"[INFO] Highlight writer opened with codec '{codec}'")
            break
        writer_try.release()

    if highlight_writer is None:
        print("[WARN] Could not open any VideoWriter codec. Highlight will be skipped.")

    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_buffer.append((frame_idx, frame.copy()))

            # Skip window after contact (min gap)
            if frame_idx > last_contact_frame and frame_idx <= skip_until:
                if post_frames_left > 0 and highlight_writer:
                    highlight_writer.write(frame)
                    post_frames_left -= 1
                    last_written_idx = frame_idx
                frame_idx += 1
                continue

            cropped = helper.adaptive_square_crop(frame, target_size=constants.CROP_SIZE)
            balls_current, bats_current = [], []

            # -------------------------
            # BALL DETECTION (YOLO)
            # -------------------------
            try:
                ball_results = constants.ball_model(
                    cropped,
                    conf=constants.CONF_THRESH,
                    iou=constants.IOU,
                    classes=[0],
                )
                if ball_results and len(ball_results) > 0:
                    r0 = ball_results[0]
                    if hasattr(r0, "boxes") and r0.boxes is not None and len(r0.boxes) > 0:
                        boxes_xyxy = r0.boxes.xyxy.cpu().numpy()
                        confs = r0.boxes.conf.cpu().numpy()
                        for (x1, y1, x2, y2), conf in zip(boxes_xyxy, confs):
                            cx = int(round((x1 + x2) / 2.0))
                            cy = int(round((y1 + y2) / 2.0))
                            balls_current.append((cx, cy, float(conf)))
            except Exception as e:
                print(f"[WARN] Ball detection failed at frame {frame_idx}: {e}")

            # -------------------------
            # BALL STATE
            # -------------------------
            if balls_current:
                ball_visible_frames += 1
                ball_missing_frames = 0
            else:
                ball_missing_frames += 1
                ball_visible_frames = max(0, ball_visible_frames - 1)

            if ball_visible_frames >= constants.BALL_SEEN_FRAMES:
                ball_active = True
                linger_counter = constants.LINGER_FRAMES
            elif ball_missing_frames >= constants.BALL_MISS_FRAMES:
                if linger_counter > 0:
                    linger_counter -= 1
                    ball_active = True
                else:
                    ball_active = False

            # -------------------------
            # BAT DETECTION (only if ball active)
            # -------------------------
            if ball_active:
                try:
                    bat_results = constants.bat_model.predict(
                        source=cropped,
                        imgsz=constants.CROP_SIZE,
                        conf=constants.CONF_THRESH,
                        verbose=False,
                    )
                    if bat_results:
                        for r in bat_results:
                            obb_attr = getattr(r, "obb", None)
                            if obb_attr is not None and getattr(obb_attr, "xyxyxyxy", None) is not None:
                                obb_boxes = obb_attr.xyxyxyxy.cpu().numpy()
                                obb_confs = obb_attr.conf.cpu().numpy()
                                for box_flat, conf in zip(obb_boxes, obb_confs):
                                    pts = box_flat.reshape(4, 2).astype(int).tolist()
                                    bats_current.append((pts, float(conf)))
                            elif hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                                boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                                confs = r.boxes.conf.cpu().numpy()
                                for (x1, y1, x2, y2), conf in zip(boxes_xyxy, confs):
                                    pts = [
                                        [int(x1), int(y1)],
                                        [int(x2), int(y1)],
                                        [int(x2), int(y2)],
                                        [int(x1), int(y2)],
                                    ]
                                    bats_current.append((pts, float(conf)))
                except Exception as e:
                    print(f"[WARN] Bat detection failed at frame {frame_idx}: {e}")

            # -------------------------
            # PREDICTIVE BALL
            # -------------------------
            if not balls_current:
                if len(last_ball) >= 2:
                    (x1, y1, c1, f1), (x2, y2, c2, f2) = last_ball[0], last_ball[1]
                    dx, dy = x2 - x1, y2 - y1
                    pred_x, pred_y = int(round(x2 + dx)), int(round(y2 + dy))
                    pred_conf = float(c2) * 0.8
                    if pred_conf >= constants.CONF_THRESH:
                        balls_current.append((pred_x, pred_y, pred_conf))
                elif len(last_ball) == 1:
                    (x, y, c, f) = last_ball[-1]
                    pred_conf = float(c) * 0.9
                    if pred_conf >= constants.CONF_THRESH:
                        balls_current.append((int(x), int(y), pred_conf))

            # -------------------------
            # PREDICTIVE BAT
            # -------------------------
            if not bats_current and len(last_bat) > 0:
                if len(last_bat) >= 2:
                    (pts1, conf1, f1), (pts2, conf2, f2) = last_bat[0], last_bat[1]
                    cx1, cy1 = helper.polygon_centroid(pts1)
                    cx2, cy2 = helper.polygon_centroid(pts2)
                    dx, dy = cx2 - cx1, cy2 - cy1
                    pred_pts = helper.translate_polygon(pts2, dx, dy)
                    pred_conf = float(conf2) * 0.8
                    if pred_conf >= constants.CONF_THRESH:
                        bats_current.append((pred_pts, pred_conf))
                else:
                    pts, conf, f = last_bat[-1]
                    pred_conf = float(conf) * 0.9
                    if pred_conf >= constants.CONF_THRESH:
                        bats_current.append((pts, pred_conf))

            # -------------------------
            # CONTACT DETECTION
            # -------------------------
            contact_found = False
            contact_ball = None
            contact_bat = None

            if balls_current and bats_current:
                for cx, cy, bconf in balls_current:
                    ball_area = Point(cx, cy).buffer(constants.CONTACT_RADIUS)
                    for pts, bat_conf in bats_current:
                        poly = Polygon(pts)
                        if poly.is_valid and ball_area.intersects(poly):
                            contact_found = True
                            contact_ball = (cx, cy, float(bconf))
                            contact_bat = (pts, float(bat_conf))
                            break
                    if contact_found:
                        break

            # -------------------------
            # HANDLE CONTACT
            # -------------------------
            if contact_found and frame_idx > last_contact_frame + constants.CONTACT_MIN_GAP:
                ann = cropped.copy()

                if contact_bat:
                    pts, conf = contact_bat
                    cv2.polylines(ann, [np.array(pts, np.int32)], True, (0, 255, 0), 2)
                    cv2.putText(
                        ann,
                        f"{conf:.2f}",
                        (pts[0][0], max(0, pts[0][1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        1,
                    )

                if contact_ball:
                    cx, cy, bconf = contact_ball
                    cv2.circle(ann, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(
                        ann,
                        f"{bconf:.2f}",
                        (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 0, 255),
                        1,
                    )

                fname = os.path.join(contact_frames_root, f"contact_{frame_idx:06d}.jpg")
                cv2.imwrite(fname, ann)

                def map_to_original(x640, y640):
                    x_in_crop = int(round(x640 * scale_from_640_to_crop))
                    y_in_crop = int(round(y640 * scale_from_640_to_crop))
                    return int(crop_x1 + x_in_crop), int(crop_y1 + y_in_crop)

                mapped_ball = None
                if contact_ball:
                    bx, by, bconf = contact_ball
                    bx_o, by_o = map_to_original(bx, by)
                    mapped_ball = {
                        "x_640": bx,
                        "y_640": by,
                        "conf": bconf,
                        "x_orig": bx_o,
                        "y_orig": by_o,
                    }

                mapped_bat = None
                if contact_bat:
                    pts640, batconf = contact_bat
                    pts_orig = [[map_to_original(px, py)[0], map_to_original(px, py)[1]] for (px, py) in pts640]
                    mapped_bat = {
                        "pts_640": pts640,
                        "conf": batconf,
                        "pts_orig": pts_orig,
                    }

                frame_highlight = (
                    (len(contacts)) * (constants.PRE_FRAMES + constants.POST_FRAMES + 1)
                    + constants.PRE_FRAMES
                    if highlight_writer
                    else None
                )

                contacts.append(
                    {
                        "frame_idx": frame_idx,
                        "frame_highlight": frame_highlight,
                        "ball": mapped_ball,
                        "bat": mapped_bat,
                        "video_dimensions": {"width": orig_w, "height": orig_h},
                    }
                )

                if highlight_writer:
                    # write pre-buffer
                    for idx, buf_frame in list(frame_buffer):
                        if idx <= last_written_idx:
                            continue
                        highlight_writer.write(buf_frame)
                        last_written_idx = idx
                    # write contact frame
                    highlight_writer.write(frame)
                    last_written_idx = frame_idx
                    post_frames_left = constants.POST_FRAMES

                last_contact_frame = frame_idx
                skip_until = frame_idx + constants.CONTACT_MIN_GAP

            # Write post frames after contact
            if post_frames_left > 0 and highlight_writer:
                highlight_writer.write(frame)
                last_written_idx = frame_idx
                post_frames_left -= 1

            # Track last detections for prediction
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

    # ---- Save contact JSON ----
    json_path = os.path.join(contact_frames_root, "contact_info.json")
    with open(json_path, "w") as jf:
        json.dump(contacts, jf, indent=2)

    # ---- Convert highlight for compatibility (optional) ----
    highlight_fixed_path = os.path.join(job_folder, "highlight_web.mp4")
    if os.path.exists(highlight_out_path):
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            highlight_out_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            highlight_fixed_path,
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True)
        except Exception as e:
            print(f"[WARN] ffmpeg re-encode failed: {e}")
            highlight_fixed_path = None
    else:
        highlight_fixed_path = None

    # ---- Zip output ----
    zip_path = os.path.join(job_folder, "process.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        if highlight_fixed_path and os.path.exists(highlight_fixed_path):
            z.write(highlight_fixed_path, "highlight.mp4")
        elif os.path.exists(highlight_out_path):
            z.write(highlight_out_path, "highlight.mp4")
        if os.path.exists(json_path):
            z.write(json_path, "contact_info.json")

    print(f"✅ process.zip ready: {zip_path}")

    # ---- Encode output zip to base64 (response payload) ----
    with open(zip_path, "rb") as f:
        zip_b64 = base64.b64encode(f.read()).decode("utf-8")

    # ---- Cleanup after response content is ready ----
    _cleanup_safely(job_folder, tmp_root)

    return {
        "job_id": job_id,
        "output_zip_base64": zip_b64,
        "output_zip_b64": zip_b64,
        "process_zip_b64": zip_b64,
        "contacts_count": int(len(contacts)),
    }


def _cleanup_safely(job_folder: str, tmp_root: str) -> None:
    try:
        shutil.rmtree(job_folder, ignore_errors=True)
    except Exception as e:
        print(f"⚠ Cleanup warning (job_folder): {e}")

    try:
        shutil.rmtree(tmp_root, ignore_errors=True)
    except Exception as e:
        print(f"⚠ Cleanup warning (tmp_root): {e}")
