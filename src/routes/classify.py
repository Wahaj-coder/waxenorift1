import base64
import glob
import os
import shutil
import subprocess
import time
import zipfile

from helper import *


def classify_video(payload: dict):
    """
    Serverless classification entry.

    Accepts either:
      - zip_base64 / zip_b64: a ZIP containing highlight.mp4 + contact_info.json (Flutter path)
      - process_zip_b64 / process_zip: same thing, different key name
      - (video_b64 + json_b64): raw base64 mp4 + json (fallback)
    Returns:
      - output_zip_base64: ZIP of classify outputs (includes merged_overall_highlights.mp4, contacts.json, predicted.json, etc.)
    """
    t0 = time.time()
    job_root = None
    tmp_root = None

    try:
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dict")

        job_id = payload.get("job_id") or f"job_{int(time.time())}"
        player_type = payload.get("player_type", "righty")
        if player_type not in ("righty", "lefty"):
            raise ValueError("player_type must be 'righty' or 'lefty'")

        # ---- Hard fail if critical models not loaded ----
        if lstm_model is None:
            raise RuntimeError("lstm_model_not_loaded")
        if yolo_model is None:
            raise RuntimeError("yolo_model_not_loaded")
        if vitpose is None:
            raise RuntimeError("vitpose_not_loaded")
        if tokenizer is None or t5_model is None:
            raise RuntimeError("t5_not_loaded")

        # Temp working area (serverless-friendly)
        tmp_root = os.path.join("/tmp", job_id)
        os.makedirs(tmp_root, exist_ok=True)

        # Inputs
        zip_b64 = (
            payload.get("zip_base64")
            or payload.get("zip_b64")
            or payload.get("process_zip_b64")
            or payload.get("process_zip")
        )
        video_b64 = payload.get("video_b64") or payload.get("video_mp4_b64")
        json_b64 = payload.get("json_b64") or payload.get("contact_json_b64")

        if zip_b64:
            inzip = os.path.join(tmp_root, "classify_input.zip")
            with open(inzip, "wb") as f:
                f.write(base64.b64decode(zip_b64))

            with zipfile.ZipFile(inzip, "r") as z:
                z.extractall(tmp_root)

            # expected filenames
            cand_video = os.path.join(tmp_root, "highlight.mp4")
            cand_json = os.path.join(tmp_root, "contact_info.json")

            # fallback search if names differ
            if not os.path.exists(cand_video):
                mp4s = glob.glob(os.path.join(tmp_root, "*.mp4"))
                if mp4s:
                    cand_video = mp4s[0]
            if not os.path.exists(cand_json):
                js = glob.glob(os.path.join(tmp_root, "*.json"))
                if js:
                    cand_json = js[0]

            if not os.path.exists(cand_video):
                raise FileNotFoundError("highlight.mp4 missing in uploaded zip")
            if not os.path.exists(cand_json):
                raise FileNotFoundError("contact_info.json missing in uploaded zip")

            video_path = cand_video
            json_path = cand_json

        else:
            if not (video_b64 and json_b64):
                raise ValueError(
                    "Provide either zip_base64 OR (video_b64 and json_b64)"
                )

            video_path = os.path.join(tmp_root, "highlight.mp4")
            with open(video_path, "wb") as f:
                f.write(base64.b64decode(video_b64))

            json_path = os.path.join(tmp_root, "contact_info.json")
            with open(json_path, "wb") as f:
                f.write(base64.b64decode(json_b64))

        # ---- Work/output dirs (use /tmp) ----
        job_root = os.path.join(PROCESSED_FOLDER, job_id)
        work_root = os.path.join(job_root, "classify_work")
        out_dir = os.path.join(job_root, "classify")
        os.makedirs(work_root, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        # Keep your file naming intact
        local_video = os.path.join(work_root, "highlight.mp4")
        local_json = os.path.join(work_root, "contact_info.json")
        shutil.copy2(video_path, local_video)
        shutil.copy2(json_path, local_json)

        # Run your existing classification pipeline (UNCHANGED)
        output_dir = process_cricket_shot_classification(
            video_path=local_video,
            json_path=local_json,
            player_type=player_type,
            output_folder=out_dir,
        )

        # Convert mp4 outputs for broad compatibility (unchanged behavior)
        for root, _, files in os.walk(out_dir):
            for file in files:
                if file.endswith(".mp4"):
                    video_file_path = os.path.join(root, file)
                    converted_video_path = os.path.join(root, f"converted_{file}")
                    ffmpeg_cmd = [
                        "ffmpeg",
                        "-y",
                        "-i",
                        video_file_path,
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        "-movflags",
                        "+faststart",
                        converted_video_path,
                    ]
                    subprocess.run(ffmpeg_cmd, check=True)
                    os.remove(video_file_path)
                    os.rename(converted_video_path, video_file_path)

        # Zip outputs (keeps your names/structure inside output_dir)
        zip_path = os.path.join(out_dir, "classify_results.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(output_dir):
                for f in files:
                    if f.endswith(".zip"):
                        continue
                    absf = os.path.join(root, f)
                    relf = os.path.relpath(absf, output_dir)
                    z.write(absf, relf)

        if not os.path.exists(zip_path):
            raise RuntimeError("classify_results.zip missing")

        with open(zip_path, "rb") as f:
            zip_out_b64 = base64.b64encode(f.read()).decode("utf-8")

        elapsed = time.time() - t0
        print(f"✅ Classify job {job_id} completed in {elapsed:.1f}s")

        # Manual cleanup (no background threads)
        try:
            shutil.rmtree(job_root, ignore_errors=True)
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception as _e:
            print(f"⚠ Cleanup warning: {_e}")

        return {
            "job_id": job_id,
            # preferred keys (for your Flutter code)
            "output_zip_base64": zip_out_b64,
            "output_zip_b64": zip_out_b64,
            # keep older keys too
            "classify_zip_b64": zip_out_b64,
            "elapsed_sec": round(elapsed, 2),
        }

    except Exception as e:
        print(f"🔥 CLASSIFY ERROR: {e}")
        import traceback

        traceback.print_exc()
        # try cleanup even on error
        try:
            if job_root:
                shutil.rmtree(job_root, ignore_errors=True)
            if tmp_root:
                shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            pass
        return {"error": str(e)}
