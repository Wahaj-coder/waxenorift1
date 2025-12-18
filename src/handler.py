# handler.py

# Force TensorFlow to CPU before any TF/Keras imports
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import os
import traceback

# Force offline mode (fail fast if any model is missing)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Optional: keep HF cache in writable temp
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("HF_HUB_CACHE", "/tmp/hf/hub")

# Import routes
from routes.classify import classify_video
from routes.process import process_video
import helper

# Load all models at startup (fail-fast)
try:
    print("🔄 Loading models...")
    helper.load_models()
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    traceback.print_exc()
    raise

# Safe RunPod import
try:
    import runpod
except ModuleNotFoundError:
    runpod = None
    print("⚠ RunPod module not found, skipping serverless startup")

def handler(event):
    """RunPod Serverless handler.

    Expected event:
      { "input": { "action": "process"|"classify", ... } }
    """
    try:
        payload = (event or {}).get("input") or {}
        action = payload.get("action", "process")

        print(f"📌 Received event. Action: {action}")

        if action == "process":
            result = process_video(payload)
            print(f"✅ Video processed successfully. Job ID: {result.get('job_id')}")
            return result

        elif action == "classify":
            result = classify_video(payload)
            print(f"✅ Video classified successfully. Job ID: {result.get('job_id')}")
            return result

        else:
            msg = f"Unknown action: {action}"
            print(f"❌ {msg}")
            return {"error": "invalid_action", "detail": msg}

    except Exception as e:
        print(f"❌ Internal server error: {e}")
        traceback.print_exc()
        return {"error": "internal_server_error", "detail": str(e)}

# Only start serverless if runpod module exists
if __name__ == "__main__" and runpod:
    print("🚀 Starting RunPod serverless...")
    runpod.serverless.start({"handler": handler})
