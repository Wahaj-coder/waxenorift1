import os

# Force offline mode (fail fast if any model is missing)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Optional: keep HF cache in writable temp
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("HF_HUB_CACHE", "/tmp/hf/hub")

from routes.classify import classify_video
from routes.process import process_video
from helper import *

try:
    load_models()
except Exception as _e:
    print(f"❌ Model load failed: {_e}")
    raise


def handler(event):
    """RunPod Serverless handler.

    Expected event:
      { "input": { "action": "process"|"classify", ... } }
    """
    try:
        payload = (event or {}).get("input") or {}
        action = payload.get("action", "process")

        if action == "process":
            return process_video(payload)
        elif action == "classify":
            return classify_video(payload)
        else:
            return {"error": "invalid_action", "detail": f"Unknown action: {action}"}
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"error": "internal_server_error", "detail": str(e)}


if __name__ == "__main__":
    # Models are loaded at import time. This is just the RunPod entrypoint.
    import runpod

    runpod.serverless.start({"handler": handler})
