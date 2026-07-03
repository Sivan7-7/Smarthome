import os
import urllib.request


MODEL_URLS = {
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}

MODEL_VARIANT = os.environ.get("POSE_MODEL_VARIANT", "heavy").lower()
if MODEL_VARIANT not in MODEL_URLS:
    raise SystemExit(f"Unknown POSE_MODEL_VARIANT={MODEL_VARIANT}. Use: heavy, full, or lite.")

os.makedirs("models", exist_ok=True)
filename = f"pose_landmarker_{MODEL_VARIANT}.task"
target_path = os.path.join("models", filename)

if os.path.exists(target_path):
    print(f"Model already exists: {target_path}")
    raise SystemExit(0)

print(f"Downloading {MODEL_VARIANT} model...")
print(MODEL_URLS[MODEL_VARIANT])
urllib.request.urlretrieve(MODEL_URLS[MODEL_VARIANT], target_path)
print(f"Saved: {target_path}")
