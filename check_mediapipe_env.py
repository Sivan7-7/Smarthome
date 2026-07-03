import importlib
import os
import sys


os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib_cache"))

print("Python executable:", sys.executable)
print("Python version:", sys.version)

try:
    mp = importlib.import_module("mediapipe")
except ImportError as exc:
    print("mediapipe import: FAILED")
    print(exc)
    raise SystemExit(1)

print("mediapipe version:", getattr(mp, "__version__", "unknown"))
print("mediapipe file:", getattr(mp, "__file__", "unknown"))
print("has mediapipe.solutions attr:", hasattr(mp, "solutions"))
print("has mediapipe.tasks attr:", hasattr(mp, "tasks"))

try:
    print("PoseLandmarker:", mp.tasks.vision.PoseLandmarker)
except Exception as exc:
    print("PoseLandmarker: FAILED - " + repr(exc))

for module_name in (
    "mediapipe.solutions.pose",
    "mediapipe.solutions.drawing_utils",
    "mediapipe.python.solutions.pose",
    "mediapipe.python.solutions.drawing_utils",
):
    try:
        importlib.import_module(module_name)
        print(module_name + ": OK")
    except Exception as exc:
        print(module_name + ": FAILED - " + repr(exc))
