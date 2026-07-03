# Smart Child Monitor

This folder contains a child safety monitor built on MediaPipe Tasks Pose Landmarker.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use Python 3.11 or 3.12 for the cleanest MediaPipe install. The project is pinned to `mediapipe==0.10.35`, which is the current package version available to this environment through pip.

Download the official Pose Landmarker model:

```bash
python download_pose_model.py
```

By default this downloads the Heavy model for higher accuracy:

```text
models/pose_landmarker_heavy.task
```

For faster inference, download another variant and update `CONFIG["MODEL_PATH"]`:

```bash
POSE_MODEL_VARIANT=full python download_pose_model.py
POSE_MODEL_VARIANT=lite python download_pose_model.py
```

If MediaPipe import errors appear, run:

```bash
python check_mediapipe_env.py
```

Then recreate the venv with Python 3.11 or 3.12 and reinstall:

```bash
python -m pip uninstall -y mediapipe
python -m pip install -r requirements.txt
```

## Run

```bash
python smart_child_monitor.py
```

Test with a local video file instead of ONVIF or a camera:

```bash
python smart_child_monitor.py --video "/path/to/test-video.mp4"
```

The app processes every frame and exits automatically at the end of the file. To replay the
same test video continuously, add `--loop`:

```bash
python smart_child_monitor.py --video "/path/to/test-video.mp4" --loop
```

Press `q` in the video window to stop at any time. Alert clips are still written to
`security_alerts/` during local-video tests.

By default the app tries to use an ONVIF network camera as the primary camera:

```python
"USE_ONVIF_CAMERA": True
```

Set the camera login before running:

```bash
export ONVIF_USERNAME=admin
export ONVIF_PASSWORD='XXXXX'
python smart_child_monitor.py
```

Startup flow:

```text
ONVIF WS-Discovery -> ONVIF Media GetStreamUri -> RTSP URL -> OpenCV/FFmpeg capture
```

If the RTSP URL returned by ONVIF does not include credentials, the app injects `ONVIF_USERNAME` and `ONVIF_PASSWORD` for OpenCV. Logs redact the password.

If multiple ONVIF cameras are on the LAN, choose one:

```bash
export ONVIF_PREFERRED_HOST=192.168.1.64
```

If ONVIF discovery or RTSP retrieval fails, the app falls back to `CONFIG["VIDEO_SOURCE"]`. You can disable ONVIF and use a local webcam by setting:

```python
"USE_ONVIF_CAMERA": False,
"VIDEO_SOURCE": 0,
```

The detector uses the newer MediaPipe Tasks `PoseLandmarker` in video mode, not the old `mp.solutions.pose` API. It detects up to four people, assigns temporary on-screen IDs (`P1`, `P2`, ...), and keeps fall timers and alert cooldowns separate for each person. It reads 33 pose landmarks per person with visibility and presence confidence, then evaluates fall, climb, and danger-zone intrusion rules using both left and right body keypoints.

Multi-person limits and matching sensitivity can be tuned in `CONFIG`:

```python
"MAX_POSES": 4,
"PERSON_TRACK_MAX_DISTANCE": 0.18,
"PERSON_TRACK_TIMEOUT": 2.0,
```

The live window and saved alert clips show the reason for each danger judgment, including trigger values such as trunk angle, body height/width ratio, danger-line position, or calibrated ground position.

## Real-World Danger Zones

The old 2D wrist rectangle is disabled by default:

```python
"ENABLE_2D_DANGER_ZONE": False
```

### Semi-automatic four-point calibration

Run the calibration tool with the measured width and depth of a rectangular floor area:

```bash
python smart_child_monitor.py --calibrate --ground-width 3.0 --ground-depth 2.0
```

To calibrate from a local video instead of the primary camera:

```bash
python smart_child_monitor.py --video "/path/to/room.mp4" --calibrate \
  --ground-width 3.0 --ground-depth 2.0
```

Press Space to freeze a useful frame. Then left-click the four floor corners in this order:
`LEFT-NEAR`, `RIGHT-NEAR`, `RIGHT-FAR`, `LEFT-FAR`. Right-click removes the last point,
`R` resets all points, Enter validates and saves, and `Q` cancels.

The result is saved as `calibration.json` and loaded automatically on later normal runs. Camera
position, zoom, resolution, or stream profile changes require a new calibration.

For real rooms, use calibrated ground-plane zones. This maps camera pixels to real floor coordinates in meters, then checks whether the child's feet enter a real danger area:

```python
"ENABLE_GROUND_PLANE_ZONE": True,
"GROUND_PLANE_IMAGE_POINTS": [[120, 420], [620, 410], [760, 700], [40, 710]],
"GROUND_PLANE_WORLD_POINTS_M": [[0, 0], [2.4, 0], [2.4, 1.8], [0, 1.8]],
"GROUND_DANGER_ZONES_M": [
    {"name": "stairs", "polygon": [[1.6, 0.2], [2.4, 0.2], [2.4, 1.4], [1.6, 1.4]]}
],
```

This is appropriate for floor hazards such as stairs, balcony thresholds, kitchen entrances, or fireplace areas. It is not enough for vertical objects such as stove burners, windowsills, sockets, or table edges; those need either depth sensing, object detection, or camera-specific calibration for the object plane.

If ground-plane detection is enabled before both sets of four calibration points are configured,
the app logs a warning and skips only ground-zone detection for that run. Fall, climb, and other
enabled checks continue normally.

If you see a Metal/OpenGL error such as `DrishtiMetalHelper` or `NSOpenGLPixelFormat` while running from an automated shell, run the script from a normal macOS Terminal session. The app is configured with CPU delegate, but MediaPipe Tasks may still initialize platform graphics services internally on macOS.

## Alert Clips

When an alert is triggered, the script saves an MP4 clip to `security_alerts/`.

You can tune clip length and frame rate in `CONFIG`:

```python
"PRE_ALERT_SECONDS": 3.0,
"POST_ALERT_SECONDS": 5.0,
"RECORD_FPS": 15.0,
```

To keep long-running monitoring memory bounded, incoming 4K streams are resized to at most
1280x720 and alert buffers are stored as compressed JPEG frames. These limits can be tuned in
`CONFIG`, but increasing them also increases pose inference load and memory use:

```python
"MAX_FRAME_WIDTH": 1280,
"MAX_FRAME_HEIGHT": 720,
"ALERT_JPEG_QUALITY": 80,
"ALERT_QUEUE_MAXSIZE": 2,
"MAX_PENDING_CLIPS": 2,
```
