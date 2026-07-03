import cv2
import numpy as np
import argparse
import json
import time
import os
import socket
import threading
import uuid
from collections import deque
from queue import Queue, Empty, Full
import logging
from urllib.parse import quote, urlparse, urlunparse
import xml.etree.ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib_cache"))

import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

POSE_CONNECTIONS = (
    (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST), (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP), (LEFT_HIP, RIGHT_HIP), (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE), (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
)

# ==========================================
# 1. 生产级日志与全局配置
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(threadName)s: %(message)s')

CONFIG = {
    "VIDEO_SOURCE": 0,                       # 可改为 "rtsp://admin:pwd@192.168.1.64:554/h264"
    "USE_ONVIF_CAMERA": True,                # 优先自动发现 ONVIF 监控摄像头并获取 RTSP
    "ONVIF_USERNAME": os.getenv("ONVIF_USERNAME", "admin"),
    "ONVIF_PASSWORD": os.getenv("ONVIF_PASSWORD", "XXXXXX"),
    "ONVIF_PREFERRED_HOST": os.getenv("ONVIF_PREFERRED_HOST", ""), # 多摄像头时指定 IP
    "ONVIF_DISCOVERY_TIMEOUT": 3.0,
    "ONVIF_STREAM_PROTOCOL": "RTSP",
    "OPENCV_FFMPEG_CAPTURE_OPTIONS": "rtsp_transport;tcp|stimeout;5000000|max_delay;500000",
    "MODEL_PATH": "models/pose_landmarker_heavy.task", # 官方 Pose Landmarker Heavy 模型
    "DELEGATE": "CPU",                       # macOS 摄像头场景优先 CPU，避免 OpenGL 初始化失败
    "DANGER_LINE_Y_PCT": 0.35,               # 警戒线位置
    "ENABLE_2D_DANGER_ZONE": False,        # 旧版二维手腕禁区仅用于调试，真实部署不建议开启
    "DANGER_ZONE_PCT": [0.1, 0.1, 0.5, 0.5], # 调试用二维区域 [x1, y1, x2, y2]
    "ENABLE_GROUND_PLANE_ZONE": True,       # 真实地面区域检测：需先完成相机地面标定
    "GROUND_PLANE_IMAGE_POINTS": [],         # 画面中地面四点，顺序需对应 WORLD_POINTS，例如 [[120,420], ...]
    "GROUND_PLANE_WORLD_POINTS_M": [],       # 地面四点真实坐标（米），例如 [[0,0], [2,0], [2,1.5], [0,1.5]]
    "GROUND_DANGER_ZONES_M": [],             # 米制危险地面多边形，例如 [{"name": "stairs", "polygon": [[1,0], [2,0], [2,1], [1,1]]}]
    "CALIBRATION_FILE": os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json"),
    "SAVE_DIR": "security_alerts",           # 报警视频存储路径
    "QUEUE_MAXSIZE": 3,                      # 极致实时性缓冲区
    "ALERT_QUEUE_MAXSIZE": 2,                # 限制等待写盘的视频，避免慢磁盘造成内存积压
    "MAX_PENDING_CLIPS": 2,                  # 同时采集中的报警片段上限
    "MAX_FRAME_WIDTH": 1280,                 # 采集后立即等比缩小，防止 4K 原始帧占满内存
    "MAX_FRAME_HEIGHT": 720,
    "ALERT_JPEG_QUALITY": 80,                # 内存中的报警帧使用 JPEG 压缩，写 MP4 时再解码
    "PRE_ALERT_SECONDS": 3.0,                # 报警前视频缓存时长
    "POST_ALERT_SECONDS": 5.0,               # 报警后继续录制时长
    "RECORD_FPS": 15.0,                      # 报警视频保存帧率

    # 儿童行为专项业务参数
    "MIN_LANDMARK_CONFIDENCE": 0.5,          # 关键点可见度阈值
    "MIN_POSE_DETECTION_CONFIDENCE": 0.65,   # 人体检测置信度阈值
    "MIN_POSE_PRESENCE_CONFIDENCE": 0.65,    # 姿态存在置信度阈值
    "MIN_TRACKING_CONFIDENCE": 0.65,         # 时序跟踪置信度阈值
    "MAX_POSES": 2,                         # 单帧最多检测人数
    "PERSON_TRACK_MAX_DISTANCE": 0.18,       # 身体中心跨帧匹配的最大归一化距离
    "PERSON_TRACK_TIMEOUT": 2.0,             # 人员消失多久后释放其临时 ID（视频时间秒）
    "FALL_ANGLE_THRESHOLD": 60,              # 躯干倾斜角阈值（度）
    "FALL_CONFIRM_DURATION": 1.5,            # 跌倒后在地面滞留的确认时间（秒）
    "HEIGHT_WIDTH_RATIO_LIMIT": 0.8,         # 外接矩形高宽比阈值
    "ALERT_COOLDOWN": 5.0,                   # 同类报警节流阀冷却时间（秒）
    "SHOW_WINDOW": True                      # 是否开启可视化窗口（部署到服务器时可设为False）
}

if not os.path.exists(CONFIG["SAVE_DIR"]):
    os.makedirs(CONFIG["SAVE_DIR"])


# ==========================================
# 2. ONVIF 自动发现与 RTSP 地址解析
# ==========================================
def discover_onvif_devices(timeout=3.0):
    probe_id = f"uuid:{uuid.uuid4()}"
    probe = f"""<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
            xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
            xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
            xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
  <e:Header>
    <w:MessageID>{probe_id}</w:MessageID>
    <w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
    <w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
  </e:Header>
  <e:Body>
    <d:Probe>
      <d:Types>dn:NetworkVideoTransmitter</d:Types>
    </d:Probe>
  </e:Body>
</e:Envelope>""".encode("utf-8")

    devices = []
    seen_xaddrs = set()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(timeout)

    try:
        sock.sendto(probe, ("239.255.255.250", 3702))
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                break

            xaddrs = extract_ws_discovery_xaddrs(data)
            for xaddr in xaddrs:
                if xaddr in seen_xaddrs:
                    continue
                seen_xaddrs.add(xaddr)
                parsed = urlparse(xaddr)
                devices.append({
                    "host": parsed.hostname or addr[0],
                    "port": parsed.port or 80,
                    "xaddr": xaddr,
                    "remote_addr": addr[0],
                })
    finally:
        sock.close()

    return devices


def extract_ws_discovery_xaddrs(data):
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return []

    xaddrs = []
    for elem in root.iter():
        if elem.tag.endswith("XAddrs") and elem.text:
            xaddrs.extend(elem.text.split())
    return xaddrs


def get_onvif_rtsp_uri(host, port, username, password):
    try:
        from onvif import ONVIFCamera
    except ImportError as exc:
        raise ImportError(
            "ONVIF 自动取流需要安装 onvif-zeep。请运行: python -m pip install -r requirements.txt"
        ) from exc

    camera = ONVIFCamera(host, port, username, password)
    media_service = camera.create_media_service()
    profiles = media_service.GetProfiles()
    if not profiles:
        raise RuntimeError("ONVIF 设备没有返回媒体 Profile。")

    profile = profiles[0]
    request = media_service.create_type("GetStreamUri")
    request.StreamSetup = {
        "Stream": "RTP-Unicast",
        "Transport": {"Protocol": CONFIG["ONVIF_STREAM_PROTOCOL"]},
    }
    request.ProfileToken = profile.token
    stream_uri = media_service.GetStreamUri(request)
    return with_rtsp_credentials(stream_uri.Uri, username, password)


def with_rtsp_credentials(rtsp_uri, username, password):
    if not username or not password:
        return rtsp_uri

    parsed = urlparse(rtsp_uri)
    if parsed.scheme.lower() != "rtsp" or parsed.username:
        return rtsp_uri

    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    userinfo = f"{quote(username)}:{quote(password)}"
    return urlunparse(parsed._replace(netloc=f"{userinfo}@{host}"))


def redact_uri(uri):
    parsed = urlparse(uri)
    if not parsed.password:
        return uri

    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    username = parsed.username or ""
    return urlunparse(parsed._replace(netloc=f"{username}:***@{host}"))


def resolve_video_source():
    if not CONFIG["USE_ONVIF_CAMERA"]:
        return CONFIG["VIDEO_SOURCE"]

    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", CONFIG["OPENCV_FFMPEG_CAPTURE_OPTIONS"])
    logging.info("开始 ONVIF 自动发现监控摄像头...")
    devices = discover_onvif_devices(CONFIG["ONVIF_DISCOVERY_TIMEOUT"])
    if CONFIG["ONVIF_PREFERRED_HOST"]:
        devices.sort(key=lambda d: d["host"] != CONFIG["ONVIF_PREFERRED_HOST"])

    for device in devices:
        try:
            logging.info(f"发现 ONVIF 设备: {device['host']}:{device['port']} ({device['xaddr']})")
            rtsp_uri = get_onvif_rtsp_uri(
                device["host"],
                device["port"],
                CONFIG["ONVIF_USERNAME"],
                CONFIG["ONVIF_PASSWORD"],
            )
            logging.info(f"已获取 ONVIF RTSP 地址: {redact_uri(rtsp_uri)}")
            return rtsp_uri
        except Exception as exc:
            logging.warning(f"ONVIF 设备 {device['host']} 获取 RTSP 失败: {exc}")

    logging.warning(f"ONVIF 自动发现/取流失败，回退到 VIDEO_SOURCE={CONFIG['VIDEO_SOURCE']}")
    return CONFIG["VIDEO_SOURCE"]


# ==========================================
# 3. 半自动地面标定
# ==========================================
CALIBRATION_POINT_LABELS = ("LEFT-NEAR", "RIGHT-NEAR", "RIGHT-FAR", "LEFT-FAR")


def load_ground_calibration():
    calibration_file = CONFIG["CALIBRATION_FILE"]
    if not os.path.isfile(calibration_file):
        return False

    try:
        with open(calibration_file, "r", encoding="utf-8") as file:
            calibration = json.load(file)
        image_points = calibration["image_points"]
        world_points = calibration["world_points_m"]
        if len(image_points) != 4 or len(world_points) != 4:
            raise ValueError("标定文件必须包含两组各 4 个点")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        logging.warning(f"无法加载地面标定文件 {calibration_file}: {exc}")
        return False

    CONFIG["GROUND_PLANE_IMAGE_POINTS"] = image_points
    CONFIG["GROUND_PLANE_WORLD_POINTS_M"] = world_points
    saved_danger_zones = calibration.get("danger_zones_m")
    if saved_danger_zones:
        CONFIG["GROUND_DANGER_ZONES_M"] = saved_danger_zones
    CONFIG["ENABLE_GROUND_PLANE_ZONE"] = True
    logging.info(f"已加载地面标定文件: {calibration_file}")
    return True


def save_ground_calibration(image_points, width_m, depth_m, frame_size):
    calibration = {
        "version": 1,
        "point_order": list(CALIBRATION_POINT_LABELS),
        "frame_size": {"width": frame_size[0], "height": frame_size[1]},
        "image_points": [[int(x), int(y)] for x, y in image_points],
        "world_points_m": [[0.0, 0.0], [width_m, 0.0], [width_m, depth_m], [0.0, depth_m]],
        "danger_zones_m": CONFIG["GROUND_DANGER_ZONES_M"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    calibration_file = CONFIG["CALIBRATION_FILE"]
    with open(calibration_file, "w", encoding="utf-8") as file:
        json.dump(calibration, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return calibration_file


def validate_calibration_points(points, frame_shape):
    if len(points) != 4:
        return False, "必须选择 4 个点"
    contour = np.array(points, dtype=np.float32).reshape((-1, 1, 2))
    frame_area = frame_shape[0] * frame_shape[1]
    if abs(cv2.contourArea(contour)) < frame_area * 0.01:
        return False, "选择的地面区域太小"
    if not cv2.isContourConvex(contour.astype(np.int32)):
        return False, "四点发生交叉，请严格按指定顺序点击"
    return True, ""


def _open_calibration_capture(source):
    if isinstance(source, str) and source.lower().startswith("rtsp://"):
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    return cv2.VideoCapture(source)


def _request_positive_measurement(value, prompt):
    if value is not None:
        return float(value)
    while True:
        try:
            measurement = float(input(prompt).strip())
            if measurement > 0:
                return measurement
        except ValueError:
            pass
        print("请输入大于 0 的数字。")


def run_ground_calibration(width_m=None, depth_m=None):
    width_m = _request_positive_measurement(width_m, "地面矩形真实宽度（米）: ")
    depth_m = _request_positive_measurement(depth_m, "地面矩形真实深度（米）: ")
    source = resolve_video_source()
    capture = _open_calibration_capture(source)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开标定视频源: {redact_uri(source) if isinstance(source, str) else source}")

    window_name = "Ground Calibration"
    is_local_video = isinstance(source, str) and os.path.isfile(source)
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    playback_delay = max(1, int(1000 / source_fps)) if is_local_video and source_fps > 0 else 1
    frozen_frame = None
    points = []

    logging.info("标定画面已打开：按空格冻结画面，按 Q 取消。")
    try:
        while frozen_frame is None:
            success, frame = capture.read()
            if not success:
                if is_local_video:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                raise RuntimeError("读取标定视频源失败。")
            frame = VideoStreamProducer._limit_frame_size(frame)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(playback_delay) & 0xFF
            if key == ord("q"):
                logging.info("已取消地面标定。")
                return False
            if key == ord(" "):
                frozen_frame = frame.copy()

        def on_mouse(event, x, y, flags, userdata):
            del flags, userdata
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((x, y))
                logging.info(f"已选择 {len(points)} {CALIBRATION_POINT_LABELS[len(points) - 1]}: ({x}, {y})")
            elif event == cv2.EVENT_RBUTTONDOWN and points:
                removed = points.pop()
                logging.info(f"已撤销标定点: {removed}")

        cv2.setMouseCallback(window_name, on_mouse)
        logging.info(
            "依次左键点击 LEFT-NEAR、RIGHT-NEAR、RIGHT-FAR、LEFT-FAR；"
            "右键撤销，R 全部重选，四点完成后按 Enter 保存，Q 取消。"
        )

        while True:
            preview = frozen_frame.copy()
            for index, point in enumerate(points):
                cv2.circle(preview, point, 7, (0, 255, 255), -1)
                cv2.putText(
                    preview,
                    f"{index + 1} {CALIBRATION_POINT_LABELS[index]}",
                    (point[0] + 10, max(24, point[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )
            if len(points) >= 2:
                cv2.polylines(preview, [np.array(points, dtype=np.int32)], len(points) == 4, (0, 220, 0), 2)
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                logging.info("已取消地面标定。")
                return False
            if key == ord("r"):
                points.clear()
                logging.info("已清除全部标定点。")
            if key in (10, 13):
                valid, reason = validate_calibration_points(points, frozen_frame.shape)
                if not valid:
                    logging.warning(f"无法保存标定: {reason}")
                    continue
                calibration_file = save_ground_calibration(
                    points,
                    width_m,
                    depth_m,
                    (frozen_frame.shape[1], frozen_frame.shape[0]),
                )
                logging.info(f"地面标定已保存: {calibration_file}")
                return True
    finally:
        capture.release()
        cv2.destroyWindow(window_name)


# ==========================================
# 4. 异步报警处理模块（消费者）
# ==========================================
class AlertHandler(threading.Thread):
    def __init__(self, alert_queue, shutdown_event):
        super().__init__(name="AlertHandlerThread", daemon=True)
        self.alert_queue = alert_queue
        self.shutdown_event = shutdown_event

    def run(self):
        logging.info("异步报警后台线程已启动。")
        while not self.shutdown_event.is_set() or not self.alert_queue.empty():
            try:
                # 使用超时机制，允许线程定期检查 shutdown_event
                alert_task = self.alert_queue.get(timeout=0.5)
                frames, alert_type, fps, reason = alert_task

                timestamp = int(time.time() * 1000)
                filename = f"ALERT_{alert_type}_{timestamp}_{uuid.uuid4().hex[:6]}.mp4"
                filepath = os.path.join(CONFIG["SAVE_DIR"], filename)

                self._write_video(filepath, frames, fps)
                logging.warning(f"🚨 [安全警报] 检测到儿童危险【{alert_type}】！原因: {reason}。视频已固化至: {filepath}")

                self.alert_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logging.error(f"报警处理线程发生异常: {e}")
        logging.info("报警后台线程安全退出。")

    def _write_video(self, filepath, frames, fps):
        if not frames:
            logging.warning("报警视频没有可写入帧，已跳过。")
            return

        first_frame = self._decode_frame(frames[0])
        if first_frame is None:
            raise RuntimeError("报警视频首帧解码失败。")

        h, w = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(filepath, fourcc, fps, (w, h))

        if not writer.isOpened():
            raise RuntimeError(f"无法创建报警视频文件: {filepath}")

        try:
            for stored_frame in frames:
                frame = self._decode_frame(stored_frame)
                if frame is None:
                    logging.warning("跳过一帧无法解码的报警画面。")
                    continue
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h))
                writer.write(frame)
        finally:
            writer.release()

    @staticmethod
    def _decode_frame(stored_frame):
        if isinstance(stored_frame, np.ndarray):
            return stored_frame
        encoded = np.frombuffer(stored_frame, dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


# ==========================================
# 3. 视频流采集模块（生产者）
# ==========================================
class VideoStreamProducer(threading.Thread):
    def __init__(self, source, frame_queue, shutdown_event, stream_finished_event, loop_local_video=False):
        super().__init__(name="VideoProducerThread", daemon=True)
        self.source = source
        self.frame_queue = frame_queue
        self.shutdown_event = shutdown_event
        self.stream_finished_event = stream_finished_event
        self.is_local_video = isinstance(source, str) and os.path.isfile(source)
        self.loop_local_video = loop_local_video
        self.lock = threading.Lock()
        self.cap = self._open_capture()
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # 强制硬件单帧缓存
        source_fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap.isOpened() else 0.0
        self.source_fps = source_fps if 1.0 <= source_fps <= 240.0 else 0.0

    def _open_capture(self):
        if isinstance(self.source, str) and self.source.lower().startswith("rtsp://"):
            return cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        return cv2.VideoCapture(self.source)

    def run(self):
        logging.info("视频流采集线程已启动。")
        while not self.shutdown_event.is_set():
            with self.lock:
                if self.cap is None or not self.cap.isOpened():
                    if self.is_local_video:
                        logging.error(f"无法打开本地测试视频: {self.source}")
                        self.stream_finished_event.set()
                        break
                    logging.error("无法打开监控源，2秒后尝试重连...")
                    time.sleep(2.0)
                    self.cap = self._open_capture()
                    continue
                success, frame = self.cap.read()

            if not success:
                if self.is_local_video:
                    if self.loop_local_video:
                        with self.lock:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        logging.info("本地测试视频已播放完毕，正在循环播放。")
                        continue
                    logging.info("本地测试视频已播放完毕。")
                    self.stream_finished_event.set()
                    break
                logging.warning("丢帧或流中断，正在尝试重连...")
                time.sleep(1.0)
                continue

            frame = self._limit_frame_size(frame)

            if self.is_local_video:
                while not self.shutdown_event.is_set():
                    try:
                        # 测试文件必须保留每一帧，避免快速读盘导致关键动作被丢弃。
                        self.frame_queue.put(frame, timeout=0.5)
                        break
                    except Full:
                        continue
                continue

            # 丢帧策略：如果队列满了，强行弹出旧帧，塞入最新鲜的帧维持高实时性
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    pass

            try:
                self.frame_queue.put_nowait(frame)
            except Full:
                pass

        self.release_hardware()
        logging.info("视频流采集线程安全退出。")

    @staticmethod
    def _limit_frame_size(frame):
        height, width = frame.shape[:2]
        scale = min(
            1.0,
            CONFIG["MAX_FRAME_WIDTH"] / width,
            CONFIG["MAX_FRAME_HEIGHT"] / height,
        )
        if scale >= 1.0:
            return frame
        return cv2.resize(
            frame,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def release_hardware(self):
        """主动释放硬件锁，防止join阻塞"""
        with self.lock:
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None


# ==========================================
# 4. AI 核心智能计算引擎（主控制单元）
# ==========================================
class SmartMonitorEngine:
    def __init__(self):
        self.shutdown_event = threading.Event()
        self.stream_finished_event = threading.Event()
        self.frame_queue = Queue(maxsize=CONFIG["QUEUE_MAXSIZE"])
        self.alert_queue = Queue(maxsize=CONFIG["ALERT_QUEUE_MAXSIZE"])

        self.video_source = resolve_video_source()
        self.producer = VideoStreamProducer(
            self.video_source,
            self.frame_queue,
            self.shutdown_event,
            self.stream_finished_event,
            loop_local_video=CONFIG.get("LOOP_LOCAL_VIDEO", False),
        )
        self.record_fps = (
            self.producer.source_fps
            if self.producer.is_local_video and self.producer.source_fps
            else float(CONFIG["RECORD_FPS"])
        )
        self.pre_alert_frame_count = max(1, int(CONFIG["PRE_ALERT_SECONDS"] * self.record_fps))
        self.post_alert_frame_count = max(1, int(CONFIG["POST_ALERT_SECONDS"] * self.record_fps))
        self.pre_alert_buffer = deque(maxlen=self.pre_alert_frame_count)
        self.pending_clips = []

        # MediaPipe Tasks Pose Landmarker 配置
        self.pose = self._create_pose_landmarker()
        self.frame_timestamp_ms = 0
        self.ground_homography = self._create_ground_homography()
        self.ground_homography_inv = np.linalg.inv(self.ground_homography) if self.ground_homography is not None else None

        # 线程实例化
        self.alert_handler = AlertHandler(self.alert_queue, self.shutdown_event)

        # 每个人独立维护姿态状态，防止多人场景下跌倒计时和报警冷却互相污染。
        self.person_tracks = {}
        self.next_person_id = 1

    def _create_ground_homography(self):
        if not CONFIG["ENABLE_GROUND_PLANE_ZONE"]:
            return None

        image_points = CONFIG["GROUND_PLANE_IMAGE_POINTS"]
        world_points = CONFIG["GROUND_PLANE_WORLD_POINTS_M"]
        if len(image_points) != 4 or len(world_points) != 4:
            logging.warning(
                "地面危险区已开启，但图像标定点和现实坐标点没有各提供 4 个；"
                "本次运行自动跳过地面区域判断，其他危险检测继续工作。"
            )
            return None

        image_points = np.array(image_points, dtype=np.float32)
        world_points = np.array(world_points, dtype=np.float32)
        if image_points.shape != (4, 2) or world_points.shape != (4, 2):
            logging.warning("地面标定点格式必须是 4x2；本次运行自动跳过地面区域判断。")
            return None

        homography = cv2.getPerspectiveTransform(image_points, world_points)
        if not np.isfinite(homography).all() or abs(np.linalg.det(homography)) < 1e-9:
            logging.warning("地面标定点无法形成有效透视变换；本次运行自动跳过地面区域判断。")
            return None

        if not CONFIG["GROUND_DANGER_ZONES_M"]:
            logging.warning("地面标定有效，但尚未配置 GROUND_DANGER_ZONES_M，不会触发地面区域报警。")
        return homography

    def _create_pose_landmarker(self):
        model_path = CONFIG["MODEL_PATH"]
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"未找到姿态模型文件: {model_path}\n"
                "请先运行: python download_pose_model.py\n"
                "或手动下载官方 Pose Landmarker Heavy 模型到该路径。"
            )

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=model_path,
                delegate=BaseOptions.Delegate.CPU if CONFIG["DELEGATE"].upper() == "CPU" else BaseOptions.Delegate.GPU,
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=CONFIG["MAX_POSES"],
            min_pose_detection_confidence=CONFIG["MIN_POSE_DETECTION_CONFIDENCE"],
            min_pose_presence_confidence=CONFIG["MIN_POSE_PRESENCE_CONFIDENCE"],
            min_tracking_confidence=CONFIG["MIN_TRACKING_CONFIDENCE"],
            output_segmentation_masks=False,
        )
        return PoseLandmarker.create_from_options(options)

    def _landmark_ok(self, landmark):
        confidence = min(
            getattr(landmark, "visibility", 1.0),
            getattr(landmark, "presence", 1.0),
        )
        return confidence >= CONFIG["MIN_LANDMARK_CONFIDENCE"]

    def _midpoint(self, landmarks, left_idx, right_idx):
        left = landmarks[left_idx]
        right = landmarks[right_idx]
        if self._landmark_ok(left) and self._landmark_ok(right):
            return np.array([(left.x + right.x) / 2, (left.y + right.y) / 2], dtype=np.float32)
        if self._landmark_ok(left):
            return np.array([left.x, left.y], dtype=np.float32)
        if self._landmark_ok(right):
            return np.array([right.x, right.y], dtype=np.float32)
        return None

    def _person_anchor(self, landmarks):
        hip_center = self._midpoint(landmarks, LEFT_HIP, RIGHT_HIP)
        shoulder_center = self._midpoint(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)
        if hip_center is not None and shoulder_center is not None:
            return (hip_center + shoulder_center) / 2
        if hip_center is not None:
            return hip_center
        if shoulder_center is not None:
            return shoulder_center

        visible = [np.array([lm.x, lm.y], dtype=np.float32) for lm in landmarks if self._landmark_ok(lm)]
        return np.mean(visible, axis=0) if visible else None

    def _assign_pose_tracks(self, pose_landmarks, timestamp_s):
        expired_ids = [
            track_id
            for track_id, track in self.person_tracks.items()
            if timestamp_s - track["last_seen"] > CONFIG["PERSON_TRACK_TIMEOUT"]
        ]
        for track_id in expired_ids:
            del self.person_tracks[track_id]

        detections = []
        for landmarks in pose_landmarks:
            anchor = self._person_anchor(landmarks)
            if anchor is not None:
                detections.append({"landmarks": landmarks, "anchor": anchor})

        candidate_pairs = []
        for detection_index, detection in enumerate(detections):
            for track_id, track in self.person_tracks.items():
                distance = float(np.linalg.norm(detection["anchor"] - track["anchor"]))
                if distance <= CONFIG["PERSON_TRACK_MAX_DISTANCE"]:
                    candidate_pairs.append((distance, detection_index, track_id))

        matched_detections = set()
        matched_tracks = set()
        assignments = {}
        for _, detection_index, track_id in sorted(candidate_pairs):
            if detection_index in matched_detections or track_id in matched_tracks:
                continue
            matched_detections.add(detection_index)
            matched_tracks.add(track_id)
            assignments[detection_index] = track_id

        tracked_poses = []
        for detection_index, detection in enumerate(detections):
            track_id = assignments.get(detection_index)
            if track_id is None:
                track_id = self.next_person_id
                self.next_person_id += 1
                self.person_tracks[track_id] = {
                    "id": track_id,
                    "anchor": detection["anchor"],
                    "last_seen": timestamp_s,
                    "is_lying_down": False,
                    "fall_start_time": None,
                    "last_alert_time": {},
                }

            track = self.person_tracks[track_id]
            track["anchor"] = detection["anchor"]
            track["last_seen"] = timestamp_s
            tracked_poses.append((track, detection["landmarks"]))

        return tracked_poses

    def _visible_points(self, landmarks, frame_w, frame_h):
        points = []
        for lm in landmarks:
            if self._landmark_ok(lm):
                points.append((lm.x * frame_w, lm.y * frame_h))
        return points

    def _point_in_polygon(self, point, polygon):
        x, y = point
        inside = False
        j = len(polygon) - 1
        for i in range(len(polygon)):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            intersects = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-9) + xi
            )
            if intersects:
                inside = not inside
            j = i
        return inside

    def _project_image_to_ground(self, pixel_point):
        if self.ground_homography is None:
            return None
        point = np.array([[[pixel_point[0], pixel_point[1]]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(point, self.ground_homography)[0][0]
        return float(projected[0]), float(projected[1])

    def _project_ground_to_image(self, world_point):
        if self.ground_homography_inv is None:
            return None
        point = np.array([[[world_point[0], world_point[1]]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(point, self.ground_homography_inv)[0][0]
        return int(projected[0]), int(projected[1])

    def _draw_landmarks(self, frame, landmarks, person_id):
        h, w = frame.shape[:2]
        for start_idx, end_idx in POSE_CONNECTIONS:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            if self._landmark_ok(start) and self._landmark_ok(end):
                cv2.line(
                    frame,
                    (int(start.x * w), int(start.y * h)),
                    (int(end.x * w), int(end.y * h)),
                    (80, 220, 120),
                    2,
                )
        for lm in landmarks:
            if self._landmark_ok(lm):
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (255, 255, 255), -1)

        anchor = self._person_anchor(landmarks)
        if anchor is not None:
            label_x = max(0, min(w - 45, int(anchor[0] * w)))
            label_y = max(24, min(h - 5, int(anchor[1] * h) - 12))
            cv2.putText(
                frame,
                f"P{person_id}",
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

    def _draw_status_overlay(self, frame, danger_line_y, danger_zone, status_text, text_color, reasons):
        h, w = frame.shape[:2]
        cv2.line(frame, (0, danger_line_y), (w, danger_line_y), (0, 0, 255), 2)
        if CONFIG["ENABLE_2D_DANGER_ZONE"]:
            cv2.rectangle(frame, (danger_zone[0], danger_zone[1]), (danger_zone[2], danger_zone[3]), (0, 165, 255), 2)
        self._draw_ground_danger_zones(frame)

        overlay_lines = [f"STATUS: {status_text}"]
        if reasons:
            overlay_lines.append(f"REASON: {reasons[0][:120]}")
            overlay_lines.extend(f"        {reason[:120]}" for reason in reasons[1:3])

        line_height = 26
        box_height = 18 + line_height * len(overlay_lines)
        cv2.rectangle(frame, (12, 12), (min(w - 12, 980), 12 + box_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (12, 12), (min(w - 12, 980), 12 + box_height), text_color, 2)

        for i, line in enumerate(overlay_lines):
            cv2.putText(
                frame,
                line,
                (20, 42 + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                text_color if i == 0 else (255, 255, 255),
                2,
            )

    def _draw_ground_danger_zones(self, frame):
        if not CONFIG["ENABLE_GROUND_PLANE_ZONE"] or self.ground_homography_inv is None:
            return

        for zone in CONFIG["GROUND_DANGER_ZONES_M"]:
            polygon = zone.get("polygon", [])
            image_points = [self._project_ground_to_image(point) for point in polygon]
            image_points = [point for point in image_points if point is not None]
            if len(image_points) < 3:
                continue

            pts = np.array(image_points, dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 120, 255), thickness=2)
            label_x, label_y = image_points[0]
            cv2.putText(
                frame,
                zone.get("name", "ground_zone"),
                (label_x, max(20, label_y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 120, 255),
                2,
            )

    def _analyze_pose(self, landmarks, frame, danger_line_y, danger_zone, person_track):
        h, w = frame.shape[:2]
        current_time = self.frame_timestamp_ms / 1000.0
        person_id = person_track["id"]
        person_label = f"P{person_id}"
        status_text = f"{person_label} SAFE"
        text_color = (0, 255, 0)
        reasons = []

        shoulder_center = self._midpoint(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)
        hip_center = self._midpoint(landmarks, LEFT_HIP, RIGHT_HIP)

        if shoulder_center is not None and hip_center is not None:
            trunk_vector = shoulder_center - hip_center
            vertical_vector = np.array([0, -1], dtype=np.float32)
            norm_prod = np.linalg.norm(trunk_vector) * np.linalg.norm(vertical_vector)
            angle = np.degrees(
                np.arccos(np.clip(np.dot(trunk_vector, vertical_vector) / norm_prod, -1.0, 1.0))
            ) if norm_prod > 1e-6 else 0.0

            points = self._visible_points(landmarks, w, h)
            if len(points) > 8:
                xs = [point[0] for point in points]
                ys = [point[1] for point in points]
                hw_ratio = (max(ys) - min(ys)) / (max(xs) - min(xs) + 1e-6)
            else:
                hw_ratio = 1.0

            if angle > CONFIG["FALL_ANGLE_THRESHOLD"] and hw_ratio < CONFIG["HEIGHT_WIDTH_RATIO_LIMIT"]:
                if not person_track["is_lying_down"]:
                    person_track["is_lying_down"] = True
                    person_track["fall_start_time"] = current_time
                else:
                    duration = current_time - person_track["fall_start_time"]
                    if duration > CONFIG["FALL_CONFIRM_DURATION"]:
                        status_text = f"{person_label} FALL DETECTED ({duration:.1f}s)"
                        text_color = (0, 0, 255)
                        reason = (
                            f"{person_label} Fall: trunk angle {angle:.1f}>{CONFIG['FALL_ANGLE_THRESHOLD']} deg, "
                            f"H/W {hw_ratio:.2f}<{CONFIG['HEIGHT_WIDTH_RATIO_LIMIT']}, "
                            f"held {duration:.1f}>{CONFIG['FALL_CONFIRM_DURATION']}s"
                        )
                        reasons.append(reason)
                        if not self._should_throttle(person_track, "FALL"):
                            self._start_alert_clip("FALL", frame, reason)
            else:
                person_track["is_lying_down"] = False
                person_track["fall_start_time"] = None

        ankles = [landmarks[LEFT_ANKLE], landmarks[RIGHT_ANKLE]]
        hips = [landmarks[LEFT_HIP], landmarks[RIGHT_HIP]]
        ankle_high = any(self._landmark_ok(ankle) and int(ankle.y * h) < danger_line_y for ankle in ankles)
        hip_high = any(self._landmark_ok(hip) and int(hip.y * h) < danger_line_y for hip in hips)
        if ankle_high and hip_high:
            status_text = f"{person_label} CLIMBING WARNING"
            text_color = (0, 0, 255)
            highest_ankle_y = min(int(ankle.y * h) for ankle in ankles if self._landmark_ok(ankle))
            highest_hip_y = min(int(hip.y * h) for hip in hips if self._landmark_ok(hip))
            reason = (
                f"{person_label} Climb: ankle y={highest_ankle_y} and hip y={highest_hip_y} "
                f"above danger line y={danger_line_y}"
            )
            reasons.append(reason)
            if not self._should_throttle(person_track, "CLIMB"):
                self._start_alert_clip("CLIMB", frame, reason)

        ground_reason = self._check_ground_danger_zones(landmarks, w, h)
        if ground_reason:
            status_text = f"{person_label} GROUND ZONE WARNING"
            text_color = (0, 0, 255)
            ground_reason = f"{person_label} {ground_reason}"
            reasons.append(ground_reason)
            if not self._should_throttle(person_track, "GROUND_ZONE"):
                self._start_alert_clip("GROUND_ZONE", frame, ground_reason)

        if not CONFIG["ENABLE_2D_DANGER_ZONE"]:
            return status_text, text_color, reasons

        wrists = [landmarks[LEFT_WRIST], landmarks[RIGHT_WRIST]]
        for wrist in wrists:
            if not self._landmark_ok(wrist):
                continue
            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
            if (danger_zone[0] < wrist_x < danger_zone[2]) and (danger_zone[1] < wrist_y < danger_zone[3]):
                status_text = f"{person_label} ZONE INTRUSION"
                text_color = (0, 0, 255)
                reason = (
                    f"{person_label} Intrusion: wrist ({wrist_x},{wrist_y}) inside zone "
                    f"({danger_zone[0]},{danger_zone[1]})-({danger_zone[2]},{danger_zone[3]})"
                )
                reasons.append(reason)
                if not self._should_throttle(person_track, "INTRUSION"):
                    self._start_alert_clip("INTRUSION", frame, reason)
                break

        return status_text, text_color, reasons

    def _check_ground_danger_zones(self, landmarks, frame_w, frame_h):
        if not CONFIG["ENABLE_GROUND_PLANE_ZONE"] or self.ground_homography is None:
            return None

        foot_points = []
        for label, idx in (("left_ankle", LEFT_ANKLE), ("right_ankle", RIGHT_ANKLE)):
            ankle = landmarks[idx]
            if not self._landmark_ok(ankle):
                continue
            pixel_point = (ankle.x * frame_w, ankle.y * frame_h)
            ground_point = self._project_image_to_ground(pixel_point)
            if ground_point is not None:
                foot_points.append((label, ground_point))

        for label, ground_point in foot_points:
            for zone in CONFIG["GROUND_DANGER_ZONES_M"]:
                polygon = zone.get("polygon", [])
                if len(polygon) < 3:
                    continue
                if self._point_in_polygon(ground_point, polygon):
                    zone_name = zone.get("name", "ground_zone")
                    return (
                        f"Ground zone: {label} at ({ground_point[0]:.2f}m,{ground_point[1]:.2f}m) "
                        f"inside calibrated zone '{zone_name}'"
                    )
        return None

    def _should_throttle(self, person_track, alert_type):
        """每个人独立的报警节流阀。"""
        current_time = self.frame_timestamp_ms / 1000.0
        last_alert_time = person_track["last_alert_time"]
        if alert_type in last_alert_time:
            if current_time - last_alert_time[alert_type] < CONFIG["ALERT_COOLDOWN"]:
                return True
        last_alert_time[alert_type] = current_time
        return False

    def _start_alert_clip(self, alert_type, current_frame, reason):
        del current_frame
        if len(self.pending_clips) >= CONFIG["MAX_PENDING_CLIPS"]:
            logging.warning("同时采集的报警片段已达上限，跳过本次录像任务。")
            return

        # JPEG bytes 是不可变对象；这里只复制列表，不复制每一帧的像素内存。
        clip_frames = list(self.pre_alert_buffer)
        self.pending_clips.append({
            "alert_type": alert_type,
            "reason": reason,
            "frames": clip_frames,
            "remaining_frames": self.post_alert_frame_count,
        })

    def _update_alert_clips(self, stored_frame):
        if not self.pending_clips:
            return

        completed_clips = []
        for clip in self.pending_clips:
            clip["frames"].append(stored_frame)
            clip["remaining_frames"] -= 1
            if clip["remaining_frames"] <= 0:
                completed_clips.append(clip)

        for clip in completed_clips:
            self.pending_clips.remove(clip)
            try:
                self.alert_queue.put_nowait((clip["frames"], clip["alert_type"], self.record_fps, clip["reason"]))
            except Full:
                logging.warning("报警视频队列已满，已丢弃一次视频保存任务。")

    def _flush_alert_clips(self):
        for clip in list(self.pending_clips):
            self.pending_clips.remove(clip)
            try:
                self.alert_queue.put_nowait((clip["frames"], clip["alert_type"], self.record_fps, clip["reason"]))
            except Full:
                logging.warning("报警视频队列已满，退出时丢弃一次未完成视频保存任务。")

    @staticmethod
    def _encode_alert_frame(frame):
        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, CONFIG["ALERT_JPEG_QUALITY"]],
        )
        if not success:
            raise RuntimeError("报警帧 JPEG 压缩失败。")
        return encoded.tobytes()

    def start_system(self):
        self.producer.start()
        self.alert_handler.start()

        logging.info("智能监控推理引擎开始运转...")

        try:
            while not self.shutdown_event.is_set():
                try:
                    frame = self.frame_queue.get(timeout=0.5)
                except Empty:
                    if self.stream_finished_event.is_set():
                        logging.info("本地视频所有帧均已处理，监控测试结束。")
                        break
                    continue

                h, w, _ = frame.shape
                danger_line_y = int(h * CONFIG["DANGER_LINE_Y_PCT"])
                danger_zone = [
                    int(w * CONFIG["DANGER_ZONE_PCT"][0]),
                    int(h * CONFIG["DANGER_ZONE_PCT"][1]),
                    int(w * CONFIG["DANGER_ZONE_PCT"][2]),
                    int(h * CONFIG["DANGER_ZONE_PCT"][3])
                ]

                status_text = "SYSTEM ACTIVE"
                text_color = (0, 255, 0)
                reasons = []

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                self.frame_timestamp_ms += max(1, int(1000 / self.record_fps))
                results = self.pose.detect_for_video(mp_image, self.frame_timestamp_ms)

                tracked_poses = self._assign_pose_tracks(
                    results.pose_landmarks or [],
                    self.frame_timestamp_ms / 1000.0,
                )
                danger_people = set()
                for person_track, landmarks in tracked_poses:
                    _, _, person_reasons = self._analyze_pose(
                        landmarks,
                        frame,
                        danger_line_y,
                        danger_zone,
                        person_track,
                    )
                    if person_reasons:
                        danger_people.add(person_track["id"])
                        reasons.extend(person_reasons)
                    if CONFIG["SHOW_WINDOW"]:
                        self._draw_landmarks(frame, landmarks, person_track["id"])

                if danger_people:
                    status_text = f"DANGER: {len(danger_people)}/{len(tracked_poses)} PEOPLE"
                    text_color = (0, 0, 255)
                elif tracked_poses:
                    status_text = f"{len(tracked_poses)} PEOPLE TRACKED"
                else:
                    status_text = "NO PERSON DETECTED"

                # UI 渲染与输出
                self._draw_status_overlay(frame, danger_line_y, danger_zone, status_text, text_color, reasons)
                if CONFIG["SHOW_WINDOW"]:
                    cv2.imshow("Production Security Engine", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                stored_frame = self._encode_alert_frame(frame)
                self.pre_alert_buffer.append(stored_frame)
                self._update_alert_clips(stored_frame)

        except KeyboardInterrupt:
            logging.info("接收到键盘中断指令...")
        finally:
            self.stop_system()

    def stop_system(self):
        if not self.shutdown_event.is_set():
            logging.info("正在安全通知后台线程退出...")
            self._flush_alert_clips()
            self.shutdown_event.set()

            # 核心改进：先强制释放硬件底层，迫使 Producer 线程从阻塞的 cap.read() 中弹回
            self.producer.release_hardware()

            # 安全回收线程
            self.producer.join(timeout=2.0)
            self.alert_handler.join(timeout=2.0)

            self.pose.close()
            if CONFIG["SHOW_WINDOW"]:
                cv2.destroyAllWindows()
            logging.info("系统已安全释放所有资源并退出。")


def parse_args():
    parser = argparse.ArgumentParser(description="儿童安全智能监控")
    parser.add_argument(
        "--video",
        metavar="PATH",
        help="读取本地视频进行测试，并跳过 ONVIF/摄像头连接",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="循环播放 --video 指定的本地测试视频",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="打开半自动地面四点标定工具，不启动姿态检测",
    )
    parser.add_argument("--ground-width", type=float, metavar="METERS", help="标定矩形真实宽度（米）")
    parser.add_argument("--ground-depth", type=float, metavar="METERS", help="标定矩形真实深度（米）")
    args = parser.parse_args()

    if args.loop and not args.video:
        parser.error("--loop 必须与 --video 一起使用")
    if (args.ground_width is not None or args.ground_depth is not None) and not args.calibrate:
        parser.error("--ground-width 和 --ground-depth 必须与 --calibrate 一起使用")
    if args.ground_width is not None and args.ground_width <= 0:
        parser.error("--ground-width 必须大于 0")
    if args.ground_depth is not None and args.ground_depth <= 0:
        parser.error("--ground-depth 必须大于 0")
    if args.video:
        video_path = os.path.abspath(os.path.expanduser(args.video))
        if not os.path.isfile(video_path):
            parser.error(f"找不到本地视频文件: {video_path}")
        CONFIG["USE_ONVIF_CAMERA"] = False
        CONFIG["VIDEO_SOURCE"] = video_path
        CONFIG["LOOP_LOCAL_VIDEO"] = args.loop

    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.calibrate:
        run_ground_calibration(arguments.ground_width, arguments.ground_depth)
    else:
        load_ground_calibration()
        engine = SmartMonitorEngine()
        engine.start_system()
