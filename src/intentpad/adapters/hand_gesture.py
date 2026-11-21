import cv2
import numpy as np
import logging
import time
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from .base import InputAdapter, AdapterFrame

logger = logging.getLogger(__name__)

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "hand_landmarker.task"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17), (0, 5), (9, 13)
]


class HandGestureAdapter(InputAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        super().__init__(config)
        self._cap = None
        self._landmarker = None
        self._frame_width = config.get("width", 640)
        self._frame_height = config.get("height", 480)
        self._fps = config.get("fps", 30)
        self._min_detection_confidence = config.get("min_detection_confidence", 0.6)
        self._min_tracking_confidence = config.get("min_tracking_confidence", 0.5)
        self._max_num_hands = config.get("max_num_hands", 1)
        self._last_frame_time = 0
        self._frame_interval = 1.0 / self._fps
        self._frame_timestamp_ms = 0

    def _ensure_model(self) -> None:
        if not MODEL_PATH.exists():
            logger.info(f"Downloading hand landmarker model from {MODEL_URL}")
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            logger.info(f"Model downloaded to {MODEL_PATH}")

    def start(self) -> None:
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            import mediapipe as mp
        except ImportError:
            raise RuntimeError("mediapipe not installed. Install with: pip install mediapipe")

        self._ensure_model()

        base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self._max_num_hands,
            min_hand_detection_confidence=self._min_detection_confidence,
            min_hand_presence_confidence=self._min_tracking_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

        self._cap = cv2.VideoCapture(0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        if not self._cap.isOpened():
            raise RuntimeError("Failed to open webcam")

        self._running = True
        self._frame_timestamp_ms = 0
        logger.info("HandGestureAdapter started with MediaPipe Tasks API")

    def stop(self) -> None:
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None
        logger.info("HandGestureAdapter stopped")

    def read_frame(self) -> AdapterFrame:
        if not self._running or not self._cap or not self._landmarker:
            return AdapterFrame()

        current_time = time.time()
        if current_time - self._last_frame_time < self._frame_interval:
            return AdapterFrame()

        ret, frame = self._cap.read()
        if not ret:
            return AdapterFrame()

        self._last_frame_time = current_time
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self._frame_timestamp_ms += int(self._frame_interval * 1000)
        result = self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        left_stick = (0.0, 0.0)
        right_stick = (0.0, 0.0)
        triggers = (0.0, 0.0)
        buttons = {}

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            self._draw_landmarks(frame, landmarks)
            left_stick, right_stick, triggers, buttons = self._process_gestures(landmarks)

        return AdapterFrame(
            left_stick=left_stick,
            right_stick=right_stick,
            triggers=triggers,
            buttons=buttons,
            annotated_frame=frame,
            metadata={"hands_detected": len(result.hand_landmarks)}
        )

    def _draw_landmarks(self, frame: np.ndarray, landmarks) -> None:
        h, w = frame.shape[:2]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        for start, end in HAND_CONNECTIONS:
            if start < len(points) and end < len(points):
                cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

        for x, y in points:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    def _process_gestures(self, landmarks) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Dict[str, bool]]:
        wrist = landmarks[0]
        index_mcp = landmarks[5]
        index_tip = landmarks[8]
        middle_mcp = landmarks[9]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]

        hand_center_x = wrist.x
        hand_center_y = wrist.y

        index_vec = np.array([index_tip.x - index_mcp.x, index_tip.y - index_mcp.y])
        middle_vec = np.array([middle_tip.x - middle_mcp.x, middle_tip.y - middle_mcp.y])
        hand_forward = (index_vec + middle_vec) / 2
        hand_forward_norm = np.linalg.norm(hand_forward)
        if hand_forward_norm > 0:
            hand_forward = hand_forward / hand_forward_norm

        hand_right = np.array([-hand_forward[1], hand_forward[0]])

        roll = np.dot(np.array([wrist.x - index_mcp.x, wrist.y - index_mcp.y]), hand_right)
        pitch = np.dot(np.array([wrist.x - index_mcp.x, wrist.y - index_mcp.y]), hand_forward)

        left_stick_x = max(-1.0, min(1.0, roll * 3.0))
        left_stick_y = max(-1.0, min(1.0, -pitch * 3.0))

        thumb_index_dist = np.linalg.norm([
            thumb_tip.x - index_tip.x,
            thumb_tip.y - index_tip.y
        ])
        pinch_distance = max(0.0, min(1.0, (0.1 - thumb_index_dist) / 0.1 * 2))

        fist_score = self._compute_fist_score(landmarks)
        open_palm_score = self._compute_open_palm_score(landmarks)

        rt_value = pinch_distance
        lt_value = fist_score

        right_stick_x = 0.0
        right_stick_y = 0.0

        buttons = {
            "A": open_palm_score > 0.7,
            "B": False,
            "X": False,
            "Y": False,
            "LB": False,
            "RB": pinch_distance > 0.8,
        }

        return (left_stick_x, left_stick_y), (right_stick_x, right_stick_y), (lt_value, rt_value), buttons

    def _compute_fist_score(self, landmarks) -> float:
        tips = [8, 12, 16, 20]
        mcps = [5, 9, 13, 17]
        curled = 0
        for tip, mcp in zip(tips, mcps):
            if landmarks[tip].y > landmarks[mcp].y:
                curled += 1
        thumb_curled = landmarks[4].x < landmarks[3].x if landmarks[4].x < landmarks[3].x else landmarks[4].x > landmarks[3].x
        return (curled + (1 if thumb_curled else 0)) / 5.0

    def _compute_open_palm_score(self, landmarks) -> float:
        tips = [8, 12, 16, 20]
        mcps = [5, 9, 13, 17]
        extended = 0
        for tip, mcp in zip(tips, mcps):
            if landmarks[tip].y < landmarks[mcp].y:
                extended += 1
        thumb_extended = landmarks[4].x > landmarks[3].x
        return (extended + (1 if thumb_extended else 0)) / 5.0

    def get_mjpeg_frame(self) -> Optional[bytes]:
        frame_data = self.read_frame()
        if frame_data.annotated_frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame_data.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                return buffer.tobytes()
        return None