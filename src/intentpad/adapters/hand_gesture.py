import cv2
import numpy as np
import logging
import time
import threading
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
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

LEFT_HAND_COLOR = (255, 255, 0)   # Cyan/Green for Left Hand
RIGHT_HAND_COLOR = (0, 165, 255)  # Amber/Red for Right Hand


class HandGestureAdapter(InputAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        super().__init__(config)
        self._frame_width = config.get("width", 640)
        self._frame_height = config.get("height", 480)
        self._fps = config.get("fps", 30)
        self._min_detection_confidence = config.get("min_detection_confidence", 0.6)
        self._min_tracking_confidence = config.get("min_tracking_confidence", 0.5)
        self._max_num_hands = config.get("max_num_hands", 2)
        
        self._cap = None
        self._landmarker = None
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        
        self._latest_frame: Optional[AdapterFrame] = None
        self._latest_frame_lock = threading.Lock()
        self._latest_annotated_frame: Optional[np.ndarray] = None
        self._latest_annotated_lock = threading.Lock()
        
        self._frame_timestamp_ms = 0
        self._frame_interval_ms = 1000.0 / self._fps
        
        self._fps_samples: List[float] = []
        self._max_fps_samples = 15
        self._current_fps = 0.0

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
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            raise RuntimeError("Failed to open webcam")

        self._running = True
        self._frame_timestamp_ms = 0
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        logger.info("HandGestureAdapter started with MediaPipe Tasks API (dual-hand tracking)")

    def stop(self) -> None:
        self._running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None
        logger.info("HandGestureAdapter stopped")

    def _capture_loop(self) -> None:
        import mediapipe as mp
        
        while self._running:
            loop_start = time.time()
            
            if not self._cap or not self._cap.isOpened():
                time.sleep(0.01)
                continue
                
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            self._frame_timestamp_ms += int(self._frame_interval_ms)
            result = self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)
            
            left_stick = (0.0, 0.0)
            right_stick = (0.0, 0.0)
            triggers = (0.0, 0.0)
            buttons = {}
            hands_detected = len(result.hand_landmarks)
            
            left_hand_landmarks = None
            right_hand_landmarks = None
            
            if hands_detected > 0:
                for i, landmarks in enumerate(result.hand_landmarks):
                    handedness = result.handedness[i]
                    if handedness and len(handedness) > 0:
                        hand_label = handedness[0].category_name  # "Left" or "Right"
                        # Camera is flipped, so user's physical left hand appears as "Right" in the flipped image
                        if hand_label == "Left":
                            right_hand_landmarks = landmarks  # User's right hand
                        else:
                            left_hand_landmarks = landmarks   # User's left hand
            
            if left_hand_landmarks:
                self._draw_landmarks(frame, left_hand_landmarks, LEFT_HAND_COLOR)
                left_stick, lt_val, left_buttons = self._process_left_hand(left_hand_landmarks)
                buttons.update(left_buttons)
                triggers = (lt_val, triggers[1])
            
            if right_hand_landmarks:
                self._draw_landmarks(frame, right_hand_landmarks, RIGHT_HAND_COLOR)
                right_stick, rt_val, right_buttons = self._process_right_hand(right_hand_landmarks)
                buttons.update(right_buttons)
                triggers = (triggers[0], rt_val)
            
            if not left_hand_landmarks and not right_hand_landmarks:
                self._draw_landmarks(frame, None, LEFT_HAND_COLOR)
            
            adapter_frame = AdapterFrame(
                left_stick=left_stick,
                right_stick=right_stick,
                triggers=triggers,
                buttons=buttons,
                annotated_frame=frame.copy(),
                metadata={
                    "hands_detected": hands_detected,
                    "left_hand_detected": left_hand_landmarks is not None,
                    "right_hand_detected": right_hand_landmarks is not None,
                    "fps": self._current_fps
                }
            )
            
            with self._latest_frame_lock:
                self._latest_frame = adapter_frame
            
            with self._latest_annotated_lock:
                self._latest_annotated_frame = frame.copy()
            
            dt = time.time() - loop_start
            if dt > 0:
                instant_fps = 1.0 / dt
                self._fps_samples.append(instant_fps)
                if len(self._fps_samples) > self._max_fps_samples:
                    self._fps_samples.pop(0)
                self._current_fps = sum(self._fps_samples) / len(self._fps_samples)
            
            elapsed_ms = (time.time() - loop_start) * 1000
            sleep_ms = max(0, self._frame_interval_ms - elapsed_ms)
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

    def read_frame(self) -> AdapterFrame:
        with self._latest_frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame
        return AdapterFrame()

    def get_mjpeg_frame(self) -> Optional[bytes]:
        with self._latest_annotated_lock:
            if self._latest_annotated_frame is not None:
                ret, buffer = cv2.imencode('.jpg', self._latest_annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    return buffer.tobytes()
        return None

    def _draw_landmarks(self, frame: np.ndarray, landmarks, color: Tuple[int, int, int] = LEFT_HAND_COLOR) -> None:
        h, w = frame.shape[:2]
        if landmarks:
            points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

            for start, end in HAND_CONNECTIONS:
                if start < len(points) and end < len(points):
                    cv2.line(frame, points[start], points[end], color, 2)

            for x, y in points:
                cv2.circle(frame, (x, y), 4, color, -1)
        
        fps_text = f"FPS: {self._current_fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def _process_left_hand(self, landmarks) -> Tuple[Tuple[float, float], float, Dict[str, bool]]:
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

        lt_value = fist_score

        buttons = {
            "A": open_palm_score > 0.7,
            "X": pinch_distance > 0.8,
        }

        return (left_stick_x, left_stick_y), lt_value, buttons

    def _process_right_hand(self, landmarks) -> Tuple[Tuple[float, float], float, Dict[str, bool]]:
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

        right_stick_x = max(-1.0, min(1.0, roll * 3.0))
        right_stick_y = max(-1.0, min(1.0, -pitch * 3.0))

        thumb_index_dist = np.linalg.norm([
            thumb_tip.x - index_tip.x,
            thumb_tip.y - index_tip.y
        ])
        pinch_distance = max(0.0, min(1.0, (0.1 - thumb_index_dist) / 0.1 * 2))

        fist_score = self._compute_fist_score(landmarks)
        open_palm_score = self._compute_open_palm_score(landmarks)

        rt_value = pinch_distance

        buttons = {
            "B": open_palm_score > 0.7,
            "Y": pinch_distance > 0.8,
        }

        return (right_stick_x, right_stick_y), rt_value, buttons

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