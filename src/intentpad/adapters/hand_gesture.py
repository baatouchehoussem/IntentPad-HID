import cv2
import numpy as np
import logging
import time
from typing import Optional, Tuple, Dict, Any
from .base import InputAdapter, AdapterFrame

logger = logging.getLogger(__name__)


class HandGestureAdapter(InputAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._cap = None
        self._hands = None
        self._mp_hands = None
        self._mp_drawing = None
        self._mp_drawing_styles = None
        self._frame_width = config.get("width", 640)
        self._frame_height = config.get("height", 480)
        self._fps = config.get("fps", 30)
        self._min_detection_confidence = config.get("min_detection_confidence", 0.7)
        self._min_tracking_confidence = config.get("min_tracking_confidence", 0.5)
        self._max_num_hands = config.get("max_num_hands", 1)
        self._last_frame_time = 0
        self._frame_interval = 1.0 / self._fps

    def start(self) -> None:
        try:
            import mediapipe as mp
            self._mp_hands = mp.solutions.hands
            self._mp_drawing = mp.solutions.drawing_utils
            self._mp_drawing_styles = mp.solutions.drawing_styles
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self._max_num_hands,
                min_detection_confidence=self._min_detection_confidence,
                min_tracking_confidence=self._min_tracking_confidence,
                model_complexity=1,
            )
        except ImportError:
            raise RuntimeError("mediapipe not installed. Install with: pip install mediapipe")

        self._cap = cv2.VideoCapture(0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        if not self._cap.isOpened():
            raise RuntimeError("Failed to open webcam")

        self._running = True
        logger.info("HandGestureAdapter started")

    def stop(self) -> None:
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        if self._hands:
            self._hands.close()
            self._hands = None
        logger.info("HandGestureAdapter stopped")

    def read_frame(self) -> AdapterFrame:
        if not self._running or not self._cap:
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
        results = self._hands.process(rgb_frame)

        left_stick = (0.0, 0.0)
        right_stick = (0.0, 0.0)
        triggers = (0.0, 0.0)
        buttons = {}

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self._mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_drawing_styles.get_default_hand_landmarks_style(),
                self._mp_drawing_styles.get_default_hand_connections_style(),
            )

            landmarks = hand_landmarks.landmark
            left_stick, right_stick, triggers, buttons = self._process_gestures(landmarks)

        return AdapterFrame(
            left_stick=left_stick,
            right_stick=right_stick,
            triggers=triggers,
            buttons=buttons,
            annotated_frame=frame,
            metadata={"hands_detected": len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0}
        )

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