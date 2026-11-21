import threading
import time
import logging
from typing import Optional, Tuple, Dict, Any
from pynput import mouse
import numpy as np
from .base import InputAdapter, AdapterFrame

logger = logging.getLogger(__name__)


class NeuralHIDAdapter(InputAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        super().__init__(config)
        self._screen_width = config.get("screen_width", 1920)
        self._screen_height = config.get("screen_height", 1080)
        self._center_x = self._screen_width / 2
        self._center_y = self._screen_height / 2
        self._max_radius = min(self._screen_width, self._screen_height) / 2
        self._dwell_time = config.get("dwell_time", 0.8)
        self._dwell_radius = config.get("dwell_radius", 20)
        self._deadzone = config.get("deadzone", 0.05)
        self._sensitivity = config.get("sensitivity", 1.0)

        self._current_pos = (self._center_x, self._center_y)
        self._dwell_start_pos = None
        self._dwell_start_time = 0
        self._dwell_triggered = False
        self._buttons = {}
        self._lock = threading.Lock()
        self._listener = None
        self._running = False

    def start(self) -> None:
        self._listener = mouse.Listener(
            on_move=self._on_move,
            on_click=self._on_click,
            on_scroll=self._on_scroll,
        )
        self._listener.start()
        self._running = True
        logger.info("NeuralHIDAdapter started")

    def stop(self) -> None:
        self._running = False
        if self._listener:
            self._listener.stop()
            self._listener = None
        logger.info("NeuralHIDAdapter stopped")

    def _on_move(self, x: int, y: int) -> None:
        with self._lock:
            self._current_pos = (x, y)
            if self._dwell_start_pos:
                dx = x - self._dwell_start_pos[0]
                dy = y - self._dwell_start_pos[1]
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if dist > self._dwell_radius:
                    self._dwell_start_pos = None
                    self._dwell_triggered = False

    def _on_click(self, x: int, y: int, button: mouse.Button, pressed: bool) -> None:
        with self._lock:
            btn_name = "LEFT" if button == mouse.Button.left else "RIGHT" if button == mouse.Button.right else "MIDDLE"
            self._buttons[btn_name] = pressed

    def _on_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        with self._lock:
            self._buttons["SCROLL_UP"] = dy > 0
            self._buttons["SCROLL_DOWN"] = dy < 0

    def read_frame(self) -> AdapterFrame:
        if not self._running:
            return AdapterFrame()

        with self._lock:
            x, y = self._current_pos
            buttons = self._buttons.copy()

            norm_x = (x - self._center_x) / self._max_radius
            norm_y = (y - self._center_y) / self._max_radius

            norm_x = max(-1.0, min(1.0, norm_x * self._sensitivity))
            norm_y = max(-1.0, min(1.0, norm_y * self._sensitivity))

            if abs(norm_x) < self._deadzone:
                norm_x = 0.0
            if abs(norm_y) < self._deadzone:
                norm_y = 0.0

            current_time = time.time()
            if self._dwell_start_pos is None:
                self._dwell_start_pos = (x, y)
                self._dwell_start_time = current_time
                self._dwell_triggered = False
            elif not self._dwell_triggered and current_time - self._dwell_start_time >= self._dwell_time:
                self._dwell_triggered = True
                buttons["DWELL"] = True

            left_stick = (norm_x, -norm_y)
            right_stick = (0.0, 0.0)
            triggers = (0.0, 0.0)

            if buttons.get("LEFT", False):
                triggers = (0.0, 1.0)
            if buttons.get("RIGHT", False):
                triggers = (1.0, 0.0)

            mapped_buttons = {
                "A": buttons.get("LEFT", False),
                "B": buttons.get("RIGHT", False),
                "X": buttons.get("MIDDLE", False),
                "Y": buttons.get("DWELL", False),
            }

            return AdapterFrame(
                left_stick=left_stick,
                right_stick=right_stick,
                triggers=triggers,
                buttons=mapped_buttons,
                metadata={
                    "cursor_x": x,
                    "cursor_y": y,
                    "dwell_active": self._dwell_start_pos is not None,
                    "dwell_progress": min(1.0, (current_time - self._dwell_start_time) / self._dwell_time) if self._dwell_start_pos else 0.0,
                }
            )

    def calibrate_center(self) -> None:
        with self._lock:
            self._center_x, self._center_y = self._current_pos
            logger.info(f"NeuralHIDAdapter calibrated center to ({self._center_x}, {self._center_y})")