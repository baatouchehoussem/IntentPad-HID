import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class SignalFilterConfig:
    deadzone: float = 0.15
    alpha: float = 0.60


class SignalFilter:
    def __init__(self, config: SignalFilterConfig = None):
        self._config = config or SignalFilterConfig()
        self._prev_x: float = 0.0
        self._prev_y: float = 0.0

    @property
    def deadzone(self) -> float:
        return self._config.deadzone

    @deadzone.setter
    def deadzone(self, value: float) -> None:
        self._config.deadzone = max(0.0, min(1.0, value))

    @property
    def alpha(self) -> float:
        return self._config.alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._config.alpha = max(0.0, min(1.0, value))

    def _apply_radial_deadzone(self, x: float, y: float) -> Tuple[float, float]:
        magnitude = math.hypot(x, y)
        if magnitude < self._config.deadzone:
            return (0.0, 0.0)
        return (x, y)

    def _normalize_vector(self, x: float, y: float) -> Tuple[float, float]:
        magnitude = math.hypot(x, y)
        if magnitude > 1.0:
            return (x / magnitude, y / magnitude)
        return (x, y)

    def _ema_smooth(self, target_x: float, target_y: float) -> Tuple[float, float]:
        smooth_x = (self._config.alpha * target_x) + ((1.0 - self._config.alpha) * self._prev_x)
        smooth_y = (self._config.alpha * target_y) + ((1.0 - self._config.alpha) * self._prev_y)
        self._prev_x = smooth_x
        self._prev_y = smooth_y
        return (smooth_x, smooth_y)

    def _clamp(self, value: float) -> float:
        return max(-1.0, min(1.0, value))

    def process(self, raw_x: float, raw_y: float) -> Tuple[float, float]:
        x, y = self._apply_radial_deadzone(raw_x, raw_y)
        x, y = self._normalize_vector(x, y)
        x, y = self._ema_smooth(x, y)
        return (self._clamp(x), self._clamp(y))

    def reset(self) -> None:
        self._prev_x = 0.0
        self._prev_y = 0.0

    def get_state(self) -> Tuple[float, float]:
        return (self._prev_x, self._prev_y)