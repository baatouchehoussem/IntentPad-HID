import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
from enum import Enum


class CurveType(Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    S_CURVE = "s_curve"


@dataclass
class AxisConfig:
    deadzone: float = 0.1
    sensitivity: float = 1.0
    curve: CurveType = CurveType.LINEAR
    invert: bool = False
    ema_alpha: float = 0.3


@dataclass
class TriggerConfig:
    deadzone: float = 0.05
    sensitivity: float = 1.0
    curve: CurveType = CurveType.LINEAR
    ema_alpha: float = 0.3


@dataclass
class SignalFilter:
    axis_configs: Dict[str, AxisConfig] = field(default_factory=dict)
    trigger_configs: Dict[str, TriggerConfig] = field(default_factory=dict)
    _ema_state: Dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self):
        defaults = {
            "lx": AxisConfig(), "ly": AxisConfig(),
            "rx": AxisConfig(), "ry": AxisConfig(),
            "lt": TriggerConfig(), "rt": TriggerConfig(),
        }
        for k, v in defaults.items():
            if k not in self.axis_configs and k not in self.trigger_configs:
                if k in ("lt", "rt"):
                    self.trigger_configs[k] = v
                else:
                    self.axis_configs[k] = v
            self._ema_state[k] = 0.0

    def _apply_deadzone(self, value: float, deadzone: float) -> float:
        if abs(value) <= deadzone:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - deadzone) / (1.0 - deadzone)

    def _apply_curve(self, value: float, curve: CurveType) -> float:
        v = abs(value)
        sign = 1.0 if value >= 0 else -1.0
        if curve == CurveType.LINEAR:
            return sign * v
        elif curve == CurveType.EXPONENTIAL:
            return sign * (v ** 2)
        elif curve == CurveType.LOGARITHMIC:
            return sign * math.log1p(v) / math.log(2) if v > 0 else 0.0
        elif curve == CurveType.S_CURVE:
            return sign * (3 * v ** 2 - 2 * v ** 3)
        return sign * v

    def _ema(self, key: str, value: float, alpha: float) -> float:
        prev = self._ema_state.get(key, 0.0)
        smoothed = alpha * value + (1.0 - alpha) * prev
        self._ema_state[key] = smoothed
        return smoothed

    def process_axis(self, axis: str, raw_value: float) -> float:
        config = self.axis_configs.get(axis)
        if not config:
            return raw_value
        v = self._apply_deadzone(raw_value, config.deadzone)
        v = self._apply_curve(v, config.curve)
        v *= config.sensitivity
        if config.invert:
            v = -v
        v = max(-1.0, min(1.0, v))
        return self._ema(axis, v, config.ema_alpha)

    def process_trigger(self, trigger: str, raw_value: float) -> float:
        config = self.trigger_configs.get(trigger)
        if not config:
            return raw_value
        v = max(0.0, min(1.0, raw_value))
        v = self._apply_deadzone(v, config.deadzone)
        v = self._apply_curve(v, config.curve)
        v *= config.sensitivity
        v = max(0.0, min(1.0, v))
        return self._ema(trigger, v, config.ema_alpha)

    def process_vector(self, x: float, y: float, axis_x: str, axis_y: str) -> Tuple[float, float]:
        return (self.process_axis(axis_x, x), self.process_axis(axis_y, y))

    def reset(self) -> None:
        for k in self._ema_state:
            self._ema_state[k] = 0.0

    def update_config(self, axis: str, **kwargs) -> None:
        if axis in self.axis_configs:
            for k, v in kwargs.items():
                if hasattr(self.axis_configs[axis], k):
                    setattr(self.axis_configs[axis], k, v)
        elif axis in self.trigger_configs:
            for k, v in kwargs.items():
                if hasattr(self.trigger_configs[axis], k):
                    setattr(self.trigger_configs[axis], k, v)

    def get_config_dict(self) -> Dict:
        return {
            "axes": {k: {
                "deadzone": v.deadzone,
                "sensitivity": v.sensitivity,
                "curve": v.curve.value,
                "invert": v.invert,
                "ema_alpha": v.ema_alpha,
            } for k, v in self.axis_configs.items()},
            "triggers": {k: {
                "deadzone": v.deadzone,
                "sensitivity": v.sensitivity,
                "curve": v.curve.value,
                "ema_alpha": v.ema_alpha,
            } for k, v in self.trigger_configs.items()},
        }


@dataclass
class ButtonMap:
    tap: Optional[str] = None
    hold: Optional[str] = None
    hold_threshold: float = 0.5


@dataclass
class Profile:
    name: str
    description: str = ""
    axis_configs: Dict[str, AxisConfig] = field(default_factory=dict)
    trigger_configs: Dict[str, TriggerConfig] = field(default_factory=dict)
    button_maps: Dict[str, ButtonMap] = field(default_factory=dict)
    gesture_map: Dict[str, str] = field(default_factory=dict)

    def to_filter(self) -> SignalFilter:
        return SignalFilter(
            axis_configs=self.axis_configs.copy(),
            trigger_configs=self.trigger_configs.copy(),
        )