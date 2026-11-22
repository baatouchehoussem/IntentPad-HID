import json
import logging
from pathlib import Path
from typing import Dict, Optional, List
from .pipeline import (
    Profile, AxisConfig, TriggerConfig, ButtonMap, CurveType, SignalFilter
)

logger = logging.getLogger(__name__)


DEFAULT_PROFILES = {
    "gta_driving": Profile(
        name="gta_driving",
        description="Optimized for GTA driving - smooth steering, responsive triggers",
        axis_configs={
            "lx": AxisConfig(deadzone=0.15, sensitivity=1.0, curve=CurveType.EXPONENTIAL, ema_alpha=0.25, radial_deadzone=0.1),
            "ly": AxisConfig(deadzone=0.15, sensitivity=0.9, curve=CurveType.EXPONENTIAL, ema_alpha=0.25, radial_deadzone=0.1),
            "rx": AxisConfig(deadzone=0.2, sensitivity=0.8, curve=CurveType.LINEAR, ema_alpha=0.3, radial_deadzone=0.1),
            "ry": AxisConfig(deadzone=0.2, sensitivity=0.8, curve=CurveType.LINEAR, ema_alpha=0.3, radial_deadzone=0.1),
        },
        trigger_configs={
            "lt": TriggerConfig(deadzone=0.05, sensitivity=1.0, curve=CurveType.LINEAR, ema_alpha=0.2),
            "rt": TriggerConfig(deadzone=0.05, sensitivity=1.0, curve=CurveType.EXPONENTIAL, ema_alpha=0.2),
        },
        gesture_map={
            "steer_left": "lx", "steer_right": "lx",
            "accelerate": "rt", "brake": "lt",
            "look_left": "rx", "look_right": "rx",
            "look_up": "ry", "look_down": "ry",
        }
    ),
    "fps_tactical": Profile(
        name="fps_tactical",
        description="Optimized for FPS - precise aim, quick trigger response",
        axis_configs={
            "lx": AxisConfig(deadzone=0.1, sensitivity=1.0, curve=CurveType.LINEAR, ema_alpha=0.35, radial_deadzone=0.05),
            "ly": AxisConfig(deadzone=0.1, sensitivity=1.0, curve=CurveType.LINEAR, ema_alpha=0.35, radial_deadzone=0.05),
            "rx": AxisConfig(deadzone=0.05, sensitivity=1.2, curve=CurveType.EXPONENTIAL, ema_alpha=0.15, radial_deadzone=0.05),
            "ry": AxisConfig(deadzone=0.05, sensitivity=1.2, curve=CurveType.EXPONENTIAL, ema_alpha=0.15, radial_deadzone=0.05),
        },
        trigger_configs={
            "lt": TriggerConfig(deadzone=0.02, sensitivity=1.0, curve=CurveType.LINEAR, ema_alpha=0.1),
            "rt": TriggerConfig(deadzone=0.02, sensitivity=1.0, curve=CurveType.LINEAR, ema_alpha=0.1),
        },
        gesture_map={
            "move_left": "lx", "move_right": "lx",
            "move_forward": "ly", "move_backward": "ly",
            "aim_x": "rx", "aim_y": "ry",
            "ads": "lt", "fire": "rt",
        }
    ),
    "desktop_assistive": Profile(
        name="desktop_assistive",
        description="Assistive desktop control - gentle curves, large deadzones for stability",
        axis_configs={
            "lx": AxisConfig(deadzone=0.25, sensitivity=0.6, curve=CurveType.LOGARITHMIC, ema_alpha=0.4, radial_deadzone=0.2),
            "ly": AxisConfig(deadzone=0.25, sensitivity=0.6, curve=CurveType.LOGARITHMIC, ema_alpha=0.4, radial_deadzone=0.2),
            "rx": AxisConfig(deadzone=0.3, sensitivity=0.5, curve=CurveType.LINEAR, ema_alpha=0.5, radial_deadzone=0.2),
            "ry": AxisConfig(deadzone=0.3, sensitivity=0.5, curve=CurveType.LINEAR, ema_alpha=0.5, radial_deadzone=0.2),
        },
        trigger_configs={
            "lt": TriggerConfig(deadzone=0.1, sensitivity=0.8, curve=CurveType.LINEAR, ema_alpha=0.3),
            "rt": TriggerConfig(deadzone=0.1, sensitivity=0.8, curve=CurveType.LINEAR, ema_alpha=0.3),
        },
        gesture_map={
            "cursor_x": "rx", "cursor_y": "ry",
            "scroll_x": "lx", "scroll_y": "ly",
            "left_click": "rt", "right_click": "lt",
        }
    ),
}


class ProfileManager:
    def __init__(self, profiles_dir: Optional[Path] = None):
        self._profiles: Dict[str, Profile] = {}
        self._active_profile: Optional[str] = None
        self._profiles_dir = profiles_dir or Path(__file__).parent / "profiles"
        self._profiles_dir.mkdir(exist_ok=True)
        self._load_builtins()
        self._load_custom()

    def _load_builtins(self) -> None:
        for profile in DEFAULT_PROFILES.values():
            self._profiles[profile.name] = profile
        logger.info(f"Loaded {len(DEFAULT_PROFILES)} built-in profiles")

    def _load_custom(self) -> None:
        for path in self._profiles_dir.glob("*.json"):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                profile = self._deserialize(data)
                self._profiles[profile.name] = profile
                logger.info(f"Loaded custom profile: {profile.name}")
            except Exception as e:
                logger.warning(f"Failed to load profile {path}: {e}")

    def _deserialize(self, data: dict) -> Profile:
        axis_configs = {}
        for k, v in data.get("axis_configs", {}).items():
            axis_configs[k] = AxisConfig(
                deadzone=v.get("deadzone", 0.1),
                sensitivity=v.get("sensitivity", 1.0),
                curve=CurveType(v.get("curve", "linear")),
                invert=v.get("invert", False),
                ema_alpha=v.get("ema_alpha", 0.3),
                radial_deadzone=v.get("radial_deadzone", 0.0),
            )
        trigger_configs = {}
        for k, v in data.get("trigger_configs", {}).items():
            trigger_configs[k] = TriggerConfig(
                deadzone=v.get("deadzone", 0.05),
                sensitivity=v.get("sensitivity", 1.0),
                curve=CurveType(v.get("curve", "linear")),
                ema_alpha=v.get("ema_alpha", 0.3),
            )
        button_maps = {}
        for k, v in data.get("button_maps", {}).items():
            button_maps[k] = ButtonMap(
                tap=v.get("tap"),
                hold=v.get("hold"),
                hold_threshold=v.get("hold_threshold", 0.5),
            )
        return Profile(
            name=data["name"],
            description=data.get("description", ""),
            axis_configs=axis_configs,
            trigger_configs=trigger_configs,
            button_maps=button_maps,
            gesture_map=data.get("gesture_map", {}),
        )

    def get_profile(self, name: str) -> Optional[Profile]:
        return self._profiles.get(name)

    def get_active_profile(self) -> Optional[Profile]:
        if self._active_profile:
            return self._profiles.get(self._active_profile)
        return None

    def set_active_profile(self, name: str) -> bool:
        if name in self._profiles:
            self._active_profile = name
            logger.info(f"Active profile set to: {name}")
            return True
        return False

    def list_profiles(self) -> List[str]:
        return list(self._profiles.keys())

    def create_filter(self, profile_name: Optional[str] = None) -> SignalFilter:
        profile = self.get_profile(profile_name or self._active_profile or "gta_driving")
        if profile:
            return profile.to_filter()
        return SignalFilter()

    def save_profile(self, profile: Profile) -> None:
        self._profiles[profile.name] = profile
        path = self._profiles_dir / f"{profile.name}.json"
        data = {
            "name": profile.name,
            "description": profile.description,
            "axis_configs": {k: {
                "deadzone": v.deadzone,
                "sensitivity": v.sensitivity,
                "curve": v.curve.value,
                "invert": v.invert,
                "ema_alpha": v.ema_alpha,
                "radial_deadzone": getattr(v, 'radial_deadzone', 0.0),
            } for k, v in profile.axis_configs.items()},
            "trigger_configs": {k: {
                "deadzone": v.deadzone,
                "sensitivity": v.sensitivity,
                "curve": v.curve.value,
                "ema_alpha": v.ema_alpha,
            } for k, v in profile.trigger_configs.items()},
            "button_maps": {k: {
                "tap": v.tap,
                "hold": v.hold,
                "hold_threshold": v.hold_threshold,
            } for k, v in profile.button_maps.items()},
            "gesture_map": profile.gesture_map,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved profile: {profile.name}")

    def delete_profile(self, name: str) -> bool:
        if name in DEFAULT_PROFILES:
            logger.warning(f"Cannot delete built-in profile: {name}")
            return False
        if name in self._profiles:
            del self._profiles[name]
            path = self._profiles_dir / f"{name}.json"
            if path.exists():
                path.unlink()
            logger.info(f"Deleted profile: {name}")
            return True
        return False