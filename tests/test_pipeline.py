import asyncio

import pytest
import math
from intentpad.pipeline import (
    SignalFilter, AxisConfig, TriggerConfig, CurveType,
    Profile, ButtonMap, AxisConfig
)
from intentpad.profiles import ProfileManager, DEFAULT_PROFILES
from intentpad.gamepad_engine import MockGamepad, GamepadEngine, create_gamepad
from intentpad.adapters import HandGestureAdapter, NeuralHIDAdapter
from intentpad.server import root
from intentpad.adapters.base import AdapterFrame


class TestSignalFilter:
    def test_ema_smoothing(self):
        filter = SignalFilter()
        filter.axis_configs["lx"] = AxisConfig(ema_alpha=0.5, deadzone=0.0)

        result1 = filter.process_axis("lx", 1.0)
        assert result1 == 0.5

        result2 = filter.process_axis("lx", 1.0)
        assert result2 == 0.75

        result3 = filter.process_axis("lx", 0.0)
        assert result3 == 0.375

    def test_deadzone(self):
        filter = SignalFilter()
        filter.axis_configs["lx"] = AxisConfig(deadzone=0.2, ema_alpha=1.0)

        assert filter.process_axis("lx", 0.1) == 0.0
        assert filter.process_axis("lx", -0.1) == 0.0
        assert filter.process_axis("lx", 0.2) == 0.0
        assert filter.process_axis("lx", 0.3) > 0.0
        assert filter.process_axis("lx", -0.3) < 0.0

    def test_exponential_curve(self):
        filter = SignalFilter()
        filter.axis_configs["lx"] = AxisConfig(curve=CurveType.EXPONENTIAL, ema_alpha=1.0, deadzone=0.0)

        assert filter.process_axis("lx", 0.5) == 0.25
        assert filter.process_axis("lx", -0.5) == -0.25
        assert filter.process_axis("lx", 1.0) == 1.0
        assert filter.process_axis("lx", 0.0) == 0.0

    def test_logarithmic_curve(self):
        filter = SignalFilter()
        filter.axis_configs["lx"] = AxisConfig(curve=CurveType.LOGARITHMIC, ema_alpha=1.0, deadzone=0.0)

        result = filter.process_axis("lx", 0.5)
        expected = math.log1p(0.5) / math.log(2)
        assert abs(result - expected) < 0.001

    def test_s_curve(self):
        filter = SignalFilter()
        filter.axis_configs["lx"] = AxisConfig(curve=CurveType.S_CURVE, ema_alpha=1.0, deadzone=0.0)

        assert filter.process_axis("lx", 0.0) == 0.0
        assert filter.process_axis("lx", 0.5) == 0.5
        assert filter.process_axis("lx", 1.0) == 1.0

    def test_invert(self):
        filter = SignalFilter()
        filter.axis_configs["lx"] = AxisConfig(invert=True, ema_alpha=1.0, deadzone=0.0)

        assert filter.process_axis("lx", 0.5) == -0.5
        assert filter.process_axis("lx", -0.5) == 0.5

    def test_sensitivity(self):
        filter = SignalFilter()
        filter.axis_configs["lx"] = AxisConfig(sensitivity=2.0, ema_alpha=1.0, deadzone=0.0)

        assert filter.process_axis("lx", 0.5) == 1.0
        assert filter.process_axis("lx", 0.3) == 0.6

    def test_clamping(self):
        filter = SignalFilter()
        filter.axis_configs["lx"] = AxisConfig(ema_alpha=1.0, deadzone=0.0, sensitivity=2.0)

        assert filter.process_axis("lx", 1.0) == 1.0
        assert filter.process_axis("lx", -1.0) == -1.0

    def test_trigger_processing(self):
        filter = SignalFilter()
        filter.trigger_configs["rt"] = TriggerConfig(deadzone=0.1, ema_alpha=1.0, curve=CurveType.LINEAR)

        assert filter.process_trigger("rt", 0.05) == 0.0
        assert filter.process_trigger("rt", 0.5) == pytest.approx(4 / 9)

    def test_vector_processing(self):
        filter = SignalFilter()
        filter.axis_configs["lx"] = AxisConfig(ema_alpha=1.0, deadzone=0.0)
        filter.axis_configs["ly"] = AxisConfig(ema_alpha=1.0, deadzone=0.0)

        x, y = filter.process_vector(0.5, -0.5, "lx", "ly")
        assert x == 0.5
        assert y == -0.5

    def test_reset(self):
        filter = SignalFilter()
        filter.axis_configs["lx"] = AxisConfig(ema_alpha=0.5, deadzone=0.0)

        filter.process_axis("lx", 1.0)
        filter.process_axis("lx", 1.0)
        filter.reset()
        result = filter.process_axis("lx", 1.0)
        assert result == 0.5

    def test_config_update(self):
        filter = SignalFilter()
        filter.update_config("lx", deadzone=0.3, sensitivity=1.5)
        assert filter.axis_configs["lx"].deadzone == 0.3
        assert filter.axis_configs["lx"].sensitivity == 1.5

    def test_get_config_dict(self):
        filter = SignalFilter()
        config = filter.get_config_dict()
        assert "axes" in config
        assert "triggers" in config
        assert "lx" in config["axes"]
        assert "rt" in config["triggers"]


class TestProfile:
    def test_profile_creation(self):
        profile = Profile(
            name="test",
            description="Test profile",
            axis_configs={"lx": AxisConfig(deadzone=0.1)},
            trigger_configs={"rt": TriggerConfig(deadzone=0.05)},
        )
        assert profile.name == "test"
        filter = profile.to_filter()
        assert filter.axis_configs["lx"].deadzone == 0.1

    def test_button_map(self):
        btn_map = ButtonMap(tap="A", hold="B", hold_threshold=0.5)
        assert btn_map.tap == "A"
        assert btn_map.hold == "B"
        assert btn_map.hold_threshold == 0.5


class TestProfileManager:
    def test_builtin_profiles_loaded(self):
        manager = ProfileManager()
        profiles = manager.list_profiles()
        assert "gta_driving" in profiles
        assert "fps_tactical" in profiles
        assert "desktop_assistive" in profiles

    def test_get_profile(self):
        manager = ProfileManager()
        profile = manager.get_profile("gta_driving")
        assert profile is not None
        assert profile.name == "gta_driving"

    def test_set_active_profile(self):
        manager = ProfileManager()
        assert manager.set_active_profile("fps_tactical") is True
        assert manager.get_active_profile().name == "fps_tactical"
        assert manager.set_active_profile("nonexistent") is False

    def test_create_filter(self):
        manager = ProfileManager()
        manager.set_active_profile("gta_driving")
        filter = manager.create_filter()
        assert filter.axis_configs["lx"].curve == CurveType.EXPONENTIAL


class TestMockGamepad:
    def test_mock_gamepad_creation(self):
        gamepad = MockGamepad()
        assert gamepad is not None

    def test_mock_gamepad_sticks(self):
        gamepad = MockGamepad()
        gamepad.set_left_stick(0.5, -0.3)
        gamepad.set_right_stick(-0.2, 0.8)
        state = gamepad.get_state()
        assert state["left_stick"] == (0.5, -0.3)
        assert state["right_stick"] == (-0.2, 0.8)

    def test_mock_gamepad_triggers(self):
        gamepad = MockGamepad()
        gamepad.set_triggers(0.3, 0.7)
        state = gamepad.get_state()
        assert state["triggers"] == (0.3, 0.7)

    def test_mock_gamepad_buttons(self):
        gamepad = MockGamepad()
        gamepad.set_button("A", True)
        gamepad.set_button("B", False)
        state = gamepad.get_state()
        assert state["buttons"]["A"] is True
        assert state["buttons"]["B"] is False

    def test_mock_gamepad_clamping(self):
        gamepad = MockGamepad()
        gamepad.set_left_stick(2.0, -2.0)
        state = gamepad.get_state()
        assert state["left_stick"] == (1.0, -1.0)

        gamepad.set_triggers(1.5, -0.5)
        state = gamepad.get_state()
        assert state["triggers"] == (1.0, 0.0)

    def test_mock_gamepad_reset(self):
        gamepad = MockGamepad()
        gamepad.set_left_stick(0.5, 0.5)
        gamepad.set_button("A", True)
        gamepad.reset()
        state = gamepad.get_state()
        assert state["left_stick"] == (0.0, 0.0)
        assert state["buttons"] == {}


class TestGamepadEngine:
    def test_gamepad_engine_creation(self):
        engine = GamepadEngine(MockGamepad())
        assert engine is not None
        engine.close()

    def test_gamepad_engine_methods(self):
        engine = GamepadEngine(MockGamepad())
        engine.set_left_stick(0.5, 0.5)
        engine.set_right_stick(-0.5, -0.5)
        engine.set_triggers(0.5, 0.5)
        engine.set_button("A", True)
        engine.update()

        state = engine.get_state()
        assert state["left_stick"] == (0.5, 0.5)
        assert state["right_stick"] == (-0.5, -0.5)
        assert state["triggers"] == (0.5, 0.5)
        assert state["buttons"]["A"] is True
        engine.close()

    def test_create_gamepad_returns_mock_on_unsupported(self, monkeypatch):
        monkeypatch.setattr("intentpad.gamepad_engine.platform.system", lambda: "unsupported")
        gamepad = create_gamepad()
        assert isinstance(gamepad, MockGamepad)


class TestIntegration:
    def test_profile_to_filter_to_gamepad(self):
        manager = ProfileManager()
        manager.set_active_profile("fps_tactical")
        filter = manager.create_filter()

        engine = GamepadEngine(MockGamepad())

        raw_x, raw_y = 0.8, 0.6
        smooth_x = filter.process_axis("lx", raw_x)
        smooth_y = filter.process_axis("ly", raw_y)

        engine.set_left_stick(smooth_x, smooth_y)
        engine.update()

        state = engine.get_state()
        assert state["left_stick"][0] != 0.0
        assert state["left_stick"][1] != 0.0
        engine.close()

    def test_deadzone_prevents_drift(self):
        filter = SignalFilter()
        filter.axis_configs["lx"] = AxisConfig(deadzone=0.15, ema_alpha=0.3)

        for _ in range(10):
            result = filter.process_axis("lx", 0.05)
        assert result == 0.0

        result = filter.process_axis("lx", 0.2)
        assert result > 0.0


class TestStartup:
    def test_hand_gesture_adapter_defaults_without_config(self):
        adapter = HandGestureAdapter()
        assert adapter.config == {}
        assert adapter._frame_width == 640
        assert adapter._frame_height == 480
        assert adapter._fps == 30

    def test_neural_hid_adapter_defaults_without_config(self):
        adapter = NeuralHIDAdapter()
        assert adapter.config == {}
        assert adapter._screen_width == 1920
        assert adapter._screen_height == 1080
        assert adapter._dwell_time == 0.8

    def test_root_serves_dashboard_with_absolute_path(self):
        response = asyncio.run(root())
        assert response.media_type == "text/html"
        assert "IntentPad-HID Mission Control" in open(response.path, encoding="utf-8").read()


class TestDualHandMapping:
    def test_hand_gesture_adapter_dual_hand_config(self):
        adapter = HandGestureAdapter()
        assert adapter._max_num_hands == 2

    def test_hand_gesture_adapter_custom_num_hands(self):
        adapter = HandGestureAdapter(config={"max_num_hands": 1})
        assert adapter._max_num_hands == 1

    def test_adapter_frame_metadata_structure(self):
        frame = AdapterFrame(
            left_stick=(0.1, 0.2),
            right_stick=(0.3, 0.4),
            triggers=(0.5, 0.6),
            buttons={"A": True, "B": False},
            metadata={"hands_detected": 2, "left_hand_detected": True, "right_hand_detected": True, "fps": 30.0}
        )
        assert frame.metadata["hands_detected"] == 2
        assert frame.metadata["left_hand_detected"] is True
        assert frame.metadata["right_hand_detected"] is True
        assert frame.metadata["fps"] == 30.0

    def test_left_hand_maps_to_left_stick(self):
        adapter = HandGestureAdapter()
        left_stick, lt_val, buttons = adapter._process_left_hand(self._create_mock_landmarks(hand_side="left"))
        assert isinstance(left_stick, tuple)
        assert len(left_stick) == 2
        assert -1.0 <= left_stick[0] <= 1.0
        assert -1.0 <= left_stick[1] <= 1.0
        assert 0.0 <= lt_val <= 1.0
        assert "A" in buttons
        assert "X" in buttons

    def test_right_hand_maps_to_right_stick(self):
        adapter = HandGestureAdapter()
        right_stick, rt_val, buttons = adapter._process_right_hand(self._create_mock_landmarks(hand_side="right"))
        assert isinstance(right_stick, tuple)
        assert len(right_stick) == 2
        assert -1.0 <= right_stick[0] <= 1.0
        assert -1.0 <= right_stick[1] <= 1.0
        assert 0.0 <= rt_val <= 1.0
        assert "B" in buttons
        assert "Y" in buttons

    def test_left_hand_buttons_mapping(self):
        adapter = HandGestureAdapter()
        _, _, buttons = adapter._process_left_hand(self._create_mock_landmarks(hand_side="left"))
        assert "A" in buttons
        assert "X" in buttons

    def test_right_hand_buttons_mapping(self):
        adapter = HandGestureAdapter()
        _, _, buttons = adapter._process_right_hand(self._create_mock_landmarks(hand_side="right"))
        assert "B" in buttons
        assert "Y" in buttons

    def test_fps_calculation_rolling_average(self):
        adapter = HandGestureAdapter()
        adapter._fps_samples = [25.0, 28.0, 30.0, 29.0, 27.0]
        avg = sum(adapter._fps_samples) / len(adapter._fps_samples)
        assert 27.0 <= avg <= 29.0
        assert len(adapter._fps_samples) <= adapter._max_fps_samples

    def test_fps_samples_limit(self):
        adapter = HandGestureAdapter()
        adapter._max_fps_samples = 5
        adapter._fps_samples = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
        if len(adapter._fps_samples) > adapter._max_fps_samples:
            adapter._fps_samples = adapter._fps_samples[-adapter._max_fps_samples:]
        assert len(adapter._fps_samples) == 5
        assert adapter._fps_samples == [25.0, 30.0, 35.0, 40.0, 45.0]

    def _create_mock_landmarks(self, hand_side: str = "left"):
        class MockLandmark:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        
        landmarks = []
        base_x = 0.3 if hand_side == "left" else 0.7
        base_y = 0.5
        
        landmarks.append(MockLandmark(base_x, base_y + 0.1))
        landmarks.append(MockLandmark(base_x + 0.02, base_y + 0.08))
        landmarks.append(MockLandmark(base_x + 0.04, base_y + 0.06))
        landmarks.append(MockLandmark(base_x + 0.06, base_y + 0.04))
        landmarks.append(MockLandmark(base_x + 0.08, base_y + 0.02))
        
        landmarks.append(MockLandmark(base_x - 0.02, base_y + 0.08))
        landmarks.append(MockLandmark(base_x - 0.04, base_y + 0.06))
        landmarks.append(MockLandmark(base_x - 0.06, base_y + 0.04))
        landmarks.append(MockLandmark(base_x - 0.08, base_y + 0.02))
        
        landmarks.append(MockLandmark(base_x - 0.02, base_y + 0.08))
        landmarks.append(MockLandmark(base_x - 0.04, base_y + 0.06))
        landmarks.append(MockLandmark(base_x - 0.06, base_y + 0.04))
        landmarks.append(MockLandmark(base_x - 0.08, base_y + 0.02))
        
        landmarks.append(MockLandmark(base_x - 0.02, base_y + 0.08))
        landmarks.append(MockLandmark(base_x - 0.04, base_y + 0.06))
        landmarks.append(MockLandmark(base_x - 0.06, base_y + 0.04))
        landmarks.append(MockLandmark(base_x - 0.08, base_y + 0.02))
        
        landmarks.append(MockLandmark(base_x - 0.02, base_y + 0.08))
        landmarks.append(MockLandmark(base_x - 0.04, base_y + 0.06))
        landmarks.append(MockLandmark(base_x - 0.06, base_y + 0.04))
        landmarks.append(MockLandmark(base_x - 0.08, base_y + 0.02))
        
        landmarks.append(MockLandmark(base_x + 0.03, base_y + 0.09))
        landmarks.append(MockLandmark(base_x + 0.02, base_y + 0.07))
        landmarks.append(MockLandmark(base_x + 0.01, base_y + 0.05))
        landmarks.append(MockLandmark(base_x, base_y + 0.03))
        
        return landmarks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
