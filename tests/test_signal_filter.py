import math
import pytest
from intentpad.signal_filter import SignalFilter, SignalFilterConfig


class TestSignalFilter:
    def test_sub_deadzone_noise_returns_zero(self):
        filter = SignalFilter(SignalFilterConfig(deadzone=0.15, alpha=1.0))
        x, y = filter.process(0.06, 0.08)
        assert x == 0.0
        assert y == 0.0

    def test_sub_deadzone_negative_noise_returns_zero(self):
        filter = SignalFilter(SignalFilterConfig(deadzone=0.15, alpha=1.0))
        x, y = filter.process(-0.06, -0.08)
        assert x == 0.0
        assert y == 0.0

    def test_diagonal_clamping_normalizes_to_unit(self):
        filter = SignalFilter(SignalFilterConfig(deadzone=0.0, alpha=1.0))
        x, y = filter.process(1.0, 1.0)
        magnitude = math.hypot(x, y)
        assert magnitude <= 1.0 + 1e-6
        assert abs(x - y) < 1e-6
        expected = 1.0 / math.sqrt(2)
        assert abs(x - expected) < 1e-6

    def test_diagonal_clamping_larger_values(self):
        filter = SignalFilter(SignalFilterConfig(deadzone=0.0, alpha=1.0))
        x, y = filter.process(2.0, 2.0)
        magnitude = math.hypot(x, y)
        assert magnitude <= 1.0 + 1e-6

    def test_ema_smoothing_step_response(self):
        filter = SignalFilter(SignalFilterConfig(deadzone=0.0, alpha=0.60))
        filter.reset()
        
        x1, y1 = filter.process(1.0, 0.0)
        assert abs(x1 - 0.60) < 1e-6
        assert y1 == 0.0
        
        x2, y2 = filter.process(1.0, 0.0)
        expected_x2 = 0.60 + 0.60 * 0.40
        assert abs(x2 - expected_x2) < 1e-6
        
        x3, y3 = filter.process(1.0, 0.0)
        expected_x3 = expected_x2 + 0.60 * (1.0 - expected_x2)
        assert abs(x3 - expected_x3) < 1e-6

    def test_ema_smoothing_decay_to_zero(self):
        filter = SignalFilter(SignalFilterConfig(deadzone=0.0, alpha=0.60))
        filter.process(1.0, 0.0)
        filter.process(1.0, 0.0)
        
        x, y = filter.process(0.0, 0.0)
        assert x < 0.84
        
        for _ in range(10):
            x, _ = filter.process(0.0, 0.0)
        assert x < 0.05

    def test_reset_clears_state(self):
        filter = SignalFilter(SignalFilterConfig(deadzone=0.0, alpha=0.60))
        filter.process(1.0, 1.0)
        filter.process(1.0, 1.0)
        filter.reset()
        x, y = filter.get_state()
        assert x == 0.0
        assert y == 0.0

    def test_deadzone_boundary(self):
        filter = SignalFilter(SignalFilterConfig(deadzone=0.15, alpha=1.0))
        x, y = filter.process(0.15, 0.0)
        assert x == 0.15
        
        x, y = filter.process(0.16, 0.0)
        assert x == 0.16

    def test_clamping_negative_values(self):
        filter = SignalFilter(SignalFilterConfig(deadzone=0.0, alpha=1.0))
        x, y = filter.process(-1.5, -1.5)
        assert x >= -1.0
        assert y >= -1.0
        magnitude = math.hypot(x, y)
        assert magnitude <= 1.0 + 1e-6

    def test_configurable_parameters(self):
        config = SignalFilterConfig(deadzone=0.25, alpha=0.30)
        filter = SignalFilter(config)
        assert filter.deadzone == 0.25
        assert filter.alpha == 0.30
        
        filter.deadzone = 0.35
        filter.alpha = 0.45
        assert filter.deadzone == 0.35
        assert filter.alpha == 0.45

    def test_config_bounds(self):
        filter = SignalFilter(SignalFilterConfig(deadzone=0.15, alpha=0.60))
        filter.deadzone = -0.5
        assert filter.deadzone == 0.0
        filter.deadzone = 1.5
        assert filter.deadzone == 1.0
        
        filter.alpha = -0.5
        assert filter.alpha == 0.0
        filter.alpha = 1.5
        assert filter.alpha == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])