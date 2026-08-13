"""Focused tests for monotonic curriculum-latch ramps."""

import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "curriculum_ramp.py"
_SPEC = importlib.util.spec_from_file_location("curriculum_ramp", _MODULE_PATH)
cr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cr)


def test_untriggered_latch_stays_zero():
    assert cr.latched_linear_alpha(1000, -1, 100) == 0.0


def test_delayed_ramp_is_linear_and_saturates():
    trigger = 100
    assert cr.latched_linear_alpha(100, trigger, 200, delay_steps=50) == 0.0
    assert cr.latched_linear_alpha(150, trigger, 200, delay_steps=50) == 0.0
    assert cr.latched_linear_alpha(250, trigger, 200, delay_steps=50) == 0.5
    assert cr.latched_linear_alpha(350, trigger, 200, delay_steps=50) == 1.0
    assert cr.latched_linear_alpha(1000, trigger, 200, delay_steps=50) == 1.0


def test_zero_duration_preserves_hard_latch_behavior():
    assert cr.latched_linear_alpha(10, 10, 0) == 1.0
