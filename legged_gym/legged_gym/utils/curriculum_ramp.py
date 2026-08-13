"""Small, simulator-independent helpers for sticky curriculum ramps."""


def latched_linear_alpha(current_step, trigger_step, ramp_steps, delay_steps=0):
    """Return a sticky latch's delayed linear progress in ``[0, 1]``.

    ``trigger_step < 0`` means that the latch has not fired. A non-positive
    ramp duration preserves the legacy hard-switch behavior once triggered.
    """
    if trigger_step < 0:
        return 0.0
    elapsed = max(0, int(current_step) - int(trigger_step) - max(0, int(delay_steps)))
    if int(ramp_steps) <= 0:
        return 1.0
    return min(float(elapsed) / float(ramp_steps), 1.0)
