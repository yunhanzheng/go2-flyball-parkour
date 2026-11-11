"""Check the config-driven checkpoint ladder resolves per task."""
import os
import sys

ROOT = '/home/ubuntu22/data/go2-flyball-parkour'
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'legged_gym'))
sys.path.insert(0, os.path.join(ROOT, 'rsl_rl'))

import isaacgym  # noqa: F401,E402
import legged_gym.envs  # noqa: F401,E402
from legged_gym.utils.task_registry import task_registry  # noqa: E402
from legged_gym.utils.helpers import class_to_dict  # noqa: E402

print(f"{'task':<12} {'max_iter':>9} {'save_int':>8} {'throttle':<34} {'<=5000':>7} {'>=5000':>7}")
for task in ('go2_teacher', 'go2_flat', 'go2_student', 'go2_depth'):
    _env_cfg, train_cfg = task_registry.get_cfgs(task)
    r = class_to_dict(train_cfg.runner)
    interval = r['save_interval']
    ladder = tuple(r.get('save_throttle') or ((0, 1), (2500, 2), (5000, 5)))

    def mult_at(it):
        m = 1
        for start, factor in ladder:
            if it >= start:
                m = factor
        return m

    print(f"{task:<12} {r['max_iterations']:>9} {interval:>8} {str(ladder):<34} "
          f"{mult_at(3000) * interval:>7} {mult_at(9000) * interval:>7}")

# Worst-case loss window at the measured 4.7 s/iter.
for label, step in (('teacher @2000-step saves', 2000), ('teacher @5000-step saves', 5000)):
    print(f"{label}: {step * 4.7 / 3600:.1f} h")
