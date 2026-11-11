"""Check the action-delay distillation guard flag resolves as intended per task."""
import isaacgym  # noqa: F401  must precede torch
import legged_gym.envs  # noqa: F401  registers tasks
from legged_gym.envs.go2.go2_teacher_config import Go2TeacherCfg
from legged_gym.envs.go2.go2_student_config import Go2StudentCfg
from legged_gym.envs.go2.go2_flat_config import Go2FlatCfg
from legged_gym.envs.go2.go2_depth_config import Go2DepthCfg

for name, cls in [('teacher', Go2TeacherCfg), ('flat', Go2FlatCfg),
                  ('student', Go2StudentCfg), ('depth', Go2DepthCfg)]:
    d = cls().domain_rand
    guard = getattr(d, 'action_delay_distillation_guard', '<absent -> default True>')
    print(f"CONFIG {name:8s} guard={guard!s:26s} "
          f"latch_mean_level={d.action_delay_latch_mean_level} "
          f"start_latched={d.action_delay_start_latched}")
