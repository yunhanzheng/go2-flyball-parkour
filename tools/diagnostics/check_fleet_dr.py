"""Verify the fleet-wide / monotonic DR + cold-start curriculum knobs per task."""
import isaacgym  # noqa: F401  must precede torch
import legged_gym.envs  # noqa: F401  registers tasks
from legged_gym.envs.go2.go2_teacher_config import Go2TeacherCfg
from legged_gym.envs.go2.go2_flat_config import Go2FlatCfg
from legged_gym.envs.go2.go2_student_config import Go2StudentCfg
from legged_gym.envs.go2.go2_depth_config import Go2DepthCfg

hdr = f"{'task':9s} {'init_lvl':>8s} {'min_lvl':>7s} {'per_env_draw':>12s} {'monotonic':>9s} {'hurdle_perenv':>13s}"
print(hdr)
for name, cls in [('teacher', Go2TeacherCfg), ('flat', Go2FlatCfg),
                  ('student', Go2StudentCfg), ('depth', Go2DepthCfg)]:
    c = cls()
    print(f"{name:9s} {c.terrain.max_init_terrain_level:>8} "
          f"{c.terrain.min_terrain_level:>7} "
          f"{str(getattr(c.domain_rand, 'dr_scale_sample_per_env', None)):>12s} "
          f"{str(getattr(c.domain_rand, 'dr_ramp_monotonic', '<absent>')):>9s} "
          f"{str(getattr(c.rewards, 'hurdle_clearance_per_env_draw', None)):>13s}")
