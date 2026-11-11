"""Print each task's resolved PPO runner budget (config-level, no CLI override)."""
import isaacgym  # noqa: F401  must precede torch
import legged_gym.envs  # noqa: F401  registers tasks
from legged_gym.envs.go2.go2_teacher_config import Go2TeacherCfgPPO
from legged_gym.envs.go2.go2_flat_config import Go2FlatCfgPPO
from legged_gym.envs.go2.go2_student_config import Go2StudentCfgPPO
from legged_gym.envs.go2.go2_depth_config import Go2DepthCfgPPO

print(f"{'task':9s} {'max_iter':>9s} {'save_int':>9s} {'nsteps':>7s}")
for name, cls in [('teacher', Go2TeacherCfgPPO), ('flat', Go2FlatCfgPPO),
                  ('student', Go2StudentCfgPPO), ('depth', Go2DepthCfgPPO)]:
    c = cls()
    print(f"{name:9s} {c.runner.max_iterations:>9d} {c.runner.save_interval:>9d} "
          f"{c.runner.num_steps_per_env:>7d}")
