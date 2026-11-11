"""Mirror the dr_scale block from legged_robot.py to prove the ceiling is sticky
while the per-env draw is untouched."""
import torch

s_lo, s_hi = 0.0, 1.0
FULL_AT = 6.0
monotonic = True
alpha_latched = 0.0

# 课程的 mean level 先升后降（模拟 demotion）
mean_levels = [0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 4.0, 0.5]
envs = 4096

print(f"{'mean_lvl':>9s} {'latched':>8s} {'ceiling':>8s} {'dr_mean':>8s} {'dr_min':>7s} {'dr_max':>7s}")
for ml in mean_levels:
    mean_level = torch.tensor(ml)
    alpha = (mean_level / max(FULL_AT, 1e-6)).clamp(0., 1.)
    if monotonic:
        alpha_latched = max(alpha_latched, float(alpha))
        alpha = torch.tensor(alpha_latched)
    ceiling = s_lo + (s_hi - s_lo) * alpha
    dr = s_lo + (ceiling - s_lo) * torch.rand(envs)   # per-env draw (main 的机制)
    print(f"{ml:>9.1f} {alpha_latched:>8.3f} {float(ceiling):>8.3f} "
          f"{dr.mean():>8.3f} {dr.min():>7.3f} {dr.max():>7.3f}")
