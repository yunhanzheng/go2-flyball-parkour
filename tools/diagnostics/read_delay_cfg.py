"""Read the resolved action-delay knobs out of a run's saved env_cfg.pkl."""
import isaacgym  # noqa: F401
import pickle
import sys
import legged_gym.envs  # noqa: F401  the pickle references legged_gym classes

path = sys.argv[1]
cfg = pickle.load(open(path, 'rb'))
print("TOP", type(cfg).__name__, list(cfg.keys())[:8] if isinstance(cfg, dict) else '')
d = cfg['domain_rand']
print("DR_TYPE", type(d).__name__)
items = d.items() if isinstance(d, dict) else [(k, getattr(d, k)) for k in dir(d) if not k.startswith('_')]
for k, v in sorted(items):
    if k.startswith('action_delay'):
        print(f"  {k:34s} {v}")
