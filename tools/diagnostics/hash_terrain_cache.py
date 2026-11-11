"""Recompute the terrain-cache digest for the live go2_teacher config.

Mirrors Terrain._cache_path (cfg.terrain fields + generator sources + numpy RNG
state at seed(1)) so we can tell whether a "CACHE MISS" came from a config change
or from a different RNG position than the process that wrote the pickle.
"""
import os
import sys
import hashlib

import numpy as np

ROOT = '/home/ubuntu22/data/go2-flyball-parkour'
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'legged_gym'))

import isaacgym  # noqa: F401,E402  must precede torch
import legged_gym.envs  # noqa: F401,E402
from legged_gym.utils.task_registry import task_registry  # noqa: E402
from legged_gym.utils import terrain as terrain_mod  # noqa: E402


def digest_for(cfg, seed=1):
    np.random.seed(seed)
    digest = hashlib.sha1()
    special = []
    for key in sorted(dir(cfg)):
        if key.startswith('__') or key in terrain_mod._CACHE_KEY_SKIP:
            continue
        value = getattr(cfg, key)
        if callable(value) or isinstance(value, type):
            continue
        if key.startswith('flyball_') and isinstance(value, (dict, list, tuple)):
            special.append(key)
        digest.update(repr((key, value)).encode())
    for path in terrain_mod._CACHE_GENERATOR_FILES:
        digest.update(os.path.basename(path).encode())
        with open(path, 'rb') as f:
            digest.update(f.read())
    state = np.random.get_state()
    digest.update(np.asarray(state[1], dtype=np.uint32).tobytes())
    digest.update(str(state[2]).encode())
    return digest.hexdigest()[:12], special


env_cfg, _train_cfg = task_registry.get_cfgs('go2_teacher')
cfg = env_cfg.terrain
print("=" * 70)
h, special = digest_for(cfg)
print(f"digest(seed=1) = {h}")
print(f"num_rows x num_cols = {cfg.num_rows} x {cfg.num_cols}  hs={cfg.horizontal_scale}")
print(f"num_goals={cfg.num_goals} terrain_dict_sum={sum(cfg.terrain_dict.values())}")
print(f"flyball_single_hurdle_height_range={getattr(cfg, 'flyball_single_hurdle_height_range', None)}")
print(f"hurdle_height_range={getattr(cfg, 'hurdle_height_range', None)}")
