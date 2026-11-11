"""Compare the live Go2FlatCfg against the env_cfg.pkl that produced model_3600."""
import isaacgym  # noqa: F401  must precede torch
import pickle
import sys

sys.path.insert(0, "/home/ubuntu22/data/go2-flyball-parkour/rsl_rl")
sys.path.insert(0, "/home/ubuntu22/data/go2-flyball-parkour/legged_gym")

import legged_gym.envs  # noqa: F401  registers tasks
from legged_gym.envs.go2.go2_flat_config import Go2FlatCfg
from legged_gym.utils.helpers import class_to_dict


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, pre + str(k) + "."))
        else:
            out[pre + str(k)] = v
    return out


live = flat(class_to_dict(Go2FlatCfg()))
old = pickle.load(open(sys.argv[1], "rb"))
# The checkpoint stores either a config object or its already-flattened dict.
old_flat = flat(old) if isinstance(old, dict) else flat(class_to_dict(old))

keys = sorted(set(live) | set(old_flat))
diffs = [(k, old_flat.get(k, "<absent>"), live.get(k, "<absent>"))
         for k in keys if str(old_flat.get(k, "<absent>")) != str(live.get(k, "<absent>"))]
print(f"总键 {len(keys)}  差异 {len(diffs)}")
for k, o, n in diffs:
    print(f"[{k}]\n   ckpt3600 : {str(o)[:180]}\n   now      : {str(n)[:180]}")
