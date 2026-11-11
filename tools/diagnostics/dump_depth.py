"""Dump the resolved depth.* block of one task from one checkout to JSON."""
import isaacgym  # noqa: F401  must be imported before torch
import importlib
import json
import sys

root, task_cls_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, root + "/rsl_rl")
sys.path.insert(0, root + "/legged_gym")

import legged_gym.envs  # noqa: F401  registers tasks
from legged_gym.utils.helpers import class_to_dict

mod_name, cls_name = task_cls_path.rsplit(".", 1)
cls = getattr(importlib.import_module(mod_name), cls_name)

d = class_to_dict(cls().depth)
json.dump({k: str(v) for k, v in sorted(d.items())}, open(out, "w"), indent=1)
print("depth keys:", len(d))
