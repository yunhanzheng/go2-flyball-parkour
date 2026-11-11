"""Dump the fully-resolved go2_teacher env+PPO config of a checkout to JSON."""
import isaacgym  # noqa: F401  must be imported before torch
import json
import sys

root = sys.argv[1]
sys.path.insert(0, root + "/rsl_rl")
sys.path.insert(0, root + "/legged_gym")

import legged_gym.envs  # noqa: F401  initialises the task registry in the usual order
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.go2.go2_teacher_config import Go2TeacherCfg, Go2TeacherCfgPPO


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, pre + str(k) + "."))
        else:
            out[pre + str(k)] = v
    return out


dump = {}
dump.update({"ENV." + k: v for k, v in flat(class_to_dict(Go2TeacherCfg())).items()})
dump.update({"PPO." + k: v for k, v in flat(class_to_dict(Go2TeacherCfgPPO())).items()})
json.dump(dump, open(sys.argv[2], "w"), indent=1, default=str, sort_keys=True)
print("keys:", len(dump))
