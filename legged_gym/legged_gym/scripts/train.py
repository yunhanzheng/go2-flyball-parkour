# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
import pickle
import subprocess
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import wandb






def dump_env_cfg(env_cfg, log_pth):
    """Pickle the fully-resolved env_cfg (after CLI + reward overrides) to
    env_cfg.pkl so play.py reloads the EXACT training settings instead of the
    live (possibly drifted) repo config. Best-effort: warn but never crash the
    run if the cfg isn't picklable."""
    path = os.path.join(log_pth, 'env_cfg.pkl')
    try:
        with open(path, 'wb') as f:
            pickle.dump(env_cfg, f)
        return path
    except Exception as e:
        print(f"[train] WARNING: could not pickle env_cfg ({e}); "
              f"play.py will fall back to the live repo config")
        return None




def _snapshot_repo(repo_root, log_pth, prefix=""):
    """Write HEAD/status/diff for one git repo into log_pth, optionally prefixed.

    Returns the list of basenames written, or [] if repo_root is not a git repo.
    """
    def _git(*cmd):
        return subprocess.run(
            ["git", "-C", repo_root, *cmd],
            check=False, capture_output=True, text=True,
        )

    toplevel = _git("rev-parse", "--show-toplevel")
    if toplevel.returncode != 0:
        print(f"[git_state] {repo_root} is not a git repo — skipping snapshot")
        return []

    head = _git("rev-parse", "HEAD").stdout.strip()
    branch = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    status = _git("status", "--porcelain=v1", "--untracked-files=all").stdout
    diff_tracked = _git("diff", "HEAD").stdout

    untracked = [
        line[3:] for line in status.splitlines()
        if line.startswith("?? ")
    ]
    diff_untracked = ""
    if untracked:
        # /dev/null vs file shows the full content of new files as additions in the patch.
        res = _git("diff", "--no-index", "--binary", "--", "/dev/null", *untracked)
        diff_untracked = res.stdout

    names = (f"{prefix}git_head.txt", f"{prefix}git_status.txt", f"{prefix}git_diff.patch")
    with open(os.path.join(log_pth, names[0]), "w") as f:
        f.write(f"branch: {branch}\ncommit: {head}\n")
    with open(os.path.join(log_pth, names[1]), "w") as f:
        f.write(status)
    with open(os.path.join(log_pth, names[2]), "w") as f:
        f.write(diff_tracked)
        if diff_untracked:
            f.write("\n# --- untracked files ---\n")
            f.write(diff_untracked)
    return list(names)


def save_git_state(log_pth):
    """Record HEAD commit, status, and diff vs last commit so the run is reproducible.

    Replaces the old full source snapshot — the repo + this patch are enough to reconstruct
    the exact code state, without copying tens of MB of unchanged files.

    Snapshots the legged_gym repo and, when rsl_rl is imported from a *different* git
    repo (e.g. pip install -e'd from a separate checkout), the rsl_rl repo too under an
    "rsl_" prefix. When rsl_rl lives inside the same repo (the default layout) its changes
    are already captured by the legged_gym diff, so the extra snapshot is skipped.

    Returns the list of basenames written, for downstream wandb.save().
    """
    repo_root = os.path.abspath(os.path.join(LEGGED_GYM_ROOT_DIR, os.pardir))
    written = _snapshot_repo(repo_root, log_pth)

    def _toplevel(path):
        res = subprocess.run(
            ["git", "-C", path, "rev-parse", "--show-toplevel"],
            check=False, capture_output=True, text=True,
        )
        return os.path.realpath(res.stdout.strip()) if res.returncode == 0 else None

    try:
        import rsl_rl
        rsl_dir = os.path.dirname(os.path.realpath(rsl_rl.__file__))
    except Exception as e:
        print(f"[git_state] could not locate rsl_rl for snapshot: {e}")
        return written

    rsl_root = _toplevel(rsl_dir)
    if rsl_root is not None and rsl_root != _toplevel(repo_root):
        written += _snapshot_repo(rsl_root, log_pth, prefix="rsl_")
    return written


def train(args):
    args.headless = True
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid
    try:
        os.makedirs(log_pth)
    except:
        pass
    git_state_files = save_git_state(log_pth)
    if args.debug:
        mode = "disabled"
        args.rows = 10
        args.cols = 8
        args.num_envs = 64
    else:
        mode = "online"

    if args.no_wandb:
        mode = "disabled"
    wandb.init(project=args.proj_name, name=args.exptid, entity=None, group=args.exptid[:3], mode=mode, dir="../../logs")
    wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot_config.py", policy="now")
    wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot.py", policy="now")
    for fname in git_state_files:
        fpath = os.path.join(log_pth, fname)
        if os.path.exists(fpath):
            wandb.save(fpath, policy="now")

    env_cfg, _ = task_registry.get_cfgs(args.task)
    # Distillation guard: a student must NEVER train on undelayed actions — deploy
    # runs the enforced 16 ms proprio pad (~= the 1-step delay), so require --delay
    # to latch it from step 0 (helpers.py sets action_delay_start_latched).
    if (args.use_camera and not args.delay
            and getattr(env_cfg.domain_rand, "action_delay_latch_mean_level", None) is not None):
        raise SystemExit(
            "[train] ERROR: distillation (--use_camera) on a config with "
            "action_delay_latch_mean_level requires --delay, so the student "
            "trains on the 1-step (20 ms) action delay from the first batch "
            "(matches the 16 ms proprio pad enforced on hardware).")
    env_cfg.terrain.log_dir = log_pth
    env_cfg.terrain.save_vts = args.save_vts
    if args.save_vts_full_res:
        env_cfg.terrain.vts_export_cell_size = env_cfg.terrain.horizontal_scale
    if args.terrain_only is not None:
        # Lock to one terrain type so gradients aren't averaged across mixed
        # terrain proportions during a focused baseline run.
        if args.terrain_only not in env_cfg.terrain.terrain_dict:
            raise ValueError(f"--terrain_only '{args.terrain_only}' not in terrain_dict keys: {list(env_cfg.terrain.terrain_dict.keys())}")
        for k in env_cfg.terrain.terrain_dict:
            env_cfg.terrain.terrain_dict[k] = 0.0
        env_cfg.terrain.terrain_dict[args.terrain_only] = 1.0
        env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
        print(f"[train] --terrain_only: locked to {args.terrain_only}")
    if getattr(args, "no_dr", False):
        # Clean, deterministic dynamics for a teacher run: mirror play.py --no-dr.
        # Every DR term is flag-gated, so clearing the randomize_* bools, pushes,
        # the curriculum-linked DR scale and obs noise fully disables DR. These
        # survive the later update_cfg_from_args (it only arms action_delay/dr-ramp).
        dr = env_cfg.domain_rand
        disabled = []
        for attr in dir(dr):
            if attr.startswith("randomize_") and isinstance(getattr(dr, attr), bool):
                setattr(dr, attr, False)
                disabled.append(attr)
        dr.push_robots = False
        if hasattr(dr, "dr_lock_to_terrain_level"):
            dr.dr_lock_to_terrain_level = False
        env_cfg.noise.add_noise = False
        print(f"[train] --no-dr: disabled DR ({', '.join(disabled)}) + pushes + obs noise")
    if getattr(args, "nodelay", False):
        # Fully lag-free teacher dynamics (NOT deploy-faithful): disable explicitly,
        # else latch-mode configs would engage the delay at action_delay_latch_step.
        env_cfg.domain_rand.action_delay = False
        env_cfg.domain_rand.action_delay_start_latched = False
        print("[train] --nodelay: action delay OFF (0 ms) — NOT deploy-faithful")
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    if args.wall_contact_scale is not None:
        env.cfg.rewards.scales.wall_contact = args.wall_contact_scale
    if args.wall_approach_scale is not None:
        env.cfg.rewards.scales.wall_approach_vel = args.wall_approach_scale
    if args.force_norm is not None:
        env.cfg.rewards.wall_force_norm = args.force_norm

    env_cfg_pkl = dump_env_cfg(env.cfg, log_pth)
    if env_cfg_pkl is not None:
        wandb.save(env_cfg_pkl, policy="now")

    print("Printing configuration...")
    print("TASK:", args.task)
    print("ASSET FILE:", env_cfg.asset.file)
    print("Configuration printed!")

    ppo_runner, train_cfg = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)




if __name__ == '__main__':
    args = get_args()
    train(args)
