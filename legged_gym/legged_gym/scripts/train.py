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

import hashlib
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


# Every optimiser the algorithm can step. A depth-distillation run (learn_vision)
# never touches `optimizer` -- that one only exists for the privileged teacher --
# and the ACTING policy there is `depth_actor`, stepped by `depth_actor_optimizer`
# (which also owns the depth encoder's parameters). Freezing PPO's lr alone would
# therefore leave the student learning.
_OPTIMIZER_ATTRS = (
    'optimizer',
    'hist_encoder_optimizer',
    'estimator_optimizer',
    'depth_encoder_optimizer',
    'depth_actor_optimizer',
    # Depth-RL only (learn_depth_rl): PPO on depth_actor + its action std.
    'depth_rl_optimizer',
    # Depth-RL only: the teacher's objective set applied to the depth chain also
    # trains the privileged critic (value loss) and the estimator, each with its
    # own optimiser, plus the history encoder (the update_dagger() regression,
    # which needs its own optimiser because it is excluded from depth_rl_optimizer).
    'critic_optimizer',
    'depth_hist_encoder_optimizer',
)

# Modules whose state_dict is fingerprinted: the privileged teacher, the estimator
# that feeds the student its deploy-like state, and the depth student itself.
_FINGERPRINT_ATTRS = ('actor_critic', 'estimator', 'depth_encoder', 'depth_actor')


def _optimizers(alg):
    return [(name, getattr(alg, name)) for name in _OPTIMIZER_ATTRS
            if getattr(alg, name, None) is not None]


def freeze_policy(runner, skip=()):
    """Make the run inference-only: nothing any optimiser touches can move.

    Only learning rates are pinned. requires_grad is deliberately KEPT on:
    PPO always calls loss.backward() during update(), and with no graph the
    backward pass raises 'element 0 of tensors does not require grad'. Leaving
    the graph intact costs one backward per minibatch (the same cost as real
    training) but an Adam step with lr=0 cannot change a single value.

    lr must be zeroed AFTER the checkpoint load (optimizer.load_state_dict
    restores the saved param_group lrs) and the adaptive-KL floor has to go too,
    otherwise ppo.py re-raises lr to min_learning_rate on a KL spike.

    `skip` names optimisers that must keep learning. --depth_rl passes
    'depth_rl_optimizer': there the whole point of the run is the PPO update of
    depth_actor, and freezing it would silently degrade to an evaluation.
    """
    alg = runner.alg
    for name, opt in _optimizers(alg):
        if name in skip:
            continue
        for group in opt.param_groups:
            group['lr'] = 0.0
    # PPO's adaptive-KL schedule reads/writes this trio every minibatch. The
    # depth-RL schedule owns a separate trio (depth_rl_learning_rate/...), so
    # zeroing these does not stop the depth student when it is skipped above.
    for attr in ('learning_rate', 'min_learning_rate', 'max_learning_rate'):
        if getattr(alg, attr, None) is not None:
            setattr(alg, attr, 0.0)
    lrs = {}
    for name, opt in _optimizers(alg):
        values = sorted({round(float(g['lr']), 12) for g in opt.param_groups})
        lrs[name] = f"KEPT {values}" if name in skip else values
    print(f"[train] --freeze: optimiser_lrs={lrs} "
          f"adaptive_floor={getattr(alg, 'min_learning_rate', None)} "
          f"requires_grad_kept=True (lr=0 makes every step a no-op)")


def policy_fingerprint(runner, attrs=_FINGERPRINT_ATTRS):
    """sha1 over every parameter AND buffer of every policy module."""
    digest = hashlib.sha1()
    for attr in attrs:
        module = getattr(runner.alg, attr, None)
        if module is None:
            continue
        for name, tensor in sorted(module.state_dict().items()):
            digest.update(f"{attr}.{name}".encode())
            digest.update(tensor.detach().to('cpu').contiguous().numpy().tobytes())
    return digest.hexdigest()






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
    if getattr(args, 'min_terrain_level', None) is not None:
        # The curriculum is left LIVE (unlike --level, which locks a row): only the
        # initial draw is floored, so promote/demote keep running. This is the single
        # terrain knob go2_student takes from the distillation flow (min_terrain_level=4).
        env_cfg.terrain.min_terrain_level = int(args.min_terrain_level)
        print(f"[train] --min_terrain_level: initial levels drawn from "
              f"U{{{env_cfg.terrain.min_terrain_level}.."
              f"{max(env_cfg.terrain.min_terrain_level, env_cfg.terrain.max_init_terrain_level)}}} "
              f"(max_init_terrain_level={env_cfg.terrain.max_init_terrain_level}), curriculum live")
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
    fp_before = None
    fp_attrs = _FINGERPRINT_ATTRS
    depth_rl = bool(getattr(ppo_runner.alg, 'depth_rl_mode', False))
    if getattr(args, 'freeze', False):
        # On the depth-RL path the frozen set is ONLY the teacher actor: depth_actor
        # (surrogate), the depth ENCODER (joint: replayed inside the PPO graph), the
        # privileged critic (value loss), the estimator (true-state regression) and
        # the history encoder (update_dagger regression) must all keep learning.
        skip = ('depth_rl_optimizer', 'critic_optimizer', 'estimator_optimizer',
                'depth_hist_encoder_optimizer') if depth_rl else ()
        if depth_rl:
            # No module is expected to stay bit-identical any more: the encoder now
            # trains jointly and actor_critic holds BOTH the frozen teacher actor and
            # the trainable critic, which a module-level fingerprint cannot separate.
            # Skip the check and rely on the optimiser-lr table above instead.
            fp_attrs = ()
        freeze_policy(ppo_runner, skip=skip)
        if fp_attrs:
            fp_before = policy_fingerprint(ppo_runner, fp_attrs)
            print(f"[train] --freeze: fp_before={fp_before} scope={fp_attrs}")
        elif depth_rl:
            print("[train] --freeze: no weight fingerprint on the depth-RL path "
                  "(teacher actor and critic share the actor_critic module); the "
                  "optimiser-lr table above is the check: depth_rl/critic/estimator/"
                  "hist_encoder KEPT, the rest pinned at 0.0.")
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    if fp_before is not None:
        fp_after = policy_fingerprint(ppo_runner, fp_attrs)
        print(f"[train] --freeze: fp_after ={fp_after}")
        frozen = "IDENTICAL (no weight moved)" if fp_after == fp_before else "CHANGED -- NOT FROZEN"
        print(f"[train] --freeze VERDICT ({'depth_encoder only' if depth_rl else 'all modules'}): "
              + frozen)
        if depth_rl:
            print("[train] --depth_rl: depth_actor (surrogate), depth_encoder (joint, "
                  "replayed in-graph), critic (value loss), estimator (true-state "
                  "regression) and history_encoder (update_dagger regression) are "
                  "trained on purpose; only the teacher actor is frozen.")




if __name__ == '__main__':
    args = get_args()
    train(args)
