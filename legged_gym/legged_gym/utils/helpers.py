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

import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil
import argparse
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    sim_params = gymapi.SimParams()

    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def latest_model_file(root, model_name_include="model"):
    """Newest `model_<iter>.pt` in `root`, picked by iteration number.

    Only `.pt` files whose name *starts* with `model_name_include` count: play.py
    writes its mp4 recordings next to the checkpoints, and an exptid such as
    'D22-newdepthmodel-4090' contains the substring "model" — the old
    substring-match + zero-padded string sort handed back the video.
    """
    models = [f for f in os.listdir(root)
              if f.startswith(model_name_include) and f.endswith(".pt")]
    if not models:
        raise FileNotFoundError(
            "no {}*.pt checkpoint in {}".format(model_name_include, root))

    def _iteration(m):
        stem = m[:-len(".pt")].split("_")[-1]
        return int(stem) if stem.isdigit() else -1

    models.sort(key=lambda m: (_iteration(m), m))
    return models[-1]

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if not os.path.isdir(root):  # use first 4 chars to mactch the run name
        model_name_cand = os.path.basename(root)
        model_parent = os.path.dirname(root)
        model_names = os.listdir(model_parent)
        model_names = [name for name in model_names if os.path.isdir(os.path.join(model_parent, name))]
        for name in model_names:
            if len(name) >= 6:
                if name[:6] == model_name_cand:
                    root = os.path.join(model_parent, name)
    if checkpoint==-1:
        model = latest_model_file(root, model_name_include)
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(root, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    if env_cfg is not None:
        if args.use_camera:
            env_cfg.depth.use_camera = args.use_camera
        if getattr(args, "depth_hole_fill", False):
            env_cfg.depth.hole_fill = True
        if getattr(args, "raw_depth", False):
            # Raw policy input: keep only crop/resize/normalization.
            env_cfg.depth.dis_noise = 0.0
            env_cfg.depth.noise_gaussian = 0.0
            env_cfg.depth.noise_dropout = 0.0
            env_cfg.depth.pw_noise = False
            env_cfg.depth.stereo_shadow = False
            env_cfg.depth.self_geometry = False
            env_cfg.depth.empty_below_m = 0.0
            env_cfg.depth.hole_fill = False
            env_cfg.depth.noise_scale_per_env = False
        if env_cfg.depth.use_camera and args.headless:
            env_cfg.env.num_envs = env_cfg.depth.camera_num_envs
            if getattr(env_cfg.depth, 'override_terrain_grid_for_camera', True):
                env_cfg.terrain.num_rows = env_cfg.depth.camera_terrain_num_rows
                env_cfg.terrain.num_cols = env_cfg.depth.camera_terrain_num_cols
                env_cfg.terrain.max_error = env_cfg.terrain.max_error_camera
                env_cfg.terrain.horizontal_scale = env_cfg.terrain.horizontal_scale_camera
                env_cfg.terrain.simplify_grid = True
            # Only rewrite the terrain mix when the config opts in: a flat-corridor
            # distillation config sets override_terrain_mix_for_camera=False so the
            # student distills on the SAME 100% parkour_flat corridor as the teacher.
            # Otherwise it faceplants on unseen obstacles (88% base-contact, ~1s
            # episodes) and the curriculum collapses to level ~0.
            if getattr(env_cfg.depth, 'override_terrain_mix_for_camera', True):
                env_cfg.terrain.terrain_dict["parkour_hurdle"] = 0.2
                env_cfg.terrain.terrain_dict["parkour_flat"] = 0.05
                env_cfg.terrain.terrain_dict["parkour_gap"] = 0.2
                env_cfg.terrain.terrain_dict["parkour_step"] = 0.2
                env_cfg.terrain.terrain_dict["demo"] = 0.15
                env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
        if env_cfg.depth.use_camera:
            if getattr(env_cfg.depth, 'override_terrain_grid_for_camera', True):
                env_cfg.terrain.y_range = [-0.1, 0.1]

        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        if args.seed is not None:
            env_cfg.seed = args.seed
        if args.task_both:
            env_cfg.env.task_both = args.task_both
        if args.rows is not None:
            env_cfg.terrain.num_rows = args.rows
        if args.cols is not None:
            env_cfg.terrain.num_cols = args.cols
        if getattr(args, "cold_start", False):
            env_cfg.terrain.max_init_terrain_level = 0
            env_cfg.terrain.min_terrain_level = 0
            print("[train] --cold_start: initializing every curriculum env at terrain row 0")
        if args.delay:
            env_cfg.domain_rand.action_delay = True
            # On latch-mode configs --delay engages the fixed 1-step (20 ms) delay
            # from step 0 instead of waiting for the fleet-mean-level latch: the
            # student must never see undelayed actions deploy cannot produce.
            # Latch mode is EITHER trigger — testing only mean_level silently
            # no-op'd --delay on mean_level=None + action_delay_latch_step configs.
            if (getattr(env_cfg.domain_rand, "action_delay_latch_mean_level", None) is not None
                    or getattr(env_cfg.domain_rand, "action_delay_latch_step", None) is not None):
                env_cfg.domain_rand.action_delay_start_latched = True
                print("[train] --delay: action delay latched from step 0 "
                      "(fixed 1 step = 20 ms)")
        if getattr(args, "prearm_term_clearance", False):
            env_cfg.rewards.wococo_armed_term_exempt_start_armed = True
            env_cfg.rewards.hurdle_clearance_start_armed = True
            print("[run] --prearm_term_clearance: termination-tolerance and "
                  "hurdle-clearance schedules triggered at step 0")
        # Clearance-only variant: the arming latch re-derives from common_step_counter,
        # which restarts at 0, so a --resume silently drops the penalty until the arm
        # step comes round again (iteration 2500 on the teachers this was traced on).
        if getattr(args, "prearm_clearance", False):
            env_cfg.rewards.hurdle_clearance_start_armed = True
            print("[run] --prearm_clearance: hurdle-clearance schedule "
                  "triggered at step 0 (arm_step/mean-level trigger bypassed)")
        # "Train from scratch" heuristic: ramp the delay in over global steps.
        # Runs AFTER play.py resolved its delay settings so it must not clobber them:
        # skip on --nodelay and in latch mode (the latch supersedes action_curr_step).
        if (not args.delay and not args.resume and not args.use_camera and args.headless
                and not getattr(args, 'nodelay', False)
                and getattr(env_cfg.domain_rand, "action_delay_latch_mean_level", None) is None):
            env_cfg.domain_rand.action_delay = True
            env_cfg.domain_rand.action_curr_step = env_cfg.domain_rand.action_curr_step_scratch
        # Resume-aware DR-ramp arming: the ramp holds dr_scale at 0 until the fleet
        # first demotes below dr_ramp_arm_below_mean_level, which a warm-started
        # --resume policy never does, so pre-arm it to ramp from iteration 0.
        if args.resume and getattr(env_cfg.domain_rand, "dr_ramp_use_mean_level", False):
            if not getattr(env_cfg.domain_rand, "dr_ramp_start_armed", False):
                env_cfg.domain_rand.dr_ramp_start_armed = True
                print("[train] --resume: pre-arming DR ramp (dr_ramp_start_armed=True) "
                      "so DR ramps from iter 0 instead of staying pinned at 0")
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        if args.use_camera:
            cfg_train.depth_encoder.if_depth = args.use_camera
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if getattr(args, "lr_floor", None) is not None:
            cfg_train.algorithm.min_learning_rate = args.lr_floor
            print(f"[train] --lr_floor: adaptive-LR floor {args.lr_floor:.2e} "
                  f"(default 1.00e-05). The KL schedule saturates at this value on warm "
                  f"resumes, so this is effectively the post-saturation learning rate.")
            if args.lr_floor > cfg_train.algorithm.learning_rate:
                cfg_train.algorithm.learning_rate = args.lr_floor
                print(f"[train] --lr_floor above algorithm.learning_rate — raising the "
                      f"initial lr to the floor as well.")
        if args.resume:
            cfg_train.runner.resume = args.resume
            cfg_train.algorithm.priv_reg_coef_schedual = cfg_train.algorithm.priv_reg_coef_schedual_resume
        # The CLI flag is an opt-out so the default recipe (encoder L2 ON) is kept.
        cfg_train.depth_encoder.use_depth_encoder_loss = not bool(getattr(args, "no_depth_encoder_loss", False))
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "go2_teacher", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int, "default": -1, "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--cold_start", "action": "store_true", "default": False, "help": "Start every curriculum environment at terrain row 0."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--lr_floor", "type": float, "default": None, "help": "Lower clamp of the KL-adaptive learning rate (algorithm.min_learning_rate, default 1e-5). The per-minibatch KL schedule saturates at this floor on warm resumes, so this is effectively the post-saturation lr."},
        {"name": "--device", "type": str, "default": "cuda:0", "help": 'Device for sim, rl, and graphics'},

        {"name": "--rows", "type": int, "help": "num_rows."},
        {"name": "--cols", "type": int, "help": "num_cols"},
        {"name": "--debug", "action": "store_true", "default": False, "help": "Disable wandb logging"},
        {"name": "--proj_name", "type": str,  "default": "extreme-parkour", "help": "run folder name."},
        
        {"name": "--teacher", "type": str, "help": "Name of the teacher policy to use when distilling"},
        {"name": "--exptid", "type": str, "help": "exptid"},
        {"name": "--resumeid", "type": str, "help": "exptid"},
        {"name": "--daggerid", "type": str, "help": "name of dagger run"},
        {"name": "--use_camera", "action": "store_true", "default": False, "help": "render camera for distillation"},
        {"name": "--depth_hole_fill", "action": "store_true", "default": False,
         "help": "apply deployment-compatible RealSense hole fill as the final depth preprocessing step"},
        {"name": "--no_depth_hole_fill", "action": "store_true", "default": False,
         "help": "disable only deployment-compatible depth hole filling; keep other camera/PW effects"},
        {"name": "--save_depth_stages", "action": "store_true", "default": False,
         "help": "save raw/noisy/filled/final depth stages once per second during play"},
        {"name": "--depth_stage_fps", "type": float, "default": 1.0,
         "help": "depth-stage capture rate in Hz (used with --save_depth_stages)"},
        {"name": "--depth_stage_start_seconds", "type": float, "default": 0.0,
         "help": "simulation time at which depth-stage capture starts"},
        {"name": "--depth_stage_seconds", "type": float, "default": None,
         "help": "optional duration of depth-stage capture"},
        {"name": "--raw_depth", "action": "store_true", "default": False,
         "help": "feed raw cropped/resized Warp depth to the policy; disable all depth noise, stereo/self geometry, min-range remap, and hole fill"},
        # Self-geometry is a training-time property that env_cfg.pkl cannot record
        # (cfg.depth pickles by class reference), so eval needs these overrides.
        {"name": "--self_geom", "action": "store_true", "default": False, "help": "force depth self-geometry ON (robot's own body rendered)"},
        {"name": "--no_self_geom", "action": "store_true", "default": False, "help": "force depth self-geometry OFF (for students distilled before it existed)"},

        {"name": "--use_jit", "action": "store_true", "default": False, "help": "Load jit script when playing"},
        {"name": "--save", "action": "store_true", "default": False, "help": "save data for evaluation"},

        {"name": "--task_both", "action": "store_true", "default": False, "help": "Both climbing and hitting policies"},
        {"name": "--nodelay", "action": "store_true", "default": False, "help": "Remove the action delay: zeroes the viewer delay on scheduled-delay tasks, and on latch-mode tasks (cliffside) disables the fixed 1-step deploy delay entirely (NOT deploy-faithful)"},
        {"name": "--full-dr", "dest": "full_dr", "action": "store_true", "default": False, "help": "play.py: use EXACT training domain randomization incl. per-env action latency under the viewer (default play is already training-faithful; this also un-flattens viewer latency). No-op for latency on latch-mode tasks, which use a fixed delay instead of latency DR"},
        {"name": "--no-dr", "dest": "no_dr", "action": "store_true", "default": False, "help": "play.py + train.py: disable ALL domain randomization (friction/mass/CoM/motor/PD/latency/pushes/restitution) + observation noise for clean deterministic dynamics. The fixed deploy action delay is kept (it is an enforced constant, not DR) — use --nodelay to drop it too"},
        {"name": "--no_depth_noise", "dest": "no_depth_noise", "action": "store_true", "default": False, "help": "play.py: disable EVERY depth-camera degradation (dis_noise, gaussian/dropout, parkour-in-the-wild block, stereo shadow, per-env noise curriculum) so the rendered depth is the clean geometric ray-cast. Sensor-physics constants (empty_below_m, far_clip) are KEPT — they are not noise."},
        {"name": "--depth_noise_scale", "dest": "depth_noise_scale", "type": float, "default": None, "help": "play.py: pin EVERY env to this depth-noise level instead of the training curriculum (which spreads envs over linspace([noise_scale_min,1])). Every degradation is LINEAR in s: 0=raw ideal ray-cast (no noise, no stereo shadow, no own legs, no min-range remap, no camera lag), 0.5=exactly the configured reality-calibrated rates, 1.0=2x those stochastic rates for robustness headroom (geometric terms saturate at 0.5). Prints the applied rates plus a realized invalid/big-jump/jitter readout every 100 frames. Requires --use_camera; conflicts with --no_depth_noise."},
        {"name": "--full_depth_noise", "dest": "full_depth_noise", "action": "store_true", "default": False, "help": "play.py: shorthand for --depth_noise_scale 1.0 — every env at the TOP of the cliffside-distill noise curriculum."},
        {"name": "--show_cam", "dest": "show_cam", "action": "store_true", "default": False, "help": "play.py: put a solid black D435i housing (90x25x25 mm) on the robot at cfg.depth.position with the mount pitch, so the camera placement can be checked. Injected as a visual-only, zero-mass fixed link into a regenerated URDF copy (<stem>_show_cam.urdf) — dynamics, contacts and depth self-geometry are unchanged. Renders in the Isaac viewer and in --web/--video; does not need --use_camera."},
        {"name": "--cam_pitch_deg", "dest": "cam_pitch_deg", "type": float, "default": None, "help": "play.py: pin the depth camera pitch (deg down) instead of play.py's hard-coded depth.angle=[0,1]. Training uses [15,25] (nominal 20). Use 20 to render what the policy actually trained on."},
        {"name": "--delay", "action": "store_true", "default": False, "help": "Add action delay. On latch-mode configs (action_delay_latch_mean_level) engages the fixed 1-step (20 ms) delay from step 0; REQUIRED for distillation. play.py latches automatically, so this flag is train-side"},
        {"name": "--hitid", "type": str, "default": None, "help": "exptid fot hitting policy"},

        {"name": "--web", "action": "store_true", "default": False, "help": "if use web viewer"},
        {"name": "--speed", "type": float, "default": None, "help": "Playback speed multiplier (e.g. 0.25, 0.5). Only active with --web."},
        {"name": "--plot", "action": "store_true", "default": False, "help": "Show live foot-force plot in web viewer."},
        {"name": "--data", "action": "store_true", "default": False, "help": "Show cmd_vel_x and current level as text overlay in web viewer."},
        {"name": "--rew", "action": "store_true", "default": False, "help": "Show reward dashboard in web viewer and print per-step breakdown to stdout."},
        {"name": "--video", "action": "store_true", "default": False, "help": "Record a follow-camera video at simulation speed (mp4)."},
        {"name": "--video_path", "type": str, "default": None, "help": "Output path for --video (default: <log_dir>/<exptid>.mp4)."},
        {"name": "--no_video", "action": "store_true", "default": False, "help": "play.py: suppress the mp4 recording that play defaults to when no video flag is given. With no --web this makes play a pure headless analysis run."},
        {"name": "--max_steps", "type": int, "default": None, "help": "Stop play loop after this many steps (default: 10 * max_episode_length)."},
        {"name": "--prearm_term_clearance", "action": "store_true", "default": False,
         "help": "Pre-trigger the WoCoCo armed-window termination-tolerance AND hurdle-clearance schedules at process step 0. Their configured delays and ramps still apply; play.py independently keeps hurdle clearance at full strength. Use --prearm_clearance for the clearance half alone."},
        {"name": "--prearm_clearance", "action": "store_true", "default": False,
         "help": "Pre-trigger ONLY the hurdle-clearance schedule at process step 0 (hurdle_clearance_start_armed), leaving the termination-tolerance window alone. Needed on --resume: the arming latch is runtime-only and re-derives from common_step_counter, which restarts at 0, so a resumed teacher would otherwise train with NO clearance penalty until its arm step despite the checkpoint having had it on. Its configured delay/ramp still apply."},
        {"name": "--no_wandb", "action": "store_true", "default": False, "help": "no wandb"},
        {"name": "--hurdle_width", "type": str, "default": None,
         "help": "play.py: set the flyball hurdle lateral width explicitly instead of taking whatever the live "
                 "terrain config happens to hold. A number pins every hurdle to that width in metres; 'lo:hi' "
                 "restores per-tile uniform sampling over that span; 'min'/'max' pin to the config's own "
                 "flyball_single_hurdle_width_range bounds. Needed for A/B evals because width is the one "
                 "flyball jitter that is difficulty-independent, so a per-level table silently mixes wide and "
                 "narrow tiles — and because env_cfg.pkl does NOT pin the flyball geometry (it restores only a "
                 "handful of instance attrs; everything else resolves from the live config class), so an eval "
                 "that relies on the file is only reproducible until someone edits it."},
        {"name": "--hurdle_height", "type": float, "default": None,
         "help": "pin every flyball hurdle height in metres"},
        {"name": "--hurdle_front_from_spawn", "type": float, "default": None,
         "help": "pin the first hurdle front distance from the nominal spawn in metres"},
        {"name": "--no_hurdle_roughness", "action": "store_true", "default": False,
         "help": "disable roughness on the top surface of flyball hurdles"},
        {"name": "--all_rough", "action": "store_true", "default": False,
         "help": "play.py: set flyball_roughness_zero_prob=0 so every tile gets the rough curriculum instead of the trained 50/50 smooth/rough split"},
        {"name": "--spawn_after_second_hurdle", "type": float, "default": None,
         "help": "spawn every environment this many metres after the second hurdle"},

        {"name": "--wall_contact_scale", "type": float, "default": None,
         "help": "Override wall_contact reward scale"},
        {"name": "--wall_approach_scale", "type": float, "default": None,
         "help": "Override wall_approach_vel reward scale"},
        {"name": "--force_norm", "type": float, "default": None,
         "help": "Override wall_force_norm (tanh normalization, default 25N)"},
        {"name": "--yaw_sentinel_hold", "action": "store_true", "default": False,
         "help": "play.py: force the WoCoCo post-sentinel yaw-tracking max-pin override ON so it shows in --rew. "
                 "Drops the curriculum min-mean-level gate to 0 (a single play env rarely reaches the fleet-mean "
                 "level the training gate needs) and enables the hold window if the saved config has it at 0. "
                 "Reward-only — does NOT change the policy's actions (play is pure inference)."},

        {"name": "--save_vts", "action": "store_true", "default": False,
         "help": "Export terrain heightfield, goals, and AABB regions as VTS/VTP/VTU files"},
        {"name": "--save_vts_full_res", "action": "store_true", "default": False,
         "help": "With --save_vts, export the terrain heightfield at its native horizontal resolution (may create a very large VTS file)."},
        {"name": "--max_level", "action": "store_true", "default": False,
         "help": "Lock all environments to the hardest terrain level (num_rows-1), disabling curriculum."},
        {"name": "--level", "type": int, "default": None,
         "help": "Lock all environments to this exact terrain level (disables curriculum). Overrides --max_level."},
        {"name": "--eval_min_level", "type": int, "default": 0,
         "help": "play.py headless analysis mode only: restrict the pinned-level round-robin to rows [eval_min_level, num_rows-1]. Use to score only the band the curriculum actually trains (e.g. 4 when terrain.resample_min_level=4). Unlike --level it keeps the deterministic per-row difficulty."},
        {"name": "--eval_max_level", "type": int, "default": None,
         "help": "play.py headless analysis mode only: upper bound (inclusive) for the pinned-level round-robin. Default None = num_rows-1."},
        {"name": "--action_noise", "type": float, "default": 0.0,
         "help": "play.py: std of Gaussian noise added to the EXECUTED action each step, matching the "
                 "noise-hardened DAgger injection (depth_encoder.action_noise_std). Use to test whether a "
                 "noise-hardened student is actually more robust: hardening COSTS 2.9pp on clean dynamics "
                 "(0.626 vs 0.655), so if it works the ranking should invert as this is raised toward 0.15."},
        {"name": "--terrain_height", "type": str, "default": None,
         "help": "play.py: override terrain.height, the add_roughness amplitude range [lo,hi] in metres, instead of "
                 "play.py's hard-coded [0.02,0.02]. A single number pins both bounds ('--terrain_height 0' = a "
                 "perfectly smooth heightfield, zero roughness); 'lo:hi' restores difficulty-scaled sampling. This is "
                 "the ONLY roughness source for the flat terrains, so it is how you get a truly flat floor for "
                 "quasi-static measurements (foot loads, base height) that ±2 cm of noise would smear."},
        {"name": "--zero_terrain_roughness", "action": "store_true", "default": False,
         "help": "play.py: force terrain.height=[0,0], disabling terrain add_roughness"},
        {"name": "--dump_actor", "action": "store_true", "default": False,
         "help": "Dump actor state_dict and first 5 (obs, action) pairs to /tmp for sim2sim comparison."},
        {"name": "--terrain_only", "type": str, "default": None,
         "help": "Force a single terrain type at play time (e.g. parkour_flat). Zeros all other terrain_proportions."},
        {"name": "--cmd_vel_x", "type": float, "default": None,
         "help": "Lock commanded forward velocity (m/s) at play time. Sets command_ranges.lin_vel_x to [v,v] and zeroes yaw/heading."},
        {"name": "--overlays", "action": "store_true", "default": False,
         "help": "play.py: draw the diagnostic overlays on rendered frames — wall-plant foothold goals and parabolic hurdle-clearance volumes. Off by default, so --video/--web produce a clean scene."},
        {"name": "--top_view", "action": "store_true", "default": False,
         "help": "Overhead camera: positions the video camera directly above the robot looking straight down."},
        {"name": "--vel_profile", "type": str, "default": None,
         "help": "Time-varying forward velocity profile as 'vel:secs' pairs, e.g. '0:3,0.5:5,0:1,0.8:5,0:5'. "
                 "Overrides --cmd_vel_x. Zeroes yaw/heading like --cmd_vel_x."},

        {"name": "--no_depth_encoder_loss", "action": "store_true", "default": False,
         "help": "Disable the depth-encoder L2 (scandots_latent vs depth_latent) during distillation. Default ON. With this flag the runner trains depth_actor + yaw only — depth_encoder still updates indirectly through the actor chain-rule gradient (same optimizer)."},



    ]
    args = parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

def export_policy_as_jit(actor_critic, path, name):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, name+".pt")
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

    
# overide gymutil
def parse_device_str(device_str):
    device = 'cpu'
    device_id = 0

    if device_str == 'cpu' or device_str == 'cuda':
        device = device_str
        device_id = 0
    else:
        device_args = device_str.split(':')
        assert len(device_args) == 2 and device_args[0] == 'cuda', f'Invalid device string "{device_str}"'
        device, device_id_s = device_args
        try:
            device_id = int(device_id_s)
        except ValueError:
            raise ValueError(f'Invalid device string "{device_str}". Cannot parse "{device_id}"" as a valid device id')
    return device, device_id

def parse_arguments(description="Isaac Gym Example", headless=False, no_graphics=False, custom_parameters=[]):
    parser = argparse.ArgumentParser(description=description)
    if headless:
        parser.add_argument('--headless', action='store_true', help='Run headless without creating a viewer window')
    if no_graphics:
        parser.add_argument('--nographics', action='store_true',
                            help='Disable graphics context creation, no viewer window is created, and no headless rendering is available')
    parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
    parser.add_argument('--pipeline', type=str, default="gpu", help='Tensor API pipeline (cpu/gpu)')
    parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')

    physics_group = parser.add_mutually_exclusive_group()
    physics_group.add_argument('--flex', action='store_true', help='Use FleX for physics')
    physics_group.add_argument('--physx', action='store_true', help='Use PhysX for physics')

    parser.add_argument('--num_threads', type=int, default=0, help='Number of cores used by PhysX')
    parser.add_argument('--subscenes', type=int, default=0, help='Number of PhysX subscenes to simulate in parallel')
    parser.add_argument('--slices', type=int, help='Number of client threads that process env slices')

    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(argument["name"], type=argument["type"], default=argument["default"], help=help_str)
                else:
                    parser.add_argument(argument["name"], type=argument["type"], help=help_str)
            elif "action" in argument:
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)

        else:
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            print("supported keys: name, type, default, action, help")
            print()

    args = parser.parse_args()

    if args.device is not None:
        args.sim_device = args.device
        args.rl_device = args.device
    args.sim_device_type, args.compute_device_id = parse_device_str(args.sim_device)
    pipeline = args.pipeline.lower()

    assert (pipeline == 'cpu' or pipeline in ('gpu', 'cuda')), f"Invalid pipeline '{args.pipeline}'. Should be either cpu or gpu."
    args.use_gpu_pipeline = (pipeline in ('gpu', 'cuda'))

    if args.sim_device_type != 'cuda' and args.flex:
        print("Can't use Flex with CPU. Changing sim device to 'cuda:0'")
        args.sim_device = 'cuda:0'
        args.sim_device_type, args.compute_device_id = parse_device_str(args.sim_device)

    if (args.sim_device_type != 'cuda' and pipeline == 'gpu'):
        print("Can't use GPU pipeline with CPU Physics. Changing pipeline to 'CPU'.")
        args.pipeline = 'CPU'
        args.use_gpu_pipeline = False

    args.physics_engine = gymapi.SIM_PHYSX
    args.use_gpu = (args.sim_device_type == 'cuda')

    if args.flex:
        args.physics_engine = gymapi.SIM_FLEX

    if no_graphics and args.nographics:
        args.headless = True

    if args.slices is None:
        args.slices = args.subscenes

    return args
