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

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import math
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch, torchvision
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import *
from legged_gym.utils.curriculum_ramp import latched_linear_alpha
from legged_gym.utils.helpers import class_to_dict
from scipy.spatial.transform import Rotation as R
from .legged_robot_config import LeggedRobotCfg

from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

def euler_from_quaternion(quat_angle):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = torch.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = torch.clip(t2, -1, 1)
        pitch_y = torch.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = torch.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        self.resize_transform = torchvision.transforms.Resize((self.cfg.depth.resized[1], self.cfg.depth.resized[0]), 
                                                              interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        self.global_counter = 0
        self.total_env_steps_counter = 0
        # Fleet-level action-delay latch (cfg.domain_rand.action_delay_latch_mean_level); sticky once True, --delay pre-engages it.
        self.action_delay_latched = bool(getattr(
            self.cfg.domain_rand, "action_delay_start_latched", False))

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.post_physics_step()

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions = self.reindex(actions)

        actions.to(self.device)
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        if getattr(self.cfg.domain_rand, "randomize_action_latency", False):
            buf_len = self.action_history_buf.shape[1]
            if self.viewer and not getattr(self.cfg.domain_rand, "force_action_latency", False):
                delay_steps = torch.full((self.num_envs,), int(self.cfg.domain_rand.action_delay_view),
                                         device=self.device, dtype=torch.long)
            else:
                delay_steps = self.action_delay_steps
            gather_idx = (buf_len - 1 - delay_steps).clamp_(0, buf_len - 1)
            env_idx = torch.arange(self.num_envs, device=self.device)
            actions = self.action_history_buf[env_idx, gather_idx]  # (num_envs, num_dofs)
        elif self.cfg.domain_rand.action_delay:
            # Latch mode fires on EITHER trigger (mean_level or step), so the step latch actually engages the delay and --delay works for those configs too.
            _latch_mode = (
                getattr(self.cfg.domain_rand, "action_delay_latch_mean_level", None) is not None
                or getattr(self.cfg.domain_rand, "action_delay_latch_step", None) is not None)
            if _latch_mode:
                # Latch mode: 0 steps until the trigger fires (latched in reset_idx), then a fixed 1 step (20 ms) permanently, superseding the action_curr_step schedule.
                self.delay = (self.action_delay_episode_enabled.to(dtype=torch.float)
                              if getattr(self.cfg.domain_rand, "action_delay_ramp_steps", 0) > 0
                              else torch.tensor(1.0 if self.action_delay_latched else 0.0,
                                                device=self.device, dtype=torch.float))
            elif self.global_counter % self.cfg.domain_rand.delay_update_global_steps == 0:
                if len(self.cfg.domain_rand.action_curr_step) != 0:
                    self.delay = torch.tensor(self.cfg.domain_rand.action_curr_step.pop(0), device=self.device, dtype=torch.float)
            if self.viewer and not _latch_mode:
                # In latch mode the delay is already a deploy-enforced constant (0 or 1 step), so the action_delay_view override must not overwrite the latch state.
                self.delay = torch.tensor(self.cfg.domain_rand.action_delay_view, device=self.device, dtype=torch.float)
            indices = (-self.delay - 1).long()
            if indices.ndim == 0:
                actions = self.action_history_buf[:, indices]  # scalar delay
            else:
                env_idx = torch.arange(self.num_envs, device=self.device)
                actions = self.action_history_buf[env_idx, indices]  # per-env delay

        self.global_counter += 1
        self.total_env_steps_counter += 1
        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        self.extras["delta_yaw_ok"] = self.delta_yaw < 0.6
        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            # Latency-aware selection (per-env depth_latency) so the sampled camera latency takes effect on every rollout step.
            self.extras["depth"] = self.get_depth_obs()
        else:
            self.extras["depth"] = None
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_history_observations(self):
        return self.obs_history_buf
    
    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip)  - 0.5
        return depth_image
    
    def _depth_noise_factors(self, n=None, env_id=None):
        """The per-env depth-degradation factors derived from depth_noise_scale s.

        ONE linear knob drives every camera degradation:

          rate = 2s              -> stochastic noise AMPLITUDES and artifact
                                    DENSITIES. s=0 is exactly zero, s=0.5 is the
                                    calibrated real-D435 rate, s=1 is 2x reality
                                    for robustness headroom.
          phys = clamp(2s, 0, 1) -> physical/geometric sensor properties that
                                    cannot meaningfully exceed 100%: stereo
                                    shadow, own-body rendering, min-range remap,
                                    blind-spot width, optical blur and transport
                                    latency. These ramp clean -> real by s=0.5
                                    and then saturate.

        So s=0 is a raw ideal ray-cast (no shadow, no legs, no min-range remap,
        no lag, no noise) and every degradation grows linearly in s from there.

        Returns (rate, phys) as [n,1,1] tensors for the batched path, or plain
        floats when env_id is given. No per-env curriculum -> (1, 1), i.e. the
        configured rates verbatim.
        """
        if not hasattr(self, "depth_noise_scale"):
            if env_id is not None:
                return 1.0, 1.0
            ones = torch.ones(n, 1, 1, device=self.device)
            return ones, ones
        if env_id is not None:
            s = float(self.depth_noise_scale[env_id])
            return 2.0 * s, min(max(2.0 * s, 0.0), 1.0)
        s = self.depth_noise_scale[:n].view(-1, 1, 1)
        return 2.0 * s, torch.clamp(2.0 * s, 0.0, 1.0)

    def _depth_phys_factor_1d(self):
        """`phys` as a [num_envs] tensor for the render-time consumers (Warp
        stereo shadow / self-geometry gate). None when there is no curriculum."""
        if not hasattr(self, "depth_noise_scale"):
            return None
        return torch.clamp(2.0 * self.depth_noise_scale, 0.0, 1.0)

    def _apply_depth_noise(self, depth):
        # MGDP-style depth noise (arclab-hku/MGDP) on raw depth in metres (Isaac negative-in-front); [H, W] or [N, H, W]; no-op when the cfg knobs are 0.
        g = getattr(self.cfg.depth, "noise_gaussian", 0.0)
        p = getattr(self.cfg.depth, "noise_dropout", 0.0)
        # Per-env rate factor scales artifact density/amplitude linearly, so artifacts stay hard at every scale (sparse-but-hard, never dense-but-faded) and vanish at s=0. None -> 1x base rates.
        rf = getattr(self, "_noise_rate_factor", None)
        if rf is None:
            rf = 1.0
        elif torch.is_tensor(rf) and depth.dim() != 3:
            rf = 1.0
        if g > 0.0:
            # Range-proportional Gaussian: per-pixel std = g * |depth|, sign preserved.
            depth = depth + torch.randn_like(depth) * (g * rf * depth)
        if p > 0.0 and getattr(self.cfg.depth, "dropout_cluster", False):
            # Model stereo invalidations as short horizontal scanline runs.
            far_v = -float(self.cfg.depth.far_clip)
            near_v = -float(self.cfg.depth.near_clip)
            ext = float(getattr(self.cfg.depth, "dropout_cluster_ext", 0.6))
            # Only on VALID pixels: the real sensor's invalid region stays uniformly far/white, so jittered specks there would be unrealistic dust.
            valid = depth > (far_v + 0.05)
            seeds = valid & (torch.rand_like(depth) < p * rf)
            add1 = seeds & (torch.rand_like(depth) < ext)      # 2nd px right
            add2 = add1 & (torch.rand_like(depth) < ext)       # 3rd px right
            near_p = float(getattr(self.cfg.depth, "dropout_near_prob", 0.35))
            blk = seeds & (torch.rand_like(depth) < near_p)
            white = seeds.clone()
            white[..., :, 1:] |= add1[..., :, :-1]             # seed + right -> far
            white[..., :, 2:] |= add2[..., :, :-2]             # + right2 -> far
            black = torch.zeros_like(seeds)
            black[..., :, :-1] |= blk[..., :, 1:]              # left of run -> near
            black &= ~white                                    # no overlap
            # Vary each cluster pixel's grayscale instead of hard white/black: white pulled up to dropout_jitter m toward the camera, black pushed the same away.
            jit = float(getattr(self.cfg.depth, "dropout_jitter", 0.0))
            white_fill = far_v + torch.rand_like(depth) * jit
            near_fill = near_v - torch.rand_like(depth) * jit
            depth = torch.where(white, white_fill, depth)
            depth = torch.where(black, near_fill, depth)
        elif p > 0.0:
            drop = torch.rand_like(depth) < p
            far_frac = getattr(self.cfg.depth, "dropout_far_frac", None)
            if far_frac is not None:
                # Split dropped pixels between far (invalid-disparity hole) and near (hot pixel): far_frac go far, the rest near, as real dead pixels saturate both ways.
                far_v = -float(self.cfg.depth.far_clip)
                near_v = -float(self.cfg.depth.near_clip)
                r2 = torch.rand_like(depth)
                depth = torch.where(drop & (r2 < far_frac),
                                    torch.full_like(depth, far_v), depth)
                depth = torch.where(drop & (r2 >= far_frac),
                                    torch.full_like(depth, near_v), depth)
            else:
                if getattr(self.cfg.depth, "dropout_to_far", False):
                    fill = -self.cfg.depth.far_clip   # dropped -> holes (far)
                else:
                    fill = 0.0                         # MGDP default: dropped -> 0 (nearest)
                depth = torch.where(drop, torch.full_like(depth, fill), depth)
        return depth

    def _gaussian_blur_batch(self, depth, sigma):
        # Depthwise Gaussian blur over [N, H, W]. Cached separable->2D kernel.
        r = max(1, int(round(3.0 * sigma)))
        key = (round(float(sigma), 3), r, depth.dtype, depth.device)
        if getattr(self, "_pw_blur_key", None) != key:
            x = torch.arange(-r, r + 1, device=depth.device, dtype=depth.dtype)
            k1 = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
            k1 = k1 / k1.sum()
            self._pw_blur_kernel = (k1[:, None] * k1[None, :]).view(1, 1, 2 * r + 1, 2 * r + 1)
            self._pw_blur_pad = r
            self._pw_blur_key = key
        d = F.conv2d(depth.unsqueeze(1), self._pw_blur_kernel, padding=self._pw_blur_pad)
        return d.squeeze(1)

    def _pw_hole_field(self, N, H, W, device, dtype):
        # Slowly-drifting low-res field (Perlin approximation) so holes persist across frames instead of flickering.
        scale = max(1, int(getattr(self.cfg.depth, "pw_hole_scale", 8)))
        lh, lw = max(1, H // scale), max(1, W // scale)
        drift = float(getattr(self.cfg.depth, "pw_hole_drift", 0.1))
        new_low = torch.rand(N, 1, lh, lw, device=device, dtype=dtype)
        if getattr(self, "_pw_low", None) is None or self._pw_low.shape != new_low.shape:
            self._pw_low = new_low
        else:
            self._pw_low = (1.0 - drift) * self._pw_low + drift * new_low
        # Smooth the coarse field before upsampling so thresholded regions are blob-like rather than pointy, while keeping the slow drift.
        smooth_sigma = float(getattr(self.cfg.depth, "pw_hole_smooth_sigma", 0.0))
        low = self._pw_low.squeeze(1)
        if smooth_sigma > 0.0 and min(lh, lw) > 1:
            # Reflective padding avoids hard zero borders and wraparound seams.
            self._gaussian_blur_batch(low, smooth_sigma)  # refresh cached kernel
            r = self._pw_blur_pad
            low = F.conv2d(
                F.pad(low.unsqueeze(1), (r, r, r, r), mode="reflect"),
                self._pw_blur_kernel).squeeze(1)
        field = F.interpolate(low.unsqueeze(1), size=(H, W),
                              mode="bilinear", align_corners=False)
        return field.squeeze(1)

    def _apply_pw_holes(self, depth):
        """Apply the temporally-drifting organic hole mask at current resolution."""
        if not getattr(self.cfg.depth, "pw_noise", False):
            return depth
        hp = float(getattr(self.cfg.depth, "pw_hole_prob", 0.0))
        if hp <= 0.0:
            return depth
        N, H, W = depth.shape
        far = -float(self.cfg.depth.far_clip)
        field = self._pw_hole_field(N, H, W, depth.device, depth.dtype)
        hp_env = torch.clamp(hp * self._noise_rate_factor.view(-1), 0.0, 1.0) \
            if torch.is_tensor(getattr(self, "_noise_rate_factor", None)) \
            else hp
        flat = field.flatten(1)
        M = flat.shape[1]
        srt, _ = flat.sort(dim=1)
        if torch.is_tensor(hp_env):
            pos = (1.0 - hp_env) * (M - 1)
            lo = pos.floor().long().clamp_(0, M - 1)
            hi = pos.ceil().long().clamp_(0, M - 1)
            w = (pos - lo.to(pos.dtype)).unsqueeze(1)
            thr = ((1.0 - w) * srt.gather(1, lo.unsqueeze(1))
                   + w * srt.gather(1, hi.unsqueeze(1))).view(N, 1, 1)
        else:
            thr = torch.quantile(flat, 1.0 - hp, dim=1).view(N, 1, 1)
        return torch.where(field > thr, torch.full_like(depth, far), depth)

    def _apply_parkour_wild_noise(self, depth):
        """Parkour-in-the-Wild depth degradation (Rudin et al., arXiv 2505.11164).

        Operates on the cropped raw depth (metres, Isaac negative-in-front
        convention: magnitude = -depth, far/empty = -far_clip), batched [N,H,W],
        mirroring the paper's 5-step model:
          (1) clip + treat too-close/invalid (<pw_empty_min m) as empty (far),
          (2) edge noise: high-gradient pixels randomly set empty or shuffled,
          (3) blind spot: a random 1..k leftmost columns set to far,
          (4) Gaussian blur the whole image.
        The hole mask is followed by deployment hole filling before resize, so
        invalid pixels do not get interpolated into the policy image.
        Gated by cfg.depth.pw_noise; each op is independently 0/off-able so the
        default config is a no-op (backward compatible).
        """
        if not getattr(self.cfg.depth, "pw_noise", False):
            return depth
        far = -float(self.cfg.depth.far_clip)
        N, H, W = depth.shape

        # Per-env linear scale factors (see _depth_noise_factors); absent -> 1x, the configured rates verbatim.
        def _fac(name):
            f = getattr(self, name, None)
            if f is None:
                f = 1.0
            if not torch.is_tensor(f):
                return torch.full((N, 1, 1), float(f), device=depth.device, dtype=depth.dtype)
            return f.to(dtype=depth.dtype)
        rate = _fac("_noise_rate_factor")   # [N,1,1], 0 .. 2
        phys = _fac("_noise_phys_factor")   # [N,1,1], 0 .. 1

        # (1) empty: too-close / invalid -> far (paper: <0.15 m set to 2 m). The threshold rides `phys`: s=0 keeps true near geometry, s>=0.5 the full min range.
        emin = float(getattr(self.cfg.depth, "pw_empty_min", 0.15))
        if emin > 0.0:
            emin_env = emin * phys
            depth = torch.where((depth.abs() < emin_env) | (depth == 0),
                                torch.full_like(depth, far), depth)

        # (2) edge noise: |grad| above threshold -> randomly near, far, or shuffled.
        # Split is configurable: pw_edge_near_frac -> near, pw_edge_far_frac -> far, remainder -> neighbor-shuffle.
        near = -float(self.cfg.depth.near_clip)
        et = float(getattr(self.cfg.depth, "pw_edge_thresh", 0.0))
        if et > 0.0:
            gx = torch.zeros_like(depth); gy = torch.zeros_like(depth)
            gx[:, :, 1:] = (depth[:, :, 1:] - depth[:, :, :-1]).abs()
            gy[:, 1:, :] = (depth[:, 1:, :] - depth[:, :-1, :]).abs()
            edge = (gx + gy) > et
            # Widen the edge band (pw_edge_dilate px each side): the real sensor's far-clip erosion eats 1-2 px into silhouettes.
            dil = int(getattr(self.cfg.depth, "pw_edge_dilate", 0))
            if dil > 0:
                edge = F.max_pool2d(edge.float().unsqueeze(1), kernel_size=2 * dil + 1,
                                    stride=1, padding=dil).squeeze(1) > 0.5
            ep = float(getattr(self.cfg.depth, "pw_edge_prob", 0.5))
            nf = float(getattr(self.cfg.depth, "pw_edge_near_frac", 1/3))
            ff = float(getattr(self.cfg.depth, "pw_edge_far_frac", 1/3))
            ep_eff = torch.clamp(ep * rate, 0.0, 1.0)
            hit = edge & (torch.rand_like(depth) < ep_eff)
            r = torch.rand_like(depth)
            to_near = hit & (r < nf)
            to_far  = hit & (r >= nf) & (r < nf + ff)
            shuf    = hit & (r >= nf + ff)
            rolled = torch.roll(depth, shifts=1, dims=2)  # neighbor pixel
            depth = torch.where(to_near, torch.full_like(depth, near), depth)
            depth = torch.where(to_far,  torch.full_like(depth, far),  depth)
            depth = torch.where(shuf, rolled, depth)

        # (3) blind spot: a PER-ENV band of columns along ONE image edge set to far/empty, where the stereo match falls outside the second imager's FOV.
        # The side is fixed per env (self.pw_bs_side); the width random-walks (pw_blindspot_drift) so the band is temporally stable rather than strobing, and self.pw_bs_slope leans it off-vertical.
        kmax = int(getattr(self.cfg.depth, "pw_blindspot_cols", 0))
        if kmax > 0 and hasattr(self, "pw_bs_k"):
            drift = float(getattr(self.cfg.depth, "pw_blindspot_drift", 0.05))
            target = torch.rand(N, device=depth.device) * kmax
            self.pw_bs_k[:N] = (1.0 - drift) * self.pw_bs_k[:N] + drift * target
            rows = torch.arange(H, device=depth.device, dtype=torch.float32)         # [H]
            k_per_row = (self.pw_bs_k[:N, None]
                         + self.pw_bs_slope[:N, None] * (rows[None, :] - H / 2.0))   # [N,H]
            # Per-row raggedness: the real invalid band's width jitters a pixel or two row-to-row and drops to zero in some rows.
            jrag = float(getattr(self.cfg.depth, "pw_blindspot_jitter", 0.0))
            if jrag > 0.0:
                k_per_row = k_per_row + (torch.rand(N, H, device=depth.device) - 0.5) * 2.0 * jrag
            # Band width rides `phys`: it shrinks to zero columns at s=0.
            k_per_row = (k_per_row * phys.view(-1, 1)).round().clamp(0, W)
            col = torch.arange(W, device=depth.device, dtype=torch.float32)          # [W]
            left = col[None, None, :] < k_per_row[:, :, None]                         # [N,H,W]
            right = col[None, None, :] >= (W - k_per_row[:, :, None])
            mask = torch.where(self.pw_bs_side[:N, None, None], right, left)          # [N,H,W]
            depth = torch.where(mask, torch.full_like(depth, far), depth)

        # (5) Gaussian blur, one shared kernel mixed in per-env by `phys`: s=0 keeps the render's edge sharpness, s>=0.5 the configured optical softness.
        sigma = float(getattr(self.cfg.depth, "pw_blur_sigma", 0.0))
        if sigma > 0.0:
            depth = depth + phys * (self._gaussian_blur_batch(depth, sigma) - depth)
        return depth

    def process_depth_image(self, depth_image, env_id):
        # These operations are replicated on the hardware
        no_noise = os.environ.get("EP_NO_CAMERA_NOISE", "0") == "1"
        far, near = self.cfg.depth.far_clip, self.cfg.depth.near_clip
        rate, phys = self._depth_noise_factors(env_id=env_id)
        if not no_noise:
            # Linear per-env noise: every amplitude and rate is multiplied by `rate`, so s=0 is exactly the clean render and no blend is needed.
            self._noise_rate_factor = rate
            self._noise_phys_factor = phys
            noised = depth_image + self.cfg.depth.dis_noise * rate * 2 * (torch.rand(1)-0.5)[0]
            noised = torch.clip(noised, -far, -near)
            noised = self._apply_parkour_wild_noise(noised.unsqueeze(0)).squeeze(0)
            noised = self._apply_depth_noise(noised)
            noised = self._apply_pw_holes(noised.unsqueeze(0)).squeeze(0)
            noised = self._apply_configured_hole_fill(noised.unsqueeze(0)).squeeze(0)
            self._noise_rate_factor = None
            self._noise_phys_factor = None
        else:
            noised = torch.clip(depth_image, -far, -near)
            noised = self._apply_configured_hole_fill(noised.unsqueeze(0)).squeeze(0)
        empty_below = float(getattr(self.cfg.depth, "empty_below_m", 0.0)) * phys
        if empty_below > 0.0:
            noised = torch.where(noised > -empty_below,
                                 torch.full_like(noised, -far), noised)
        noised = torch.clip(noised, -far, -near)
        noised = F.interpolate(noised[None, None], size=(60, 106),
                               mode="bicubic", align_corners=False).squeeze(0).squeeze(0)
        depth_image = self.crop_depth_image(noised)
        depth_image = self.resize_transform(depth_image[None]).squeeze()
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image

    def _sample_blindspot_side(self, n):
        """Per-env blind-spot edge as a bool tensor [n] (True=right, False=left).
        cfg.depth.pw_blindspot_side: 'left' (default), 'right', or 'random'."""
        mode = str(getattr(self.cfg.depth, "pw_blindspot_side", "left")).lower()
        if mode == "random":
            return torch.rand(n, device=self.device) < 0.5
        return torch.full((n,), mode == "right", dtype=torch.bool, device=self.device)

    def crop_depth_image(self, depth_image):
        return depth_image[:-2, 4:-4]

    def _apply_configured_hole_fill(self, depth):
        """Apply deployment hole fill at the physical camera resolution."""
        if not getattr(self.cfg.depth, "hole_fill", False):
            return depth
        from legged_gym.utils.depth_hole_fill import hole_fill
        positive = (-depth).detach().cpu().numpy()
        invalid = positive >= float(self.cfg.depth.far_clip) - 1e-4
        positive[invalid] = 0.0
        if getattr(self.cfg.depth, "hole_fill_upscale", False):
            fill_h = int(getattr(self.cfg.depth, "hole_fill_height", 270))
            fill_w = int(getattr(self.cfg.depth, "hole_fill_width", 480))
            # Deployment upsamples (nearest-neighbour) before the RealSense filter, so the one-pixel iteration radius is in physical-camera pixels.
            if positive.shape[-2:] != (fill_h, fill_w):
                positive = F.interpolate(
                    torch.from_numpy(positive).to(self.device).unsqueeze(1),
                    size=(fill_h, fill_w), mode="nearest").squeeze(1).cpu().numpy()
        filled = [
            hole_fill(frame,
                      int(getattr(self.cfg.depth, "hole_fill_mode", 1)),
                      int(getattr(self.cfg.depth, "hole_fill_iters", 8)))
            for frame in positive
        ]
        return -torch.from_numpy(np.stack(filled)).to(
            device=depth.device, dtype=depth.dtype)

    def process_depth_batch(self, depth):
        # Batched equivalent of process_depth_image for the Warp path; `depth` is [N, H, W] in the Isaac negative-in-front convention.
        no_noise = os.environ.get("EP_NO_CAMERA_NOISE", "0") == "1"
        far, near = self.cfg.depth.far_clip, self.cfg.depth.near_clip
        # Noise-scale semantics -- ONE LINEAR KNOB (see _depth_noise_factors):
        #   rate = 2s scales every stochastic amplitude and artifact density (0 at s=0, reality-calibrated at s=0.5, 2x at s=1),
        #   phys = min(2s, 1) scales physical sensor properties (min range, blind-spot width, blur; at render time also stereo shadow, own-body geometry, transport latency).
        # Nothing is pixel-blended: the old scheme faded artifacts toward mid-gray below s=0.5, which the real sensor never does. Now s=0 IS the raw ideal ray-cast.
        rate, phys = self._depth_noise_factors(depth.shape[0])
        _dbg_id = getattr(self, "_depth_debug_env_id", None)
        if _dbg_id is not None and 0 <= int(_dbg_id) < depth.shape[0]:
            _dbg_id = int(_dbg_id)
            self._depth_debug_stages = {
                "01_raw_warp": depth[_dbg_id].detach().clone()
            }
        if not no_noise:
            self._noise_rate_factor = rate
            self._noise_phys_factor = phys
            depth = depth + self.cfg.depth.dis_noise * rate \
                * 2 * (torch.rand(depth.shape[0], 1, 1, device=self.device) - 0.5)
            depth = torch.clip(depth, -far, -near)
            depth = self._apply_parkour_wild_noise(depth)
            depth = self._apply_depth_noise(depth)
            depth = self._apply_pw_holes(depth)
            if _dbg_id is not None:
                self._depth_debug_stages["02_noisy_pre_fill"] = depth[_dbg_id].detach().clone()
            depth = self._apply_configured_hole_fill(depth)
            if _dbg_id is not None:
                self._depth_debug_stages["03_hole_filled"] = depth[_dbg_id].detach().clone()
            self._noise_rate_factor = None
            self._noise_phys_factor = None
        else:
            depth = torch.clip(depth, -far, -near)
            depth = self._apply_configured_hole_fill(depth)
        empty_below = float(getattr(self.cfg.depth, "empty_below_m", 0.0))
        if empty_below > 0.0:
            depth = torch.where(depth > -(empty_below * phys),
                                torch.full_like(depth, -far), depth)
        depth = torch.clip(depth, -far, -near)
        depth = F.interpolate(depth.unsqueeze(1), size=(60, 106),
                              mode="bicubic", align_corners=False).squeeze(1)
        depth = depth[:, :-2, 4:-4]
        depth = self.resize_transform(depth)  # [N, resized_h, resized_w]
        depth = self.normalize_depth_image(depth)
        if _dbg_id is not None:
            self._depth_debug_stages["04_policy_final"] = depth[_dbg_id].detach().clone()
        return depth

    def _depth_on_demand(self):
        """Per-env on-demand capture (cfg.depth.render_on_demand).

        Default (off): every env is ray-cast on every capture step and the frames
        are stacked into a ring buffer that get_depth_obs indexes by per-env lag.
        With capture_interval < update_interval that casts update_interval frames
        per env per encoder tick and feeds the network exactly ONE of them.

        On: each env is cast only on the single capture step whose frame it will
        actually consume, i.e. when (global_counter + lag_i) % update_interval == 0.
        Same frame, same pose, same lag distribution, ~update_interval x fewer
        rays. Requires the Warp backend (the Isaac path renders whole-scene) and
        capture_interval == 1 (the 1-step grid is what the lag phase is defined
        on). Off-tick reads of depth_buffer / get_depth_obs return each env's
        latest cast frame, which can be up to update_interval-1 steps stale —
        fine for the runners (they only read on ticks) and for debug views.
        """
        if not getattr(self.cfg.depth, "render_on_demand", False):
            return False
        if not self._warp_enabled():
            raise ValueError("cfg.depth.render_on_demand requires the Warp depth "
                             "backend (cfg.depth.use_warp = True)")
        cap = int(getattr(self.cfg.depth, "capture_interval",
                          self.cfg.depth.update_interval))
        if cap != 1:
            raise ValueError("cfg.depth.render_on_demand requires capture_interval "
                             f"== 1 (got {cap}): the per-env lag phase is defined "
                             "on the 1-step capture grid")
        return True

    def update_depth_buffer(self):
        if not self.cfg.depth.use_camera:
            return

        capture_interval = int(getattr(
            self.cfg.depth, "capture_interval", self.cfg.depth.update_interval))
        if self.global_counter % capture_interval != 0:
            return

        if self._depth_on_demand():
            self._update_depth_buffer_on_demand()
            return

        if self._warp_enabled():
            # GPU ray-cast for all envs at once, then one batched crop/clip/resize/normalize and a vectorized buffer roll; rigid_body_states is already current.
            # Per-env geometric degradation: the stereo shadow is thinned to `phys` of its occluded pixels (real block matching recovers most), own-body rendering is gated per episode.
            depth = self.warp_cam.render(
                self.root_states[:, :3], self.base_quat,
                body_pos=self.rigid_body_states[..., 0:3],
                body_quat=self.rigid_body_states[..., 3:7],
                shadow_keep=self._depth_phys_factor_1d(),
                self_on=getattr(self, "depth_self_geom_on", None))  # [N, H, W]
            new_frame = self.process_depth_batch(depth)  # [N, resized_h, resized_w]
            self.depth_buffer = torch.cat([self.depth_buffer[:, 1:], new_frame.unsqueeze(1)], dim=1)
            init_flag = self.episode_length_buf <= 1
            if init_flag.any():
                self.depth_buffer[init_flag] = new_frame[init_flag].unsqueeze(1).repeat(1, self.cfg.depth.buffer_len, 1, 1)
            return

        self.gym.step_graphics(self.sim) # required to render in headless mode
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        for i in range(self.num_envs):
            depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim, 
                                                                self.envs[i], 
                                                                self.cam_handles[i],
                                                                gymapi.IMAGE_DEPTH)
            
            depth_image = gymtorch.wrap_tensor(depth_image_)
            depth_image = self.process_depth_image(depth_image, i)

            init_flag = self.episode_length_buf <= 1
            if init_flag[i]:
                self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
            else:
                self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)], dim=0)

        self.gym.end_access_image_tensors(self.sim)

    def _update_depth_buffer_on_demand(self):
        """Cast ONLY the envs whose frame the encoder is about to consume.

        See _depth_on_demand. Env i is due on the capture step exactly lag_i
        steps before the next encoder tick, so it is cast once per encoder
        window instead of update_interval times, and the frame it lands on is
        bit-for-bit the one the legacy ring buffer would have handed back.
        """
        U = int(self.cfg.depth.update_interval)
        due = ((self.global_counter + self.depth_latency) % U == 0)
        # First two steps of an episode: prefill every ring slot as the legacy path does (lag suppressed until the buffer refills).
        fresh = self.episode_length_buf <= 1
        active = due | fresh
        depth = self.warp_cam.render(
            self.root_states[:, :3], self.base_quat,
            body_pos=self.rigid_body_states[..., 0:3],
            body_quat=self.rigid_body_states[..., 3:7],
            shadow_keep=self._depth_phys_factor_1d(),
            self_on=getattr(self, "depth_self_geom_on", None),
            active=active)                      # only these rows are valid
        # The post-process runs on the FULL batch on purpose: the degradation carries per-env temporal state (blind-spot random walk, drifting hole field) that must advance once per call.
        # Rows of envs that were not cast are computed and dropped; the rays are the cost.
        new_frame = self.process_depth_batch(depth)
        # Masked writes, not index lists: avoids a per-step nonzero()/host sync in the rollout hot loop.
        n_slots = self.depth_buffer.shape[1]
        roll = active.view(-1, 1, 1)
        for s in range(n_slots - 1):            # shift older slots down one
            self.depth_buffer[:, s] = torch.where(roll, self.depth_buffer[:, s + 1],
                                                  self.depth_buffer[:, s])
        self.depth_buffer[:, -1] = torch.where(roll, new_frame, self.depth_buffer[:, -1])
        if n_slots > 1:
            # Fresh episodes: every slot holds the new frame, so no pre-reset frame can be served.
            init = fresh.view(-1, 1, 1)
            for s in range(n_slots - 1):
                self.depth_buffer[:, s] = torch.where(init, new_frame,
                                                      self.depth_buffer[:, s])

    def get_depth_obs(self):
        """The depth frame the policy/encoder consumes, with per-env camera
        latency applied. The freshest render is depth_buffer[:, -1]; a latency
        of L capture frames returns depth_buffer[:, -1 - L] (older = laggier).
        Returns a clone (matches the previous .clone() call-site semantics).
        None if the camera is off."""
        if not self.cfg.depth.use_camera:
            return None
        if self.depth_latency_max == 0:
            return self.depth_buffer[:, -1].clone()
        if self._depth_on_demand():
            # Lag is baked into the capture step, so only the encoder window is indexed: lag L lands in the current window for L < update_interval, else L // U windows back (L == U is the previous window's tick frame).
            k = torch.div(self.depth_latency, int(self.cfg.depth.update_interval),
                          rounding_mode='floor')
            idx = (self.depth_buffer.shape[1] - 1 - k).clamp_min(0)               # [N]
        else:
            idx = (self.depth_buffer.shape[1] - 1 - self.depth_latency).clamp_min(0)  # [N]
        env_idx = torch.arange(self.num_envs, device=self.device)
        return self.depth_buffer[env_idx, idx].clone()

    def _update_goals(self):
        next_flag = self.reach_goal_timer > self.cfg.env.reach_goal_delay / self.dt
        # Freeze the goal index during the post-circuit hard-stop hold so it can't advance past num_goals and overflow the goal gather.
        if hasattr(self, 'final_goal_reached'):
            next_flag = next_flag & ~self.final_goal_reached
        self.cur_goal_idx[next_flag] += 1
        self.reach_goal_timer[next_flag] = 0
        self.max_goal_idx_ep = torch.maximum(self.max_goal_idx_ep, self.cur_goal_idx)

        # env19 hurdle wococo: force-arm slot 1 (backward crossing) once the robot enters the return phase, however the forward crossing finished. Idempotent.
        if hasattr(self, 'wococo_hurdle_idx'):
            arm_slot1 = (
                (self.env_class == 19)
                & (self.cur_goal_idx >= self._flyball_return_goal_idx)
                & (self.wococo_hurdle_idx == 0)
            )
            if arm_slot1.any():
                self.wococo_hurdle_idx[arm_slot1]   = 1
                self.wococo_hurdle_stage[arm_slot1] = 0
                self.wococo_hurdle_s1_time[arm_slot1] = 0
                # Clear slot-1 episode-max so one-shot reward modes can still fire on the new slot's stages.
                self.wococo_hurdle_episode_max_stage[arm_slot1, 1] = 0

        self.reached_goal_ids = torch.norm(self.root_states[:, :2] - self.cur_goals[:, :2], dim=1) < self.cfg.env.next_goal_threshold

        # Wall goal advancement. env19 (flyball) wall is g2, env15 (two_hurdles) is g4; the default distance threshold cannot be used because the analytic wall goal sits inside the wall surface -- the robot can never reach it geometrically.
        # With wococo_stage active both advance on wococo_contact_stage >= plant_complete_stage (P3=3 LANDING, P4/P2=2, P1=1); with wococo off, fall back to all four feet contacting the wall front face (feet_x>=pit_x1).
        at_wall_19 = (self.env_class == 19) & (self.cur_goal_idx == self._flyball_wall_goal_idx)
        at_wall_15 = (self.env_class == 15) & (self.cur_goal_idx == 4)
        wall_goal_active = at_wall_19 | at_wall_15

        feet_pos_x = self.rigid_body_states[:, self.feet_indices, 0]
        feet_contact_norm = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.0
        feet_in_wall_x = feet_pos_x >= self.env_flyball_pit_x1.unsqueeze(1)
        on_wall = feet_in_wall_x & feet_contact_norm
        # Optional stricter wall-touch gate above a terrain level: foot fx < cfg.env.wall_touched_fx_threshold, else the legacy magnitude > 1 N check. Both still require feet_x >= pit_x1.
        fx_thresh = getattr(self.cfg.env, 'wall_touched_fx_threshold', None)
        if fx_thresh is not None:
            min_level = int(getattr(self.cfg.env, 'wall_touched_min_level', 0))
            feet_fx = self.contact_forces[:, self.feet_indices, 0]
            strong_press = feet_fx < float(fx_thresh)
            on_wall_strict = feet_in_wall_x & strong_press
            use_strict = (self.terrain_levels > min_level).unsqueeze(1)
            on_wall = torch.where(use_strict, on_wall_strict, on_wall)
        self.wall_feet_touched |= wall_goal_active.unsqueeze(1) & on_wall
        all_feet_touched = self.wall_feet_touched.all(dim=1)

        # The wall goal (g2) always advances on the plant FSM reaching its landing stage (P4L plant_complete_stage = 2 = LANDING).
        # The gate reads wococo_contact_stage (driven by _reward_wococo_contact_pts in every ablation), NOT the wococo_stage reward scale, so ablations don't silently fall back to the easy all-feet gate; that gate needs wococo_goal_advance_mode == 'all_feet'.
        adv_mode = str(getattr(self.cfg.rewards, 'wococo_goal_advance_mode', 'plant_stage'))
        if adv_mode != 'all_feet':
            _pcs = self._wococo_plant_complete_stage   # == 2 (LANDING) for P4L
            env19_reached = at_wall_19 & (self.wococo_contact_stage >= _pcs)
            env15_reached = at_wall_15 & (self.wococo_contact_stage >= _pcs)
        else:
            env19_reached = at_wall_19 & all_feet_touched
            env15_reached = at_wall_15 & all_feet_touched

        wall_goal_reached = env19_reached | env15_reached
        self.reached_goal_ids = torch.where(wall_goal_active, wall_goal_reached, self.reached_goal_ids)
        self.reach_goal_timer[wall_goal_reached] = self.cfg.env.reach_goal_delay / self.dt + 1

        self.reach_goal_timer[self.reached_goal_ids] += 1

        # Padded goals (cur position == next position) skip the delay and advance immediately, gated on reached_goal_ids so a real waypoint sharing xy with the next slot isn't skipped before arrival.
        padded_goal = torch.all(torch.abs(self.cur_goals[:, :2] - self.next_goals[:, :2]) < 1e-3, dim=1)
        self.reach_goal_timer[padded_goal & self.reached_goal_ids] = self.cfg.env.reach_goal_delay / self.dt + 1

        self.target_pos_rel = self.cur_goals[:, :2] - self.root_states[:, :2]
        self.next_target_pos_rel = self.next_goals[:, :2] - self.root_states[:, :2]

        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt

        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        feet_force_norm = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1)
        contact = feet_force_norm > 2.
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        # Observed contact bit with the per-env/per-foot randomized threshold; rewards/FSM keep the clean 2 N contact_filt.
        if getattr(self.cfg.domain_rand, "randomize_contact_threshold", False):
            contact_obs = feet_force_norm > self.contact_thresh
            self.contact_filt_obs = torch.logical_or(contact_obs, self.last_contacts_obs)
            self.last_contacts_obs = contact_obs
        else:
            self.contact_filt_obs = self.contact_filt
        
        self._update_goals()
        self._post_physics_step_callback()
        self._update_soft_curriculum_latches()

        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

        self.update_depth_buffer()

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self._draw_goals()
            self._draw_feet()
            if self.cfg.depth.use_camera:
                window_name = "Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
                cv2.waitKey(1)

    def reindex_feet(self, vec):
        return vec[:, [1, 0, 3, 2]]

    def reindex(self, vec):
        return vec[:, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]

    def _update_soft_curriculum_latches(self):
        """Trigger and advance sticky wall/clearance curriculum ramps once/step."""
        evaluation_mode = bool(getattr(self, "evaluation_mode", False))
        mean_level = None

        def _mean_level():
            nonlocal mean_level
            if mean_level is None:
                mean_level = float(self.terrain_levels.float().mean())
            return mean_level

        term_enabled = bool(getattr(
            self.cfg.rewards, 'enable_wococo_armed_term_exempt', False))
        if term_enabled:
            threshold = float(getattr(
                self.cfg.rewards, 'wococo_armed_term_exempt_min_mean_level', 6.0))
            if (self.wococo_armed_term_exempt_trigger_step < 0
                    and (evaluation_mode or _mean_level() >= threshold)):
                self.wococo_armed_term_exempt_trigger_step = self.common_step_counter
            self.wococo_armed_term_exempt_armed = (
                self.wococo_armed_term_exempt_trigger_step >= 0)
            if evaluation_mode and self.wococo_armed_term_exempt_armed:
                self.wococo_armed_term_exempt_alpha = 1.0
            else:
                self.wococo_armed_term_exempt_alpha = latched_linear_alpha(
                    self.common_step_counter,
                    self.wococo_armed_term_exempt_trigger_step,
                    getattr(self.cfg.rewards,
                            'wococo_armed_term_exempt_ramp_steps', 0))
        else:
            self.wococo_armed_term_exempt_armed = False
            self.wococo_armed_term_exempt_alpha = 0.0

        clearance_enabled = 'hurdle_clearance' in self.reward_scales
        if clearance_enabled:
            # hurdle_clearance_arm_step, when set, REPLACES the fleet-mean-level trigger with a fixed env-step count (iteration * num_steps_per_env; cliffside runs 24 steps/iteration, so iteration 2500 = 60 000).
            # A wall-clock trigger lands once at a predictable time; a mean-level trigger fires when the fleet is doing well and stays latched after a demotion.
            _arm_step = getattr(self.cfg.rewards, 'hurdle_clearance_arm_step', None)
            _arm_lvl = getattr(self.cfg.rewards, 'hurdle_clearance_arm_mean_level', 6.0)
            if self.hurdle_clearance_trigger_step < 0:
                if _arm_step is not None:
                    _fire = evaluation_mode or self.common_step_counter >= int(_arm_step)
                else:
                    _fire = evaluation_mode or _mean_level() >= float(_arm_lvl)
                if _fire:
                    self.hurdle_clearance_trigger_step = self.common_step_counter
                    print(f"[hurdle_clearance] ARMED at step {self.common_step_counter}"
                          + (f" (>= {int(_arm_step)})" if _arm_step is not None
                             else f" (fleet mean level >= {_arm_lvl})"))
            self.hurdle_clearance_armed = self.hurdle_clearance_trigger_step >= 0
            if evaluation_mode and self.hurdle_clearance_armed:
                self.hurdle_clearance_alpha = 1.0
            else:
                self.hurdle_clearance_alpha = latched_linear_alpha(
                    self.common_step_counter,
                    self.hurdle_clearance_trigger_step,
                    getattr(self.cfg.rewards, 'hurdle_clearance_ramp_steps', 0),
                    getattr(self.cfg.rewards,
                            'hurdle_clearance_ramp_delay_steps', 0))
        else:
            self.hurdle_clearance_armed = False
            self.hurdle_clearance_alpha = 0.0

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.zeros((self.num_envs, ), dtype=torch.bool, device=self.device)
        roll_cutoff = torch.abs(self.roll) > self._roll_term_rad
        pitch_cutoff = torch.abs(self.pitch) > 1.5
        reach_goal_cutoff = self.cur_goal_idx >= self.cfg.terrain.num_goals
        height_cutoff = self.root_states[:, 2] < -0.25

        # Lateral-deviation termination (opt-in via cfg.env.terminate_on_lateral_deviation): terminate when |root_y - env_origin_y| exceeds lateral_deviation_term_m off the spawn corridor centerline.
        # A real termination (reset_buf, NOT time_out) so the critic learns straying off course is a failure.
        lateral_cutoff = torch.zeros_like(roll_cutoff)
        if bool(getattr(self.cfg.env, 'terminate_on_lateral_deviation', False)):
            lat_thresh = float(getattr(self.cfg.env, 'lateral_deviation_term_m', 1.5))
            lateral_dev = torch.abs(self.root_states[:, 1] - self.env_origins[:, 1])
            lateral_cutoff = lateral_dev > lat_thresh

        # Yaw-tracking termination (distillation only): once the student's heading drifts too far from the oracle heading the teacher's labels stop being trustworthy, so cut the episode before it corrupts the DAgger dataset.
        # Computed fresh here (NOT self.delta_yaw, which is unwrapped and refreshed every 5 steps) and wrap_to_pi'd so the heading flip at the wall turn doesn't trip a false positive. Gated on use_camera.
        yaw_track_cutoff = torch.zeros_like(roll_cutoff)
        if self.cfg.depth.use_camera and bool(getattr(self.cfg.env, "terminate_on_yaw_tracking", True)):
            yaw_thresh = float(getattr(self.cfg.env, "yaw_tracking_term_thresh", 0.6))
            yaw_err = torch.abs(wrap_to_pi(self.target_yaw - self.yaw))
            yaw_track_cutoff = yaw_err > yaw_thresh
            # Wall-turn exemption: silence the yaw-tracking termination at the wall goal AND the goal right after it, where the turn legitimately swings the heading far off the goal direction.
            # Wall goal is g2 for flyball (env19), g4 for two_hurdles (env15); non-wall envs are never exempted.
            wall_env  = (self.env_class == 19) | (self.env_class == 15)
            wall_goal = torch.full_like(self.cur_goal_idx, self._flyball_wall_goal_idx)
            wall_goal[self.env_class == 15] = 4
            yaw_exempt = wall_env & ((self.cur_goal_idx == wall_goal) | (self.cur_goal_idx == wall_goal + 1))
            yaw_track_cutoff = yaw_track_cutoff & ~yaw_exempt

        # Post-circuit hard-stop hold (cliffside, opt-in): on final goal reached, zero the velocity command and hold the episode open for final_goal_stop_s so braking and standing still are rewarded.
        # The reach-goal timeout fires only once the hold has elapsed; final_goal_reached freezes cur_goal_idx (see _update_goals).
        if bool(getattr(self.cfg.rewards, 'enable_final_goal_hard_stop', False)):
            hs_env = (self.env_class == 19)
            stop_steps = int(round(float(getattr(self.cfg.rewards, 'final_goal_stop_s', 5.0)) / self.dt))
            newly_done = reach_goal_cutoff & hs_env & ~self.final_goal_reached
            self.final_goal_stop_steps[newly_done] = stop_steps
            self.final_goal_reached |= (reach_goal_cutoff & hs_env)
            in_stop = self.final_goal_reached & (self.final_goal_stop_steps > 0)
            self.commands[in_stop, :3] = 0.0   # zero lin_x / lin_y / ang_vel_yaw
            # Pin the heading objective to the CURRENT yaw so _reward_tracking_yaw rewards holding the stopped heading instead of chasing the degenerate target_yaw.
            self.target_yaw[in_stop] = self.yaw[in_stop]
            self.next_target_yaw[in_stop] = self.yaw[in_stop]
            self.final_goal_stop_steps[self.final_goal_reached] = \
                (self.final_goal_stop_steps[self.final_goal_reached] - 1).clamp(min=0)
            hold_elapsed = self.final_goal_reached & (self.final_goal_stop_steps <= 0)
            reach_goal_cutoff = torch.where(hs_env, hold_elapsed, reach_goal_cutoff)

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.time_out_buf |= reach_goal_cutoff

        # Wococo-only safety terminations. When the wococo state machine is off
        # (Phase-1 foundation training), behaviour matches ad05628: no base-contact
        # termination, no premature-wall termination.
        wococo_env = (self.env_class == 19) | (self.env_class == 15)
        # Hard-contact termination on cfg.asset.terminate_after_contacts_on; an empty cfg list makes it a no-op and leaves avoidance to the soft _reward_collision penalty.
        if self.termination_contact_indices.numel() > 0:
            base_contact = torch.any(
                torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 10.0,
                dim=1,
            )
        else:
            base_contact = torch.zeros_like(wococo_env)
        # Premature-wall termination (opt-in via cfg.rewards.enable_premature_wall_termination): a hard wall slam before the robot commits to the plant resets the episode. 'Before the plant' is ~plant_armed, NOT stage==0, since for P1/P2/P4 stage 0 is already a plant stage.
        if bool(getattr(self.cfg.rewards, 'enable_premature_wall_termination', False)):
            pre_plant = wococo_env & ~self.wococo_armed
            feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
            foot_force = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
            foot_in_wall = self._in_region(feet_pos, self.env_flyball_wall_region)
            prem_force = float(getattr(self.cfg.rewards, 'wococo_premature_wall_force_threshold', 50.0))
            premature_wall = pre_plant & (foot_in_wall & (foot_force > prem_force)).any(dim=-1)
        else:
            premature_wall = torch.zeros_like(wococo_env)

        # Latch first sentinel entry for post-plant roll and yaw grace periods.
        at_sentinel = self.wococo_contact_stage >= self._wococo_sentinel_stage
        newly_sentinel = at_sentinel & (self.wococo_sentinel_reached_step < 0)
        self.wococo_sentinel_reached_step[newly_sentinel] = self.episode_length_buf[newly_sentinel]
        if bool(getattr(self.cfg.rewards, 'exempt_penalties_at_wall_plant', False)):
            in_plant = self.wococo_armed & (self.wococo_contact_stage < self._wococo_sentinel_stage)
            # Hold the roll exemption for _wococo_roll_exempt_post_steps after the sentinel-reach latched above.
            post_sentinel = (self.wococo_sentinel_reached_step >= 0) & (
                (self.episode_length_buf - self.wococo_sentinel_reached_step)
                < self._wococo_roll_exempt_post_steps)
            roll_exempt = in_plant | post_sentinel
            # Expose the plant window so reward terms reuse the EXACT same window (_reward_dof_error scales its penalty to 25% inside it).
            self.wococo_plant_window = roll_exempt
            roll_thresh = torch.full_like(self.roll, self._roll_term_rad)
            roll_thresh[roll_exempt] = self._roll_exempt_rad
            roll_cutoff = torch.abs(self.roll) > roll_thresh

        # Reset plants that remain armed without advancing before the stage budget.
        plant_dwell = torch.zeros_like(wococo_env)
        if bool(getattr(self.cfg.rewards, 'enable_plant_dwell_termination', False)):
            advanced = self.wococo_contact_stage > self.wococo_plant_dwell_stage
            self.wococo_plant_dwell[advanced] = 0
            self.wococo_plant_dwell_stage[:] = self.wococo_contact_stage
            armed = wococo_env & self.wococo_armed & \
                    (self.wococo_contact_stage < self._wococo_sentinel_stage)
            self.wococo_plant_dwell[~armed] = 0   # only count while committed & not yet done
            self.wococo_plant_dwell[armed] += 1
            dwell_max = int(getattr(self.cfg.rewards, 'wococo_plant_dwell_max_steps', 200))
            plant_dwell = armed & (self.wococo_plant_dwell > dwell_max)

        # Plant-failure early terminations (P4L, opt-in): cut the episode the moment the wall plant / floor landing fails, so the robot never rides onto the return path still armed -- where the clip-exempt contact_pts penalty taught it to suicide via base contact.
        # Both triggers are hard failures (reset_buf, NOT time_out); env_class 19/15 only.
        stage1_ground    = torch.zeros_like(wococo_env)
        stage2_overshoot = torch.zeros_like(wococo_env)
        # Curriculum latch: hold both terminations off until the fleet-average terrain level first reaches the arm threshold; once armed it stays armed.
        if not self.wococo_plant_fail_term_armed:
            arm_level = float(getattr(self.cfg.rewards, 'wococo_plant_fail_term_arm_mean_level', 4.0))
            if self.terrain_levels.float().mean() >= arm_level:
                self.wococo_plant_fail_term_armed = True
                self.wococo_plant_fail_arm_step = self.common_step_counter
        # Optional ramp on the arming, which is otherwise a STEP that hands every env two instant-death conditions at once. Measured cost of that step (E02): term_stage1_ground 0.000 -> 0.377 within ~40 updates and fleet level 3.95 -> 3.16.
        # With wococo_plant_fail_term_ramp_steps set, the kill applies to a Bernoulli(alpha) subset with alpha 0 -> 1 over that many steps. Unset = the original step behaviour.
        _pf_ramp = int(getattr(self.cfg.rewards, 'wococo_plant_fail_term_ramp_steps', 0) or 0)
        if _pf_ramp > 0 and self.wococo_plant_fail_term_armed:
            self.wococo_plant_fail_alpha = latched_linear_alpha(
                self.common_step_counter,
                getattr(self, 'wococo_plant_fail_arm_step', self.common_step_counter),
                _pf_ramp)
        else:
            self.wococo_plant_fail_alpha = 1.0
        if (bool(getattr(self.cfg.rewards, 'enable_wococo_plant_fail_termination', False))
                and self.wococo_plant_fail_term_armed):
            stage = self.wococo_contact_stage
            # (1) REAR_PLANT: ANY foot in contact with the FLOOR (x < wall_front; wall feet have x >= env_flyball_pit_x1) means the wall plant collapsed.
            feet_pos_x = self.rigid_body_states[:, self.feet_indices, 0]          # (N,4)
            feet_pos_z = self.rigid_body_states[:, self.feet_indices, 2]          # (N,4)
            on_wall_x = feet_pos_x >= self.env_flyball_pit_x1.unsqueeze(1)        # (N,4)
            # Height gate: the wall is a RAMP, so a foot planted on its shallow toe is at x < pit_x1 yet is a good wall contact. Requiring the contact to also sit below ground_max_z means only a floor-level foot reads as a collapsed plant.
            # 1e9 = gate disabled (pre-gate behaviour), so the raw termination and this refinement stay separable in ablations.
            ground_max_z = float(getattr(self.cfg.rewards, 'wococo_stage1_ground_max_z', 1.0e9))
            foot_low = feet_pos_z < ground_max_z                                  # (N,4)
            foot_on_ground = self.contact_filt & ~on_wall_x & foot_low            # (N,4)
            stage1_ground = wococo_env & (stage == 1) & foot_on_ground.any(dim=-1)
            # (2) LANDING: base x has overshot the front landing footholds (at wall_front - landing_foot_standoff) toward spawn, so the body flew past the landing zone and can't plant its front feet.
            if hasattr(self, 'env_flyball_landing_foot_target'):
                margin = float(getattr(self.cfg.rewards, 'wococo_stage2_overshoot_margin_m', 0.15))
                # FL/FR are staggered in x, so take the pair midpoint.
                foot_tgt_x = self.env_flyball_landing_foot_target[:, :, 0].mean(dim=1)   # (N,)
                base_overshot = self.root_states[:, 0] < (foot_tgt_x - margin)
                stage2_overshoot = wococo_env & (stage == 2) & base_overshot
            # Apply the arming ramp (alpha == 1.0 unless the ramp is configured).
            if float(self.wococo_plant_fail_alpha) < 1.0:
                _keep = (torch.rand(self.num_envs, device=self.device)
                         < float(self.wococo_plant_fail_alpha))
                stage1_ground = stage1_ground & _keep
                stage2_overshoot = stage2_overshoot & _keep

        # Late-curriculum attitude / base-contact tolerance while armed: the sticky, rate-limited global alpha (see _update_soft_curriculum_latches) applies only from plant commit to the sentinel.
        # Finite task-specific limits avoid the old discontinuity where roll, pitch and base-contact terminations all vanished on one step.
        term_alpha = float(self.wococo_armed_term_exempt_alpha)
        armed_term_exempt = (
            wococo_env & self.wococo_armed & (term_alpha > 0.0))
        if term_alpha > 0.0:
            roll_max_deg = getattr(
                self.cfg.rewards, 'wococo_armed_roll_term_max_deg', None)
            if roll_max_deg is None:
                # Backward-compatible legacy mode: full exemption only at alpha=1.
                if term_alpha >= 1.0:
                    roll_cutoff = roll_cutoff & ~armed_term_exempt
            else:
                roll_limit = self._roll_term_rad + term_alpha * (
                    math.radians(float(roll_max_deg)) - self._roll_term_rad)
                roll_cutoff = torch.where(
                    armed_term_exempt,
                    torch.abs(self.roll) > roll_limit,
                    roll_cutoff)

            pitch_max_deg = getattr(
                self.cfg.rewards, 'wococo_armed_pitch_term_max_deg', None)
            if pitch_max_deg is None:
                if term_alpha >= 1.0:
                    pitch_cutoff = pitch_cutoff & ~armed_term_exempt
            else:
                pitch_base = 1.5
                pitch_limit = pitch_base + term_alpha * (
                    math.radians(float(pitch_max_deg)) - pitch_base)
                pitch_cutoff = torch.where(
                    armed_term_exempt,
                    torch.abs(self.pitch) > pitch_limit,
                    pitch_cutoff)

            # Explicit mask for tasks that want the attitude terminations completely absent once the soft ramp finishes, rather than relying on unreachable Euler bounds.
            if (term_alpha >= 1.0 and bool(getattr(
                    self.cfg.rewards,
                    'wococo_armed_disable_attitude_terms_at_full', False))):
                roll_cutoff = roll_cutoff & ~armed_term_exempt
                pitch_cutoff = pitch_cutoff & ~armed_term_exempt

        # Base contact uses persistence rather than a randomized hard kill: at alpha=0 it terminates immediately, and cliffside ramps to 99 grace samples,
        # so the 100th consecutive 20 ms sample ends the episode at exactly 2 s.
        armed_base_contact = armed_term_exempt & base_contact
        self.wococo_base_contact_streak = torch.where(
            armed_base_contact,
            self.wococo_base_contact_streak + 1,
            torch.zeros_like(self.wococo_base_contact_streak))
        grace_max = getattr(
            self.cfg.rewards, 'wococo_armed_base_contact_grace_steps', None)
        if grace_max is None:
            if term_alpha >= 1.0:
                base_contact = base_contact & ~armed_term_exempt
        else:
            grace_steps = term_alpha * max(0, int(grace_max))
            grace_expired = self.wococo_base_contact_streak.float() > grace_steps
            base_contact = base_contact & (~armed_term_exempt | grace_expired)

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= roll_cutoff
        self.reset_buf |= pitch_cutoff
        self.reset_buf |= height_cutoff
        self.reset_buf |= base_contact
        self.reset_buf |= premature_wall
        self.reset_buf |= yaw_track_cutoff
        self.reset_buf |= plant_dwell
        self.reset_buf |= lateral_cutoff
        self.reset_buf |= stage1_ground
        self.reset_buf |= stage2_overshoot

        # Expose per-cause flags so reset_idx can log which termination actually fired.
        self.term_roll            = roll_cutoff
        self.term_wococo_stage1_ground    = stage1_ground
        self.term_wococo_stage2_overshoot = stage2_overshoot
        self.term_pitch           = pitch_cutoff
        self.term_height          = height_cutoff
        self.term_base_contact    = base_contact
        self.term_reach_goal      = reach_goal_cutoff
        self.term_timeout         = self.episode_length_buf > self.max_episode_length
        self.term_premature_wall  = premature_wall
        self.term_yaw_tracking    = yaw_track_cutoff
        self.term_plant_dwell     = plant_dwell
        self.term_lateral         = lateral_cutoff
        # Not a termination: the per-env mask of envs whose roll / pitch / base-contact kills were suppressed this step (logged as wococo/armed_term_exempt).
        self.wococo_armed_term_exempt = armed_term_exempt

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        if getattr(self.cfg.rewards, "hurdle_clearance_per_env_draw", False):
            _clear_alpha = 1.0 if bool(getattr(self, "evaluation_mode", False)) else float(self.hurdle_clearance_alpha)
            self.hurdle_clearance_env_enabled[env_ids] = (
                torch.rand(len(env_ids), device=self.device) < _clear_alpha)
        evaluation_mode = bool(getattr(self, "evaluation_mode", False))
        if self.cfg.terrain.curriculum and not evaluation_mode:
            self._update_terrain_curriculum(env_ids)
        # Expand command ranges independently for successful reset envs.
        if self.cfg.commands.curriculum and not evaluation_mode:
            self.update_command_curriculum(env_ids)

        # Fleet-level action-delay latch: once the fleet-mean terrain level first reaches the threshold, the fixed 1-step (20 ms) delay engages permanently (see step()).
        # Checked here because terrain levels only change on reset. Evaluation latches immediately -- a mature policy must be evaluated with the delay it deploys with.
        # action_delay_latch_step, when set, REPLACES the mean-level trigger with a fixed env-step count (iteration * num_steps_per_env; cliffside runs 24 steps/iteration, so iteration 5000 = 120 000), decoupling the delay from the curriculum.
        _latch_step = getattr(self.cfg.domain_rand, "action_delay_latch_step", None)
        _latch_at = getattr(self.cfg.domain_rand, "action_delay_latch_mean_level", None)
        if (not self.action_delay_latched and self.cfg.domain_rand.action_delay
                and (_latch_step is not None or _latch_at is not None)):
            _mean_level = self.terrain_levels.float().mean()
            if _latch_step is not None:
                _fire = evaluation_mode or self.common_step_counter >= int(_latch_step)
                _why = f"step {self.common_step_counter} >= {int(_latch_step)}"
            else:
                _fire = evaluation_mode or _mean_level >= float(_latch_at)
                _why = f"fleet mean level {_mean_level:.2f} >= {_latch_at}"
            if _fire:
                self.action_delay_latched = True
                print(f"[action_delay] 1-step (20 ms) action delay LATCHED "
                      f"({_why}, fleet mean level {_mean_level:.2f}"
                      f"{', evaluation' if evaluation_mode else ''})")

        # Sample the action-delay domain for each new episode. During the ramp the fleet holds a controlled mixture of exact 0-step and exact 1-step episodes; no fractional delay is used.
        _delay_ramp = int(getattr(self.cfg.domain_rand, "action_delay_ramp_steps", 0) or 0)
        if self.cfg.domain_rand.action_delay:
            if getattr(self.cfg.domain_rand, "action_delay_start_latched", False):
                self.action_delay_episode_enabled[env_ids] = True
            elif _delay_ramp > 0 and _latch_step is not None:
                _delay_alpha = min(1.0, max(0.0, (self.common_step_counter - int(_latch_step)) / float(_delay_ramp)))
                self.action_delay_episode_enabled[env_ids] = (
                    torch.rand(len(env_ids), device=self.device) < _delay_alpha)
            else:
                self.action_delay_episode_enabled[env_ids] = self.action_delay_latched

        # Scale DR with either each env's level or the fleet-mean curriculum level.
        if getattr(self.cfg.domain_rand, "dr_lock_to_terrain_level", False):
            s_lo, s_hi = self.cfg.domain_rand.dr_scale_range
            if getattr(self.cfg.domain_rand, "dr_ramp_use_mean_level", False):
                mean_level = self.terrain_levels.float().mean()
                # Ignore the initial level spread until a cold fleet first demotes.
                if not self.dr_ramp_armed and mean_level < float(getattr(self.cfg.domain_rand, "dr_ramp_arm_below_mean_level", 0.5)):
                    self.dr_ramp_armed = True
                if not self.dr_ramp_armed:
                    self.dr_scale[env_ids] = 0.0
                    # Unarmed fleet => zero sensor lag too (clean cold start).
                    self._dr_fleet_latency_ramp = 0.0
                else:
                    full_at = float(getattr(self.cfg.domain_rand, "dr_ramp_full_at_mean_level", self.max_terrain_level - 1))
                    alpha = (mean_level / max(full_at, 1e-6)).clamp(0., 1.)
                    ceiling = s_lo + (s_hi - s_lo) * alpha
                    # Sensor latency tracks the fleet ceiling, exempt from the per-env draw below; alpha is already normalized 0..1 progress.
                    self._dr_fleet_latency_ramp = float(alpha)
                    if getattr(self.cfg.domain_rand, "dr_scale_sample_per_env", False):
                        self.dr_scale[env_ids] = s_lo + (ceiling - s_lo) * torch.rand(len(env_ids), device=self.device)
                    else:
                        self.dr_scale[env_ids] = ceiling
            else:
                alpha = (self.terrain_levels[env_ids].float() / max(self.max_terrain_level - 1, 1)).clamp(0., 1.)
                self.dr_scale[env_ids] = s_lo + (s_hi - s_lo) * alpha

            # Ramp the reset-time DR terms with the per-env dr_scale: PD-gain and motor-offset are re-rolled here so their deviation from nominal scales 0 (cold) -> full, and dr_scale=0 gives nominal gains / zero offset.
            _sc = self.dr_scale[env_ids].unsqueeze(-1)   # (n, 1)
            if getattr(self.cfg.domain_rand, "randomize_pd_gain", False):
                kp_rng = self.cfg.domain_rand.kp_range
                kd_rng = self.cfg.domain_rand.kd_range
                self.kp_factors[env_ids] = 1.0 + (torch_rand_float(kp_rng[0], kp_rng[1], (len(env_ids), self.num_actions), device=self.device) - 1.0) * _sc
                self.kd_factors[env_ids] = 1.0 + (torch_rand_float(kd_rng[0], kd_rng[1], (len(env_ids), self.num_actions), device=self.device) - 1.0) * _sc
            if getattr(self.cfg.domain_rand, "randomize_motor_offset", False):
                off_rng = self.cfg.domain_rand.motor_offset_range
                self.motor_offsets[env_ids] = torch_rand_float(off_rng[0], off_rng[1], (len(env_ids), self.num_actions), device=self.device) * _sc
            # Restitution is a creation-time rigid-shape property, so re-apply it per reset env scaled by dr_scale (0 => fully inelastic, so cold-start wall contacts don't rebound the plant).
            if getattr(self.cfg.domain_rand, "randomize_restitution", False):
                r_lo, r_hi = self.cfg.domain_rand.restitution_range
                _eids = env_ids.tolist() if torch.is_tensor(env_ids) else list(env_ids)
                _rest = ((r_lo + (r_hi - r_lo) * torch.rand(len(_eids), device=self.device)) * self.dr_scale[env_ids]).tolist()
                for _k, _eid in enumerate(_eids):
                    _props = self.gym.get_actor_rigid_shape_properties(self.envs[_eid], self.actor_handles[_eid])
                    for _p in _props:
                        _p.restitution = _rest[_k]
                    self.gym.set_actor_rigid_shape_properties(self.envs[_eid], self.actor_handles[_eid], _props)

        # Randomize only the observed contact threshold; rewards use clean forces.
        if getattr(self.cfg.domain_rand, "randomize_contact_threshold", False):
            t_lo, t_hi = self.cfg.domain_rand.contact_threshold_range
            _t = torch_rand_float(t_lo, t_hi, (len(env_ids), len(self.feet_indices)), device=self.device)
            if getattr(self.cfg.domain_rand, "dr_lock_to_terrain_level", False):
                _t = 2. + (_t - 2.) * self.dr_scale[env_ids].unsqueeze(-1)
            self.contact_thresh[env_ids] = _t
            self.last_contacts_obs[env_ids] = False

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_root_vel[:] = 0.
        self.reset_buf[env_ids] = 1
        # Retain the most recent termination reason for rendering and diagnostics.
        if not hasattr(self, 'last_term_code'):
            self.last_term_code = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        if hasattr(self, 'term_roll'):
            _c = torch.full((len(env_ids),), 6, dtype=torch.long, device=self.device)  # 6 = timeout
            _c[self.term_reach_goal[env_ids]]     = 0
            _c[self.term_premature_wall[env_ids]] = 5
            _c[self.term_height[env_ids]]         = 3
            _c[self.term_pitch[env_ids]]          = 2
            _c[self.term_base_contact[env_ids]]   = 4
            _c[self.term_yaw_tracking[env_ids]]   = 7
            _c[self.term_plant_dwell[env_ids]]    = 8
            _c[self.term_lateral[env_ids]]        = 9
            _c[self.term_wococo_stage1_ground[env_ids]]    = 10
            _c[self.term_wococo_stage2_overshoot[env_ids]] = 11
            _c[self.term_roll[env_ids]]           = 1   # roll assigned last → wins ties (wall-plant cause)
            self.last_term_code[env_ids] = _c
        self.obs_history_buf[env_ids, :, :] = 0.  # reset obs history buffer TODO no 0s
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        if getattr(self.cfg.domain_rand, "randomize_action_latency", False):
            lo, hi = self.cfg.domain_rand.action_latency_range
            if getattr(self.cfg.domain_rand, "dr_lock_to_terrain_level", False):
                # Harder terrain → larger max latency. Sample uniform in [lo, hi_env].
                hi_env = (self.dr_scale[env_ids] * hi).round().long().clamp(int(lo), int(hi))
                span = (hi_env - int(lo) + 1).float()
                self.action_delay_steps[env_ids] = int(lo) + (torch.rand(len(env_ids), device=self.device) * span).long()
            else:
                self.action_delay_steps[env_ids] = torch.randint(int(lo), int(hi) + 1, (len(env_ids),), device=self.device)

        # Resample the per-env depth-noise scale each episode (uniform in [noise_scale_min, 1]) so an env sees varying noise over training. Drawn BEFORE the camera-latency draw that rides it, and before the own-body gate.
        # depth_noise_scale_pinned (play.py --depth_noise_scale, render scripts) suppresses the redraw; without it an eval pin drifted back to the training spread after the env's first reset.
        if hasattr(self, "depth_noise_scale") and not getattr(self, "depth_noise_scale_pinned", False):
            smin = float(getattr(self.cfg.depth, "noise_scale_min", 0.0))
            self.depth_noise_scale[env_ids] = (
                torch.rand(len(env_ids), device=self.device) * (1.0 - smin) + smin)
        if hasattr(self, "depth_self_geom_on"):
            # Own-body rendering ramps in as a per-EPISODE Bernoulli draw with p = phys, not per-pixel (a leg either occludes or it does not): p=0 at s=0, p=1 from s=0.5 up.
            p = torch.clamp(2.0 * self.depth_noise_scale[env_ids], 0.0, 1.0)
            self.depth_self_geom_on[env_ids] = torch.rand(len(env_ids), device=self.device) < p

        # Sensor latency must ramp from zero even where general DR keeps a nonzero floor (Go2 uses dr_scale_range=[0.3, 1.0]), so convert DR strength back into normalized curriculum progress. The cliffside range [0, 1] is unchanged.
        latency_ramp = None
        if getattr(self.cfg.domain_rand, "dr_lock_to_terrain_level", False):
            if self._dr_fleet_latency_ramp is not None:
                # Fleet-ceiling ramp: identical for every env, so the per-env dr_scale_sample_per_env draw does not halve the trained latency mean -- transport lag is a pipeline property shared by all units. Dynamics/noise/push DR keep the per-env draw on purpose.
                latency_ramp = torch.full((len(env_ids),), self._dr_fleet_latency_ramp,
                                          device=self.device)
            else:
                s_lo, s_hi = self.cfg.domain_rand.dr_scale_range
                if s_hi > s_lo:
                    latency_ramp = ((self.dr_scale[env_ids] - s_lo) / (s_hi - s_lo)).clamp(0., 1.)
                else:
                    latency_ramp = torch.ones(len(env_ids), device=self.device)

        if getattr(self.cfg.domain_rand, "randomize_proprio_latency", False):
            p_lo, p_hi = self.cfg.domain_rand.proprio_latency_range_s
            if latency_ramp is not None:
                # Ramp the per-env upper bound with terrain difficulty. With the Go2 range [0, 20 ms], progress=0 gives zero lag and progress=1 the full range.
                p_hi_env = p_lo + latency_ramp * (p_hi - p_lo)
                latency_s = p_lo + torch.rand(len(env_ids), device=self.device) * (p_hi_env - p_lo)
            else:
                latency_s = torch.rand(len(env_ids), device=self.device) * (p_hi - p_lo) + p_lo
            lag_steps = latency_s / self.dt
            self.proprio_lag_floor[env_ids] = lag_steps.floor().long()
            self.proprio_lag_frac[env_ids] = lag_steps - lag_steps.floor()
        # Resample per-env camera latency each episode (real lag drifts run to run).
        if self.cfg.depth.use_camera and self.depth_latency_max > 0:
            lat_lo, lat_hi = (int(min(self.cfg.depth.latency_range)),
                              int(max(self.cfg.depth.latency_range)))
            if not getattr(self.cfg.depth, "latency_ride_dr_ramp", True):
                # Deploy-matched fixed range from step 0 (no maturity ramp): latency_range mirrors the enforced 80 ms exposure->latent pad, so even the first DAgger batches must not see fresher depth than the hardware pipeline delivers.
                self.depth_latency[env_ids] = torch.randint(
                    lat_lo, lat_hi + 1, (len(env_ids),), device=self.device)
            elif latency_ramp is not None:
                # Depth lag is quantized in camera-capture frames. Ramp BOTH inclusive bounds from 0 toward [lat_lo, lat_hi], so a nonzero lat_lo becomes a deploy-centering FLOOR at full DR while a cold fleet still starts at exactly zero lag (reduces to ramping the upper bound only when lat_lo == 0).
                lo_env = (latency_ramp * lat_lo).round().long().clamp_(0, lat_lo)
                hi_env = (latency_ramp * lat_hi).round().long().clamp_(0, lat_hi)
                hi_env = torch.maximum(hi_env, lo_env)
                span = (hi_env - lo_env + 1).float()
                self.depth_latency[env_ids] = lo_env + (torch.rand(len(env_ids), device=self.device) * span).long()
            else:
                self.depth_latency[env_ids] = torch.randint(lat_lo, lat_hi + 1,
                                                            (len(env_ids),), device=self.device)
            # Keep the legacy latency scaling opt-in per config: some deployments need transport latency independently of the depth-noise curriculum.
            if (getattr(self.cfg.depth, "latency_scale_with_noise", True)
                    and hasattr(self, "depth_noise_scale")):
                _p = torch.clamp(2.0 * self.depth_noise_scale[env_ids], 0.0, 1.0)
                self.depth_latency[env_ids] = (
                    self.depth_latency[env_ids].float() * _p).round().long()
        # Resample blind-spot side + slope each episode (which imager edge is blind varies env-to-env); the width re-seeds and then drifts during the episode.
        if self.cfg.depth.use_camera and hasattr(self, "pw_bs_side"):
            n = len(env_ids)
            slope_max = float(getattr(self.cfg.depth, "pw_blindspot_slope_max", 0.15))
            self.pw_bs_side[env_ids] = self._sample_blindspot_side(n)
            self.pw_bs_slope[env_ids] = (torch.rand(n, device=self.device) * 2.0 - 1.0) * slope_max
            self.pw_bs_k[env_ids] = torch.rand(n, device=self.device) * int(self.cfg.depth.pw_blindspot_cols)
        # Snapshot BEFORE the zeroing below -- this is the curriculum's own promote/demote criterion and the logging block needs the value the finished episode reached.
        max_goal_idx_pre_reset = self.max_goal_idx_ep[env_ids].clone()
        self.cur_goal_idx[env_ids] = 0
        self.max_goal_idx_ep[env_ids] = 0
        self.reach_goal_timer[env_ids] = 0
        self.final_goal_reached[env_ids] = False
        self.final_goal_stop_steps[env_ids] = 0
        self.wall_g1_time[env_ids] = 1e9
        self.wall_g3_rewarded[env_ids] = False
        self.yaw_penalty_latched[env_ids] = False
        self.yaw_prev[env_ids] = 0.0  # spawn yaw is 0 (facing +x)
        self.wall_approach_prev_vel[env_ids] = 0.0
        self.wall_push_rewarded[env_ids] = False
        self.wall_align_rewarded_front[env_ids] = False
        self.wall_align_rewarded_back[env_ids]  = False
        self.wall_feet_touch_count[env_ids] = 0
        self.wall_feet_prev_on_wall[env_ids] = False
        self.wall_feet_touched[env_ids] = False
        self.wall_align_earned_once[env_ids] = False
        self.wall_align_steps_since_earn[env_ids] = 0
        # Snapshot stage BEFORE clearing so the logging branch can read the max stage reached during the just-finished episode.
        wococo_stage_pre_reset = self.wococo_contact_stage[env_ids].clone()
        p4l_stage2_steps_pre_reset = self.wococo_p4l_stage2_steps[env_ids].clone()
        p4l_stage2_front_land_ok_pre_reset = self.wococo_p4l_stage2_front_land_ok[env_ids].clone()
        p4l_stage2_land_latched_pre_reset = self.wococo_p4l_stage2_land_latched[env_ids].clone()
        p4l_stage2_rear_air_pre_reset = self.wococo_p4l_stage2_rear_air[env_ids].clone()
        p4l_stage2_roll_ok_pre_reset = self.wococo_p4l_stage2_roll_ok[env_ids].clone()
        p4l_stage2_yaw_ok_pre_reset = self.wococo_p4l_stage2_yaw_ok[env_ids].clone()
        p4l_stage2_hold_ok_pre_reset = self.wococo_p4l_stage2_hold_ok[env_ids].clone()
        p4l_stage2_advance_pre_reset = self.wococo_p4l_stage2_advance[env_ids].clone()
        p4l_stage2_pitch_ok_pre_reset = self.wococo_p4l_stage2_pitch_ok[env_ids].clone()
        p4l_stage2_height_ok_pre_reset = self.wococo_p4l_stage2_height_ok[env_ids].clone()
        p4l_stage2_speed_ok_pre_reset = self.wococo_p4l_stage2_speed_ok[env_ids].clone()
        s0diag_pre_reset = {
            k: getattr(self, f'wococo_s0diag_{k}')[env_ids].clone()
            for k in ('steps', 'near', 'press', 'rear_air', 'roll_ok', 'hold_ok')}
        s1diag_pre_reset = {
            k: getattr(self, f'wococo_s1diag_{k}')[env_ids].clone()
            for k in ('steps', 'near', 'press', 'yaw_ok', 'hold_ok')}
        ever_pre_reset = {
            k: getattr(self, f'wococo_{k}')[env_ids].clone()
            for k in ('s0ever_near', 's1ever_near', 's1ever_yaw')}
        self.wococo_contact_stage[env_ids]       = 0
        self.wococo_stage2_time[env_ids]         = 0
        self.wococo_stage3_time[env_ids]         = 0
        self.wococo_stage4_time[env_ids]         = 0
        self.wococo_sentinel_reached_step[env_ids] = -1
        self.wococo_s0_rear_touched[env_ids]     = False
        self.wococo_foot_target_paid[env_ids]    = False
        self.wococo_landing_foot_target_paid[env_ids] = False
        self.wococo_foot_target_pending[env_ids] = 0.
        self.wococo_foot_target_prev_stage[env_ids] = 0
        self.wococo_plant_dwell[env_ids]         = 0
        self.wococo_plant_dwell_stage[env_ids]   = 0
        self.wococo_base_contact_streak[env_ids] = 0
        self.wococo_corr_paid[env_ids]           = False
        self.wococo_armed[env_ids]         = False
        self.wococo_stage_hold_counter[env_ids]  = 0
        self.wococo_p4l_stage2_steps[env_ids]         = 0.
        self.wococo_p4l_stage2_front_land_ok[env_ids] = 0.
        self.wococo_p4l_stage2_land_latched[env_ids]  = 0.
        self.wococo_p4l_land_latch[env_ids]     = False
        self.wococo_p4l_land_latch_age[env_ids] = 0
        self.wococo_p4l_stage2_rear_air[env_ids]      = 0.
        self.wococo_p4l_stage2_roll_ok[env_ids]       = 0.
        self.wococo_p4l_stage2_yaw_ok[env_ids]        = 0.
        self.wococo_p4l_stage2_hold_ok[env_ids]       = 0.
        self.wococo_p4l_stage2_advance[env_ids]       = 0.
        self.wococo_p4l_stage2_pitch_ok[env_ids]      = 0.
        self.wococo_p4l_stage2_height_ok[env_ids]     = 0.
        self.wococo_p4l_stage2_speed_ok[env_ids]      = 0.
        for _k in ('steps', 'near', 'press', 'rear_air', 'roll_ok', 'hold_ok'):
            getattr(self, f'wococo_s0diag_{_k}')[env_ids] = 0.
        for _k in ('steps', 'near', 'press', 'yaw_ok', 'hold_ok'):
            getattr(self, f'wococo_s1diag_{_k}')[env_ids] = 0.
        for _k in ('s0ever_near', 's1ever_near', 's1ever_yaw'):
            getattr(self, f'wococo_{_k}')[env_ids] = False
        # Sticky WoCoCo unlock telemetry; the historic "stage4" name meant "plant complete" (buffer kept for compatibility, polished names emitted below).
        reached_plant_complete = wococo_stage_pre_reset >= self._wococo_plant_complete_stage
        self.wococo_stage4_unlocked[env_ids] = (
            self.wococo_stage4_unlocked[env_ids] | reached_plant_complete)
        reached_sentinel = wococo_stage_pre_reset >= self._wococo_sentinel_stage
        self.wococo_sentinel_unlocked[env_ids] = (
            self.wococo_sentinel_unlocked[env_ids] | reached_sentinel)
        for _stage_idx in range(1, self.wococo_stage_unlocked.shape[1]):
            self.wococo_stage_unlocked[env_ids, _stage_idx] |= (
                wococo_stage_pre_reset >= _stage_idx)
        self.wococo_episode_max_stage[env_ids]   = 0
        self.wococo_income_epsum[env_ids]        = 0.
        self.wococo_hurdle_stage[env_ids]              = 0
        self.wococo_hurdle_idx[env_ids]                = 0
        self.wococo_hurdle_episode_max_stage[env_ids]  = 0
        self.wococo_hurdle_s1_time[env_ids]            = 0
        self.wococo_hurdle_n_wrong_armed[env_ids]      = False
        # Invalidate the cached hip-pair x so the first post-reset step cannot read a stale value and register a phantom crossing.
        self.hurdle_sym_prev_valid[env_ids]            = False
        self.hurdle_sym_fired[env_ids]                 = False
        # A fresh episode starting in the timed initial standstill must ramp in from 1.0, not inherit the previous episode's held count.
        self.standstill_hold_steps[env_ids]            = 0

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            log_key = 'rew_' + key
            if key.startswith('wococo_'):
                log_key = 'wococo/rew_' + key[len('wococo_'):]
            self.extras["episode"][log_key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.episode_length_buf[env_ids] = 0

        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
            if getattr(self.cfg.rewards,
                       'wococo_stage_anneal_start_mean_level', None) is not None:
                self.extras["episode"]["stage_anneal_frac"] = torch.tensor(
                    getattr(self, '_wococo_stage_anneal_latched', 1.0),
                    device=self.device)
            # DR ramp telemetry: mean = avg per-env DR multiplier applied, max ~= current ramp ceiling, armed = the cold-start latch.
            self.extras["episode"]["dr_scale_mean"]  = torch.mean(self.dr_scale)
            self.extras["episode"]["dr_scale_max"]   = torch.max(self.dr_scale)
            self.extras["episode"]["dr_ramp_armed"]  = torch.tensor(float(self.dr_ramp_armed), device=self.device)
            # env19-specific keys must be emitted on EVERY env-step where
            # extras["episode"] is populated; the PPO runner aggregates across
            # steps and KeyErrors on absent keys (rsl_rl/.../on_policy_runner.py:404).
            # Sentinel 0.0 when no env19 envs are present (won't pollute the
            # average for mixed runs since env19 is always part of the env mix).
            if hasattr(self, 'env_flyball_wall_angle_deg'):
                wall_envs = (self.env_class == 19) | (self.env_class == 15)
                self.extras["episode"]["wall_angle_deg"] = (
                    self.env_flyball_wall_angle_deg[wall_envs].mean().item()
                    if wall_envs.any() else 0.0)
        # WoCoCo stage at reset == max stage reached (stages only increment); sentinel 0.0 when no wococo-eligible envs reset this step.
        wococo_reset = (self.env_class[env_ids] == 19) | (self.env_class[env_ids] == 15)
        if wococo_reset.any():
            stage_at_reset = wococo_stage_pre_reset[wococo_reset].float()
            self.extras["episode"]["wococo/stage_max"] = stage_at_reset.max()
            p4l_denom = p4l_stage2_steps_pre_reset[wococo_reset].clamp(min=1.0)
            self.extras["episode"]["wococo/p4l_stage2_active_frac"] = (
                p4l_stage2_steps_pre_reset[wococo_reset] > 0).float().mean()
            self.extras["episode"]["wococo/p4l_stage2_front_land_ok_frac"] = (
                p4l_stage2_front_land_ok_pre_reset[wococo_reset] / p4l_denom).mean()
            self.extras["episode"]["wococo/p4l_stage2_land_latched_frac"] = (
                p4l_stage2_land_latched_pre_reset[wococo_reset] / p4l_denom).mean()
            self.extras["episode"]["wococo/p4l_stage2_rear_air_frac"] = (
                p4l_stage2_rear_air_pre_reset[wococo_reset] / p4l_denom).mean()
            self.extras["episode"]["wococo/p4l_stage2_roll_ok_frac"] = (
                p4l_stage2_roll_ok_pre_reset[wococo_reset] / p4l_denom).mean()
            self.extras["episode"]["wococo/p4l_stage2_yaw_ok_frac"] = (
                p4l_stage2_yaw_ok_pre_reset[wococo_reset] / p4l_denom).mean()
            self.extras["episode"]["wococo/p4l_stage2_hold_ok_frac"] = (
                p4l_stage2_hold_ok_pre_reset[wococo_reset] / p4l_denom).mean()
            self.extras["episode"]["wococo/p4l_stage2_advance_frac"] = (
                p4l_stage2_advance_pre_reset[wococo_reset] / p4l_denom).mean()
            # Posture-only landing gate occupancy: zero unless wococo_landing_posture_only is set, in which case these name the binding conjunct.
            self.extras["episode"]["wococo/p4l_stage2_pitch_ok_frac"] = (
                p4l_stage2_pitch_ok_pre_reset[wococo_reset] / p4l_denom).mean()
            self.extras["episode"]["wococo/p4l_stage2_height_ok_frac"] = (
                p4l_stage2_height_ok_pre_reset[wococo_reset] / p4l_denom).mean()
            self.extras["episode"]["wococo/p4l_stage2_speed_ok_frac"] = (
                p4l_stage2_speed_ok_pre_reset[wococo_reset] / p4l_denom).mean()
            # Stage-0 / stage-1 gate occupancy: *_active_frac is the share of resetting envs that entered the stage; the rest are shares of in-stage steps with the named conjunct satisfied, so the smallest names the binding gate.
            for _pfx, _dct, _keys in (
                    ('s0', s0diag_pre_reset, ('near', 'press', 'rear_air', 'roll_ok', 'hold_ok')),
                    ('s1', s1diag_pre_reset, ('near', 'press', 'yaw_ok', 'hold_ok'))):
                _steps = _dct['steps'][wococo_reset]
                _den = _steps.clamp(min=1.0)
                self.extras["episode"][f"wococo/{_pfx}_active_frac"] = (_steps > 0).float().mean()
                for _k in _keys:
                    self.extras["episode"][f"wococo/{_pfx}_{_k}_frac"] = (
                        _dct[_k][wococo_reset] / _den).mean()
            for _k, _v in ever_pre_reset.items():
                self.extras["episode"][f"wococo/{_k}"] = _v[wococo_reset].float().mean()
            self.extras["episode"]["wococo/max_goal_idx"] = (
                max_goal_idx_pre_reset[wococo_reset].float().mean())
        else:
            self.extras["episode"]["wococo/stage_max"] = torch.tensor(0.0, device=self.device)
            self.extras["episode"]["wococo/p4l_stage2_active_frac"] = torch.tensor(0.0, device=self.device)
            self.extras["episode"]["wococo/p4l_stage2_front_land_ok_frac"] = torch.tensor(0.0, device=self.device)
            self.extras["episode"]["wococo/p4l_stage2_land_latched_frac"] = torch.tensor(0.0, device=self.device)
            self.extras["episode"]["wococo/p4l_stage2_rear_air_frac"] = torch.tensor(0.0, device=self.device)
            self.extras["episode"]["wococo/p4l_stage2_roll_ok_frac"] = torch.tensor(0.0, device=self.device)
            self.extras["episode"]["wococo/p4l_stage2_yaw_ok_frac"] = torch.tensor(0.0, device=self.device)
            self.extras["episode"]["wococo/p4l_stage2_hold_ok_frac"] = torch.tensor(0.0, device=self.device)
            self.extras["episode"]["wococo/p4l_stage2_advance_frac"] = torch.tensor(0.0, device=self.device)
            self.extras["episode"]["wococo/p4l_stage2_pitch_ok_frac"] = torch.tensor(0.0, device=self.device)
            self.extras["episode"]["wococo/p4l_stage2_height_ok_frac"] = torch.tensor(0.0, device=self.device)
            self.extras["episode"]["wococo/p4l_stage2_speed_ok_frac"] = torch.tensor(0.0, device=self.device)
            for _pfx, _keys in (
                    ('s0', ('near', 'press', 'rear_air', 'roll_ok', 'hold_ok')),
                    ('s1', ('near', 'press', 'yaw_ok', 'hold_ok'))):
                self.extras["episode"][f"wococo/{_pfx}_active_frac"] = torch.tensor(0.0, device=self.device)
                for _k in _keys:
                    self.extras["episode"][f"wococo/{_pfx}_{_k}_frac"] = torch.tensor(0.0, device=self.device)
            for _k in ('s0ever_near', 's1ever_near', 's1ever_yaw', 'max_goal_idx'):
                self.extras["episode"][f"wococo/{_k}"] = torch.tensor(0.0, device=self.device)
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        self.extras["episode"]["wococo/plant_complete_unlocked_frac"] = (
            self.wococo_stage4_unlocked.float().mean())
        self.extras["episode"]["wococo/sentinel_unlocked_frac"] = (
            self.wococo_sentinel_unlocked.float().mean())
        for _stage_idx in range(1, self.wococo_stage_unlocked.shape[1]):
            self.extras["episode"][f"wococo/stage_{_stage_idx}_unlocked_frac"] = (
                self.wococo_stage_unlocked[:, _stage_idx].float().mean())

        # Per-cause termination rates among envs resetting this step (diagnoses short-episode pathologies).
        for key in ("term_roll", "term_pitch", "term_height",
                    "term_base_contact", "term_reach_goal", "term_timeout",
                    "term_yaw_tracking", "term_plant_dwell",
                    "term_lateral"):
            if hasattr(self, key):
                self.extras["episode"][key] = getattr(self, key)[env_ids].float().mean()
        # Plant-failure early terminations, with the curriculum arm-latch state (0 until the fleet reaches the arm level).
        if hasattr(self, 'term_wococo_stage1_ground'):
            self.extras["episode"]["wococo/term_stage1_ground"] = \
                self.term_wococo_stage1_ground[env_ids].float().mean()
            self.extras["episode"]["wococo/term_stage2_overshoot"] = \
                self.term_wococo_stage2_overshoot[env_ids].float().mean()
            self.extras["episode"]["wococo/plant_fail_term_armed"] = \
                torch.tensor(float(self.wococo_plant_fail_term_armed), device=self.device)
            self.extras["episode"]["wococo/plant_fail_alpha"] = \
                torch.tensor(float(getattr(self, 'wococo_plant_fail_alpha', 1.0)), device=self.device)
        # Armed-window termination tolerance: sticky trigger, effective alpha, and the fraction of resetting envs inside the window.
        if hasattr(self, 'wococo_armed_term_exempt'):
            self.extras["episode"]["wococo/armed_term_exempt_armed"] = \
                torch.tensor(float(self.wococo_armed_term_exempt_armed), device=self.device)
            self.extras["episode"]["wococo/term_tolerance_alpha"] = \
                torch.tensor(float(self.wococo_armed_term_exempt_alpha), device=self.device)
            self.extras["episode"]["wococo/armed_term_exempt"] = \
                self.wococo_armed_term_exempt[env_ids].float().mean()
        # Hurdle-clearance scale ramp: armed is the sticky trigger, alpha the fraction of the configured reward scale applied.
        self.extras["episode"]["hurdle_clearance/armed"] = \
            torch.tensor(float(self.hurdle_clearance_armed), device=self.device)
        self.extras["episode"]["hurdle_clearance/alpha"] = \
            torch.tensor(float(self.hurdle_clearance_alpha), device=self.device)
        if getattr(self.cfg.rewards, "hurdle_clearance_per_env_draw", False):
            self.extras["episode"]["hurdle_clearance/active_frac"] = self.hurdle_clearance_env_enabled[env_ids].float().mean()
        # Action-delay latch state, so the iteration the 1-step (20 ms) delay engages is visible in wandb; the trigger's progress is logged too, so a run that has not latched still shows how far off it is.
        self.extras["episode"]["action_delay/latched"] = \
            torch.tensor(float(self.action_delay_latched), device=self.device)
        if getattr(self.cfg.domain_rand, "action_delay_ramp_steps", 0) > 0:
            self.extras["episode"]["action_delay/active_frac"] = self.action_delay_episode_enabled[env_ids].float().mean()
        _latch_step = getattr(self.cfg.domain_rand, "action_delay_latch_step", None)
        if _latch_step is not None:
            self.extras["episode"]["action_delay/latch_progress"] = torch.tensor(
                min(1.0, max(0.0, (self.common_step_counter - int(_latch_step)) / float(max(int(getattr(self.cfg.domain_rand, "action_delay_ramp_steps", 0)), 1)))),
                device=self.device)
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        # Update the canonical-WoCoCo plant-arm latch ONCE per step before any reward reads it; reward call order is not guaranteed, so the latch can't live in a reward term.
        if hasattr(self, 'wococo_armed'):
            self._wococo_arm()
        # Standstill posture-lock ramp: counts consecutive steps commanded to stand still so _reward_dof_error eases its penalty boost in over standstill_dof_error_ramp_s.
        # Updated here rather than in _post_physics_step_callback because the post-circuit hard stop zeroes the command later inside check_termination -- by this point self.commands is final.
        if hasattr(self, 'standstill_hold_steps'):
            _ss_now = torch.abs(self.commands[:, 0]) < self.cfg.commands.lin_vel_clip
            self.standstill_hold_steps = torch.where(
                _ss_now, self.standstill_hold_steps + 1,
                torch.zeros_like(self.standstill_hold_steps))
        # Reward functions run alphabetically and contact_pts can advance the landing FSM before the force terms, so snapshot stage 2 to keep the touchdown step penalized.
        if ('wococo_landing_z_force' in self.reward_scales
                or 'wococo_landing_positive_x_force' in self.reward_scales):
            self._wococo_force_reward_stage = self.wococo_contact_stage.clone()
        # exempt_wococo_contact_pts_from_clip defers wococo_contact_pts past the only_positive_rewards clip (same pattern as 'termination'), letting the n_wrong penalty drive total reward negative.
        exempt_wococo = (
            bool(getattr(self.cfg.rewards, 'exempt_wococo_contact_pts_from_clip', False))
            and 'wococo_contact_pts' in self.reward_scales
        )
        # exempt_collision_from_clip: same pattern for the collision penalty, for when collision is the dominant shaping signal.
        exempt_collision = (
            bool(getattr(self.cfg.rewards, 'exempt_collision_from_clip', False))
            and 'collision' in self.reward_scales
        )
        exempt_wococo_hurdle = (
            bool(getattr(self.cfg.rewards, 'exempt_wococo_hurdle_contact_pts_from_clip', False))
            and 'wococo_hurdle_contact_pts' in self.reward_scales
        )
        exempt_wall_multi_touch = (
            bool(getattr(self.cfg.rewards, 'exempt_wall_multi_touch_from_clip', False))
            and 'wall_multi_touch' in self.reward_scales
        )
        # exempt_hurdle_contact_from_clip: same pattern for the per-step hurdle-contact penalty.
        exempt_hurdle_contact = (
            bool(getattr(self.cfg.rewards, 'exempt_hurdle_contact_from_clip', False))
            and 'hurdle_contact' in self.reward_scales
        )
        # Same pattern for the AABB clearance penalty: it arms at a high fleet mean, but demoted envs can sit below only_positive_rewards_clip_below_level where the clip would swallow it.
        exempt_hurdle_clearance = (
            bool(getattr(self.cfg.rewards, 'exempt_hurdle_clearance_from_clip', False))
            and 'hurdle_clearance' in self.reward_scales
        )
        # Same pattern for the point-symmetric pair-symmetry score, the only form of that term that can go negative.
        exempt_hurdle_pair_sym = (
            bool(getattr(self.cfg.rewards, 'exempt_hurdle_pair_sym_from_clip', False))
            and 'hurdle_pair_sym' in self.reward_scales
        )
        deferred_hurdle_pair_sym_rew = None
        deferred_wococo_rew = None
        deferred_collision_rew = None
        deferred_wococo_hurdle_rew = None
        deferred_wall_multi_touch_rew = None
        deferred_hurdle_contact_rew = None
        deferred_hurdle_clearance_rew = None

        self.rew_buf[:] = 0.
        self.last_step_rews = {}
        # Cap positive sub-sentinel shaping so repeated tapping cannot farm reward.
        wococo_income_cap = getattr(self.cfg.rewards, 'wococo_income_episode_cap', None)
        _capped_income_terms = ('wococo_contact_pts', 'wococo_foot_track', 'wococo_base_orient')
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            if wococo_income_cap is not None and name in _capped_income_terms:
                headroom = (float(wococo_income_cap) - self.wococo_income_epsum).clamp(min=0.)
                pos = rew.clamp(min=0.)
                allowed = torch.minimum(pos, headroom)
                rew = rew - (pos - allowed)
                self.wococo_income_epsum += allowed
            self.episode_sums[name] += rew
            # Split the velocity-command tracking reward into standstill-carrot and moving streams for wandb; they sum exactly to rew_tracking_goal_vel and are never added to rew_buf.
            if name == 'tracking_goal_vel' and hasattr(self, '_track_goal_vel_standstill'):
                _ss = self._track_goal_vel_standstill.float()
                self.episode_sums['tracking_goal_vel_standstill'] += rew * _ss
                self.episode_sums['tracking_goal_vel_moving']     += rew * (1.0 - _ss)
            self.last_step_rews[name] = rew
            if exempt_wococo and name == 'wococo_contact_pts':
                deferred_wococo_rew = rew
                continue
            if exempt_collision and name == 'collision':
                deferred_collision_rew = rew
                continue
            if exempt_wococo_hurdle and name == 'wococo_hurdle_contact_pts':
                deferred_wococo_hurdle_rew = rew
                continue
            if exempt_wall_multi_touch and name == 'wall_multi_touch':
                deferred_wall_multi_touch_rew = rew
                continue
            if exempt_hurdle_contact and name == 'hurdle_contact':
                deferred_hurdle_contact_rew = rew
                continue
            if exempt_hurdle_clearance and name == 'hurdle_clearance':
                deferred_hurdle_clearance_rew = rew
                continue
            if exempt_hurdle_pair_sym and name == 'hurdle_pair_sym':
                deferred_hurdle_pair_sym_rew = rew
                continue
            self.rew_buf += rew
        if self.cfg.rewards.only_positive_rewards:
            if getattr(self.cfg.rewards, 'only_positive_rewards_exempt_high_levels', False):
                # Preserve penalty gradients above the configured curriculum level.
                _clip_below = int(getattr(self.cfg.rewards, 'only_positive_rewards_clip_below_level', 2))
                _clip_mask = self.terrain_levels < _clip_below
                self.rew_buf[_clip_mask] = torch.clip(self.rew_buf[_clip_mask], min=0.)
            else:
                # Default: clip negative totals to zero at ALL terrain levels.
                self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)

        if deferred_wococo_rew is not None:
            self.rew_buf += deferred_wococo_rew
        if deferred_collision_rew is not None:
            self.rew_buf += deferred_collision_rew
        if deferred_wococo_hurdle_rew is not None:
            self.rew_buf += deferred_wococo_hurdle_rew
        if deferred_wall_multi_touch_rew is not None:
            self.rew_buf += deferred_wall_multi_touch_rew
        if deferred_hurdle_contact_rew is not None:
            self.rew_buf += deferred_hurdle_contact_rew
        if deferred_hurdle_clearance_rew is not None:
            self.rew_buf += deferred_hurdle_clearance_rew
        if deferred_hurdle_pair_sym_rew is not None:
            self.rew_buf += deferred_hurdle_pair_sym_rew

        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        # Same treatment for the soft terminal price (see _reward_timeout): added past the clip so it bites on low rows, where an unpriced timeout is the cheapest way to end an episode.
        if "timeout" in self.reward_scales:
            rew = self._reward_timeout() * self.reward_scales["timeout"]
            self.rew_buf += rew
            self.episode_sums["timeout"] += rew
    
    def _flat_terrain_mask(self):
        """Boolean (num_envs,) — True where the env's terrain is 'flat' (smooth
        ground the policy should walk level/smoothly over), False on obstacle
        terrain (where vertical motion / tilt are needed to clear obstacles).

        Historically this was hardcoded as env_class == 17 (parkour_flat). With
        a hurdle-curriculum mix there are TWO flat terrains — parkour_flat (17,
        curved corridor) and smooth flat (9, straight walk) — and one obstacle
        terrain (16, hurdles). cfg.terrain.flat_terrain_env_classes lists the
        flat classes; default [17] reproduces the original behaviour exactly for
        every existing task (none of which use env_class 9)."""
        if not hasattr(self, '_flat_classes_t'):
            _cls = getattr(self.cfg.terrain, 'flat_terrain_env_classes', [17])
            self._flat_classes_t = torch.tensor(_cls, device=self.device, dtype=self.env_class.dtype)
        return torch.isin(self.env_class, self._flat_classes_t)

    def _flat_obs_mask(self):
        """Boolean (num_envs,) — flat-vs-obstacle mask for the OBSERVATION terrain
        flags (slots 11/12), kept SEPARATE from _flat_terrain_mask which gates
        reward shaping (lin_vel_z / orientation). Defaults to the reward mask's
        classes, so every existing task is unchanged. Cliffside sets
        flat_obs_env_classes=[] so the flat corridor (env_class 17) presents the
        SAME terrain flags as the wall terrain (19) — the policy can't read the
        flag as a which-lane oracle — while rewards still treat 17 as flat."""
        if not hasattr(self, '_flat_obs_classes_t'):
            _default = getattr(self.cfg.terrain, 'flat_terrain_env_classes', [17])
            _cls = getattr(self.cfg.terrain, 'flat_obs_env_classes', _default)
            self._flat_obs_classes_t = torch.tensor(_cls, device=self.device, dtype=self.env_class.dtype)
        return torch.isin(self.env_class, self._flat_obs_classes_t)

    def _get_proprio_noise_scale_vec(self):
        """Per-dimension noise magnitude for the proprioceptive obs_buf, laid
        out to match its concatenation order. Privileged/derived terms
        (commands, terrain flags, last action) get zero noise; only the real
        on-robot sensors (IMU angular vel, roll/pitch, joint pos/vel) are
        perturbed. Noise is expressed in the same scaled space as the obs."""
        ns = self.cfg.noise.noise_scales
        lvl = self.cfg.noise.noise_level
        s = self.obs_scales
        dev = self.device
        vec = torch.cat([
            torch.ones(3, device=dev) * ns.ang_vel * s.ang_vel,          # base_ang_vel
            torch.ones(2, device=dev) * ns.rotation,                     # imu roll/pitch
            torch.zeros(1, device=dev),                                  # 0*delta_yaw
            torch.zeros(1, device=dev),                                  # delta_yaw (derived)
            torch.zeros(1, device=dev),                                  # delta_next_yaw (derived)
            torch.zeros(2, device=dev),                                  # 0*commands[:2]
            torch.zeros(1, device=dev),                                  # commands[0:1]
            torch.zeros(1, device=dev),                                  # obstacle-terrain flag
            torch.zeros(1, device=dev),                                  # flat-terrain flag
            torch.ones(self.num_dof, device=dev) * ns.dof_pos * s.dof_pos,   # dof_pos
            torch.ones(self.num_dof, device=dev) * ns.dof_vel * s.dof_vel,   # dof_vel
            torch.zeros(self.num_actions, device=dev),                   # last action (policy's own)
            torch.zeros(4, device=dev),                                  # contact_filt
        ]) * lvl
        return vec.unsqueeze(0)

    def _delayed_proprio(self):
        """Push the current *true* proprioceptive readings into per-sensor ring
        buffers and return latency-delayed copies for observation construction.

        Only the observation copy is delayed — self.dof_pos/.dof_vel/.base_ang_vel
        and roll/pitch keep zero-latency ground truth for PD control, rewards and
        terminations. Latency is per-env (sampled in reset_idx) and realized at
        sub-control-step resolution by linearly interpolating between the two
        snapshots bracketing the requested lag. When randomize_proprio_latency is
        off this is a pure pass-through (no buffering cost)."""
        cur = {
            'dof_pos': self.dof_pos,
            'dof_vel': self.dof_vel,
            'ang_vel': self.base_ang_vel,
            'imu':     torch.stack((self.roll, self.pitch), dim=1),
        }
        if not getattr(self.cfg.domain_rand, "randomize_proprio_latency", False):
            return cur

        depth = self._proprio_buf_depth
        # Prime reset envs to prevent proprio samples leaking across episodes.
        fresh = (self.episode_length_buf <= 1)[:, None, None]
        env_idx = torch.arange(self.num_envs, device=self.device)
        newer = (depth - 1 - self.proprio_lag_floor).clamp_(0, depth - 1)
        older = (depth - 2 - self.proprio_lag_floor).clamp_(0, depth - 1)
        frac = self.proprio_lag_frac[:, None]
        out = {}
        for k, c in cur.items():
            b = self._proprio_bufs[k]
            rolled = torch.cat([b[:, 1:], c.unsqueeze(1)], dim=1)
            b = torch.where(fresh, c.unsqueeze(1).expand_as(b), rolled)
            self._proprio_bufs[k] = b
            out[k] = (1.0 - frac) * b[env_idx, newer] + frac * b[env_idx, older]
        return out

    def compute_observations(self):
        """
        Computes observations
        """
        _p = self._delayed_proprio()
        imu_obs = _p['imu']
        if self.global_counter % 5 == 0:
            self.delta_yaw = self.target_yaw - self.yaw
            self.delta_next_yaw = self.next_target_yaw - self.yaw
        obs_buf = torch.cat((#skill_vector,
                            _p['ang_vel']  * self.obs_scales.ang_vel,   #[1,3]
                            imu_obs,    #[1,2]
                            0*self.delta_yaw[:, None], 
                            self.delta_yaw[:, None],
                            self.delta_next_yaw[:, None],
                            0*self.commands[:, 0:2],
                            self.commands[:, 0:1],  #[1,1]
                            (~self._flat_obs_mask()).float()[:, None],   # obstacle-terrain flag
                            (self._flat_obs_mask().float()[:, None]      # flat-terrain flag
                             if getattr(self.cfg.env, "give_flat_terrain_flag", True)
                             else torch.zeros(self.num_envs, 1, device=self.device)),  # gated off -> policy gets no "I'm on flat" bit
                            self.reindex((_p['dof_pos'] - self.default_dof_pos_all) * self.obs_scales.dof_pos),
                            self.reindex(_p['dof_vel'] * self.obs_scales.dof_vel),
                            self.reindex(self.action_history_buf[:, -1]),
                            self.reindex_feet(self.contact_filt_obs.float()-0.5),
                            ),dim=-1)
        if self.cfg.noise.add_noise:
            if not hasattr(self, "noise_scale_vec"):
                self.noise_scale_vec = self._get_proprio_noise_scale_vec()
            noise = (2.0 * torch.rand_like(obs_buf) - 1.0) * self.noise_scale_vec
            if getattr(self.cfg.domain_rand, "dr_lock_to_terrain_level", False):
                noise = noise * self.dr_scale[:, None]
            obs_buf = obs_buf + noise
        priv_explicit = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                   0 * self.base_lin_vel,
                                   0 * self.base_lin_vel), dim=-1)
        priv_latent = torch.cat((
            self.mass_params_tensor,
            self.friction_coeffs_tensor,
            self.motor_strength[0] - 1, 
            self.motor_strength[1] - 1
        ), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.)
            if self.cfg.noise.add_noise:
                h_noise = (2.0 * torch.rand_like(heights) - 1.0) * self.cfg.noise.noise_scales.height_measurements * self.cfg.noise.noise_level
                if getattr(self.cfg.domain_rand, "dr_lock_to_terrain_level", False):
                    h_noise = h_noise * self.dr_scale[:, None]
                heights = heights + h_noise
            self.obs_buf = torch.cat([obs_buf, heights, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        else:
            self.obs_buf = torch.cat([obs_buf, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)

        obs_buf[:, 6:8] = 0  # mask yaw in proprioceptive history
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )

        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([
                self.contact_buf[:, 1:],
                self.contact_filt.float().unsqueeze(1)
            ], dim=1)
        )
        
        
        
    def get_noisy_measurement(self, x, scale):
        if self.cfg.noise.add_noise:
            x = x + (2.0 * torch.rand_like(x) - 1) * scale * self.cfg.noise.noise_level
        return x

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        # Isaac Gym camera sensors need a graphics/Vulkan context, which segfaults on headless compute nodes (exit 139); the Warp renderer raycasts on CUDA with no graphics context, so force one only for the Isaac Gym camera path.
        if self.cfg.depth.use_camera and not self._warp_enabled():
            self.graphics_device_id = self.sim_device_id  # required in headless mode
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        start = time()
        print("*"*80)
        print("Start creating ground...")
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        print("Finished creating ground. Time taken {:.2f} s".format(time() - start))
        print("*"*80)
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        if getattr(self.cfg.domain_rand, "randomize_restitution", False):
            if env_id == 0:
                restitution_range = self.cfg.domain_rand.restitution_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                restitution_buckets = torch_rand_float(restitution_range[0], restitution_range[1], (num_buckets, 1), device='cpu')
                self.restitution_coeffs = restitution_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

        rand_friction = getattr(self.cfg.domain_rand, "randomize_joint_friction", False)
        rand_damping = getattr(self.cfg.domain_rand, "randomize_joint_damping", False)
        if rand_friction or rand_damping:
            if env_id == 0:
                fr = self.cfg.domain_rand.joint_friction_range
                dr = self.cfg.domain_rand.joint_damping_range
                self.joint_friction_coeffs = np.random.uniform(fr[0], fr[1], (self.num_envs, len(props)))
                self.joint_damping_coeffs = np.random.uniform(dr[0], dr[1], (self.num_envs, len(props)))
            for i in range(len(props)):
                if rand_friction:
                    props["friction"][i] = self.joint_friction_coeffs[env_id, i]
                if rand_damping:
                    props["damping"][i] = self.joint_damping_coeffs[env_id, i]
        return props

    def _process_rigid_body_props(self, props, env_id):
        # No need to use tensors as only called upon env creation
        if self.cfg.domain_rand.randomize_base_mass:
            # base_mass_range (multiplicative, centred on the robot's ACTUAL mass) takes precedence over added_mass_range (additive payload). props[0].mass is the post-collapse_fixed_joints trunk mass, so fixed-joint payloads are inside the nominal being scaled.
            rng_factor = getattr(self.cfg.domain_rand, "base_mass_range", None)
            if rng_factor is not None:
                nominal = float(props[0].mass)
                factor = float(np.random.uniform(rng_factor[0], rng_factor[1]))
                props[0].mass = nominal * factor
                # Report the signed kg deviation, NOT the factor, so the priv_latent mass channel keeps identical units and scale in either mode.
                rand_mass = np.array([nominal * (factor - 1.0)])
            else:
                rng_mass = self.cfg.domain_rand.added_mass_range
                rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
                props[0].mass += rand_mass
        else:
            rand_mass = np.zeros((1, ))
        if self.cfg.domain_rand.randomize_base_com:
            rng_com = self.cfg.domain_rand.added_com_range
            rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3, ))
            props[0].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros(3)
        # Per-link mass scaling on every non-base link (base/trunk = props[0] is handled by the payload randomization above).
        if getattr(self.cfg.domain_rand, "randomize_link_mass", False):
            rng_link = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                props[i].mass *= np.random.uniform(rng_link[0], rng_link[1])
        mass_params = np.concatenate([rand_mass, rand_com])
        return props, mass_params
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # Periodic mid-episode command resampling. With commands.resample_only_on_reset the command is drawn ONCE per episode, so a resample can't flip the intent mid-plant or overwrite the hard stop's zeroed command.
        if not bool(getattr(self.cfg.commands, 'resample_only_on_reset', False)):
            env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0)
            self._resample_commands(env_ids.nonzero(as_tuple=False).flatten())

        # Timed initial standstill (commands.initial_standstill_max_s > 0): zero the command for the episode's first cmd_standstill_steps steps, then restore the sampled approach speed. Envs that finished the circuit are skipped so this never fights the hard stop.
        if float(getattr(self.cfg.commands, 'initial_standstill_max_s', 0.0)) > 0.0:
            owned = self.cmd_standstill_active & ~self.final_goal_reached
            if owned.any():
                in_hold = owned & (self.episode_length_buf < self.cmd_standstill_steps)
                self.commands[:, 0] = torch.where(
                    in_hold, torch.zeros_like(self.commands[:, 0]),
                    torch.where(owned, self.cmd_post_standstill_x, self.commands[:, 0]))

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.8*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
            self.commands[:, 2] *= torch.abs(self.commands[:, 2]) > self.cfg.commands.ang_vel_clip
        
        if self.cfg.terrain.measure_heights:
            if self.global_counter % self.cfg.depth.update_interval == 0:
                self.measured_heights = self._get_heights()
                if getattr(self.cfg.terrain, 'scandot_occlusion_mask', False):
                    self.measured_heights = self._mask_occluded_scandots(self.measured_heights)
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        
    def _gather_cur_goals(self, future=0):
        # Clamp the gather index: during the post-circuit hold cur_goal_idx can sit at num_goals, so (idx+future) must not run past the padded env_goals.
        idx = (self.cur_goal_idx[:, None, None] + future).clamp(max=self.env_goals.shape[1] - 1)
        return self.env_goals.gather(1, idx.expand(-1, -1, self.env_goals.shape[-1])).squeeze(1)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 2] *= torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_clip

        self.commands[env_ids, :2] *= torch.abs(self.commands[env_ids, 0:1]) > self.cfg.commands.lin_vel_clip

        # Standing commands: expose the policy to zero velocity so it doesn't drift in deployment when speed drops to zero (flat-corridor configs use 15%).
        p_zero = getattr(self.cfg.commands, 'zero_command_probability', 0.0)
        if p_zero > 0.0 and len(env_ids) > 0:
            zero_mask = torch.rand(len(env_ids), device=self.device) < p_zero
            self.commands[env_ids[zero_mask], 0] = 0.0

        # Wall lanes sample either a standstill or the configured fast band.
        # NOTE the `wococo_stage > 0` term is a legacy PROXY for "this is a wall task": a config that strips the dense shaping but wants the wall COMMAND regime loses the bimodal sampler silently and falls through to the generic U[ranges.lin_vel_x] draw (observed: continuous commands like 0.38 m/s, above lin_vel_clip=0.2).
        # `wococo_bimodal_cmd_force` is an explicit opt-in that does not depend on a reward magnitude; it defaults False, so no existing task changes behaviour; the proxy stays because wococo_bimodal_cmd DEFAULTS TRUE, and dropping it would switch on bimodal commands for every task that never set it.
        _bimodal_force = bool(getattr(self.cfg.commands, 'wococo_bimodal_cmd_force', False))
        if (getattr(self.cfg.commands, 'wococo_bimodal_cmd', True)
                and (float(getattr(self.cfg.rewards.scales, 'wococo_stage', 0.0)) > 0
                     or _bimodal_force)
                and getattr(self, 'custom_origins', False)
                and len(env_ids) > 0):
            # Which env classes use the bimodal sampler; config-driven (default [19,15] = the wall terrains), so cliffside can add parkour_flat (17) without touching other tasks.
            if not hasattr(self, '_wococo_bimodal_classes_t'):
                _bc = getattr(self.cfg.commands, 'wococo_bimodal_env_classes', [19, 15])
                self._wococo_bimodal_classes_t = torch.tensor(_bc, device=self.device, dtype=self.env_class.dtype)
            wococo_envs = torch.isin(self.env_class[env_ids], self._wococo_bimodal_classes_t)
            if wococo_envs.any():
                ids = env_ids[wococo_envs]
                n = int(wococo_envs.sum())
                max_lo = self.cfg.commands.max_ranges.lin_vel_x[0]
                max_hi = self.cfg.commands.max_ranges.lin_vel_x[1]
                fast = torch_rand_float(max_lo, max_hi, (n, 1), device=self.device).squeeze(1)
                # Optional FAST-COMMAND MIX. max_ranges.lin_vel_x is a degenerate [v, v] draw for the wall lanes, so widening it to [1.0, 1.5] would give a uniform spread rather than the intended 50/50 split.
                # Instead, with probability wococo_cmd_fast_prob the sampled approach speed is REPLACED by wococo_cmd_fast_speed, applied before the standstill prefix and the zero draw. Default prob 0.0.
                _fast_prob = float(getattr(self.cfg.commands, 'wococo_cmd_fast_prob', 0.0))
                if _fast_prob > 0.0:
                    _fast_speed = float(getattr(self.cfg.commands, 'wococo_cmd_fast_speed', max_hi))
                    _pick = torch.rand(n, device=self.device) < _fast_prob
                    fast = torch.where(_pick, torch.full_like(fast, _fast_speed), fast)
                hold_max_s = float(getattr(self.cfg.commands, 'initial_standstill_max_s', 0.0))
                if hold_max_s > 0.0:
                    # Timed standstill PREFIX instead of a whole-episode coin flip: hold zero for a per-episode U[0, hold_max_s], then run the sampled speed to the timeout / hard stop, so every episode can finish the circuit and standstill is still trained.
                    # Envs resampled here but NOT in the bimodal classes must release ownership, otherwise a stale prefix would keep driving them.
                    self.cmd_standstill_active[env_ids] = False
                    self.cmd_post_standstill_x[ids] = fast
                    self.cmd_standstill_active[ids] = True
                    self.cmd_standstill_steps[ids] = (
                        torch.rand(n, device=self.device) * (hold_max_s / self.dt)
                    ).long()
                    self.commands[ids, 0] = torch.where(
                        self.cmd_standstill_steps[ids] > 0, torch.zeros_like(fast), fast)
                else:
                    zero_prob = float(getattr(self.cfg.commands, 'wococo_zero_cmd_prob', 0.15))
                    if zero_prob > 0:
                        fast[torch.rand(n, device=self.device) < zero_prob] = 0.0
                    self.commands[ids, 0] = fast

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            # Position error including per-joint motor offset (calibration bias).
            pos_err = actions_scaled + self.default_dof_pos_all - self.dof_pos + self.motor_offsets
            if not self.cfg.domain_rand.randomize_motor:  # TODO add strength to gain directly
                torques = self.kp_factors * self.p_gains * pos_err - self.kd_factors * self.d_gains * self.dof_vel
            else:
                torques = self.motor_strength[0] * self.kp_factors * self.p_gains * pos_err - self.motor_strength[1] * self.kd_factors * self.d_gains * self.dof_vel
                
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(0., 0.9, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if self.cfg.env.randomize_start_pos:
                self.root_states[env_ids, :2] += torch_rand_float(-0.3, 0.3, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            if self.cfg.env.randomize_start_yaw:
                _yaw_scale = (self.terrain_levels[env_ids].float()
                              / max(self.max_terrain_level - 1, 1)).clamp(0., 1.)
                rand_yaw = self.cfg.env.rand_yaw_range * _yaw_scale * torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                if self.cfg.env.randomize_start_pitch:
                    rand_pitch = self.cfg.env.rand_pitch_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                else:
                    rand_pitch = torch.zeros(len(env_ids), device=self.device)
                quat = quat_from_euler_xyz(0*rand_yaw, rand_pitch, rand_yaw) 
                self.root_states[env_ids, 3:7] = quat[:, :]  
            if self.cfg.env.randomize_start_y:
                self.root_states[env_ids, 1] += self.cfg.env.rand_y_range * torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
            
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        # Per-env push strength scaling (1.0 when not locked to terrain level).
        scale = self.dr_scale[:, None] if getattr(self.cfg.domain_rand, "dr_lock_to_terrain_level", False) else 1.0
        # Reset the telemetry vectors so components not pushed this round don't show stale values in the overlay.
        self.last_push_lin.zero_()
        self.last_push_ang.zero_()
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        lin_xy = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) * scale # lin vel x/y
        self.root_states[:, 7:9] = lin_xy
        self.last_push_lin[:, :2] = lin_xy
        max_vel_z = getattr(self.cfg.domain_rand, "max_push_vel_z", 0.0)
        if max_vel_z > 0.0:
            lin_z = torch_rand_float(-max_vel_z, max_vel_z, (self.num_envs, 1), device=self.device) * scale # lin vel z
            self.root_states[:, 9:10] = lin_z
            self.last_push_lin[:, 2:3] = lin_z
        max_ang_vel = getattr(self.cfg.domain_rand, "max_push_ang_vel", 0.0)
        if max_ang_vel > 0.0:
            ang = torch_rand_float(-max_ang_vel, max_ang_vel, (self.num_envs, 3), device=self.device) * scale # ang vel roll/pitch/yaw
            self.root_states[:, 10:13] = ang
            self.last_push_ang[:] = ang
        self.last_push_step[:] = self.common_step_counter
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if not self.init_done:
            # don't change on initial reset
            return
        
        # Per-env_class curriculum criterion. env_class in {15, 19} (wococo wall terrains) use a goal-count rule: promote on >=70% of the choreographed goal sequence, demote below 30%, since the sequence is short and skill-specific.
        # All other terrains (parkour, parkour_flat, ...) use the main-branch distance-vs-commanded-velocity rule; on parkour's 6 stepping stones the goal-count rule is too strict early and cascades the population to level 0.
        ec = self.env_class[env_ids]
        wococo_envs = (ec == 15) | (ec == 19)
        use_goal_rule = wococo_envs
        if bool(getattr(self.cfg.terrain, 'flat_step_goal_curriculum', False)):
            use_goal_rule = wococo_envs | (ec == 17)
        # Short choreographed courses promote by goals instead of travel distance.
        _goal_classes = getattr(self.cfg.terrain, 'goal_curriculum_env_classes', None)
        if _goal_classes:
            for _gc in _goal_classes:
                use_goal_rule = use_goal_rule | (ec == int(_gc))

        # Promote/demote goal fractions are cfg-tunable (kills the band dead zone).
        promote_frac = float(getattr(self.cfg.terrain, 'promote_goal_frac', 0.7))
        demote_frac  = float(getattr(self.cfg.terrain, 'demote_goal_frac', 0.3))
        promote_threshold = int(promote_frac * self.cfg.terrain.num_goals)  # e.g. 5 of 8
        demote_threshold  = int(demote_frac * self.cfg.terrain.num_goals)   # e.g. 2 of 8
        goal_move_up   = self.max_goal_idx_ep[env_ids] >= promote_threshold
        goal_move_down = self.max_goal_idx_ep[env_ids] <  demote_threshold

        dis_to_origin = torch.norm(
            self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        threshold = self.commands[env_ids, 0] * self.cfg.env.episode_length_s
        dist_move_up   = dis_to_origin > 0.8 * threshold
        dist_move_down = dis_to_origin < 0.4 * threshold

        move_up   = torch.where(use_goal_rule, goal_move_up,   dist_move_up)
        move_down = torch.where(use_goal_rule, goal_move_down, dist_move_down)
        # cfg.terrain.no_demotion suppresses move_down so promoted envs never get demoted back. Useful when curriculum cascade-collapse destroys progress (envs forget, all demote, wococo learning resets).
        if bool(getattr(self.cfg.terrain, 'no_demotion', False)):
            move_down = torch.zeros_like(move_down)
        # cfg.terrain.max_promoted_level caps promotion (e.g. 3) so the policy can consolidate skill before being pushed onto walls it cannot plant cleanly.
        max_promoted = getattr(self.cfg.terrain, 'max_promoted_level', None)
        if max_promoted is not None:
            move_up = move_up & (self.terrain_levels[env_ids] < int(max_promoted))

        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # cfg.terrain.min_terrain_level is a hard floor on the row an env may occupy, applied ONLY to envs that actually demoted this call, never as a blanket clamp.
        # play.py's eval sweep assigns terrain_levels directly and calls this method purely to refresh derived buffers, so flooring just the demotions leaves an explicit --eval_min_level 0 pin intact while keeping the floor airtight in training.
        _min_lvl = int(getattr(self.cfg.terrain, 'min_terrain_level', 0))
        _min_lvl = max(0, min(_min_lvl, self.max_terrain_level - 1))
        if _min_lvl > 0:
            _demoted = move_down & ~move_up
            self.terrain_levels[env_ids] = torch.where(
                _demoted,
                self.terrain_levels[env_ids].clamp(min=_min_lvl),
                self.terrain_levels[env_ids])
        # cfg.terrain.resample_at_max_level controls promotion past the last curriculum row: False (default) clamps to [0, max-1]; True uniformly resamples to [resample_min_level, max-1] (ad05628's game-inspired curriculum), which avoids top-row over-fit but stops mean-level converging at the ceiling.
        # cfg.terrain.resample_min_level floors that resample so finished envs aren't dropped onto trivial rows they have long since mastered.
        if bool(getattr(self.cfg.terrain, 'resample_at_max_level', False)):
            _resample_lo = int(getattr(self.cfg.terrain, 'resample_min_level', 0))
            _resample_lo = max(_resample_lo, _min_lvl)  # the hard floor also bounds the resample
            _resample_lo = max(0, min(_resample_lo, self.max_terrain_level - 1))
            self.terrain_levels[env_ids] = torch.where(
                self.terrain_levels[env_ids] >= self.max_terrain_level,
                torch.randint_like(self.terrain_levels[env_ids],
                                   _resample_lo, self.max_terrain_level),
                torch.clip(self.terrain_levels[env_ids], 0))
        else:
            self.terrain_levels[env_ids] = torch.clip(
                self.terrain_levels[env_ids], 0, self.max_terrain_level - 1)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self.env_class[env_ids] = self.terrain_class[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        
        temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
        last_col = temp[:, -1].unsqueeze(1)
        self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)
        self.env_flyball_pit_x0  = self.terrain_flyball_pit_x0[self.terrain_levels, self.terrain_types]
        self.env_flyball_pit_x1  = self.terrain_flyball_pit_x1[self.terrain_levels, self.terrain_types]
        self.env_flyball_pit_mid = (self.env_flyball_pit_x0 + self.env_flyball_pit_x1) * 0.5
        self.env_flyball_forbidden_x0 = self.terrain_flyball_forbidden_x0[self.terrain_levels, self.terrain_types]
        self.env_flyball_forbidden_x1 = self.terrain_flyball_forbidden_x1[self.terrain_levels, self.terrain_types]
        self.env_flyball_wall_angle_deg  = self.terrain_flyball_wall_angle_deg[self.terrain_levels, self.terrain_types]
        self.env_flyball_wall_delta_z    = self.terrain_flyball_wall_delta_z[self.terrain_levels, self.terrain_types]
        self.env_flyball_liftoff_region  = self.terrain_flyball_liftoff_region[self.terrain_levels, self.terrain_types]
        self.env_flyball_wall_region     = self.terrain_flyball_wall_region[self.terrain_levels, self.terrain_types]
        self.env_flyball_landing_region  = self.terrain_flyball_landing_region[self.terrain_levels, self.terrain_types]
        self.env_flyball_air_region      = self.terrain_flyball_air_region[self.terrain_levels, self.terrain_types]
        self.env_flyball_foot_target     = self.terrain_flyball_foot_target[self.terrain_levels, self.terrain_types]
        self.env_flyball_landing_foot_target = self.terrain_flyball_landing_foot_target[self.terrain_levels, self.terrain_types]
        self.env_flyball_hurdle_x_ranges = self.terrain_flyball_hurdle_x_ranges[self.terrain_levels, self.terrain_types]
        self.env_flyball_hurdle_liftoff_region = self.terrain_flyball_hurdle_liftoff_region[self.terrain_levels, self.terrain_types]
        self.env_flyball_hurdle_air_region     = self.terrain_flyball_hurdle_air_region[self.terrain_levels, self.terrain_types]
        self.env_flyball_hurdle_landing_region = self.terrain_flyball_hurdle_landing_region[self.terrain_levels, self.terrain_types]
        self.env_flyball_hurdle_clear_region   = self.terrain_flyball_hurdle_clear_region[self.terrain_levels, self.terrain_types]
        self._refresh_wococo_angle_radii()

    def set_terrain_levels(self, levels):
        """Assign terrain rows and refresh all row-dependent environment data."""
        if isinstance(levels, int):
            levels = torch.full_like(self.terrain_levels, levels)
        self.terrain_levels[:] = torch.clamp(
            levels.to(device=self.device, dtype=torch.long),
            0,
            self.max_terrain_level - 1,
        )
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.env_origins[:] = self.terrain_origins[
            self.terrain_levels, self.terrain_types
        ]
        self.env_class[:] = self.terrain_class[
            self.terrain_levels, self.terrain_types
        ]
        temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
        last_col = temp[:, -1].unsqueeze(1)
        self.env_goals[:] = torch.cat(
            (
                temp,
                last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1),
            ),
            dim=1,
        )
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)
        self.env_flyball_pit_x0 = self.terrain_flyball_pit_x0[
            self.terrain_levels, self.terrain_types
        ]
        self.env_flyball_pit_x1 = self.terrain_flyball_pit_x1[
            self.terrain_levels, self.terrain_types
        ]
        self.env_flyball_pit_mid = (
            self.env_flyball_pit_x0 + self.env_flyball_pit_x1
        ) * 0.5
        self.env_flyball_forbidden_x0 = self.terrain_flyball_forbidden_x0[
            self.terrain_levels, self.terrain_types
        ]
        self.env_flyball_forbidden_x1 = self.terrain_flyball_forbidden_x1[
            self.terrain_levels, self.terrain_types
        ]
        self.env_flyball_wall_angle_deg = self.terrain_flyball_wall_angle_deg[
            self.terrain_levels, self.terrain_types
        ]
        self.env_flyball_wall_delta_z = self.terrain_flyball_wall_delta_z[
            self.terrain_levels, self.terrain_types
        ]
        self.env_flyball_liftoff_region = self.terrain_flyball_liftoff_region[
            self.terrain_levels, self.terrain_types
        ]
        self.env_flyball_wall_region = self.terrain_flyball_wall_region[
            self.terrain_levels, self.terrain_types
        ]
        self.env_flyball_landing_region = self.terrain_flyball_landing_region[
            self.terrain_levels, self.terrain_types
        ]
        self.env_flyball_air_region = self.terrain_flyball_air_region[
            self.terrain_levels, self.terrain_types
        ]
        self.env_flyball_hurdle_x_ranges = self.terrain_flyball_hurdle_x_ranges[
            self.terrain_levels, self.terrain_types
        ]
        self.env_flyball_hurdle_liftoff_region = (
            self.terrain_flyball_hurdle_liftoff_region[
                self.terrain_levels, self.terrain_types
            ]
        )
        self.env_flyball_hurdle_air_region = (
            self.terrain_flyball_hurdle_air_region[
                self.terrain_levels, self.terrain_types
            ]
        )
        self.env_flyball_hurdle_landing_region = (
            self.terrain_flyball_hurdle_landing_region[
                self.terrain_levels, self.terrain_types
            ]
        )
        self.env_flyball_hurdle_clear_region = (
            self.terrain_flyball_hurdle_clear_region[
                self.terrain_levels, self.terrain_types
            ]
        )
        self.env_flyball_foot_target = self.terrain_flyball_foot_target[
            self.terrain_levels, self.terrain_types
        ]
        self.env_flyball_landing_foot_target = self.terrain_flyball_landing_foot_target[
            self.terrain_levels, self.terrain_types
        ]
        self._refresh_wococo_angle_radii()

    def update_command_curriculum(self, env_ids):
        """ Expand command ranges toward max_ranges based on robot performance.

        F02-10: reverted from per-env (introduced in 3daf02f) back to the
        legacy fleet-wide rule. Per-env let easy-terrain envs ratchet
        cmd_vel_x_max independently while hard-terrain envs stayed slow.
        Because the *policy* is shared across all envs, the gait that the
        easy/fast envs learnt at high vmax was applied at low actual speed
        on the slow envs → ugly mismatched gait. Global keeps the whole
        fleet at the same vmax: vmax cannot run ahead of the slowest envs,
        so the policy never has to bridge fast-gait↔slow-cmd.
        """
        if not self.init_done:
            return
        max_ranges = class_to_dict(self.cfg.commands.max_ranges)
        increments = class_to_dict(self.cfg.commands.crclm_incremnt)
        # promote if robots are doing well (S00-03 parity: hardcoded 0.7)
        promote_threshold = int(0.7 * self.cfg.terrain.num_goals)
        if self.max_goal_idx_ep[env_ids].float().mean() >= promote_threshold:
            for key in self.command_ranges:
                if key not in max_ranges or key not in increments:
                    continue
                inc = increments[key]
                lo, hi = self.command_ranges[key]
                max_lo, max_hi = max_ranges[key][0], max_ranges[key][1]
                self.command_ranges[key] = [
                    max(lo - inc, max_lo) if max_lo < lo else min(lo + inc, max_lo),
                    min(hi + inc, max_hi),
                ]

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
            
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(self.num_envs, 4, 6) # for feet only, see create_env()
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # Post-circuit hard-stop hold (cliffside): on final goal reached the command is zeroed and the episode held open for final_goal_stop_s so the robot learns to hard-stop and stand still. final_goal_reached latches the event; final_goal_stop_steps counts the hold down.
        self.final_goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.final_goal_stop_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Timed initial standstill (commands.initial_standstill_max_s > 0): every episode opens standing still for a random 0..max seconds, then runs the sampled speed. Replaces wococo_zero_cmd_prob, which capped term_reach_goal at 1 - p_zero because a standstill episode can never reach the final goal.
        # steps = hold length in control steps, active = env owned by the prefix, post_standstill_x = the approach speed to restore.
        self.cmd_standstill_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.cmd_standstill_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cmd_post_standstill_x = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.wall_g1_time = torch.full((self.num_envs,), 1e9, device=self.device)
        self.wall_g3_rewarded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Sticky latch for _reward_ang_vel_z: set when yaw wraps +pi -> -pi on a wall env (exact -x axis crossing during a left turn), which disables the yaw-penalty silence for the rest of the episode. yaw_prev tracks the previous yaw; both reset on episode reset.
        self.yaw_penalty_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.yaw_prev = torch.zeros(self.num_envs, device=self.device)
        self.wall_approach_prev_vel = torch.zeros(self.num_envs, device=self.device)
        self.wall_push_rewarded = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self.wall_align_rewarded_front = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.wall_align_rewarded_back  = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.wall_feet_touch_count = torch.zeros(self.num_envs, 4, dtype=torch.long, device=self.device)
        self.wall_feet_prev_on_wall = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        # Per-episode latch + step counter for wall_feet_alignment decay: the latch arms the first step the unscaled bonus exceeds the alignment threshold, then the counter drives an exp(-k/tau) decay so the bonus can't be farmed.
        self.wall_align_earned_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.wall_align_steps_since_earn = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Per-foot wall-touch flags for the wococo-off fallback goal-advance gate (advance once all 4 feet touch at the wall goal).
        self.wall_feet_touched = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self.wococo_contact_stage    = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Per-env highest wococo stage ever reached, sticky across episode resets: once an env reliably hits stage N, tracking is silenced in stages 1..N.
        self.max_stage_unlocked      = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.wococo_stage2_time      = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.wococo_stage3_time      = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.wococo_stage4_time      = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Episode step at which each env FIRST reached the wococo sentinel stage (-1 = not yet), driving the post-sentinel roll-term exemption window in check_termination.
        self.wococo_sentinel_reached_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        # Committed-plant window (in_plant | post_sentinel), set each step in check_termination -- the SAME window the roll-term exemption uses, so reward relaxations (e.g. _reward_dof_error's 25%) stay in lockstep.
        self.wococo_plant_window = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.wococo_s0_rear_touched  = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        # Per-foot one-shot latch for _reward_wococo_foot_target: each foot banks its exp(-d/kappa) precision bonus once per episode, at its first hard wall contact during the plant.
        self.wococo_foot_target_paid = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        # P4L landing one-shot precision latch (front FL/FR only): banked at first floor touchdown on its landing foothold (stage 2).
        self.wococo_landing_foot_target_paid = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        # Defer foot-target payouts until the FSM advances past that foot's stage.
        self.wococo_foot_target_pending    = torch.zeros(self.num_envs, 4, device=self.device)
        self.wococo_foot_target_prev_stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.wococo_plant_dwell       = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.wococo_plant_dwell_stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Single-touch n_corr latch (P4 few-stage FSM): each foot's contact bonus (n_corr in Eq.3) is banked once per episode at its first correct wall contact, so repeated tapping can't farm it.
        self.wococo_corr_paid = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self.wococo_armed      = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Per-env persistence counter for the pause-on-error stage advance used by P1/P2/P4 (success must hold N_persist consecutive steps).
        self.wococo_stage_hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Per-episode max-stage buffer for the one-shot stage-reward modes (cfg.rewards.wococo_stage_reward_mode): the reward fires when the current stage exceeds this, which is then bumped.
        self.wococo_episode_max_stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Cumulative scaled income of the camp-farmable wococo shaping terms this episode (see wococo_income_episode_cap).
        self.wococo_income_epsum = torch.zeros(self.num_envs, device=self.device)
        # P4L staggered-landing gate: per-front-foot latch of "landed on its foothold", held for wococo_landing_stagger_window_steps so the fronts may touch down sequentially. Gate only -- F_con/jackpot stay instantaneous, so a foot that touches and lifts never keeps earning the Eq. 3 bonus.
        self.wococo_p4l_land_latch     = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.wococo_p4l_land_latch_age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # P4L stage-2 landing-gate diagnostics: per-episode step counts, reset in reset_idx, emitted as fractions under Episode_rew/wococo/...
        self.wococo_p4l_stage2_steps         = torch.zeros(self.num_envs, device=self.device)
        self.wococo_p4l_stage2_front_land_ok = torch.zeros(self.num_envs, device=self.device)
        # Per-episode accumulator for the LATCHED gate, alongside the instantaneous front_land_ok counter.
        self.wococo_p4l_stage2_land_latched  = torch.zeros(self.num_envs, device=self.device)
        self.wococo_p4l_stage2_rear_air      = torch.zeros(self.num_envs, device=self.device)
        self.wococo_p4l_stage2_roll_ok       = torch.zeros(self.num_envs, device=self.device)
        self.wococo_p4l_stage2_yaw_ok        = torch.zeros(self.num_envs, device=self.device)
        self.wococo_p4l_stage2_hold_ok       = torch.zeros(self.num_envs, device=self.device)
        self.wococo_p4l_stage2_advance       = torch.zeros(self.num_envs, device=self.device)
        # Posture-only landing (wococo_landing_posture_only) gate counters: under that mode the foothold counters stay 0 and these name the blocking conjunct -- pitch, height, or speed (never reached the commanded fraction).
        self.wococo_p4l_stage2_pitch_ok      = torch.zeros(self.num_envs, device=self.device)
        self.wococo_p4l_stage2_height_ok     = torch.zeros(self.num_envs, device=self.device)
        self.wococo_p4l_stage2_speed_ok      = torch.zeros(self.num_envs, device=self.device)
        # Stage-0 (FRONT_PLANT) and stage-1 (REAR_PLANT) gate diagnostics, the wall-side mirror of the p4l_stage2_* counters.
        # stage_max alone says the FSM is stuck but not on which conjunct, and stage 1 is an AND of three conditions (both rears within their target radius, both pressing the wall hard enough, body yawed ~90 deg).
        self.wococo_s0diag_steps    = torch.zeros(self.num_envs, device=self.device)
        self.wococo_s0diag_near     = torch.zeros(self.num_envs, device=self.device)
        self.wococo_s0diag_press    = torch.zeros(self.num_envs, device=self.device)
        self.wococo_s0diag_rear_air = torch.zeros(self.num_envs, device=self.device)
        self.wococo_s0diag_roll_ok  = torch.zeros(self.num_envs, device=self.device)
        self.wococo_s0diag_hold_ok  = torch.zeros(self.num_envs, device=self.device)
        self.wococo_s1diag_steps    = torch.zeros(self.num_envs, device=self.device)
        self.wococo_s1diag_near     = torch.zeros(self.num_envs, device=self.device)
        self.wococo_s1diag_press    = torch.zeros(self.num_envs, device=self.device)
        self.wococo_s1diag_yaw_ok   = torch.zeros(self.num_envs, device=self.device)
        self.wococo_s1diag_hold_ok  = torch.zeros(self.num_envs, device=self.device)
        # Per-episode EVER flags. The *_frac counters divide by every step spent in the stage, and stage 0 spans the whole run-up, so arriving-and-missing is indistinguishable from never arriving; these latch "did this condition hold at any point".
        self.wococo_s0ever_near = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.wococo_s1ever_near = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.wococo_s1ever_yaw  = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Sticky unlock telemetry; stage4_unlocked is the legacy buffer name for "plant complete".
        self.wococo_stage4_unlocked = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)
        self.wococo_sentinel_unlocked = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)
        self.wococo_stage_unlocked = torch.zeros(
            self.num_envs, 5, dtype=torch.bool, device=self.device)
        # Same Eq.3/Eq.4 pattern as the wall, but a separate counter that resets
        # to 0 after each hurdle is cleared (idx 0 → idx 1). Stages:
        #   0 LIFTOFF  — rear feet in pre-hurdle AABB (stance), front airborne
        #   1 AIRBORNE — all 4 feet airborne above the hurdle footprint
        #   2 LANDED   — front feet in post-hurdle AABB (hard contact)
        #   3 terminal — on entry: idx 0 → reset stage:=0, idx:=1; idx 1 → pin
        self.wococo_hurdle_stage              = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.wococo_hurdle_idx                = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.wococo_hurdle_episode_max_stage  = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)
        self.wococo_hurdle_s1_time            = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Per-slot n_wrong gate latch (the hurdle-FSM analog of wococo_armed): armed the first step base xy enters the active liftoff AABB, cleared on slot-0 -> slot-1 hand-off, stage-3 entry, or episode reset.
        self.wococo_hurdle_n_wrong_armed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)

        # One-shot hurdle leg-pair symmetry (_reward_hurdle_pair_sym): previous-step x of each hip-pair centroid (col 0 = front, 1 = rear) plus a (pair, slot, direction) fired latch, so each pair pays out once per bar per travel direction.
        # Every slot is evaluated every step; the latch deliberately does NOT key off wococo_hurdle_idx (overloaded, and only advanced by the hurdle FSM), and the direction axis separates outbound from return crossings.
        # Consecutive steps commanded to stand still, driving the standstill_dof_error_scale ramp in _reward_dof_error.
        self.standstill_hold_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)

        self.hurdle_sym_prev_x = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device)
        self.hurdle_sym_prev_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)
        _n_slots = int(getattr(getattr(self, 'terrain', None), 'n_hurdle_slots', 2) or 2)
        self.hurdle_sym_fired = torch.zeros(
            self.num_envs, 2, _n_slots, 2, dtype=torch.bool, device=self.device)

        # motor_strength feeds both compute_torques (only on the randomize_motor branch) and the priv_latent obs, so guard the draw on the flag: with randomize_motor off the obs channel reads exactly 0 instead of a deviation the dynamics never apply.
        if getattr(self.cfg.domain_rand, "randomize_motor", False):
            str_rng = self.cfg.domain_rand.motor_strength_range
            self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(2, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + str_rng[0]
        else:
            self.motor_strength = torch.ones(2, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        # Per-env, per-joint PD gain multipliers (applied on top of motor_strength).
        if getattr(self.cfg.domain_rand, "randomize_pd_gain", False):
            kp_rng = self.cfg.domain_rand.kp_range
            kd_rng = self.cfg.domain_rand.kd_range
            self.kp_factors = torch_rand_float(kp_rng[0], kp_rng[1], (self.num_envs, self.num_actions), device=self.device)
            self.kd_factors = torch_rand_float(kd_rng[0], kd_rng[1], (self.num_envs, self.num_actions), device=self.device)
        else:
            self.kp_factors = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            self.kd_factors = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        # Per-env, per-joint constant motor offset (encoder/calibration bias).
        if getattr(self.cfg.domain_rand, "randomize_motor_offset", False):
            off_rng = self.cfg.domain_rand.motor_offset_range
            self.motor_offsets = torch_rand_float(off_rng[0], off_rng[1], (self.num_envs, self.num_actions), device=self.device)
        else:
            self.motor_offsets = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.env.history_encoding:
            self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.n_proprio, device=self.device, dtype=torch.float)
        self.action_history_buf = torch.zeros(self.num_envs, self.cfg.domain_rand.action_buf_len, self.num_dofs, device=self.device, dtype=torch.float)
        # Per-env action latency in control steps (0 = no delay); re-sampled per episode in reset_idx when randomize_action_latency.
        self.action_delay_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # Episode-level 0/1-step delay draw used by the action-delay curriculum.
        self.action_delay_episode_enabled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if getattr(self.cfg.domain_rand, "randomize_action_latency", False):
            lo, hi = self.cfg.domain_rand.action_latency_range
            self.action_delay_steps = torch.randint(int(lo), int(hi) + 1, (self.num_envs,), device=self.device)
        # Per-env DR strength multiplier (1.0 = full), driven by terrain level in reset_idx when domain_rand.dr_lock_to_terrain_level is enabled.
        self.dr_scale = torch.ones(self.num_envs, device=self.device, dtype=torch.float)
        # Normalized (0..1) FLEET-level DR progress, i.e. dr_scale's ceiling mapped back onto [0, 1]. Sensor latency (proprio + depth) rides this rather than the per-env dr_scale because transport lag is identical on every unit, and per-env U[0, ceiling] sampling would dilute the trained mean to half the configured range.
        # Dynamics/noise/push DR deliberately KEEP the per-env draw. None => no fleet ramp available; latency falls back to per-env dr_scale.
        self._dr_fleet_latency_ramp = None
        # Warm starts may pre-arm the fleet-level DR ramp.
        self.dr_ramp_armed = bool(getattr(self.cfg.domain_rand, "dr_ramp_start_armed", False))
        # Latch for the plant-failure early terminations: they stay OFF until the
        # fleet-average terrain level first reaches wococo_plant_fail_term_arm_mean_level,
        # then latch ON for the rest of the run (never disarm if the mean dips).
        self.wococo_plant_fail_term_armed = False
        # 1.0 = kills applied to every env (the historic step behaviour); < 1.0 only while wococo_plant_fail_term_ramp_steps is ramping them in.
        self.wococo_plant_fail_alpha = 1.0
        # Sticky, soft ramp for armed-window roll/pitch/base-contact tolerance.
        self.wococo_armed_term_exempt_armed = bool(getattr(
            self.cfg.rewards, 'wococo_armed_term_exempt_start_armed', False))
        self.wococo_armed_term_exempt_trigger_step = (
            self.common_step_counter if self.wococo_armed_term_exempt_armed else -1)
        self.wococo_armed_term_exempt_alpha = latched_linear_alpha(
            self.common_step_counter,
            self.wococo_armed_term_exempt_trigger_step,
            getattr(self.cfg.rewards, 'wococo_armed_term_exempt_ramp_steps', 0))
        self.wococo_base_contact_streak = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)
        # Sticky, delayed scale ramp for the hurdle-clearance reward only.
        self.hurdle_clearance_armed = bool(getattr(
            self.cfg.rewards, 'hurdle_clearance_start_armed', False))
        self.hurdle_clearance_trigger_step = (
            self.common_step_counter if self.hurdle_clearance_armed else -1)
        # Per-environment hurdle-clearance activation draw; sampled on episode reset.
        self.hurdle_clearance_env_enabled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.hurdle_clearance_alpha = latched_linear_alpha(
            self.common_step_counter,
            self.hurdle_clearance_trigger_step,
            getattr(self.cfg.rewards, 'hurdle_clearance_ramp_steps', 0),
            getattr(self.cfg.rewards, 'hurdle_clearance_ramp_delay_steps', 0))

        # Telemetry for the play.py --data overlay: the most recent push impulse per env and the global step it landed on.
        self.last_push_lin = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_push_ang = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_push_step = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)

        # Per-env proprioception latency (joint encoders + IMU), split into an integer floor (control steps) plus a fractional part so a sub-control-step delay (paper: 0.016 s < 20 ms step) is realized by interpolating the two bracketing snapshots.
        # Both are sampled per episode in reset_idx; the ring buffers hold the recent true readings and the delayed copy (_delayed_proprio) is used only for observations, never for rewards/PD/terminations.
        self.proprio_lag_floor = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.proprio_lag_frac  = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        if getattr(self.cfg.domain_rand, "randomize_proprio_latency", False):
            _lo, _hi = self.cfg.domain_rand.proprio_latency_range_s
            self._proprio_buf_depth = int(np.floor(max(_lo, _hi) / self.dt)) + 2
        else:
            self._proprio_buf_depth = 2
        _d = self._proprio_buf_depth
        self._proprio_bufs = {
            'dof_pos': torch.zeros(self.num_envs, _d, self.num_dof, device=self.device),
            'dof_vel': torch.zeros(self.num_envs, _d, self.num_dof, device=self.device),
            'ang_vel': torch.zeros(self.num_envs, _d, 3, device=self.device),
            'imu':     torch.zeros(self.num_envs, _d, 2, device=self.device),
        }

        self.contact_buf = torch.zeros(self.num_envs, self.cfg.env.contact_buf_len, 4, device=self.device, dtype=torch.float)

        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self._resample_commands(torch.arange(self.num_envs, device=self.device, requires_grad=False))
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_contacts_obs = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.contact_thresh = torch.full((self.num_envs, len(self.feet_indices)), 2., device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos_all = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        default_joint_angles = self.cfg.init_state.default_joint_angles
        if hasattr(self.cfg.init_state, "default_joint_angles_from_base_height"):
            default_joint_angles_fn = getattr(self.cfg.init_state, "default_joint_angles_from_base_height")
            if callable(default_joint_angles_fn):
                base_height_target = getattr(getattr(self.cfg, "rewards", object()), "base_height_target", None)
                if base_height_target is not None:
                    default_joint_angles = default_joint_angles_fn(base_height_target)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.default_dof_pos_all[:] = self.default_dof_pos[0]

        self.height_update_interval = 1
        if hasattr(self.cfg.env, "height_update_dt"):
            self.height_update_interval = int(self.cfg.env.height_update_dt / (self.cfg.sim.dt * self.cfg.control.decimation))

        if self.cfg.depth.use_camera:
            # Per-env camera latency in capture frames; buffer_len auto-grows so the delayed frame [:, -1 - lag] is always retrievable.
            lat_lo, lat_hi = (int(min(self.cfg.depth.latency_range)),
                              int(max(self.cfg.depth.latency_range)))
            self.depth_latency_max = lat_hi
            if self._depth_on_demand():
                # On-demand capture bakes the lag into WHEN each env is rendered, so the buffer only has to span whole encoder windows: lag L reads the frame cast L capture steps back, L // update_interval windows in the past.
                n_slots = lat_hi // int(self.cfg.depth.update_interval) + 1
                if self.cfg.depth.buffer_len != n_slots:
                    print(f"[depth] on-demand capture: buffer_len "
                          f"{self.cfg.depth.buffer_len} -> {n_slots} slots "
                          f"(latency_range {self.cfg.depth.latency_range}, "
                          f"update_interval {self.cfg.depth.update_interval})")
                    self.cfg.depth.buffer_len = n_slots
            elif self.cfg.depth.buffer_len < self.depth_latency_max + 1:
                print(f"[depth] growing buffer_len {self.cfg.depth.buffer_len} -> "
                      f"{self.depth_latency_max + 1} to cover latency_range {self.cfg.depth.latency_range}")
                self.cfg.depth.buffer_len = self.depth_latency_max + 1
            # Per-env depth-noise scale s in [noise_scale_min, 1]: ONE linear knob for every camera degradation (see _depth_noise_factors). s=0 is the raw ideal ray-cast (no noise, shadow, legs, min-range remap or lag), s=0.5 is the calibrated real D435, s=1 is 2x its stochastic rates.
            # The initial population is a deterministic linspace by default; configs can request the independent uniform initialization used by the other DR ranges. Opt-in via cfg.depth.noise_scale_per_env; absent -> the configured rates verbatim (s == 0.5 equivalent).
            if getattr(self.cfg.depth, "noise_scale_per_env", False):
                smin = float(getattr(self.cfg.depth, "noise_scale_min", 0.0))
                if getattr(self.cfg.depth, "noise_scale_init_random", False):
                    self.depth_noise_scale = (
                        torch.rand(self.num_envs, device=self.device)
                        * (1.0 - smin) + smin)
                else:
                    self.depth_noise_scale = torch.linspace(
                        smin, 1.0, self.num_envs, device=self.device)
                # Own-body rendering: per-episode Bernoulli gate with p = phys.
                self.depth_self_geom_on = (
                    torch.rand(self.num_envs, device=self.device)
                    < torch.clamp(2.0 * self.depth_noise_scale, 0.0, 1.0))
            self.depth_latency = torch.randint(lat_lo, lat_hi + 1, (self.num_envs,),
                                               device=self.device)
            if (getattr(self.cfg.depth, "latency_scale_with_noise", True)
                    and hasattr(self, "depth_noise_scale")):
                # Optionally scale lag with s; see the matching block in reset_idx.
                _p = torch.clamp(2.0 * self.depth_noise_scale, 0.0, 1.0)
                self.depth_latency = (self.depth_latency.float() * _p).round().long()
            kmax_bs = int(self.cfg.depth.pw_blindspot_cols)
            slope_max = float(getattr(self.cfg.depth, "pw_blindspot_slope_max", 0.15))
            self.pw_bs_side = self._sample_blindspot_side(self.num_envs)   # True=right edge
            self.pw_bs_k = torch.rand(self.num_envs, device=self.device) * max(kmax_bs, 0)
            self.pw_bs_slope = (torch.rand(self.num_envs, device=self.device) * 2.0 - 1.0) * slope_max
            self.depth_buffer = torch.zeros(self.num_envs,
                                            self.cfg.depth.buffer_len,
                                            self.cfg.depth.resized[1],
                                            self.cfg.depth.resized[0]).to(self.device)
            if self._warp_enabled():
                from legged_gym.utils.warp_cam import WarpDepthCamera
                if not hasattr(self.terrain, "vertices"):
                    raise RuntimeError("Warp depth camera requires a trimesh terrain "
                                       "(self.terrain.vertices/triangles).")
                self.warp_cam = WarpDepthCamera(self.num_envs, self.cfg.depth, self.device)
                border = self.terrain.cfg.border_size
                self.warp_cam.set_mesh(self.terrain.vertices, self.terrain.triangles, (-border, -border, 0.0))
                print(f"[warp] depth camera enabled: {self.warp_cam.H}x{self.warp_cam.W}, "
                      f"mesh verts={self.terrain.vertices.shape[0]} tris={self.terrain.triangles.shape[0]}")
                if getattr(self.cfg.depth, "self_geometry", False):
                    # Render the robot's own body into the depth image (the real D435 sees its legs; a terrain-only ray-cast does not).
                    from legged_gym.utils.self_geometry import load_urdf_self_geometry
                    self_geom = load_urdf_self_geometry(
                        self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR),
                        collapse_fixed_joints=self.cfg.asset.collapse_fixed_joints,
                        replace_cylinder_with_capsule=self.cfg.asset.replace_cylinder_with_capsule,
                        link_filter=getattr(self.cfg.depth, "self_geometry_links", None))
                    # Resolve body names the same way feet_indices does, so the indices match the rigid body state tensor.
                    body_index_map = {
                        nm: self.gym.find_actor_rigid_body_handle(
                            self.envs[0], self.actor_handles[0], nm)
                        for nm in sorted(set(self_geom.body_names))}
                    self.warp_cam.set_self_geometry(self_geom, body_index_map)
                    print(f"[warp] self-geometry enabled: {self_geom.summary()}")

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            # 'timeout' rides with 'termination': both are terminal-state prices applied AFTER the only_positive_rewards clip in compute_reward, so neither may be summed into rew_buf here -- otherwise the timeout penalty is swallowed on exactly the low rows where loitering is cheapest.
            if name in ("termination", "timeout"):
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}
        # Diagnostic split of tracking_goal_vel (standstill carrot vs moving), logged as rew_tracking_goal_vel_{standstill,moving}: they sum to rew_tracking_goal_vel and never touch rew_buf.
        for _k in ('tracking_goal_vel_standstill', 'tracking_goal_vel_moving'):
            self.episode_sums[_k] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.cfg.terrain.horizontal_scale
        hf_params.row_scale = self.cfg.terrain.horizontal_scale
        hf_params.vertical_scale = self.cfg.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.border
        hf_params.transform.p.y = -self.terrain.border
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples.flatten(order='C'), hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
            Very slow when horizontal_scale is small
        """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        print("Adding trimesh to simulation...")
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)  
        print("Trimesh added")
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)


    def _warp_enabled(self):
        return getattr(self.cfg.depth, "use_warp", False) or os.environ.get("EP_USE_WARP", "0") == "1"

    def attach_camera(self, i, env_handle, actor_handle):
        if self.cfg.depth.use_camera and not self._warp_enabled():
            config = self.cfg.depth
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg.depth.original[0]
            camera_props.height = self.cfg.depth.original[1]
            camera_props.enable_tensors = True
            camera_horizontal_fov = self.cfg.depth.horizontal_fov 
            camera_props.horizontal_fov = camera_horizontal_fov

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.cam_handles.append(camera_handle)
            
            local_transform = gymapi.Transform()
            
            camera_position = np.copy(config.position)
            camera_angle = np.random.uniform(config.angle[0], config.angle[1])
            
            local_transform.p = gymapi.Vec3(*camera_position)
            local_transform.r = gymapi.Quat.from_euler_zyx(0, np.radians(camera_angle), 0)
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)

            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]


        for s in ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]:
            feet_idx = self.gym.find_asset_rigid_body_index(robot_asset, s)
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
            self.gym.create_asset_force_sensor(robot_asset, feet_idx, sensor_pose)
        
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.cam_handles = []
        self.cam_tensors = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

        print("Creating env...")
        for i in tqdm(range(self.num_envs)):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            if self.cfg.env.randomize_start_pos:
                pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            if self.cfg.env.randomize_start_yaw:
                rand_yaw_quat = gymapi.Quat.from_euler_zyx(0., 0., self.cfg.env.rand_yaw_range*np.random.uniform(-1, 1))
                start_pose.r = rand_yaw_quat
            start_pose.p = gymapi.Vec3(*(pos + self.base_init_state[:3]))

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "anymal", i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)

            
            self.attach_camera(i, env_handle, anymal_handle)

            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device).to(torch.float)
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)
        else:
            # randomize_friction disabled (e.g. --no-dr): the priv_latent obs still needs this tensor, so fill it with the nominal friction (midpoint of friction_range) to match the training distribution centre.
            friction_range = getattr(self.cfg.domain_rand, "friction_range", [1.0, 1.0])
            nominal_friction = 0.5 * (friction_range[0] + friction_range[1])
            self.friction_coeffs_tensor = torch.full((self.num_envs, 1), nominal_friction,
                                                      dtype=torch.float, device=self.device)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        # The parabolic clearance reward uses the actor's actual collision primitives (collision origins, collapsed fixed joints, Isaac's cylinder->capsule conversion). Initialise only for tasks that enable the term, so others carry no parsing/GPU cost.
        if self.reward_scales.get('hurdle_clearance', 0.0) != 0.0:
            self._init_hurdle_clearance_collision_geometry(asset_path)

        hip_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.dof_names.index(name)
        thigh_names = ["FR_thigh_joint", "FL_thigh_joint", "RR_thigh_joint", "RL_thigh_joint"]
        self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(thigh_names):
            self.thigh_indices[i] = self.dof_names.index(name)
        calf_names = ["FR_calf_joint", "FL_calf_joint", "RR_calf_joint", "RL_calf_joint"]
        self.calf_indices = torch.zeros(len(calf_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(calf_names):
            self.calf_indices[i] = self.dof_names.index(name)

        # RIGID-BODY indices of the four hip links, in the same [FR, FL, RR, RL] order as the DOF index tensors. Distinct from self.hip_indices (which indexes dof_pos); these index rigid_body_states to locate each leg pair relative to a hurdle.
        hip_body_names = ["FR_hip", "FL_hip", "RR_hip", "RL_hip"]
        self.hip_body_indices = torch.zeros(len(hip_body_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_body_names):
            self.hip_body_indices[i] = body_names.index(name)
    
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        # Cache per-step GPU constants so we don't reallocate them every call.
        self._roll_term_rad = float(self.cfg.env.roll_term_deg) * (np.pi / 180.0)
        self._roll_exempt_rad = float(getattr(self.cfg.rewards, 'wococo_roll_exempt_deg', 100.0)) * (np.pi / 180.0)
        self._wococo_roll_exempt_post_steps = int(round(
            float(getattr(self.cfg.rewards, 'wococo_roll_exempt_post_sentinel_s', 2.0)) / self.dt))
        # Extra window (in sim steps) after the FSM first reaches the sentinel during which _reward_tracking_yaw is pinned to its MAXIMAL value (1.0) instead of gated to 0, keeping the reward stream alive through the unwind beat without paying the policy to chase its heading.
        # 0 (default) = re-enable normal yaw tracking the instant the plant gate drops; reads the same wococo_sentinel_reached_step latch.
        self._yaw_post_sentinel_hold_steps = int(round(
            float(getattr(self.cfg.rewards, 'yaw_tracking_post_sentinel_hold_s', 0.0)) / self.dt))
        # Curriculum gate: the max-pin hold engages only once the FLEET-MEAN terrain level reaches this; below it, normal yaw tracking. 0 (default) = no gate.
        self._yaw_post_sentinel_hold_min_mean_level = float(getattr(
            self.cfg.rewards, 'yaw_post_sentinel_hold_min_mean_level', 0.0))
        # Goal-index gates for the flyball wall circuit (env_class 19). The default 8-slot layout is g0 approach / g1 takeoff / g2 contact / g3 landing / g4+ return, so the wall sits at index 2 and the return phase opens at 4.
        # Compact layouts move both (go2_teacher: g0 between-hurdles / g1 wall / g2 between-hurdles / g3 final), so every env19 goal gate reads these instead of literals.
        self._flyball_wall_goal_idx   = int(getattr(self.cfg.terrain, 'flyball_wall_goal_idx', 2))
        self._flyball_return_goal_idx = int(getattr(self.cfg.terrain, 'flyball_return_goal_idx', 4))
        self._wococo_wall_goal_idx_19 = torch.tensor(
            self._flyball_wall_goal_idx, device=self.device, dtype=torch.long)
        self._wococo_goal_idx_4 = torch.tensor(4, device=self.device, dtype=torch.long)
        # Canonical WoCoCo 4-design selector (reward-only): cfg.rewards.wococo_design in {'P1','P2','P3','P4'} selects which contact FSM the wall reward uses.
        # plant_complete_stage is the stage index meaning "plant done" -- goal-advance, n_wrong disarm, tracking/hip silencing and termination readers key off it, so the stage count can differ per design (P3 keeps 0..4; P4/P2 use 0..2; P1 uses 0..1).
        self._wococo_design = str(getattr(self.cfg.rewards, 'wococo_design', 'P3')).upper()
        # Default plant_complete_stage per design when the cfg knob is unset. P4L = P4 plus a stage-2 floor LANDING before the sentinel: plant_complete stays 2, sentinel becomes 3.
        _pcs_default = {'P1': 1, 'P2': 2, 'P3': 3, 'P4': 2, 'P4L': 2}.get(self._wococo_design, 3)
        self._wococo_plant_complete_stage = int(
            getattr(self.cfg.rewards, 'wococo_plant_complete_stage', _pcs_default))
        # Sentinel stage = the terminal/return stage the FSM clips to after the plant. For P3 the LANDING (stage 3) is a real plant stage so the sentinel is 4; for P1/P2/P4 plant_complete IS the sentinel.
        # Readers meaning "mid-plant" use `armed & stage < sentinel`; goal-advance/unlock use `>= plant_complete`.
        self._wococo_sentinel_stage = (
            self._wococo_plant_complete_stage + 1 if self._wococo_design in ('P3', 'P4L')
            else self._wococo_plant_complete_stage)
        self._wococo_n_persist = int(getattr(self.cfg.rewards, 'wococo_n_persist', 3))

        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.env_class = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            fixed = bool(getattr(self.cfg.terrain, 'fixed_level', False))
            # Fixed levels remain valid even when live curriculum updates are off.
            if not self.cfg.terrain.curriculum and not fixed:
                max_init_level = self.cfg.terrain.num_rows - 1
            if fixed:
                self.terrain_levels = torch.full((self.num_envs,), max_init_level, dtype=torch.long, device=self.device)
            else:
                # cfg.terrain.min_terrain_level floors the initial spread. Levels are NOT restored from a checkpoint, so a --resume run re-draws them here and would otherwise spend its first iterations back on rows 0..3. Clamped below max_init_level.
                min_init_level = max(0, min(
                    int(getattr(self.cfg.terrain, 'min_terrain_level', 0)), max_init_level))
                self.terrain_levels = torch.randint(min_init_level, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            
            
            print(self.terrain_levels.min().item(), self.terrain_levels.max().item(), "num_rows", self.cfg.terrain.num_rows)
            print(self.terrain_types.min().item(),  self.terrain_types.max().item(),  "num_cols", self.cfg.terrain.num_cols)
            
            
            self.terrain_class = torch.from_numpy(self.terrain.terrain_type).to(self.device).to(torch.float)
            self.env_class[:] = self.terrain_class[self.terrain_levels, self.terrain_types]

            self.terrain_goals = torch.from_numpy(self.terrain.goals).to(self.device).to(torch.float)
            self.terrain_flyball_pit_x0 = torch.from_numpy(self.terrain.flyball_pit_x0).to(self.device).to(torch.float)
            self.terrain_flyball_pit_x1 = torch.from_numpy(self.terrain.flyball_pit_x1).to(self.device).to(torch.float)
            self.terrain_flyball_forbidden_x0 = torch.from_numpy(self.terrain.flyball_forbidden_x0).to(self.device).to(torch.float)
            self.terrain_flyball_forbidden_x1 = torch.from_numpy(self.terrain.flyball_forbidden_x1).to(self.device).to(torch.float)
            self.terrain_flyball_wall_angle_deg = torch.from_numpy(self.terrain.flyball_wall_angle_deg).to(self.device).float()
            self.terrain_flyball_wall_delta_z   = torch.from_numpy(self.terrain.flyball_wall_delta_z).to(self.device).float()
            self.terrain_flyball_liftoff_region = torch.from_numpy(self.terrain.flyball_liftoff_region).to(self.device).float()
            self.terrain_flyball_wall_region    = torch.from_numpy(self.terrain.flyball_wall_region).to(self.device).float()
            self.terrain_flyball_landing_region = torch.from_numpy(self.terrain.flyball_landing_region).to(self.device).float()
            self.terrain_flyball_air_region     = torch.from_numpy(self.terrain.flyball_air_region).to(self.device).float()
            # Per-foot WoCoCo targets (rows, cols, 4, 3) in world metres.
            self.terrain_flyball_foot_target    = torch.from_numpy(self.terrain.flyball_foot_target).to(self.device).float()
            # Front-foot landing footholds (rows, cols, 2, 3) for the P4L stage-2 floor landing.
            self.terrain_flyball_landing_foot_target = torch.from_numpy(self.terrain.flyball_landing_foot_target).to(self.device).float()
            self.terrain_flyball_hurdle_x_ranges = torch.from_numpy(self.terrain.flyball_hurdle_x_ranges).to(self.device).float()
            self.terrain_flyball_hurdle_liftoff_region = torch.from_numpy(self.terrain.flyball_hurdle_liftoff_region).to(self.device).float()
            self.terrain_flyball_hurdle_air_region     = torch.from_numpy(self.terrain.flyball_hurdle_air_region).to(self.device).float()
            self.terrain_flyball_hurdle_landing_region = torch.from_numpy(self.terrain.flyball_hurdle_landing_region).to(self.device).float()
            # Per-hurdle parabolic-clearance bounds (rows, cols, slots, 2, 3).
            self.terrain_flyball_hurdle_clear_region = torch.from_numpy(self.terrain.flyball_hurdle_clear_region).to(self.device).float()
            self.env_goals = torch.zeros(self.num_envs, self.cfg.terrain.num_goals + self.cfg.env.num_future_goal_obs, 3, device=self.device, requires_grad=False)
            self.cur_goal_idx = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
            self.max_goal_idx_ep = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
            temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
            last_col = temp[:, -1].unsqueeze(1)
            self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]
            self.cur_goals = self._gather_cur_goals()
            self.next_goals = self._gather_cur_goals(future=1)
            self.env_flyball_pit_x0  = self.terrain_flyball_pit_x0[self.terrain_levels, self.terrain_types]
            self.env_flyball_pit_x1  = self.terrain_flyball_pit_x1[self.terrain_levels, self.terrain_types]
            self.env_flyball_pit_mid = (self.env_flyball_pit_x0 + self.env_flyball_pit_x1) * 0.5
            self.env_flyball_forbidden_x0 = self.terrain_flyball_forbidden_x0[self.terrain_levels, self.terrain_types]
            self.env_flyball_forbidden_x1 = self.terrain_flyball_forbidden_x1[self.terrain_levels, self.terrain_types]
            self.env_flyball_wall_angle_deg  = self.terrain_flyball_wall_angle_deg[self.terrain_levels, self.terrain_types]
            self.env_flyball_wall_delta_z    = self.terrain_flyball_wall_delta_z[self.terrain_levels, self.terrain_types]
            self.env_flyball_liftoff_region  = self.terrain_flyball_liftoff_region[self.terrain_levels, self.terrain_types]
            self.env_flyball_wall_region     = self.terrain_flyball_wall_region[self.terrain_levels, self.terrain_types]
            self.env_flyball_landing_region  = self.terrain_flyball_landing_region[self.terrain_levels, self.terrain_types]
            self.env_flyball_air_region      = self.terrain_flyball_air_region[self.terrain_levels, self.terrain_types]
            self.env_flyball_foot_target     = self.terrain_flyball_foot_target[self.terrain_levels, self.terrain_types]
            self.env_flyball_landing_foot_target = self.terrain_flyball_landing_foot_target[self.terrain_levels, self.terrain_types]
            self.env_flyball_hurdle_x_ranges = self.terrain_flyball_hurdle_x_ranges[self.terrain_levels, self.terrain_types]
            self.env_flyball_hurdle_liftoff_region = self.terrain_flyball_hurdle_liftoff_region[self.terrain_levels, self.terrain_types]
            self.env_flyball_hurdle_air_region     = self.terrain_flyball_hurdle_air_region[self.terrain_levels, self.terrain_types]
            self.env_flyball_hurdle_landing_region = self.terrain_flyball_hurdle_landing_region[self.terrain_levels, self.terrain_types]
            self.env_flyball_hurdle_clear_region   = self.terrain_flyball_hurdle_clear_region[self.terrain_levels, self.terrain_types]
            self._refresh_wococo_angle_radii()

        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        reward_norm_factor = 1#np.sum(list(self.reward_scales.values()))
        for rew in self.reward_scales:
            self.reward_scales[rew] = self.reward_scales[rew] / reward_norm_factor
        if self.cfg.commands.curriculum:
            self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        else:
            self.command_ranges = class_to_dict(self.cfg.commands.max_ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_height_samples(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        i = self.lookat_id
        base_pos = (self.root_states[i, :3]).cpu().numpy()
        heights = self.measured_heights[i].cpu().numpy()
        height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
        for j in range(heights.shape[0]):
            x = height_points[j, 0] + base_pos[0]
            y = height_points[j, 1] + base_pos[1]
            z = heights[j]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
    
    def _draw_goals(self):
        if not (hasattr(self, 'env_class') and hasattr(self, 'wococo_contact_stage')):
            return
        env_id = self.lookat_id
        if int(self.wococo_contact_stage[env_id].item()) >= getattr(
                self, '_wococo_sentinel_stage', 99):
            return
        if not bool(((self.env_class[env_id] == 19)
                     | (self.env_class[env_id] == 15)).item()):
            return

        stage = int(self.wococo_contact_stage[env_id].item())
        design = getattr(self, '_wococo_design', '')
        if design == 'P4L' and stage == 2 and self._wococo_landing_posture_only():
            # Posture-only landing: there are no foothold targets to draw, and falling through to the wall-target branch would draw the PLANT footholds during the landing.
            return
        if (design == 'P4L' and stage == 2
                and hasattr(self, 'env_flyball_landing_foot_target')):
            targets = self.env_flyball_landing_foot_target[env_id]
            radii = self._wococo_landing_accept_radius()
            color = np.array([[1.0, 0.15, 0.8]], dtype=np.float32)
            active_ids = [0, 1]  # landing target array is front-feet only
        elif hasattr(self, 'env_flyball_foot_target'):
            targets = self.env_flyball_foot_target[env_id]
            radii = self._wococo_rad('wococo_r_wall',
                                     'wococo_foot_region_radius', 0.18)
            color = np.array([[1.0, 0.55, 0.05]], dtype=np.float32)
            active = self._wococo_active_feet_mask(self.wococo_contact_stage)[env_id]
            active_ids = torch.nonzero(active, as_tuple=False).flatten().tolist()
        else:
            return
        if not active_ids:
            return
        if torch.is_tensor(radii):
            radii = radii[env_id].detach().cpu().numpy()
        else:
            radii = np.full(4 if targets.shape[0] == 4 else 2,
                             float(radii), dtype=np.float32)

        target_geom = gymutil.WireframeSphereGeometry(
            0.035, 16, 16, None, color=tuple(color[0]))
        theta = np.linspace(0.0, 2.0 * np.pi, 49, dtype=np.float32)
        for foot_id in active_ids:
            target = targets[foot_id].detach().cpu().numpy()
            pose = gymapi.Transform(gymapi.Vec3(*target), r=None)
            gymutil.draw_lines(target_geom, self.gym, self.viewer,
                               self.envs[env_id], pose)
            xy = float(radii[foot_id]) * np.stack(
                (np.cos(theta), np.sin(theta)), axis=1)
            circle = np.zeros((48, 2, 3), dtype=np.float32)
            circle[:, 0, :2] = target[None, :2] + xy[:-1]
            circle[:, 1, :2] = target[None, :2] + xy[1:]
            circle[:, :, 2] = target[2] + 0.012
            colors = np.repeat(color, 48, axis=0)
            self.gym.add_lines(self.viewer, self.envs[env_id],
                               48, circle, colors)
        
    def _draw_feet(self):
        if hasattr(self, 'feet_at_edge'):
            non_edge_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0))
            edge_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0, 0))

            feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
            for i in range(4):
                pose = gymapi.Transform(gymapi.Vec3(feet_pos[self.lookat_id, i, 0], feet_pos[self.lookat_id, i, 1], feet_pos[self.lookat_id, i, 2]), r=None)
                if self.feet_at_edge[self.lookat_id, i]:
                    gymutil.draw_lines(edge_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
                else:
                    gymutil.draw_lines(non_edge_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
    
    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        for i in range(self.num_envs):
            offset = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points,2), device=self.device).squeeze()
            xy_noise = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points,2), device=self.device).squeeze() + offset
            points[i, :, 0] = grid_x.flatten() + xy_noise[:, 0]
            points[i, :, 1] = grid_y.flatten() + xy_noise[:, 1]
        return points

    def get_foot_contacts(self):
        foot_contacts_bool = self.contact_forces[:, self.feet_indices, 2] > 10
        if self.cfg.env.include_foot_contacts:
            return foot_contacts_bool
        else:
            return torch.zeros_like(foot_contacts_bool).to(self.device)

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        num_query_envs = self.num_envs if env_ids is None else len(env_ids)
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(num_query_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids is not None:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        world_points = points
        map_points = ((points + self.terrain.cfg.border_size) /
                      self.terrain.cfg.horizontal_scale).long()
        px = map_points[:, :, 0].view(-1)
        py = map_points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(torch.min(heights1, heights2), heights3)

        # Optional conservative local elevation aggregation for thin obstacles.
        # Pool only the sparse scandot footprints that overlap a real hurdle;
        # never pool the full terrain heightfield. This keeps the cost proportional
        # to num_envs * num_scandots * kernel_area rather than map resolution.
        # Deterministic max-elevation analogue, for the sparse teacher scandots, of Kuthirummal, Das & Samarasekera, "A Graph Traversal Based Algorithm for Obstacle Detection Using Lidar or Stereo," IROS 2011, DOI 10.1109/IROS.2011.6048725.
        # It preserves the scandot output dimension.
        pool_rx = max(0, int(getattr(self.cfg.terrain, 'scandot_maxpool_radius_x', 0)))
        pool_ry = max(0, int(getattr(self.cfg.terrain, 'scandot_maxpool_radius_y', 0)))
        pool_mask = None
        if getattr(self.cfg.terrain, 'scandot_maxpool', False) and (pool_rx or pool_ry):
            if getattr(self.cfg.terrain, 'scandot_maxpool_hurdles_only', False):
                hurdle_ranges = getattr(self, 'env_flyball_hurdle_x_ranges', None)
                if hurdle_ranges is not None:
                    if env_ids is not None:
                        hurdle_ranges = hurdle_ranges[env_ids]
                    dot_lo = world_points[:, :, 0] - pool_rx * self.terrain.cfg.horizontal_scale
                    dot_hi = world_points[:, :, 0] + pool_rx * self.terrain.cfg.horizontal_scale
                    range_lo = hurdle_ranges[:, :, 0].unsqueeze(2)
                    range_hi = hurdle_ranges[:, :, 1].unsqueeze(2)
                    valid = range_hi > range_lo
                    pool_mask = ((dot_hi.unsqueeze(1) >= range_lo) &
                                 (dot_lo.unsqueeze(1) <= range_hi) &
                                 valid).any(dim=1)
                else:
                    pool_mask = torch.zeros(
                        world_points.shape[:2], dtype=torch.bool, device=self.device)
            else:
                pool_mask = torch.ones(
                    world_points.shape[:2], dtype=torch.bool, device=self.device)

        if pool_mask is not None:
            pool_indices = torch.nonzero(pool_mask.view(-1), as_tuple=False).squeeze(1)
            offset_x = torch.arange(-pool_rx, pool_rx + 1, device=self.device)
            offset_y = torch.arange(-pool_ry, pool_ry + 1, device=self.device)
            sample_px = torch.clip(
                px[pool_indices, None, None] + offset_x[None, :, None],
                0, self.height_samples.shape[0] - 1)
            sample_py = torch.clip(
                py[pool_indices, None, None] + offset_y[None, None, :],
                0, self.height_samples.shape[1] - 1)
            pooled_heights = self.height_samples[sample_px, sample_py].amax(dim=(1, 2))
            heights[pool_indices] = pooled_heights

        return heights.view(num_query_envs, -1) * self.terrain.cfg.vertical_scale

    def _mask_occluded_scandots(self, heights):
        """Replace scandot heights the front depth camera could not see
        (scandot_occlusion_mask). For each dot, walk a line of sight from the
        camera mount (cfg.depth.position, yaw-aligned to the base like the dot
        grid itself) to the dot's terrain surface point; if the heightfield
        rises above the ray anywhere along the way, the dot reports the highest
        blocking terrain instead of the true height — the occluder's front face
        is all a real depth sensor would deliver. Orientation of the camera is
        deliberately ignored (occlusion is a property of geometry, not gaze),
        so the mask stays stable through the banked wall plant.

        heights: (num_envs, num_height_points) world-frame terrain z [m].
        Returns the masked tensor (same shape).
        """
        K   = int(getattr(self.cfg.terrain, 'scandot_occlusion_samples', 12))
        eps = float(getattr(self.cfg.terrain, 'scandot_occlusion_eps', 0.05))
        hs  = self.terrain.cfg.horizontal_scale
        vs  = self.terrain.cfg.vertical_scale
        border = self.terrain.cfg.border_size

        # Dot surface points in world xy (same construction as _get_heights).
        pts = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points),
                             self.height_points) + self.root_states[:, :3].unsqueeze(1)
        if not hasattr(self, '_scandot_cam_offset'):
            self._scandot_cam_offset = torch.tensor(
                self.cfg.depth.position, dtype=torch.float, device=self.device)
        cam = self.root_states[:, :3] + quat_apply_yaw(
            self.base_quat, self._scandot_cam_offset.expand(self.num_envs, 3))

        seg_xy = pts[:, :, :2] - cam[:, None, :2]              # (N, P, 2)
        seg_z  = heights - cam[:, None, 2]                     # (N, P) ray ends on the dot's surface
        occluded  = torch.zeros_like(heights, dtype=torch.bool)
        blocker_h = torch.full_like(heights, -1.0e6)
        # Loop over the K samples (not vectorized over K) to keep transient index tensors at (N, P) instead of (N, P, K); this runs on the depth.update_interval cadence, so small gathers are cheap.
        for t in torch.linspace(0.08, 0.92, K, device=self.device):
            sample_xy = cam[:, None, :2] + t * seg_xy
            px = ((sample_xy[:, :, 0] + border) / hs).long().view(-1)
            py = ((sample_xy[:, :, 1] + border) / hs).long().view(-1)
            px = torch.clip(px, 0, self.height_samples.shape[0]-2)
            py = torch.clip(py, 0, self.height_samples.shape[1]-2)
            # Same min-of-3 stencil as _get_heights: biases the terrain LOW, so marginal samples err toward "visible" rather than false occlusion.
            h_terr = torch.min(torch.min(self.height_samples[px, py],
                                         self.height_samples[px+1, py]),
                               self.height_samples[px, py+1]).view(self.num_envs, -1) * vs
            z_ray = cam[:, None, 2] + t * seg_z
            blocked = h_terr > z_ray + eps
            occluded |= blocked
            blocker_h = torch.where(blocked, torch.maximum(blocker_h, h_terr), blocker_h)
        return torch.where(occluded, blocker_h, heights)

    def _get_heights_points(self, coords, env_ids=None):
        if env_ids:
            points = coords[env_ids]
        else:
            points = coords

        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale


    def _tracking_fade_factor(self):
        # Tracking-reward fade: 1.0 until `start` episodes per env, then decays to 0.0 over the next `N`. Curve is "linear" or "log" (concave: fast early drop, slow tail); returns a Python float.
        N = float(getattr(self.cfg.rewards, 'tracking_fadeout_episodes', 0.0))
        if N <= 0.0:
            return 1.0
        start = float(getattr(self.cfg.rewards, 'tracking_fadeout_start_episodes', 0.0))
        ep_per_env = float(self.common_step_counter) / float(self.max_episode_length)
        if ep_per_env <= start:
            return 1.0
        progress = (ep_per_env - start) / N
        if progress >= 1.0:
            return 0.0
        curve = str(getattr(self.cfg.rewards, 'tracking_fadeout_curve', 'log')).lower()
        if curve == 'linear':
            return 1.0 - progress
        import math
        return 1.0 - math.log1p(progress) / math.log(2.0)

    def _tracking_stage_mask(self, unmask_stage=None):
        """Per-env multiplicative mask for the wococo-stage gating of
        tracking rewards. Returns a (N,) float tensor in {0.0, 1.0}, or
        None if no stage gating is active (caller can short-circuit).
        Two modes:
          * progressive_tracking_gating=True: silence iff 1 ≤ stage ≤ max_stage_unlocked
            (per-env, sticky across episode resets).
          * gate_tracking_during_wococo_plant=True: silence during the plant
            phase (plant_armed and not yet past plant_complete_stage). This is
            DESIGN-ROBUST — the plant stages differ per design (P3 plants in
            stages 1-3, P4/P1 plant from stage 0), so we key off the plant_armed
            latch + plant_complete_stage rather than fixed stage numbers.
          * neither: returns None (no mask).
        """
        if not hasattr(self, 'wococo_contact_stage'):
            return None
        stage = self.wococo_contact_stage
        if bool(getattr(self.cfg.rewards, 'progressive_tracking_gating', False)):
            torch.maximum(self.max_stage_unlocked, stage, out=self.max_stage_unlocked)
            # Silence only in the wall-plant stages 1-3; stage 0 (approach) and stage 4 (terminal/return) keep tracking on so the policy has a goal-direction signal back toward spawn.
            silence = (stage >= 1) & (stage <= 3) & (stage <= self.max_stage_unlocked)
            return (~silence).float()
        if bool(getattr(self.cfg.rewards, 'gate_tracking_during_wococo_plant', True)):
            # Plant phase: committed to the wall (plant_armed) and the FSM below unmask_stage (default the sentinel). P4L turns tracking_goal_vel on at the LANDING (stage 2) while tracking_yaw stays off until the sentinel (stage 3).
            thr = self._wococo_sentinel_stage if unmask_stage is None else int(unmask_stage)
            in_plant = self.wococo_armed & (stage < thr)
            return (~in_plant).float()
        return None

    def _reward_tracking_goal_vel(self):
        # Two regimes, split on lin_vel_clip. The cliffside command curriculum runs lin_vel_x [0,1] (easy) -> [1,2] (hard) and lin_vel_clip=0.2 zeros any sampled |cmd|<=0.2, so after clipping cmd is either 0 or >0.2 and the split below is clean:
        #  * MOVING (|c| >= lin_vel_clip): S00-03 form min(v_goal, c)/(c+eps), BIT-IDENTICAL to the faithful teacher repro.
        #  * STANDSTILL (c == 0): the S00-03 min/div form gives NO positive signal at rest and blows up on backward drift, so use the low-speed reward exp(-||v_xy||^2/sigma) in (0,1] -- the standstill carrot the parkour tracking form structurally lacks.
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        cur_vel = self.root_states[:, 7:9]
        cmd = self.commands[:, 0]
        rew_move = torch.minimum(torch.sum(target_vec_norm * cur_vel, dim=-1),
                                 cmd) / (cmd + 1e-5)
        rew_stand = torch.exp(-torch.sum(torch.square(cur_vel), dim=-1)
                              / self.cfg.rewards.tracking_sigma)
        standstill = torch.abs(cmd) < self.cfg.commands.lin_vel_clip
        rew = torch.where(standstill, rew_stand, rew_move)
        # Stash the per-env standstill mask so compute_reward can split this term's logged episode-sum into standstill and moving streams; the returned reward is unchanged.
        self._track_goal_vel_standstill = standstill
        mask = self._tracking_stage_mask(
            unmask_stage=getattr(self.cfg.rewards, 'tracking_goal_vel_unmask_stage', None))
        if mask is not None:
            rew = rew * mask
        return rew * self._tracking_fade_factor()

    def _reward_tracking_yaw(self):
        # Silence goal-facing reward during gated plant stages.
        rew = torch.exp(-torch.abs(self.target_yaw - self.yaw))
        mask = self._tracking_stage_mask(
            unmask_stage=getattr(self.cfg.rewards, 'tracking_yaw_unmask_stage', None))
        if mask is not None:
            rew = rew * mask
        # Standstill yaw gate (opt-in, default off): when commanded to stand still (|cmd_x| < lin_vel_clip) zero the yaw-facing reward, so a stopped robot isn't nudged to pivot in place toward the goal.
        if bool(getattr(self.cfg.rewards, 'gate_tracking_yaw_at_standstill', False)):
            moving = torch.abs(self.commands[:, 0]) >= self.cfg.commands.lin_vel_clip
            rew = rew * moving.float()
        # Post-sentinel yaw hold (opt-in, default 0 s): for yaw_tracking_post_sentinel_hold_s after the FSM first reaches the sentinel, OVERRIDE yaw tracking to its maximal value (1.0) instead of letting the plant/standstill gates zero it.
        # This spans the unwind beat where the robot is still rotating out of the banked plant: a constant max pays no gradient to snap its heading to the return goal, yet keeps the reward stream alive so the value doesn't collapse there.
        # GATED on the fleet-mean terrain level reaching yaw_post_sentinel_hold_min_mean_level, applied LAST so it wins over both gates above, and inert for non-wococo envs (latch stays -1). Yaw-only.
        if self._yaw_post_sentinel_hold_steps > 0:
            # Keep the fleet-mean-level gate as a 0-dim TENSOR and fold it into the per-env mask by broadcasting, so the comparison adds no per-step CPU/GPU sync.
            mean_level_ok = (self.terrain_levels.float().mean()
                             >= self._yaw_post_sentinel_hold_min_mean_level)
            in_hold = mean_level_ok & (self.wococo_sentinel_reached_step >= 0) & (
                (self.episode_length_buf - self.wococo_sentinel_reached_step)
                < self._yaw_post_sentinel_hold_steps)
            rew = torch.where(in_hold, torch.ones_like(rew), rew)
        return rew * self._tracking_fade_factor()

    def _reward_termination(self):
        """1.0 on a FAILURE terminal step, 0 otherwise (time-outs excluded).

        compute_reward() has always called this whenever a `termination` scale is
        set, but the method did not exist anywhere in the repo — so any config
        that switched the term on crashed with AttributeError. For a long time
        nothing set it and the latent bug never fired; `go2_teacher` sets
        termination = -150.0, so the term is live.

        Time-outs are excluded via time_out_buf: reaching the episode limit is not
        a failure, and charging for it would penalise long survival. This matches
        the intent of the comment where time_out_buf is built ("no terminal reward
        for time-outs").

        Deliberately NOT masked by the WoCoCo arm latch: the armed plant is
        precisely where failures should be charged for. Note that the latch exists
        so the dense plant shaping is not swamped by attitude penalties, so on a
        config that keeps that shaping — as the swimmer teacher does — this term
        stacks on top of it; weigh that when setting the scale. Note
        `enable_wococo_armed_term_exempt` still governs whether those
        terminations FIRE at all; this term only prices the ones that do.

        Scale note: _prepare_reward_function multiplies EVERY non-zero scale by
        dt, termination included (the `continue` there only skips adding it to
        reward_functions, since compute_reward applies it after the
        only_positive_rewards clip so it can drive the total negative). Effective
        penalty is therefore scale * dt = scale * 0.02.
        """
        return (self.reset_buf & ~self.time_out_buf).float()

    def _reward_timeout(self):
        """1.0 on a genuine CLOCK-EXPIRY terminal step, 0 otherwise.

        Deliberately NOT just `time_out_buf`. check_termination folds
        reach_goal_cutoff INTO time_out_buf (line ~941) so that finishing the
        circuit is not billed as a failure by `_reward_termination` — which means
        the raw buffer reads "timed out OR succeeded". Penalising it directly
        would charge the robot for COMPLETING the course, the exact opposite of
        the intent. Masking term_reach_goal back out leaves only "ran out of the
        20 s clock without finishing", i.e. survived but never got there:
        loitering, stalling short of the wall, or dawdling on the return.

        A soft failure, so it is meant to carry a much smaller scale than
        `termination`. check_termination runs before compute_reward, so both
        buffers are current for this step.
        """
        return (self.time_out_buf & ~self.term_reach_goal).float()


    def _reward_lin_vel_z(self):
        rew = torch.square(self.base_lin_vel[:, 2])
        # Relax the vertical-bounce penalty on obstacle terrain (jumping a hurdle needs vertical motion) and keep it full on flat terrain; flat set defaults to {17}, see _flat_terrain_mask.
        rew[~self._flat_terrain_mask()] *= 0.5
        over_pit = (self.env_class == 19) & (self.root_states[:, 0] > self.env_flyball_pit_mid)
        rew[over_pit] = 0.
        return rew

    def _reward_ang_vel_xy(self):
        rew = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        return rew

    def _reward_ang_vel_z(self):
        """Penalise base yaw rate (square of z-component of base angular velocity).

        Spec: penalty is on by default. When a wall env (env_class 15/19) has
        cur_goal_idx == 4 AND is on the left-turn arc (yaw > 0, CCW from spawn
        yaw=0), the penalty is silenced — letting the policy rotate left
        toward the -x axis without paying the quadratic yaw cost. The latch
        fires on the *exact* -x crossing: a left-turn wrap from +π to -π is
        detected as Δyaw < -π in one step. Once latched the silence is
        disabled for the rest of the episode. Both latch and yaw_prev clear
        on episode reset.

        Wococo plant stages (1-3) keep their own independent silencing for
        wococo runs; foundation never enables wococo so this branch is inert.
        """
        rew = torch.square(self.base_ang_vel[:, 2])
        wall_env = (self.env_class == 19) | (self.env_class == 15)
        # Exact -x crossing via left-turn wrap: yaw jumps from near +pi down to near -pi in one step, so delta_yaw < -pi. Right-turn wrap gives +delta and is not detected, by design.
        delta_yaw = self.yaw - self.yaw_prev
        wrapped_left = delta_yaw < -math.pi
        self.yaw_penalty_latched |= wall_env & wrapped_left
        in_return_turn = (
            wall_env
            & (self.cur_goal_idx == 4)
            & (self.yaw > 0.0)
            & (~self.yaw_penalty_latched)
        )
        if hasattr(self, 'wococo_contact_stage'):
            stage = self.wococo_contact_stage
            # Design-robust plant window: committed to the wall (plant_armed) and not yet at the sentinel. The yaw-rate penalty is silenced here because the deliberate ~90 deg body rotation is the task.
            in_plant = wall_env & self.wococo_armed & (stage < self._wococo_sentinel_stage)
        else:
            in_plant = torch.zeros_like(in_return_turn)
        silence = in_return_turn | in_plant
        rew = torch.where(silence, torch.zeros_like(rew), rew)
        self.yaw_prev = self.yaw.clone()
        return rew
     
    def _reward_orientation(self):
        rew = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        # Orientation penalty only on flat terrain (keep the gait level), zeroed on obstacle terrain where clearing a hurdle legitimately tilts the base. Flat set defaults to {17}, or {9,17} where a config lists both.
        rew[~self._flat_terrain_mask()] = 0.
        over_pit = (self.env_class == 19) & (self.root_states[:, 0] > self.env_flyball_pit_mid)
        rew[over_pit] = 0.
        return rew

    def _reward_standstill_orientation(self):
        """Positive base-level carrot while commanded to stand still.

        _reward_orientation is a PENALTY masked to the flat classes only, so on
        the cliffside wall lanes (env_class 19) there is NO attitude shaping at
        all — including through the initial_standstill hold and the post-circuit
        hard stop, the two windows where a level base is exactly the behaviour we
        want. This pays exp(-‖g_xy‖²/σ) ∈ (0,1] for a level base on EVERY terrain
        class, gated on the same standstill test the dof-error posture lock and
        the tracking carrot use (|cmd_x| < lin_vel_clip). It completes the
        standstill triple: hold position (tracking carrot), hold the default
        joint stance (dof_error boost), hold the base level (here).

        Not farmable: the zero command is set by the env, not chosen by the
        policy. The one path the policy controls is finishing the circuit into
        the hard stop — the same incentive the standstill tracking carrot already
        pays, and a wanted one.

        Silenced during the committed plant (armed, pre-sentinel), where the
        banked posture IS the task — computed inline rather than read from
        self.wococo_plant_window because that buffer is only maintained when
        exempt_penalties_at_wall_plant is on (cliffside runs it off).

        σ ≈ sin²(tilt): the 0.05 default gives ~0.86 at 5°, ~0.55 at 10°, ~0.10
        at 20°. Scale defaults to 0, so the term is popped in
        _prepare_reward_function and every other task is bit-identical.
        """
        sigma = float(getattr(self.cfg.rewards, 'standstill_orientation_sigma', 0.05))
        rew = torch.exp(-torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) / sigma)
        standstill = torch.abs(self.commands[:, 0]) < self.cfg.commands.lin_vel_clip
        in_plant = self.wococo_armed & (self.wococo_contact_stage < self._wococo_sentinel_stage)
        return rew * (standstill & ~in_plant).float()

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_collision(self):
        contact_count = torch.sum(
            1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
            dim=1
        )
        return contact_count

    def _reward_feet_contact_forces(self):
        """Penalize per-foot 3-D contact-force magnitude above the limit.

        This is the standard thresholded excess cost used in legged_gym:
        each foot contributes max(||abs(F_xyz)||_2 - limit, 0). The WoCoCo
        armed window is exempt so wall-plant and landing shaping can remain
        active without this generic contact constraint suppressing it.
        """
        force = torch.linalg.vector_norm(
            torch.abs(self.contact_forces[:, self.feet_indices, :]), dim=-1)
        threshold = float(getattr(self.cfg.rewards, 'max_contact_force', 150.0))
        excess = torch.clamp(force - threshold, min=0.0)
        armed = getattr(self, 'wococo_armed',
                        torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        return excess.sum(dim=1) * (~armed).float()

    def _reward_action_rate(self):
        return torch.norm(self.last_actions - self.actions, dim=1)

    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)
    
    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)



    def _reward_hip_pos(self):
        base = torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)
        episode_multiplier = getattr(self.cfg.rewards, 'hip_pos_episode_multiplier', None)
        if episode_multiplier is not None:
            return base * float(episode_multiplier)
        # Increase hip regularization during the active env19 plant window.
        env19_wall = ((self.env_class == 19) & self.wococo_armed
                      & (self.wococo_contact_stage < self._wococo_sentinel_stage))
        return base * torch.where(env19_wall, 5.0, 1.0)

    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        # Standstill posture lock (HEAD): ramp the deviation-from-default penalty by standstill_dof_error_scale when commanded to stand still, so the policy holds a clean default stance. Factor 1.0 -> locomotion unchanged.
        boost = float(getattr(self.cfg.rewards, 'standstill_dof_error_scale', 1.0))
        if boost != 1.0:
            standstill = torch.abs(self.commands[:, 0]) < self.cfg.commands.lin_vel_clip
            ramp_s = float(getattr(self.cfg.rewards, 'standstill_dof_error_ramp_s', 0.0))
            if ramp_s > 0.0 and hasattr(self, 'standstill_hold_steps'):
                # Ease 1.0 -> boost over ramp_s of held standstill instead of stepping there the instant cmd_x crosses lin_vel_clip: a 1 -> 0 command asks the robot to brake from a sprint, and billing the full 20x on the first braking step penalises the deceleration itself rather than the failure to settle.
                # standstill_hold_steps is reset to 0 the moment a move command
                # returns, so RELEASE is instantaneous — the ramp must never
                # linger into an acceleration, where large joint excursions are
                # exactly what is wanted.
                alpha = (self.standstill_hold_steps.float() * self.dt / ramp_s).clamp(max=1.0)
                factor = 1.0 + (boost - 1.0) * alpha
                dof_error = dof_error * torch.where(standstill, factor,
                                                    torch.ones_like(factor))
            else:
                dof_error = dof_error * torch.where(standstill, boost, 1.0)
        # Plant-window 25% scaling (wococo cliffside): the committed wall plant plus post-sentinel unwind (self.wococo_plant_window, set in check_termination) demands large joint excursions, so keep only a quarter penalty during the maneuver.
        if hasattr(self, 'wococo_plant_window'):
            dof_error = dof_error * torch.where(self.wococo_plant_window, 0.25, 1.0)
        return dof_error

    def _reward_dof_pos_limits(self):
        # L1 distance outside the soft joint limits computed from the URDF range.
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)   # lower-limit violation
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)   # upper-limit violation
        return torch.sum(out_of_limits, dim=1)

    def _reward_feet_stumble(self):
        # Intentional wall presses are exempt from the vertical-impact penalty.
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             4 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        rew = rew.float()
        at_wall_19 = (self.env_class == 19) & (self.root_states[:, 0] > self.env_flyball_pit_mid)
        at_wall_15 = (self.env_class == 15) & (self.cur_goal_idx == 4)
        exempt = at_wall_19 | at_wall_15
        if bool(getattr(self.cfg.rewards, 'exempt_penalties_at_wall_plant', False)):
            exempt = exempt | (self.wococo_armed & (self.wococo_contact_stage < self._wococo_sentinel_stage))
        rew[exempt] = 0.
        return rew

    def _reward_feet_edge(self):
        feet_pos_xy = ((self.rigid_body_states[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0]-1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1]-1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
    
        self.feet_at_edge = self.contact_filt & feet_at_edge
        rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        # env19 exemption keys on the wococo ARM LATCH, not the pit-midpoint x test _reward_feet_stumble still uses: the latch is the uniform "we are in the wall maneuver" signal (base past the liftoff AABB far edge AND tracked to the wall goal), so this arms in lock-step with the other wococo terms.
        # Consequences: the approach over the near half of the pit is exempt only after the robot commits, and the latch clears at the sentinel, so the post-plant return trip is policed again.
        at_wall_19 = ((self.env_class == 19) & self.wococo_armed
                      & (self.wococo_contact_stage < self._wococo_sentinel_stage))
        at_wall_15 = (self.env_class == 15) & (self.cur_goal_idx == 4)
        exempt = at_wall_19 | at_wall_15
        if bool(getattr(self.cfg.rewards, 'exempt_penalties_at_wall_plant', False)):
            exempt = exempt | (self.wococo_armed & (self.wococo_contact_stage < self._wococo_sentinel_stage))
        rew[exempt] = 0.
        return rew

    def _at_wall_goal(self):
        return ((self.env_class == 19) & (self.cur_goal_idx == self._flyball_wall_goal_idx)) | \
               ((self.env_class == 15) & (self.cur_goal_idx == 4))

    def _reward_wall_approach_decel(self):
        # Penalise deceleration during the wall run-up, until the robot leaves the ground to jump: active for all goals before wall contact while at least one foot is on the ground.
        # env_class 19 (flyball) contacts the wall at goal 2, so goals < 2; env_class 15 (two_hurdles) has the wall at g4 and fires only on idx in {3,4}, so the penalty misses hurdle landings (idx 1,2).
        pre_jump = ((self.env_class == 19) & (self.cur_goal_idx < self._flyball_wall_goal_idx)) | \
                   ((self.env_class == 15) & ((self.cur_goal_idx == 3) | (self.cur_goal_idx == 4)))

        feet_contact = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.0
        on_ground = feet_contact.any(dim=1)

        vel_x = self.base_lin_vel[:, 0]
        decel = torch.clamp(self.wall_approach_prev_vel - vel_x, min=0.0)
        self.wall_approach_prev_vel = vel_x.clone()

        return decel * (pre_jump & on_ground).float()

    def _reward_wall_turn_left(self):
        # Reward positive yaw rate (turning left) while at the wall goal
        at_wall_goal = self._at_wall_goal()
        rew = torch.clamp(self.base_ang_vel[:, 2], min=0.0)
        return rew * at_wall_goal.float()

    def _reward_wall_push_force(self):
        # One-shot bonus per foot for high horizontal contact force while on the wall, gated on the wall-contact goal (flyball g2 / two_hurdles g4) and terrain_level>4. Rear legs only (2=RR, 3=RL; order is FR, FL, RR, RL).
        at_wall = self._at_wall_goal() & (self.terrain_levels > 4)  # (num_envs,)
        horiz_force = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=-1)  # (num_envs, 4)
        rear_mask = torch.zeros(4, dtype=torch.bool, device=self.device)
        rear_mask[2:] = True
        eligible = at_wall.unsqueeze(1) & rear_mask.unsqueeze(0) & ~self.wall_push_rewarded & (horiz_force > 0.)
        self.wall_push_rewarded |= eligible
        return (eligible.float() * horiz_force).sum(dim=-1)

    def _reward_forbidden_zone_contact(self):
        """Penalize foot or body contact in the forbidden zone in front of the wall.
        Replaces the old physical pit: robot must jump over this zone to reach the wall.
        Active on both flyball (env_class==19) and two_hurdles_wall_turn (env_class==15);
        both terrains populate `flyball_forbidden_x0/x1` metadata in `terrain.py`."""
        active = (self.env_class == 19) | (self.env_class == 15)

        feet_x = self.rigid_body_states[:, self.feet_indices, 0]
        feet_in_zone = (feet_x >= self.env_flyball_forbidden_x0.unsqueeze(1)) & \
                       (feet_x <  self.env_flyball_forbidden_x1.unsqueeze(1))
        feet_contact = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.0
        feet_penalty = torch.sum(feet_in_zone & feet_contact, dim=1).float()

        body_x = self.rigid_body_states[:, self.penalised_contact_indices, 0]
        body_in_zone = (body_x >= self.env_flyball_forbidden_x0.unsqueeze(1)) & \
                       (body_x <  self.env_flyball_forbidden_x1.unsqueeze(1))
        body_contact = torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1
        body_penalty = torch.sum(body_in_zone & body_contact, dim=1).float()

        return active.float() * (feet_penalty + body_penalty)

    def _reward_hurdle_contact(self):
        """Penalise feet/body parts touching any hurdle. Terrain-driven: fires
        wherever a parabolic clearance volume is recorded. Hurdles are meant
        to be cleared, not stepped on. Detection requires the rigid-body origin
        to lie between the parabola's exact floor roots (15 cm before/after the
        hurdle centre) AND that body to be in soft contact. This deliberately
        shares x bounds with `_reward_hurdle_clearance`; it no longer grows the
        physical hurdle x-range with an independent margin.
        """
        bounds = self.env_flyball_hurdle_clear_region          # (N, slots, 2, 3)
        x_lo = bounds[:, :, 0, 0]
        x_hi = bounds[:, :, 1, 0]
        real_slot = x_hi > x_lo
        active = real_slot.any(dim=1)
        if not active.any():
            return torch.zeros(self.num_envs, device=self.device)

        feet_x = self.rigid_body_states[:, self.feet_indices, 0]
        feet_in_h = ((feet_x.unsqueeze(-1) >= x_lo.unsqueeze(1)) &
                     (feet_x.unsqueeze(-1) <  x_hi.unsqueeze(1)))
        feet_in_any = feet_in_h.any(dim=-1)
        feet_contact = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.0
        feet_penalty = torch.sum(feet_in_any & feet_contact, dim=1).float()

        body_x = self.rigid_body_states[:, self.penalised_contact_indices, 0]
        body_in_h = ((body_x.unsqueeze(-1) >= x_lo.unsqueeze(1)) &
                     (body_x.unsqueeze(-1) <  x_hi.unsqueeze(1)))
        body_in_any = body_in_h.any(dim=-1)
        body_contact = torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1
        body_penalty = torch.sum(body_in_any & body_contact, dim=1).float()

        return active.float() * (feet_penalty + body_penalty)

    def _init_hurdle_clearance_collision_geometry(self, asset_path):
        """Load the actor collision primitives used by parabolic clearance."""
        from legged_gym.utils.self_geometry import load_urdf_self_geometry

        geometry = load_urdf_self_geometry(
            asset_path,
            collapse_fixed_joints=self.cfg.asset.collapse_fixed_joints,
            replace_cylinder_with_capsule=self.cfg.asset.replace_cylinder_with_capsule,
            strict=True)
        body_index_map = {
            name: self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name)
            for name in sorted(set(geometry.body_names))
        }
        body_indices = geometry.body_indices(body_index_map)
        self._hurdle_clearance_primitive_body_idx = torch.as_tensor(
            body_indices, dtype=torch.long, device=self.device)
        self._hurdle_clearance_primitive_type = torch.as_tensor(
            geometry.types, dtype=torch.long, device=self.device)
        self._hurdle_clearance_primitive_size = torch.as_tensor(
            geometry.sizes, dtype=torch.float, device=self.device)
        self._hurdle_clearance_primitive_offset_pos = torch.as_tensor(
            geometry.offset_pos, dtype=torch.float, device=self.device)
        self._hurdle_clearance_primitive_offset_rot = torch.as_tensor(
            geometry.offset_rot, dtype=torch.float, device=self.device)
        print(f"[hurdle-clearance] exact collision geometry: {geometry.summary()}")

    def _reward_hurdle_clearance(self):
        """Penalise every robot collision body penetrating a parabolic prism.

        Each hurdle's x/z cross-section peaks 30 cm above its floor at the
        hurdle centre and reaches the floor 15 cm before/after the centre. It
        is extruded across the hurdle's physical lateral width plus the existing
        cfg.terrain.flyball_hurdle_clear_pad_y margin on each side.

        This is an exact convex collision calculation against every URDF
        collision primitive after applying its current rigid-body pose,
        collision-origin transform, fixed-joint collapse, and Isaac Gym's
        cylinder-to-capsule conversion. It does not approximate links by their
        origins or by bounding boxes. A broad-phase sphere only rejects
        impossible pairs; every candidate is resolved by analytical-support
        GJK against the actual parabolic volume.

        Independent of `hurdle_contact`, which keeps its own contact-based
        x-band semantics; both can be active at once.

        The collision geometry and body count are constant. Curriculum changes
        only the effective reward scale through ``hurdle_clearance_alpha``:
        zero before its sticky trigger, then a delayed linear ramp to one.

        Returns the COUNT of rigid bodies whose collision geometry overlaps at
        least one prism this step. Multiple collision primitives belonging to
        one rigid body count that body once.
        """
        clearance_alpha = float(self.hurdle_clearance_alpha)
        if getattr(self.cfg.rewards, "hurdle_clearance_per_env_draw", False):
            clearance_alpha = 1.0
            active_mask = self.hurdle_clearance_env_enabled
        if clearance_alpha <= 0.0:
            return torch.zeros(self.num_envs, device=self.device)

        bounds = self.env_flyball_hurdle_clear_region        # (N, slots, 2, 3)
        real_slot = bounds[:, :, 1, 0] > bounds[:, :, 0, 0]
        if getattr(self.cfg.rewards, "hurdle_clearance_per_env_draw", False):
            active_mask = active_mask.to(dtype=bounds.dtype)
        if not real_slot.any():
            return torch.zeros(self.num_envs, device=self.device)

        # Curriculum on the PRISM GEOMETRY, not just the reward scale: with a fixed apex, a fleet that arms at mean level 5 and then demotes must clear the full-height volume on rows whose bars are far shorter, so the keep-out stops describing the obstacle.
        # Scale the apex above the floor linearly in fleet-mean level: 0 at mean 0, nominal at hurdle_clearance_apex_full_mean_level. x roots and y extent are untouched.
        _apex_full = getattr(self.cfg.rewards, 'hurdle_clearance_apex_full_mean_level', None)
        if _apex_full is not None:
            _apex_scale = float((self.terrain_levels.float().mean()
                                 / max(float(_apex_full), 1e-6)).clamp(0.0, 1.0))
            if _apex_scale <= 0.0:
                return torch.zeros(self.num_envs, device=self.device)
            bounds = bounds.clone()
            _z_lo = bounds[:, :, 0, 2]
            bounds[:, :, 1, 2] = _z_lo + (bounds[:, :, 1, 2] - _z_lo) * _apex_scale

        from legged_gym.utils.parabolic_clearance import (
            collision_primitive_world_poses,
            primitive_parabolic_prism_overlaps,
        )
        primitive_pos, primitive_rot = collision_primitive_world_poses(
            self.rigid_body_states[:, :, :3],
            self.rigid_body_states[:, :, 3:7],
            self._hurdle_clearance_primitive_body_idx,
            self._hurdle_clearance_primitive_offset_pos,
            self._hurdle_clearance_primitive_offset_rot)
        primitive_hit = primitive_parabolic_prism_overlaps(
            primitive_pos, primitive_rot,
            self._hurdle_clearance_primitive_type,
            self._hurdle_clearance_primitive_size,
            bounds).any(dim=2)                               # (N, primitives)

        body_hit_count = torch.zeros(
            (self.num_envs, self.num_bodies), dtype=torch.int32,
            device=self.device)
        body_indices = self._hurdle_clearance_primitive_body_idx.unsqueeze(0).expand(
            self.num_envs, -1)
        body_hit_count.scatter_add_(1, body_indices, primitive_hit.to(torch.int32))
        _count = (body_hit_count > 0).sum(dim=1).float()
        if getattr(self.cfg.rewards, "hurdle_clearance_per_env_draw", False):
            _count = _count * self.hurdle_clearance_env_enabled.float()
        return clearance_alpha * _count

    def _reward_wall_multi_touch(self):
        """Penalize each foot that touches the wall more than once per episode.
        A 'touch' is a rising edge: foot transitions from off-wall to on-wall.
        The penalty fires at the moment of the 2nd, 3rd, ... touch per foot.

        Both flyball (env_class==19) and two_hurdles_wall_turn (env_class==15)
        populate `flyball_pit_x1` metadata (env15 collapses pit_x0/x1 to wall_front_x
        in terrain.py:1272-1273), so the same precise position-based check
        (`feet_x >= pit_x1` AND in contact) works on both.

        cfg.rewards.wall_multi_touch_min_level (default 0) gates the penalty:
        only envs whose terrain_level is strictly > min_level can incur it.
        Foundation sets this to 4 so easy rows (0-4) train without the
        penalty pressure and only harder rows (5+) require single-touch
        plants."""
        min_level = float(getattr(self.cfg.rewards, 'wall_multi_touch_min_level', 0.0))
        feet_pos_x = self.rigid_body_states[:, self.feet_indices, 0]          # (num_envs, 4)
        feet_contact = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.0

        wall_terrain = (self.env_class == 19) | (self.env_class == 15)
        level_ok     = self.terrain_levels.float() > min_level
        active_env   = (wall_terrain & level_ok).unsqueeze(1)
        on_wall = (feet_pos_x >= self.env_flyball_pit_x1.unsqueeze(1)) & feet_contact & active_env

        rising_edge = on_wall & ~self.wall_feet_prev_on_wall                   # new touches this step
        extra_touch = rising_edge & (self.wall_feet_touch_count >= 1)          # 2nd+ touch per foot

        self.wall_feet_touch_count += rising_edge.long()
        self.wall_feet_prev_on_wall = on_wall.clone()

        return extra_touch.sum(dim=1).float()

    def _reward_wall_feet_alignment(self):
        # Every-step bonus for FL/FR + RL/RR lateral (y) alignment at the wall waypoint (goal 2 on env_class 19 flyball_box, goal 4 on env_class 15 two_hurdles_wall_turn).
        # Decay arms after the first step the unscaled bonus crosses an alignment threshold (i.e. the reward was actually earned), after which the bonus is weighted exp(-k/tau), so the per-episode payout is bounded and dwelling at the waypoint can't farm it. Latch is per-episode.
        wall_goal_active = (
            ((self.env_class == 19) & (self.cur_goal_idx == self._flyball_wall_goal_idx))
            | ((self.env_class == 15) & (self.cur_goal_idx == 4))
        )
        if not wall_goal_active.any():
            return torch.zeros(self.num_envs, device=self.device)

        feet_y = self.rigid_body_states[:, self.feet_indices, 1]  # (N, 4)
        front_dy = torch.abs(feet_y[:, 0] - feet_y[:, 1])
        back_dy  = torch.abs(feet_y[:, 2] - feet_y[:, 3])
        sigma = 0.05
        front_align = torch.exp(-front_dy / sigma)
        back_align  = torch.exp(-back_dy / sigma)

        # Optional per-pair gate: a pair's alignment bonus fires only when both its feet are physically on the wall (feet_x >= pit_x1 AND contact > 1 N); without it the bonus also fires during the approach, encouraging tight lateral alignment everywhere.
        if bool(getattr(self.cfg.rewards, 'wall_feet_alignment_on_wall_only', False)):
            feet_x = self.rigid_body_states[:, self.feet_indices, 0]
            feet_in_wall_x = feet_x >= self.env_flyball_pit_x1.unsqueeze(1)
            feet_contact = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.0
            on_wall = feet_in_wall_x & feet_contact
            front_pair_on_wall = on_wall[:, 0] & on_wall[:, 1]
            back_pair_on_wall  = on_wall[:, 2] & on_wall[:, 3]
            front_align = torch.where(front_pair_on_wall, front_align, torch.zeros_like(front_align))
            back_align  = torch.where(back_pair_on_wall,  back_align,  torch.zeros_like(back_align))

        raw = front_align + back_align

        # Threshold: at least one pair aligned to ~sigma gives exp(-1) ~= 0.37, so raw >= 1.0 means one pair is solidly aligned -- the moment the reward counts as earned and the decay clock starts.
        earn_threshold = 1.0
        new_earn = wall_goal_active & (raw >= earn_threshold) & ~self.wall_align_earned_once
        self.wall_align_earned_once = self.wall_align_earned_once | new_earn

        # Linear decay to zero over 2 s post-latch. At decimation 4 / sim dt
        # 0.005 → 50 Hz, so 100 steps == 2 s. Cumulative payout post-latch is
        # bounded by ~0.5 * 2 * decay_steps = decay_steps bonus-units (≈100).
        decay_steps = 100.0
        decay = torch.clamp(
            1.0 - self.wall_align_steps_since_earn.float() / decay_steps,
            min=0.0,
        )
        # Pre-latch envs see decay==1 (counter still at 0), post-latch envs see exp(-k/tau); increment after using so the latch step itself pays the full bonus.
        rew = raw * decay
        self.wall_align_steps_since_earn = torch.where(
            self.wall_align_earned_once,
            self.wall_align_steps_since_earn + 1,
            self.wall_align_steps_since_earn,
        )
        return rew * wall_goal_active.float()

    @staticmethod
    def _in_region(pos, region):
        """Return True for each position inside the AABB.

        pos:    (N, K, 3)
        region: (N, 2, 3)  — region[:, 0, :] = min corner, region[:, 1, :] = max corner
        returns: (N, K) bool
        """
        lo = region[:, 0:1, :]   # (N, 1, 3)
        hi = region[:, 1:2, :]   # (N, 1, 3)
        return ((pos >= lo) & (pos <= hi)).all(dim=-1)

    def _near_target(self, feet_pos, targets, radius):
        """Per-foot isotropic region gate (cliffside-mirror): foot f is 'in its
        region' iff within `radius` of ITS OWN target. A radius beats an AABB on
        the banked wall face (no corner leakage) and stays definitionally
        consistent with the dense exp(-d/kappa) term (same centre).
        feet_pos (N,K,3), targets (N,K,3) -> (N,K) bool. `radius` may be a float
        or a per-env/per-foot tensor broadcastable to (N,K)."""
        return torch.norm(feet_pos - targets, dim=-1) <= radius

    # Per-wall-angle wococo radii / kappas: (attr, n_feet, cfg fallback key, hard default). Authored per angle in scripts/foothold_editor.html and stored as cfg.rewards.wococo_per_angle_radii; the tuple ORDER in that table must match this list.
    _WOCOCO_ANGLE_RADII_FIELDS = (
        ('wococo_r_wall',   4, 'wococo_foot_region_radius',                0.18),
        ('wococo_k_wall',   4, 'wococo_foot_target_kappa_oneshot',         0.12),
        ('wococo_k_track',  1, 'wococo_foot_track_kappa',                  0.30),
        ('wococo_k_orient', 1, 'wococo_foot_target_kappa',                 0.20),
        ('wococo_r_land',   2, 'wococo_landing_foot_region_radius',        0.35),
        ('wococo_k_land',   2, 'wococo_landing_foot_target_kappa_oneshot', 0.12),
    )

    def _init_wococo_angle_radii(self):
        """Parse the nominal and optional low-level angle tables.

        ``wococo_per_angle_radii_low`` is selected per environment below
        ``wococo_per_angle_radii_low_below_level``.  This keeps the E12
        bootstrap gates on levels 0--3 without loosening the mature rows.
        """
        def parse(tbl):
            if not tbl:
                return None
            rows = sorted((float(k), v) for k, v in tbl.items())
            th = np.array([r[0] for r in rows], dtype=np.float64)
            cols = []
            for j, (name, k, _, _) in enumerate(self._WOCOCO_ANGLE_RADII_FIELDS):
                per_angle = []
                for _, vals in rows:
                    v = np.asarray(vals[j], dtype=np.float64)
                    if v.ndim == 0:
                        v = np.full(k, float(v))
                    elif v.shape != (k,):
                        raise ValueError(f"wococo_per_angle_radii field {name} needs a "
                                         f"scalar or {k} per-foot values, got {v.shape}")
                    per_angle.append(v)
                cols.append(np.stack(per_angle))
            return th, cols

        self._wococo_angle_radii = None
        self._wococo_angle_radii = parse(
            getattr(self.cfg.rewards, 'wococo_per_angle_radii', None))
        self._wococo_angle_radii_low = parse(
            getattr(self.cfg.rewards, 'wococo_per_angle_radii_low', None))
        if self._wococo_angle_radii is None and self._wococo_angle_radii_low is None:
            return
        th, cols = self._wococo_angle_radii or self._wococo_angle_radii_low
        # Printed once at env construction, next to the foothold-table banner.
        print(f"[wococo] PER-ANGLE RADII: {len(th)} rows, angles "
              f"{[float(t) for t in th]} deg (linear interp, ends clamp)")
        for (name, k, cfg_key, _), tab in zip(self._WOCOCO_ANGLE_RADII_FIELDS, cols):
            print(f"[wococo]   {cfg_key:<41} {tab.min():.3f} .. {tab.max():.3f}"
                  f"   (overrides the cfg scalar)")

    def _refresh_wococo_angle_radii(self):
        """Resample the table at each env's own wall angle. Called wherever
        env_flyball_wall_angle_deg is (re)assigned — init, curriculum promote/
        demote, set_terrain_levels — never per step."""
        if not hasattr(self, '_wococo_angle_radii'):
            self._init_wococo_angle_radii()
        if (self._wococo_angle_radii is None
                and getattr(self, '_wococo_angle_radii_low', None) is None):
            return
        if not hasattr(self, 'env_flyball_wall_angle_deg'):
            return
        ang = self.env_flyball_wall_angle_deg.detach().cpu().numpy().astype(np.float64)
        low = getattr(self, '_wococo_angle_radii_low', None)
        low_below = getattr(self.cfg.rewards,
                            'wococo_per_angle_radii_low_below_level', 0)
        low_mask = (self.terrain_levels < int(low_below)).cpu().numpy()
        for j, (name, k, cfg_key, default) in enumerate(self._WOCOCO_ANGLE_RADII_FIELDS):
            if self._wococo_angle_radii is not None:
                th, cols = self._wococo_angle_radii
                out = np.stack([np.interp(ang, th, cols[j][:, c])
                                for c in range(k)], axis=-1)
            else:
                scalar = getattr(self.cfg.rewards, cfg_key, None)
                out = np.full((self.num_envs, k),
                              float(default if scalar is None else scalar))
            if low is not None:
                th_low, cols_low = low
                low_out = np.stack([np.interp(ang, th_low, cols_low[j][:, c])
                                    for c in range(k)], axis=-1)
                out[low_mask] = low_out[low_mask]
            t = torch.from_numpy(out).to(self.device).float()
            setattr(self, name, t[:, 0] if k == 1 else t)

    def _wococo_rad(self, attr, cfg_key, default, fallback_key=None):
        """Per-angle tensor for `attr` when the table is active, else the cfg
        scalar (so non-cliffside wococo configs behave exactly as before)."""
        t = getattr(self, attr, None)
        if t is not None:
            return t
        v = getattr(self.cfg.rewards, cfg_key, None)
        if v is None and fallback_key is not None:
            v = getattr(self.cfg.rewards, fallback_key, None)
        return float(default if v is None else v)

    @staticmethod
    def _rad_feet(v, sl):
        """Slice a per-foot radius tensor to a foot subset; scalars pass through."""
        return v[:, sl] if torch.is_tensor(v) and v.dim() == 2 else v

    @staticmethod
    def _rad_bcast(v):
        """Shape a per-env (N,) kappa for division against per-foot (N,K) dists."""
        return v.unsqueeze(-1) if torch.is_tensor(v) else v

    # P4L posture-only landing (cfg.rewards.wococo_landing_posture_only) replaces the stage-2 foothold plan with a BASE-STATE gate on attitude, height and return speed: no foothold placement, no per-foot contact plan, no n_wrong, and n_con = 0.
    # Regulating which feet land where confined the search for a good landing; the stage is kept only to shape the RETURN direction. These helpers feed the four per-stage extension points in _wococo_contact_pts_few_stage plus _reward_wococo_base_orient.

    def _wococo_landing_posture_only(self):
        return bool(getattr(self.cfg.rewards, 'wococo_landing_posture_only', False))

    def _wococo_landing_base_height(self):
        """Base height above the floor the landing happens on.

        The front-foot landing targets are written AT platform_height
        (flyball_wall.py), so their z IS the tile's floor datum. We keep that z
        and discard their xy — the placement constraint this design removes.
        Falls back to raw world z (platform_height defaults to 0.0 on flyball)
        when the array is absent.
        """
        if hasattr(self, 'env_flyball_landing_foot_target'):
            floor_z = self.env_flyball_landing_foot_target[:, :, 2].mean(dim=1)
            return self.root_states[:, 2] - floor_z
        return self.root_states[:, 2]

    def _wococo_landing_return_vec(self):
        """(N,2) unit vector from the base toward the first waypoint of the
        RETURN leg, indexed by cfg.terrain.flyball_return_goal_idx.

        Layout-indexed on purpose. NOT target_pos_rel: under the legacy 8-goal
        layout the current goal at stage 2 is wp_land, the waypoint under the
        robot's feet, whose bearing is meaningless inside next_goal_threshold.
        NOT next_target_pos_rel either: under the compact swimmer layout the
        return waypoint IS the current goal, so 'next' overshoots it by one.
        """
        rel = (self.env_goals[:, self._flyball_return_goal_idx, :2]
               - self.root_states[:, :2])
        return rel / (torch.norm(rel, dim=-1, keepdim=True) + 1e-5)

    def _wococo_landing_return_yaw_err(self):
        """|yaw - bearing of the return leg|, wrapped to [0, pi]."""
        d = self._wococo_landing_return_vec()
        return wrap_to_pi(torch.atan2(d[:, 1], d[:, 0]) - self.yaw).abs()

    def _wococo_landing_return_vel(self):
        """Base xy velocity projected on the return bearing. SIGNED: bouncing
        back toward the wall reads negative, so it can never pass the gate."""
        return (self._wococo_landing_return_vec()
                * self.root_states[:, 7:9]).sum(dim=-1)

    def _wococo_landing_posture_terms(self):
        """(roll_ok, pitch_ok, yaw_ok) for the stage-2 gate. Per axis so a large
        yaw error cannot hide behind a small roll one, and so the diagnostics can
        report WHICH axis is blocking. Yaw is measured from the RETURN bearing,
        not a hard-coded 180 deg, so it follows the goal layout."""
        r_tol = math.radians(float(getattr(
            self.cfg.rewards, 'wococo_landing_roll_tol_deg', 25.0)))
        p_tol = math.radians(float(getattr(
            self.cfg.rewards, 'wococo_landing_pitch_tol_deg', 35.0)))
        y_tol = math.radians(float(getattr(
            self.cfg.rewards, 'wococo_landing_yaw_tol_deg', 70.0)))
        return ((self.roll.abs() < r_tol),
                (self.pitch.abs() < p_tol),
                (self._wococo_landing_return_yaw_err() < y_tol))

    def _wococo_landing_posture_ok(self):
        roll_ok, pitch_ok, yaw_ok = self._wococo_landing_posture_terms()
        return roll_ok & pitch_ok & yaw_ok

    def _wococo_landing_support_terms(self):
        """(height_ok, speed_ok) — the stage-2 task condition (the FSM's F_con):
        supported, and running home.

        h_lo rejects a belly-slide (the base sits at ~0.08-0.10 m when the trunk
        is down). h_hi bounds how long a BALLISTIC arc can satisfy the band: a
        body entering it from above dwells at most sqrt(2*(h_hi-h_lo)/g) =
        0.202 s = 10.1 steps for [0.15, 0.35], against wococo_landing_n_persist
        = 15. That is what stops the stage completing in mid-air off the wall,
        so h_hi MUST NOT be raised without also raising n_persist.

        The speed term is the same projection _reward_tracking_goal_vel saturates
        on, at wococo_landing_speed_frac of the command — it is signed, so
        bouncing back toward the wall can never pass.
        """
        h_lo = float(getattr(self.cfg.rewards, 'wococo_landing_h_lo', 0.15))
        h_hi = float(getattr(self.cfg.rewards, 'wococo_landing_h_hi', 0.35))
        frac = float(getattr(self.cfg.rewards, 'wococo_landing_speed_frac', 0.8))
        base_h = self._wococo_landing_base_height()
        return ((base_h > h_lo) & (base_h < h_hi),
                self._wococo_landing_return_vel() >= frac * self.commands[:, 0])

    def _wococo_landing_support_ok(self):
        height_ok, speed_ok = self._wococo_landing_support_terms()
        return height_ok & speed_ok

    def _wococo_landing_accept_radius(self):
        """P4L stage-2 landing foothold acceptance radius, level-gated.

        Base value is the per-angle table entry (wococo_r_land) or the cfg scalar.
        Rows at/above wococo_landing_region_shrink_from_level get it multiplied by
        wococo_landing_region_shrink_factor, so the precision demanded of the
        landing tightens with the curriculum instead of staying at the bootstrap
        width. Returns a float when the gate is off and the base is scalar, else
        an (N,2) tensor — both are valid inputs to _near_target.

        NOTE the returned radius defines BOTH sides of the landing plan: a front
        foot inside it and loaded earns n_corr credit, and a front foot loaded
        OUTSIDE it counts as n_wrong. Shrinking therefore both withholds the
        jackpot and starts billing landings that used to pass.
        """
        rad = self._wococo_rad('wococo_r_land',
                               'wococo_landing_foot_region_radius', 0.35)
        from_level = getattr(self.cfg.rewards,
                             'wococo_landing_region_shrink_from_level', None)
        if from_level is None:
            return rad
        factor = float(getattr(self.cfg.rewards,
                               'wococo_landing_region_shrink_factor', 0.5))
        # (N,1) multiplier: 1.0 on the easy rows, `factor` from the threshold up.
        mult = torch.where(self.terrain_levels >= int(from_level),
                           torch.full_like(self.terrain_levels, factor,
                                           dtype=torch.float),
                           torch.ones(self.num_envs, device=self.device))
        if not torch.is_tensor(rad):
            rad = torch.full((self.num_envs, 2), float(rad), device=self.device)
        elif rad.dim() == 1:
            rad = rad.unsqueeze(-1).expand(-1, 2)
        return rad * mult.unsqueeze(-1)

    def _wococo_arm(self):
        """Latch self.wococo_armed once an env has committed to the wall
        plant: base x past the FAR (wall-side) edge of the liftoff AABB AND the
        robot has tracked to (at least) the wall goal. Stays armed for the rest
        of the episode (cleared on reset / disarmed at the sentinel stage). This
        is the SINGLE uniform 'we are in the wall maneuver' latch — every wococo
        shaping term (foot_track / base_orient / foot_target / contact_pts
        jackpot) AND the contact_pts n_wrong penalty gate on it, so reward and
        penalty arm in exact lock-step."""
        wococo_env = (self.env_class == 19) | (self.env_class == 15)
        wall_goal_idx = torch.where(self.env_class == 15,
                                    self._wococo_goal_idx_4,
                                    self._wococo_wall_goal_idx_19)
        # Plant-arm trigger x: the liftoff AABB far edge, optionally shifted toward the wall by wococo_arm_x_offset_m so the robot must get closer before the plant reward latches (anti stop-short-and-farm).
        arm_x = self.env_flyball_liftoff_region[:, 1, 0] + float(getattr(self.cfg.rewards, 'wococo_arm_x_offset_m', 0.0))
        committed = (self.root_states[:, 0] >= arm_x) & (self.cur_goal_idx >= wall_goal_idx)
        self.wococo_armed |= wococo_env & committed
        # Disarm once the FSM reaches the sentinel (return trip): the post-plant locomotion home should not be gated by the plant latch.
        self.wococo_armed &= (self.wococo_contact_stage < self._wococo_sentinel_stage)

    def _wococo_active_feet_mask(self, stage=None):
        """(N,4) bool: which feet are the 'active' plant feet at the given
        stage for the running design (defaults to the current contact stage).
        Used by the dense foot-track term and the deferred foot-target payout
        (which queries the *previous* stage's active feet on advance).
          P3: stage1→front(FL,FR), stage2→rear(RL,RR), stage3(landing)→front;
              elsewhere none.
          P4/P2: stage0→front, stage1→rear; elsewhere none.
          P4L: stage0→front, stage1→rear, stage2(landing)→front; elsewhere none.
          P1: stage0→all four; elsewhere none."""
        if stage is None:
            stage = self.wococo_contact_stage
        mask = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        front = torch.tensor([True, True, False, False], device=self.device)
        rear  = torch.tensor([False, False, True, True], device=self.device)
        allf  = torch.ones(4, dtype=torch.bool, device=self.device)
        d = self._wococo_design
        if d == 'P1':
            mask[stage == 0] = allf
        elif d in ('P2', 'P4'):
            mask[stage == 0] = front
            mask[stage == 1] = rear
        elif d == 'P4L':
            mask[stage == 0] = front
            mask[stage == 1] = rear
            if not self._wococo_landing_posture_only():
                mask[stage == 2] = front   # landing: front feet on the floor
            # posture_only: the landing plans NO feet, so foot_track pays nothing at stage 2 and base_orient's landing branch gates on the stage directly.
        else:  # P3 (and any unknown design): keep the 4-stage front/rear/landing
            mask[stage == 1] = front
            mask[stage == 2] = rear
            mask[stage == 3] = front
        return mask

    def _reward_wococo_foot_target(self):
        """ONE-SHOT, contact-gated per-foot foothold reward (canonical WoCoCo
        precision term). For each foot f, exp(-||foot_f - target_f|| / kappa) is
        banked exactly ONCE per episode — at the moment of that foot's FIRST hard
        wall contact (force > thresh, x >= pit_x1) during the plant phase
        (plant_armed). Hovering pays nothing (contact gate); lingering pays
        nothing more (latch); re-touches pay nothing (single-touch). NOTE scales
        are dt-multiplied so a one-shot needs a large cfg scale (~50) to matter.
        """
        if not hasattr(self, 'env_flyball_foot_target'):
            return torch.zeros(self.num_envs, device=self.device)
        kappa        = self._wococo_rad('wococo_k_wall', 'wococo_foot_target_kappa_oneshot',
                                        0.12, 'wococo_foot_target_kappa')
        force_thresh = float(getattr(self.cfg.rewards, 'wococo_foot_target_force_thresh', 20.0))
        plant_phase = ((self.env_class == 19) | (self.env_class == 15)) & self.wococo_armed
        feet_pos = self.rigid_body_states[:, self.feet_indices, :3]              # (N,4,3)
        force    = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)  # (N,4)
        on_wall  = (force > force_thresh) & \
                   (feet_pos[..., 0] >= self.env_flyball_pit_x1.unsqueeze(1))    # (N,4)
        trigger  = plant_phase.unsqueeze(1) & on_wall & ~self.wococo_foot_target_paid
        d        = torch.norm(feet_pos - self.env_flyball_foot_target, dim=-1)   # (N,4)
        per_foot = torch.exp(-d / kappa) * trigger.float()                       # paid once
        self.wococo_foot_target_paid = self.wococo_foot_target_paid | trigger
        return per_foot.sum(dim=-1)

    def _reward_wococo_landing_foot_target(self):
        """ONE-SHOT, contact-gated per-foot LANDING foothold reward (P4L floor
        analogue of _reward_wococo_foot_target). For each FRONT foot f in {FL,FR},
        exp(-||foot_f - landing_target_f|| / kappa) is banked exactly ONCE per
        episode — at the moment of that foot's FIRST contact above the configured
        force threshold while the FSM is in the LANDING stage (stage 2, armed).
        Hovering / lingering / re-touches pay nothing. dt-multiplied, so needs a
        large cfg scale (~25) to matter. Inert off P4L."""
        if getattr(self, '_wococo_design', '') != 'P4L' or not hasattr(self, 'env_flyball_landing_foot_target'):
            return torch.zeros(self.num_envs, device=self.device)
        kappa = self._wococo_rad('wococo_k_land', 'wococo_landing_foot_target_kappa_oneshot',
                                 0.12, 'wococo_foot_target_kappa_oneshot')
        force_thresh = float(getattr(self.cfg.rewards, 'wococo_landing_contact_force_threshold', 15.0))
        landing_phase = (((self.env_class == 19) | (self.env_class == 15))
                         & (self.wococo_contact_stage == 2) & self.wococo_armed)
        feet_pos = self.rigid_body_states[:, self.feet_indices[:2], :3]              # (N,2,3) FRONT
        force    = torch.norm(self.contact_forces[:, self.feet_indices[:2], :], dim=-1)  # (N,2)
        in_contact = force > force_thresh                                           # (N,2)
        trigger  = landing_phase.unsqueeze(1) & in_contact & ~self.wococo_landing_foot_target_paid
        d        = torch.norm(feet_pos - self.env_flyball_landing_foot_target, dim=-1)   # (N,2)
        per_foot = torch.exp(-d / kappa) * trigger.float()                          # paid once
        self.wococo_landing_foot_target_paid = self.wococo_landing_foot_target_paid | trigger
        return per_foot.sum(dim=-1)

    def _wococo_p4l_landing_gate(self):
        """Per-env gate shared by P4L landing-only force shaping terms."""
        if getattr(self, '_wococo_design', '') != 'P4L':
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        stage = getattr(self, '_wococo_force_reward_stage', self.wococo_contact_stage)
        return (((self.env_class == 19) | (self.env_class == 15))
                & (stage == 2) & self.wococo_armed)

    def _reward_wococo_landing_z_force(self):
        """Quadratic front-foot penalty for vertical landing force above threshold."""
        gate = self._wococo_p4l_landing_gate()
        threshold = float(getattr(self.cfg.rewards,
                                  'wococo_landing_z_force_threshold', 100.0))
        norm = max(float(getattr(self.cfg.rewards,
                                 'wococo_landing_force_normalization', 100.0)), 1e-6)
        front_fz = self.contact_forces[:, self.feet_indices[:2], 2]
        excess = torch.relu((front_fz - threshold) / norm)
        return gate.float() * excess.square().sum(dim=-1)

    def _reward_wococo_landing_positive_x_force(self):
        """Signed quadratic/leaky cost on world-X force at landing front feet.

        With a negative reward scale, positive Fx is penalized quadratically
        while negative Fx receives a leaky linear credit toward the -X return.
        """
        gate = self._wococo_p4l_landing_gate()
        norm = max(float(getattr(self.cfg.rewards,
                                 'wococo_landing_force_normalization', 100.0)), 1e-6)
        slope = float(getattr(self.cfg.rewards,
                              'wococo_landing_x_leaky_slope', 0.01))
        front_fx = self.contact_forces[:, self.feet_indices[:2], 0]
        normalized_fx = front_fx / norm
        cost = torch.where(normalized_fx >= 0.0,
                           normalized_fx.square(),
                           slope * normalized_fx)
        return gate.float() * cost.sum(dim=-1)

    def _reward_wococo_base_orient(self):
        """Cliffside w_base analog, POSITIVE form: exp(-|attitude error|/0.5)
        toward the banked turn posture, proximity-gated by the active feet's
        nearness to their targets. Positive shaping (not the paper's -|err|
        penalty) so ENTERING the plant always pays — a pure -|err| made the
        cold-start net-negative and the robot avoided the plant. The hard
        roll/yaw gates still drive ADVANCE inside the contact FSM.

        Design-robust: the 'front-pair / rear-pair' posture targets key off the
        active-feet mask (front pair → bank roll to the wall angle; rear pair →
        |yaw| ~ 90 deg), so it works for the 4-stage (P3) and 2-stage (P1/P2/P4)
        FSMs alike."""
        wococo_env = (self.env_class == 19) | (self.env_class == 15)
        active = self._wococo_active_feet_mask()                                  # (N,4)
        front_active = active[:, :2].any(dim=-1) & ~active[:, 2:].any(dim=-1)
        rear_active  = active[:, 2:].any(dim=-1)
        gate = wococo_env & self.wococo_armed & (front_active | rear_active)
        kappa = self._rad_bcast(self._wococo_rad('wococo_k_orient',
                                                 'wococo_foot_target_kappa', 0.2))
        feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
        err  = torch.zeros(self.num_envs, device=self.device)
        prox = torch.zeros(self.num_envs, device=self.device)
        # Per-env output multiplier, 1.0 everywhere except the posture-only landing (wococo_landing_base_orient_gain). Applied here rather than on cfg.rewards.scales.wococo_base_orient, which is shared with the stage-0/1 wall-plant branches and must not move.
        gain = torch.ones(self.num_envs, device=self.device)
        if front_active.any():
            roll_target = -torch.deg2rad(self.env_flyball_wall_angle_deg)
            err = torch.where(front_active, (self.roll - roll_target).abs(), err)
            d_front = torch.norm(feet_pos[:, :2, :] - self.env_flyball_foot_target[:, :2, :], dim=-1)
            prox = torch.where(front_active, torch.exp(-d_front / kappa).mean(dim=-1), prox)
        if rear_active.any():
            yaw_err = (self.yaw.abs() - math.pi / 2.0).abs()
            err = torch.where(rear_active, yaw_err, err)
            d_rear = torch.norm(feet_pos[:, 2:, :] - self.env_flyball_foot_target[:, 2:, :], dim=-1)
            prox = torch.where(rear_active, torch.exp(-d_rear / kappa).mean(dim=-1), prox)
        # P4L stage-2 LANDING orientation shaping: front-active, but the posture target is zero roll plus complete turnaround and proximity is to the FLOOR landing footholds. Overrides the front-plant branch above.
        if self._wococo_design == 'P4L' and self._wococo_landing_posture_only():
            # Posture-only landing: stage 2 plans NO feet, so those envs are not in `gate` above (which keys off the active-feet mask) and need their own gate term.
            # Shape ONLY what the live tracking rewards do not cover at stage 2: tracking_yaw carries the return HEADING and tracking_goal_vel the return SPEED, both at higher scale. What nothing else pays for on env_class 19 is ROLL/PITCH (_reward_orientation is zeroed off the flat classes) and BASE HEIGHT.
            land = wococo_env & self.wococo_armed & (self.wococo_contact_stage == 2)
            gate = gate | land
            if land.any():
                w_p  = float(getattr(self.cfg.rewards, 'wococo_landing_pitch_weight', 1.0))
                w_h  = float(getattr(self.cfg.rewards, 'wococo_landing_height_weight', 10.0))
                h_lo = float(getattr(self.cfg.rewards, 'wococo_landing_h_lo', 0.15))
                # One-sided in height: being BELOW h_lo (trunk down) is shaped, being above h_hi is NOT -- the upper bound only bounds ballistic dwell inside the gate, and penalising height would penalise the push-off arc.
                shortfall = torch.clamp(
                    h_lo - self._wococo_landing_base_height(), min=0.)
                err_land = (self.roll.abs() + w_p * self.pitch.abs()
                            + w_h * shortfall)
                err  = torch.where(land, err_land, err)
                prox = torch.where(land, torch.ones_like(prox), prox)
                # Stage-2-only boost: this is the ONLY dense signal for roll / pitch / base height at the landing (tracking_yaw and tracking_goal_vel cover the other conjuncts) and it carries a 15-step pause-on-error hold, so it is worth more per step than at the wall stages.
                _g = float(getattr(self.cfg.rewards,
                                   'wococo_landing_base_orient_gain', 1.0))
                gain = torch.where(land, torch.full_like(gain, _g), gain)
        elif self._wococo_design == 'P4L' and hasattr(self, 'env_flyball_landing_foot_target'):
            land = wococo_env & (self.wococo_contact_stage == 2)
            if land.any():
                err_land = self.roll.abs() + (self.yaw.abs() - math.pi).abs()
                err = torch.where(land, err_land, err)
                d_land = torch.norm(feet_pos[:, :2, :] - self.env_flyball_landing_foot_target, dim=-1)
                prox = torch.where(land, torch.exp(-d_land / kappa).mean(dim=-1), prox)
        return torch.exp(-err / 0.5) * prox * gate.float() * gain

    def _reward_wococo_foot_track(self):
        """DENSE per-step per-foot tracking reward (new): for the feet that are
        active at the current stage, Σ exp(-||foot_f - target_f|| / kappa_dense).
        Gated on plant_armed. Provides the long-range pull onto the
        per-foot foothold goals that the (tight) one-shot term cannot. kappa_dense from
        cfg.rewards.wococo_foot_track_kappa (default 0.30)."""
        if not hasattr(self, 'env_flyball_foot_target'):
            return torch.zeros(self.num_envs, device=self.device)
        kappa = self._rad_bcast(self._wococo_rad('wococo_k_track',
                                                 'wococo_foot_track_kappa', 0.30))
        wococo_env = (self.env_class == 19) | (self.env_class == 15)
        gate = wococo_env & self.wococo_armed
        active = self._wococo_active_feet_mask().float()                          # (N,4)
        feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
        target = self.env_flyball_foot_target
        # P4L landing (stage 2): the FRONT feet track the FLOOR landing footholds instead of the wall footholds, so the same dense Gaussian pull that shapes the plant pulls them onto the landing spots.
        if self._wococo_design == 'P4L' and hasattr(self, 'env_flyball_landing_foot_target'):
            land = (self.wococo_contact_stage == 2).view(-1, 1, 1)                # (N,1,1)
            front_tgt = torch.where(land, self.env_flyball_landing_foot_target, target[:, :2, :])
            target = torch.cat([front_tgt, target[:, 2:, :]], dim=1)
        d = torch.norm(feet_pos - target, dim=-1)                                 # (N,4)
        per_foot = torch.exp(-d / kappa) * active
        return per_foot.sum(dim=-1) * gate.float()


    def _wococo_stage_anneal_factor(self):
        """Latched fleet-mean-level anneal on the one-shot stage bonus.

        The bonus's measured value is curriculum onset, which accrues below the
        anneal window, while its measured cost is a rushed, cue-locked plant
        trigger whose residual gradient keeps optimizing timing after the task
        is solved (median plant-to-turn dwell 0.86 s at w=25 vs 1.32 s at w=0).
        The factor ramps 1 -> final_frac linearly on the FLEET-MEAN terrain
        level between start and end and LATCHES at its running minimum, so a
        later demotion never resurrects the bonus and a resumed run re-derives
        the correct value from the restored levels.
        wococo_stage_anneal_start_mean_level = None disables (factor 1.0).
        """
        start = getattr(self.cfg.rewards,
                        'wococo_stage_anneal_start_mean_level', None)
        if start is None:
            return 1.0
        end = float(getattr(self.cfg.rewards,
                            'wococo_stage_anneal_end_mean_level',
                            float(start) + 2.0))
        final = float(getattr(self.cfg.rewards,
                              'wococo_stage_anneal_final_frac', 0.0))
        mean_lvl = float(self.terrain_levels.float().mean().item())
        if end <= float(start):
            cand = final if mean_lvl >= float(start) else 1.0
        else:
            t = (mean_lvl - float(start)) / (end - float(start))
            cand = 1.0 - min(max(t, 0.0), 1.0) * (1.0 - final)
        cand = min(getattr(self, '_wococo_stage_anneal_latched', 1.0), cand)
        self._wococo_stage_anneal_latched = cand
        return cand

    def _reward_wococo_stage(self):
        """WoCoCo r_stage. Three modes selectable via cfg.rewards.wococo_stage_reward_mode:

          "per_step"       — Eq. 4: n_stage · F_task per step (the original).
          "oneshot_const"  — +1 the moment stage ever exceeds the per-episode max,
                             zero otherwise. F_task not gated (the stage-advance gate
                             inside _reward_wococo_contact_pts is the gate).
          "oneshot_linear" — same as oneshot_const but magnitude = new stage index
                             (entering stage k pays +k).
        """
        wococo_env = (self.env_class == 19) | (self.env_class == 15)
        mode = str(getattr(self.cfg.rewards, 'wococo_stage_reward_mode', 'per_step'))
        if mode in ('oneshot_const', 'oneshot_linear'):
            stage = self.wococo_contact_stage
            new_stage = wococo_env & (stage > self.wococo_episode_max_stage)
            if mode == 'oneshot_const':
                reward = new_stage.float()
            else:
                reward = new_stage.float() * stage.float()
            torch.maximum(self.wococo_episode_max_stage, stage,
                          out=self.wococo_episode_max_stage)
            return reward * self._wococo_stage_anneal_factor()
        # Per-env wall-goal index: env19→2, env15→4. Used to gate stage-0 farming.
        wall_goal_idx = torch.where(self.env_class == 15,
                                    self._wococo_goal_idx_4,
                                    self._wococo_wall_goal_idx_19)
        stage = self.wococo_contact_stage
        soft_contact = self.contact_filt   # smoothed >2 N signal

        F_task = torch.zeros(self.num_envs, device=self.device)

        s0 = wococo_env & (stage == 0)
        if s0.any():
            rear_pos       = self.rigid_body_states[:, self.feet_indices[2:], :3]
            in_lo_per      = self._in_region(rear_pos, self.env_flyball_liftoff_region)
            in_lo          = in_lo_per.all(dim=-1)
            front_airborne = (~soft_contact[:, :2]).all(dim=-1)
            # Approach velocity along the current-goal direction, compared to commanded velocity (absolute thresholds conflict with low commands).
            goal_vec       = self.target_pos_rel
            goal_dir       = goal_vec / (torch.norm(goal_vec, dim=-1, keepdim=True) + 1e-5)
            approach_vel   = (goal_dir * self.root_states[:, 7:9]).sum(dim=-1)
            min_approach   = torch.clamp(0.5 * self.commands[:, 0], min=0.3)
            # cur_goal_idx >= wall_goal_idx blocks stage-0 farming until the robot has tracked past the takeoff waypoint before the wall (env19 g2, env15 g4; per-env tensor).
            goal_reached = self.cur_goal_idx >= wall_goal_idx
            # Match the stage-0 wrong-contact gate used by the contact reward.
            front_pos_s0   = self.rigid_body_states[:, self.feet_indices[:2], :3]
            in_wall_f_s0   = self._in_region(front_pos_s0, self.env_flyball_wall_region)
            n_wrong_s0     = ((soft_contact[:, 2:] & ~in_lo_per).float().sum(dim=-1)
                              + (soft_contact[:, :2] & ~in_wall_f_s0).float().sum(dim=-1))
            no_wrong_s0    = (n_wrong_s0 == 0)
            F_task[s0] = (in_lo
                          & (self.root_states[:, 2] > 0.20)
                          & front_airborne
                          & (approach_vel >= min_approach)
                          & goal_reached
                          & no_wrong_s0)[s0].float()

        s1 = wococo_env & (stage == 1)
        if s1.any():
            rear_airborne = (~soft_contact[:, 2:]).all(dim=-1)
            roll_target = -torch.deg2rad(self.env_flyball_wall_angle_deg)
            roll_tol    = torch.deg2rad(torch.tensor(float(getattr(self.cfg.rewards, 'wococo_front_roll_tol_deg', 45.0)), device=self.device))
            roll_ok     = (self.roll >= roll_target - roll_tol) & (self.roll <= roll_target + roll_tol)
            # Match the stage-1 wrong-contact gate used by the contact reward.
            front_pos_s1  = self.rigid_body_states[:, self.feet_indices[:2], :3]
            in_wall_f_s1  = self._in_region(front_pos_s1, self.env_flyball_wall_region)
            n_wrong_s1    = ((soft_contact[:, :2] & ~in_wall_f_s1).float().sum(dim=-1)
                             + soft_contact[:, 2:].float().sum(dim=-1))
            no_wrong_s1   = (n_wrong_s1 == 0)
            F_task[s1]    = (rear_airborne & roll_ok & no_wrong_s1)[s1].float()

        s2 = wococo_env & (stage == 2)
        if s2.any():
            # Stage-2 dwell-time decay: same shape as stage 3 (1-(t/T)^2),
            # T tunable via cfg.rewards.wococo_stage2_decay_T (default 50).
            self.wococo_stage2_time[s2] += 1
            t2 = self.wococo_stage2_time[s2].float()
            T2 = float(getattr(self.cfg.rewards, 'wococo_stage2_decay_T', 50.0))
            decay_s2 = torch.clamp(1.0 - (t2 / T2) ** 2, min=0.0)
            _yaw_tol = math.radians(float(getattr(self.cfg.rewards, 'wococo_rear_yaw_tol_deg', 40.0)))
            yaw_ok = (self.yaw.abs() >= (math.pi / 2 - _yaw_tol)) & (self.yaw.abs() <= (math.pi / 2 + _yaw_tol))
            # Stage 2 requires no off-target rear-wall contacts.
            rear_pos_s2  = self.rigid_body_states[:, self.feet_indices[2:], :3]
            in_wall_r_s2 = self._in_region(rear_pos_s2, self.env_flyball_wall_region)
            n_wrong_s2   = (soft_contact[:, 2:] & ~in_wall_r_s2).float().sum(dim=-1)
            no_wrong_s2  = (n_wrong_s2 == 0)
            F_task[s2] = (yaw_ok & no_wrong_s2)[s2].float() * decay_s2

        s3 = wococo_env & (stage == 3)
        if s3.any():
            # Share landing roll/yaw gates with the contact reward.
            self.wococo_stage3_time[s3] += 1
            t = self.wococo_stage3_time[s3].float()
            T = float(getattr(self.cfg.rewards, 'wococo_stage3_decay_T', 100.0))
            # rear feet airborne (no-contact) — relaxed from positional air-AABB
            # gate. Cfg knob wococo_landing_yaw_tol_deg controls the |yaw|
            # tolerance from 180° (default 45° → threshold |yaw| > 2.356).
            rear_in_air    = (~soft_contact[:, 2:]).all(dim=-1)
            roll_land_tol  = torch.deg2rad(torch.tensor(float(getattr(self.cfg.rewards, 'wococo_landing_roll_tol_deg', 25.0)), device=self.device))
            roll_ok_land   = self.roll.abs() < roll_land_tol
            yaw_tol_rad    = math.pi - math.radians(float(getattr(self.cfg.rewards, 'wococo_landing_yaw_tol_deg', 60.0)))
            yaw_ok_land    = self.yaw.abs() > yaw_tol_rad
            landing_center = 0.5 * (self.env_flyball_landing_region[:, 0, :]
                                    + self.env_flyball_landing_region[:, 1, :])
            to_target      = landing_center[:, :2] - self.root_states[:, :2]
            goal_dir       = to_target / (torch.norm(to_target, dim=-1, keepdim=True) + 1e-5)
            approach_vel   = (goal_dir * self.root_states[:, 7:9]).sum(dim=-1)
            min_app        = torch.clamp(0.3 * self.commands[:, 0], min=0.2)
            making_progress = approach_vel >= min_app
            front_pos      = self.rigid_body_states[:, self.feet_indices[:2], :3]
            in_land_front  = self._in_region(front_pos, self.env_flyball_landing_region)
            already_in     = in_land_front.all(dim=-1)
            decay          = torch.clamp(1.0 - (t / T) ** 2, min=0.0)
            n_wrong_s3     = ((soft_contact[:, :2] & ~in_land_front).float().sum(dim=-1)
                              + soft_contact[:, 2:].float().sum(dim=-1))
            no_wrong       = (n_wrong_s3 == 0)
            gated          = rear_in_air & (making_progress | already_in) & roll_ok_land & yaw_ok_land & no_wrong
            F_task[s3]     = gated[s3].float() * decay

        # Stage 4 (terminal sentinel) pays no wococo reward -- the return trip must be earned via tracking_goal_vel, not by standing still after the wall turn.
        s4 = wococo_env & (stage == 4)
        if s4.any():
            self.wococo_stage4_time[s4] += 1  # still tracked for episode logging

        n_stage = torch.clamp(stage.float(), min=1.0, max=3.0)
        per_step_reward = n_stage * F_task * wococo_env.float()

        # per_step-mode stage-4 entry bonus: without it per_step pays nothing for the s3->s4 transition (F_task[s4]=0, n_stage clamped at 3). First-time entry is detected via wococo_episode_max_stage, maintained here too because the oneshot branches return early.
        stage4_entry = wococo_env & (stage == 4) & (self.wococo_episode_max_stage < 4)
        bonus = stage4_entry.float() * float(getattr(self.cfg.rewards, 'wococo_stage4_entry_bonus', 0.0))
        torch.maximum(self.wococo_episode_max_stage, stage,
                      out=self.wococo_episode_max_stage)
        return per_step_reward + bonus

    def _reward_wococo_contact_pts(self):
        """Design-aware contact reward (WoCoCo Eq. 3). cfg.rewards.wococo_design
        selects the FSM:
          P3 — keep the existing 4-stage FSM (0 LIFTOFF → 1 FRONT → 2 REAR →
               3 LANDING → 4 sentinel). s1/s2 now use per-foot _near_target
               gating + a SMOOTH exp(-posture_err/0.5) F_task multiplier.
          P4/P2 — 2 contact stages (0 FRONT_PLANT, 1 REAR_PLANT, 2 sentinel),
               success-gated PAUSE-ON-ERROR advance (N_persist hold).
          P1 — 1 contact stage (0 ALL_FOUR_PLANT, 1 sentinel), F_task ≡ 1.

        r_con = n_corr − w_nw·n_con·n_wrong·1(n_stage>0) + 2·n_con²·F_con·F_task
        """
        if self._wococo_design in ('P1', 'P2', 'P4', 'P4L'):
            return self._wococo_contact_pts_few_stage()
        return self._wococo_contact_pts_p3()

    def _wococo_wall_press_threshold(self):
        """Return the scalar or per-env force gate for a valid wall press."""
        thresholds = getattr(
            self.cfg.rewards, 'wococo_wall_press_force_threshold_by_level', None)
        if thresholds is None:
            return float(getattr(
                self.cfg.rewards, 'wococo_wall_press_force_threshold', 15.0))

        threshold_table = getattr(
            self, '_wococo_wall_press_force_threshold_table', None)
        if threshold_table is None:
            threshold_table = torch.as_tensor(
                thresholds, dtype=torch.float, device=self.device)
            if threshold_table.ndim != 1 or threshold_table.numel() == 0:
                raise ValueError(
                    'wococo_wall_press_force_threshold_by_level must be a non-empty 1-D sequence')
            self._wococo_wall_press_force_threshold_table = threshold_table
        level_idx = self.terrain_levels.long().clamp(
            min=0, max=threshold_table.numel() - 1)
        return threshold_table[level_idx].unsqueeze(1)

    def _wococo_contact_pts_p3(self):
        """P3 minimal: the existing eased 4-stage FSM, upgraded with per-foot
        _near_target gating + a smooth posture F_task multiplier in s1/s2.
        Stages: 0 LIFTOFF, 1 FRONT_WALL, 2 REAR_WALL, 3 LANDING, 4 sentinel."""
        wococo_env = (self.env_class == 19) | (self.env_class == 15)
        wall_goal_idx = torch.where(self.env_class == 15,
                                    self._wococo_goal_idx_4,
                                    self._wococo_wall_goal_idx_19)
        reward = torch.zeros(self.num_envs, device=self.device)

        feet_pos     = self.rigid_body_states[:, self.feet_indices, :3]             # (N, 4, 3)
        force        = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)  # (N, 4)
        landing_force_thresh = float(getattr(self.cfg.rewards, 'wococo_landing_contact_force_threshold', 15.0))
        hard_contact = (force > landing_force_thresh)   # used by stage-3 landing
        wall_press_threshold = self._wococo_wall_press_threshold()
        hard_contact_wall = (force > wall_press_threshold)
        stance_threshold = float(getattr(self.cfg.rewards, 'wococo_liftoff_stance_force_threshold', 25.0))
        stance_contact = (force > stance_threshold)
        soft_contact = self.contact_filt

        stage = self.wococo_contact_stage
        n_wrong_weight = float(getattr(self.cfg.rewards, 'wococo_n_wrong_weight', 0.3))

        def _wococo(n_corr_inst, n_con, F_con_inst, F_task, n_wrong, stage_mask):
            # Gate n_wrong on the latched arm flag (set once base x crosses liftoff_x_far, cleared on stage-4 entry or episode reset).
            n_wrong_active = n_wrong * self.wococo_armed.float()
            r = n_corr_inst + 2.0 * (n_con ** 2) * F_con_inst * F_task - n_wrong_weight * n_con * n_wrong_active
            return r * stage_mask.float()

        # Stage 0: LIFTOFF — rear feet in liftoff region, sprint approach
        s0 = wococo_env & (stage == 0)
        if s0.any():
            rear_pos       = feet_pos[:, 2:, :]
            in_lo          = self._in_region(rear_pos, self.env_flyball_liftoff_region)
            correct_now    = in_lo & stance_contact[:, 2:]
            n_corr_inst    = correct_now.float().sum(dim=-1)
            # Graded F_con: one foot in the liftoff AABB earns half credit on the
            # 2·n_con²·F_con·F_task term. Advance still requires both feet (F_con_full).
            F_con_inst     = n_corr_inst / 2.0
            F_con_full     = correct_now.all(dim=-1)
            front_pos      = feet_pos[:, :2, :]
            in_wall_f      = self._in_region(front_pos, self.env_flyball_wall_region)
            front_on_floor = soft_contact[:, :2] & ~in_wall_f
            n_wrong        = ((soft_contact[:, 2:] & ~in_lo).float().sum(dim=-1)
                              + front_on_floor.float().sum(dim=-1))
            goal_vec       = self.target_pos_rel
            goal_dir       = goal_vec / (torch.norm(goal_vec, dim=-1, keepdim=True) + 1e-5)
            approach_vel   = (goal_dir * self.root_states[:, 7:9]).sum(dim=-1)
            min_approach   = torch.clamp(0.5 * self.commands[:, 0], min=0.3)
            front_airborne = (~soft_contact[:, :2]).all(dim=-1)
            # cur_goal_idx >= wall_goal_idx blocks stage-0 farming until the robot has tracked past the takeoff waypoint (env19 g2, env15 g4); it also gates n_corr_inst so the dense per-foot term can't be farmed by standing in the liftoff AABB.
            goal_reached  = (self.cur_goal_idx >= wall_goal_idx)
            n_corr_inst   = n_corr_inst * goal_reached.float()
            F_task        = ((self.root_states[:, 2] > 0.20)
                              & front_airborne
                              & (approach_vel >= min_approach)
                              & goal_reached).float()
            reward += _wococo(n_corr_inst, 2.0, F_con_inst, F_task, n_wrong, s0)
            advance = s0 & F_con_full & F_task.bool()
            self.wococo_contact_stage[advance] = 1

        # Per-foot region radius for the _near_target gate (cliffside-mirror).
        foot_radius = self._wococo_rad('wococo_r_wall', 'wococo_foot_region_radius', 0.18)

        # Stage 1 uses smooth roll credit while its boolean gate controls advance.
        s1 = wococo_env & (stage == 1)
        if s1.any():
            front_pos     = feet_pos[:, :2, :]
            tgt_front     = self.env_flyball_foot_target[:, :2, :]
            near_front    = self._near_target(front_pos, tgt_front,
                                             self._rad_feet(foot_radius, slice(0, 2)))
            front_on_wall = front_pos[..., 0] >= self.env_flyball_pit_x1.unsqueeze(1)
            correct_now   = near_front & hard_contact_wall[:, :2] & front_on_wall
            n_corr_inst   = correct_now.float().sum(dim=-1)
            F_con_inst    = correct_now.all(dim=-1).float()
            front_wall_contact = soft_contact[:, :2] & front_on_wall
            n_wrong       = ((front_wall_contact & ~near_front).float().sum(dim=-1)
                             + soft_contact[:, 2:].float().sum(dim=-1))
            rear_airborne = (~soft_contact[:, 2:]).all(dim=-1)
            roll_target   = -torch.deg2rad(self.env_flyball_wall_angle_deg)
            roll_tol      = torch.deg2rad(torch.tensor(float(getattr(self.cfg.rewards, 'wococo_front_roll_tol_deg', 45.0)), device=self.device))
            roll_ok       = (self.roll >= roll_target - roll_tol) & (self.roll <= roll_target + roll_tol)
            # SMOOTH F_task multiplier: exp(-roll_err/0.5), gated by rear_airborne.
            roll_err      = (self.roll - roll_target).abs()
            F_task        = rear_airborne.float() * torch.exp(-roll_err / 0.5)
            reward += _wococo(n_corr_inst, 2.0, F_con_inst, F_task, n_wrong, s1)
            advance = s1 & F_con_inst.bool() & rear_airborne & roll_ok
            self.wococo_contact_stage[advance] = 2
            self.wococo_stage2_time[advance] = 0

        # Stage 2: REAR_WALL — rear feet on THEIR OWN targets, yaw ~90°, front
        # allowed on the wall. SMOOTH yaw-error F_task multiplier.
        s2 = wococo_env & (stage == 2)
        if s2.any():
            rear_pos    = feet_pos[:, 2:, :]
            tgt_rear    = self.env_flyball_foot_target[:, 2:, :]
            near_rear   = self._near_target(rear_pos, tgt_rear,
                                           self._rad_feet(foot_radius, slice(2, 4)))
            rear_on_wall = rear_pos[..., 0] >= self.env_flyball_pit_x1.unsqueeze(1)
            correct_now = near_rear & hard_contact_wall[:, 2:] & rear_on_wall
            n_corr_inst = correct_now.float().sum(dim=-1)
            F_con_inst  = correct_now.all(dim=-1).float()
            rear_wall_contact = soft_contact[:, 2:] & rear_on_wall
            n_wrong     = (rear_wall_contact & ~near_rear).float().sum(dim=-1)
            _yaw_tol    = math.radians(float(getattr(self.cfg.rewards, 'wococo_rear_yaw_tol_deg', 40.0)))
            yaw_ok      = (self.yaw.abs() >= (math.pi / 2 - _yaw_tol)) & (self.yaw.abs() <= (math.pi / 2 + _yaw_tol))
            yaw_err     = (self.yaw.abs() - math.pi / 2.0).abs()
            F_task      = torch.exp(-yaw_err / 0.5)
            reward += _wococo(n_corr_inst, 2.0, F_con_inst, F_task, n_wrong, s2)
            advance = s2 & F_con_inst.bool() & yaw_ok
            self.wococo_contact_stage[advance] = 3
            self.wococo_stage3_time[advance] = 0

        # Stage 3: LANDING — front feet in landing region, rear feet in air-AABB,
        # roll ≈ 0, body yaw aligned with world -x.
        # rear feet airborne (no-contact) — relaxed from positional air-AABB.
        # Cfg knob wococo_landing_yaw_tol_deg controls |yaw| tolerance from 180°
        # (default 45° → threshold |yaw| > 2.356 rad ≈ facing back along -x).
        s3 = wococo_env & (stage == 3)
        if s3.any():
            front_pos     = feet_pos[:, :2, :]
            in_land       = self._in_region(front_pos, self.env_flyball_landing_region)
            correct_now   = in_land & hard_contact[:, :2]
            n_corr_inst   = correct_now.float().sum(dim=-1)
            F_con_inst    = correct_now.all(dim=-1).float()
            rear_in_air   = (~soft_contact[:, 2:]).all(dim=-1)
            n_wrong       = ((soft_contact[:, :2] & ~in_land).float().sum(dim=-1)
                             + soft_contact[:, 2:].float().sum(dim=-1))
            roll_land_tol = torch.deg2rad(torch.tensor(float(getattr(self.cfg.rewards, 'wococo_landing_roll_tol_deg', 25.0)), device=self.device))
            roll_ok_land  = self.roll.abs() < roll_land_tol
            yaw_tol_rad   = math.pi - math.radians(float(getattr(self.cfg.rewards, 'wococo_landing_yaw_tol_deg', 60.0)))
            yaw_ok_land   = self.yaw.abs() > yaw_tol_rad
            F_task        = (rear_in_air & roll_ok_land & yaw_ok_land).float()
            reward += _wococo(n_corr_inst, 2.0, F_con_inst, F_task, n_wrong, s3)
            advance = s3 & F_con_inst.bool() & rear_in_air & roll_ok_land & yaw_ok_land
            self.wococo_contact_stage[advance] = 4

        return reward

    def _wococo_contact_pts_few_stage(self):
        """Reduced-stage contact FSM (P4/P2 = 2 contact stages, P1 = 1).

        Stages:
          P4/P2: 0 FRONT_PLANT (FL,FR on targets), 1 REAR_PLANT (RL,RR on
                 targets), 2 sentinel.
          P1:    0 ALL_FOUR_PLANT (all 4 on targets), 1 sentinel.

        Eq.3:
          r = n_corr + 2·n_con²·F_con·F_task − w_nw·n_con·n_wrong·1(n_stage>0)
        n_con = size of the stage's contact plan. P1/P2: 4 throughout. P4
        (spec 2026-06-12): stage 0 plans ALL FOUR feet — fronts on targets,
        rears AIRBORNE — so n_con=4 and rear-air counts toward n_corr/F_con
        (air credit gated on plant_armed); stage 1 plans the rear pair only
        (fronts don't-care) so n_con=2. P4L stage 2 (landing) plans the FRONT
        pair only (rears don't-care) so n_con=2. F_con = full plan satisfied. F_task is
        a SMOOTH posture multiplier (P4/P2) or ≡1 (P1, paper-purist — no
        posture gate on the bonus). Advance is success-gated with
        PAUSE-ON-ERROR persistence:
        stage_hold_counter += hold_ok (reset to 0 on ~hold_ok); advance when the
        counter reaches N_persist; stage is monotonic clip(stage+1, 0, max)."""
        wococo_env = (self.env_class == 19) | (self.env_class == 15)
        reward = torch.zeros(self.num_envs, device=self.device)
        stage = self.wococo_contact_stage
        feet_pos = self.rigid_body_states[:, self.feet_indices, :3]               # (N,4,3)
        # Absolute contact-force magnitude per foot: ||[Fx, Fy, Fz]||_2 in N.
        force    = torch.linalg.vector_norm(self.contact_forces[:, self.feet_indices, :], dim=-1)  # (N,4)
        soft_contact = self.contact_filt
        wall_press_threshold = self._wococo_wall_press_threshold()
        hard_contact_wall = (force > wall_press_threshold)                        # (N,4)
        on_wall_x = feet_pos[..., 0] >= self.env_flyball_pit_x1.unsqueeze(1)      # (N,4)
        foot_radius = self._wococo_rad('wococo_r_wall', 'wococo_foot_region_radius', 0.18)
        near = self._near_target(feet_pos, self.env_flyball_foot_target, foot_radius)  # (N,4)
        n_wrong_weight = float(getattr(self.cfg.rewards, 'wococo_n_wrong_weight', 0.3))
        n_persist = self._wococo_n_persist
        low_persist = getattr(self.cfg.rewards, 'wococo_n_persist_low_level', None)
        if low_persist is not None:
            low_below = int(getattr(
                self.cfg.rewards, 'wococo_n_persist_low_below_level', 4))
            low_end = int(getattr(
                self.cfg.rewards, 'wococo_n_persist_low_end', low_persist))
            low_level = self.terrain_levels.clamp(min=0, max=max(low_below - 1, 0)).float()
            low_persist_t = torch.round(
                float(low_persist) + (low_end - float(low_persist))
                * low_level / max(low_below - 1, 1))
            n_persist = torch.where(
                self.terrain_levels < low_below,
                low_persist_t.to(dtype=self.terrain_levels.dtype),
                torch.full_like(self.terrain_levels, int(n_persist)))
        # For the reduced designs the sentinel == plant_complete (P1=1, P4/P2=2);
        # the FSM clips here and `stage < max_stage` marks the contact stages.
        max_stage = self._wococo_sentinel_stage
        design = self._wococo_design
        # P4L only: replace the stage-2 foothold plan with a base-state gate.
        posture_only = (design == 'P4L') and self._wococo_landing_posture_only()

        roll_target = -torch.deg2rad(self.env_flyball_wall_angle_deg)
        roll_err = (self.roll - roll_target).abs()
        yaw_err  = (self.yaw.abs() - math.pi / 2.0).abs()

        # Low-level plant easing (PER-ENV on terrain_levels, not the fleet mean). Plant discovery is gated by a CONJUNCTION whose expensive conjuncts on the easiest rows are stage 0's both-rears-AIRBORNE while the fronts press the wall (a reared posture a trotting gait essentially never produces by accident, so the front placement it teaches never gets credited) and stage 1's |yaw| within wococo_rear_yaw_tol_deg of 90 deg.
        # Both stay fully in force from wococo_low_level_ease_below_level upward, so the graduated skill is unchanged and only a bootstrapping fleet's rows get the wider gate. None/absent = no easing, and the arithmetic below is bit-identical to before.
        _ease_below = getattr(self.cfg.rewards, 'wococo_low_level_ease_below_level', None)
        if _ease_below is None:
            low_ease = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            low_ease = self.terrain_levels < int(_ease_below)

        # Per-stage active-foot mask + posture F_task.
        active = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        F_task = torch.ones(self.num_envs, device=self.device)
        if design == 'P1':
            # Stage 0: all four feet on their targets. F_task ≡ 1 (purist).
            s0 = wococo_env & (stage == 0)
            active[s0] = torch.ones(4, dtype=torch.bool, device=self.device)
            # F_task stays 1 everywhere (no posture gate on the bonus).
        else:  # P2 / P4 / P4L
            front = torch.tensor([True, True, False, False], device=self.device)
            rear  = torch.tensor([False, False, True, True], device=self.device)
            s0 = wococo_env & (stage == 0)   # FRONT_PLANT
            s1 = wococo_env & (stage == 1)   # REAR_PLANT
            active[s0] = front
            active[s1] = rear
            F_task = torch.where(s0, torch.exp(-roll_err / 0.5), F_task)
            F_task = torch.where(s1, torch.exp(-yaw_err / 0.5), F_task)
            if design == 'P4L' and self._wococo_landing_posture_only():
                # Stage 2 LANDING, posture-only: NO feet are planned, so `active` stays all-False and n_corr / F_con / n_wrong fall out of the arithmetic (F_con is overridden with the base-state support condition, n_con forced to 0). F_task is still computed from the attitude error so an n_con != 0 ablation keeps a sensible multiplier.
                s2 = wococo_env & (stage == 2)
                _ls = float(getattr(self.cfg.rewards, 'wococo_landing_F_task_sigma', 0.5))
                _w_p = float(getattr(self.cfg.rewards, 'wococo_landing_pitch_weight', 1.0))
                F_task = torch.where(
                    s2,
                    torch.exp(-(self.roll.abs() + _w_p * self.pitch.abs()) / _ls),
                    F_task)
            elif design == 'P4L':
                # Stage 2 LANDING: front feet active on the FLOOR. Posture F_task rewards zero roll AND a complete turnaround (|yaw| -> 180 deg), mirroring the stage-0/1 form.
                s2 = wococo_env & (stage == 2)
                active[s2] = front
                # Kernel width is cfg-tunable (default 0.5 = the historic value). The stage-2 posture GATE is roll < wococo_landing_roll_tol_deg (20 deg) and |yaw| within wococo_landing_yaw_tol_deg (70 deg) of pi,
                # but sigma=0.5 makes F_task ~0.04 at that gate corner and ~0.25 at a good landing, so the completion jackpot was worth less than one step of standing on the footholds. Widen sigma to keep a live gradient over the range the gate admits.
                _ls = float(getattr(self.cfg.rewards, 'wococo_landing_F_task_sigma', 0.5))
                roll_err_land = self.roll.abs()
                yaw_err_land  = (self.yaw.abs() - math.pi).abs()
                F_task = torch.where(s2, torch.exp(-(roll_err_land + yaw_err_land) / _ls), F_task)

        in_contact_stage = wococo_env & (stage < max_stage)
        correct = near & hard_contact_wall & active                              # (N,4)
        n_corr_inst = correct.float().sum(dim=-1)
        # F_con: every active foot is correct (correct ⊆ active, so equality of
        # counts ⇔ all-active-correct), and at least one foot is active.
        F_con = (correct.float().sum(dim=-1) >= active.float().sum(dim=-1)) & active.any(dim=-1)
        F_con_inst = F_con.float()

        n_con = 4.0
        n_con_pen = n_con      # rebound to a tensor alongside n_con for P4/P4L
        if design in ('P4', 'P4L'):
            # Stage 0 plans all feet; stage 1 plans only the rear pair.
            s0_p4 = wococo_env & (stage == 0)
            rear_air = ~soft_contact[:, 2:]                                   # (N,2)
            rear_air_credit = rear_air & (s0_p4 & self.wococo_armed).unsqueeze(1)
            n_corr_inst = n_corr_inst + rear_air_credit.float().sum(dim=-1)
            # Stage 0 requires correct front contacts and airborne rear feet.
            front_pair_ok = correct[:, :2].all(dim=-1)
            # wococo_stage0_ease_rear_air: on the eased rows, front placement alone completes stage 0. The rear-air n_corr credit is untouched, so keeping the rears up still pays more -- it just stops being a hard precondition.
            _rear_air_ok = rear_air.all(dim=-1)
            if bool(getattr(self.cfg.rewards, 'wococo_low_level_ease_ramp', False)):
                # Three-rung ladder instead of on/off, so the rear-air demand arrives in stages: bottom third of the eased band needs no rear foot up, middle needs one, top (and above the band) needs both. One foot up is the posture a trot already passes through.
                _span = max(int(_ease_below or 1), 1)
                _rear_air_ok = torch.where(
                    self.terrain_levels < max(_span // 2, 1),
                    torch.ones_like(_rear_air_ok),
                    torch.where(self.terrain_levels < _span,
                                rear_air.any(dim=-1), _rear_air_ok))
            if bool(getattr(self.cfg.rewards, 'wococo_stage0_ease_rear_air', False)):
                _rear_air_ok = _rear_air_ok | low_ease
            F_con = torch.where(s0_p4, front_pair_ok & _rear_air_ok, F_con)
            F_con_inst = F_con.float()
            n_con = torch.full((self.num_envs,), 4.0, device=self.device)
            n_con[stage == 1] = 2.0
            # Plan size used to price the n_wrong penalty. Identical to n_con except at the P4L landing, where the credit plan may include the rears but only the fronts can ever be wrong.
            n_con_pen = n_con.clone()

            # Stage 1 (REAR_PLANT) FRONT-AIR requirement, the mirror of the stage-0 rear-air demand: from wococo_stage1_front_air_from_level upward both fronts must be OFF the surface while the rears plant, so the rear stage can't complete with the front feet loaded.
            # Below that level the gate is inactive and the arithmetic is bit-identical. The credit plan (n_con = 2, the rear pair) is untouched: this is a REQUIREMENT on the advance gate, not extra income.
            _fa_from = getattr(self.cfg.rewards,
                               'wococo_stage1_front_air_from_level', None)
            if _fa_from is not None:
                s1_p4 = wococo_env & (stage == 1)
                front_air_ok = (~soft_contact[:, :2]).all(dim=-1) | (
                    self.terrain_levels < int(_fa_from))
                F_con = torch.where(s1_p4, F_con & front_air_ok, F_con)
                F_con_inst = F_con.float()

        # P4L stage-2 LANDING contact plan (n_con=2): FRONT FEET ONLY, both touching their Gaussian-radius landing footholds. The rears are DON'T-CARE -- no n_corr credit, no F_con gate, and rear ground contact is not n_wrong.
        s2_l = None
        near_land = None
        front_landing_contact = None
        front_land_all = None
        front_land_latched = None
        rear_air_all = None
        land_support_ok = None
        if posture_only:
            # POSTURE-ONLY LANDING (cfg.rewards.wococo_landing_posture_only). The stage-2 contact plan is EMPTY: n_con = 0 collapses the Eq. 3 reward to exactly zero here -- no n_corr income to farm by loitering, no jackpot, and no n_wrong (only feet can be wrong and no feet are planned).
            # The stage survives purely as a GATE on the return posture, and every conjunct of that gate has its own dense source elsewhere (tracking_yaw and tracking_goal_vel are both live at stage 2; roll/pitch/height come from _reward_wococo_base_orient's landing branch).
            # F_con keeps its meaning -- "this stage's task condition holds" -- so the generic hold_ok / persistence / advance machinery needs no stage-2 special case.
            s2_l = wococo_env & (stage == 2)
            land_height_ok, land_speed_ok = self._wococo_landing_support_terms()
            land_support_ok = land_height_ok & land_speed_ok
            F_con = torch.where(s2_l, land_support_ok, F_con)
            F_con_inst = F_con.float()
            n_con[stage == 2] = 0.0
            n_con_pen[stage == 2] = 0.0
        elif design == 'P4L':
            s2_l       = wococo_env & (stage == 2)
            land_tgt   = self.env_flyball_landing_foot_target                 # (N,2,3)
            land_rad   = self._wococo_landing_accept_radius()
            land_force = float(getattr(self.cfg.rewards, 'wococo_landing_contact_force_threshold', 15.0))
            front_pos  = feet_pos[:, :2, :]
            near_land  = self._near_target(front_pos, land_tgt, land_rad)     # (N,2)
            front_landing_contact = force[:, :2] > land_force                 # (N,2)
            front_land_ok = near_land & front_landing_contact                 # (N,2)
            rear_air_l = ~soft_contact[:, 2:]                                 # (N,2)
            # INSTANTANEOUS pair condition. Feeds n_corr, F_con and therefore the
            # Eq. 3 jackpot below — never latch this one, or a foot that touches
            # and lifts keeps earning 2*n_con^2*F_con*F_task.
            front_land_all = front_land_ok.all(dim=-1)
            # LATCHED pair condition, gate only. With rewards.wococo_landing_stagger_window_steps = W > 0 a front foot that lands on its foothold is remembered for W steps, so the two fronts may touch down sequentially
            # (a galloping dog's fore-fore interval is ~50-80 ms = 2.5-4 steps at 50 Hz) instead of in the same step. W = 0 keeps this bit-identical to the instantaneous gate.
            stagger_w = int(getattr(self.cfg.rewards, 'wococo_landing_stagger_window_steps', 0))
            if stagger_w > 0:
                # Expire a stale window, and drop the latch whenever we are not in stage 2, so a re-entered landing starts from scratch.
                expired = (self.wococo_p4l_land_latch_age >= stagger_w) | ~s2_l
                self.wococo_p4l_land_latch[expired]     = False
                self.wococo_p4l_land_latch_age[expired] = 0
                newly = s2_l.unsqueeze(1) & front_land_ok & ~self.wococo_p4l_land_latch
                # Start the clock at the FIRST foot to land.
                first = newly.any(dim=-1) & ~self.wococo_p4l_land_latch.any(dim=-1)
                self.wococo_p4l_land_latch_age[first] = 0
                self.wococo_p4l_land_latch |= newly
                open_window = s2_l & self.wococo_p4l_land_latch.any(dim=-1)
                self.wococo_p4l_land_latch_age[open_window] += 1
                front_land_latched = self.wococo_p4l_land_latch.all(dim=-1)
            else:
                front_land_latched = front_land_all
            rear_air_all = rear_air_l.all(dim=-1)

            # ---- rear pair in the landing plan: REQUIRED AIRBORNE -----------
            # wococo_landing_require_rear_air promotes the rears from don't-care to a planned condition (both off any surface) for the whole of stage 2, so the landing is "fronts loaded on their footholds AND rears clear" and n_con goes back to 4.
            # GRACE: stage 2 is entered the step the rear plant completes, when the rear feet are still ON THE WALL by construction -- that is what stage 1 just rewarded -- so wococo_landing_rear_air_grace_steps counts steps since entering stage 2 and treats the rears as satisfied throughout, letting the requirement bite only once there has been time to push off. Rear contact is NOT n_wrong at any point (matching the
            # F_con but is never "wrong"), so nothing negative can fire here —
            # the rears can only cost the jackpot, never charge for it.
            # The age counter is the existing wococo_p4l_stage2_steps diagnostic: zeroed per episode, incremented once per step while stage == 2, and incremented AFTER this block, so it reads 0 on the first stage-2 step and `< N` gives exactly N steps of grace.
            if bool(getattr(self.cfg.rewards,
                            'wococo_landing_require_rear_air', False)):
                _grace_n = int(getattr(self.cfg.rewards,
                                       'wococo_landing_rear_air_grace_steps', 15))
                in_grace = (self.wococo_p4l_stage2_steps < _grace_n).unsqueeze(1)  # (N,1)
                # wococo_landing_rear_air_min_level: per-env level gate on the requirement. Rows BELOW the threshold keep the front-only plan of the else-branch bit-identically (no rear n_corr credit, no rear F_con conjunct, n_con 2 so the jackpot stays 8) rather than auto-satisfied rears, which would hand those rows a free 32-point jackpot the front-only plan never priced.
                # None/absent = required on every row, as before.
                _rr_min = getattr(self.cfg.rewards,
                                  'wococo_landing_rear_air_min_level', None)
                if _rr_min is None:
                    rr_required = torch.ones(self.num_envs, dtype=torch.bool,
                                             device=self.device)
                else:
                    rr_required = self.terrain_levels >= int(_rr_min)
                rear_ok = rear_air_l | in_grace                                  # (N,2)
                n_corr_land = (front_land_ok.float().sum(dim=-1)
                               + rr_required.float() * rear_ok.float().sum(dim=-1))
                land_F_con = front_land_all & (rear_ok.all(dim=-1) | ~rr_required)
                _n_con_land = torch.where(rr_required,
                                          torch.full_like(n_con, 4.0),
                                          torch.full_like(n_con, 2.0))
            else:
                n_corr_land = front_land_ok.float().sum(dim=-1)
                land_F_con = front_land_all
                _n_con_land = 2.0

            n_corr_inst = torch.where(s2_l, n_corr_land, n_corr_inst)
            F_con = torch.where(s2_l, land_F_con, F_con)
            F_con_inst = F_con.float()
            # Landing-stage plan size (overrides the 4 set by the P4/P4L block above): front pair only = 2, front + rear-airborne = 4 (per-env where the rear-air requirement is level-gated).
            if torch.is_tensor(_n_con_land):
                n_con[stage == 2] = _n_con_land[stage == 2]
            else:
                n_con[stage == 2] = _n_con_land
            # ...but only the FRONT pair can be n_wrong at the landing, so the penalty keeps the front-pair plan size regardless.
            n_con_pen[stage == 2] = 2.0

        # n_wrong: a foot pressing the wall (on_wall_x) OFF any target. For P4 in the REAR stage, FRONT contact is NEUTRAL per spec; for P2 and the FRONT/ALL stages, any off-target wall press is wrong.
        # (A grounded rear foot in P4 stage 0 is NOT n_wrong -- it just forfeits its n_corr credit and blocks F_con; n_wrong stays reserved for wrong WALL contact.)
        wall_contact = soft_contact & on_wall_x
        off_target_press = wall_contact & ~near
        if design in ('P4', 'P4L'):
            rear_stage = wococo_env & (stage == 1)
            # Zero the FRONT feet's wrong-contribution during the rear stage.
            front_idx = torch.tensor([True, True, False, False], device=self.device)
            off_target_press = off_target_press & ~(rear_stage.unsqueeze(1) & front_idx)
        n_wrong = off_target_press.float().sum(dim=-1)
        if posture_only:
            # No feet are planned at the landing, so nothing can be WRONG there (n_con_pen is 0 at stage 2 anyway; this also keeps the logged n_wrong honest).
            n_wrong = torch.where(s2_l, torch.zeros_like(n_wrong), n_wrong)
        elif design == 'P4L' and s2_l is not None:
            # Landing stage: wrong = a contacting front foot off its landing
            # foothold. Rear contact is NEUTRAL (front-pair-only plan).
            n_wrong_land = (front_landing_contact & ~near_land).float().sum(dim=-1)
            n_wrong = torch.where(s2_l, n_wrong_land, n_wrong)
        n_wrong_active = n_wrong * self.wococo_armed.float()

        # PAUSE-ON-ERROR persistence: hold_ok = the active pair all correct AND (P4/P2) the posture gate passes; the counter increments on hold_ok, resets otherwise, and advance fires at N_persist. Computed BEFORE the reward so the jackpot can be persistence-gated.
        if design == 'P1':
            hold_ok = in_contact_stage & F_con
        else:  # P2 / P4 / P4L
            roll_tol = torch.deg2rad(torch.tensor(float(getattr(self.cfg.rewards, 'wococo_front_roll_tol_deg', 45.0)), device=self.device))
            roll_ok = (self.roll >= roll_target - roll_tol) & (self.roll <= roll_target + roll_tol)
            _yaw_tol = math.radians(float(getattr(self.cfg.rewards, 'wococo_rear_yaw_tol_deg', 40.0)))
            # Per-env widened yaw window on the eased rows (see low_ease above).
            _yaw_tol_low = getattr(self.cfg.rewards, 'wococo_rear_yaw_tol_deg_low', None)
            if _yaw_tol_low is None:
                _yaw_tol_t = torch.full((self.num_envs,), _yaw_tol, device=self.device)
            elif bool(getattr(self.cfg.rewards, 'wococo_low_level_ease_ramp', False)):
                # Continuous form: interpolate from *_low at level 0 down to the nominal tolerance at wococo_low_level_ease_below_level, rather than dropping it in one step -- a step makes the gate strictly harder on the exact iteration an env earns promotion, which builds a curriculum that oscillates around that row.
                _lo = math.radians(float(_yaw_tol_low))
                _span = max(int(_ease_below or 1), 1)
                _t = (self.terrain_levels.float() / _span).clamp(0.0, 1.0)
                _yaw_tol_t = _lo + _t * (_yaw_tol - _lo)
            else:
                _yaw_tol_t = torch.where(
                    low_ease,
                    torch.full((self.num_envs,), math.radians(float(_yaw_tol_low)), device=self.device),
                    torch.full((self.num_envs,), _yaw_tol, device=self.device))
            yaw_ok = ((self.yaw.abs() >= (math.pi / 2 - _yaw_tol_t))
                      & (self.yaw.abs() <= (math.pi / 2 + _yaw_tol_t)))
            posture_ok = torch.where(stage == 0, roll_ok, yaw_ok)
            pitch_ok_land = None
            if posture_only:
                # Per-axis attitude gate against the RETURN bearing, kept as separate booleans so the stage-2 diagnostics can say which axis is blocking.
                roll_ok_land, pitch_ok_land, yaw_ok_land = \
                    self._wococo_landing_posture_terms()
                posture_ok = torch.where(
                    stage == 2, roll_ok_land & pitch_ok_land & yaw_ok_land, posture_ok)
            elif design == 'P4L':
                roll_land_tol = torch.deg2rad(torch.tensor(float(getattr(self.cfg.rewards, 'wococo_landing_roll_tol_deg', 10.0)), device=self.device))
                yaw_land_tol  = math.radians(float(getattr(self.cfg.rewards, 'wococo_landing_yaw_tol_deg', 45.0)))
                roll_ok_land  = self.roll.abs() < roll_land_tol
                yaw_ok_land   = self.yaw.abs() > (math.pi - yaw_land_tol)
                posture_ok = torch.where(stage == 2, roll_ok_land & yaw_ok_land, posture_ok)
            hold_ok = in_contact_stage & F_con & posture_ok
            # posture_only needs NO stage-2 override here: F_con already carries the landing's task condition, so the generic form above is exact.
            if design == 'P4L' and s2_l is not None and not posture_only:
                # Stage 2 advances on the LATCHED pair; s2_l is a subset of in_contact_stage (max_stage == 3 for P4L), so this only swaps the contact term. Everything else, jackpot included, still sees F_con.
                hold_ok = torch.where(s2_l, s2_l & front_land_latched & posture_ok, hold_ok)

        # Ramp the contact jackpot over a sustained hold to suppress tap farming.
        # NOTE on what the ramp does: `advance` fires on the SAME step hold_ok first becomes true and the counter is zeroed on advance, so wococo_stage_hold_counter can never exceed the stage's n_persist and the ramp never reaches 1.0 -- it is a CONSTANT per-stage discount on the completion jackpot: n_persist/K, i.e. 3/6 = 0.5 for the wall stages and 1/6 = 0.167 for the P4L landing (landing_n_persist = 1).
        # At the landing that 6x discount removed the urgency to finish stage 2: loitering on the footholds with the posture gate unmet pays n_corr = 2 every step, while completing paid 8*F_task/6 -- hence the ~1 s pause after landing before the robot runs again.
        # wococo_landing_jackpot_persist_K overrides K for stage 2 only; the wall stages keep the global K so warm starts stay on their trained ratios.
        jackpot_K = float(getattr(self.cfg.rewards, 'wococo_jackpot_persist_K', 1.0))
        K = torch.full((self.num_envs,), jackpot_K, device=self.device)
        if design == 'P4L':
            _lk = getattr(self.cfg.rewards, 'wococo_landing_jackpot_persist_K', None)
            if _lk is not None:
                K = torch.where(stage == 2, torch.full_like(K, float(_lk)), K)
        run_len = torch.where(
            hold_ok, (self.wococo_stage_hold_counter + 1).float(),
            torch.zeros(self.num_envs, device=self.device))
        # Per-env branch: K<=1 keeps the legacy always-full jackpot (no posture/persistence coupling), K>1 ramps. Bit-identical to the old scalar branch whenever every env shares K.
        persist_factor = torch.where(
            K > 1.0,
            torch.clamp(run_len / K.clamp(min=1e-6), max=1.0),
            torch.ones(self.num_envs, device=self.device))
        # The n_wrong term is priced by the size of the plan that can actually be WRONG. At the P4L landing with wococo_landing_require_rear_air the plan is 4 (two fronts on footholds + two rears airborne) but only the two FRONT feet can be n_wrong -- rear contact is never counted -- so n_con_pen stays 2 rather than doubling the off-target front-foot charge.
        r = (n_corr_inst
             + 2.0 * (n_con ** 2) * F_con_inst * F_task * persist_factor
             - n_wrong_weight * n_con_pen * n_wrong_active)
        reward += r * in_contact_stage.float()

        self.wococo_stage_hold_counter = torch.where(
            hold_ok, self.wococo_stage_hold_counter + 1,
            torch.zeros_like(self.wococo_stage_hold_counter))
        persist_required = (n_persist if torch.is_tensor(n_persist)
                            else torch.full_like(self.wococo_stage_hold_counter,
                                                 n_persist))
        if design == 'P4L':
            front_factor = getattr(self.cfg.rewards, 'wococo_front_n_persist_factor', None)
            if front_factor is not None:
                front_required = torch.round(persist_required.float() * float(front_factor)).clamp(min=1)
                persist_required = torch.where(stage == 0, front_required.to(persist_required.dtype), persist_required)
            stage0_persist = getattr(self.cfg.rewards, 'wococo_stage0_n_persist', None)
            if stage0_persist is not None:
                persist_required = torch.where(
                    stage == 0,
                    torch.full_like(persist_required, max(1, int(stage0_persist))),
                    persist_required)
            stage1_persist = getattr(self.cfg.rewards, 'wococo_stage1_n_persist', None)
            if stage1_persist is not None:
                persist_required = torch.where(
                    stage == 1,
                    torch.full_like(persist_required, max(1, int(stage1_persist))),
                    persist_required)
            landing_n_persist = getattr(self.cfg.rewards, 'wococo_landing_n_persist', None)
            if landing_n_persist is not None:
                persist_required = torch.where(
                    stage == 2,
                    torch.full_like(persist_required, max(1, int(landing_n_persist))),
                    persist_required)
        advance = in_contact_stage & (self.wococo_stage_hold_counter >= persist_required)
        if posture_only:
            # Posture-only stage-2 occupancy: the foothold counters (front_land_ok / land_latched / rear_air) stay at zero since there is no foothold plan, and the per-axis gate counters below take their place, so a stalled landing says WHICH conjunct is blocking.
            s2_float = s2_l.float()
            self.wococo_p4l_stage2_steps     += s2_float
            self.wococo_p4l_stage2_roll_ok   += (s2_l & roll_ok_land).float()
            self.wococo_p4l_stage2_pitch_ok  += (s2_l & pitch_ok_land).float()
            self.wococo_p4l_stage2_yaw_ok    += (s2_l & yaw_ok_land).float()
            self.wococo_p4l_stage2_height_ok += (s2_l & land_height_ok).float()
            self.wococo_p4l_stage2_speed_ok  += (s2_l & land_speed_ok).float()
            self.wococo_p4l_stage2_hold_ok   += (s2_l & hold_ok).float()
            self.wococo_p4l_stage2_advance   += (s2_l & advance).float()
        elif design == 'P4L' and s2_l is not None:
            s2_float = s2_l.float()
            self.wococo_p4l_stage2_steps += s2_float
            self.wococo_p4l_stage2_front_land_ok += (s2_l & front_land_all).float()
            self.wococo_p4l_stage2_land_latched += (s2_l & front_land_latched).float()
            self.wococo_p4l_stage2_rear_air += (s2_l & rear_air_all).float()
            self.wococo_p4l_stage2_roll_ok += (s2_l & roll_ok_land).float()
            self.wococo_p4l_stage2_yaw_ok += (s2_l & yaw_ok_land).float()
            self.wococo_p4l_stage2_hold_ok += (s2_l & hold_ok).float()
            self.wococo_p4l_stage2_advance += (s2_l & advance).float()
        # Stage-0 / stage-1 gate occupancy (see the buffers in _init_buffers), recorded for every non-P1 design, whose stage 0/1 are FRONT/REAR_PLANT.
        if design != 'P1':
            _s0 = wococo_env & (stage == 0)
            _s1 = wococo_env & (stage == 1)
            _rear_air_all = (~soft_contact[:, 2:]).all(dim=-1)
            self.wococo_s0diag_steps    += _s0.float()
            self.wococo_s0diag_near     += (_s0 & near[:, :2].all(dim=-1)).float()
            self.wococo_s0diag_press    += (_s0 & hard_contact_wall[:, :2].all(dim=-1)).float()
            self.wococo_s0diag_rear_air += (_s0 & _rear_air_all).float()
            self.wococo_s0diag_roll_ok  += (_s0 & roll_ok).float()
            self.wococo_s0diag_hold_ok  += (_s0 & hold_ok).float()
            self.wococo_s1diag_steps    += _s1.float()
            self.wococo_s1diag_near     += (_s1 & near[:, 2:].all(dim=-1)).float()
            self.wococo_s1diag_press    += (_s1 & hard_contact_wall[:, 2:].all(dim=-1)).float()
            self.wococo_s1diag_yaw_ok   += (_s1 & yaw_ok).float()
            self.wococo_s1diag_hold_ok  += (_s1 & hold_ok).float()
            self.wococo_s0ever_near |= _s0 & near[:, :2].all(dim=-1)
            self.wococo_s1ever_near |= _s1 & near[:, 2:].all(dim=-1)
            self.wococo_s1ever_yaw  |= _s1 & yaw_ok
        # Monotonic single-stage advance; reset the hold counter on advance so the next stage must earn its own N_persist.
        new_stage = torch.clamp(self.wococo_contact_stage + 1, max=max_stage)
        self.wococo_contact_stage = torch.where(advance, new_stage, self.wococo_contact_stage)
        self.wococo_stage_hold_counter = torch.where(
            advance, torch.zeros_like(self.wococo_stage_hold_counter),
            self.wococo_stage_hold_counter)
        return reward

    # WoCoCo hurdle-foothold stages:
    #   0 LIFTOFF  — both rear feet in pre-hurdle AABB with stance force, front airborne
    #   1 AIRBORNE — all 4 feet airborne, base inside air-AABB above the hurdle
    #   2 LANDED   — both front feet in post-hurdle AABB with hard contact, rear in air-AABB
    #   3 terminal — on entry: idx 0 → reset stage:=0, idx:=1; idx 1 → pin
    def _hurdle_active_region(self, region_4d):
        """Slice a per-hurdle AABB tensor of shape (N, 2_h, 2, 3) by wococo_hurdle_idx
        to return the active hurdle's AABB tensor of shape (N, 2, 3)."""
        idx = torch.arange(self.num_envs, device=self.device)
        return region_4d[idx, self.wococo_hurdle_idx]

    def _wococo_hurdle_front_sym(self):
        """Per-env bool: True iff front-leg joint angles are mirror-symmetric.
        hip_indices / thigh_indices / calf_indices are stored as [FR, FL, RR, RL].
        Hip defaults are sign-mirrored (FR=-FL), so check |FR_hip + FL_hip|;
        thigh/calf share defaults, so check |FR - FL|. Tolerance is configurable
        via cfg.rewards.wococo_hurdle_front_sym_tol_rad (default 0.3 rad ≈ 17°)."""
        tol = float(getattr(self.cfg.rewards, 'wococo_hurdle_front_sym_tol_rad', 0.3))
        fr_hip   = self.dof_pos[:, self.hip_indices[0]]
        fl_hip   = self.dof_pos[:, self.hip_indices[1]]
        fr_thigh = self.dof_pos[:, self.thigh_indices[0]]
        fl_thigh = self.dof_pos[:, self.thigh_indices[1]]
        fr_calf  = self.dof_pos[:, self.calf_indices[0]]
        fl_calf  = self.dof_pos[:, self.calf_indices[1]]
        return (
            ((fr_hip + fl_hip).abs() < tol)
            & ((fr_thigh - fl_thigh).abs() < tol)
            & ((fr_calf  - fl_calf ).abs() < tol)
        )

    def _hurdle_pair_sym_score(self):
        """Smooth mirror-symmetry score in (0, 1] for BOTH leg pairs: (N, 2),
        column 0 = front (FR, FL), column 1 = rear (RR, RL).

        Sign convention matches _wococo_hurdle_front_sym: hip defaults are
        mirrored (right = -left) so the abduction error is |R + L|, while thigh
        and calf share defaults so their error is |R - L|.

        Both pairs and all six joints are read with ONE gather through the
        cached (2, 6) index table rather than twelve scalar-indexed gathers:
        the term is launch-bound, not bandwidth-bound, so the kernel count is
        what costs. Returns exp(-total_error / sigma), sigma from
        cfg.rewards.hurdle_pair_sym_sigma_rad (default 0.25 rad).
        """
        if not hasattr(self, 'hurdle_sym_dof_idx'):
            # Rows = leg pair, cols = [hip_R, hip_L, thigh_R, thigh_L, calf_R, calf_L].
            # The *_indices tensors are ordered [FR, FL, RR, RL].
            self.hurdle_sym_dof_idx = torch.stack([
                torch.stack([self.hip_indices[a], self.hip_indices[b],
                             self.thigh_indices[a], self.thigh_indices[b],
                             self.calf_indices[a], self.calf_indices[b]])
                for a, b in ((0, 1), (2, 3))
            ]).to(self.device)
            self._hurdle_sym_inv_sigma = 1.0 / max(
                float(getattr(self.cfg.rewards, 'hurdle_pair_sym_sigma_rad', 0.25)), 1e-6)
            # Point-symmetric (odd) variant knobs. hurdle_pair_sym_zero_rad is the error at which the score crosses 0, defaulting to sigma so the crossover sits where the exp form had decayed to 1/e.
            self._hurdle_sym_signed = bool(getattr(
                self.cfg.rewards, 'hurdle_pair_sym_signed', False))
            self._hurdle_sym_zero = float(getattr(
                self.cfg.rewards, 'hurdle_pair_sym_zero_rad',
                float(getattr(self.cfg.rewards, 'hurdle_pair_sym_sigma_rad', 0.25))))
            self._hurdle_sym_inv_width = 1.0 / max(float(getattr(
                self.cfg.rewards, 'hurdle_pair_sym_width_rad', 0.25)), 1e-6)

        q = self.dof_pos[:, self.hurdle_sym_dof_idx]            # (N, 2, 6), one gather
        err = ((q[:, :, 0] + q[:, :, 1]).abs()
               + (q[:, :, 2] - q[:, :, 3]).abs()
               + (q[:, :, 4] - q[:, :, 5]).abs())               # (N, 2)
        if self._hurdle_sym_signed:
            # POINT-SYMMETRIC score: -tanh((err - e0) / w) is odd about (e0, 0), so err 0 -> ~+1 (mirror form), err = e0 -> exactly 0, err >> e0 -> ~-1.
            # Unlike the exp form, a pair whose angles are COMPLETELY off is penalised rather than merely paid nothing, and the penalty saturates at -1 so one grotesque crossing cannot dominate the return.
            return -torch.tanh((err - self._hurdle_sym_zero) * self._hurdle_sym_inv_width)
        return torch.exp(-err * self._hurdle_sym_inv_sigma)

    def _reward_hurdle_pair_sym(self):
        """One-shot bonus, per leg pair, on the single step that pair passes
        directly over a hurdle, scaled by how mirror-symmetric its joint
        angles are at that instant.

        The hip-link centroid of each SCORED pair is tracked in world x against
        EVERY hurdle slot. The bonus fires on the step a centroid's offset from
        a bar's x centre changes sign — exactly one step per pair per bar per
        direction, regardless of approach speed, so there is nothing to farm by
        loitering.

        cfg.rewards.hurdle_pair_sym_front_only (default True) scores the FRONT
        pair (FR/FL) alone. The rear pair crosses mid-bound with the trunk
        pitched, still unloading from push-off, so its mirror error is largely a
        readout of how the front legs took off rather than a posture the policy
        can shape independently at that instant — and paying for it there works
        against the push-off. Front-only gives 4 payouts per episode (2 bars x 2
        directions); scoring both pairs gives 8.

        Hurdle geometry is read from env_flyball_hurdle_clear_region rather than
        env_flyball_hurdle_x_ranges: _hurdle_clear_boxes emits an all-zero
        sentinel for any slot whose bar height is <= 1e-6, so the `real_slot`
        test below makes this term identically zero on zero-height (level-0 and
        demoted) rows, where there is no bar to be symmetric over. The x_ranges
        buffer carries a span for placed-but-flat bars and would pay out there.

        NOTE: the crossing latch lives in this function, so a zero scale (which
        _prepare_reward_function pops) disables the bookkeeping too. That is
        safe here — no other term reads these buffers — but it is the same
        pattern that makes the wococo stage machine fragile, so keep it that way.
        """
        # n_hurdle_slots is sized by num_goals (terrain.py:78), so the buffer carries 8 slots while the flyball writer fills only 2. Cache the highest slot any env uses and evaluate only that prefix.
        if not hasattr(self, '_hurdle_sym_n_slots'):
            _any_real = (self.env_flyball_hurdle_clear_region[:, :, 1, 0]
                         > self.env_flyball_hurdle_clear_region[:, :, 0, 0]).any(dim=0)
            _nz = torch.nonzero(_any_real)
            self._hurdle_sym_n_slots = int(_nz.max().item()) + 1 if _nz.numel() else 0
        S = self._hurdle_sym_n_slots
        if S == 0:
            return torch.zeros(self.num_envs, device=self.device)

        bounds = self.env_flyball_hurdle_clear_region[:, :S]          # (N, S, 2, 3)
        real_slot = bounds[:, :, 1, 0] > bounds[:, :, 0, 0]           # (N, S)
        x_centre = 0.5 * (bounds[:, :, 0, 0] + bounds[:, :, 1, 0])    # (N, S)

        # Front pair is index 0, so front-only is a prefix slice: P = 1 or 2.
        P = 1 if bool(getattr(self.cfg.rewards,
                              'hurdle_pair_sym_front_only', True)) else 2

        hip_x = self.rigid_body_states[:, self.hip_body_indices, 0]                 # (N,4)
        pair_x = torch.stack([0.5 * (hip_x[:, 0] + hip_x[:, 1]),                   # front
                              0.5 * (hip_x[:, 2] + hip_x[:, 3])], dim=-1)          # (N,2)

        # (N, pair, slot): offset of each scored pair's centroid from each bar.
        d      = pair_x[:, :P].unsqueeze(-1) - x_centre.unsqueeze(1)
        d_prev = self.hurdle_sym_prev_x[:, :P].unsqueeze(-1) - x_centre.unsqueeze(1)
        # Split by travel direction so the outbound and return crossings of the SAME bar are latched independently: (N, pair, slot, dir), dir 0 = +x.
        crossed = torch.stack([(d_prev < 0) & (d >= 0),
                               (d_prev > 0) & (d <= 0)], dim=-1)
        crossed = crossed & self.hurdle_sym_prev_valid.view(-1, 1, 1, 1)
        crossed = crossed & real_slot.view(self.num_envs, 1, -1, 1)

        env_ok = ((self.env_class == 15) | (self.env_class == 19))
        latch = self.hurdle_sym_fired[:, :P, :S]
        fire = crossed & ~latch & env_ok.view(-1, 1, 1, 1)

        score = self._hurdle_pair_sym_score()[:, :P]                                # (N,P)
        reward = (fire.float() * score.view(self.num_envs, P, 1, 1)).sum(dim=(1, 2, 3))

        # latch is a basic-slice VIEW of hurdle_sym_fired, so this writes through.
        latch |= fire
        # In-place: these run every step, so avoid re-allocating both buffers.
        self.hurdle_sym_prev_x.copy_(pair_x)
        self.hurdle_sym_prev_valid.fill_(True)
        return reward

    def _reward_wococo_hurdle_stage(self):
        """WoCoCo r_stage applied to the hurdle phase. Same three modes as the wall
        version (per_step / oneshot_const / oneshot_linear), gated on
        env_class ∈ {15, 19}. On env19 (single-hurdle flyball), slot 0 is the
        forward crossing and slot 1 is the backward crossing; slot 1 only
        becomes active once cur_goal_idx >= the return-phase goal index
        (4 in the default layout; cfg.terrain.flyball_return_goal_idx).
        """
        wococo_env = (self.env_class == 15) | (self.env_class == 19)
        # Slot-1 active gate: env15 keeps the existing "any-time" semantics, while env19 requires the robot to be in the return phase before slot 1 evaluates, so slot-1 stages can't trigger during the run-up to the wall.
        slot_idx_active = (
            (self.env_class == 15)
            | ((self.env_class == 19) & ((self.wococo_hurdle_idx == 0)
                                        | (self.cur_goal_idx >= self._flyball_return_goal_idx)))
        )
        wococo_env = wococo_env & slot_idx_active
        mode = str(getattr(self.cfg.rewards, 'wococo_hurdle_stage_reward_mode', 'per_step'))
        stage = self.wococo_hurdle_stage
        idx   = self.wococo_hurdle_idx

        # Track per-slot episode-max regardless of reward mode so wandb can log peak hurdle stage even under per_step; the oneshot branch consumes the same buffer below.
        arange  = torch.arange(self.num_envs, device=self.device)
        cur_max = self.wococo_hurdle_episode_max_stage[arange, idx]
        new_stage = wococo_env & (stage > cur_max)
        updated = torch.maximum(cur_max, stage)
        self.wococo_hurdle_episode_max_stage[arange, idx] = updated

        if mode in ('oneshot_const', 'oneshot_linear'):
            if mode == 'oneshot_const':
                reward = new_stage.float()
            else:
                reward = new_stage.float() * stage.float()
            return reward

        # per_step: n_stage · F_task per step
        soft_contact = self.contact_filt
        F_task = torch.zeros(self.num_envs, device=self.device)

        liftoff_region = self._hurdle_active_region(self.env_flyball_hurdle_liftoff_region)
        air_region     = self._hurdle_active_region(self.env_flyball_hurdle_air_region)
        landing_region = self._hurdle_active_region(self.env_flyball_hurdle_landing_region)

        s0 = wococo_env & (stage == 0)
        if s0.any():
            rear_pos       = self.rigid_body_states[:, self.feet_indices[2:], :3]
            in_lo          = self._in_region(rear_pos, liftoff_region).all(dim=-1)
            front_airborne = (~soft_contact[:, :2]).all(dim=-1)
            front_pos      = self.rigid_body_states[:, self.feet_indices[:2], :3]
            in_air_f_s0    = self._in_region(front_pos, air_region)
            n_wrong_s0     = ((soft_contact[:, 2:] & ~self._in_region(rear_pos, liftoff_region)).float().sum(dim=-1)
                              + (soft_contact[:, :2] & ~in_air_f_s0).float().sum(dim=-1))
            no_wrong_s0    = (n_wrong_s0 == 0)
            F_task[s0] = (in_lo & front_airborne & no_wrong_s0)[s0].float()

        s1 = wococo_env & (stage == 1)
        if s1.any():
            self.wococo_hurdle_s1_time[s1] += 1
            t1 = self.wococo_hurdle_s1_time[s1].float()
            T1 = float(getattr(self.cfg.rewards, 'wococo_hurdle_air_decay_T', 10.0))
            decay_s1 = torch.clamp(1.0 - (t1 / T1) ** 2, min=0.0)
            all_airborne = (~soft_contact).all(dim=-1)
            base_pos     = self.root_states[:, :3].unsqueeze(1)  # (N, 1, 3)
            base_in_air  = self._in_region(base_pos, air_region).squeeze(-1)
            no_wrong_s1  = (~soft_contact).all(dim=-1)  # any contact = wrong
            front_sym    = self._wococo_hurdle_front_sym()
            F_task[s1] = (all_airborne & base_in_air & no_wrong_s1 & front_sym)[s1].float() * decay_s1

        s2 = wococo_env & (stage == 2)
        if s2.any():
            front_pos    = self.rigid_body_states[:, self.feet_indices[:2], :3]
            rear_pos     = self.rigid_body_states[:, self.feet_indices[2:], :3]
            in_land      = self._in_region(front_pos, landing_region).all(dim=-1)
            rear_in_air  = self._in_region(rear_pos, air_region).all(dim=-1)
            n_wrong_s2   = ((soft_contact[:, :2] & ~self._in_region(front_pos, landing_region)).float().sum(dim=-1)
                            + soft_contact[:, 2:].float().sum(dim=-1))
            no_wrong_s2  = (n_wrong_s2 == 0)
            F_task[s2] = (in_land & rear_in_air & no_wrong_s2)[s2].float()

        n_stage = torch.clamp(stage.float(), min=1.0, max=2.0)
        return n_stage * F_task * wococo_env.float()

    def _reward_wococo_hurdle_contact_pts(self):
        """WoCoCo Eq.3 applied to the hurdle phase. Three stages on a per-env counter
        that resets to 0 after each hurdle is cleared (idx 0 → idx 1).

          r_con = n_corr − n_con · n_wrong · 1(base_over_liftoff) · w + 2·n_con²·F_con·F_task

        Active on env_class ∈ {15, 19}. On env19 slot 1 (backward crossing)
        is gated on the return-phase goal index (4 by default,
        cfg.terrain.flyball_return_goal_idx) — without that gate the policy could
        latch slot-1 LIFTOFF during the run-up to the wall.

        n_wrong is now gated on "base xy inside the liftoff AABB" and fires
        across all three stages (0/1/2) — previously the term lived inside
        the stage-1 block, so the policy only got penalised after it had
        already advanced to AIRBORNE. Anchoring the gate on base position
        gives a contact penalty as soon as the base enters the takeoff
        zone, regardless of whether the state machine has advanced.
        """
        wococo_env = (self.env_class == 15) | (self.env_class == 19)
        slot_idx_active = (
            (self.env_class == 15)
            | ((self.env_class == 19) & ((self.wococo_hurdle_idx == 0)
                                        | (self.cur_goal_idx >= self._flyball_return_goal_idx)))
        )
        wococo_env = wococo_env & slot_idx_active

        reward = torch.zeros(self.num_envs, device=self.device)

        feet_pos     = self.rigid_body_states[:, self.feet_indices, :3]
        force        = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        hard_contact = (force > 30.0)
        stance_threshold = float(getattr(self.cfg.rewards, 'wococo_hurdle_stance_force_threshold', 20.0))
        stance_contact   = (force > stance_threshold)
        soft_contact     = self.contact_filt

        stage = self.wococo_hurdle_stage
        idx   = self.wococo_hurdle_idx
        n_wrong_weight = float(getattr(self.cfg.rewards, 'wococo_hurdle_n_wrong_weight', 0.3))

        liftoff_region = self._hurdle_active_region(self.env_flyball_hurdle_liftoff_region)
        air_region     = self._hurdle_active_region(self.env_flyball_hurdle_air_region)
        landing_region = self._hurdle_active_region(self.env_flyball_hurdle_landing_region)

        # Stage-agnostic n_wrong term with a "no escape" latch (the hurdle-FSM analog of wococo_armed): armed the first step the base crosses the END of the active liftoff AABB, and stays armed even if the policy retreats.
        # Cleared on the slot-0 -> slot-1 hand-off (so slot 1 starts fresh) and on stage-3 entry / episode reset. n_con is held at 2.0 to match the s1 shaping coefficient.
        base_x   = self.root_states[:, 0]
        lo_x_min = liftoff_region[:, 0, 0]
        lo_x_max = liftoff_region[:, 1, 0]
        forward_slot  = (self.env_class == 15) | ((self.env_class == 19) & (self.wococo_hurdle_idx == 0))
        backward_slot = (self.env_class == 19) & (self.wococo_hurdle_idx == 1)
        past_lo_end   = (forward_slot & (base_x >= lo_x_max)) | (backward_slot & (base_x <= lo_x_min))
        self.wococo_hurdle_n_wrong_armed |= (wococo_env & past_lo_end)
        # Disarm at/after stage 3 (terminal, both slots cleared) so the post-completion locomotion phase is not penalised.
        self.wococo_hurdle_n_wrong_armed &= (self.wococo_hurdle_stage < 3)
        n_wrong_inst = soft_contact.float().sum(dim=-1)
        n_wrong_gate = wococo_env & self.wococo_hurdle_n_wrong_armed
        reward += -n_wrong_weight * 2.0 * n_wrong_inst * n_wrong_gate.float()

        def _wococo(n_corr_inst, n_con, F_con_inst, F_task, stage_mask):
            r = n_corr_inst + 2.0 * (n_con ** 2) * F_con_inst * F_task
            return r * stage_mask.float()

        # Stage 0: LIFTOFF -- rear feet in pre-hurdle AABB with stance force, front airborne. Reward intentionally OFF here; only the advance logic runs.
        s0 = wococo_env & (stage == 0)
        if s0.any():
            rear_pos       = feet_pos[:, 2:, :]
            in_lo          = self._in_region(rear_pos, liftoff_region)
            correct_now    = in_lo & stance_contact[:, 2:]
            F_con_full     = correct_now.all(dim=-1)
            front_airborne = (~soft_contact[:, :2]).all(dim=-1)
            advance = s0 & F_con_full & front_airborne
            self.wococo_hurdle_stage[advance] = 1
            self.wococo_hurdle_s1_time[advance] = 0

        # Stage 1: AIRBORNE — all 4 feet airborne, base over hurdle footprint
        s1 = wococo_env & (stage == 1)
        if s1.any():
            front_pos     = feet_pos[:, :2, :]
            rear_pos      = feet_pos[:, 2:, :]
            in_air_f      = self._in_region(front_pos, air_region)
            in_air_r      = self._in_region(rear_pos, air_region)
            front_airborne = (~soft_contact[:, :2])
            rear_airborne  = (~soft_contact[:, 2:])
            # correct_now per "pair": pair counts as correct iff both feet in the
            # pair are airborne and inside the air-AABB. n_corr_inst ∈ [0, 2].
            front_pair_ok = (front_airborne & in_air_f).all(dim=-1)
            rear_pair_ok  = (rear_airborne  & in_air_r).all(dim=-1)
            correct_now   = torch.stack([front_pair_ok, rear_pair_ok], dim=-1)
            n_corr_inst   = correct_now.float().sum(dim=-1)
            F_con_inst    = correct_now.all(dim=-1).float()
            base_pos      = self.root_states[:, :3].unsqueeze(1)
            base_in_air   = self._in_region(base_pos, air_region).squeeze(-1)
            all_airborne  = (~soft_contact).all(dim=-1)
            front_sym     = self._wococo_hurdle_front_sym()
            F_task        = (all_airborne & base_in_air & front_sym).float()
            reward += _wococo(n_corr_inst, 2.0, F_con_inst, F_task, s1)
            # Advance once any front foot enters the landing AABB (start of touchdown).
            in_land_f     = self._in_region(front_pos, landing_region)
            advance       = s1 & in_land_f.any(dim=-1)
            self.wococo_hurdle_stage[advance] = 2

        # Landing advances the hurdle FSM but intentionally pays no contact reward.
        s2 = wococo_env & (stage == 2)
        if s2.any():
            front_pos     = feet_pos[:, :2, :]
            in_land       = self._in_region(front_pos, landing_region)
            correct_now   = in_land & stance_contact[:, :2]
            F_con_inst    = correct_now.all(dim=-1).float()
            advance = s2 & F_con_inst.bool()
            # On terminal entry: idx 0 -> reset stage:=0, idx:=1 (env15 starts hurdle 2; env19 arms the slot-1 backward crossing, awaiting cur_goal_idx>=4); idx 1 -> pin at stage 3.
            adv_first  = advance & (idx == 0)
            adv_second = advance & (idx == 1)
            self.wococo_hurdle_stage[adv_first]  = 0
            self.wococo_hurdle_idx[adv_first]    = 1
            self.wococo_hurdle_s1_time[adv_first] = 0
            # Slot 1 should re-arm fresh from its own liftoff AABB entry.
            self.wococo_hurdle_n_wrong_armed[adv_first] = False
            self.wococo_hurdle_stage[adv_second] = 3

        return reward
