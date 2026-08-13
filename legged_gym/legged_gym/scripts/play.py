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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import code
import csv
import pickle
import sys
import importlib.util

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.helpers import latest_model_file
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.math import quat_apply_yaw
import numpy as np
import re
import socket
import torch
import cv2
from collections import deque, defaultdict
import statistics
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time, sleep


def _save_depth_stage_set(env, out_dir, frame_idx):
    """Save the latest selected-env depth stages as grayscale PNGs + NPZ."""
    stages = getattr(env, "_depth_debug_stages", None)
    if not stages:
        return False
    os.makedirs(out_dir, exist_ok=True)
    near = float(env.cfg.depth.near_clip)
    far = float(env.cfg.depth.far_clip)
    raw = {}
    for name, tensor in stages.items():
        arr = tensor.detach().float().cpu().numpy()
        raw[name] = arr.astype(np.float32)
        if name == "04_policy_final":
            gray = np.clip((arr + 0.5) * 255.0, 0, 255)
        else:
            gray = np.clip(((-arr) - near) / (far - near) * 255.0, 0, 255)
        cv2.imwrite(os.path.join(out_dir, f"{frame_idx:06d}_{name}.png"),
                    gray.astype(np.uint8))
    # Keep the geometry that produced the frame next to the image data.
    if hasattr(env, "_depth_debug_spawn_xyz"):
        raw["spawn_root_xyz"] = env._depth_debug_spawn_xyz.detach().cpu().numpy().astype(np.float32)
    raw["current_root_xyz"] = env.root_states[:, :3].detach().cpu().numpy().astype(np.float32)
    if hasattr(env, "_depth_debug_hurdle_x_ranges"):
        raw["hurdle_x_ranges"] = env._depth_debug_hurdle_x_ranges.detach().cpu().numpy().astype(np.float32)
    if hasattr(env, "env_flyball_forbidden_x1"):
        raw["wall_front_x"] = env.env_flyball_forbidden_x1.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(os.path.join(out_dir, f"{frame_idx:06d}_stages.npz"), **raw)
    return True
from legged_gym.utils import webviewer

try:
    import imageio
except ImportError:
    imageio = None

# Mirrors env.last_term_code in legged_robot.reset_idx — keep the two in sync.
TERM_CODE_NAMES = {
    -1: "none yet",
    0: "reach_goal",
    1: "roll",
    2: "pitch",
    3: "height",
    4: "base_contact",
    5: "premature_wall",
    6: "timeout",
    7: "yaw_tracking",
    8: "plant_dwell",
    9: "lateral",
    10: "wococo_stage1_ground",
    11: "wococo_stage2_overshoot",
}


def _write_foot_force_outputs(rows, episodes, output_dir, exptid, level, dt):
    """Write phase-only foot-force samples and an aligned aggregate chart."""
    os.makedirs(output_dir, exist_ok=True)
    tag = os.environ.get('EP_FOOT_FORCE_TAG', '').strip()
    stem = f"{exptid}_level{level}_foot_forces_{len(episodes)}ep"
    if tag:
        stem += f"_{tag}"
    samples_path = os.path.join(output_dir, stem + ".csv")
    episodes_path = os.path.join(output_dir, stem + "_episodes.csv")
    chart_path = os.path.join(output_dir, stem + ".png")

    completed_ids = {ep['episode'] for ep in episodes}
    rows = [row for row in rows if row['episode'] in completed_ids]
    sample_fields = [
        'episode', 'env', 'episode_step', 'phase', 'phase_step', 'phase_time_s',
        'zone', 'foot', 'fx_n', 'fy_n', 'fz_n', 'force_n', 'contact', 'foot_x_m',
        'foot_y_m', 'foot_z_m',
    ]
    with open(samples_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sample_fields)
        writer.writeheader()
        writer.writerows(rows)

    episode_fields = ['episode', 'env', 'length_steps', 'length_s',
                      'term_code', 'termination', 'wall_samples', 'landing_samples']
    counts = defaultdict(lambda: defaultdict(int))
    for row in rows:
        counts[row['episode']][row['phase']] += 1
    for ep in episodes:
        ep['wall_samples'] = (counts[ep['episode']]['wall_front']
                              + counts[ep['episode']]['wall_rear'])
        ep['landing_samples'] = counts[ep['episode']]['landing']
    with open(episodes_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=episode_fields)
        writer.writeheader()
        writer.writerows(episodes)

    phase_styles = {
        'wall_front': ('#c7512c', 'Front-foot wall plant'),
        'wall_rear': ('#d79b27', 'Rear-foot wall plant'),
        'landing': ('#247b68', 'Landing'),
    }
    feet = list(dict.fromkeys(row['foot'] for row in rows))
    if rows and feet:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
        for ax, foot in zip(axes.flat, feet):
            for phase, (color, label) in phase_styles.items():
                phase_rows = [r for r in rows if r['foot'] == foot and r['phase'] == phase]
                by_step = defaultdict(list)
                for row in phase_rows:
                    by_step[row['phase_step']].append(row['force_n'])
                if not by_step:
                    continue
                # Suppress long tails from one or two unusually slow episodes.
                steps = np.array([s for s in sorted(by_step) if len(by_step[s]) >= 5])
                if not steps.size:
                    continue
                values = [np.asarray(by_step[s], dtype=float) for s in steps]
                median = np.array([np.median(v) for v in values])
                lo = np.array([np.percentile(v, 10) for v in values])
                hi = np.array([np.percentile(v, 90) for v in values])
                time_s = steps * dt
                ax.plot(time_s, median, color=color, linewidth=2, label=label)
                ax.fill_between(time_s, lo, hi, color=color, alpha=0.16)
            ax.set_title(foot.replace('_foot', '').upper())
            ax.grid(True, alpha=0.2)
            ax.set_xlabel('Time from phase entry (s)')
            ax.set_ylabel('Resultant contact force (N)')
        for ax in axes.flat[len(feet):]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.935),
                       ncol=3, frameon=False)
        fig.suptitle(f'Foot forces during wall plant and landing\n'
                     f'{exptid}, level {level}, {len(episodes)} completed episodes', y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.87))
        fig.savefig(chart_path, dpi=180, bbox_inches='tight')
        plt.close(fig)
    else:
        chart_path = None

    print(f'[foot-force] samples: {samples_path} ({len(rows)} rows)')
    print(f'[foot-force] episodes: {episodes_path}')
    if chart_path is not None:
        print(f'[foot-force] chart: {chart_path}')
    else:
        print('[foot-force] no wall-plant or landing samples; chart not generated')
    return samples_path, episodes_path, chart_path

def _print_eval_summary(stats, wall_angles):
    """Print the headless failure-mode tables (no --web/--video runs only).

    Three views of the same episode set: which termination fires, at which
    curriculum level, and where on the circuit (goal index / WoCoCo stage /
    signed distance to the wall face) it fires.
    """
    levels = sorted(stats['ep'].keys())
    codes_seen = sorted({c for lvl in levels for c in stats['term'][lvl]})
    total_ep = sum(stats['ep'][l] for l in levels)
    if total_ep == 0:
        print('[eval] no episodes finished yet — nothing to summarise')
        return
    print('\n' + '=' * 100)
    print(f'[eval] TERMINATION MODE x LEVEL   ({total_ep} episodes)')
    print('=' * 100)
    hdr = f"{'level':>5} {'wall°':>6} {'eps':>5} {'ep_len':>7} " + ''.join(
        f"{TERM_CODE_NAMES.get(c, c)[:13]:>14}" for c in codes_seen)
    print(hdr)
    for lvl in levels:
        n = stats['ep'][lvl]
        row = (f"{lvl:>5} {wall_angles.get(lvl, float('nan')):>6.1f} {n:>5} "
               f"{stats['ep_len'][lvl] / max(n, 1):>7.1f} ")
        row += ''.join(f"{stats['term'][lvl].get(c, 0) / max(n, 1):>14.3f}" for c in codes_seen)
        print(row)
    agg = {c: sum(stats['term'][l].get(c, 0) for l in levels) for c in codes_seen}
    print(f"{'ALL':>5} {'':>6} {total_ep:>5} "
          f"{sum(stats['ep_len'].values()) / total_ep:>7.1f} "
          + ''.join(f"{agg[c] / total_ep:>14.3f}" for c in codes_seen))

    print('\n' + '=' * 100)
    print('[eval] WHERE IT ENDS — per termination mode (mean over episodes)')
    print('=' * 100)
    print(f"{'termination':>24} {'eps':>6} {'share':>7} {'goal_idx':>9} "
          f"{'stage':>7} {'x-wall[m]':>10} {'lvl':>6}")
    for c in sorted(codes_seen, key=lambda c: -agg[c]):
        n = agg[c]
        if n == 0:
            continue
        print(f"{TERM_CODE_NAMES.get(c, c):>24} {n:>6} {n / total_ep:>7.3f} "
              f"{stats['goal'][c] / n:>9.2f} {stats['stage'][c] / n:>7.2f} "
              f"{stats['xwall'][c] / n:>10.2f} {stats['lvl_of_code'][c] / n:>6.2f}")

    if stats.get('zone'):
        znames = stats['zone_names']
        print('\n' + '=' * 100)
        print('[eval] WHICH OBSTACLE — termination x circuit zone (episode counts)')
        print('       zone from the env\'s own hurdle span / wall face / landing '
              'footholds; out vs back split by WoCoCo stage')
        print('=' * 100)
        print(f"{'termination':>24} " + ''.join(f"{z[:13]:>14}" for z in znames))
        _ztot = defaultdict(int)
        for c in sorted(codes_seen, key=lambda c: -agg[c]):
            if agg[c] == 0:
                continue
            row = ''.join(f"{stats['zone'][c].get(zi, 0):>14}" for zi in range(len(znames)))
            print(f"{TERM_CODE_NAMES.get(c, c):>24} {row}")
            for zi in range(len(znames)):
                _ztot[zi] += stats['zone'][c].get(zi, 0)
        print(f"{'TOTAL':>24} " + ''.join(f"{_ztot[zi]:>14}" for zi in range(len(znames))))
        print(f"{'share of all eps':>24} "
              + ''.join(f"{_ztot[zi] / total_ep:>14.3f}" for zi in range(len(znames))))
        _fail = total_ep - agg.get(0, 0)
        if _fail > 0:
            _zf = defaultdict(int)
            for c in codes_seen:
                if c == 0:
                    continue
                for zi in range(len(znames)):
                    _zf[zi] += stats['zone'][c].get(zi, 0)
            print(f"{'share of FAILURES':>24} "
                  + ''.join(f"{_zf[zi] / _fail:>14.3f}" for zi in range(len(znames))))

    print('\n' + '=' * 100)
    print('[eval] REWARD STREAM x LEVEL (mean per step, terms sorted by |mean|)')
    print('=' * 100)
    names = sorted(stats['rew_names'],
                   key=lambda k: -abs(sum(stats['rew'][l].get(k, 0.) for l in levels)
                                      / max(sum(stats['steps'].values()), 1)))
    print(f"{'reward term':>34} " + ''.join(f"{('L%d' % l):>9}" for l in levels) + f"{'ALL':>9}")
    tot_steps = sum(stats['steps'].values())
    for k in names:
        vals = ''.join(f"{stats['rew'][l].get(k, 0.) / max(stats['steps'][l], 1):>9.4f}"
                       for l in levels)
        allv = sum(stats['rew'][l].get(k, 0.) for l in levels) / max(tot_steps, 1)
        print(f"{k[:34]:>34} {vals}{allv:>9.4f}")
    print('=' * 100 + '\n', flush=True)


def _compose_video_frame(cam_rgb, plot_rgb, rew_rgb, w, h, depth_gray=None):
    """Composite the camera view (+ optional plot/reward panels) into one RGB frame.

    The camera image is the hero and fills every pixel not claimed by a panel:
      - No plot/reward panels (the common --use_camera case): the camera fills
        the whole frame.
      - Plot/reward panels present: they take the right column and the camera
        fills the rest on the left.
    The depth image, when present, is drawn as a small picture-in-picture inset
    in the bottom-left of the camera region instead of consuming a whole column
    (which used to leave the right 70% black when no plots were shown).

    Telemetry text is already burnt onto `cam_rgb` by webviewer._overlay_data,
    so it is deliberately NOT redrawn here — it appears exactly once.
    """
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    panel_imgs = [im for im in (plot_rgb, rew_rgb) if im is not None]
    if cam_rgb is None:
        cam_w, plot_w = 0, w
    elif panel_imgs:
        plot_w = int(w * 0.62)
        cam_w = w - plot_w
    else:
        plot_w, cam_w = 0, w

    if cam_rgb is not None and cam_w > 0:
        ch, cw = cam_rgb.shape[:2]
        scale = min(cam_w / cw, h / ch)
        new_w, new_h = max(1, int(cw * scale)), max(1, int(ch * scale))
        cam_resized = cv2.resize(cam_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x_off, y_off = (cam_w - new_w) // 2, (h - new_h) // 2
        frame[y_off:y_off + new_h, x_off:x_off + new_w] = cam_resized

        if depth_gray is not None:
            dh, dw = depth_gray.shape[:2]
            pip_w = max(1, int(cam_w * 0.28))
            pip_h = max(1, int(dh * pip_w / dw))
            if pip_w < cam_w and pip_h < h:
                depth_resized = cv2.resize(depth_gray, (pip_w, pip_h), interpolation=cv2.INTER_AREA)
                if depth_resized.ndim == 2:
                    depth_resized = cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2RGB)
                m = 12
                y1, x1 = m, cam_w - pip_w - m
                frame[y1:y1 + pip_h, x1:x1 + pip_w] = depth_resized
                cv2.rectangle(frame, (x1 - 2, y1 - 2),
                              (x1 + pip_w + 1, y1 + pip_h + 1), (90, 90, 90), 1)
                cv2.putText(frame, "depth", (x1, y1 + pip_h + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    if panel_imgs and plot_w > 0:
        y = 0
        for img_rgb in panel_imgs:
            ph = min(int(img_rgb.shape[0] * plot_w / img_rgb.shape[1]), h - y)
            if ph <= 0:
                break
            frame[y:y + ph, w - plot_w:] = cv2.resize(
                img_rgb, (plot_w, ph), interpolation=cv2.INTER_AREA)
            y += ph

    return frame


def _write_show_cam_urdf(env_cfg):
    """Ensure the play asset contains a solid D435i housing.

    The Go2 URDF contains the permanent, mass-bearing D435i model. Return that
    asset unchanged when its camera link is present; the generated visual-only
    fallback below remains useful for other robot assets.

    Isaac Gym's debug drawing can only produce wireframes, so a genuinely solid
    box has to be scene geometry. The camera is rigidly base-mounted, so a
    fixed-joint link at cfg.depth.position with rpy pitch = the mount angle
    sits exactly where warp_cam.render() places the optical centre (same
    R_y(pitch) convention as attach_camera). The link carries a VISUAL only —
    no <collision>, zero mass, mirroring the URDF's existing radar link — so
    contacts, self-geometry ray-casts and dynamics are all untouched, and
    collapse_fixed_joints merges it into `base` leaving the body count and
    every body index unchanged.

    The copy is regenerated on every --show_cam run and must live next to the
    original (mesh paths inside the URDF are relative). Its name keeps a single
    dot — Isaac Gym guesses the asset format from the extension and rejects
    anything like "foo.autogen.urdf". Returns its path.
    """
    src = env_cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    x, y, z = [float(v) for v in env_cfg.depth.position]
    pitch = np.radians(0.5 * (float(env_cfg.depth.angle[0]) + float(env_cfg.depth.angle[1])))
    # Real D435i: 90 mm wide x 25 mm deep x 25 mm tall. Camera frame is x
    # forward / y left / z up, so depth is the x extent and width the y extent.
    link = f"""
  <!-- AUTOGENERATED by play.py show_cam (visual only; regenerated each run).
       No double hyphens in this comment: XML forbids them, and the depth
       self-geometry loader parses this file with ElementTree, which (unlike
       Isaac's own URDF parser) rejects them. -->
  <link name="depth_camera_housing">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <box size="0.025 0.090 0.025" />
      </geometry>
      <material name="d435i_black">
        <color rgba="0.05 0.05 0.05 1" />
      </material>
    </visual>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <mass value="0" />
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0" />
    </inertial>
  </link>
  <joint name="depth_camera_housing_joint" type="fixed">
    <origin xyz="{x:g} {y:g} {z:g}" rpy="0 {pitch:.6f} 0" />
    <parent link="base" />
    <child link="depth_camera_housing" />
    <axis xyz="0 0 0" />
  </joint>
</robot>"""
    text = open(src).read()
    if '<link name="depth_camera_housing">' in text:
        return src
    if "</robot>" not in text:
        raise SystemExit(f"[show_cam] no </robot> tag in {src}")
    stem = os.path.splitext(os.path.basename(src))[0]
    dst = os.path.join(os.path.dirname(src), f"{stem}_show_cam.urdf")
    with open(dst, "w") as f:
        f.write(text.replace("</robot>", link, 1))
    return dst


def _load_env_cfg_from_log(log_pth, task):
    """Load env_cfg from log_pth.

    Tries env_cfg.pkl first (written by train.py with CLI overrides baked in).
    Falls back to importing the saved go2_config.py / legged_robot_config.py.
    Returns (None, None) if nothing loadable is found.
    """
    pkl_path = os.path.join(log_pth, 'env_cfg.pkl')
    if os.path.isfile(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                env_cfg = pickle.load(f)
            print(f"[play] Loaded env_cfg from {pkl_path}")
            return env_cfg, None
        except Exception as e:
            print(f"[play] Warning: could not load env_cfg.pkl: {e}, falling back to .py config")

    task_cfg_path = os.path.join(log_pth, f"{task}_config.py")
    base_cfg_path = os.path.join(log_pth, "legged_robot_config.py")
    if not os.path.isfile(task_cfg_path):
        return None, None

    BASE_KEY = 'legged_gym.envs.base.legged_robot_config'
    original_base = sys.modules.get(BASE_KEY)
    try:
        if os.path.isfile(base_cfg_path):
            spec = importlib.util.spec_from_file_location(BASE_KEY, base_cfg_path)
            saved_base = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(saved_base)
            sys.modules[BASE_KEY] = saved_base

        spec = importlib.util.spec_from_file_location(f'_saved_{task}_config', task_cfg_path)
        task_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(task_mod)

        cap = ''.join(w.capitalize() for w in task.split('_'))
        # Naming is inconsistent (Go2RoughCfg vs Go2StudentCfg): try both forms.
        env_cls = (getattr(task_mod, f'{cap}Cfg', None)
                   or getattr(task_mod, f'{cap}RoughCfg', None))
        train_cls = (getattr(task_mod, f'{cap}CfgPPO', None)
                     or getattr(task_mod, f'{cap}RoughCfgPPO', None))
        if env_cls is None:
            return None, None
        return env_cls(), (train_cls() if train_cls else None)
    except Exception as e:
        print(f"[play] Warning: could not load saved config: {e}")
        return None, None
    finally:
        if original_base is not None:
            sys.modules[BASE_KEY] = original_base
        elif BASE_KEY in sys.modules:
            del sys.modules[BASE_KEY]


def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if checkpoint==-1:
        model = latest_model_file(root, model_name_include)
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint

def _parabolic_prism_lines(bounds, samples=33, ribs=9):
    """Return line segments tracing one extruded parabolic clearance volume."""
    lo, hi = np.asarray(bounds[0]), np.asarray(bounds[1])
    xs = np.linspace(lo[0], hi[0], samples, dtype=np.float32)
    center_x = 0.5 * (lo[0] + hi[0])
    half_span = 0.5 * (hi[0] - lo[0])
    height = hi[2] - lo[2]
    zs = lo[2] + height * (1.0 - ((xs - center_x) / half_span) ** 2)

    segments = []
    for y in (lo[1], hi[1]):
        curve = np.stack((xs, np.full_like(xs, y), zs), axis=1)
        segments.extend(np.stack((curve[:-1], curve[1:]), axis=1))
    for idx in np.linspace(0, samples - 1, ribs, dtype=int):
        segments.append(np.array(
            [[xs[idx], lo[1], zs[idx]], [xs[idx], hi[1], zs[idx]]],
            dtype=np.float32))
    segments.extend((
        np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]]], dtype=np.float32),
        np.array([[lo[0], hi[1], lo[2]], [hi[0], hi[1], lo[2]]], dtype=np.float32),
        np.array([[lo[0], lo[1], lo[2]], [lo[0], hi[1], lo[2]]], dtype=np.float32),
        np.array([[hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]]], dtype=np.float32),
    ))
    return np.asarray(segments, dtype=np.float32), xs, zs


def _draw_hurdle_clearance_parabolas(env, color=(1.0, 0.35, 0.0)):
    """Draw the exact parabolic clearance prisms for the tracked environment.

    Isaac Gym's debug API supports lines but not translucent faces, so the live
    viewer uses dense curves and lateral ribs. Camera/video overlays use filled
    faces in `_overlay_clearance_parabolas`. Both are sampled from the exact
    analytical bounds used by the collision reward.

    Must be called AFTER env.step(): legged_robot.render() does the
    gym.clear_lines(), so lines added here survive until the next frame's clear.
    Viewer-only — Isaac Gym debug lines do not appear in camera-sensor images
    (same limitation as _draw_goals). For the --video / --web camera frames the
    prisms are projected in software instead, see _overlay_clearance_parabolas().
    """
    if getattr(env, 'viewer', None) is None:
        return
    region = getattr(env, 'env_flyball_hurdle_clear_region', None)
    if region is None:
        return
    bounds = region[env.lookat_id].cpu().numpy()
    for prism in bounds:
        if prism[1, 0] <= prism[0, 0]:                   # degenerate/unfilled slot
            continue
        segments, _, _ = _parabolic_prism_lines(prism)
        colors = np.repeat(np.asarray(color, dtype=np.float32)[None, :],
                           segments.shape[0], axis=0)
        env.gym.add_lines(env.viewer, env.envs[env.lookat_id],
                          segments.shape[0], segments, colors)


def _perceived_scandot_world_data(env):
    """Return tracked-env scandot XYZ and pre-scale observation values.

    `height_points` contains robot-relative XY locations with Z=0. Terrain Z
    must therefore come from `measured_heights`; adding the base position in all
    three axes incorrectly puts every dot at base height.
    """
    if not hasattr(env, 'measured_heights') or not hasattr(env, 'height_points'):
        return None, None
    env_id = env.lookat_id
    heights = env.measured_heights[env_id].detach().cpu().numpy()
    local_points = quat_apply_yaw(
        env.base_quat[env_id].repeat(env.height_points[env_id].shape[0]),
        env.height_points[env_id]).detach().cpu().numpy()
    base = env.root_states[env_id, :3].detach().cpu().numpy()
    world = local_points + base[None, :]
    world[:, 2] = heights
    obs_val = np.clip(base[2] - 0.3 - heights, -1.0, 1.0)
    return world, obs_val


def _scandot_turbo_rgb(obs_val, num_bins=256):
    """Map scandot observations to RGB TURBO colours."""
    bins = np.clip(
        np.round((1.0 - obs_val) * 0.5 * (num_bins - 1)),
        0, num_bins - 1)
    color_codes = np.round(bins * 255.0 / max(num_bins - 1, 1)).astype(np.uint8)
    bgr = cv2.applyColorMap(color_codes.reshape(-1, 1), cv2.COLORMAP_TURBO)[:, 0]
    return bgr[:, ::-1].copy()


def _draw_perceived_scandots(env):
    """Draw the post-pooling/post-occlusion scandots used by the policy."""
    if getattr(env, 'viewer', None) is None:
        return
    world, obs_val = _perceived_scandot_world_data(env)
    if world is None:
        return

    num_bins = 13
    if not hasattr(env, '_play_scandot_geoms'):
        lut_obs = np.linspace(1.0, -1.0, num_bins)
        lut_rgb = _scandot_turbo_rgb(lut_obs, num_bins=num_bins)
        env._play_scandot_geoms = [
            gymutil.WireframeSphereGeometry(
                0.02, 6, 6, None, color=tuple(float(c) / 255.0 for c in rgb))
            for rgb in lut_rgb
        ]
    bins = np.clip(
        np.round((1.0 - obs_val) * 0.5 * (num_bins - 1)),
        0, num_bins - 1).astype(int)
    for point, bin_id in zip(world, bins):
        pose = gymapi.Transform(gymapi.Vec3(*[float(v) for v in point]), r=None)
        gymutil.draw_lines(env._play_scandot_geoms[bin_id], env.gym, env.viewer,
                           env.envs[env.lookat_id], pose)


def _pick_proj_convention(view, proj, ref_world):
    """Resolve the camera matrix convention ONCE for a frame.

    get_camera_view_matrix / get_camera_proj_matrix are column/row-major
    ambiguous (webviewer._world_to_pixel probes all four combinations per
    point). Probing per point is fine for independent circles but not for box
    edges: two corners resolved under different conventions would be joined by
    a nonsense line. So probe with one reference point that is known to be on
    screen — the tracked robot's base, which every camera here looks at — and
    return that (V, P) pair for all corners of the frame.
    """
    p = np.array([ref_world[0], ref_world[1], ref_world[2], 1.0])
    for V in (view, view.T):
        for P in (proj, proj.T):
            clip = P @ (V @ p)
            if clip[3] <= 0:
                continue
            nx, ny = clip[0] / clip[3], clip[1] / clip[3]
            if -1.0 <= nx <= 1.0 and -1.0 <= ny <= 1.0:
                return V, P
    return None


def _overlay_clearance_parabolas(image, env, gym, sim, env_handle, cam_handle,
                                 color=(255, 120, 0), thickness=2, alpha=0.20):
    """Draw translucent parabolic clearance prisms into a camera RGB frame.

    Software equivalent of _draw_hurdle_clearance_parabolas() for paths that
    read pixels out of a camera sensor (--video, --web), where Isaac Gym debug
    lines are not rendered. Same source buffer, so the two agree.

    No depth test is possible here, so the translucent surface draws over the
    terrain rather than being occluded by it. Returns a new image; the input is
    not modified.
    """
    region = getattr(env, 'env_flyball_hurdle_clear_region', None)
    if region is None or image is None:
        return image
    bounds = region[env.lookat_id].cpu().numpy()
    real = [b for b in bounds if b[1, 0] > b[0, 0]]      # skip sentinel slots
    if not real:
        return image

    try:
        view = np.array(gym.get_camera_view_matrix(sim, env_handle, cam_handle))
        proj = np.array(gym.get_camera_proj_matrix(sim, env_handle, cam_handle))
    except Exception:
        return image

    conv = _pick_proj_convention(view, proj,
                                 env.root_states[env.lookat_id, :3].cpu().numpy())
    if conv is None:
        return image
    V, P = conv

    h, w = image.shape[:2]
    img = np.ascontiguousarray(image.copy())
    fill = img.copy()

    def project(points):
        pix = []
        for point in points:
            pt = np.array([point[0], point[1], point[2], 1.0])
            clip = P @ (V @ pt)
            if clip[3] <= 0:
                pix.append(None)
                continue
            nx, ny = clip[0] / clip[3], clip[1] / clip[3]
            pix.append((int((nx + 1.0) * 0.5 * w),
                        int((1.0 - (ny + 1.0) * 0.5) * h)))
        return pix

    all_segments = []
    for prism in real:
        segments, xs, zs = _parabolic_prism_lines(prism)
        lo, hi = prism
        faces = []
        for i in range(len(xs) - 1):
            faces.extend((
                [[xs[i], lo[1], zs[i]], [xs[i + 1], lo[1], zs[i + 1]],
                 [xs[i + 1], hi[1], zs[i + 1]], [xs[i], hi[1], zs[i]]],
                [[xs[i], lo[1], lo[2]], [xs[i + 1], lo[1], lo[2]],
                 [xs[i + 1], lo[1], zs[i + 1]], [xs[i], lo[1], zs[i]]],
                [[xs[i], hi[1], lo[2]], [xs[i], hi[1], zs[i]],
                 [xs[i + 1], hi[1], zs[i + 1]], [xs[i + 1], hi[1], lo[2]]],
            ))
        for face in faces:
            projected = project(face)
            if any(p is None for p in projected):
                continue
            cv2.fillConvexPoly(fill, np.asarray(projected, dtype=np.int32),
                               color, cv2.LINE_AA)
        all_segments.extend(segments)

    img = cv2.addWeighted(fill, alpha, img, 1.0 - alpha, 0.0)
    for segment in all_segments:
        pix = project(segment)
        if pix[0] is not None and pix[1] is not None:
            cv2.line(img, pix[0], pix[1], color, thickness, cv2.LINE_AA)
    return img


def _overlay_foothold_targets(image, env, gym, sim, env_handle, cam_handle,
                              color=(255, 40, 210), thickness=2):
    """Project active foothold targets and flat acceptance-radius circles.

    The live viewer gets these from LeggedRobot._draw_goals(), but Isaac Gym
    debug lines are not rendered by camera sensors.  The large 3-D acceptance
    balls are intentionally omitted; the flat circles use the exact stage/radius
    pair for the active footholds.
    """
    if image is None:
        return image
    if not (hasattr(env, 'env_class')
            and hasattr(env, 'env_flyball_foot_target')
            and hasattr(env, 'wococo_contact_stage')
            and bool(((env.env_class[env.lookat_id] == 19)
                      | (env.env_class[env.lookat_id] == 15)).item())):
        return image
    stage = int(env.wococo_contact_stage[env.lookat_id].item())
    if stage >= getattr(env, '_wococo_sentinel_stage', 99):
        return image
    if (getattr(env, '_wococo_design', '') == 'P4L' and stage == 2
            and getattr(env, '_wococo_landing_posture_only', lambda: False)()):
        # Posture-only landing has no stage-2 footholds (the branch below draws PLANT).
        return image
    if (getattr(env, '_wococo_design', '') == 'P4L' and stage == 2
            and hasattr(env, 'env_flyball_landing_foot_target')):
        targets = env.env_flyball_landing_foot_target[env.lookat_id].detach().cpu().numpy()
        radii = env._wococo_landing_accept_radius()
        color = (255, 40, 210)
        active_ids = [0, 1]  # landing target array is front-feet only
    else:
        targets = env.env_flyball_foot_target[env.lookat_id].detach().cpu().numpy()
        radii = env._wococo_rad('wococo_r_wall',
                                'wococo_foot_region_radius', 0.18)
        color = (255, 140, 20)
        active = env._wococo_active_feet_mask(env.wococo_contact_stage)[env.lookat_id]
        active_ids = torch.nonzero(active, as_tuple=False).flatten().tolist()
    if not active_ids:
        return image
    if torch.is_tensor(radii):
        radii = radii[env.lookat_id].detach().cpu().numpy()
    else:
        radii = np.full(targets.shape[0], float(radii), dtype=np.float32)
    try:
        view = np.array(gym.get_camera_view_matrix(sim, env_handle, cam_handle))
        proj = np.array(gym.get_camera_proj_matrix(sim, env_handle, cam_handle))
    except Exception:
        return image

    base = env.root_states[env.lookat_id, :3].detach().cpu().numpy()
    conv = _pick_proj_convention(view, proj, base)
    if conv is None:
        return image
    V, P = conv
    img = np.ascontiguousarray(image.copy())

    h, w = img.shape[:2]

    def project(point):
        clip = P @ (V @ np.array([point[0], point[1], point[2], 1.0]))
        if clip[3] <= 0:
            return None
        nx, ny = clip[0] / clip[3], clip[1] / clip[3]
        return int((nx + 1.0) * 0.5 * w), int((1.0 - (ny + 1.0) * 0.5) * h)

    # Landing targets hold only the front feet: ignore incompatible IDs.
    for foot_id in active_ids:
        if foot_id >= len(targets) or foot_id >= len(radii):
            continue
        centre, radius = targets[foot_id], radii[foot_id]
        centre_px = project(centre)
        if centre_px is not None:
            cv2.circle(img, centre_px, 4, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, centre_px, 3, color, -1, cv2.LINE_AA)
        theta = np.linspace(0.0, 2.0 * np.pi, 49, dtype=np.float32)
        circle = np.column_stack((
            centre[0] + float(radius) * np.cos(theta),
            centre[1] + float(radius) * np.sin(theta),
            np.full_like(theta, centre[2] + 0.012)))
        pixels = [project(point) for point in circle]
        for p0, p1 in zip(pixels[:-1], pixels[1:]):
            if p0 is not None and p1 is not None:
                cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)
    return img


def _overlay_perceived_scandots(image, env, gym, sim, env_handle, cam_handle,
                                radius=4):
    """Overlay the actual policy scandot heights onto a camera RGB frame."""
    if image is None:
        return image
    try:
        view = np.array(gym.get_camera_view_matrix(sim, env_handle, cam_handle))
        proj = np.array(gym.get_camera_proj_matrix(sim, env_handle, cam_handle))
    except Exception:
        return image

    world, obs_val = _perceived_scandot_world_data(env)
    if world is None:
        return image
    base = env.root_states[env.lookat_id, :3].detach().cpu().numpy()
    conv = _pick_proj_convention(view, proj, base)
    if conv is None:
        return image
    V, P = conv
    img = np.ascontiguousarray(image.copy())
    colors = _scandot_turbo_rgb(obs_val)
    h, w = img.shape[:2]
    for point, color in zip(world, colors):
        clip = P @ (V @ np.array([point[0], point[1], point[2], 1.0]))
        if clip[3] <= 0:
            continue
        nx, ny = clip[0] / clip[3], clip[1] / clip[3]
        px = int((nx + 1.0) * 0.5 * w)
        py = int((1.0 - (ny + 1.0) * 0.5) * h)
        if -radius <= px < w + radius and -radius <= py < h + radius:
            rgb = tuple(int(v) for v in color)
            cv2.circle(img, (px, py), radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), radius, rgb, -1, cv2.LINE_AA)
    return img


def _display_reachable():
    """True if the X display named by $DISPLAY can actually be opened.

    A dead ssh X11-forwarding channel leaves DISPLAY set but unusable, and the
    Isaac Gym viewer then dies mid-run with "The X11 connection broke (error 1)"
    — after terrain generation, so minutes of setup are lost. Probing up front
    lets play.py drop to offscreen rendering instead: NOTHING here needs a
    display except the GUI viewer window. Camera sensors (mp4, --web) render
    through Vulkan on the GPU with no X server at all.
    """
    disp = os.environ.get('DISPLAY', '')
    m = re.match(r'^(?P<host>[^:]*?)(?:/[^:]*)?:(?P<num>\d+)', disp)
    if not m:
        return False
    host, num = m.group('host'), int(m.group('num'))
    if host in ('', 'unix'):
        return os.path.exists('/tmp/.X11-unix/X{}'.format(num))
    try:
        # Forwarded displays are TCP on 6000+N; refused means the ssh session is gone.
        socket.create_connection((host, 6000 + num), timeout=2.0).close()
        return True
    except OSError:
        return False


def play(args):
    if not args.exptid:
        raise SystemExit("[play] --exptid is required and must exactly match a log directory")
    if not args.headless and not _display_reachable():
        args.headless = True
        print("[play] no reachable X display (DISPLAY={}) — running headless with "
              "offscreen rendering (no GUI window; mp4/--web still render)"
              .format(os.environ.get('DISPLAY', '') or 'unset'))
    _need_viewer = args.web or args.video
    if _need_viewer:
        web_viewer = webviewer.WebViewer(start_server=args.web, top_view=getattr(args, 'top_view', False))
    faulthandler.enable()
    exptid = args.exptid
    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    resolved_log_pth = os.path.abspath(log_pth)
    if not os.path.isdir(resolved_log_pth):
        raise SystemExit(
            f"[play] --exptid must exactly match an existing log directory: "
            f"{resolved_log_pth}"
        )
    saved_env_cfg, _ = _load_env_cfg_from_log(resolved_log_pth, args.task)
    use_live_config = True
    if saved_env_cfg is not None and not use_live_config:
        saved_env_cfg.seed = train_cfg.seed
        env_cfg = saved_env_cfg
        print(f"[play] Loaded config from {resolved_log_pth}")
    elif saved_env_cfg is not None:
        print(f"[play] Using live {args.task} config instead of the saved terrain config")
    else:
        print(f"[play] No saved config in {resolved_log_pth}, using default")
    if args.nodelay:
        # Legacy/scheduled-delay tasks: flatten the viewer delay to 0.
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = 16 if not args.save else 64
    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 10
    # One column when only rendering: on the 1 cm heightfield 5 columns is ~47 M
    # triangles per frame for ONE robot. Analysis keeps 5 for per-column tile jitter.
    env_cfg.terrain.num_cols = 1 if _need_viewer else 5
    env_cfg.terrain.max_init_terrain_level = env_cfg.terrain.num_rows - 1
    env_cfg.terrain.height = [0.02, 0.02]
    if getattr(args, 'zero_terrain_roughness', False):
        if getattr(args, 'terrain_height', None) is not None:
            raise ValueError("--zero_terrain_roughness conflicts with --terrain_height")
        args.terrain_height = '0'
        print("[play] --zero_terrain_roughness: terrain roughness forced to zero")
    if getattr(args, 'terrain_height', None) is not None:
        # terrain.height is the add_roughness amplitude: pin to 0 for quasi-static
        # measurements, where ±2 cm of noise smears the contact points.
        _th = str(args.terrain_height).strip()
        if ':' in _th:
            _lo, _hi = (float(t) for t in _th.split(':', 1))
            if _hi < _lo:
                raise ValueError(f"--terrain_height '{args.terrain_height}': hi < lo")
        else:
            _lo = _hi = float(_th)
        env_cfg.terrain.height = [_lo, _hi]
        print(f"[play] --terrain_height: roughness amplitude "
              + (f"pinned to {_lo:.3f} m" if _lo == _hi else f"sampled from [{_lo:.3f}, {_hi:.3f}] m")
              + "  [play default was 0.020 m]")
    if not getattr(env_cfg.terrain, 'terrain_dict', None):
        raise RuntimeError(f"[play] task '{args.task}' config has no terrain_dict to render")
    print(f"[play] terrain mix from {args.task} config: "
          + ", ".join(f"{k}={v}" for k, v in env_cfg.terrain.terrain_dict.items() if v))

    if args.terrain_only is not None:
        if args.terrain_only not in env_cfg.terrain.terrain_dict:
            raise ValueError(f"--terrain_only '{args.terrain_only}' not in terrain_dict keys: {list(env_cfg.terrain.terrain_dict.keys())}")
        for k in env_cfg.terrain.terrain_dict:
            env_cfg.terrain.terrain_dict[k] = 0.0
        env_cfg.terrain.terrain_dict[args.terrain_only] = 1.0
        print(f"[play] --terrain_only: locked to {args.terrain_only}")

    # Hurdle lateral width is the one flyball jitter NOT tied to difficulty, so when
    # randomised the per-level tables mix wide and narrow tiles; pin it for A/B evals.
    # It is also the only way to reproduce a past run's hurdle geometry: env_cfg.pkl
    # restores INSTANCE attributes only, so flyball_* resolves from the live config.
    if getattr(args, "hurdle_width", None) is not None:
        _wr = getattr(env_cfg.terrain, 'flyball_single_hurdle_width_range', None)
        _wfix = getattr(env_cfg.terrain, 'flyball_single_hurdle_width', None)
        _tok = str(args.hurdle_width).strip().lower()
        if _tok in ('min', 'max'):
            if _wr is None:
                raise ValueError(
                    f"--hurdle_width {_tok} needs flyball_single_hurdle_width_range in the "
                    f"{args.task} terrain config, but it is None (fixed width "
                    f"{_wfix} m). Pass an explicit width in metres, or 'lo:hi' to sample.")
            _lo = _hi = float(_wr[0] if _tok == 'min' else _wr[1])
        elif ':' in _tok:
            _a, _b = _tok.split(':', 1)
            _lo, _hi = float(_a), float(_b)
            if _hi < _lo:
                raise ValueError(f"--hurdle_width '{args.hurdle_width}': hi < lo")
        else:
            _lo = _hi = float(_tok)
        # Always go through the RANGE, even when pinning (lo == hi), so the same RNG
        # stream is consumed and every other per-tile draw stays bit-identical.
        env_cfg.terrain.flyball_single_hurdle_width_range = (_lo, _hi)
        env_cfg.terrain.flyball_single_hurdle_width = _lo
        _was = (f"sampled {tuple(_wr)}" if _wr is not None else f"fixed {_wfix} m")
        print(f"[play] --hurdle_width {args.hurdle_width}: hurdle width "
              + (f"pinned to {_lo:.2f} m" if _lo == _hi
                 else f"sampled per tile from ({_lo:.2f}, {_hi:.2f}) m")
              + f"  [live config had: {_was}]")

    if getattr(args, "hurdle_height", None) is not None:
        _hh = float(args.hurdle_height)
        if _hh < 0.0:
            raise ValueError("--hurdle_height must be non-negative")
        env_cfg.terrain.flyball_single_hurdle_height_range = (_hh, _hh)
        env_cfg.terrain.flyball_single_hurdle_height_max = _hh
        print(f"[play] --hurdle_height: pinned to {_hh:.3f} m")

    if getattr(args, "hurdle_front_from_spawn", None) is not None:
        _hf = float(args.hurdle_front_from_spawn)
        if _hf < 0.0:
            raise ValueError("--hurdle_front_from_spawn must be non-negative")
        env_cfg.terrain.flyball_hurdle_front_from_spawn = _hf
        env_cfg.terrain.flyball_hurdle_front_from_spawn_jitter = 0.0
        print(f"[play] --hurdle_front_from_spawn: pinned to {_hf:.3f} m")

    if getattr(args, "no_hurdle_roughness", False):
        env_cfg.terrain.flyball_hurdle_roughness_height = (0.0, 0.0)
        print("[play] --no_hurdle_roughness: hurdle top roughness disabled")

    if getattr(args, "all_rough", False):
        # Training draws smooth tiles with prob flyball_roughness_zero_prob (0.5);
        # force every tile rough for renders that must show roughness.
        env_cfg.terrain.flyball_roughness_zero_prob = 0.0
        print("[play] --all_rough: flyball_roughness_zero_prob=0 — every tile rough")

    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_difficulty = True
    # cfg.terrain.curriculum does DOUBLE DUTY: BUILD mode (row i gets difficulty
    # i/(num_rows-1), else U(0.7, 1.0) on EVERY row) plus the live promote/demote.
    # Switching it off to freeze levels rebuilt every row hard — the intended level-1
    # 35 deg wall came out at 61.5-75 deg (+-5 deg jitter). Freeze with fixed_level +
    # no_demotion + max_promoted_level=-1 instead.
    def _pin_level(lvl):
        env_cfg.terrain.max_init_terrain_level = lvl
        env_cfg.terrain.curriculum = True      # deterministic per-row difficulty
        env_cfg.terrain.max_difficulty = False # ...not U(0.7, 1.0) on every row
        env_cfg.terrain.fixed_level = True     # every env spawns on row `lvl`
        env_cfg.terrain.no_demotion = True     # freeze: no move_down
        env_cfg.terrain.max_promoted_level = -1  # freeze: no move_up
        env_cfg.commands.curriculum = False
        _rng = getattr(env_cfg.terrain, 'flyball_wall_angle_deg_range', None)
        if _rng is not None and env_cfg.terrain.num_rows > 1:
            _d = lvl / (env_cfg.terrain.num_rows - 1)
            print(f"[play] --level {lvl}: difficulty {_d:.3f} -> wall "
                  f"{_rng[0] + _d * (_rng[1] - _rng[0]):.1f} deg "
                  f"(+-{getattr(env_cfg.terrain, 'flyball_wall_angle_jitter_deg', 0.0):.0f} deg jitter)")

    if args.level is not None:
        _pin_level(max(0, min(int(args.level), env_cfg.terrain.num_rows - 1)))
    elif args.max_level:
        # uses max_ranges [1.5, 2.0], no near-zero cmd_vel_x
        _pin_level(env_cfg.terrain.num_rows - 1)
    env_cfg.terrain.save_vts = args.save_vts
    if args.save_vts_full_res:
        env_cfg.terrain.vts_export_cell_size = env_cfg.terrain.horizontal_scale
    
    # Depth camera pitch (deg down). Default 20 = training nominal (depth.angle=[15,25],
    # i.e. 20 +-5 deg mount noise); any other value is a camera it never trained on.
    env_cfg.depth.angle = [20, 20]

    # --cam_pitch_deg: override the 20 deg default (e.g. 7.6, the real Go2 mount).
    if getattr(args, "cam_pitch_deg", None) is not None:
        _p = float(args.cam_pitch_deg)
        env_cfg.depth.angle = [_p, _p]
        print(f"[depth] camera pitch pinned to {_p:g} deg down "
              f"(default is 20; training samples [15, 25])")

    # --show_cam: give the URDF a solid D435i housing at the mount pose (Go2 has one;
    # others get a visual copy). Must run AFTER pitch pinning and BEFORE make_env.
    if getattr(args, "show_cam", False):
        env_cfg.asset.file = _write_show_cam_urdf(env_cfg)
        print(f"[show_cam] solid D435i housing (90x25x25 mm) at base-frame "
              f"{list(env_cfg.depth.position)} m, "
              f"{0.5 * (env_cfg.depth.angle[0] + env_cfg.depth.angle[1]):g} deg down "
              f"-> {env_cfg.asset.file}")

    # --no_depth_noise: stochastic degradation off, but the sensor-physics constants
    # are KEPT — real D435 properties, not noise.
    if getattr(args, "no_depth_noise", False):
        dcfg = env_cfg.depth
        _off = {
            "dis_noise": 0.0, "noise_gaussian": 0.0, "noise_dropout": 0.0,
            "dropout_cluster": False, "dropout_jitter": 0.0,
            "dropout_to_far": False, "dropout_far_frac": None,
            "pw_noise": False, "stereo_shadow": False,
            "noise_scale_per_env": False,
        }
        _changed = []
        for k, v in _off.items():
            if hasattr(dcfg, k) and getattr(dcfg, k) != v:
                _changed.append(f"{k}: {getattr(dcfg, k)} -> {v}")
                setattr(dcfg, k, v)
        if hasattr(dcfg, "noise_scale_min"):
            dcfg.noise_scale_min = 0.0
        print(f"[depth] noise DISABLED ({len(_changed)} knobs): " + "; ".join(_changed)
              if _changed else "[depth] noise DISABLED (already clean)")
    if getattr(args, "no_depth_hole_fill", False):
        env_cfg.depth.hole_fill = False
        print("[depth] deployment hole fill DISABLED (PW holes/noise remain active)")
    # --depth_noise_scale / --full_depth_noise: pin every env to ONE noise level; the
    # curriculum otherwise spreads envs over linspace([noise_scale_min, 1]).
    #   ONE linear knob: rate = 2s (amplitudes/densities), phys = clamp(2s, 0, 1)
    #   (stereo shadow / own legs / min range / blur / lag). s=0 is the ideal ray-cast,
    #   s=0.5 the configured reality-calibrated rates, s=1.0 the 2x headroom end.
    _dns = getattr(args, "depth_noise_scale", None)
    if getattr(args, "full_depth_noise", False):
        if _dns is not None and _dns != 1.0:
            raise SystemExit("[play] --full_depth_noise conflicts with --depth_noise_scale "
                             f"{_dns}; pass only one")
        _dns = 1.0
    if _dns is not None:
        if getattr(args, "no_depth_noise", False):
            raise SystemExit("[play] --depth_noise_scale/--full_depth_noise conflicts with "
                             "--no_depth_noise")
        if not (0.0 <= _dns <= 1.0):
            raise SystemExit(f"[play] --depth_noise_scale must be in [0, 1], got {_dns}")
        if not args.use_camera:
            raise SystemExit("[play] --depth_noise_scale/--full_depth_noise needs --use_camera")

    if args.use_camera:
        # Force Warp so all three noise draws match training.
        env_cfg.depth.use_warp = True

        # Self-geometry is NOT recoverable from env_cfg.pkl, so a student distilled
        # before it existed would be evaluated with its legs in frame. Announce it.
        if args.self_geom and args.no_self_geom:
            raise SystemExit("[play] pass at most one of --self_geom / --no_self_geom")
        if args.self_geom:
            env_cfg.depth.self_geometry = True
        elif args.no_self_geom:
            env_cfg.depth.self_geometry = False
        _sg = getattr(env_cfg.depth, 'self_geometry', False)
        _src = 'CLI override' if (args.self_geom or args.no_self_geom) else f'{args.task} config'
        print(f"[depth] self-geometry {'ON' if _sg else 'OFF'} ({_src}) — must match how "
              f"'{args.exptid}' was distilled, or the student sees an unfamiliar depth image")
    # By DEFAULT play.py mirrors the task's training DR as the config resolved it. One
    # gap: the env flattens per-env action latency under a GUI viewer, unless --full-dr.
    if getattr(args, "no_dr", False):
        # Every DR term is flag-gated: clearing randomize_* + pushes + noise is enough.
        dr = env_cfg.domain_rand
        disabled = []
        for attr in dir(dr):
            if attr.startswith("randomize_") and isinstance(getattr(dr, attr), bool):
                setattr(dr, attr, False)
                disabled.append(attr)
        dr.push_robots = False
        if hasattr(dr, "dr_lock_to_terrain_level"):
            dr.dr_lock_to_terrain_level = False
        # The fixed action delay is NOT randomization but a deploy-ENFORCED constant
        # (16 ms proprio pad + ~3 ms firmware/DDS = one step), so it stays on.
        if getattr(dr, "action_delay_latch_mean_level", None) is None:
            dr.action_delay = False
        env_cfg.noise.add_noise = False
        print(f"[play] --no-dr: disabled DR ({', '.join(disabled)}) + pushes + obs noise")
    elif getattr(args, "full_dr", False):
        env_cfg.domain_rand.force_action_latency = True
        if not getattr(env_cfg.domain_rand, "randomize_action_latency", False):
            print("[play] --full-dr: note — this task has randomize_action_latency=False "
                  "(fixed deploy delay instead), so there is no per-env latency to un-flatten")
        print("[play] --full-dr: exact training DR (incl. per-env action latency under viewer)")
    else:
        print("[play] using training DR from env_cfg (pass --full-dr to also un-flatten viewer action latency)")

    # Deploy-matched latency: latch-mode configs use a FIXED 1-step (20 ms) delay that
    # engages at action_delay_latch_mean_level. Play is EVALUATION, so latch from step
    # 0 — 1-16 envs may never reach that mean, leaving play lag-free and flattering.
    _dr = env_cfg.domain_rand
    _latch_at = getattr(_dr, "action_delay_latch_mean_level", None)
    if args.nodelay:
        # Kill the delay outright: action_delay_view only reaches step() under a GUI
        # viewer, so on headless latch-STEP configs --nodelay was silently ignored.
        _dr.action_delay = False
        _dr.action_delay_start_latched = False
        print("[play] --nodelay: action delay OFF (0 ms) — the policy sees "
              "its own actions with zero lag; NOT deploy-faithful")
    elif _latch_at is not None:
        if _dr.action_delay:
            _dr.action_delay_start_latched = True
            print(f"[play] action delay LATCHED from step 0: fixed 1 step "
                  f"(20 ms, deploy-enforced; training latch was mean level {_latch_at})")
        else:
            print("[play] WARNING: action_delay=False in this config — evaluating without "
                  "the 20 ms delay the deploy stack enforces")
    if getattr(_dr, "randomize_action_latency", False) or getattr(_dr, "randomize_proprio_latency", False):
        if _latch_at is not None:
            print("[play] WARNING: latency DR is on alongside the fixed deploy delay — "
                  "these double-count the same physical lag (see go2_teacher_config)")

    if args.use_camera:
        _d = env_cfg.depth
        _lat = list(getattr(_d, 'latency_range', [0, 0]))
        _lo, _hi = int(min(_lat)), int(max(_lat))
        # Latency is quantized in camera-capture frames, capture_interval steps apart.
        _cap = int(getattr(_d, 'capture_interval', _d.update_interval))
        _ms = 1000.0 * env_cfg.control.decimation * env_cfg.sim.dt * _cap
        _ramp = getattr(_d, 'latency_ride_dr_ramp', True)
        print(f"[play] depth latency: {_lo}-{_hi} capture frames "
              f"({_lo * _ms:.0f}-{_hi * _ms:.0f} ms)"
              + ("" if _ramp else " — full range from step 0 (no maturity ramp)"))
        _eb = float(getattr(_d, 'empty_below_m', 0.0))
        print(f"[play] RealSense min-range emulation: "
              + (f"pixels closer than {_eb:.2f} m -> far (matches deploy invalid->far)"
                 if _eb > 0.0 else "off (empty_below_m=0)"))

    # Pin the WoCoCo yaw-tracking hold ON so --rew shows it. Reward-only: play is pure
    # inference. Both knobs are read once in _init_buffers, so set them BEFORE make_env
    # — non-zero hold window, and min-mean-level gate to 0 (the training gate is 5.0).
    if getattr(args, "yaw_sentinel_hold", False):
        hold_s = float(getattr(env_cfg.rewards, "yaw_tracking_post_sentinel_hold_s", 0.0))
        if hold_s <= 0.0:
            hold_s = 1.0
            env_cfg.rewards.yaw_tracking_post_sentinel_hold_s = hold_s
        env_cfg.rewards.yaw_post_sentinel_hold_min_mean_level = 0.0
        print(f"[play] --yaw_sentinel_hold: post-sentinel yaw max-pin ON "
              f"(hold={hold_s:.2f}s, curriculum gate→0; affects --rew only)")

    # Camera sensors (mp4, --web) need a render context that base_task switches off
    # under --headless (handles come back -1, frames vanish). It is offscreen Vulkan.
    if _need_viewer:
        env_cfg.env.offscreen_render = True

    depth_latent_buffer = []
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    if getattr(args, "spawn_after_second_hurdle", None) is not None:
        _clearance = float(args.spawn_after_second_hurdle)
        if _clearance < 0.0:
            raise ValueError("--spawn_after_second_hurdle must be non-negative")
        _ranges = getattr(env, "env_flyball_hurdle_x_ranges", None)
        if _ranges is None or _ranges.ndim != 3 or _ranges.shape[1] < 2:
            raise RuntimeError("--spawn_after_second_hurdle requires two generated flyball hurdles")
        _second_back = _ranges[:, 1, 1]
        # Empty hurdle slots use the finite sentinel (-1, -1); an isfinite()-only check
        # would place robots at x=-0.5 m.
        _valid = torch.isfinite(_second_back) & (_ranges[:, 1, 0] >= 0.0)
        if not bool(_valid.any()):
            raise RuntimeError("--spawn_after_second_hurdle found no valid second hurdle")
        # Keep each env's y/z and heading; place its base past hurdle slot 1.
        env.root_states[_valid, 0] = _second_back[_valid] + _clearance
        env.root_states[_valid, 1] = env.env_origins[_valid, 1]
        env.root_states[_valid, 7:13] = 0.0
        env.gym.set_actor_root_state_tensor(
            env.sim, gymtorch.unwrap_tensor(env.root_states))
        env._depth_debug_spawn_xyz = env.root_states[:, :3].clone()
        env._depth_debug_hurdle_x_ranges = _ranges.clone()
        print(f"[play] --spawn_after_second_hurdle: spawned {int(_valid.sum())}/"
              f"{env.num_envs} envs {_clearance:.2f} m after hurdle 2")
        for _eid in torch.nonzero(_valid, as_tuple=False).flatten().tolist():
            print(f"[play] env {_eid}: spawn_x={env.root_states[_eid, 0].item():.3f} "
                  f"hurdle1={_ranges[_eid, 0].tolist()} "
                  f"hurdle2={_ranges[_eid, 1].tolist()}")

    _depth_stage_dir = None
    if getattr(args, "save_depth_stages", False) and args.use_camera:
        _depth_stage_dir = os.path.join(resolved_log_pth, "depth_stages")
        env._depth_debug_env_id = int(env.lookat_id)
        if args.depth_stage_fps <= 0.0:
            raise ValueError("--depth_stage_fps must be positive")
        if args.depth_stage_start_seconds < 0.0:
            raise ValueError("--depth_stage_start_seconds must be non-negative")
        if args.depth_stage_seconds is not None and args.depth_stage_seconds <= 0.0:
            raise ValueError("--depth_stage_seconds must be positive")
        print(f"[depth] saving raw/noisy/filled/final stages at {args.depth_stage_fps:g} Hz "
              f"from {args.depth_stage_start_seconds:g} s to "
              f"{('end' if args.depth_stage_seconds is None else str(args.depth_stage_start_seconds + args.depth_stage_seconds) + ' s')} "
              f"in {_depth_stage_dir}")
    if args.use_camera:
        if getattr(args, "raw_depth", False):
            print("[depth] RAW mode ON: crop/resize/normalize only; all depth "
                  "noise, stereo/self geometry, min-range remap, and hole fill OFF")
        else:
            _hf = getattr(env.cfg.depth, "hole_fill", False)
            print(f"[depth] deployment hole fill {'ON' if _hf else 'OFF'}"
                  + (f" (mode={getattr(env.cfg.depth, 'hole_fill_mode', 1)}, "
                     f"iters={getattr(env.cfg.depth, 'hole_fill_iters', 8)})"
                     if _hf else ""))

    # Play evaluates a mature policy: skip the training-only 6k-step delay + 24k-step
    # clearance ramp. Local to this reward, not the broader evaluation_mode flag.
    if 'hurdle_clearance' in env.reward_scales:
        env.cfg.rewards.hurdle_clearance_ramp_delay_steps = 0
        env.cfg.rewards.hurdle_clearance_ramp_steps = 0
        env.hurdle_clearance_trigger_step = env.common_step_counter
        env.hurdle_clearance_armed = True
        env.hurdle_clearance_alpha = 1.0
        print('[play] hurdle-clearance penalty ON from step 0 '
              f'(scale={env.reward_scales["hurdle_clearance"] / env.dt:g})')

    # depth_noise_scale comes from _init_buffers, so pin it AFTER make_env.
    if _dns is not None:
        if hasattr(env, "depth_noise_scale"):
            env.depth_noise_scale[:] = _dns
            # Make the pin STICKY: reset_idx otherwise redraws it every episode.
            env.depth_noise_scale_pinned = True
            _kind = ("clean (raw ideal ray-cast)" if _dns == 0 else
                     "configured cliffside-distill rates" if _dns == 0.5 else
                     "FULL (2x configured rates)" if _dns == 1.0 else "custom")
            print(f"[depth] noise scale pinned to s={_dns:g} on all {env.num_envs} envs — {_kind}")
            # Redraw the s-dependent per-EPISODE values at the pinned s.
            _phys = min(max(2.0 * _dns, 0.0), 1.0)
            if hasattr(env, "depth_self_geom_on"):
                env.depth_self_geom_on[:] = _phys > 0.0
            if hasattr(env, "depth_latency") and env.depth_latency_max > 0:
                _lo, _hi = (int(min(env.cfg.depth.latency_range)),
                            int(max(env.cfg.depth.latency_range)))
                import torch as _t
                env.depth_latency[:] = (
                    _t.randint(_lo, _hi + 1, (env.num_envs,), device=env.device
                               ).float() * _phys).round().long()
            # Report what is ACTUALLY applied at this s.
            _d = env.cfg.depth
            _r = 2.0 * _dns
            def _g(n, dflt=0.0):
                return float(getattr(_d, n, dflt))
            print(f"[depth] applied at s={_dns:g}: rate x{_r:.2f} (stochastic), "
                  f"phys x{_phys:.2f} (geometric)")
            print(f"[depth]   dis_noise {_g('dis_noise'):.4f} -> {_g('dis_noise')*_r:.4f} m | "
                  f"gaussian {_g('noise_gaussian'):.4f} -> {_g('noise_gaussian')*_r:.4f} | "
                  f"dropout seed {_g('noise_dropout'):.5f} -> {_g('noise_dropout')*_r:.5f}")
            print(f"[depth]   pw_edge {_g('pw_edge_prob'):.3f} -> {min(_g('pw_edge_prob')*_r, 1.0):.3f} | "
                  f"pw_hole {_g('pw_hole_prob'):.4f} -> {min(_g('pw_hole_prob')*_r, 1.0):.4f} | "
                  f"pw_blur sigma {_g('pw_blur_sigma'):.2f} x{_phys:.2f} | "
                  f"blindspot {int(_g('pw_blindspot_cols'))} px x{_phys:.2f}")
            print(f"[depth]   stereo shadow kept {100.0*_phys:.0f}% | "
                  f"own legs {'ON' if (_phys > 0 and getattr(_d, 'self_geometry', False)) else 'OFF'} | "
                  f"min range {_g('empty_below_m')*_phys:.3f} m | "
                  f"latency x{_phys:.2f} of {list(_d.latency_range)} capture frames")
        elif _dns != 1.0:
            raise SystemExit(f"[play] this env has no per-env noise curriculum "
                             f"(noise_scale_per_env off) so it is fixed at s=1; "
                             f"--depth_noise_scale {_dns} cannot be honoured")
        else:
            print("[depth] noise already at s=1 (no per-env curriculum on this env)")

    if args.cmd_vel_x is not None:
        import torch as _torch
        v = float(args.cmd_vel_x)
        env.command_ranges["lin_vel_x"] = [v, v]
        if "lin_vel_y" in env.command_ranges:
            env.command_ranges["lin_vel_y"] = [0.0, 0.0]
        env.command_ranges["ang_vel_yaw"] = [0.0, 0.0]
        if "heading" in env.command_ranges:
            env.command_ranges["heading"] = [0.0, 0.0]
        env.cfg.commands.curriculum = False
        env._resample_commands(_torch.arange(env.num_envs, device=env.device))
        print(f"[play] --cmd_vel_x: locked lin_vel_x at {v} m/s (yaw/heading zeroed)")

    _vel_profile_segments = []  # list of (vel_m_s, n_steps)
    if args.vel_profile is not None:
        if "lin_vel_y" in env.command_ranges:
            env.command_ranges["lin_vel_y"] = [0.0, 0.0]
        env.command_ranges["ang_vel_yaw"] = [0.0, 0.0]
        if "heading" in env.command_ranges:
            env.command_ranges["heading"] = [0.0, 0.0]
        env.cfg.commands.curriculum = False
        steps_per_sec = 1.0 / env.dt
        for token in args.vel_profile.replace(" ", "").split(","):
            vel_s, dur_s = token.split(":")
            n_steps = max(1, int(round(float(dur_s) * steps_per_sec)))
            _vel_profile_segments.append((float(vel_s), n_steps))
        total_profile_steps = sum(s for _, s in _vel_profile_segments)
        print(f"[play] --vel_profile: {_vel_profile_segments} (total {total_profile_steps} steps / {total_profile_steps * env.dt:.1f}s)")
        env.commands[:, 0] = _vel_profile_segments[0][0]
    _vp_seg_idx = 0
    _vp_seg_step = 0

    obs = env.get_observations()

    if _need_viewer:
        web_viewer.setup(env)
        web_viewer.show_plot = args.plot
        web_viewer.show_data = args.data
        web_viewer.show_rew = args.rew
        web_viewer.run_name = args.exptid or ''
        if not getattr(args, 'top_view', False):
            web_viewer.cam_pos_rel = np.array([0, 2.4, 0.0])

    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = resolved_log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)
    
    
    
    
    if args.use_jit:
        path = os.path.join(log_pth, "traced")
        model, checkpoint = get_load_path(root=path, checkpoint=args.checkpoint)
        path = os.path.join(path, model)
        print("Loading jit for policy: ", path)
        policy_jit = torch.jit.load(path, map_location=env.device)
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)
    estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    if ppo_runner.if_depth:
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    infos = {}
    infos["depth"] = env.get_depth_obs().to(ppo_runner.device) if ppo_runner.if_depth else None
    depth_latent = None   # safe default before first depth update
    _pred_yaw = None      # safe default before first depth update (EP_USE_YAW)

    def _distillation_student_obs(obs_tensor):
        """Mirror the distillation trainer's student observation construction."""
        teacher_obs = obs_tensor
        student_obs = obs_tensor.clone()
        priv_start = ppo_runner.alg.num_prop + ppo_runner.alg.num_scan

        if getattr(ppo_runner.alg, "train_with_estimated_states", False):
            with torch.no_grad():
                priv_states_estimated = estimator(obs_tensor[:, :ppo_runner.alg.num_prop])
            student_obs[:, priv_start:priv_start + ppo_runner.alg.priv_states_dim] = priv_states_estimated

        if getattr(ppo_runner, "zero_student_velocity_obs", False):
            velocity_dim = min(3, ppo_runner.alg.priv_states_dim)
            student_obs[:, priv_start:priv_start + velocity_dim] = 0.0

        return teacher_obs, student_obs

    # Reset the lookat env if it hasn't moved in STUCK_WINDOW steps (not a termination).
    STUCK_RESET_ENABLED = False
    STUCK_WINDOW = 300       # ~6s at 50Hz
    STUCK_THRESHOLD = 0.15   # m total displacement over window
    pos_history = deque(maxlen=STUCK_WINDOW)

    video_writer = None
    video_cam = None
    VIDEO_W, VIDEO_H = 1920, 1080
    VIDEO_CAM_W, VIDEO_CAM_H = 1920, 1080
    # Render every VIDEO_STEP_DECIM-th step; writer fps is divided to match.
    VIDEO_STEP_DECIM = 2
    if args.video:
        video_fps = int(round(1.0 / env.dt / VIDEO_STEP_DECIM))
        if args.video_path:
            video_path = args.video_path
        else:
            if args.checkpoint != -1:
                _ckpt = args.checkpoint
            else:
                try:
                    _, _ckpt = get_load_path(root=log_pth, checkpoint=-1)
                except Exception:
                    _ckpt = "unknown"
            video_path = os.path.join(log_pth, f"{args.exptid}_ckpt{_ckpt}.mp4")
        if imageio is None:
            print(f"[video] WARNING: imageio not installed — no frames will be saved")
            video_writer = None
        else:
            try:
                video_writer = imageio.get_writer(
                    video_path, fps=video_fps,
                    codec='libx264',
                    quality=None,
                    output_params=['-crf', '23', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p'],
                )
            except Exception as e:
                print(f"[video] WARNING: imageio writer failed ({e}) — no frames will be saved")
                video_writer = None
        if video_writer is not None:
            if not _need_viewer:
                cam_props = gymapi.CameraProperties()
                cam_props.width = VIDEO_CAM_W
                cam_props.height = VIDEO_CAM_H
                # No supersampling: 2x2 SSAA quadruples shaded pixels for AA x264 eats.
                cam_props.supersampling_horizontal = 1
                cam_props.supersampling_vertical = 1
                video_cam = env.gym.create_camera_sensor(env.envs[env.lookat_id], cam_props)
            print(f"[video] Recording to {video_path} at {video_fps} fps  ({VIDEO_W}×{VIDEO_H}, "
                  f"every {VIDEO_STEP_DECIM} sim steps)")

    num_steps = args.max_steps if args.max_steps is not None else 10 * int(env.max_episode_length)
    # Progress line: a silent terminal is indistinguishable from a hang while rendering.
    _step_log_every = 1 if video_writer is not None else 100
    _t_loop_start = time()
    _depth_stage_next_t = args.depth_stage_start_seconds
    _depth_stage_interval = 1.0 / args.depth_stage_fps
    _depth_stage_idx = 0
    _last_depth_frame = None   # carry-over for JIT between update_interval steps
    _dstat = {'n': 0, 'nd': 0, 'invalid': 0.0, 'jump': 0.0, 'jitter': 0.0, 'prev': None}
    # Last-episode termination reason per env (lookat_id can switch mid-run).
    _last_term_text = ["none yet"] * env.num_envs

    # Headless failure-mode analysis: levels are PINNED (evaluation_mode gates the
    # curriculum updates) and row i has difficulty i/(num_rows-1) — do NOT use
    # --level/--max_level here, those rebuild every row at U(0.7,1.0).
    _eval_stats = None
    if not _need_viewer:
        # Round-robin envs over ALL rows for equal sample counts, then FREEZE with
        # max_promoted_level=-1 + no_demotion (buffers still refresh on reset).
        env.cfg.terrain.no_demotion = True
        env.cfg.terrain.max_promoted_level = -1
        env.cfg.commands.curriculum = False
        _all_ids = torch.arange(env.num_envs, device=env.device)
        if args.level is not None:
            _lo = _hi = max(0, min(int(args.level), env.max_terrain_level - 1))
        else:
            _lo = max(0, min(int(getattr(args, 'eval_min_level', 0) or 0),
                             env.max_terrain_level - 1))
            _hi = getattr(args, 'eval_max_level', None)
            _hi = env.max_terrain_level - 1 if _hi is None else max(
                _lo, min(int(_hi), env.max_terrain_level - 1))
        env.terrain_levels[:] = _lo + (_all_ids % (_hi - _lo + 1))
        env._update_terrain_curriculum(_all_ids)
        env.reset_idx(_all_ids)
        _eval_stats = {
            'ep': defaultdict(int), 'ep_len': defaultdict(float),
            'term': defaultdict(lambda: defaultdict(int)),
            'goal': defaultdict(float), 'stage': defaultdict(float),
            'xwall': defaultdict(float), 'lvl_of_code': defaultdict(float),
            'rew': defaultdict(lambda: defaultdict(float)),
            'steps': defaultdict(int), 'rew_names': set(),
        }
        _lvl_all = env.terrain_levels.clone()
        _wall_sum, _wall_n = defaultdict(float), defaultdict(int)
        for _l, _a in zip(_lvl_all.tolist(),
                          env.env_flyball_wall_angle_deg.tolist()
                          if hasattr(env, 'env_flyball_wall_angle_deg')
                          else [float('nan')] * env.num_envs):
            _wall_sum[_l] += _a
            _wall_n[_l] += 1
        _wall_angles = {_l: _wall_sum[_l] / _wall_n[_l] for _l in _wall_sum}
        print(f'[eval] headless analysis mode: {env.num_envs} envs pinned across '
              f'levels {sorted(set(_lvl_all.tolist()))}')
        # Zone classifier: WHICH OBSTACLE the robot died on. The hurdle sits ~1.7 m in
        # front of the wall, so bucket by geometry and split outbound/return by stage.
        _ZONE_NAMES = ['spawn→hurdle', 'HURDLE out', 'hurdle→wall',
                       'WALL face', 'LANDING', 'HURDLE back', 'return', 'other']

        def _zone_of(eid, x, stage):
            _hx = env.env_flyball_hurdle_x_ranges[eid]           # (slots, 2)
            _wall = float(env.env_flyball_pit_x1[eid].item())     # wall front face
            _m = 0.15
            _in_hurdle = False
            for _s in range(_hx.shape[0]):
                _x0, _x1 = float(_hx[_s, 0].item()), float(_hx[_s, 1].item())
                if _x0 < 0:      # -1 sentinel = empty slot
                    continue
                if (_x0 - _m) <= x <= (_x1 + _m):
                    _in_hurdle = True
                    _hx0, _hx1 = _x0, _x1
            _outbound = stage <= 1
            if _in_hurdle:
                return 1 if _outbound else 5
            if x >= _wall - 0.30:
                return 3
            _land = (float(env.env_flyball_landing_foot_target[eid, :, 0].mean().item())
                     if hasattr(env, 'env_flyball_landing_foot_target') else _wall - 0.75)
            _first_h = min((float(_hx[_s, 0].item()) for _s in range(_hx.shape[0])
                            if float(_hx[_s, 0].item()) >= 0), default=-1.0)
            if _outbound:
                return 0 if (_first_h >= 0 and x < _first_h) else 2
            if x >= _land - 0.45:
                return 4
            return 6

        _eval_stats['zone'] = defaultdict(lambda: defaultdict(int))
        _eval_stats['zone_names'] = _ZONE_NAMES
        # Circuit state one step back: reset_idx wipes goal/stage before a done.
        _prev_goal = env.cur_goal_idx.clone()
        _prev_stage = (env.wococo_contact_stage.clone()
                       if hasattr(env, 'wococo_contact_stage')
                       else torch.zeros_like(_prev_goal))
        _prev_xwall = torch.zeros(env.num_envs, device=env.device)
        _prev_x = env.root_states[:, 0].clone()
        _prev_eplen = env.episode_length_buf.clone()

    # Opt-in phase force capture, via env vars to keep it out of the shared parser.
    _action_noise = float(getattr(args, 'action_noise', 0.0) or 0.0)
    if _action_noise > 0:
        print(f"[eval] --action_noise {_action_noise}: Gaussian noise on the executed action "
              f"(pre-scaling, matching learn_vision's DAgger injection)")

    _ff_target = int(os.environ.get('EP_FOOT_FORCE_EPISODES', '0'))
    _ff_rows = []
    _ff_episodes = []
    _ff_stop = False
    if _ff_target > 0:
        if _need_viewer:
            raise SystemExit('[foot-force] EP_FOOT_FORCE_EPISODES requires --no_video and no --web')
        _ff_output_dir = os.path.abspath(os.environ.get(
            'EP_FOOT_FORCE_OUT', os.path.join(os.getcwd(), 'foot_force_logs')))
        _ff_episode_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        _ff_next_episode_id = env.num_envs
        _ff_last_phase = [''] * env.num_envs
        _ff_phase_steps = [0] * env.num_envs
        _body_names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
        _ff_foot_names = [_body_names[int(idx)] for idx in env.feet_indices.tolist()]
        print(f'[foot-force] capturing {_ff_target} completed episodes at {1.0 / env.dt:.1f} Hz; '
              f'feet={_ff_foot_names}; mode=full episode; '
              f'output={_ff_output_dir}')

        def _ff_zone_of(eid, x):
            if hasattr(env, 'env_flyball_hurdle_x_ranges'):
                ranges = env.env_flyball_hurdle_x_ranges[eid]
                for slot in range(ranges.shape[0]):
                    x0 = float(ranges[slot, 0].item())
                    x1 = float(ranges[slot, 1].item())
                    if x0 >= 0.0 and (x0 - 0.15) <= x <= (x1 + 0.15):
                        return 'hurdle'
            if hasattr(env, 'env_flyball_pit_x1'):
                wall = float(env.env_flyball_pit_x1[eid].item())
                if x >= wall - 0.30:
                    return 'wall'
            return 'other'

    # Evaluation only. Without this the depth student OOMs: RecurrentDepthBackbone
    # carries its GRU state across steps, so with grad enabled the autograd graph
    # chains the whole rollout. Invisible at 600 steps, fatal by ~3k.
    torch.set_grad_enabled(False)
    for i in range(num_steps):
        if i % _step_log_every == 0 or i == num_steps - 1:
            _el = time() - _t_loop_start
            _rate = i / _el if (i and _el > 0) else 0.0
            _eta = (num_steps - i) / _rate if _rate > 0 else float('nan')
            print(f"[play] step {i + 1}/{num_steps}  "
                  f"{_rate:.1f} steps/s  elapsed {_el:.0f}s  eta {_eta:.0f}s",
                  flush=True)
        if args.use_jit:
            if ppo_runner.if_depth:
                if infos["depth"] is not None:
                    _last_depth_frame = infos["depth"]
                _depth_in = _last_depth_frame if _last_depth_frame is not None else torch.zeros(
                    (env_cfg.env.num_envs, env_cfg.depth.resized[0], env_cfg.depth.resized[1]),
                    device=env.device)
                actions, depth_latent = policy_jit(obs.detach(), infos["depth"] is not None, _depth_in, depth_latent)
            else:
                obs_jit = torch.cat((obs.detach()[:, :env_cfg.env.n_proprio+env_cfg.env.n_priv], obs.detach()[:, -env_cfg.env.history_len*env_cfg.env.n_proprio:]), dim=1)
                actions = policy(obs_jit)
        else:
            obs_student = obs.detach()
            if ppo_runner.if_depth:
                teacher_obs, obs_student = _distillation_student_obs(obs.detach())
                if infos["depth"] is not None:
                    obs_prop_depth = obs_student[:, :env.cfg.env.n_proprio].clone()
                    obs_prop_depth[:, 6:8] = 0
                    depth_out = depth_encoder(infos["depth"], obs_prop_depth)
                    if getattr(ppo_runner, "use_yaw", False):
                        # Yaw predictor on: scale the 2-D heading by ×1.5 as in training.
                        depth_latent = depth_out[:, :-2]
                        _pred_yaw = 1.5 * depth_out[:, -2:]
                    else:
                        depth_latent = depth_out
                        _pred_yaw = None
                if getattr(ppo_runner, "use_yaw", False) and _pred_yaw is not None:
                    obs_student[:, 6:8] = _pred_yaw
                else:
                    obs_student[:, 6:8] = 0

            else:
                depth_latent = None
            
            if hasattr(ppo_runner.alg, "depth_actor"):
                actions = ppo_runner.alg.depth_actor(obs_student, hist_encoding=True, scandots_latent=depth_latent)
            else:
                actions = policy(obs_student, hist_encoding=True, scandots_latent=depth_latent)
                if args.dump_actor:
                    if not hasattr(env, "_play_dump_idx"):
                        env._play_dump_idx = 0
                    if env._play_dump_idx < 5:
                        if env._play_dump_idx == 0:
                            torch.save(ppo_runner.alg.actor_critic.state_dict(), "/tmp/play_actor_state.pt")
                            actions_via_actor_call = ppo_runner.alg.actor_critic.actor(obs_student, hist_encoding=True, eval=False, scandots_latent=depth_latent)
                            print(f"[play_dump] direct actor call action_norm={float(actions_via_actor_call.norm()):.3f}")
                        np.savez(f"/tmp/play_step_{env._play_dump_idx}.npz",
                                 obs_into_policy=obs_student[0].cpu().numpy(),
                                 actions_out=actions.detach()[0].cpu().numpy())
                        print(f"[play_dump] step {env._play_dump_idx}: action_norm={float(actions.norm()):.3f}")
                        env._play_dump_idx += 1
            
        if _vel_profile_segments:
            cur_vel, cur_dur = _vel_profile_segments[_vp_seg_idx]
            env.commands[:, 0] = cur_vel
            _vp_seg_step += 1
            if _vp_seg_step >= cur_dur:
                _vp_seg_step = 0
                if _vp_seg_idx + 1 < len(_vel_profile_segments):
                    _vp_seg_idx += 1
                    next_vel = _vel_profile_segments[_vp_seg_idx][0]
                    print(f"[vel_profile] step={i}  → vel={next_vel} m/s (seg {_vp_seg_idx})")

        # --action_noise: Gaussian noise on the EXECUTED action, as hardened DAgger
        # does. The hardened prior scores WORSE on clean dynamics (0.626 vs 0.655), so
        # the ranking should invert as this noise is raised if hardening works.
        if _action_noise > 0:
            actions = actions + _action_noise * torch.randn_like(actions)

        if _ff_target > 0:
            _ff_stage_pre = env.wococo_contact_stage.clone()
            _ff_armed_pre = env.wococo_armed.clone()
            _ff_episode_ids_pre = _ff_episode_ids.clone()
            _ff_episode_steps_pre = env.episode_length_buf.clone()
            _ff_root_x_pre = env.root_states[:, 0].clone()

        obs, _, rews, dones, infos = env.step(actions.detach())

        _stage_t = i * env.dt
        _stage_in_window = (
            _stage_t >= args.depth_stage_start_seconds and
            (args.depth_stage_seconds is None or
             _stage_t < args.depth_stage_start_seconds + args.depth_stage_seconds))
        if _depth_stage_dir is not None and _stage_in_window and _stage_t >= _depth_stage_next_t:
            if _save_depth_stage_set(env, _depth_stage_dir, _depth_stage_idx):
                if hasattr(env, "_depth_debug_hurdle_x_ranges"):
                    _eid = int(getattr(env, "_depth_debug_env_id", env.lookat_id))
                    _cx = env.root_states[_eid, 0].item()
                    _h2b = env._depth_debug_hurdle_x_ranges[_eid, 1, 1].item()
                    _wx = env.env_flyball_forbidden_x1[_eid].item()
                    print(f"[depth] frame {_depth_stage_idx}: root_x={_cx:.3f}, "
                          f"hurdle2_back={_h2b:.3f}, wall_front={_wx:.3f}")
                _depth_stage_idx += 1
                _depth_stage_next_t += _depth_stage_interval

        if _ff_target > 0:
            _forces = env.contact_forces[:, env.feet_indices, :].detach()
            _positions = env.rigid_body_states[:, env.feet_indices, :3].detach()
            for _eid in range(env.num_envs):
                _stage = int(_ff_stage_pre[_eid].item())
                _armed = bool(_ff_armed_pre[_eid].item())
                _phase = ('wall_front' if _armed and _stage == 0 else
                          'wall_rear' if _armed and _stage == 1 else
                          'landing' if _armed and _stage == 2 else '')
                _zone = _ff_zone_of(_eid, float(_ff_root_x_pre[_eid].item()))
                if _phase and _ff_last_phase[_eid] != _phase:
                    _ff_last_phase[_eid] = _phase
                    _ff_phase_steps[_eid] = 0
                _phase_step = _ff_phase_steps[_eid] if _phase else -1
                for _foot_i, _foot_name in enumerate(_ff_foot_names):
                    _f = _forces[_eid, _foot_i].cpu().tolist()
                    _p = _positions[_eid, _foot_i].cpu().tolist()
                    _norm = float(np.linalg.norm(_f))
                    _ff_rows.append({
                        'episode': int(_ff_episode_ids_pre[_eid].item()),
                        'env': _eid,
                        'episode_step': int(_ff_episode_steps_pre[_eid].item()),
                        'phase': _phase,
                        'phase_step': _phase_step,
                        'phase_time_s': _phase_step * env.dt,
                        'zone': _zone,
                        'foot': _foot_name,
                        'fx_n': _f[0], 'fy_n': _f[1], 'fz_n': _f[2],
                        'force_n': _norm, 'contact': int(_norm > 1.0),
                        'foot_x_m': _p[0], 'foot_y_m': _p[1], 'foot_z_m': _p[2],
                    })
                if _phase:
                    _ff_phase_steps[_eid] += 1

            for _eid in dones.nonzero(as_tuple=False).flatten().tolist():
                if len(_ff_episodes) < _ff_target:
                    _term_code = int(env.last_term_code[_eid].item())
                    _length_steps = int(_ff_episode_steps_pre[_eid].item()) + 1
                    _ff_episodes.append({
                        'episode': int(_ff_episode_ids_pre[_eid].item()),
                        'env': _eid,
                        'length_steps': _length_steps,
                        'length_s': _length_steps * env.dt,
                        'term_code': _term_code,
                        'termination': TERM_CODE_NAMES.get(_term_code, 'unknown'),
                    })
                _ff_episode_ids[_eid] = _ff_next_episode_id
                _ff_next_episode_id += 1
                _ff_last_phase[_eid] = ''
                _ff_phase_steps[_eid] = 0
            if len(_ff_episodes) >= _ff_target:
                _ff_stop = True

        # Exact parabolic clearance volumes (what the reward tests); --overlays.
        if args.overlays:
            _draw_hurdle_clearance_parabolas(env)

        # Realized degradation on the frames the ENCODER consumed, in the three metrics
        # the model was calibrated on (1.30% invalid, 0.558% big-jump, 0.0137 m jitter).
        if _dns is not None and infos.get("depth", None) is not None:
            _far, _near = env.cfg.depth.far_clip, env.cfg.depth.near_clip
            _dm = (infos["depth"][env.lookat_id].float() + 0.5) * (_far - _near) + _near
            _dstat['n'] += 1
            _dstat['invalid'] += float((_dm > _far - 0.05).float().mean())
            if _dstat['prev'] is not None and _dstat['prev'].shape == _dm.shape:
                _delta = (_dm - _dstat['prev']).abs()
                _smooth = _delta[_delta <= 0.25]
                _dstat['jump'] += float((_delta > 0.25).float().mean())
                _dstat['jitter'] += float(_smooth.mean()) if _smooth.numel() else 0.0
                _dstat['nd'] += 1
            _dstat['prev'] = _dm.clone()
            if _dstat['n'] % 100 == 0:
                _nd = max(_dstat['nd'], 1)
                print(f"[depth] realized over {_dstat['n']} consumed frames "
                      f"(env {env.lookat_id}, s={_dns:g}): "
                      f"invalid {100*_dstat['invalid']/_dstat['n']:.2f}% | "
                      f"big-jump {100*_dstat['jump']/_nd:.3f}% | "
                      f"smooth jitter {_dstat['jitter']/_nd:.4f} m")

        # Record why each ended episode terminated; show the tracked env's reason.
        if hasattr(env, 'last_term_code'):
            for eid in dones.nonzero(as_tuple=False).flatten().tolist():
                code = int(env.last_term_code[eid].item())
                _last_term_text[eid] = (f"{TERM_CODE_NAMES.get(code, 'unknown')} "
                                        f"(code {code}, step {i})")
        if _need_viewer:
            web_viewer.last_term_reason = _last_term_text[env.lookat_id]

        if _eval_stats is not None:
            # Reward streams: every env, bucketed by its pinned level.
            if getattr(env, 'last_step_rews', None):
                for _k, _v in env.last_step_rews.items():
                    _eval_stats['rew_names'].add(_k)
                    _sums = torch.bincount(_lvl_all, weights=_v.detach().float(),
                                           minlength=env.max_terrain_level)
                    for _l, _s in enumerate(_sums.tolist()):
                        if _s != 0.0:
                            _eval_stats['rew'][_l][_k] += _s
            for _l in _lvl_all.tolist():
                _eval_stats['steps'][_l] += 1
            # Terminations: attribute to the pinned level and the prior circuit position.
            _done_ids = dones.nonzero(as_tuple=False).flatten().tolist()
            for _eid in _done_ids:
                _l = int(_lvl_all[_eid].item())
                _c = int(env.last_term_code[_eid].item())
                _eval_stats['ep'][_l] += 1
                _eval_stats['ep_len'][_l] += float(_prev_eplen[_eid].item())
                _eval_stats['term'][_l][_c] += 1
                _eval_stats['goal'][_c] += float(_prev_goal[_eid].item())
                _eval_stats['stage'][_c] += float(_prev_stage[_eid].item())
                _eval_stats['xwall'][_c] += float(_prev_xwall[_eid].item())
                _eval_stats['lvl_of_code'][_c] += _l
                _eval_stats['zone'][_c][_zone_of(_eid, float(_prev_x[_eid].item()),
                                                 float(_prev_stage[_eid].item()))] += 1
            _prev_goal = env.cur_goal_idx.clone()
            if hasattr(env, 'wococo_contact_stage'):
                _prev_stage = env.wococo_contact_stage.clone()
            if hasattr(env, 'env_flyball_pit_x1'):
                _prev_xwall = env.root_states[:, 0] - env.env_flyball_pit_x1
            _prev_x = env.root_states[:, 0].clone()
            _prev_eplen = env.episode_length_buf.clone()
            if (i + 1) % 1000 == 0:
                print(f'[eval] --- interim summary at step {i + 1}/{num_steps} ---')
                _print_eval_summary(_eval_stats, _wall_angles)

        if args.rew and getattr(env, 'last_step_rews', None):
            id = env.lookat_id
            names = list(env.last_step_rews.keys())
            vals = torch.stack([env.last_step_rews[k][id] for k in names] + [rews[id]]).tolist()
            total = vals[-1]
            entries = [(k, v) for k, v in zip(names, vals[:-1]) if v != 0.0]
            if entries:
                entries.sort(key=lambda x: abs(x[1]), reverse=True)
                parts = '  '.join(f'{k}: {v:+.4f}' for k, v in entries)
                print(f'[rew] step={i:5d}  total={total:+.4f}  |  {parts}')

        if _ff_stop:
            print(f'[foot-force] reached {_ff_target} completed episodes at step {i + 1}')
            break

        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)
            if args.speed is not None and args.speed > 0:
                sleep(env.dt * (1.0 / args.speed - 1.0))
        elif _need_viewer and i % VIDEO_STEP_DECIM == 0:
            # Headless render, only on written steps — the 1080p readback dominates.
            web_viewer._notified = False
            web_viewer.render(fetch_results=True,
                              step_graphics=True,
                              render_all_camera_sensors=True,
                              wait_for_page_load=False)

        id = env.lookat_id

        if video_writer is not None and i % VIDEO_STEP_DECIM == 0:
            if _need_viewer and web_viewer._image is not None:
                plot_img = web_viewer._image_plot if (args.plot and not args.rew) else None
                rew_img  = web_viewer._image_rew  if args.rew else None
                depth_img = web_viewer._image_depth if env.cfg.depth.use_camera else None
                # Start from the raw camera image: play.py never renders scandots.
                cam_rgb = web_viewer._image
                if args.overlays:
                    cam_rgb = _overlay_clearance_parabolas(
                        cam_rgb, env, env.gym, env.sim,
                        web_viewer._envs[web_viewer._camera_id],
                        web_viewer._cameras[web_viewer._camera_id])
                    cam_rgb = _overlay_foothold_targets(
                        cam_rgb, env, env.gym, env.sim,
                        web_viewer._envs[web_viewer._camera_id],
                        web_viewer._cameras[web_viewer._camera_id])
                frame = _compose_video_frame(cam_rgb, plot_img, rew_img, VIDEO_W, VIDEO_H,
                                             depth_gray=depth_img)
            else:
                root_pos = env.root_states[id, :3].cpu().numpy()
                if args.top_view:
                    cam_pos = root_pos + np.array([0, 0.01, 4.5])
                    cam_target = root_pos.copy()
                else:
                    cam_pos = root_pos + np.array([0, 2.4, 0.0])
                    cam_target = root_pos
                env.gym.set_camera_location(video_cam, env.envs[id],
                                            gymapi.Vec3(*cam_pos), gymapi.Vec3(*cam_target))
                env.gym.fetch_results(env.sim, True)
                env.gym.step_graphics(env.sim)
                env.gym.render_all_camera_sensors(env.sim)
                raw = env.gym.get_camera_image(env.sim, env.envs[id], video_cam, gymapi.IMAGE_COLOR)
                frame = raw.reshape(VIDEO_CAM_H, VIDEO_CAM_W, 4)[..., :3]  # RGBA→RGB
                if args.overlays:
                    frame = _overlay_clearance_parabolas(
                        frame, env, env.gym, env.sim, env.envs[id], video_cam)
                if frame.shape[1] != VIDEO_W or frame.shape[0] != VIDEO_H:
                    frame = cv2.resize(frame, (VIDEO_W, VIDEO_H), interpolation=cv2.INTER_LANCZOS4)
            try:
                video_writer.append_data(frame)
            except Exception:
                pass

        # OFF in headless analysis: a forced reset would hide wall-stalls counted as timeouts.
        if not STUCK_RESET_ENABLED or _eval_stats is not None:
            continue
        cur_pos = env.root_states[id, :2].cpu().numpy().copy()
        pos_history.append(cur_pos)
        if len(pos_history) == STUCK_WINDOW:
            total_disp = float(np.linalg.norm(cur_pos - pos_history[0]))
            if total_disp < STUCK_THRESHOLD:
                print(f"[play] Robot stuck (disp={total_disp:.3f}m over {STUCK_WINDOW} steps) — resetting env {id}")
                env.reset_idx(torch.tensor([id], device=env.device))
                pos_history.clear()

    if video_writer is not None:
        video_writer.close()
        print(f"[video] Saved to {video_path}")

    if _eval_stats is not None:
        print(f'[eval] === FINAL summary after {i + 1} steps ===')
        _print_eval_summary(_eval_stats, _wall_angles)

    if _ff_target > 0:
        _write_foot_force_outputs(
            _ff_rows, _ff_episodes, _ff_output_dir, args.exptid,
            int(args.level) if args.level is not None else 'sweep', env.dt)


if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()

    # play.py convenience defaults so the common eval command is just
    #   python play.py --exptid xxx
    # Anything explicit wins; get_args() is shared with train.py, hence not there.
    _argv = sys.argv[1:]
    def _given(flag):
        return any(a == flag or a.startswith(flag + '=') for a in _argv)
    if not _given('--task'):
        args.task = 'go2_teacher'
    if not _given('--num_envs') and not args.save:
        # --num_envs overrides the count above; left None under --save to keep 64.
        args.num_envs = 1
    if not _given('--video'):
        args.video = True
    if args.no_video:
        # Explicit opt-out beats the default: no mp4, and without --web no graphics.
        args.video = False
    if not _given('--data'):
        args.data = True
    if not _given('--max_steps'):
        args.max_steps = 600

    play(args)
