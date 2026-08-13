"""Browser-based viewer for headless runs (``play.py --web``).

Adapted from skrl (https://github.com/Toni-SM/skrl), MIT licensed, (c) Toni-SM.
"""
from typing import List, Optional

import io
import logging
import math
import threading
from collections import deque

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

try:
    import flask
except ImportError:
    flask = None

try:
    import imageio
    import isaacgym
    import isaacgym.torch_utils as torch_utils
    from isaacgym import gymapi
except ImportError:
    imageio = None
    isaacgym = None
    torch_utils = None
    gymapi = None


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r) if r != 0 else 0
    phi = np.arctan2(y, x)
    return r, theta, phi

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

class WebViewer:
    def __init__(self, host: str = "127.0.0.1", port: int = 5000,
                 start_server: bool = True, top_view: bool = False) -> None:
        """
        Web viewer for Isaac Gym

        :param host: Host address (default: "127.0.0.1")
        :type host: str
        :param port: Port number (default: 5000)
        :type port: int
        :param start_server: Start the Flask HTTP server (default: True).
            Set False for headless use (e.g. video-only without a browser).
        :type start_server: bool
        :param top_view: Position camera directly above robot looking down.
        :type top_view: bool
        """
        self.top_view = top_view
        self._app = flask.Flask(__name__)
        self._app.add_url_rule("/", view_func=self._route_index)
        self._app.add_url_rule("/_route_stream", view_func=self._route_stream)
        self._app.add_url_rule("/_route_stream_depth", view_func=self._route_stream_depth)
        self._app.add_url_rule("/_route_stream_plot", view_func=self._route_stream_plot)
        self._app.add_url_rule("/_route_stream_rew", view_func=self._route_stream_rew)
        self._app.add_url_rule("/_route_input_event", view_func=self._route_input_event, methods=["POST"])

        self._log = logging.getLogger('werkzeug')
        self._log.disabled = True
        self._app.logger.disabled = True

        self._image = None
        self._image_depth = None
        self._image_plot = None
        self._image_rew = None
        # render() calls between matplotlib panel rebuilds. --rew rebuilds every
        # reward-producing step so short events survive; this slows playback.
        self._fxfz_plot_decimation = 5
        self._rew_plot_decimation = 1
        self._camera_id = 0
        self._camera_type = gymapi.IMAGE_COLOR
        self._notified = False
        self._wait_for_page = True
        self._pause_stream = False
        self._event_load = threading.Event()
        self._event_stream = threading.Event()
        self._event_stream_depth = threading.Event()
        self._event_stream_plot = threading.Event()
        self._event_stream_rew = threading.Event()

        if start_server:
            self._thread = threading.Thread(target=lambda: \
                self._app.run(host=host, port=port, debug=False, use_reloader=False), daemon=True)
            self._thread.start()
            print(f"\nStarting web viewer on http://{host}:{port}/\n")

    def _route_index(self) -> 'flask.Response':
        """Render the web page

        :return: Flask response
        :rtype: flask.Response
        """
        with open(os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "utils", "webviewer.html"), 'r', encoding='utf-8') as file:
            template = file.read()
        self._event_load.set()
        return flask.render_template_string(template)

    def _route_stream(self) -> 'flask.Response':
        """Stream the image to the web page

        :return: Flask response
        :rtype: flask.Response
        """
        return flask.Response(self._stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def _route_stream_depth(self) -> 'flask.Response':
        return flask.Response(self._stream_depth(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def _route_input_event(self) -> 'flask.Response':

        data = flask.request.get_json()
        key, mouse = data.get("key", None), data.get("mouse", None)
        dx, dy, dz = data.get("dx", None), data.get("dy", None), data.get("dz", None)

        transform = self._gym.get_camera_transform(self._sim,
                                                   self._envs[self._camera_id],
                                                   self._cameras[self._camera_id])

        if mouse == "wheel":
            r, theta, phi = cartesian_to_spherical(*self.cam_pos_rel)
            r += 0.05 * dz
            self.cam_pos_rel = spherical_to_cartesian(r, theta, phi)

        elif mouse == "left":
            dx *= 0.2 * math.pi / 180
            dy *= 0.2 * math.pi / 180

            r, theta, phi = cartesian_to_spherical(*self.cam_pos_rel)
            theta -= dy
            phi -= dx
            self.cam_pos_rel = spherical_to_cartesian(r, theta, phi)

        elif mouse == "right":
            dx *= -0.2 * math.pi / 180
            dy *= -0.2 * math.pi / 180

            r, theta, phi = cartesian_to_spherical(*self.cam_pos_rel)
            theta += dy
            phi += dx
            self.cam_pos_rel = spherical_to_cartesian(r, theta, phi)

        elif key == 219:  # prev
            self._camera_id = (self._camera_id-1) % self._env.num_envs
            self._foot_force_history = [deque(maxlen=200) for _ in range(4)]
            self._yaw_history.clear()
            self._roll_history.clear()
            self._base_height_history.clear()
            self._pitch_history.clear()
            self._land_yaw_err_history.clear()
            self._return_vel_history.clear()
            self._land_hold_history.clear()
            self._plot_step = 0; self._stage_advance_markers = []; self._prev_wococo_stage = -1
            self._wococo_stage_history.clear()
            self._rew_history = {k: deque(maxlen=200) for k in self._rew_history}
            print(f"Switching to previous environment {self._camera_id}")
            return flask.Response(status=200)

        elif key == 221:  # next
            self._camera_id = (self._camera_id+1) % self._env.num_envs
            self._foot_force_history = [deque(maxlen=200) for _ in range(4)]
            self._yaw_history.clear()
            self._roll_history.clear()
            self._base_height_history.clear()
            self._pitch_history.clear()
            self._land_yaw_err_history.clear()
            self._return_vel_history.clear()
            self._land_hold_history.clear()
            self._plot_step = 0; self._stage_advance_markers = []; self._prev_wococo_stage = -1
            self._wococo_stage_history.clear()
            self._rew_history = {k: deque(maxlen=200) for k in self._rew_history}
            print(f"Switching to next environment {self._camera_id}")
            return flask.Response(status=200)
        
        # pause stream (V: 86)
        elif key == 86:
            print("Toggling stream pause...")
            self._pause_stream = not self._pause_stream
            return flask.Response(status=200)

        # change image type (T: 84)
        elif key == 84:
            print("Switching camera type...")
            if self._camera_type == gymapi.IMAGE_COLOR:
                self._camera_type = gymapi.IMAGE_DEPTH
            elif self._camera_type == gymapi.IMAGE_DEPTH:
                self._camera_type = gymapi.IMAGE_COLOR
            return flask.Response(status=200)
        
        

        else:
            return flask.Response(status=200)

        return flask.Response(status=200)

    def _stream(self) -> bytes:
        """Format the image to be streamed

        :return: Image encoded as Content-Type
        :rtype: bytes
        """
        while True:
            self._event_stream.wait()

            image = imageio.imwrite("<bytes>", self._image, format="JPEG", quality=90)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

            self._event_stream.clear()
            self._notified = False

    def _stream_depth(self) -> bytes:
        while self._env.cfg.depth.use_camera:
            self._event_stream_depth.wait()

            image = imageio.imwrite("<bytes>", self._image_depth, format="JPEG")

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

            self._event_stream_depth.clear()

    def _route_stream_plot(self) -> 'flask.Response':
        return flask.Response(self._stream_plot(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def _stream_plot(self) -> bytes:
        while True:
            self._event_stream_plot.wait()
            if self._image_plot is not None:
                image = imageio.imwrite("<bytes>", self._image_plot, format="JPEG", quality=85)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            self._event_stream_plot.clear()

    def _route_stream_rew(self) -> 'flask.Response':
        return flask.Response(self._stream_rew(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def _stream_rew(self) -> bytes:
        while True:
            self._event_stream_rew.wait()
            if self._image_rew is not None:
                image = imageio.imwrite("<bytes>", self._image_rew, format="JPEG", quality=85)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            self._event_stream_rew.clear()

    def attach_view_camera(self, i, env_handle, actor_handle, root_pos):
        if True:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 1920
            camera_props.height = 1080
            # 1x, not 2x: 2x renders 3840x2160 (4x shaded pixels/frame), which on the
            # 1 cm cliffside trimesh dominates wall-clock and dies in the x264 encode.
            camera_props.supersampling_horizontal = 1
            camera_props.supersampling_vertical = 1

            camera_handle = self._gym.create_camera_sensor(env_handle, camera_props)
            self._cameras.append(camera_handle)
            
            cam_pos = root_pos + np.array([0, 1, 0.5])
            self._gym.set_camera_location(camera_handle, env_handle, gymapi.Vec3(*cam_pos), gymapi.Vec3(*root_pos))

    def _world_to_pixel(self, world_pos, view, proj, w, h):
        """Project a 3D world position to 2D pixel coordinates.
        Returns (px, py) or None if outside frustum / behind camera.
        Tries both matrix conventions (raw and transposed) and returns
        whichever produces an in-frustum result first.
        """
        p = np.array([world_pos[0], world_pos[1], world_pos[2], 1.0])
        for V in (view, view.T):
            for P in (proj, proj.T):
                clip = P @ (V @ p)
                if clip[3] <= 0:
                    continue
                nx, ny = clip[0] / clip[3], clip[1] / clip[3]
                if -1.0 <= nx <= 1.0 and -1.0 <= ny <= 1.0:
                    px = int((nx + 1.0) * 0.5 * w)
                    py = int((1.0 - (ny + 1.0) * 0.5) * h)
                    return px, py
        return None

    def _overlay_goals(self, image: np.ndarray) -> np.ndarray:
        """Project 3D goal positions into the camera image and draw coloured circles.

        Current goal: yellow  |  Future goals: cyan  |  Reached: green
        """
        env = self._env
        id = self._camera_id
        if not hasattr(env, 'terrain_goals') or not hasattr(env, 'height_samples'):
            return image

        try:
            view = np.array(self._gym.get_camera_view_matrix(
                self._sim, self._envs[id], self._cameras[id]))
            proj = np.array(self._gym.get_camera_proj_matrix(
                self._sim, self._envs[id], self._cameras[id]))
        except Exception:
            return image

        h, w = image.shape[:2]
        img = image.copy()

        goals = env.terrain_goals[
            env.terrain_levels[id], env.terrain_types[id]].cpu().numpy()
        cur_idx = int(env.cur_goal_idx[id].cpu().item())
        reached = bool(env.reached_goal_ids[id].cpu().item()) if hasattr(env, 'reached_goal_ids') else False

        n_drawn = 0
        _debug_once = not hasattr(self, '_goal_debug_printed')
        if _debug_once:
            self._goal_debug_printed = True
            print(f"[overlay_goals] {len(goals)} goals, cur={cur_idx}, "
                  f"view shape={view.shape}, proj shape={proj.shape}")
            robot_pos = env.root_states[id, :3].cpu().numpy()
            print(f"[overlay_goals] robot world pos = {robot_pos}")
            print(f"[overlay_goals] cam_pos_rel = {self.cam_pos_rel}")

        for i, goal in enumerate(goals):
            goal_xy = goal[:2] + env.terrain.cfg.border_size
            pts = (goal_xy / env.terrain.cfg.horizontal_scale).astype(int)
            pts[0] = int(np.clip(pts[0], 0, env.height_samples.shape[0] - 1))
            pts[1] = int(np.clip(pts[1], 0, env.height_samples.shape[1] - 1))
            goal_z = (env.height_samples[pts[0], pts[1]].cpu().item()
                      * env.terrain.cfg.vertical_scale)

            result = self._world_to_pixel([goal[0], goal[1], goal_z], view, proj, w, h)
            if _debug_once:
                print(f"[overlay_goals] goal[{i}] world=({goal[0]:.2f},{goal[1]:.2f},{goal_z:.2f}) → {result}")
            if result is None:
                continue
            px, py = result
            n_drawn += 1

            if i == cur_idx and reached:
                color, radius, thick = (0, 255, 80), 16, 3   # green: reached
            elif i == cur_idx:
                color, radius, thick = (255, 255, 0), 14, 3  # yellow: current
            else:
                color, radius, thick = (0, 200, 255), 9, 2   # cyan: future

            cv2.circle(img, (px, py), radius, color, thick)
            cv2.putText(img, str(i), (px + 6, py - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        if _debug_once:
            print(f"[overlay_goals] drew {n_drawn}/{len(goals)} goals")
        return img

    def _collect_step_data(self) -> None:
        """Collect per-step data into shared buffers.

        Called unconditionally every render step so both --plot and --rew share
        the same history regardless of which flags are active.
        """
        env = self._env
        id = self._camera_id

        forces = env.contact_forces[id, env.feet_indices, :]
        magnitudes = torch.norm(forces, dim=-1).cpu().numpy()
        for foot_idx, mag in enumerate(magnitudes):
            self._foot_force_history[foot_idx].append(float(mag))

        if hasattr(env, 'yaw'):
            self._yaw_history.append(float(math.degrees(env.yaw[id].item())))
        if hasattr(env, 'roll'):
            self._roll_history.append(float(math.degrees(env.roll[id].item())))
        if hasattr(env, 'pitch'):
            self._pitch_history.append(float(math.degrees(env.pitch[id].item())))

        # Posture-only landing gate quantities, read from the env's OWN gate helpers
        # so a panel can never disagree with the FSM: base height against the tile
        # floor datum, yaw error off the RETURN bearing (not the absolute |yaw| ~ 180 deg).
        if self._landing_posture_only():
            self._base_height_history.append(
                float(env._wococo_landing_base_height()[id].item()))
            self._land_yaw_err_history.append(
                float(math.degrees(env._wococo_landing_return_yaw_err()[id].item())))
            self._return_vel_history.append(
                float(env._wococo_landing_return_vel()[id].item()))
        else:
            self._base_height_history.append(float(env.root_states[id, 2].item()))
        if hasattr(env, 'wococo_stage_hold_counter'):
            self._land_hold_history.append(
                int(env.wococo_stage_hold_counter[id].item()))

        if hasattr(env, 'wococo_contact_stage'):
            cur_stage = int(env.wococo_contact_stage[id].item())
            if self._prev_wococo_stage >= 0:
                if cur_stage > self._prev_wococo_stage:
                    self._stage_advance_markers.append((self._plot_step, cur_stage))
                elif cur_stage < self._prev_wococo_stage:
                    self._stage_advance_markers.clear()
                    self._wococo_stage_history.clear()
            self._prev_wococo_stage = cur_stage
            self._wococo_stage_history.append(cur_stage)

        if self.show_rew:
            self._collect_reward_data()

        self._plot_step += 1

    def _collect_reward_data(self) -> None:
        """Record exact per-term and total rewards for the current env step."""
        env = self._env
        if not getattr(env, 'last_step_rews', None):
            return

        id = self._camera_id
        names = list(env.last_step_rews.keys())
        # One device sync for all terms. rew_buf is the final reward after clipping,
        # deferred terms and termination; summing last_step_rews would not match it.
        values = torch.stack(
            [env.last_step_rews[name][id] for name in names] + [env.rew_buf[id]]
        ).detach().cpu().tolist()
        for name, value in zip(names, values[:-1]):
            if name not in self._rew_history:
                self._rew_history[name] = deque(maxlen=200)
            self._rew_history[name].append(float(value))
        self._total_rew_history.append(float(values[-1]))

    def _update_fxfz_plot(self) -> None:
        """Render per-foot contact force plot. Data collection is done by _collect_step_data."""
        id = self._camera_id

        foot_labels = ['FL', 'FR', 'RL', 'RR']
        foot_colors = ['#ef9a9a', '#90caf9', '#a5d6a7', '#ffe082']
        stage_colors = {1: '#ff9800', 2: '#ff5722', 3: '#ce93d8'}

        fig, axes = plt.subplots(2, 2, figsize=(12, 6), dpi=112)
        fig.patch.set_facecolor('#1e1e1e')
        axes_flat = axes.flatten()

        for foot_idx in range(4):
            ax = axes_flat[foot_idx]
            ax.set_facecolor('#1e1e1e')
            data = np.array(self._foot_force_history[foot_idx])
            if len(data) > 0:
                t = np.arange(len(data))
                ax.plot(t, data, color=foot_colors[foot_idx], linewidth=1.2)
                max_val = float(data.max())
            else:
                max_val = 0.0

            window_start = self._plot_step - len(self._foot_force_history[foot_idx])
            for marker_step, new_stage in self._stage_advance_markers:
                x = marker_step - window_start
                if 0 <= x < len(self._foot_force_history[foot_idx]):
                    c = stage_colors.get(new_stage, '#ef5350')
                    ax.axvline(x, color=c, linewidth=1.2, alpha=0.85)

            ax.axhline(5.0, color='white', linestyle='--', linewidth=0.9, alpha=0.6)
            ax.set_ylim(0, max(50.0, max_val * 1.15))
            ax.set_title(f'{foot_labels[foot_idx]} ‖F‖ [N]', color='white', fontsize=8, pad=2)
            ax.set_xlabel('steps', color='white', fontsize=7)
            ax.tick_params(colors='white', labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor('#555')

        fig.suptitle(f'foot contact forces — env {id}', color='white', fontsize=9)
        fig.tight_layout(pad=0.5)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        img_bgr = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
        self._image_plot = img_bgr[..., ::-1].copy()

    def _landing_posture_only(self) -> bool:
        """True when the env runs the posture-only P4L landing, so the landing
        panels should plot its base-state gate instead of the foothold one."""
        env = self._env
        try:
            return bool(env._wococo_landing_posture_only())
        except AttributeError:
            return False

    def _landing_gate_cfg(self):
        """(roll_tol, pitch_tol, yaw_tol, h_lo, h_hi, speed_thresh, n_persist)
        for the posture-only landing, in degrees / metres / m·s⁻¹ / steps."""
        r = self._env.cfg.rewards
        cmd_hi = 1.0
        try:
            cmd_hi = float(self._env.commands[self._camera_id, 0].item())
        except Exception:
            pass
        n_persist = getattr(r, 'wococo_landing_n_persist', None)
        n_persist = int(n_persist if n_persist is not None
                        else getattr(r, 'wococo_n_persist', 5))
        return (float(getattr(r, 'wococo_landing_roll_tol_deg', 25.0)),
                float(getattr(r, 'wococo_landing_pitch_tol_deg', 35.0)),
                float(getattr(r, 'wococo_landing_yaw_tol_deg', 70.0)),
                float(getattr(r, 'wococo_landing_h_lo', 0.15)),
                float(getattr(r, 'wococo_landing_h_hi', 0.35)),
                float(getattr(r, 'wococo_landing_speed_frac', 0.8)) * cmd_hi,
                n_persist)

    @staticmethod
    def _rew_panel_style(last: float, data: np.ndarray):
        """Return (bg_hex, line_color) graded by |last| relative to history range."""
        max_abs = float(np.max(np.abs(data))) if len(data) else 0.0
        max_abs = max(max_abs, 0.005)
        t = float(np.clip(last / max_abs, -1.0, 1.0))

        nr, ng, nb = 30, 30, 30
        if t > 0:
            r = int(nr + t * (13 - nr))
            g = int(ng + t * (51 - ng))
            b = int(nb + t * (32 - nb))
            line = '#00e676' if t > 0.5 else '#a5d6a7'
        elif t < 0:
            ta = -t
            r = int(nr + ta * (62 - nr))
            g = int(ng + ta * (10 - ng))
            b = int(nb + ta * (10 - nb))
            line = '#ff1744' if ta > 0.5 else '#ef9a9a'
        else:
            r, g, b = nr, ng, nb
            line = '#90caf9'

        return f'#{r:02x}{g:02x}{b:02x}', line

    def _update_rew_plot(self) -> None:
        """Render a unified grid: force panels (0-3) + total rew + yaw + roll + base-z + reward terms."""
        env = self._env
        id = self._camera_id

        if not hasattr(env, 'last_step_rews'):
            return

        rew_names = list(self._rew_history.keys())
        # Posture-only adds pitch + return-speed panels, pushing terms to cell 10.
        posture_only = self._landing_posture_only()
        n_gate_extra = 2 if posture_only else 0
        rew_cell0 = 8 + n_gate_extra
        n_panels = rew_cell0 + len(rew_names)
        ncols = 4
        nrows = math.ceil(n_panels / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 2.5), dpi=96)
        fig.patch.set_facecolor('#1e1e1e')
        axes_flat = np.array(axes).flatten()

        foot_labels = ['FL', 'FR', 'RL', 'RR']
        foot_colors = ['#ef9a9a', '#90caf9', '#a5d6a7', '#ffe082']
        stage_colors = {1: '#ff9800', 2: '#ff5722', 3: '#ce93d8'}
        for foot_idx in range(4):
            ax = axes_flat[foot_idx]
            ax.set_facecolor('#1e1e1e')
            data = np.array(self._foot_force_history[foot_idx])
            max_val = 0.0
            if len(data):
                ax.plot(np.arange(len(data)), data, color=foot_colors[foot_idx],
                        linewidth=0.8, label=foot_labels[foot_idx])
                max_val = float(data.max())
            window_start = self._plot_step - len(self._foot_force_history[foot_idx])
            for marker_step, new_stage in self._stage_advance_markers:
                x = marker_step - window_start
                if 0 <= x < len(self._foot_force_history[foot_idx]):
                    c = stage_colors.get(new_stage, '#ef5350')
                    ax.axvline(x, color=c, linewidth=0.9, alpha=0.85)
            ax.axhline(5.0, color='white', linestyle='--', linewidth=0.7, alpha=0.6)
            ax.set_ylim(0, max(50.0, max_val * 1.15))
            ax.set_title(f'{foot_labels[foot_idx]} ‖F‖ [N]', color='white', fontsize=6, pad=2)
            ax.tick_params(colors='white', labelsize=5, length=2, pad=1)
            for spine in ax.spines.values():
                spine.set_edgecolor('#444')

        ax = axes_flat[4]
        tr_data = np.array(self._total_rew_history)
        tr_last = float(tr_data[-1]) if len(tr_data) else 0.0
        tr_bg, tr_line = self._rew_panel_style(tr_last, tr_data)
        ax.set_facecolor(tr_bg)
        ax.plot(np.arange(len(tr_data)), tr_data, color=tr_line, linewidth=0.8)
        ax.axhline(0, color='#777', linewidth=0.5)
        ax.set_title('total reward / step', color='white', fontsize=6, pad=2)
        ax.tick_params(colors='white', labelsize=5, length=2, pad=1)
        for spine in ax.spines.values():
            spine.set_edgecolor(tr_bg)
        ax.annotate(f'{tr_last:+.4f}', xy=(1, 1), xycoords='axes fraction',
                    ha='right', va='top', color='white', fontsize=6,
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='#0008', alpha=0.8))

        if posture_only:
            self._draw_posture_landing_panels(axes_flat, stage_colors)
        else:
            self._draw_legacy_landing_panels(axes_flat, stage_colors)

        self._draw_reward_term_panels(axes_flat, rew_cell0, rew_names, env, id)

        for j in range(n_panels, len(axes_flat)):
            axes_flat[j].set_visible(False)

        title = f'rewards — env {id}'
        if posture_only and len(self._land_hold_history):
            *_, n_persist = self._landing_gate_cfg()
            stage = int(env.wococo_contact_stage[id].item()) if hasattr(env, 'wococo_contact_stage') else -1
            if stage == 2:
                title += (f'   |   LANDING hold {self._land_hold_history[-1]}/{n_persist}')
        fig.suptitle(title, color='white', fontsize=8, y=1.0)
        fig.tight_layout(pad=0.4, rect=[0, 0, 1, 0.98])

        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        img_bgr = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
        self._image_rew = img_bgr[..., ::-1].copy()

    def _band_panel(self, ax, data, lo, hi, color, title_fmt, stage_colors,
                    ylim=None, yticks=None, unit='', fmt='{:+.1f}'):
        """Shared band-gate panel: series + green accept band + ✓/✗ title.

        `lo`/`hi` may be None for a one-sided gate (>= lo, or <= hi)."""
        ax.set_facecolor('#1e1e1e')
        arr = np.array(data)
        cur = float(arr[-1]) if len(arr) else 0.0
        ok = ((lo is None or cur >= lo) and (hi is None or cur <= hi))
        if len(arr):
            ax.plot(np.arange(len(arr)), arr, color=color, linewidth=0.8)
        _lo = lo if lo is not None else (ax.get_ylim()[0] if ylim is None else ylim[0])
        _hi = hi if hi is not None else (ax.get_ylim()[1] if ylim is None else ylim[1])
        ax.axhspan(_lo, _hi, color='#66bb6a', alpha=0.15)
        for edge in (lo, hi):
            if edge is not None:
                ax.axhline(edge, color='#66bb6a', linewidth=0.7, linestyle='--', alpha=0.7)
        ax.axhline(0, color='#555', linewidth=0.5)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if yticks is not None:
            ax.set_yticks(yticks)
        window_start = self._plot_step - len(arr)
        for marker_step, new_stage in self._stage_advance_markers:
            x = marker_step - window_start
            if 0 <= x < len(arr):
                ax.axvline(x, color=stage_colors.get(new_stage, '#ef5350'),
                           linewidth=0.9, alpha=0.85)
        ax.set_title(title_fmt.format(mark='✓' if ok else '✗', cur=fmt.format(cur)),
                     color=color, fontsize=6, pad=2)
        ax.tick_params(colors='white', labelsize=5, length=2, pad=1)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2e7d32' if ok else '#444')
        return ok

    def _draw_posture_landing_panels(self, axes_flat, stage_colors) -> None:
        """Cells 5-9 for the posture-only P4L landing (stage 2 -> 3 sentinel).

        One panel per gate conjunct, all five plotted against the SAME accept
        band the FSM tests, so the reason a landing is not advancing is readable
        straight off the dashboard: whichever panel shows ✗ is the blocker.
        """
        roll_tol, pitch_tol, yaw_tol, h_lo, h_hi, spd, n_persist = self._landing_gate_cfg()

        # Cell 5 plots yaw error off the RETURN bearing, NOT absolute |yaw|.
        self._band_panel(
            axes_flat[5], self._land_yaw_err_history, None, yaw_tol, '#80deea',
            f'yaw err vs return [°] s2→s3  {{mark}} {{cur}}° (≤{yaw_tol:.0f})',
            stage_colors, ylim=(0, 185),
            yticks=sorted({0, int(round(yaw_tol)), 90, 180}), fmt='{:.1f}')

        self._band_panel(
            axes_flat[6], self._roll_history, -roll_tol, roll_tol, '#a5d6a7',
            f'roll [°] s2→s3  {{mark}} {{cur}}° (|·|≤{roll_tol:.0f})',
            stage_colors, ylim=(-185, 185),
            yticks=sorted({-180, -90, -int(round(roll_tol)), 0,
                           int(round(roll_tol)), 90, 180}))

        # h_hi bounds ballistic dwell so the landing cannot complete mid-air:
        # plot base height as a closed band, not a floor.
        self._band_panel(
            axes_flat[7], self._base_height_history, h_lo, h_hi, '#ce93d8',
            f'base z [m] s2→s3  {{mark}} {{cur}} m  ([{h_lo:.2f}, {h_hi:.2f}])',
            stage_colors, fmt='{:.3f}')

        self._band_panel(
            axes_flat[8], self._pitch_history, -pitch_tol, pitch_tol, '#ffe082',
            f'pitch [°] s2→s3  {{mark}} {{cur}}° (|·|≤{pitch_tol:.0f})',
            stage_colors, ylim=(-185, 185),
            yticks=sorted({-180, -90, -int(round(pitch_tol)), 0,
                           int(round(pitch_tol)), 90, 180}))

        # Cell 9: velocity projected on the return bearing. Signed -- running
        # back at the wall reads negative and can never satisfy the gate.
        self._band_panel(
            axes_flat[9], self._return_vel_history, spd, None, '#ef9a9a',
            f'return vel [m/s] s2→s3  {{mark}} {{cur}} (≥{spd:.2f})',
            stage_colors, fmt='{:+.2f}')

    def _draw_legacy_landing_panels(self, axes_flat, stage_colors) -> None:
        """Cells 5-7 for the foothold landing (wococo_landing_posture_only off)."""
        env = self._env
        # Cell 5: yaw plot — s3→s4 LANDING tolerance: |yaw| > 180° − yaw_tol_deg.
        yaw_tol_deg = float(getattr(self._env.cfg.rewards, 'wococo_landing_yaw_tol_deg', 45.0))
        yaw_thresh  = 180.0 - yaw_tol_deg
        ax = axes_flat[5]
        ax.set_facecolor('#1e1e1e')
        yaw_data = np.array(self._yaw_history)
        if len(yaw_data):
            t = np.arange(len(yaw_data))
            ax.plot(t, yaw_data, color='#80deea', linewidth=0.8)
            ax.plot(t, np.abs(yaw_data), color='#80deea', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axhspan(yaw_thresh, 180, color='#66bb6a', alpha=0.15)
        ax.axhspan(-180, -yaw_thresh, color='#66bb6a', alpha=0.15)
        ax.axhline(yaw_thresh,  color='#66bb6a', linewidth=0.7, linestyle='--', alpha=0.7)
        ax.axhline(-yaw_thresh, color='#66bb6a', linewidth=0.7, linestyle='--', alpha=0.7)
        ax.axhline(0,  color='#555', linewidth=0.5)
        ax.set_ylim(-185, 185)
        ax.set_yticks(sorted({-180, -int(round(yaw_thresh)), 0, int(round(yaw_thresh)), 180}))
        window_start = self._plot_step - len(self._yaw_history)
        for marker_step, new_stage in self._stage_advance_markers:
            x = marker_step - window_start
            if 0 <= x < len(self._yaw_history):
                c = stage_colors.get(new_stage, '#ef5350')
                ax.axvline(x, color=c, linewidth=0.9, alpha=0.85)
        cur_yaw = float(yaw_data[-1]) if len(yaw_data) else 0.0
        in_zone = abs(cur_yaw) >= yaw_thresh
        ax.set_title(f'yaw [°] s3→s4  {"✓" if in_zone else "✗"} {cur_yaw:+.1f}° (|·|≥{yaw_thresh:.0f})',
                     color='#80deea', fontsize=6, pad=2)
        ax.tick_params(colors='white', labelsize=5, length=2, pad=1)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

        # Cell 6: roll plot — s3→s4 LANDING tolerance: |roll| < roll_tol_deg.
        roll_tol_deg = float(getattr(self._env.cfg.rewards, 'wococo_landing_roll_tol_deg', 10.0))
        ax = axes_flat[6]
        ax.set_facecolor('#1e1e1e')
        roll_data = np.array(self._roll_history)
        cur_roll = float(roll_data[-1]) if len(roll_data) else 0.0
        roll_lo, roll_hi = -roll_tol_deg, roll_tol_deg
        in_roll_zone = roll_lo <= cur_roll <= roll_hi
        if len(roll_data):
            t = np.arange(len(roll_data))
            ax.plot(t, roll_data, color='#a5d6a7', linewidth=0.8)
        ax.axhspan(roll_lo, roll_hi, color='#66bb6a', alpha=0.15)
        ax.axhline(roll_lo, color='#66bb6a', linewidth=0.7, linestyle='--', alpha=0.7)
        ax.axhline(roll_hi, color='#66bb6a', linewidth=0.7, linestyle='--', alpha=0.7)
        ax.axhline(0, color='#555', linewidth=0.5)
        ax.set_ylim(-185, 185)
        ax.set_yticks(sorted({-180, -90, int(round(roll_lo)), 0, int(round(roll_hi)), 90, 180}))
        window_start = self._plot_step - len(self._roll_history)
        for marker_step, new_stage in self._stage_advance_markers:
            x = marker_step - window_start
            if 0 <= x < len(self._roll_history):
                c = stage_colors.get(new_stage, '#ef5350')
                ax.axvline(x, color=c, linewidth=0.9, alpha=0.85)
        ax.set_title(f'roll [°] s3→s4  {"✓" if in_roll_zone else "✗"} {cur_roll:+.1f}° (|·|≤{roll_tol_deg:.0f})',
                     color='#a5d6a7', fontsize=6, pad=2)
        ax.tick_params(colors='white', labelsize=5, length=2, pad=1)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

        ax = axes_flat[7]
        ax.set_facecolor('#1e1e1e')
        bh_data = np.array(self._base_height_history)
        cur_bh = float(bh_data[-1]) if len(bh_data) else 0.0
        above = cur_bh > 0.35
        if len(bh_data):
            t = np.arange(len(bh_data))
            ax.plot(t, bh_data, color='#ce93d8', linewidth=0.8)
        ax.axhline(0.35, color='#ff9800', linewidth=0.9, linestyle='--', alpha=0.9, label='stage-0 gate')
        ax.fill_between(np.arange(len(bh_data)), 0.35, bh_data,
                        where=(bh_data > 0.35), color='#ce93d8', alpha=0.18)
        window_start = self._plot_step - len(self._base_height_history)
        for marker_step, new_stage in self._stage_advance_markers:
            x = marker_step - window_start
            if 0 <= x < len(self._base_height_history):
                c = stage_colors.get(new_stage, '#ef5350')
                ax.axvline(x, color=c, linewidth=0.9, alpha=0.85)
        ax.set_title(f'base z [m]  {"✓" if above else "✗"} {cur_bh:.3f} m', color='#ce93d8', fontsize=6, pad=2)
        ax.tick_params(colors='white', labelsize=5, length=2, pad=1)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    def _draw_reward_term_panels(self, axes_flat, rew_cell0, rew_names, env, id) -> None:
        """Individual reward-term panels, starting at cell `rew_cell0`."""
        STAGE_NAMES = {0: 'liftoff', 1: 'front_wall', 2: 'rear_wall', 3: 'landing'}
        for i, name in enumerate(rew_names):
            ax = axes_flat[rew_cell0 + i]
            data = np.array(self._rew_history[name])
            last = float(data[-1]) if len(data) else 0.0
            bg_color, line_color = self._rew_panel_style(last, data)
            ax.set_facecolor(bg_color)
            ax.plot(np.arange(len(data)), data, color=line_color, linewidth=0.8)
            ax.axhline(0, color='#777', linewidth=0.5)
            raw_scale = env.reward_scales.get(name, 0.0) / env.dt
            title_str = f"{name.replace('_', ' ')} (×{raw_scale:.2g})"

            if name == 'wococo_contact_pts' and len(self._wococo_stage_history):
                cur_stage = int(env.wococo_contact_stage[id].item()) if hasattr(env, 'wococo_contact_stage') else '?'
                title_str += f'  [stage {cur_stage}: {STAGE_NAMES.get(cur_stage, "?")}]'
                stage_data = np.array(self._wococo_stage_history)
                ax2 = ax.twinx()
                ax2.step(np.arange(len(stage_data)), stage_data, color='#80cbc4',
                         linewidth=1.0, alpha=0.75, where='post')
                ax2.set_ylim(-0.3, 2.8)
                ax2.set_yticks([0, 1, 2])
                ax2.set_yticklabels(['0\nlift', '1\nwall', '2\nland'], fontsize=4)
                ax2.tick_params(colors='#80cbc4', labelsize=4, length=2, pad=1)
                for spine in ax2.spines.values():
                    spine.set_edgecolor(bg_color)

            ax.set_title(title_str, color='white', fontsize=6, pad=2)
            ax.tick_params(colors='white', labelsize=5, length=2, pad=1)
            for spine in ax.spines.values():
                spine.set_edgecolor(bg_color)
            ax.annotate(f'{last:+.3f}', xy=(1, 1), xycoords='axes fraction',
                        ha='right', va='top', color='white', fontsize=6,
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='#0008', alpha=0.8))

    def _overlay_contacts(self, image: np.ndarray) -> np.ndarray:
        """Draw the WoCoCo *reward* zones for the wall-turn terrains
        (env_class == 19 flyball, env_class == 15 two_hurdles_wall_turn).

        Instead of the coarse contact AABB boxes, this renders the regions that
        actually pay wococo reward at the current stage: the per-foot foothold
        targets (env_flyball_foot_target) of the feet that are *active* this
        stage, each as a translucent disc at the dense foot-track kappa radius
        (the per-step _reward_wococo_foot_track zone) with a tight inner ring at
        the one-shot precision kappa (_reward_wococo_foot_target). Active feet
        are read from env._wococo_active_feet_mask(), so the design-aware
        mapping is correct — P4 is stage0=front / stage1=rear (the old
        hard-coded P3 indices highlighted the wrong feet on the cliffside runs).
        """
        env = self._env
        id = self._camera_id

        if not hasattr(env, 'env_class'):
            return image
        env_cls = int(env.env_class[id].item())
        if env_cls not in (19, 15):
            return image
        if not hasattr(env, 'env_flyball_foot_target'):
            return image

        try:
            view = np.array(self._gym.get_camera_view_matrix(
                self._sim, self._envs[id], self._cameras[id]))
            proj = np.array(self._gym.get_camera_proj_matrix(
                self._sim, self._envs[id], self._cameras[id]))
        except Exception:
            return image

        h, w = image.shape[:2]
        img = image.copy()

        rcfg = env.cfg.rewards
        kappa_dense = float(getattr(rcfg, 'wococo_foot_track_kappa', 0.30))
        kappa_os    = float(getattr(rcfg, 'wococo_foot_target_kappa_oneshot',
                            getattr(rcfg, 'wococo_foot_target_kappa', 0.12)))

        FOOT_COLORS_VIVID = [
            ( 66, 165, 245),
            (239,  83,  80),
            (255, 213,  79),
            (102, 187, 106),
        ]
        FOOT_COLORS_DIM = [tuple(max(0, int(c * 0.45)) for c in col)
                           for col in FOOT_COLORS_VIVID]
        FOOT_LABELS = ['FL', 'FR', 'RL', 'RR']

        # Feet paying wococo reward THIS stage (P4 s0=front/s1=rear; P3
        # s1=front/s2=rear/s3=front; P1 s0=all four).
        try:
            active = env._wococo_active_feet_mask()[id].cpu().numpy().astype(bool)
        except Exception:
            active = np.zeros(4, dtype=bool)

        targets  = env.env_flyball_foot_target[id].cpu().numpy()
        feet_pos = env.rigid_body_states[id, env.feet_indices, :3].cpu().numpy()

        # P4L landing (stage 2): mirror _reward_wococo_foot_track — front feet aim at
        # the FLOOR landing foothold with a LARGER acceptance radius.
        dense_r = [kappa_dense] * 4
        _is_p4l_land = (getattr(env, '_wococo_design', '') == 'P4L'
                        and hasattr(env, 'env_flyball_landing_foot_target')
                        and int(env.wococo_contact_stage[id].item()) == 2)
        if _is_p4l_land:
            targets[:2] = env.env_flyball_landing_foot_target[id].cpu().numpy()   # FL, FR
            _lr = float(getattr(rcfg, 'wococo_landing_foot_region_radius', 0.35))
            dense_r[0] = dense_r[1] = _lr

        def _proj_radius(center, r):
            """Approx screen-pixel radius of a world-space sphere of radius r,
            by projecting the centre and three axis-offset points."""
            c2 = self._world_to_pixel(center.tolist(), view, proj, w, h)
            if c2 is None:
                return None, None
            best = 0
            for ax in ([r, 0, 0], [0, r, 0], [0, 0, r]):
                o2 = self._world_to_pixel((center + np.array(ax)).tolist(), view, proj, w, h)
                if o2 is not None:
                    best = max(best, int(round(np.hypot(o2[0] - c2[0], o2[1] - c2[1]))))
            return c2, max(best, 3)

        overlay = img.copy()
        for fi in range(4):
            if not active[fi]:
                continue
            c2, rad = _proj_radius(targets[fi], dense_r[fi])
            if c2 is None:
                continue
            cv2.circle(overlay, c2, rad, FOOT_COLORS_VIVID[fi], -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.30, img, 0.70, 0, img)

        for fi in range(4):
            if not active[fi]:
                continue
            c2, rad = _proj_radius(targets[fi], dense_r[fi])
            if c2 is None:
                continue
            rad_os = None
            if not (_is_p4l_land and fi < 2):
                _, rad_os = _proj_radius(targets[fi], kappa_os)   # no one-shot term on the floor landing
            col = FOOT_COLORS_VIVID[fi]
            cv2.circle(img, c2, rad, col, 1, cv2.LINE_AA)
            if rad_os:
                cv2.circle(img, c2, rad_os, col, 2, cv2.LINE_AA)
            cv2.drawMarker(img, c2, col, cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
            cv2.putText(img, FOOT_LABELS[fi] + '*', (c2[0] + rad + 3, c2[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)

        for fi in range(4):
            fp = feet_pos[fi]
            result = self._world_to_pixel(fp.tolist(), view, proj, w, h)
            if result is None:
                continue
            inside = bool(active[fi] and np.linalg.norm(fp - targets[fi]) <= dense_r[fi])
            dot_color = FOOT_COLORS_VIVID[fi] if inside else FOOT_COLORS_DIM[fi]
            cv2.circle(img, result, 5, dot_color, -1, cv2.LINE_AA)
            cv2.putText(img, FOOT_LABELS[fi], (result[0] + 6, result[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, dot_color, 1, cv2.LINE_AA)
            if active[fi]:
                c2 = self._world_to_pixel(targets[fi].tolist(), view, proj, w, h)
                if c2 is not None:
                    cv2.line(img, result, c2, FOOT_COLORS_DIM[fi], 1, cv2.LINE_AA)

        # Always-on P4L LANDING footholds (magenta, at acceptance radius); skipped in
        # the landing stage, where the active-foot overlay above already draws them.
        if (getattr(env, '_wococo_design', '') == 'P4L'
                and hasattr(env, 'env_flyball_landing_foot_target')
                and not _is_p4l_land):
            land_tgt = env.env_flyball_landing_foot_target[id].cpu().numpy()      # (2,3)
            land_rad = float(getattr(rcfg, 'wococo_landing_foot_region_radius', 0.35))
            LAND_COL = (255, 0, 255)   # magenta
            for li, lbl in enumerate(['L-FL', 'L-FR']):
                c2, rad = _proj_radius(land_tgt[li], land_rad)
                if c2 is None:
                    continue
                cv2.circle(img, c2, rad, LAND_COL, 1, cv2.LINE_AA)
                cv2.drawMarker(img, c2, LAND_COL, cv2.MARKER_TILTED_CROSS, 10, 1, cv2.LINE_AA)
                cv2.putText(img, lbl, (c2[0] + rad + 3, c2[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, LAND_COL, 1, cv2.LINE_AA)

        return img

    def _overlay_data(self, image: np.ndarray) -> np.ndarray:
        """Overlay telemetry text on the image and cache lines for video compose."""
        env = self._env
        id = self._camera_id
        img = image.copy()

        cmd_vel_x = env.commands[id, 0].item()
        level = int(env.terrain_levels[id].item()) if hasattr(env, 'terrain_levels') else -1
        env_cls = int(env.env_class[id].item()) if hasattr(env, 'env_class') else -1
        cur_goal = int(env.cur_goal_idx[id].item()) if hasattr(env, 'cur_goal_idx') else -1
        scales = getattr(env, 'reward_scales', {}) or {}

        lines = []
        if self.run_name:
            lines.append(self.run_name)
        lines.append(f"env_class: {env_cls}   level: {level}   goal: {cur_goal}")
        if hasattr(env, 'env_flyball_wall_angle_deg') and env_cls in (15, 19):
            angle = env.env_flyball_wall_angle_deg[id].item()
            lines.append(f"wall angle: {angle:.1f} deg")

        # Wall WoCoCo — tagged [reward off] when the wococo_stage scale is 0, so the
        # operator knows the counter drives no reward.
        wococo_armed = None
        if hasattr(env, 'wococo_armed'):
            wococo_armed = bool(env.wococo_armed[id].item())
        if hasattr(env, 'wococo_contact_stage'):
            wstage = int(env.wococo_contact_stage[id].item())
            wall_stage_name = {
                0: 'APPROACH',
                1: 'FRONT_WALL',
                2: 'REAR_WALL',
                3: 'LANDING',
                4: 'RETURN',
            }.get(wstage, str(wstage))
            wall_wococo_active = scales.get('wococo_stage', 0.0) != 0.0
            tag = '' if wall_wococo_active else '  [reward off]'
            latch = '' if wococo_armed is None else (
                '  latch: ARMED' if wococo_armed else '  latch: disarmed')
            lines.append(f"wall WoCoCo: stage {wstage} ({wall_stage_name}){latch}{tag}")

        # Hurdle WoCoCo — only when its reward is scaled in. env19: idx 0 = forward,
        # 1 = backward; env15 (two hurdles): idx 0 = h1, 1 = h2.
        hurdle_wococo_active = scales.get('wococo_hurdle_stage', 0.0) != 0.0
        if hurdle_wococo_active and hasattr(env, 'wococo_hurdle_stage'):
            hstage = int(env.wococo_hurdle_stage[id].item())
            hidx   = int(env.wococo_hurdle_idx[id].item())
            stage_name = {0: 'LIFTOFF', 1: 'AIRBORNE', 2: 'LANDED', 3: 'done'}.get(hstage, str(hstage))
            if env_cls == 19:
                slot_label = 'fwd' if hidx == 0 else 'bwd'
            else:
                slot_label = f'h{hidx + 1}'
            lines.append(f"hurdle WoCoCo: slot {slot_label}  stage {hstage} ({stage_name})")
            if hstage == 1 and hasattr(env, 'wococo_hurdle_s1_time'):
                t1 = int(env.wococo_hurdle_s1_time[id].item())
                T1 = float(getattr(env.cfg.rewards, 'wococo_hurdle_air_decay_T', 10.0))
                lines.append(f"  airborne dwell: {t1}/{int(T1)} steps")

        lines.append(f"cmd_vel_x: {cmd_vel_x:.2f} m/s")

        dr = []
        if hasattr(env, 'dr_scale'):
            dr.append(f"dr_scale {env.dr_scale[id].item():.2f}")
        if hasattr(env, 'friction_coeffs_tensor'):
            dr.append(f"friction {env.friction_coeffs_tensor[id].item():.2f}")
        if hasattr(env, 'restitution_coeffs'):
            dr.append(f"restitution {float(env.restitution_coeffs[id].reshape(-1)[0]):.2f}")
        if dr:
            lines.append("DR: " + "  ".join(dr))
        if hasattr(env, 'mass_params_tensor'):
            mp = env.mass_params_tensor[id]
            lines.append(f"  +mass {mp[0].item():+.2f} kg  com ("
                         f"{mp[1].item():+.02f},{mp[2].item():+.02f},{mp[3].item():+.02f}) m")
        if hasattr(env, 'motor_strength'):
            ms = env.motor_strength[:, id, :]
            lines.append(f"  motor_str {ms.mean().item():.2f} "
                         f"[{ms.min().item():.2f}, {ms.max().item():.2f}]")
        # Latencies (action + proprioception), reported in ms.
        dt = float(getattr(env, 'dt', 0.02))
        lat = []
        if getattr(env.cfg.domain_rand, 'randomize_action_latency', False):
            a = int(env.action_delay_steps[id].item())
            lat.append(f"action {a*dt*1000:.0f} ms ({a} st)")
        elif getattr(env.cfg.domain_rand, 'action_delay', False):
            # Fixed delay (deploy latch, 0/1 step sticky, or legacy step schedule):
            # action_delay_steps stays all-zero, so read what the env applies.
            a = int(getattr(env, 'delay', torch.zeros(1)).item())
            latched = getattr(env, 'action_delay_latched', None)
            tag = '' if latched is None else (' latched' if latched else ' pre-latch')
            lat.append(f"action {a*dt*1000:.0f} ms ({a} st{tag})")
        if getattr(env.cfg.domain_rand, 'randomize_proprio_latency', False) and hasattr(env, 'proprio_lag_floor'):
            pl = (env.proprio_lag_floor[id].item() + env.proprio_lag_frac[id].item()) * dt
            lat.append(f"proprio {pl*1000:.0f} ms")
        if env.cfg.depth.use_camera and getattr(env, 'depth_latency_max', 0) > 0:
            dl = int(env.depth_latency[id].item())
            cap = int(getattr(env.cfg.depth, 'capture_interval', env.cfg.depth.update_interval))
            lat.append(f"depth {dl*cap*dt*1000:.0f} ms ({dl} fr)")
        if lat:
            lines.append("  latency: " + "  ".join(lat))
        if hasattr(env, 'last_push_step'):
            pstep = int(env.last_push_step[id].item())
            if pstep < 0:
                lines.append("push: none yet")
            else:
                ago = int(getattr(env, 'common_step_counter', 0)) - pstep
                lmag = float(torch.linalg.norm(env.last_push_lin[id]).item())
                amag = float(torch.linalg.norm(env.last_push_ang[id]).item())
                tag = "  <== PUSH" if ago * dt <= 0.5 else ""
                lines.append(f"push: |v| {lmag:.2f} m/s  |w| {amag:.2f} rad/s  ({ago} st ago){tag}")

        # Why the tracked env's previous episode ended (filled in by play.py).
        if self.last_term_reason:
            lines.append(f"last term: {self.last_term_reason}")

        self._data_lines = lines

        # Without the reward dashboard the camera fills the screen — bump the font.
        big = not self.show_rew
        font_scale = 1.1 if big else 0.55
        thickness = 2 if big else 1
        line_h    = 44 if big else 22
        y0 = 40 if big else 20
        for line in lines:
            cv2.putText(img, line, (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            y0 += line_h

        # WoCoCo-armed badge, drawn bottom-right with a font_scale INDEPENDENT of
        # `big` so the plant latch stays legible in both overlay layouts.
        if wococo_armed is not None:
            badge = 'WoCoCo: ARMED' if wococo_armed else 'WoCoCo: disarmed'
            badge_color = (0, 255, 80) if wococo_armed else (150, 150, 150)
            badge_scale = 1.1
            badge_th = 2
            (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, badge_scale, badge_th)
            bx = max(10, img.shape[1] - bw - 12)
            by = max(bh + 10, img.shape[0] - 16)
            cv2.putText(img, badge, (bx, by), cv2.FONT_HERSHEY_SIMPLEX,
                        badge_scale, badge_color, badge_th, cv2.LINE_AA)
        return img

    def setup(self, env) -> None:
        """Setup the web viewer

        :param gym: The gym
        :type gym: isaacgym.gymapi.Gym
        :param sim: Simulation handle
        :type sim: isaacgym.gymapi.Sim
        :param envs: Environment handles
        :type envs: list of ints
        :param cameras: Camera handles
        :type cameras: list of ints
        """
        self._gym = env.gym
        self._sim = env.sim
        self._envs = env.envs
        self._cameras = []
        self._env = env
        # Small forward tilt (0.01 m) avoids a degenerate look-at when the
        # camera is exactly overhead — Isaac Gym's matrix breaks at (0,0,-1).
        self.cam_pos_rel = np.array([0, 0.01, 4.5]) if self.top_view else np.array([0, 1.4, 0.7])
        self._linvelx_history = deque(maxlen=200)
        self._cmd_velx_history = deque(maxlen=200)
        self._foot_force_history = [deque(maxlen=200) for _ in range(4)]
        self._yaw_history = deque(maxlen=200)
        self._roll_history = deque(maxlen=200)
        self._base_height_history = deque(maxlen=200)
        # Posture-only P4L landing gate (cfg.rewards.wococo_landing_posture_only):
        # none of its five conjuncts is the absolute |yaw|~180 the legacy panel plots.
        self._pitch_history = deque(maxlen=200)
        self._land_yaw_err_history = deque(maxlen=200)
        self._return_vel_history = deque(maxlen=200)
        self._land_hold_history = deque(maxlen=200)
        self._plot_step = 0
        self._stage_advance_markers = []
        self._prev_wococo_stage = -1
        self._wococo_stage_history = deque(maxlen=200)
        self._rew_history = {name: deque(maxlen=200) for name in env.reward_names}
        self._total_rew_history = deque(maxlen=200)
        self.show_plot = False
        self.show_data = False
        self.show_rew = False
        self.run_name = ''
        # Set by play.py after each step: why the tracked env's last episode ended.
        self.last_term_reason = ''
        self._data_lines = []
        for i in range(self._env.num_envs):
            root_pos = self._env.root_states[i, :3].cpu().numpy()
            self.attach_view_camera(i, self._envs[i], self._env.actor_handles[i], root_pos)
    
    def render(self,
               fetch_results: bool = True,
               step_graphics: bool = True,
               render_all_camera_sensors: bool = True,
               wait_for_page_load: bool = True) -> None:
        """Render and get the image from the current camera

        This function must be called after the simulation is stepped (post_physics_step).
        The following Isaac Gym functions are called before get the image.
        Their calling can be skipped by setting the corresponding argument to False

        - fetch_results
        - step_graphics
        - render_all_camera_sensors

        :param fetch_results: Call Gym.fetch_results method (default: True)
        :type fetch_results: bool
        :param step_graphics: Call Gym.step_graphics method (default: True)
        :type step_graphics: bool
        :param render_all_camera_sensors: Call Gym.render_all_camera_sensors method (default: True)
        :type render_all_camera_sensors: bool
        :param wait_for_page_load: Wait for the page to load (default: True)
        :type wait_for_page_load: bool
        """
        if self._wait_for_page:
            if wait_for_page_load:
                if not self._event_load.is_set():
                    print("Waiting for web page to begin loading...")
                self._event_load.wait()
                self._event_load.clear()
            self._wait_for_page = False

        if self._pause_stream:
            return

        if self._notified and self.show_rew:
            # The browser has not consumed the previous camera frame yet. Do not
            # drop this reward step: update its histories independently instead.
            self._collect_step_data()
            self._plot_step_counter = getattr(self, '_plot_step_counter', 0) + 1
            self._update_rew_plot()
            self._event_stream_rew.set()
            return

        if self._notified:
            return

        if fetch_results:
            self._gym.fetch_results(self._sim, True)
        if step_graphics:
            self._gym.step_graphics(self._sim)
        if render_all_camera_sensors:
            self._gym.render_all_camera_sensors(self._sim)

        image = self._gym.get_camera_image(self._sim,
                                           self._envs[self._camera_id],
                                           self._cameras[self._camera_id],
                                           self._camera_type)
        if self._camera_type == gymapi.IMAGE_COLOR:
            self._image = image.reshape(image.shape[0], -1, 4)[..., :3]
            if self.show_data:
                self._image = self._overlay_data(self._image)
        elif self._camera_type == gymapi.IMAGE_DEPTH:
            self._image = -image.reshape(image.shape[0], -1)
            minimum = 0 if np.isinf(np.min(self._image)) else np.min(self._image)
            maximum = 5 if np.isinf(np.max(self._image)) else np.max(self._image)
            self._image = np.clip(1 - (self._image - minimum) / (maximum - minimum), 0, 1)
            self._image = np.uint8(255 * self._image)
        else:
            raise ValueError("Unsupported camera type")

        if self._env.cfg.depth.use_camera:
            # The frame the POLICY actually consumed: get_depth_obs applies the per-env
            # camera latency, unlike depth_buffer[:, -1] which hides the transport delay.
            _consumed = self._env.get_depth_obs()
            self._image_depth = _consumed[self._camera_id].cpu().numpy() + 0.5
            self._image_depth = np.uint8(255 * np.clip(self._image_depth, 0.0, 1.0))
        
        root_pos = self._env.root_states[self._camera_id, :3].cpu().numpy()
        cam_pos = root_pos + self.cam_pos_rel
        self._gym.set_camera_location(self._cameras[self._camera_id], self._envs[self._camera_id], gymapi.Vec3(*cam_pos), gymapi.Vec3(*root_pos))

        self._collect_step_data()

        self._plot_step_counter = getattr(self, '_plot_step_counter', 0) + 1
        if self.show_plot and self._plot_step_counter % self._fxfz_plot_decimation == 0:
            self._update_fxfz_plot()
        if self.show_rew and self._plot_step_counter % self._rew_plot_decimation == 0:
            self._update_rew_plot()

        self._event_stream.set()
        if self._env.cfg.depth.use_camera:
            self._event_stream_depth.set()
        self._event_stream_plot.set()
        self._event_stream_rew.set()
        self._notified = True


def ik(jacobian_end_effector: torch.Tensor,
       current_position: torch.Tensor,
       current_orientation: torch.Tensor,
       goal_position: torch.Tensor,
       goal_orientation: Optional[torch.Tensor] = None,
       damping_factor: float = 0.05,
       squeeze_output: bool = True) -> torch.Tensor:
    """
    Inverse kinematics using damped least squares method

    :param jacobian_end_effector: End effector's jacobian
    :type jacobian_end_effector: torch.Tensor
    :param current_position: End effector's current position
    :type current_position: torch.Tensor
    :param current_orientation: End effector's current orientation
    :type current_orientation: torch.Tensor
    :param goal_position: End effector's goal position
    :type goal_position: torch.Tensor
    :param goal_orientation: End effector's goal orientation (default: None)
    :type goal_orientation: torch.Tensor or None
    :param damping_factor: Damping factor (default: 0.05)
    :type damping_factor: float
    :param squeeze_output: Squeeze output (default: True)
    :type squeeze_output: bool

    :return: Change in joint angles
    :rtype: torch.Tensor
    """
    if goal_orientation is None:
        goal_orientation = current_orientation

    q = torch_utils.quat_mul(goal_orientation, torch_utils.quat_conjugate(current_orientation))
    error = torch.cat([goal_position - current_position,
                       q[:, 0:3] * torch.sign(q[:, 3]).unsqueeze(-1)],
                      dim=-1).unsqueeze(-1)

    # solve damped least squares (dO = J.T * V)
    transpose = torch.transpose(jacobian_end_effector, 1, 2)
    lmbda = torch.eye(6, device=jacobian_end_effector.device) * (damping_factor ** 2)
    if squeeze_output:
        return (transpose @ torch.inverse(jacobian_end_effector @ transpose + lmbda) @ error).squeeze(dim=2)
    else:
        return transpose @ torch.inverse(jacobian_end_effector @ transpose + lmbda) @ error

def print_arguments(args):
    print("")
    print("Arguments")
    for a in args.__dict__:
        print(f"  |-- {a}: {args.__getattribute__(a)}")

def print_asset_options(asset_options: 'isaacgym.gymapi.AssetOptions', asset_name: str = ""):
    attrs = ["angular_damping", "armature", "collapse_fixed_joints", "convex_decomposition_from_submeshes",
             "default_dof_drive_mode", "density", "disable_gravity", "fix_base_link", "flip_visual_attachments",
             "linear_damping", "max_angular_velocity", "max_linear_velocity", "mesh_normal_mode", "min_particle_mass",
             "override_com", "override_inertia", "replace_cylinder_with_capsule", "tendon_limit_stiffness", "thickness",
             "use_mesh_materials", "use_physx_armature", "vhacd_enabled"]  # vhacd_params
    print("\nAsset options{}".format(f" ({asset_name})" if asset_name else ""))
    for attr in attrs:
        print("  |-- {}: {}".format(attr, getattr(asset_options, attr) if hasattr(asset_options, attr) else "--"))
        if attr == "vhacd_enabled" and hasattr(asset_options, attr) and getattr(asset_options, attr):
            vhacd_attrs = ["alpha", "beta", "concavity", "convex_hull_approximation", "convex_hull_downsampling",
                           "max_convex_hulls", "max_num_vertices_per_ch", "min_volume_per_ch", "mode", "ocl_acceleration",
                           "pca", "plane_downsampling", "project_hull_vertices", "resolution"]
            print("  |-- vhacd_params:")
            for vhacd_attr in vhacd_attrs:
                print("  |   |-- {}: {}".format(vhacd_attr, getattr(asset_options.vhacd_params, vhacd_attr) \
                    if hasattr(asset_options.vhacd_params, vhacd_attr) else "--"))

def print_sim_components(gym, sim):
    print("")
    print("Sim components")
    print("  |--  env count:", gym.get_env_count(sim))
    print("  |--  actor count:", gym.get_sim_actor_count(sim))
    print("  |--  rigid body count:", gym.get_sim_rigid_body_count(sim))
    print("  |--  joint count:", gym.get_sim_joint_count(sim))
    print("  |--  dof count:", gym.get_sim_dof_count(sim))
    print("  |--  force sensor count:", gym.get_sim_force_sensor_count(sim))

def print_env_components(gym, env):
    print("")
    print("Env components")
    print("  |--  actor count:", gym.get_actor_count(env))
    print("  |--  rigid body count:", gym.get_env_rigid_body_count(env))
    print("  |--  joint count:", gym.get_env_joint_count(env))
    print("  |--  dof count:", gym.get_env_dof_count(env))

def print_actor_components(gym, env, actor):
    print("")
    print("Actor components")
    print("  |--  rigid body count:", gym.get_actor_rigid_body_count(env, actor))
    print("  |--  joint count:", gym.get_actor_joint_count(env, actor))
    print("  |--  dof count:", gym.get_actor_dof_count(env, actor))
    print("  |--  actuator count:", gym.get_actor_actuator_count(env, actor))
    print("  |--  rigid shape count:", gym.get_actor_rigid_shape_count(env, actor))
    print("  |--  soft body count:", gym.get_actor_soft_body_count(env, actor))
    print("  |--  tendon count:", gym.get_actor_tendon_count(env, actor))

def print_dof_properties(gymapi, props):
    print("")
    print("DOF properties")
    print("  |--  hasLimits:", props["hasLimits"])
    print("  |--  lower:", props["lower"])
    print("  |--  upper:", props["upper"])
    print("  |--  driveMode:", props["driveMode"])
    print("  |      |-- {}: gymapi.DOF_MODE_NONE".format(int(gymapi.DOF_MODE_NONE)))
    print("  |      |-- {}: gymapi.DOF_MODE_POS".format(int(gymapi.DOF_MODE_POS)))
    print("  |      |-- {}: gymapi.DOF_MODE_VEL".format(int(gymapi.DOF_MODE_VEL)))
    print("  |      |-- {}: gymapi.DOF_MODE_EFFORT".format(int(gymapi.DOF_MODE_EFFORT)))
    print("  |--  stiffness:", props["stiffness"])
    print("  |--  damping:", props["damping"])
    print("  |--  velocity (max):", props["velocity"])
    print("  |--  effort (max):", props["effort"])
    print("  |--  friction:", props["friction"])
    print("  |--  armature:", props["armature"])

def print_links_and_dofs(gym, asset):
    link_dict = gym.get_asset_rigid_body_dict(asset)
    dof_dict = gym.get_asset_dof_dict(asset)

    print("")
    print("Links")
    for k in link_dict:
        print(f"  |-- {k}: {link_dict[k]}")
    print("DOFs")
    for k in dof_dict:
        print(f"  |-- {k}: {dof_dict[k]}")
