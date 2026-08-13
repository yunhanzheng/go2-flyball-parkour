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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


_GO2_REFERENCE_BASE_HEIGHT = 0.25

_GO2_HIP_OFFSETS = {
    "FL": np.array([0.1934, 0.0465, 0.0], dtype=np.float64),
    "FR": np.array([0.1934, -0.0465, 0.0], dtype=np.float64),
    "RL": np.array([-0.1934, 0.0465, 0.0], dtype=np.float64),
    "RR": np.array([-0.1934, -0.0465, 0.0], dtype=np.float64),
}
_GO2_HIP_TO_THIGH = np.array([0.0, 0.0955, 0.0], dtype=np.float64)
_GO2_THIGH_TO_CALF = np.array([0.0, 0.0, -0.213], dtype=np.float64)
_GO2_CALF_TO_FOOT = np.array([0.0, 0.0, -0.213], dtype=np.float64)
_GO2_JOINT_LIMITS = np.array([
    [-1.0472, 1.0472],
    [-1.5708, 3.4907],
    [-2.7227, -0.83776],
], dtype=np.float64)

_GO2_REFERENCE_STAND_ANGLES = {
    "FL": np.array([0.1, 0.8, -1.5], dtype=np.float64),
    "FR": np.array([-0.1, 0.8, -1.5], dtype=np.float64),
    "RL": np.array([0.1, 1.0, -1.5], dtype=np.float64),
    "RR": np.array([-0.1, 1.0, -1.5], dtype=np.float64),
}


def _rot_x(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ], dtype=np.float64)


def _rot_y(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=np.float64)


def _go2_leg_fk(leg, angles):
    q1, q2, q3 = np.asarray(angles, dtype=np.float64)
    p = _GO2_HIP_OFFSETS[leg].copy()
    rx = _rot_x(q1)
    ry2 = _rot_y(q2)
    ry3 = _rot_y(q3)
    p += rx @ _GO2_HIP_TO_THIGH
    p += rx @ ry2 @ _GO2_THIGH_TO_CALF
    p += rx @ ry2 @ ry3 @ _GO2_CALF_TO_FOOT
    return p


_GO2_REFERENCE_FOOT_TARGETS = {
    leg: _go2_leg_fk(leg, angles)
    for leg, angles in _GO2_REFERENCE_STAND_ANGLES.items()
}


def _go2_leg_ik(leg, target_pos, seed_angles):
    q = np.asarray(seed_angles, dtype=np.float64).copy()
    target = np.asarray(target_pos, dtype=np.float64)

    for _ in range(50):
        current = _go2_leg_fk(leg, q)
        err = target - current
        if np.linalg.norm(err) < 1e-12:
            break

        jacobian = np.zeros((3, 3), dtype=np.float64)
        eps = 1e-6
        for i in range(3):
            dq = np.zeros(3, dtype=np.float64)
            dq[i] = eps
            jacobian[:, i] = (_go2_leg_fk(leg, q + dq) - current) / eps

        damping = 1e-4
        lhs = jacobian @ jacobian.T + damping * np.eye(3, dtype=np.float64)
        dq = jacobian.T @ np.linalg.solve(lhs, err)
        q = np.clip(q + dq, _GO2_JOINT_LIMITS[:, 0], _GO2_JOINT_LIMITS[:, 1])

    return q


def _go2_default_joint_angles_from_base_height(base_height_target):
    """Return the Go2 standing pose used by the reward penalties.

    The solver keeps the nominal foot placement from the 0.25 m stance and
    adjusts the foot height by the requested base height. The result is used
    for both the zero-action target and the regularization penalties, so one
    base-height knob controls the whole standing pose.
    """
    base_height_target = float(base_height_target)
    delta = base_height_target - _GO2_REFERENCE_BASE_HEIGHT
    joint_angles = {}

    for leg, seed_angles in _GO2_REFERENCE_STAND_ANGLES.items():
        target_pos = _GO2_REFERENCE_FOOT_TARGETS[leg].copy()
        target_pos[2] -= delta
        solved = _go2_leg_ik(leg, target_pos, seed_angles)
        joint_angles[f"{leg}_hip_joint"] = float(solved[0])
        joint_angles[f"{leg}_thigh_joint"] = float(solved[1])
        joint_angles[f"{leg}_calf_joint"] = float(solved[2])

    return joint_angles

class Go2RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        # Do not expose the unavailable-on-robot flat-terrain oracle bit.
        give_flat_terrain_flag = False

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles_from_base_height = staticmethod(_go2_default_joint_angles_from_base_height)
        default_joint_angles = _go2_default_joint_angles_from_base_height(_GO2_REFERENCE_BASE_HEIGHT)

    class init_state_slope( LeggedRobotCfg.init_state ):
        pos = [0.56, 0.0, 0.24] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.03,   # [rad]
            'RL_hip_joint': 0.03,   # [rad]
            'FR_hip_joint': -0.03,  # [rad]
            'RR_hip_joint': -0.03,   # [rad]

            'FL_thigh_joint': 1.0,     # [rad]
            'RL_thigh_joint': 1.9,   # [rad]1.8
            'FR_thigh_joint': 1.0,     # [rad]
            'RR_thigh_joint': 1.9,   # [rad]

            'FL_calf_joint': -2.2,   # [rad]
            'RL_calf_joint': -0.9,    # [rad]
            'FR_calf_joint': -2.2,  # [rad]
            'RR_calf_joint': -0.9,    # [rad]
        }
        
    class control( LeggedRobotCfg.control ):
        control_type = 'P'
        # Deployment must use the same PD gains as training.
        stiffness = {'joint': 30.}  # [N*m/rad]
        damping = {'joint': 0.6}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class sim( LeggedRobotCfg.sim ):
        class physx( LeggedRobotCfg.sim.physx ):
            max_gpu_contact_pairs = 2**24 
            default_buffer_size_multiplier =  20 

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["base", "thigh", "calf", "Head"]
        terminate_after_contacts_on = ["base","Head"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class domain_rand( LeggedRobotCfg.domain_rand ):
        # Go2-specific DR on top of the inherited friction/mass/push/motor-strength/
        # action-delay randomizations. Dynamics only, not observed.

        # Per-env, per-joint Kp/Kd multiplier (controller stiffness mismatch).
        randomize_pd_gain = False
        kp_range = [0.9, 1.1]
        kd_range = [0.9, 1.1]

        # Per-env, per-joint PD target bias (encoder/zero-calibration error).
        randomize_motor_offset = True
        motor_offset_range = [-0.035, 0.035]   # [rad]

        randomize_joint_friction = False
        joint_friction_range = [0.0, 0.04]     # [N.m]
        randomize_joint_damping = False
        joint_damping_range = [0.0, 0.04]      # [N.m.s/rad]

        randomize_link_mass = False            # disabled, kept for reference
        link_mass_range = [0.9, 1.1]

        # Foot contact-shape restitution (bounciness).
        randomize_restitution = True
        restitution_range = [0.0, 0.5]         # 0 = inelastic, 1 = elastic

        # Push disturbance disabled for all go2 tasks.
        push_robots = False
        push_interval_s = 4
        max_push_vel_xy = 0.25
        max_push_vel_z = 0.0
        max_push_ang_vel = 0.25

        randomize_action_latency = True
        action_latency_range = [0, 2]           # control steps -> 0..40 ms

        # Per-env proprio (joint/IMU) latency, ramped by DR level. Observation copy only — PD control, rewards, terminations stay zero-latency.
        randomize_proprio_latency = True
        proprio_latency_range_s = [0.0, 0.030]  # seconds

        # DR strength scales with terrain level (easy -> gentle, hard -> full range); runtime DR only, not creation-time friction/mass.
        dr_lock_to_terrain_level = True
        dr_scale_range = [0.3, 1.0]

    class noise( LeggedRobotCfg.noise ):
        # Base config defines noise scales but leaves add_noise off; enable it.
        add_noise = True
        noise_level = 1.0
        class noise_scales( LeggedRobotCfg.noise.noise_scales ):
            rotation = 0.05    # roll/pitch from IMU [rad] (base had 0.0)

    class depth( LeggedRobotCfg.depth ):
        # Perception DR shared by every go2 task; inert unless a task turns on use_camera (lidar/privileged policies never touch this).

        use_warp = True            # required for pw_noise/latency to apply
        capture_interval = 1       # 50 Hz capture, 10 Hz encoder cadence

        # Calibrated at depth_noise_scale=0.5; scales linearly with the per-env draw s in [0,1].
        dis_noise = 0.05
        noise_gaussian = 0.006
        noise_dropout = 0.0025     # per-pixel seed prob for a horizontal run
        dropout_cluster = True
        dropout_cluster_ext = 0.55
        dropout_jitter = 0.8       # [m]
        dropout_near_prob = 0.4

        # Parkour-in-the-Wild depth degradation (arXiv 2505.11164 §2.4): edge erosion, holes, blind spot, far-empties, blur.
        pw_noise = True
        pw_empty_min = 0.15
        pw_edge_thresh = 0.50      # confine edge noise to strong true silhouettes
        pw_edge_prob = 0.85
        pw_edge_near_frac = 0.03
        pw_edge_far_frac = 0.90
        pw_edge_dilate = 1
        pw_hole_prob = 0.004
        pw_hole_scale = 8
        pw_hole_drift = 0.1
        # Pinned to the left image edge (matches the real sensor's invalid band).
        pw_blindspot_cols = 3
        pw_blindspot_side = 'left'
        pw_blindspot_drift = 0.05
        pw_blindspot_slope_max = 0.3
        pw_blindspot_jitter = 1.0
        pw_blur_sigma = 0.45

        # Camera transport/processing latency, in 20 ms capture frames; both bounds
        # ramp with the fleet DR curriculum. buffer_len auto-grows to 11.
        latency_range = [3, 10]

    class commands( LeggedRobotCfg.commands ):
        resampling_time = 5.

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh'
        hf2mesh_method = "grid"
        max_error = 0.1
        max_error_camera = 2
        y_range = [-0.4, 0.4]
        edge_width_thresh = 0.05
        horizontal_scale = 0.05
        horizontal_scale_camera = 0.1
        vertical_scale = 0.005
        border_size = 5
        height = [0.02, 0.06]
        simplify_grid = False
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        flyball_pit_width_range          = (0.20, 0.65)
        flyball_pit_height_offset_range  = (0.05, 0.20)
        flyball_pit_depth                = 2.0
        flyball_wall_angle_deg_range     = (20.0, 50.0)
        flyball_wall_contact_fxfz_threshold = -0.5
        # two_hurdles_wall_turn hurdle geometry
        flyball_hurdle_thickness         = 0.10           # [m] depth along corridor
        flyball_hurdle_width             = 2.0            # [m] lateral span
        flyball_hurdle_height_range      = (0.05, 0.25)   # [m] easy→hard
        flyball_hurdle_gap_range         = (1.1, 2.4)     # [m] start → hurdle 1
        flyball_inter_hurdle_range       = (1.5, 2.8)     # [m] hurdle 1 → hurdle 2
        # WoCoCo contact regions (offsets relative to wall_front_x)
        flyball_liftoff_x_range   = (-1.3500, -0.900)
        flyball_liftoff_y_mid     = -0.5600
        flyball_liftoff_y_half    =  0.5300
        flyball_liftoff_z_above   =  0.0500
        flyball_wall_y_half       =  0.8200
        flyball_wall_z_min_above  =  0.1800
        flyball_wall_x_depth_scale =  1.0
        flyball_landing_x_range   = (-1.6500, -0.8500)
        flyball_landing_y_mid     =  0.6500
        flyball_landing_y_half    =  1.0000
        flyball_landing_z_above   =  0.0500
        flyball_air_x_range       = (-1.6500, -0.1000)
        flyball_air_y_mid         =  0.6500
        flyball_air_y_half        =  1.0000
        flyball_air_z_min_above   =  0.0500
        flyball_air_z_max_above   =  0.8000
        curriculum = True
        all_vertical = False
        no_flat = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # Height-field measurement (proprio obs)
        measure_heights = True
        measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2]
        measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]
        measure_horizontal_noise = 0.0
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5
        resample_at_max_level = False  # clamp at the top row; do NOT redistribute
        terrain_length = 18.
        terrain_width = 4
        num_rows = 10
        num_cols = 40
        slope_treshold = 11.43
        origin_zero_z = True
        num_goals = 8
        # Terrain mix — exclusive flyball_box training
        terrain_dict = {"smooth slope": 0.,
                        "rough slope up": 0.,
                        "rough slope down": 0.,
                        "rough stairs up": 0.,
                        "rough stairs down": 0.,
                        "discrete": 0.,
                        "stepping stones": 0.,
                        "gaps": 0.,
                        "smooth flat": 0.0,
                        "parkour_stones": 0.,
                        "wall": 0.,
                        "platform": 0.,
                        "large stairs up": 0.,
                        "large stairs down": 0.,
                        "parkour": 0.,
                        "two_hurdles_wall_turn": 0., # exclusive training terrain
                        "parkour_flat": 0.,
                        "parkour_step": 0.,
                        "flyball_box": 1.,       # wall-turn training terrain
                        "demo": 0.,
                        }
        terrain_proportions = list(terrain_dict.values())

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.26   # runtime IK stance target
        # tracking_goal_vel/tracking_yaw always on here; go2_teacher_config sets this
        # True to silence tracking during the wall-plant stages.
        gate_tracking_during_wococo_plant = False

        class scales( LeggedRobotCfg.rewards.scales ):
            action_rate = -0.1
            collision   = -10.

class Go2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go2'

  
