"""
Go2 平地训练配置（与 flat-baseline-512 相同，只是地形改为纯平地）
- 纯平地地形（smooth flat = 1.0）
- 继承 go2_teacher_config 的随机化配置，使用平地专用奖励
- 双编码器（为后续复杂地形热启动准备）
"""

from legged_gym.envs.go2.go2_teacher_config import Go2TeacherCfg, Go2TeacherCfgPPO
from legged_gym.utils.helpers import class_to_dict


class Go2FlatCfg(Go2TeacherCfg):
    """平地训练的环境配置（地形改为纯平地）"""
    
    class env(Go2TeacherCfg.env):
        num_envs = 512
        episode_length_s = 20
        # 低速终止：只在 cmd > 0.1 时启用（避免静止命令触发）
        terminate_on_low_velocity = True
        low_velocity_term_mps = 0.1  # 速度阈值 0.1 m/s
        low_velocity_term_duration_steps = 50  # 持续 2 秒
        low_velocity_cmd_thresh = 0.1  # 命令阈值 0.1 m/s
        
    class terrain(Go2TeacherCfg.terrain):
        # smooth flat 位于第 9 个槽位（索引 8），生成 env_class=9。
        # 索引 9 是 parkour_stones（env_class=10），不能用于纯平地训练。
        terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # 同时兼容当前生成器的 class 9 和旧缓存中的 class 10。
        flat_terrain_env_classes = [9, 10]
        
        # 显式设置 terrain_dict（覆盖继承的 flyball_box）
        terrain_dict = {
            'smooth slope': 0.0, 'rough slope up': 0.0, 'rough slope down': 0.0,
            'rough stairs up': 0.0, 'rough stairs down': 0.0, 'discrete': 0.0,
            'stepping stones': 0.0, 'gaps': 0.0, 'smooth flat': 1.0,
            'parkour_stones': 0.0, 'wall': 0.0, 'platform': 0.0,
            'large stairs up': 0.0, 'large stairs down': 0.0, 'parkour': 0.0,
            'two_hurdles_wall_turn': 0.0, 'parkour_flat': 0.0, 'parkour_step': 0.0,
            'flyball_box': 0.0, 'demo': 0.0
        }
        
        # 平地不需要大地形网格，使用更小的配置以加速初始化
        num_cols = 8
        num_rows = 10
        horizontal_scale = 0.05
        terrain_length = 10.0
        border_size = 1.5
        
        curriculum = False
        max_init_terrain_level = 0
        resample_type_on_level_change = False
        
    class rewards(Go2TeacherCfg.rewards):
        # 平地没有贴墙状态机，也不应保留 parkour 的奖励塑形。
        gate_tracking_during_wococo_plant = False
        enable_final_goal_hard_stop = False
        only_positive_rewards = False

        class scales(Go2TeacherCfg.rewards.scales):
            # 前进是主任务；降低单纯对准目标方向就能获得的收益。
            tracking_goal_vel = 3.0
            tracking_yaw = 0.25
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            feet_air_time = 1.0

            # 保留基本姿态与安全约束，减轻抑制平地起步的运动惩罚。
            lin_vel_z = -1.0
            lin_vel_y = -0.5
            ang_vel_xy = -0.05
            orientation = -0.5
            dof_acc = -2.5e-8
            action_rate = -0.01
            delta_torques = 0.0
            torques = -0.00001
            hip_pos = -0.1
            dof_error = -0.01
            collision = -10.0
            dof_pos_limits = -10.0
            feet_stumble = 0.0
            feet_edge = 0.0
            feet_contact_forces = 0.0
            standstill_orientation = 0.0

            # 纯平地任务不启用 WoCoCo / hurdle 奖励。
            wococo_contact_pts = 0.0
            wococo_stage = 0.0
            wococo_foot_track = 0.0
            wococo_foot_target = 0.0
            wococo_landing_foot_target = 0.0
            wococo_base_orient = 0.0
            hurdle_contact = 0.0
            hurdle_clearance = 0.0
            hurdle_pair_sym = 0.0

            # 摔倒等真实失败保留终局惩罚；正常 timeout 不额外惩罚。
            termination = -150.0
            timeout = 0.0

    class commands(Go2TeacherCfg.commands):
        curriculum = False
        
    class depth(Go2TeacherCfg.depth):
        use_camera = True
        camera_num_envs = 512  # dual encoder + depth rendering VRAM 限制


class Go2FlatCfgPPO(Go2TeacherCfgPPO):
    """平地训练的 PPO 配置"""
    
    class runner(Go2TeacherCfgPPO.runner):
        experiment_name = 'go2_flat'
        run_name = 'flat_baseline'
        max_iterations = 5000
        save_interval = 100  # 每 100 轮保存一次
