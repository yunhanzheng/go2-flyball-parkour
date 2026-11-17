"""go2_teacher_maxdr — teacher task on the frozen row-difficulty grid.

The "already learned the curriculum" variant: envs are spread UNIFORMLY across
every difficulty tile (row) once and never promote/demote, every randomization
is pinned at full strength.

Rewards: gating values reverted to main (user decision 2026-11-14). The parent
Go2TeacherCfg's 8 "warm-start full-arm" overrides are pinned back to main here
(radii below_level=4, clearance arm_step=2000*24, plant_fail arm=4.0, etc.).
Reward SCALES tuned vs main on 2026-11-14 (termination / motion-reg down, task
tracking up; standstill_orientation kept at main). See rewards.scales below.

  terrain (frozen row-difficulty design; inherits the MODIFIED teacher terrain)
    curriculum = False            -> row-based difficulty grid d = i/9 (same
                                     build as main); envs draw a random row once
                                     (uniform, randint(0, num_rows)) and never
                                     promote/demote, so row == difficulty == wall
                                     angle and exposure is uniform across levels.
    height = [0.02, 0.06]         -> ground roughness: 50% flat (p=0.5 coin flip,
                                     unchanged) + 50% per-tile U(0.02, 0.06)
                                     (geomax=True splices the range).
    max_error = 2.0               -> collision-mesh simplification (fix SIGSEGV):
                                     all-level-9 roughness makes the delatin mesh
                                     too dense at max_error=1.0 (27.8M tris);
                                     2.0 -> ~11.8M tris. heightsamples stay full-res.
    flyball_return_goal_gap_jitter = 0.30  -> g5 return-goal jitter = main (0.30).
    log_per_level_metrics = True  -> per_level/L0..L9/{success,goal_idx,rew_total,
                                     ep_len_s} TB telemetry (bins reset episodes by
                                     frozen row; runner nanmeans empty-row bins).
    inherited unchanged: wall angle (30,75)+-5 lerp; delta_x = main lerp(0.20,0.65,d)
                                     difficulty-coupled; corridor jitter +-0.50 full
                                     (scale_with_difficulty=False); hurdle-top
                                     roughness U[0,0.01]; hurdle count zero_prob 0.25;
                                     zero_below_level=4; geometry_at_max_difficulty=True.

  domain_rand
    dr_scale_range = [1.0, 1.0] and dr_ramp_use_mean_level = False
                                  -> dr_scale identically 1.0 (full DR every reset).
    action_delay_start_latched = True
                                  -> fixed 20 ms delay active from step 0.

  rewards (gating = main; scales tuned 2026-11-14)
    only_positive_rewards_clip_below_level = 2
    hurdle_clearance_arm_step = 0, ramp_steps = 0 (2026-11-17: full tax from
                                  step 0; the arm-then-ramp schedule made the value
                                  target unpredictable mid-ramp via per_env_draw and
                                  its first arm on this line drove the it13500-16000
                                  goal drop)
    wococo_stage_anneal_start/end = 5.0/7.0 (inert: frozen mean 4.5 < 5)
    wococo_per_angle_radii_low_below_level = 4 (lenient shallow walls)
    wococo_plant_fail_term_arm_mean_level = 4.0 (frozen 4.5 -> armed)
    wococo_armed_term_exempt_min_mean_level = 6.0 (frozen 4.5 -> never arms)
    scales tuned vs main: termination -150->-60, ang_vel_xy -0.05->-0.02,
      action_rate -0.1->-0.05, dof_acc -2.5e-7->-1.2e-7, collision -20->-12,
      tracking_goal_vel 1.5->2.2, wococo_contact_pts 0.5->1.0,
      wococo_foot_track 0.75->1.2, tracking_yaw 0.5->0.6.
      standstill_orientation KEPT 0.5 (env-gated, not the standstill driver).
    2026-11-16 anti-crouch retune (it0 vs it10000 |rew| decomposition): the
      motion-reg block grew 15.6%->23.3% (dof_acc x4.4, action_rate x3.5) while
      the posture anchor dof_error shrank 5.5%->3.7% and orientation stays 0 on
      wall lanes -> policy settled into a low-crouch shuffle that saves motion
      cost while keeping the 39% tracking income. Ease motion + strengthen the
      posture anchor: dof_acc -1.2e-7->-0.6e-7, action_rate -0.05->-0.03,
      dof_error -0.04->-0.08 (reference stance IS the 0.25 m tall pose, so this
      is the weight-only base_height proxy; plant window x0.25 protects the wall
      maneuver). Falls still end ~60% of episodes (roll 34.6% + base 25.2%) but
      termination fear is only 2.7% -> termination KEPT -60 this round.
    2026-11-16 collision bump (user decision, bundled into the ac-warm8000 leg):
      knee telemetry shows 73% of episodes take a >10 N thigh/calf hit while
      rew_collision was only ~1.6% of the budget (-0.059 at it10000) -> impacts
      were nearly free. collision -12->-20 (back to the teacher main value,
      x1.67). Kept moderate: the term is a binary >0.1 N per-body count, so
      legitimate shin-on-wall grazes during plant/clearance pay full price too;
      harder weights risk shy wall approaches. Not an anti-crouch lever per se
      (knee contact is only 0.5% of steps; crouch saves motion cost, not contact).

Launch:
    python train.py --task go2_teacher_maxdr --exptid <id> --device cuda:0
Warm-start from the teacher checkpoint:
    python train.py --task go2_teacher_maxdr --exptid <new_id> \
        --resume --resumeid teacher --checkpoint 3200 --device cuda:0
"""

from legged_gym.envs.go2.go2_teacher_config import (
    Go2TeacherCfg, Go2TeacherCfgPPO)


class Go2TeacherMaxDrCfg(Go2TeacherCfg):

    class terrain(Go2TeacherCfg.terrain):
        # Frozen row-difficulty design (experiment-log.md §12.8).
        # Inherits the MODIFIED teacher terrain: geometry_at_max_difficulty=True,
        # corridor jitter +-0.50 full (scale_with_difficulty=False), wall angle
        # (30,75)+-5 lerp, hurdle count zero_prob=0.25, zero_below_level=4,
        # hurdle-top roughness (0.0,0.01). Only the overrides below differ.
        #
        # Live curriculum OFF, but the BUILD is main's row-difficulty grid
        # (d = i/9; terrain.py non-curriculum branch calls curiculum()). envs draw
        # a random row once (uniform) and never promote/demote, so row == difficulty
        # == wall angle and exposure is uniform across all levels (mean ~4.5).
        curriculum     = False
        max_difficulty = False
        min_terrain_level = 0

        # Per-difficulty TB telemetry: bin finished episodes by terrain row
        # (== difficulty == wall angle) -> per_level/L0..L9/{success,goal_idx,
        # rew_total,ep_len_s}. Opt-in; the runner nanmeans empty-row NaN bins and
        # the writer skips all-NaN rows. Other runs (flag absent) are unaffected.
        log_per_level_metrics = True

        # delta_x (forbidden-zone width) is IDENTICAL to main: delta_x =
        # lerp(0.20, 0.65, d), difficulty-coupled (NOT pinned). The active-tree
        # flyball_wall.py always lerps delta_x (the delta_x_follow_difficulty flag
        # is inert here), so the result matches main regardless. geomax=True pins
        # the OTHER geometry (roughness / jitter / hurdle-top).
        flyball_delta_x_follow_difficulty = True

        # Ground roughness: the p=0.5 coin flip (flyball_roughness_zero_prob) keeps
        # half the tiles completely flat (unchanged). The rough half now draws a
        # per-tile amplitude U[height[0],height[1]] = U(0.02, 0.06) instead of the
        # fixed 0.06, because geometry_at_max_difficulty=True splices the range.
        height = [0.02, 0.06]

        # Collision-mesh simplification (fix SIGSEGV at add_triangle_mesh).
        # All-level-9 roughness makes the delatin mesh too dense for PhysX:
        # max_error=1.0 -> 27.8M triangles -> segfault. max_error=2.0 -> ~11.8M
        # triangles, ~40% below the known-good level. Only the PhysX collision
        # mesh is simplified; heightsamples (rewards / terrain-level / depth)
        # stay full-resolution. ~10mm vertical tolerance is negligible.
        max_error = 2.0

        # Hurdle-top roughness stays U[0,0.01] (level-9 ceiling) via inherited
        # geometry_at_max_difficulty=True + hurdle_roughness_height=(0.0,0.01).
        # Wall angle stays main lerp(30,75,d)+-5 (inherited; NOT pinned/uniform).
        # Corridor jitter stays +-0.50 full (inherited scale_with_difficulty=False).

        # Return-goal (g5) gap jitter = main's 0.30 (g5_gap = 1.70 +- 0.30).
        flyball_return_goal_gap_jitter = 0.30

    class domain_rand(Go2TeacherCfg.domain_rand):
        # dr_scale identically 1.0: no per-level alpha, no unarmed zeroing.
        dr_scale_range         = [1.0, 1.0]
        dr_ramp_use_mean_level = False
        dr_scale_sample_per_env = True

        # Fixed 20 ms action delay from step 0 (no fleet-mean-level latch).
        action_delay               = True
        action_delay_start_latched = True

        # Creation-time draws, restated so the full span is explicit.
        randomize_friction  = True
        friction_range      = [0.6, 2.0]      # 64 buckets
        randomize_base_mass = True
        added_mass_range    = [0.0, 3.0]      # [kg]
        randomize_base_com  = True
        added_com_range     = [-0.2, 0.2]     # [m] per axis
        randomize_motor     = True
        motor_strength_range = [0.8, 1.2]

        # Per-episode draws (now always at the ceiling of their ranges).
        randomize_pd_gain      = True
        kp_range               = [0.95, 1.05]
        kd_range               = [0.95, 1.05]
        randomize_motor_offset = True
        motor_offset_range     = [-0.035, 0.035]   # [rad]
        randomize_restitution  = True
        restitution_range      = [0.0, 0.5]

    class rewards(Go2TeacherCfg.rewards):
        # ALL rewards reverted to main (user decision 2026-11-14).
        # The parent Go2TeacherCfg has 8 "warm-start full-arm" overrides;
        # every one is pinned back to main's value here.

        # main: clip negative totals only below level 2 (cold-start rows).
        only_positive_rewards_clip_below_level = 2

        # 2026-11-17 (user decision): full tax from step 0. The old arm-then-ramp
        # (arm it2000, ramp it2000-6000) hurt the critic two ways: per_env_draw
        # makes the same state yield two different returns mid-ramp (draw is not in
        # the obs), and the schedule's first arm on THIS line (resume counter restart
        # -> it12000) drove the it13500-16000 goal drop. The tax itself is kept
        # (scale -1.0): it penalises the touch-free graze-over that collision -20
        # cannot see. arm_step=0 arms at counter 0; ramp_steps=0 = legacy immediate
        # full scale. per_env_draw stays True: with alpha identically 1.0 the draw
        # is always armed, so no value ambiguity remains.
        hurdle_clearance_arm_step         = 0
        hurdle_clearance_ramp_steps       = 0

        # main: stage bonus anneals over fleet-mean 5->7 (frozen mean 4.5
        # never crosses 5.0, so factor stays 1.0 = full bonus; inert).
        wococo_stage_anneal_start_mean_level = 5.0
        wococo_stage_anneal_end_mean_level   = 7.0

        # main: lenient low radius table on rows 0-3 (shallow walls).
        wococo_per_angle_radii_low_below_level = 4

        # main: plant-fail arms at fleet mean >= 4.0 (frozen 4.5 -> armed).
        wococo_plant_fail_term_arm_mean_level = 4.0

        # main: armed-term exempt at fleet mean >= 6.0 (frozen 4.5 -> NEVER
        # arms; roll/pitch termination stays live during wall plant).
        wococo_armed_term_exempt_min_mean_level = 6.0

        # User decision 2026-11-15: stage1 hold 3 -> 1 step. it500 L9 A/B proved
        # the 3-step (60 ms) static hold was the ONLY binding gate: the policy's
        # dynamic plant presses both rears >50 N in-radius at yaw ~90 deg for
        # 2 steps then pushes off; n_persist=1 lets that exact motion advance
        # stage1->2->3 and clear the wall goal (play-only np1 video verified).
        wococo_stage1_n_persist = 1

        # Reward-WEIGHT tuning (user decision 2026-11-14). it0/it600 reward
        # decomposition showed termination fear (it0 #1 penalty, 73% approach
        # roll-flips) drove the policy to stand still; motion regularization then
        # accumulated over long idle episodes (it600 #1). Reduce termination fear
        # + motion cost, boost task tracking. standstill_orientation KEPT at main
        # 0.5: it is env-gated (|cmd_x|<lin_vel_clip), not farmable during approach,
        # so it is NOT the standstill driver.

        # 2026-11-16 knee-clearance penalty REMOVED at user request: _reward_knee_clearance
        # and its knobs are gone. The residual knee scrape is now handled by collision,
        # restored to the main -20 below (penalize_contacts_on = thigh/calf/Head already
        # covers the offending bodies; foot stays legal).

        class scales(Go2TeacherCfg.rewards.scales):
            termination        = -60.0    # main -150: reduce fear of flipping
            ang_vel_xy         = -0.02    # main -0.05: allow trunk rotation (recovery / wall turn)
            action_rate        = -0.05    # REVERTED §13.5 (2026-11-16): -0.03 back to the 11-14 value
            dof_acc            = -1.2e-7  # REVERTED §13.5 (2026-11-16): -0.6e-7 back to the 11-14 value
            collision          = -20.0    # 2026-11-16: back to main -20 (user decision). Penalizes thigh/calf/Head contact on wall OR ground; foot stays legal.
            dof_error          = -0.08    # 2026-11-16: DOUBLED from inherited base -0.04. Stiffer/taller stance anchor lifts the knees, suppressing the residual scrape.
            tracking_goal_vel  = 2.2      # main 1.5: reward forward progress
            wococo_contact_pts = 1.0      # main 0.5: denser wall-plant guidance (drives FSM, must stay != 0)
            wococo_foot_track  = 1.2      # main 0.75: denser foot guidance
            tracking_yaw       = 0.6      # main 0.5
            # wococo_stage / wococo_foot_target stay 25 (main);
            # standstill_orientation stays 0.5 (main).


class Go2TeacherMaxDrCfgPPO(Go2TeacherCfgPPO):
    pass