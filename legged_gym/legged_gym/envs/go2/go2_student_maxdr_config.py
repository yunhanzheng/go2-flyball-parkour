"""go2_student_maxdr — warm-started pure-RL depth student on the maxdr task.

Initialization: legged_gym/logs/extreme-parkour/student/model_10000.pt (the
DAgger-distilled depth student). Objective from there on: PURE RL — no teacher
anchor, no L2 imitation, no BC:
  * actor  = depth_encoder(depth image) + proprio + history_encoder(proprio history)
             -> action. Deployable (depth-only perception, no privileged leak).
  * critic = privileged obs (height scan + true dynamics state). Training-only.
  * depth_encoder, depth_actor (incl. history_encoder), critic and estimator all
    keep training against the env reward. The checkpoint's teacher actor
    (actor_critic.actor) loads but stays frozen and unused (anchor coef = 0).

Checkpoint facts (verified 2026-11-15): model_10000.pt carries NO depth_actor_std
(sigma starts at rl_init_std below and trains), NO depth-RL/critic optimizer
state (fresh Adam), and iter=0 (the counter restarts at 0, so max_iterations
counts NEW iterations).

Task = go2_teacher_maxdr EXACTLY (terrain / domain_rand / rewards inherited from
Go2TeacherMaxDrCfg): frozen row-difficulty grid (curriculum=False, rows 0-9
uniform), ground roughness 0.06, max_error=2.0 (SIGSEGV fix), full DR
(dr_scale=1.0) + latched 20 ms action delay, and the 2026-11-14 tuned reward
scales (termination -60, tracking_goal_vel 2.2, ...).

Depth camera pipeline = main student (Go2StudentCfg.depth): 480x270, hole_fill,
self_geometry, latency [4,5], render_on_demand. override_terrain_{mix,grid}_for_camera
= False (inherited) so the camera path does NOT rewrite the maxdr terrain into the
standard parkour mix/grid.

Hyperparameters = main teacher PPO + main student depth camera. Go2StudentCfgPPO
already inherits the teacher algorithm/policy unchanged (lr 2e-4 adaptive,
entropy 0.01, desired_kl 0.01, init/max_noise_std 1.0, epochs 5, minibatches 4,
actor/critic hidden [512,256,128]). Only the depth-RL optimization knobs are
restated here (NOT the ARM-A fine-tune recipe): encoder lr>0, anchor=0, sigma
trainable + entropy>0, history_encoder via the PPO surrogate.

CRITIC WARM-START DIVERGENCE (verified on ARM-B long6): the checkpoint critic was
trained on MAIN student env returns; maxdr (frozen full-difficulty rows + the
2026-11-14 tuned scales) is a different return distribution. At 3e-4 it overshot
and diverged (rl_value 0.3 -> 2.2e9), contaminating the un-normalized advantages.
rl_critic_learning_rate is therefore 3e-5 (the verified fix). Early warning:
Loss_depth/rl_value climbing monotonically within it30.

Launch (warm start, pure RL; NO anchor / NO BC):
    python train.py --task go2_student_maxdr --exptid <id> --proj_name extreme-parkour \
        --headless --no_wandb --device cuda:0 --num_envs 1024 \
        --max_iterations 10000 --save_interval 100 \
        --resume --resumeid student --checkpoint 10000 \
        --freeze --depth_rl --use_camera --delay
"""

from legged_gym.envs.go2.go2_teacher_maxdr_config import Go2TeacherMaxDrCfg
from legged_gym.envs.go2.go2_student_config import (
    Go2StudentCfg, Go2StudentCfgPPO)


class Go2StudentMaxDrCfg(Go2StudentCfg):
    # env inherits Go2StudentCfg.env: terminate_on_yaw_tracking=False (depth-only
    # cannot observe heading). terminate_on_lateral_deviation stays True / 1.5 m
    # (inherited from teacher; lenient). Flip it here if depth-only false-kills
    # become a problem.

    class terrain(Go2TeacherMaxDrCfg.terrain):
        # EXACTLY maxdr terrain: curriculum=False, rows 0-9 uniform, height=[0.06,
        # 0.06], max_error=2.0, log_per_level_metrics=True, return-goal jitter 0.30.
        # Inherits the modified teacher geometry (geometry_at_max_difficulty=True,
        # wall-angle lerp, corridor jitter, hurdle count zero_prob=0.25, ...).
        pass

    class domain_rand(Go2TeacherMaxDrCfg.domain_rand):
        # EXACTLY maxdr DR: dr_scale identically 1.0, fixed 20 ms delay latched from
        # step 0, full friction/mass/com/motor/pd/restitution randomization.
        pass

    class rewards(Go2TeacherMaxDrCfg.rewards):
        # EXACTLY maxdr rewards: gating values = main, scales tuned 2026-11-14.
        pass

    class depth(Go2StudentCfg.depth):
        # main student camera pipeline. override_terrain_{mix,grid}_for_camera=False
        # (inherited) keeps the maxdr terrain intact on the camera path.
        pass


class Go2StudentMaxDrCfgPPO(Go2StudentCfgPPO):
    # algorithm + policy already == teacher (Go2StudentCfgPPO inherits
    # Go2TeacherCfgPPO without overriding them). Restated explicitly for clarity.

    class runner(Go2StudentCfgPPO.runner):
        save_interval = 100
        max_iterations = 10000

    class depth_encoder(Go2StudentCfgPPO.depth_encoder):
        # ---- pure-RL overrides (warm-started; NOT the ARM-A fine-tune recipe) ----
        rl_depth = True

        # actor: the teacher's exploration recipe (adaptive KL, full sigma, entropy).
        rl_learning_rate = 2.e-4
        rl_adaptive_lr = True
        rl_min_learning_rate = 1.e-5
        rl_max_learning_rate = 1.e-2
        rl_desired_kl = 0.01
        rl_step_ratio = None          # sigma~1.0 -> use desired_kl directly (not the
                                      # 0.4 ratio calibrated for the 0.05 fine-tune sigma)
        rl_init_std = 1.0             # teacher init_noise_std; the ckpt has NO
                                      # depth_actor_std, so this IS the starting sigma
        rl_max_noise_std = 1.0        # teacher max_noise_std
        rl_freeze_std = False         # sigma trains (ARM-A froze it at 0.05)
        rl_entropy_coef = 0.01        # teacher entropy_coef (ARM-A used 0)

        # encoder: keeps learning depth features end-to-end (ARM-A froze it at 0).
        rl_encoder_learning_rate = 3.e-4
        rl_encoder_anchor_coef = 0.0  # no teacher latent anchor — pure RL, no BC

        # history_encoder: initialized from the distilled checkpoint, but trained
        # THROUGH the PPO surrogate (end-to-end), NOT via an L2 regression onto the
        # frozen priv_encoder — the objective stays pure RL (no BC). Requires the
        # ppo.py rl_hist_via_surrogate flag (keeps the head inside depth_rl_optimizer).
        rl_hist_via_surrogate = True
        rl_hist_encoder_learning_rate = 0.0

        # critic: warm-started from the DAgger ckpt onto a MISMATCHED return
        # distribution (maxdr terrain + tuned scales). 3e-4 diverged on ARM-B
        # long6 (rl_value -> 2.2e9); 3e-5 is the verified fix. Watch rl_value it0-30.
        rl_critic_learning_rate = 3.e-5

        # minibatches = teacher's 4; encoder-replay gradient-accumulation width.
        rl_num_mini_batches = 4
        rl_replay_chunk = 512

        # DAgger-only knobs, inert on the depth-RL path; zeroed for clarity (no BC).
        action_noise_std = 0.0
        use_depth_encoder_loss = False