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

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import wandb
# import ml_runlog
import datetime

from rsl_rl.algorithms import PPO
from rsl_rl.modules import *
from rsl_rl.env import VecEnv
import sys
from copy import copy, deepcopy
import warnings

class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 init_wandb=True,
                 device='cpu', **kwargs):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.estimator_cfg = train_cfg["estimator"]
        self.depth_encoder_cfg = train_cfg["depth_encoder"]
        self.zero_student_velocity_obs = bool(
            self.depth_encoder_cfg.get("zero_student_velocity_obs", False))
        # Gaussian action noise for DAgger collection (see cfg.depth_encoder).
        self.dagger_action_noise_std = float(
            self.depth_encoder_cfg.get("action_noise_std", 0.0))
        self.device = device
        self.env = env

        print("Using MLP and Priviliged Env encoder ActorCritic structure")
        actor_critic: ActorCriticRMA = ActorCriticRMA(self.env.cfg.env.n_proprio,
                                                      self.env.cfg.env.n_scan,
                                                      self.env.num_obs,
                                                      self.env.cfg.env.n_priv_latent,
                                                      self.env.cfg.env.n_priv,
                                                      self.env.cfg.env.history_len,
                                                      self.env.num_actions,
                                                      **self.policy_cfg).to(self.device)
        estimator = Estimator(input_dim=env.cfg.env.n_proprio, output_dim=env.cfg.env.n_priv, hidden_dims=self.estimator_cfg["hidden_dims"]).to(self.device)
        # Depth encoder
        self.if_depth = self.depth_encoder_cfg["if_depth"]
        # Yaw predictor toggle (EP_USE_YAW=1): the depth encoder regresses a
        # 2-D heading that is fed into the student's obs[:,6:8] and supervised
        # against the teacher's commanded heading. Off => zeroed heading slots.
        self.use_yaw = os.environ.get("EP_USE_YAW", "0") == "1"
        if self.if_depth:
            depth_backbone = DepthOnlyFCBackbone58x87(env.cfg.env.n_proprio, 
                                                    self.policy_cfg["scan_encoder_dims"][-1], 
                                                    self.depth_encoder_cfg["hidden_dims"],
                                                    )
            depth_encoder = RecurrentDepthBackbone(depth_backbone, env.cfg).to(self.device)
            depth_actor = deepcopy(actor_critic.actor)
        else:
            depth_encoder = None
            depth_actor = None
        # self.depth_encoder = depth_encoder
        # self.depth_encoder_optimizer = optim.Adam(self.depth_encoder.parameters(), lr=self.depth_encoder_cfg["learning_rate"])
        # self.depth_encoder_paras = self.depth_encoder_cfg
        # self.depth_encoder_criterion = nn.MSELoss()
        # Create algorithm
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, 
                                  estimator, self.estimator_cfg, 
                                  depth_encoder, self.depth_encoder_cfg, depth_actor,
                                  device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]
        # learn_vision (DAgger: teacher labels, L2 imitation) vs learn_depth_rl
        # (PPO on the depth student against env reward). Both run the camera
        # pipeline; only the objective applied to depth_actor differs.
        self.depth_rl = bool(self.depth_encoder_cfg.get("rl_depth", False))
        # The depth-RL update keeps the teacher's step count (num_mini_batches x
        # num_learning_epochs = 4 x 5 = 20 Adam steps per iteration) and bounds the
        # encoder-replay memory with `rl_replay_chunk` instead: update_depth_rl
        # accumulates gradients over that many samples and steps once per
        # minibatch. So this override only exists to keep the two knobs separate.
        if self.depth_rl:
            rl_nb = self.depth_encoder_cfg.get("rl_num_mini_batches")
            if rl_nb is not None:
                self.alg.num_mini_batches = int(rl_nb)
            print(f"[ppo] depth-RL minibatches = {self.alg.num_mini_batches} "
                  f"x {self.alg.num_learning_epochs} epochs; encoder replay in "
                  f"{self.alg.replay_chunk}-sample chunks, one step per minibatch")

        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
        )

        if self.if_depth:
            self.learn = self.learn_depth_rl if self.depth_rl else self.learn_vision
        else:
            self.learn = self.learn_RL
            
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        

    def _tb_log(self, wandb_dict, step):
        """Mirror the same scalars into TensorBoard.

        wandb_dict mixes plain numbers, single-element tensors (every
        Episode_rew/* entry) and the occasional non-scalar, so normalise
        first and skip anything that is not a finite float.

        The writer is created lazily here rather than only in learn_RL: a
        depth run goes through learn_vision, whose loop never touched the
        writer, so a --use_camera run wrote no TensorBoard data at all.
        """
        if self.log_dir is None:
            return
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        for key, val in wandb_dict.items():
            if isinstance(val, bool):
                continue
            if isinstance(val, torch.Tensor):
                if val.numel() != 1:
                    continue
                val = float(val.item())
            if isinstance(val, (int, float)):
                val = float(val)
                if val == val:  # NaN guard
                    self.writer.add_scalar(key, val, step)
        self.writer.flush()

    @staticmethod
    def _flush_episode_stats(rew_snapshots, len_snapshots, done_masks, rewbuffer, lenbuffer):
        """Drain a rollout's finished-episode reward sums / lengths into the deques
        with a SINGLE host transfer.

        Per step we stash fixed-size (num_envs,) clones of the running reward sum and
        episode length plus the done mask — no GPU `nonzero()` / boolean-index, both
        of which force a per-step CUDA sync. Here we stack them to (T, num_envs),
        move to the host ONCE, and filter the finished episodes on the CPU (NumPy
        boolean indexing, no sync). Row-major flatten preserves step-then-env order,
        so the maxlen-100 deques keep the same recency window as the per-step code.
        Returns the number of finished episodes drained (for lockstep zero-padding of
        the explr/entropy deques)."""
        if len(done_masks) == 0:
            return 0
        rew_mat = torch.stack(rew_snapshots)
        len_mat = torch.stack(len_snapshots)
        done_mat = torch.stack(done_masks).to(rew_mat.dtype)
        packed = torch.stack((rew_mat, len_mat, done_mat)).cpu().numpy()  # one transfer
        done_np = packed[2] > 0
        rews = packed[0][done_np]
        rewbuffer.extend(rews.tolist())
        lenbuffer.extend(packed[1][done_np].tolist())
        return int(rews.shape[0])

    def _distillation_observations(self, obs):
        """Return untouched teacher observations and deploy-like student observations."""
        teacher_obs = obs
        student_obs = obs.clone()
        priv_start = self.alg.num_prop + self.alg.num_scan
        if self.alg.train_with_estimated_states:
            with torch.no_grad():
                priv_states_estimated = self.alg.estimator(obs[:, :self.alg.num_prop])
            student_obs[:, priv_start:
                priv_start + self.alg.priv_states_dim] = priv_states_estimated
        if self.zero_student_velocity_obs:
            velocity_dim = min(3, self.alg.priv_states_dim)
            student_obs[:, priv_start:priv_start + velocity_dim] = 0.0
        return teacher_obs, student_obs

    def learn_RL(self, num_learning_iterations, init_at_random_ep_len=False):
        mean_value_loss = 0.
        mean_surrogate_loss = 0.
        mean_estimator_loss = 0.
        mean_disc_loss = 0.
        mean_disc_acc = 0.
        mean_hist_latent_loss = 0.
        mean_priv_reg_loss = 0. 
        priv_reg_coef = 0.
        entropy_coef = 0.
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        infos = {}
        infos["depth"] = self.env.depth_buffer.clone().to(self.device) if self.if_depth else None
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        rew_explr_buffer = deque(maxlen=100)
        rew_entropy_buffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            hist_encoding = it % self.dagger_update_freq == 0

            # Per-iteration episode bookkeeping, collected ON-GPU each step with NO
            # sync (no nonzero / boolean-index / .cpu()), then drained to the deques
            # in one host transfer after the rollout (see _flush_episode_stats).
            rew_snapshots, len_snapshots, done_masks = [], [], []

            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, infos, hist_encoding)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)  # obs has changed to next_obs !! if done obs has been reset
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    total_rew = self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += total_rew
                        cur_episode_length += 1
                        # Snapshot BEFORE the reset so finished episodes keep their
                        # terminal-step reward. clone() because dones aliases the env's
                        # reset buffer and the sums are reset in-place just below.
                        rew_snapshots.append(cur_reward_sum.clone())
                        len_snapshots.append(cur_episode_length.clone())
                        done_masks.append(dones.clone())
                        # Reset finished envs via mask-multiply (no nonzero/index sync).
                        not_done = (dones == 0).to(cur_reward_sum.dtype)
                        cur_reward_sum *= not_done
                        cur_episode_length *= not_done

                # Single host transfer per iteration. explr/entropy rewards are
                # structurally 0 here, so pad those deques with matching-length zeros
                # to keep the three reward deques in lockstep (their means stay 0).
                if self.log_dir is not None:
                    n_done = self._flush_episode_stats(
                        rew_snapshots, len_snapshots, done_masks, rewbuffer, lenbuffer)
                    rew_explr_buffer.extend([0.0] * n_done)
                    rew_entropy_buffer.extend([0.0] * n_done)

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_disc_loss, mean_disc_acc, mean_priv_reg_loss, priv_reg_coef = self.alg.update()
            if hist_encoding:
                print("Updating dagger...")
                mean_hist_latent_loss = self.alg.update_dagger()
            
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it < 2500:
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            elif it < 5000:
                if it % (2*self.save_interval) == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            else:
                if it % (5*self.save_interval) == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            # Absolute progress for the next save()/resume: `it` indexes the iteration
            # that just finished, so the counter becomes it+1. Deliberately set AFTER
            # the save above, which therefore writes iter == the number in its filename.
            self.current_learning_iteration = it + 1
            ep_infos.clear()
        
        # self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def learn_vision(self, num_learning_iterations, init_at_random_ep_len=False):
        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        obs = self.env.get_observations()
        infos = {}
        infos["depth"] = self.env.get_depth_obs().to(self.device) if self.if_depth else None
        self.alg.depth_encoder.train()
        self.alg.depth_actor.train()

        num_pretrain_iter = 0
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            depth_latent_buffer = []
            scandots_latent_buffer = []
            actions_teacher_buffer = []
            actions_student_buffer = []
            yaw_student_buffer = []
            yaw_teacher_buffer = []
            # On-GPU episode bookkeeping; one host transfer after the rollout (no
            # per-step nonzero/.cpu() sync — same as learn_RL).
            rew_snapshots, len_snapshots, done_masks = [], [], []
            for i in range(self.depth_encoder_cfg["num_steps_per_env"]):
                # Keep the teacher on simulator ground truth while giving the
                # student the estimator-backed explicit state used at deploy.
                teacher_obs, student_obs = self._distillation_observations(obs)

                if infos["depth"] != None:
                    with torch.no_grad():
                        scandots_latent = self.alg.actor_critic.actor.infer_scandots_latent(teacher_obs)
                    scandots_latent_buffer.append(scandots_latent)
                    obs_prop_depth = student_obs[:, :self.env.cfg.env.n_proprio].clone()
                    obs_prop_depth[:, 6:8] = 0
                    # Encoder output. With EP_USE_YAW the head emits 32 latent +
                    # 2 heading dims; the heading is scaled (×1.5, upstream) and
                    # fed into the student's obs[:,6:8] + supervised against the
                    # teacher's commanded heading. Without it, the head emits the
                    # 32-D latent only and the student gets zeroed heading slots.
                    depth_out = self.alg.depth_encoder(infos["depth"].clone(), obs_prop_depth)  # clone is crucial to avoid in-place operation
                    if self.use_yaw:
                        depth_latent = depth_out[:, :-2]
                        yaw = 1.5 * depth_out[:, -2:]
                        yaw_student_buffer.append(yaw)
                        yaw_teacher_buffer.append(teacher_obs[:, 6:8])
                    else:
                        depth_latent = depth_out
                    depth_latent_buffer.append(depth_latent)

                with torch.no_grad():
                    actions_teacher = self.alg.actor_critic.act_inference(
                        teacher_obs, hist_encoding=True, scandots_latent=None)
                    actions_teacher_buffer.append(actions_teacher)

                obs_student = student_obs
                if self.use_yaw:
                    obs_student[:, 6:8] = yaw.detach()
                else:
                    obs_student[:, 6:8] = 0
                actions_student = self.alg.depth_actor(obs_student, hist_encoding=True, scandots_latent=depth_latent)
                actions_student_buffer.append(actions_student)

                # detach actions before feeding the env
                if it < num_pretrain_iter:
                    step_actions = actions_teacher.detach()
                else:
                    step_actions = actions_student.detach()
                # Explore around the executed action (PiW Alg. 1). Applied ONLY
                # here, after actions_student_buffer has already captured the
                # clean output, so the BC target is unchanged and only the state
                # distribution widens.
                if self.dagger_action_noise_std > 0.0:
                    step_actions = step_actions + self.dagger_action_noise_std * torch.randn_like(step_actions)
                obs, privileged_obs, rewards, dones, infos = self.env.step(step_actions)  # obs has changed to next_obs !! if done obs has been reset
                critic_obs = privileged_obs if privileged_obs is not None else obs
                obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        # Snapshot BEFORE reset; clone (dones aliases the env buffer,
                        # sums reset in-place). Mask-multiply resets finished envs
                        # without a nonzero/index sync.
                        rew_snapshots.append(cur_reward_sum.clone())
                        len_snapshots.append(cur_episode_length.clone())
                        done_masks.append(dones.clone())
                        not_done = (dones == 0).to(cur_reward_sum.dtype)
                        cur_reward_sum *= not_done
                        cur_episode_length *= not_done

            # One host transfer per iteration for the episode-stat deques.
            if self.log_dir is not None:
                self._flush_episode_stats(
                    rew_snapshots, len_snapshots, done_masks, rewbuffer, lenbuffer)

            stop = time.time()
            collection_time = stop - start
            start = stop

            scandots_latent_buffer = torch.cat(scandots_latent_buffer, dim=0)
            depth_latent_buffer = torch.cat(depth_latent_buffer, dim=0)
            depth_encoder_loss = 0
            # depth_encoder_loss = self.alg.update_depth_encoder(depth_latent_buffer, scandots_latent_buffer)

            actions_teacher_buffer = torch.cat(actions_teacher_buffer, dim=0)
            actions_student_buffer = torch.cat(actions_student_buffer, dim=0)
            yaw_loss = 0
            if self.use_yaw:
                yaw_student_buffer = torch.cat(yaw_student_buffer, dim=0)
                yaw_teacher_buffer = torch.cat(yaw_teacher_buffer, dim=0)
                depth_actor_loss, yaw_loss = self.alg.update_depth_actor(
                    actions_student_buffer, actions_teacher_buffer,
                    yaw_student_buffer, yaw_teacher_buffer)
            else:
                depth_actor_loss = self.alg.update_depth_actor(actions_student_buffer, actions_teacher_buffer)

            # depth_encoder_loss, depth_actor_loss = self.alg.update_depth_both(depth_latent_buffer, scandots_latent_buffer, actions_student_buffer, actions_teacher_buffer)
            stop = time.time()
            learn_time = stop - start

            self.alg.depth_encoder.detach_hidden_states()

            if self.log_dir is not None:
                self.log_vision(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            # Absolute progress (see learn_RL); set after save() so the checkpoint's
            # iter matches its filename.
            self.current_learning_iteration = it + 1
            ep_infos.clear()
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(tot_iter)))

    def learn_depth_rl(self, num_learning_iterations, init_at_random_ep_len=False):
        """Reinforcement learning on the depth student (PPO, env reward).

        Same observations and same camera pipeline as learn_vision — the policy is
        still depth -> depth_encoder -> latent -> depth_actor — but the objective is
        the PPO surrogate on the environment reward instead of an L2 fit to the
        teacher's action, so `terrain_level` measures what the depth policy can
        actually achieve rather than how well it copies the teacher.

        Trained: depth_actor + its action std (surrogate), the depth ENCODER
        (surrogate: its inputs are stored and the forward is replayed inside the PPO
        graph, so the policy gradient reaches it — the same joint relationship the
        teacher's scan_encoder has with its actor), the privileged critic (value
        loss) and the estimator (regression on the true privileged state). Only the
        teacher actor is frozen: it is the privileged reference and the label source
        for DAgger.
        """
        # Teacher cadence: run the SAME 24 steps/iteration the privileged teacher
        # used, not learn_vision's camera cadence (update_interval*24 = 120). Two
        # reasons: (1) PPO gets 5x more gradient steps per environment step, which is
        # what a surrogate objective needs (DAgger did not — one L2 fit per 120 steps
        # is enough); (2) every step-counted curriculum constant, notably
        # hurdle_clearance_arm_step/ramp_steps = 2000*24 / 4000*24, was written for
        # the 24-step clock and armed 5x too early at 120. learn_vision keeps its own
        # cadence; this only changes the depth-RL path.
        num_steps = self.cfg["num_steps_per_env"]
        self.alg.init_storage(
            self.env.num_envs,
            num_steps,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
        )
        # log_vision derives fps/timesteps from this; it must be the depth cadence.
        self.num_steps_per_env = num_steps

        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        infos = {}

        # The encoder is TRAINED (jointly with depth_actor through the replayed
        # forward in update_depth_rl).
        self.alg.depth_encoder.train()
        self.alg.depth_actor.train()
        # The teacher actor must stay in eval and out of every optimiser; the critic
        # is now TRAINED, so it is switched to train mode explicitly (the module has
        # no dropout/BN today, but this keeps the intent unambiguous).
        self.alg.actor_critic.eval()
        self.alg.actor_critic.critic.train()

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # On-GPU episode bookkeeping; one host transfer after the rollout.
            rew_snapshots, len_snapshots, done_masks = [], [], []

            # Depth is only published every cfg.depth.update_interval steps (4 of 5
            # steps return infos['depth'] = None). Re-read the freshest frame HERE,
            # not once before the loop: env.step() overwrites infos every step, and
            # an out-of-loop fetch left the second iteration starting from None.
            # The hold-last-frame state must also reset per iteration so a replayed
            # encoder input never pairs a stale frame with a fresh hidden state.
            infos["depth"] = self.env.get_depth_obs().to(self.device) if self.if_depth else None
            depth_latent = None
            depth_frame = None      # last frame fed to the encoder (hold-last source)
            depth_prop = None       # the proprio input that went with it
            depth_hidden_in = None  # GRU state BEFORE that frame's forward
            depth_scan_target = None  # teacher scan latent for the same frame
            for i in range(num_steps):
                # Teacher on simulator ground truth, student on the
                # estimator-backed explicit state it sees at deploy.
                teacher_obs, student_obs = self._distillation_observations(obs)

                if infos["depth"] is not None:
                    with torch.no_grad():
                        obs_prop_depth = student_obs[:, :self.env.cfg.env.n_proprio].clone()
                        obs_prop_depth[:, 6:8] = 0
                        # Capture the GRU input state BEFORE the forward: the PPO
                        # update replays this exact step from it (truncated BPTT).
                        hidden_in = self.alg.depth_encoder.hidden_states
                        if hidden_in is None:
                            hidden_in = torch.zeros(
                                1, self.env.num_envs,
                                self.alg.depth_encoder.rnn.hidden_size,
                                device=self.device)
                        depth_out = self.alg.depth_encoder(
                            infos["depth"].clone(), obs_prop_depth)
                        # Teacher anchor for the encoder replay: the frozen
                        # scan_encoder's latent for the SAME frame. It is the stable
                        # label update_depth_rl regresses depth_latent onto, and the
                        # only thing stopping the encoder from drifting with every
                        # actor step (see update_depth_rl). Computed here, inside the
                        # no_grad block, so it costs one MLP forward.
                        depth_scan_target = (
                            self.alg.actor_critic.actor.infer_scandots_latent(
                                teacher_obs))
                    if self.use_yaw:
                        depth_latent = depth_out[:, :-2]
                        yaw = 1.5 * depth_out[:, -2:]
                    else:
                        depth_latent = depth_out
                    # Hold-last sources: reused verbatim on the 4 of 5 steps that
                    # publish no new frame, so the replay reproduces what the actor
                    # actually saw.
                    depth_frame = infos["depth"].clone()
                    depth_prop = obs_prop_depth.detach().clone()
                    depth_hidden_in = hidden_in.detach().clone()

                if depth_latent is None:
                    raise RuntimeError(
                        "learn_depth_rl needs the camera pipeline but "
                        "infos['depth'] is None: run with --use_camera.")

                obs_student = student_obs
                if self.use_yaw:
                    obs_student[:, 6:8] = yaw.detach()
                else:
                    obs_student[:, 6:8] = 0

                actions = self.alg.act_depth(
                    obs_student, critic_obs, depth_latent,
                    depth_images=depth_frame, depth_props=depth_prop,
                    depth_hiddens=depth_hidden_in,
                    scandots_target=depth_scan_target)

                obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                critic_obs = privileged_obs if privileged_obs is not None else obs
                obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                total_rew = self.alg.process_env_step(rewards, dones, infos)

                if self.log_dir is not None:
                    if 'episode' in infos:
                        ep_infos.append(infos['episode'])
                    cur_reward_sum += total_rew
                    cur_episode_length += 1
                    rew_snapshots.append(cur_reward_sum.clone())
                    len_snapshots.append(cur_episode_length.clone())
                    done_masks.append(dones.clone())
                    not_done = (dones == 0).to(cur_reward_sum.dtype)
                    cur_reward_sum *= not_done
                    cur_episode_length *= not_done

            if self.log_dir is not None:
                self._flush_episode_stats(
                    rew_snapshots, len_snapshots, done_masks, rewbuffer, lenbuffer)

            stop = time.time()
            collection_time = stop - start

            # Learning step
            start = stop
            self.alg.compute_returns(critic_obs)

            depth_rl_surrogate_loss, depth_rl_entropy, depth_rl_value_loss, \
                depth_rl_estimator_loss = self.alg.update_depth_rl()
            self.alg.depth_encoder.detach_hidden_states()

            stop = time.time()
            learn_time = stop - start

            # Keys log_vision already expects, filled with the RL equivalents so the
            # PPO run reuses the same TensorBoard/console plumbing.
            #
            # depth_encoder_loss on THIS path is the teacher ANCHOR term, not an L2
            # distillation loss: the encoder is trained by the surrogate plus the
            # anchor. It used to be hard-coded 0.0, which read as "converged" while
            # the encoder was in fact completely unconstrained.
            depth_encoder_loss = getattr(self.alg, 'last_encoder_anchor_loss', 0.0)
            depth_actor_loss = depth_rl_surrogate_loss
            # yaw_loss stays 0.0 here: the heading head has no supervised target on
            # the RL path (it would need the teacher's delta_yaw), so log_vision only
            # prints it when use_yaw is on.
            yaw_loss = 0.0
            depth_rl_lr = getattr(self.alg, 'last_depth_lr', 0.0)
            depth_rl_kl_mean = getattr(self.alg, 'last_depth_kl_mean', float('nan'))
            depth_encoder_grad_norm = getattr(self.alg, 'last_encoder_grad_norm', 0.0)
            depth_hist_latent_loss = getattr(self.alg, 'last_hist_latent_loss', 0.0)
            # Annealed action-BC anchor diagnostics (all zero when rl_bc_coef=0).
            depth_rl_bc_coef = getattr(self.alg, 'last_depth_rl_bc_coef', 0.0)
            depth_bc_loss = getattr(self.alg, 'last_depth_bc_loss', 0.0)
            depth_rl_bc_frozen = getattr(self.alg, 'last_depth_rl_bc_frozen', False)
            depth_std_mean = float(self.alg.depth_actor_std.mean().item())

            if self.log_dir is not None:
                self.log_vision(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            # Absolute progress (see learn_RL). This is what makes a watchdog restart
            # land on a monotonically increasing filename instead of rewriting
            # model_0/100/200 and losing the real position.
            self.current_learning_iteration = it + 1
            ep_infos.clear()
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(tot_iter)))

    def log_vision(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # per_level/* bins are NaN on steps where no env of that row
                # reset; nanmean ignores those empty steps (all-NaN -> NaN, which
                # the writer's NaN guard skips). Same handling as log(): the
                # depth path (learn_vision / learn_depth_rl) lands here, not there.
                if key.startswith('per_level/'):
                    value = (torch.nanmean(infotensor)
                             if not bool(torch.isnan(infotensor).all())
                             else torch.tensor(float('nan'), device=self.device))
                wandb_key = key if (key.startswith('wococo/') or key.startswith('per_level/')) else 'Episode_rew/' + key
                wandb_dict[wandb_key] = value
                # Per-level curves go to TB only; don't spam the console with
                # 10 rows x 4 metrics every iteration.
                if not key.startswith('per_level/'):
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        if 'depth_rl_surrogate_loss' in locs:
            # Depth-RL path: actor_critic is the FROZEN teacher, so its std is a
            # constant 0.60 that says nothing about the student's exploration width
            # (and misreads as "sigma is growing" in the console). Report the
            # student's own sigma here and keep the teacher's on a labelled line.
            # On every other path locs has no depth_rl_* key and this is unchanged.
            mean_std = torch.as_tensor(locs['depth_std_mean'], device=self.device)
        else:
            mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        wandb_dict['Loss_depth/depth_encoder'] = locs['depth_encoder_loss']
        wandb_dict['Loss_depth/depth_actor'] = locs['depth_actor_loss']
        if 'depth_rl_surrogate_loss' in locs:
            # Depth-RL run (learn_depth_rl): report the PPO quantities and the
            # curriculum level the depth policy is actually holding.
            wandb_dict['Loss_depth/rl_surrogate'] = locs['depth_rl_surrogate_loss']
            wandb_dict['Loss_depth/rl_entropy'] = locs['depth_rl_entropy']
            wandb_dict['Loss_depth/rl_value'] = locs['depth_rl_value_loss']
            wandb_dict['Loss_depth/rl_estimator'] = locs['depth_rl_estimator_loss']
            wandb_dict['Loss_depth/rl_learning_rate'] = locs['depth_rl_lr']
            wandb_dict['Loss_depth/rl_kl_mean'] = locs['depth_rl_kl_mean']
            wandb_dict['Loss_depth/rl_encoder_grad'] = locs['depth_encoder_grad_norm']
            wandb_dict['Loss_depth/rl_hist_latent'] = locs['depth_hist_latent_loss']
            wandb_dict['Loss_depth/rl_bc'] = locs['depth_bc_loss']
            wandb_dict['Loss_depth/rl_bc_coef'] = locs['depth_rl_bc_coef']
            # 0/1 float, not bool: _tb_log skips bools outright.
            wandb_dict['Loss_depth/rl_bc_frozen'] = float(locs['depth_rl_bc_frozen'])
            wandb_dict['Policy/depth_action_std'] = locs['depth_std_mean']
            terrain_levels = getattr(self.env, 'terrain_levels', None)
            if terrain_levels is not None:
                wandb_dict['Train/terrain_level_mean'] = float(terrain_levels.float().mean().item())
                wandb_dict['Train/terrain_level_max'] = float(terrain_levels.max().item())
        if self.use_yaw:
            wandb_dict['Loss_depth/yaw'] = locs['yaw_loss']
        wandb_dict['Policy/mean_noise_std'] = mean_std.item()
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
        
        wandb.log(wandb_dict, step=locs['it'])
        self._tb_log(wandb_dict, locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.start_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            # Labels matter here. mean_std is the STUDENT's sigma on the depth-RL path
            # and the teacher's elsewhere; the depth label is anchor-vs-distillation;
            # and a yaw loss does not exist on the RL path, so the line is dropped
            # instead of printing a hard-coded 0.0000 that reads as convergence.
            depth_rl_run = 'depth_rl_surrogate_loss' in locs
            std_label = ('Mean action noise std (student):' if depth_rl_run
                         else 'Mean action noise std:')
            enc_label = ('Depth encoder anchor loss:' if depth_rl_run
                         else 'Depth encoder loss:')
            yaw_line = (f"""{'Yaw loss:':>{pad}} {locs['yaw_loss']:.4f}\n"""
                        if ('yaw_loss' in locs and
                            (not depth_rl_run or getattr(self, 'use_yaw', False)))
                        else '')
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{std_label:>{pad}} {mean_std.item():.3f}\n"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{enc_label:>{pad}} {locs['depth_encoder_loss']:.4f}\n"""
                          f"""{'Depth actor loss:':>{pad}} {locs['depth_actor_loss']:.4f}\n"""
                          + yaw_line
                          + (f"""{'RL surrogate loss:':>{pad}} {locs['depth_rl_surrogate_loss']:.4f}\n"""
                             f"""{'RL entropy:':>{pad}} {locs['depth_rl_entropy']:.3f}\n"""
                             f"""{'RL learning rate:':>{pad}} {locs['depth_rl_lr']:.2e}\n"""
                             f"""{'RL kl_mean:':>{pad}} {locs['depth_rl_kl_mean']:.4f}\n"""
                             f"""{'Encoder grad norm:':>{pad}} {locs['depth_encoder_grad_norm']:.3f}\n"""
                             f"""{'Hist latent loss:':>{pad}} {locs['depth_hist_latent_loss']:.4f}\n"""
                             f"""{'RL BC coef:':>{pad}} {locs['depth_rl_bc_coef']:.3f}\n"""
                             f"""{'RL BC loss:':>{pad}} {locs['depth_bc_loss']:.4f}\n"""
                             f"""{'Depth action std:':>{pad}} {locs['depth_std_mean']:.3f}\n"""
                             f"""{'Terrain level (mean):':>{pad}} {wandb_dict.get('Train/terrain_level_mean', float('nan')):.2f}\n"""
                             if 'depth_rl_surrogate_loss' in locs else ''))
        else:
            log_string = (f"""{'#' * width}\n""")

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        curr_it = locs['it'] - self.start_learning_iteration
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
        mins = eta // 60
        secs = eta % 60
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
        print(log_string)

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # per_level/* bins are NaN on steps where no env of that row
                # reset; nanmean ignores those empty steps (all-NaN -> NaN, which
                # the writer's NaN guard skips). Other keys keep the plain mean.
                if key.startswith('per_level/'):
                    value = (torch.nanmean(infotensor)
                             if not bool(torch.isnan(infotensor).all())
                             else torch.tensor(float('nan'), device=self.device))
                wandb_key = key if (key.startswith('wococo/') or key.startswith('per_level/')) else 'Episode_rew/' + key
                wandb_dict[wandb_key] = value
                # Per-level curves go to TB only; don't spam the console with
                # 10 rows x 4 metrics every iteration.
                if not key.startswith('per_level/'):
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        if 'depth_rl_surrogate_loss' in locs:
            # Depth-RL path: actor_critic is the FROZEN teacher, so its std is a
            # constant 0.60 that says nothing about the student's exploration width
            # (and misreads as "sigma is growing" in the console). Report the
            # student's own sigma here and keep the teacher's on a labelled line.
            # On every other path locs has no depth_rl_* key and this is unchanged.
            mean_std = torch.as_tensor(locs['depth_std_mean'], device=self.device)
        else:
            mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        wandb_dict['Loss/value_function'] = locs['mean_value_loss']
        wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        wandb_dict['Loss/estimator'] = locs['mean_estimator_loss']
        wandb_dict['Loss/hist_latent_loss'] = locs['mean_hist_latent_loss']
        wandb_dict['Loss/priv_reg_loss'] = locs['mean_priv_reg_loss']
        wandb_dict['Loss/priv_ref_lambda'] = locs['priv_reg_coef']
        wandb_dict['Loss/entropy_coef'] = locs['entropy_coef']
        wandb_dict['Loss/learning_rate'] = self.alg.learning_rate
        # KL-schedule diagnostics: the driver of the adaptive lr and how often it
        # saturated at the floor this iteration. Without these the lr trace alone
        # can't distinguish "KL on target" from "KL pinned way above target".
        wandb_dict['Loss/kl_mean'] = getattr(self.alg, 'last_kl_mean', float('nan'))
        wandb_dict['Loss/lr_floor_frac'] = getattr(self.alg, 'last_lr_floor_frac', float('nan'))
        wandb_dict['Loss/discriminator'] = locs['mean_disc_loss']
        wandb_dict['Loss/discriminator_accuracy'] = locs['mean_disc_acc']

        wandb_dict['Policy/mean_noise_std'] = mean_std.item()
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_reward_explr'] = statistics.mean(locs['rew_explr_buffer'])
            wandb_dict['Train/mean_reward_task'] = wandb_dict['Train/mean_reward'] - wandb_dict['Train/mean_reward_explr']
            wandb_dict['Train/mean_reward_entropy'] = statistics.mean(locs['rew_entropy_buffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
            # wandb_dict['Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            # wandb_dict['Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        wandb.log(wandb_dict, step=locs['it'])
        self._tb_log(wandb_dict, locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.start_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Discriminator loss:':>{pad}} {locs['mean_disc_loss']:.4f}\n"""
                          f"""{'Discriminator accuracy:':>{pad}} {locs['mean_disc_acc']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean reward (task):':>{pad}} {statistics.mean(locs['rewbuffer']) - statistics.mean(locs['rew_explr_buffer']):.2f}\n"""
                          f"""{'Mean reward (exploration):':>{pad}} {statistics.mean(locs['rew_explr_buffer']):.2f}\n"""
                          f"""{'Mean reward (entropy):':>{pad}} {statistics.mean(locs['rew_entropy_buffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Estimator loss:':>{pad}} {locs['mean_estimator_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        curr_it = locs['it'] - self.start_learning_iteration
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
        mins = eta // 60
        secs = eta % 60
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
        print(log_string)

    def save(self, path, infos=None):
        state_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'estimator_state_dict': self.alg.estimator.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }
        if self.if_depth:
            state_dict['depth_encoder_state_dict'] = self.alg.depth_encoder.state_dict()
            state_dict['depth_actor_state_dict'] = self.alg.depth_actor.state_dict()
            # Only meaningful once a depth-RL rollout has seeded it (learn_vision
            # never touches it); a fresh key so old checkpoints load unchanged.
            if self.alg.depth_std_initialised:
                state_dict['depth_actor_std'] = self.alg.depth_actor_std.detach().cpu()
                state_dict['depth_rl_optimizer_state_dict'] = self.alg.depth_rl_optimizer.state_dict()
            # Teacher-style critic/estimator training on the depth chain; saved under
            # fresh keys so older checkpoints (incl. released model_10000) still load.
            state_dict['critic_optimizer_state_dict'] = self.alg.critic_optimizer.state_dict()
            # Same treatment for the history-encoder regression head: it has its own
            # optimiser, so its Adam moments need their own key or a resumed run
            # restarts the head from zero momentum each time.
            if (self.alg.depth_std_initialised
                    and getattr(self.alg, 'depth_hist_encoder_optimizer', None) is not None):
                state_dict['depth_hist_encoder_optimizer_state_dict'] = \
                    self.alg.depth_hist_encoder_optimizer.state_dict()
        torch.save(state_dict, path)

    def load(self, path, load_optimizer=True):
        print("*" * 80)
        print("Loading model from {}...".format(path))
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        self.alg.estimator.load_state_dict(loaded_dict['estimator_state_dict'])
        if self.if_depth:
            if 'depth_encoder_state_dict' not in loaded_dict:
                warnings.warn("'depth_encoder_state_dict' key does not exist, not loading depth encoder...")
            else:
                print("Saved depth encoder detected, loading...")
                self.alg.depth_encoder.load_state_dict(loaded_dict['depth_encoder_state_dict'])
            if 'depth_actor_state_dict' in loaded_dict:
                print("Saved depth actor detected, loading...")
                self.alg.depth_actor.load_state_dict(loaded_dict['depth_actor_state_dict'])
            else:
                print("No saved depth actor, Copying actor critic actor to depth actor...")
                self.alg.depth_actor.load_state_dict(self.alg.actor_critic.actor.state_dict())
            # Resume a depth-RL run (or skip and let act_depth seed from
            # actor_critic.std on a checkpoint that only ever ran DAgger).
            if 'depth_actor_std' in loaded_dict:
                print("Saved depth-RL action std detected, loading...")
                self.alg.depth_actor_std.data.copy_(
                    loaded_dict['depth_actor_std'].to(self.alg.depth_actor_std.device))
                self.alg.depth_std_initialised = True
                if load_optimizer and 'depth_rl_optimizer_state_dict' in loaded_dict:
                    self.alg.depth_rl_optimizer.load_state_dict(
                        loaded_dict['depth_rl_optimizer_state_dict'])
            if load_optimizer and 'critic_optimizer_state_dict' in loaded_dict:
                self.alg.critic_optimizer.load_state_dict(
                    loaded_dict['critic_optimizer_state_dict'])
            if (load_optimizer
                    and getattr(self.alg, 'depth_hist_encoder_optimizer', None) is not None
                    and 'depth_hist_encoder_optimizer_state_dict' in loaded_dict):
                self.alg.depth_hist_encoder_optimizer.load_state_dict(
                    loaded_dict['depth_hist_encoder_optimizer_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        # Restore the ABSOLUTE iteration counter. It drives three things at once, and
        # all three were silently wrong while this line was commented out:
        #   * the TensorBoard x-axis -- _tb_log(wandb_dict, locs['it']) takes the
        #     loop variable, so every resumed segment rewrote steps 0..N and the
        #     merged curve overlapped instead of continuing;
        #   * the checkpoint filename -- save() is called with 'model_{it}', so a
        #     resume restarted the numbering at 0 and a watchdog that picks the
        #     newest checkpoint by the LARGEST filename then mis-read the progress;
        #   * save()'s 'iter' field, which is read right here and was therefore
        #     always 0 for every checkpoint this repo has ever written.
        # Checkpoints written before this change store iter=0 (the counter was never
        # incremented on the old path), so a released warm start such as
        # student/model_10000.pt still legitimately starts the counter at 0.
        self.current_learning_iteration = int(loaded_dict.get('iter', 0))
        print(f"[runner] resuming at learning iteration {self.current_learning_iteration}")
        print("*" * 80)
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
    
    def get_depth_actor_inference_policy(self, device=None):
        self.alg.depth_actor.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.depth_actor.to(device)
        return self.alg.depth_actor
    
    def get_actor_critic(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic
    
    def get_estimator_inference_policy(self, device=None):
        self.alg.estimator.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.estimator.to(device)
        return self.alg.estimator.inference

    def get_depth_encoder_inference_policy(self, device=None):
        self.alg.depth_encoder.eval()
        if device is not None:
            self.alg.depth_encoder.to(device)
        return self.alg.depth_encoder
    
    def get_disc_inference_policy(self, device=None):
        self.alg.discriminator.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.discriminator.to(device)
        return self.alg.discriminator.inference
