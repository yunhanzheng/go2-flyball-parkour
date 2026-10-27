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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from rsl_rl.modules import ActorCriticRMA
from rsl_rl.storage import RolloutStorage
import wandb
from rsl_rl.utils import unpad_trajectories


class RMS(object):
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape, device=device)
        self.S = torch.ones(shape, device=device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs + (delta**2) * self.n * bs / (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S

class PPO:
    actor_critic: ActorCriticRMA
    def __init__(self,
                 actor_critic,
                 estimator,
                 estimator_paras,
                 depth_encoder,
                 depth_encoder_paras,
                 depth_actor,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 min_learning_rate=1e-5,
                 max_learning_rate=1e-2,
                 device='cpu',
                 dagger_update_freq=20,
                 priv_reg_coef_schedual = [0, 0, 0],
                 **kwargs
                 ):

        
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        # Clamps of the KL-adaptive schedule below. The floor matters on warm
        # resumes: the KL check runs PER MINIBATCH (num_learning_epochs *
        # num_mini_batches times per iteration), so a stale value function or a
        # mid-run change to the termination/reward set drives kl_mean past
        # 2*desired_kl on every minibatch and the lr saturates at the floor and
        # stays there. Raising the floor trades trust-region safety for the
        # ability to keep making progress in that regime.
        self.min_learning_rate = float(min_learning_rate)
        self.max_learning_rate = float(max_learning_rate)
        if self.min_learning_rate != 1e-5:
            print(f"[ppo] adaptive-LR floor raised to {self.min_learning_rate:.2e} "
                  f"(default 1.00e-05), ceiling {self.max_learning_rate:.2e}")

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Adaptation
        self.hist_encoder_optimizer = optim.Adam(self.actor_critic.actor.history_encoder.parameters(), lr=learning_rate)
        self.priv_reg_coef_schedual = priv_reg_coef_schedual
        self.counter = 0

        # Estimator
        self.estimator = estimator
        self.priv_states_dim = estimator_paras["priv_states_dim"]
        self.num_prop = estimator_paras["num_prop"]
        self.num_scan = estimator_paras["num_scan"]
        self.estimator_optimizer = optim.Adam(self.estimator.parameters(), lr=estimator_paras["learning_rate"])
        self.train_with_estimated_states = estimator_paras["train_with_estimated_states"]

        # Depth encoder
        self.if_depth = depth_encoder != None
        # Single-stage "teacher + student merged": the depth encoder is a TRAINABLE part
        # of the policy (PPO gradients through the actor + an auxiliary scan-latent
        # regression), not a DAgger/frozen target. Off by default so the existing
        # two-stage distillation path is unchanged byte for byte.
        self.train_depth_ppo = bool(self.if_depth and depth_encoder_paras.get("train_with_ppo", False))
        self.depth_aux_scan_coef = float(depth_encoder_paras.get("aux_scan_coef", 0.0))
        # Deploy-side observation knobs, mirrored from the distillation runner: the
        # camera policy must be optimised on the observation it will actually see.
        self.zero_heading_obs = bool(depth_encoder_paras.get("zero_heading_obs", True))
        self.zero_student_velocity_obs = bool(depth_encoder_paras.get("zero_student_velocity_obs", False))
        # Encoder lr / policy lr, used to rescale the second param group whenever the
        # KL-adaptive schedule moves the policy lr.
        self.depth_encoder_lr_ratio = 1.0
        self.last_depth_aux_loss = float('nan')
        if self.if_depth:
            self.depth_encoder = depth_encoder
            self.depth_encoder_optimizer = optim.Adam(self.depth_encoder.parameters(), lr=depth_encoder_paras["learning_rate"])
            self.depth_encoder_paras = depth_encoder_paras
            if self.train_depth_ppo:
                # The encoder joins the PPO optimizer as a SECOND parameter group with
                # its own base lr. There is no separate depth actor: actor_critic.actor
                # IS the depth actor (driven through scandots_latent), so nothing is
                # cloned and nothing is distilled.
                self.depth_encoder_lr_ratio = float(depth_encoder_paras["learning_rate"]) / float(learning_rate)
                self.optimizer = optim.Adam([
                    {'params': list(self.actor_critic.parameters()), 'lr': learning_rate},
                    {'params': list(self.depth_encoder.parameters()),
                     'lr': float(depth_encoder_paras["learning_rate"])},
                ], lr=learning_rate)
                self.depth_encoder_optimizer = self.optimizer
                print(f"[ppo] depth PPO: encoder in the PPO optimizer (policy lr "
                      f"{learning_rate:.1e}, encoder lr {depth_encoder_paras['learning_rate']:.1e}), "
                      f"aux scan coef {self.depth_aux_scan_coef}")
            else:
                self.depth_actor = depth_actor
                self.depth_actor_optimizer = optim.Adam([*self.depth_actor.parameters(), *self.depth_encoder.parameters()], lr=depth_encoder_paras["learning_rate"])

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape,  critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def depth_actor_obs(self, obs):
        """The observation the DEPLOYED depth actor sees.

        Mirrors OnPolicyRunner._distillation_observations plus the heading blanking
        that play.py applies before handing the frame to the encoder/actor:

          * privileged explicit state -> the estimator's prediction (the robot has no
            vicon, so the deploy policy must be trained on the estimate),
          * obs[:, 6:8] -> 0, the two command slots the camera policy has to infer
            (direction-to-goal) from depth instead of reading off the command vector.

        Used for BOTH the rollout action and the PPO recompute, so the optimised
        distribution is the one that will actually run on hardware.
        """
        out = obs.clone()
        if self.train_with_estimated_states:
            with torch.no_grad():
                out[:, self.num_prop + self.num_scan:
                       self.num_prop + self.num_scan + self.priv_states_dim] = \
                    self.estimator(out[:, :self.num_prop])
        if self.zero_student_velocity_obs:
            velocity_dim = min(3, self.priv_states_dim)
            out[:, self.num_prop + self.num_scan:
                   self.num_prop + self.num_scan + velocity_dim] = 0.0
        if self.zero_heading_obs:
            out[:, 6:8] = 0.0
        return out

    def _depth_latent_batch(self, depth_batch, depth_hidden_batch, obs_batch):
        """Replay the depth encoder for a minibatch at its ROLLOUT-time GRU state.

        The rollout caches (frame, hidden-before) at every 10 Hz encoder tick and
        reuses that pair on the intermediate 50 Hz steps, so replaying the pair
        reproduces the depth latent the action was taken with (up to the weights that
        have moved since) while keeping the encoder cadence identical to deployment.
        Gradients flow through the CNN / combination MLP / output head; the recurrent
        path is truncated at the stored state (BPTT length 1, as at rollout).

        Returns (latent, aux_loss_or_None).
        """
        enc = self.depth_encoder
        prop = obs_batch[:, :self.num_prop]
        if self.zero_heading_obs:
            prop = prop.clone()
            prop[:, 6:8] = 0.0
        prev_hidden = enc.hidden_states
        enc.hidden_states = depth_hidden_batch.unsqueeze(0)
        latent = enc(depth_batch, prop)
        enc.hidden_states = prev_hidden

        aux_loss = None
        if self.depth_aux_scan_coef > 0.0 and self.actor_critic.actor.if_scan_encode:
            # Target is the actor's OWN height-scan latent, stop-grad: this is not
            # distillation of a teacher's actions, only a frame-of-reference anchor that
            # keeps the camera bottleneck where the privileged actor expects it.
            with torch.no_grad():
                target = self.actor_critic.actor.infer_scandots_latent(obs_batch)
            aux_loss = torch.nn.functional.mse_loss(
                latent[:, :target.shape[1]], target)
        return latent, aux_loss

    def act(self, obs, critic_obs, info, hist_encoding=False, depth_latent=None,
            depth_img=None, depth_hidden=None):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values, use proprio to compute estimated priv_states then actions, but store true priv_states
        if depth_latent is not None:
            # Depth PPO: the actor is driven by the camera latent in place of the height
            # scan, on the deployable observation built by depth_actor_obs().
            self.transition.actions = self.actor_critic.act(
                self.depth_actor_obs(obs), hist_encoding,
                scandots_latent=depth_latent).detach()
        elif self.train_with_estimated_states:
            obs_est = obs.clone()
            priv_states_estimated = self.estimator(obs_est[:, :self.num_prop])
            obs_est[:, self.num_prop+self.num_scan:self.num_prop+self.num_scan+self.priv_states_dim] = priv_states_estimated
            self.transition.actions = self.actor_critic.act(obs_est, hist_encoding).detach()
        else:
            self.transition.actions = self.actor_critic.act(obs, hist_encoding).detach()

        # Recorded on EVERY transition (the tick pair is reused between ticks), so the
        # storage carries one replayable encoder input per stored step.
        self.transition.depth = depth_img
        self.transition.depth_hidden = depth_hidden

        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs

        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        rewards_total = rewards.clone()

        self.transition.rewards = rewards_total.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

        return rewards_total
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
    

    def update(self):
        # Accumulate the per-minibatch losses as DETACHED GPU scalars and sync to the
        # host ONCE after the update loop — was 4 .item() calls (each a CUDA sync) per
        # minibatch, i.e. 4 * num_learning_epochs * num_mini_batches syncs/iteration.
        mean_value_loss = torch.zeros((), device=self.device)
        mean_surrogate_loss = torch.zeros((), device=self.device)
        mean_estimator_loss = torch.zeros((), device=self.device)
        mean_priv_reg_loss = torch.zeros((), device=self.device)
        mean_depth_aux_loss = torch.zeros((), device=self.device)
        mean_discriminator_loss = 0   # no per-minibatch value; stays Python 0
        mean_discriminator_acc = 0
        # Diagnostics for the KL-adaptive schedule (logged, not used for control):
        # the mean per-minibatch KL that drives the lr, and the fraction of
        # minibatches whose adapted lr ended up pinned at the floor. kl_mean is
        # already a host float below, so this costs no extra CUDA sync.
        kl_sum, kl_n, lr_floor_hits = 0.0, 0, 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, depth_batch, depth_hidden_batch in generator:

                depth_aux_loss = None
                if depth_batch is not None:
                    # Depth PPO: rebuild the camera latent from the cached
                    # (frame, hidden-before) pair and feed it as the scan slot, so the
                    # distribution being clipped is the deployed one. hist_encoding is
                    # always on here — the deploy path has no privileged latent.
                    depth_latent, depth_aux_loss = self._depth_latent_batch(
                        depth_batch, depth_hidden_batch, obs_batch)
                    self.actor_critic.act(self.depth_actor_obs(obs_batch), hist_encoding=True,
                                          scandots_latent=depth_latent)
                else:
                    self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]) # match distribution dimension

                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy
                
                # Adaptation module update
                priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
                with torch.inference_mode():
                    hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_batch)
                priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(p=2, dim=1).mean()
                priv_reg_stage = min(max((self.counter - self.priv_reg_coef_schedual[2]), 0) / self.priv_reg_coef_schedual[3], 1)
                priv_reg_coef = priv_reg_stage * (self.priv_reg_coef_schedual[1] - self.priv_reg_coef_schedual[0]) + self.priv_reg_coef_schedual[0]

                # Estimator
                priv_states_predicted = self.estimator(obs_batch[:, :self.num_prop])  # obs in batch is with true priv_states
                estimator_loss = (priv_states_predicted - obs_batch[:, self.num_prop+self.num_scan:self.num_prop+self.num_scan+self.priv_states_dim]).pow(2).mean()
                self.estimator_optimizer.zero_grad()
                estimator_loss.backward()
                nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
                self.estimator_optimizer.step()
                
                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        # Pull kl_mean to the host ONCE (.item()) so the comparisons
                        # below are Python-float compares, not tensor compares — the
                        # latter each force their own CUDA sync (up to 3/minibatch via
                        # the if/elif-and). The remaining single sync is inherent: the
                        # adapted LR is a Python float consumed by this same minibatch's
                        # optimizer.step(), so it must be resolved before the step.
                        kl_mean = torch.mean(kl).item()

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(self.min_learning_rate, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(self.max_learning_rate, self.learning_rate * 1.5)
                        
                        kl_sum += kl_mean
                        kl_n += 1
                        lr_floor_hits += int(self.learning_rate <= self.min_learning_rate * 1.0001)

                        for param_group in self.optimizer.param_groups:
                            # Group 0 is the policy, group 1 (depth PPO only) the encoder;
                            # the encoder keeps its own lr scale relative to the policy.
                            ratio = 1.0 if param_group is self.optimizer.param_groups[0] \
                                else self.depth_encoder_lr_ratio
                            param_group['lr'] = self.learning_rate * ratio


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + \
                       self.value_loss_coef * value_loss - \
                       self.entropy_coef * entropy_batch.mean() + \
                       priv_reg_coef * priv_reg_loss
                if depth_aux_loss is not None:
                    loss = loss + self.depth_aux_scan_coef * depth_aux_loss
                # loss = self.teacher_alpha * imitation_loss + (1 - self.teacher_alpha) * loss

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                # Both parameter groups (policy, and the encoder under depth PPO).
                nn.utils.clip_grad_norm_(
                    [p for group in self.optimizer.param_groups for p in group['params']],
                    self.max_grad_norm)
                self.optimizer.step()

                # Detached GPU accumulation — no host sync inside the loop.
                mean_value_loss += value_loss.detach()
                mean_surrogate_loss += surrogate_loss.detach()
                mean_estimator_loss += estimator_loss.detach()
                mean_priv_reg_loss += priv_reg_loss.detach()
                if depth_aux_loss is not None:
                    mean_depth_aux_loss = mean_depth_aux_loss + depth_aux_loss.detach()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        # Single host sync: stack the four GPU-accumulated means and pull them over in
        # one .tolist() (was four sequential .item() calls = four sync points).
        mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_priv_reg_loss, mean_depth_aux_loss = (
            torch.stack([mean_value_loss, mean_surrogate_loss,
                         mean_estimator_loss, mean_priv_reg_loss,
                         mean_depth_aux_loss]) / num_updates).tolist()
        # Surfaced for the runner's log line (kept off the return tuple so the
        # signature and every call site stay untouched).
        self.last_depth_aux_loss = mean_depth_aux_loss
        mean_discriminator_loss /= num_updates
        mean_discriminator_acc /= num_updates
        # Read by the runner for logging (kept off the return tuple so the
        # signature and every call site stay untouched).
        self.last_kl_mean = (kl_sum / kl_n) if kl_n else float('nan')
        self.last_lr_floor_frac = (lr_floor_hits / kl_n) if kl_n else float('nan')
        self.storage.clear()
        self.update_counter()
        return mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_discriminator_loss, mean_discriminator_acc, mean_priv_reg_loss, priv_reg_coef

    def update_dagger(self):
        mean_hist_latent_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, _depth_batch, _depth_hidden_batch in generator:
                with torch.inference_mode():
                    self.actor_critic.act(obs_batch, hist_encoding=True, masks=masks_batch, hidden_states=hid_states_batch[0])

                # Adaptation module update
                with torch.inference_mode():
                    priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
                hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_batch)
                hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()
                self.hist_encoder_optimizer.zero_grad()
                hist_latent_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.actor.history_encoder.parameters(), self.max_grad_norm)
                self.hist_encoder_optimizer.step()
                
                mean_hist_latent_loss += hist_latent_loss.item()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_hist_latent_loss /= num_updates
        self.storage.clear()
        self.update_counter()
        return mean_hist_latent_loss

    def update_depth_encoder(self, depth_latent_batch, scandots_latent_batch):
        # Depth encoder ditillation
        if self.if_depth:
            # TODO: needs to save hidden states
            depth_encoder_loss = (scandots_latent_batch.detach() - depth_latent_batch).norm(p=2, dim=1).mean()

            self.depth_encoder_optimizer.zero_grad()
            depth_encoder_loss.backward()
            nn.utils.clip_grad_norm_(self.depth_encoder.parameters(), self.max_grad_norm)
            self.depth_encoder_optimizer.step()
            return depth_encoder_loss.item()
    
    def update_depth_actor(self, actions_student_batch, actions_teacher_batch,
                           yaw_student_batch=None, yaw_teacher_batch=None):
        if self.if_depth:
            # Action behaviour-cloning term: student matches teacher's action.
            depth_actor_loss = (actions_teacher_batch.detach() - actions_student_batch).norm(p=2, dim=1).mean()

            loss = depth_actor_loss

            # Optional yaw-prediction term (EP_USE_YAW=1): the encoder's heading
            # output is regressed against the teacher's commanded heading
            # (obs[:,6:8]). One combined backward + step over the shared
            # depth_image -> encoder -> latent -> actor graph.
            yaw_loss = None
            if yaw_student_batch is not None and yaw_teacher_batch is not None:
                yaw_loss = (yaw_teacher_batch.detach() - yaw_student_batch).norm(p=2, dim=1).mean()
                loss = loss + yaw_loss

            self.depth_actor_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.depth_actor.parameters(), self.max_grad_norm)
            self.depth_actor_optimizer.step()
            if yaw_loss is not None:
                return depth_actor_loss.item(), yaw_loss.item()
            return depth_actor_loss.item()
    
    def update_depth_both(self, depth_latent_batch, scandots_latent_batch, actions_student_batch, actions_teacher_batch):
        if self.if_depth:
            depth_encoder_loss = (scandots_latent_batch.detach() - depth_latent_batch).norm(p=2, dim=1).mean()
            depth_actor_loss = (actions_teacher_batch.detach() - actions_student_batch).norm(p=2, dim=1).mean()

            depth_loss = depth_encoder_loss + depth_actor_loss

            self.depth_actor_optimizer.zero_grad()
            depth_loss.backward()
            nn.utils.clip_grad_norm_([*self.depth_actor.parameters(), *self.depth_encoder.parameters()], self.max_grad_norm)
            self.depth_actor_optimizer.step()
            return depth_encoder_loss.item(), depth_actor_loss.item()
    
    def update_counter(self):
        self.counter += 1
    
    def compute_apt_reward(self, source, target):

        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
        # sim_matrix = torch.norm(source[:, None, ::2].view(b1, 1, -1) - target[None, :, ::2].view(1, b2, -1), dim=-1, p=2)
        # sim_matrix = torch.norm(source[:, None, :2].view(b1, 1, -1) - target[None, :, :2].view(1, b2, -1), dim=-1, p=2)
        sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1), dim=-1, p=2)

        reward, _ = sim_matrix.topk(self.knn_k, dim=1, largest=False, sorted=True)  # (b1, k)

        if not self.knn_avg:  # only keep k-th nearest neighbor
            reward = reward[:, -1]
            reward = reward.reshape(-1, 1)  # (b1, 1)
            if self.rms:
                moving_mean, moving_std = self.disc_state_rms(reward)
                reward = reward / moving_std
            reward = torch.clamp(reward - self.knn_clip, 0)  # (b1, )
        else:  # average over all k nearest neighbors
            reward = reward.reshape(-1, 1)  # (b1 * k, 1)
            if self.rms:
                moving_mean, moving_std = self.disc_state_rms(reward)
                reward = reward / moving_std
            reward = torch.clamp(reward - self.knn_clip, 0)
            reward = reward.reshape((b1, self.knn_k))  # (b1, k)
            reward = reward.mean(dim=1)  # (b1,)
        reward = torch.log(reward + 1.0)
        return reward