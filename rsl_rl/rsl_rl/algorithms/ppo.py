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
        if self.if_depth:
            self.depth_encoder = depth_encoder
            self.depth_encoder_optimizer = optim.Adam(self.depth_encoder.parameters(), lr=depth_encoder_paras["learning_rate"])
            self.depth_encoder_paras = depth_encoder_paras
            self.depth_actor = depth_actor
            self.depth_actor_optimizer = optim.Adam([*self.depth_actor.parameters(), *self.depth_encoder.parameters()], lr=depth_encoder_paras["learning_rate"])
            # Depth-RL path (runner.learn_depth_rl): the student actor is optimised
            # with the PPO surrogate against env reward instead of the DAgger L2
            # imitation target. Actor has no mean/std head of its own, so PPO needs
            # an explicit state-independent sigma; it is seeded from
            # actor_critic.std on the first rollout step (see _init_depth_std) so a
            # resumed student explores at the teacher's calibrated width, and it is
            # saved under its own key since it is not part of depth_actor.
            self.depth_rl_mode = bool(depth_encoder_paras.get("rl_depth", False))
            self.depth_actor_std = nn.Parameter(
                torch.ones(self.actor_critic.actor.num_actions, device=self.device))
            self.depth_std_initialised = False
            # ARM-A control switches. Both are counters to assumptions this class used
            # to state as facts, and both were measured false over long4's 217 iters.
            #
            # rl_freeze_std: sigma was believed immovable at rl_entropy_coef=0 ("the
            # surrogate's gradient w.r.t. sigma is zero in expectation"). It is not; it
            # drifts anyway, because ratio = exp(logpi - logpi_old) still depends on
            # sigma on every unclipped sample: measured 0.05006 -> 0.08760 (+75%) in
            # 217 iterations, while rl_init_std only fill_()s once at start. That drift
            # silently widens the trust region -- kl_target is calibrated as
            # 6*step_ratio^2 at sigma 0.05, and KL = 6*(dmu/sigma)^2, so the SAME dmu
            # measures (0.05/0.088)^2 = 0.32x as much KL once sigma has drifted, which
            # lets each later step grow ~3x. Frozen, sigma is a plain constant: no
            # graph edge, no optimiser membership, no drift.
            self.depth_rl_freeze_std = bool(
                depth_encoder_paras.get("rl_freeze_std", False))
            # rl_adaptive_lr: the KL schedule cannot help once it sits on its floor.
            # long4 measured 125/197 late iterations at 1e-5 with a 2.43 median kl
            # (max 24.3), i.e. it was already at the bottom and still overshooting, so
            # all the controller did was add step-size noise. False pins the lr at
            # rl_learning_rate while KEEPING the kl diagnostic in the log.
            self.depth_rl_adaptive_lr = bool(
                depth_encoder_paras.get("rl_adaptive_lr", True))
            depth_rl_lr = float(depth_encoder_paras.get("rl_learning_rate", 1.e-4))
            # Entropy bonus is a separate knob on the depth-RL objective. cfg's
            # entropy_coef (0.01) maximises entropy, i.e. pushes sigma UP, which is
            # the opposite of what a deploy-time-deterministic student wants; the
            # Gaussian sigma has no zero-cost way to shrink against it.
            self.depth_rl_entropy_coef = float(
                depth_encoder_paras.get("rl_entropy_coef", self.entropy_coef))
            self.depth_rl_learning_rate = depth_rl_lr
            self.depth_rl_lr_min = float(depth_encoder_paras.get("rl_min_learning_rate", depth_rl_lr / 10.))
            self.depth_rl_lr_max = float(depth_encoder_paras.get("rl_max_learning_rate", depth_rl_lr * 10.))
            # Trust region of the depth-RL update, as its own knob. cfg's desired_kl is
            # only comparable at the SAME sigma: KL = 6*dmu^2/sigma^2, so an lr that
            # lands on 0.01 at the teacher's 0.598 produces ~143x more at 0.05. Falls
            # back to cfg's value when the key is absent.
            self.depth_rl_desired_kl = depth_encoder_paras.get(
                "rl_desired_kl", self.desired_kl)
            # Trust-region target expressed WITHOUT the sigma scale. For this Gaussian
            # KL = 6 * (dmu_rms/sigma)^2, so a fixed desired_kl silently rescales the
            # allowed policy step whenever sigma changes: 0.01 permits dmu_rms = 0.024
            # rad at sigma 0.6 but only 0.002 rad at sigma 0.05 -- the same number
            # means a 12x smaller step, the adaptive schedule hits its floor and never
            # comes back (measured: RL learning rate = 1.00e-05 on 96 of 96 iterations,
            # i.e. the depth actor never moved). rl_step_ratio states the target in the
            # only units that survive a sigma change: RMS action-mean displacement as a
            # fraction of sigma. 0.05 reproduces desired_kl = 0.01 at sigma ~0.598.
            _step_ratio = depth_encoder_paras.get("rl_step_ratio", None)
            self.depth_rl_step_ratio = (
                None if _step_ratio is None else float(_step_ratio))
            self.depth_rl_kl_target = (
                self.depth_rl_desired_kl if self.depth_rl_step_ratio is None
                else 6.0 * self.depth_rl_step_ratio ** 2)
            # Hard ceiling on the action std, mirroring the clamp ActorCritic applies
            # to actor_critic.std in update_distribution (cfg.policy.max_noise_std).
            # depth_actor_std is a PPO-owned parameter outside that module, so nothing
            # else bounds it while the entropy bonus pushes it up.
            self.depth_rl_max_noise_std = depth_encoder_paras.get("rl_max_noise_std", None)
            if self.depth_rl_max_noise_std is not None:
                self.depth_rl_max_noise_std = float(self.depth_rl_max_noise_std)
            # The encoder is optimised JOINTLY with the actor, the way the teacher's
            # scan_encoder is: update_depth_rl replays depth_encoder.forward_step on
            # the stored rollout inputs, so the PPO surrogate's gradient flows into
            # the encoder. It gets its own (smaller) lr because the signal is now a
            # noisy policy gradient instead of a clean L2 regression.
            self.depth_encoder_rl_lr = float(
                depth_encoder_paras.get("rl_encoder_learning_rate", 3.e-5))
            # Teacher anchor on the encoder replay. The encoder is optimised jointly
            # with the actor, but a policy-gradient signal alone leaves it free to
            # drift: nothing pins depth_latent to the teacher's scan latent, and the
            # measured encoder grad norm is ~9. The teacher's scan_encoder is frozen
            # and maps the SAME privileged scan to a 32-D latent (scan_encoder_dims
            # ends at 32, matching the no-yaw depth head), so it is a stable label for
            # this head -- exactly the DAgger objective learn_vision trains with, and
            # the term the commented-out update_depth_encoder call used to provide.
            # Coef 0 restores the anchor-free behaviour.
            self.depth_encoder_anchor_coef = float(
                depth_encoder_paras.get("rl_encoder_anchor_coef", 1.0))
            # Annealed ACTION-level BC anchor (the stage-B hybrid: RL reward with an
            # imitation term that decays to zero). The target is the frozen teacher's
            # action mean recomputed inside update_depth_rl from the stored obs plus
            # the true priv slice, so it needs no rollout-side or storage changes.
            # rl_bc_coef = 0.0 (default) keeps the pure-RL objective byte-identical.
            # The anneal clock counts only iterations whose value loss is within
            # rl_bc_value_guard of its recent EMA: a diverging critic holds the
            # anchor instead of letting it anneal away mid-divergence (the warm-start
            # critic lesson -- value loss climbing above its recent level is the
            # earliest visible symptom, well before the advantage is polluted).
            self.depth_rl_bc_coef_start = float(
                depth_encoder_paras.get("rl_bc_coef", 0.0))
            self.depth_rl_bc_coef_end = float(
                depth_encoder_paras.get("rl_bc_coef_end", 0.0))
            self.depth_rl_bc_anneal_iters = int(
                depth_encoder_paras.get("rl_bc_anneal_iters", 1500))
            _bc_guard = depth_encoder_paras.get("rl_bc_value_guard", 2.0)
            self.depth_rl_bc_value_guard = (
                None if _bc_guard is None or float(_bc_guard) <= 0.0
                else float(_bc_guard))
            # Anneal state: the clock (healthy iterations), the current weight, and
            # the value-loss EMA the guard compares against (tracks the healthy
            # baseline only -- it is NOT updated on a frozen iteration, so recovery
            # is detected against the pre-divergence level).
            self.depth_rl_bc_coef = 0.0
            self._depth_rl_bc_step = 0
            self._depth_rl_bc_value_ema = None
            self.last_depth_rl_bc_coef = 0.0
            self.last_depth_bc_loss = 0.0
            self.last_depth_rl_bc_frozen = False
            # Gradient-accumulation width for the encoder replay. update_depth_rl
            # walks each minibatch in slices of this many samples, so the conv
            # activations stay bounded (~1 GB at 512) while the step still sees the
            # WHOLE minibatch. Decoupling the two is what lets num_mini_batches
            # stay at the teacher's 4 instead of being inflated 10x for memory.
            self.replay_chunk = int(depth_encoder_paras.get("rl_replay_chunk", 512))
            self.replay_chunk = max(1, self.replay_chunk)
            # History-encoder supervision: the depth-RL port of update_dagger().
            # depth_actor runs with hist_encoding=True, so history_encoder IS the
            # privileged-latent head at deployment, and the surrogate only trains it
            # through the action output -- no direct incentive to actually recover
            # the privileged latent. It gets its own optimiser and its own lr, and is
            # EXCLUDED from depth_rl_optimizer below: the teacher's arrangement lets
            # both self.optimizer and hist_encoder_optimizer own the same parameters,
            # which applies every history-encoder gradient twice.
            # The privileged encoder is the FROZEN supervision TARGET for the history
            # encoder: priv_encoder(29-D true privileged latent) is the label,
            # history_encoder(10 x 53 proprio history) is the learner. Nothing on this
            # path ever feeds priv_encoder -- _depth_distribution hard-codes
            # hist_encoding=True, so Actor.forward takes the history-encoder branch, and
            # the only priv_encoder call sits inside torch.inference_mode() -- hence its
            # .grad is always None and Adam silently skips it. That is a property of two
            # code conventions, not of the parameter, so state it structurally: a target
            # that moved would be chased by the head it is supposed to anchor and the L2
            # loss would lose its floor. The parameters stay inside actor_rl_params
            # (frozen tensors are inert in Adam) so the optimizer group layout and every
            # saved checkpoint remain byte-identical.
            priv_enc_params = list(self.depth_actor.priv_encoder.parameters())
            if self.depth_rl_mode:
                for p in priv_enc_params:
                    p.requires_grad_(False)
            # nn.Identity (empty priv_encoder_dims) contributes no parameters, in which
            # case the target is the raw 29-D slice and both guards hold trivially.
            assert not any(p.requires_grad for p in priv_enc_params) or not self.depth_rl_mode, (
                "depth_actor.priv_encoder is the frozen supervision target for "
                "history_encoder and must never be trained on the depth-RL path")
            hist_params = (list(self.depth_actor.history_encoder.parameters())
                           if getattr(self.depth_actor, 'history_encoder', None) is not None
                           else [])
            # From-scratch pure RL (rl_hist_via_surrogate): route history_encoder
            # through the PPO surrogate instead of the DAgger L2 regression onto a
            # frozen priv_encoder. With no distilled checkpoint priv_encoder is
            # random, so an L2 target would be meaningless; the surrogate is the
            # only supervision the head needs. Empty hist_ids keeps the head inside
            # depth_rl_optimizer group 0, and a None optimiser makes update_depth_rl
            # skip the L2 backward / clip / step (all three are guarded on it).
            self.depth_rl_hist_via_surrogate = bool(
                depth_encoder_paras.get("rl_hist_via_surrogate", False))
            if self.depth_rl_hist_via_surrogate:
                self.depth_hist_encoder_lr = 0.0
                self.depth_hist_encoder_optimizer = None
                hist_ids = set()
            else:
                self.depth_hist_encoder_lr = float(
                    depth_encoder_paras.get("rl_hist_encoder_learning_rate", depth_rl_lr))
                self.depth_hist_encoder_optimizer = (
                    optim.Adam(hist_params, lr=self.depth_hist_encoder_lr) if hist_params else None)
                hist_ids = {id(p) for p in hist_params}
            # Includes priv_encoder by design (see the freeze above): keeping it in the
            # group means the group's parameter list matches depth_actor's layout, and a
            # frozen tensor simply never receives a gradient, so Adam never steps it.
            actor_rl_params = [p for p in self.depth_actor.parameters()
                               if id(p) not in hist_ids]
            # Frozen sigma is excluded from the group rather than merely zeroed, so
            # neither Adam's moment estimates nor clip_grad_norm_ touch it.
            std_rl_params = ([] if self.depth_rl_freeze_std
                             else [self.depth_actor_std])
            self.depth_rl_optimizer = optim.Adam(
                [{'params': [*actor_rl_params, *std_rl_params],
                  'lr': depth_rl_lr},
                 {'params': list(self.depth_encoder.parameters()),
                  'lr': self.depth_encoder_rl_lr}])
            if self.depth_rl_freeze_std:
                self.depth_actor_std.requires_grad_(False)
            # Teacher-style completeness on the depth chain: the privileged critic and
            # the estimator are no longer frozen. The critic gets its own optimiser
            # because `self.optimizer` also owns the TEACHER actor, which must stay
            # put (it is the label source for DAgger and the privileged reference).
            # Its lr is separate and higher: a value function that starts from the
            # teacher's scale has to catch up with a policy that is still moving.
            self.critic_lr = float(
                depth_encoder_paras.get("rl_critic_learning_rate", 3.e-4))
            self.critic_optimizer = optim.Adam(
                self.actor_critic.critic.parameters(), lr=self.critic_lr)
            if self.depth_rl_mode:
                print(f"[ppo] depth-RL ON: depth_actor trained by PPO (lr {depth_rl_lr:.2e}, "
                      f"adaptive [{self.depth_rl_lr_min:.1e}, {self.depth_rl_lr_max:.1e}], "
                      f"desired_kl {self.depth_rl_desired_kl}, "
                      f"kl target {self.depth_rl_kl_target:.5f}"
                      f"{' (step_ratio ' + format(self.depth_rl_step_ratio, '.3f') + ')' if self.depth_rl_step_ratio is not None else ''}, "
                      f"sigma <= {self.depth_rl_max_noise_std}); "
                      f"depth_encoder JOINT (lr {self.depth_encoder_rl_lr:.1e}, "
                      f"replayed in-graph, {self.replay_chunk}-sample chunks, "
                      f"teacher anchor coef {self.depth_encoder_anchor_coef}); "
                      f"history_encoder {'via PPO surrogate (no L2)' if self.depth_rl_hist_via_surrogate else 'supervised (lr %.1e, own optimiser)' % self.depth_hist_encoder_lr}; "
                      f"entropy_coef {self.depth_rl_entropy_coef}; "
                      f"critic lr {self.critic_lr:.1e}, estimator trains; "
                      f"teacher actor stays frozen")
                print(f"[ppo] depth-RL control: sigma "
                      f"{'FROZEN (requires_grad=False, not in optimiser)' if self.depth_rl_freeze_std else 'trainable'}; "
                      f"actor lr {'FIXED' if not self.depth_rl_adaptive_lr else 'adaptive'} "
                      f"at {depth_rl_lr:.2e}; encoder lr {self.depth_encoder_rl_lr:.1e}"
                      f"{' (FROZEN)' if self.depth_encoder_rl_lr == 0.0 else ''}")

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape,  critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, info, hist_encoding=False):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values, use proprio to compute estimated priv_states then actions, but store true priv_states
        if self.train_with_estimated_states:
            obs_est = obs.clone()
            priv_states_estimated = self.estimator(obs_est[:, :self.num_prop])
            obs_est[:, self.num_prop+self.num_scan:self.num_prop+self.num_scan+self.priv_states_dim] = priv_states_estimated
            self.transition.actions = self.actor_critic.act(obs_est, hist_encoding).detach()
        else:
            self.transition.actions = self.actor_critic.act(obs, hist_encoding).detach()

        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs

        return self.transition.actions
    
    def _init_depth_std(self):
        """Seed the depth-RL action noise.

        `rl_init_std` (cfg.depth_encoder) wins when set; when it is None the
        teacher's per-action std is copied, which is the width the teacher-recipe
        probe runs at.

        Why the override exists: the teacher's std is calibrated for the TEACHER's
        action scale, and a distilled student emits a flatter action, so the same
        sigma is a much larger relative perturbation (measured once on a 60-iter run:
        sigma 0.60, term_roll 50 %, episodes 4 s, wococo/stage_max 0.008 -- while
        play, which uses the deterministic mean, clears the same level). PPO cannot
        shrink it by itself: the Gaussian sigma's gradient against the clipped
        surrogate is zero in expectation over samples and shares one clip_grad_norm_
        with the whole actor, while the entropy term actively pushes it up. So if the
        inherited width turns out to be too wide, pin it here (e.g. 0.05) -- that is
        the intended rollback for the teacher-recipe probe.
        """
        override = None
        if self.depth_encoder_paras is not None:
            override = self.depth_encoder_paras.get("rl_init_std", None)
        if override is not None:
            with torch.no_grad():
                self.depth_actor_std.fill_(float(override))
            print(f"[ppo] depth-RL: action std OVERRIDDEN by rl_init_std "
                  f"(mean={self.depth_actor_std.mean().item():.3f}, "
                  f"min={self.depth_actor_std.min().item():.3f}, "
                  f"max={self.depth_actor_std.max().item():.3f})")
            self.depth_std_initialised = True
            return
        ref = self.actor_critic.std.detach()
        if ref.numel() == self.depth_actor_std.numel():
            with torch.no_grad():
                self.depth_actor_std.copy_(ref)
        print(f"[ppo] depth-RL: action std initialised (mean="
              f"{self.depth_actor_std.mean().item():.3f}, "
              f"min={self.depth_actor_std.min().item():.3f}, "
              f"max={self.depth_actor_std.max().item():.3f})")
        self.depth_std_initialised = True

    def _depth_distribution(self, observations, depth_latent):
        """Gaussian policy of the depth student for one batch of samples."""
        mean = self.depth_actor(observations, hist_encoding=True, scandots_latent=depth_latent)
        # Same per-forward clamp the teacher gets (ActorCritic.update_distribution):
        # without it the entropy term can walk sigma past the teacher's calibrated
        # width with nothing to stop it.
        if self.depth_rl_max_noise_std is not None:
            with torch.no_grad():
                self.depth_actor_std.data.clamp_(max=self.depth_rl_max_noise_std)
        return Normal(mean, mean * 0. + self.depth_actor_std)

    def act_depth(self, observations, critic_obs, depth_latent,
                  depth_images=None, depth_props=None, depth_hiddens=None,
                  scandots_target=None):
        """Rollout step for learn_depth_rl: sample the depth student, store the PPO data.

        observations are the DEPLOY-like (estimated-state) student observations and
        depth_latent is the encoder output for this step; both are stored so
        update_depth_rl can re-evaluate the same distribution after the weights
        move. The value baseline is the frozen privileged critic.

        depth_images / depth_props / depth_hiddens are the encoder's INPUTS for
        this step. They are what lets update_depth_rl replay the encoder inside
        the PPO graph (joint optimisation) instead of using the detached latent.

        scandots_target is the teacher's frozen scan-encoder latent for the same
        frame. It rides along as the anchor label, so the encoder keeps a stable
        target while the surrogate pulls on it.
        """
        if not self.depth_std_initialised:
            self._init_depth_std()
        with torch.no_grad():
            distribution = self._depth_distribution(observations, depth_latent)
            actions = distribution.sample()
            self.transition.actions = actions.detach()
            self.transition.actions_log_prob = distribution.log_prob(actions).sum(dim=-1).detach()
            self.transition.action_mean = distribution.mean.detach()
            self.transition.action_sigma = distribution.stddev.detach()
            self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.observations = observations
        self.transition.critic_observations = critic_obs
        self.transition.depth_latent = depth_latent.detach()
        # Encoder replay inputs (see update_depth_rl). depth_hiddens is the GRU
        # state BEFORE the forward, stored as (num_envs, 512); the generator
        # restores the (1, batch, 512) layout GRU expects.
        self.transition.depth_images = None if depth_images is None else depth_images.detach()
        self.transition.depth_props = None if depth_props is None else depth_props.detach()
        self.transition.depth_hiddens = (
            None if depth_hiddens is None
            else depth_hiddens.squeeze(0).detach())
        self.transition.scandots_latent = (
            None if scandots_target is None else scandots_target.detach())
        # The acting observations have the estimator's output spliced into the
        # privileged-state slice, so the supervised target must be taken from the
        # untouched env observation (critic_obs) and stored separately.
        self.transition.priv_states_true = critic_obs[
            :, self.num_prop + self.num_scan:
            self.num_prop + self.num_scan + self.priv_states_dim].detach()
        return actions

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
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

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
                            param_group['lr'] = self.learning_rate


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
                # loss = self.teacher_alpha * imitation_loss + (1 - self.teacher_alpha) * loss

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Detached GPU accumulation — no host sync inside the loop.
                mean_value_loss += value_loss.detach()
                mean_surrogate_loss += surrogate_loss.detach()
                mean_estimator_loss += estimator_loss.detach()
                mean_priv_reg_loss += priv_reg_loss.detach()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        # Single host sync: stack the four GPU-accumulated means and pull them over in
        # one .tolist() (was four sequential .item() calls = four sync points).
        mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_priv_reg_loss = (
            torch.stack([mean_value_loss, mean_surrogate_loss,
                         mean_estimator_loss, mean_priv_reg_loss]) / num_updates).tolist()
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
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:
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

    def update_depth_rl(self):
        """PPO update for the depth student, with the teacher's full objective set.

        What moves: depth_actor + its action std (surrogate), the depth encoder
        (surrogate, replayed in-graph), the privileged critic (value loss), the
        estimator (regression on the true privileged state) and the history encoder
        (regression onto priv_encoder(true privileged latent), the update_dagger()
        term: hist_encoding=True routes the actor through that head at deployment, so
        the surrogate alone gives it no direct target). The teacher actor stays
        frozen.

        Why the critic is no longer a frozen baseline: the advantages were previously
        anchored to a value function trained for the TEACHER's state distribution,
        while the rollouts come from the depth student. That mismatch biases every
        advantage, so the surrogate can only make locally-consistent progress.

        Why the encoder is replayed instead of using the stored latent: the rollout
        latent is computed under no_grad, so it carries no gradient and any weight
        change would be invisible to the objective. forward_step re-runs the encoder
        on the stored (frame, proprio, GRU hidden-in), which puts it in the same
        autograd graph as the actor — exactly how the teacher's scan_encoder is
        trained. The GRU hidden-in is detached, i.e. truncated BPTT(1).

        Why the encoder ALSO carries an anchor term: the surrogate alone is a noisy
        policy-gradient signal with no fixed target for this head, so the encoder
        drifts whenever the actor does (measured: encoder grad norm ~9 per update).
        Because the trust-region KL compares the replayed latent's action mean with
        the rollout one, that drift is charged to the KL as if it were policy
        movement, which at a narrow sigma pins the lr to its floor. The teacher's
        scan latent is frozen and 32-D, matching the no-yaw depth head, so the L2
        term restores the stable target the DAgger objective has.

        Action-level BC anchor (rl_bc_coef > 0): an L2 term between the frozen
        teacher's action mean and the replayed student mean, with a weight that
        anneals linearly to rl_bc_coef_end over rl_bc_anneal_iters healthy
        iterations. The teacher action is recomputed per chunk from the stored
        student obs (true priv slice spliced back in, hist_encoding=False -- the
        teacher's own update() path, no GRU state involved), so it needs no
        rollout-side changes and stays a stationary target while the critic
        warms up. The clock freezes on a value-loss spike (rl_bc_value_guard).
        """
        if not self.if_depth:
            return 0.0, 0.0, 0.0, 0.0
        mean_surrogate_loss = torch.zeros((), device=self.device)
        mean_entropy = torch.zeros((), device=self.device)
        mean_value_loss = torch.zeros((), device=self.device)
        mean_estimator_loss = torch.zeros((), device=self.device)
        mean_hist_latent_loss = torch.zeros((), device=self.device)
        mean_encoder_anchor_loss = torch.zeros((), device=self.device)
        mean_encoder_grad = torch.zeros((), device=self.device)
        kl_sum, kl_n = 0.0, 0
        # Annealed action-BC weight for THIS update. The clock counts healthy
        # iterations (advanced at the end of the update, gated on value-loss
        # stability), so a watchdog restart re-anneals from rl_bc_coef -- the
        # conservative direction (more anchoring), never a stronger RL signal.
        mean_bc_loss = torch.zeros((), device=self.device)
        if self.depth_rl_bc_coef_start > 0.0:
            _progress = min(
                self._depth_rl_bc_step / max(self.depth_rl_bc_anneal_iters, 1),
                1.0)
            self.depth_rl_bc_coef = self.depth_rl_bc_coef_end + (
                self.depth_rl_bc_coef_start - self.depth_rl_bc_coef_end) * _progress
        else:
            self.depth_rl_bc_coef = 0.0
        generator = self.storage.depth_mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs)
        for (obs_batch, _stored_latent, critic_obs_batch, actions_batch, target_values_batch,
             advantages_batch, returns_batch, old_actions_log_prob_batch,
             old_mu_batch, old_sigma_batch, priv_true_batch, scandots_target_batch,
             depth_images_batch, depth_props_batch, depth_hiddens_batch) in generator:

            # GRADIENT ACCUMULATION, and why it is needed here. num_mini_batches
            # stays at the teacher's 4, so one minibatch is 6144 samples and the step
            # count is the teacher's 20/iteration. The encoder replay is what broke
            # the memory budget: a single 6144-sample conv forward+backward peaks
            # around 12 GB. Walking the minibatch in chunks of replay_chunk samples,
            # back-propagating one chunk at a time and stepping ONCE at the end, is
            # the SAME update -- each chunk is weighted by its share of the batch, so
            # the accumulated gradient is the whole-batch gradient -- at 512 samples
            # of activation (~1 GB). The alternative, raising num_mini_batches to 40
            # to shrink the batch, would give 200 steps/iteration against the
            # teacher's 20, i.e. 10x the trust-region movement, and the adaptive KL
            # would sit pinned on min_learning_rate for the whole run.
            n_samples = depth_images_batch.shape[0]
            chunk = min(self.replay_chunk, n_samples)
            n_chunks = (n_samples + chunk - 1) // chunk

            self.depth_rl_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            self.estimator_optimizer.zero_grad()
            if self.depth_hist_encoder_optimizer is not None:
                # history_encoder is NOT in depth_rl_optimizer (see __init__), so this
                # is the only thing that clears its gradient.
                self.depth_hist_encoder_optimizer.zero_grad()

            kl_batch = 0.0
            for c in range(n_chunks):
                lo = c * chunk
                hi = min(lo + chunk, n_samples)
                # Share of the batch this chunk carries. Equal chunks give
                # 1/n_chunks; the general form also covers a trailing short chunk
                # and keeps the accumulated loss equal to the whole-batch mean.
                w = (hi - lo) / n_samples

                # Replay the encoder in-graph so the surrogate reaches it.
                latent = self.depth_encoder.forward_step(
                    depth_images_batch[lo:hi], depth_props_batch[lo:hi],
                    depth_hiddens_batch[:, lo:hi, :])
                # Mirror the rollout's slice: with EP_USE_YAW the head also emits
                # the two heading dims, which the rollout writes into obs[:,6:8] and
                # never feeds to depth_actor. Slicing keeps the two paths identical.
                if getattr(self.depth_encoder, 'output_yaw', False):
                    latent = latent[:, :-2]

                # Teacher anchor (see the docstring). The label is the frozen
                # scan_encoder's output on the same frame, detached so the teacher is
                # never dragged toward the student. This is the only term that gives
                # the encoder a STATIONARY target; without it the latent moves with
                # every actor step and that movement is misread as policy KL.
                encoder_anchor_loss = (
                    scandots_target_batch[lo:hi].detach() - latent
                ).norm(p=2, dim=1).mean()

                distribution = self._depth_distribution(obs_batch[lo:hi], latent)
                mu_batch = distribution.mean
                sigma_batch = distribution.stddev
                actions_log_prob_batch = distribution.log_prob(
                    actions_batch[lo:hi]).sum(dim=-1)
                entropy_batch = distribution.entropy().sum(dim=-1)

                ratio = torch.exp(actions_log_prob_batch
                                  - torch.squeeze(old_actions_log_prob_batch[lo:hi]))
                adv = torch.squeeze(advantages_batch[lo:hi])
                surrogate = -adv * ratio
                surrogate_clipped = -adv * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value loss. The storage has no privileged buffer (num_privileged_obs
                # is None), so critic_obs_batch holds the ACTING observation, whose
                # privileged-state slice carries the estimator's own output. Splice the
                # stored ground truth back in: the critic is asymmetric and must see
                # the true state, exactly as it did when its values were collected.
                priv_start = self.num_prop + self.num_scan
                critic_input = torch.cat((
                    critic_obs_batch[lo:hi, :priv_start],
                    priv_true_batch[lo:hi],
                    critic_obs_batch[lo:hi, priv_start + self.priv_states_dim:]), dim=1)
                value_batch = self.actor_critic.evaluate(critic_input)
                if self.use_clipped_value_loss:
                    value_target = target_values_batch[lo:hi]
                    returns_chunk = returns_batch[lo:hi]
                    value_clipped = value_target + (value_batch - value_target).clamp(
                        -self.clip_param, self.clip_param)
                    value_losses = (value_batch - returns_chunk).pow(2)
                    value_losses_clipped = (value_clipped - returns_chunk).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch[lo:hi] - value_batch).pow(2).mean()

                # Estimator: the same supervised objective the teacher's update() uses,
                # but the target is the stored ground-truth slice (the acting obs has
                # the estimator's own prediction in that slot).
                priv_states_predicted = self.estimator(
                    obs_batch[lo:hi, :self.num_prop])
                estimator_loss = (priv_states_predicted
                                  - priv_true_batch[lo:hi]).pow(2).mean()

                # History-encoder supervision: the teacher's update_dagger() term.
                # The target is the SAME actor's priv_encoder applied to the stored
                # privileged-latent slice. That slice is untouched by
                # _distillation_observations (only the EXPLICIT-state slice is
                # replaced by the estimator), and priv_encoder is bypassed whenever
                # hist_encoding=True, so it keeps the teacher's weights and the
                # target does not drift while the head chases it. Whole block is
                # guarded on the optimiser so this stays consistent with __init__
                # deciding whether the head is supervised at all.
                if self.depth_hist_encoder_optimizer is not None:
                    with torch.inference_mode():
                        priv_latent_target = self.depth_actor.infer_priv_latent(obs_batch[lo:hi])
                    hist_latent = self.depth_actor.infer_hist_latent(obs_batch[lo:hi])
                    hist_latent_loss = (priv_latent_target
                                        - hist_latent).norm(p=2, dim=1).mean()
                    (hist_latent_loss * w).backward()
                    with torch.inference_mode():
                        mean_hist_latent_loss += hist_latent_loss.detach() * w

                # Annealed action-BC anchor. The acting obs carries the estimator's
                # output in the priv-explicit slice; the priv-latent and scan slices
                # were never masked, so splicing the stored ground truth back in
                # reconstructs exactly the obs the teacher would see. The target
                # uses hist_encoding=False -- the teacher's own PPO path (update()
                # runs it) and one that touches no GRU state, so it is stationary
                # and reproducible; the history head is supervised toward this same
                # priv_encoder, so the path gap is second-order for an anchor whose
                # weight decays to zero. The BC gradient reaches depth_actor AND
                # the replayed encoder (same graph as the surrogate).
                bc_loss = None
                if self.depth_rl_bc_coef > 0.0:
                    with torch.inference_mode():
                        teacher_obs_batch = torch.cat((
                            obs_batch[lo:hi, :priv_start],
                            priv_true_batch[lo:hi],
                            obs_batch[lo:hi,
                                      priv_start + self.priv_states_dim:]), dim=1)
                        actions_teacher_batch = self.actor_critic.act_inference(
                            teacher_obs_batch, hist_encoding=False,
                            scandots_latent=None)
                    bc_loss = (actions_teacher_batch
                               - mu_batch).norm(p=2, dim=1).mean()
                    mean_bc_loss += bc_loss.detach() * w

                loss = surrogate_loss \
                       - self.depth_rl_entropy_coef * entropy_batch.mean() \
                       + self.value_loss_coef * value_loss

                # The anchor rides in the SAME backward as the surrogate, not a
                # second one. Both terms reach the encoder through the replayed
                # `latent`, so a separate .backward() would re-traverse a graph the
                # surrogate's backward has already freed (RuntimeError: trying to
                # backward through the graph a second time). Adding it to `loss` is
                # arithmetically identical and costs no extra traversal; the anchor
                # depends only on `latent` and a detached target, so it still puts
                # gradient on the encoder and nowhere else.
                if self.depth_encoder_anchor_coef > 0.0:
                    loss = loss + self.depth_encoder_anchor_coef * encoder_anchor_loss
                if bc_loss is not None:
                    loss = loss + self.depth_rl_bc_coef * bc_loss

                # Weighted so the accumulation over chunks equals the whole-batch
                # loss; autograd frees each chunk's graph before the next one.
                (loss * w).backward()
                (estimator_loss * w).backward()

                with torch.inference_mode():
                    mean_surrogate_loss += surrogate_loss.detach() * w
                    mean_entropy += entropy_batch.mean().detach() * w
                    mean_value_loss += value_loss.detach() * w
                    mean_estimator_loss += estimator_loss.detach() * w
                    mean_encoder_anchor_loss += encoder_anchor_loss.detach() * w
                    if self.depth_rl_desired_kl is not None and self.schedule == 'adaptive':
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch[lo:hi] + 1.e-5) + (torch.square(old_sigma_batch[lo:hi]) + torch.square(old_mu_batch[lo:hi] - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_batch += torch.mean(kl).item() * w

            # --- one trust-region update and one optimiser step per minibatch ---
            # Same KL-adaptive schedule as update(), but on the depth-RL lr: a warm
            # start sits close to the distilled policy, so the trust region is what
            # stops the first batches from tearing the actor away from it. Evaluated
            # once per MINIBATCH (not per chunk), so the schedule still gets the
            # teacher's 20 lr updates per iteration rather than 240.
            if self.depth_rl_desired_kl is not None and self.schedule == 'adaptive':
                kl_mean = kl_batch
                # depth_rl_kl_target is the sigma-free form of the trust region (see
                # __init__): with rl_step_ratio set it is 6*(dmu_rms/sigma)^2, which
                # keeps the SAME policy step regardless of how narrow sigma is.
                if self.depth_rl_adaptive_lr:
                    if kl_mean > self.depth_rl_kl_target * 2.0:
                        self.depth_rl_learning_rate = max(
                            self.depth_rl_lr_min, self.depth_rl_learning_rate / 1.5)
                    elif kl_mean < self.depth_rl_kl_target / 2.0 and kl_mean > 0.0:
                        self.depth_rl_learning_rate = min(
                            self.depth_rl_lr_max, self.depth_rl_learning_rate * 1.5)
                    # Only the actor group follows the KL schedule; the encoder keeps
                    # its fixed (smaller) lr -- its gradient has a different scale.
                    self.depth_rl_optimizer.param_groups[0]['lr'] = self.depth_rl_learning_rate
                # kl_sum / kl_n keep accumulating with the schedule off: the KL is the
                # only evidence of how far the actor actually moved this iteration.
                kl_sum += kl_mean
                kl_n += 1

            # Clipping must run AFTER every chunk has added its gradient, otherwise
            # each chunk would be normalised on its own norm and the step would no
            # longer be the whole-batch step.
            # Group 0 of depth_rl_optimizer is exactly (actor params MINUS the
            # history encoder) + depth_actor_std. Clipping self.depth_actor instead
            # would fold the history encoder's regression gradient into the actor's
            # norm, coupling two objectives that are scaled independently.
            nn.utils.clip_grad_norm_(
                self.depth_rl_optimizer.param_groups[0]['params'], self.max_grad_norm)
            encoder_grad_norm = nn.utils.clip_grad_norm_(
                self.depth_encoder.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(
                self.actor_critic.critic.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
            if self.depth_hist_encoder_optimizer is not None:
                # Own clip: history_encoder is no longer inside depth_rl_optimizer, so
                # nothing else bounds this gradient.
                nn.utils.clip_grad_norm_(
                    self.depth_actor.history_encoder.parameters(), self.max_grad_norm)
            self.depth_rl_optimizer.step()
            self.critic_optimizer.step()
            self.estimator_optimizer.step()
            if self.depth_hist_encoder_optimizer is not None:
                self.depth_hist_encoder_optimizer.step()

            mean_encoder_grad += torch.as_tensor(
                encoder_grad_norm, device=self.device, dtype=torch.float)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        # Diagnostics for the runner's log (same convention as last_kl_mean).
        self.last_depth_kl_mean = (kl_sum / kl_n) if kl_n else float('nan')
        self.last_depth_lr = self.depth_rl_learning_rate
        self.last_encoder_grad_norm = (mean_encoder_grad / num_updates).item()
        self.last_encoder_anchor_loss = (mean_encoder_anchor_loss / num_updates).item()
        self.last_hist_latent_loss = (mean_hist_latent_loss / num_updates).item()
        self.last_depth_bc_loss = (mean_bc_loss / num_updates).item()
        self.last_depth_rl_bc_coef = float(self.depth_rl_bc_coef)
        # Advance the anneal clock only when the value loss is stable: frozen means
        # the BC weight HOLDS at its current value (and the EMA keeps the healthy
        # baseline, so recovery against the pre-divergence level re-arms the clock).
        self.last_depth_rl_bc_frozen = False
        if self.depth_rl_bc_coef_start > 0.0:
            _vl = (mean_value_loss / num_updates).item()
            if self._depth_rl_bc_value_ema is None:
                self._depth_rl_bc_value_ema = _vl
            elif (self.depth_rl_bc_value_guard is not None
                    and _vl > self.depth_rl_bc_value_guard
                    * self._depth_rl_bc_value_ema):
                self.last_depth_rl_bc_frozen = True
            else:
                self._depth_rl_bc_step += 1
                self._depth_rl_bc_value_ema = (
                    0.9 * self._depth_rl_bc_value_ema + 0.1 * _vl)
        self.storage.clear()
        self.update_counter()
        return ((mean_surrogate_loss / num_updates).item(),
                (mean_entropy / num_updates).item(),
                (mean_value_loss / num_updates).item(),
                (mean_estimator_loss / num_updates).item())

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