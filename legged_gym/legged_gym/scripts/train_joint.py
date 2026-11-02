"""One-stage: teacher PPO at full fleet + depth-encoder BC on a small rendered prefix.

Why this script exists instead of train.py: train.py routes on cfg.depth.use_camera
(helpers.py sets depth_encoder.if_depth, and if_depth picks the learn method), so any
--use_camera run leaves learn_RL entirely -- either for learn_vision (pure BC, no PPO)
or for learn_depth_ppo (8810e91, encoder inside the PPO batch). Neither can keep the
actor on scan and render depth for a subset only, because in both of them the acting
policy consumes depth in EVERY env.

Here the acting policy never touches depth:

  * rollout + PPO        = learn_RL over the full env.num_envs fleet (6144), the
                           go2_teacher observation set, scan channel included.
  * every update_interval-th env step, for the FIRST num_depth_envs envs only, we
    snapshot (frame, obs) and after the rollout run

        z   = depth_encoder(frame, obs with the two heading slots blanked)
        a_s = actor(obs, scandots_latent=z)        # same actor, perception swapped
        L   = act_w * ||a_s - a_t|| + lat_w * ||z - scan_lat||,  Adam_encoder(L)

    with a_t / scan_lat recomputed from the SAME snapshot obs under the current actor
    weights, so the target tracks the policy (no stale BC labels).
  * only the encoder parameters receive that gradient: the actor is frozen for the BC
    step (requires_grad_(False)) and go2_teacher has depth_encoder.train_with_ppo off,
    so PPO contributes no encoder gradient either.

This is what cfg.depth.num_depth_envs was built for: rendering is the cost that scales
with the fleet, and the BC target does not need a frame in every env. It requires
cfg.depth.render_on_demand (each env cast only on its own due step) plus the two
camera terrain overrides turned off, so the fleet still runs the teacher's own 10x80
grid and terrain mix instead of helpers.py's 10x20 camera grid.

Deployment artifact is one checkpoint: model_state_dict (the PPO actor, scan-fed) +
depth_encoder_state_dict, played through the same scan->depth latent swap the BC step
trains, so play.py needs no separate student head.

Usage (the t18 path-B fleet, 6144 physics envs / 280 rendered):

    python -u legged_gym/legged_gym/scripts/train_joint.py --task go2_teacher \
        --use_camera --num_envs 6144 --num_depth_envs 280 --render_on_demand \
        --exptid joint-6144-280 --max_iterations 10000 --device cuda:0 --headless
"""
import argparse
import os
import sys
import time

# isaacgym has to be imported before torch (gymdeps enforces the order, same as train.py).
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import get_args, update_cfg_from_args
import torch
import torch.nn as nn

from legged_gym import LEGGED_GYM_ROOT_DIR


def parse_extra_args():
    """Knobs that belong to this loop only, so get_args() stays task-agnostic."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--act_w", type=float, default=1.0, help="weight of the action-BC term")
    p.add_argument("--lat_w", type=float, default=1.0,
                   help="weight of the latent-consistency term (depth latent vs the actor's own scan latent)")
    p.add_argument("--enc_lr", type=float, default=None,
                   help="encoder Adam lr (default: cfg.depth_encoder.learning_rate)")
    p.add_argument("--enc_clip", type=float, default=1.0, help="grad-norm clip for the BC step")
    p.add_argument("--save_every", type=int, default=400, help="iterations between checkpoints")
    p.add_argument("--resume_ckpt", type=str, default=None,
                   help="explicit model_*.pt to resume this loop from (for a cold "
                        "--resume, task_registry already loads the teacher ckpt)")
    p.add_argument("--minibatches", type=int, default=None, help="override algorithm.num_mini_batches")
    p.add_argument("--epochs", type=int, default=None, help="override algorithm.num_learning_epochs")
    p.add_argument("--steps_per_env", type=int, default=None, help="override runner.num_steps_per_env")
    p.add_argument("--desired_kl", type=float, default=None, help="override algorithm.desired_kl")
    p.add_argument("--max_lr", type=float, default=None, help="override algorithm.max_learning_rate")
    p.add_argument("--no_tf32", action="store_true", default=False,
                   help="disable TF32 matmuls (default on, matching train.py)")
    known, rest = p.parse_known_args()
    # Hand everything else to the shared Isaac Gym parser.
    sys.argv = [sys.argv[0]] + rest
    return known


def main():
    ex = parse_extra_args()
    args = get_args()
    if not ex.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # Terrain parity with the camera-off teacher runs: helpers.py rewrites the grid
    # (10x20, horizontal_scale 0.1, y_range [-0.1,0.1]) and, headless, the terrain mix
    # for any use_camera config. The student config disables both for the same reason.
    # These two lines MUST precede update_cfg_from_args: that rewrite is one-way, so
    # turning the overrides off afterwards leaves the camera grid in place.
    env_cfg.depth.override_terrain_mix_for_camera = False
    env_cfg.depth.override_terrain_grid_for_camera = False
    # Apply the CLI once here (train.py does the same) so the checks below see final
    # values: make_env/make_alg_runner only re-apply update_cfg_from_args internally,
    # which is after the env is already built. It is idempotent, so running it twice
    # is harmless.
    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)

    nd = int(getattr(env_cfg.depth, "num_depth_envs", 0) or 0)
    if not env_cfg.depth.use_camera:
        sys.exit("[joint] needs --use_camera: without a rendered prefix there is no BC signal")
    if not getattr(env_cfg.depth, "render_on_demand", False):
        sys.exit("[joint] needs --render_on_demand; the eager path ray-casts the whole fleet "
                 "on every capture step, so a prefix subset would save no ray")
    if not 0 < nd < env_cfg.env.num_envs:
        sys.exit(f"[joint] --num_depth_envs must be in (0, num_envs={env_cfg.env.num_envs}); "
                 f"got {nd}. Set it to the number of envs that should actually render depth.")

    for attr, dst in [("minibatches", "num_mini_batches"), ("epochs", "num_learning_epochs"),
                      ("desired_kl", "desired_kl"), ("max_lr", "max_learning_rate")]:
        val = getattr(ex, attr)
        if val is not None:
            setattr(train_cfg.algorithm, dst, val)
    if ex.steps_per_env is not None:
        train_cfg.runner.num_steps_per_env = ex.steps_per_env

    # --nodelay is applied in train.py, not in helpers.update_cfg_from_args, and this
    # script bypasses train.py -- so it has to be mirrored here or the flag is parsed
    # and silently dropped. Without it go2_teacher keeps action_delay=True with
    # latch_mean_level 6.0: no delay while the curriculum climbs, then a hard 20 ms
    # step the moment the fleet mean reaches row 6 (ramp_steps = 0), which is exactly
    # what the new-run-v2 teacher runs turned off.
    if getattr(args, "nodelay", False):
        env_cfg.domain_rand.action_delay = False
        env_cfg.domain_rand.action_delay_start_latched = False
        print("[joint] --nodelay: action delay OFF (0 ms) - NOT deploy-faithful", flush=True)
    else:
        print(f"[joint] action delay: {env_cfg.domain_rand.action_delay} "
              f"latch_mean_level={getattr(env_cfg.domain_rand, 'action_delay_latch_mean_level', None)} "
              f"start_latched={getattr(env_cfg.domain_rand, 'action_delay_start_latched', None)} "
              f"(pass --nodelay for the new-run-v2 lag-free dynamics)", flush=True)

    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args,
                                                      train_cfg=train_cfg, init_wandb=False)
    alg, ac, enc = runner.alg, runner.alg.actor_critic, runner.alg.depth_encoder
    assert alg.if_depth, "need --use_camera"
    if getattr(alg, "train_depth_ppo", False):
        sys.exit(f"[joint] task {args.task} has depth_encoder.train_with_ppo on; that task trains "
                 "the encoder inside the PPO batch and wants go2_depth via train.py instead")

    dev = runner.device
    IV = int(env_cfg.depth.update_interval)
    NBC = min(nd, env.num_envs)
    NUM_STEPS = runner.num_steps_per_env
    n_prop = env_cfg.env.n_proprio
    enc_lr = ex.enc_lr if ex.enc_lr is not None else float(
        getattr(train_cfg.depth_encoder, "learning_rate", 1e-3))
    enc_opt = torch.optim.Adam(enc.parameters(), lr=enc_lr)

    log_pth = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.proj_name, args.exptid or "joint")
    os.makedirs(log_pth, exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(os.path.join(log_pth, "tb"))

    print(f"[joint] envs={env.num_envs} bc_envs={NBC} steps/iter={NUM_STEPS} IV={IV} "
          f"act_w={ex.act_w} lat_w={ex.lat_w} enc_lr={enc_lr} "
          f"minibatches={train_cfg.algorithm.num_mini_batches} epochs={train_cfg.algorithm.num_learning_epochs} "
          f"lr=[{train_cfg.algorithm.min_learning_rate:.1e},{train_cfg.algorithm.max_learning_rate:.1e}] "
          f"desired_kl={train_cfg.algorithm.desired_kl}", flush=True)
    print(f"[joint] terrain: {env_cfg.terrain.num_rows}x{env_cfg.terrain.num_cols} "
          f"h_scale={env_cfg.terrain.horizontal_scale} y_range={env_cfg.terrain.y_range} "
          f"mix={ {k: v for k, v in env_cfg.terrain.terrain_dict.items() if v} } "
          f"(camera overrides disabled for parity)", flush=True)

    start_it = 0
    if ex.resume_ckpt:
        runner.load(ex.resume_ckpt, load_optimizer=True)
        import re
        m = re.search(r"model_(\d+)\.pt", ex.resume_ckpt)
        start_it = int(m.group(1)) + 1 if m else 0
        print(f"[joint] resumed {ex.resume_ckpt} -> start iter {start_it}", flush=True)

    def new_hidden():
        return torch.zeros(1, NBC, enc.rnn.hidden_size, device=dev)

    def bc_tick(obs_snap, frame):
        """One encoder update from a (obs, frame) snapshot pair taken during the rollout."""
        scan_lat = ac.actor.infer_scandots_latent(obs_snap)
        a_t = ac.act_inference(obs_snap, hist_encoding=False, scandots_latent=None).detach()
        op = obs_snap.clone()
        op[:, 6:8] = 0.0                      # heading commands are not resolvable from depth
        prop = op[:, :n_prop]
        if enc.hidden_states is None:
            enc.hidden_states = new_hidden()
        for p in ac.parameters():
            p.requires_grad_(False)           # BC gradient goes to the encoder only
        enc.train()
        z = enc(frame, prop)
        a_s = ac.act_inference(op, hist_encoding=False, scandots_latent=z)
        act_err = (a_s - a_t).norm(p=2, dim=1).mean()
        lat_err = (z[:, :scan_lat.shape[1]] - scan_lat).norm(p=2, dim=1).mean()
        loss = ex.act_w * act_err + ex.lat_w * lat_err
        enc_opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(enc.parameters(), ex.enc_clip)
        enc_opt.step()
        enc.detach_hidden_states()
        for p in ac.parameters():
            p.requires_grad_(True)
        return loss.item(), act_err.item(), lat_err.item()

    # Same random episode phase as learn_RL's init_at_random_ep_len.
    env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))
    obs = env.get_observations().to(dev)
    privileged_obs = env.get_privileged_observations()
    critic_obs = privileged_obs.to(dev) if privileged_obs is not None else obs

    tot = args.max_iterations if args.max_iterations is not None else train_cfg.runner.max_iterations
    for it in range(start_it, tot):
        t0 = time.time()
        snaps = []
        with torch.no_grad():
            for i in range(NUM_STEPS):
                if i % IV == 0:
                    # env already serves the encoder-shaped 58x87 frame; no resize here.
                    snaps.append((obs[:NBC].clone(), env.get_depth_obs()[:NBC].to(dev)))
                actions = alg.act(obs, critic_obs, {"depth": None}, False)
                obs, privileged_obs, rewards, dones, infos = env.step(actions)
                critic_obs = privileged_obs if privileged_obs is not None else obs
                obs, critic_obs, rewards, dones = (obs.to(dev), critic_obs.to(dev),
                                                  rewards.to(dev), dones.to(dev))
                alg.process_env_step(rewards, dones, infos)
                if enc.hidden_states is not None and dones.any():
                    h = enc.hidden_states.clone()
                    h[0][dones[:NBC].bool()] = 0     # a reset env restarts its encoder sequence
                    enc.hidden_states = h
        t_roll = time.time() - t0

        act_err = lat_err = 0.0
        for o_, f_ in snaps:
            with torch.enable_grad():
                _, a_, l_ = bc_tick(o_, f_)
            act_err += a_ / len(snaps)
            lat_err += l_ / len(snaps)
        del snaps
        t_bc = time.time() - t0 - t_roll

        with torch.no_grad():
            alg.compute_returns(critic_obs)
        mv, ms, me, md, ma, mpr, prc = alg.update()
        t_upd = time.time() - t0 - t_roll - t_bc

        lvl = env.terrain_levels.float()
        nrow = env_cfg.terrain.num_rows
        mean_lvl = lvl.mean().item()
        print(f"[joint] iter {it:4d} | act_err={act_err:.4f} lat_err={lat_err:.4f} | value={mv:.4f} "
              f"surr={ms:.4f} est={me:.4f} lr={alg.learning_rate:.2e} "
              f"LVL mean={mean_lvl:.2f} max={int(lvl.max().item())} "
              f"top%={100.0 * (lvl >= nrow - 1).float().mean().item():.0f} "
              f"| t={time.time() - t0:.1f}s [roll={t_roll:.2f} bc={t_bc:.2f} upd={t_upd:.2f}]",
              flush=True)
        for k, v in [('curriculum/mean_level', mean_lvl),
                     ('curriculum/max_level', float(lvl.max().item())),
                     ('curriculum/frac_top_row', (lvl >= nrow - 1).float().mean().item()),
                     ('curriculum/frac_row_ge_7', (lvl >= 7).float().mean().item()),
                     ('bc/act_err', act_err), ('bc/lat_err', lat_err),
                     ('bc/enc_loss_scale', ex.act_w + ex.lat_w),
                     ('ppo/value_loss', mv), ('ppo/surr', ms), ('ppo/explained_var', me),
                     ('ppo/lr', alg.learning_rate), ('perf/iter_time', time.time() - t0)]:
            writer.add_scalar(k, v, it)
        ep = infos.get('episode') if isinstance(infos, dict) else None
        if ep:
            for k, v in ep.items():
                try:
                    fv = float(v.item()) if torch.is_tensor(v) else float(v)
                except (TypeError, ValueError):
                    continue
                writer.add_scalar('ep/' + str(k).replace(' ', '_').replace('/', '_'), fv, it)
        if (it + 1) % ex.save_every == 0 or it == tot - 1:
            runner.save(os.path.join(log_pth, f'model_{it}.pt'))
            print(f"[joint] saved model_{it}.pt", flush=True)
    print('[joint] done', flush=True)


if __name__ == "__main__":
    main()
