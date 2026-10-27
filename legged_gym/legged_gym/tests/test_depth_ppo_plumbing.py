"""CPU plumbing guards for the single-stage depth PPO (go2_depth).

No Isaac Gym here: this only checks the wiring between RolloutStorage, PPO and the
depth encoder, i.e. the failure modes that are silent at runtime:

  * the depth frames cached during rollout must be NORMAL tensors (rollout runs under
    torch.inference_mode(), so a lazily allocated buffer would otherwise be an
    inference tensor and reject copy_/backward),
  * PPO gradients must actually reach the encoder,
  * the aux scan-latent loss must be finite,
  * update() must restore the encoder GRU state it borrowed for replay,
  * the encoder and the policy must stay in two parameter groups with their own lr,
  * the no-camera (teacher) and DAgger (go2_student) paths must be untouched.

Run:  python -m pytest legged_gym/legged_gym/tests/test_depth_ppo_plumbing.py -s
"""
import os
import sys
import types

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_ROOT, "rsl_rl"))

from rsl_rl.modules import (                                    # noqa: E402
    ActorCriticRMA, Estimator, RecurrentDepthBackbone,
    DepthOnlyFCBackbone58x87)
from rsl_rl.algorithms import PPO                               # noqa: E402

N_PROP, N_SCAN, N_HIST, N_PRIV_LAT, N_PRIV, N_ACT = 53, 132, 10, 29, 9, 12
N_OBS = N_PROP + N_SCAN + N_HIST * N_PROP + N_PRIV_LAT + N_PRIV
N_ENV, T, TICK = 8, 6, 3
TORCH_SEED = 0


def _actor_critic():
    return ActorCriticRMA(
        N_PROP, N_SCAN, N_OBS, N_PRIV_LAT, N_PRIV, N_HIST, N_ACT,
        scan_encoder_dims=[128, 64, 32], actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128], activation='elu', init_noise_std=1.0,
        priv_encoder_dims=[64, 20], tanh_encoder_output=False)


def _estimator():
    return Estimator(N_PROP, N_PRIV, [128, 64])


_ESTIMATOR_PARAS = dict(priv_states_dim=N_PRIV, num_prop=N_PROP, num_scan=N_SCAN,
                        learning_rate=1e-4, hidden_dims=[128, 64],
                        train_with_estimated_states=True)
_DEPTH_PARAS = dict(if_depth=True, hidden_dims=512, learning_rate=1e-3,
                    train_with_ppo=True, aux_scan_coef=1.0,
                    zero_heading_obs=True, depth_shape=(87, 58))


def _backbone():
    stub = types.SimpleNamespace(env=types.SimpleNamespace(n_proprio=N_PROP))
    return DepthOnlyFCBackbone58x87(N_PROP, 32, 512), stub


def _rollout(ppo, enc, with_depth):
    """Mimic learn_depth_ppo: encoder ticks every TICK steps, latent held in between."""
    ppo.init_storage(N_ENV, T, [N_OBS], [None], [N_ACT])
    obs = torch.randn(N_ENV, N_OBS)
    latent = frame = hidden = None
    for t in range(T):
        if with_depth and t % TICK == 0:
            if enc.hidden_states is None:
                enc.hidden_states = torch.zeros(1, N_ENV, enc.rnn.hidden_size)
            hidden = enc.hidden_states[0].clone()
            frame = -torch.rand(N_ENV, 58, 87) * 2.0
            prop = obs[:, :N_PROP].clone()
            prop[:, 6:8] = 0.0
            with torch.no_grad():
                latent = enc(frame.clone(), prop)
        with torch.inference_mode():
            ppo.act(obs, obs, {}, hist_encoding=True, depth_latent=latent,
                    depth_img=frame, depth_hidden=hidden)
            ppo.process_env_step(torch.randn(N_ENV), torch.zeros(N_ENV), {})
        obs = torch.randn(N_ENV, N_OBS)
    return obs


def test_depth_ppo_plumbing():
    torch.manual_seed(TORCH_SEED)
    ac = _actor_critic()
    backbone, stub = _backbone()
    enc = RecurrentDepthBackbone(backbone, stub)
    ppo = PPO(ac, _estimator(), _ESTIMATOR_PARAS, enc, dict(_DEPTH_PARAS), None,
              num_learning_epochs=2, num_mini_batches=2, learning_rate=3e-4,
              schedule='fixed', desired_kl=0.01, max_grad_norm=1.0, device='cpu',
              priv_reg_coef_schedual=[0, 0.1, 5000, 3000])
    assert ppo.train_depth_ppo, "train_with_ppo did not switch the depth PPO on"
    assert not hasattr(ppo, 'depth_actor'), \
        "depth PPO must not clone an actor (play.py keys off hasattr)"

    conv_w0 = backbone.image_compression[0].weight.detach().clone()
    obs = _rollout(ppo, enc, with_depth=True)
    hidden_after_rollout = enc.hidden_states.clone()

    st = ppo.storage
    assert st.depth is not None and st.depth_hidden is not None
    assert not st.depth.is_inference(), "depth buffer became an inference tensor"
    assert st.depth.dtype == torch.float16
    assert tuple(st.depth.shape) == (T, N_ENV, 58, 87)
    assert tuple(st.depth_hidden.shape) == (T, N_ENV, enc.rnn.hidden_size)

    ppo.compute_returns(obs)
    ppo.update()

    assert ppo.last_depth_aux_loss == ppo.last_depth_aux_loss, "aux loss is NaN"
    moved = (backbone.image_compression[0].weight.detach() - conv_w0).abs().max()
    assert moved.item() > 0.0, "PPO did not update the depth encoder"
    assert torch.allclose(enc.hidden_states, hidden_after_rollout), \
        "encoder GRU state was not restored after the replay in update()"
    assert [g['lr'] for g in ppo.optimizer.param_groups] == [3e-4, 1e-3], \
        "policy and encoder must be two parameter groups with their own lr"
    assert ppo.depth_encoder_optimizer is ppo.optimizer


def test_no_camera_path_unchanged():
    torch.manual_seed(TORCH_SEED)
    ppo = PPO(_actor_critic(), _estimator(), _ESTIMATOR_PARAS, None,
              dict(_DEPTH_PARAS, if_depth=False), None, num_learning_epochs=2,
              num_mini_batches=2, learning_rate=3e-4, schedule='fixed',
              device='cpu', priv_reg_coef_schedual=[0, 0.1, 5000, 3000])
    ppo.depth_actor = ppo.actor_critic.actor
    ppo.depth_actor_optimizer = torch.optim.Adam(ppo.depth_actor.parameters(),
                                                 lr=1e-3)
    obs = _rollout(ppo, None, with_depth=False)
    assert ppo.storage.depth is None, "no camera must mean no depth buffer"
    ppo.compute_returns(obs)
    ppo.update()
    ppo.update_dagger()


def test_dagger_path_keeps_depth_actor():
    torch.manual_seed(TORCH_SEED)
    ac = _actor_critic()
    backbone, stub = _backbone()
    enc = RecurrentDepthBackbone(backbone, stub)
    ppo = PPO(ac, _estimator(), _ESTIMATOR_PARAS, enc,
              dict(_DEPTH_PARAS, train_with_ppo=False),
              types.SimpleNamespace(parameters=lambda: iter([])),
              num_learning_epochs=2, num_mini_batches=2, learning_rate=3e-4,
              schedule='fixed', device='cpu',
              priv_reg_coef_schedual=[0, 0.1, 5000, 3000])
    ppo.depth_actor = ac.actor
    assert not ppo.train_depth_ppo and hasattr(ppo, 'depth_actor')
    obs = _rollout(ppo, enc, with_depth=False)
    ppo.compute_returns(obs)
    ppo.update()
    ppo.update_dagger()
