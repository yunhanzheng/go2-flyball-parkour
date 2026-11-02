"""Build a stage-3 warm start: student actor + teacher critic + student encoder.

WHY THIS EXISTS
---------------
`go2_depth` sets `depth_encoder.train_with_ppo = True`, and OnPolicyRunner then
builds `depth_actor = None` (on_policy_runner.py: `None if self.train_depth_ppo
else deepcopy(actor_critic.actor)`).  PPO's `load()` therefore skips the
`depth_actor_state_dict` branch, so resuming straight from a DAgger student
directory gives you:

    actor   = the TEACHER's weights (trained against real height scans)
    encoder = the STUDENT's CNN (emits depth latents)

i.e. a policy wired to a latent space it has never seen.  A DAgger student keeps
its deployable policy in `depth_actor_state_dict`, and that module is literally
the same `Actor` class the teacher uses, so the fix is a key remap:

    model_state_dict['actor.<k>'] <- depth_actor_state_dict['<k>']

What is copied, what is kept:
    actor        <- student's depth_actor            (deployable policy)
    critic       <- teacher (753-D privileged obs; asymmetric actor-critic)
    std          <- teacher (kept: `continue_from_last_std = True`)
    encoder      <- student's depth_encoder
    estimator    <- student's estimator
    optimizer    -> DROPPED by default: the student's groups were built for the
                    DAgger objective; depth PPO rebuilds its own (KL-scheduled)
                    groups, and PPO.load() already degrades gracefully.

USAGE
-----
    python make_stage3_ckpt.py                  # student/model_10000 -> stage3-from-student/model_0.pt
    python make_stage3_ckpt.py --src logs/extreme-parkour/v2-student/model_5000.pt \\
                               --dst logs/extreme-parkour/stage3-from-student/model_0.pt

Then, with a DIFFERENT exptid (never reuse the resumeid directory):

    python train.py --task go2_depth --exptid stage3-run1 \\
        --resume --resumeid stage3-from-student --checkpoint 0 \\
        --use_camera --delay --headless --device cuda:0

The output must be named model_<int>.pt: --checkpoint N resolves to exactly
`model_{N}.pt` (helpers.get_load_path), and --resumeid is a DIRECTORY, not a file.
"""

import argparse
import os
from pathlib import Path

import torch

# scripts/ -> legged_gym(pkg) -> legged_gym(dir, holds logs/)
_LOGS = Path(__file__).resolve().parents[2] / 'logs' / 'extreme-parkour'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--src', default=str(_LOGS / 'student' / 'model_10000.pt'),
                   help='DAgger student checkpoint (needs depth_actor_state_dict)')
    p.add_argument('--dst', default=str(_LOGS / 'stage3-from-student' / 'model_0.pt'),
                   help='output checkpoint; must be named model_<int>.pt')
    p.add_argument('--keep-optimizer', action='store_true',
                   help='keep the student optimizer state instead of dropping it')
    p.add_argument('--dry-run', action='store_true', help='validate and report only')
    return p.parse_args()


def main():
    a = parse_args()
    src, dst = Path(a.src), Path(a.dst)
    if not src.is_file():
        raise SystemExit('src not found: {}'.format(src))

    d = torch.load(str(src), map_location='cpu')
    for k in ('model_state_dict', 'depth_actor_state_dict'):
        if k not in d:
            raise SystemExit('{} has no {}: not a DAgger student checkpoint'.format(src, k))
    ms, da = d['model_state_dict'], d['depth_actor_state_dict']

    # Every student-actor key must land on an existing, same-shaped teacher-actor key.
    problems, diffs = [], []
    for k, v in da.items():
        tgt = 'actor.' + k
        if tgt not in ms:
            problems.append('{}: missing in model_state_dict'.format(tgt))
        elif tuple(ms[tgt].shape) != tuple(v.shape):
            problems.append('{}: {} vs student {}'.format(tgt, tuple(ms[tgt].shape), tuple(v.shape)))
        else:
            diffs.append(float((ms[tgt] - v).abs().max()))
    if problems:
        raise SystemExit('key remap failed:\n  ' + '\n  '.join(problems))
    if 'critic.0.weight' not in ms:
        raise SystemExit('no critic in model_state_dict: cannot keep the privileged critic')

    print('src  {}'.format(src))
    print('  iterations      {}'.format(d.get('iter')))
    print('  actor keys      {}  (all remapped)'.format(len(da)))
    print('  actor match     all {} keys land on actor.* with equal shape'.format(len(da)))
    print('  critic in       {}'.format(tuple(ms['critic.0.weight'].shape)))
    print('  overwrite delta max |teacher - student|: {:.3f}'.format(max(diffs)))
    if 'std' in ms:
        print('  std             {} dims, mean {:.3f} (kept)'.format(
            tuple(ms['std'].shape), float(ms['std'].mean())))
    else:
        print('  std             MISSING (PPO will re-init from policy.init_noise_std)')
    print('  encoder         {} keys'.format(
        len(d.get('depth_encoder_state_dict', {})) or 'MISSING'))
    print('  estimator       {} keys'.format(
        len(d.get('estimator_state_dict', {})) or 'MISSING'))

    if a.keep_optimizer:
        print('  optimizer       kept (student groups: {})'.format(
            len(d.get('optimizer_state_dict', {}).get('param_groups', []))))
    else:
        d.pop('optimizer_state_dict', None)
        print('  optimizer       dropped')

    for k, v in da.items():
        ms['actor.' + k] = v.detach().clone()
    d['iter'] = 0
    d['infos'] = None

    if a.dry_run:
        print('\n[dry-run] nothing written')
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, str(dst))
    print('\nwrote {} ({:.1f} MB)'.format(dst, dst.stat().st_size / 1e6))
    print('next:  python train.py --task go2_depth --exptid stage3-run1 \\')
    print('           --resume --resumeid {} --checkpoint {} \\'.format(
        dst.parent.name, dst.stem.replace('model_', '')))
    print('           --use_camera --delay --headless --device cuda:0')


if __name__ == '__main__':
    main()
