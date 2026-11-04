"""Focused tests for the level-change terrain-column resample.

Mirrors, not imports: legged_gym.envs.base.legged_robot pulls in isaacgym at module
import, so this file cannot exercise the real method. It replays the two blocks it is
testing -- the round-robin initial mapping and the moved-row resample -- with the same
tensor shapes and the same index arithmetic as the env code:

  * legged_robot.py:3287-3295  self.terrain_types = arange(num_envs) % num_train_cols
  * legged_robot.py:2411-2452  _old_levels snapshot, level update, clip/floor, then
                               terrain_types[env_ids[_moved]] = randint(0, _ntc, ...)

If either block is edited, keep this mirror in step; the arithmetic it checks
(env_ids is a SUBSET tensor while _moved is a mask over that subset) is exactly the
part that would silently no-op if written as terrain_types[_moved].
"""

import torch


NUM_ENVS = 6144
NUM_COLS = 80
NUM_ROWS = 10
NUM_DEPTH = 280


def initial_mapping_is_round_robin(num_envs=NUM_ENVS, num_train_cols=NUM_COLS):
    types = torch.remainder(torch.arange(num_envs), num_train_cols).to(torch.long)
    return types


def resample_after_level_move(levels, types, env_ids, move_up, move_down,
                              num_rows=NUM_ROWS, num_train_cols=NUM_COLS):
    """Replays the curriculum block: clip first, then resample rows that ACTUALLY moved."""
    old_levels = levels[env_ids].clone()
    levels = levels.clone()
    types = types.clone()
    levels[env_ids] = levels[env_ids] + move_up.long() - move_down.long()
    levels[env_ids] = torch.clip(levels[env_ids], 0, num_rows - 1)  # resample_at_max_level=False
    moved = levels[env_ids] != old_levels
    n_moved = int(moved.sum())
    if n_moved > 0 and num_train_cols > 1:
        types[env_ids[moved]] = torch.randint(0, num_train_cols, (n_moved,))
    return levels, types, n_moved


def test_blocked_mapping_pins_the_depth_prefix_to_4_columns():
    """The regression this change fixes: floor division + 6144/80 -> prefix sees cols 0..3 only."""
    blocked = torch.div(torch.arange(NUM_ENVS), NUM_ENVS / NUM_COLS, rounding_mode='floor').to(torch.long)
    assert sorted(set(blocked[:NUM_DEPTH].tolist())) == [0, 1, 2, 3]
    assert blocked.max().item() == NUM_COLS - 1  # the fleet does reach every column, the prefix never does


def test_round_robin_mapping_covers_all_columns_in_the_prefix():
    types = initial_mapping_is_round_robin()
    assert sorted(set(types[:NUM_DEPTH].tolist())) == list(range(NUM_COLS))
    # 6144 = 80*76 + 64, so the first 64 residues occur 77 times and the last 16 occur 76 --
    # the same non-integrality the floor-division mapping already had, just redistributed.
    assert int((types == 0).sum()) == 77
    assert int((types == 63).sum()) == 77
    assert int((types == 64).sum()) == 76
    assert int((types == 79).sum()) == 76
    assert int(types.sum()) == int(torch.remainder(torch.arange(NUM_ENVS), NUM_COLS).sum())


def test_holdout_cols_are_never_dealt():
    types = initial_mapping_is_round_robin(num_train_cols=76)
    assert types.max().item() == 75
    assert not bool((types >= 76).any())


def test_only_moved_rows_get_a_new_column():
    levels = torch.zeros(NUM_ENVS, dtype=torch.long)
    types = initial_mapping_is_round_robin()
    env_ids = torch.tensor([5, 100, 300, 4000])
    # env 5 promotes, env 100 demotes from row 3, 300 and 4000 do nothing.
    levels[100] = 3
    move_up = torch.tensor([True, False, False, False])
    move_down = torch.tensor([False, True, False, False])
    new_levels, new_types, n_moved = resample_after_level_move(
        levels, types, env_ids, move_up, move_down)
    assert n_moved == 2
    assert new_levels[5].item() == 1 and new_levels[100].item() == 2
    # The two moved envs are the ONLY ones rewritten. Draw the resample many times: an env that keeps its
    # column is allowed (1/80 chance per draw), but a moved env must never leave the legal range.
    for _ in range(200):
        _, nt, _ = resample_after_level_move(levels, types, env_ids, move_up, move_down)
        assert 0 <= int(nt[5]) < NUM_COLS and 0 <= int(nt[100]) < NUM_COLS
        assert nt[300].item() == types[300].item()
        assert nt[4000].item() == types[4000].item()
    # Untouched envs must keep their exact column, including envs outside this reset batch.
    assert new_types[300].item() == types[300].item()
    assert new_types[4000].item() == types[4000].item()
    assert new_types[1000].item() == types[1000].item()
    assert new_types[99].item() == types[99].item()


def test_clipped_move_at_the_boundary_does_not_resample():
    """Row 9 promoting further, or row 0 demoting further, is not a scene change."""
    levels = torch.full((NUM_ENVS,), NUM_ROWS - 1, dtype=torch.long)
    types = initial_mapping_is_round_robin()
    env_ids = torch.tensor([7, 8])
    _, _, n_moved = resample_after_level_move(
        levels, types, env_ids, torch.tensor([True, False]), torch.tensor([False, False]))
    assert n_moved == 0  # clipped back to row 9 -> nothing moved -> column kept

    levels0 = torch.zeros(NUM_ENVS, dtype=torch.long)
    _, _, n_moved = resample_after_level_move(
        levels0, types, env_ids, torch.tensor([False, False]), torch.tensor([True, False]))
    assert n_moved == 0  # demotion below row 0 is clipped away too


def test_prefix_column_coverage_reaches_the_whole_pool_within_a_few_row_changes():
    """One column re-roll per level change: how many changes until the prefix spans all 80?"""
    for n_changes in (1, 3, 10):
        seen = set()
        for _ in range(200):  # Monte Carlo over the cohort of 280 prefix envs
            types = initial_mapping_is_round_robin()
            for _ in range(n_changes):
                env_ids = torch.arange(NUM_ENVS)
                move_up = torch.rand(NUM_ENVS) < 0.5
                move_down = torch.zeros(NUM_ENVS, dtype=torch.bool)
                _, types, _ = resample_after_level_move(
                    torch.full((NUM_ENVS,), 4, dtype=torch.long), types, env_ids, move_up, move_down)
            seen |= set(types[:NUM_DEPTH].tolist())
        print(f"prefix cols spanned after {n_changes} re-roll(s)/env: {len(seen)}/{NUM_COLS}")
        assert len(seen) >= {1: 44, 3: 77, 10: 80}[n_changes]


def test_distinct_tile_coverage_of_the_prefix():
    """Tiles, not columns: row x col is what the encoder actually sees. 280 envs span 76-80
    columns but only a few rows at once, so per-iter tile count stays well under 280."""
    torch.manual_seed(0)
    types = initial_mapping_is_round_robin()
    levels = torch.randint(0, 6, (NUM_ENVS,))  # max_init_terrain_level=5, as in the real init
    tiles = set(zip(levels[:NUM_DEPTH].tolist(), types[:NUM_DEPTH].tolist()))
    assert len(tiles) <= NUM_DEPTH
    assert len(tiles) >= 200  # ~250 expected with rows spread over [0,5]
