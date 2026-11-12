import math

import numpy as np
import pytest

from quip_solver_dwave import cascade_codec as codec
from quip_solver_dwave import wins


def test_lane_words_put_read_r_at_bit_r_mod_64_of_word_r_div_64():
    states = np.ones((66, 3), dtype=np.int8)
    states[0, 0] = -1          # lane 0, node 0
    states[63, 1] = -1         # lane 63, node 1
    states[65, 2] = -1         # lane 1 of word 1, node 2
    lanes = codec.lane_words(states, words=2)
    assert lanes.shape == (3, 2)
    assert lanes[0].tolist() == [1, 0]
    assert lanes[1].tolist() == [1 << 63, 0]
    assert lanes[2].tolist() == [0, 2]


def test_lane_words_refuse_too_many_states_and_bad_values():
    with pytest.raises(ValueError, match="do not fit 64 lanes"):
        codec.lane_words(np.ones((65, 2), dtype=np.int8), words=1)
    with pytest.raises(ValueError, match="other than -1 and \\+1"):
        codec.lane_words(np.zeros((1, 2), dtype=np.int8), words=1)


# The Rust miner asserts this string too. The fixture has 9 nodes, 2 words, and 128 lanes. State r has the spin -1 at
# node i when bit i of (37 * r) % 512 is set. The hex holds the little-endian words, node by node, word 0 first.
PINNED_LANES_HEX = (
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"  # node 0
    "cccccccccccccccccccccccccccccccc"  # node 1
    "5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a"  # node 2
    "6c936c936c936c936c936c936c936c93"  # node 3
    "701c8fe3701c8fe3701c8fe3701c8fe3"  # node 4
    "2ab55aa9d54aa5562ab55aa9d54aa556"  # node 5
    "4c2693cd6693c964b3d96c32996c369b"  # node 6
    "70381c0e87e3f1783c1e8fc3e170381c"  # node 7
    "803fe00ff803fe803fe00ffc017fc01f"  # node 8
)


def test_encode_lanes_pins_the_layout_that_the_miner_parses():
    states = np.array([[-1 if ((37 * r) % 512 >> i) & 1 else 1 for i in range(9)] for r in range(128)], dtype=np.int8)
    assert len(PINNED_LANES_HEX) == 9 * 2 * 16
    assert codec.encode_lanes(states, 2) == PINNED_LANES_HEX


def test_reads_round_trip_and_use_a_set_bit_for_minus_one():
    rng = np.random.default_rng(5)
    states = (1 - 2 * rng.integers(0, 2, size=(7, 4577))).astype(np.int8)
    assert (codec.decode_reads(codec.encode_reads(states), 7, 4577) == states).all()
    one = np.ones((1, 9), dtype=np.int8)
    one[0, 0] = one[0, 8] = -1
    assert codec.encode_reads(one) == "0101"


def test_decode_reads_refuses_a_wrong_length():
    with pytest.raises(ValueError, match="do not hold 2 reads"):
        codec.decode_reads(codec.encode_reads(np.ones((1, 16), dtype=np.int8)), 2, 16)


def test_poisson_win_share_counts_only_the_time_left_after_the_latency():
    assert wins.poisson_win_share([100.0], 0.0, 10.0) == 0.0
    assert wins.poisson_win_share([100.0], 1.0, 200.0) == 0.0
    assert wins.poisson_win_share([110.0], 0.01, 10.0) == pytest.approx(1 - math.exp(-1.0))


def test_replay_win_share_needs_the_find_and_the_clear_inside_the_round():
    # One clear, found at 10 s and cleared at 130 s, in a 1,000 s run.
    share = wins.replay_win_share([(10.0, 130.0)], 1000.0, [200.0], step_seconds=10.0)
    # Starts 0 and 10 win: the find is at or after the start and the clear is inside 200 s.
    assert share == pytest.approx(2 / 81)
    assert wins.replay_win_share([(10.0, 130.0)], 1000.0, [100.0], step_seconds=10.0) == 0.0
    assert wins.replay_win_share([], 1000.0, [200.0]) == 0.0


def test_replay_uses_the_earliest_clear_among_later_finds():
    clears = [(10.0, 500.0), (20.0, 60.0)]
    # A round of 100 s that starts at 0 wins through the second clear.
    assert wins.replay_win_share(clears, 100.0, [100.0], step_seconds=10.0) == 1.0


def test_equilibrium_share_is_the_share_of_clearing_power():
    assert wins.equilibrium_share(0.002, 0.002) == 0.5
    assert wins.equilibrium_share(0.0, 0.0) == 0.0
