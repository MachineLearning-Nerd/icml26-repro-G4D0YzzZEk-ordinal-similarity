from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ordinal_audit import (
    brute_qsi,
    brute_tsi,
    joint_mnn_all_scales,
    no_distance_ties,
    pairwise_distances,
    rank_tsi,
)


def test_brute_and_rank_tsi_agree_on_no_tie_input():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(7, 3)), rng.normal(size=(7, 3))
    dx, dy = pairwise_distances(x), pairwise_distances(y)
    assert abs(brute_tsi(dx, dy) - rank_tsi(dx, dy)) < 1e-12


def test_ordinal_indices_are_one_for_rotation_scale_and_translation():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(6, 3))
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    y = 2.5 * x @ rotation + 4.0
    dx, dy = pairwise_distances(x), pairwise_distances(y)
    assert brute_tsi(dx, dy) == 1.0
    assert brute_qsi(dx, dy) == 1.0


def test_tsi_perfection_equals_all_scale_mnn_agreement_for_permutation_family():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(5, 2))
    dx = pairwise_distances(x)
    assert no_distance_ties(dx)
    for permutation in [(0, 1, 2, 3, 4), (1, 0, 2, 3, 4), (4, 3, 2, 1, 0)]:
        dy = pairwise_distances(x[list(permutation)])
        tsi_perfect = brute_tsi(dx, dy) == 1.0
        mnn_perfect, _ = joint_mnn_all_scales(dx, dy)
        assert tsi_perfect == mnn_perfect


def test_ties_are_explicitly_rejected_by_the_equivalence_precondition():
    tied = np.array([[0.0], [1.0], [-1.0], [3.0]])
    assert not no_distance_ties(pairwise_distances(tied))
