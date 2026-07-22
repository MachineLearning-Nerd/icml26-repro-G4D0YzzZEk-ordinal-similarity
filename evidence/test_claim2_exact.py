from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claim2_exact import (
    _inversion_count,
    all_scale_prefix_mnn,
    audit_point_sets,
    exact_tsi_from_orders,
    validate_exactly_ten_routes,
)


def test_route_contract_accepts_exactly_ten_and_rejects_eleven():
    validate_exactly_ten_routes(range(1, 11))
    with pytest.raises(ValueError):
        validate_exactly_ten_routes(range(1, 12))
    with pytest.raises(ValueError):
        validate_exactly_ten_routes([1, 2, 3, 4, 5, 6, 7, 8, 10, 10])


def test_inversion_count_known_permutations():
    assert _inversion_count(np.array([0, 1, 2, 3])) == 0
    assert _inversion_count(np.array([3, 2, 1, 0])) == 6
    assert _inversion_count(np.array([0, 2, 1, 3])) == 1


def test_exact_tsi_and_prefix_mnn_agree_for_all_width_six_permutations():
    import itertools

    identity = np.stack(
        [np.delete(np.arange(7, dtype=np.int32), anchor) for anchor in range(7)]
    )
    for permutation in itertools.permutations(identity[0]):
        candidate = identity.copy()
        candidate[0] = permutation
        discordant, comparisons, tsi = exact_tsi_from_orders(identity, candidate)
        mnn_all, _, _, _ = all_scale_prefix_mnn(identity, candidate)
        assert comparisons == 105
        assert (discordant == 0) == (tsi == 1.0) == mnn_all


def test_similarity_and_deformation_cover_both_directions():
    rng = np.random.default_rng(20)
    x = rng.normal(size=(30, 5))
    aligned = audit_point_sets(x, -x, selected_k=(1, 10))
    assert aligned.equivalence_holds
    assert aligned.tsi_perfect
    assert aligned.mnn_all_scales_perfect

    deformed = x * np.array([0.2, 0.5, 1.0, 2.0, 4.0])
    adverse = audit_point_sets(x, deformed, selected_k=(1, 10))
    assert adverse.equivalence_holds
    assert not adverse.tsi_perfect
    assert not adverse.mnn_all_scales_perfect


def test_tied_input_is_refused_instead_of_forced_through_theorem():
    tied = np.array([[0.0], [1.0], [-1.0], [3.0]])
    audit = audit_point_sets(tied, tied)
    assert not audit.no_ties_x
    assert audit.tsi is None
    assert audit.mnn_all_scales_perfect is None
    assert audit.equivalence_holds is None
