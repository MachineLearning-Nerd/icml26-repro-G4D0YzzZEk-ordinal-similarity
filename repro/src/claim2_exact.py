"""Exact, fail-closed machinery for the Claim 2 TSI/MutualNN audit.

The paper's no-tie corollary is combinatorial: the distance order around each
anchor is a strict permutation.  TSI=1 exactly when all of those permutations
agree, while MutualNN=1 at every k exactly when all of their prefix sets agree.
This module deliberately computes both sides independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


EXPECTED_ROUTES = tuple(range(1, 11))


def validate_exactly_ten_routes(route_numbers: Iterable[int]) -> None:
    """Reject missing, duplicated, reordered, or additional approaches."""
    observed = tuple(route_numbers)
    if observed != EXPECTED_ROUTES:
        raise ValueError(
            f"Claim 2 requires routes {EXPECTED_ROUTES} exactly once and in order; "
            f"observed {observed}"
        )


def squared_euclidean_distances(points: np.ndarray) -> np.ndarray:
    """Return a float64 squared-distance matrix with an infinite diagonal."""
    points64 = np.asarray(points, dtype=np.float64)
    if points64.ndim != 2 or len(points64) < 3:
        raise ValueError("points must have shape (n, d) with n >= 3")
    norms = np.einsum("ij,ij->i", points64, points64)
    distances = norms[:, None] + norms[None, :] - 2.0 * (points64 @ points64.T)
    np.maximum(distances, 0.0, out=distances)
    np.fill_diagonal(distances, np.inf)
    return distances


def strict_orders(distances: np.ndarray) -> tuple[np.ndarray, bool, float, int]:
    """Sort every anchor and report exact floating-point distance ties."""
    matrix = np.asarray(distances, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distances must be a square matrix")
    n = len(matrix)
    orders = np.argsort(matrix, axis=1, kind="stable")[:, : n - 1]
    ordered = np.take_along_axis(matrix, orders, axis=1)
    gaps = np.diff(ordered, axis=1)
    tie_count = int(np.count_nonzero(gaps <= 0.0))
    minimum_gap = float(np.min(gaps)) if gaps.size else float("inf")
    return orders, tie_count == 0, minimum_gap, tie_count


def _inversion_count(values: np.ndarray) -> int:
    """Count inversions in a permutation with an iterative merge pass."""
    source = [int(value) for value in values]
    n = len(source)
    target = [0] * n
    inversions = 0
    width = 1
    while width < n:
        for start in range(0, n, 2 * width):
            middle = min(start + width, n)
            stop = min(start + 2 * width, n)
            left, right, output = start, middle, start
            while left < middle and right < stop:
                if source[left] <= source[right]:
                    target[output] = source[left]
                    left += 1
                else:
                    target[output] = source[right]
                    inversions += middle - left
                    right += 1
                output += 1
            while left < middle:
                target[output] = source[left]
                left += 1
                output += 1
            while right < stop:
                target[output] = source[right]
                right += 1
                output += 1
        source, target = target, source
        width *= 2
    return inversions


def exact_tsi_from_orders(orders_x: np.ndarray, orders_y: np.ndarray) -> tuple[int, int, float]:
    """Compute exact TSI via discordant distance-pair inversion counts."""
    if orders_x.shape != orders_y.shape:
        raise ValueError("the two strict-order arrays must have equal shape")
    n, width = orders_x.shape
    if width != n - 1:
        raise ValueError("strict-order arrays must have shape (n, n-1)")
    if np.array_equal(orders_x, orders_y):
        comparisons = n * width * (width - 1) // 2
        return 0, int(comparisons), 1.0
    inverse_y = np.empty((n, n), dtype=np.int32)
    rows = np.arange(n)[:, None]
    inverse_y[rows, orders_y] = np.arange(width, dtype=np.int32)
    discordant = 0
    for anchor in range(n):
        discordant += _inversion_count(inverse_y[anchor, orders_x[anchor]])
    comparisons = n * width * (width - 1) // 2
    tsi = 1.0 - discordant / comparisons
    return int(discordant), int(comparisons), float(tsi)


def all_scale_prefix_mnn(
    orders_x: np.ndarray, orders_y: np.ndarray
) -> tuple[bool, int | None, int | None, int]:
    """Independently compare neighbor-set prefixes for every k=1,...,n-2."""
    if orders_x.shape != orders_y.shape:
        raise ValueError("the two strict-order arrays must have equal shape")
    n, width = orders_x.shape
    comparisons = 0
    for anchor in range(n):
        prefix_x = 0
        prefix_y = 0
        for offset in range(width - 1):
            prefix_x |= 1 << int(orders_x[anchor, offset])
            prefix_y |= 1 << int(orders_y[anchor, offset])
            comparisons += 1
            if prefix_x != prefix_y:
                return False, anchor, offset + 1, comparisons
    return True, None, None, comparisons


def mnn_score_at_k(orders_x: np.ndarray, orders_y: np.ndarray, k: int) -> float:
    """Compute the paper's mean mutual-neighbor overlap at one fixed k."""
    n, width = orders_x.shape
    if orders_y.shape != orders_x.shape or not 1 <= k <= width:
        raise ValueError("invalid order arrays or k")
    total_overlap = 0
    for anchor in range(n):
        left = set(int(value) for value in orders_x[anchor, :k])
        right = set(int(value) for value in orders_y[anchor, :k])
        total_overlap += len(left & right)
    return float(total_overlap / (n * k))


@dataclass(frozen=True)
class ExactAudit:
    n: int
    d: int
    no_ties_x: bool
    no_ties_y: bool
    minimum_gap_x: float
    minimum_gap_y: float
    tie_count_x: int
    tie_count_y: int
    tsi_discordant: int | None
    tsi_comparisons: int | None
    tsi: float | None
    tsi_perfect: bool | None
    mnn_all_scales_perfect: bool | None
    first_mnn_mismatch_anchor: int | None
    first_mnn_mismatch_k: int | None
    mnn_prefix_comparisons: int
    equivalence_holds: bool | None
    selected_mnn_scores: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def audit_point_sets(
    points_x: np.ndarray,
    points_y: np.ndarray,
    selected_k: Iterable[int] = (),
) -> ExactAudit:
    """Run the two exact sides, refusing to apply the theorem when ties exist."""
    x = np.asarray(points_x)
    y = np.asarray(points_y)
    if x.shape != y.shape:
        raise ValueError("representations must have identical shape")
    distances_x = squared_euclidean_distances(x)
    distances_y = squared_euclidean_distances(y)
    orders_x, no_ties_x, gap_x, ties_x = strict_orders(distances_x)
    orders_y, no_ties_y, gap_y, ties_y = strict_orders(distances_y)
    if not (no_ties_x and no_ties_y):
        return ExactAudit(
            n=len(x), d=x.shape[1], no_ties_x=no_ties_x, no_ties_y=no_ties_y,
            minimum_gap_x=gap_x, minimum_gap_y=gap_y,
            tie_count_x=ties_x, tie_count_y=ties_y,
            tsi_discordant=None, tsi_comparisons=None, tsi=None, tsi_perfect=None,
            mnn_all_scales_perfect=None, first_mnn_mismatch_anchor=None,
            first_mnn_mismatch_k=None, mnn_prefix_comparisons=0,
            equivalence_holds=None, selected_mnn_scores={},
        )
    discordant, comparisons, tsi = exact_tsi_from_orders(orders_x, orders_y)
    mnn_all, mismatch_anchor, mismatch_k, prefix_comparisons = all_scale_prefix_mnn(
        orders_x, orders_y
    )
    scores = {str(k): mnn_score_at_k(orders_x, orders_y, int(k)) for k in selected_k}
    tsi_perfect = discordant == 0
    return ExactAudit(
        n=len(x), d=x.shape[1], no_ties_x=True, no_ties_y=True,
        minimum_gap_x=gap_x, minimum_gap_y=gap_y,
        tie_count_x=0, tie_count_y=0,
        tsi_discordant=discordant, tsi_comparisons=comparisons, tsi=tsi,
        tsi_perfect=tsi_perfect, mnn_all_scales_perfect=mnn_all,
        first_mnn_mismatch_anchor=mismatch_anchor,
        first_mnn_mismatch_k=mismatch_k,
        mnn_prefix_comparisons=prefix_comparisons,
        equivalence_holds=tsi_perfect == mnn_all,
        selected_mnn_scores=scores,
    )
