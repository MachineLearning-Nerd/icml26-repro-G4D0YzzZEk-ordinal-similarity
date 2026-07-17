"""Independent finite and sampled audits for ordinal representation alignment."""

from __future__ import annotations

from itertools import combinations, permutations
from typing import Iterable

import numpy as np


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    differences = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.einsum("...d,...d->...", differences, differences))


def no_distance_ties(distances: np.ndarray, atol: float = 1e-10) -> bool:
    n = len(distances)
    for anchor in range(n):
        values = np.sort(np.delete(distances[anchor], anchor))
        if np.any(np.diff(values) <= atol):
            return False
    return True


def brute_tsi(distances_x: np.ndarray, distances_y: np.ndarray) -> float:
    """Literal ordered-triplet definition, independent of the released code."""
    n = len(distances_x)
    agreements = 0
    for i, j, k in permutations(range(n), 3):
        left = np.sign(distances_x[i, j] - distances_x[i, k])
        right = np.sign(distances_y[i, j] - distances_y[i, k])
        agreements += left == right
    return agreements / (n * (n - 1) * (n - 2))


def brute_qsi(distances_x: np.ndarray, distances_y: np.ndarray) -> float:
    """Literal ordered-quadruplet definition."""
    n = len(distances_x)
    agreements = 0
    for i, j, k, ell in permutations(range(n), 4):
        left = np.sign(distances_x[i, j] - distances_x[k, ell])
        right = np.sign(distances_y[i, j] - distances_y[k, ell])
        agreements += left == right
    return agreements / (n * (n - 1) * (n - 2) * (n - 3))


def rank_tsi(distances_x: np.ndarray, distances_y: np.ndarray) -> float:
    """Rank-based TSI computation using unordered pairs at every anchor."""
    n = len(distances_x)
    agreements = 0
    possible = 0
    for anchor in range(n):
        indices = [index for index in range(n) if index != anchor]
        rank_x = np.argsort(np.argsort(distances_x[anchor, indices], kind="stable"), kind="stable")
        rank_y = np.argsort(np.argsort(distances_y[anchor, indices], kind="stable"), kind="stable")
        for left, right in combinations(range(n - 1), 2):
            agreements += (rank_x[left] - rank_x[right]) * (rank_y[left] - rank_y[right]) > 0
            possible += 1
    return agreements / possible


def joint_mnn_all_scales(distances_x: np.ndarray, distances_y: np.ndarray) -> tuple[bool, float]:
    """Return whether all k-neighborhood sets agree and the smallest MNN score."""
    n = len(distances_x)
    minimum = 1.0
    all_match = True
    for anchor in range(n):
        indices = np.array([index for index in range(n) if index != anchor])
        order_x = indices[np.argsort(distances_x[anchor, indices], kind="stable")]
        order_y = indices[np.argsort(distances_y[anchor, indices], kind="stable")]
        for k in range(1, n - 1):
            neighbors_x = set(order_x[:k])
            neighbors_y = set(order_y[:k])
            overlap = len(neighbors_x & neighbors_y) / k
            minimum = min(minimum, overlap)
            all_match = all_match and neighbors_x == neighbors_y
    return all_match, minimum


def _sample_distinct(rng: np.random.Generator, n: int, width: int, count: int) -> np.ndarray:
    chunks = []
    remaining = count
    while remaining:
        candidates = rng.integers(0, n, size=(max(remaining * 2, 64), width))
        valid = candidates[np.all(np.diff(np.sort(candidates, axis=1), axis=1) != 0, axis=1)]
        take = min(remaining, len(valid))
        chunks.append(valid[:take])
        remaining -= take
    return np.concatenate(chunks, axis=0)


def sampled_tsi(points_x: np.ndarray, points_y: np.ndarray, samples: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    triples = _sample_distinct(rng, len(points_x), 3, samples)
    left = np.linalg.norm(points_x[triples[:, 0]] - points_x[triples[:, 1]], axis=1)
    right = np.linalg.norm(points_x[triples[:, 0]] - points_x[triples[:, 2]], axis=1)
    left_y = np.linalg.norm(points_y[triples[:, 0]] - points_y[triples[:, 1]], axis=1)
    right_y = np.linalg.norm(points_y[triples[:, 0]] - points_y[triples[:, 2]], axis=1)
    return float(np.mean(np.sign(left - right) == np.sign(left_y - right_y)))


def sampled_qsi(points_x: np.ndarray, points_y: np.ndarray, samples: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    quads = _sample_distinct(rng, len(points_x), 4, samples)
    first = np.linalg.norm(points_x[quads[:, 0]] - points_x[quads[:, 1]], axis=1)
    second = np.linalg.norm(points_x[quads[:, 2]] - points_x[quads[:, 3]], axis=1)
    first_y = np.linalg.norm(points_y[quads[:, 0]] - points_y[quads[:, 1]], axis=1)
    second_y = np.linalg.norm(points_y[quads[:, 2]] - points_y[quads[:, 3]], axis=1)
    return float(np.mean(np.sign(first - second) == np.sign(first_y - second_y)))
