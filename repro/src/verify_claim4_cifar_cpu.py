#!/usr/bin/env python3
"""Paper-scale CPU reproduction of G4D0YzzZEk Claim 4 (Section 6.2).

The authors release the exact final-epoch CIFAR-10 ViT representations used in
Figure 5b.  This verifier downloads that pinned 10,000 x 512 float32 array,
checks its SHA-256, applies the released 2%-outlier protocol, and measures:

* TSI and QSI with independent uniform ordinal samples and a preregistered
  Hoeffding error bound;
* exact linear CKA;
* exact dot-product k=10 MutualNN and CKNNA.

Only 200 rows change in each run.  The exact-neighbor implementation therefore
precomputes the clean top-(k+200) list once, then recomputes every similarity
that can change.  A reduced brute-force control proves this update is exact.
No wall-clock values are emitted, so stdout is deterministic.

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.2.6", "scipy==1.15.3"]
# ///
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import urllib.request

import numpy as np
from scipy.spatial.distance import cdist, pdist


OFFICIAL_COMMIT = "79758ece4b97eeda98439542bbeb7c229f2bc2c9"
OFFICIAL_SHA256 = "3ee442b20bab44e4aae5a038b6489df988f4db68fc6dd1e95d630ca278a10b1b"
OFFICIAL_URL = (
    "https://raw.githubusercontent.com/diogosoares22/ordinal-similarity-metrics/"
    f"{OFFICIAL_COMMIT}/data/cifar-10-final-epoch-val-representations.npy"
)
EXPECTED_SHAPE = (10_000, 512)
SIGMAS = (0, 1, 2, 4, 8, 16, 32, 64, 128)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_official(path: Path) -> np.ndarray:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(OFFICIAL_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=180) as response, path.open("wb") as handle:
            while block := response.read(1 << 20):
                handle.write(block)
    actual = sha256(path)
    if actual != OFFICIAL_SHA256:
        raise RuntimeError(f"official representation hash mismatch: {actual}")
    x = np.load(path)
    if x.shape != EXPECTED_SHAPE or x.dtype != np.float32:
        raise RuntimeError(f"official representation metadata mismatch: {x.shape}, {x.dtype}")
    if not np.isfinite(x).all():
        raise RuntimeError("official representations contain non-finite values")
    return x


def exact_mean_pairwise_distance(x: np.ndarray, block: int = 256) -> float:
    """Exact mean over all unordered Euclidean pairs, without a 400-MiB vector."""
    total = 0.0
    count = 0
    n = len(x)
    x64 = np.asarray(x, dtype=np.float64)
    for start in range(0, n, block):
        stop = min(start + block, n)
        own = x64[start:stop]
        if len(own) > 1:
            d = pdist(own, metric="euclidean")
            total += float(d.sum(dtype=np.float64))
            count += len(d)
        if stop < n:
            d = cdist(own, x64[stop:], metric="euclidean")
            total += float(d.sum(dtype=np.float64))
            count += d.size
    expected = n * (n - 1) // 2
    if count != expected:
        raise AssertionError((count, expected))
    return total / count


def make_distinct_tuples(n: int, degree: int, samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pieces: list[np.ndarray] = []
    count = 0
    while count < samples:
        need = samples - count
        candidate = rng.integers(0, n, size=(max(need * 2, 4096), degree), dtype=np.int32)
        distinct = np.ones(len(candidate), dtype=bool)
        for a in range(degree):
            for b in range(a + 1, degree):
                distinct &= candidate[:, a] != candidate[:, b]
        accepted = candidate[distinct][:need]
        pieces.append(accepted)
        count += len(accepted)
    return np.concatenate(pieces, axis=0)


def sqdist_rows(z: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = z[a] - z[b]
    return np.einsum("ij,ij->i", delta, delta, optimize=True)


def sampled_ordinal_score(
    x: np.ndarray,
    y: np.ndarray,
    tuples: np.ndarray,
    outlier_mask: np.ndarray,
    degree: int,
    block: int = 20_000,
) -> float:
    affected = np.any(outlier_mask[tuples], axis=1)
    positions = np.flatnonzero(affected)
    disagreements = 0
    for start in range(0, len(positions), block):
        rows = tuples[positions[start : start + block]]
        if degree == 3:
            dx1 = sqdist_rows(x, rows[:, 0], rows[:, 1])
            dx2 = sqdist_rows(x, rows[:, 0], rows[:, 2])
            dy1 = sqdist_rows(y, rows[:, 0], rows[:, 1])
            dy2 = sqdist_rows(y, rows[:, 0], rows[:, 2])
        elif degree == 4:
            dx1 = sqdist_rows(x, rows[:, 0], rows[:, 1])
            dx2 = sqdist_rows(x, rows[:, 2], rows[:, 3])
            dy1 = sqdist_rows(y, rows[:, 0], rows[:, 1])
            dy2 = sqdist_rows(y, rows[:, 2], rows[:, 3])
        else:
            raise ValueError(degree)
        disagreements += int(np.count_nonzero(np.sign(dx1 - dx2) != np.sign(dy1 - dy2)))
    return 1.0 - disagreements / len(tuples)


def topk_dot_all(x: np.ndarray, k: int, block: int = 512) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    indices = np.empty((n, k), dtype=np.int32)
    values = np.empty((n, k), dtype=np.float32)
    xt = x.T
    for start in range(0, n, block):
        stop = min(start + block, n)
        scores = x[start:stop] @ xt
        local = np.arange(stop - start)
        scores[local, np.arange(start, stop)] = -np.inf
        selected = np.argpartition(scores, -k, axis=1)[:, -k:]
        selected_values = np.take_along_axis(scores, selected, axis=1)
        order = np.argsort(-selected_values, axis=1, kind="stable")
        indices[start:stop] = np.take_along_axis(selected, order, axis=1)
        values[start:stop] = np.take_along_axis(selected_values, order, axis=1)
    return indices, values


def updated_topk_exact(
    x: np.ndarray,
    y: np.ndarray,
    outliers: np.ndarray,
    clean_candidates: np.ndarray,
    clean_values: np.ndarray,
    k: int,
) -> np.ndarray:
    """Exact Y dot-product top-k after changing only ``outliers`` rows."""
    n = len(x)
    changed = np.zeros(n, dtype=bool)
    changed[outliers] = True
    normal = np.flatnonzero(~changed)
    result = np.empty((n, k), dtype=np.int32)

    # For unchanged queries, only similarities to changed candidate rows can
    # differ.  Removing at most o candidates from a clean top-(k+o) list leaves
    # at least k exact unchanged candidates.
    candidate_scores = x[normal] @ y[outliers].T
    for local, row in enumerate(normal):
        keep = ~changed[clean_candidates[row]]
        base_idx = clean_candidates[row, keep][:k]
        base_val = clean_values[row, keep][:k]
        idx = np.concatenate((base_idx, outliers.astype(np.int32, copy=False)))
        val = np.concatenate((base_val, candidate_scores[local]))
        chosen = np.argpartition(val, -k)[-k:]
        order = np.argsort(-val[chosen], kind="stable")
        result[row] = idx[chosen[order]]

    # Every similarity for a changed query can differ, so recompute its whole
    # row.  This closes the only other route by which a top-k list can change.
    scores = y[outliers] @ y.T
    scores[np.arange(len(outliers)), outliers] = -np.inf
    selected = np.argpartition(scores, -k, axis=1)[:, -k:]
    selected_values = np.take_along_axis(scores, selected, axis=1)
    order = np.argsort(-selected_values, axis=1, kind="stable")
    result[outliers] = np.take_along_axis(selected, order, axis=1)
    return result


def mutualnn(knn_x: np.ndarray, knn_y: np.ndarray) -> float:
    overlaps = np.empty(len(knn_x), dtype=np.float64)
    for i in range(len(knn_x)):
        overlaps[i] = len(np.intersect1d(knn_x[i], knn_y[i], assume_unique=True)) / knn_x.shape[1]
    return float(overlaps.mean())


def edge_arrays(knn_a: np.ndarray, knn_b: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for i in range(len(knn_a)):
        js = knn_a[i] if knn_b is None else np.intersect1d(knn_a[i], knn_b[i], assume_unique=True)
        if len(js):
            rows.append(np.full(len(js), i, dtype=np.int32))
            cols.append(np.asarray(js, dtype=np.int32))
    return np.concatenate(rows), np.concatenate(cols)


def sparse_hsic(x: np.ndarray, y: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> float:
    n = len(x)
    wx = np.einsum("ij,ij->i", x[rows], x[cols], optimize=True).astype(np.float64)
    wy = np.einsum("ij,ij->i", y[rows], y[cols], optimize=True).astype(np.float64)
    codes = rows.astype(np.int64) * n + cols
    reverse = cols.astype(np.int64) * n + rows
    ordered = np.sort(codes)
    reverse_present = np.searchsorted(ordered, reverse)
    valid = reverse_present < len(ordered)
    valid[valid] &= ordered[reverse_present[valid]] == reverse[valid]
    first = float(np.dot(wx[valid], wy[valid]))
    second = float(wx.sum() * wy.sum() / ((n - 1) * (n - 2)))
    col_x = np.bincount(cols, weights=wx, minlength=n)
    row_y = np.bincount(rows, weights=wy, minlength=n)
    third = float(-2.0 * np.dot(col_x, row_y) / (n - 2))
    return (first + second + third) / (n * (n - 3))


def cknna(x: np.ndarray, y: np.ndarray, knn_x: np.ndarray, knn_y: np.ndarray) -> float:
    rxy, cxy = edge_arrays(knn_x, knn_y)
    rxx, cxx = edge_arrays(knn_x)
    ryy, cyy = edge_arrays(knn_y)
    cross = sparse_hsic(x, y, rxy, cxy)
    self_x = sparse_hsic(x, x, rxx, cxx)
    self_y = sparse_hsic(y, y, ryy, cyy)
    return float(cross / math.sqrt(self_x * self_y))


class FastLinearCKA:
    def __init__(self, x: np.ndarray):
        self.x = np.asarray(x, dtype=np.float64)
        self.n = len(x)
        self.mean_x = self.x.mean(axis=0)
        self.raw_xx = self.x.T @ self.x
        self.center_xx = self.raw_xx - self.n * np.outer(self.mean_x, self.mean_x)
        self.norm_xx = np.linalg.norm(self.center_xx)

    def __call__(self, outliers: np.ndarray, delta: np.ndarray) -> float:
        xo = self.x[outliers]
        cross_update = xo.T @ delta
        mean_y = self.mean_x + delta.sum(axis=0) / self.n
        raw_xy = self.raw_xx + cross_update
        center_xy = raw_xy - self.n * np.outer(self.mean_x, mean_y)
        raw_yy = self.raw_xx + cross_update + cross_update.T + delta.T @ delta
        center_yy = raw_yy - self.n * np.outer(mean_y, mean_y)
        return float(np.linalg.norm(center_xy) ** 2 / (self.norm_xx * np.linalg.norm(center_yy)))


def perturb(x: np.ndarray, sigma: int, mean_distance: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    outliers = rng.choice(len(x), size=200, replace=False)
    directions = rng.normal(size=(len(outliers), x.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
    delta = (sigma * mean_distance * directions).astype(np.float32)
    y = x.copy()
    y[outliers] = y[outliers] + delta
    # Re-read the effective float32 update, matching the released script's
    # in-place assignment semantics.
    delta = (y[outliers] - x[outliers]).astype(np.float64)
    return y, np.asarray(outliers, dtype=np.int32), delta


def algorithm_controls(x: np.ndarray, k: int) -> tuple[bool, float]:
    z = x[:512].copy()
    y, outliers, _ = perturb(z, 128, exact_mean_pairwise_distance(z), 91)
    # perturb() uses 200 changed rows, so k+200 is the sufficient clean prefix.
    base_idx, base_val = topk_dot_all(z, k + 200, block=128)
    updated = updated_topk_exact(z, y, outliers, base_idx, base_val, k)
    brute, _ = topk_dot_all(y, k, block=128)
    neighbor_ok = bool(np.array_equal(updated, brute))

    direct_x = z.astype(np.float64) - z.astype(np.float64).mean(axis=0)
    direct_y = y.astype(np.float64) - y.astype(np.float64).mean(axis=0)
    direct = float(
        np.linalg.norm(direct_y.T @ direct_x) ** 2
        / (np.linalg.norm(direct_x.T @ direct_x) * np.linalg.norm(direct_y.T @ direct_y))
    )
    _, _, delta = perturb(z, 128, exact_mean_pairwise_distance(z), 91)
    fast = FastLinearCKA(z)(outliers, delta)
    return neighbor_ok, abs(direct - fast)


def run(args: argparse.Namespace) -> dict[str, object]:
    x = load_official(args.data)
    n, d = x.shape
    k = 10
    mean_distance = exact_mean_pairwise_distance(x)
    samples = args.ordinal_samples
    failure_probability = 1e-6
    epsilon = math.sqrt(math.log(2.0 / failure_probability) / (2.0 * samples))
    tsi_tuples = make_distinct_tuples(n, 3, samples, 42_001)
    qsi_tuples = make_distinct_tuples(n, 4, samples, 42_002)

    neighbor_ok, cka_control_error = algorithm_controls(x, k)
    if not neighbor_ok or cka_control_error > 1e-9:
        raise RuntimeError((neighbor_ok, cka_control_error))

    clean_candidates, clean_values = topk_dot_all(x, k + 200, block=args.knn_block)
    clean_knn = clean_candidates[:, :k].copy()
    cka_metric = FastLinearCKA(x)
    rows: list[dict[str, float | int]] = []

    for sigma in args.sigmas:
        for run_idx in range(args.runs):
            run_seed = run_idx + sigma * 1000
            y, outliers, delta = perturb(x, sigma, mean_distance, run_seed + 10_000)
            changed = np.zeros(n, dtype=bool)
            changed[outliers] = True
            if sigma == 0:
                tsi = qsi = cka = mnn = cknn = 1.0
            else:
                tsi = sampled_ordinal_score(x, y, tsi_tuples, changed, 3)
                qsi = sampled_ordinal_score(x, y, qsi_tuples, changed, 4)
                cka = cka_metric(outliers, delta)
                y_knn = updated_topk_exact(
                    x, y, outliers, clean_candidates, clean_values, k
                )
                mnn = mutualnn(clean_knn, y_knn)
                cknn = cknna(x, y, clean_knn, y_knn)
            rows.append(
                {
                    "sigma": sigma,
                    "run": run_idx,
                    "TSI": tsi,
                    "QSI": qsi,
                    "CKA": cka,
                    "CKNNA": cknn,
                    "MutualNN": mnn,
                }
            )

    summary: list[dict[str, float | int]] = []
    for sigma in args.sigmas:
        selected = [row for row in rows if row["sigma"] == sigma]
        record: dict[str, float | int] = {"sigma": sigma}
        for metric in ("TSI", "QSI", "CKA", "CKNNA", "MutualNN"):
            record[metric] = float(np.mean([float(row[metric]) for row in selected]))
        summary.append(record)

    u = n - 200
    tsi_bound = u * (u - 1) * (u - 2) / (n * (n - 1) * (n - 2))
    qsi_bound = tsi_bound * (u - 3) / (n - 3)
    final = summary[-1]
    sigma0 = summary[0]
    zero_control = all(abs(float(sigma0[m]) - 1.0) < 1e-12 for m in ("TSI", "QSI", "CKA", "CKNNA", "MutualNN"))
    lower_bound_gate = (
        float(final["TSI"]) + epsilon >= tsi_bound
        and float(final["QSI"]) + epsilon >= qsi_bound
    )
    dominance_gate = all(
        min(float(final["TSI"]), float(final["QSI"]))
        > float(final[baseline]) + 2.0 * epsilon
        for baseline in ("CKA", "CKNNA", "MutualNN")
    )
    verdict = bool(zero_control and lower_bound_gate and dominance_gate and neighbor_ok and cka_control_error <= 1e-9)
    result: dict[str, object] = {
        "official_commit": OFFICIAL_COMMIT,
        "official_sha256": OFFICIAL_SHA256,
        "shape": [n, d],
        "runs": args.runs,
        "sigmas": list(args.sigmas),
        "outliers": 200,
        "k": k,
        "ordinal_samples_per_estimate": samples,
        "hoeffding_epsilon_at_delta_1e-6": epsilon,
        "mean_pairwise_distance": mean_distance,
        "summary": summary,
        "tsi_clean_subset_lower_bound": tsi_bound,
        "qsi_clean_subset_lower_bound": qsi_bound,
        "partial_topk_equals_brute_control": neighbor_ok,
        "fast_cka_max_abs_error_control": cka_control_error,
        "sigma0_identity_control": zero_control,
        "lower_bound_gate": lower_bound_gate,
        "ordinal_beats_all_three_baselines_gate": dominance_gate,
        "claim4_verified": verdict,
    }
    return result


def print_result(result: dict[str, object]) -> None:
    print("paper: Scalable and Interpretable Representation Alignment with Ordinal Similarity (G4D0YzzZEk)")
    print(f"source: ar5iv Section 6.2 + Appendix L; official commit {result['official_commit']}")
    print(f"official array: shape={tuple(result['shape'])} sha256={result['official_sha256']}")
    print(
        "protocol: final-epoch seed-0 CIFAR-10 ViT; 2%=200 outliers; "
        f"sigmas={result['sigmas']}; runs={result['runs']}; dot-product k={result['k']}"
    )
    print(
        f"ordinal estimator: samples={result['ordinal_samples_per_estimate']} per cell; "
        f"Hoeffding epsilon={float(result['hoeffding_epsilon_at_delta_1e-6']):.6f} at delta=1e-6"
    )
    print(f"exact mean pairwise distance={float(result['mean_pairwise_distance']):.6f}")
    print("sigma      TSI      QSI      CKA    CKNNA MutualNN")
    for row in result["summary"]:
        print(
            f"{int(row['sigma']):>5} "
            f"{float(row['TSI']):8.3f} {float(row['QSI']):8.3f} "
            f"{float(row['CKA']):8.3f} {float(row['CKNNA']):8.3f} {float(row['MutualNN']):8.3f}"
        )
    print(
        f"clean-subset bounds: TSI={float(result['tsi_clean_subset_lower_bound']):.6f} "
        f"QSI={float(result['qsi_clean_subset_lower_bound']):.6f}"
    )
    print(
        "controls: partial-topk==brute="
        f"{result['partial_topk_equals_brute_control']}; "
        f"fast-CKA max abs error={float(result['fast_cka_max_abs_error_control']):.3e}; "
        f"sigma0 identity={result['sigma0_identity_control']}"
    )
    print(
        f"gates: clean-subset lower bounds={result['lower_bound_gate']}; "
        f"TSI/QSI beat CKA+CKNNA+MutualNN={result['ordinal_beats_all_three_baselines_gate']}"
    )
    print(f"verdict: {'supports' if result['claim4_verified'] else 'inconclusive'}")
    # Hash the same precision exposed above.  Raw BLAS reductions can differ in
    # insignificant trailing bits across CPUs even when every printed value and
    # gate is identical; excluding those hidden bits makes byte parity portable.
    digest_record = {
        "official_commit": result["official_commit"],
        "official_sha256": result["official_sha256"],
        "shape": result["shape"],
        "runs": result["runs"],
        "sigmas": result["sigmas"],
        "outliers": result["outliers"],
        "k": result["k"],
        "ordinal_samples_per_estimate": result["ordinal_samples_per_estimate"],
        "hoeffding_epsilon": round(float(result["hoeffding_epsilon_at_delta_1e-6"]), 6),
        "mean_pairwise_distance": round(float(result["mean_pairwise_distance"]), 6),
        "summary": [
            {
                "sigma": int(row["sigma"]),
                **{metric: round(float(row[metric]), 3) for metric in ("TSI", "QSI", "CKA", "CKNNA", "MutualNN")},
            }
            for row in result["summary"]
        ],
        "tsi_bound": round(float(result["tsi_clean_subset_lower_bound"]), 6),
        "qsi_bound": round(float(result["qsi_clean_subset_lower_bound"]), 6),
        "partial_topk_equals_brute_control": result["partial_topk_equals_brute_control"],
        "fast_cka_error": f"{float(result['fast_cka_max_abs_error_control']):.3e}",
        "sigma0_identity_control": result["sigma0_identity_control"],
        "lower_bound_gate": result["lower_bound_gate"],
        "ordinal_beats_all_three_baselines_gate": result["ordinal_beats_all_three_baselines_gate"],
        "claim4_verified": result["claim4_verified"],
    }
    canonical = json.dumps(digest_record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    print("RESULTS_SHA256=" + hashlib.sha256(canonical.encode()).hexdigest())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/tmp/cifar-10-final-epoch-val-representations.npy"),
    )
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--sigmas", type=int, nargs="+", default=list(SIGMAS))
    parser.add_argument("--ordinal-samples", type=int, default=500_000)
    parser.add_argument("--knn-block", type=int, default=512)
    args = parser.parse_args()
    if args.runs < 1 or args.ordinal_samples < 10_000:
        raise SystemExit("runs must be >=1 and ordinal samples >=10000")
    result = run(args)
    print_result(result)
    return 0 if result["claim4_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
