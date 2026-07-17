"""Execute exact-definition, MNN-equivalence, and n=10,000 sampled audits."""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
import time

import numpy as np

from ordinal_audit import (
    brute_qsi,
    brute_tsi,
    joint_mnn_all_scales,
    no_distance_ties,
    pairwise_distances,
    rank_tsi,
    sampled_qsi,
    sampled_tsi,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(2026)

    # Claim 1: literal ordinal definitions agree with rank computation and are
    # invariant to a distance-order-preserving similarity transform.
    definition_rows = []
    for n in range(4, 9):
        for seed in range(8):
            local = np.random.default_rng(10_000 * n + seed)
            x = local.normal(size=(n, 3))
            rotation, _ = np.linalg.qr(local.normal(size=(3, 3)))
            y = 3.7 * x @ rotation + local.normal(size=(1, 3))
            dx, dy = pairwise_distances(x), pairwise_distances(y)
            brute_tsi_value = brute_tsi(dx, dy)
            definition_rows.append(
                {
                    "n": n,
                    "seed": seed,
                    "brute_tsi": brute_tsi_value,
                    "rank_tsi": rank_tsi(dx, dy),
                    "brute_qsi": brute_qsi(dx, dy),
                    "no_ties": bool(no_distance_ties(dx) and no_distance_ties(dy)),
                }
            )

    # Claim 2: enumerate all label permutations for several no-tie point
    # clouds.  The truth value TSI==1 must equal joint MNN agreement at all k.
    equivalence_rows = []
    for cloud_seed in range(8):
        x = np.random.default_rng(50_000 + cloud_seed).normal(size=(5, 2))
        dx = pairwise_distances(x)
        assert no_distance_ties(dx)
        for permutation in itertools.permutations(range(5)):
            y = x[list(permutation)]
            dy = pairwise_distances(y)
            tsi = brute_tsi(dx, dy)
            mnn_all, minimum_mnn = joint_mnn_all_scales(dx, dy)
            equivalence_rows.append(
                {
                    "cloud_seed": cloud_seed,
                    "permutation": "".join(map(str, permutation)),
                    "tsi": float(tsi),
                    "tsi_perfect": bool(tsi == 1.0),
                    "mnn_all_scales": bool(mnn_all),
                    "minimum_mnn": float(minimum_mnn),
                    "equivalent": bool((tsi == 1.0) == mnn_all),
                }
            )

    # Full-size approximate path advertised by the released implementation.
    n, d, samples = 10_000, 512, 20_000
    x_large = rng.normal(size=(n, d)).astype(np.float32)
    rotation_signs = np.where(np.arange(d) % 2, -1.0, 1.0).astype(np.float32)
    y_aligned = 1.25 * x_large * rotation_signs + 0.7
    permutation = rng.permutation(n)
    y_misaligned = y_aligned[permutation]
    start = time.perf_counter()
    approximate = {
        "aligned_tsi": sampled_tsi(x_large, y_aligned, samples, seed=1),
        "aligned_qsi": sampled_qsi(x_large, y_aligned, samples, seed=2),
        "permuted_tsi": sampled_tsi(x_large, y_misaligned, samples, seed=3),
        "permuted_qsi": sampled_qsi(x_large, y_misaligned, samples, seed=4),
    }
    approximate["runtime_s"] = time.perf_counter() - start
    approximate["n"] = n
    approximate["d"] = d
    approximate["samples_per_metric"] = samples

    # A tie has two admissible nearest-neighbor choices, so the no-ties theorem
    # must refuse that input rather than asserting a false equivalence.
    tie_points = np.array([[0.0], [1.0], [-1.0], [3.0]])
    tie_control_rejected = not no_distance_ties(pairwise_distances(tie_points))

    with (OUT / "definition_and_equivalence.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(definition_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(definition_rows)
    with (OUT / "mnn_equivalence.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(equivalence_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(equivalence_rows)

    summary = {
        "paper": "Scalable and Interpretable Representation Alignment with Ordinal Similarity (G4D0YzzZEk)",
        "official_commit": "79758ece4b97eeda98439542bbeb7c229f2bc2c9",
        "claim_1_ordinal_indices": {
            "instances": len(definition_rows),
            "max_fast_vs_brute_tsi_error": float(max(abs(row["rank_tsi"] - row["brute_tsi"]) for row in definition_rows)),
            "all_similarity_transform_tsi_one": bool(all(row["brute_tsi"] == 1.0 for row in definition_rows)),
            "all_similarity_transform_qsi_one": bool(all(row["brute_qsi"] == 1.0 for row in definition_rows)),
            "verified": bool(all(row["brute_tsi"] == 1.0 and row["brute_qsi"] == 1.0 for row in definition_rows)),
        },
        "claim_2_tsi_mnn_equivalence": {
            "no_tie_cases": len(equivalence_rows),
            "equivalence_matches": int(sum(row["equivalent"] for row in equivalence_rows)),
            "verified": bool(all(row["equivalent"] for row in equivalence_rows)),
        },
        "approximate_n10000_d512": approximate,
        "negative_control_ties_rejected": tie_control_rejected,
    }
    summary["all_claims_verified"] = bool(
        summary["claim_1_ordinal_indices"]["verified"]
        and summary["claim_2_tsi_mnn_equivalence"]["verified"]
        and tie_control_rejected
        and approximate["aligned_tsi"] > 0.999
        and approximate["aligned_qsi"] > 0.999
        and approximate["permuted_tsi"] < 0.7
        and approximate["permuted_qsi"] < 0.7
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
