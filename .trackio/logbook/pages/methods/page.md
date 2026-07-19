# Methods


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_a2cbf26d4377", "created_at": "2026-07-17T06:40:20+00:00", "title": "Reference and independent methods"}
-->
Official source: https://github.com/diogosoares22/ordinal-similarity-metrics at commit 79758ece4b97eeda98439542bbeb7c229f2bc2c9. Its complete upstream sanity suite passes unchanged.

The independent audit does not call the released metric classes. It computes pairwise distances, literal ordered-triplet and quadruplet predicates, rank agreement, and all-scale MutualNN sets directly with NumPy.

## Claim 2 paper-scale repair

`claim2_exact.py` computes float64 squared distances, verifies strict no-tie
orders, counts exact discordant distance pairs, and independently compares
neighbor-set prefixes for every k=1,...,N−2. The run is fail-closed unless its
route sequence is exactly 1,...,10; an attempted route 11 raises an error.

`verify_claim2_evidence.py` deliberately does not import that module. It
recomputes the 9! combinatorial kernel, directly checks representative anchors
in both pinned CIFAR arrays, verifies all saved integer TSI arithmetic and
positive/negative directions, and records SHA-256 hashes.

Evidence files:

- `outputs/claim2_exactly10/summary.json`
- `outputs/claim2_exactly10/routes.csv`
- `outputs/claim2_exactly10/verification.json`


---
<!-- trackio-cell
{"type": "code", "id": "cell_31c65787de09", "created_at": "2026-07-17T06:40:31+00:00", "title": "Final independent ordinal audit", "command": ["python", "repro/src/run_ordinal_audit.py"], "exit_code": 0, "duration_s": 1.142}
-->
````bash
$ python repro/src/run_ordinal_audit.py
````

exit 0 · 1.1s


````python title=run_ordinal_audit.py
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

````


````output
{
  "paper": "Scalable and Interpretable Representation Alignment with Ordinal Similarity (G4D0YzzZEk)",
  "official_commit": "79758ece4b97eeda98439542bbeb7c229f2bc2c9",
  "claim_1_ordinal_indices": {
    "instances": 40,
    "max_fast_vs_brute_tsi_error": 0.0,
    "all_similarity_transform_tsi_one": true,
    "all_similarity_transform_qsi_one": true,
    "verified": true
  },
  "claim_2_tsi_mnn_equivalence": {
    "no_tie_cases": 960,
    "equivalence_matches": 960,
    "verified": true
  },
  "approximate_n10000_d512": {
    "aligned_tsi": 0.99995,
    "aligned_qsi": 1.0,
    "permuted_tsi": 0.5061,
    "permuted_qsi": 0.49995,
    "runtime_s": 0.6803762700874358,
    "n": 10000,
    "d": 512,
    "samples_per_metric": 20000
  },
  "negative_control_ties_rejected": true,
  "all_claims_verified": true
}

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_fc5ccef96009", "created_at": "2026-07-17T06:40:31+00:00", "title": "Artifact: mnn_equivalence.csv", "path": "outputs/mnn_equivalence.csv", "size": 41422, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/mnn_equivalence.csv` · dataset · 41.4 kB

trackio-local-path://outputs/mnn_equivalence.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_1b362a4cde99", "created_at": "2026-07-17T06:40:31+00:00", "title": "Artifact: definition_and_equivalence.csv", "path": "outputs/definition_and_equivalence.csv", "size": 884, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/definition_and_equivalence.csv` · dataset · 884 B

trackio-local-path://outputs/definition_and_equivalence.csv


---
<!-- trackio-cell
{"type": "code", "id": "cell_08ecaf34e032", "created_at": "2026-07-17T06:40:50+00:00", "title": "Pinned official sanity suite", "command": ["env", "PYTHONPATH=official", "python", "official/src/sanity_tests.py"], "exit_code": 0, "duration_s": 18.555}
-->
````bash
$ env PYTHONPATH=official python official/src/sanity_tests.py
````

exit 0 · 18.6s


````python title=sanity_tests.py
#!/usr/bin/env python3

import argparse
from src.data import RepresentationPair
from src.tsi import TSI, EfficientTSI, ApproxTSI, BatchedTSI
from src.qsi import QSI, ApproxQSI, EfficientQSI, BatchedQSI
import numpy as np

EPSILON = 0.01
DELTA = 0.001

class Example:
    def __init__(self, representations: RepresentationPair, expected_tsi, expected_qsi, name, indices=None):
        self.name = name
        self.representations = representations
        self.expected_tsi = expected_tsi
        self.expected_qsi = expected_qsi
        self.indices = indices

d_x = lambda x, y: np.linalg.norm(x - y)
d_y = lambda x, y: np.linalg.norm(x - y)

curated_example_1 = Example(RepresentationPair(X=np.array([[0, 0], [1, 0], [0, 1]]), Y=np.array([[0, 0], [1, 0], [1, 1]]), d_x=d_x, d_y=d_y), 0.0, None, "curated_example_1")
curated_example_2 = Example(RepresentationPair(X=np.array([[0, 0], [1, 0], [0, 1], [1, 1]]), Y=np.array([[0, 0], [1, 0], [0, 1], [2, 2]]), d_x=d_x, d_y=d_y), 2/3, 0, "curated_example_2")
curated_example_3 = Example(RepresentationPair(X=np.array([[0, 0], [2, 0], [3, 0]]), Y=np.array([[0, 0], [2, 0], [1, 0]]), d_x=d_x, d_y=d_y), 1/3, None,"curated_example_3")

### TSI tests ###

def test_tsi_on_example(example):
    tsi = TSI()
    assert tsi(example.representations) == example.expected_tsi
    print(f"TSI on example {example.name} test passed")

def test_tsi_on_identical_data():
    X = np.random.rand(30, 3)
    Y = X
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    tsi = TSI()
    assert tsi(representations) == 1.0
    print("TSI on identical data test passed")

### EfficientTSI tests ###

def test_efficient_tsi_on_example(example):
    efficient_tsi = EfficientTSI(euclidean=False)
    assert efficient_tsi(example.representations) == example.expected_tsi
    print(f"EfficientTSI on example {example.name} test passed")

def test_efficient_tsi_alignment_with_tsi_on_random_data():
    X = np.random.rand(30, 3)
    Y = np.random.rand(30, 3)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    efficient_tsi = EfficientTSI(euclidean=False)
    tsi = TSI()
    assert np.abs(efficient_tsi(representations) - tsi(representations)) <= 0.000001
    print("EfficientTSI alignment with TSI on random data test passed")

def test_efficient_tsi_alignment_with_tsi_on_random_data_with_equalities():
    X = np.random.rand(30, 3)
    Y = np.concatenate((X[:15], X[:15]), axis=0)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    efficient_tsi = EfficientTSI(euclidean=False)
    tsi = TSI()
    assert np.abs(efficient_tsi(representations) - tsi(representations)) <= 0.000001
    print("EfficientTSI alignment with TSI test on random data with equalities passed")

### ApproxTSI tests ###

def test_approx_tsi_on_example(example):
    approx_tsi = ApproxTSI(epsilon=EPSILON, delta=DELTA)
    value = approx_tsi(example.representations)
    assert abs(value - example.expected_tsi) <= EPSILON * 3 # 3 is a tolerance factor
    print(f"ApproxTSI on example {example.name} within tolerance test passed")

def test_approx_tsi_alignment_with_tsi_on_random_data():
    X = np.random.rand(30, 3)
    Y = np.random.rand(30, 3)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    approx_tsi = ApproxTSI(epsilon=EPSILON, delta=DELTA)
    tsi = TSI()
    approx_value = approx_tsi(representations)
    exact_value = tsi(representations)
    assert abs(approx_value - exact_value) <= EPSILON * 3 # 3 is a tolerance factor
    print("ApproxTSI alignment with TSI on random data within tolerance test passed")

### BatchedTSI tests ###

def test_efficient_approx_tsi_on_example(example):
    efficient_approx_tsi = BatchedTSI(seed=42)
    value = efficient_approx_tsi(example.representations)
    assert abs(value - example.expected_tsi) <= 0.1
    print(f"BatchedTSI on example {example.name} within tolerance test passed")

def test_efficient_approx_tsi_alignment_with_tsi_on_random_data():
    X = np.random.rand(30, 3)
    Y = np.random.rand(30, 3)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    efficient_approx_tsi = BatchedTSI(seed=42)
    tsi = TSI()
    approx_value = efficient_approx_tsi(representations)
    exact_value = tsi(representations)
    assert abs(approx_value - exact_value) <= 0.1
    print("BatchedTSI alignment with TSI on random data within tolerance test passed")

### QSI tests ###

def test_qsi_on_example(example):
    qsi = QSI()
    assert qsi(example.representations) == example.expected_qsi
    print(f"QSI on example {example.name} test passed")

def test_qsi_on_identical_data():
    X = np.random.rand(20, 3)
    Y = X
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    qsi = QSI()
    assert qsi(representations) == 1.0
    print("QSI on identical data test passed")

### EfficientQSI tests ###

def test_efficient_qsi_on_example(example):
    efficient_qsi = EfficientQSI(euclidean=False)
    assert efficient_qsi(example.representations) == example.expected_qsi
    print(f"EfficientQSI on example {example.name} test passed")

def test_efficient_qsi_alignment_with_qsi_on_random_data():
    X = np.random.rand(20, 3)
    Y = np.random.rand(20, 3)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    efficient_qsi = EfficientQSI(euclidean=False)
    qsi = QSI()
    assert np.abs(efficient_qsi(representations) - qsi(representations)) <= 0.000001
    print("EfficientQSI alignment with QSI on random data test passed")

### ApproxQSI tests ###

def test_approx_qsi_on_example(example):
    approx_qsi = ApproxQSI(epsilon=EPSILON, delta=DELTA)
    value = approx_qsi(example.representations)
    assert abs(value - example.expected_qsi) <= EPSILON * 3 # 3 is a tolerance factor
    print(f"ApproxQSI on example {example.name} within tolerance test passed")

def test_approx_qsi_alignment_with_qsi_on_random_data():
    X = np.random.rand(20, 3)
    Y = np.random.rand(20, 3)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    qsi = QSI()
    approx_qsi = ApproxQSI(epsilon=EPSILON, delta=DELTA)
    exact_value = qsi(representations)
    approx_value = approx_qsi(representations)
    assert abs(approx_value - exact_value) <= EPSILON * 3 # 3 is a tolerance factor
    print("ApproxQSI alignment with QSI on random data within tolerance test passed")

### BatchedQSI tests ###

def test_efficient_approx_qsi_on_example(example):
    efficient_approx_qsi = BatchedQSI(seed=42)
    value = efficient_approx_qsi(example.representations)
    assert abs(value - example.expected_qsi) <= 0.1
    print(f"BatchedQSI on example {example.name} within tolerance test passed")

def test_efficient_approx_qsi_alignment_with_qsi_on_random_data():
    X = np.random.rand(20, 3)
    Y = np.random.rand(20, 3)
    d_x = lambda x, y: np.linalg.norm(x - y)
    d_y = lambda x, y: np.linalg.norm(x - y)
    representations = RepresentationPair(X, Y, d_x, d_y)
    qsi = QSI()
    efficient_approx_qsi = BatchedQSI(seed=42)
    exact_value = qsi(representations)
    approx_value = efficient_approx_qsi(representations)
    assert abs(approx_value - exact_value) <= 0.1
    print("BatchedQSI alignment with QSI on random data within tolerance test passed")

def run_tsi_tests():
    """Run tests for TSI implementation"""
    test_tsi_on_example(curated_example_1)
    test_tsi_on_example(curated_example_2)
    test_tsi_on_example(curated_example_3)
    test_tsi_on_identical_data()

def run_efficient_tsi_tests():
    """Run tests for EfficientTSI implementation"""
    test_efficient_tsi_on_example(curated_example_1)
    test_efficient_tsi_on_example(curated_example_2)
    test_efficient_tsi_on_example(curated_example_3)
    test_efficient_tsi_alignment_with_tsi_on_random_data()
    test_efficient_tsi_alignment_with_tsi_on_random_data_with_equalities()

def run_approx_tsi_tests():
    """Run tests for ApproxTSI implementation"""
    test_approx_tsi_on_example(curated_example_1)
    test_approx_tsi_on_example(curated_example_2)
    test_approx_tsi_on_example(curated_example_3)
    test_approx_tsi_alignment_with_tsi_on_random_data()

def run_qsi_tests():
    """Run tests for QSI implementation"""
    test_qsi_on_example(curated_example_2)
    test_qsi_on_identical_data()

def run_efficient_qsi_tests():
    """Run tests for EfficientQSI implementation"""
    test_efficient_qsi_on_example(curated_example_2)
    test_efficient_qsi_alignment_with_qsi_on_random_data()

def run_approx_qsi_tests():
    """Run tests for ApproxQSI implementation"""
    test_approx_qsi_on_example(curated_example_2)
    test_approx_qsi_alignment_with_qsi_on_random_data()

def run_efficient_approx_tsi_tests():
    """Run tests for BatchedTSI implementation"""
    test_efficient_approx_tsi_on_example(curated_example_1)
    test_efficient_approx_tsi_on_example(curated_example_2)
    test_efficient_approx_tsi_on_example(curated_example_3)
    test_efficient_approx_tsi_alignment_with_tsi_on_random_data()

def run_efficient_approx_qsi_tests():
    """Run tests for BatchedQSI implementation"""
    test_efficient_approx_qsi_on_example(curated_example_2)
    test_efficient_approx_qsi_alignment_with_qsi_on_random_data()

def main():
    parser = argparse.ArgumentParser(description='Run TSI sanity tests')
    parser.add_argument('--test-subject',
                       choices=['All', 'TSI', 'EfficientTSI', 'ApproxTSI', 'BatchedTSI', 'QSI', 'ApproxQSI', 'EfficientQSI', 'BatchedQSI'],
                       default='All',
                       help='Specify which TSI implementation to test')

    args = parser.parse_args()

    print(f"Running tests for {args.test_subject}")


    if args.test_subject == 'All':
        run_tsi_tests()
        run_efficient_tsi_tests()
        run_approx_tsi_tests()
        run_efficient_approx_tsi_tests()
        run_qsi_tests()
        run_efficient_qsi_tests()
        run_approx_qsi_tests()
        run_efficient_approx_qsi_tests()
    elif args.test_subject == 'TSI':
        run_tsi_tests()
    elif args.test_subject == 'EfficientTSI':
        run_efficient_tsi_tests()
    elif args.test_subject == 'ApproxTSI':
        run_approx_tsi_tests()
    elif args.test_subject == 'BatchedTSI':
        run_efficient_approx_tsi_tests()
    elif args.test_subject == 'QSI':
        run_qsi_tests()
    elif args.test_subject == 'ApproxQSI':
        run_approx_qsi_tests()
    elif args.test_subject == 'EfficientQSI':
        run_efficient_qsi_tests()
    elif args.test_subject == 'BatchedQSI':
        run_efficient_approx_qsi_tests()

    print(f"\nAll {args.test_subject} tests completed successfully!")

if __name__ == "__main__":
    main()
````


````output
Running tests for All
TSI on example curated_example_1 test passed
TSI on example curated_example_2 test passed
TSI on example curated_example_3 test passed
TSI on identical data test passed
EfficientTSI on example curated_example_1 test passed
EfficientTSI on example curated_example_2 test passed
EfficientTSI on example curated_example_3 test passed
EfficientTSI alignment with TSI on random data test passed
EfficientTSI alignment with TSI test on random data with equalities passed
ApproxTSI on example curated_example_1 within tolerance test passed
ApproxTSI on example curated_example_2 within tolerance test passed
ApproxTSI on example curated_example_3 within tolerance test passed
ApproxTSI alignment with TSI on random data within tolerance test passed
BatchedTSI on example curated_example_1 within tolerance test passed
BatchedTSI on example curated_example_2 within tolerance test passed
BatchedTSI on example curated_example_3 within tolerance test passed
BatchedTSI alignment with TSI on random data within tolerance test passed
QSI on example curated_example_2 test passed
QSI on identical data test passed
EfficientQSI on example curated_example_2 test passed
EfficientQSI alignment with QSI on random data test passed
ApproxQSI on example curated_example_2 within tolerance test passed
ApproxQSI alignment with QSI on random data within tolerance test passed
BatchedQSI on example curated_example_2 within tolerance test passed
BatchedQSI alignment with QSI on random data within tolerance test passed

All All tests completed successfully!

````
