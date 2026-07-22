#!/usr/bin/env python3
"""Run the exact Section 6.2 CIFAR-10 outlier benchmark on HF cpu-upgrade.

The job clones the authors' repository at the audited commit, executes the
released 10,000 x 512, 20-run, nine-sigma protocol, validates every cell, and
emits the raw CSV as base64 so the terminal result can be reconstructed without
using an artifact or binary-upload route.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys


OFFICIAL_COMMIT = "79758ece4b97eeda98439542bbeb7c229f2bc2c9"
OFFICIAL_REPOSITORY = "https://github.com/diogosoares22/ordinal-similarity-metrics.git"
SOURCE_URL = "https://ar5iv.labs.arxiv.org/html/2606.16379"
SOURCE_SCOPE = "Section 6.2, Figure 5, Appendix K.2, and Appendix L/Table 4"
EXPECTED_DATA_SHA256 = "3ee442b20bab44e4aae5a038b6489df988f4db68fc6dd1e95d630ca278a10b1b"
EXPECTED_REFERENCE_SHA256 = "617ff892ae9eee4eb73ab543a6d29b1fa96f208eb02fa3eeb1aee6651356daf3"
EXPECTED_SIGMAS = [0, 1, 2, 4, 8, 16, 32, 64, 128]
SCORE_COLUMNS = [
    "TSI_score",
    "QSI_score",
    "CKA_score",
    "CKNNA_score",
    "MutualNN_score",
    "SVCCA_score",
    "PWCCA_score",
    "CKNNA_Euclidean_score",
    "MutualNN_Euclidean_score",
]
STANDARD_BASELINES = ["CKA_score", "CKNNA_score", "MutualNN_score"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(rows: list[dict[str, str]], column: str) -> float:
    values = [float(row[column]) for row in rows]
    require(all(math.isfinite(value) for value in values), f"non-finite {column}")
    return sum(values) / len(values)


def keyed(rows: list[dict[str, str]]) -> dict[tuple[int, int], dict[str, str]]:
    return {(int(row["sigma"]), int(row["run"])): row for row in rows}


def main() -> None:
    cpu_count = os.cpu_count() or 0
    memory_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    require(cpu_count >= 8, f"cpu-upgrade exposes only {cpu_count} CPUs")
    require(memory_bytes >= 30 * 1024**3, f"cpu-upgrade exposes only {memory_bytes} bytes RAM")

    root = Path("/tmp/g4d-c4-fullscale")
    if root.exists():
        shutil.rmtree(root)
    repository = root / "official"
    subprocess.run(
        ["git", "clone", "--quiet", OFFICIAL_REPOSITORY, str(repository)],
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "--quiet", OFFICIAL_COMMIT],
        cwd=repository,
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    require(revision == OFFICIAL_COMMIT, f"unexpected official revision {revision}")

    data_path = repository / "data/cifar-10-final-epoch-val-representations.npy"
    reference_path = repository / (
        "experiments/results/benchmark_outliers/"
        "outliers_benchmark_cifar10_synthetic_per_outlier_sigma0-128_pct2.0_runs20_seed0.csv"
    )
    require(sha256_file(data_path) == EXPECTED_DATA_SHA256, "primary ViT array hash mismatch")
    require(sha256_file(reference_path) == EXPECTED_REFERENCE_SHA256, "official result hash mismatch")

    command = [
        sys.executable,
        "experiments/scripts/benchmark_outliers.py",
        "--data-sources",
        "cifar10",
        "--cifar10-n",
        "10000",
        "--sigma-max",
        "128",
        "--sigma-step",
        "2",
        "--outlier-pct",
        "2.0",
        "--outlier-direction-mode",
        "per_outlier",
        "--runs",
        "20",
        "--seed",
        "0",
    ]
    completed = subprocess.run(
        command,
        cwd=repository,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    sys.stdout.write(completed.stdout)
    sys.stdout.flush()
    require(completed.returncode == 0, f"official benchmark exited {completed.returncode}")

    output_path = repository / (
        "experiments/results/benchmark_outliers/"
        "outliers_benchmark_cifar10_per_outlier_sigma0-128_pct2.0_runs20_seed0.csv"
    )
    require(output_path.is_file(), "full benchmark CSV missing")
    with output_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with reference_path.open(newline="") as handle:
        reference_rows = [
            row for row in csv.DictReader(handle) if row["data_source"] == "cifar10"
        ]

    require(len(rows) == 180, f"expected 180 rows, observed {len(rows)}")
    require(len(reference_rows) == 180, "pinned reference does not contain 180 CIFAR rows")
    require(sorted({int(row["sigma"]) for row in rows}) == EXPECTED_SIGMAS, "sigma grid mismatch")
    require(sorted({int(row["run"]) for row in rows}) == list(range(1, 21)), "run grid mismatch")
    require({int(row["n_points"]) for row in rows} == {10000}, "N mismatch")
    require({int(row["dim"]) for row in rows} == {512}, "dimension mismatch")
    require({int(row["k_outliers"]) for row in rows} == {200}, "outlier count mismatch")
    require({float(row["outlier_pct"]) for row in rows} == {2.0}, "outlier percentage mismatch")

    observed_by_key = keyed(rows)
    reference_by_key = keyed(reference_rows)
    require(set(observed_by_key) == set(reference_by_key), "reference key grid mismatch")
    max_reference_abs_error = 0.0
    for key in sorted(observed_by_key):
        for column in SCORE_COLUMNS:
            observed = float(observed_by_key[key][column])
            expected = float(reference_by_key[key][column])
            require(math.isfinite(observed), f"non-finite {column} at {key}")
            max_reference_abs_error = max(max_reference_abs_error, abs(observed - expected))
    require(max_reference_abs_error <= 5e-4, f"reference drift {max_reference_abs_error}")

    aggregates: dict[str, dict[str, float]] = {}
    for sigma in EXPECTED_SIGMAS:
        subset = [row for row in rows if int(row["sigma"]) == sigma]
        require(len(subset) == 20, f"sigma {sigma} does not contain 20 runs")
        aggregates[str(sigma)] = {column: mean(subset, column) for column in SCORE_COLUMNS}

    clean_gate = all(abs(aggregates["0"][column] - 1.0) <= 0.005 for column in SCORE_COLUMNS)
    ordinal_retention_gate = min(
        aggregates[str(sigma)][column]
        for sigma in EXPECTED_SIGMAS[1:]
        for column in ("TSI_score", "QSI_score")
    ) >= 0.94
    dominance_gate = all(
        aggregates[str(sigma)][ordinal] > aggregates[str(sigma)][baseline]
        for sigma in EXPECTED_SIGMAS
        if sigma >= 4
        for ordinal in ("TSI_score", "QSI_score")
        for baseline in STANDARD_BASELINES
    )
    require(clean_gate, "clean self-alignment gate failed")
    require(ordinal_retention_gate, "ordinal retention fell below 0.94")
    require(dominance_gate, "strong-outlier standard-baseline dominance failed")

    summary = {
        "paper": "Scalable and Interpretable Representation Alignment with Ordinal Similarity",
        "openreview_id": "G4D0YzzZEk",
        "claim": 4,
        "source_url": SOURCE_URL,
        "source_scope": SOURCE_SCOPE,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "primary_array_sha256": EXPECTED_DATA_SHA256,
        "official_reference_sha256": EXPECTED_REFERENCE_SHA256,
        "hardware": {"flavor": "cpu-upgrade", "cpu_count": cpu_count, "memory_bytes": memory_bytes},
        "protocol": {
            "n": 10000,
            "d": 512,
            "outlier_fraction": 0.02,
            "outliers": 200,
            "sigmas": EXPECTED_SIGMAS,
            "runs_per_sigma": 20,
            "rows": len(rows),
        },
        "aggregates": aggregates,
        "max_abs_error_vs_pinned_official_result": max_reference_abs_error,
        "gates": {
            "clean_self_alignment": clean_gate,
            "tsi_qsi_minimum_retention_at_least_0_94": ordinal_retention_gate,
            "tsi_qsi_dominate_standard_baselines_for_sigma_ge_4": dominance_gate,
        },
        "verdict": "supports",
        "raw_csv_sha256": sha256_file(output_path),
    }
    canonical = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    print("G4D_SUMMARY_JSON=" + canonical)
    print("G4D_SUMMARY_SHA256=" + hashlib.sha256(canonical.encode()).hexdigest())
    print("G4D_CSV_B64_BEGIN")
    encoded = base64.b64encode(output_path.read_bytes()).decode()
    for offset in range(0, len(encoded), 120):
        print(encoded[offset : offset + 120])
    print("G4D_CSV_B64_END")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"G4D C4 full-scale job failed closed: {error}", file=sys.stderr, flush=True)
        raise
