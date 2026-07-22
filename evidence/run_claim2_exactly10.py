"""Execute exactly ten non-toy approaches for paper G4D0YzzZEk Claim 2."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np

from claim2_exact import audit_point_sets, validate_exactly_ten_routes


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "claim2_exactly10"
OFFICIAL = ROOT / "official"
OFFICIAL_COMMIT = "79758ece4b97eeda98439542bbeb7c229f2bc2c9"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_jitter(points: np.ndarray, seed: int) -> np.ndarray:
    points64 = np.asarray(points, dtype=np.float64)
    scale = np.maximum(np.std(points64, axis=0, keepdims=True), 1.0)
    noise = np.random.default_rng(seed).normal(size=points64.shape)
    return points64 + noise * scale * 1e-9


def standardize(points: np.ndarray) -> np.ndarray:
    points64 = np.asarray(points, dtype=np.float64)
    scale = np.std(points64, axis=0)
    scale[scale == 0] = 1.0
    return (points64 - np.mean(points64, axis=0)) / scale


def route1_exhaustive_orders() -> dict[str, object]:
    width = 9
    identity = tuple(range(width))
    cases = 0
    equivalence_matches = 0
    tsi_perfect_cases = 0
    mnn_perfect_cases = 0
    for permutation in itertools.permutations(identity):
        cases += 1
        inversions = sum(
            permutation[left] > permutation[right]
            for left in range(width)
            for right in range(left + 1, width)
        )
        tsi_perfect = inversions == 0
        prefix_sets_match = all(
            set(identity[:k]) == set(permutation[:k]) for k in range(1, width)
        )
        tsi_perfect_cases += int(tsi_perfect)
        mnn_perfect_cases += int(prefix_sets_match)
        equivalence_matches += int(tsi_perfect == prefix_sets_match)
    return {
        "route": 1,
        "name": "exhaustive_strict_order_proof_checker",
        "order_width": width,
        "permutations": cases,
        "tsi_perfect_cases": tsi_perfect_cases,
        "mnn_all_scales_perfect_cases": mnn_perfect_cases,
        "equivalence_matches": equivalence_matches,
        "verified": equivalence_matches == cases and cases == 362_880,
    }


def audited_route(route: int, name: str, x: np.ndarray, y: np.ndarray, k: tuple[int, ...]) -> dict[str, object]:
    start = time.perf_counter()
    audit = audit_point_sets(x, y, selected_k=k)
    result = {
        "route": route,
        "name": name,
        "runtime_seconds": time.perf_counter() - start,
        **audit.as_dict(),
    }
    result["verified"] = bool(
        audit.no_ties_x and audit.no_ties_y and audit.equivalence_holds
    )
    return result


def load_sklearn_datasets() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.datasets import load_breast_cancer, load_digits, load_wine

    return load_digits().data, load_breast_cancer().data, load_wine().data


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    final_path = OFFICIAL / "data" / "cifar-10-final-epoch-val-representations.npy"
    initial_path = OFFICIAL / "data" / "cifar-10-initial-epoch-val-representations.npy"
    final = np.load(final_path, mmap_mode="r")[:2048].astype(np.float64)
    initial = np.load(initial_path, mmap_mode="r")[:2048].astype(np.float64)

    routes: list[dict[str, object]] = [route1_exhaustive_orders()]
    routes.append(audited_route(2, "official_cifar_final_isometry", final, -final, (1, 32, 256, 1024)))
    routes.append(audited_route(3, "official_cifar_initial_vs_final", initial, final, (1, 32, 256, 1024)))

    contaminated = final.copy()
    rng = np.random.default_rng(20260719)
    outlier_indices = np.sort(rng.choice(len(contaminated), size=41, replace=False))
    contaminated[outlier_indices] += rng.normal(scale=25.0, size=(len(outlier_indices), final.shape[1]))
    route4 = audited_route(4, "official_cifar_final_two_percent_outliers", final, contaminated, (1, 32, 256, 1024))
    route4["outlier_count"] = len(outlier_indices)
    route4["outlier_fraction"] = len(outlier_indices) / len(final)
    routes.append(route4)

    digits_raw, cancer_raw, wine_raw = load_sklearn_datasets()
    digits = deterministic_jitter(standardize(digits_raw), seed=5)
    routes.append(audited_route(5, "sklearn_digits_isometry", digits, -digits, (1, 16, 128, 512)))
    digit_scales = np.linspace(0.35, 2.5, digits.shape[1])
    digit_deformed = digits * digit_scales
    digit_deformed[:, 0] += 0.4 * digits[:, 1] ** 2
    routes.append(audited_route(6, "sklearn_digits_anisotropic_deformation", digits, digit_deformed, (1, 16, 128, 512)))

    cancer = deterministic_jitter(standardize(cancer_raw), seed=7)
    cancer_warped = cancer * np.linspace(0.5, 2.0, cancer.shape[1])
    cancer_warped[:, 0] += 0.5 * cancer[:, 1] ** 2
    routes.append(audited_route(7, "wisconsin_breast_cancer_nonlinear_warp", cancer, cancer_warped, (1, 8, 64, 256)))

    wine = deterministic_jitter(standardize(wine_raw), seed=8)
    routes.append(audited_route(8, "sklearn_wine_isometry", wine, -wine, (1, 8, 32, 96)))

    cluster_rng = np.random.default_rng(9)
    cluster_parts = [cluster_rng.normal(scale=0.5, size=(400, 64)) for _ in range(3)]
    cluster_parts[0][:, 0] -= 100.0
    cluster_parts[2][:, 0] += 100.0
    clusters = np.vstack(cluster_parts)
    translated = clusters.copy()
    translated[800:, 0] += 200.0
    route9 = audited_route(9, "clustered_global_translation_fixed_k_trap", clusters, translated, (32, 399, 600, 1000))
    route9["fixed_k_trap_verified"] = bool(
        route9["selected_mnn_scores"]["399"] == 1.0
        and not route9["mnn_all_scales_perfect"]
        and not route9["tsi_perfect"]
    )
    route9["verified"] = bool(route9["verified"] and route9["fixed_k_trap_verified"])
    routes.append(route9)

    grid_x, grid_y = np.meshgrid(np.arange(32), np.arange(16), indexing="ij")
    tied_grid = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float64)
    tied_audit = audit_point_sets(tied_grid, tied_grid)
    jittered_grid = deterministic_jitter(tied_grid, seed=10)
    perturbed_grid = jittered_grid.copy()
    perturbed_grid[0] += np.array([0.37, -0.21])
    route10 = audited_route(10, "tied_grid_boundary_then_no_tie_perturbation", jittered_grid, perturbed_grid, (1, 8, 64, 256))
    route10["tied_boundary_rejected"] = tied_audit.equivalence_holds is None
    route10["tied_boundary_tie_count_x"] = tied_audit.tie_count_x
    route10["verified"] = bool(
        route10["verified"]
        and route10["tied_boundary_rejected"]
        and not route10["tsi_perfect"]
        and not route10["mnn_all_scales_perfect"]
    )
    routes.append(route10)

    validate_exactly_ten_routes(route["route"] for route in routes)
    route11_rejected = False
    try:
        validate_exactly_ten_routes(range(1, 12))
    except ValueError:
        route11_rejected = True

    summary = {
        "paper": "Scalable and Interpretable Representation Alignment with Ordinal Similarity",
        "openreview_id": "G4D0YzzZEk",
        "claim": 2,
        "official_repository_commit": OFFICIAL_COMMIT,
        "primary_artifacts": {
            "cifar_final": {"shape": [10000, 512], "sha256": sha256_file(final_path)},
            "cifar_initial": {"shape": [10000, 512], "sha256": sha256_file(initial_path)},
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "approaches_executed": len(routes),
        "route_numbers": [route["route"] for route in routes],
        "route11_rejected": route11_rejected,
        "all_routes_verified": all(bool(route["verified"]) for route in routes),
        "claim2_verified": bool(
            len(routes) == 10
            and route11_rejected
            and all(bool(route["verified"]) for route in routes)
        ),
        "routes": routes,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "routes.csv").open("w", newline="") as handle:
        fields = ["route", "name", "n", "d", "tsi", "tsi_perfect", "mnn_all_scales_perfect", "equivalence_holds", "verified", "runtime_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for route in routes:
            writer.writerow({field: route.get(field) for field in fields})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
