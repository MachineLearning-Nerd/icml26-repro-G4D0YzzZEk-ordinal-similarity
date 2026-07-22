"""Independent, fail-closed verifier for the exactly-ten Claim 2 evidence."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "claim2_exactly10"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def independent_order_checks() -> dict[str, object]:
    final_path = ROOT / "official" / "data" / "cifar-10-final-epoch-val-representations.npy"
    initial_path = ROOT / "official" / "data" / "cifar-10-initial-epoch-val-representations.npy"
    # Promote before subtracting so the independent direct-difference path has
    # the same precision contract as the main float64 Gram-matrix path.
    final = np.asarray(np.load(final_path, mmap_mode="r")[:2048], dtype=np.float64)
    initial = np.asarray(np.load(initial_path, mmap_mode="r")[:2048], dtype=np.float64)
    anchors = (0, 1, 17, 255, 1023, 2047)
    isometry_orders_equal = True
    initial_final_mismatches = 0
    exact_ties_seen = 0
    for anchor in anchors:
        final_distances = np.einsum(
            "ij,ij->i", final - final[anchor], final - final[anchor]
        )
        negated = -final
        negated_distances = np.einsum(
            "ij,ij->i", negated - negated[anchor], negated - negated[anchor]
        )
        initial_distances = np.einsum(
            "ij,ij->i", initial - initial[anchor], initial - initial[anchor]
        )
        keep = np.arange(len(final)) != anchor
        final_order = np.flatnonzero(keep)[np.argsort(final_distances[keep], kind="stable")]
        negated_order = np.flatnonzero(keep)[np.argsort(negated_distances[keep], kind="stable")]
        initial_order = np.flatnonzero(keep)[np.argsort(initial_distances[keep], kind="stable")]
        isometry_orders_equal &= bool(np.array_equal(final_order, negated_order))
        initial_final_mismatches += int(not np.array_equal(initial_order, final_order))
        exact_ties_seen += int(np.count_nonzero(np.diff(np.sort(final_distances[keep])) <= 0.0))
        exact_ties_seen += int(np.count_nonzero(np.diff(np.sort(initial_distances[keep])) <= 0.0))
    return {
        "anchors": list(anchors),
        "isometry_orders_equal": isometry_orders_equal,
        "initial_final_order_mismatches": initial_final_mismatches,
        "exact_ties_seen": exact_ties_seen,
        "verified": bool(
            isometry_orders_equal
            and initial_final_mismatches == len(anchors)
            and exact_ties_seen == 0
        ),
    }


def independent_exhaustive_kernel() -> dict[str, object]:
    identity = tuple(range(9))
    cases = 0
    mismatches = 0
    for candidate in itertools.permutations(identity):
        cases += 1
        zero_inversions = all(
            candidate[left] < candidate[right]
            for left in range(9)
            for right in range(left + 1, 9)
        )
        every_prefix_equal = all(
            frozenset(candidate[:k]) == frozenset(identity[:k])
            for k in range(1, 9)
        )
        mismatches += int(zero_inversions != every_prefix_equal)
    return {
        "permutations": cases,
        "equivalence_mismatches": mismatches,
        "verified": cases == 362_880 and mismatches == 0,
    }


def main() -> None:
    summary_path = OUT / "summary.json"
    routes_path = OUT / "routes.csv"
    summary = json.loads(summary_path.read_text())
    checks: list[dict[str, object]] = []

    checks.append({
        "name": "exactly_ten_contract",
        "passed": summary["approaches_executed"] == 10
        and summary["route_numbers"] == list(range(1, 11))
        and summary["route11_rejected"] is True,
    })
    checks.append({
        "name": "every_route_retained_and_verified",
        "passed": len(summary["routes"]) == 10
        and all(route["verified"] is True for route in summary["routes"]),
    })
    expected_perfection = {2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: True, 9: False, 10: False}
    direction_ok = True
    arithmetic_ok = True
    for route in summary["routes"][1:]:
        direction_ok &= route["tsi_perfect"] is expected_perfection[route["route"]]
        direction_ok &= route["mnn_all_scales_perfect"] is expected_perfection[route["route"]]
        direction_ok &= route["equivalence_holds"] is True
        expected_tsi = 1.0 - route["tsi_discordant"] / route["tsi_comparisons"]
        arithmetic_ok &= abs(expected_tsi - route["tsi"]) < 1e-15
    checks.append({"name": "both_theorem_directions_and_controls", "passed": bool(direction_ok)})
    checks.append({"name": "tsi_integer_arithmetic", "passed": bool(arithmetic_ok)})

    route9 = summary["routes"][8]
    route10 = summary["routes"][9]
    checks.append({
        "name": "fixed_k_trap_and_tie_boundary",
        "passed": route9["selected_mnn_scores"]["399"] == 1.0
        and route9["fixed_k_trap_verified"] is True
        and route10["tied_boundary_rejected"] is True
        and route10["tied_boundary_tie_count_x"] > 0,
    })
    exhaustive = independent_exhaustive_kernel()
    cifar = independent_order_checks()
    checks.append({"name": "independent_exhaustive_kernel", "passed": exhaustive["verified"]})
    checks.append({"name": "independent_primary_data_orders", "passed": cifar["verified"]})

    report = {
        "verifier": "independent standard-library/NumPy verifier; does not import claim2_exact",
        "checks": checks,
        "independent_exhaustive_kernel": exhaustive,
        "independent_primary_data_orders": cifar,
        "artifact_sha256": {
            "summary.json": sha256_file(summary_path),
            "routes.csv": sha256_file(routes_path),
            "claim2_exact.py": sha256_file(ROOT / "repro" / "src" / "claim2_exact.py"),
            "run_claim2_exactly10.py": sha256_file(ROOT / "repro" / "src" / "run_claim2_exactly10.py"),
            "cifar_final.npy": sha256_file(ROOT / "official" / "data" / "cifar-10-final-epoch-val-representations.npy"),
            "cifar_initial.npy": sha256_file(ROOT / "official" / "data" / "cifar-10-initial-epoch-val-representations.npy"),
        },
        "verified": all(check["passed"] for check in checks),
    }
    if not report["verified"]:
        raise SystemExit(json.dumps(report, indent=2))
    (OUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
