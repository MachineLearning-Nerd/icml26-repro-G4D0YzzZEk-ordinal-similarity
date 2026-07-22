#!/usr/bin/env python3
"""Independent, dependency-free audit of the G4D0 Claim 4 result bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "claim4_fullscale_summary.json"
OUTPUT = ROOT / "claim4_fullscale_job_output.txt"
EXPECTED_SIGMAS = [0, 1, 2, 4, 8, 16, 32, 64, 128]
METRICS = ["TSI", "QSI", "CKA", "CKNNA", "MutualNN"]


def fail(message: str) -> None:
    raise SystemExit("FAIL: " + message)


def main() -> int:
    record = json.loads(SUMMARY.read_text())
    raw = OUTPUT.read_text()

    if record["paper"] != "G4D0YzzZEk" or record["claim"] != 4:
        fail("paper or claim identity")
    if record["job_id"] != "6a60f87213e6ef894d54c100" or record["job_status"] != "COMPLETED":
        fail("terminal job provenance")
    if record["official_commit"] != "79758ece4b97eeda98439542bbeb7c229f2bc2c9":
        fail("official commit")
    if record["official_array_sha256"] != "3ee442b20bab44e4aae5a038b6489df988f4db68fc6dd1e95d630ca278a10b1b":
        fail("official array hash")
    if record["official_array_shape"] != [10000, 512]:
        fail("official array shape")
    if record["sigmas"] != EXPECTED_SIGMAS or record["runs_per_sigma"] != 20:
        fail("20-run nine-sigma protocol")
    if record["metric_rows"] != 180 or record["outliers_per_run"] != 200:
        fail("metric-row or outlier count")

    row_pattern = re.compile(
        r"^\s*(0|1|2|4|8|16|32|64|128)\s+"
        r"([0-9]+\.[0-9]{5})\s+([0-9]+\.[0-9]{5})\s+"
        r"([0-9]+\.[0-9]{5})\s+([0-9]+\.[0-9]{5})\s+([0-9]+\.[0-9]{5})$",
        re.MULTILINE,
    )
    parsed = [
        {"sigma": int(match.group(1)), **dict(zip(METRICS, map(float, match.groups()[1:])))}
        for match in row_pattern.finditer(raw)
    ]
    if parsed != record["summary"]:
        fail("raw table and JSON summary differ")
    if [row["sigma"] for row in parsed] != EXPECTED_SIGMAS:
        fail("raw sigma rows")

    identity = all(parsed[0][metric] == 1.0 for metric in METRICS)
    tail = parsed[-1]
    lower_bound = (
        tail["TSI"] + record["hoeffding_epsilon_delta_1e_6"] >= record["tsi_clean_subset_lower_bound"]
        and tail["QSI"] + record["hoeffding_epsilon_delta_1e_6"] >= record["qsi_clean_subset_lower_bound"]
    )
    margin = 2.0 * record["hoeffding_epsilon_delta_1e_6"]
    dominance = all(
        min(tail["TSI"], tail["QSI"]) > tail[baseline] + margin
        for baseline in ("CKA", "CKNNA", "MutualNN")
    )
    controls = (
        record["partial_topk_equals_brute_control"]
        and record["fast_cka_max_abs_error_control"] <= 1e-9
        and record["sigma0_identity_control"]
    )
    gates = (
        identity
        and lower_bound
        and dominance
        and controls
        and record["lower_bound_gate"]
        and record["ordinal_beats_all_three_baselines_gate"]
        and record["claim4_verified"]
        and "verdict: supports\n" in raw
    )
    if not gates:
        fail("one or more preregistered gates")

    digest = hashlib.sha256(SUMMARY.read_bytes()).hexdigest()
    print("paper=G4D0YzzZEk claim=4 job=6a60f87213e6ef894d54c100 status=COMPLETED")
    print("protocol=10000x512 CIFAR-10 ViT; 200 outliers; 20 runs x 9 sigmas; rows=180")
    print(
        "sigma128="
        + ",".join(f"{metric}:{tail[metric]:.5f}" for metric in METRICS)
    )
    print(f"controls=identity:{identity},exact-neighbor:{record['partial_topk_equals_brute_control']},cka-error:{record['fast_cka_max_abs_error_control']:.3e}")
    print(f"gates=clean-subset:{lower_bound},ordinal-dominance:{dominance},claim4:{gates}")
    print("SUMMARY_SHA256=" + digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
