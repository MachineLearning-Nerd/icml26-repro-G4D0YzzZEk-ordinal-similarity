# Claim 2 — TSI and MutualNN


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_8f62b7dae2a7", "created_at": "2026-07-17T06:40:20+00:00", "title": "Claim and result"}
-->
## Claim
For representations without distance ties, perfect TSI alignment is equivalent to exact Mutual Nearest Neighbor agreement at every neighborhood scale.

## Exhaustive audit
For eight independently drawn no-tie point clouds of five points, all 120 label permutations were enumerated. In all 960 cases, the truth value of TSI = 1 exactly matched joint MutualNN agreement for every k from 1 to N−2.

**Result:** 960/960 equivalence checks pass.
