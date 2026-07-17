# Repro - Ordinal Similarity Indices

## Pages

| Page |
| --- |
| [Claim 1 — Ordinal TSI and QSI](#/claim-1-ordinal-tsi-and-qsi) |
| [Claim 2 — TSI and MutualNN](#/claim-2-tsi-and-mutualnn) |
| [Methods](#/methods) |
| [Negative controls](#/negative-controls) |
| [Conclusion](#/conclusion) |

## Outcome

**Both scored ordinal-similarity claims are verified.** Literal ordinal
enumeration and an independent rank computation agree exactly; TSI equals one
iff every Mutual Nearest Neighbor set agrees in all 960 no-tie permutation
cases. At `n=10,000`, `d=512`, the public sampled path scores near one for
aligned representations and near one half for a correspondence-breaking
permutation.

The theorem's no-ties precondition is tested explicitly and rejected for a
tied-distance control.
