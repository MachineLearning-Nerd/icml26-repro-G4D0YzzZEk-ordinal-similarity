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
enumeration and an independent rank computation agree exactly. The Claim 2
repair executes exactly 10 frozen approaches: all 362,880 strict orders,
authors' CIFAR-10 ViT data at n=2,048,d=512, three real datasets, outliers,
a fixed-k trap, and a tie boundary. Every route passes; route 11 is rejected.

The theorem's no-ties precondition is tested explicitly and rejected for a
tied-distance control. A separate verifier rechecks primary-data orders,
integer arithmetic, artifact hashes, and the exhaustive kernel without
importing the audit implementation.
