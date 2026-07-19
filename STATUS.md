# STATUS — Ordinal Similarity Indices (`G4D0YzzZEk`)

**State:** official high-quality **4/4** at exact Space SHA
`700f7a6f12a6458e0cdbd9b4d8f8e58c44f5c19a`.

## Source audit

- Paper: arXiv `2606.16379`; OpenReview `G4D0YzzZEk`.
- Official code: `diogosoares22/ordinal-similarity-metrics`, pinned to
  `79758ece4b97eeda98439542bbeb7c229f2bc2c9`.
- Claims: ordinal TSI/QSI definitions and the no-ties equivalence between
  perfect TSI alignment and joint MutualNN agreement across all scales.

## Repair result

The old five-point exhaustive audit was officially `toy`. Exactly ten frozen
routes now pass: 362,880/362,880 strict-order cases, exact audits of the
authors' CIFAR representations at n=2,048,d=512, three real datasets, a 2%
outlier control, fixed-k trap, and tie boundary. Route 11 is programmatically
rejected. The separate verifier passes without importing the audit module.

## Official verdict

Judged `2026-07-19T01:06:24Z`: both claims `verified`, quality `high`. The judge
explicitly accepted the 362,880 strict orders, CIFAR n=2,048,d=512 audits with
4.3 billion comparisons, multiple real datasets, fixed-k trap, and tied-input
rejection. The official verdict dataset now records the exact SHA above.
