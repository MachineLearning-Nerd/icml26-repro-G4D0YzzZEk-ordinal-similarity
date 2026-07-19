# STATUS — Ordinal Similarity Indices (`G4D0YzzZEk`)

**State:** official 3/4; exactly-10 Claim-2 repair executed and independently
verified, awaiting exact-SHA official re-judgment.

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

Next: publish, perform anonymous public readback, and require an exact-SHA
official 4/4 verdict before declaring this paper complete.
