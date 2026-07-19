# Claim 2 repair ledger — exactly 10 approaches

Target claim: for two representation sets without distance ties, perfect TSI
alignment is equivalent to MutualNN alignment at every neighborhood scale
`k=1,...,N-2` (paper Corollary 1). The official verdict is `toy` because the
first reproduction exhaustively used only five-point clouds (the judge
summarized these as eight independently drawn small clouds).

The route list is frozen at exactly ten. Seeds, transforms, neighborhood
scales, official/independent implementations, and negative controls are
subchecks within a route—not extra approaches.

Pinned primary sources:

- paper: arXiv `2606.16379v1`, Corollary 1 and proof N.2.1;
- official repository: `diogosoares22/ordinal-similarity-metrics` at
  `79758ece4b97eeda98439542bbeb7c229f2bc2c9`;
- official CIFAR-10 ViT representations: 10,000×512 initial/final arrays,
  committed by the authors and SHA-256 pinned in the result manifest.

| # | Approach | Decisive non-toy output | State |
|---:|---|---|---|
| 1 | Exhaustive strict-order proof checker | All 9! orderings: zero inversions iff every prefix-neighbor set agrees | planned |
| 2 | Official CIFAR-10 ViT final representations, isometry | Exact n=2,048,d=512 perfect/perfect direction | planned |
| 3 | Official CIFAR-10 ViT initial vs final | Exact n=2,048,d=512 imperfect/imperfect direction | planned |
| 4 | Official CIFAR-10 ViT final with 2% outliers | Exact n=2,048,d=512 robustness regime | planned |
| 5 | Real sklearn Digits, isometry | Exact n=1,797,d=64 perfect/perfect direction | planned |
| 6 | Real sklearn Digits, anisotropic deformation | Exact n=1,797,d=64 imperfect/imperfect direction | planned |
| 7 | Real Wisconsin Breast Cancer, nonlinear warp | Exact n=569,d=30 imperfect/imperfect direction | planned |
| 8 | Real Wine, isometry | Exact n=178,d=13 perfect/perfect direction | planned |
| 9 | Swiss-roll global cluster translation | n=1,200; fixed-k MNN trap but all-scales MNN and TSI both imperfect | planned |
| 10 | Tied-grid boundary with no-tie perturbation | n=512; tied input rejected, jittered adversarial input yields imperfect/imperfect | planned |

Fail-closed invariant: the final result must report
`approaches_executed == 10`, contain route numbers 1 through 10 exactly once,
and reject route 11 or higher. Failed, mixed, or adverse results remain in
their original route.
