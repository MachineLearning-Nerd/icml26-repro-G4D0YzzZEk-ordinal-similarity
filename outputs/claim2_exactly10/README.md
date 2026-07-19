# Claim 2 — exactly 10 approach evidence

This directory is the complete machine-readable result for the paper-scale
repair of OpenReview `G4D0YzzZEk`, Claim 2.

- `summary.json`: all 10 frozen routes, exact TSI integer counts, all-scale
  MutualNN results, no-tie checks, primary-array hashes, and route-11 rejection.
- `routes.csv`: compact one-row-per-route result table.
- `verification.json`: results and SHA-256 manifest from the independent
  verifier, which does not import the main exact-audit module.

The decisive checks are `approaches_executed == 10`, route numbers exactly
`1,...,10`, `route11_rejected == true`, and `claim2_verified == true`.
Route 1 covers all 362,880 strict orders. Routes 2–4 use the authors' pinned
CIFAR-10 ViT arrays at n=2,048,d=512; each contains 4,288,677,888 exact ordinal
comparisons. Supporting, imperfect, and boundary outcomes are all retained.
