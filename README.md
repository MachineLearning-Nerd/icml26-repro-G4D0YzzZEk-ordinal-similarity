# Ordinal Similarity Indices (TSI and QSI)

Reproduction of *Scalable and Interpretable Representation Alignment with
Ordinal Similarity* (ICML 2026, OpenReview `G4D0YzzZEk`, arXiv `2606.16379`).

## Result

Both claim-level results are verified with an independent implementation and
the public reference implementation:

1. TSI and QSI measure agreement of ordinal distance relationships. Literal
   ordered-triplet/quadruplet enumeration and an independent rank computation
   agree exactly; both indices remain 1 under rotation, scaling, and
   translation.
2. With no distance ties, `TSI = 1` iff all Mutual Nearest Neighbor sets agree
   at every neighborhood scale. The paper-scale repair executes exactly 10
   frozen approaches—never 11—including all 9! strict orders, the authors'
   CIFAR-10 ViT representations at `n=2,048,d=512`, three real sklearn
   datasets, outliers, a fixed-k trap, and a tied-distance boundary.

The 10/10 routes pass. Each CIFAR route evaluates 4,288,677,888 exact ordinal
comparisons. An independent verifier that does not import the audit module
rechecks the exhaustive kernel, primary-data orders, integer arithmetic,
artifact hashes, and the exactly-10 contract.

The sampled path also runs at `n=10,000`, `d=512`, using 20,000 comparisons
per metric—the released repository's advertised large-data regime.

## Run

```bash
source .venv/bin/activate
python repro/src/run_ordinal_audit.py
python repro/src/run_claim2_exactly10.py
python repro/src/verify_claim2_evidence.py
PYTHONPATH=official python official/src/sanity_tests.py
pytest -q repro/tests
```

## Scope

The source implementation is `diogosoares22/ordinal-similarity-metrics`, pinned
to `79758ece4b97eeda98439542bbeb7c229f2bc2c9`. The two 10,000×512 CIFAR-10
representation arrays are pinned by SHA-256; the exact audit uses the first
2,048 aligned rows. This artifact verifies the two scored definitions/theorem
and the public approximate computation path. It does not claim to repeat every
CIFAR-10 or CLIP training experiment in the paper.
