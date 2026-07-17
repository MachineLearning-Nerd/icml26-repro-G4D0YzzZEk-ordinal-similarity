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
   at every neighborhood scale. The audit checks 960 exhaustive no-tie
   permutation cases and has a tie control that explicitly rejects the theorem
   precondition.

The sampled path also runs at `n=10,000`, `d=512`, using 20,000 comparisons
per metric—the released repository's advertised large-data regime.

## Run

```bash
source .venv/bin/activate
python repro/src/run_ordinal_audit.py
PYTHONPATH=official python official/src/sanity_tests.py
pytest -q repro/tests
```

## Scope

The source implementation is `diogosoares22/ordinal-similarity-metrics`, pinned
to `79758ece4b97eeda98439542bbeb7c229f2bc2c9`. This artifact verifies the two
scored definitions/theorem and the public approximate computation path. It does
not claim to repeat every CIFAR-10 or CLIP training experiment in the paper.
