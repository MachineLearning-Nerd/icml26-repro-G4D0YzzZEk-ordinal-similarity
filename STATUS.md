# STATUS — Ordinal Similarity Indices (`G4D0YzzZEk`)

**State:** in progress.

## Source audit

- Paper: arXiv `2606.16379`; OpenReview `G4D0YzzZEk`.
- Official code: `diogosoares22/ordinal-similarity-metrics`, pinned to
  `79758ece4b97eeda98439542bbeb7c229f2bc2c9`.
- Claims: ordinal TSI/QSI definitions and the no-ties equivalence between
  perfect TSI alignment and joint MutualNN agreement across all scales.

## Current step

Independent enumerator, MNN equivalence audit, tie control, and n=10,000,
d=512 sampled path are scaffolded. Next: run the complete suite, cross-check
against the official sanity tests, then prepare Trackio and the public repo.
