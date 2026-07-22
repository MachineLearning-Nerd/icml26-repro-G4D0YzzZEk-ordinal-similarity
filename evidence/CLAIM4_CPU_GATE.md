# Claim 4 CPU gate contract

Paper: `G4D0YzzZEk`, *Scalable and Interpretable Representation Alignment
with Ordinal Similarity* (`arXiv:2606.16379`).

## Audited claim and sources

Challenge Claim 4 says that under a 2% outlier perturbation on synthetic data
(`N=1000`, `D=50`) **and on a ViT trained on CIFAR-10**, TSI and QSI retain
roughly 94% alignment and outperform CKA, CKNNA, and MutualNN.

The exact protocol was audited from:

- ar5iv Section 6.2 and Figure 5:
  `https://ar5iv.labs.arxiv.org/html/2606.16379#S6.SS2`
- ar5iv Appendix J, Table 2 (dot-product CKNNA/MutualNN, `k=10`):
  `https://ar5iv.labs.arxiv.org/html/2606.16379#A10.T2`
- ar5iv Appendix L and Table 4 (final-epoch seed-0 CIFAR-10 ViT `[CLS]`
  representations, `N=10000`, `D=512`):
  `https://ar5iv.labs.arxiv.org/html/2606.16379#A12`
- authors' released code/data at commit
  `79758ece4b97eeda98439542bbeb7c229f2bc2c9`:
  `https://github.com/diogosoares22/ordinal-similarity-metrics`

The released `data/cifar-10-final-epoch-val-representations.npy` has shape
`(10000, 512)`, dtype `float32`, and SHA-256
`3ee442b20bab44e4aae5a038b6489df988f4db68fc6dd1e95d630ca278a10b1b`.

## Three-attempt record

1. The judged revision's synthetic-only experiment used newly sampled Gaussian
   data and an ad hoc perturbation. It reproduced the synthetic panel but not
   the ViT/CIFAR-10 half; the judge correctly left Claim 4 inconclusive.
2. A local official-array feasibility preflight ran the full `N=10000,D=512`
   input at identity and `sigma=128` for one seed. It passed the exact-neighbor
   and CKA implementation controls and showed the required separation, but a
   one-run/two-sigma preflight is not the paper's 20-run curve and cannot be
   published as the final claim result.
3. The final attempt runs all nine released sigma values
   `0,1,2,4,8,16,32,64,128`, with 20 runs per value, on the pinned official
   array. This is the claim-grade attempt governed by the gates below.

## Frozen protocol and gates

- Exactly 200 of 10,000 points are perturbed in each run.
- Each outlier receives its own independently sampled random unit direction,
  with magnitude `mean_pairwise_distance * sigma`, matching the released code.
- Seeds follow the released schedule: `run + 1000*sigma + 10000`.
- TSI and QSI use 500,000 independent uniform distinct tuples per cell. The
  preregistered per-estimate Hoeffding radius is `0.003809` at `delta=1e-6`.
- Linear CKA is exact. Dot-product CKNNA and MutualNN use exact `k=10` neighbor
  lists. The partial-neighbor update must equal brute force on the frozen
  reduced control, and optimized CKA must agree with direct CKA within `1e-9`.
- At `sigma=0`, all five metrics must equal 1.
- At `sigma=128`, the TSI and QSI estimates plus their Hoeffding radius must
  remain above the clean-subset lower bounds (`TSI=0.941186`, `QSI=0.922357`).
- At `sigma=128`, the smaller of TSI/QSI must exceed each of CKA, CKNNA, and
  MutualNN by more than twice the Hoeffding radius.
- Any failed control or gate makes the final result `inconclusive`; it must not
  be published as support.

## Compute record

- Local official-array preflight: exit 0, about 30.0 seconds. At `sigma=128`,
  TSI `0.97470`, QSI `0.96060`, CKA `0.00944`, CKNNA `0.00000`, and MutualNN
  `0.00084`; partial top-k equaled brute force, and CKA error was `4.496e-11`.
- HF job `6a60f82dd09dc1f57c6c2d07`: failed before experiment execution because
  malformed PEP 723 delimiters prevented NumPy installation. It is not a
  scientific attempt and contributes no result.
- HF `cpu-upgrade` job `6a60f87213e6ef894d54c100`: full frozen protocol.
  It completed successfully in 571 seconds (579 seconds including scheduling).
  All controls and both claim gates passed. At `sigma=128`, the 20-run means
  were TSI `0.97072`, QSI `0.96111`, CKA `0.00952`, CKNNA `0.00009`, and
  MutualNN `0.00110`. The job's pre-normalization raw-float result hash was
  `8a9f87335460f12ef1eda4d83c7acdf4ed84de9a9dab29c05194e6974b30f00b`.
- A local full-protocol byte-parity run also exited 0 with every gate true.
  Linux and macOS BLAS differed only in the fifth printed decimal of one
  aggregate (`sigma=1` TSI, `0.97875` versus `0.97876`). The published
  verifier therefore exposes and hashes three-decimal aggregates; both tested
  stacks produce normalized `RESULTS_SHA256`
  `ef8e10098741d45adb87aab45af65dfa54e8a99d03654ba4ba27f973a8abad21`.

The judged baseline remains 9/12 until a new Hugging Face revision is scored.
Claim 5 remains GPU-bound and outside the authorized compute policy.
