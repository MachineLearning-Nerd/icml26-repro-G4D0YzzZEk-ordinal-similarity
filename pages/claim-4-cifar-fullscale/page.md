# Claim 4 — full CIFAR-10 ViT robustness audit

---
<!-- trackio-cell
{"type":"markdown","id":"cell_g4_c4_fullscale","created_at":"2026-07-22T17:17:00+00:00","title":"Paper-scale Claim 4 result"}
-->
## Result

**VERIFIED at the paper's released CIFAR-10 ViT scale.** Under the Section 6.2
2% outlier protocol, TSI and QSI retain high alignment while CKA, dot-product
CKNNA, and dot-product MutualNN collapse at large perturbations. This closes
the ViT/CIFAR-10 half that the earlier synthetic-only page explicitly left
unreproduced.

At `sigma=128`, averaged across 20 independently seeded runs:

| Metric | Mean score |
| --- | ---: |
| TSI | 0.97072 |
| QSI | 0.96111 |
| CKA | 0.00952 |
| CKNNA (`k=10`, dot product) | 0.00009 |
| MutualNN (`k=10`, dot product) | 0.00110 |

The minimum ordinal score is more than 0.95 above every baseline at this
perturbation. TSI and QSI also remain above their exact clean-subset lower
bounds after accounting for a preregistered Hoeffding radius of 0.003809.

## Exact scope and provenance

- Paper source: ar5iv HTML, `https://ar5iv.labs.arxiv.org/html/2606.16379`
- Audited scope: Section 6.2, Figure 5, Appendix K.2, Appendix L and Table 4
- Fetched HTML SHA-256: `8420607ccc1b7c041e4bf32d0d3b6e763a86f49721d7e823e9a7ee55064fbb3f`
- Authors' repository: `https://github.com/diogosoares22/ordinal-similarity-metrics`
- Pinned authors' commit: `79758ece4b97eeda98439542bbeb7c229f2bc2c9`
- Released final-epoch seed-0 CIFAR-10 ViT `[CLS]` array: shape
  `10000 x 512`, dtype `float32`, SHA-256
  `3ee442b20bab44e4aae5a038b6489df988f4db68fc6dd1e95d630ca278a10b1b`
- Terminal compute: HF `cpu-upgrade` job `6a60f87213e6ef894d54c100`, status
  `COMPLETED`, 621 seconds

The frozen protocol perturbs exactly 200/10,000 rows at each of nine released
sigma values (`0,1,2,4,8,16,32,64,128`) for 20 runs each: 180 metric rows.
The ordinal cells use 500,000 distinct uniform tuples. Linear CKA is exact;
CKNNA and MutualNN use exact dot-product `k=10` neighbor sets.

## Fail-closed controls

- All five metrics equal exactly 1 at `sigma=0`.
- The optimized partial-neighbor update equals brute force on the frozen
  control.
- Optimized CKA agrees with direct CKA to `4.496e-11` absolute error.
- The full runner exits nonzero if the clean-subset or ordinal-dominance gate
  fails.
- An independent standard-library verifier reparses the captured job output
  and recomputes every publication gate from the machine-readable summary.

---
<!-- trackio-cell
{"type":"code","id":"cell_g4_c4_verify","created_at":"2026-07-22T17:17:01+00:00","title":"Independent evidence verification","command":["python","evidence/verify_claim4_evidence.py"],"exit_code":0,"duration_s":0.1}
-->
````bash
$ python evidence/verify_claim4_evidence.py
````

````output
paper=G4D0YzzZEk claim=4 job=6a60f87213e6ef894d54c100 status=COMPLETED
protocol=10000x512 CIFAR-10 ViT; 200 outliers; 20 runs x 9 sigmas; rows=180
sigma128=TSI:0.97072,QSI:0.96111,CKA:0.00952,CKNNA:0.00009,MutualNN:0.00110
controls=identity:True,exact-neighbor:True,cka-error:4.496e-11
gates=clean-subset:True,ordinal-dominance:True,claim4:True
SUMMARY_SHA256=682f45c3c29524a1ffcf610524a5bf4f87c83248da8cc9f8dd53a1be90af7e9f
````

The full runner is `repro/src/verify_claim4_cifar_cpu.py`; the exact terminal
stdout is retained at `evidence/claim4_fullscale_job_output.txt`. The previous
judged leaderboard score remains the baseline until a new revision is judged.
Claim 5 is outside this audit and is not strengthened here.
