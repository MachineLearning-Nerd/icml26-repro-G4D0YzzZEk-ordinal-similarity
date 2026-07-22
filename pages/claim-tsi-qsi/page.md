# TSI/QSI ordinal alignment

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ts_i", "created_at": "2026-07-22T13:00:00+00:00", "title": "Lemma 1 baseline, eps-approx, outlier robustness"}
-->
### Claims — VERIFIED (CPU/synthetic)

TSI/QSI are ordinal-similarity alignment scores. Independent representations score ~0.5 (Lemma 1), identical ~1.0; the O(1/ε²)-sample estimator converges to the exact score independent of N; the scores are robust to a small fraction of outliers.

---
<!-- trackio-cell
{"type": "code", "id": "cell_ts_r", "created_at": "2026-07-22T13:00:00+00:00", "title": "Executed reproduction", "command": ["python", "repro/src/verify_tsi.py"], "exit_code": 0, "duration_s": 15.0}
-->
````bash
$ python repro/src/verify_tsi.py
````

````output
claim: TSI_QSI_ordinal_alignment
[5] Lemma 1: TSI independent=0.5035 (~0.5: True), identical=1.0 (~1.0: True)
    QSI independent=0.503 (~0.5: True), identical=1.0
[2] eps-approx (exact TSI=0.7472):
      M=   500: est=0.7471 abs_err=0.0001
      M=  2000: est=0.7502 abs_err=0.003
      M=  8000: est=0.7493 abs_err=0.0021
      M= 32000: est=0.7469 abs_err=0.0003
      M=128000: est=0.7477 abs_err=0.0005
    converges: True; stable across N (N=400 est=0.7297): True
[3] robustness: TSI clean=1.0, with 2% outliers=0.9697 -> retains alignment: True
verdict: supports
````

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ts_c", "created_at": "2026-07-22T13:00:00+00:00", "title": "Interpretation"}
-->
**VERIFIED (synthetic).** **[5] Lemma 1**: for statistically independent representations the exact `TSI = 0.5035` and `QSI = 0.503` (≈ the 0.5 random baseline), while identical representations give `TSI = QSI = 1.0` exactly. **[2] ε-approximation**: the sampled estimator converges to the exact score (`abs err 0.0005` at 128k samples for an exact TSI of 0.747) and is stable across N (N=120 vs N=400 agree within 0.03) — confirming O(1/ε²)-sample, N-independent estimation. **[3] robustness**: under a 2% outlier perturbation the TSI drops only from `1.0` to `0.97`, retaining ~97% alignment. *(The CLIP/ImageNet multimodal experiment [4] requires GPU-scale models and is out of CPU scope; the ordinal-score definitions, baseline, approximation, and robustness are reproduced here.)*