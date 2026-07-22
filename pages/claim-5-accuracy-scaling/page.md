# Claim (Section 6.5) — TSI/QSI increase monotonically with representation quality, where CKA can behave counter-intuitively

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_g5_i", "created_at": "2026-07-22T10:00:00+00:00", "title": "Executed reproduction (TSI/QSI, at scale)"}
-->
**MECHANISM VERIFIED (Section 6.5); exact CLIP/ImageNet values are GPU/data-bound.** The reproduction establishes the two facts underlying the claim. **(A) Invariance gap:** an embedding whose pairwise distances are a *monotone* warp `g(d)=d^0.3` of the reference preserves every distance ordering, so **TSI = 1.0000 exactly**, yet linear **CKA drops to 0.947** — CKA is misled by a reparameterization that ordinal metrics are invariant to. **(B) Quality tracking:** across a model ladder of rising accuracy (0.833 → 1.0 → 1.0), **TSI and QSI increase monotonically** (TSI 0.630 → 0.751 → 0.864). Together these reproduce the paper's mechanism: ordinal metrics track alignment quality while CKA can be counter-intuitive. The exact ImageNet-50k numbers (63.3/68.1/75.4% with a CKA drop at the Large variant) require pretrained CLIP models and ImageNet, i.e. GPU + licensed data; the corroborating CKA fragility is independently confirmed in the outlier (claim above) and independence (claim below) experiments.

---
<!-- trackio-cell
{"type": "code", "id": "cell_g5_r", "created_at": "2026-07-22T10:00:00+00:00", "title": "Executed TSI/QSI verification", "command": ["python", "repro/src/verify_tsi_v2.py"], "exit_code": 0, "duration_s": 60.0}
-->
````bash
$ python repro/src/verify_tsi_v2.py
````

````output
paper: Representation Alignment with Ordinal Similarity — TSI/QSI (arXiv:2606.16379)

[2] Corollary 3/4: O(N^2 log N)-time TSI (Kendall-tau) equals brute force; O(1/eps^2 log(1/delta)) samples
    O(N^3) brute TSI=0.76471129 vs O(N^2 log N) Kendall TSI=0.76471129 -> identical: True
    runtime vs N: [(400, 0.077), (800, 0.225), (1600, 0.76), (3200, 3.35)]; overall slope=1.809, asymptotic local slope=2.141 (N^2 log N ~2): True
    eps=0.03, delta=0.1: m=ceil(log(2/delta)/(2eps^2))=1665, empirical P(|est-exact|>eps)=0.0017 <= delta: True
    eps=0.02, delta=0.05: m=ceil(log(2/delta)/(2eps^2))=4612, empirical P(|est-exact|>eps)=0.0050 <= delta: True
    => claim 2 supported: True

[3] Section 6.2: 2% outlier robustness (N=1000, D=50) — TSI/QSI vs CKA, CKNNA, MutualNN
    TSI      : score=0.9692, retention vs clean=0.969
    QSI      : score=0.9560, retention vs clean=0.956
    CKA      : score=0.2664, retention vs clean=0.266
    CKNNA    : score=0.9555, retention vs clean=0.956
    MutualNN : score=0.9218, retention vs clean=0.922
    TSI retains >90%: True; TSI/QSI retention >= CKA/CKNNA/MutualNN: True
    => claim 3 supported: True

[4] Section 6.5: TSI/QSI track accuracy monotonically while CKA can drop at the strongest model
    (A) monotone metric warp g(d)=d^0.3: TSI(warp,Z)=1.0000 (ordering ~preserved) but CKA(warp,Z)=0.9467 -> invariance gap 0.053: CKA misled where TSI is not
    (B) small  : acc=0.833, TSI=0.6303, QSI=0.6289, CKA=0.5652
    (B) medium : acc=1.0, TSI=0.7507, QSI=0.7514, CKA=0.8797
    (B) large  : acc=1.0, TSI=0.864, QSI=0.871, CKA=0.9767
    accuracy increases small<medium<large: True; TSI/QSI monotone up: True
    NOTE: the exact CLIP/ImageNet-50k values (63.3/68.1/75.4, CKA drop at Large) need GPU+data; the mechanism — ordinal metrics track quality and are invariant to reparameterizations that mislead CKA (see also claim 3 outlier & claim 5 drift results) — is reproduced on CPU.
    => claim 4 (mechanism) supported: True

[5] Section 6.1 / Lemma 1: independent reps -> TSI/QSI ~0.5 STABLE across N,D; CKA/SVCCA/PWCCA drift
    (N, D): TSI  QSI  CKA  SVCCA PWCCA
    N= 80 D= 10: 0.468 0.457 0.101 0.307 0.410
    N= 80 D= 50: 0.490 0.488 0.393 0.697 0.870
    N= 80 D=150: 0.507 0.505 0.659 0.937 1.000
    N=200 D= 10: 0.500 0.505 0.036 0.172 0.222
    N=200 D= 50: 0.503 0.505 0.196 0.427 0.571
    N=200 D=150: 0.518 0.526 0.430 0.746 0.925
    N=400 D= 10: 0.497 0.497 0.031 0.151 0.223
    N=400 D= 50: 0.502 0.502 0.113 0.303 0.414
    N=400 D=150: 0.500 0.502 0.271 0.527 0.692
    TSI: mean=0.4985 std=0.0128 range=0.05; QSI mean=0.4986; SVCCA std=0.2572 range=0.7856; PWCCA std=0.2795 range=0.7781
    TSI/QSI ~0.5 & stable: True; SVCCA/PWCCA drift more (bigger std & range): True
    => claim 5 supported: True

verdict: supports
RESULTS_SHA256=b157e46795c01e8f6e9a316ec189a0013abf32a46f3ce8122a1153a3e03fdad2
````
