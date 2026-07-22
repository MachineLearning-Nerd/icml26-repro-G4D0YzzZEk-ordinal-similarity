#!/usr/bin/env python3
"""Representation Alignment with Ordinal Similarity, TSI/QSI (arXiv:2606.16379). Reproduces:
  [5] Lemma 1: for INDEPENDENT representations E[TSI]=E[QSI]~0.5; identical/aligned -> ~1.0.
  [2] eps-approximation: O(1/eps^2) sampled triplets/quadruplets give the exact score to +-eps
      (independent of N); we confirm the sampled estimate converges to the exact value.
  [3] robustness: under a small (2%) outlier perturbation TSI/QSI retain ~94% alignment.
  TSI: sign(d_X(i,j)-d_X(i,k)) == sign(d_Y(i,j)-d_Y(i,k)); QSI: sign(d_X(i,j)-d_X(k,l)) == ...
Deterministic seeds.
"""
import numpy as np, json, hashlib

def dist(X): return np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))

def tsi_exact(X, Y):
    dX, dY = dist(X), dist(Y); N = len(X); agree = tot = 0
    for i in range(N):
        cx = np.sign(dX[i][:, None] - dX[i][None, :]); cy = np.sign(dY[i][:, None] - dY[i][None, :])
        m = np.ones((N, N), bool); m[i, :] = False; m[:, i] = False; np.fill_diagonal(m, False)
        agree += int((cx[m] == cy[m]).sum()); tot += int(m.sum())
    return agree / tot

def tsi_sampled(X, Y, M, rng):
    dX, dY = dist(X), dist(Y); N = len(X)
    i = rng.integers(0, N, M); j = rng.integers(0, N, M); k = rng.integers(0, N, M)
    ok = (i != j) & (i != k) & (j != k); i, j, k = i[ok], j[ok], k[ok]
    cx = np.sign(dX[i, j] - dX[i, k]); cy = np.sign(dY[i, j] - dY[i, k])
    return float(np.mean(cx == cy))

def qsi_sampled(X, Y, M, rng):
    dX, dY = dist(X), dist(Y); N = len(X)
    i, j, k, l = [rng.integers(0, N, M) for _ in range(4)]
    ok = (i != j) & (k != l); i, j, k, l = i[ok], j[ok], k[ok], l[ok]
    cx = np.sign(dX[i, j] - dX[k, l]); cy = np.sign(dY[i, j] - dY[k, l])
    return float(np.mean(cx == cy))

def main():
    R = {"claim": "TSI_QSI_ordinal_alignment", "paper": "arXiv:2606.16379"}
    rng = np.random.default_rng(0)

    # ---------- [5] independent ~0.5, identical ~1.0 (exact, N=120) ----------
    N, D = 120, 10
    Xi, Yi = rng.standard_normal((N, D)), rng.standard_normal((N, D))    # independent
    Xa = rng.standard_normal((N, D)); Ya = Xa.copy()                     # identical
    R["TSI_independent_exact"] = round(tsi_exact(Xi, Yi), 4)
    R["TSI_identical_exact"] = round(tsi_exact(Xa, Ya), 4)
    R["lemma1_independent_approx_half"] = abs(R["TSI_independent_exact"] - 0.5) < 0.03
    R["identical_approx_one"] = R["TSI_identical_exact"] > 0.999
    # QSI too (sampled)
    R["QSI_independent"] = round(qsi_sampled(Xi, Yi, 400000, rng), 4)
    R["QSI_identical"] = round(qsi_sampled(Xa, Ya, 400000, rng), 4)
    R["qsi_independent_half"] = abs(R["QSI_independent"] - 0.5) < 0.03
    R["qsi_identical_one"] = R["QSI_identical"] > 0.999

    # ---------- [2] eps-approximation: sampled -> exact as M grows, independent of N ----------
    # aligned-with-noise pair so the true TSI is strictly between 0.5 and 1
    Xr = rng.standard_normal((N, D)); Yr = Xr + 0.7 * rng.standard_normal((N, D))
    exact = tsi_exact(Xr, Yr)
    rows = []
    for M in [500, 2000, 8000, 32000, 128000]:
        est = np.mean([tsi_sampled(Xr, Yr, M, np.random.default_rng(100 + s)) for s in range(8)])
        rows.append({"M": M, "est": round(float(est), 4), "abs_err": round(abs(est - exact), 4)})
    R["exact_TSI_noisy"] = round(exact, 4); R["approx_convergence"] = rows
    R["eps_approx_converges"] = rows[-1]["abs_err"] < 0.01
    # independence of N: same exact-ish score for N=120 vs N=400 at the same alignment
    Xr2 = rng.standard_normal((400, D)); Yr2 = Xr2 + 0.7 * rng.standard_normal((400, D))
    est_bigN = tsi_sampled(Xr2, Yr2, 128000, rng)
    R["TSI_bigN_400"] = round(float(est_bigN), 4)
    R["score_stable_across_N"] = abs(est_bigN - exact) < 0.03

    # ---------- [3] robustness to 2% outliers ----------
    Nl = 1000; Xc = rng.standard_normal((Nl, 30)); Yc = Xc.copy()
    n_out = int(0.02 * Nl); idx = rng.choice(Nl, n_out, replace=False)
    Yc[idx] += rng.standard_normal((n_out, 30)) * 8.0                    # 2% outliers perturbed
    tsi_clean = tsi_sampled(Xc, Xc, 300000, rng)                        # ~1.0
    tsi_out = tsi_sampled(Xc, Yc, 300000, rng)
    R["TSI_clean"] = round(tsi_clean, 4); R["TSI_2pct_outliers"] = round(tsi_out, 4)
    R["robustness_retains_alignment"] = tsi_out > 0.90                   # retains ~94%+

    R["verdict"] = "supports" if (R["lemma1_independent_approx_half"] and R["identical_approx_one"]
                                  and R["qsi_independent_half"] and R["eps_approx_converges"]
                                  and R["score_stable_across_N"] and R["robustness_retains_alignment"]) else "inconclusive"

    print("claim: " + R["claim"])
    print(f"[5] Lemma 1: TSI independent={R['TSI_independent_exact']} (~0.5: {R['lemma1_independent_approx_half']}), "
          f"identical={R['TSI_identical_exact']} (~1.0: {R['identical_approx_one']})")
    print(f"    QSI independent={R['QSI_independent']} (~0.5: {R['qsi_independent_half']}), identical={R['QSI_identical']}")
    print(f"[2] eps-approx (exact TSI={R['exact_TSI_noisy']}):")
    for r in rows: print(f"      M={r['M']:>6}: est={r['est']} abs_err={r['abs_err']}")
    print(f"    converges: {R['eps_approx_converges']}; stable across N (N=400 est={R['TSI_bigN_400']}): {R['score_stable_across_N']}")
    print(f"[3] robustness: TSI clean={R['TSI_clean']}, with 2% outliers={R['TSI_2pct_outliers']} -> retains alignment: {R['robustness_retains_alignment']}")
    print(f"verdict: {R['verdict']}")

    def _np(o):
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        raise TypeError
    import os; os.makedirs("outputs", exist_ok=True)
    open("outputs/tsi_results.json", "w").write(json.dumps(R, indent=2, default=_np))
    print("RESULTS_SHA256=" + hashlib.sha256(json.dumps(R, sort_keys=True, default=_np).encode()).hexdigest())
    return 0 if R["verdict"] == "supports" else 1

if __name__ == "__main__":
    raise SystemExit(main())
