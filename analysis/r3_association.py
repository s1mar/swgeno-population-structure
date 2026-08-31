"""R3: the motif-outcome association scan, pooled against task-conditioned, with the inflation
diagnostic and its permutation null.

For each alphabet:
  * every k-gram (k <= 3) with at least MIN_SUPPORT runs on each side is tested for association
    with the outcome, once by the pooled chi-square that ignores the task and once by the
    Cochran-Mantel-Haenszel chi-square stratified by task;
  * lambda, the genomic inflation factor, is reported for both;
  * the labels are permuted WITHIN task and both lambdas recomputed. That null preserves each
    task's resolve rate and each task's motif distribution and destroys only the run-level
    association, so pooled lambda under it is stratification alone and conditioned lambda under it
    is the calibration check;
  * survivors are motifs passing BH-FDR with a common-odds-ratio floor, both fixed in the spec;
  * everything is repeated on the first PREFIX steps of each run, which is the only version in
    which the behaviour is observed before the outcome exists.

Writes ../data/r3_association.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llib  # noqa: E402

MIN_SUPPORT = 20
KMAX = 3
N_PERM = 200
FDR_Q = 0.05
OR_FLOOR = 1.5          # a survivor needs a common OR outside [1/1.5, 1.5]
PREFIX = 10
BAND = (0.10, 0.90)     # prevalence band for the survivor catalogue
# A median over a handful of motifs is not an inflation factor. Below this count lambda is
# computed but flagged unreportable rather than quietly used.
LAMBDA_MIN_MOTIFS = 50
ALPHABETS = ("xepv", "xepvb", "l1", "l2", "l3")


def permute_within(y: np.ndarray, strata: np.ndarray, rng) -> np.ndarray:
    out = y.copy()
    order = np.argsort(strata, kind="stable")
    s = strata[order]
    bounds = np.flatnonzero(np.r_[True, s[1:] != s[:-1], True])
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        idx = order[lo:hi]
        out[idx] = rng.permutation(y[idx])
    return out


def scan(rows, alpha, prefix=None, n_perm=N_PERM, seed=0, drop_submit=False,
         support_band=None, drop_terminal=False) -> dict:
    motifs, P, y, strata = llib.presence_matrix(rows, alpha, KMAX, MIN_SUPPORT, prefix,
                                                drop_submit=drop_submit,
                                                support_band=support_band,
                                                drop_terminal=drop_terminal)
    if not motifs:
        return {"n_motifs": 0}
    chi_p = llib.pooled_chi2(P, y)
    chi_c, orr = llib.cmh(P, y, strata)

    rng = np.random.default_rng(seed)
    lp, lc = [], []
    # The mean SORTED null statistic is what a QQ plot needs: it is the curve the observed
    # quantiles would trace if there were no run-level association at all.
    qp = np.zeros(len(motifs))
    qc = np.zeros(len(motifs))
    for _ in range(n_perm):
        yp = permute_within(y, strata, rng)
        cp_ = llib.pooled_chi2(P, yp)
        cc_ = llib.cmh(P, yp, strata)[0]
        lp.append(llib.lambda_gc(cp_))
        lc.append(llib.lambda_gc(cc_))
        qp += np.sort(cp_)
        qc += np.sort(cc_)
    qp /= n_perm
    qc /= n_perm

    p_pool = llib.chi2_sf(chi_p)
    p_cond = llib.chi2_sf(chi_c)
    keep_c = llib.bh_fdr(p_cond, FDR_Q)
    keep_p = llib.bh_fdr(p_pool, FDR_Q)
    big = np.isfinite(orr) & ((orr >= OR_FLOOR) | (orr <= 1 / OR_FLOOR))
    surv = keep_c & big

    # pooled odds ratio, for showing the shrinkage side by side
    a = (P & y[:, None]).sum(0).astype(float)
    b = (P & ~y[:, None]).sum(0).astype(float)
    c = (~P & y[:, None]).sum(0).astype(float)
    d = (~P & ~y[:, None]).sum(0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        or_pool = ((a + .5) * (d + .5)) / ((b + .5) * (c + .5))

    order = np.argsort(-chi_c)
    top = [{"motif": motifs[j], "chi_cond": float(chi_c[j]), "p_cond": float(p_cond[j]),
            "or_cond": float(orr[j]) if np.isfinite(orr[j]) else None,
            "chi_pooled": float(chi_p[j]), "p_pooled": float(p_pool[j]),
            "or_pooled": float(or_pool[j]),
            "n_present": int(P[:, j].sum()), "survivor": bool(surv[j])}
           for j in order[:40]]

    return {
        "n_motifs": len(motifs), "n_runs": int(len(y)),
        "n_strata": int(len(set(strata.tolist()))),
        "lambda_reportable": bool(len(motifs) >= LAMBDA_MIN_MOTIFS),
        "lambda_pooled": llib.lambda_gc(chi_p),
        "lambda_cond": llib.lambda_gc(chi_c),
        "lambda_pooled_null_mean": float(np.mean(lp)),
        "lambda_pooled_null_ci": [float(np.quantile(lp, .025)), float(np.quantile(lp, .975))],
        "lambda_cond_null_mean": float(np.mean(lc)),
        "lambda_cond_null_ci": [float(np.quantile(lc, .025)), float(np.quantile(lc, .975))],
        "n_fdr_pooled": int(keep_p.sum()), "n_fdr_cond": int(keep_c.sum()),
        "n_survivors": int(surv.sum()),
        "median_abs_log_or_pooled": float(np.median(np.abs(np.log(or_pool)))),
        "median_abs_log_or_cond": float(np.median(np.abs(np.log(
            orr[np.isfinite(orr) & (orr > 0)])))),
        "top": top,
        "survivors": [{"motif": motifs[j], "or_cond": float(orr[j]),
                       "or_pooled": float(or_pool[j]), "p_cond": float(p_cond[j]),
                       "n_present": int(P[:, j].sum())}
                      for j in np.argsort(-chi_c) if surv[j]][:60],
        # kept for the figures
        "_chi_pooled": chi_p.tolist(), "_chi_cond": chi_c.tolist(), "_motifs": motifs,
        "_or_cond": [float(x) if np.isfinite(x) else None for x in orr],
        "_p_cond": p_cond.tolist(),
        "_null_q_pooled": qp.tolist(), "_null_q_cond": qc.tolist(),
    }


def main() -> int:
    rows = llib.load_dual()
    out = {"config": {"min_support": MIN_SUPPORT, "kmax": KMAX, "n_perm": N_PERM,
                      "fdr_q": FDR_Q, "or_floor": OR_FLOOR, "prefix": PREFIX,
                      "n_runs": len(rows),
                      "n_instances": len({r["instance_id"] for r in rows})}}
    # Amendment 2: the pre-registered configuration is kept and reported, and a second
    # configuration removes the two ways a motif can be associated with the outcome for reasons
    # that are not behavioural: the submit action (an episode that never submits cannot be scored
    # resolved) and near-universal motifs (whose test is really a test of a degenerate minority).
    out["config"]["configurations"] = {
        "raw": "pre-registered: every k-gram with >= 20 runs on each side",
        "nosubmit": "raw, with the submit action AND the terminal action removed; this is the "
                    "configuration lambda is read from, because it keeps a wide, mostly-null "
                    "motif set",
        "nosubmit_band": "nosubmit, restricted to motifs with prevalence in [0.10, 0.90]; this is "
                         "the configuration the survivor catalogue is read from, because a motif "
                         "present in 95% of runs tests whether the other 5% are degenerate",
    }
    out["config"]["band"] = list(BAND)
    out["config"]["lambda_min_motifs"] = LAMBDA_MIN_MOTIFS
    for alpha in ALPHABETS:
        print(f"  scanning {alpha} ...", flush=True)
        out.setdefault("raw", {})[alpha] = scan(rows, alpha)
        out.setdefault("nosubmit", {})[alpha] = scan(rows, alpha, drop_submit=True,
                                                     drop_terminal=True)
        out.setdefault("nosubmit_band", {})[alpha] = scan(
            rows, alpha, drop_submit=True, support_band=BAND, drop_terminal=True)
        out.setdefault("prefix_nosubmit", {})[alpha] = scan(
            rows, alpha, prefix=PREFIX, drop_submit=True, drop_terminal=True)
        out.setdefault("prefix_nosubmit_band", {})[alpha] = scan(
            rows, alpha, prefix=PREFIX, drop_submit=True, support_band=BAND, drop_terminal=True)
    dst = os.path.join(llib.DATA, "r3_association.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
