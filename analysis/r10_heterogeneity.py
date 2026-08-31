"""R10: does the Cochran-Mantel-Haenszel common odds ratio hide task-to-task variation?

All three reviewers asked the same question: a common odds ratio summarises heterogeneous
per-stratum relationships, and a motif could help on one kind of task and hurt on another. This
script keeps the per-stratum 2x2 tables that `llib.cmh` accumulates and discards, and reports, for
every motif in the survivor catalogue:

  * the number of INFORMATIVE strata (those varying in both motif and outcome; the rest carry no
    within-task information and contribute nothing to CMH either);
  * the fraction of informative strata whose own odds ratio falls on the same side of 1 as the
    common odds ratio, which is the question reviewer 33C asked literally;
  * Breslow-Day, the canonical homogeneity test for a CMH analysis, and the I^2 derived from
    Cochran's Q on the Haldane-corrected per-stratum log odds ratios;
  * an exact calibration for Q. Strata here are small, and Q on sparse 2x2 tables is not
    trustworthy against its asymptotic chi-square: the Haldane correction shrinks each per-stratum
    log odds ratio toward zero by an amount that depends on the stratum's size, which inflates Q
    on its own. The reference therefore has to be a HOMOGENEOUS world with this motif's own common
    odds ratio, not a null world with no effect. For each motif we hold every stratum's margins
    fixed and redraw its 2x2 table from Fisher's noncentral hypergeometric distribution at exactly
    the estimated common odds ratio, which is homogeneity by construction, and read Q against the
    distribution that produces. Permuting the labels instead would set the odds ratio to one and
    calibrate the wrong quantity.

Adds to and changes nothing in the published results; writes ../data/r10_heterogeneity.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import nchypergeom_fisher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llib  # noqa: E402
import r3_association as r3  # noqa: E402

HALDANE = 0.5


def strata_tables(P: np.ndarray, y: np.ndarray, strata: np.ndarray):
    """Per-stratum 2x2 counts for every motif, shape (n_strata, n_motifs)."""
    order = np.argsort(strata, kind="stable")
    P, y, s = P[order], y[order], strata[order]
    bounds = np.flatnonzero(np.r_[True, s[1:] != s[:-1], True])
    A, B, C, D = [], [], [], []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi - lo < 2:
            continue
        Pk, yk = P[lo:hi], y[lo:hi]
        A.append((Pk & yk[:, None]).sum(0))
        B.append((Pk & ~yk[:, None]).sum(0))
        C.append((~Pk & yk[:, None]).sum(0))
        D.append((~Pk & ~yk[:, None]).sum(0))
    return (np.array(A, float), np.array(B, float), np.array(C, float), np.array(D, float))


def _informative(a, b, c, d):
    return ((a + b) > 0) & ((c + d) > 0) & ((a + c) > 0) & ((b + d) > 0)


def breslow_day(a, b, c, d, or_common):
    """Breslow-Day homogeneity statistic for one motif over its informative strata."""
    keep = _informative(a, b, c, d)
    a, b, c, d = a[keep], b[keep], c[keep], d[keep]
    k = a.size
    if k < 2 or not np.isfinite(or_common) or or_common <= 0:
        return float("nan"), 0
    n1 = a + b
    n2 = c + d
    m1 = a + c
    n = n1 + n2
    psi = float(or_common)
    if abs(psi - 1.0) < 1e-9:
        afit = n1 * m1 / n
    else:
        aa = psi - 1.0
        bb = (n1 + m1) * (1.0 - psi) - n
        cc = psi * n1 * m1
        disc = np.maximum(bb * bb - 4 * aa * cc, 0.0)
        r1 = (-bb - np.sqrt(disc)) / (2 * aa)
        r2 = (-bb + np.sqrt(disc)) / (2 * aa)
        lo = np.maximum(0.0, m1 - n2)
        hi = np.minimum(n1, m1)
        afit = np.where((r1 >= lo - 1e-9) & (r1 <= hi + 1e-9), r1, r2)
        afit = np.clip(afit, lo, hi)
    eps = 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        var = 1.0 / (1.0 / np.maximum(afit, eps)
                     + 1.0 / np.maximum(n1 - afit, eps)
                     + 1.0 / np.maximum(m1 - afit, eps)
                     + 1.0 / np.maximum(n2 - m1 + afit, eps))
        term = (a - afit) ** 2 / var
    term = term[np.isfinite(term)]
    return float(term.sum()), int(k - 1)


def cochran_q(a, b, c, d, or_common):
    """Cochran's Q, I^2 and directional consistency on Haldane-corrected per-stratum log ORs."""
    keep = _informative(a, b, c, d)
    a = a[keep] + HALDANE
    b = b[keep] + HALDANE
    c = c[keep] + HALDANE
    d = d[keep] + HALDANE
    k = int(a.size)
    if k < 2 or not np.isfinite(or_common) or or_common <= 0:
        return float("nan"), 0, float("nan"), k, float("nan")
    lor = np.log((a * d) / (b * c))
    w = 1.0 / (1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    q = float(np.sum(w * (lor - np.log(or_common)) ** 2))
    df = k - 1
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
    # Directional agreement. A stratum whose corrected log odds ratio is exactly zero agrees with
    # neither direction, so it is EXCLUDED rather than silently counted as disagreeing: about 2%
    # of informative strata are exact ties, and scoring them as disagreement biases the fraction
    # downward, which is the direction the surrounding argument is least entitled to.
    sgn = np.sign(lor)
    tied = sgn == 0
    same = (float(np.mean(sgn[~tied] == np.sign(np.log(or_common))))
            if (~tied).any() else float("nan"))
    return q, df, float(i2), k, same


def homogeneous_q_null(a, b, c, d, or_common, n_rep, rng):
    """Q under homogeneity at this motif's own common odds ratio, margins held fixed.

    For every informative stratum the row and column totals are kept and the cell count `a` is
    redrawn from Fisher's noncentral hypergeometric distribution at odds ratio `or_common`. That
    is a world in which the effect is identical in every task, so the Q it produces is what Q
    looks like on THESE strata when there is nothing to detect. Reading the observed Q against it
    absorbs both the sparseness of the tables and the Haldane shrinkage.
    """
    keep = _informative(a, b, c, d)
    a, b, c, d = a[keep], b[keep], c[keep], d[keep]
    if a.size < 2 or not np.isfinite(or_common) or or_common <= 0:
        return np.empty(0)
    n1 = (a + b).astype(int)          # motif present
    n2 = (c + d).astype(int)          # motif absent
    m1 = (a + c).astype(int)          # resolved
    ntot = n1 + n2
    out, agree = [], []
    for _ in range(n_rep):
        aa = nchypergeom_fisher.rvs(ntot, n1, m1, float(or_common),
                                    random_state=rng).astype(float)
        bb = n1 - aa
        cc = m1 - aa
        dd = n2 - cc
        q, _df, _i2, _k, same = cochran_q(aa, bb, cc, dd, or_common)
        if np.isfinite(q):
            out.append(q)
        if np.isfinite(same):
            agree.append(same)
    return np.asarray(out, float), np.asarray(agree, float)


def analyse(rows, alpha, n_perm, seed, tag):
    motifs, P, y, strata = llib.presence_matrix(
        rows, alpha, r3.KMAX, r3.MIN_SUPPORT, None,
        drop_submit=True, support_band=r3.BAND, drop_terminal=True)
    if not motifs:
        return {"n_motifs": 0}
    chi_c, orr = llib.cmh(P, y, strata)
    p_cond = llib.chi2_sf(chi_c)
    keep_c = llib.bh_fdr(p_cond, r3.FDR_Q)
    big = np.isfinite(orr) & ((orr >= r3.OR_FLOOR) | (orr <= 1 / r3.OR_FLOOR))
    surv = [int(j) for j in np.flatnonzero(keep_c & big)]
    A, B, C, D = strata_tables(P, y, strata)

    rng = np.random.default_rng(seed)
    nulls = {j: homogeneous_q_null(A[:, j], B[:, j], C[:, j], D[:, j], orr[j], n_perm, rng)
             for j in surv}

    rowsout = []
    for j in surv:
        q, df, i2, k, same = cochran_q(A[:, j], B[:, j], C[:, j], D[:, j], orr[j])
        bd, bddf = breslow_day(A[:, j], B[:, j], C[:, j], D[:, j], orr[j])
        nul, agree_null = nulls[j]
        rowsout.append({
            "motif": motifs[j],
            "or_cond": float(orr[j]),
            "chi_cond": float(chi_c[j]),
            "n_present": int(P[:, j].sum()),
            "n_informative_strata": k,
            "frac_strata_same_sign": None if not np.isfinite(same) else round(float(same), 4),
            # The agreement fraction means nothing on its own: even a perfectly homogeneous world
            # produces disagreement, because each stratum's odds ratio is estimated from a handful
            # of runs. The same homogeneous reference used for Q says what agreement to expect.
            "frac_same_sign_homog_mean": None if agree_null.size == 0 else round(
                float(agree_null.mean()), 4),
            "frac_same_sign_homog_lo": None if agree_null.size == 0 else round(
                float(np.quantile(agree_null, .025)), 4),
            "cochran_q": None if not np.isfinite(q) else round(q, 3),
            "q_df": df,
            "i2": None if not np.isfinite(i2) else round(i2, 4),
            "breslow_day": None if not np.isfinite(bd) else round(bd, 3),
            "breslow_day_df": bddf,
            # I^2 against the chi-square df is conservative here: the simulation shows Q's
            # homogeneous expectation on these sparse strata is well below df. Recentring I^2 on
            # the measured homogeneous mean is the honest version, and it is the one reported.
            "i2_calibrated": None if (nul.size == 0 or not np.isfinite(q) or q <= 0) else round(
                max(0.0, float((q - nul.mean()) / q)), 4),
            "q_homog_mean": None if nul.size == 0 else round(float(nul.mean()), 3),
            "q_homog_p95": None if nul.size == 0 else round(float(np.quantile(nul, .95)), 3),
            "q_homog_p": None if nul.size == 0 or not np.isfinite(q) else round(
                float((1 + int((nul >= q).sum())) / (1 + nul.size)), 4),
        })
    rowsout.sort(key=lambda r: -r["chi_cond"])
    # Multiplicity. The motif scan itself is false-discovery controlled, so reporting a count of
    # homogeneity rejections at a nominal 0.05 across dozens of survivors would hold the
    # heterogeneity question to a looser standard than the paper holds its own findings to.
    pv = np.array([r["q_homog_p"] if r["q_homog_p"] is not None else 1.0 for r in rowsout])
    keep = llib.bh_fdr(pv, r3.FDR_Q) if pv.size else np.zeros(0, bool)
    for r, k in zip(rowsout, keep):
        r["q_homog_rejected_bh"] = bool(k)
    return {"alphabet": alpha, "tag": tag, "n_motifs": len(motifs),
            "n_survivors": len(rowsout), "n_strata_total": int(len(set(strata.tolist()))),
            "n_perm": n_perm, "survivors": rowsout}


def main() -> int:
    rows = llib.load_dual()
    n_perm = int(os.environ.get("R10_PERM", "200"))
    out = {"config": {"source": "A1 (tokens_dual.jsonl)", "n_runs": len(rows),
                      "n_instances": len({r["instance_id"] for r in rows}),
                      "configuration": "nosubmit_band, the survivor-catalogue configuration",
                      "n_perm": n_perm, "haldane": HALDANE,
                      "min_support": r3.MIN_SUPPORT, "band": list(r3.BAND),
                      "fdr_q": r3.FDR_Q, "or_floor": r3.OR_FLOOR}}
    for alpha in ("l3", "l2"):
        print(f"  heterogeneity {alpha} ...", flush=True)
        out[alpha] = analyse(rows, alpha, n_perm, seed=0, tag="nosubmit_band")
    dst = os.path.join(llib.DATA, "r10_heterogeneity.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
