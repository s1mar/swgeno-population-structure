"""R7: a positive and a negative control for the two estimators, on the real strata of corpus A1.

The paper argues that the pooled test is inflated by task structure and that the stratified test is
not. Both halves of that are checkable directly by planting motifs whose truth is known and seeing
what each test says. The strata, their sizes and their resolve rates are the real ones from A1, so
this is not a generic simulation: it is this corpus's structure with a synthetic feature laid over
it.

Two kinds of planted motif:

  REAL      present with a within-task odds ratio of OR_TRUE against the outcome, identical in
            every task. Both tests should find it, and the stratified estimate should recover
            OR_TRUE.
  SPURIOUS  present with a prevalence that tracks the task's own resolve rate, and drawn
            INDEPENDENTLY of each run's outcome. There is no run-level association at all. The
            pooled test should find one anyway; the stratified test should not.

Writes ../data/r7_control.json.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llib  # noqa: E402

N_REP = 200
OR_TRUE = 2.0
BASE = 0.40          # prevalence in failed runs for the REAL motif
ALPHA = 0.05


def plant_real(y, rng, or_true=OR_TRUE, base=BASE):
    """Within each task, present with odds base-odds x or_true when resolved."""
    odds = base / (1 - base)
    p_res = (odds * or_true) / (1 + odds * or_true)
    p = np.where(y, p_res, base)
    return rng.random(len(y)) < p


def plant_spurious(y, strata, rng, lo=0.15, hi=0.75):
    """Prevalence tracks the task's resolve rate; presence is independent of the run's outcome."""
    out = np.zeros(len(y), dtype=bool)
    for s in np.unique(strata):
        m = strata == s
        rate = y[m].mean()
        q = lo + (hi - lo) * rate
        out[m] = rng.random(m.sum()) < q
    return out


def run(rows):
    y = np.array([bool(r["resolved"]) for r in rows])
    strata = np.array([r["instance_id"] for r in rows])
    rng = np.random.default_rng(0)
    res = {}
    for kind, fn in (("real", lambda: plant_real(y, rng)),
                     ("spurious", lambda: plant_spurious(y, strata, rng))):
        chi_p, chi_c, or_p, or_c = [], [], [], []
        for _ in range(N_REP):
            P = fn().reshape(-1, 1)
            cp = llib.pooled_chi2(P, y)
            cc, oc = llib.cmh(P, y, strata)
            a = float((P[:, 0] & y).sum())
            b = float((P[:, 0] & ~y).sum())
            c = float((~P[:, 0] & y).sum())
            d = float((~P[:, 0] & ~y).sum())
            chi_p.append(float(cp[0]))
            chi_c.append(float(cc[0]))
            or_p.append(((a + .5) * (d + .5)) / ((b + .5) * (c + .5)))
            or_c.append(float(oc[0]))
        crit = 3.841458820694124        # chi-square(1) at 0.05
        res[kind] = {
            "n_replicates": N_REP,
            "pooled_reject_rate": float(np.mean(np.array(chi_p) > crit)),
            "cond_reject_rate": float(np.mean(np.array(chi_c) > crit)),
            "pooled_or_median": float(np.median(or_p)),
            "cond_or_median": float(np.median(or_c)),
            "pooled_chi_median": float(np.median(chi_p)),
            "cond_chi_median": float(np.median(chi_c)),
        }
    res["config"] = {"or_true": OR_TRUE, "base_prevalence": BASE, "alpha": ALPHA,
                     "n_runs": int(len(y)), "n_strata": int(len(set(strata.tolist())))}
    return res


def power_of_stratification(rows, strata_key, or_true, n_rep=N_REP, seed=1):
    """Rejection rate of the stratified test when a real effect of known size is planted.

    A stratified test that finds nothing is only informative if it could have found something.
    Corpus B is stratified two ways and reports an association under one and not the other; this
    measures whether the second stratification has the power to detect an effect at all.
    """
    y = np.array([bool(r["resolved"]) for r in rows])
    strata = np.array([r[strata_key] for r in rows])
    rng = np.random.default_rng(seed)
    crit = 3.841458820694124
    chi, orr = [], []
    for _ in range(n_rep):
        P = plant_real(y, rng, or_true=or_true).reshape(-1, 1)
        c, o = llib.cmh(P, y, strata)
        chi.append(float(c[0]))
        orr.append(float(o[0]))
    return {"or_planted": or_true, "n_runs": int(len(y)),
            "n_strata": int(len(set(strata.tolist()))),
            "reject_rate": float(np.mean(np.array(chi) > crit)),
            "or_median": float(np.median(orr)),
            "chi_median": float(np.median(chi))}


def main() -> int:
    rows = llib.load_dual()
    out = {"a1": run(rows)}

    # Corpus B: the paper reports an association when stratifying by task and none when
    # stratifying by model, and reads the second as evidence. That reading is only allowed if the
    # model-stratified test could detect an effect. Measure it on the same runs.
    import collections
    b = llib.load_tokens(os.path.join(llib.DATA, "tokens_frontier.jsonl"))
    by_inst = collections.defaultdict(list)
    for r in b:
        by_inst[r["instance_id"]].append(r)
    disc = [r for rs in by_inst.values() if len(rs) >= 2
            and any(x["resolved"] for x in rs) and any(not x["resolved"] for x in rs)
            for r in rs]
    out["b_power"] = {
        "by_model": [power_of_stratification(disc, "model", o) for o in (1.5, 2.0, 3.0, 5.0)],
        "by_task": [power_of_stratification(disc, "instance_id", o) for o in (1.5, 2.0, 3.0, 5.0)],
    }
    dst = os.path.join(llib.DATA, "r7_control.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
