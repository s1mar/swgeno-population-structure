"""R12: does the calibrated homogeneity test do what it claims?

The heterogeneity analysis in R10 reads Cochran's Q against a simulated homogeneous reference
rather than against the asymptotic chi-square, and reports that homogeneity is rejected for most
survivors. That conclusion is entirely a product of the calibration: against the textbook
chi-square not one survivor rejects. The calibration is defensible, because the Haldane correction
deflates Q on sparse strata, but this paper's own argument is that a test which finds something
means nothing until you show it could have found nothing. Every other analysis here has a planted
control. This one did not, which is the one place the paper failed to take its own advice.

So: plant motifs whose homogeneity is KNOWN, on A1's real strata at their real sizes and real
resolve rates, and ask what the calibrated test says.

  HOMOGENEOUS  the same within-task odds ratio in every task (r7_control.plant_real, which is
               exactly what "homogeneous" means). The rejection rate here is the FALSE POSITIVE
               rate of the calibrated test and should sit near its nominal level.
  HETEROGENEOUS the odds ratio is or_hi in half the tasks and 1/or_hi in the other half, assigned
               by a fixed hash of the task id so the split is deterministic. The rejection rate
               here is POWER. A test that never rejects a homogeneous motif is worthless if it
               also never rejects this one, which is why the control is two-sided.

Writes ../data/r12_het_control.json. Reads only data already in data/; plants nothing in the
manuscript's own results.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llib  # noqa: E402
import r7_control as r7  # noqa: E402
import r10_heterogeneity as r10  # noqa: E402

N_REP = int(os.environ.get("R12_REP", "200"))
N_NULL = int(os.environ.get("R12_NULL", "200"))
ALPHA = 0.05
OR_TRUE = r7.OR_TRUE          # 2.0, the same planted size the existing controls use
OR_HI = 3.0                   # heterogeneous arm: 3.0 in half the tasks, 1/3 in the other half


def plant_heterogeneous(y, strata, rng, or_hi=OR_HI, base=r7.BASE):
    """A motif whose odds ratio is or_hi in half the tasks and 1/or_hi in the other half.

    The split alternates over the task ids in sorted order, so which tasks are in which half is
    fixed both between replicates and between PROCESSES: the heterogeneity is a property of the
    planted world, not of the draw or of the interpreter.

    This used Python's built-in hash() of the task id, which is randomised per process unless
    PYTHONHASHSEED is set. The split therefore moved between runs, and while both headline
    rejection counts were stable, the result file was not byte-reproducible and the package's
    determinism claim was false. Caught by re-running the stage in a fresh clone and diffing.
    """
    odds = base / (1 - base)
    hi_task = {s: (i % 2 == 0) for i, s in enumerate(sorted(set(strata.tolist())))}
    ors = np.array([or_hi if hi_task[s] else 1.0 / or_hi for s in strata])
    p_res = (odds * ors) / (1 + odds * ors)
    p = np.where(y, p_res, base)
    return rng.random(len(y)) < p


def one_replicate(P, y, strata, rng):
    """Q, its homogeneous reference, and the resulting p, for a single planted motif."""
    chi, orr = llib.cmh(P, y, strata)
    A, B, C, D = r10.strata_tables(P, y, strata)
    q, _df, _i2, k, _same = r10.cochran_q(A[:, 0], B[:, 0], C[:, 0], D[:, 0], orr[0])
    if not np.isfinite(q) or k < 2:
        return None
    nul, _agree = r10.homogeneous_q_null(A[:, 0], B[:, 0], C[:, 0], D[:, 0], orr[0], N_NULL, rng)
    if nul.size == 0:
        return None
    p = (1 + int((nul >= q).sum())) / (1 + nul.size)
    return {"q": float(q), "k": int(k), "or_cmh": float(orr[0]), "p": float(p)}


def arm(rows, kind, n_rep, seed):
    y = np.array([bool(r["resolved"]) for r in rows])
    strata = np.array([r["instance_id"] for r in rows])
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_rep):
        if kind == "homogeneous":
            present = r7.plant_real(y, rng, or_true=OR_TRUE)
        else:
            present = plant_heterogeneous(y, strata, rng)
        rec = one_replicate(present.reshape(-1, 1), y, strata, rng)
        if rec is not None:
            out.append(rec)
        if (i + 1) % 25 == 0:
            print(f"    {kind}: {i + 1}/{n_rep}", flush=True)
    ps = np.array([r["p"] for r in out])
    return {"kind": kind, "n_rep": len(out), "alpha": ALPHA,
            "n_reject": int((ps <= ALPHA).sum()),
            "reject_rate": float((ps <= ALPHA).mean()) if ps.size else None,
            "median_p": float(np.median(ps)) if ps.size else None,
            "median_q_over_df": float(np.median([r["q"] / (r["k"] - 1) for r in out])),
            "median_or_cmh": float(np.median([r["or_cmh"] for r in out]))}


def main() -> int:
    rows = llib.load_dual()
    out = {"config": {"n_rep": N_REP, "n_null": N_NULL, "alpha": ALPHA,
                      "or_homogeneous": OR_TRUE, "or_heterogeneous_split": [OR_HI, 1.0 / OR_HI],
                      "base_prevalence": r7.BASE,
                      "source": "A1 (tokens_dual.jsonl), real strata at real sizes and rates",
                      "n_runs": len(rows),
                      "n_instances": len({r["instance_id"] for r in rows})}}
    for kind, seed in (("homogeneous", 11), ("heterogeneous", 12)):
        print(f"  planting {kind} ...", flush=True)
        out[kind] = arm(rows, kind, N_REP, seed)
        a = out[kind]
        print(f"    {kind}: rejects {a['n_reject']}/{a['n_rep']} "
              f"({a['reject_rate']:.3f}), median p {a['median_p']:.3f}")
    dst = os.path.join(llib.DATA, "r12_het_control.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
