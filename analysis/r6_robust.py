"""R6: the three robustness checks an adversarial reader asked for, plus one they did not.

1. JOINT STRATIFICATION. Corpus A1 mixes three model sizes, so a within-task contrast still lets
   model identity vary inside a stratum. Redo every contrast and the motif scan stratified by
   (task, model) jointly, which removes it.
2. A CORPUS-WIDE INTERVAL. The whole-corpus pooled difference was reported as a point estimate.
   Without an interval it cannot be compared with the within-task estimate, which is exactly the
   comparison the sign-reversal claim rests on. Bootstrap it over tasks.
3. A SECOND ICC ESTIMATOR. The ANOVA intraclass correlation is not the only defensible estimator
   for a clustered binary outcome. Fleiss-Cuzick is the standard moment-based alternative; if the
   two agree the choice of estimator is not carrying the claim.

Writes ../data/r6_robust.json.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llib  # noqa: E402
import r1_replicate as r1  # noqa: E402
import r2_variance as r2  # noqa: E402
import r3_association as r3  # noqa: E402

MIN_RUNS = 4


def fleiss_cuzick(groups: list[list[int]]) -> float:
    """Moment estimator of the intraclass correlation for clustered binary data.

    rho = 1 - sum_i x_i (n_i - x_i) / (n_i - 1)  /  ((N - k) * pbar * qbar)
    """
    groups = [g for g in groups if len(g) >= 2]
    k = len(groups)
    N = sum(len(g) for g in groups)
    x = np.array([sum(g) for g in groups], float)
    n = np.array([len(g) for g in groups], float)
    p = x.sum() / N
    num = float((x * (n - x) / (n - 1)).sum())
    den = (N - k) * p * (1 - p)
    return 1.0 - num / den if den > 0 else float("nan")


def main() -> int:
    out = {}
    a0 = llib.load_tokens()
    a1 = llib.load_dual()

    # ---------------------------------------------------------------- 3. second ICC
    cells = defaultdict(list)
    for r in a0:
        if r["model"] == "swe-agent-llama-70b":
            cells[r["instance_id"]].append(int(bool(r["resolved"])))
    groups = [v for v in cells.values() if len(v) >= MIN_RUNS]
    fc = fleiss_cuzick(groups)
    lo, hi = r2.boot_over_tasks(groups, fleiss_cuzick, n_boot=1000)
    out["icc_fleiss_cuzick"] = {"icc": fc, "ci": [lo, hi], "k_tasks": len(groups),
                                "n_runs": int(sum(len(g) for g in groups))}

    # ---------------------------------------------------------------- 2. corpus interval
    print("  corpus-wide bootstrap ...", flush=True)
    out["corpus_contrast"] = {
        alpha: r1.contrast(a0, alpha, r1.group_stats, n_boot=1000)["pooled"]
        for alpha in ("xepvb",)
    }

    # ---------------------------------------------------------------- 1. joint strata
    # The stratum becomes (task, model). Everything else is unchanged, so any difference from the
    # task-only result is attributable to model identity varying inside a task.
    joint = [dict(r, instance_id=f"{r['instance_id']}|{r['model']}") for r in a1]
    n_joint = len({r["instance_id"] for r in joint})
    n_usable = sum(1 for _iid, rs in llib.group_by_instance(joint).items()
                   if any(x["resolved"] for x in rs) and any(not x["resolved"] for x in rs))
    out["joint"] = {"n_strata": n_joint, "n_strata_both_outcomes": n_usable,
                    "n_runs": len(joint),
                    "model_mix": {m: sum(1 for r in a1 if r["model"] == m)
                                  for m in sorted({r["model"] for r in a1})}}
    print("  joint-strata contrast ...", flush=True)
    out["joint"]["contrast"] = {
        alpha: r1.contrast(joint, alpha, r1.group_stats, n_boot=2000)
        for alpha in ("xepvb",)
    }
    for alpha in ("l1", "l2", "l3"):
        print(f"  joint-strata scan {alpha} ...", flush=True)
        out["joint"].setdefault("scan", {})[alpha] = r3.scan(joint, alpha, n_perm=200,
                                                             drop_submit=True)

    # Within-task model composition: is the resolved arm systematically a stronger model?
    comp = defaultdict(lambda: defaultdict(int))
    for r in a1:
        comp["resolved" if r["resolved"] else "failed"][r["model"]] += 1
    out["a1_model_by_outcome"] = {k: dict(v) for k, v in comp.items()}

    dst = os.path.join(llib.DATA, "r6_robust.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
