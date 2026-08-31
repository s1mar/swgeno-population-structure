"""R2: how much of a coding agent's outcome is fixed by the task, and how much is run-level noise.

Run on A0, the UNSELECTED corpus (every run of the public corpus), because the dual-outcome corpus
is ascertained on the phenotype and would force the answer.

Three quantities, all per action-generator model so that model identity is never the source of the
between-task variance:
  * ICC, the intraclass correlation of the binary outcome across repeated runs of the same task,
    by the one-way random-effects estimator with the unequal-group-size correction;
  * the same on the liability scale (Dempster-Lerner), which is what a geneticist would call the
    repeatability of a threshold trait and is an upper bound on heritability, not heritability;
  * the discordance rate, a statistic with no model behind it: among tasks run more than once,
    how often do two runs of the same task disagree?

Writes ../data/r2_variance.json.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llib  # noqa: E402

MIN_RUNS = 4          # cells smaller than this carry almost no within-task information
N_BOOT = 2000


def icc_oneway(groups: list[list[int]]) -> dict:
    """One-way random-effects ICC for a binary outcome, ANOVA estimator.

    groups: one list of 0/1 outcomes per task. Unequal sizes handled by the standard n0 term.
    """
    groups = [g for g in groups if len(g) >= 2]
    k = len(groups)
    if k < 2:
        return {"icc": None}
    ns = np.array([len(g) for g in groups], dtype=float)
    means = np.array([np.mean(g) for g in groups], dtype=float)
    N = ns.sum()
    grand = float(sum(sum(g) for g in groups) / N)
    msb = float((ns * (means - grand) ** 2).sum() / (k - 1))
    ssw = float(sum(((np.array(g) - m) ** 2).sum() for g, m in zip(groups, means)))
    msw = ssw / (N - k) if N > k else float("nan")
    n0 = (N - (ns ** 2).sum() / N) / (k - 1)
    denom = msb + (n0 - 1) * msw
    icc = (msb - msw) / denom if denom > 0 else float("nan")
    return {"icc": float(icc), "k_tasks": k, "n_runs": int(N), "n0": float(n0),
            "p": grand, "msb": msb, "msw": msw}


def liability(icc: float, p: float) -> float:
    """Dempster-Lerner: observed-scale ICC for a 0/1 trait -> liability scale.

    z is the height of the standard normal density at the threshold that gives prevalence p.
    """
    if not (0 < p < 1) or icc is None or not np.isfinite(icc):
        return float("nan")
    # inverse normal CDF by bisection: no scipy dependency anywhere in this pipeline
    lo, hi = -10.0, 10.0
    target = 1 - p
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < target:
            lo = mid
        else:
            hi = mid
    t = (lo + hi) / 2
    z = math.exp(-t * t / 2) / math.sqrt(2 * math.pi)
    return float(icc * p * (1 - p) / (z * z))


def discordance(groups: list[list[int]]) -> dict:
    """Pairwise disagreement within a task, against the value expected if runs were independent
    draws at the overall rate (which is what ICC = 0 means)."""
    groups = [g for g in groups if len(g) >= 2]
    num = den = 0.0
    both = 0
    for g in groups:
        n, s = len(g), sum(g)
        num += 2.0 * s * (n - s)
        den += n * (n - 1.0)
        both += 1 if 0 < s < n else 0
    p = float(sum(sum(g) for g in groups) / sum(len(g) for g in groups))
    obs = num / den if den else float("nan")
    exp = 2 * p * (1 - p)
    return {"observed": obs, "expected_if_independent": exp,
            "ratio": obs / exp if exp else float("nan"),
            "frac_tasks_discordant": both / len(groups) if groups else float("nan"),
            "k_tasks": len(groups), "p": p}


def boot_over_tasks(groups, fn, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    k = len(groups)
    for _ in range(n_boot):
        idx = rng.integers(0, k, k)
        v = fn([groups[i] for i in idx])
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return None, None
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def main() -> int:
    rows = llib.load_tokens()
    out = {"corpus": {"n_runs": len(rows),
                      "n_instances": len({r["instance_id"] for r in rows}),
                      "n_models": len({r["model"] for r in rows}),
                      "resolution_rate": float(np.mean([r["resolved"] for r in rows]))}}

    by_model = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_model[r["model"]][r["instance_id"]].append(int(bool(r["resolved"])))

    out["models"] = {}
    for model, cells in sorted(by_model.items()):
        groups = [v for v in cells.values() if len(v) >= MIN_RUNS]
        if len(groups) < 20:
            out["models"][model] = {"skipped": True, "n_tasks_with_min_runs": len(groups),
                                    "n_runs_total": sum(len(v) for v in cells.values()),
                                    "resolution_rate": float(np.mean(
                                        [x for v in cells.values() for x in v]))}
            continue
        a = icc_oneway(groups)
        d = discordance(groups)
        lo, hi = boot_over_tasks(groups, lambda gs: icc_oneway(gs)["icc"])
        dlo, dhi = boot_over_tasks(groups, lambda gs: discordance(gs)["ratio"])
        out["models"][model] = {
            "min_runs": MIN_RUNS, **a,
            "icc_ci": [lo, hi],
            "icc_liability": liability(a["icc"], a["p"]),
            "discordance": d, "discordance_ratio_ci": [dlo, dhi],
            "n_runs_total": sum(len(v) for v in cells.values()),
            "resolution_rate_all_runs": float(np.mean([x for v in cells.values() for x in v])),
        }

    # Confounding needs BOTH arms: the task must predict the outcome AND predict the behaviour.
    # The outcome side is above; this is the behaviour side, on the same cells and the same
    # estimator, so the two numbers are comparable.
    import r1_replicate as r1
    beh = defaultdict(lambda: defaultdict(list))
    for r in rows:
        s = llib.seq(r, "xepvb", drop_submit=True)
        u = r1.summarize(s)
        n = len(s)
        if n == 0:
            continue
        beh[r["model"]][r["instance_id"]].append(
            {"x_ratio": s.count("X") / n, "v_ratio": s.count("V") / n,
             "e_ratio": s.count("E") / n, "steps": float(n),
             "max_x_run": float(u["maxx"])})
    out["behaviour_icc"] = {}
    for model, cells in sorted(beh.items()):
        cells = {i: v for i, v in cells.items() if len(v) >= MIN_RUNS}
        if len(cells) < 20:
            continue
        entry = {"k_tasks": len(cells), "n_runs": sum(len(v) for v in cells.values())}
        for key in ("x_ratio", "v_ratio", "e_ratio", "steps", "max_x_run"):
            groups = [[d[key] for d in v] for v in cells.values()]
            a = icc_oneway(groups)
            lo, hi = boot_over_tasks(groups, lambda gs: icc_oneway(gs)["icc"], n_boot=500)
            entry[key] = {"icc": a["icc"], "ci": [lo, hi]}
        out["behaviour_icc"][model] = entry

    # Repo level: is difficulty a property of the project rather than the issue?
    by_repo = defaultdict(list)
    for r in rows:
        by_repo[r["repo"]].append(int(bool(r["resolved"])))
    rgroups = [v for v in by_repo.values() if len(v) >= MIN_RUNS]
    ra = icc_oneway(rgroups)
    rlo, rhi = boot_over_tasks(rgroups, lambda gs: icc_oneway(gs)["icc"])
    out["repo_level"] = {**ra, "icc_ci": [rlo, rhi],
                         "icc_liability": liability(ra["icc"], ra["p"])}

    # Model main effect, for comparison with the task effect.
    out["model_marginals"] = {m: {"n": sum(len(v) for v in cells.values()),
                                  "rate": float(np.mean([x for v in cells.values() for x in v]))}
                              for m, cells in sorted(by_model.items())}

    dst = os.path.join(llib.DATA, "r2_variance.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
