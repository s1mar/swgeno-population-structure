"""R5: does the whole thing replicate on a different scaffold, model and benchmark?

Corpus C is OpenHands with Qwen3-Coder-480B on SWE-rebench. Scaffold, action-generator model and
benchmark all differ from corpus A, so a behavioural difference between A and C cannot be
attributed to any one of them and no such attribution is made. What can be tested here is the
claim that does not need them separated: that pooling runs across tasks inflates a motif-outcome
association, and that conditioning on the task removes the inflation.

Unlike corpus B, corpus C has many replicate runs of the same (task, model), so it supports the
same estimators as corpus A: outcome repeatability, behaviour repeatability, the pooled-versus-
conditioned contrast, and the permutation null.

Writes ../data/r5_openhands.json.
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
CAP = 12          # same per-outcome cap as corpus A1, so the two are built the same way


SYMS = ("EXEC", "VIEW", "EDIT", "CREATE", "OTHER")


def generic_contrast(dual, n_boot=2000, seed=0):
    """Pooled and within-task resolved-minus-failed differences for symbol shares and length."""
    per = defaultdict(lambda: ([], []))
    for r in dual:
        s = llib.seq(r, "l1", drop_submit=True)
        n = len(s) or 1
        v = {f"{k.lower()}_share": s.count(k) / n for k in SYMS}
        v["steps"] = float(len(s))
        per[r["instance_id"]][0 if r["resolved"] else 1].append(v)
    iids = sorted(per)
    keys = [f"{k.lower()}_share" for k in SYMS] + ["steps"]

    def stats(vs):
        return {k: float(np.mean([v[k] for v in vs])) for k in keys} if vs else None

    def pooled(idx):
        res = [v for j in idx for v in per[iids[j]][0]]
        fail = [v for j in idx for v in per[iids[j]][1]]
        a, b = stats(res), stats(fail)
        return {k: (a[k] - b[k]) if a and b else np.nan for k in keys}

    delta_i = np.full((len(iids), len(keys)), np.nan)
    for j, i in enumerate(iids):
        a, b = stats(per[i][0]), stats(per[i][1])
        if a and b:
            delta_i[j] = [a[k] - b[k] for k in keys]

    def within(idx):
        sub = delta_i[idx]
        return {k: float(np.nanmean(sub[:, c])) if np.isfinite(sub[:, c]).any() else np.nan
                for c, k in enumerate(keys)}

    allidx = np.arange(len(iids))
    point = {"pooled": pooled(allidx), "within": within(allidx)}
    rng = np.random.default_rng(seed)
    boots = {k: {m: [] for m in keys} for k in point}
    for _ in range(n_boot):
        idx = rng.integers(0, len(iids), len(iids))
        for k, dd in (("pooled", pooled(idx)), ("within", within(idx))):
            for m in keys:
                boots[k][m].append(dd[m])
    res = {"n_instances": len(iids)}
    for k in point:
        res[k] = {}
        for m in keys:
            b = np.array(boots[k][m], dtype=float)
            b = b[np.isfinite(b)]
            res[k][m] = {"delta": float(point[k][m]) if np.isfinite(point[k][m]) else None,
                         "lo": float(np.quantile(b, .025)) if b.size else None,
                         "hi": float(np.quantile(b, .975)) if b.size else None}
    return res


def main() -> int:
    rows = llib.load_tokens(os.path.join(llib.DATA, "tokens_openhands.jsonl"))
    out = {"corpus": {
        "n": len(rows), "n_instances": len({r["instance_id"] for r in rows}),
        "model": sorted({r["model"] for r in rows}),
        "resolution_rate": float(np.mean([r["resolved"] for r in rows])),
    }}

    by_inst = defaultdict(list)
    for r in rows:
        by_inst[r["instance_id"]].append(r)
    out["corpus"]["n_instances_ge2"] = sum(1 for v in by_inst.values() if len(v) >= 2)

    # --- repeatability, on the unselected corpus
    groups = [[int(bool(r["resolved"])) for r in v] for v in by_inst.values() if len(v) >= MIN_RUNS]
    a = r2.icc_oneway(groups)
    lo, hi = r2.boot_over_tasks(groups, lambda gs: r2.icc_oneway(gs)["icc"], n_boot=1000)
    d = r2.discordance(groups)
    out["outcome_icc"] = {**a, "icc_ci": [lo, hi], "discordance": d, "min_runs": MIN_RUNS}

    beh = defaultdict(list)
    for iid, v in by_inst.items():
        if len(v) < MIN_RUNS:
            continue
        for r in v:
            s = llib.seq(r, "l1", drop_submit=True)
            n = len(s) or 1
            beh[iid].append({"exec_ratio": s.count("EXEC") / n, "view_ratio": s.count("VIEW") / n,
                             "edit_ratio": s.count("EDIT") / n, "steps": float(len(s))})
    out["behaviour_icc"] = {}
    for key in ("exec_ratio", "view_ratio", "edit_ratio", "steps"):
        gs = [[x[key] for x in v] for v in beh.values()]
        aa = r2.icc_oneway(gs)
        blo, bhi = r2.boot_over_tasks(gs, lambda g: r2.icc_oneway(g)["icc"], n_boot=500)
        out["behaviour_icc"][key] = {"icc": aa["icc"], "ci": [blo, bhi]}

    # --- dual-outcome subcorpus, built exactly like A1
    dual = []
    n_dual = 0
    for iid, v in sorted(by_inst.items()):
        res = [r for r in v if r["resolved"]][:CAP]
        fail = [r for r in v if not r["resolved"]][:CAP]
        if res and fail:
            n_dual += 1
            dual.extend(res + fail)
    out["dual"] = {"n_instances": n_dual, "n_runs": len(dual), "cap": CAP,
                   "n_resolved": sum(1 for r in dual if r["resolved"])}

    # The published XEPV adapter does not transfer to this scaffold: OpenHands routes execution and
    # testing through one `bash` tool, so "execute" and "verify" are not separable from the action
    # name. The contrast here therefore uses alphabet-agnostic quantities (symbol shares and
    # length) rather than the published statistics.
    out["contrast_generic"] = generic_contrast(dual)

    for alpha in ("l1", "l3"):
        print(f"  scan {alpha} ...", flush=True)
        out.setdefault("scan", {})[alpha] = r3.scan(dual, alpha, n_perm=200, drop_submit=True,
                                                    drop_terminal=True)
        out.setdefault("scan_band", {})[alpha] = r3.scan(dual, alpha, n_perm=200,
                                                         drop_submit=True, support_band=r3.BAND,
                                                         drop_terminal=True)

    dst = os.path.join(llib.DATA, "r5_openhands.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
