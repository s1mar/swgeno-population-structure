"""R1: reproduce the published pooled behavioural signature, then condition it on the task.

Three things happen here.

1. REPLICATION. The summary statistics reported by prior work on 2,000 runs of this corpus are
   recomputed on our own draw (the first 2,000 runs in stream order, fixed in the frozen spec)
   using that work's published SWE-agent adapter.

2. CONTRAST. The same statistics are recomputed as a resolved-minus-failed difference at three
   levels: the whole corpus, the dual-outcome corpus pooled, and the dual-outcome corpus within
   instance. The first-to-second step is ascertainment; the second-to-third step is conditioning.

3. WEIGHTING CHECK. The within-instance estimator weights every task equally and the pooled one
   weights every run equally, so a difference between them could in principle be weighting rather
   than conditioning. A run-weighted within-instance estimator is reported to settle that.

Estimator note (frozen spec amendment 1): transition probabilities and symbol shares are POOLED
over a group of runs, not averaged per run. The pooled form is what reproduces the published
table; the per-run form is kept as a declared sensitivity.

Writes ../data/r1_replicate.json.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llib  # noqa: E402

METRICS = ["pr_v_given_e", "pr_e_given_e", "pr_x_given_x", "max_x_run",
           "v_ratio", "x_ratio", "e_ratio", "steps"]

# Published values on 2,000 runs of this corpus, resolved / unresolved. Kept in the code so the
# comparison cannot drift away from the prose.
PUBLISHED = {
    "pr_v_given_e": (0.542, 0.281), "pr_e_given_e": (0.415, 0.636),
    "pr_x_given_x": (0.746, 0.848), "max_x_run": (4.8, 11.0),
    "v_ratio": (0.247, 0.157), "x_ratio": (0.336, 0.449),
    "steps": (16.0, 30.1),
}
PUBLISHED_N = {"resolved": 338, "unresolved": 1662, "total": 2000,
               "models": {"70B": 1793, "8B": 167, "405B": 40}}


def _max_run(s, sym):
    best = cur = 0
    for x in s:
        cur = cur + 1 if x == sym else 0
        best = max(best, cur)
    return best


def summarize(s: list[str]) -> dict:
    """Sufficient statistics for one run. Group statistics are sums of these, so the bootstrap
    never re-walks a sequence."""
    tr = Counter()
    frm = Counter()
    sym = Counter(s)
    for i in range(len(s) - 1):
        frm[s[i]] += 1
        tr[(s[i], s[i + 1])] += 1
    return {"tr": tr, "frm": frm, "sym": sym, "n": len(s),
            "maxx": _max_run(s, "X")}


def group_stats(sums: list[dict]) -> dict:
    if not sums:
        return {m: None for m in METRICS}
    tr, frm, sym = Counter(), Counter(), Counter()
    n_tot = 0
    maxx = []
    lens = []
    for u in sums:
        tr.update(u["tr"])
        frm.update(u["frm"])
        sym.update(u["sym"])
        n_tot += u["n"]
        maxx.append(u["maxx"])
        lens.append(u["n"])

    def t(a, b):
        return tr[(a, b)] / frm[a] if frm[a] else None

    return {"pr_v_given_e": t("E", "V"), "pr_e_given_e": t("E", "E"),
            "pr_x_given_x": t("X", "X"),
            "max_x_run": float(np.mean(maxx)),
            "v_ratio": sym["V"] / n_tot if n_tot else None,
            "x_ratio": sym["X"] / n_tot if n_tot else None,
            "e_ratio": sym["E"] / n_tot if n_tot else None,
            "steps": float(np.mean(lens))}


def per_run_stats(sums: list[dict]) -> dict:
    """Declared sensitivity: quantities averaged over runs instead of pooled over the group."""
    def avg(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    def t(u, a, b):
        return u["tr"][(a, b)] / u["frm"][a] if u["frm"][a] else None

    return {"pr_v_given_e": avg([t(u, "E", "V") for u in sums]),
            "pr_e_given_e": avg([t(u, "E", "E") for u in sums]),
            "pr_x_given_x": avg([t(u, "X", "X") for u in sums]),
            "max_x_run": avg([float(u["maxx"]) for u in sums]),
            "v_ratio": avg([u["sym"]["V"] / u["n"] if u["n"] else None for u in sums]),
            "x_ratio": avg([u["sym"]["X"] / u["n"] if u["n"] else None for u in sums]),
            "e_ratio": avg([u["sym"]["E"] / u["n"] if u["n"] else None for u in sums]),
            "steps": avg([float(u["n"]) for u in sums])}


def outcome_split(rows, sums, fn):
    out = {}
    for name, want in (("resolved", True), ("failed", False)):
        sub = [u for r, u in zip(rows, sums) if bool(r["resolved"]) is want]
        out[name] = fn(sub)
        out[name]["n"] = len(sub)
    return out


def _delta(res, fail, fn):
    a, b = fn(res), fn(fail)
    return {m: (a[m] - b[m]) if a[m] is not None and b[m] is not None else np.nan
            for m in METRICS}


def _vectorize(sums: list[dict], keys) -> np.ndarray:
    """Additive sufficient statistics for a set of runs, as one fixed-length vector.

    Group statistics are ratios of sums, so once each (instance, outcome) cell is one vector the
    bootstrap is a matrix sum instead of 700,000 re-walks of the corpus.
    """
    tkeys, skeys = keys
    v = np.zeros(len(tkeys) + len(skeys) + len(skeys) + 3)
    nt, ns = len(tkeys), len(skeys)
    for u in sums:
        for i, k in enumerate(tkeys):
            v[i] += u["tr"][k]
        for i, k in enumerate(skeys):
            v[nt + i] += u["frm"][k]
            v[nt + ns + i] += u["sym"][k]
        v[-3] += u["n"]
        v[-2] += u["maxx"]
        v[-1] += 1
    return v


def _stats_from_vec(v: np.ndarray, keys) -> dict:
    tkeys, skeys = keys
    nt, ns = len(tkeys), len(skeys)
    ti = {k: i for i, k in enumerate(tkeys)}
    si = {k: i for i, k in enumerate(skeys)}
    n_runs, n_steps = v[-1], v[-3]
    if n_runs == 0 or n_steps == 0:
        return {m: None for m in METRICS}

    def t(a, b):
        if (a, b) not in ti or a not in si:
            return None
        den = v[nt + si[a]]
        return float(v[ti[(a, b)]] / den) if den else None

    def share(a):
        return float(v[nt + ns + si[a]] / n_steps) if a in si else None

    return {"pr_v_given_e": t("E", "V"), "pr_e_given_e": t("E", "E"),
            "pr_x_given_x": t("X", "X"), "max_x_run": float(v[-2] / n_runs),
            "v_ratio": share("V"), "x_ratio": share("X"), "e_ratio": share("E"),
            "steps": float(n_steps / n_runs)}


def contrast(rows, alpha, fn=group_stats, n_boot=2000, seed=0):
    sums = [summarize(llib.seq(r, alpha)) for r in rows]
    by_inst: dict[str, tuple[list, list]] = {}
    for r, u in zip(rows, sums):
        res, fail = by_inst.setdefault(r["instance_id"], ([], []))
        (res if r["resolved"] else fail).append(u)
    iids = sorted(by_inst)

    vectorized = fn is group_stats
    if vectorized:
        skeys = sorted({s for u in sums for s in u["sym"]})
        tkeys = sorted({k for u in sums for k in u["tr"]})
        keys = (tkeys, skeys)
        VR = np.stack([_vectorize(by_inst[i][0], keys) for i in iids])
        VF = np.stack([_vectorize(by_inst[i][1], keys) for i in iids])

        def pooled_idx(idx):
            a = _stats_from_vec(VR[idx].sum(0), keys)
            b = _stats_from_vec(VF[idx].sum(0), keys)
            return {m: (a[m] - b[m]) if a[m] is not None and b[m] is not None else np.nan
                    for m in METRICS}

        per_inst = np.full((len(iids), len(METRICS)), np.nan)
        for j, i in enumerate(iids):
            a = _stats_from_vec(VR[j], keys)
            b = _stats_from_vec(VF[j], keys)
            for c, m in enumerate(METRICS):
                if a[m] is not None and b[m] is not None:
                    per_inst[j, c] = a[m] - b[m]
        wts = np.array([len(by_inst[i][0]) + len(by_inst[i][1]) for i in iids], dtype=float)

        def within_idx(idx, weighted=False):
            sub = per_inst[idx]
            w = wts[idx] if weighted else np.ones(len(idx))
            out = {}
            for c, m in enumerate(METRICS):
                col = sub[:, c]
                ok = np.isfinite(col)
                out[m] = float(np.average(col[ok], weights=w[ok])) if ok.any() else np.nan
            return out
    else:
        def pooled_idx(idx):
            res = [u for j in idx for u in by_inst[iids[j]][0]]
            fail = [u for j in idx for u in by_inst[iids[j]][1]]
            return _delta(res, fail, fn)

        per = []
        for i in iids:
            res, fail = by_inst[i]
            d = _delta(res, fail, fn) if res and fail else {m: np.nan for m in METRICS}
            per.append([d[m] for m in METRICS])
        per_inst = np.array(per, dtype=float)
        wts = np.array([len(by_inst[i][0]) + len(by_inst[i][1]) for i in iids], dtype=float)

        def within_idx(idx, weighted=False):
            sub = per_inst[idx]
            w = wts[idx] if weighted else np.ones(len(idx))
            out = {}
            for c, m in enumerate(METRICS):
                col = sub[:, c]
                ok = np.isfinite(col)
                out[m] = float(np.average(col[ok], weights=w[ok])) if ok.any() else np.nan
            return out

    allidx = np.arange(len(iids))
    point = {"pooled": pooled_idx(allidx), "within": within_idx(allidx),
             "within_runweighted": within_idx(allidx, weighted=True)}
    rng = np.random.default_rng(seed)
    boots = {k: {m: [] for m in METRICS} for k in point}
    for _ in range(n_boot):
        idx = rng.integers(0, len(iids), len(iids))
        for k, d in (("pooled", pooled_idx(idx)), ("within", within_idx(idx)),
                     ("within_runweighted", within_idx(idx, weighted=True))):
            for m in METRICS:
                boots[k][m].append(d[m])

    out = {"n_instances": len(iids),
           "n_instances_both_outcomes": sum(1 for i in iids
                                            if by_inst[i][0] and by_inst[i][1])}
    for k in point:
        out[k] = {}
        for m in METRICS:
            b = np.array(boots[k][m], dtype=float)
            b = b[np.isfinite(b)]
            out[k][m] = {"delta": float(point[k][m]) if np.isfinite(point[k][m]) else None,
                         "lo": float(np.quantile(b, .025)) if b.size else None,
                         "hi": float(np.quantile(b, .975)) if b.size else None}
    return out


def plain_delta(rows, alpha, fn=group_stats):
    sums = [summarize(llib.seq(r, alpha)) for r in rows]
    res = [u for r, u in zip(rows, sums) if r["resolved"]]
    fail = [u for r, u in zip(rows, sums) if not r["resolved"]]
    d = _delta(res, fail, fn)
    return {m: (float(d[m]) if np.isfinite(d[m]) else None) for m in METRICS}


def main() -> int:
    out = {"published": {"table15": PUBLISHED, "n": PUBLISHED_N}}

    a0 = llib.load_tokens()
    draw = a0[:2000]
    out["replication_draw"] = {
        "n": len(draw), "n_resolved": sum(1 for r in draw if r["resolved"]),
        "resolution_rate": sum(1 for r in draw if r["resolved"]) / len(draw),
        "n_instances": len({r["instance_id"] for r in draw}),
        "models": {k: sum(1 for r in draw if r["model"] == k)
                   for k in sorted({r["model"] for r in draw})},
        "unparsed_steps_frac": float(sum(r["n_unparsed"] for r in draw)
                                     / sum(r["n_steps"] for r in draw)),
    }
    out["corpus"] = {
        "n": len(a0), "n_instances": len({r["instance_id"] for r in a0}),
        "n_resolved": sum(1 for r in a0 if r["resolved"]),
        "resolution_rate": float(np.mean([r["resolved"] for r in a0])),
        "models": {k: sum(1 for r in a0 if r["model"] == k)
                   for k in sorted({r["model"] for r in a0})},
    }
    for alpha in ("xepv", "xepvb"):
        sums = [summarize(llib.seq(r, alpha)) for r in draw]
        out.setdefault("replication_pooled", {})[alpha] = outcome_split(draw, sums, group_stats)
        out.setdefault("replication_perrun", {})[alpha] = outcome_split(draw, sums, per_run_stats)
        out.setdefault("corpus_delta", {})[alpha] = plain_delta(a0, alpha)
        out.setdefault("draw_delta", {})[alpha] = plain_delta(draw, alpha)

    a1 = llib.load_dual()
    out["a1"] = {
        "n": len(a1), "n_instances": len({r["instance_id"] for r in a1}),
        "n_resolved": sum(1 for r in a1 if r["resolved"]),
        "models": {k: sum(1 for r in a1 if r["model"] == k)
                   for k in sorted({r["model"] for r in a1})},
    }
    for alpha in ("xepv", "xepvb", "l1"):
        print(f"  contrast {alpha} ...", flush=True)
        out.setdefault("contrast", {})[alpha] = contrast(a1, alpha, group_stats)
    out["contrast_perrun_xepvb"] = contrast(a1, "xepvb", per_run_stats)

    dst = os.path.join(llib.DATA, "r1_replicate.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
