"""Shared machinery for the SWGeno study: alphabets, k-grams, and the two association tests.

Everything the paper reports is computed from `data/tokens.jsonl` (unselected corpus A0) or from
the dual-outcome pickle (A1) through the functions here, so there is exactly one implementation of
each estimator. See notes/FROZEN_SPEC.md; nothing in this file may drift from it silently.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Sibling data/ when run from the repository, ./data when run from inside a review packet.
# Packet-local data/ wins, for the reason recorded in paper/mknumbers.py.
DATA = (os.path.join(HERE, "data") if os.path.isdir(os.path.join(HERE, "data"))
        else os.path.join(HERE, "..", "data"))

# --------------------------------------------------------------------------- alphabets

# The replication target's published SWE-agent adapter, applied literally.
_XEPV_X = {"search_dir", "search_file", "find_file", "open", "goto", "scroll_up", "scroll_down",
           "ls"}
_XEPV_E = {"edit", "create", "insert", "str_replace"}
_XEPV_V = {"pytest", "submit"}
# The one pre-registered sensitivity: anything that runs code or tests counts as verification.
_XEPV_V_WIDE = _XEPV_V | {"python", "python3", "tox", "make", "bash", "sh", "nosetests",
                          "unittest", "coverage"}


def _xepv(cmds, wide: bool):
    v = _XEPV_V_WIDE if wide else _XEPV_V
    out = []
    for c in cmds:
        if c in _XEPV_X:
            out.append("X")
        elif c in v:
            out.append("V")
        else:
            # The target's stated default. P is empty for SWE-agent by construction: every turn
            # carries a command, so there is no pure-planning step to label.
            out.append("E")
    return out


ALPHABETS = ("xepv", "xepvb", "l1", "l2", "l3")


def seq(row: dict, alpha: str, drop_submit: bool = False,
        drop_terminal: bool = False) -> list[str]:
    """Symbol sequence for one run.

    `drop_submit` removes the `submit` action. Submission is how an episode ENDS, not a behaviour
    that could be observed while it is running, and a run that never submits cannot be scored
    resolved at all, so leaving it in makes the strongest "behavioural" association a mechanical
    one. See frozen-spec amendment 2.

    `drop_terminal` removes the FINAL action as well, which generalises the same rule. Submission
    is only one way an episode can end; ending on an edit because the context ran out is another,
    and it is just as entailed by the outcome. Amendment 3, after the terminal step was found to
    carry the strongest apparent survivor in the L3 alphabet.
    """
    cmds = row["cmd"]
    keep = None
    if drop_submit or drop_terminal:
        last = len(cmds) - 1
        keep = [i for i, c in enumerate(cmds)
                if not (drop_submit and c == "submit") and not (drop_terminal and i == last)]
    if alpha == "xepv":
        s = _xepv(cmds, wide=False)
    elif alpha == "xepvb":
        s = _xepv(cmds, wide=True)
    else:
        s = row[alpha]
    return [s[i] for i in keep] if keep is not None else s


# --------------------------------------------------------------------------- loading

def open_stream(path: str):
    """Open a token stream, transparently accepting a gzipped copy.

    The released artifact ships the large streams gzipped: the A0 stream is 112 MB of JSON lines
    and 4.7 MB compressed, which is the difference between a package a reader can clone and one
    they cannot. Working trees keep the plain file, so the plain path wins when both exist.
    """
    import gzip  # noqa: PLC0415  (only needed on the compressed path)
    if os.path.exists(path):
        return open(path, encoding="utf-8")
    if os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rt", encoding="utf-8")
    raise FileNotFoundError(f"{path} (and no .gz beside it)")


def load_tokens(path: str | None = None, limit: int | None = None) -> list[dict]:
    path = path or os.path.join(DATA, "tokens.jsonl")
    rows = []
    with open_stream(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def load_dual(path: str | None = None) -> list[dict]:
    """A1 as plain dicts with the same field names as A0, so every estimator takes one shape.

    The pickle holds `llab.schema.Trajectory` objects, so unpickling it needs the trajectory
    library on the path. A released artifact ships `tokens_dual.jsonl` instead, which needs
    nothing; that is preferred when present.
    """
    import json as _json
    import pickle
    import sys

    flat = os.path.join(DATA, "tokens_dual.jsonl")
    if path is None and (os.path.exists(flat) or os.path.exists(flat + ".gz")):
        with open_stream(flat) as f:
            return [_json.loads(line) for line in f]

    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
    from extract_tokens import row_of  # noqa: PLC0415  (same directory)
    path = path or os.path.join(DATA, "swgeno_dual_corpus.pkl")
    with open(path, "rb") as f:
        trajs = pickle.load(f)
    return [row_of(t, i) for i, t in enumerate(trajs)]


# --------------------------------------------------------------------------- k-grams

def kgrams(s: list[str], kmax: int = 3) -> set[str]:
    """Set of contiguous k-grams, k = 1..kmax, as 'a>b>c' strings."""
    out = set()
    n = len(s)
    for k in range(1, kmax + 1):
        for i in range(n - k + 1):
            out.add(">".join(s[i:i + k]))
    return out


def kgram_counts(s: list[str], kmax: int = 3) -> Counter:
    out = Counter()
    n = len(s)
    for k in range(1, kmax + 1):
        for i in range(n - k + 1):
            out[">".join(s[i:i + k])] += 1
    return out


def presence_matrix(rows: list[dict], alpha: str, kmax: int = 3, min_support: int = 20,
                    prefix: int | None = None, drop_submit: bool = False,
                    support_band: tuple[float, float] | None = None,
                    drop_terminal: bool = False):
    """Return (motifs, P, y, strata) with P[i, j] = motif j present in run i.

    `support_band` (lo, hi) as fractions of runs keeps only motifs whose prevalence sits inside the
    band. A motif present in 95% of runs is a test of whether the 5% that lack it are degenerate,
    which is not the question being asked.
    """
    sets = []
    for r in rows:
        s = seq(r, alpha, drop_submit=drop_submit, drop_terminal=drop_terminal)
        if prefix is not None:
            s = s[:prefix]
        sets.append(kgrams(s, kmax))
    n = len(rows)
    present = Counter()
    for st in sets:
        present.update(st)
    lo, hi = (0.0, 1.0) if support_band is None else support_band
    motifs = sorted(m for m, c in present.items()
                    if c >= min_support and n - c >= min_support and lo * n <= c <= hi * n)
    idx = {m: j for j, m in enumerate(motifs)}
    P = np.zeros((n, len(motifs)), dtype=bool)
    for i, st in enumerate(sets):
        for m in st:
            j = idx.get(m)
            if j is not None:
                P[i, j] = True
    y = np.array([bool(r["resolved"]) for r in rows])
    strata = np.array([r["instance_id"] for r in rows])
    return motifs, P, y, strata


# --------------------------------------------------------------------------- tests

def pooled_chi2(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson chi-square (1 df) for each motif column against outcome, vectorised.

    This is the test the pooled literature runs: every trajectory is one independent observation
    and the task it came from is ignored.
    """
    n = len(y)
    a = (P & y[:, None]).sum(0).astype(float)          # present & resolved
    b = (P & ~y[:, None]).sum(0).astype(float)         # present & failed
    c = (~P & y[:, None]).sum(0).astype(float)
    d = (~P & ~y[:, None]).sum(0).astype(float)
    num = n * (a * d - b * c) ** 2
    den = (a + b) * (c + d) * (a + c) * (b + d)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = np.where(den > 0, num / den, 0.0)
    return chi


def cmh(P: np.ndarray, y: np.ndarray, strata: np.ndarray):
    """Cochran-Mantel-Haenszel chi-square and common odds ratio, stratified by task instance.

    Strata with no variation in motif or in outcome contribute zero to both numerator and
    denominator, which is the correct behaviour: they carry no within-task information.
    """
    order = np.argsort(strata, kind="stable")
    P, y, strata = P[order], y[order], strata[order]
    bounds = np.flatnonzero(np.r_[True, strata[1:] != strata[:-1], True])

    m = P.shape[1]
    num = np.zeros(m)
    var = np.zeros(m)
    r_sum = np.zeros(m)
    s_sum = np.zeros(m)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        Pk, yk = P[lo:hi], y[lo:hi]
        nk = hi - lo
        if nk < 2:
            continue
        a = (Pk & yk[:, None]).sum(0).astype(float)
        b = (Pk & ~yk[:, None]).sum(0).astype(float)
        c = (~Pk & yk[:, None]).sum(0).astype(float)
        d = (~Pk & ~yk[:, None]).sum(0).astype(float)
        r1, r0 = a + b, c + d                    # motif present / absent
        c1, c0 = a + c, b + d                    # resolved / failed
        exp = r1 * c1 / nk
        num += a - exp
        denom = nk * nk * (nk - 1)
        var += np.where(denom > 0, r1 * r0 * c1 * c0 / denom, 0.0)
        r_sum += a * d / nk
        s_sum += b * c / nk
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = np.where(var > 0, num ** 2 / var, 0.0)
        orr = np.where(s_sum > 0, r_sum / s_sum, np.nan)
    return chi, orr


def lambda_gc(chi: np.ndarray) -> float:
    """Genomic inflation factor: median chi-square over the expected median of chi-square(1)."""
    chi = chi[np.isfinite(chi)]
    if chi.size == 0:
        return float("nan")
    return float(np.median(chi) / 0.4549364231195736)


def bh_fdr(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg. Returns a boolean mask of rejections at level q."""
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    keep = np.zeros(n, dtype=bool)
    if passed.any():
        kmax = np.flatnonzero(passed).max()
        keep[order[:kmax + 1]] = True
    return keep


def chi2_sf(chi: np.ndarray) -> np.ndarray:
    """Upper tail of chi-square with 1 df, via the error function (no scipy dependency)."""
    z = np.sqrt(np.maximum(chi, 0.0))
    return np.array([math.erfc(v / math.sqrt(2.0)) for v in z])


# --------------------------------------------------------------------------- bootstrap

def boot_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile interval for the mean of `values`, resampling the values themselves.

    Callers pass one value per INSTANCE, so this is a cluster bootstrap over tasks.
    """
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def group_by_instance(rows: list[dict]) -> dict[str, list[dict]]:
    g = defaultdict(list)
    for r in rows:
        g[r["instance_id"]].append(r)
    return dict(g)
