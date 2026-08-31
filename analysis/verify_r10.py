"""Independent re-derivation of the heterogeneity numbers, sharing no code with analysis/r10.

Same principle as verify.py: a second implementation is the only check that catches a bug in the
first one. This one deliberately avoids llib and r3 entirely. It rebuilds the symbol sequences, the
motif presence vector, the per-task 2x2 tables, the Cochran-Mantel-Haenszel common odds ratio,
Cochran's Q and the directional-consistency fraction from the raw token stream with plain Python
loops and dictionaries, then compares against data/r10_heterogeneity.json.

It also checks the one thing most likely to be silently wrong: the parameterisation of
scipy's Fisher noncentral hypergeometric sampler used to build the homogeneity reference. If the
argument order is wrong the sampler still returns plausible integers and every downstream number
looks reasonable. Here the sampler is asked for a known odds ratio on known margins, and the
empirical odds ratio it produces has to come back.

Run: python verify_r10.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import nchypergeom_fisher

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

FAILS: list[str] = []
CHECKS = 0


def check(name: str, got, want, tol=1e-9):
    global CHECKS
    CHECKS += 1
    ok = (abs(got - want) <= tol) if isinstance(want, float) else (got == want)
    if not ok:
        FAILS.append(f"{name}: re-derived {got!r}, file says {want!r}")


def load(path):
    """Read a token stream, accepting a gzipped copy, as the released artifact ships."""
    import gzip  # noqa: PLC0415
    full = os.path.join(DATA, path)
    opener = (lambda: open(full, encoding="utf-8")) if os.path.exists(full) \
        else (lambda: gzip.open(full + ".gz", "rt", encoding="utf-8"))
    with opener() as f:
        return [json.loads(line) for line in f]


def kgrams_present(seq, kmax=3):
    """Every contiguous k-gram, k<=kmax, as a set. Written from the definition, not imported."""
    out = set()
    for k in range(1, kmax + 1):
        for i in range(len(seq) - k + 1):
            out.add(">".join(seq[i:i + k]))
    return out


def drop_submit_and_terminal(cmds, symbols):
    """The 'nosubmit' configuration, restated from the frozen spec rather than imported.

    Two exclusions, and they are applied to the ORIGINAL step indices in a single pass, not one
    after the other. The submit action is outcome-entailed, and a run's final action has no
    following observation so its observation class is fictitious. When a run's last command IS
    `submit`, both rules name the same step and exactly one step is removed. Applying them in
    sequence instead would remove a second, real action from every such run, which is the error
    this checker made on its first attempt.

    The test is on the raw command verb, not on the alphabet symbol, because that is what the
    specification names.
    """
    last = len(cmds) - 1
    return [symbols[i] for i, c in enumerate(cmds) if c != "submit" and i != last]


def main() -> int:
    ref = json.load(open(os.path.join(DATA, "r10_heterogeneity.json"), encoding="utf-8"))
    rows = load("tokens_dual.jsonl")

    # --- 1. the sampler's parameterisation ------------------------------------------------
    # A known 2x2 world: 40 runs, 15 carrying the motif, 18 resolved, true odds ratio 3.0.
    rng = np.random.default_rng(0)
    M, n, N, psi = 40, 15, 18, 3.0
    draws = nchypergeom_fisher.rvs(M, n, N, psi, size=200000, random_state=rng).astype(float)
    a = draws
    b = n - a
    c = N - a
    d = M - n - c
    with np.errstate(divide="ignore", invalid="ignore"):
        emp = np.median((a + .5) * (d + .5) / ((b + .5) * (c + .5)))
    if not (2.5 < emp < 3.6):
        FAILS.append(f"nchypergeom_fisher parameterisation: asked for odds 3.0, "
                     f"empirical median odds ratio came back {emp:.3f}. Argument order is wrong.")
    else:
        globals()["CHECKS"] = CHECKS + 1

    # --- 2. per-task tables and CMH for each reported survivor -----------------------------
    alpha = "l3"
    seqs = {}
    for r in rows:
        seqs[id(r)] = drop_submit_and_terminal(r["cmd"], r[alpha])

    by_task = defaultdict(list)
    for r in rows:
        by_task[r["instance_id"]].append(r)

    n_tasks = len(by_task)
    check("n_instances", n_tasks, ref["config"]["n_instances"])
    check("n_runs", len(rows), ref["config"]["n_runs"])

    for entry in ref[alpha]["survivors"][:4]:          # the four strongest, spot-checked
        motif = entry["motif"]
        # present/absent per run, from the sequence, independently of any matrix builder
        present = {id(r): (motif in kgrams_present(seqs[id(r)])) for r in rows}
        check(f"{motif}: n_present", sum(present.values()), entry["n_present"])

        num = var = r_sum = s_sum = 0.0
        lors, weights, informative = [], [], 0
        for _tid, runs in by_task.items():
            if len(runs) < 2:
                continue
            A = sum(1 for r in runs if present[id(r)] and r["resolved"])
            B = sum(1 for r in runs if present[id(r)] and not r["resolved"])
            C = sum(1 for r in runs if not present[id(r)] and r["resolved"])
            D = sum(1 for r in runs if not present[id(r)] and not r["resolved"])
            nk = A + B + C + D
            r1, r0, c1, c0 = A + B, C + D, A + C, B + D
            num += A - r1 * c1 / nk
            if nk > 1:
                var += r1 * r0 * c1 * c0 / (nk * nk * (nk - 1))
            r_sum += A * D / nk
            s_sum += B * C / nk
            if r1 > 0 and r0 > 0 and c1 > 0 and c0 > 0:
                informative += 1
                aa, bb, cc, dd = A + .5, B + .5, C + .5, D + .5
                lors.append(math.log((aa * dd) / (bb * cc)))
                weights.append(1.0 / (1 / aa + 1 / bb + 1 / cc + 1 / dd))

        or_cmh = r_sum / s_sum
        chi = num * num / var
        check(f"{motif}: or_cond", round(or_cmh, 6), round(entry["or_cond"], 6), 1e-6)
        check(f"{motif}: chi_cond", round(chi, 4), round(entry["chi_cond"], 4), 1e-4)
        check(f"{motif}: informative strata", informative, entry["n_informative_strata"])

        lg = math.log(or_cmh)
        q = sum(w * (l - lg) ** 2 for w, l in zip(weights, lors))
        check(f"{motif}: cochran_q", round(q, 3), entry["cochran_q"], 5e-4)
        # Directional agreement, with exact ties EXCLUDED rather than scored as disagreement.
        # A stratum whose corrected log odds ratio is exactly zero points in no direction, and
        # about 2% of informative strata are such ties; counting them as disagreement biases the
        # fraction downward. Implemented here from the rule, not copied from the analysis.
        untied = [l for l in lors if l != 0.0]
        same = sum(1 for l in untied if (l > 0) == (lg > 0)) / len(untied)
        check(f"{motif}: frac same sign", round(same, 4), entry["frac_strata_same_sign"], 5e-5)

    print(f"re-derived {CHECKS} quantities with an independent implementation")
    if FAILS:
        print("\nMISMATCHES")
        for f in FAILS:
            print("  " + f)
        print("\nGATE: FAIL")
        return 1
    print("\nGATE: pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
