"""R9: do the seven headline statistics survive removing the outcome-entailed actions?

A reviewer found the gap this closes. The motif scan excludes `submit` and the terminal action
because a run that never finishes cannot be scored resolved, so a feature that encodes finishing is
entailed by the outcome rather than predictive of it. The seven summary statistics in the
replication table were never put through that filter, and under the published adapter `submit` maps
to the verification class. Two of the five statistics reported as surviving conditioning are
verification statistics, so part of what survives could be the act of finishing.

The replication arm itself must keep the published adapter untouched, or it is not a replication.
This is a sensitivity on the CONTRAST arm only: the same estimator, the same corpus, the same
bootstrap, with submission and the terminal action removed from every sequence.

Writes ../data/r9_entailment.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llib  # noqa: E402
import r1_replicate as r1  # noqa: E402


def contrast_stripped(rows, alpha, n_boot=2000, seed=0):
    """r1.contrast, but on sequences with submit and the terminal action removed."""
    stripped = []
    for r in rows:
        s = llib.seq(r, alpha, drop_submit=True, drop_terminal=True)
        # r1.contrast re-derives the sequence from `cmd`, so hand it a row whose cmd list is
        # already filtered and whose alphabet field matches.
        keep = [i for i, c in enumerate(r["cmd"])
                if c != "submit" and i != len(r["cmd"]) - 1]
        q = dict(r)
        q["cmd"] = [r["cmd"][i] for i in keep]
        for a in ("l1", "l2", "l3"):
            if a in q:
                q[a] = [r[a][i] for i in keep]
        assert len(llib.seq(q, alpha)) == len(s)
        stripped.append(q)
    return r1.contrast(stripped, alpha, r1.group_stats, n_boot=n_boot, seed=seed)


def main() -> int:
    a1 = llib.load_dual()
    a0 = llib.load_tokens()
    out = {"note": "Sensitivity on the contrast arm only. The replication arm keeps the published "
                   "adapter unchanged. Here submission and the terminal action are removed from "
                   "every sequence, matching the motif scan's clean configuration."}

    for alpha in ("xepvb",):
        print(f"  stripped contrast {alpha} ...", flush=True)
        out[f"contrast_{alpha}"] = contrast_stripped(a1, alpha)

        # whole-corpus delta with the same stripping, for the corpus column
        keep_rows = []
        for r in a0:
            keep = [i for i, c in enumerate(r["cmd"])
                    if c != "submit" and i != len(r["cmd"]) - 1]
            q = dict(r)
            q["cmd"] = [r["cmd"][i] for i in keep]
            for a in ("l1", "l2", "l3"):
                if a in q:
                    q[a] = [r[a][i] for i in keep]
            keep_rows.append(q)
        out[f"corpus_delta_{alpha}"] = r1.plain_delta(keep_rows, alpha)

    # Which of the five previously-surviving statistics still clear zero?
    base = json.load(open(os.path.join(llib.DATA, "r1_replicate.json"), encoding="utf-8"))
    survived = ["pr_v_given_e", "pr_e_given_e", "max_x_run", "v_ratio", "steps"]
    verdict = {}
    for m in survived:
        b = base["contrast"]["xepvb"]["within"][m]
        s = out["contrast_xepvb"]["within"][m]
        clears = (s["lo"] is not None and s["hi"] is not None
                  and (s["lo"] > 0 or s["hi"] < 0))
        verdict[m] = {"before": [b["delta"], b["lo"], b["hi"]],
                      "after": [s["delta"], s["lo"], s["hi"]],
                      "still_excludes_zero": bool(clears),
                      "retained_fraction": (s["delta"] / b["delta"]) if b["delta"] else None}
    out["verdict"] = verdict

    dst = os.path.join(llib.DATA, "r9_entailment.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    for m, v in verdict.items():
        b, s = v["before"], v["after"]
        print(f"  {m:14} before {b[0]:8.3f} [{b[1]:7.3f},{b[2]:7.3f}]   "
              f"after {s[0]:8.3f} [{s[1]:7.3f},{s[2]:7.3f}]   "
              f"{'SURVIVES' if v['still_excludes_zero'] else 'GONE'}")
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
