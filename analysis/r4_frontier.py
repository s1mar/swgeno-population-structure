"""R4: does the same thing happen for frontier models, and does the correction replicate?

Corpus B is SWE-bench Verified under the same SWE-agent scaffold with three frontier
action-generator models, one run per (instance, model). So the within-instance contrast is between
MODELS rather than between repeated runs of one model. That is a different confound structure:
here the task is held fixed and the model varies, which is the comparison a reader who suspects
"this is just weak models" will want.

Caveat recorded in the output: this cache was collected under a quota (a fixed number of resolved
runs per submission), so its resolution rate is not any model's true rate. Only within-instance
contrasts are read from it.

Writes ../data/r4_frontier.json.
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
import r3_association as r3  # noqa: E402


def main() -> int:
    rows = llib.load_tokens(os.path.join(llib.DATA, "tokens_frontier.jsonl"))
    out = {"corpus": {
        "n": len(rows), "n_instances": len({r["instance_id"] for r in rows}),
        "models": {m: {"n": sum(1 for r in rows if r["model"] == m),
                       "resolved": sum(1 for r in rows if r["model"] == m and r["resolved"])}
                   for m in sorted({r["model"] for r in rows})},
        "note": "quota-sampled cache; resolution rate here is not a model's true rate",
    }}

    # instances covered by more than one model, with both outcomes present across models
    by_inst = defaultdict(list)
    for r in rows:
        by_inst[r["instance_id"]].append(r)
    disc = {i: rs for i, rs in by_inst.items()
            if len(rs) >= 2 and any(r["resolved"] for r in rs) and any(not r["resolved"] for r in rs)}
    out["corpus"]["n_instances_multi_model"] = sum(1 for rs in by_inst.values() if len(rs) >= 2)
    out["corpus"]["n_instances_discordant"] = len(disc)
    out["corpus"]["n_runs_in_discordant"] = sum(len(rs) for rs in disc.values())

    for alpha in ("xepvb", "l1", "l2"):
        out.setdefault("contrast", {})[alpha] = r1.contrast(rows, alpha, r1.group_stats,
                                                            n_boot=2000)

    # Motif scan on the discordant instances only: this is the corpus B replication of R3.
    #
    # Corpus B has exactly one run per (instance, model), so on a discordant instance the run that
    # resolved and the run that failed came from DIFFERENT models. Conditioning on the task
    # therefore leaves model identity confounded with the outcome, and the literature already
    # reports that a trajectory identifies its own model with high accuracy. To make that visible
    # rather than to hide it, the same scan is run stratified by model instead of by task: neither
    # stratification can remove the other confounder in a corpus without replicate runs.
    sub = [r for rs in disc.values() for r in rs]
    by_model = [dict(r, instance_id=r["model"]) for r in sub]
    for alpha in ("xepvb", "l1", "l2"):
        out.setdefault("scan_clean", {})[alpha] = r3.scan(
            sub, alpha, n_perm=200, drop_submit=True, support_band=r3.BAND, drop_terminal=True)
        out.setdefault("scan_model_strata", {})[alpha] = r3.scan(
            by_model, alpha, n_perm=200, drop_submit=True, support_band=r3.BAND,
            drop_terminal=True)
    out["model_mix_in_discordant"] = {
        "resolved": {m: sum(1 for r in sub if r["resolved"] and r["model"] == m)
                     for m in sorted({r["model"] for r in sub})},
        "failed": {m: sum(1 for r in sub if not r["resolved"] and r["model"] == m)
                   for m in sorted({r["model"] for r in sub})},
    }

    # Do the A1 survivors point the same way here?
    a1 = json.load(open(os.path.join(llib.DATA, "r3_association.json"), encoding="utf-8"))
    rep = {}
    for alpha in ("xepvb", "l1", "l2"):
        a1s = {s["motif"]: s["or_cond"] for s in a1["nosubmit_band"][alpha]["survivors"]}
        here = {t["motif"]: t["or_cond"] for t in out["scan_clean"][alpha]["top"]
                if t["or_cond"] is not None}
        both = [(m, a1s[m], here[m]) for m in a1s if m in here]
        agree = [1 for _m, x, z in both if (x - 1) * (z - 1) > 0]
        rep[alpha] = {"n_survivors_a1": len(a1s), "n_testable_here": len(both),
                      "n_same_direction": len(agree),
                      "pairs": [{"motif": m, "or_a1": x, "or_b": z} for m, x, z in both]}
    out["replication_of_survivors"] = rep

    dst = os.path.join(llib.DATA, "r4_frontier.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
