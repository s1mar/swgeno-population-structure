"""R11: does the within-task contrast depend on how A1 was built?

Reviewers asked whether the reported disappearance, attenuation and sign reversal survive other
task-selection and capping rules. Answering that first required establishing what A1's rule
actually is, because the manuscript's description of it is wrong.

WHAT A1 ACTUALLY IS. Reconstructed here and verified bit-exactly against `tokens_dual.jsonl`:
take the first 40,000 rows of the corpus in stream order, keep runs with at least 15 agent steps,
keep the tasks that carry both outcomes, and cap each task at 12 runs per outcome. That yields
exactly 348 tasks, 5,004 runs and 1,777 resolved, matching the shipped A1 run for run. Two
consequences:

  * inside that window there are exactly 348 both-outcome candidates, so the ranking by comparison
    support that the manuscript describes selected nothing. It was not a binding rule.
  * the 15-step floor IS binding: 42.4% of A0's runs fall below it. A0 itself has no such floor,
    so A1 and A0 are drawn from populations that differ by more than the both-outcome requirement.

This script therefore varies one factor at a time from the real rule: the scan window, the step
floor, the per-outcome cap, and a minimum-replication requirement. It re-uses
`r1_replicate.contrast` unchanged, so the estimator, the metric definitions and the cluster
bootstrap are identical to the published analysis; only the row set changes.

`A1_reconstructed` must reproduce `data/r1_replicate.json` exactly. If it does not, every other row
here is meaningless and the script says so. Nothing here modifies a published result.

Writes ../data/r11_a1_sensitivity.json.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llib  # noqa: E402
import r1_replicate as r1  # noqa: E402

ALPHA = "xepvb"
N_BOOT = 2000

# A1's real construction, recovered from the shipped corpus and verified run for run.
A1_WINDOW = 40000
A1_MIN_STEPS = 15
A1_CAP = 12


def load_a0():
    with llib.open_stream(os.path.join(llib.DATA, "tokens.jsonl")) as f:
        return [json.loads(line) for line in f]


def build(a0, window, min_steps, cap, min_arm=1):
    """A1's rule with any of its three parameters changed.

    Runs are taken in corpus stream order, which is what A1 did, so the choice is deterministic
    and needs no seed.
    """
    sub = [r for r in a0
           if (window is None or r["order"] < window) and r["n_steps"] >= min_steps]
    by = defaultdict(list)
    for r in sub:
        by[r["instance_id"]].append(r)
    rows = []
    n_tasks = 0
    for _, v in sorted(by.items()):
        res = sorted((r for r in v if r["resolved"]), key=lambda r: r["order"])
        fail = sorted((r for r in v if not r["resolved"]), key=lambda r: r["order"])
        if len(res) < min_arm or len(fail) < min_arm:
            continue
        n_tasks += 1
        rows.extend(res if cap is None else res[:cap])
        rows.extend(fail if cap is None else fail[:cap])
    return rows, n_tasks


def summarise(tag, rows, note):
    print(f"  {tag}: {len(rows)} runs ...", flush=True)
    c = r1.contrast(rows, ALPHA, r1.group_stats, n_boot=N_BOOT, seed=0)
    c["tag"] = tag
    c["note"] = note
    c["n_runs"] = len(rows)
    c["n_resolved"] = sum(1 for r in rows if r["resolved"])
    # the share of THIS build that falls under A1 own step floor; the corpus-wide
    # share is a different number and does not describe any particular build
    c["frac_below_floor"] = round(
        sum(1 for r in rows if r["n_steps"] < A1_MIN_STEPS) / max(len(rows), 1), 4)
    return c


def main() -> int:
    a0 = load_a0()
    out = {"config": {
        "alphabet": ALPHA, "n_boot": N_BOOT, "metrics": list(r1.METRICS),
        "a1_rule": {"window": A1_WINDOW, "min_steps": A1_MIN_STEPS, "cap": A1_CAP},
        "a0_runs": len(a0), "a0_tasks": len({r["instance_id"] for r in a0}),
        "a0_frac_runs_below_15_steps": round(
            sum(1 for r in a0 if r["n_steps"] < A1_MIN_STEPS) / len(a0), 4),
    }}

    # --- the identity check the rest of the file depends on -------------------------------
    recon, n_recon = build(a0, A1_WINDOW, A1_MIN_STEPS, A1_CAP)
    shipped = llib.load_dual()
    ok = (n_recon == len({r["instance_id"] for r in shipped})
          and len(recon) == len(shipped)
          and sum(1 for r in recon if r["resolved"]) == sum(1 for r in shipped if r["resolved"])
          and {r["instance_id"] for r in recon} == {r["instance_id"] for r in shipped})
    out["config"]["a1_reconstruction_matches_shipped"] = bool(ok)
    print(f"  A1 reconstruction matches shipped corpus: {ok} "
          f"({n_recon} tasks, {len(recon)} runs)")
    if not ok:
        print("  REFUSING to report sensitivity: the reconstruction of A1 does not match.")
        with open(os.path.join(llib.DATA, "r11_a1_sensitivity.json"), "w",
                  encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        return 1

    variants = [
        ("A1_reconstructed", A1_WINDOW, A1_MIN_STEPS, A1_CAP, 1,
         "A1's real rule, rebuilt from A0. Must reproduce the published contrast exactly."),
        ("nocap", A1_WINDOW, A1_MIN_STEPS, None, 1,
         "A1's window and step floor, no per-outcome cap"),
        ("cap5", A1_WINDOW, A1_MIN_STEPS, 5, 1, "cap 5 per outcome"),
        ("cap20", A1_WINDOW, A1_MIN_STEPS, 20, 1, "cap 20 per outcome"),
        ("nofloor_cap12", A1_WINDOW, 1, A1_CAP, 1,
         "the step floor dropped and nothing else: A1's window and cap kept. This is the variant "
         "that isolates the 15-step rule, which is the one binding rule the submitted paper did "
         "not disclose."),
        ("nowindow_cap12", None, A1_MIN_STEPS, A1_CAP, 1,
         "the whole corpus rather than the first 40,000 rows, step floor and cap kept"),
        ("nowindow_nocap", None, A1_MIN_STEPS, None, 1,
         "the whole corpus, step floor kept, no cap: every both-outcome task A0 supports"),
        ("nosteps_nowindow_nocap", None, 1, None, 1,
         "every constraint dropped: whole corpus, no step floor, no cap. This is a different "
         "population from the rest of the paper, because 42.4% of A0's runs are shorter than "
         "the 15 steps A1 requires."),
        ("nowindow_nocap_minarm3", None, A1_MIN_STEPS, None, 3,
         "whole corpus, step floor kept, no cap, and at least three runs on each side"),
    ]
    for tag, win, ms, cap, arm, note in variants:
        rows, n_tasks = build(a0, win, ms, cap, arm)
        rec = summarise(tag, rows, note)
        rec["n_tasks"] = n_tasks
        rec["rule"] = {"window": win, "min_steps": ms, "cap": cap, "min_arm": arm}
        out[tag] = rec

    ref = json.load(open(os.path.join(llib.DATA, "r1_replicate.json"),
                         encoding="utf-8"))["contrast"][ALPHA]
    diffs = {m: abs(out["A1_reconstructed"]["within"][m]["delta"] - ref["within"][m]["delta"])
             for m in r1.METRICS}
    worst = max(diffs.values())
    out["config"]["max_abs_deviation_from_published_within"] = worst
    print(f"  max deviation from published within-task deltas: {worst:.2e}")
    # This is the check that actually establishes the reconstruction, and it has to be ASSERTED,
    # not merely printed. The aggregate check above (task, run and resolved counts, and instance-id
    # set equality) can be satisfied by a wrong selection rule that happens to pick the same
    # instances; reproducing all eight within-task deltas to the last bit cannot.
    if worst != 0.0:
        print("  REFUSING to report: the rebuilt A1 does not reproduce the published contrast.")
        with open(os.path.join(llib.DATA, "r11_a1_sensitivity.json"), "w",
                  encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        return 1

    dst = os.path.join(llib.DATA, "r11_a1_sensitivity.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
