"""R8: measure the terminal-step artefact that Section 11 of the paper reports.

The paper claims that labelling the final action's missing observation as an empty one created a
fictitious no-op at the end of every run, and that the resulting motif passed both significance and
effect-size filters while being a marker of how the episode stopped rather than of behaviour. That
is a claim about the corpus, so it needs a result file like every other claim.

Everything here is measured on the token streams as they are NOW, i.e. after the fix, by
reconstructing what the old labelling would have produced: the final token's observation class is
`end`, and the pre-fix labelling would have called it `noop`.

Writes ../data/r8_terminal.json.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llib  # noqa: E402


def measure(rows: list[dict], name: str) -> dict:
    n = len(rows)
    ends_with_end = sum(1 for r in rows if r["l3"] and r["l3"][-1].endswith(":end"))
    # Under the pre-fix labelling every one of these was a `:noop`, so the count of `:noop`
    # tokens that were really terminal is exactly the number of runs.
    genuine_noop = sum(sum(1 for t in r["l3"][:-1] if t.endswith(":noop")) for r in rows)
    terminal_class = Counter(r["l3"][-1].split(":")[0] for r in rows if r["l3"])

    # The specific motif the paper names: an EDIT as the final action.
    edit_terminal = [r for r in rows if r["l3"] and r["l3"][-1] == "EDIT:end"]
    exits = Counter(r["exit_status"] for r in edit_terminal)
    ctx = sum(v for k, v in exits.items() if "exit_context" in k)
    # Under the old labelling, could a run carry a mid-run EDIT:noop as well?
    mid_edit_noop = sum(1 for r in rows
                        if any(t == "EDIT:noop" for t in r["l3"][:-1]))
    return {
        "n_runs": n,
        "runs_whose_last_token_is_terminal": ends_with_end,
        "frac_runs_terminal": ends_with_end / n if n else None,
        "genuine_mid_run_noop_tokens": genuine_noop,
        "terminal_tokens": ends_with_end,
        "frac_of_all_noop_that_was_terminal":
            ends_with_end / (ends_with_end + genuine_noop) if (ends_with_end + genuine_noop) else None,
        "terminal_action_class": dict(terminal_class.most_common()),
        "edit_terminal_runs": len(edit_terminal),
        "edit_terminal_exit_status": dict(exits.most_common()),
        "edit_terminal_context_exhausted": ctx,
        "edit_terminal_frac_context_exhausted": ctx / len(edit_terminal) if edit_terminal else None,
        "runs_with_a_mid_run_edit_noop": mid_edit_noop,
        "edit_terminal_resolved": sum(1 for r in edit_terminal if r["resolved"]),
    }


def prefix_odds_ratio(rows: list[dict]) -> dict:
    """What the artefact motif scored under the pre-fix labelling.

    The paper quotes the odds ratio the discarded motif used to have. Quoting it from memory would
    be exactly the habit this pipeline exists to prevent, so it is recomputed here with the same
    estimator the scan uses, on the same strata: presence of a terminal EDIT, which is what
    `EDIT:noop` was.
    """
    import numpy as np
    present = np.array([bool(r["l3"]) and r["l3"][-1] == "EDIT:end" for r in rows])
    y = np.array([bool(r["resolved"]) for r in rows])
    strata = np.array([r["instance_id"] for r in rows])
    chi, orr = llib.cmh(present[:, None], y, strata)
    a = float((present & y).sum())
    b = float((present & ~y).sum())
    c = float((~present & y).sum())
    d = float((~present & ~y).sum())
    return {"motif": "EDIT:noop (pre-fix) == terminal EDIT",
            "or_cond": float(orr[0]), "chi_cond": float(chi[0]),
            "or_pooled": ((a + .5) * (d + .5)) / ((b + .5) * (c + .5)),
            "n_present": int(present.sum())}


def runs_per_task() -> dict:
    """The runs-per-task distribution for every corpus.

    This belongs in a result file, not in the macro generator. It used to be recomputed from the
    raw token streams by mknumbers.py, which meant the macro generator could not run without the
    full corpora present, which meant a review packet regenerated a corrupted numbers.tex.
    """
    import collections
    import statistics
    out = {}
    for tag, fn in (("A", "tokens.jsonl"), ("Aone", "tokens_dual.jsonl"),
                    ("B", "tokens_frontier.jsonl"), ("C", "tokens_openhands.jsonl")):
        path = os.path.join(llib.DATA, fn)
        counts: dict[str, int] = collections.Counter()
        with llib.open_stream(path) as f:
            for line in f:
                counts[json.loads(line)["instance_id"]] += 1
        v = sorted(counts.values())
        out[tag] = {"median": int(statistics.median(v)),
                    "q1": v[len(v) // 4], "q3": v[(3 * len(v)) // 4],
                    "n_tasks": len(v),
                    "frac_single_run": sum(1 for x in v if x == 1) / len(v)}
    return out


def main() -> int:
    out = {
        "note": "Under the pre-fix labelling the final action's missing observation was called "
                "`noop`. Every run therefore ended in a fictitious no-op. `EDIT:noop` was the "
                "strongest negative survivor in the L3 scan and was entirely this artefact.",
        "A1": measure(llib.load_dual(), "A1"),
        "prefix_artefact_effect": prefix_odds_ratio(llib.load_dual()),
        "A0": measure(llib.load_tokens(), "A0"),
        "runs_per_task": runs_per_task(),
    }
    dst = os.path.join(llib.DATA, "r8_terminal.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["A1"], indent=1))
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
