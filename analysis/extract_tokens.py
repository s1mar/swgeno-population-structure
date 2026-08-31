"""Stream the Nebius SWE-agent trajectory corpus and emit ONE compact JSONL row per run.

Everything downstream in this paper reads this file, not the raw corpus. Keeping only the derived
token stream (rather than 5 GB of trajectory text) is what makes the whole study re-runnable on a
laptop, and it is what gets released as the artifact.

The sample is UNSELECTED with respect to outcome: every run of every instance seen in the scanned
prefix is emitted, subject only to a minimum step count. That matters because the dual-outcome
corpus used for the association study is ascertained on the phenotype, so it cannot be used to
estimate how much outcome variance the task explains.

Row schema:
  instance_id, model, resolved, n_steps, repo, exit_status,
  l1  list[str]  coarse action class per step
  l2  list[str]  action class + target class
  l3  list[str]  action class + observation outcome
  gen_len       length of the submitted patch (0 if the run produced none)

Run: python extract_tokens.py --scan 120000 --out ../data/tokens.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys

# The trajectory ingest modules are vendored in ./llab/, so this tree runs standalone.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llab import ingest  # noqa: E402

NEBIUS = "nebius/SWE-agent-trajectories"

# ---------------------------------------------------------------- alphabet L1
# One class per harness affordance, not per command string. `python`/`pytest`/`bash` all put the
# agent in the same position (it ran something and must read output), so they are one class.
VIEW = {"open", "goto", "scroll_up", "scroll_down", "cat", "head", "tail", "view"}
SEARCH = {"search_dir", "search_file", "find_file", "grep", "find", "rg", "ag"}
EXEC = {"python", "python3", "pytest", "tox", "make", "bash", "sh", "npm", "node", "go",
        "cargo", "java", "javac", "gcc", "coverage", "nosetests", "unittest", "pip", "poetry"}
LIST = {"ls", "tree", "pwd", "cd"}


def l1_of(cmd: str) -> str:
    c = (cmd or "").strip().lower()
    if c in ("edit", "insert", "str_replace"):
        return "EDIT"
    if c == "create":
        return "CREATE"
    if c in VIEW:
        return "VIEW"
    if c in SEARCH:
        return "SEARCH"
    if c in LIST:
        return "LIST"
    if c == "submit":
        return "SUBMIT"
    if c in ("rm", "mv", "cp", "mkdir", "touch", "chmod"):
        return "FS"
    if c in EXEC:
        return "EXEC"
    # Unparsed or repo-specific binaries (`dvc`, `beet`, ...) are still the agent running something.
    if c and re.fullmatch(r"[a-z0-9_.\-/]+", c):
        return "EXEC"
    return "OTHER"


_TEST_RE = re.compile(r"(^|/)(tests?|testing)(/|$)|test_[^/]*\.py$|_test\.py$|conftest\.py$",
                      re.I)
_CFG_RE = re.compile(r"\.(cfg|ini|toml|yaml|yml|json|txt|md|rst)$", re.I)


def target_class(path: str | None) -> str:
    if not path:
        return "none"
    if _TEST_RE.search(path):
        return "test"
    if _CFG_RE.search(path):
        return "cfg"
    if path.endswith(".py"):
        return "src"
    return "other"


_ERR_RE = re.compile(
    r"traceback \(most recent call last\)|^\s*[A-Za-z_.]*(Error|Exception)\b|"
    r"\bcommand not found\b|\bno such file or directory\b|\bsyntaxerror\b|"
    r"\byour proposed edit has introduced|\bfailed\b|\b\d+ failed\b|\berror:",
    re.I | re.M)
_NOOP_RE = re.compile(r"no matches found|no such file|found 0 matches|^\s*$", re.I)


def obs_class(obs: str) -> str:
    o = obs or ""
    if _ERR_RE.search(o):
        return "err"
    if _NOOP_RE.search(o[:400]):
        return "noop"
    return "ok"


def row_of(t, order: int) -> dict:
    cmds, l1, l2, l3 = [], [], [], []
    n_unparsed = 0
    for s in t.steps:
        a = s.action
        if a is None:
            n_unparsed += 1
        # The raw verb is kept so that ANY alphabet can be rebuilt downstream from this file,
        # including the ones defined by other people's papers. Without it we would be locked
        # into the three abstractions chosen here.
        cmds.append((a.cmd if a else "").strip().lower()[:24])
        c = l1_of(a.cmd if a else "")
        l1.append(c)
        l2.append(f"{c}:{target_class(a.target_file if a else None)}")
        # The LAST action has no following observation, because the run ended. Its observation is
        # missing, not empty, and labelling it `noop` makes every trajectory end in a fake no-op:
        # measured on this corpus that was 100% of runs and 45% of all `noop` tokens, and it turned
        # the terminal token into a marker of HOW THE RUN ENDED, which is outcome-entailed. `end`
        # keeps the action and marks the observation as unobserved.
        is_last = s is t.steps[-1]
        l3.append(f"{c}:{'end' if is_last else obs_class(s.observation)}")
    return {
        # `order` is the position in the corpus stream. Any "first N rows" sample is then
        # reproducible from this file alone, without re-streaming 200k rows from the hub.
        "order": order,
        "instance_id": t.instance_id, "model": t.model_name, "resolved": bool(t.resolved),
        "n_steps": t.n_steps, "repo": t.instance_id.rsplit("-", 1)[0],
        "exit_status": t.exit_status, "n_unparsed": n_unparsed,
        "cmd": cmds, "l1": l1, "l2": l2, "l3": l3,
        "gen_len": len(t.generated_patch or ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=int, default=120000)
    # Default 1, not 15: the replication target reports a resolved-run mean of 16 steps, so any
    # long-horizon filter would delete most of the runs it is being compared against.
    ap.add_argument("--min-steps", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(NEBIUS, split="train", streaming=True)
    n_seen = n_kept = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for row in itertools.islice(ds, a.scan):
            n_seen += 1
            traj = row.get("trajectory")
            if not isinstance(traj, list):
                continue
            if sum(1 for s in traj if s.get("role") == "ai") < a.min_steps:
                continue
            try:
                t = ingest.normalize(row)
            except Exception:                                   # noqa: BLE001
                continue
            f.write(json.dumps(row_of(t, n_seen - 1)) + "\n")
            n_kept += 1
            if n_seen % 10000 == 0:
                print(f"  scanned {n_seen:7d}  kept {n_kept:7d}", flush=True)
    print(f"done: scanned {n_seen}, kept {n_kept} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
