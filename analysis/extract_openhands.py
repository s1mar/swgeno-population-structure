"""Corpus C: OpenHands trajectories (Qwen3-Coder-480B) on SWE-rebench, as token rows.

A different scaffold, a different action-generator model and a different benchmark, all changing
at once. That makes it useless for attributing a behavioural difference to any one of them, and it
is not used for that. It is used for the one claim that does not need them held apart: whether
pooling runs across tasks inflates a motif-outcome association somewhere other than SWE-agent.

Unselected: no outcome quota, no gold join, every run in the scanned prefix.

Run: python extract_openhands.py --scan 20000 --out ../data/tokens_openhands.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

# The trajectory ingest modules are vendored in ./llab/, so this tree runs standalone.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llab import ingest_openhands as oh  # noqa: E402
from extract_tokens import row_of  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=int, default=20000)
    ap.add_argument("--min-steps", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data",
                                                  "tokens_openhands.jsonl"))
    a = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(oh.DS, split="train", streaming=True)
    n_seen = n_kept = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for row in itertools.islice(ds, a.scan):
            n_seen += 1
            try:
                t = oh._normalize(row)
            except Exception:                              # noqa: BLE001
                continue
            if t.n_steps < a.min_steps:
                continue
            f.write(json.dumps(row_of(t, n_seen - 1)) + "\n")
            n_kept += 1
            if n_seen % 2000 == 0:
                print(f"  scanned {n_seen:6d}  kept {n_kept:6d}", flush=True)
    print(f"done: scanned {n_seen}, kept {n_kept} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
