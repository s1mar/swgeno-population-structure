"""Turn the cached SWE-bench Verified `.traj` files into the same token rows as A0.

Corpus B: three frontier action-generator models under the same SWE-agent scaffold, one run per
(instance, model). Its role in the paper is replication across model families: the instance-level
contrast here is BETWEEN models rather than between repeated runs of one model, which is a
different confound structure and therefore a real second test.

Run: python extract_frontier.py --out ../data/tokens_frontier.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# The trajectory ingest modules are vendored in ./llab/, so this tree runs standalone.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llab import ingest_traj  # noqa: E402
from extract_tokens import row_of  # noqa: E402

# Corpus B is built from published SWE-bench Verified submission trajectories, which are files on
# disk rather than a Hugging Face dataset, so unlike the other two extractors this one cannot fetch
# its own input. Point FRONTIER_TRAJ_CACHE at a directory of those submission folders to re-run it.
# The derived token stream (data/tokens_frontier.jsonl) ships, so nothing in the paper depends on
# being able to run this.
CACHE = os.path.abspath(os.environ.get(
    "FRONTIER_TRAJ_CACHE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "frontier_trajs")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data",
                                                  "tokens_frontier.jsonl"))
    a = ap.parse_args()

    folders = {label: (folder, name)
               for label, (folder, name) in ingest_traj.SUBMISSIONS.items()
               if os.path.isdir(os.path.join(CACHE, folder))}
    print(f"cached submissions: {sorted(folders)}")

    n = 0
    with open(a.out, "w", encoding="utf-8") as out:
        for label, (folder, model) in sorted(folders.items()):
            res = ingest_traj.results(folder)
            resolved = set(res.get("resolved", []))
            d = os.path.join(CACHE, folder)
            files = sorted(f for f in os.listdir(d) if f.endswith(".traj"))
            kept = 0
            for fn in files:
                iid = fn[:-5]
                t = ingest_traj.parse_traj(os.path.join(d, fn), iid, model, iid in resolved)
                if t is None or t.n_steps == 0:
                    continue
                r = row_of(t, kept)
                r["model"] = model
                out.write(json.dumps(r) + "\n")
                kept += 1
                n += 1
            print(f"  {label:20s} {kept:4d} runs, {sum(1 for f in files if f[:-5] in resolved):4d} resolved")
    print(f"wrote {n} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
