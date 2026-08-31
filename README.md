# Same Task, Different Fate: Population Structure in Trajectory-Based Analyses of Coding Agents

Replication package for the SWGeno 2026 paper. Everything the paper reports is regenerated from
this tree: the result files, every number in the manuscript, the figures and the survivor table.
All four token streams are included, so nothing here needs a download to run.

## Quick start

```
pip install numpy scipy matplotlib
cd analysis
python mknumbers.py          # regenerates numbers.tex and survivors.tex from data/
python verify_r10.py         # independent re-derivation of the heterogeneity numbers
```

`mknumbers.py` writes the LaTeX macros the manuscript prints. Every numeral in the paper comes from
one of them; none is typed by hand. If your `numbers.tex` differs from the one that built the
paper, something has changed.

## Every file in this package

```
analysis/
  llib.py               estimator library: symbol sequences, k-gram presence matrices,
                        pooled chi-square, Cochran-Mantel-Haenszel, genomic inflation, BH-FDR,
                        cluster bootstrap. Every stage imports this.
  extract_tokens.py     builds the A0 token stream from the public SWE-agent corpus
  extract_frontier.py   builds the corpus B token stream
  extract_openhands.py  builds the corpus C token stream
  r1_replicate.py       reproduction of a published signature, and the estimand contrast
  r2_variance.py        intraclass correlations: both arms of the confound
  r3_association.py     the motif scan, pooled against task-stratified, with its permutation null
  r4_frontier.py        corpus B: where conditioning on the task is not enough
  r5_openhands.py       corpus C: replication on another scaffold, model and benchmark
  r6_robust.py          alphabet and configuration robustness
  r7_control.py         planted positive and negative controls
  r8_terminal.py        the terminal-action artefact
  r9_entailment.py      outcome entailment
  r10_heterogeneity.py  heterogeneity of the common odds ratio across tasks
  r11_a1_sensitivity.py sensitivity of the within-task contrast to how corpus A1 is built
  make_figures.py       the paper's figures
  mknumbers.py          result files -> numbers.tex and survivors.tex
  verify.py             independent re-derivation of 23 headline quantities from the raw streams
  verify_r10.py         independent re-derivation of the heterogeneity numbers, sharing no code
                        with r10; also checks that the Fisher noncentral hypergeometric sampler
                        returns the odds ratio it is asked for
data/
  tokens.jsonl.gz            A0, the unselected corpus: 80,035 runs over 3,591 tasks
  tokens_dual.jsonl          A1, the both-outcome corpus: 5,004 runs over 348 tasks
  tokens_frontier.jsonl      B, one run per (task, model): 629 runs
  tokens_openhands.jsonl.gz  C, a different scaffold, model and benchmark
  tokens_sample.jsonl        a prefix of A0, used by verify.py for its faster pass
  r1..r11_*.json             the result files each stage writes
llab/
  __init__.py, ingest.py, ingest_openhands.py, ingest_traj.py, schema.py, actions.py
                        trajectory ingest, vendored so this tree runs standalone. Only the
                        extract_*.py scripts use these.
spec/
  FROZEN_SPEC.md        the pre-registration and its five amendments
```

The two large streams are gzipped (112 MB and 57 MB uncompressed, 4.7 MB and 3.5 MB here). Every
loader opens either form transparently, so no manual decompression is needed.

## A token stream

One compact JSON line per agent run: the outcome, the step count, the raw command verbs, and the
run rendered under three alphabets. Deriving these once is what makes the study re-runnable on a
laptop instead of on gigabytes of trajectory text.

## Re-running the analyses

Every stage reads only files in `data/` and writes its own result file there. Run them from
`analysis/`, in any order:

```
python r1_replicate.py   python r2_variance.py    python r3_association.py
python r4_frontier.py    python r5_openhands.py   python r6_robust.py
python r7_control.py     python r8_terminal.py    python r9_entailment.py
python r10_heterogeneity.py                       python r11_a1_sensitivity.py
```

Runtimes range from seconds to several minutes. `r2_variance.py` is the longest, about seven
minutes, and prints nothing at all while it works, so give it time rather than assuming it has
hung. `r3_association.py` and `r5_openhands.py` are next, because each runs 200 within-task label
permutations.

Every stage is seeded and deterministic: re-running one overwrites its result file with the same
bytes. `r10_heterogeneity.py` reads an optional `R10_PERM` environment variable to run a faster
smoke test; the paper uses its default of 200.

## Rebuilding the token streams from source (optional)

The streams are included, so this is only needed to verify the extraction itself. It downloads
large public datasets and needs `pip install datasets`.

```
cd analysis
python extract_tokens.py    --scan 200000 --min-steps 1 --out ../data/tokens.jsonl
python extract_openhands.py --out ../data/tokens_openhands.jsonl
```

Those two fetch the Hugging Face datasets `nebius/SWE-agent-trajectories` and
`nebius/SWE-rebench-openhands-trajectories`. Third-party corpus text is not redistributed here.

`extract_frontier.py` is different and cannot fetch its own input. Corpus B is built from published
SWE-bench Verified submission trajectories, which are directories of files rather than a dataset,
so the script reads them from a local cache:

```
FRONTIER_TRAJ_CACHE=/path/to/submission/folders python extract_frontier.py --out ../data/tokens_frontier.jsonl
```

The derived stream `data/tokens_frontier.jsonl` ships, so nothing in the paper depends on being
able to re-run it; the script is included so the derivation is inspectable.

## How corpus A1 was built

Stated here as well as in the paper, because the exact rule matters for interpreting every
within-task number:

1. take the first 40,000 rows of `nebius/SWE-agent-trajectories` in stream order;
2. keep runs with at least 15 agent steps;
3. keep the tasks that carry both a resolved and an unresolved run;
4. cap each task at 12 runs per outcome.

That yields 348 tasks, 5,004 runs, 1,777 of them resolved. Inside that window those 348 tasks are
every both-outcome candidate, so no further selection is applied. `r11_a1_sensitivity.py` rebuilds
the population with each rule relaxed and reports what changes; it refuses to report anything
unless its reconstruction of A1 matches the shipped corpus run for run.

## Specification

`spec/FROZEN_SPEC.md` is the pre-registration. It fixes the alphabets, the k-gram space, the
estimators, the corrections and the direction of every prediction, and was written before any
association result was inspected. Its five amendments are recorded in it, in order, with the reason
for each; the fifth covers the two analyses added after acceptance. Two of the three pre-registered
predictions held, and the one that missed is reported in the paper rather than dropped.

## Requirements

Python 3.11 or newer with `numpy`, `scipy` and `matplotlib`. Add `datasets` only to rebuild the
token streams. No network access is needed otherwise.

## Licence and data provenance

The code here is released for replication. The trajectory corpora are third-party datasets under
their own terms; the derived token streams are included, and the extraction scripts fetch the
originals.
