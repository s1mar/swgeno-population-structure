# FROZEN SPEC, SWGeno 2026 paper

Written **before** any association result was computed. Nothing below may be changed after a result
is seen; if something has to change, the change is appended at the bottom with the reason and the
date, and the paper reports both. This is the house rule and it is the only
thing standing between an honest study and one fitted to the first plot.

Frozen 2026-07-26.

---

## 0. Question

Published behavioural analyses of coding agents associate trajectory structure with task outcome by
**pooling runs across different task instances**. Task instances differ enormously in difficulty.
How much of that association survives holding the task fixed?

The study is about the **validity of a measurement**, not about causes of failure. No causal claim
about behaviour is made anywhere, because behaviour is emitted during the episode by the same
process that produces the outcome.

## 1. Corpora

- **A0, unselected.** `data/tokens.jsonl`: every run in the first 200,000 rows of
  `nebius/SWE-agent-trajectories` with at least one agent step. No outcome filter, no length filter.
  Field `order` records stream position, so any prefix sample is reproducible from the file.
- **A1, dual-outcome.** `data/swgeno_dual_corpus.pkl`: 348 instances that each carry both outcomes,
  capped at 12 runs per outcome per instance; 5,004 runs. Ascertained on the phenotype, so it is
  used **only** for within-instance contrasts, never for prevalence or variance components.
- **B, frontier.** Cached SWE-bench Verified `.traj` for GPT-4o, Claude 3.5 Sonnet, Claude 4 Sonnet
  under the same SWE-agent scaffold. One run per (instance, model).
- **C, other scaffold.** OpenHands / Qwen3-Coder-480B on SWE-rebench. Descriptive only.

## 2. Alphabets (all four are computed; none is privileged)

- **XEPV**: the replication target's alphabet, its published SWE-agent adapter applied literally.
  `search_dir, find_file, open, goto, scroll_up, scroll_down, ls` -> X; `edit, create` -> E;
  `pytest, submit` -> V; everything else -> E (the target's stated default); P is empty by
  construction because every SWE-agent turn carries a command.
  - **XEPV-b**, one pre-registered sensitivity: any command that executes code or tests
    (`python, python3, pytest, tox, make, bash, sh, nosetests, unittest, coverage`) -> V.
    Declared now because the literal adapter sends `python reproduce.py` to E, which is arguable.
- **L1**: EDIT, CREATE, VIEW, SEARCH, LIST, EXEC, FS, SUBMIT, OTHER (9 symbols).
- **L2**: L1 symbol x target class (test / src / cfg / other / none).
- **L3**: L1 symbol x observation class (err / noop / ok).

## 3. Motif space

k-grams for k = 1, 2, 3 over each alphabet, contiguous, within a single run. A motif enters the
scan only if it is **present in at least 20 runs and absent in at least 20 runs** of the analysis
set (both arms need support for a 2x2 test). Primary encoding is presence/absence per run; a
count-based encoding is a declared sensitivity, not a second bite.

## 4. Estimators

**Pooled** (what prior work does): 2x2 association between motif presence and `resolved` over all
runs, chi-square with 1 df.

**Conditioned**: Cochran-Mantel-Haenszel chi-square stratified by `instance_id`, on the same runs.
Strata with no variation in motif or in outcome contribute nothing, which is correct.
Effect size: the CMH common odds ratio.

Runs are the unit. The 16,877 "pairs" are **not** independent observations and no per-pair test is
run; they are reported only as a description of the design.

**Metric contrasts** (for the seven published summary statistics in section 6): pooled difference
of means (resolved minus failed) versus the within-instance difference, the latter computed per
instance and then averaged with equal weight per instance. Uncertainty by bootstrap over instances,
2,000 resamples, percentile interval.

## 5. Stratification diagnostic

lambda = median(chi-square) / 0.4549, the genomic inflation factor, computed over all motifs in the
scan, separately for the pooled and the conditioned test.

**Null:** permute the `resolved` labels **within instance**, 200 permutations. This preserves each
instance's resolve rate and each instance's motif distribution, and destroys only the run-level
association. It therefore isolates two distinct things:
- pooled lambda under the null > 1 means task stratification alone inflates the pooled test;
- conditioned lambda under the null ~ 1 means the conditioned test is calibrated, and any excess in
  the observed conditioned lambda is signal rather than overlapping-k-gram correlation.

Reported as a 2x2 table: {pooled, conditioned} x {observed, null}.

**Predictions, recorded now.** Pooled null lambda > 1.2. Conditioned null lambda in [0.9, 1.1].
Pooled observed lambda > conditioned observed lambda. If any of these fails, it is reported as a
failed prediction, in the paper, in those words.

Prior evidence that this could go the other way, and is therefore a real test: on a 60-instance
version of A1, recent work found that for its four detector scores the pooled estimate fell inside
the within-instance interval, i.e. difficulty was *not* the confound for those scores.

## 6. The replication arm

The target reports, on 2,000 runs of this corpus: resolution 16.9% (338/1,662), model mix
70B 1,793 / 8B 167 / 405B 40, and seven statistics split by outcome, namely Pr(V|E) 54.2 / 28.1,
Pr(E|E) 41.5 / 63.6, Pr(X|X) 74.6 / 84.8, mean max X-run 4.8 / 11.0, V ratio 24.7 / 15.7, X ratio
33.6 / 44.9, mean steps 16.0 / 30.1 (resolved / unresolved).

Our replication draw is the **first 2,000 runs of A0 in stream order**, fixed now. We report each
statistic beside the published value, then the same statistic conditioned on instance in A1. We do
not claim the published numbers are wrong; the replication exists to show that the conditioned
estimate answers a different question from the pooled one.

## 7. Variance components (A0 only)

For instances with at least 4 runs of the **same** model (so model identity is not the source of
the variance), the outcome is binary and:
- observed-scale intraclass correlation from a one-way random-effects decomposition with the
  standard unequal-group-size correction;
- liability-scale value by the Dempster-Lerner transformation, reported as a repeatability, which
  in quantitative genetics is an upper bound on heritability, not heritability itself;
- a model-free companion: the **discordance rate**, the fraction of same-(instance, model) run
  pairs whose outcomes disagree, against the rate expected if runs were independent Bernoulli draws
  at the instance's own rate.

Bootstrap over instances for all three.

## 8. Survivor catalogue

A motif is a survivor if, on A1 with the conditioned test, it passes Benjamini-Hochberg FDR at 0.05
**and** has a CMH common odds ratio outside [1/1.5, 1.5]. Both thresholds are fixed now, because
with thousands of strata almost anything passes FDR alone. Survivors are reported with their pooled
counterpart so the shrinkage is visible. Replication of the survivor list is checked on B.

## 9. Temporal precedence

Every motif is recomputed on the **first 10 steps only** and the conditioned test is repeated.
Motifs still associated when only the prefix is observed are labelled *prognostic*; the rest are
*concomitant*. This is a descriptive split, not a detector. No abort policy, no operating point, no
savings estimate: recent work already reports that surface stopping signals fail at deployable
false-abort budgets, and re-litigating that is out of scope.

## 10. Scaffold contrast (C)

One paragraph, descriptive, explicitly labelled unidentifiable: scaffold, model and benchmark all
change together, so a difference cannot be attributed to the scaffold.

## 11. Multiple testing and reporting

BH-FDR within each alphabet, with the number of motifs tested stated. Every table reports n. No
p-value is reported without an effect size. Every number in the paper is produced by a script in
`analysis/` and re-derived by `verify.py`; anything not yet computed appears as
`[[MEASURE:id | what | source | expected direction]]` and blocks submission.

---

## Amendments

**A1, 2026-07-26. Estimator for the seven summary statistics: pooled, not per-run.**
Discovered while attempting the replication, before any conditioned contrast was inspected. The
published table is reproduced only if transition probabilities and symbol shares are computed by
pooling counts over a group of runs; averaging per-run rates gives visibly different values
(P(V|E) 0.221 against a published 0.542). Both estimators are now computed; the pooled one is
primary because it is the one being replicated, the per-run one is reported as a sensitivity. The
same amendment fixed the adapter question: the literal published adapter (`pytest, submit` -> V)
does **not** reproduce the published V statistics, and the pre-registered XEPV-b variant (any
command that runs code or tests -> V) does, to within 0.03 on every V and X statistic. XEPV-b is
therefore treated as the target's effective adapter, and both are reported.

**A2, 2026-07-26. A second, "clean" configuration of the motif scan.**
The first scan showed that its strongest hits are not behavioural. Two mechanisms:
- *Outcome entailment.* `submit` is how an episode ends; a run that never submits cannot be scored
  resolved. Under the published adapter `submit` is a V, so it contaminates every V statistic.
- *Degenerate minorities.* A motif present in 95% of runs (`X>E`, "looked at something, then
  edited") tests whether the remaining 5% are degenerate runs, which is not the question.
The pre-registered scan is kept and reported unchanged. A second configuration removes `submit`
from every sequence and restricts the scan to motifs with prevalence in [0.10, 0.90]. Both are in
the results file; the paper reports the clean configuration as primary and the difference between
them as a finding in its own right.

**A3, 2026-07-27. The terminal step is excluded from the clean configurations.**
Found by an audit of this project's own changes, not by a reviewer. The final action of a run has
no following observation, because the run ended. The extractor labelled that missing observation
with the same class it uses for an observation that came back empty, so every run in both corpora
ended in a fictitious no-op: 100% of runs, and 45% of all `noop` tokens in A1. The terminal token
therefore encoded HOW THE EPISODE STOPPED, which is entailed by the outcome in exactly the sense
Section 9 of the paper describes.

The consequence was not hypothetical. `EDIT:noop` was the strongest negative survivor in the L3
scan, common odds ratio 0.19 within task on 894 runs, and it was this artefact in its entirety:
it was the last step in 894 of 894 runs carrying it, 0 runs carried one mid-run, and 92% of those
runs had exhausted their context. It passed FDR and the effect-size floor.

Fix: the extractor now labels the final observation `end` rather than `noop`, and `drop_terminal`
removes the final action in every clean configuration, generalising the existing `drop_submit`
rule (submission is only one of the ways a run can stop). The pre-registered `raw` configuration is
unchanged and still reported. Only the L3 alphabet is affected; L1, L2 and both XEPV variants do
not use the observation class. The headline inflation result does not move: the pooled null stays
at 1.63 to 2.11 and the stratified null at 1.02 to 1.05.

**A4, 2026-07-27. A minimum motif count before lambda is read.**
Adopted after the fact and recorded here because it is the amendment that touches a prediction.
Section 5 above says lambda is computed "over all motifs in the scan". It still is; what changed is
which alphabets it is *read from*. The two four-symbol alphabets carry only 30 and 39 motifs, and on
them the conditioned null came out at 1.17 to 1.19, outside the pre-registered [0.9, 1.1] band. A
median over that few statistics is not a stable estimate of anything, so lambda is now read only
from alphabets with at least **50** motifs in the scan (`lambda_min_motifs` in the analysis config).

This rule was chosen after seeing the values it excludes, which is exactly the situation the
pre-registration exists to make visible, so: the miss is reported in the paper either way, one of
the two alphabets is printed in the lambda table with a footnote marking it as under the floor, and
the same footnote is now on the corpus B table, whose four-symbol row (22 motifs) is under it too.
No headline number depends on the rule. It removes rows from being *read*, not from being shown.

**A5, 2026-08-31. Two analyses added after acceptance, at reviewers' request.**
Neither is pre-registered and neither is presented as such. Both are re-analyses of data already
collected; no new runs were made, and no number reported in the accepted version changed.

*Heterogeneity of the common odds ratio* (`analysis/r10_heterogeneity.py`). Section 5 above fixes
the Cochran-Mantel-Haenszel estimator but says nothing about its homogeneity assumption, which all
three reviewers asked about. For each motif in the survivor catalogue we retain the per-stratum 2x2
tables and report the number of informative strata, the fraction whose own odds ratio falls on the
same side of one as the common estimate, Breslow-Day, and Cochran's Q with the derived I-squared.
One decision here is a judgement call and is recorded as such: Q is read against a reference in
which every stratum carries the motif's own common odds ratio, generated by holding each stratum's
margins fixed and redrawing the table from Fisher's noncentral hypergeometric distribution. The
within-task label permutation used elsewhere in this specification would NOT serve, because it sets
the odds ratio to one and so calibrates a different quantity. Per-stratum log odds ratios use a
Haldane correction of 0.5, which shrinks each estimate toward zero by an amount that depends on the
stratum size; the calibration above absorbs that, and the I-squared recentred on the measured
homogeneous mean is the one reported, because the textbook I-squared against the chi-square degrees
of freedom is conservative here.

*Sensitivity of the within-task contrast to how A1 is built* (`analysis/r11_a1_sensitivity.py`).
Reviewers asked whether the results depend on A1's construction. Establishing that first required
recovering what A1's rule actually is, because Section 3 above and the accepted manuscript both
described a selection on comparison support that never bound. The rule, reconstructed and matched
run for run against the shipped corpus, is: the first 40,000 rows in stream order; runs of at least
15 agent steps; tasks carrying both outcomes; a cap of 12 runs per outcome. Inside that window
there are exactly 348 both-outcome candidates, so the ranking selected nothing. The 15-step floor
does bind and removes 42.4% of A0's runs. The script varies each of the three rules in turn and
refuses to report anything unless its reconstruction of A1 reproduces the published contrast
exactly. The manuscript's description of A1 and its threats section are corrected accordingly.

---

## Corrections to this document

- Section 6 said "eight statistics" and then listed seven. Seven is right; the paper reports seven
  throughout. Corrected in place, noted here so a reader diffing versions sees why.
- Section 7 promises three quantities and the paper reports two. The third, the liability-scale
  value, is in the released `r2_variance.json` as `icc_liability` and reads **1.146** for the 70B
  cells: the Dempster-Lerner transformation has no ceiling at 1 and returns an out-of-range value
  when the observed-scale correlation is high relative to the base rate. It is therefore not
  reportable as a correlation and the paper does not report it. It stays in the released file with
  this note rather than being deleted, because deleting a promised quantity is worse than
  explaining a broken one.
