"""Emit every number the paper quotes as a LaTeX macro, straight from the result JSONs.

No figure in the manuscript is typed by hand. `main.tex` says \\NoutcomeICC, never "0.52". If a
result changes, this file regenerates and the paper follows; if a macro is missing, LaTeX fails
loudly instead of printing a stale value silently.

Run: python mknumbers.py   ->   numbers.tex
"""
from __future__ import annotations

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
# Resolve to the sibling data/ when run from the repo, and to ./data when run from inside a
# review packet. A reviewer who cannot execute this script ends up reviewing a corrupted
# regeneration of it, which has happened.
# Packet-local data/ WINS. The other order silently resolves to the repository's real data
# when a packet sits inside it, so the packet's self-test passes while a reviewer working from a
# copy regenerates a corrupted numbers.tex and reviews that. That has now happened twice.
DATA = (os.path.join(HERE, "data") if os.path.isdir(os.path.join(HERE, "data"))
        else os.path.join(HERE, "..", "data"))
OUT = os.path.join(HERE, "numbers.tex")

LINES: list[str] = []

# LaTeX macro names cannot contain digits, so the alphabet labels are spelled out.
ALPHA_TAG = {"l1": "LONE", "l2": "LTWO", "l3": "LTHREE", "xepv": "XEPV", "xepvb": "XEPVB"}


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


def mac(name: str, value, fmt: str = "{:.3f}"):
    if value is None:
        s = "??"
    elif isinstance(value, str):
        s = value
    elif isinstance(value, int):
        s = f"{value:,}".replace(",", "{,}")
    else:
        s = fmt.format(value)
    # An ASCII hyphen is a hyphen, not a minus sign. \ensuremath survives in both text and math
    # mode, and these macros are used in both.
    s = s.replace("-", r"\ensuremath{-}")
    LINES.append(f"\\newcommand{{\\{name}}}{{{s}}}")


def pct(name, value, fmt="{:.1f}"):
    mac(name, 100 * value, fmt)


def ci(name, lo, hi, fmt="{:.3f}"):
    mac(name, f"[{fmt.format(lo)}, {fmt.format(hi)}]")


def main() -> int:
    r1 = load("r1_replicate.json")
    r2 = load("r2_variance.json")
    r3 = load("r3_association.json")
    r4 = load("r4_frontier.json")
    r5 = load("r5_openhands.json")
    r6 = load("r6_robust.json")
    r7 = load("r7_control.json")
    r8 = load("r8_terminal.json")
    r9 = load("r9_entailment.json")

    # ---------------------------------------------------------------- corpora
    c = r1["corpus"]
    # The one figure in this file that is not computed: the row count the dataset card states.
    mac("Acorpuspublished", 80036)
    mac("Acorpusruns", c["n"])
    mac("Acorpustasks", c["n_instances"])
    pct("Acorpusrate", c["resolution_rate"])
    a1 = r1["a1"]
    mac("Aoneruns", a1["n"])
    mac("Aonetasks", a1["n_instances"])
    mac("Aoneresolved", a1["n_resolved"])
    pct("Aoneratepct", a1["n_resolved"] / a1["n"])
    d = r1["replication_draw"]
    mac("Drawruns", d["n"])
    mac("Drawresolved", d["n_resolved"])
    pct("Drawrate", d["resolution_rate"])
    mac("Drawtasks", d["n_instances"])
    mac("Drawrunspertask", d["n"] / d["n_instances"], "{:.0f}")
    mac("Drawseventyb", d["models"]["swe-agent-llama-70b"])
    mac("Draweightb", d["models"]["swe-agent-llama-8b"])
    mac("Drawfourohfiveb", d["models"]["swe-agent-llama-405b"])

    b = r4["corpus"]
    mac("Bruns", b["n"])
    mac("Btasks", b["n_instances"])
    mac("Bmulti", b["n_instances_multi_model"])
    mac("Bdiscordant", b["n_instances_discordant"])
    mac("Bdiscordantruns", b["n_runs_in_discordant"])
    pct("Bratepct", sum(v["resolved"] for v in b["models"].values()) / b["n"])

    cc = r5["corpus"]
    mac("Cruns", cc["n"])
    mac("Ctasks", cc["n_instances"])
    pct("Crate", cc["resolution_rate"])
    mac("Cdualtasks", r5["dual"]["n_instances"])
    mac("Cdualruns", r5["dual"]["n_runs"])

    # Protocol item 1 says report the runs-per-task distribution. A paper that recommends that
    # and does not do it has written a rule it does not follow. Computed in r8, not here: this
    # file reads result JSONs only, so it runs anywhere the JSONs do.
    for tag, v in r8["runs_per_task"].items():
        mac(f"{tag}rpt", f"{v['median']} ({v['q1']} to {v['q3']})")
        pct(f"{tag}rptone", v["frac_single_run"], "{:.0f}")

    # ---------------------------------------------------- repeatability (A, C)
    m = r2["models"]["swe-agent-llama-70b"]
    mac("outcomeICC", m["icc"], "{:.2f}")
    ci("outcomeICCci", *m["icc_ci"], fmt="{:.2f}")
    mac("outcomeICCtasks", m["k_tasks"])
    mac("outcomeICCruns", m["n_runs"])
    pct("outcomediscordfrac", m["discordance"]["frac_tasks_discordant"], "{:.0f}")
    mac("outcomediscordobs", m["discordance"]["observed"], "{:.3f}")
    mac("outcomediscordexp", m["discordance"]["expected_if_independent"], "{:.3f}")
    mac("outcomediscordratio", m["discordance"]["ratio"], "{:.2f}")
    mac("repoICC", r2["repo_level"]["icc"], "{:.2f}")
    ci("repoICCci", *r2["repo_level"]["icc_ci"], fmt="{:.2f}")
    bi = r2["behaviour_icc"]["swe-agent-llama-70b"]
    for k, nm in (("x_ratio", "behXICC"), ("v_ratio", "behVICC"),
                  ("e_ratio", "behEICC"), ("steps", "behstepsICC"),
                  ("max_x_run", "behmaxxICC")):
        mac(nm, bi[k]["icc"], "{:.2f}")
    lo = min(bi[k]["icc"] for k in ("x_ratio", "v_ratio", "e_ratio", "steps", "max_x_run"))
    hi = max(bi[k]["icc"] for k in ("x_ratio", "v_ratio", "e_ratio", "steps", "max_x_run"))
    mac("behICClo", lo, "{:.2f}")
    mac("behICChi", hi, "{:.2f}")
    # The interval matters here, not only the point: the sentence these feed says the two smaller
    # samples are "clear of zero", and the paper gives an interval everywhere else.
    for mm, nm in (("swe-agent-llama-8b", "outcomeICCeightb"),
                   ("swe-agent-llama-405b", "outcomeICCfourohfiveb")):
        mac(nm, r2["models"][mm]["icc"], "{:.2f}")
        ci(nm + "ci", *r2["models"][mm]["icc_ci"], fmt="{:.2f}")

    co = r5["outcome_icc"]
    mac("CoutcomeICC", co["icc"], "{:.2f}")
    ci("CoutcomeICCci", *co["icc_ci"], fmt="{:.2f}")
    mac("CoutcomeICCtasks", co["k_tasks"])
    mac("CoutcomeICCruns", co["n_runs"])
    pct("Cdiscordfrac", co["discordance"]["frac_tasks_discordant"], "{:.0f}")
    mac("CbehICClo", min(v["icc"] for v in r5["behaviour_icc"].values()), "{:.2f}")
    mac("CbehICChi", max(v["icc"] for v in r5["behaviour_icc"].values()), "{:.2f}")

    # ------------------------------------------------- replication of Table 15
    pub = r1["published"]["table15"]
    ours = r1["replication_pooled"]["xepvb"]
    lit = r1["replication_pooled"]["xepv"]
    short = {"pr_v_given_e": "PVE", "pr_e_given_e": "PEE", "pr_x_given_x": "PXX",
             "max_x_run": "MAXX", "v_ratio": "VR", "x_ratio": "XR", "steps": "STEPS"}
    worst = 0.0
    for k, s in short.items():
        f = "{:.1f}" if k in ("max_x_run", "steps") else "{:.3f}"
        mac(f"pub{s}res", pub[k][0], f)
        mac(f"pub{s}unres", pub[k][1], f)
        mac(f"our{s}res", ours["resolved"][k], f)
        mac(f"our{s}unres", ours["failed"][k], f)
        mac(f"lit{s}res", lit["resolved"][k], f)
        mac(f"lit{s}unres", lit["failed"][k], f)
        for a, b_ in ((pub[k][0], ours["resolved"][k]), (pub[k][1], ours["failed"][k])):
            scale = max(abs(pub[k][0]), abs(pub[k][1]))
            worst = max(worst, abs(a - b_) / scale)
    pct("replmaxdev", worst, "{:.0f}")
    # A single percentage hides which statistic is off, and scaling a small statistic by the larger
    # of its pair flatters it. Report the absolute deviation for the five quantities that live on
    # [0, 1], and quote the two that do not separately in the text.
    share_keys = ["pr_v_given_e", "pr_e_given_e", "pr_x_given_x", "v_ratio", "x_ratio"]
    devs = [max(abs(pub[k][0] - ours["resolved"][k]), abs(pub[k][1] - ours["failed"][k]))
            for k in share_keys]
    # Round the bound UP, not to nearest. The prose says the statistics agree "to within" this
    # number, so a value that rounds down turns a true statement into a false one: the worst
    # deviation is 0.0501, which to nearest is 0.050 and is then exceeded by the thing it bounds.
    import math as _m
    mac("repldevshare", _m.ceil(max(devs) * 1000) / 1000, "{:.3f}")
    mac("repldevsteps", max(abs(pub["steps"][0] - ours["resolved"]["steps"]),
                            abs(pub["steps"][1] - ours["failed"]["steps"])), "{:.2f}")
    mac("repldevmaxx", abs(pub["max_x_run"][0] - ours["resolved"]["max_x_run"]), "{:.1f}")
    mac("litVR", lit["resolved"]["v_ratio"], "{:.3f}")
    mac("pubVR", pub["v_ratio"][0], "{:.3f}")

    # ------------------------------------------------- pooled vs within (A1)
    con = r1["contrast"]["xepvb"]
    corpus_delta = r1["corpus_delta"]["xepvb"]
    for k, s in short.items():
        f = "{:.1f}" if k in ("max_x_run", "steps") else "{:.3f}"
        mac(f"corp{s}", corpus_delta[k], f)
        for lvl, tag in (("pooled", "pool"), ("within", "wthn"),
                         ("within_runweighted", "wrwt")):
            v = con[lvl][k]
            mac(f"{tag}{s}", v["delta"], f)
            ci(f"{tag}{s}ci", v["lo"], v["hi"], fmt=f)
    mac("Aonecontrasttasks", con["n_instances"])

    # How much of each pooled effect the within-task estimate retains. Computed only over the
    # statistics that actually survive conditioning; the two that do not are quoted separately.
    survived = ["pr_v_given_e", "pr_e_given_e", "max_x_run", "v_ratio", "steps"]
    retained = [100 * con["within"][k]["delta"] / con["pooled"][k]["delta"] for k in survived]
    mac("survretainlo", min(retained), "{:.0f}")
    mac("survretainhi", max(retained), "{:.0f}")
    mac("nsurvived", len(survived))
    mac("PXXshrinkpct",
        100 * (1 - abs(con["within"]["pr_x_given_x"]["delta"])
               / abs(con["pooled"]["pr_x_given_x"]["delta"])), "{:.0f}")

    # ------------------------------------------------------------ lambda table
    for scope, tag in (("raw", "raw"), ("nosubmit", "ns"), ("nosubmit_band", "nsb"),
                       ("prefix_nosubmit", "pre"), ("prefix_nosubmit_band", "preb")):
        for alpha in ("l1", "l2", "l3", "xepvb"):
            v = r3[scope][alpha]
            A = ALPHA_TAG[alpha]
            mac(f"lam{tag}{A}P", v["lambda_pooled"], "{:.2f}")
            mac(f"lam{tag}{A}C", v["lambda_cond"], "{:.2f}")
            mac(f"lam{tag}{A}PN", v["lambda_pooled_null_mean"], "{:.2f}")
            mac(f"lam{tag}{A}CN", v["lambda_cond_null_mean"], "{:.2f}")
            mac(f"nm{tag}{A}", v["n_motifs"])
            mac(f"surv{tag}{A}", v["n_survivors"])
            mac(f"fdrP{tag}{A}", v["n_fdr_pooled"])
            mac(f"fdrC{tag}{A}", v["n_fdr_cond"])
    # How well the null MEAN is pinned. The stored interval is the percentile spread of the
    # individual permutation lambdas, so sd ~ width/3.92 and the standard error of the mean is
    # that over sqrt(n_perm). Quoted so a reader knows the null is not one draw.
    # This bounds EVERY row the table prints, xepv-b included. It used to be maxed over the five
    # fine-alphabet rows only, while the table carried a sixth: the caption said "in every row" and
    # the xepv-b row missed the bound by a factor of three. The row set here must track the table.
    import math as _math
    widths = []
    for src, keys in ((r3["nosubmit"], ("l1", "l2", "l3", "xepvb")), (r5["scan"], ("l1", "l3"))):
        for a in keys:
            for k in ("lambda_pooled_null_ci", "lambda_cond_null_ci"):
                lo_, hi_ = src[a][k]
                widths.append(hi_ - lo_)
    mac("nullse", _math.ceil(max(widths) / 3.92 / _math.sqrt(r3["config"]["n_perm"]) * 100) / 100,
        "{:.2f}")

    ns = [r3["nosubmit"][a] for a in ("l1", "l2", "l3")]
    mac("nullPlo", min(v["lambda_pooled_null_mean"] for v in ns), "{:.2f}")
    mac("nullPhi", max(v["lambda_pooled_null_mean"] for v in ns), "{:.2f}")
    mac("nullClo", min(v["lambda_cond_null_mean"] for v in ns), "{:.2f}")
    mac("nullChi", max(v["lambda_cond_null_mean"] for v in ns), "{:.2f}")
    xe = [r3["nosubmit"][a] for a in ("xepv", "xepvb")]
    mac("nullCxepvlo", min(v["lambda_cond_null_mean"] for v in xe), "{:.2f}")
    mac("nullCxepvhi", max(v["lambda_cond_null_mean"] for v in xe), "{:.2f}")
    mac("nmxepvlo", min(v["n_motifs"] for v in xe))
    mac("nmxepvhi", max(v["n_motifs"] for v in xe))
    # The floor below which a median is not read as an inflation factor. Two tables carry a row
    # under it and both must mark it, so the threshold is a macro rather than a typed "50".
    mac("lammin", r3["config"]["lambda_min_motifs"])
    # NOTE the bootstrap resample count is deliberately NOT a macro. It is 2,000, and so is the
    # replication draw size \Drawruns, but they are unrelated quantities; a shared macro would
    # couple them silently. Sourcing it would mean importing the analysis package, which would
    # break this file's rule of reading result JSONs only and make the review packet non-portable.

    # ------------------------------------------------------- corpus B and C lambdas
    for alpha in ("l1", "l2", "xepvb"):
        A = ALPHA_TAG[alpha]
        v = r4["scan_clean"][alpha]
        w = r4["scan_model_strata"][alpha]
        mac(f"BtaskP{A}", v["lambda_pooled"], "{:.2f}")
        mac(f"BtaskC{A}", v["lambda_cond"], "{:.2f}")
        mac(f"BtaskCN{A}", v["lambda_cond_null_mean"], "{:.2f}")
        mac(f"BmodelC{A}", w["lambda_cond"], "{:.2f}")
        mac(f"BmodelCN{A}", w["lambda_cond_null_mean"], "{:.2f}")
        mac(f"BmodelPN{A}", w["lambda_pooled_null_mean"], "{:.2f}")
        mac(f"Bsurvtask{A}", v["n_survivors"])
        mac(f"Bsurvmodel{A}", w["n_survivors"])
        mac(f"Bnm{A}", v["n_motifs"])
    mm = r4["model_mix_in_discordant"]
    mac("Bclaudefourres", mm["resolved"]["Claude 4 Sonnet"])
    mac("Bclaudefourfail", mm["failed"]["Claude 4 Sonnet"])
    mac("Bgptfores", mm["resolved"]["GPT-4o"])
    mac("Bgptfofail", mm["failed"]["GPT-4o"])

    for alpha in ("l1", "l3"):
        A = ALPHA_TAG[alpha]
        v = r5["scan"][alpha]
        mac(f"CP{A}", v["lambda_pooled"], "{:.2f}")
        mac(f"CC{A}", v["lambda_cond"], "{:.2f}")
        mac(f"CPN{A}", v["lambda_pooled_null_mean"], "{:.2f}")
        mac(f"CCN{A}", v["lambda_cond_null_mean"], "{:.2f}")
        mac(f"Cnm{A}", v["n_motifs"])
        mac(f"Csurv{A}", v["n_survivors"])
    # Corpus C's behavioural contrast, reported rather than left in the results file: the
    # methodological failure mode replicates across scaffolds while the behavioural signature does
    # not, and that contrast is only visible if both halves are quoted.
    _cg = r5["contrast_generic"]
    _shares = [k for k in _cg["pooled"] if k.endswith("_share")]
    mac("Cnshares", len(_shares))
    mac("Cstepsretain",
        100 * _cg["within"]["steps"]["delta"] / _cg["pooled"]["steps"]["delta"], "{:.0f}")
    mac("Cstepsattenuate",
        100 * (1 - _cg["within"]["steps"]["delta"] / _cg["pooled"]["steps"]["delta"]), "{:.0f}")
    # Every share must be null under BOTH estimands for the sentence in Section 7 to be true.
    _allnull = all(v["lo"] < 0 < v["hi"]
                   for k in _shares for v in (_cg["pooled"][k], _cg["within"][k]))
    mac("Csharesallnull", "yes" if _allnull else "NO")

    cg = r5["contrast_generic"]
    mac("Cstepspool", cg["pooled"]["steps"]["delta"], "{:.1f}")
    ci("Cstepspoolci", cg["pooled"]["steps"]["lo"], cg["pooled"]["steps"]["hi"], fmt="{:.1f}")
    mac("Cstepswthn", cg["within"]["steps"]["delta"], "{:.1f}")
    ci("Cstepswthnci", cg["within"]["steps"]["lo"], cg["within"]["steps"]["hi"], fmt="{:.1f}")

    # ------------------------------------------------------------- survivors
    sv = r3["nosubmit_band"]["l3"]["survivors"]
    mac("survLthreelist", ", ".join(s["motif"] for s in sv[:5]).replace("_", "\\_"))
    for i, s in enumerate(sv[:6]):
        mac(f"survmotif{'abcdef'[i]}", s["motif"].replace(">", "$\\rightarrow$"))
        mac(f"survor{'abcdef'[i]}", s["or_cond"], "{:.2f}")
        mac(f"survorp{'abcdef'[i]}", s["or_pooled"], "{:.2f}")
    # R9: the seven statistics with the outcome-entailed actions removed. The paper's own rule
    # (protocol item 4) applied to the paper's own headline table.
    ent = r9["contrast_xepvb"]["within"]
    for k, tag in short.items():
        f = "{:.1f}" if k in ("max_x_run", "steps") else "{:.3f}"
        v = ent[k]
        mac(f"ent{tag}", v["delta"], f)
        ci(f"ent{tag}ci", v["lo"], v["hi"], fmt=f)
    # The two outcome-entailed hits, quoted separately. They were previously described with one
    # motif's odds ratio and prevalence applied to both, which is false of the submission motif.
    for motif, stem in (("EDIT:src", "entailEDIT"), ("SUBMIT:none", "entailSUB")):
        hit = [t for t in r3["raw"]["l2"]["top"] if t["motif"] == motif]
        if hit:
            h = hit[0]
            frac = h["n_present"] / r3["raw"]["l2"]["n_runs"]
            mac(stem + "or", h["or_cond"], "{:.0f}" if h["or_cond"] > 20 else "{:.2f}")
            pct(stem + "pct", frac, "{:.0f}")
            pct(stem + "rest", 1 - frac, "{:.0f}")

    # The per-run estimator, named as such. The prose previously quoted the POOLED literal-adapter
    # value and called it the per-run one; they differ by more than a factor of two.
    mac("perrunPVEres", r1["replication_perrun"]["xepvb"]["resolved"]["pr_v_given_e"], "{:.3f}")
    mac("perrunlitPVEres", r1["replication_perrun"]["xepv"]["resolved"]["pr_v_given_e"], "{:.3f}")
    # The size of the replicated statistic set, so the abstract cannot mix "5 of seven".
    mac("nstats", len(short))
    surv = [k for k, v in r9["verdict"].items() if v["still_excludes_zero"]]
    mac("nsurvivedent", len(surv))
    mac("entVRretain", 100 * r9["verdict"]["v_ratio"]["retained_fraction"], "{:.0f}")
    mac("entPVEretain", 100 * r9["verdict"]["pr_v_given_e"]["retained_fraction"], "{:.0f}")
    # The complement, named separately. The prose once attributed the RETAINED fraction to
    # submission, which inverts the claim: what submission accounts for is what was removed.
    mac("entVRremoved", 100 * (1 - r9["verdict"]["v_ratio"]["retained_fraction"]), "{:.0f}")

    # Named lookups, so the prose never depends on a motif's rank in a sorted list.
    def by_motif(scope, alpha, motif, stem):
        hit = [s for s in r3[scope][alpha]["survivors"] if s["motif"] == motif]
        if not hit:
            return
        mac(stem + "OR", hit[0]["or_cond"], "{:.2f}")
        mac(stem + "ORpool", hit[0]["or_pooled"], "{:.2f}")
        mac(stem + "N", hit[0]["n_present"])
        # The same motif restricted to a prefix. Two of the catalogue's strongest rows cross 1
        # here, which is a sign reversal in a paper whose sharpest result is a sign reversal, so
        # the prose has to be able to name the value rather than leave it in the table alone.
        pf = r3["prefix_" + scope.replace("nosubmit", "nosubmit")][alpha]
        p = dict(zip(pf["_motifs"], pf["_or_cond"])).get(motif)
        if p:
            mac(stem + "ORpre", p, "{:.2f}")

    # `EDIT:noop` used to be the headline negative survivor. It was an artefact: the terminal
    # action of a run has no following observation, so every run ended in a fake no-op. With the
    # terminal step excluded, the strongest negative survivor is a genuine mid-run pattern.
    by_motif("nosubmit_band", "l3", "EDIT:err>EDIT:err>EDIT:err", "editerr")
    by_motif("nosubmit_band", "l3", "VIEW:ok>EDIT:err>EDIT:err", "viewediterr")
    by_motif("nosubmit_band", "l3", "EDIT:ok>EXEC:ok", "editexec")
    by_motif("nosubmit_band", "l3", "EXEC:ok", "execok")

    # How many of the rows the survivor table prints reverse direction inside the prefix. Counted,
    # not asserted: the number is a claim about the generated table and has to follow it.
    _full = r3["nosubmit_band"]["l3"]["survivors"][:8]
    _pre = dict(zip(r3["prefix_nosubmit_band"]["l3"]["_motifs"],
                    r3["prefix_nosubmit_band"]["l3"]["_or_cond"]))
    mac("survrevn", sum(1 for s in _full
                        if _pre.get(s["motif"]) and (_pre[s["motif"]] - 1) * (s["or_cond"] - 1) < 0))
    # The terminal-step artefact the paper reports in Section 11.
    t = r8["A1"]
    pct("Aendterminal", t["frac_runs_terminal"], "{:.0f}")
    pct("Aendofnoop", t["frac_of_all_noop_that_was_terminal"], "{:.0f}")
    pct("Aendexit", t["edit_terminal_frac_context_exhausted"], "{:.0f}")
    mac("Aendruns", t["edit_terminal_runs"])
    mac("Aendmidrun", t["runs_with_a_mid_run_edit_noop"])
    mac("AendOR", r8["prefix_artefact_effect"]["or_cond"], "{:.2f}")

    negs = [s for s in r3["nosubmit_band"]["l3"]["survivors"] if s["or_cond"] < 1]
    mac("survLthreeneg", len(negs))
    mac("survLthreepos", r3["nosubmit_band"]["l3"]["n_survivors"] - len(negs))

    # ------------------------------------------------------------ robustness (R6)
    fc = r6["icc_fleiss_cuzick"]
    mac("iccFC", fc["icc"], "{:.2f}")
    ci("iccFCci", *fc["ci"], fmt="{:.2f}")
    cw = r6["corpus_contrast"]["xepvb"]
    for k, s in short.items():
        f = "{:.1f}" if k in ("max_x_run", "steps") else "{:.3f}"
        ci(f"corp{s}ci", cw[k]["lo"], cw[k]["hi"], fmt=f)
    j = r6["joint"]
    mac("jointstrata", j["n_strata"])
    mac("jointboth", j["n_strata_both_outcomes"])
    # A1's model composition. "A1 mixes three model sizes" is true and, on its own, misleading:
    # one model supplies nine runs in ten, which is the real reason joint stratification changes
    # nothing. Stating the share makes that argument stronger rather than weaker.
    _mix = j["model_mix"]
    pct("Aonedompct", max(_mix.values()) / sum(_mix.values()), "{:.0f}")
    jc = j["contrast"]["xepvb"]["within"]
    for k, s in short.items():
        f = "{:.1f}" if k in ("max_x_run", "steps") else "{:.3f}"
        mac(f"jnt{s}", jc[k]["delta"], f)
        ci(f"jnt{s}ci", jc[k]["lo"], jc[k]["hi"], fmt=f)
    for alpha in ("l1", "l2", "l3"):
        A = ALPHA_TAG[alpha]
        v = j["scan"][alpha]
        mac(f"jntlam{A}C", v["lambda_cond"], "{:.2f}")
        mac(f"jntlam{A}CN", v["lambda_cond_null_mean"], "{:.2f}")
        mac(f"jntsurv{A}", v["n_survivors"])
    mo = r6["a1_model_by_outcome"]
    mac("Aonebigres", mo["resolved"]["swe-agent-llama-405b"])
    mac("Aonebigfail", mo["failed"]["swe-agent-llama-405b"])

    # ------------------------------------------------------------ controls (R7)
    for kind, tag in (("real", "ctlreal"), ("spurious", "ctlspur")):
        v = r7["a1"][kind]
        pct(tag + "poolrej", v["pooled_reject_rate"], "{:.1f}")
        pct(tag + "condrej", v["cond_reject_rate"], "{:.1f}")
        mac(tag + "poolor", v["pooled_or_median"], "{:.2f}")
        mac(tag + "condor", v["cond_or_median"], "{:.2f}")
        mac(tag + "poolchi", v["pooled_chi_median"], "{:.1f}")
        mac(tag + "condchi", v["cond_chi_median"], "{:.2f}")
        mac(tag + "poolrejn", int(round(v["pooled_reject_rate"] * v["n_replicates"])))
        mac(tag + "condrejn", int(round(v["cond_reject_rate"] * v["n_replicates"])))
    mac("ctlreps", r7["a1"]["real"]["n_replicates"])
    mac("ctlortrue", r7["a1"]["config"]["or_true"], "{:.1f}")
    # Power of each corpus B stratification, so a null result there can be read.
    WORD = {1.5: "onefive", 2.0: "two", 3.0: "three", 5.0: "five"}
    for key, tag in (("by_model", "Bpowmodel"), ("by_task", "Bpowtask")):
        for e in r7["b_power"][key]:
            pct(tag + WORD[e["or_planted"]], e["reject_rate"], "{:.0f}")
        mac(tag + "strata", r7["b_power"][key][0]["n_strata"])
    # The planted odds ratios themselves, so the prose does not hand-type a configuration value.
    for e in r7["b_power"]["by_model"]:
        mac("ORplanted" + WORD[e["or_planted"]], e["or_planted"], "{:.1f}")

    # ---- camera-ready additions: heterogeneity of the common odds ratio (R10) -------------
    r10 = load("r10_heterogeneity.json")
    s10 = r10["l3"]["survivors"]
    mac("hetperm", r10["config"]["n_perm"])
    mac("hetn", len(s10))
    mac("hetstratalo", min(r["n_informative_strata"] for r in s10))
    mac("hetstratahi", max(r["n_informative_strata"] for r in s10))
    same = sorted(r["frac_strata_same_sign"] for r in s10)
    pct("hetsamelo", same[0], "{:.0f}")
    pct("hetsamehi", same[-1], "{:.0f}")
    pct("hetsamemed", same[len(same) // 2], "{:.0f}")
    # The agreement fraction read against the SAME homogeneous reference used for Q. Without it
    # the raw fraction reads as reassurance, and it points the other way: a world with one
    # identical odds ratio in every task produces MORE agreement than we observe.
    mac("hetsamebelow", sum(1 for r in s10
                            if r["frac_strata_same_sign"] < r["frac_same_sign_homog_mean"]))
    mac("hetsamebelowlo", sum(1 for r in s10
                              if r["frac_strata_same_sign"] < r["frac_same_sign_homog_lo"]))
    top_null = max(s10, key=lambda r: r["or_cond"])
    pct("hettopsamenull", top_null["frac_same_sign_homog_mean"], "{:.0f}")
    i2 = sorted(r["i2_calibrated"] for r in s10)
    mac("hetitwolo", i2[0], "{:.2f}")
    mac("hetitwohi", i2[-1], "{:.2f}")
    mac("hetitwomed", i2[len(i2) // 2], "{:.2f}")
    # Benjamini-Hochberg, not a nominal 0.05: the motif scan itself is FDR controlled,
    # so counting homogeneity rejections at a raw threshold would hold this question to
    # a looser standard than the paper holds its own findings to.
    mac("hetrejn", sum(1 for r in s10 if r["q_homog_rejected_bh"]))
    top = max(s10, key=lambda r: r["or_cond"])
    mac("hettopor", top["or_cond"], "{:.2f}")
    pct("hettopsame", top["frac_strata_same_sign"], "{:.0f}")
    mac("hetLTWOn", len(r10["l2"]["survivors"]))
    mac("hetLTWOrejn", sum(1 for r in r10["l2"]["survivors"] if r["q_homog_rejected_bh"]))

    # ---- camera-ready additions: how A1 was actually built, and its sensitivity (R11) -----
    r11 = load("r11_a1_sensitivity.json")
    c11 = r11["config"]
    rule = c11["a1_rule"]
    mac("Aonewindow", rule["window"])
    mac("Aoneminsteps", rule["min_steps"])
    mac("Aonecap", rule["cap"])
    pct("Aonebelowfloor", c11["a0_frac_runs_below_15_steps"], "{:.1f}")
    VARIANTS = [k for k in r11 if k != "config"]
    mac("sensn", len(VARIANTS))

    def _w(tag, metric, which="within"):
        return r11[tag][which][metric]

    def excl(d):
        return d["lo"] > 0 or d["hi"] < 0

    # Which of the seven statistics in Table 2 hold their sign AND stay clear of zero in every
    # build. This is counted, not asserted: an earlier draft claimed all seven and was wrong,
    # because the exploration self-loop is precisely the one that does not.
    SEVEN = ["pr_v_given_e", "pr_e_given_e", "pr_x_given_x", "max_x_run",
             "v_ratio", "x_ratio", "steps"]
    stable = [m for m in SEVEN
              if len({_w(v, m)["delta"] > 0 for v in VARIANTS}) == 1
              and all(excl(_w(v, m)) for v in VARIANTS)]
    mac("sensstable", len(stable))
    mac("sensseven", len(SEVEN))

    xr = [_w(v, "x_ratio") for v in VARIANTS]
    # A printed range must contain the values it claims to bound, so the low end floors and the
    # high end ceils. Rounding to nearest printed a lower bound ABOVE the smallest value it
    # covered, which makes the sentence quoting it false.
    mac("sensXRlo", math.floor(min(d["delta"] for d in xr) * 1000) / 1000, "{:+.3f}")
    mac("sensXRhi", math.ceil(max(d["delta"] for d in xr) * 1000) / 1000, "{:+.3f}")
    mac("sensXRexcl", sum(1 for d in xr if excl(d)))
    xrp = [_w(v, "x_ratio", "pooled") for v in VARIANTS]
    mac("sensXRpoolneg", sum(1 for d in xrp if d["hi"] < 0))
    pxx = {v: _w(v, "pr_x_given_x") for v in VARIANTS}
    mac("sensPXXzero", sum(1 for v in VARIANTS if not excl(pxx[v])))
    nf = pxx["nosteps_nowindow_nocap"]
    mac("sensPXXnofloor", nf["delta"], "{:+.3f}")
    ci("sensPXXnofloorci", nf["lo"], nf["hi"])
    mac("senswidetasks", r11["nosteps_nowindow_nocap"]["n_tasks"])
    mac("senswideruns", r11["nosteps_nowindow_nocap"]["n_runs"])
    wide = _w("nosteps_nowindow_nocap", "x_ratio")
    mac("senswideXR", wide["delta"], "{:+.3f}")
    ci("senswideXRci", wide["lo"], wide["hi"])
    # The share of THIS build's runs that fall under A1's step floor, which is the number the
    # sentence about it actually needs. The corpus-wide share is a different quantity.
    pct("sensnofloorpct", r11["nosteps_nowindow_nocap"]["frac_below_floor"], "{:.0f}")

    # ---- camera-ready additions: the replication target's own adapter (33A-3) -------------
    xe = r1["contrast"]["xepv"]
    mac("xepvPXXwthn", xe["within"]["pr_x_given_x"]["delta"], "{:+.3f}")
    ci("xepvPXXwthnci", xe["within"]["pr_x_given_x"]["lo"], xe["within"]["pr_x_given_x"]["hi"])
    mac("xepvXRwthn", xe["within"]["x_ratio"]["delta"], "{:+.3f}")
    ci("xepvXRwthnci", xe["within"]["x_ratio"]["lo"], xe["within"]["x_ratio"]["hi"])

    # ---- camera-ready additions: what the motif floor is worth (33C-Q4) -------------------
    # The null lambda's own spread against the motif count, which is the measured justification
    # for the floor. Read from the configuration lambda is read from.
    for a in ("xepv", "l3"):
        v = r3["nosubmit"][a]
        tag = ALPHA_TAG[a]
        mac("floornm" + tag, v["n_motifs"])
        ci("floorci" + tag, v["lambda_cond_null_ci"][0], v["lambda_cond_null_ci"][1], "{:.2f}")

    # config
    cfg = r3["config"]
    mac("minsupport", cfg["min_support"])
    mac("nperm", cfg["n_perm"])
    mac("prefixK", cfg["prefix"])
    mac("orfloor", cfg["or_floor"], "{:.1f}")
    mac("fdrq", cfg["fdr_q"], "{:.2f}")
    mac("bandlo", cfg["band"][0], "{:.2f}")
    mac("bandhi", cfg["band"][1], "{:.2f}")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("% GENERATED by mknumbers.py. Do not edit.\n")
        f.write("\n".join(LINES) + "\n")
    print(f"wrote {len(LINES)} macros -> {OUT}")

    write_survivor_table(r3)
    return 0


def write_survivor_table(r3, alpha="l3", n=8):
    """The survivor catalogue as a table body, generated rather than transcribed.

    Each row carries the pooled odds ratio beside the task-conditioned one so the shrinkage is
    visible, and the conditioned odds ratio when the motif is only allowed to be observed in the
    first K steps, which is the only version in which the behaviour precedes most of the episode.
    """
    full = r3["nosubmit_band"][alpha]
    pf = r3["prefix_nosubmit_band"][alpha]
    pre = dict(zip(pf["_motifs"], pf["_or_cond"]))
    pre_surv = {s["motif"] for s in pf["survivors"]}
    rows = []
    for s in full["survivors"][:n]:
        m = s["motif"]
        p = pre.get(m)
        # "not tested" means the motif did not clear the support rule inside a 10-step prefix,
        # which is itself informative: it is a late-episode pattern.
        pre_or = f"{p:.2f}" if p else "not tested"
        star = "$^{\\ast}$" if m in pre_surv else ""
        label = m.replace("_", "\\_").replace(">", "}$\\rightarrow$\\texttt{")
        rows.append(f"\\texttt{{{label}}} & "
                    f"{s['n_present']:,} & {s['or_pooled']:.2f} & {s['or_cond']:.2f} & "
                    f"{pre_or}{star} \\\\".replace(",", "{,}"))
    # The whole tabular is generated, not just its rows: an \input that ends mid-alignment
    # confuses booktabs' \bottomrule.
    body = ["\\footnotesize",
            "\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
            "motif & runs & OR pooled & OR within & prefix \\\\", "\\midrule",
            *rows, "\\bottomrule", "\\end{tabular}%"]
    dst = os.path.join(HERE, "survivors.tex")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("% GENERATED by mknumbers.py. Do not edit.\n")
        f.write("\n".join(body) + "\n")
    print(f"wrote {len(rows)} survivor rows -> {dst}")


if __name__ == "__main__":
    raise SystemExit(main())
