"""Build the paper's figures from the result JSONs. No number is typed in here by hand."""
from __future__ import annotations

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llib  # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
    # Matplotlib's default embeds Type 3 fonts, which the ACM upload checker rejects. Type 42
    # is TrueType and passes. This has to be set at the generator: nothing downstream of the
    # figure PDF can undo it.
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
GREY, BLUE, RED = "#555555", "#1f5fa8", "#b2182b"

# Declare the size the text should be ON THE PAGE, not in the figure.
#
# savefig.bbox="tight" crops each figure to its content, which lands BELOW the width it is
# then drawn at (\textwidth 506.295pt for the two figure*, \columnwidth 241.15pt for the
# single-column one), so LaTeX MAGNIFIES every figure, by a different factor each. Font sizes
# are absolute points inside the figure, so they are magnified too. Measured from the shipped
# camera-ready with pdfplumber: fig_forest's panel titles rendered at 9.06pt against 8.97pt
# body text, and the three figures carried three different text sizes (ticks at 7.19, 7.82 and
# 7.85pt) purely because each got a different upscale. Declaring a size here and assuming it
# is what prints is how that happened.
#
# So divide the target by the measured upscale. Oversized furniture is also what forced the
# data area to be crushed during the page-limit fight: shrinking it back buys roughly 30% more
# plotting area at an identical page footprint, which is the whole point of this table.
#
# UPSCALE is measured, not guessed. Each figure's rendered width is its /MediaBox width times
# the factor below, and the factor is read back off the built figure as DRAWN_WIDTH / MediaBox
# width. It is mildly self-referential, because narrower tick labels make a narrower tight bbox
# and so a LARGER upscale, so it takes two iterations to settle. Changing a figure's geometry
# makes this table stale: re-measure rather than assume the declared sizes are what print.
PAGE_PT = {"title": 7.5, "label": 7.0, "tick": 6.5}   # all below the 8.97pt body text
UPSCALE = {"fig_qq": 1.2146, "fig_forest": 1.2245, "fig_repeat": 1.0998}


def fonts(name):
    """rcParams that render at PAGE_PT once LaTeX has scaled this figure up."""
    u = UPSCALE[name]
    return {"font.size": PAGE_PT["label"] / u,
            "axes.labelsize": PAGE_PT["label"] / u,
            "axes.titlesize": PAGE_PT["title"] / u,
            "xtick.labelsize": PAGE_PT["tick"] / u,
            "ytick.labelsize": PAGE_PT["tick"] / u,
            "legend.fontsize": PAGE_PT["tick"] / u}


def save(fig, name):
    """PDF for the paper, PNG beside it so the figure can actually be looked at before it ships."""
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=170)


def load(name):
    with open(os.path.join(llib.DATA, name), encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ figure 1
def fig_manhattan(r3, alpha="l2", config="nosubmit"):
    v = r3[config][alpha]
    motifs = v["_motifs"]
    cp = np.array(v["_chi_pooled"])
    cc = np.array(v["_chi_cond"])
    lp = -np.log10(np.maximum(llib.chi2_sf(cp), 1e-300))
    lc = -np.log10(np.maximum(llib.chi2_sf(cc), 1e-300))
    # order by the motif's leading symbol so the axis has a readable structure
    order = np.argsort([m.split(">")[0] + f"{len(m.split('>'))}" + m for m in motifs])
    lead = [motifs[i].split(">")[0] for i in order]
    bounds, prev = [], None
    for i, s in enumerate(lead):
        if s != prev:
            bounds.append((i, s))
            prev = s

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 2.9), sharex=True)
    for ax, y, lam, nul, name in (
            (axes[0], lp[order], v["lambda_pooled"], v["lambda_pooled_null_mean"],
             "pooled (task ignored)"),
            (axes[1], lc[order], v["lambda_cond"], v["lambda_cond_null_mean"],
             "conditioned on task (CMH)")):
        colors = [BLUE if i % 2 == 0 else GREY for i in range(len(bounds))]
        cmap = {}
        for j, (b, s) in enumerate(bounds):
            cmap[s] = colors[j]
        ax.scatter(range(len(y)), y, s=4, c=[cmap[s] for s in lead], linewidths=0)
        ax.axhline(-math.log10(0.05), color=RED, lw=0.6, ls=":")
        ax.set_ylabel(r"$-\log_{10} p$")
        ax.text(0.995, 0.92, f"{name}:  $\\lambda={lam:.2f}$   "
                             f"(null $\\lambda={nul:.2f}$)",
                transform=ax.transAxes, ha="right", va="top", fontsize=7)
    axes[1].set_xticks([b for b, _ in bounds])
    axes[1].set_xticklabels([s for _, s in bounds], rotation=45, ha="right", fontsize=5.5)
    axes[1].set_xlabel(f"{v['n_motifs']} motifs (k$\\leq$3), grouped by leading action class")
    fig.align_ylabels(axes)
    save(fig, "fig_manhattan")
    plt.close(fig)


def fig_qq(r3, r5, alpha="l2", alpha_c="l3", config="nosubmit"):
    """The genomic-control figure: observed quantiles against the theoretical chi-square(1) with
    the within-task permutation null drawn beside them, for both corpora that have replicate runs.
    The null is the whole argument, so it is the heavier element."""
    panels = [(r3[config][alpha], "A1 pooled", "_chi_pooled", "_null_q_pooled",
               "lambda_pooled", "lambda_pooled_null_mean"),
              (r3[config][alpha], "A1 stratified", "_chi_cond", "_null_q_cond",
               "lambda_cond", "lambda_cond_null_mean"),
              (r5["scan"][alpha_c], "C pooled", "_chi_pooled", "_null_q_pooled",
               "lambda_pooled", "lambda_pooled_null_mean"),
              (r5["scan"][alpha_c], "C stratified", "_chi_cond", "_null_q_cond",
               "lambda_cond", "lambda_cond_null_mean")]
    ymax = 20.0
    # Height is page budget, not taste: this figure spans both columns, so every 0.1in here frees
    # roughly a line in each. Anything below about 1.3in starts crowding the legends into the
    # points, so check the rendered page after changing it. The height stayed at 0.96 through the
    # font repair; what grew was the share of it the data gets, once the furniture stopped being
    # magnified to 8.4pt.
    with plt.rc_context(fonts("fig_qq")):
        _fig_qq_body(panels, ymax)


def _fig_qq_body(panels, ymax):
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 0.96), sharey=True)
    for ax, (v, name, ko, kn, lk, nk) in zip(axes, panels):
        m = v["n_motifs"]
        exp = -np.log10((np.arange(1, m + 1) - 0.5) / m)
        obs = np.sort(np.array(v[ko]))[::-1]
        nullq = np.sort(np.array(v[kn]))[::-1]
        ax.plot([0, exp.max()], [0, exp.max()], color="k", lw=0.6)
        ax.plot(exp, -np.log10(np.maximum(llib.chi2_sf(nullq), 1e-300)),
                color=RED, lw=1.3, label=f"null $\\lambda$={v[nk]:.2f}")
        ax.plot(exp, np.minimum(-np.log10(np.maximum(llib.chi2_sf(obs), 1e-300)), ymax), ".",
                color=BLUE, ms=2.2, label=f"obs. $\\lambda$={v[lk]:.2f}")
        ax.set_title(name, pad=2)
        ax.set_xlabel(r"expected $-\log_{10}p$")
        # Headroom, not padding. The legend sits upper-left in axes coordinates and the clipped
        # points pile up at y=ymax, so at 1.05 the second legend line lands on top of them: in the
        # A1 pooled panel a data point printed through the "4" of "obs. lambda=3.14". Leaving the
        # top ~25% of the range empty puts both legend lines above the ceiling row of points.
        ax.set_ylim(0, ymax * 1.34)
        ax.set_xlim(0, exp.max() * 1.02)
        ax.legend(loc="upper left", frameon=False, handlelength=1.0, borderpad=0.1,
                  labelspacing=0.15)
    axes[0].set_ylabel(r"observed $-\log_{10}p$")
    save(fig, "fig_qq")
    plt.close(fig)


# ------------------------------------------------------------------ figure 2
LABEL = {"pr_v_given_e": r"$P(V\!\mid\! E)$", "pr_e_given_e": r"$P(E\!\mid\! E)$",
         "pr_x_given_x": r"$P(X\!\mid\! X)$", "max_x_run": "mean max X-run",
         "v_ratio": "V share", "x_ratio": "X share", "steps": "steps"}


def fig_forest(r1, r6, alpha="xepvb"):
    keys = ["pr_v_given_e", "pr_e_given_e", "pr_x_given_x", "max_x_run",
            "v_ratio", "x_ratio", "steps"]
    c = r1["contrast"][alpha]
    # The whole-corpus row used to be drawn as a bare point while the caption and the alt text
    # both promised an interval on every level. r6 bootstraps that estimate over tasks, the same
    # resampling unit as the other two rows, so plot it rather than weaken the caption.
    corpus = r6["corpus_contrast"][alpha]
    with plt.rc_context(fonts("fig_forest")):
        _fig_forest_body(keys, c, corpus)


def _fig_forest_body(keys, c, corpus):
    # wspace: 0.45 was tried once and is KNOWN BAD. It does separate the outermost tick labels of
    # adjacent panels, but it narrows every panel, and the X-share panel carries three ticks
    # (-0.04, 0.00, 0.04) whose first two then collide INSIDE the panel and render as "-0.040.00".
    # Trading a near-touch between panels for an overlap within one is a worse figure.
    #
    # The between-panel touch was real, not cosmetic: pdfplumber extracted the mean-max-X-run
    # panel's "0.0" and the V-share panel's "0.00" from the shipped PDF as the single token
    # "0.00.00", i.e. a ZERO gap, and it shipped that way in the submitted version too. What makes
    # a modest value safe now is that the tick labels are no longer magnified to 7.85pt, so they
    # are about 17% narrower and the X-share panel has interior room it did not have before. Check
    # BOTH the between-panel gaps and the X-share panel's internal ones after changing this.
    # The height is load-bearing for the page limit; do not touch it.
    fig, axes = plt.subplots(1, len(keys), figsize=(7.0, 0.98),
                             gridspec_kw={"wspace": 0.20})
    for ax, k in zip(axes, keys):
        rows = [("whole corpus\n(pooled)", corpus[k]["delta"], corpus[k]["lo"],
                 corpus[k]["hi"], GREY),
                ("dual-outcome\n(pooled)", c["pooled"][k]["delta"], c["pooled"][k]["lo"],
                 c["pooled"][k]["hi"], BLUE),
                ("dual-outcome\n(within task)", c["within"][k]["delta"], c["within"][k]["lo"],
                 c["within"][k]["hi"], RED)]
        for i, (_lab, d, lo, hi, col) in enumerate(rows):
            yy = 2 - i
            if lo is not None:
                ax.plot([lo, hi], [yy, yy], color=col, lw=1.2, solid_capstyle="butt")
            ax.plot([d], [yy], "o", color=col, ms=3.2)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels(["corpus", "pooled", "within"] if k == keys[0] else [])
        ax.set_title(LABEL[k], pad=3)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, 2.6)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2, prune="both"))
        lo_, hi_ = ax.get_xlim()
        pad = 0.08 * (hi_ - lo_)
        ax.set_xlim(min(lo_, -pad * 0.2), max(hi_, pad * 0.2))
    # No shared x-label. It was positioned in FIGURE coordinates, so shrinking the figure to fit
    # the page limit walked it up into the tick labels. The axis is named in the caption instead,
    # which cannot collide with anything.
    save(fig, "fig_forest")
    plt.close(fig)


# ------------------------------------------------------------------ figure 3
def fig_repeat(r2, model="swe-agent-llama-70b", min_runs=4):
    rows = llib.load_tokens()
    cells = {}
    for r in rows:
        if r["model"] != model:
            continue
        cells.setdefault(r["instance_id"], []).append(int(bool(r["resolved"])))
    rates = np.array([np.mean(v) for v in cells.values() if len(v) >= min_runs])
    m = r2["models"][model]
    with plt.rc_context(fonts("fig_repeat")):
        _fig_repeat_body(rates, m, min_runs)


def _fig_repeat_body(rates, m, min_runs):
    fig, ax = plt.subplots(figsize=(3.33, 1.06))
    ax.hist(rates, bins=np.linspace(0, 1, 21), color=BLUE, edgecolor="white", linewidth=0.3)
    # Log scale: the never-solved bar is an order of magnitude taller than everything else, and
    # the shape of the rest is the point of the figure.
    ax.set_yscale("log")
    # Autoscaling left only two labelled ticks, 10^2 and 10^3, over a range whose bars run from 18
    # to 2,673, so a reader could not put a number on any of the short bars. Pinning the floor at
    # 10 gives three labelled decades. The floor is BELOW the smallest bin (18 of 3,387 tasks, in
    # the 0.55-0.60 bucket) and no bin is empty, so nothing is clipped by it; re-check that if the
    # binning or the model ever changes.
    ax.set_ylim(bottom=10)
    ax.set_yticks([10, 100, 1000])
    # Plain integers rather than 10^n. A log tick's exponent is mathtext at ~0.68 of the tick size,
    # which at this scale renders about 4.5pt on the page, the smallest thing in any figure here.
    # "10 / 100 / 1000" says the same thing with no glyph below the tick size, and three decades
    # is short enough that the powers of ten buy nothing.
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(f"per-task resolve rate ({m['k_tasks']} tasks with $\\geq${min_runs} runs)")
    ax.set_ylabel("tasks")
    ax.text(0.5, 0.95,
            f"ICC $=$ {m['icc']:.2f}\n{100 * m['discordance']['frac_tasks_discordant']:.0f}% of tasks"
            f" give both outcomes",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=PAGE_PT["tick"] / UPSCALE["fig_repeat"])
    save(fig, "fig_repeat")
    plt.close(fig)


def main() -> int:
    r1 = load("r1_replicate.json")
    r2 = load("r2_variance.json")
    r3 = load("r3_association.json")
    r5 = load("r5_openhands.json")
    r6 = load("r6_robust.json")
    fig_qq(r3, r5)
    fig_forest(r1, r6)
    fig_repeat(r2)
    print("figures ->", os.path.abspath(FIG))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
