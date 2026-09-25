#!/usr/bin/env python3
"""Context Sensitivity, Question Quality and Trend Alignment, all on [0, 1].

Three model-level metrics for one visibility x cost x gamma condition, all
computed from the trajectory exports via bootstrap_overall. Each is
higher-is-better, so they can be read and ranked the same way.

Panel A -- Sens against Qual@5:

    Sens(theta)   = max_l r(theta, l) - min_l r(theta, l)          in [0, 1]
    Qual@5(theta) = (1/c) * sum_l e(theta, l)                      in [0, 1]

Sens is the spread of the exploration rate across the context levels: how much
behaviour moves with context, with no claim about the direction. Qual@5 is
the plain mean of the per-level Exp@5 values e(theta, l), each level counting
equally.

Panel B -- the Pearson correlations of the per-level rates against context,
rho_VA and rho_EXP, summarized as Trend Alignment:

    Align(theta) = 1 - sqrt((1 - rho_VA)^2 + rho_EXP^2) / sqrt(5)  in [0, 1]

rho_VA is high when exploration falls as context grows; rho_EXP is near zero
when question quality is not systematically driven by context. Both live on
[-1, 1], so the farthest point from the ideal (1, 0) is (-1, +-1) at sqrt(5),
and dividing by that puts the distance on [0, 1]. Whiskers are percentile
CIs from a paired bootstrap over users.

model_ranks.csv joins the two and adds the aggregate:

    Overall(theta) = (Sens + Qual@5 + Align) / 3                   in [0, 1]

All four columns rank descending, so rank 1 always means best on that metric.
Overall is reported after the three, never instead of them: the mean hides
the trade-off they exist to show.

Outputs:
    sens_qual_<vis>_<cost>_g<gamma>.csv           + fig_... {pdf,png}
    align_<vis>_<cost>_g<gamma>.csv               + fig_... {pdf,png}
    fig_combined_<vis>_<cost>_g<gamma>.{pdf,png}  both panels, one legend
    model_ranks.csv

Usage:
    python make_normalized_metrics.py
    python make_normalized_metrics.py --gamma 0.5 --cost cheap
    python make_normalized_metrics.py --n-boot 0 --formats pdf
"""
import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_STYLE = [
    ("Aya-Expanse-32B", "Aya-32B", "#c2610a"),
    ("Aya-Expanse-8B", "Aya-8B", "#f7b98a"),
    ("DeepSeek-R1-Llama-8B", "DeepSeek-R1-8B", "#d1483a"),
    ("Gemini-2.5-Flash", "Gemini-2.5-Flash", "#cf94d9"),
    ("Gemini-2.5-Pro", "Gemini-2.5-Pro", "#8e46ab"),
    ("Gemma-2-2B", "Gemma-2B", "#8fd28f"),
    ("Gemma-2-9B", "Gemma-9B", "#2f8f40"),
    ("Llama-3.1-70B", "Llama-70B", "#2f6ba8"),
    ("Llama-3.1-8B", "Llama-8B", "#9dc8ec"),
    ("Qwen3.5-27B", "Qwen-27B", "#6b6b6b"),
    ("Qwen3.5-9B", "Qwen-9B", "#b4b4b4"),
]
MODELS = [(name, label) for name, label, _ in MODEL_STYLE]
MODEL_LABELS = {name: label for name, label, _ in MODEL_STYLE}
MODEL_COLORS = {name: color for name, _, color in MODEL_STYLE}

IDEAL = (1.0, 0.0)
MAX_DIST = math.sqrt(5.0)
GOLD = "#c8961e"
FIGSIZE = (4.4, 4.4)
SENS_FIGSIZE = (4.8, 3.6)
COMBINED_FIGSIZE = (8.0, 4.2)
LEGEND_TOP = 0.88
LEGEND_NCOL = 6
LEGEND_FONTSIZE = 9.0
LEGEND_ORDER = [
    "Aya-32B", "Aya-8B",
    "Gemini-2.5-Pro", "Gemini-2.5-Flash",
    "Gemma-9B", "Gemma-2B",
    "Llama-70B", "Llama-8B",
    "Qwen-27B", "Qwen-9B",
    "DeepSeek-R1-8B",
]
MARKER_SIZE = 62
EDGE_COLOR = "black"
EDGE_WIDTH = 1.1
COSTS = {"expensive": -5, "cheap": -2}

plt.rcParams.update({
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def sensitivity(va_values):
    return max(va_values) - min(va_values)


def overall(sens, qual, align):
    return (sens + qual + align) / 3.0


def alignment(rho_va, rho_exp):
    return 1.0 - math.hypot(IDEAL[0] - rho_va, IDEAL[1] - rho_exp) / MAX_DIST


def nan_to_zero(value):
    return 0.0 if value != value else value


def rho_cis(terms_by_model, n_boot, seed=0):
    import numpy as np
    from bootstrap_overall import corr_rows, level_rates

    models = list(terms_by_model)
    n_users = len(terms_by_model[models[0]]["users"])
    rng = np.random.default_rng(seed)
    out = {m: np.empty((n_boot, 2)) for m in models}
    for b in range(n_boot):
        take = rng.integers(0, n_users, n_users)
        for model in models:
            t = terms_by_model[model]
            va = level_rates(t["va_num"][take], t["va_den"][take])
            ex = level_rates(t["ex_num"][take], t["ex_den"][take])
            out[model][b] = (-nan_to_zero(corr_rows(va)[0]),
                             -nan_to_zero(corr_rows(ex)[0]))
    return {m: (tuple(np.percentile(v[:, 0], [2.5, 97.5])),
                tuple(np.percentile(v[:, 1], [2.5, 97.5])))
            for m, v in out.items()}


def rows_from_trajectories(visibility, cost, gamma, n_boot=2000, seed=0):
    from bootstrap_overall import (TRAJECTORIES, corr_rows, level_rates,
                                   load_model)

    label_of = dict(MODELS)
    terms_by_model = {
        model: load_model(model, visibility, COSTS[cost], float(gamma),
                          strict=True)
        for model in TRAJECTORIES}
    cis = rho_cis(terms_by_model, n_boot, seed) if n_boot else {}

    sens_rows, s_rows = [], []
    for model, terms in terms_by_model.items():
        va_rate = level_rates(terms["va_num"], terms["va_den"])
        ex_rate = level_rates(terms["ex_num"], terms["ex_den"])
        rho_va = -float(nan_to_zero(corr_rows(va_rate)[0]))
        rho_ex = -float(nan_to_zero(corr_rows(ex_rate)[0]))

        sens_rows.append({
            "model": model,
            "label": label_of.get(model, model),
            "va_min": float(min(va_rate)),
            "va_max": float(max(va_rate)),
            "sensitivity": sensitivity(list(va_rate)),
            "qual_exp5": float(sum(ex_rate) / len(ex_rate)),
        })
        s_rows.append({
            "model": model,
            "label": label_of.get(model, model),
            "s_va": rho_va, "s_exp5": rho_ex,
            "align": alignment(rho_va, rho_ex),
            "va_ci": cis.get(model, (None, None))[0],
            "exp_ci": cis.get(model, (None, None))[1],
        })

    s_rows.sort(key=lambda r: -r["align"])
    for i, r in enumerate(s_rows, 1):
        r["rank"] = i
    return sens_rows, s_rows


def rounded(row, fields):
    return {k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in row.items() if k in fields}


def write_sens_csv(rows, path):
    fields = ["model", "va_min", "va_max", "sensitivity", "qual_exp5"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(rounded(r, fields))


def write_dist_csv(rows, path):
    fields = ["rank", "model", "s_va", "s_exp5", "align"]
    has_ci = any(r.get("va_ci") or r.get("exp_ci") for r in rows)
    if has_ci:
        fields += ["s_va_ci_low", "s_va_ci_high",
                   "s_exp5_ci_low", "s_exp5_ci_high"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = dict(r)
            if has_ci:
                va, ex = r.get("va_ci"), r.get("exp_ci")
                row["s_va_ci_low"], row["s_va_ci_high"] = va or ("", "")
                row["s_exp5_ci_low"], row["s_exp5_ci_high"] = ex or ("", "")
            w.writerow(rounded(row, fields))


def write_model_ranks(sens_rows, s_rows, path):
    by_model = {r["model"]: r for r in sens_rows}
    merged = []
    for s in s_rows:
        sens = by_model.get(s["model"])
        if sens is None:
            continue
        row = {
            "model": s["model"],
            "sens": sens["sensitivity"],
            "qual@5": sens["qual_exp5"],
            "align": s["align"],
        }
        row["overall"] = overall(row["sens"], row["qual@5"], row["align"])
        merged.append(row)

    for key in ("sens", "qual@5", "align", "overall"):
        ordered = sorted(merged, key=lambda r: r[key], reverse=True)
        prev, prev_rank = None, 0
        for i, r in enumerate(ordered, 1):
            rank = prev_rank if prev is not None and r[key] == prev else i
            r[f"{key}_rank"] = rank
            prev, prev_rank = r[key], rank

    merged.sort(key=lambda r: r["overall_rank"])
    fields = ["model", "sens", "sens_rank", "qual@5", "qual@5_rank",
              "align", "align_rank", "overall", "overall_rank"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in merged:
            w.writerow(rounded(r, fields))
    return merged


def draw_points(ax, points):
    handles = []
    for x, y, color, label in sorted(points, key=lambda p: -(p[0] + p[1])):
        ax.scatter(x, y, s=MARKER_SIZE * 1.9, marker="o", color="white",
                   edgecolors="none", zorder=4, alpha=0.95)
        h = ax.scatter(x, y, s=MARKER_SIZE, marker="o", color=color,
                       edgecolors=EDGE_COLOR, linewidths=EDGE_WIDTH,
                       alpha=0.95, zorder=5, label=label)
        handles.append((label, h))
    order = {lab: i for i, (_, lab, _) in enumerate(MODEL_STYLE)}
    handles.sort(key=lambda t: order.get(t[0], len(order)))
    return [h for _, h in handles]


def draw_sens_quality(ax, rows):
    by_model = {r["model"]: r for r in rows}
    ax.axvline(0.0, color="0.8", lw=0.8, zorder=1)
    ax.set_xlabel(r"$\mathrm{Sens}(\theta)$")
    ax.set_ylabel(r"$\mathrm{Qual}@5(\theta)$")

    xs = [r["sensitivity"] for r in rows] + [0.0]
    ys = [r["qual_exp5"] for r in rows]
    xpad = max(0.02, (max(xs) - min(xs)) * 0.12)
    ypad = max(0.02, (max(ys) - min(ys)) * 0.12)
    ax.set_xlim(max(0.0, min(xs) - xpad), min(1.0, max(xs) + xpad))
    ax.set_ylim(max(0.0, min(ys) - ypad), min(1.0, max(ys) + ypad))
    ax.set_box_aspect(1)

    points = [(r["sensitivity"], r["qual_exp5"], color, label)
              for model, label, color in MODEL_STYLE
              for r in [by_model.get(model)] if r is not None]
    return draw_points(ax, points)


def draw_s_scores(ax, rows, show_ci=True):
    for r in rows:
        col = MODEL_COLORS.get(r["model"], "#888888")
        if show_ci and (r["va_ci"] or r["exp_ci"]):
            xerr = ([[r["s_va"] - r["va_ci"][0]], [r["va_ci"][1] - r["s_va"]]]
                    if r["va_ci"] else None)
            yerr = ([[r["s_exp5"] - r["exp_ci"][0]], [r["exp_ci"][1] - r["s_exp5"]]]
                    if r["exp_ci"] else None)
            ax.errorbar(r["s_va"], r["s_exp5"], xerr=xerr, yerr=yerr,
                        fmt="none", ecolor=col, elinewidth=1.0, capsize=2.0,
                        capthick=1.0, alpha=0.75, zorder=3)

    ax.plot(*IDEAL, marker="*", ms=16, mfc=GOLD, mec=GOLD, mew=1.0,
            ls="none", zorder=2)

    ax.set_xlabel(r"$\rho_{\mathrm{VA}}(\theta)$")
    ax.set_ylabel(r"$\rho_{\mathrm{EXP}}(\theta)$")

    xs = [r["s_va"] for r in rows] + [IDEAL[0]]
    ys = [r["s_exp5"] for r in rows] + [IDEAL[1]]
    if show_ci:
        xs += [b for r in rows if r["va_ci"] for b in r["va_ci"]]
        ys += [b for r in rows if r["exp_ci"] for b in r["exp_ci"]]
    pad = 0.22
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + pad
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")

    points = [(r["s_va"], r["s_exp5"],
               MODEL_COLORS.get(r["model"], "#888888"),
               MODEL_LABELS.get(r["model"], r["model"])) for r in rows]
    return draw_points(ax, points)


def save_single(fig, ax, handles, path_stem, formats):
    leg = ax.legend(handles=handles, loc="center left",
                    bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=11,
                    title="Model", title_fontsize=11.5, handletextpad=0.4,
                    labelspacing=0.7)
    leg._legend_box.align = "left"
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(f"{path_stem}.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sens_quality(rows, path_stem, formats):
    fig, ax = plt.subplots(figsize=SENS_FIGSIZE)
    save_single(fig, ax, draw_sens_quality(ax, rows), path_stem, formats)


def plot_s_scores(rows, path_stem, formats, show_ci=True):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    save_single(fig, ax, draw_s_scores(ax, rows, show_ci=show_ci),
                path_stem, formats)


def plot_combined(sens_rows, s_rows, path_stem, formats, show_ci=True):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=COMBINED_FIGSIZE)
    draw_sens_quality(ax_a, sens_rows)
    handles = draw_s_scores(ax_b, s_rows, show_ci=show_ci)
    fig.tight_layout(rect=(0, 0, 1.0, LEGEND_TOP))

    rank = {lab: i for i, lab in enumerate(LEGEND_ORDER)}
    handles = sorted(handles,
                     key=lambda h: rank.get(h.get_label(), len(rank)))
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), frameon=False,
               fontsize=LEGEND_FONTSIZE,
               ncol=LEGEND_NCOL, handletextpad=0.4, columnspacing=1.1,
               borderaxespad=0.0)

    for fmt in formats:
        fig.savefig(f"{path_stem}.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gamma", default="0.9")
    p.add_argument("--visibility", default="seen", choices=["seen", "unseen"])
    p.add_argument("--cost", default="expensive", choices=list(COSTS))
    p.add_argument("--no-ci", action="store_true",
                   help="omit the CI whiskers on panel B")
    p.add_argument("--n-boot", type=int, default=2000,
                   help="bootstrap replicates for panel B's whiskers; 0 omits them")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default=here, type=Path,
                   help="where the CSVs go")
    p.add_argument("--figdir", default=here / "figures", type=Path,
                   help="parent of the per-figure-type folders")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    args = p.parse_args()

    gamma = f"{float(args.gamma):g}"
    args.outdir.mkdir(parents=True, exist_ok=True)
    for sub in ("sens_qual", "align", "combined"):
        (args.figdir / sub).mkdir(parents=True, exist_ok=True)

    n_boot = 0 if args.no_ci else args.n_boot
    rows, s_rows = rows_from_trajectories(
        args.visibility, args.cost, gamma, n_boot=n_boot, seed=args.seed)
    if not rows:
        p.error(f"no trajectories for {args.visibility}/{args.cost}/gamma={gamma}")

    unknown = [r["model"] for r in s_rows if r["model"] not in MODEL_COLORS]
    if unknown:
        print("warning: no colour mapped for", ", ".join(sorted(set(unknown))))

    rank_of = {r["model"]: r["rank"] for r in s_rows}
    rows.sort(key=lambda r: rank_of.get(r["model"], len(rank_of) + 1))

    tag = f"{args.visibility}_{args.cost}_g{gamma.replace('.', '')}"
    show_ci = not args.no_ci

    dist_csv = args.outdir / f"align_{tag}.csv"
    write_dist_csv(s_rows, dist_csv)
    dist_fig = args.figdir / "align" / f"fig_align_{tag}"
    plot_s_scores(s_rows, str(dist_fig), args.formats, show_ci=show_ci)

    sens_csv = args.outdir / f"sens_qual_{tag}.csv"
    write_sens_csv(rows, sens_csv)
    sens_fig = args.figdir / "sens_qual" / f"fig_sens_qual_{tag}"
    plot_sens_quality(rows, str(sens_fig), args.formats)

    comb_fig = args.figdir / "combined" / f"fig_combined_{tag}"
    plot_combined(rows, s_rows, str(comb_fig), args.formats, show_ci=show_ci)

    ranks_csv = args.outdir / "model_ranks.csv"
    ranked = write_model_ranks(rows, s_rows, ranks_csv)

    print(f"{args.visibility} / {args.cost} / gamma = {gamma}   "
          f"({len(rows)} models; every column higher is better)")
    width = max(len(r["model"]) for r in ranked)
    print(f"  {'#':>2}  {'model':<{width}}  {'Sens':>6} {'Qual@5':>7} "
          f"{'Align':>6} | {'Overall':>7}")
    for i, r in enumerate(ranked, 1):
        print(f"  {i:>2}  {r['model']:<{width}}  {r['sens']:>6.3f} "
              f"{r['qual@5']:>7.3f} {r['align']:>6.3f} | {r['overall']:>7.3f}")

    print(f"\nwrote {sens_csv}")
    for fmt in args.formats:
        print(f"wrote {sens_fig}.{fmt}")
    print(f"wrote {dist_csv}")
    for fmt in args.formats:
        print(f"wrote {dist_fig}.{fmt}")
    for fmt in args.formats:
        print(f"wrote {comb_fig}.{fmt}")
    print(f"wrote {ranks_csv}")


if __name__ == "__main__":
    main()
