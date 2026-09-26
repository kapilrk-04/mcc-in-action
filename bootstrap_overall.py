#!/usr/bin/env python3
"""Paired user-level bootstrap of Sens, Qual@K, Align and Overall across ablations.

Cells: the baseline (seen, expensive, gamma 0.9, K = 5) and one ablated cell
per factor in FACTORS: unseen, cheap, gamma 0.7 / 0.5 / 0.3, and K = 10.
Every cell and model is resampled with the same user draws, so all
comparisons are paired.

Per cell: the four metrics with 95% CIs, the point rank by Overall, the
median bootstrap rank with its 95% interval, and significant pairwise wins.

Per ablation, against the baseline: each model's change in Overall and in
rank, with 95% CIs and corrected p-values (a change is significant when the
corrected p < alpha and the CI excludes 0), plus Kendall tau and Spearman rho
between the two rankings and a noise floor from two independent resamples of
the baseline.

Input: one CSV per model from export_trajectories.py, named as in
TRAJECTORIES, in trajectories/ beside this file or in --traj-dir. Every model
and cell must cover the same users, with one run per (user, context level).

Outputs, under --outdir (default bootstrap_with_rank/):
    bootstrap_overall_<tag>.csv   one row per model per cell
    bootstrap_pairs_<tag>.csv     pairwise tests within each cell
    bootstrap_draws_<tag>.csv     every replicate's metrics
    ablation_models.csv           per-model changes for every ablation
    ablation_summary.csv          tau, rho, noise floor and counts
    ablation_rank_tables.tex      one LaTeX table per cell, plus a summary

Usage:
    python bootstrap_overall.py
    python bootstrap_overall.py --n-boot 2000 --outdir quick --traj-dir path/to/trajectories
"""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def multipletests(pvals, alpha=0.05, method="fdr_bh"):
    """Multiplicity correction, matching statsmodels' multipletests.

    Supports "fdr_bh" (Benjamini-Hochberg, step-up) and "holm" (step-down).
    Returns (reject, p_adjusted, None, None) so call sites read as before.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    if method == "fdr_bh":
        scaled = ranked * m / np.arange(1, m + 1)
        adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    elif method == "holm":
        scaled = ranked * (m - np.arange(m))
        adjusted = np.maximum.accumulate(scaled)
    else:
        raise ValueError(f"unsupported correction: {method}")
    adjusted = np.minimum(adjusted, 1.0)
    out = np.empty(m)
    out[order] = adjusted
    return out <= alpha, out, None, None


# ----------------------------------------------------------------------
# Metrics and bootstrap
# ----------------------------------------------------------------------

TRAJ = HERE / "trajectories"

LEVELS = ["none", "demographics_only", "context_only", "demographics_context"]

LEVEL_INDEX = np.arange(1.0, len(LEVELS) + 1)

IDEAL = (1.0, 0.0)

MAX_DIST = math.sqrt(5.0)

METRICS = ("sens", "qual", "align", "overall")

TRAJECTORIES = {
    "Llama-3.1-8B": "llama8b_trajectories.csv",
    "Llama-3.1-70B": "llama-70b-trajectories.csv",
    "Aya-Expanse-8B": "aya8b_trajectories.csv",
    "Aya-Expanse-32B": "aya-32B-trajectories.csv",
    "Gemma-2-2B": "gemma_2b-trajectories.csv",
    "Gemma-2-9B": "gemma9b_trajectories.csv",
    "Qwen3.5-9B": "qwen_35_9b-trajectories.csv",
    "Qwen3.5-27B": "qwen_35_27b-trajectories.csv",
    "Gemini-2.5-Flash": "gemini-2.5-flash-trajectories.csv",
    "Gemini-2.5-Pro": "gemini-2.5-pro-trajectories.csv",
    "DeepSeek-R1-Llama-8B": "deepseek_r1_llama8b_trajectories.csv",
}

EXCLUDE: list[str] = []

def corr_rows(values: np.ndarray) -> np.ndarray:
    """Pearson r against the level index, along the last axis.

    A NaN level -- one no resampled user explored -- would otherwise make the
    whole correlation NaN, so it is dropped and the remaining levels are
    correlated against their own positions. Fewer than two levels leaves
    nothing to correlate, and the result is NaN.
    """
    values = np.atleast_2d(values)
    out = np.full(values.shape[0], np.nan)
    for i, row in enumerate(values):
        ok = ~np.isnan(row)
        if ok.sum() < 2:
            continue
        y = row[ok]
        x = LEVEL_INDEX[ok] - LEVEL_INDEX[ok].mean()
        y = y - y.mean()
        denom = np.sqrt((y * y).sum() * (x * x).sum())
        if denom > 0:
            out[i] = (x * y).sum() / denom
    return out

def va_terms(trajectory: str, gamma: float) -> tuple[float, float]:
    """Eq. 1's numerator and denominator from one trajectory string.

    Tokens: R an exploit, R! an exploit the budget forced, anything else an
    explore slot's utility. Eq. 1 sums over *affordable* attempts, so R! is
    excluded from both sums -- that is what the marker is for.
    """
    tokens = str(trajectory).split()
    weight = gamma ** np.arange(len(tokens))
    explore = np.fromiter((t not in ("R", "R!") for t in tokens), bool,
                          len(tokens))
    affordable = np.fromiter((t != "R!" for t in tokens), bool, len(tokens))
    return weight[explore & affordable].sum(), weight[affordable].sum()

def load_model(label: str, visibility: str, cost: int, gamma: float,
               strict: bool, k: int = 5) -> dict:
    """Per-user, per-level Eq. 1 and Eq. 2 terms for one model, by user.

    The exported num_exp<K>_g<g>/den_g<g> are already Eq. 2's terms: the
    evaluator's eq2_terms skips every exploit, so den is sum over explore
    turns of gamma^(t-1) and num weights each by whether the question was in
    the live top-K. Eq. 1's terms are rebuilt from the trajectory string,
    which is the only place the R! marker survives. K only enters Eq. 2: the
    explore/exploit sequence, and so Eq. 1 and Sens, is the same for any K.

    There is exactly one run per (user, level) cell, so a cell's ratio is
    both the per-run and the per-user ratio -- no within-user pooling choice
    arises, and Eq. 1/Eq. 2's (1/m) sum is just the column mean in
    level_rates().
    """
    path = TRAJ / TRAJECTORIES[label]
    if not path.is_file():
        raise SystemExit(f"missing trajectory file: {path}")
    key = f"g{gamma:g}"
    frame = pd.read_csv(path, usecols=[
        "user_id", "condition", "explore_cost", "movie_visibility",
        "trajectory_exp5", f"num_exp{k}_{key}", f"den_{key}"])
    frame = frame[(frame.movie_visibility == visibility)
                  & (frame.explore_cost == cost)]

    marked = frame.trajectory_exp5.str.contains("R!", regex=False).any()
    if strict and not marked:
        raise SystemExit(
            f"{label}: trajectories carry no R! marker, so Eq. 1's "
            f"denominator cannot be rebuilt. Re-export with "
            f"export_trajectories.py, or leave {label} in EXCLUDE.")

    if frame.duplicated(["user_id", "condition"]).any():
        raise SystemExit(
            f"{label}: more than one run per (user, level); level_rates "
            f"assumes one, so the per-user ratio would be taken over a "
            f"single arbitrary run")

    terms = np.array([va_terms(t, gamma) for t in frame.trajectory_exp5])
    frame = frame.assign(va_num=terms[:, 0], va_den=terms[:, 1])

    # (users, levels) with a stable user order, so models stay paired.
    def grid(column):
        wide = frame.pivot(index="user_id", columns="condition", values=column)
        return wide[LEVELS].sort_index()

    va_num = grid("va_num")
    return {
        "label": label,
        "users": va_num.index.to_numpy(),
        "va_num": va_num.to_numpy(float),
        "va_den": grid("va_den").to_numpy(float),
        "ex_num": grid(f"num_exp{k}_{key}").to_numpy(float),
        "ex_den": grid(f"den_{key}").to_numpy(float),
        "marked": bool(marked),
    }

def level_rates(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """The (1/m) sum over users inside Eq. 1 and Eq. 2, per level.

        r_l = (1/m) * sum_u ( sum_t ... / sum_t ... )

    The mean of per-user ratios, not the ratio of the sums: every user counts
    once regardless of how many turns they took. A user whose denominator is
    zero -- no affordable turn under Eq. 1, no explore turn under Eq. 2, so
    the ratio says nothing about them -- is left out of that level's mean
    rather than counted as 0, which would conflate "explored badly" with
    "never explored". A level no resampled user explored at all is NaN, for
    the caller to skip.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(den > 0, num / den, np.nan)
    empty = np.isnan(ratio).all(axis=0)
    out = np.full(ratio.shape[1], np.nan)
    if (~empty).any():
        out[~empty] = np.nanmean(ratio[:, ~empty], axis=0)
    return out

def metrics(va_num, va_den, ex_num, ex_den) -> tuple[float, float, float, float]:
    """(Sens, Qual, Align, Overall) from one set of term stacks.

    Both rates are per-level means over users first (Eq. 1, Eq. 2), and the
    across-level reductions come after, never the other way round: averaging
    per-user ranges or correlations would give a different, biased quantity.
    """
    va_rate = level_rates(va_num, va_den)        # Eq. 1: exploration rate r_l
    ex_rate = level_rates(ex_num, ex_den)        # Eq. 2: question quality e_l

    # Sens: the spread of the exploration rate across levels. Unsigned -- it
    # says how much behaviour moves, not whether it moves the right way,
    # which is what Align is for.
    sens = float(np.nanmax(va_rate) - np.nanmin(va_rate))

    # Qual@K (Eq. 2): the plain mean of the per-level e_l, each level counting
    # equally. Not the ratio pooled over all levels at once, which would
    # weight a level by how often it was explored; the two differ by up to
    # 0.02 here. A resampled draw can leave a level with no explore turn at
    # all, so the mean skips an undefined level rather than propagating NaN.
    qual = float(np.nanmean(ex_rate))

    # Align: closeness to (rho_Rate, rho_EXP) = (1, 0). Both correlations are
    # negated, so rho_Rate = +1 means exploration
    # falls as context rises. rho_EXP's ideal is 0, where the sign does not
    # matter -- but it does everywhere else, so it is flipped too rather than
    # left as the raw r.
    # An undefined correlation means the rates showed no variation to
    # correlate, which is "no trend" -- rho = 0 -- not a missing value.
    rho_rate = -float(np.nan_to_num(corr_rows(va_rate)[0], nan=0.0))
    rho_ex = -float(np.nan_to_num(corr_rows(ex_rate)[0], nan=0.0))
    align = 1.0 - math.hypot(IDEAL[0] - rho_rate, IDEAL[1] - rho_ex) / MAX_DIST

    return sens, qual, align, (sens + qual + align) / 3.0

def point_metrics(model) -> tuple[float, float, float, float]:
    return metrics(model["va_num"], model["va_den"],
                   model["ex_num"], model["ex_den"])

def run_bootstrap(models: list[dict], n_boot: int, rng) -> np.ndarray:
    """(n_boot, n_models, 4) array of replicate metrics.

    One draw of user indices per replicate, shared by every model: the models
    are paired on users, and resampling them independently would break that
    and inflate every pairwise interval. Each replicate recomputes Eq. 1 and
    Eq. 2 from the resampled users' terms, so Sens's range and Align's
    correlation are re-derived across levels inside the draw.
    """
    n_users = len(models[0]["users"])
    out = np.empty((n_boot, len(models), 4))
    for r in range(n_boot):
        take = rng.integers(0, n_users, n_users)
        for j, m in enumerate(models):
            out[r, j] = metrics(m["va_num"][take], m["va_den"][take],
                                m["ex_num"][take], m["ex_den"][take])
    return out

def rank_stats(overall: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median and 95% interval of each model's rank, ranks recomputed per draw.

    Rank 1 is the highest Overall. Ranking inside the replicate is the point:
    it carries the variation of the ordering, which a rank taken once from the
    point estimates cannot show.
    """
    # argsort twice turns scores into 0-based ranks; negate for descending.
    order = np.argsort(-overall, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(overall.shape[0])[:, None]
    ranks[rows, order] = np.arange(overall.shape[1])[None, :]
    ranks = ranks + 1
    return (np.median(ranks, axis=0),
            np.percentile(ranks, 2.5, axis=0),
            np.percentile(ranks, 97.5, axis=0))

def pairwise(models, draws, alpha, correction):
    """One row per model pair, with the corrected p and the CI on the gap."""
    overall = draws[:, :, 3]
    n_boot, n_models = overall.shape
    rows = []
    for i, j in itertools.combinations(range(n_models), 2):
        delta = overall[:, i] - overall[:, j]
        lo, hi = np.percentile(delta, [2.5, 97.5])
        # Two-sided bootstrap p, floored at 1/R: with R draws nothing smaller
        # is resolvable, and quoting 0 would overstate the evidence.
        tail = min((delta <= 0).sum(), (delta >= 0).sum())
        p_raw = max(2.0 * tail / n_boot, 1.0 / n_boot)
        rows.append({
            "model_a": models[i]["label"], "model_b": models[j]["label"],
            "delta_mean": float(delta.mean()),
            "delta_ci_low": float(lo), "delta_ci_high": float(hi),
            "p_raw": p_raw,
        })

    frame = pd.DataFrame(rows)
    # multipletests does the multiplicity correction; `reject` is its own
    # decision at this alpha, which for a step-down or step-up procedure is
    # not always the same as comparing the adjusted p to alpha.
    reject, p_adj, _, _ = multipletests(frame.p_raw.to_numpy(), alpha=alpha,
                                        method=correction)
    frame["p_adj"] = p_adj
    # Both criteria must agree: the correction rejects and the interval
    # excludes 0. The CI is uncorrected, so on its own it would be liberal.
    frame["significant"] = (reject
                            & ((frame.delta_ci_low > 0)
                               | (frame.delta_ci_high < 0)).to_numpy())
    # Which model the pair favours, or "similar" when the test does not
    # separate them -- a gap this test cannot resolve is not a win.
    frame["winner"] = np.where(
        ~frame.significant, "similar",
        np.where(frame.delta_mean > 0, frame.model_a, frame.model_b))
    return frame

def win_loss(frame, labels) -> tuple[dict, dict]:
    """Significant wins and losses per model, from the pair table."""
    wins = {label: 0 for label in labels}
    losses = {label: 0 for label in labels}
    for _, r in frame[frame.significant].iterrows():
        better, worse = ((r.model_a, r.model_b) if r.delta_mean > 0
                         else (r.model_b, r.model_a))
        wins[better] += 1
        losses[worse] += 1
    return wins, losses


# ----------------------------------------------------------------------
# Ablation factors
# ----------------------------------------------------------------------

FACTORS = {
    "cost": [("expensive vs cheap", ("seen", -5, 0.9, 5),
              ("seen", -2, 0.9, 5))],
    "visibility": [("seen vs unseen", ("seen", -5, 0.9, 5),
                    ("unseen", -5, 0.9, 5))],
    "gamma": [(f"g0.9 vs g{g}", ("seen", -5, 0.9, 5), ("seen", -5, g, 5))
              for g in (0.7, 0.5, 0.3)],
    # K only changes Qual@K and, through rho_EXP, Align; Sens is identical in
    # the two cells by construction.
    "k": [("K=5 vs K=10", ("seen", -5, 0.9, 5), ("seen", -5, 0.9, 10))],
}


# ----------------------------------------------------------------------
# LaTeX tables
# ----------------------------------------------------------------------

def escape(text: str) -> str:
    """LaTeX-safe label. Only these characters occur in these names."""
    for char in ("\\", "&", "_", "#", "%"):
        text = text.replace(char, "\\" + char)
    return text

def cell_label(visibility, cost, gamma, k=5):
    """Human-readable cell name, e.g. 'seen, expensive, gamma=0.9'.

    K is named only when it is not the default 5, so the existing captions
    and labels are unchanged.
    """
    return (f"{visibility}, {COST_NAME.get(cost, cost)}, "
            rf"$\gamma={gamma:g}$" + ("" if k == 5 else f", $K={k}$"))

def cell_tag(visibility, cost, gamma, k=5):
    return (f"{visibility}-{COST_NAME.get(cost, cost)}-g{gamma:g}".replace(
        ".", "") + ("" if k == 5 else f"-k{k}"))

def file_tag(visibility, cost, gamma, k=5):
    """The suffix of each cell's output files."""
    return (f"{visibility}_{COST_NAME[cost]}_g{gamma:g}".replace(".", "")
            + ("" if k == 5 else f"_k{k}"))

CORRECTION_NAMES = {"fdr_bh": "Benjamini--Hochberg", "holm": "Holm"}

def point_ranks(table):
    """Rank 1 = highest Overall, from the point estimates."""
    return table["overall"].rank(ascending=False, method="min").astype(int)

def rank_cell(rank, lo, hi):
    return (rf"{rank} {{\scriptsize $[{lo:.0f},\,{hi:.0f}]$}}")

def delta_rank_cell(test):
    d = int(test["delta_rank"])
    text = f"{d:+d}" if d else "0"
    ci = rf"{{\scriptsize $[{test['ci_low']:+.0f},\,{test['ci_high']:+.0f}]$}}"
    ci = ci.replace("+0,", "0,").replace("+0]", "0]").replace("[-0", "[0")
    mark = rf"$\mathbf{{{text}}}^{{*}}$" if test["moved"] else f"${text}$"
    return f"{mark} {ci}"

RANK_CAPTION = (
    r" Rank is the model's position by Overall in this cell (1 = best), "
    r"with its 95\% bootstrap interval over users.")

DELTA_RANK_CAPTION = (
    r" $\Delta$Rank is this cell's rank minus the baseline rank (positive: "
    r"the model ranks lower here), with its 95\% interval from a paired "
    r"bootstrap in which each resample of users is applied to both cells "
    r"and every model is re-ranked within each; "
    r"$\mathbf{\Delta}^{*}$ marks a rank change with Benjamini--Hochberg "
    r"corrected $p<0.05$ whose interval excludes $0$.")

def fmt_rank(value):
    """A median rank: whole numbers bare, a tie between two ranks as x.5."""
    return f"{value:.0f}" if float(value).is_integer() else f"{value:.1f}"

def render_change(table, metric_columns, best, visibility, cost, gamma, k,
                  base, change, rank_change=None):
    """An ablated cell: its scores, then each metric's change from the
    baseline cell and the paired test of the change in Overall.

    Delta = ablated - baseline, per model. The Sens/Qual/Align/Overall
    deltas are differences of the two cells' point estimates. The Overall
    change carries its paired-bootstrap 95% interval (one resample of users
    applied to both cells); it is significant when its Benjamini-Hochberg p
    across the 11 models is < 0.05 and the interval
    excludes 0.
    """
    # Each metric followed by its change; then the test of the Overall change.
    heads = []
    for _, h in metric_columns:
        heads += [h, rf"$\Delta${h}"]
    with_ranks = rank_change is not None
    n_cols = 11 if with_ranks else 9
    rank_heads = r" & Rank & $\Delta$Rank" if with_ranks else ""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}l" + "c" * n_cols + "@{}}",
        r"\toprule",
        "Model & " + " & ".join(heads) + r" & 95\% CI" + rank_heads + r" \\",
        r"\midrule",
    ]
    ranks = point_ranks(table.set_index("model")) if with_ranks else None
    for _, r in table.iterrows():
        model = r["model"]
        test = change.loc[model]
        cells = []
        for column, _ in metric_columns:
            text = f"{r[column]:.3f}"
            if abs(r[column] - best[column]) < 5e-4:
                text = rf"\textbf{{{text}}}"
            cells.append(text)
            if column == "overall":
                # The Overall change is the one with a test; mark it when
                # significant.
                delta = f"{test['delta']:+.3f}"
                cells.append(rf"$\mathbf{{{delta}}}^{{*}}$"
                             if test["significant"] else f"${delta}$")
            else:
                cells.append(f"${r[column] - base.loc[model, column]:+.3f}$")
        lines.append(escape(model))
        lines.append("& " + " & ".join(cells))
        rank_text = ""
        if with_ranks:
            rt = rank_change.loc[model]
            if int(rt["rank_b"]) != ranks[model]:
                raise SystemExit(f"{cell_tag(visibility, cost, gamma, k)} "
                                 f"{model}: rank_ablation rank {rt['rank_b']} "
                                 f"!= table rank {ranks[model]} -- stale run")
            rank_text = (" & " + rank_cell(ranks[model], r["rank_ci_low"],
                                           r["rank_ci_high"])
                         + " & " + delta_rank_cell(rt))
        lines.append(rf"& $[{test['ci_low']:+.3f},\,{test['ci_high']:+.3f}]$"
                     rf"{rank_text} \\")
    lines += [r"\bottomrule", r"\end{tabular}}"]
    lines.append(
        rf"\caption{{\textbf{{Overall ranking: "
        rf"{cell_label(visibility, cost, gamma, k)}.}} "
        r"Sensitivity (Sens), Question Quality "
        rf"(Qual@{k}), and Alignment (Align) lie in $[0,1]$; Overall is their "
        r"arithmetic mean. Higher scores are better; the best score in each "
        r"column is in \textbf{bold}. Each $\Delta$ column follows its metric "
        r"and gives the change from the baseline cell "
        r"(Table~\ref{tab:ablation-seen-expensive-g09}), this cell minus the "
        r"baseline. The change in Overall is tested with a paired bootstrap "
        r"(one resample of users applied to both cells); the 95\% CI column "
        r"gives its interval. $\mathbf{\Delta}^{*}$: significant at "
        r"$\alpha=0.05$ ($p<0.05$ after Benjamini--Hochberg correction "
        r"across the 11 models, and the interval excludes $0$)."
        + (RANK_CAPTION + DELTA_RANK_CAPTION if with_ranks else "") + "}")
    lines.append(rf"\label{{tab:ablation-{cell_tag(visibility, cost, gamma, k)}}}")
    lines.append(r"\end{table*}")
    return lines

def render_cell(frame, visibility, cost, gamma, k, baseline, boot=None,
                correction="Benjamini--Hochberg", base=None, change=None,
                ranks=False, rank_change=None):
    """One ablation cell in the main results table's form.

    Model, the three metrics and Overall, the Overall 95% interval, the median
    bootstrap rank and its 95% interval, and the win rate. Every column is
    read from the cell's bootstrap_overall run, so the point estimates and
    their intervals come from one resampling -- exactly as in the main table.
    Rows are ordered by median rank, then Overall.
    """
    if boot is None:
        raise SystemExit(
            f"cell {cell_tag(visibility, cost, gamma, k)} has no "
            f"bootstrap_overall file; the main-table form needs one")
    if ranks:
        table = boot.reset_index().sort_values("overall", ascending=False)
    else:
        table = boot.reset_index().sort_values(["median_rank", "overall"],
                                               ascending=[True, False])
    metric_columns = [("sens", "Sens"), ("qual", f"Qual@{k}"),
                      ("align", "Align"), ("overall", "Overall")]
    best = {c: table[c].max() for c, _ in metric_columns}
    if not baseline:
        return render_change(table, metric_columns, best, visibility, cost,
                             gamma, k, base, change,
                             rank_change if ranks else None)

    # The baseline cell: the same layout as the ablated cells, minus the
    # change columns -- it is the reference they are measured against.
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lcccc" + ("c" if ranks else "") + "@{}}",
        r"\toprule",
        "Model & " + " & ".join(h for _, h in metric_columns)
        + (" & Rank" if ranks else "") + r" \\",
        r"\midrule",
    ]
    point = point_ranks(table.set_index("model")) if ranks else None
    for _, r in table.iterrows():
        cells = []
        for column, _ in metric_columns:
            text = f"{r[column]:.3f}"
            # Compared at the printed precision, so a tie at three decimals
            # is bolded for both rather than broken by the sixth.
            if abs(r[column] - best[column]) < 5e-4:
                text = rf"\textbf{{{text}}}"
            cells.append(text)
        if ranks:
            cells.append(rank_cell(point[r["model"]], r["rank_ci_low"],
                                   r["rank_ci_high"]))
        lines.append(f"{escape(r['model'])} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    lines.append(
        rf"\caption{{\textbf{{Overall ranking: "
        rf"{cell_label(visibility, cost, gamma, k)} (baseline cell).}} "
        r"Sensitivity (Sens), Question Quality "
        rf"(Qual@{k}), and Alignment (Align) lie in $[0,1]$; Overall is their "
        r"arithmetic mean. Higher scores are better; the best score in each "
        r"column is in \textbf{bold}. This is the reference setting of the "
        r"ablation: every other ablation table reports its change from these "
        r"scores." + (RANK_CAPTION if ranks else "") + "}")
    lines.append(rf"\label{{tab:ablation-{cell_tag(visibility, cost, gamma, k)}}}")
    lines.append(r"\end{table}")
    return lines


# ----------------------------------------------------------------------
# Ablation comparisons
# ----------------------------------------------------------------------

BASELINE = ("seen", -5, 0.9, 5)
COST_NAME = {-5: "expensive", -2: "cheap"}


def cells_to_run():
    cells = [BASELINE]
    comparisons = []
    for factor, items in FACTORS.items():
        for name, cell_a, cell_b in items:
            if tuple(cell_a) != BASELINE:
                raise SystemExit(f"{name}: expected the baseline as cell A")
            comparisons.append((factor, name, tuple(cell_b)))
            if tuple(cell_b) not in cells:
                cells.append(tuple(cell_b))
    return cells, comparisons


def load_cell(cell, labels):
    visibility, cost, gamma, k = cell
    return [load_model(l, visibility, cost, gamma, strict=True, k=k)
            for l in labels]


def ranks_of(overall: np.ndarray) -> np.ndarray:
    return (-overall).argsort(axis=-1).argsort(axis=-1) + 1


def kendall_tau(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(a.shape[-1], 1)
    s = np.sign(a[..., i] - a[..., j]) * np.sign(b[..., i] - b[..., j])
    return s.mean(axis=-1)


def spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ra = ranks_of(a).astype(float)
    rb = ranks_of(b).astype(float)
    ra -= ra.mean(axis=-1, keepdims=True)
    rb -= rb.mean(axis=-1, keepdims=True)
    return (ra * rb).sum(-1) / np.sqrt((ra * ra).sum(-1) * (rb * rb).sum(-1))


def paired_test(delta: np.ndarray, point: np.ndarray, alpha, correction):
    n_boot = delta.shape[0]
    lo, hi = np.percentile(delta, [2.5, 97.5], axis=0)
    tail = np.minimum((delta <= 0).sum(0), (delta >= 0).sum(0))
    p_raw = np.clip(2.0 * tail / n_boot, 1.0 / n_boot, 1.0)
    reject, p_adj, _, _ = multipletests(p_raw, alpha=alpha, method=correction)
    return pd.DataFrame({
        "delta": point, "ci_low": lo, "ci_high": hi,
        "p_raw": p_raw, "p_adj": p_adj,
        "significant": reject & ((lo > 0) | (hi < 0)),
    })


def cell_table(models, draws, alpha, correction):
    median_rank, rank_lo, rank_hi = rank_stats(draws[:, :, 3])
    pairs = pairwise(models, draws, alpha, correction)
    wins, losses = win_loss(pairs, [m["label"] for m in models])
    points = np.array([point_metrics(m) for m in models])
    rows = []
    for j, m in enumerate(models):
        row = {"model": m["label"]}
        row.update(dict(zip(METRICS, points[j])))
        for k, name in enumerate(METRICS):
            lo, hi = np.percentile(draws[:, j, k], [2.5, 97.5])
            row[f"{name}_ci_low"], row[f"{name}_ci_high"] = float(lo), float(hi)
        row.update({
            "overall_boot_median": float(np.median(draws[:, j, 3])),
            "rank": int(ranks_of(points[:, 3])[j]),
            "median_rank": float(median_rank[j]),
            "rank_ci_low": float(rank_lo[j]), "rank_ci_high": float(rank_hi[j]),
            "sig_wins": wins[m["label"]], "sig_losses": losses[m["label"]],
        })
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("overall", ascending=False)
    return table.reset_index(drop=True), pairs, points


def compare(name, labels, base, cell, alpha, correction):
    base_draws, base_points = base["draws"][:, :, 3], base["points"][:, 3]
    draws, points = cell["draws"][:, :, 3], cell["points"][:, 3]

    overall = paired_test(draws - base_draws, points - base_points,
                          alpha, correction)
    rank_draws = ranks_of(draws) - ranks_of(base_draws)
    rank_point = ranks_of(points) - ranks_of(base_points)
    rank = paired_test(rank_draws, rank_point, alpha, correction)

    per_model = pd.DataFrame({"comparison": name, "model": labels,
                              "rank_baseline": ranks_of(base_points),
                              "rank_b": ranks_of(points)})
    per_model = pd.concat([
        per_model,
        overall.add_prefix("overall_"),
        rank.rename(columns={"delta": "delta_rank"}).add_prefix("rank_")
            .rename(columns={"rank_delta_rank": "delta_rank"}),
    ], axis=1)

    tau = kendall_tau(draws, base_draws)
    rho = spearman(draws, base_draws)
    noise = kendall_tau(base_draws, np.roll(base_draws, -1, axis=0))
    summary = {
        "comparison": name,
        "kendall_tau": float(kendall_tau(points, base_points)),
        "tau_ci_low": float(np.percentile(tau, 2.5)),
        "tau_ci_high": float(np.percentile(tau, 97.5)),
        "noise_tau_median": float(np.median(noise)),
        "noise_tau_ci_low": float(np.percentile(noise, 2.5)),
        "noise_tau_ci_high": float(np.percentile(noise, 97.5)),
        "p_tau_not_below_noise": float(np.mean(tau[:, None] >= noise[None, :])),
        "spearman_rho": float(spearman(points, base_points)),
        "rho_ci_low": float(np.percentile(rho, 2.5)),
        "rho_ci_high": float(np.percentile(rho, 97.5)),
        "n_overall_changed": int(overall.significant.sum()),
        "n_rank_moved": int(rank.significant.sum()),
        "mean_abs_delta_overall": float(np.abs(overall.delta).mean()),
        "mean_abs_delta_rank": float(np.abs(rank.delta).mean()),
    }
    return per_model, summary


def for_render(per_model):
    m = per_model.set_index("model")
    change = pd.DataFrame({
        "delta": m.overall_delta, "ci_low": m.overall_ci_low,
        "ci_high": m.overall_ci_high, "p_adj": m.overall_p_adj,
        "significant": m.overall_significant})
    rank_change = pd.DataFrame({
        "rank_b": m.rank_b, "delta_rank": m.delta_rank,
        "ci_low": m.rank_ci_low, "ci_high": m.rank_ci_high,
        "moved": m.rank_significant})
    return change, rank_change


def render_summary(summary, factors, n_models, correction):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        r"Factor & Comparison & Kendall $\tau$ & $\Delta$Overall & "
        r"$\Delta$Rank \\",
        r"\midrule",
    ]
    previous = None
    for (_, r), factor in zip(summary.iterrows(), factors):
        if previous is not None and factor != previous:
            lines.append(r"\addlinespace")
        shown = ({"k": "$K$"}.get(factor, escape(factor))
                 if factor != previous else "")
        previous = factor
        tau = (rf"${r['kendall_tau']:.3f}$ {{\scriptsize "
               rf"$[{r['tau_ci_low']:.2f},\,{r['tau_ci_high']:.2f}]$}}")
        lines.append(
            f"{shown} & {escape(r['comparison'])} & {tau} & "
            f"{int(r['n_overall_changed'])}\\,/\\,{n_models} & "
            f"{int(r['n_rank_moved'])}\\,/\\,{n_models} \\\\")
    noise = summary.iloc[0]
    lines += [r"\bottomrule", r"\end{tabular}"]
    lines.append(
        r"\caption{\textbf{Ablation: does the ranking survive each factor?} "
        r"Kendall $\tau$ between the baseline and ablated cells' Overall "
        r"rankings, with its 95\% bootstrap interval over users (one set of "
        r"resamples shared by every cell and model). For reference, two "
        r"independent resamples of the baseline agree at "
        rf"$\tau={noise['noise_tau_median']:.2f}$ "
        rf"$[{noise['noise_tau_ci_low']:.2f},\,{noise['noise_tau_ci_high']:.2f}]$, "
        r"the level expected from sampling noise alone. $\Delta$Overall and "
        r"$\Delta$Rank count the models whose Overall score or rank changed "
        rf"significantly ({correction}-corrected $p<0.05$, interval "
        r"excluding $0$).}")
    lines.append(r"\label{tab:ablation-summary}")
    lines.append(r"\end{table}")
    return lines


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--correction", default="fdr_bh",
                   choices=list(CORRECTION_NAMES),
                   help="multipletests method for every test (default: fdr_bh)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default=HERE / "bootstrap_with_rank", type=Path)
    p.add_argument("--traj-dir", default=None, type=Path,
                   help="folder holding the trajectory CSVs (default: "
                        "trajectories/ beside this file)")
    args = p.parse_args()

    global TRAJ
    if args.traj_dir is not None:
        TRAJ = args.traj_dir
    if not TRAJ.is_dir():
        raise SystemExit(f"trajectory folder not found: {TRAJ}; "
                         f"pass --traj-dir")
    print(f"reading trajectories from {TRAJ}")

    labels = [l for l in TRAJECTORIES if l not in EXCLUDE]
    cells, comparisons = cells_to_run()
    args.outdir.mkdir(parents=True, exist_ok=True)

    results, reference = {}, None
    for cell in cells:
        models = load_cell(cell, labels)
        for m in models:
            if reference is None:
                reference = m["users"]
            if not np.array_equal(m["users"], reference):
                raise SystemExit(f"{cell} {m['label']}: different user set, "
                                 f"so the cells cannot be paired")
        print(f"{cell_tag(*cell)}: {len(models)} models, {len(reference)} "
              f"users, {args.n_boot} draws")
        draws = run_bootstrap(models, args.n_boot,
                              np.random.default_rng(args.seed))
        table, pairs, points = cell_table(models, draws, args.alpha,
                                          args.correction)
        results[cell] = {"draws": draws, "points": points, "table": table}

        tag = file_tag(*cell)
        table.to_csv(args.outdir / f"bootstrap_overall_{tag}.csv", index=False)
        pairs.to_csv(args.outdir / f"bootstrap_pairs_{tag}.csv", index=False)
        frame = pd.DataFrame(
            draws.reshape(args.n_boot, -1),
            columns=[f"{m['label']}|{n}" for m in models for n in METRICS])
        frame.insert(0, "draw", np.arange(1, args.n_boot + 1))
        frame.to_csv(args.outdir / f"bootstrap_draws_{tag}.csv", index=False,
                     float_format="%.6f")

    per_models, summaries, factors = [], [], []
    for factor, name, cell in comparisons:
        per_model, summary = compare(name, labels, results[BASELINE],
                                     results[cell], args.alpha,
                                     args.correction)
        per_model.insert(0, "factor", factor)
        per_model.insert(2, "cell", cell_tag(*cell))
        per_models.append(per_model)
        summaries.append({"factor": factor, "cell": cell_tag(*cell), **summary})
        factors.append(factor)
        results[cell]["per_model"] = per_model

        print(f"\n{name}: tau = {summary['kendall_tau']:+.3f} "
              f"[{summary['tau_ci_low']:+.3f}, {summary['tau_ci_high']:+.3f}]"
              f"  noise {summary['noise_tau_median']:+.3f}   "
              f"Overall changed {summary['n_overall_changed']}/{len(labels)}"
              f"   rank moved {summary['n_rank_moved']}/{len(labels)}")
        moved = per_model[per_model.rank_significant]
        for _, r in moved.sort_values("delta_rank").iterrows():
            print(f"    {r['model']:<22} rank {r['rank_baseline']:>2} -> "
                  f"{r['rank_b']:>2}  ({r['delta_rank']:+d}, "
                  f"[{r['rank_ci_low']:+.0f}, {r['rank_ci_high']:+.0f}], "
                  f"p_adj {r['rank_p_adj']:.3f})")

    models_csv = args.outdir / "ablation_models.csv"
    summary_csv = args.outdir / "ablation_summary.csv"
    pd.concat(per_models, ignore_index=True).to_csv(models_csv, index=False,
                                                    float_format="%.6f")
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(summary_csv, index=False, float_format="%.6f")

    correction = CORRECTION_NAMES[args.correction]
    base_boot = results[BASELINE]["table"].set_index("model")
    blocks = [r"% Requires \usepackage{booktabs,graphicx}."]
    for cell in cells:
        boot = results[cell]["table"].set_index("model")
        change = rank_change = None
        if cell != BASELINE:
            change, rank_change = for_render(results[cell]["per_model"])
        blocks.append("\n".join(render_cell(
            None, *cell, baseline=cell == BASELINE, boot=boot,
            correction=correction, base=base_boot, change=change,
            ranks=True, rank_change=rank_change)))
    blocks.append("\n".join(render_summary(summary_frame, factors,
                                            len(labels), correction)))
    tex = args.outdir / "ablation_rank_tables.tex"
    tex.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")

    print(f"\nwrote {len(cells)} cells of bootstrap_overall/pairs/draws CSVs "
          f"to {args.outdir}")
    for path in (models_csv, summary_csv, tex):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
