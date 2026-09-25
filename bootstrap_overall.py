#!/usr/bin/env python3
"""User-level bootstrap over Sens, Qual@K, Align and Overall.

With m users, c = 4 context levels and decay g:

    r(theta, l)  = (1/m) * sum_u  sum_t g^(t-1) 1[B>=lam] 1[explore]
                                / sum_t g^(t-1) 1[B>=lam]          (Eq. 1)
    Sens(theta)  = max_l r(theta, l) - min_l r(theta, l)

    e(theta, l)  = (1/m) * sum_u  sum_t g^(t-1) 1[top-K] 1[explore]
                                / sum_t g^(t-1) 1[explore]
    Qual@K(theta)= (1/c) * sum_l e(theta, l)                       (Eq. 2)

    Align(theta) = 1 - sqrt((1-rho_VA)^2 + rho_EXP^2)/sqrt(5)
    Overall      = (Sens + Qual@K + Align) / 3

All four lie in [0, 1]. Users with a zero denominator at a level are left out
of that level's mean rather than scored 0. Users are averaged before Sens and
Align are taken across levels, so every replicate recomputes the metrics from
resampled per-user terms, never from per-user metric values.

Each replicate draws m users with replacement, keeping all their context
levels, and applies the same draw to every model so comparisons are paired.
For each model pair, the two-sided bootstrap p of the Overall difference is
corrected by Holm (default) or Benjamini-Hochberg; a win needs the corrected p
below alpha and a 95% interval that excludes 0. Ranks are recomputed in every
replicate.

Point estimates come from the trajectory exports, so they can differ slightly
from model_ranks.csv; treat that file as the point estimates and this one as
their uncertainty.

Outputs:
    bootstrap_overall_<vis>_<cost>_g<gamma>[_k10].csv  one row per model
    bootstrap_pairs_<vis>_<cost>_g<gamma>.csv          one row per pair
    bootstrap_draws_<vis>_<cost>_g<gamma>.csv          one row per replicate

Usage:
    python bootstrap_overall.py
    python bootstrap_overall.py --n-boot 10000 --correction fdr_bh
    python bootstrap_overall.py --gamma 0.5 --cost -2
    python bootstrap_overall.py --k 10 --correction fdr_bh
"""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
TRAJ = HERE.parent / "trajectories"

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
    tokens = str(trajectory).split()
    weight = gamma ** np.arange(len(tokens))
    explore = np.fromiter((t not in ("R", "R!") for t in tokens), bool,
                          len(tokens))
    affordable = np.fromiter((t != "R!" for t in tokens), bool, len(tokens))
    return weight[explore & affordable].sum(), weight[affordable].sum()


def load_model(label: str, visibility: str, cost: int, gamma: float,
               strict: bool, k: int = 5) -> dict:
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
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(den > 0, num / den, np.nan)
    empty = np.isnan(ratio).all(axis=0)
    out = np.full(ratio.shape[1], np.nan)
    if (~empty).any():
        out[~empty] = np.nanmean(ratio[:, ~empty], axis=0)
    return out


def self_test() -> int:
    checks, failed = [], 0

    num = np.array([[1.0], [0.5], [0.0], [0.0]])
    den = np.array([[1.0], [1.0], [0.0], [0.0]])
    checks.append(("skips zero-denominator users",
                   float(level_rates(num, den)[0]), 0.75))

    empty = level_rates(np.zeros((3, 1)), np.zeros((3, 1)))[0]
    checks.append(("empty level is NaN", float(np.isnan(empty)), 1.0))

    ex_num = np.array([[1.0, 0.5, 0.75, 0.0], [0.0, 0.0, 0.0, 0.0]])
    ex_den = np.array([[1.0, 1.0, 1.00, 0.0], [0.0, 0.0, 0.0, 0.0]])
    va = np.ones((2, 4))
    checks.append(("Qual@K skips an empty level",
                   metrics(va, va, ex_num, ex_den)[1], 0.75))

    checks.append(("Eq. 1 excludes forced exploits",
                   va_terms("R! 1 R", 1.0)[1], 2.0))
    checks.append(("Eq. 1 numerator counts explores",
                   va_terms("R! 1 R", 1.0)[0], 1.0))

    for name, got, want in checks:
        ok = abs(got - want) < 1e-12
        failed += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:<34} "
              f"got {got:.6f}  want {want:.6f}")
    print(f"\n{len(checks) - failed} of {len(checks)} checks passed")
    return 1 if failed else 0


def metrics(va_num, va_den, ex_num, ex_den) -> tuple[float, float, float, float]:
    va_rate = level_rates(va_num, va_den)
    ex_rate = level_rates(ex_num, ex_den)

    sens = float(np.nanmax(va_rate) - np.nanmin(va_rate))

    qual = float(np.nanmean(ex_rate))

    rho_va = -float(np.nan_to_num(corr_rows(va_rate)[0], nan=0.0))
    rho_ex = -float(np.nan_to_num(corr_rows(ex_rate)[0], nan=0.0))
    align = 1.0 - math.hypot(IDEAL[0] - rho_va, IDEAL[1] - rho_ex) / MAX_DIST

    return sens, qual, align, (sens + qual + align) / 3.0


def point_metrics(model) -> tuple[float, float, float, float]:
    return metrics(model["va_num"], model["va_den"],
                   model["ex_num"], model["ex_den"])


def run_bootstrap(models: list[dict], n_boot: int, rng) -> np.ndarray:
    n_users = len(models[0]["users"])
    out = np.empty((n_boot, len(models), 4))
    for r in range(n_boot):
        take = rng.integers(0, n_users, n_users)
        for j, m in enumerate(models):
            out[r, j] = metrics(m["va_num"][take], m["va_den"][take],
                                m["ex_num"][take], m["ex_den"][take])
    return out


def rank_stats(overall: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-overall, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(overall.shape[0])[:, None]
    ranks[rows, order] = np.arange(overall.shape[1])[None, :]
    ranks = ranks + 1
    return (np.median(ranks, axis=0),
            np.percentile(ranks, 2.5, axis=0),
            np.percentile(ranks, 97.5, axis=0))


def pairwise(models, draws, alpha, correction):
    overall = draws[:, :, 3]
    n_boot, n_models = overall.shape
    rows = []
    for i, j in itertools.combinations(range(n_models), 2):
        delta = overall[:, i] - overall[:, j]
        lo, hi = np.percentile(delta, [2.5, 97.5])
        tail = min((delta <= 0).sum(), (delta >= 0).sum())
        p_raw = max(2.0 * tail / n_boot, 1.0 / n_boot)
        rows.append({
            "model_a": models[i]["label"], "model_b": models[j]["label"],
            "delta_mean": float(delta.mean()),
            "delta_ci_low": float(lo), "delta_ci_high": float(hi),
            "p_raw": p_raw,
        })

    frame = pd.DataFrame(rows)
    reject, p_adj, _, _ = multipletests(frame.p_raw.to_numpy(), alpha=alpha,
                                        method=correction)
    frame["p_adj"] = p_adj
    frame["significant"] = (reject
                            & ((frame.delta_ci_low > 0)
                               | (frame.delta_ci_high < 0)).to_numpy())
    frame["winner"] = np.where(
        ~frame.significant, "similar",
        np.where(frame.delta_mean > 0, frame.model_a, frame.model_b))
    return frame


def win_loss(frame, labels) -> tuple[dict, dict]:
    wins = {label: 0 for label in labels}
    losses = {label: 0 for label in labels}
    for _, r in frame[frame.significant].iterrows():
        better, worse = ((r.model_a, r.model_b) if r.delta_mean > 0
                         else (r.model_b, r.model_a))
        wins[better] += 1
        losses[worse] += 1
    return wins, losses


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--visibility", default="seen", choices=["seen", "unseen"])
    p.add_argument("--cost", type=int, default=-5,
                   help="explore cost: -5 expensive, -2 cheap")
    p.add_argument("--k", type=int, default=5, choices=[5, 10],
                   help="top-K for Qual@K (default 5; 10 adds _k10 to the "
                        "output names)")
    p.add_argument("--n-boot", type=int, default=5000,
                   help="bootstrap replicates (1000-10000; 5000 is plenty)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--correction", default="holm",
                   choices=["holm", "fdr_bh", "bonferroni", "holm-sidak",
                            "fdr_by"],
                   help="multipletests method (default: holm)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--include-excluded", action="store_true")
    p.add_argument("--outdir", default=HERE, type=Path)
    p.add_argument("--self-test", action="store_true",
                   help="check the zero-denominator rule and exit")
    args = p.parse_args()

    if args.self_test:
        return self_test()

    labels = [l for l in TRAJECTORIES
              if args.include_excluded or l not in EXCLUDE]
    models = []
    for label in labels:
        models.append(load_model(label, args.visibility, args.cost,
                                 args.gamma,
                                 strict=not args.include_excluded,
                                 k=args.k))

    reference = models[0]["users"]
    for m in models[1:]:
        if not np.array_equal(m["users"], reference):
            raise SystemExit(
                f"{m['label']} does not share the user set of "
                f"{models[0]['label']}, so the comparison cannot be paired")

    cost_name = {-5: "expensive", -2: "cheap"}.get(args.cost, str(args.cost))
    print(f"{args.visibility} / {cost_name} / gamma = {args.gamma:g} / "
          f"K = {args.k}   "
          f"({len(models)} models, {len(reference)} users, "
          f"{args.n_boot} bootstrap draws)")

    rng = np.random.default_rng(args.seed)
    draws = run_bootstrap(models, args.n_boot, rng)
    median_rank, rank_lo, rank_hi = rank_stats(draws[:, :, 3])
    pairs = pairwise(models, draws, args.alpha, args.correction)
    wins, losses = win_loss(pairs, [m["label"] for m in models])

    rows = []
    for j, m in enumerate(models):
        sens, qual, align, overall = point_metrics(m)
        row = {
            "model": m["label"],
            "sens": sens, "qual": qual, "align": align, "overall": overall,
        }
        for k, name in enumerate(METRICS):
            lo, hi = np.percentile(draws[:, j, k], [2.5, 97.5])
            row[f"{name}_ci_low"] = float(lo)
            row[f"{name}_ci_high"] = float(hi)
        row.update({
            "overall_boot_median": float(np.median(draws[:, j, 3])),
            "median_rank": float(median_rank[j]),
            "rank_ci_low": float(rank_lo[j]), "rank_ci_high": float(rank_hi[j]),
            "sig_wins": wins[m["label"]], "sig_losses": losses[m["label"]],
        })
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(
        ["median_rank", "overall"], ascending=[True, False]).reset_index(drop=True)

    tag = (f"{args.visibility}_{cost_name}"
           f"_g{args.gamma:g}".replace(".", "")
           + ("" if args.k == 5 else f"_k{args.k}"))
    args.outdir.mkdir(parents=True, exist_ok=True)
    main_csv = args.outdir / f"bootstrap_overall_{tag}.csv"
    pair_csv = args.outdir / f"bootstrap_pairs_{tag}.csv"
    draw_csv = args.outdir / f"bootstrap_draws_{tag}.csv"
    table.to_csv(main_csv, index=False)
    pairs.to_csv(pair_csv, index=False)

    draws_frame = pd.DataFrame(
        draws.reshape(args.n_boot, -1),
        columns=[f"{m['label']}|{name}" for m in models for name in METRICS])
    draws_frame.insert(0, "draw", np.arange(1, args.n_boot + 1))
    draws_frame.to_csv(draw_csv, index=False, float_format="%.6f")

    width = max(len(r) for r in table.model)
    print(f"\n  {'model':<{width}}  {'Sens':>14} {f'Qual@{args.k}':>14} {'Align':>14} "
          f"{'Overall':>14} {'MedRank':>8} {'RankCI':>9} {'W':>3} {'L':>3}")
    for _, r in table.iterrows():
        cells = "".join(
            f" {r[name]:.3f}[{r[f'{name}_ci_low']:.2f},{r[f'{name}_ci_high']:.2f}]"
            for name in METRICS)
        rank_ci = f"[{r['rank_ci_low']:.0f},{r['rank_ci_high']:.0f}]"
        print(f"  {r['model']:<{width}} {cells} "
              f"{r['median_rank']:>8.1f} {rank_ci:>9} "
              f"{r['sig_wins']:>3} {r['sig_losses']:>3}")

    n_sig = int(pairs.significant.sum())
    print(f"\n{n_sig} of {len(pairs)} pairs significant at alpha = "
          f"{args.alpha} after {args.correction} correction")
    print(f"wrote {main_csv}")
    print(f"wrote {pair_csv}")
    print(f"wrote {draw_csv}  ({args.n_boot} draws x {len(models)} models "
          f"x {len(METRICS)} metrics)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
