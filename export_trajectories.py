#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from trajectory_utils import VARIANTS, discover_runs, gamma_key, load_tier_map

HERE = Path(__file__).resolve().parent


def exploit_token(item) -> str:
    return "R!" if item["forced"] else "R"


def render(sequence, variant: str) -> str:
    return " ".join(
        exploit_token(item) if item["kind"] == "exploit" else f"{item['u'][variant]:g}"
        for item in sequence
    )


def render_questions(sequence) -> str:
    parts = []
    for item in sequence:
        if item["kind"] == "exploit":
            parts.append(exploit_token(item))
        elif item["kind"] == "invalid":
            parts.append("X")
        else:
            label = f"Q{item['q']}" if item["q"] is not None else "Q?"
            parts.append(label + ("*" if item["kind"] == "duplicate" else ""))
    return " ".join(parts)


def main() -> int:
    p = argparse.ArgumentParser(description="Export per-run interaction trajectories and Exp terms.")
    p.add_argument("--results-root", type=Path, default=Path("rl_explore_exploit_results"))
    p.add_argument("--question-rankings", type=Path, default=HERE / "question_rankings.json")
    p.add_argument("--model-name", help="label every run with this model name")
    p.add_argument("--visibility", choices=["seen", "unseen"])
    p.add_argument("--gammas", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    p.add_argument("--not-in-list", choices=["zero", "skip"], default="zero")
    p.add_argument("--out", type=Path, default=HERE / "trajectories.csv")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    runs, warnings = discover_runs(
        args.results_root,
        load_tier_map(args.question_rankings),
        args.gammas,
        not_in_list=args.not_in_list,
        visibility=args.visibility,
        model_name=args.model_name,
        workers=args.workers,
    )
    for warning in warnings[:5]:
        print(f"warning: {warning}", file=sys.stderr)
    if not runs:
        print("error: no eligible completed runs found", file=sys.stderr)
        return 1

    keys = [gamma_key(g) for g in args.gammas]
    fields = [
        "path", "user_id", "model", "condition", "level", "explore_cost",
        "movie_visibility", "schema",
        "trajectory_exp5", "trajectory_exp10", "questions",
        "n_slots", "n_exploits", "n_forced_exploits", "n_asks",
        "n_duplicates", "n_invalid",
    ]
    for key in keys:
        fields += [f"num_exp5_g{key}", f"num_exp10_g{key}", f"den_g{key}"]

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            row = {
                "path": str(run.path),
                "user_id": run.user_id,
                "model": run.model,
                "condition": run.condition,
                "level": run.level_label,
                "explore_cost": run.explore_cost,
                "movie_visibility": run.movie_visibility,
                "schema": run.schema,
                "trajectory_exp5": render(run.sequence, "exp5"),
                "trajectory_exp10": render(run.sequence, "exp10"),
                "questions": render_questions(run.sequence),
                "n_slots": sum(1 for item in run.sequence if item["kind"] != "exploit"),
                "n_exploits": run.counts.get("n_exploits", 0),
                "n_forced_exploits": run.counts.get("n_forced_exploits", 0),
                "n_asks": run.counts.get("n_asks", 0),
                "n_duplicates": run.counts.get("n_duplicates", 0),
                "n_invalid": run.counts.get("n_invalid", 0),
            }
            for key in keys:
                terms = run.terms[key]
                for variant in VARIANTS:
                    row[f"num_{variant}_g{key}"] = terms["num"][variant]
                row[f"den_g{key}"] = terms["den"]
            writer.writerow(row)

    print(f"wrote {args.out}  ({len(runs)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
