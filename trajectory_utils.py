from __future__ import annotations

import concurrent.futures
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm

QUESTIONS = [
    "What are your top 3 favourite movie genres?",
    "Which genres do you tend to dislike or avoid?",
    "Do you tend to enjoy older classic films, or do you gravitate more toward modern productions?",
    "Do you tend to rate movies within your favourite genres consistently highly, or does quality vary a lot even within genres you like?",
    "Overall, are you an easy-to-please viewer or a hard-to-impress one?",
    "Are you an avid, moderate, or light movie watcher?",
    "Are you part of a generation that grew up with cinema as a cultural centerpiece (pre-1970s), the blockbuster/VHS era (1970s-1990s), or the streaming/digital era (post-1990s)?",
    "How would you describe your area — urban, suburban, or rural?",
    "Would you describe your work as more intellectual/creative, practical/hands-on, or service-oriented?",
    "What is your occupation?",
    "What is your age group?",
    "What is your gender?",
    "What is your favorite color?",
    "What is your lucky number?",
    "Do you prefer tea or coffee?",
]

CONTEXT_HIERARCHY = ("none", "demographics_only", "context_only", "demographics_context")
LEVEL_LABELS = ("L0", "L1", "L2", "L3")
TIERS = ("T1", "T2", "T3")

VARIANTS = ("exp5", "exp10")
BANDS: dict[str, tuple[tuple[str, ...], ...]] = {
    "exp5": (("T1",), ("T2",), ("T3",)),
    "exp10": (("T1", "T2"), ("T3",)),
}


def gamma_key(gamma: float) -> str:
    return f"{gamma:g}"


def normalize_question(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[‒–—―−]|--", "-", text)
    text = re.sub(r"\s*-\s*", " - ", text)
    return re.sub(r"\s+", " ", text)


QUESTION_INDEX = {
    normalize_question(question): index for index, question in enumerate(QUESTIONS, start=1)
}


def canonical_index(question: Any) -> int | None:
    if isinstance(question, int) and not isinstance(question, bool):
        return question if 1 <= question <= len(QUESTIONS) else None
    if not isinstance(question, str):
        return None
    return QUESTION_INDEX.get(normalize_question(question))


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value is not an object")
    return value


def load_tier_map(path: Path) -> dict[str, dict[int, str]]:
    rankings = load_json(path)
    if set(rankings) != set(CONTEXT_HIERARCHY):
        raise ValueError(f"{path}: expected rankings for exactly {list(CONTEXT_HIERARCHY)}")
    tier_by_condition: dict[str, dict[int, str]] = {}
    for condition in CONTEXT_HIERARCHY:
        tiers = rankings[condition]
        if not isinstance(tiers, dict) or set(tiers) != set(TIERS):
            raise ValueError(f"{path}: {condition} must have exactly the keys {list(TIERS)}")
        mapping: dict[int, str] = {}
        for tier in TIERS:
            questions = tiers[tier]
            if not isinstance(questions, list) or len(questions) != 5:
                raise ValueError(f"{path}: {condition}.{tier} must list exactly 5 questions")
            for question in questions:
                index = canonical_index(question)
                if index is None:
                    raise ValueError(f"{path}: unknown question {question!r} in {condition}.{tier}")
                if index in mapping:
                    raise ValueError(f"{path}: Q{index} assigned more than once in {condition}")
                mapping[index] = tier
        tier_by_condition[condition] = mapping
    return tier_by_condition


def band_of(tier: str, variant: str) -> int:
    for index, band in enumerate(BANDS[variant]):
        if tier in band:
            return index
    raise ValueError(f"tier {tier!r} is not in any {variant} band")


def question_utility(tier: str, unasked: Counter, variant: str) -> float:
    mine = band_of(tier, variant)
    for better in BANDS[variant][:mine]:
        if any(unasked[t] > 0 for t in better):
            return 0.0
    return 1.0


def _int_or(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_sequence(
    turns: Sequence[dict[str, Any]],
    retry_log: Sequence[dict[str, Any]],
    tiers: dict[int, str],
    not_in_list: str = "zero",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    retries_by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in retry_log or []:
        if isinstance(entry, dict) and _int_or(entry.get("turn"), None) is not None:
            retries_by_turn[int(entry["turn"])].append(entry)

    unasked = Counter(tiers.values())
    asked: set[int] = set()
    sequence: list[dict[str, Any]] = []
    counts: Counter = Counter()
    zero = {variant: 0.0 for variant in VARIANTS}

    def slot(kind: str, turn: int, index: int | None, tier: str | None,
             utilities: dict[str, float] | None, forced: bool = False) -> None:
        sequence.append({"kind": kind, "turn": turn, "q": index, "tier": tier,
                         "u": utilities, "forced": forced})

    def add_retry(entry: dict[str, Any], turn: int) -> None:
        index = canonical_index(entry.get("canonical", entry.get("question")))
        if entry.get("reason") == "duplicate":
            counts["n_duplicates"] += 1
            slot("duplicate", turn, index, tiers.get(index), dict(zero))
            return
        counts["n_invalid"] += 1
        if not_in_list == "zero":
            slot("invalid", turn, index, None, dict(zero))

    def retries_for(turn: int) -> list[dict[str, Any]]:
        return sorted(retries_by_turn.pop(turn, []),
                      key=lambda entry: _int_or(entry.get("attempt"), 0))

    for turn in sorted(turns, key=lambda t: _int_or(t.get("turn"), 0)):
        number = _int_or(turn.get("turn"), 0)
        for entry in retries_for(number):
            add_retry(entry, number)

        if turn.get("decision") != "EXPLORE":
            forced = bool(turn.get("forced", False))
            counts["n_exploits"] += 1
            counts["n_forced_exploits"] += forced
            slot("exploit", number, None, None, None, forced)
            continue

        index = canonical_index(turn.get("question"))
        if index is None or index not in tiers:
            counts["n_invalid"] += 1
            counts["n_unmatched_questions"] += 1
            if not_in_list == "zero":
                slot("invalid", number, None, None, dict(zero))
            continue
        tier = tiers[index]
        if index in asked:
            counts["n_duplicates"] += 1
            slot("duplicate", number, index, tier, dict(zero))
            continue
        utilities = {variant: question_utility(tier, unasked, variant) for variant in VARIANTS}
        asked.add(index)
        unasked[tier] -= 1
        counts["n_asks"] += 1
        slot("ask", number, index, tier, utilities)

    for number in sorted(retries_by_turn):
        for entry in retries_for(number):
            add_retry(entry, number)
        counts["n_orphan_retry_turns"] += 1

    return sequence, dict(counts)


def eq2_terms(sequence: Sequence[dict[str, Any]], gamma: float) -> dict[str, Any]:
    numerators = {variant: 0.0 for variant in VARIANTS}
    denominator = 0.0
    for position, item in enumerate(sequence):
        if item["kind"] == "exploit":
            continue
        weight = gamma ** position
        denominator += weight
        for variant in VARIANTS:
            numerators[variant] += weight * item["u"][variant]
    return {"num": numerators, "den": denominator}


@dataclass
class Run:
    path: Path
    user_id: str
    model: str
    condition: str
    schema: int
    explore_cost: int
    movie_visibility: str
    has_retry_log: bool
    sequence: list[dict[str, Any]]
    counts: dict[str, int]
    terms: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def level_label(self) -> str:
        return LEVEL_LABELS[CONTEXT_HIERARCHY.index(self.condition)]


def retry_log_from(*sources: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    for source in sources:
        if isinstance(source, dict) and "duplicate_retry_log" in source:
            value = source["duplicate_retry_log"]
            return value if isinstance(value, list) else []
    return None


def resolve_model(summary: dict[str, Any], path: Path, results_root: Path,
                  model_name: str | None) -> str:
    if model_name:
        return model_name
    if summary.get("model"):
        return str(summary["model"])
    try:
        parts = path.resolve().relative_to(results_root.resolve()).parts
    except (ValueError, OSError):
        return "unknown"
    return parts[0] if len(parts) >= 4 else "unknown"


def load_run(
    path: Path,
    results_root: Path,
    tier_by_condition: dict[str, dict[int, str]],
    gammas: Sequence[float],
    not_in_list: str,
    model_name: str | None,
) -> tuple[Run | None, str | None]:
    turn_path = path.with_name("turn_log.json")
    if not turn_path.exists():
        return None, f"skipped run without turn_log.json: {path}"
    try:
        summary = load_json(path)
        turn_log = load_json(turn_path)
    except (OSError, ValueError) as exc:
        return None, f"skipped unreadable run {path}: {exc}"

    turns = turn_log.get("turns", []) if isinstance(turn_log, dict) else turn_log
    if not isinstance(turns, list) or not turns:
        return None, f"skipped run without turns: {path}"
    condition = str(summary.get("condition", ""))
    if condition not in CONTEXT_HIERARCHY:
        return None, None

    schema = _int_or(summary.get("schema"), 0)
    retry_log = retry_log_from(turn_log, summary)
    sequence, counts = build_sequence(
        turns, retry_log or [], tier_by_condition.get(condition, {}), not_in_list
    )
    run = Run(
        path=path,
        user_id=str(summary.get("user_id", path.parent.parent.name)),
        model=resolve_model(summary, path, results_root, model_name),
        condition=condition,
        schema=schema,
        explore_cost=_int_or(summary.get("explore_cost"), -5 if schema == 1 else -2),
        movie_visibility=str(summary.get("movie_visibility", "unknown")),
        has_retry_log=retry_log is not None,
        sequence=sequence,
        counts=counts,
    )
    run.terms = {gamma_key(gamma): eq2_terms(sequence, gamma) for gamma in gammas}

    warning = None
    if retry_log is None:
        warning = f"no duplicate_retry_log in {path}: duplicate attempts cannot be recovered"
    elif counts.get("n_unmatched_questions"):
        warning = (f"{counts['n_unmatched_questions']} EXPLORE question(s) not matched "
                   f"to the canonical list in {path}")
    return run, warning


def discover_runs(
    results_root: Path,
    tier_by_condition: dict[str, dict[int, str]],
    gammas: Sequence[float],
    not_in_list: str = "zero",
    visibility: str | None = None,
    model_name: str | None = None,
    workers: int = 8,
) -> tuple[list[Run], list[str]]:
    paths = sorted(results_root.rglob("summary.json"))

    def load(path: Path) -> tuple[Run | None, str | None]:
        return load_run(path, results_root, tier_by_condition, gammas, not_in_list, model_name)

    runs: list[Run] = []
    warnings: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for run, warning in tqdm(executor.map(load, paths), total=len(paths),
                                 desc="Loading runs", unit="run"):
            if warning is not None:
                warnings.append(warning)
            if run is not None and (visibility is None or run.movie_visibility == visibility):
                runs.append(run)
    return runs, warnings
