"""
Explore/exploit movie-preference experiment on an open-weight model (vLLM).
See README.md for setup, the run grid, and how batching/sharding work.

CLI:
  --model           : Hugging Face model id (default meta-llama/Llama-3.1-70B-Instruct)
  --model_dir       : results subfolder name (default: derived from --model)
  --condition       : demographics_only | context_only | demographics_context
                      | none
  --schema          : 1 | 2
  --movie_visibility: seen | unseen
  --tensor_parallel_size: number of GPUs to shard the model across
  --batch_size      : users driven concurrently (1 == sequential)
  --shard / --num_shards: split users across N processes of the same config
  --overwrite       : re-run users that already have a summary.json

Results go to rl_explore_exploit_results/<model_dir>/user_<id>/<run_tag>/.
"""

import argparse
import pandas as pd
import numpy as np
import json
import re
import random
import os
from vllm import LLM, SamplingParams
from collections import Counter
import io
import contextlib

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--condition",        required=True,
                    choices=["demographics_only", "context_only",
                             "demographics_context", "none"])
parser.add_argument("--schema",           required=True, type=int, choices=[1, 2])
parser.add_argument("--movie_visibility", required=True, choices=["seen", "unseen"])
parser.add_argument("--model", default="meta-llama/Llama-3.1-70B-Instruct",
                    help="Hugging Face model id to load with vLLM.")
parser.add_argument("--model_dir", default=None,
                    help=("Results subfolder under rl_explore_exploit_results/. "
                          "Defaults to a slug of --model, e.g. "
                          "llama_3_1_70b_instruct."))
parser.add_argument("--tensor_parallel_size", type=int,
                    default=int(os.environ.get("TENSOR_PARALLEL_SIZE", "4")),
                    help=("Number of visible GPUs vLLM should shard the model "
                          "across. It must divide the model's attention "
                          "head count (1, 2, 4 or 8 works for most models)."))
parser.add_argument("--batch_size", type=int,
                    default=int(os.environ.get("BATCH_SIZE", "32")),
                    help=("Number of users driven concurrently. Their LLM\n"
                          "calls are merged into single batched vLLM\n"
                          "requests. 1 runs users one at a time."))
parser.add_argument("--shard", type=int, default=0,
                    help="This process's shard index (0-based).")
parser.add_argument("--num_shards", type=int, default=1,
                    help=("Split eligible users across N processes by "
                          "round-robin so concurrent runs of the SAME "
                          "config take disjoint users. Resume-by-skip "
                          "alone does NOT prevent duplication, because "
                          "in-flight users carry no completion marker."))
parser.add_argument("--overwrite", action="store_true",
                    help=("Re-run users that already have a completed "
                          "summary.json for this RUN_TAG. Default is to skip "
                          "them (resume)."))
args = parser.parse_args()

CONDITION        = args.condition
SCHEMA           = args.schema
MOVIE_VISIBILITY = args.movie_visibility
TENSOR_PARALLEL_SIZE = args.tensor_parallel_size
BATCH_SIZE           = args.batch_size
SHARD                = args.shard
NUM_SHARDS           = max(1, args.num_shards)
assert 0 <= SHARD < NUM_SHARDS, "--shard must be in [0, --num_shards)"

# ─────────────────────────────────────────────────────────────────────────────
# FIXED CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TEMPERATURE  = 1.0
N_HISTORY    = 10
NATURE       = "diverse"

if SCHEMA == 1:
    EXPLORE_COST = -5
    EXPLOIT_GAIN = +2
elif SCHEMA == 2:
    EXPLORE_COST = -2
    EXPLOIT_GAIN = +2

CORRECT_REWARD   = +1
INCORRECT_REWARD = -1

RANDOM_SEED      = 42
N_TURNS          = 20
MOVIES_PER_TURN  = 4
INITIAL_BUDGET   = 10
BUDGET_FLOOR     = 0
MIN_MOVIES_NEEDED = 80   # users with fewer exploit rows are skipped

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

MODEL_NAME = args.model
MODEL_DIR  = args.model_dir or re.sub(
    r"[^a-z0-9]+", "_", MODEL_NAME.split("/")[-1].lower()).strip("_")
OUT_ROOT   = os.path.join("rl_explore_exploit_results", MODEL_DIR)

# Llama-3 models keep the hand-written template below, so their prompts stay
# identical to earlier runs. Every other model uses its tokenizer's template.
USE_LLAMA3_TEMPLATE = MODEL_NAME.startswith("meta-llama/") and "Llama-3" in MODEL_NAME
TOKENIZER = None

# Models advertise long contexts, but the prompts here fit well within 8192.
# A longer limit makes vLLM reserve KV cache that may not fit next to the
# weights.
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))

# Fraction of each GPU's memory vLLM may use for weights + KV cache. Lower it
# if the engine runs out of memory during start-up.
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.94"))

RUN_TAG = (
    f"schema{SCHEMA}_{CONDITION}_temp{TEMPERATURE}_{MOVIE_VISIBILITY}"
    if CONDITION in ("demographics_only", "none")
    else f"schema{SCHEMA}_{CONDITION}_{N_HISTORY}movies_{NATURE}"
         f"_temp{TEMPERATURE}_{MOVIE_VISIBILITY}"
)

# ─────────────────────────────────────────────────────────────────────────────
# FIXED QUESTION LIST
# Shuffled once per user in main() (seeded), before sharding/resume checks.
# ─────────────────────────────────────────────────────────────────────────────

QUESTIONS = [
    "What is your age group?",
    "What is your gender?",
    "What is your occupation?",
    "How would you describe your area — urban, suburban, or rural?",
    "Would you describe your work as more intellectual/creative, "
    "practical/hands-on, or service-oriented?",
    "Are you part of a generation that grew up with cinema as a cultural "
    "centerpiece (pre-1970s), the blockbuster/VHS era (1970s-1990s), "
    "or the streaming/digital era (post-1990s)?",
    "What are your top 3 favourite movie genres?",
    "Which genres do you tend to dislike or avoid?",
    "Overall, are you an easy-to-please viewer or a hard-to-impress one?",
    "Are you an avid, moderate, or light movie watcher?",
    "Do you tend to enjoy older classic films, or do you gravitate more "
    "toward modern productions?",
    "Do you tend to rate movies within your favourite genres consistently "
    "highly, or does quality vary a lot even within genres you like?",
    "What is your favorite color?",
    "What is your lucky number?",
    "Do you prefer tea or coffee?",
]

# ─────────────────────────────────────────────────────────────────────────────
# SAMPLING PARAMS
# ─────────────────────────────────────────────────────────────────────────────

SAMPLING_DECIDE = SamplingParams(
    temperature=TEMPERATURE, max_tokens=120, top_p=0.9,
    stop=["<|eot_id|>", "<|end_of_text|>"],
)
SAMPLING_PREDICT = SamplingParams(
    temperature=TEMPERATURE, max_tokens=400, top_p=0.9,
    stop=["<|eot_id|>", "<|end_of_text|>"],
)

# ─────────────────────────────────────────────────────────────────────────────
# LLAMA-3.1 CHAT TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

def llama_prompt(system: str, user: str) -> str:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def build_prompt(system: str, user: str) -> str:
    if USE_LLAMA3_TEMPLATE:
        return llama_prompt(system, user)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    try:
        text = TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Some templates (e.g. Gemma) reject a system role; fold it into the user turn.
        text = TOKENIZER.apply_chat_template(
            [{"role": "user", "content": f"{system}\n\n{user}"}],
            tokenize=False, add_generation_prompt=True)
    # vLLM adds BOS itself when it tokenizes the prompt.
    bos = TOKENIZER.bos_token
    if bos and text.startswith(bos):
        text = text[len(bos):]
    return text

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def clip(text, n):
    if not text:
        return ""
    s = str(text).strip()
    return s if len(s) <= n else s[:n] + "..."


def extract_demographics(movie_df):
    row = movie_df.iloc[0]
    age = str(row.get("Age",        "unknown"))
    gen = str(row.get("Gender",     "unknown"))
    occ = str(row.get("Occupation", "unknown"))
    return age, gen, occ


def llm_request(prompt_str, sampling_params, retries=2):
    """Request one LLM completion from inside a run_user() coroutine.

    Instead of calling the engine itself it YIELDS the work item and is
    resumed with the generated text (or None on failure). The scheduler in
    run_cohort() collects one such item from every in-flight user and
    submits them as a single batched vLLM request. Up to `retries` attempts,
    then None.
    """
    for attempt in range(retries):
        response = yield (prompt_str, sampling_params)
        if response is not None:
            return response.strip()
        print(f"    [llm_request] attempt {attempt+1} produced no output")
    return None


def is_duplicate_question(new_q, qa_history, threshold=0.7):
    if not qa_history:
        return False
    new_words = set(new_q.lower().split())
    for qa in qa_history:
        prev_words = set(qa["question"].lower().split())
        if not prev_words:
            continue
        overlap = len(new_words & prev_words) / len(new_words | prev_words)
        if overlap >= threshold:
            return True
    return False


def match_question_to_list(raw_q: str, shuffled_questions: list) -> tuple:
    """
    Match model output to closest canonical question using Jaccard similarity.
    Returns (canonical_question, position_in_shuffled_list) or (None, None).
    Position is 1-indexed.
    """
    if not raw_q:
        return None, None

    raw_words     = set(raw_q.lower().split())
    best_score    = 0.0
    best_question = None

    for q in QUESTIONS:
        q_words = set(q.lower().split())
        if not q_words:
            continue
        score = len(raw_words & q_words) / len(raw_words | q_words)
        if score > best_score:
            best_score    = score
            best_question = q

    if best_score < 0.5:
        return None, None

    try:
        position = shuffled_questions.index(best_question) + 1
    except ValueError:
        position = None

    return best_question, position


def lookup_answer(simulator_data: dict, user_id: int, question: str) -> str:
    user_profile = (simulator_data.get(str(user_id))
                    or simulator_data.get(int(user_id), {}))
    return str(user_profile.get(question, "Not available"))

# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT HISTORY SELECTION
# Diverse, balanced 5 Like + 5 Dislike from context pool
# ─────────────────────────────────────────────────────────────────────────────

def get_dominant_genre(movies_df):
    genre_counts = Counter()
    for genres_str in movies_df["Genres"]:
        for g in str(genres_str).split("|"):
            genre_counts[g.strip()] += 1
    return genre_counts.most_common(1)[0][0]


def select_history_diverse_balanced(context_pool: pd.DataFrame,
                                    n: int = 10) -> tuple:
    """
    Select n movies from context_pool with diverse genres and
    exactly n//2 Like and n//2 Dislike.

    Steps:
      1. Greedy diverse selection (2n candidates, ignoring label).
      2. From candidates, take n//2 Likes and n//2 Dislikes.
      3. Pad from remaining context pool if one label is short.
    """
    likes_needed    = n // 2
    dislikes_needed = n // 2

    if context_pool.empty:
        return pd.DataFrame(), "N/A"

    # Step 1: greedy diverse candidates
    shuffled         = context_pool.sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)
    seen_genres      = set()
    selected_indices = []

    for idx, row in shuffled.iterrows():
        movie_genres = set(str(row["Genres"]).split("|"))
        if movie_genres - seen_genres:
            selected_indices.append(idx)
            seen_genres.update(movie_genres)
        if len(selected_indices) == n * 2:
            break

    if len(selected_indices) < n:
        already = set(selected_indices)
        for idx in shuffled.index:
            if idx not in already:
                selected_indices.append(idx)
            if len(selected_indices) == n * 2:
                break

    candidates = shuffled.loc[selected_indices].reset_index(drop=True)

    # Step 2: take n//2 of each label
    chosen_likes    = candidates[candidates["Preference"] == "Like"].head(likes_needed)
    chosen_dislikes = candidates[candidates["Preference"] == "Dislike"].head(dislikes_needed)

    # Step 3: pad if short
    chosen_titles = set(chosen_likes["Title"].tolist()
                        + chosen_dislikes["Title"].tolist())
    remaining     = context_pool[~context_pool["Title"].isin(chosen_titles)]

    if len(chosen_likes) < likes_needed:
        need         = likes_needed - len(chosen_likes)
        pad          = remaining[remaining["Preference"] == "Like"].head(need)
        chosen_likes = pd.concat([chosen_likes, pad])

    if len(chosen_dislikes) < dislikes_needed:
        need             = dislikes_needed - len(chosen_dislikes)
        pad              = remaining[remaining["Preference"] == "Dislike"].head(need)
        chosen_dislikes  = pd.concat([chosen_dislikes, pad])

    result   = pd.concat([chosen_likes, chosen_dislikes]).reset_index(drop=True)
    dominant = get_dominant_genre(result)
    return result, dominant

# ─────────────────────────────────────────────────────────────────────────────
# PARSERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_decision(text, debug_label=""):
    if not text:
        if debug_label:
            print(f"    [parse:{debug_label}] empty response")
        return None, None
    if debug_label:
        print(f"    [parse:{debug_label}] raw='{clip(text, 120)}'")

    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return None, None

    for i, line in enumerate(lines):
        lu = line.upper()
        if re.match(r"^EXPLOIT[\.\:\s]*$", lu):
            return "EXPLOIT", None
        if re.match(r"^EXPLORE[\.\:\s]*$", lu):
            after = line[7:].strip(" :.")
            if after:
                return "EXPLORE", clip(after, 300)
            elif i + 1 < len(lines):
                return "EXPLORE", clip(lines[i + 1], 300)
            else:
                return "EXPLORE", ""

    upper_all = text.upper()
    if "EXPLOIT" in upper_all:
        return "EXPLOIT", None
    if "EXPLORE" in upper_all:
        for i, line in enumerate(lines):
            if "EXPLORE" in line.upper():
                q = lines[i + 1] if i + 1 < len(lines) else ""
                return "EXPLORE", clip(q, 300)

    return None, None


def parse_predictions(text):
    if not text:
        return None
    text = re.sub(r"```[a-z]*", "", text).strip()

    try:
        s = text.find("{"); e = text.rfind("}") + 1
        if s != -1 and e > s:
            data = json.loads(text[s:e])
            if isinstance(data.get("predictions"), list):
                return data["predictions"]
    except Exception:
        pass

    try:
        s = text.find("["); e = text.rfind("]") + 1
        if s != -1 and e > s:
            arr = json.loads(text[s:e])
            if isinstance(arr, list) and arr and "title" in arr[0]:
                return arr
    except Exception:
        pass

    try:
        s = text.find("[")
        if s != -1:
            partial    = text[s:]
            last_brace = partial.rfind("}")
            if last_brace != -1:
                fixed = partial[:last_brace + 1] + "]"
                arr   = json.loads(fixed)
                if isinstance(arr, list) and arr and "title" in arr[0]:
                    return arr
    except Exception:
        pass

    results = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if '"title"' in line.lower() and '"prediction"' in line.lower():
            try:
                s = line.find("{"); e = line.rfind("}") + 1
                if s != -1 and e > s:
                    obj = json.loads(line[s:e])
                    if "title" in obj and "prediction" in obj:
                        results.append(obj)
            except Exception:
                pass
    return results if results else None

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM MESSAGES
# ─────────────────────────────────────────────────────────────────────────────

def build_sys_decide(budget: int, shuffled_questions: list) -> str:
    question_list = "\n".join(
        f"  {i+1:2d}. {q}"
        for i, q in enumerate(shuffled_questions)
    )

    forced_note = (
        f"\nNOTE: Your current budget is {budget} points, which is less than "
        f"the cost of exploring ({abs(EXPLORE_COST)} points). You cannot choose "
        f"EXPLORE this turn. You must choose EXPLOIT."
        if budget < abs(EXPLORE_COST)
        else ""
    )

    return (
        "You are a movie recommendation assistant. You are helping predict\n"
        "whether a user will like or dislike movies.\n\n"

        "You are given a points budget. Each turn, you must choose between\n"
        "EXPLORE and EXPLOIT.\n\n"

        "WHAT EACH CHOICE MEANS:\n\n"

        "  EXPLORE: Choose this if you feel you do not yet have enough\n"
        "  information about the user to make confident predictions. You\n"
        "  select one question from the list below to learn more about the\n"
        "  user's preferences or background. You do not make any predictions\n"
        "  this turn. You must choose a question that has not already been\n"
        "  asked in this session.\n\n"

        "  EXPLOIT: Choose this if you feel you have enough information about\n"
        "  the user to make predictions. You predict whether the user will\n"
        "  Like or Dislike 4 movies based on what you know about them so far.\n"
        "  You do not ask any questions this turn.\n\n"

        "AVAILABLE QUESTIONS (choose exactly one if you select EXPLORE):\n"
        f"{question_list}\n\n"

        "SCORING RULES:\n\n"
        "  Choosing EXPLORE costs points:\n"
        f"    Your budget changes by {EXPLORE_COST} points.\n\n"
        "  Choosing EXPLOIT earns points:\n"
        f"    You receive +{EXPLOIT_GAIN} points for choosing to exploit.\n"
        "    For each prediction:\n"
        f"      If your prediction is correct: +{CORRECT_REWARD} point\n"
        f"      If your prediction is incorrect: {INCORRECT_REWARD} point\n\n"
        "  So if you exploit and predict 4 movies:\n"
        f"    Best case  (4 correct, 0 wrong): +{EXPLOIT_GAIN} +4 = "
        f"+{EXPLOIT_GAIN + 4} points\n"
        f"    Worst case (0 correct, 4 wrong): +{EXPLOIT_GAIN} -4 = "
        f"{EXPLOIT_GAIN - 4} points\n\n"

        "BUDGET RULES:\n"
        "  You start with a budget of 10 points.\n"
        "  Your budget cannot exceed 10.\n"
        "  Your budget cannot go below 0.\n"
        f"  If your budget is less than {abs(EXPLORE_COST)}, you cannot\n"
        "  choose EXPLORE and must choose EXPLOIT.\n\n"

        "WORKED EXAMPLE:\n"
        "  Current budget: 8\n"
        f"  You choose EXPLORE: budget becomes 8 + ({EXPLORE_COST}) = "
        f"{8 + EXPLORE_COST}\n"
        "  You choose EXPLOIT and get 3 correct, 1 wrong:\n"
        f"    budget becomes 8 + {EXPLOIT_GAIN} + 3 - 1 = "
        f"{8 + EXPLOIT_GAIN + 3 - 1}, capped at 10.\n\n"

        "IMPORTANT RULES FOR EXPLORE:\n"
        "  - You must copy the question exactly as it appears in the list above.\n"
        "  - You may not rephrase, shorten, or modify the question in any way.\n"
        "  - You may not ask a question that has already been asked this session.\n"
        "  - You may not invent a question that is not in the list.\n\n"

        "Reply with ONLY one of these two formats, nothing before, nothing after:\n\n"
        "EXPLORE\n"
        "[copy the exact question from the list above]\n\n"
        "EXPLOIT\n"
        "[one short reason why you feel ready to predict]"
        + forced_note
    )


def build_sys_predict() -> str:
    return (
        "You are a movie recommendation system. Your task is to predict\n"
        "whether a specific user will Like or Dislike each movie given to you.\n\n"
        "Base your predictions only on the information provided about this user:\n"
        "their demographics, their rating history, and any preferences they\n"
        "have expressed in the Questions and Answers.\n\n"
        "Output ONLY a valid JSON object. No explanation. No markdown.\n"
        "No extra text before or after the JSON.\n\n"
        "Required format:\n"
        "{\n"
        '  "predictions": [\n'
        '    {"title": "Movie Title Here", "prediction": "Like"},\n'
        '    {"title": "Another Movie",    "prediction": "Dislike"}\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        '  - Use exactly "Like" or "Dislike" for each prediction, nothing else.\n'
        "  - Include every movie from the input, no exceptions.\n"
        "  - Copy each title exactly as given, character for character.\n"
        "  - Begin your response immediately with the opening brace {"
    )

# ─────────────────────────────────────────────────────────────────────────────
# USER MESSAGE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def history_block(history_movies):
    lines = []
    for _, row in history_movies.iterrows():
        lines.append(
            f"  \"{row['Title']}\" [{row['Genres']}] - {row['Preference']}"
        )
    return "\n".join(lines)


def movies_block(movies: list) -> str:
    lines = []
    for m in movies:
        lines.append(
            f"  title=\"{m['title']}\" year={m['year']} "
            f"genres=[{m['genres']}]"
        )
    return "\n".join(lines)


def user_msg_decide(age, gender, occupation,
                    history_movies, qa_history,
                    budget, turn,
                    last_batch_result,
                    duplicate_warning,
                    upcoming_movies=None):
    parts = [
        f"Turn {turn} of {N_TURNS}.",
        f"Current budget: {budget} / 10.",
    ]

    if age is not None:
        parts.append(
            f"\nUser demographics: Age {age}, Gender {gender}, "
            f"Occupation {occupation}"
        )

    if history_movies is not None and len(history_movies) > 0:
        parts.append(
            f"\nUser's rated movie history ({len(history_movies)} movies):"
        )
        parts.append(history_block(history_movies))

    if qa_history:
        parts.append("\nInformation gathered so far from previous questions:")
        for i, qa in enumerate(qa_history, 1):
            parts.append(f"  Question {i}: {qa['question']}")
            parts.append(f"  Answer {i}: {clip(qa['answer'], 200)}")

    if last_batch_result:
        parts.append("\nResults from last exploit turn:")
        for r in last_batch_result["details"]:
            outcome = "correct" if r["correct"] else "incorrect"
            parts.append(
                f"  \"{r['title']}\" - predicted {r['prediction']}, "
                f"actual {r['actual']} - {outcome}"
            )
        parts.append(
            f"  Correct: {last_batch_result['correct']} out of "
            f"{last_batch_result['total']}. "
            f"Budget change: {last_batch_result['budget_change']:+d} points."
        )

    # Seen condition: show upcoming movies without labels
    if upcoming_movies is not None:
        parts.append(
            "\nIf you choose EXPLOIT this turn, these are the 4 movies "
            "you will be asked to predict (labels not shown):"
        )
        parts.append(movies_block(upcoming_movies))

    if duplicate_warning:
        parts.append(
            f"\nNOTE: Your last question was too similar to a previous one "
            f"(\"{duplicate_warning}\"). "
            "If you EXPLORE, choose a different question from the list."
        )

    parts.append("\nWill you EXPLORE or EXPLOIT?")
    return "\n".join(parts)


def user_msg_predict(age, gender, occupation,
                     history_movies, qa_history, movies):
    parts = []

    if age is not None:
        parts.append(
            f"USER DEMOGRAPHICS: Age {age}, Gender {gender}, "
            f"Occupation {occupation}"
        )

    if history_movies is not None and len(history_movies) > 0:
        parts.append(f"\nUSER RATED HISTORY ({len(history_movies)} movies):")
        parts.append(history_block(history_movies))

    if qa_history:
        parts.append("\nUSER PREFERENCES (from Questions and Answers):")
        for qa in qa_history:
            parts.append(f"  Q: {qa['question']}")
            parts.append(f"  A: {clip(qa['answer'], 150)}")

    parts.append("\nMOVIES TO PREDICT:")
    for m in movies:
        parts.append(
            f"  title=\"{m['title']}\" year={m['year']} "
            f"genres=[{m['genres']}]"
        )
    parts.append("\nOutput the JSON object now:")
    return "\n".join(parts)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_precision(details):
    tp = sum(1 for d in details
             if d["prediction"] == "Like" and d["actual"] == "Like")
    fp = sum(1 for d in details
             if d["prediction"] == "Like" and d["actual"] == "Dislike")
    return tp / (tp + fp) if (tp + fp) > 0 else None


FAKE_TITLES = {"The Matrix", "Gigli", "Inception", "Forrest Gump", "Pulp Fiction"}


def do_exploit(age, gender, occupation,
               history_movies, qa_history, movies):
    prompt   = build_prompt(
        build_sys_predict(),
        user_msg_predict(age, gender, occupation,
                         history_movies, qa_history, movies)
    )
    response = yield from llm_request(prompt, SAMPLING_PREDICT)

    if response is None:
        print("    no LLM response")
        return None

    print(f"    raw (first 300): {clip(response, 300)}")
    preds = parse_predictions(response)

    if not preds:
        print("    parse failed")
        return None

    preds = [
        p for p in preds
        if p.get("title", "").strip()
        and not p.get("title", "").startswith("<")
        and p.get("title", "") not in FAKE_TITLES
    ]

    if not preds:
        print("    only placeholders after filtering")
        return None

    matched = []
    for movie in movies:
        t_lower    = movie["title"].lower()
        pred_entry = None
        for p in preds:
            if p.get("title", "").lower() == t_lower:
                pred_entry = p
                break
        if pred_entry is None:
            for p in preds:
                pt = p.get("title", "").lower()
                if t_lower in pt or pt in t_lower:
                    pred_entry = p
                    break

        prediction = "Like"
        if pred_entry is None:
            print(f"    no match for '{movie['title']}' – defaulting to Like")
        else:
            # exact match first; note "like" is a substring of "dislike"
            raw = pred_entry.get("prediction", "Like").strip().lower()
            if raw == "dislike":
                prediction = "Dislike"
            elif raw == "like":
                prediction = "Like"
            else:
                prediction = "Dislike" if "dislike" in raw else "Like"

        matched.append({
            "title":      movie["title"],
            "year":       movie["year"],
            "genres":     movie["genres"],
            "prediction": prediction,
        })
    return matched

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE USER RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_user(user_id, exploit_pool, context_pool, simulator_data, out_root,
             shuffled_questions):
    print(f"\n{'='*70}")
    print(f"USER {user_id}  |  {RUN_TAG}")
    print(f"{'='*70}")

    age, gender, occupation = extract_demographics(exploit_pool)
    print(f"Demographics: Age={age}, Gender={gender}, Occupation={occupation}")

    # NOTE: shuffled_questions is drawn by the caller (see main()) so that the
    # global RNG stream advances once per eligible user whether or not the user
    # is skipped on resume. Do not draw it here.

    # ── context history ───────────────────────────────────────────────────────
    if CONDITION in ("demographics_only", "none"):
        history_movies = None
        dominant_genre = "N/A"
    else:
        history_movies, dominant_genre = select_history_diverse_balanced(
            context_pool, n=N_HISTORY
        )
        n_like    = (history_movies["Preference"] == "Like").sum()
        n_dislike = (history_movies["Preference"] == "Dislike").sum()
        print(f"Context: {len(history_movies)} movies "
              f"({n_like} Like, {n_dislike} Dislike), "
              f"dominant genre: {dominant_genre}")

    # ── what predictor sees ───────────────────────────────────────────────────
    pred_age        = age        if CONDITION in ("demographics_only",
                                                   "demographics_context") else None
    pred_gender     = gender     if CONDITION in ("demographics_only",
                                                   "demographics_context") else None
    pred_occupation = occupation if CONDITION in ("demographics_only",
                                                   "demographics_context") else None
    pred_history    = history_movies

    # ── predict pool ──────────────────────────────────────────────────────────
    predict_pool = exploit_pool.reset_index(drop=True)
    pool_idx     = 0

    budget              = INITIAL_BUDGET
    qa_history          = []
    turn_logs           = []
    all_prediction_rows = []
    last_batch_result   = None
    duplicate_warning   = None
    duplicate_retry_log = []

    for turn in range(1, N_TURNS + 1):
        print(f"\n{'─'*50}")
        print(f"TURN {turn}/{N_TURNS}  |  budget={budget}")

        # ── peek at upcoming movies for 'seen' condition ──────────────────────
        upcoming_movies = None
        if MOVIE_VISIBILITY == "seen" and pool_idx < len(predict_pool):
            peek_df = predict_pool.iloc[pool_idx: pool_idx + MOVIES_PER_TURN]
            upcoming_movies = []
            for _, row in peek_df.iterrows():
                full_title = str(row["Title"]).strip()
                m          = re.match(r"^(.*)\((\d{4})\)\s*$", full_title)
                t, y = (m.group(1).strip(), int(m.group(2))) if m else (full_title, 0)
                upcoming_movies.append({
                    "title":  t,
                    "year":   y,
                    "genres": str(row["Genres"]),
                })

        # ── forced exploit check ──────────────────────────────────────────────
        forced = budget < abs(EXPLORE_COST)
        if forced:
            print("  (forced EXPLOIT – budget too low)")
            decision = "EXPLOIT"
            question = None
        else:
            prompt   = build_prompt(
                build_sys_decide(budget, shuffled_questions),
                user_msg_decide(
                    pred_age, pred_gender, pred_occupation,
                    history_movies, qa_history,
                    budget, turn,
                    last_batch_result,
                    duplicate_warning,
                    upcoming_movies=upcoming_movies,
                )
            )
            response  = yield from llm_request(prompt, SAMPLING_DECIDE)
            decision, question = parse_decision(
                response, debug_label=f"turn{turn}_decide"
            )
            duplicate_warning = None

            if decision is None:
                print("  (unparseable → defaulting to EXPLOIT)")
                decision = "EXPLOIT"
                forced   = True

        log_entry = {
            "turn":          turn,
            "budget_before": budget,
            "decision":      decision,
            "forced":        forced,
        }

        # ── EXPLORE ───────────────────────────────────────────────────────────
        if decision == "EXPLORE":
            if not question or len(question.strip()) < 5:
                print("  EXPLORE but no valid question → treating as EXPLOIT")
                decision              = "EXPLOIT"
                log_entry["decision"] = "EXPLOIT"
                log_entry["note"]     = "explore_no_question"

            else:
                MAX_RETRIES   = 3
                retry_count   = 0
                resolved      = False   # True once a valid new question is found
                current_q     = question

                while retry_count <= MAX_RETRIES:
                    canonical_q, q_position = match_question_to_list(
                        current_q, shuffled_questions
                    )

                    # Question not in list at all
                    if canonical_q is None:
                        print(f"  EXPLORE question not in list: "
                            f"'{clip(current_q, 80)}'")
                        duplicate_retry_log.append({
                            "turn":        turn,
                            "attempt":     retry_count + 1,
                            "raw_question": current_q,
                            "canonical":   None,
                            "reason":      "not_in_list",
                        })
                        decision              = "EXPLOIT"
                        log_entry["decision"] = "EXPLOIT"
                        log_entry["note"]     = "explore_question_not_in_list"
                        log_entry["raw_question"] = current_q
                        break

                    # Duplicate question
                    if is_duplicate_question(canonical_q, qa_history):
                        retry_count += 1
                        print(f"  EXPLORE duplicate (attempt {retry_count}): "
                            f"'{clip(canonical_q, 60)}'")

                        duplicate_retry_log.append({
                            "turn":        turn,
                            "attempt":     retry_count,
                            "raw_question": current_q,
                            "canonical":   canonical_q,
                            "reason":      "duplicate",
                            "question_position_in_list": q_position,
                        })

                        if retry_count > MAX_RETRIES:
                            print(f"  Max retries ({MAX_RETRIES}) reached "
                                f"→ treating as EXPLOIT")
                            decision              = "EXPLOIT"
                            log_entry["decision"] = "EXPLOIT"
                            log_entry["note"]     = (
                                f"explore_duplicate_max_retries_{MAX_RETRIES}"
                            )
                            break

                        # Retry: full prompt with explicit error message appended
                        already_asked = [qa["question"] for qa in qa_history]
                        already_str   = "\n".join(
                            f"    - {q}" for q in already_asked
                        )
                        error_note = (
                            f"\n\nERROR: You chose \"{canonical_q}\" but this "
                            f"question has already been asked in this session.\n"
                            f"Questions already asked:\n{already_str}\n"
                            f"You must choose a DIFFERENT question from the list "
                            f"that has NOT been asked yet. "
                            f"Try again (attempt {retry_count + 1} of {MAX_RETRIES})."
                        )

                        retry_prompt = build_prompt(
                            build_sys_decide(budget, shuffled_questions) + error_note,
                            user_msg_decide(
                                pred_age, pred_gender, pred_occupation,
                                history_movies, qa_history,
                                budget, turn,
                                last_batch_result,
                                duplicate_warning=None,
                                upcoming_movies=upcoming_movies,
                            )
                        )
                        retry_response = yield from llm_request(retry_prompt, SAMPLING_DECIDE)
                        retry_decision, retry_question = parse_decision(
                            retry_response,
                            debug_label=f"turn{turn}_retry{retry_count}"
                        )

                        # Model switched to EXPLOIT on retry — accept it
                        if retry_decision == "EXPLOIT":
                            print(f"  Model switched to EXPLOIT on retry {retry_count}")
                            decision              = "EXPLOIT"
                            log_entry["decision"] = "EXPLOIT"
                            log_entry["note"]     = "explore_switched_to_exploit_on_retry"
                            resolved              = True
                            break

                        # Update current_q for next loop iteration
                        current_q = retry_question or ""
                        if not current_q or len(current_q.strip()) < 5:
                            print(f"  No valid question on retry {retry_count} "
                                f"→ treating as EXPLOIT")
                            decision              = "EXPLOIT"
                            log_entry["decision"] = "EXPLOIT"
                            log_entry["note"]     = "explore_no_question_on_retry"
                            break

                        continue   # loop back to check canonical_q again

                    # Valid new question — ask it
                    else:
                        answer = lookup_answer(simulator_data, user_id, canonical_q)
                        print(f"  EXPLORE  Q: {canonical_q} "
                            f"[list pos {q_position}]")
                        print(f"           A: {answer}")

                        qa_history.append({
                            "turn":                      turn,
                            "question":                  canonical_q,
                            "raw_question":              current_q,
                            "answer":                    answer,
                            "question_position_in_list": q_position,
                            "retries_needed":            retry_count,
                        })
                        budget                                += EXPLORE_COST
                        budget                                 = max(budget, BUDGET_FLOOR)
                        log_entry["question"]                  = canonical_q
                        log_entry["raw_question"]              = current_q
                        log_entry["question_position_in_list"] = q_position
                        log_entry["answer"]                    = answer
                        log_entry["budget_change"]             = EXPLORE_COST
                        log_entry["budget_after"]              = budget
                        log_entry["retries_needed"]            = retry_count
                        turn_logs.append(log_entry)
                        last_batch_result                      = None
                        resolved                               = True
                        break

                # If resolved as EXPLORE with a valid question, continue to next turn
                if resolved and decision == "EXPLORE":
                    continue
                # Otherwise fall through to EXPLOIT block below

        # ── EXPLOIT ───────────────────────────────────────────────────────────
        if decision == "EXPLOIT":
            batch_df = predict_pool.iloc[pool_idx: pool_idx + MOVIES_PER_TURN]
            if len(batch_df) == 0:
                print("  prediction pool exhausted – stopping")
                log_entry["note"]         = "pool_exhausted"
                log_entry["budget_after"] = budget
                turn_logs.append(log_entry)
                break
            pool_idx += MOVIES_PER_TURN

            movies = []
            for _, row in batch_df.iterrows():
                full_title = str(row["Title"]).strip()
                m          = re.match(r"^(.*)\((\d{4})\)\s*$", full_title)
                title, year = (
                    (m.group(1).strip(), int(m.group(2))) if m
                    else (full_title, 0)
                )
                movies.append({
                    "title":  title,
                    "year":   year,
                    "genres": str(row["Genres"]),
                    "label":  str(row["Preference"]),
                })

            preds = yield from do_exploit(
                pred_age, pred_gender, pred_occupation,
                pred_history,
                qa_history,
                movies,
            )

            if preds is None:
                print("  prediction failed – skipping turn")
                log_entry["note"]         = "prediction_failed"
                log_entry["budget_after"] = budget
                turn_logs.append(log_entry)
                continue

            correct      = sum(
                1 for p, m in zip(preds, movies)
                if p["prediction"] == m["label"]
            )
            incorrect    = len(preds) - correct
            score_change = (
                EXPLOIT_GAIN
                + correct   * CORRECT_REWARD
                + incorrect * INCORRECT_REWARD
            )
            budget = max(min(budget + score_change, INITIAL_BUDGET),
                         BUDGET_FLOOR)

            print(f"  EXPLOIT  {correct}/{len(preds)} correct  "
                  f"score_change={score_change:+d}  new_budget={budget}")

            details = []
            for p, m in zip(preds, movies):
                is_correct = p["prediction"] == m["label"]
                details.append({
                    "title":      p["title"],
                    "prediction": p["prediction"],
                    "actual":     m["label"],
                    "correct":    is_correct,
                })
                all_prediction_rows.append({
                    "turn":       turn,
                    "title":      p["title"],
                    "year":       p["year"],
                    "genres":     p["genres"],
                    "prediction": p["prediction"],
                    "actual":     m["label"],
                    "correct":    is_correct,
                })

            turn_accuracy  = correct / len(preds)
            turn_precision = compute_precision(details)

            print(f"           accuracy={turn_accuracy:.2f}  "
                  f"precision="
                  f"{f'{turn_precision:.2f}' if turn_precision is not None else 'N/A'}")

            last_batch_result = {
                "correct":       correct,
                "total":         len(preds),
                "budget_change": score_change,
                "details":       details,
            }
            log_entry["exploit_results"] = details
            log_entry["correct"]         = correct
            log_entry["total"]           = len(preds)
            log_entry["score_change"]    = score_change
            log_entry["accuracy"]        = turn_accuracy
            log_entry["precision"]       = turn_precision
            log_entry["budget_after"]    = budget
            turn_logs.append(log_entry)

    # ── save ──────────────────────────────────────────────────────────────────
    out_dir = os.path.join(out_root, f"user_{user_id}", RUN_TAG)
    os.makedirs(out_dir, exist_ok=True)

    explore_turns = sum(1 for t in turn_logs if "EXPLORE" in t["decision"])
    exploit_turns = sum(1 for t in turn_logs if t["decision"] == "EXPLOIT")
    forced_turns  = sum(
        1 for t in turn_logs
        if t.get("forced") and t["decision"] == "EXPLOIT"
    )

    question_counts = {}
    for qa in qa_history:
        q = qa["question"]
        question_counts[q] = question_counts.get(q, 0) + 1

    full_log = {
        "user_id":                 int(user_id),
        "run_tag":                 RUN_TAG,
        "model":                   MODEL_NAME,
        "condition":               CONDITION,
        "schema":                  SCHEMA,
        "movie_visibility":        MOVIE_VISIBILITY,
        "n_history":               N_HISTORY,
        "nature":                  NATURE,
        "temperature":             TEMPERATURE,
        "explore_cost":            EXPLORE_COST,
        "exploit_gain":            EXPLOIT_GAIN,
        "demographics":            f"Age:{age}, Gender:{gender}, "
                                   f"Occupation:{occupation}",
        "dominant_genre":          dominant_genre,
        "shuffled_question_order": shuffled_questions,
        "turns":                   turn_logs,
        "duplicate_retry_log": duplicate_retry_log,
    }
    with open(os.path.join(out_dir, "turn_log.json"), "w") as f:
        json.dump(full_log, f, indent=2)

    if not all_prediction_rows:
        print("\n  No successful predictions.")
        summary = {
            "user_id":                 int(user_id),
            "run_tag":                 RUN_TAG,
        "model":                   MODEL_NAME,
            "condition":               CONDITION,
            "schema":                  SCHEMA,
            "movie_visibility":        MOVIE_VISIBILITY,
            "n_history":               N_HISTORY,
            "nature":                  NATURE,
            "temperature":             TEMPERATURE,
            "explore_turns":           explore_turns,
            "exploit_turns":           exploit_turns,
            "forced_exploit_turns":    forced_turns,
            "final_budget":            budget,
            "overall_accuracy":        None,
            "overall_precision":       None,
            "question_counts":         question_counts,
            "shuffled_question_order": shuffled_questions,
            "error":                   "no_predictions",
        }
    else:
        pred_df      = pd.DataFrame(all_prediction_rows)
        pred_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

        overall_acc  = float(pred_df["correct"].mean())
        all_details  = [
            {"prediction": r["prediction"], "actual": r["actual"]}
            for r in all_prediction_rows
        ]
        overall_prec = compute_precision(all_details)

        exploit_turn_metrics = {
            t["turn"]: {
                "accuracy":  t.get("accuracy"),
                "precision": t.get("precision"),
            }
            for t in turn_logs
            if t["decision"] == "EXPLOIT" and "exploit_results" in t
        }

        summary = {
            "user_id":                  int(user_id),
            "run_tag":                  RUN_TAG,
            "model":                    MODEL_NAME,
            "condition":                CONDITION,
            "schema":                   SCHEMA,
            "movie_visibility":         MOVIE_VISIBILITY,
            "n_history":                N_HISTORY,
            "nature":                   NATURE,
            "temperature":              TEMPERATURE,
            "explore_cost":             EXPLORE_COST,
            "exploit_gain":             EXPLOIT_GAIN,
            "demographics":             f"Age:{age}, Gender:{gender}, "
                                        f"Occupation:{occupation}",
            "dominant_genre":           dominant_genre,
            "total_turns":              N_TURNS,
            "explore_turns":            explore_turns,
            "exploit_turns":            exploit_turns,
            "forced_exploit_turns":     forced_turns,
            "total_questions":          len(qa_history),
            "final_budget":             budget,
            "overall_accuracy":         overall_acc,
            "overall_precision":        overall_prec,
            "metrics_per_exploit_turn": exploit_turn_metrics,
            "question_counts":          question_counts,
            "shuffled_question_order":  shuffled_questions,
            "qa_history":               qa_history,
            "duplicate_retry_log":         duplicate_retry_log,
            "total_duplicate_attempts":    len(duplicate_retry_log),
            "duplicate_question_counts":   {
                q: sum(1 for e in duplicate_retry_log if e["canonical"] == q)
                for q in set(
                    e["canonical"] for e in duplicate_retry_log
                    if e["canonical"] is not None
                )
            },
        }

        print(f"\n  overall accuracy   : {overall_acc*100:.1f}%")
        print(f"  overall precision  : "
              f"{f'{overall_prec*100:.1f}%' if overall_prec is not None else 'N/A'}")
        print(f"  explore turns      : {explore_turns}")
        print(f"  exploit turns      : {exploit_turns}")
        print(f"  forced exploits    : {forced_turns}")
        print(f"  final budget       : {budget}")
        print(f"  questions asked    : {len(qa_history)}")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved → {out_dir}")

# ─────────────────────────────────────────────────────────────────────────────
# BATCHED SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

def run_cohort(task_stream, llm, cohort_size):
    """Drive up to `cohort_size` run_user() coroutines concurrently.

    Every in-flight user is advanced until it needs the LLM. All of those
    prompts are then submitted as ONE vLLM call, and each user is resumed with
    its own reply. Users stay completely independent - a user only ever sees
    its own prompts, in its own order - so batching affects only wall-clock
    time: decode runs at batch N instead of batch 1.

    Each user's stdout is captured into its own buffer and flushed as one block
    when that user finishes, so the log stays readable per user instead of
    interleaving N users line by line.
    """
    live      = []
    processed = 0
    exhausted = False

    while True:
        while not exhausted and len(live) < cohort_size:
            nxt = next(task_stream, None)
            if nxt is None:
                exhausted = True
                break
            uid, gen = nxt
            live.append({"uid": uid, "gen": gen, "buf": io.StringIO(),
                         "send": None, "pending": None})

        if not live:
            break

        # advance every in-flight user to its next LLM request
        still = []
        for slot in live:
            try:
                with contextlib.redirect_stdout(slot["buf"]):
                    slot["pending"] = slot["gen"].send(slot["send"])
            except StopIteration:
                print(slot["buf"].getvalue(), end="", flush=True)
                processed += 1
                continue
            except Exception as exc:
                print(slot["buf"].getvalue(), end="")
                print(f"\n  !! user {slot['uid']} aborted: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                continue
            slot["send"] = None
            still.append(slot)
        live = still

        if not live:
            continue

        # one batched engine call for every waiting user
        prompts = [s["pending"][0] for s in live]
        params  = [s["pending"][1] for s in live]
        try:
            outs = llm.generate(prompts, params)
        except Exception as exc:
            print(f"    [batch] generate failed for {len(prompts)} prompts: "
                  f"{exc}", flush=True)
            continue   # llm_request() retries on its own
        for slot, out in zip(live, outs):
            slot["send"] = out.outputs[0].text if out.outputs else None

    return processed

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"RL EXPLORE/EXPLOIT – {RUN_TAG}")
    print(f"Schema {SCHEMA}  |  explore={EXPLORE_COST:+d}  "
          f"exploit={EXPLOIT_GAIN:+d}  temp={TEMPERATURE}")
    print(f"Condition={CONDITION}  Visibility={MOVIE_VISIBILITY}")
    print(f"N_history={N_HISTORY}  Nature={NATURE}")
    print(f"Turns={N_TURNS}  Movies/turn={MOVIES_PER_TURN}  "
          f"Budget={INITIAL_BUDGET}")
    print("=" * 70)

    os.environ.setdefault("HF_HUB_CACHE", os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "hf_models"))

    # Gated repos (e.g. meta-llama) need a valid HuggingFace token. It is read
    # from the environment - never hardcoded here.
    #   export HF_TOKEN=hf_...     (run_ablations.sh also sources .env / ../.env)
    # (or rely on the token saved at ~/.cache/huggingface/token)
    if not (os.environ.get("HF_TOKEN")
            or os.path.exists(os.path.expanduser("~/.cache/huggingface/token"))):
        raise SystemExit(
            "HF_TOKEN is not set and no ~/.cache/huggingface/token exists.\n"
            "Run:  export HF_TOKEN=hf_...   (or: huggingface-cli login)\n"
            f"For gated models the token must belong to an account that has "
            f"accepted the {MODEL_NAME} licence."
        )

    print("Loading dataset...")
    df        = pd.read_csv("filtered_dataset_balanced.csv")
    all_users = sorted(df["UserID"].unique().tolist())
    print(f"  Total users: {len(all_users)}")

    print("Loading simulator data...")
    with open("user_simulator_output.json") as f:
        simulator_data = json.load(f)
    print(f"  Profiles loaded: {len(simulator_data)}")

    print("\nLoading LLM...")
    llm = LLM(model=MODEL_NAME,
              dtype="bfloat16",
              tensor_parallel_size=TENSOR_PARALLEL_SIZE,
              max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
              disable_custom_all_reduce=True,
            )
    global TOKENIZER
    TOKENIZER = llm.get_tokenizer()
    print(f"Ready. {MODEL_NAME} on {TENSOR_PARALLEL_SIZE} GPU(s), "
          f"{'Llama-3' if USE_LLAMA3_TEMPLATE else 'tokenizer'} chat template.\n")

    out_root = OUT_ROOT
    os.makedirs(out_root, exist_ok=True)
    print(f"Results: {out_root}")

    counts = {"skipped": 0, "resumed": 0}

    if not args.overwrite:
        print("Resume mode ON: users with an existing summary.json for "
              f"'{RUN_TAG}' will be skipped (use --overwrite to force re-run).")
    print(f"Batch size: {BATCH_SIZE} users driven concurrently "
          f"({'sequential' if BATCH_SIZE == 1 else 'batched'} mode).")
    if NUM_SHARDS > 1:
        print(f"Shard {SHARD} of {NUM_SHARDS}: taking every "
              f"{NUM_SHARDS}th eligible user (disjoint from other shards).")

    def task_stream():
        """Yield (user_id, run_user-coroutine) for every user that must run.

        Consumed lazily and strictly in sorted user order, so the global RNG
        stream advances exactly as it does in a sequential run.
        """
        for user_id in all_users:
            user_df      = df[df["UserID"] == user_id].reset_index(drop=True)
            exploit_pool = user_df[user_df["pool"] == "exploit"].reset_index(drop=True)
            context_pool = user_df[user_df["pool"] == "context"].reset_index(drop=True)

            if len(exploit_pool) < MIN_MOVIES_NEEDED:
                print(f"User {user_id}: exploit pool has {len(exploit_pool)} "
                      f"movies (need {MIN_MOVIES_NEEDED}) - skipping.")
                counts["skipped"] += 1
                continue

            if (CONDITION not in ("demographics_only", "none")
                    and len(context_pool) < N_HISTORY):
                print(f"User {user_id}: context pool has {len(context_pool)} "
                      f"movies (need {N_HISTORY}) - skipping.")
                counts["skipped"] += 1
                continue

            if (str(user_id) not in simulator_data
                    and user_id not in simulator_data):
                print(f"User {user_id}: no simulator profile - skipping.")
                counts["skipped"] += 1
                continue

            # per-user question shuffle - drawn BEFORE the resume check so the
            # global RNG stream advances once per eligible user whether or not
            # the user is skipped. Keeps question order paired across runs.
            shuffled_questions = QUESTIONS[:]
            random.shuffle(shuffled_questions)

            # shard filter: placed AFTER the shuffle draw so every process
            # advances the global RNG stream identically, and each user
            # still gets the same question order in every shard.
            eligible_idx[0] += 1
            if NUM_SHARDS > 1 and (eligible_idx[0] - 1) % NUM_SHARDS != SHARD:
                continue

            # resume: summary.json is the last file run_user writes, so its
            # presence means the user finished.
            done_marker = os.path.join(out_root, f"user_{user_id}", RUN_TAG,
                                       "summary.json")
            if not args.overwrite and os.path.exists(done_marker):
                counts["resumed"] += 1
                if counts["resumed"] % 100 == 0 or counts["resumed"] == 1:
                    print(f"User {user_id}: already complete - skipping "
                          f"({counts['resumed']} so far).")
                continue

            yield user_id, run_user(user_id, exploit_pool, context_pool,
                                    simulator_data, out_root,
                                    shuffled_questions)

    eligible_idx = [0]
    processed = run_cohort(task_stream(), llm, max(1, BATCH_SIZE))
    skipped   = counts["skipped"]
    resumed   = counts["resumed"]

    print("\n" + "=" * 70)
    print(f"ALL DONE.  Processed: {processed}  "
          f"Already-complete (resumed past): {resumed}  "
          f"Ineligible: {skipped}")
    print("=" * 70)


if __name__ == "__main__":
    main()