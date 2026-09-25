# -*- coding: utf-8 -*-
"""
Schema-level batch executor for the RL explore/exploit study.

Runs all 8 ablations for a given schema simultaneously using the Gemini Batch Jobs API.

Execution model:
  For turn t = 1..20:
    1. Submit one DECIDE batch job per ablation (up to 8 jobs) simultaneously
    2. Poll all jobs concurrently until all complete
    3. Submit one PREDICT batch job per ablation simultaneously
    4. Poll all jobs concurrently until all complete
    5. Apply results, checkpoint all users across all ablations

Ablations (fixed per schema):
  conditions       : demographics_only | context_only | demographics_context | none
  movie_visibility : unseen | seen

Condition semantics (what the model is shown in BOTH the DECIDE and PREDICT prompts):
  demographics_only    : demographics,  no rated history
  context_only         : no demographics, rated history
  demographics_context : demographics,  rated history
  none                 : no demographics, no rated history (Q&A from EXPLORE only)

Resume:
  - Users with summary.json are fully done — skipped.
  - Users with checkpoint.json resume from the next turn.
  - --ignore_summary redoes completed users but still resumes from checkpoint.
  - --rerun redoes everything from turn 1, ignoring both files.

CLI:
  --schema        : 1 | 2
  --model         : gemini-2.5-flash | gemini-2.5-pro
  --thinking      : none | medium
  --poll_interval : seconds between status polls (default: 30)
  --rerun         : rerun all ablations from turn 1, ignoring existing output
  --ignore_summary: redo completed users, resuming each from its checkpoint
  --conditions    : subset of conditions to run (default: all four)
"""

import argparse
import json
import os
import re
import random
import signal
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import numpy as np
import pandas as pd
from google import genai
from google.genai import types
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


parser = argparse.ArgumentParser()
parser.add_argument("--schema",         required=True,  type=int, choices=[1, 2])
parser.add_argument("--model",          required=False,
                    choices=["gemini-2.5-flash", "gemini-2.5-pro"],
                    default="gemini-2.5-flash")
parser.add_argument("--thinking",       required=False,
                    choices=["none", "medium"], default="none")
parser.add_argument("--poll_interval",  required=False, type=int, default=30)
parser.add_argument("--rerun",          action="store_true", default=False)
parser.add_argument("--ignore_summary", action="store_true", default=False,
                    help="Redo users that already have summary.json, but still "
                         "resume each from checkpoint.json instead of turn 1")
parser.add_argument("--temperature",    required=False, type=float,
                    choices=[0.0, 1.0], default=1.0)
parser.add_argument("--n_history",      required=False, type=int,
                    choices=[10, 20], default=10)
parser.add_argument("--nature",         required=False,
                    choices=["homo", "semi", "diverse"], default="diverse")
parser.add_argument("--conditions",     required=False, nargs="+",
                    choices=["demographics_only", "context_only",
                             "demographics_context", "none"],
                    default=None,
                    help="Subset of conditions to run (default: all four)")

args = parser.parse_args()

SCHEMA        = args.schema
MODEL_NAME    = args.model
THINKING      = args.thinking
POLL_INTERVAL = args.poll_interval
RERUN         = args.rerun
IGNORE_SUMMARY = args.ignore_summary or args.rerun
TEMPERATURE   = args.temperature
N_HISTORY     = args.n_history
NATURE        = args.nature
CONDITION_SEL = args.conditions


EXPLORE_COST = -5 if SCHEMA == 1 else -2
EXPLOIT_GAIN = +2
CORRECT_REWARD   = +1
INCORRECT_REWARD = -1


RANDOM_SEED       = 42
N_TURNS           = 20
MOVIES_PER_TURN   = 4
INITIAL_BUDGET    = 10
BUDGET_FLOOR      = 0
MIN_MOVIES_NEEDED = 100

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

THINKING_BUDGET_PREDICT = 2048 if THINKING == "medium" else 0
THINKING_BUDGET_DECIDE  = 1024 if THINKING == "medium" else 0
MAX_TOKENS_DECIDE  = THINKING_BUDGET_DECIDE  + 512  if THINKING_BUDGET_DECIDE  > 0 else 256
MAX_TOKENS_PREDICT = THINKING_BUDGET_PREDICT + 2048 if THINKING_BUDGET_PREDICT > 0 else 1024

if MODEL_NAME == "gemini-2.5-flash":
    MODEL_SUFFIX  = "flash"
    OUT_DIR_MODEL = "gemini_25_flash"
elif THINKING == "medium":
    MODEL_SUFFIX  = "pro_think_medium"
    OUT_DIR_MODEL = "gemini_25_pro_thinking"
else:
    MODEL_SUFFIX  = "pro_think_none"
    OUT_DIR_MODEL = "gemini_25_pro_no_thinking"

PRED_SUFFIX = "showpred"
OUT_ROOT    = os.path.join("rl_explore_exploit_results", OUT_DIR_MODEL)

BATCH_COMPLETED = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
                   "JOB_STATE_CANCELLED", "JOB_STATE_PAUSED"}


CONDITIONS = ["demographics_only", "context_only", "demographics_context", "none"]

CONTEXT_CONDITIONS     = {"context_only", "demographics_context"}
DEMOGRAPHIC_CONDITIONS = {"demographics_only", "demographics_context"}

SELECTED_CONDITIONS = ([c for c in CONDITIONS if c in set(CONDITION_SEL)]
                       if CONDITION_SEL else CONDITIONS)

ABLATIONS = [
    {"condition": c, "movie_visibility": v}
    for c in SELECTED_CONDITIONS
    for v in ["unseen", "seen"]
]


def run_tag_for(condition, movie_visibility):
    if condition not in CONTEXT_CONDITIONS:
        return (f"schema{SCHEMA}_{condition}_temp{TEMPERATURE}_{PRED_SUFFIX}"
                f"_{movie_visibility}_{MODEL_SUFFIX}")
    return (f"schema{SCHEMA}_{condition}_{N_HISTORY}movies_{NATURE}_temp{TEMPERATURE}"
            f"_{PRED_SUFFIX}_{movie_visibility}_{MODEL_SUFFIX}")


QUESTIONS = [
    "What is your age group?",
    "What is your gender?",
    "What is your occupation?",
    "How would you describe your area — urban, suburban, or rural?",
    "Would you describe your work as more intellectual/creative, practical/hands-on, or service-oriented?",
    "Are you part of a generation that grew up with cinema as a cultural centerpiece (pre-1970s), the blockbuster/VHS era (1970s-1990s), or the streaming/digital era (post-1990s)?",
    "What are your top 3 favourite movie genres?",
    "Which genres do you tend to dislike or avoid?",
    "Overall, are you an easy-to-please viewer or a hard-to-impress one?",
    "Are you an avid, moderate, or light movie watcher?",
    "Do you tend to enjoy older classic films, or do you gravitate more toward modern productions?",
    "Do you tend to rate movies within your favourite genres consistently highly, or does quality vary a lot even within genres you like?",
    "What is your favorite color?",
    "What is your lucky number?",
    "Do you prefer tea or coffee?",
]


CLIENT = None


def build_batch_request(system_prompt, user_prompt, max_output_tokens, thinking_budget):
    return {
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "config": {
            "system_instruction": system_prompt,
            "temperature": TEMPERATURE,
            "max_output_tokens": max_output_tokens,
            "top_p": 0.9,
            "thinking_config": {"thinking_budget": thinking_budget},
        },
    }


IN_FLIGHT = {}
IN_FLIGHT_LOG = "in_flight_batches.jsonl"


def _record_in_flight(job, display_name, event):
    try:
        with open(IN_FLIGHT_LOG, "a") as f:
            f.write(json.dumps({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "event": event, "job": job, "display_name": display_name,
            }) + "\n")
    except OSError:
        pass


def submit_batch(requests, display_name):
    job = CLIENT.batches.create(
        model=MODEL_NAME,
        src=requests,
        config={"display_name": display_name},
    )
    IN_FLIGHT[job.name] = display_name
    _record_in_flight(job.name, display_name, "submitted")
    print(f"  [batch] submitted '{display_name}'  job={job.name}  n={len(requests)}")
    return job


POLL_MAX_RETRIES   = 8
POLL_BACKOFF_BASE  = 5
POLL_BACKOFF_MAX   = 120


def batches_get(job_name, display_name):
    attempt = 0
    while True:
        try:
            return CLIENT.batches.get(name=job_name)
        except (httpx.TransportError, OSError) as e:
            attempt += 1
            if attempt > POLL_MAX_RETRIES:
                print(f"  [batch] '{display_name}' get FAILED after "
                      f"{POLL_MAX_RETRIES} retries: {type(e).__name__}: {e}")
                raise
            wait = min(POLL_BACKOFF_BASE * (2 ** (attempt - 1)), POLL_BACKOFF_MAX)
            print(f"  [batch] '{display_name}' transient get error "
                  f"({type(e).__name__}: {e}); retry {attempt}/{POLL_MAX_RETRIES} "
                  f"in {wait}s")
            time.sleep(wait)


def poll_one(job_name, display_name):
    job = batches_get(job_name, display_name)
    while job.state not in BATCH_COMPLETED:
        time.sleep(POLL_INTERVAL)
        job = batches_get(job_name, display_name)
    IN_FLIGHT.pop(job_name, None)
    _record_in_flight(job_name, display_name, f"terminal:{job.state}")
    print(f"  [batch] '{display_name}'  state={job.state}")
    return job


def poll_all(jobs):
    completed = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futures = {
            ex.submit(poll_one, job.name, name): i
            for i, (job, name) in enumerate(jobs)
        }
        for future in as_completed(futures):
            idx = futures[future]
            completed[idx] = future.result()
    return completed


def extract_text(inlined):
    try:
        if getattr(inlined, "error", None) is not None:
            return None
        response = getattr(inlined, "response", inlined)
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    return part.text.strip()
    except Exception:
        pass
    return None


def collect_results(job, display_name):
    if job.state != "JOB_STATE_SUCCEEDED":
        print(f"  [batch] '{display_name}' did not succeed ({job.state}) — None for all")
        return None
    inlined = getattr(job.dest, "inlined_responses", None) if job.dest else None
    if inlined is None:
        print(f"  [batch] '{display_name}' succeeded but has no inlined_responses — None for all")
        return None
    results = [extract_text(r) for r in inlined]
    failed  = sum(1 for r in results if r is None)
    print(f"  [batch] '{display_name}'  n={len(results)}  failed={failed}")
    return results


def clip(text, n):
    if not text:
        return ""
    s = str(text).strip()
    return s if len(s) <= n else s[:n] + "..."


def extract_demographics(movie_df):
    row = movie_df.iloc[0]
    return (str(row.get("Age", "unknown")),
            str(row.get("Gender", "unknown")),
            str(row.get("Occupation", "unknown")))


def get_dominant_genre(movies_df):
    counts = Counter()
    for g in movies_df["Genres"]:
        for genre in str(g).split("|"):
            counts[genre.strip()] += 1
    return counts.most_common(1)[0][0]


def select_history_diverse_balanced(context_pool, n=10):
    likes_needed    = n // 2
    dislikes_needed = n // 2
    if context_pool.empty:
        return pd.DataFrame(), "N/A"
    shuffled = context_pool.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
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
    candidates      = shuffled.loc[selected_indices].reset_index(drop=True)
    chosen_likes    = candidates[candidates["Preference"] == "Like"].head(likes_needed)
    chosen_dislikes = candidates[candidates["Preference"] == "Dislike"].head(dislikes_needed)
    chosen_titles   = set(chosen_likes["Title"].tolist() + chosen_dislikes["Title"].tolist())
    remaining       = context_pool[~context_pool["Title"].isin(chosen_titles)]
    if len(chosen_likes) < likes_needed:
        pad          = remaining[remaining["Preference"] == "Like"].head(likes_needed - len(chosen_likes))
        chosen_likes = pd.concat([chosen_likes, pad])
    if len(chosen_dislikes) < dislikes_needed:
        pad             = remaining[remaining["Preference"] == "Dislike"].head(dislikes_needed - len(chosen_dislikes))
        chosen_dislikes = pd.concat([chosen_dislikes, pad])
    result = pd.concat([chosen_likes, chosen_dislikes]).reset_index(drop=True)
    return result, get_dominant_genre(result)


def is_duplicate_question(new_q, qa_history, threshold=0.7):
    new_words = set(new_q.lower().split())
    for qa in qa_history:
        prev_words = set(qa["question"].lower().split())
        if not prev_words:
            continue
        if len(new_words & prev_words) / len(new_words | prev_words) >= threshold:
            return True
    return False


def match_question_to_list(raw_q, shuffled_questions):
    if not raw_q:
        return None, None
    raw_words  = set(raw_q.lower().split())
    best_score = 0.0
    best_q     = None
    for q in QUESTIONS:
        q_words = set(q.lower().split())
        if not q_words:
            continue
        score = len(raw_words & q_words) / len(raw_words | q_words)
        if score > best_score:
            best_score = score
            best_q     = q
    if best_score < 0.5:
        return None, None
    try:
        position = shuffled_questions.index(best_q) + 1
    except ValueError:
        position = None
    return best_q, position


def lookup_answer(simulator_data, user_id, question):
    profile = simulator_data.get(str(user_id)) or simulator_data.get(int(user_id), {})
    return str(profile.get(question, "Not available"))


def parse_decision(text, debug_label=""):
    if not text:
        return None, None
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
    for attempt in [
        lambda t: json.loads(t[t.find("{"):t.rfind("}")+1]).get("predictions"),
        lambda t: json.loads(t[t.find("["):t.rfind("]")+1]) if t.find("[") != -1 else None,
    ]:
        try:
            result = attempt(text)
            if isinstance(result, list) and result and "title" in result[0]:
                return result
        except Exception:
            pass
    try:
        s = text.find("[")
        if s != -1:
            partial    = text[s:]
            last_brace = partial.rfind("}")
            if last_brace != -1:
                arr = json.loads(partial[:last_brace + 1] + "]")
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
                obj = json.loads(line[s:e])
                if "title" in obj and "prediction" in obj:
                    results.append(obj)
            except Exception:
                pass
    return results if results else None


def compute_precision(details):
    tp = sum(1 for d in details if d["prediction"] == "Like" and d["actual"] == "Like")
    fp = sum(1 for d in details if d["prediction"] == "Like" and d["actual"] == "Dislike")
    return tp / (tp + fp) if (tp + fp) > 0 else None


def build_sys_decide(budget, shuffled_questions, explore_cost, exploit_gain):
    question_list = "\n".join(f"  {i+1:2d}. {q}" for i, q in enumerate(shuffled_questions))
    forced_note = (
        f"\nNOTE: Your current budget is {budget} points, which is less than the "
        f"cost of exploring ({abs(explore_cost)} points). You cannot choose "
        f"EXPLORE this turn. You must choose EXPLOIT."
        if budget < abs(explore_cost) else ""
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
        f"  EXPLORE costs {explore_cost} points.\n"
        f"  EXPLOIT earns +{exploit_gain} points, plus +1 per correct / -1 per wrong prediction.\n"
        f"  Best case (4 correct): +{exploit_gain + 4}  Worst case: {exploit_gain - 4}\n\n"
        "BUDGET RULES:\n"
        "  Start at 10. Cannot exceed 10 or go below 0.\n"
        f"  If budget < {abs(explore_cost)}, you must EXPLOIT.\n\n"
        "IMPORTANT RULES FOR EXPLORE:\n"
        "  - Copy the question exactly as listed above.\n"
        "  - Do not ask a question already asked this session.\n\n"
        "Reply with ONLY one of:\n\n"
        "EXPLORE\n[exact question from the list]\n\n"
        "EXPLOIT\n[one short reason]"
        + forced_note
    )


def build_sys_predict():
    return (
        "You are a movie recommendation system. Predict whether a specific user\n"
        "will Like or Dislike each movie given to you.\n\n"
        "Output ONLY a valid JSON object. No explanation. No markdown.\n\n"
        "{\n"
        '  "predictions": [\n'
        '    {"title": "Movie Title Here", "prediction": "Like"},\n'
        '    {"title": "Another Movie",    "prediction": "Dislike"}\n'
        "  ]\n"
        "}\n\n"
        'Use exactly "Like" or "Dislike". Include every movie. Copy titles exactly.\n'
        "Begin your response immediately with {"
    )


def history_block(history_movies):
    return "\n".join(
        f"  \"{row['Title']}\" [{row['Genres']}] - {row['Preference']}"
        for _, row in history_movies.iterrows()
    )


def movies_block(movies):
    return "\n".join(
        f"  title=\"{m['title']}\" year={m['year']} genres=[{m['genres']}]"
        for m in movies
    )


def user_msg_decide(state, turn, show_predictions):
    parts = [f"Turn {turn} of {N_TURNS}.", f"Current budget: {state.budget} / 10."]
    has_demo    = state.age is not None
    has_history = state.history_movies is not None and len(state.history_movies) > 0
    if has_demo:
        parts.append(f"\nUser demographics: Age {state.age}, Gender {state.gender}, Occupation {state.occupation}")
    if has_history:
        parts.append(f"\nUser's rated movie history ({len(state.history_movies)} movies):")
        parts.append(history_block(state.history_movies))
    if not has_demo and not has_history and not state.qa_history:
        parts.append("\nYou have no information about this user yet.")
    if state.qa_history:
        parts.append("\nInformation gathered so far:")
        for i, qa in enumerate(state.qa_history, 1):
            parts.append(f"  Question {i}: {qa['question']}")
            parts.append(f"  Answer {i}: {clip(qa['answer'], 200)}")
    if state.last_batch_result and show_predictions:
        parts.append("\nResults from last exploit turn:")
        for r in state.last_batch_result["details"]:
            outcome = "correct" if r["correct"] else "incorrect"
            parts.append(f"  \"{r['title']}\" - predicted {r['prediction']}, actual {r['actual']} - {outcome}")
        parts.append(
            f"  Correct: {state.last_batch_result['correct']} out of {state.last_batch_result['total']}. "
            f"Budget change: {state.last_batch_result['budget_change']:+d} points."
        )
    upcoming = state.peek_upcoming()
    if upcoming is not None:
        parts.append("\nIf you choose EXPLOIT, these are the 4 movies you will predict:")
        parts.append(movies_block(upcoming))
    parts.append("\nWill you EXPLORE or EXPLOIT?")
    return "\n".join(parts)


def user_msg_predict(state):
    parts = []
    has_demo    = state.pred_age is not None
    has_history = state.history_movies is not None and len(state.history_movies) > 0
    if has_demo:
        parts.append(f"USER DEMOGRAPHICS: Age {state.pred_age}, Gender {state.pred_gender}, Occupation {state.pred_occupation}")
    if has_history:
        parts.append(f"\nUSER RATED HISTORY ({len(state.history_movies)} movies):")
        parts.append(history_block(state.history_movies))
    if not has_demo and not has_history and not state.qa_history:
        parts.append("NO INFORMATION AVAILABLE ABOUT THIS USER.")
    if state.qa_history:
        parts.append("\nUSER PREFERENCES (from Q&A):")
        for qa in state.qa_history:
            parts.append(f"  Q: {qa['question']}")
            parts.append(f"  A: {clip(qa['answer'], 150)}")
    parts.append("\nMOVIES TO PREDICT:")
    for m in state.pending_movies:
        parts.append(f'  title="{m["title"]}" year={m["year"]} genres=[{m["genres"]}]')
    parts.append("\nOutput the JSON object now:")
    return "\n".join(parts)


FAKE_TITLES = {"The Matrix", "Gigli", "Inception", "Forrest Gump", "Pulp Fiction"}


class UserState:
    def __init__(self, user_id, exploit_pool, context_pool, simulator_data,
                 condition, movie_visibility, run_tag, out_root):
        self.user_id          = user_id
        self.simulator_data   = simulator_data
        self.condition        = condition
        self.movie_visibility = movie_visibility
        self.run_tag          = run_tag
        self.out_dir          = os.path.join(out_root, f"user_{user_id}", run_tag)

        age, gender, occupation = extract_demographics(exploit_pool)
        self.true_age        = age
        self.true_gender     = gender
        self.true_occupation = occupation

        self.shuffled_questions = QUESTIONS[:]
        random.shuffle(self.shuffled_questions)

        if condition in CONTEXT_CONDITIONS:
            self.history_movies, self.dominant_genre = select_history_diverse_balanced(
                context_pool, n=N_HISTORY
            )
        else:
            self.history_movies = None
            self.dominant_genre = "N/A"

        show_demo       = condition in DEMOGRAPHIC_CONDITIONS
        self.age        = age        if show_demo else None
        self.gender     = gender     if show_demo else None
        self.occupation = occupation if show_demo else None

        self.pred_age        = self.age
        self.pred_gender     = self.gender
        self.pred_occupation = self.occupation

        self.predict_pool        = exploit_pool.reset_index(drop=True)
        self.pool_idx            = 0
        self.budget              = INITIAL_BUDGET
        self.qa_history          = []
        self.turn_logs           = []
        self.all_prediction_rows = []
        self.last_batch_result   = None
        self.duplicate_retry_log = []
        self.done                = False
        self.resume_from_turn    = 1
        self.pending_movies      = None
        self._pending_log_entry  = None

    def peek_upcoming(self):
        if self.movie_visibility != "seen" or self.pool_idx >= len(self.predict_pool):
            return None
        peek_df = self.predict_pool.iloc[self.pool_idx: self.pool_idx + MOVIES_PER_TURN]
        upcoming = []
        for _, row in peek_df.iterrows():
            full_title = str(row["Title"]).strip()
            m_ = re.match(r"^(.*)\((\d{4})\)\s*$", full_title)
            t, y = (m_.group(1).strip(), int(m_.group(2))) if m_ else (full_title, 0)
            upcoming.append({"title": t, "year": y, "genres": str(row["Genres"])})
        return upcoming

    def next_movies(self):
        batch_df = self.predict_pool.iloc[self.pool_idx: self.pool_idx + MOVIES_PER_TURN]
        if len(batch_df) == 0:
            return None
        self.pool_idx += MOVIES_PER_TURN
        movies = []
        for _, row in batch_df.iterrows():
            full_title = str(row["Title"]).strip()
            m = re.match(r"^(.*)\((\d{4})\)\s*$", full_title)
            title, year = (m.group(1).strip(), int(m.group(2))) if m else (full_title, 0)
            movies.append({
                "title": title, "year": year,
                "genres": str(row["Genres"]), "label": str(row["Preference"]),
            })
        return movies

    def save_checkpoint(self, turn):
        os.makedirs(self.out_dir, exist_ok=True)
        ckpt = {
            "completed_turn":      turn,
            "pool_idx":            self.pool_idx,
            "budget":              self.budget,
            "qa_history":          self.qa_history,
            "turn_logs":           self.turn_logs,
            "all_prediction_rows": self.all_prediction_rows,
            "last_batch_result":   self.last_batch_result,
            "duplicate_retry_log": self.duplicate_retry_log,
            "shuffled_questions":  self.shuffled_questions,
        }
        with open(os.path.join(self.out_dir, "checkpoint.json"), "w") as f:
            json.dump(ckpt, f)

    def load_checkpoint(self):
        ckpt_path = os.path.join(self.out_dir, "checkpoint.json")
        if not os.path.exists(ckpt_path):
            return False
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        self.resume_from_turn    = ckpt["completed_turn"] + 1
        self.pool_idx            = ckpt["pool_idx"]
        self.budget              = ckpt["budget"]
        self.qa_history          = ckpt["qa_history"]
        self.turn_logs           = ckpt["turn_logs"]
        self.all_prediction_rows = ckpt["all_prediction_rows"]
        self.last_batch_result   = ckpt["last_batch_result"]
        self.duplicate_retry_log = ckpt["duplicate_retry_log"]
        self.shuffled_questions  = ckpt["shuffled_questions"]
        return True

    def delete_checkpoint(self):
        ckpt_path = os.path.join(self.out_dir, "checkpoint.json")
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)


def apply_decide(state, turn, text, explore_cost):
    forced   = state.budget < abs(explore_cost)
    decision, question = parse_decision(text)
    if decision is None or forced:
        decision = "EXPLOIT"
        forced   = True

    log_entry = {
        "turn": turn, "budget_before": state.budget,
        "decision": decision, "forced": forced,
    }

    if decision == "EXPLORE":
        if not question or len(question.strip()) < 5:
            decision = "EXPLOIT"
            log_entry["decision"] = "EXPLOIT"
            log_entry["note"]     = "explore_no_question"
        else:
            MAX_RETRIES = 3
            retry_count = 0
            current_q   = question
            resolved    = False

            while retry_count <= MAX_RETRIES:
                canonical_q, q_position = match_question_to_list(current_q, state.shuffled_questions)

                if canonical_q is None:
                    state.duplicate_retry_log.append({
                        "turn": turn, "attempt": retry_count + 1,
                        "raw_question": current_q, "canonical": None, "reason": "not_in_list",
                    })
                    decision = "EXPLOIT"
                    log_entry["decision"]     = "EXPLOIT"
                    log_entry["note"]         = "explore_question_not_in_list"
                    log_entry["raw_question"] = current_q
                    break

                if is_duplicate_question(canonical_q, state.qa_history):
                    retry_count += 1
                    state.duplicate_retry_log.append({
                        "turn": turn, "attempt": retry_count,
                        "raw_question": current_q, "canonical": canonical_q,
                        "reason": "duplicate", "question_position_in_list": q_position,
                    })
                    if retry_count > MAX_RETRIES:
                        decision = "EXPLOIT"
                        log_entry["decision"] = "EXPLOIT"
                        log_entry["note"]     = f"explore_duplicate_max_retries_{MAX_RETRIES}"
                        break
                    already_asked = [qa["question"] for qa in state.qa_history]
                    already_str   = "\n".join(f"    - {q}" for q in already_asked)
                    error_note    = (
                        f"\n\nERROR: You chose \"{canonical_q}\" but it has already been asked.\n"
                        f"Questions already asked:\n{already_str}\n"
                        f"Choose a DIFFERENT question that has NOT been asked yet."
                    )
                    retry_resp = CLIENT.models.generate_content(
                        model=MODEL_NAME,
                        contents=user_msg_decide(state, turn, show_predictions=True),
                        config=types.GenerateContentConfig(
                            system_instruction=build_sys_decide(
                                state.budget, state.shuffled_questions, explore_cost, EXPLOIT_GAIN
                            ) + error_note,
                            temperature=TEMPERATURE,
                            max_output_tokens=MAX_TOKENS_DECIDE,
                            top_p=0.9,
                            thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET_DECIDE),
                        ),
                    )
                    retry_text    = retry_resp.text.strip() if retry_resp.text else None
                    rd, rq        = parse_decision(retry_text)
                    if rd == "EXPLOIT":
                        decision = "EXPLOIT"
                        log_entry["decision"] = "EXPLOIT"
                        log_entry["note"]     = "explore_switched_to_exploit_on_retry"
                        resolved = True
                        break
                    current_q = rq or ""
                    if not current_q or len(current_q.strip()) < 5:
                        decision = "EXPLOIT"
                        log_entry["decision"] = "EXPLOIT"
                        log_entry["note"]     = "explore_no_question_on_retry"
                        break
                    continue

                else:
                    answer = lookup_answer(state.simulator_data, state.user_id, canonical_q)
                    state.qa_history.append({
                        "turn": turn, "question": canonical_q,
                        "raw_question": current_q, "answer": answer,
                        "question_position_in_list": q_position,
                        "retries_needed": retry_count,
                    })
                    state.budget                          += explore_cost
                    state.budget                           = max(state.budget, BUDGET_FLOOR)
                    log_entry["question"]                  = canonical_q
                    log_entry["raw_question"]              = current_q
                    log_entry["question_position_in_list"] = q_position
                    log_entry["answer"]                    = answer
                    log_entry["budget_change"]             = explore_cost
                    log_entry["budget_after"]              = state.budget
                    log_entry["retries_needed"]            = retry_count
                    state.turn_logs.append(log_entry)
                    state.last_batch_result = None
                    resolved = True
                    break

            if resolved and decision == "EXPLORE":
                return "EXPLORE"

    if decision == "EXPLOIT":
        movies = state.next_movies()
        if movies is None:
            log_entry["note"]         = "pool_exhausted"
            log_entry["budget_after"] = state.budget
            state.turn_logs.append(log_entry)
            state.done = True
            return "EXPLOIT"
        state.pending_movies     = movies
        state._pending_log_entry = log_entry
        return "EXPLOIT"

    return "EXPLORE"


def apply_predict(state, turn, text):
    log_entry = state._pending_log_entry
    movies    = state.pending_movies
    preds     = parse_predictions(text) if text else None

    if preds:
        cleaned = []
        for p in preds:
            if not isinstance(p, dict):
                continue
            title = p.get("title", "")
            if not isinstance(title, str):
                continue
            title = title.strip()
            if not title or title.startswith("<") or title in FAKE_TITLES:
                continue
            pred = p.get("prediction", "Like")
            cleaned.append({**p, "title": title,
                            "prediction": pred if isinstance(pred, str) else "Like"})
        preds = cleaned

    if not preds:
        log_entry["note"]         = "prediction_failed"
        log_entry["budget_after"] = state.budget
        state.turn_logs.append(log_entry)
        return

    matched = []
    for movie in movies:
        t_lower    = movie["title"].lower()
        pred_entry = next((p for p in preds if p.get("title", "").lower() == t_lower), None)
        if pred_entry is None:
            pred_entry = next(
                (p for p in preds if t_lower in p.get("title", "").lower()
                 or p.get("title", "").lower() in t_lower), None
            )
        raw        = (pred_entry or {}).get("prediction", "Like").strip().lower()
        prediction = "Dislike" if "dislike" in raw else "Like"
        matched.append({"title": movie["title"], "year": movie["year"],
                        "genres": movie["genres"], "prediction": prediction})

    correct      = sum(1 for p, m in zip(matched, movies) if p["prediction"] == m["label"])
    incorrect    = len(matched) - correct
    score_change = EXPLOIT_GAIN + correct * CORRECT_REWARD + incorrect * INCORRECT_REWARD
    state.budget = max(min(state.budget + score_change, INITIAL_BUDGET), BUDGET_FLOOR)

    details = []
    for p, m in zip(matched, movies):
        is_correct = p["prediction"] == m["label"]
        details.append({"title": p["title"], "prediction": p["prediction"],
                        "actual": m["label"], "correct": is_correct})
        state.all_prediction_rows.append({
            "turn": turn, "title": p["title"], "year": p["year"],
            "genres": p["genres"], "prediction": p["prediction"],
            "actual": m["label"], "correct": is_correct,
        })

    turn_accuracy  = correct / len(matched)
    turn_precision = compute_precision(details)
    state.last_batch_result = {
        "correct": correct, "total": len(matched),
        "budget_change": score_change, "details": details,
    }
    log_entry.update({
        "exploit_results": details, "correct": correct, "total": len(matched),
        "score_change": score_change, "accuracy": turn_accuracy,
        "precision": turn_precision, "budget_after": state.budget,
    })
    state.turn_logs.append(log_entry)


def save_user(state, explore_cost):
    out_dir = state.out_dir
    os.makedirs(out_dir, exist_ok=True)

    explore_turns   = sum(1 for t in state.turn_logs if t["decision"] == "EXPLORE")
    exploit_turns   = sum(1 for t in state.turn_logs if t["decision"] == "EXPLOIT")
    forced_turns    = sum(1 for t in state.turn_logs if t.get("forced") and t["decision"] == "EXPLOIT")
    question_counts = {}
    for qa in state.qa_history:
        question_counts[qa["question"]] = question_counts.get(qa["question"], 0) + 1

    full_log = {
        "user_id": int(state.user_id), "run_tag": state.run_tag,
        "condition": state.condition, "schema": SCHEMA,
        "model": MODEL_NAME, "thinking": THINKING,
        "movie_visibility": state.movie_visibility,
        "n_history": N_HISTORY, "nature": NATURE,
        "explore_cost": explore_cost, "exploit_gain": EXPLOIT_GAIN,
        "demographics": f"Age:{state.age}, Gender:{state.gender}, Occupation:{state.occupation}",
        "dominant_genre": state.dominant_genre,
        "shuffled_question_order": state.shuffled_questions,
        "turns": state.turn_logs,
        "duplicate_retry_log": state.duplicate_retry_log,
    }
    with open(os.path.join(out_dir, "turn_log.json"), "w") as f:
        json.dump(full_log, f, indent=2)

    if not state.all_prediction_rows:
        summary = {
            "user_id": int(state.user_id), "run_tag": state.run_tag,
            "condition": state.condition, "schema": SCHEMA,
            "model": MODEL_NAME, "thinking": THINKING,
            "movie_visibility": state.movie_visibility,
            "explore_turns": explore_turns, "exploit_turns": exploit_turns,
            "forced_exploit_turns": forced_turns, "final_budget": state.budget,
            "overall_accuracy": None, "overall_precision": None,
            "question_counts": question_counts, "error": "no_predictions",
        }
    else:
        pred_df      = pd.DataFrame(state.all_prediction_rows)
        pred_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
        overall_acc  = float(pred_df["correct"].mean())
        all_details  = [{"prediction": r["prediction"], "actual": r["actual"]}
                        for r in state.all_prediction_rows]
        overall_prec = compute_precision(all_details)
        exploit_turn_metrics = {
            t["turn"]: {"accuracy": t.get("accuracy"), "precision": t.get("precision")}
            for t in state.turn_logs if t["decision"] == "EXPLOIT" and "exploit_results" in t
        }
        summary = {
            "user_id": int(state.user_id), "run_tag": state.run_tag,
            "condition": state.condition, "schema": SCHEMA,
            "model": MODEL_NAME, "thinking": THINKING,
            "movie_visibility": state.movie_visibility,
            "n_history": N_HISTORY, "nature": NATURE,
            "explore_cost": explore_cost, "exploit_gain": EXPLOIT_GAIN,
            "demographics": f"Age:{state.true_age}, Gender:{state.true_gender}, Occupation:{state.true_occupation}",
            "demographics_shown_to_model": state.condition in DEMOGRAPHIC_CONDITIONS,
            "history_shown_to_model": state.condition in CONTEXT_CONDITIONS,
            "dominant_genre": state.dominant_genre, "total_turns": N_TURNS,
            "explore_turns": explore_turns, "exploit_turns": exploit_turns,
            "forced_exploit_turns": forced_turns, "total_questions": len(state.qa_history),
            "final_budget": state.budget,
            "overall_accuracy": overall_acc, "overall_precision": overall_prec,
            "metrics_per_exploit_turn": exploit_turn_metrics,
            "question_counts": question_counts,
            "shuffled_question_order": state.shuffled_questions,
            "qa_history": state.qa_history,
            "duplicate_retry_log": state.duplicate_retry_log,
            "total_duplicate_attempts": len(state.duplicate_retry_log),
            "duplicate_question_counts": {
                q: sum(1 for e in state.duplicate_retry_log if e["canonical"] == q)
                for q in set(e["canonical"] for e in state.duplicate_retry_log if e["canonical"] is not None)
            },
        }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    state.delete_checkpoint()


class AblationPool:

    def __init__(self, condition, movie_visibility, df, simulator_data):
        self.condition        = condition
        self.movie_visibility = movie_visibility
        self.run_tag          = run_tag_for(condition, movie_visibility)
        self.explore_cost     = EXPLORE_COST
        self.states           = []
        self.skipped          = 0
        self.already_done     = 0
        self.resumed          = 0

        for user_id in sorted(df["UserID"].unique().tolist()):
            summary_path = os.path.join(OUT_ROOT, f"user_{user_id}", self.run_tag, "summary.json")
            if os.path.exists(summary_path) and not IGNORE_SUMMARY:
                self.already_done += 1
                continue
            user_df      = df[df["UserID"] == user_id].reset_index(drop=True)
            exploit_pool = user_df[user_df["pool"] == "exploit"].reset_index(drop=True)
            context_pool = user_df[user_df["pool"] == "context"].reset_index(drop=True)
            if len(exploit_pool) < MIN_MOVIES_NEEDED:
                self.skipped += 1; continue
            if condition in CONTEXT_CONDITIONS and len(context_pool) < N_HISTORY:
                self.skipped += 1; continue
            if str(user_id) not in simulator_data and user_id not in simulator_data:
                self.skipped += 1; continue
            state = UserState(
                user_id, exploit_pool, context_pool, simulator_data,
                condition, movie_visibility, self.run_tag, OUT_ROOT,
            )
            if not RERUN and state.load_checkpoint():
                self.resumed += 1
            self.states.append(state)

    def active(self, turn):
        return [s for s in self.states if not s.done and s.resume_from_turn <= turn]

    def all_done(self):
        return all(s.done for s in self.states)


def main():
    global CLIENT

    with open("config.json") as f:
        cfg = json.load(f)

    CLIENT = genai.Client(api_key=cfg["GEMINI_API_KEY"])

    print(f"RL EXPLORE/EXPLOIT (schema-level batch) – schema={SCHEMA}")
    print(f"Model: {MODEL_NAME}  |  thinking: {THINKING}  |  poll_interval: {POLL_INTERVAL}s")
    print(f"Schema {SCHEMA}  |  explore={EXPLORE_COST:+d}  exploit={EXPLOIT_GAIN:+d}")
    print(f"Running {len(ABLATIONS)} ablations concurrently "
          f"({len(SELECTED_CONDITIONS)} conditions x 2 visibilities)")
    print("=" * 70)

    print("Loading dataset...")
    df = pd.read_csv("filtered_dataset_balanced.csv")
    print("Loading simulator data...")
    with open("user_simulator_output.json") as f:
        simulator_data = json.load(f)
    print(f"  Profiles: {len(simulator_data)}")
    os.makedirs(OUT_ROOT, exist_ok=True)

    pools = []
    for abl in ABLATIONS:
        pool = AblationPool(abl["condition"], abl["movie_visibility"], df, simulator_data)
        print(f"  [{pool.run_tag}]  done={pool.already_done}  resuming={pool.resumed}  "
              f"fresh={len(pool.states)-pool.resumed}  skipped={pool.skipped}")
        if pool.states:
            pools.append(pool)
        else:
            print(f"    → nothing to do, skipping")

    if not pools:
        print("All ablations complete.")
        return

    for turn in range(1, N_TURNS + 1):
        active_pools = [p for p in pools if p.active(turn)]
        if not active_pools:
            if all(p.all_done() for p in pools):
                break
            continue

        print(f"\n{'='*70}")
        print(f"TURN {turn}/{N_TURNS}  |  active ablations: {len(active_pools)}")
        print(f"{'='*70}")
        t0 = time.time()

        decide_jobs = []
        for pool in active_pools:
            active = pool.active(turn)
            need_llm = [s for s in active if s.budget >= abs(pool.explore_cost)]
            if not need_llm:
                decide_jobs.append((None, None, pool, active))
                continue
            requests = [
                build_batch_request(
                    build_sys_decide(s.budget, s.shuffled_questions, pool.explore_cost, EXPLOIT_GAIN),
                    user_msg_decide(s, turn, show_predictions=True),
                    MAX_TOKENS_DECIDE, THINKING_BUDGET_DECIDE,
                )
                for s in need_llm
            ]
            display_name = f"{pool.run_tag}_T{turn:02d}_DECIDE"
            job          = submit_batch(requests, display_name)
            decide_jobs.append((job, display_name, pool, active))

        jobs_to_poll = [(job, name) for job, name, _, _ in decide_jobs if job is not None]
        if jobs_to_poll:
            print(f"  Polling {len(jobs_to_poll)} DECIDE jobs ...")
            completed_decide = poll_all(jobs_to_poll)
            result_map = {}
            poll_idx = 0
            for job, name, pool, active in decide_jobs:
                if job is not None:
                    result_map[(pool.run_tag, "DECIDE")] = collect_results(completed_decide[poll_idx], name)
                    poll_idx += 1

        predict_jobs = []
        for job, name, pool, active in decide_jobs:
            active_need_llm = [s for s in active if s.budget >= abs(pool.explore_cost)]
            active_forced   = [s for s in active if s.budget < abs(pool.explore_cost)]
            texts           = result_map.get((pool.run_tag, "DECIDE")) if job is not None else None

            exploit_states = []
            explore_count  = 0

            for s in active_forced:
                decision = apply_decide(s, turn, None, pool.explore_cost)
                if decision == "EXPLOIT" and not s.done and s.pending_movies is not None:
                    exploit_states.append(s)

            for s, text in zip(active_need_llm, (texts or [None]*len(active_need_llm))):
                decision = apply_decide(s, turn, text, pool.explore_cost)
                if decision == "EXPLORE":
                    explore_count += 1
                elif not s.done and s.pending_movies is not None:
                    exploit_states.append(s)

            print(f"  [{pool.run_tag}] EXPLORE={explore_count}  EXPLOIT={len(exploit_states)}  "
                  f"done={sum(s.done for s in active)}")

            if not exploit_states:
                predict_jobs.append((None, None, pool, []))
                continue

            requests = [
                build_batch_request(
                    build_sys_predict(),
                    user_msg_predict(s),
                    MAX_TOKENS_PREDICT, THINKING_BUDGET_PREDICT,
                )
                for s in exploit_states
            ]
            display_name = f"{pool.run_tag}_T{turn:02d}_PREDICT"
            job          = submit_batch(requests, display_name)
            predict_jobs.append((job, display_name, pool, exploit_states))

        jobs_to_poll = [(job, name) for job, name, _, _ in predict_jobs if job is not None]
        if jobs_to_poll:
            print(f"  Polling {len(jobs_to_poll)} PREDICT jobs ...")
            completed_predict = poll_all(jobs_to_poll)
            poll_idx = 0
            for job, name, pool, exploit_states in predict_jobs:
                if job is not None:
                    texts = collect_results(completed_predict[poll_idx], name)
                    poll_idx += 1
                    for s, text in zip(exploit_states, (texts or [None]*len(exploit_states))):
                        apply_predict(s, turn, text)

        for pool in active_pools:
            for s in pool.active(turn):
                if turn == N_TURNS:
                    s.done = True
                elif not s.done:
                    s.save_checkpoint(turn)

        print(f"  Turn {turn} elapsed: {time.time()-t0:.1f}s")

    print(f"\n{'='*70}\nSaving results ...")
    for pool in pools:
        for state in tqdm(pool.states, desc=pool.run_tag, unit="user"):
            save_user(state, pool.explore_cost)

    print(f"\nALL DONE.")
    print("=" * 70)


def report_in_flight(reason):
    if not IN_FLIGHT:
        print(f"\n[shutdown] {reason}: no batches in flight.")
        return
    print(f"\n[shutdown] {reason}: {len(IN_FLIGHT)} batch(es) still in flight "
          f"— left running, recorded in {IN_FLIGHT_LOG}:")
    for job_name, display_name in list(IN_FLIGHT.items()):
        _record_in_flight(job_name, display_name, f"orphaned:{reason}")
        print(f"  {display_name}  ({job_name})")


def _on_signal(signum, frame):
    name = signal.Signals(signum).name
    report_in_flight(f"received {name}")
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    main()
