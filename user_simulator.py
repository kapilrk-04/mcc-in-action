"""
User Simulator — Pre-computed Q&A profiles for all users.

15 questions total:
  Q1–Q6   : Demographics
  Q7–Q12  : Preferences
  Q13–Q15 : Distractors (random, never-watched genres as placeholders)

Descriptive questions use sentence templates.
Yes/No questions return a short direct answer only.

Zip code → area classification uses a prefix-based population density
heuristic derived from US Census data (no network or library required).

Run:
    python user_simulator.py
Output:
    user_simulator_output.json
"""

import pandas as pd
import json
import random
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_SEED  = 42
INPUT_CSV    = "filtered_dataset_100_or_more_movies.csv"
OUTPUT_JSON  = "user_simulator_output.json"

random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# ZIP CODE → AREA CLASSIFICATION
# Based on US Census population density by 3-digit zip prefix.
# Dense urban cores, suburban rings, and rural zones are mapped below.
# Non-US / malformed codes fall back to "Not available".
# ─────────────────────────────────────────────────────────────────────────────

# Urban prefixes: major city cores (NYC, LA, Chicago, Houston, Philly, etc.)
URBAN_PREFIXES = {
    # New York City area
    "100", "101", "102", "103", "104", "111", "112", "113", "114", "116",
    # Los Angeles
    "900", "901", "902", "903", "904", "905", "906", "907", "908",
    # Chicago
    "606", "607", "608",
    # Houston
    "770", "771", "772",
    # Philadelphia
    "191", "192",
    # Phoenix
    "850", "851", "852", "853",
    # San Antonio
    "782",
    # San Diego
    "921", "922",
    # Dallas
    "752", "753",
    # San Jose / SF Bay Area
    "940", "941", "942", "943", "944", "945", "946", "947", "948", "949",
    # Washington DC
    "200", "201", "202", "203", "204", "205",
    # Boston
    "021", "022", "023", "024",
    # Seattle
    "981", "982",
    # Denver
    "802", "803",
    # Atlanta
    "303", "304",
    # Miami
    "331", "332", "333",
    # Minneapolis
    "554", "555",
    # Detroit
    "482", "483",
    # Baltimore
    "212", "211",
    # Portland OR
    "972", "973",
    # Las Vegas
    "891", "892",
    # Cleveland
    "441",
    # Pittsburgh
    "152",
    # St Louis
    "631", "632",
    # Tampa
    "336",
    # New Orleans
    "701",
    # Sacramento
    "958",
    # Kansas City
    "641",
    # Cincinnati
    "452",
    # Columbus OH
    "432",
    # Indianapolis
    "462",
    # Charlotte
    "282",
    # Memphis
    "381",
    # Louisville
    "402",
    # Richmond VA
    "232",
    # Hartford CT
    "061",
    # Providence RI
    "029",
    # Buffalo NY
    "142",
    # Rochester NY
    "146",
    # Salt Lake City
    "841",
    # Albuquerque
    "871",
    # Tucson
    "857",
    # Honolulu
    "968",
    # Anchorage
    "995",
    # Newark NJ
    "071",
    # Jersey City
    "073",
}

# Rural prefixes: states / regions that are predominantly rural
RURAL_PREFIXES = {
    # Montana
    "590", "591", "592", "593", "594", "595", "596", "597", "598", "599",
    # Wyoming
    "820", "821", "822", "823", "824", "825", "826", "827", "828", "829", "830",
    # North Dakota
    "580", "581", "582", "583", "584", "585", "586", "587", "588",
    # South Dakota
    "570", "571", "572", "573", "574", "575", "576", "577",
    # Alaska (non-Anchorage)
    "996", "997", "998", "999",
    # Vermont
    "050", "051", "052", "053", "054", "055", "056", "057", "058",
    # Maine
    "039", "040", "041", "042", "043", "044", "045", "046", "047", "048", "049",
    # Idaho (rural)
    "832", "833", "834", "835", "836", "837",
    # Nebraska (rural)
    "685", "686", "687", "688", "689", "690", "691", "692", "693",
    # Kansas (rural)
    "674", "675", "676", "677", "678", "679",
    # Oklahoma (rural)
    "739", "740", "741",
    # Mississippi (rural)
    "386", "387", "388", "389", "390", "391", "392", "393", "394", "395",
    # West Virginia
    "247", "248", "249", "250", "251", "252", "253", "254", "255", "256", "257", "258",
    # New Mexico (rural)
    "875", "876", "877", "878", "879",
    # Arkansas (rural)
    "716", "717", "718", "719",
    # Iowa (rural)
    "520", "521", "522", "523", "524", "525", "526",
}


def zip_to_area(zipcode: str) -> str:
    """
    Classify a US zip code as urban, suburban, or rural.
    Uses 3-digit prefix heuristic. Non-US / malformed → 'Not available'.
    """
    if not zipcode or not isinstance(zipcode, str):
        return "Not available"

    z = zipcode.strip().split("-")[0]   # handle ZIP+4 format
    if not z.isdigit() or len(z) < 3:
        return "Not available"

    prefix = z[:3]

    if prefix in URBAN_PREFIXES:
        return "urban"
    if prefix in RURAL_PREFIXES:
        return "rural"
    # Everything else: suburban (covers most of the US zip space)
    return "suburban"


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading dataset: {INPUT_CSV}")
movie_ds = pd.read_csv(INPUT_CSV)
user_ids = movie_ds["UserID"].unique()
print(f"  Users found: {len(user_ids)}")

all_genres = set()
for genres in movie_ds["Genres"]:
    all_genres.update(genres.split("|"))

# ─────────────────────────────────────────────────────────────────────────────
# BUILD BASE USER DICT
# ─────────────────────────────────────────────────────────────────────────────

user_sim_dict = {}
for user_id in tqdm(user_ids, desc="Building base user dict"):
    user_movies = movie_ds[movie_ds["UserID"] == user_id]
    first_row   = user_movies.iloc[0]
    user_sim_dict[user_id] = {
        "age_group":  first_row["Age"],
        "gender":     first_row["Gender"],
        "occupation": first_row["Occupation"],
        "zipcode": str(first_row.get("ZipCode", "")).strip(),
        "movies": {
            row["Title"]: {
                "rating": row["Rating"],
                "genres": row["Genres"].split("|"),
            }
            for _, row in user_movies.iterrows()
        },
    }

# ─────────────────────────────────────────────────────────────────────────────
# Q1 — What is your age group?   [Descriptive]
# Template: "My age group is []."
# ─────────────────────────────────────────────────────────────────────────────

AGE_GROUP_LABELS = {
    "Under 18": "Under 18",
    "18-24":    "18-24",
    "25-34":    "25-34",
    "35-44":    "35-44",
    "45-49":    "45-49",
    "50-55":    "50-55",
    "56+":      "56 and above",
}

for user_id in user_ids:
    raw = str(user_sim_dict[user_id]["age_group"]).strip()
    label = AGE_GROUP_LABELS.get(raw, raw)
    user_sim_dict[user_id]["age_group"]        = label
    user_sim_dict[user_id]["age_group_answer"] = f"My age group is {label}."

# ─────────────────────────────────────────────────────────────────────────────
# Q2 — What is your gender?   [Descriptive]
# Template: "My gender is []."
# ─────────────────────────────────────────────────────────────────────────────

GENDER_LABELS = {"M": "Male", "F": "Female"}

for user_id in user_ids:
    raw    = str(user_sim_dict[user_id]["gender"]).strip()
    label  = GENDER_LABELS.get(raw, raw)
    user_sim_dict[user_id]["gender"]        = label
    user_sim_dict[user_id]["gender_answer"] = f"My gender is {label}."

# ─────────────────────────────────────────────────────────────────────────────
# Q3 — What is your occupation?   [Descriptive]
# Template: "My occupation is []."
# ─────────────────────────────────────────────────────────────────────────────

for user_id in user_ids:
    occ = str(user_sim_dict[user_id]["occupation"]).strip()
    user_sim_dict[user_id]["occupation_answer"] = f"My occupation is {occ}."

# ─────────────────────────────────────────────────────────────────────────────
# Q4 — How would you describe your area?   [Yes/No choice]
# Answer: just the classification word.
# ─────────────────────────────────────────────────────────────────────────────

for user_id in user_ids:
    zipcode = user_sim_dict[user_id]["zipcode"]
    area    = zip_to_area(zipcode)
    user_sim_dict[user_id]["area"]        = area
    user_sim_dict[user_id]["area_answer"] = area

# ─────────────────────────────────────────────────────────────────────────────
# Q5 — Work type   [Yes/No choice]
# Answer: just the work type label.
# ─────────────────────────────────────────────────────────────────────────────

OCCUPATION_WORK_TYPE = {
    "college/grad student": "intellectual/creative",
    "academic/educator":    "intellectual/creative",
    "writer":               "intellectual/creative",
    "programmer":           "intellectual/creative",
    "artist":               "intellectual/creative",
    "scientist":            "intellectual/creative",
    "K-12 student":         "intellectual/creative",
    "lawyer":               "intellectual/creative",
    "technician/engineer":  "practical/hands-on",
    "tradesman/craftsman":  "practical/hands-on",
    "farmer":               "practical/hands-on",
    "homemaker":            "practical/hands-on",
    "executive/managerial": "service-oriented",
    "sales/marketing":      "service-oriented",
    "self-employed":        "service-oriented",
    "doctor/health care":   "service-oriented",
    "clerical/admin":       "service-oriented",
    "customer service":     "service-oriented",
    "retired":              "service-oriented",
    "unemployed":           "service-oriented",
    "other":                "service-oriented",
}

for user_id in user_ids:
    occ       = str(user_sim_dict[user_id]["occupation"]).strip()
    work_type = OCCUPATION_WORK_TYPE.get(occ, "service-oriented")
    user_sim_dict[user_id]["work_type"]        = work_type
    user_sim_dict[user_id]["work_type_answer"] = work_type

# ─────────────────────────────────────────────────────────────────────────────
# Q6 — Cinema generation   [Yes/No choice]
# Answer: just the era label.
# ─────────────────────────────────────────────────────────────────────────────

AGE_TO_ERA_KEY = {
    "Under 18":    "Under 18",
    "18-24":       "18-24",
    "25-34":       "25-34",
    "35-44":       "35-44",
    "45-49":       "45-49",
    "50-55":       "50-55",
    "56 and above":"56+",
}

CINEMA_GENERATION = {
    "Under 18": "streaming/digital era (post-1990s)",
    "18-24":    "streaming/digital era (post-1990s)",
    "25-34":    "blockbuster/VHS era (1970s-1990s)",
    "35-44":    "blockbuster/VHS era (1970s-1990s)",
    "45-49":    "blockbuster/VHS era (1970s-1990s)",
    "50-55":    "blockbuster/VHS era (1970s-1990s)",
    "56+":      "cinema as a cultural centerpiece (pre-1970s)",
}

for user_id in user_ids:
    age_label = user_sim_dict[user_id]["age_group"]
    era_key   = AGE_TO_ERA_KEY.get(age_label, age_label)
    era       = CINEMA_GENERATION.get(era_key, "streaming/digital era (post-1990s)")
    user_sim_dict[user_id]["cinema_generation"]        = era
    user_sim_dict[user_id]["cinema_generation_answer"] = era

# ─────────────────────────────────────────────────────────────────────────────
# GENRE RATINGS — used by Q7, Q8, Q12
# ─────────────────────────────────────────────────────────────────────────────

user_genre_ratings = {}
for user_id in tqdm(user_ids, desc="Computing genre ratings"):
    genre_ratings = {}
    for title, data in user_sim_dict[user_id]["movies"].items():
        for genre in data["genres"]:
            entry = genre_ratings.setdefault(genre, {"count": 0, "total_rating": 0.0})
            entry["count"]        += 1
            entry["total_rating"] += data["rating"]
    for genre in genre_ratings:
        genre_ratings[genre]["average_rating"] = (
            genre_ratings[genre]["total_rating"] / genre_ratings[genre]["count"]
        )
    user_genre_ratings[user_id] = genre_ratings

# ─────────────────────────────────────────────────────────────────────────────
# Q7 — Top 3 favourite genres   [Descriptive]
# Template: "My top 3 favourite genres are [], [], and []."
# ─────────────────────────────────────────────────────────────────────────────

for user_id in user_ids:
    gr            = user_genre_ratings[user_id]
    sorted_genres = sorted(gr.items(), key=lambda x: x[1]["average_rating"], reverse=True)
    top3          = [g for g, _ in sorted_genres[:3]]

    if len(top3) >= 3:
        phrase = f"My top 3 favourite genres are {top3[0]}, {top3[1]}, and {top3[2]}."
    elif len(top3) == 2:
        phrase = f"My favourite genres are {top3[0]} and {top3[1]}."
    else:
        phrase = f"My favourite genre is {top3[0]}." if top3 else "I have no strong genre preference."

    user_sim_dict[user_id]["top3_genres"]        = top3
    user_sim_dict[user_id]["top3_genres_answer"] = phrase

# ─────────────────────────────────────────────────────────────────────────────
# Q8 — Disliked/avoided genres   [Descriptive]
# Template: "I tend to avoid [], [], and []."
# ─────────────────────────────────────────────────────────────────────────────

for user_id in user_ids:
    gr            = user_genre_ratings[user_id]
    sorted_genres = sorted(gr.items(), key=lambda x: x[1]["average_rating"], reverse=True)
    avoided       = [g for g in all_genres if g not in gr]
    low3          = [g for g, _ in sorted_genres[-3:]]
    disliked      = (avoided + low3)[:3]

    if disliked:
        phrase = f"I tend to avoid {', '.join(disliked[:3])}."
    else:
        phrase = "I do not particularly avoid any genre."

    user_sim_dict[user_id]["disliked_genres"]        = disliked
    user_sim_dict[user_id]["disliked_genres_answer"] = phrase

# ─────────────────────────────────────────────────────────────────────────────
# OVERALL RATINGS — used by Q9
# ─────────────────────────────────────────────────────────────────────────────

user_overall_ratings = {}
for user_id in user_ids:
    gr           = user_genre_ratings[user_id]
    total_rating = sum(r["total_rating"] for r in gr.values())
    total_count  = sum(r["count"]        for r in gr.values())
    avg          = total_rating / total_count if total_count else 0
    std          = pd.Series([r["average_rating"] for r in gr.values()]).std() or 0

    if avg > 4.0 and std < 0.4:
        category = "easy to please"
    elif avg < 3.2 and std > 0.8:
        category = "hard to impress"
    else:
        category = "somewhere in between"

    user_overall_ratings[user_id] = {
        "average_rating": avg,
        "std_dev":        std,
        "category":       category,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Q9 — Easy to please or hard to impress?   [Yes/No choice]
# Answer: just the category.
# ─────────────────────────────────────────────────────────────────────────────

for user_id in user_ids:
    cat = user_overall_ratings[user_id]["category"]
    user_sim_dict[user_id]["viewer_type"]        = cat
    user_sim_dict[user_id]["viewer_type_answer"] = cat


# ─────────────────────────────────────────────────────────────────────────────
# Q10 — Are you an avid, moderate, or light movie watcher?   [Yes/No choice]
# Categories determined by number of movies rated (equal-thirds quantile split).
# ─────────────────────────────────────────────────────────────────────────────

movie_counts  = {user_id: len(user_sim_dict[user_id]["movies"]) for user_id in user_ids}
counts_series = pd.Series(list(movie_counts.values()))

low_thresh  = counts_series.quantile(1/3)   # bottom third -- light
high_thresh = counts_series.quantile(2/3)   # top third -- avid

for user_id in user_ids:
    count = movie_counts[user_id]
    if count <= low_thresh:
        freq = "light"
    elif count <= high_thresh:
        freq = "moderate"
    else:
        freq = "avid"

    user_sim_dict[user_id]["watch_frequency"]        = freq
    user_sim_dict[user_id]["watch_frequency_answer"] = freq

# ─────────────────────────────────────────────────────────────────────────────
# Q11 — Classic or modern films?   [Yes/No choice]
# Answer: just the era preference label.
# ─────────────────────────────────────────────────────────────────────────────

for user_id in user_ids:
    titles    = list(user_sim_dict[user_id]["movies"].keys())
    old_count = new_count = 0
    for title in titles:
        try:
            year = int(title.strip()[-5:-1])
            if year < 1990:
                old_count += 1
            else:
                new_count += 1
        except (ValueError, IndexError):
            pass

    if old_count > new_count:
        era = "classic films"
    elif new_count > old_count:
        era = "modern productions"
    else:
        era = "both equally"

    user_sim_dict[user_id]["movie_era_preference"]        = era
    user_sim_dict[user_id]["movie_era_preference_answer"] = era
    user_sim_dict[user_id]["_era_old_count"]              = old_count
    user_sim_dict[user_id]["_era_new_count"]              = new_count

# ─────────────────────────────────────────────────────────────────────────────
# Q12 — Consistent ratings or varied within favourite genres?   [Yes/No choice]
# Answer: just the consistency label.
# ─────────────────────────────────────────────────────────────────────────────

for user_id in user_ids:
    top3_names = user_sim_dict[user_id]["top3_genres"]
    movies     = user_sim_dict[user_id]["movies"]

    ratings_in_top = [
        data["rating"]
        for data in movies.values()
        for genre in data["genres"]
        if genre in top3_names
    ]

    std_dev = pd.Series(ratings_in_top).std() if ratings_in_top else 0

    if std_dev < 0.4:
        consistency = "consistent"
    elif std_dev > 0.8:
        consistency = "varies a lot"
    else:
        consistency = "somewhat consistent"

    user_sim_dict[user_id]["genre_quality_consistency"]        = consistency
    user_sim_dict[user_id]["genre_quality_consistency_answer"] = consistency
    user_sim_dict[user_id]["_std_dev_top_genres"]              = round(float(std_dev), 4)

# ─────────────────────────────────────────────────────────────────────────────
# Q13 — What is your favorite color?   [Descriptive — distractor]
# Template: "My favourite color is []."
# ─────────────────────────────────────────────────────────────────────────────

COLORS = ["violet", "indigo", "blue", "green", "yellow", "orange", "red"]

for user_id in user_ids:
    color = random.choice(COLORS)
    user_sim_dict[user_id]["favorite_color"]        = color
    user_sim_dict[user_id]["favorite_color_answer"] = f"My favourite color is {color}."

# ─────────────────────────────────────────────────────────────────────────────
# Q14 — What is your lucky number?   [Yes/No — distractor]
# Answer: just the number.
# ─────────────────────────────────────────────────────────────────────────────

for user_id in user_ids:
    number = random.randint(1, 100)
    user_sim_dict[user_id]["lucky_number"]        = number
    user_sim_dict[user_id]["lucky_number_answer"] = number

# ─────────────────────────────────────────────────────────────────────────────
# Q15 — Tea or coffee?   [Yes/No — distractor]
# Answer: just the choice.
# ─────────────────────────────────────────────────────────────────────────────

for user_id in user_ids:
    drink = random.choice(["tea", "coffee"])
    user_sim_dict[user_id]["preferred_drink"]        = drink
    user_sim_dict[user_id]["preferred_drink_answer"] = drink

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS BLOCK — per-user statistics stored alongside Q&A
# ─────────────────────────────────────────────────────────────────────────────

def build_analysis(user_id):
    u   = user_sim_dict[user_id]
    gr  = user_genre_ratings[user_id]
    ora = user_overall_ratings[user_id]

    sorted_genres = sorted(gr.items(), key=lambda x: x[1]["average_rating"], reverse=True)

    genre_breakdown = {
        genre: {
            "movies_rated": data["count"],
            "avg_rating":   round(data["average_rating"], 2),
        }
        for genre, data in sorted_genres
    }

    top3_breakdown = [
        {"genre": g, "avg_rating": round(d["average_rating"], 2), "movies_rated": d["count"]}
        for g, d in sorted_genres[:3]
    ]
    low3_breakdown = [
        {"genre": g, "avg_rating": round(d["average_rating"], 2), "movies_rated": d["count"]}
        for g, d in sorted_genres[-3:]
    ]
    never_watched = [g for g in all_genres if g not in gr]

    return {
        "total_movies_rated":               len(u["movies"]),
        "overall_avg_rating":               round(ora["average_rating"], 3),
        "overall_rating_std_across_genres": round(float(ora["std_dev"]), 3),
        "pre_2000_movies":                  u["_era_old_count"],
        "post_2000_movies":                 u["_era_new_count"],
        "std_dev_within_top3_genres":       u["_std_dev_top_genres"],
        "top3_genres_breakdown":            top3_breakdown,
        "low_rated_genres_breakdown":       low3_breakdown,
        "never_watched_genres":             never_watched,
        "genre_breakdown":                  genre_breakdown,
    }

# ─────────────────────────────────────────────────────────────────────────────
# BUILD FINAL OUTPUT
# Maps each question string → answer string
# ─────────────────────────────────────────────────────────────────────────────

QUESTIONS = [
    # (question_text, answer_key)
    ("What is your age group?",
     "age_group_answer"),
    ("What is your gender?",
     "gender_answer"),
    ("What is your occupation?",
     "occupation_answer"),
    ("How would you describe your area — urban, suburban, or rural?",
     "area_answer"),
    ("Would you describe your work as more intellectual/creative, practical/hands-on, or service-oriented?",
     "work_type_answer"),
    ("Are you part of a generation that grew up with cinema as a cultural centerpiece (pre-1970s), the blockbuster/VHS era (1970s-1990s), or the streaming/digital era (post-1990s)?",
     "cinema_generation_answer"),
    ("What are your top 3 favourite movie genres?",
     "top3_genres_answer"),
    ("Which genres do you tend to dislike or avoid?",
     "disliked_genres_answer"),
    ("Overall, are you an easy-to-please viewer or a hard-to-impress one?",
     "viewer_type_answer"),
    ("Are you an avid, moderate, or light movie watcher?",
     "watch_frequency_answer"),
    ("Do you tend to enjoy older classic films, or do you gravitate more toward modern productions?",
     "movie_era_preference_answer"),
    ("Do you tend to rate movies within your favourite genres consistently highly, or does quality vary a lot even within genres you like?",
     "genre_quality_consistency_answer"),
    ("What is your favorite color?",
     "favorite_color_answer"),
    ("What is your lucky number?",
     "lucky_number_answer"),
    ("Do you prefer tea or coffee?",
     "preferred_drink_answer"),
]

output = {}
for user_id in tqdm(user_ids, desc="Building output"):
    u     = user_sim_dict[user_id]
    entry = {q: u[key] for q, key in QUESTIONS}
    entry["analysis"] = build_analysis(user_id)
    output[int(user_id)] = entry

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved {len(output)} user profiles to '{OUTPUT_JSON}'.")

# ── Preview first user ────────────────────────────────────────────────────────
first_id = list(output.keys())[0]
print(f"\nSample — User {first_id}:")
for q, a in output[first_id].items():
    if q == "analysis":
        print("  [analysis block omitted from preview]")
    else:
        print(f"  Q: {q}")
        print(f"  A: {a}")
        print()