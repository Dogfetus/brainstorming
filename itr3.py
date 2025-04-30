import sqlite3
import numpy as np
from datetime import datetime
from bisect import bisect_right

DB_PATH = "new_data.db"
REFERENCE_DATE = datetime(2024, 10, 25)

PLAYER_USERNAMES = [
    "jimbosmx", "hintz", "spencer", "swagman", "paranoiaboi", "masongos",
    "grady", "jjk.", "chezmix", "inglomi", "jellyslosh", "wdrm", "eesa",
    "senpi", "janus5k", "tayman", "emcat", "pilot", "mxl100", "snowstorm",
    "datcoreedoe", "big matt", "werdwerdus", "cathadan", "shinobee",
    "mesyr", "zom585", "ctemi", "zephyrnobar", "sydia", "kren", "xosen",
    "sweetjohnnycage", "cheesecake", "ditz", "enderstevemc", "jiajiabinks",
    "meeko", "momosapien", "auby", "arual", "dogfetus", "noir", "dali",
    " peter", "jokko", "butterbutt", "jewel", "beatao", "maverick"
]

BASE_RATING = 800
K_FACTOR = 69 
MAX_DELTA = 10 #max delta set to 20 with k-factor 64 and exponent 1.15 gave mae at 2.52
MIN_RATING = 100 
MAX_RATING = 30000
DIFFICULTY_FIELD = "c.I_DIFF_7"  # <- just change this if you want to switch difficulty source
THRESHOLD = 2240

def connect_db():
    return sqlite3.connect(DB_PATH)

def fetch_scores_with_difficulty(difficulty_field=DIFFICULTY_FIELD):
    conn = connect_db()
    cursor = conn.cursor()
    
    query = f"""
        SELECT g.username, s.score, c._id, {difficulty_field}
        FROM scores s
        JOIN charts c ON s.chart_id = c._id
        JOIN gamers g ON s.gamer_id = g._id
        WHERE s.created_at < ?
            AND s.score IS NOT NULL
            AND {DIFFICULTY_FIELD} IS NOT NULL
            AND {DIFFICULTY_FIELD} != 0
            AND c.difficulty_display NOT IN ('full', 'dual', 'full+', 'dual+', 'full2', 'dual2')
    """
    
    cursor.execute(query, (REFERENCE_DATE.isoformat(),))
    rows = cursor.fetchall()
    conn.close()
    return rows

def build_chart_score_lookup(scores):
    chart_score_lookup = {}
    for _, score, chart_id, _ in scores:
        chart_score_lookup.setdefault(chart_id, []).append(score)
    for chart_id in chart_score_lookup:
        chart_score_lookup[chart_id].sort()
    return chart_score_lookup

def rank_itr3(
        player_list,
        difficulty_field,
        threshold=THRESHOLD,
        printing=False,
        k_factor=K_FACTOR,
        max_delta=MAX_DELTA,
        min_rating=MIN_RATING,
        max_rating=MAX_RATING,
        base_rating=BASE_RATING,
        exponent=1.05,
        chart_score_lookup=None,
        all_scores=None):

    def normalize_difficulty(difficulty, log_max, is_log_scale=True):
        if is_log_scale:
            return np.log1p(difficulty) / log_max
        else:
            return difficulty / log_max

    def percentile_score(score, sorted_scores):
        index = bisect_right(sorted_scores, score)
        percentile = index / len(sorted_scores)
        return np.clip(percentile, 0, 1)

    def elo_rating_system(all_scores, chart_score_lookup):
        ratings = {uname: BASE_RATING for uname in player_list}

        # Calculate max difficulty for normalization (once)
        max_difficulty = max(d for _, _, _, d in all_scores if d is not None)
        # Detect if values are large enough to benefit from log scaling
        is_log_scale = max_difficulty > 1500  # tweak this threshold if needed
        log_max = np.log1p(max_difficulty) if is_log_scale else max_difficulty


        for username, score, chart_id, difficulty in all_scores:
            uname = username.strip().lower()
            if uname not in ratings or chart_id not in chart_score_lookup:
                continue

            sorted_scores = chart_score_lookup[chart_id]
            if len(sorted_scores) < 2:
                continue

            player_rating = ratings[uname]

            # Normalize difficulty
            norm_difficulty = normalize_difficulty(difficulty, log_max, is_log_scale)
            chart_rating = norm_difficulty * threshold  # you can tune this scale

            # 1. Calculate actual score as percentile
            actual = percentile_score(score, sorted_scores)

            # 2. Calculate expected score from ELO formula
            expected = 1 / (1 + 10 ** ((chart_rating - player_rating) / 400))

            # 3. Difficulty-weighted scaled K-factor (soft curve)
            difficulty_weight = norm_difficulty ** exponent 
            scaled_k = k_factor * difficulty_weight

            # 4. Delta capped to avoid explosion
            delta = np.clip(scaled_k * (actual - expected), -max_delta, max_delta)

            # 5. Update rating
            new_rating = player_rating + delta
            ratings[uname] = max(MIN_RATING, min(new_rating, MAX_RATING))

        return sorted(ratings.items(), key=lambda x: x[1], reverse=True)


    if all_scores is None:
        all_scores = fetch_scores_with_difficulty(difficulty_field)
    if chart_score_lookup is None:
        chart_score_lookup = build_chart_score_lookup(all_scores)

    ranked_players = elo_rating_system(all_scores, chart_score_lookup)
    player_list = [player for player, _ in ranked_players]


    if printing:
        print("\nFinal Chart-as-Opponent ELO Ranking (Iteration 3.9 – Precision Tuned):")
        for rank, (player, rating) in enumerate(ranked_players, 1):
            print(f"\033[1;36m{rank:2}.\033[0m \033[1;33m{player:15}\033[0m: {rating:.2f}")

    return player_list 

