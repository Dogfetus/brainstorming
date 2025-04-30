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
K_FACTOR = 64
MAX_DELTA = 20 #max delta set to 20 with k-factor 64 and exponent 1.15 gave mae at 2.52
MIN_RATING = 400
MAX_RATING = 3000

def connect_db():
    return sqlite3.connect(DB_PATH)

def fetch_scores_with_difficulty():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT g.username, s.score, c._id, c.difficulty
        FROM scores s
        JOIN charts c ON s.chart_id = c._id
        JOIN gamers g ON s.gamer_id = g._id
        WHERE s.created_at < ?
            AND s.score IS NOT NULL
            AND c.difficulty IS NOT NULL
            AND c.difficulty != 0
            AND c.difficulty_display NOT IN ('full', 'dual', 'full+', 'dual+', 'full2', 'dual2')
    """, (REFERENCE_DATE.isoformat(),))
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

def percentile_score(score, sorted_scores):
    index = bisect_right(sorted_scores, score)
    percentile = index / len(sorted_scores)
    return np.clip(percentile, 0, 1)

def elo_rating_system(all_scores, chart_score_lookup):
    ratings = {uname: BASE_RATING for uname in PLAYER_USERNAMES}

    for username, score, chart_id, difficulty in all_scores:
        uname = username.strip().lower()
        if uname not in ratings or chart_id not in chart_score_lookup:
            continue

        sorted_scores = chart_score_lookup[chart_id]
        if len(sorted_scores) < 2:
            continue

        player_rating = ratings[uname]
        chart_rating = difficulty * 100

        # 1. Calculate actual score as percentile
        actual = percentile_score(score, sorted_scores)

        # 2. Calculate expected score from ELO formula
        expected = 1 / (1 + 10 ** ((chart_rating - player_rating) / 400))

        # 3. Difficulty-weighted scaled K-factor (soft curve)
        difficulty_weight = (difficulty / 27.0) ** 1.15
        scaled_k = K_FACTOR * difficulty_weight

        # 4. Delta capped to avoid explosion
        delta = np.clip(scaled_k * (actual - expected), -MAX_DELTA, MAX_DELTA)

        # 5. Update rating
        new_rating = player_rating + delta
        ratings[uname] = max(MIN_RATING, min(new_rating, MAX_RATING))

    return sorted(ratings.items(), key=lambda x: x[1], reverse=True)

def main():
    all_scores = fetch_scores_with_difficulty()
    chart_score_lookup = build_chart_score_lookup(all_scores)
    ranked_players = elo_rating_system(all_scores, chart_score_lookup)

    print("\nFinal Chart-as-Opponent ELO Ranking (Iteration 3.9 – Precision Tuned):")
    for rank, (player, rating) in enumerate(ranked_players, 1):
        print(f"{rank:2}. {player:15} - Rating: {rating:.2f}")

if __name__ == "__main__":
    main()
