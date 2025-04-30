import sqlite3
from collections import Counter
from last_scores import best_x_scr_for_chart, get_played_charts
import score_evaluater 
import requests
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict
from typing import List, Tuple
from difficulty import song_from_chart



index = {
    "_id": 0,
    "calories": 1,
    "chart_id": 2,
    "cleared": 3,
    "created_at": 4,
    "early": 5,
    "flags": 6,
    "full_combo": 7,
    "gamer_id": 8,
    "global_flags": 9,
    "grade": 10,
    "green": 11,
    "late": 12,
    "max_combo": 13,
    "misses": 14,
    "music_speed": 15,
    "perfect1": 16,
    "perfect2": 17,
    "red": 18,
    "score": 19,
    "side": 20,
    "song_chart_id": 21,
    "steps": 22,
    "updated_at": 23,
    "uuid": 24,
    "yellow": 25
}

cindex = {
    "_id": 0,
    "created_at": 1,
    "difficulty": 2,
    "difficulty_display": 3,
    "difficulty_id": 4,
    "difficulty_name": 5,
    "game_difficulty_id": 6,
    "graph": 7,
    "id": 8,
    "is_enabled": 9,
    "meter": 10,
    "pass_count": 11,
    "play_count": 12,
    "song_id": 13,
    "steps_author": 14,
    "steps_index": 15,
    "updated_at": 16,
    "diff_1": 17,
    "diff_2": 18,
    "diff_3": 19,
    "diff_4": 20,
    "diff_5": 21,
    "diff_6": 22,
    "diff_7": 23,
}

value_of_difficulty = {
    "basic": 1,
    "easy": 2,
    "easy+": 3,
    "hard": 4,
    "hard+": 5,
    "wild": 6,
}






def get_charts():
    """
    Gets the charts from the database
    """
    conn = sqlite3.connect("new_data.db")
    cursor = conn.cursor()

    # Get the charts from the database
    cursor.execute("SELECT * FROM charts")
    charts = cursor.fetchall()

    conn.close()

    return charts



def rank_by_avg(player, date="2026-10-10"):
    """
    rank players based on a predefined score
    """

    played_charts = get_played_charts(player, date=date)
    total = 0

    if not played_charts:
        return 0

    for c in played_charts:
        # Get the chart id
        chart_id = c[cindex["_id"]]
        chart_diff = c[cindex["diff_7"]]

        best_score = best_x_scr_for_chart(player, chart_id, 1, date=date, ignore_cleared=True)

        if not best_score or len(best_score) == 0:
            continue


        score = best_score[0][1] * chart_diff
        total += score 

    return total / len(played_charts)



rank_by_avg("kramsen")
