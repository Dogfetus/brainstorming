import sqlite3
from collections import Counter
import last_scores
import score_evaluater 
import requests
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict
from typing import List, Tuple





EXTRA_EP = """https://api.smx.573.no/extra/{}"""
DATE = "2024-10-17"

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
}

# value_of_difficulty = {
#     "basic": 1,
#     "easy": 2,
#     "easy+": 2**2,
#     "hard": 2**3,
#     "hard+": 2**4,
#     "wild": 2**5,
# }

value_of_difficulty = {
    "basic": 1,
    "easy": 2,
    "easy+": 3,
    "hard": 4,
    "hard+": 5,
    "wild": 6,
}


# to add difficulty to database run this:
def save_difficulty(db_path: str, chart_difficulties: List[Tuple[int, float]]) -> None:
    """
    Add a single new I_DIFF_X column to the charts table and populate it with values
    for multiple charts.
    
    Args:
        db_path: Path to the SQLite database file
        chart_difficulties: List of tuples (chart_id, difficulty_value)
        both chart_id and difficulty_value should be single integers (or float for difficulty_value)
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get the highest existing I_DIFF column number
    cursor.execute("PRAGMA table_info(charts)")
    table_info = cursor.fetchall()
    
    # Find the highest existing I_DIFF column number
    max_idiff = 0
    for column in table_info:
        column_name = column[1]
        if column_name.startswith('I_DIFF_'):
            try:
                idiff_num = int(column_name.split('_')[2])
                max_idiff = max(max_idiff, idiff_num)
            except (IndexError, ValueError):
                continue
    
    # Create the new column name with the next number
    next_idiff = max_idiff + 1
    new_column = f"I_DIFF_{next_idiff}"
    
    # Add the new column
    cursor.execute(f"ALTER TABLE charts ADD COLUMN {new_column} REAL")
    print(f"Added new column: {new_column}")
    
    # Update each chart's difficulty value in the new column
    for chart_id, difficulty_value in chart_difficulties:
        update_query = f"UPDATE charts SET {new_column} = ? WHERE _id = ?"
        cursor.execute(update_query, (difficulty_value, chart_id))
        print(f"Updated chart ID {chart_id} with difficulty {difficulty_value}")
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    print(f"Successfully added column {new_column} and populated it with {len(chart_difficulties)} chart difficulties")




# get the graph for a score
def read_graph(score):
    if isinstance(score, int):
        graph = requests.get(EXTRA_EP.format(score)).json()
    elif isinstance(score, tuple):
        graph = requests.get(EXTRA_EP.format(score[0])).json()
    else:
        return None

    # check if the graph has the offsets
    if "offsets" not in graph:
        return None
    time = graph["offsets"][1:][0::3]
    offsets = graph["offsets"][1:][1::3]
    judgement = graph["offsets"][1:][2::3]
    return time, offsets, judgement




def avg_best_score_for_chart(chart_id, cleared=1, force_cleared=False):
    conn = sqlite3.connect("new_data.db")
    cursor = conn.cursor()
    query = """
        SELECT AVG(best_score) 
        FROM (
            SELECT MAX(score) AS best_score
            FROM scores
            WHERE chart_id = ?
    """

    params = [chart_id]

    if force_cleared:
        query += " AND cleared = ?"
        params.append(cleared)

    query += """ 
            GROUP BY gamer_id
        )
    """

    cursor.execute(query, params)
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


avg_best_score_for_chart(3477)



# get all scores for a chart
def scores_for_all_charts():
    conn = sqlite3.connect("new_data.db")
    cursor = conn.cursor()
    query = """  
        SELECT scores.* 
        FROM scores
        JOIN charts ON scores.chart_id = charts._id;
    """
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    return results



def steps_for_chart(chart_id):
    conn = sqlite3.connect("new_data.db")
    cursor = conn.cursor()
    query = """  
        SELECT MAX(scores.steps) 
        FROM scores 
        JOIN charts ON scores.chart_id = charts._id
        WHERE charts._id = ?;
    """
    cursor.execute(query, (chart_id,))
    results = cursor.fetchall()
    conn.close()
    return results[0][0] if results else 0


def avg_score_for_chart_per_user(chart_id, cleared=1):
    conn = sqlite3.connect("new_data.db")
    cursor = conn.cursor()
    query = """  
        SELECT scores.gamer_id, AVG(scores.score) 
        FROM scores 
        JOIN charts ON scores.chart_id = charts._id
        WHERE charts._id = ?
        GROUP BY scores.gamer_id;
    """
    cursor.execute(query, (chart_id,))
    results = cursor.fetchall()
    conn.close()
    return results


def avg_score_for_chart(chart_id, cleared=1):
    conn = sqlite3.connect("new_data.db")
    cursor = conn.cursor()
    query = """  
        SELECT AVG(scores.score) 
        FROM scores 
        JOIN charts ON scores.chart_id = charts._id
        WHERE charts._id = ?
        AND scores.cleared = ?;
    """
    cursor.execute(query, (chart_id, cleared))
    results = cursor.fetchall()
    conn.close()
    return results[0][0] if results else 0


def scores_for_chart(chart_id, cleared=1, player=None):
    conn = sqlite3.connect("new_data.db")
    cursor = conn.cursor()

    if player:
        query = """  
            SELECT scores.* 
            FROM scores
            JOIN charts ON scores.chart_id = charts._id
            WHERE charts._id = ?
            AND scores.cleared = ?
            AND scores.gamer_id = ?;
        """
        cursor.execute(query, (chart_id, cleared, player))


    else:
        query = """  
            SELECT scores.* 
            FROM scores
            JOIN charts ON scores.chart_id = charts._id
            WHERE charts._id = ?
            AND scores.cleared = ?;
        """
        cursor.execute(query, (chart_id, cleared))

    results = cursor.fetchall()
    conn.close()
    return results


def fulls():
    conn = sqlite3.connect("new_data.db")
    cursor = conn.cursor()
    query = """  
        SELECT charts.* 
        FROM charts
        WHERE charts.difficulty_display IN ('full', 'full+');
    """
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    return results


def charts(additional_exclude=[]):
    conn = sqlite3.connect("new_data.db")
    cursor = conn.cursor()
    exclude = ['dual', 'dual+', 'full', 'full+', 'edit'] + additional_exclude
    
    placeholders = ','.join('?' for _ in exclude)

    query = f"""  
        SELECT charts.* 
        FROM charts
        WHERE charts.difficulty_display NOT IN ({placeholders});
    """

    # Execute the query with the values provided
    cursor.execute(query, exclude)
    
    results = cursor.fetchall()
    conn.close()
    return results



def song_from_chart(chart):
    if isinstance(chart, tuple):
        chart_id = chart[cindex["id"]]
    else:
        chart_id = chart
    conn = sqlite3.connect("new_data.db")
    cursor = conn.cursor()
    query = """  
        SELECT songs.title 
        FROM songs
        JOIN charts ON songs._id = charts.song_id
        WHERE charts._id = ?;
    """
    cursor.execute(query, (chart_id,))
    results = cursor.fetchall()
    conn.close()
    return results[0][0] if results else None



def get_players():
    conn = sqlite3.connect("new_data.db")
    cursor = conn.cursor()
    query = """  
        SELECT DISTINCT(gamer_id) 
        FROM scores;
    """
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    return [x[0] for x in results]






































def get_clear_ratios_per_player():
    conn = sqlite3.connect("new_data.db")
    cursor = conn.cursor()

    query = """
    SELECT
        gamer_id,
        COUNT(*) AS attempts,
        SUM(CASE WHEN cleared = 1 THEN 1 ELSE 0 END) AS clears,
        ROUND(1.0 * SUM(CASE WHEN cleared = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS clear_ratio
    FROM scores
    WHERE cleared IS NOT NULL
    GROUP BY gamer_id
    """
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    return results  # Each row is (gamer_id, attempts, clears, clear_ratio)




def average_clear_ratio_for_chart(chart_id):
    conn = sqlite3.connect("new_data.db")
    cursor = conn.cursor()

    # Get all scores for this chart
    query = """
    SELECT gamer_id, cleared
    FROM scores
    WHERE chart_id = ?
    AND cleared IS NOT NULL
    """
    cursor.execute(query, (chart_id,))
    rows = cursor.fetchall()
    conn.close()

    # Count clears/attempts per player
    player_attempts = {}
    for gamer_id, cleared in rows:
        if gamer_id not in player_attempts:
            player_attempts[gamer_id] = {"clears": 0, "attempts": 0}
        player_attempts[gamer_id]["attempts"] += 1
        if cleared:
            player_attempts[gamer_id]["clears"] += 1

    # Calculate per-player clear ratios
    ratios = []
    for stats in player_attempts.values():
        if stats["attempts"] > 0:
            ratios.append(stats["clears"] / stats["attempts"])

    # Average the ratios
    return sum(ratios) / len(ratios) if ratios else 0.0





print(avg_best_score_for_chart(3477))

def create_difficulty(chart):
    id = chart[cindex["id"]]

    # ratio = average_clear_ratio_for_chart(id)
    #
    # return 1/ratio if ratio != 0 else -1 


    # clear ratio
    # play_count = chart[cindex["play_count"]]
    # pass_count = chart[cindex["pass_count"]]
    # clear_ratio = pass_count / play_count if play_count != 0 else 0


    # get average clear ratio per player for a chart


    # median score per user (does not work)
    # medians = []
    # players = get_players()
    # for player in players:
    #     scores = scores_for_chart(id, player=player)
    #     if scores:
    #         sort = sorted(scores, key=lambda x: x[index["score"]], reverse=True)
    #         median = sort[len(sort) // 2]
    #         medians.append(median[index["score"]])




    # scores = scores_for_chart(id)
    # if scores:
    #     avg = sum(score[index["score"]] for score in scores) / len(scores)
    # else:
    #     avg = 0

    # avg = avg_score_for_chart(id)
    #
    # if avg:
    #     b = avg / 100000
    # else:
    #     b = 0


    avges = avg_score_for_chart_per_user(id, 1)
    if avges:
        avg = sum(score[1] for score in avges) / len(avges)
    #     b = avg / 100000
    else:    
        avg = 0
    #     b=0
    #
    # steps = steps_for_chart(id)
    #
    # print("AVG: ", avg)
    #
    # steps
    steps = steps_for_chart(id)
    
    # yesavg = avg_best_score_for_chart(id)
    # if yesavg:
    #     avg = yesavg
    # else:
    #     avg = 0

    # combined
    # difficulty_score = (avg / 100000) * clear_ratio
    # difficulty_score = clear_ratio
    # difficulty_score = clear_ratio + (avg / 100000)
    # return difficulty_score
    # return steps
    # return (1/clear_ratio if clear_ratio != 0 else 0) + (1/avg if avg != 0 else 0) 
    # return (12.5/clear_ratio if clear_ratio != 0 else 0) + steps
    # return ((1/difficulty_score if difficulty_score != 0 else 0) * steps * (1000*value_of_difficulty[chart[cindex["difficulty_display"]]])) * 0.001
    # return steps
    # return 1/avg if avg != 0 else 0 
    # return (100_000 - avg)   # Scale down the values
    # return (100_000 - avg)**2 * steps**2 * value_of_difficulty[chart[cindex["difficulty_display"]]] * chart[cindex["difficulty"]] * 0.001 
    return ((100_000 - avg) * steps ** chart[cindex["difficulty"]]) * 0.0000001 
    # return (100_000 - avg)
    # return (1/b if b != 0 else 0) + 0.1 * value_of_difficulty[chart[cindex["difficulty_display"]]]


    # mode
    # if scores:
    #     score_values = [score[index["score"]] for score in scores]
    #     counter = Counter(score_values)
    #     most_common = counter.most_common(1)
    # else :
    #     most_common = []

    # return most_common[0][0] if most_common else None
    #
    #



    # # median score:
    # if scores:
    #     sort = sorted(scores, key=lambda x: x[index["score"]], reverse=True)
    #     median = sort[len(sort) // 2]
    #     median_score = median[index["score"]]
    # else:
    #     median_score = 0
    #
    # return 1/median_score if median_score != 0 else 0
    #































# faster to get scores and the sort by charts but this is fine
def get_all_avg_score():
    slist = {}

    for chart in charts():
        id = chart[cindex["id"]]
        scores = scores_for_chart(id)
        if scores:
            avg = sum(score[index["score"]] for score in scores) / len(scores)
            slist[id] = avg
        else:
            slist[id] = 0

    return slist


def rate_all_charts():
    dif = {} 
    for chart in charts(additional_exclude=[]): 
        id = chart[cindex["id"]]
        print("CHART: ", id)

        difficulty = create_difficulty(chart)


        print("DIFFICULTY: ", difficulty)
        dif[id] = difficulty 

    dif_sorted = sorted(dif.items(), key=lambda x: x[1], reverse=True)
    return dif_sorted


















def main():
    dif_sorted = rate_all_charts()




# dif_sorted[-missing:]
#
#
#
#
    excluded = []
# excluded = [x[0] for x in dif_sorted[:6]]
#
# # exclude 26 and 27
# for chart in dif_sorted[:6]:
#     excluded.append(chart[0])
    included = []
    included = [(x[0], x[1]) for x in dif_sorted if x[0] not in excluded]


    _charts = charts()
    sorted_charts = sorted(_charts, key=lambda x: x[cindex["difficulty"]])
    sorted_combined = [(x[cindex["id"]], x[cindex["difficulty"]]) for x in sorted_charts if x[cindex["id"]] not in excluded]
    sorted_charts_ids = [x[cindex["id"]] for x in sorted_charts if x[cindex["id"]] not in excluded]
    len(sorted_charts_ids)
    len(sorted_combined)
    len(included)




# Example inputs
# included = [(chart_id, continuous_score), ...]
# sorted_combined = [(chart_id, discrete_difficulty), ...]

# Convert sorted_combined to a lookup dict
    id_to_discrete = dict(sorted_combined)

# Group continuous difficulties by discrete difficulty
    grouped = defaultdict(list)
    for chart_id, continuous_score in included:
        if chart_id in id_to_discrete:
            discrete_diff = id_to_discrete[chart_id]
            grouped[discrete_diff].append(continuous_score)

# Plot setup
    plt.figure(figsize=(14, 6))
    colors = plt.get_cmap("tab20")
    patches = []

# Plot each group
    for i, (discrete_diff, scores) in enumerate(sorted(grouped.items())):
        x_vals = [discrete_diff] * len(scores)
        y_vals = scores
        color = colors(i % 20)
        plt.scatter(x_vals, y_vals, color=color, alpha=0.7, label=f'Difficulty {discrete_diff}')
        patches.append((discrete_diff, color))

# Labels and aesthetics
    plt.xlabel("Discrete Difficulty", fontsize=12)
    plt.ylabel("Continuous Difficulty", fontsize=12)
    plt.title("Continuous vs Discrete Difficulty", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(sorted(grouped.keys()))

# Legend
    plt.legend(
        handles=[
            Line2D([0], [0], marker='o', color='w', label=f'Difficulty {d}', markerfacecolor=c, markersize=8)
            for d, c in patches
        ],
        title="Discrete Difficulty",
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    plt.tight_layout()
    plt.show()







if __name__ == "__main__":
    main()






















# # Connect to the database
# conn = sqlite3.connect("new_data.db")
# cursor = conn.cursor()
#
# # Update each chart with its rank
# for chart_id, rank in dif_sorted:
#     cursor.execute("UPDATE charts SET I_DIFF_2 = ? WHERE _id = ?", (rank, chart_id))
#
# # Commit and close
# conn.commit()
# conn.close()



# bare tull
#
# def fac(num):
#     if num == 0:
#         return 1
#
#     return num + fac(num - 1)
#
# def ply(num):
#     return (num*(num-1)) // 2
# ply(50)
# fac(50)
#
#
#
#
#
#
#
#
# full = charts()
# lowest = 9999999
# for chart in full:
#     steps = steps_for_chart(chart[cindex["_id"]])
#     if steps < lowest:
#         lowest = steps
#         print(chart[cindex["_id"]], chart[cindex["difficulty_display"]], steps)
# #
# #
# #
# song_from_chart(4679)
#














# get all charts on beforehand
# all_chart_avg = get_all_avg_score() 
# last_scores.last_scr_for_dif_and_dis("dogfetus", 10, 20, "wild", DATE)
# score_evaluater.difficulties_for_display("wild")
# def get_rank():
        
