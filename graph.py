import sqlite3
import json
import time
# from collections import Counter
import last_scores
import score_evaluater 
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# from collections import defaultdict
import requests
import statistics
from difficulty import index, cindex, value_of_difficulty

EXTRA_EP = """https://api.smx.573.no/extra/{}"""
DATE = "2024-10-17"


players = [
    "JimboSMX",
    "hintz",
    "spencer",
    "swagman",
    "paranoiaboi",
    "masongos",
    "grady",
    "jjk.",
    "chezmix",
    "inglomi",
    "jellyslosh",
    "wdrm",
    "eesa",
    "senpi",
    "Janus5k",
    "tayman",
    "emcat",
    "pilot",
    "mxl100",
    "snowstorm",
    "datcoreedoe",
    "big matt",
    "werdwerdus",
    "cathadan",
    "shinobee",
    "mesyr",
    "zom585",
    "ctemi",
    "zephyrnobar",
    "sydia",
    "kren",
    "xosen",
    "sweetjohnnycage",
    "cheesecake",
    "ditz",
    "enderstevemc",
    "jiajiabinks",
    "meeko",
    "momosapien",
    "auby",
    "arual",
    "dogfetus",
    "noir",
    "dali",
    " peter",
    "jokko",
    "butterbutt",
    "jewel",
    "beatao",
    "maverick",
]




CHUNK_SIZE = 100



def read_graph(score):
    score_id = score if isinstance(score, int) else score[0] if isinstance(score, tuple) else None
    if score_id is None:
        return None, None, None

    with sqlite3.connect("new_data.db") as conn:
        cursor = conn.execute("SELECT offsets FROM score_graphs WHERE score_id = ?", (score_id,))
        row = cursor.fetchone()

        if row:
            if row[0] is None:
                return None, None, None
            offsets = json.loads(row[0])
        else:
            try:
                graph = requests.get(EXTRA_EP.format(score_id)).json()
            except Exception:
                return None, None, None

            time.sleep(1)

            if "offsets" not in graph or not graph["offsets"]:
                conn.execute(
                    "INSERT INTO score_graphs (score_id, offsets) VALUES (?, ?)",
                    (score_id, None)
                )
                return None, None, None

            offsets = graph["offsets"]
            conn.execute(
                "INSERT INTO score_graphs (score_id, offsets) VALUES (?, ?)",
                (score_id, json.dumps(offsets))
            )

    # Parse the triplets
    _time = offsets[1:][0::3]
    offset_vals = offsets[1:][1::3]
    judgement = offsets[1:][2::3]

    return _time, offset_vals, judgement




def stddev(data):
    if not data:
        return -1

    if len(data) < 2:
        return -1 

    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return variance ** 0.5

def stddev2(data):
    if not data:
        return -1

    if len(data) < 2:
        return -1

    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return variance


def get_scores(player, x, dif_display=["hard", "wild"], date=DATE):
    scores = []
    for display in dif_display:
        difficulties = score_evaluater.difficulties_for_display(display)
        for difficulty in difficulties:
            last = last_scores.last_scr_for_dif_and_dis(player, x, difficulty, display, date) 
            scores.append(last)
    return scores


def get_rank_graph(player, x=10, date="2026-10-10"):

    scores = get_scores(player, x, date=date)
    stddevs = []

    # one score section is multipole scores for the same difficulty + display
    for score_section in scores:
        for score in score_section:
            _, offsets, _ = read_graph(score[0])
            if offsets is None:
                continue
            mean_offset = abs(sum(offsets) / len(offsets))
            stddevs.append((stddev2(offsets), mean_offset , score[3], score[4]))

    values = []

    for stddev, distance, diff, display in stddevs:
        if stddev == -1 or distance == -1:
            continue

        base_score = 1 / stddev * diff * value_of_difficulty[display]
        penalty = distance * 0.01
        values.append(base_score)

    if not values: 
        return 0

    rank = sum(values) / len(values)

    return rank






#
# _, test, _ = read_graph(3253977)
#
# if test:
#     mean_offset = abs(sum(test) / len(test))
#
# mean_offset


# result = []
# for player in players:
#     rank = get_rank_graph(player)
#     print(f"\033[1;36mPlayer: {player}: \t\033[1;33m{rank}\033[0m")
#     result.append((player, rank))
#
#
# result.sort(key=lambda x: x[1], reverse=True)
# print("\n\n\n")
# for player, rank in result:
#     print(f"\033[1;36mPlayer: {player}: \t\033[1;33m{rank}\033[0m")
#
#




# def read_graphs_bulk(score_ids):
#     graph_data = {}
#
#     with sqlite3.connect("new_data.db") as conn:
#         conn.execute("PRAGMA synchronous = OFF")
#         conn.execute("PRAGMA journal_mode = MEMORY")
#         conn.execute("PRAGMA temp_store = MEMORY")
#
#         for i in range(0, len(score_ids), CHUNK_SIZE):
#             chunk = score_ids[i:i+CHUNK_SIZE]
#             cursor = conn.execute(
#                 f"SELECT score_id, offsets FROM score_graphs WHERE score_id IN ({','.join('?' for _ in chunk)})",
#                 chunk
#             )
#             rows = cursor.fetchall()
#             for score_id, offsets_json in rows:
#                 if offsets_json:
#                     offsets = json.loads(offsets_json)
#                     graph_data[score_id] = offsets
#                 else:
#                     graph_data[score_id] = None
#
#     return graph_data
def read_graphs_bulk(score_ids):
    graph_data = {}
    to_fetch = []

    with sqlite3.connect("new_data.db") as conn:
        cursor = conn.execute(
            f"SELECT score_id, offsets FROM score_graphs WHERE score_id IN ({','.join('?' for _ in score_ids)})",
            score_ids
        )
        rows = cursor.fetchall()
        for score_id, offsets_json in rows:
            if offsets_json:
                offsets = json.loads(offsets_json)
                graph_data[score_id] = offsets
            else:
                graph_data[score_id] = None

        # Determine which IDs are missing
        found_ids = set(row[0] for row in rows)
        to_fetch = [sid for sid in score_ids if sid not in found_ids]

        for sid in to_fetch:
            try:
                graph = requests.get(EXTRA_EP.format(sid)).json()
                time.sleep(1)  # Respect rate limit
            except Exception:
                conn.execute("INSERT INTO score_graphs (score_id, offsets) VALUES (?, ?)", (sid, None))
                graph_data[sid] = None
                continue

            if "offsets" not in graph or not graph["offsets"]:
                conn.execute("INSERT INTO score_graphs (score_id, offsets) VALUES (?, ?)", (sid, None))
                graph_data[sid] = None
                continue

            offsets = graph["offsets"]
            conn.execute("INSERT INTO score_graphs (score_id, offsets) VALUES (?, ?)", (sid, json.dumps(offsets)))
            graph_data[sid] = offsets

    return graph_data






def get_rank_graph_bulk(player, x=10, date="2026-10-10"):
    scores = get_scores(player, x, date=date)

    # Flatten score IDs
    all_score_ids = [score[0] for group in scores for score in group]
    graph_cache = read_graphs_bulk(all_score_ids)

    stddevs = []

    for score_section in scores:
        for score in score_section:
            score_id = score[0]
            offsets = graph_cache.get(score_id)
            if not offsets:
                continue
            offset_vals = offsets[1:][1::3]

            mean_offset = abs(sum(offset_vals) / len(offset_vals))
            stddevs.append((stddev2(offset_vals), mean_offset, score[3], score[4]))

    values = []

    for stddev, distance, diff, display in stddevs:
        if stddev == -1 or distance == -1:
            continue
        base_score = 1 / stddev * diff * value_of_difficulty[display]
        penalty = distance * 0.01
        values.append(base_score)

    if not values:
        return 0

    rank = sum(values) / len(values)
    return rank


def get_rank_graphs_for_list(players, x=12, date=DATE):
    all_scores = {}       # player -> list of scores
    all_score_ids = set() # all score_ids to fetch graphs for

    # Step 1: Gather scores for all players
    for player in players:
        scores = get_scores(player, x, date=date)
        all_scores[player] = scores
        for group in scores:
            for score in group:
                all_score_ids.add(score[0])

    # Step 2: Batch fetch all graphs
    graph_cache = read_graphs_bulk(list(all_score_ids))

    # Step 3: Calculate rankings
    player_ranks = []

    for player in players:
        scores = all_scores[player]
        stddevs = []

        for score_section in scores:
            for score in score_section:
                score_id = score[0]
                offsets = graph_cache.get(score_id)
                if not offsets:
                    continue

                offset_vals = offsets[1:][1::3]

                if not offset_vals:
                    continue

                mean_offset = abs(sum(offset_vals) / len(offset_vals))
                stddevs.append((stddev2(offset_vals), mean_offset, score[3], score[4]))

        values = []

        for stddev, distance, diff, display in stddevs:
            if stddev == -1 or distance == -1:
                continue
            safe_stddev = 1 / stddev if stddev != 0 else 1e-6 
            base_score = (safe_stddev * diff * value_of_difficulty[display])
            values.append(base_score)


        rank = sum(values) / len(values) if values else 0
        player_ranks.append((player, rank))

        print(f"\033[1;36mPlayer: {player}: \t\033[1;33m{rank}\033[0m")

    # Step 4: Sort and return
    player_ranks.sort(key=lambda x: x[1], reverse=True)

    #temporary:
    fin_res = [player for player, _ in player_ranks]
    
    return fin_res 
    # return player_ranks
    #





def get_rank_graphs_for_list_better(players, x=5, date=DATE):
    """
    Optimized version of the function that calculates and returns player rankings.
    """
    # Step 1: Gather scores for all players - using flat lists for efficiency
    all_scores = {}  # player -> list of scores
    all_score_ids = set()  # all score_ids to fetch graphs for
    
    for player in players:
        scores = get_scores(player, x, date=date)
        all_scores[player] = scores
        for group in scores:
            all_score_ids.update(score[0] for score in group)
    
    # Step 2: Batch fetch all graphs
    graph_cache = read_graphs_bulk(list(all_score_ids))
    
    # Step 3: Calculate rankings with optimized data processing
    player_ranks = []
    
    for player in players:
        values = []
        
        for score_section in all_scores[player]:
            for score in score_section:
                score_id = score[0]
                offsets = graph_cache.get(score_id)
                
                if not offsets:
                    continue
                
                # More efficient slicing - get offset values directly
                offset_vals = offsets[2::3]  # Instead of offsets[1:][1::3]
                
                if not offset_vals:
                    continue
                
                # Calculate mean and variance in one pass
                n = len(offset_vals)
                mean_offset = sum(offset_vals) / n
                mean_abs_offset = abs(mean_offset)
                
                # Calculate variance directly
                variance = sum((x - mean_offset) ** 2 for x in offset_vals) / (n - 1) if n > 1 else -1
                
                if variance > 0:
                    # Calculate base score in one step - avoid multiple conditionals
                    diff_value = score[3] * value_of_difficulty[score[4]]
                    base_score = (1 / variance) * diff_value
                    values.append(base_score)
        
        # Calculate rank - avoid division if empty list
        rank = sum(values) / len(values) if values else 0
        player_ranks.append((player, rank))
        print(f"\033[1;36mPlayer: {player}: \t\033[1;33m{rank}\033[0m")
    
    # Step 4: Sort and return
    player_ranks.sort(key=lambda x: x[1], reverse=True)
    
    # Extract just the player names in ranked order
    return [player for player, _ in player_ranks]



from concurrent.futures import ThreadPoolExecutor
import math

def get_rank_graphs_for_list_superfast(players, x=5, date=DATE):
    """
    Super optimized version for ranking players based on graphs.
    """
    all_scores = {}  
    all_score_ids = []  
    
    for player in players:
        scores = get_scores(player, x, date=date)
        all_scores[player] = scores
        for group in scores:
            for score in group:
                all_score_ids.append(score[0])

    graph_cache = read_graphs_bulk(all_score_ids)

    difficulty_values = value_of_difficulty.copy()

    player_ranks = []

    for player in players:
        values = []
        
        for score_section in all_scores[player]:
            for score in score_section:
                score_id = score[0]
                offsets = graph_cache.get(score_id)
                
                if not offsets:
                    continue

                offset_vals = offsets[2::3]

                if not offset_vals:
                    continue

                # Welford's one-pass variance calculation
                n = 0
                mean = 0
                M2 = 0

                for x in offset_vals:
                    n += 1
                    delta = x - mean
                    mean += delta / n
                    delta2 = x - mean
                    M2 += delta * delta2

                variance = (M2 / (n - 1)) if n > 1 else -1

                if variance > 0:
                    diff_value = score[3] * difficulty_values.get(score[4], 1)
                    base_score = (1 / variance) * diff_value
                    values.append(base_score)
        
        rank = sum(values) / len(values) if values else 0
        player_ranks.append((player, rank))
        print(f"\033[1;36mPlayer: {player}: \t\033[1;33m{rank}\033[0m")
    
    player_ranks.sort(key=lambda x: x[1], reverse=True)

    return [player for player, _ in player_ranks]

