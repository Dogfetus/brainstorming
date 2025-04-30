import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from collections import defaultdict

from itr3 import fetch_scores_with_difficulty, rank_itr3, build_chart_score_lookup
# from avg import rank_by_avg
from score import rank_players_by_categories 
from graph import get_rank_graphs_for_list, get_rank_graphs_for_list_better, get_rank_graphs_for_list_superfast

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

players = [player.lower() for player in players]








data=""""""

def extract_player_names(text):
    # Pattern looks for number followed by dot, player name, and then dash
    pattern = r'\d+\.\s+([^—]+)—'
    
    # Find all matches
    matches = re.findall(pattern, text)
    
    # Strip whitespace from each name
    player_names = [name.strip() for name in matches]
    
    return player_names



def read_csv(file_path="final_ml_rankings.csv"):

    df = pd.read_csv(file_path)
    sorted_df = df.sort_values(by="final_rank")
    player_names = sorted_df["player"].str.lower().tolist()
    for name in player_names:
        print(name)

    return player_names 



def optimizer(start, end, step, difficulty_field="I_DIFF_7", optimizing_field="max_delta", **kwargs):
    list1 = players

    all_scores = fetch_scores_with_difficulty(difficulty_field=difficulty_field) 
    chart_score_lookup = build_chart_score_lookup(all_scores) 

    best_result = {"threshold": None, "mae": float("inf")}
    for i in np.arange(start, end, step):

        params = {
            **kwargs,
            optimizing_field: i,
            "player_list": players[::-1],
            "difficulty_field": difficulty_field,
            "all_scores": all_scores,
            "chart_score_lookup": chart_score_lookup,
        }


        list2 = rank_itr3(**params) 
        mae = np.mean(np.abs([i - list2.index(p) for i, p in enumerate(list1)]))
        if mae < best_result["mae"]:
            best_result["threshold"] = i
            best_result["mae"] = mae
            print(f"\033[1;32minput: {i}, Mean Absolute Error: {mae:.2f}\033[0m")
        else:
            print(f"input: {i}, Mean Absolute Error: {mae:.2f}")
    return best_result



def merge_rankings_by_average(rank1, rank2):
    """
    rank1 and rank2 are dicts: player -> rank (1-based)
    Returns a list of (player, combined_rank), sorted by combined rank
    """
    merged = defaultdict(list)

    # Add ranks from both lists
    for player, r in rank1.items():
        merged[player].append(r)
    for player, r in rank2.items():
        merged[player].append(r)

    # Calculate average rank
    average_ranks = {p: sum(rs)/len(rs) for p, rs in merged.items()}

    # Sort by average
    return sorted(average_ranks.items(), key=lambda x: x[1])



def merge_helper(ranks):
    result = 1
    for rank in ranks:
        result *= rank
    return result

def new_merge(list1, list2):
    merged = defaultdict(list)

    for player, rank in list1:
        if rank < 100:
            merged[player].append(rank*1000)
        else:
            merged[player].append(rank)

    for player, rank in list2:
        if rank < 100:
            merged[player].append(rank*1000)
        else:
            merged[player].append(rank)

    # Calculate average rank
    average_ranks = {p: sum(rs)/len(rs) for p, rs in merged.items()}

    # Sort by average in reverse order
    return sorted(average_ranks.items(), key=lambda x: x[1], reverse=True)




























#
# # # #graph and such
# player_ranks = []
# for player in players:
#     rank = get_rank(player, dis=["wild"], date=DATE)
#     # rank = get_rank_graph_bulk(player, x=5, date=DATE)
#     # rank = rank_by_avg(player, date=DATE)
#     print(f"\033[1;36m{player} : \t\033[1;33m{rank}\033[0m")
#     player_ranks.append((player, rank))
#
# groupd_player_ranks = sorted(player_ranks, key=lambda x: x[1], reverse=True) 
# graph_list = [group[0] for group in groupd_player_ranks]
#
# mae = np.mean(np.abs([i - graph_list.index(p) for i, p in enumerate(players)]))
# print(f"Mean Absolute Error: {mae:.2f}")
#
# for rank, (player, rating) in enumerate(groupd_player_ranks, start=1):
#     print(f"\033[1;36m{rank:2}.\033[0m \033[1;33m{player:15}\033[0m: {rating:.2f}")














result = rank_players_by_categories(players, ["wild"], date=DATE)
result

# MAE 2.12 2.16?
# graph_list = get_rank_graphs_for_list(players, x=5, date=DATE)
# graph_list = get_rank_graphs_for_list_superfast(players, x=5, date=DATE)
elo_list = rank_itr3(players[::-1], "I_DIFF_7", threshold=2240)
other_list = [player for player, _ in sorted(result, key=lambda x: x[1], reverse=True)] 
#
# mae = np.mean(np.abs([i - graph_list.index(p) for i, p in enumerate(players)]))
mae2 = np.mean(np.abs([i - elo_list.index(p) for i, p in enumerate(players)]))
mae3 = np.mean(np.abs([i - other_list.index(p) for i, p in enumerate(players)]))
#
# rank_dict_2 = {player: i+1 for i, player in enumerate(graph_list)}
rank_dict_1 = {player: i+1 for i, player in enumerate(elo_list)}
rank_dict_3 = {player: i+1 for i, player in enumerate(other_list)}
#
merged = merge_rankings_by_average(rank_dict_1, rank_dict_3)
















# score2 = rank_itr3(players[::-1], difficulty_field="I_DIFF_7", threshold=2240)
# score = get_rank_graphs_for_list(players, x=5, date=DATE)
#
# merged = new_merge(score, score2)















#exponent=15
# results = optimizer(0, 100, 1, difficulty_field="I_DIFF_7", optimizing_field="exponent")

list1 = players

# list2 = [player for player, _ in sorted(groupd_player_ranks, key=lambda x: x[1], reverse=True)]
# list2 = rank_itr3(players[::-1], "I_DIFF_7")
# list2 = rank_itr3(players[::-1], "I_DIFF_3", threshold=1900)
#
# list2 = rank_itr3(players[::-1], difficulty_field="I_DIFF_1", exponent=0.5, max_delta=10, threshold=2300, k_factor=50)
# list2 = rank_itr3(players[::-1], difficulty_field="I_DIFF_2",max_delta=10, threshold=2200)
# list2 = rank_itr3(players[::-1], difficulty_field="I_DIFF_3", max_delta=3, threshold=1900, k_factor=2, exponent=0.8)
# list2 = rank_itr3(players[::-1], difficulty_field="I_DIFF_4", max_delta=12, threshold=2000, k_factor=42, exponent=15)
# list2 = rank_itr3(players[::-1], difficulty_field="I_DIFF_5", threshold=16200, max_delta=3, k_factor=69, exponent=1.2)
# list2 = rank_itr3(players[::-1], difficulty_field="I_DIFF_6", max_delta=8, threshold=1800, k_factor=20, exponent=1.2)

list2 = [m[0] for m in merged] 
# list2 = get_rank_graphs_for_list(players, x=14, date=DATE)
# list2 = get_rank_graphs_for_list(players, x=5, date=DATE)


for p in list2:
    print(p)

deviations = []
for i, player in enumerate(list1):
    lower_list2 = [p.lower() for p in list2]
    if player.lower() in lower_list2:
        new_index = lower_list2.index(player.lower())
        original_index = i
        diff = original_index - new_index
        deviations.append(diff)
    else:
        # throw an error print as error with red color
        print(f"\033[91mError: Player '{player}' not found in the new ranking list.\033[0m")
        exit(1)


if deviations:
    mae = np.mean(np.abs(deviations))
else:
    print("\033[91mError: No valid deviations found. 'deviations' list is empty.\033[0m")
    exit(1)

print (f"\nMean Absolute Error (MAE): {mae:.2f}")


x = np.arange(len(list1))
plt.figure(figsize=(14, 6))
plt.bar(x, deviations, color="purple")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")

plt.xticks(x, list1, rotation=90)
plt.xlabel("Players (Original Order)", fontsize=16)
plt.ylabel("Ranking Deviation (Predicted - Acutal)", fontsize=16)
plt.title(f"Deviation from Actual Tournament Placement \n Mean Absolute Error: {mae:.2f}", fontsize=18)
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.savefig("placement_change.png", dpi=600, bbox_inches="tight")
plt.show()


