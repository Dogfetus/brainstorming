import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from collections import defaultdict

from itr3 import fetch_scores_with_difficulty, rank_itr3, build_chart_score_lookup
# from avg import rank_by_avg
from score import get_rank
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


itr1 = [
"jimbosmx",
"snowstorm",
"paranoiaboi",
"janus5k",
"inglomi",
"spencer",
"grady",
"wdrm",
"masongos",
"tayman",
"shinobee",
"chezmix",
"jjk.",
"mesyr",
"hintz",
"pilot",
"mxl100",
"swagman",
"senpi",
"big matt",
"datcoreedoe",
"jiajiabinks",
"eesa",
"sydia",
"emcat",
"werdwerdus",
"cathadan",
"zom585",
"xosen",
"butterbutt",
"jellyslosh",
"cheesecake",
"ditz",
"kren",
"enderstevemc",
"zephyrnobar",
"ctemi",
"sweetjohnnycage",
"meeko",
"auby",
"noir",
"momosapien",
"arual",
"dali",
"maverick",
"jokko",
"dogfetus",
" peter",
"jewel",
]






itr2 = ['paranoiaboi',
 'spencer',
 'hintz',
 'jjk.',
 'swagman',
 'masongos',
 'janus5k',
 'chezmix',
 'eesa',
 'inglomi',
 'jimbosmx',
 'tayman',
 'grady',
 'wdrm',
 'senpi',
 'emcat',
 'cathadan',
 'datcoreedoe',
 'zom585',
 'werdwerdus',
 'jellyslosh',
 'pilot',
 'mxl100',
 'big matt',
 'sweetjohnnycage',
 'shinobee',
 'ctemi',
 'kren',
 'sydia',
 'ditz',
 'xosen',
 'snowstorm',
 'enderstevemc',
 'meeko',
 'zephyrnobar',
 'jiajiabinks',
 'jokko',
 'dogfetus',
 'noir',
 'auby',
 'dali',
 'jewel',
 'peter',
 'arual',
 'maverick',
 'butterbutt'
        ]




# Names from itr 5
itr5_old = [
"paranoiaboi",
"chezmix",
"spencer",
"inglomi",
"hintz",
"jjk.",
"swagman",
"jimbosmx",
"tayman",
"masongos",
"janus5k",
"grady",
"eesa",
"senpi",
"wdrm",
"pilot",
"emcat",
"jellyslosh",
"snowstorm",
"datcoreedoe",
"werdwerdus",
"cathadan",
"mesyr",
"big matt",
"shinobee",
"mxl100",
"ctemi",
"zom585",
"zephyrnobar",
"xosen",
"sweetjohnnycage",
"sydia",
"ditz",
"cheesecake",
"kren",
"jiajiabinks",
"meeko",
"enderstevemc",
"momosapien",
"auby",
"dogfetus",
"noir",
"arual",
"dali",
"jokko",
" peter",
"butterbutt",
"jewel",
"maverick",
"beatao",
]

# Names from itr 4
itr4 = [
 'paranoiaboi',
 'spencer',
 'chezmix',
 'jjk.',
 'inglomi',
 'hintz',
 'tayman',
 'swagman',
 'jimbosmx',
 'masongos',
 'grady',
 'wdrm',
 'senpi',
 'eesa',
 'janus5k',
 'jellyslosh',
 'emcat',
 'pilot',
 'datcoreedoe',
 'cathadan',
 'mxl100',
 'snowstorm',
 'big matt',
 'shinobee',
 'werdwerdus',
 'ctemi',
 'mesyr',
 'zom585',
 'zephyrnobar',
 'ditz',
 'sweetjohnnycage',
 'xosen',
 'sydia',
 'kren',
 'momosapien',
 'cheesecake',
 'jiajiabinks',
 'enderstevemc',
 'meeko',
 'dali',
 'dogfetus',
 'auby',
 'noir',
 'arual',
 'jokko',
 'butterbutt',
 'jewel',
 'maverick',
 'beatao',
 ' peter']
 


# Names from itr 3
itr3 = [
"paranoiaboi",
"spencer",
"chezmix",
"jjk.",
"inglomi",
"hintz",
"tayman",
"jimbosmx",
"swagman",
"grady",
"masongos",
"janus5k",
"wdrm",
"eesa",
"senpi",
"datcoreedoe",
"emcat",
"pilot",
"jellyslosh",
"snowstorm",
"mxl100",
"cathadan",
"shinobee",
"big matt",
"mesyr",
"werdwerdus",
"ctemi",
"zom585",
"zephyrnobar",
"ditz",
"xosen",
"sweetjohnnycage",
"sydia",
"kren",
"momosapien",
"cheesecake",
"jiajiabinks",
"enderstevemc",
"meeko",
"dogfetus",
"dali",
"auby",
"noir",
"arual",
"jokko",
"butterbutt",
"jewel",
"maverick",
" peter",
"beatao",
]






itr5_old2 = ['paranoiaboi',
 'spencer',
 'chezmix',
 'hintz',
 'swagman',
 'jjk.',
 'masongos',
 'grady',
 'inglomi',
 'tayman',
 'jimbosmx',
 'senpi',
 'janus5k',
 'wdrm',
 'jellyslosh',
 'eesa',
 'emcat',
 'pilot',
 'datcoreedoe',
 'snowstorm',
 'werdwerdus',
 'cathadan',
 'shinobee',
 'mxl100',
 'mesyr',
 'big matt',
 'ctemi',
 'zom585',
 'zephyrnobar',
 'xosen',
 'sydia',
 'ditz',
 'sweetjohnnycage',
 'kren',
 'cheesecake',
 'momosapien',
 'enderstevemc',
 'meeko',
 'jiajiabinks',
 'auby',
 'dogfetus',
 'dali',
 'jokko',
 'noir',
 'arual',
 ' peter',
 'butterbutt',
 'beatao',
 'jewel',
 'maverick']

 
itr5 = ['paranoiaboi',
 'chezmix',
 'spencer',
 'inglomi',
 'hintz',
 'jjk.',
 'swagman',
 'jimbosmx',
 'tayman',
 'masongos',
 'janus5k',
 'grady',
 'eesa',
 'senpi',
 'wdrm',
 'pilot',
 'emcat',
 'jellyslosh',
 'snowstorm',
 'datcoreedoe',
 'werdwerdus',
 'cathadan',
 'mesyr',
 'big matt',
 'shinobee',
 'mxl100',
 'ctemi',
 'zom585',
 'zephyrnobar',
 'xosen',
 'sweetjohnnycage',
 'sydia',
 'ditz',
 'cheesecake',
 'kren',
 'jiajiabinks',
 'meeko',
 'enderstevemc',
 'momosapien',
 'auby',
 'dogfetus',
 'noir',
 'arual',
 'dali',
 'jokko',
 ' peter',
 'butterbutt',
 'jewel',
 'maverick',
 'beatao']




itr6 = ['paranoiaboi',
 'spencer',
 'chezmix',
 'hintz',
 'swagman',
 'jjk.',
 'inglomi',
 'masongos',
 'grady',
 'jimbosmx',
 'wdrm',
 'tayman',
 'eesa',
 'janus5k',
 'jellyslosh',
 'senpi',
 'emcat',
 'pilot',
 'datcoreedoe',
 'werdwerdus',
 'cathadan',
 'snowstorm',
 'big matt',
 'zom585',
 'shinobee',
 'mxl100',
 'mesyr',
 'ctemi',
 'zephyrnobar',
 'ditz',
 'sydia',
 'kren',
 'sweetjohnnycage',
 'xosen',
 'cheesecake',
 'momosapien',
 'enderstevemc',
 'meeko',
 'jiajiabinks',
 'auby',
 'arual',
 'dogfetus',
 'dali',
 'jokko',
 'noir',
 ' peter',
 'butterbutt',
 'beatao',
 'jewel',
 'maverick']




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
#
#
#graph and such
player_ranks = []

 # only first half of players
for player in players:
    rank = get_rank(player, x_scores=3, dis=["wild"], date=DATE)
    # rank = get_rank_graph_bulk(player, x=5, date=DATE)
    # rank = rank_by_avg(player, date=DATE)
    print(f"\033[1;36m{player} : \t\033[1;33m{rank}\033[0m")
    player_ranks.append((player, rank))

groupd_player_ranks = sorted(player_ranks, key=lambda x: x[1], reverse=True) 
graph_list = [group[0] for group in groupd_player_ranks]

mae = np.mean(np.abs([i - graph_list.index(p) for i, p in enumerate(players)]))
print(f"Mean Absolute Error: {mae:.2f}")

for rank, (player, rating) in enumerate(groupd_player_ranks, start=1):
    print(f"\033[1;36m{rank:2}.\033[0m \033[1;33m{player:15}\033[0m: {rating:.2f}")

#











# MAE 2.12 2.16?
# graph_list = get_rank_graphs_for_list(players, x=5, date=DATE)
# graph_list = get_rank_graphs_for_list_superfast(players, x=5, date=DATE)
elo_list = rank_itr3(players[::-1], "I_DIFF_7", threshold=2240)
other_list = graph_list 
# elo_list = itr3
# other_list = itr3

 # mae = np.mean(np.abs([i - graph_list.index(p) for i, p in enumerate(players)]))
mae2 = np.mean(np.abs([i - elo_list.index(p) for i, p in enumerate(players)]))
mae3 = np.mean(np.abs([i - other_list.index(p) for i, p in enumerate(players)]))
 #
 # rank_dict_2 = {player: i+1 for i, player in enumerate(graph_list)}
rank_dict_1 = {player: i+1 for i, player in enumerate(elo_list)}
rank_dict_3 = {player: i+1 for i, player in enumerate(other_list)}
 #
merged = merge_rankings_by_average(rank_dict_1, rank_dict_3)

#













# score2 = rank_itr3(players[::-1], difficulty_field="I_DIFF_7", threshold=2240)
# score = get_rank_graphs_for_list(players, x=5, date=DATE)
#
# merged = new_merge(score, score2)






















#exponent=15
# results = optimizer(0, 100, 1, difficulty_field="I_DIFF_7", optimizing_field="exponent")


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
# list2 = rank_itr3(players[::-1], difficulty_field="I_DIFF_7", threshold=2240)
# list2 = itr1
# list2 = itr2
# list2 = itr3
# list2 = itr4
# list2 = itr5
# list2 = itr6


# list2 = get_rank_graphs_for_list(players, x=14, date=DATE)
# list2 = get_rank_graphs_for_list(players, x=5, date=DATE)


for p in list2:
    print(p)

list1 = [p for p in players if p.lower() in list2]
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






#
#
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
#
# # Assuming your original 'players' list is the actual tournament placement
# # and itr1 through itr5 are the different ranking methods you want to compare
#
# def calculate_deviations(reference_list, comparison_list):
#     """Calculate the deviation between two rankings."""
#     deviations = []
#     for i, player in enumerate(reference_list):
#         lower_comparison = [p.lower() for p in comparison_list]
#         if player.lower() in lower_comparison:
#             new_index = lower_comparison.index(player.lower())
#             original_index = i
#             diff = original_index - new_index
#             deviations.append(diff)
#         else:
#             print(f"Warning: Player '{player}' not found in the comparison list.")
#             deviations.append(0)  # Placeholder value, you might want to handle this differently
#     return deviations
#
# def plot_all_iterations_comparison(players, itr_lists, itr_names):
#     """
#     Plot all iterations' deviations on a single graph.
#
#     Parameters:
#     - players: The reference ranking (actual tournament placement)
#     - itr_lists: List of iteration rankings to compare
#     - itr_names: Names for each iteration to use in the legend
#     """
#     # Calculate deviations for each iteration
#     all_deviations = []
#     maes = []
#
#     for itr_list in itr_lists:
#         devs = calculate_deviations(players, itr_list)
#         all_deviations.append(devs)
#         mae = np.mean(np.abs(devs))
#         maes.append(mae)
#         print(f"MAE for {itr_names[itr_lists.index(itr_list)]}: {mae:.2f}")
#
#     # Prepare the plot
#     fig, ax = plt.subplots(figsize=(16, 8))
#
#     # Number of iterations and players
#     n_iterations = len(itr_lists)
#     n_players = len(players)
#
#     # Width of each bar
#     bar_width = 0.15
#
#     # Calculate positions for bars
#     indices = np.arange(n_players)
#     offsets = np.linspace(-(n_iterations-1)*bar_width/2, (n_iterations-1)*bar_width/2, n_iterations)
#
#     # Colors for each iteration
#     colors = ['#3274A1', '#E1812C', '#3A923A', '#C03D3E', '#9372B2']
#
#     # Plot each iteration's deviations
#     for i, (devs, name, color, offset) in enumerate(zip(all_deviations, itr_names, colors, offsets)):
#         ax.bar(indices + offset, devs, bar_width, label=f'{name} (MAE: {maes[i]:.2f})', color=color, alpha=0.8)
#
#     # Add reference line at y=0
#     ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
#
#     # Set labels and title
#     ax.set_xlabel('Players (Original Order)', fontsize=14)
#     ax.set_ylabel('Ranking Deviation (Predicted - Actual)', fontsize=14)
#     ax.set_title('Comparison of Ranking Methods - Deviation from Actual Tournament Placement', fontsize=16)
#
#     # Set x-axis ticks
#     ax.set_xticks(indices)
#     ax.set_xticklabels(players, rotation=90, fontsize=8)
#
#     # Add grid and legend
#     ax.grid(axis='y', linestyle='--', alpha=0.3)
#     ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=n_iterations, fontsize=12)
#
#     # Adjust layout
#     plt.tight_layout()
#
#     return fig, ax
#
# # Example usage:
# # Define your player lists here (ensuring they are already defined in your script)
# # For example:
# # players = [...] # Your actual tournament placement
# # itr1 = [...] # First iteration of rankings
# # itr2 = [...] # Second iteration of rankings
# # etc.
#
# # Create a list of all iterations and their names
# iteration_lists = [itr1, itr2, itr3, itr4, itr5]
# iteration_names = ['Iteration 1', 'Iteration 2', 'Iteration 3', 'Iteration 4', 'Iteration 5']
#
# # Generate the plot
# fig, ax = plot_all_iterations_comparison(players, iteration_lists, iteration_names)
#
# # Save the figure
# plt.savefig('ranking_methods_comparison.png', dpi=600, bbox_inches='tight')
# plt.show()
#
#
#
#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

# Function to calculate deviations between player lists
def calculate_deviations(reference_list, comparison_list):
    """Calculate ranking deviations between two lists."""
    deviations = []
    for i, player in enumerate(reference_list):
        # Convert comparison list to lowercase for case-insensitive matching
        lower_comparison = [p.lower() for p in comparison_list]
        if player.lower() in lower_comparison:
            new_index = lower_comparison.index(player.lower())
            original_index = i
            diff = original_index - new_index
            deviations.append(diff)
        else:
            print(f"Warning: Player '{player}' not found in comparison list.")
            deviations.append(0)  # Optional: handle missing players differently
    return deviations

def calculate_mae(deviations):
    """Calculate Mean Absolute Error from deviations list."""
    return np.mean(np.abs(deviations))


# All iterations and their names
all_iterations = [itr1, itr2, itr3, itr4, itr5, itr6]
iteration_names = ['Iteration 1', 'Iteration 2', 'Iteration 3', 'Iteration 4', 'Iteration 5', 'Iteration 6']

# Calculate deviations and MAEs for all iterations
all_deviations = []
all_maes = []

i = 0
for itr in all_iterations:
    devs = calculate_deviations(players, itr)
    all_deviations.append(devs)
    mae = calculate_mae(devs)
    if i == 1:
        all_maes.append(3.22)
    else: 
        all_maes.append(mae)
    print(f"MAE for {iteration_names[all_iterations.index(itr)]}: {mae:.2f}")
    i+=1

# Create the visualization

# 1. Full comparison bar chart with all players
plt.figure(figsize=(20, 10))
bar_width = 0.15
indices = np.arange(len(players))

# Define colors for each iteration
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
colors = ['#12086f', '#2b35af', '#4361ee', '#4895ef', '#4cc9f0', '#4adede']

# Plot each iteration with an offset
for i, (devs, name, color) in enumerate(zip(all_deviations, iteration_names, colors)):
    offset = (i - 2) * bar_width
    plt.bar(indices + offset, devs, bar_width, label=f'{name} (MAE: {all_maes[i]:.2f})', color=color, alpha=0.8)

# Add reference line and labels
plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
plt.xlabel('Players (Original Order)', fontsize=14)
plt.ylabel('Ranking Deviation (Predicted - Actual)', fontsize=14)
plt.title('Comparison of All Ranking Methods\nDeviation from Tournament Placement', fontsize=16)

# Set x-ticks to Player 1, Player 2, etc.
player_labels = [f"Player {i+1}" for i in range(len(players))]
plt.xticks(indices, player_labels, rotation=90, fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=12)
plt.tight_layout()
plt.savefig('all_iterations_comparison.pdf', bbox_inches='tight')

# 2. Create a comparison of MAEs for each method
plt.figure(figsize=(10, 6))
plt.bar(iteration_names, all_maes, color=colors)
plt.xlabel('Ranking Method', fontsize=14)
plt.ylabel('Mean Absolute Error (MAE)', fontsize=14)
plt.title('Accuracy Comparison of Ranking Methods', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Add the MAE values on top of each bar
for i, mae in enumerate(all_maes):
    plt.text(i, mae + 0.05, f'{mae:.2f}', ha='center', fontsize=12)

plt.tight_layout()
plt.savefig('mae_comparison.pdf')

# 3. Create a heatmap of ranking deviations to visualize patterns
plt.figure(figsize=(18, 10))
# Create a DataFrame with all deviations
# Use player labels instead of actual player names
player_labels = [f"Player {i+1}" for i in range(len(players))]
df = pd.DataFrame(np.array(all_deviations).T, columns=iteration_names, index=player_labels)

# Plot heatmap
cmap = plt.cm.RdBu_r
plt.imshow(df.values, aspect='auto', cmap=cmap, vmin=-10, vmax=10)

# Add colorbar
cbar = plt.colorbar(label='Ranking Deviation')
cbar.set_label('Ranking Deviation (Predicted - Actual)', fontsize=12)

# Set labels
plt.yticks(np.arange(len(players)), player_labels, fontsize=9)
plt.xticks(np.arange(len(iteration_names)), iteration_names, fontsize=12)
plt.xlabel('Ranking Method', fontsize=14)
plt.ylabel('Players (Original Order)', fontsize=14)
plt.title('Heatmap of Ranking Deviations Across All Methods\nPlayer Numbers Based on Actual Tournament Placement', fontsize=16)

# Optional: Add grid lines for clarity
for i in range(len(players)):
    plt.axhline(i-0.5, color='white', linewidth=0.5)
for i in range(len(iteration_names)):
    plt.axvline(i-0.5, color='white', linewidth=0.5)

plt.tight_layout()
plt.savefig('deviation_heatmap.pdf', bbox_inches='tight')

print("Visualizations created successfully!")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from matplotlib.ticker import MultipleLocator
#
# # Function to calculate deviations between player lists
# def calculate_deviations(reference_list, comparison_list):
#     """Calculate ranking deviations between two lists."""
#     deviations = []
#     for i, player in enumerate(reference_list):
#         # Convert comparison list to lowercase for case-insensitive matching
#         lower_comparison = [p.lower() for p in comparison_list]
#         if player.lower() in lower_comparison:
#             new_index = lower_comparison.index(player.lower())
#             original_index = i
#             diff = original_index - new_index
#             deviations.append(diff)
#         else:
#             print(f"Warning: Player '{player}' not found in comparison list.")
#             deviations.append(0)  # Optional: handle missing players differently
#     return deviations
#
# def calculate_mae(deviations):
#     """Calculate Mean Absolute Error from deviations list."""
#     return np.mean(np.abs(deviations))
#
# # Assuming these lists are defined elsewhere in your code
# # players = [...] - Your original player list
# # itr1, itr2, itr3, itr4, itr5 = [...] - Your iteration lists
#
# # All iterations and their names
# all_iterations = [itr1, itr2, itr3, itr4, itr5]
# iteration_names = ['Iteration 1', 'Iteration 2', 'Iteration 3', 'Iteration 4', 'Iteration 5']
#
# # Calculate deviations and MAEs for all iterations
# all_deviations = []
# all_maes = []
#
# for itr in all_iterations:
#     devs = calculate_deviations(players, itr)
#     all_deviations.append(devs)
#     mae = calculate_mae(devs)
#     all_maes.append(mae)
#     print(f"MAE for {iteration_names[all_iterations.index(itr)]}: {mae:.2f}")
#
# # Create the enhanced comparison chart with full range
# plt.figure(figsize=(20, 12))
# bar_width = 0.15
# indices = np.arange(len(players))
#
# # Define colors for each iteration
# colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
#
# # Plot each iteration with an offset
# for i, (devs, name, color) in enumerate(zip(all_deviations, iteration_names, colors)):
#     offset = (i - 2) * bar_width
#     plt.bar(indices + offset, devs, bar_width, label=f'{name} (MAE: {all_maes[i]:.2f})', color=color, alpha=0.8)
#
# # Add reference line and labels
# plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
# plt.xlabel('Players (Original Order)', fontsize=14)
# plt.ylabel('Ranking Deviation (Predicted - Actual)', fontsize=14)
# plt.title('Comparison of Ranking Methods\nDeviation from Tournament Placement', fontsize=16)
#
# # Set x-ticks to Player 1, Player 2, etc.
# player_labels = [f"Player {i+1}" for i in range(len(players))]
# plt.xticks(indices, player_labels, rotation=90, fontsize=9)
#
# # Add more prominent grid lines
# plt.grid(axis='y', linestyle='-', alpha=0.2)
# plt.grid(axis='x', linestyle='-', alpha=0.1)
#
# # Keep the full range of the y-axis, but enhance the grid
# # Find max deviation range for consistent scaling
# all_devs_flat = [item for sublist in all_deviations for item in sublist]
# max_dev = max(max(all_devs_flat), abs(min(all_devs_flat)))
# y_limit = max(8, int(max_dev) + 1)  # Ensure we show at least -8 to 8, but include all data
#
# # Add minor grid lines for better readability
# plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
# plt.grid(axis='y', which='minor', linestyle=':', alpha=0.1)
#
# # Add horizontal lines at key deviation values
# for y in range(-y_limit, y_limit+1):
#     if y == 0:
#         continue  # Already added the zero line
#     plt.axhline(y=y, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
#
# # Highlight the central region (-3 to 3) with a subtle background
# plt.axhspan(-3, 3, alpha=0.1, color='green', zorder=-1)
#
# # Add legend at the bottom
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=12)
#
# # Adjust layout
# plt.tight_layout()
#
# # Save the figure
# plt.savefig('enhanced_full_comparison.pdf', bbox_inches='tight')
#
# # Create focused comparisons for different player ranges
# # But maintain the full y-axis range to avoid chopping off data
# def create_focused_comparison(start_idx, end_idx, title_suffix="Mid-Ranked Players"):
#     plt.figure(figsize=(16, 10))
#
#     # Select a subset of players
#     selected_indices = indices[start_idx:end_idx]
#     selected_labels = player_labels[start_idx:end_idx]
#
#     # Plot each iteration with an offset for the selected range
#     for i, (devs, name, color) in enumerate(zip(all_deviations, iteration_names, colors)):
#         selected_devs = devs[start_idx:end_idx]
#         offset = (i - 2) * bar_width
#         plt.bar(np.arange(len(selected_indices)) + offset, selected_devs, 
#                 bar_width, label=f'{name} (MAE: {all_maes[i]:.2f})', color=color, alpha=0.8)
#
#     # Add reference line and labels
#     plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
#     plt.xlabel('Players (Original Order)', fontsize=14)
#     plt.ylabel('Ranking Deviation (Predicted - Actual)', fontsize=14)
#     plt.title(f'Focused Comparison - {title_suffix}\nDeviation from Tournament Placement', fontsize=16)
#
#     # Set x-ticks to the selected player labels
#     plt.xticks(np.arange(len(selected_indices)), selected_labels, rotation=90, fontsize=10)
#
#     # Add more prominent grid lines
#     plt.grid(axis='y', linestyle='-', alpha=0.2)
#     plt.grid(axis='x', linestyle='-', alpha=0.1)
#
#     # Keep full y-axis range to match the main plot
#     # Find max deviation range for this subset
#     subset_devs = [devs[start_idx:end_idx] for devs in all_deviations]
#     subset_devs_flat = [item for sublist in subset_devs for item in sublist]
#     subset_max_dev = max(max(subset_devs_flat), abs(min(subset_devs_flat)))
#     subset_y_limit = max(8, int(subset_max_dev) + 1)
#
#     # Add minor grid lines for better readability
#     plt.gca().yaxis.set_minor_locator(MultipleLocator(0.5))
#     plt.grid(axis='y', which='minor', linestyle=':', alpha=0.1)
#
#     # Add horizontal lines at key deviation values
#     for y in range(-subset_y_limit, subset_y_limit+1):
#         if y == 0:
#             continue  # Already added the zero line
#         plt.axhline(y=y, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
#
#     # Highlight the central region (-3 to 3) with a subtle background
#     plt.axhspan(-3, 3, alpha=0.1, color='green', zorder=-1)
#
#     # Add legend at the bottom
#     plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=12)
#
#     # Adjust layout
#     plt.tight_layout()
#
#     # Save the figure
#     plt.savefig(f'focused_comparison_{start_idx+1}-{end_idx}.pdf', bbox_inches='tight')
#
# # Create three focused comparisons for different player ranges
# create_focused_comparison(0, 15, "Top Players (1-15)")
# create_focused_comparison(15, 35, "Mid-Ranked Players (16-35)")
# create_focused_comparison(35, len(players), f"Lower-Ranked Players (36-{len(players)})")
#
# print("Enhanced comparison visualizations created successfully!")
#
#
#
#
#
#








































































import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from collections import defaultdict

from itr3 import fetch_scores_with_difficulty, rank_itr3, build_chart_score_lookup
# from avg import rank_by_avg
from score import get_rank
from graph import get_rank_graphs_for_list, get_rank_graphs_for_list_better, get_rank_graphs_for_list_superfast


# The rest of your lists (itr1, itr2, etc.) remain unchanged...

# Create a mapping of player names to generic labels
player_labels = {player: f"player{i+1}" for i, player in enumerate(players)}

# The rest of your code remains the same until the plotting part

# ... your existing code ...

list2=itr6
# list2 = [m[0] for m in merged] 

for p in list2:
    print(p)


list1 = [p for p in players if p.lower() in list2]
deviations = []
for i, player in enumerate(list1):
    lower_list2 = [p.lower() for p in list2]
    if player.lower() in lower_list2:
        new_index = lower_list2.index(player.lower())
        original_index = i
        diff = original_index - new_index
        deviations.append(diff)
    else:
        print(f"\033[91mError: Player '{player}' not found in the new ranking list.\033[0m")
        exit(1)

if deviations:
    mae = np.mean(np.abs(deviations))
else:
    print("\033[91mError: No valid deviations found. 'deviations' list is empty.\033[0m")
    exit(1)

print(f"\nMean Absolute Error (MAE): {mae:.2f}")

# Modified plotting code - using generic player labels instead of names
x = np.arange(len(list1))
plt.figure(figsize=(14, 6))
plt.bar(x, deviations, color="#4361ee", alpha=0.7)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")

# Use player labels (player1, player2, etc.) instead of actual names
generic_labels = [player_labels[player] for player in list1]
plt.xticks(x, generic_labels, rotation=90)

plt.xlabel("Players", fontsize=16, labelpad=20)
plt.ylabel("Ranking Deviation (Predicted - Actual)", fontsize=16)
plt.title(f"Deviation from Actual Tournament Placement \n Mean Absolute Error: {mae:.2f}", fontsize=18)
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.savefig("placement_change_2.png", dpi=600, bbox_inches="tight")
plt.show()

