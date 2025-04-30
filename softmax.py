
# > sqlite3 new_data.db
# SQLite version 3.45.3 2024-04-15 13:34:05
# Enter ".help" for usage hints.
# sqlite> .schema
# CREATE TABLE songs (
#         _id INTEGER PRIMARY KEY,
#         allow_edits INTEGER,
#         artist TEXT,
#         bpm TEXT,
#         cover TEXT,
#         cover_path TEXT,
#         cover_thumb TEXT,
#         created_at TEXT,
#         extra TEXT,  -- JSON-like data stored as TEXT
#         first_beat INTEGER,
#         first_ms INTEGER,
#         game_song_id INTEGER,
#         genre TEXT,
#         id INTEGER,  -- Duplicate of _id (consider removing)
#         is_enabled INTEGER,
#         label TEXT,
#         last_beat INTEGER,
#         last_ms INTEGER,
#         release_date TEXT,
#         subtitle TEXT,
#         timing_bpms TEXT,
#         timing_offset_ms INTEGER,
#         timing_stops TEXT,
#         title TEXT,
#         updated_at TEXT,
#         website TEXT
#     );
# CREATE TABLE charts (
#         _id INTEGER PRIMARY KEY,
#         created_at TEXT,
#         difficulty INTEGER,
#         difficulty_display TEXT,
#         difficulty_id INTEGER,
#         difficulty_name TEXT,
#         game_difficulty_id INTEGER,
#         graph TEXT,  -- JSON-like array stored as TEXT
#         id INTEGER,  -- Duplicate of _id (consider removing)
#         is_enabled INTEGER,
#         meter INTEGER,
#         pass_count INTEGER,
#         play_count INTEGER,
#         song_id INTEGER,
#         steps_author TEXT,
#         steps_index INTEGER,
#         updated_at TEXT,
#         FOREIGN KEY (song_id) REFERENCES songs(_id) ON DELETE CASCADE
#     );
# CREATE TABLE gamers (
#         _id INTEGER PRIMARY KEY,
#         country TEXT,
#         description TEXT,
#         hex_color TEXT,  -- Nullable field
#         id INTEGER,  -- Duplicate of _id (consider removing)
#         picture_path TEXT,  -- Nullable field
#         private BOOLEAN,
#         published_edits INTEGER,
#         rival INTEGER,
#         username TEXT
#     );
# CREATE TABLE scores (
#         _id INTEGER PRIMARY KEY,
#         calories INTEGER,
#         chart_id INTEGER,
#         cleared BOOLEAN,
#         created_at TEXT,
#         early INTEGER,
#         flags INTEGER,
#         full_combo BOOLEAN,
#         gamer_id INTEGER,
#         global_flags INTEGER,
#         grade INTEGER,
#         green INTEGER,
#         late INTEGER,
#         max_combo INTEGER,
#         misses INTEGER,
#         music_speed INTEGER,
#         perfect1 INTEGER,
#         perfect2 INTEGER,
#         red INTEGER,
#         score INTEGER,
#         side TEXT,
#         song_chart_id INTEGER,
#         steps INTEGER,
#         updated_at TEXT,
#         uuid TEXT,
#         yellow INTEGER,
#         FOREIGN KEY (gamer_id) REFERENCES gamers(_id) ON DELETE CASCADE,
#         FOREIGN KEY (song_chart_id) REFERENCES charts(_id) ON DELETE CASCADE
#     );
# sqlite> 
#

import sqlite3
import requests
import math
import statistics as stats
from datetime import datetime



EXTRA_EP = """https://api.smx.573.no/extra/{}"""

def player_exists(gamer_name):
    con = sqlite3.connect("new_data.db")
    cur = con.cursor()

    dogfetus = cur.execute("""
        SELECT _id
        FROM gamers
        WHERE username LIKE ?
    """, (gamer_name,)).fetchone()

    con.close()

    return dogfetus


def number_of_charts_per_difficulty(difficulty, display):
    con = sqlite3.connect("new_data.db")
    cur = con.cursor()

    dogfetus = cur.execute("""
        SELECT COUNT(charts._id)
        FROM charts
        WHERE difficulty = ?
        AND difficulty_display LIKE ?
    """, (difficulty, display)).fetchone()

    con.close()

    return dogfetus[0]


def last_x_scores_per_difficulty(gamer_name, difficulty, x=10, before_time=None, full=False):
    con = sqlite3.connect("new_data.db")
    cur = con.cursor()

    if before_time and "T" in before_time and "Z" in before_time:
        before_time = before_time.replace("T", " ").replace("Z", "")

    query = """
        SELECT scores.*
        FROM scores
        JOIN gamers ON scores.gamer_id = gamers._id
        JOIN charts ON scores.chart_id = charts._id
        WHERE gamers.username LIKE ?
        AND charts.difficulty = ?
        AND charts.difficulty_name IN ({})
    """.format(
        "'dual', 'dual2', 'full', 'full2'" if full else "'basic','easy','easy2','hard','hard 2','wild'"
    )

    params = [gamer_name, difficulty]

    if before_time:
        query += " AND DATETIME(replace(replace(scores.created_at, 'T', ' '), 'Z', '')) < DATETIME(?)"
        params.append(before_time)

    query += " ORDER BY scores._id DESC LIMIT ?"
    params.append(x)

    print("Executing SQL Query:")
    print(query)
    print("With parameters:", params)

    dogfetus = cur.execute(query, params).fetchall()
    con.close()
    return dogfetus


def get_scores(gamer_name, difficulty, limit=10, before_time=None):
    con = sqlite3.connect("new_data.db")
    cur = con.cursor()

    scores = cur.execute("""
        SELECT scores.*
        FROM scores
        JOIN gamers ON scores.gamer_id = gamers._id
        JOIN charts ON scores.chart_id = charts._id
        WHERE gamers.username LIKE ?
        AND charts.difficulty = ?
        AND charts.difficulty_name IN ('basic','easy','easy2','hard','hard 2','wild')
    """, (gamer_name, difficulty)).fetchall()

    con.close()

    if before_time is None:
        return scores[:limit]

    before_time = datetime.strptime(before_time, "%Y-%m-%dT%H:%M:%SZ")

    filtered_scores = []
    for score in scores:
        created_at = score[4] 
        
        created_at_dt = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")

        if created_at_dt < before_time:
            filtered_scores.append(score)

    return filtered_scores[:limit] 

scores = get_scores("dogfetus", 10, limit=5, before_time="2024-10-17T23:59:59Z")
scores2 = get_scores("dogfetus", 10, limit=5)
score3 = last_x_scores_per_difficulty("dogfetus", 10, x=5)

for score in scores:
    print(score[0])

for score in scores2:
    print(score[0])

for score in score3:
    print(score[0])


def get_last_score(gamer_name):
    con = sqlite3.connect("new_data.db")
    cur = con.cursor()

    dogfetus = cur.execute("""
        SELECT scores._id 
        FROM scores
        JOIN gamers ON scores.gamer_id = gamers._id
        WHERE gamers.username LIKE ?
        ORDER BY scores._id DESC
        LIMIT 1
    """, (gamer_name,)).fetchall()

    con.close()

    return dogfetus[0]


def get_score(id):
    con = sqlite3.connect("new_data.db")
    cur = con.cursor()

    dogfetus = cur.execute("""
        SELECT score
        FROM scores
        WHERE scores._id = ?
    """, (id,)).fetchone()

    con.close()

    return dogfetus[0]


score = get_last_score("micha")
score = 2800109 

graph = requests.get(EXTRA_EP.format(score)).json() 

data = graph["offsets"][1:]
time = data[0::3]
offset = data[1::3]
judgment = data[2::3]


def graph_worth(graph):
    mi = min(graph) 
    ma = max(graph)
    mean = sum(graph) / len(graph)
    stdev = stats.stdev(graph)
    median = stats.median(graph)
    mode = stats.mode(graph)
    variance = stats.variance(graph)
    above = len([x for x in graph if x > 0])
    below = len([x for x in graph if x < 0])
    k = 1.0

    # print(f"Min: {mi}")
    # print(f"Max: {ma}")
    # print(f"Mean: {mean}")
    # print(f"Stdev: {stdev}")
    # print(f"Median: {median}")
    # print(f"Mode: {mode}")
    # print(f"Variance: {variance}")
    # print(f"Above: {above}")
    # print(f"Below: {below}")
    #

    result = (abs(mean) + variance * k) or 0.0001

    return 100/result

print(graph_worth(offset))

def score_worth(score, difficulty=0):
    k = 1
    if isinstance(score, int):
        return score * difficulty * k
    else:
        return score[19] * difficulty * k


def get_higher_score(scores, index, limit=27):
    """for a given list, this retreives the first score that is not null from the higher difficulties"""
    for i in range(index, limit):
        if scores[i][0]:
            return scores[i][0], i+1
    return None, None

def get_lower_score(scores, index):
    """for a given list, this retreives the first score that is not null from the lower difficulties"""
    for i in range(index, 0, -1):
        if scores[i-1][0]:
            return scores[i-1][0], i
    return None, None


def get_rank(user, min_difficulty=1, max_difficulty=27):
    if not player_exists(user):
        return None

# get last 10 scores for each difficulty (mean for each difficulty)
    pre_scores = []
    mean_scores = []
    for i in range(min_difficulty, max_difficulty+1):
        values = last_x_scores_per_difficulty(user, i, before_time="2024-10-17 00:00:00")
        pre_scores.append((values, i))
        if values:
            mean_scores.append((stats.mean([x[19] for x in values]), i))
        else:
            mean_scores.append(([], i))


# fill in the missing scores
    all_scores = []
    for score, i in mean_scores:
        new_score = score
        if not new_score:
            new_score, new_i = get_higher_score(mean_scores, i, max_difficulty)
            if new_score:
                new_score = new_score - (1000-1000*math.log(abs(new_i - i)))
            else:
                new_score, new_i = get_lower_score(mean_scores, i)
                new_score = new_score * (i/new_i)**1.2 

        all_scores.append((new_score, i)) 


# get the graph worth for each score (mean for each difficulty)
    mean_graphs = []
    for scores, i in pre_scores:
        graphs = [] 
        for score in scores:
            graph = requests.get(EXTRA_EP.format(score[0])).json()
            if "offsets" not in graph:
                continue
            offsets = graph["offsets"][1:][1::3]
            graphs.append(offsets)
        if not graphs:
            mean_graphs.append(([], i))
        else:
            mean_graphs.append((stats.mean([graph_worth(x) for x in graphs]), i))


# fill in the missing graph worths
    all_graphs = []
    for graph, i in mean_graphs:
        new_graph = graph
        if not new_graph:
            higher, i_h = get_higher_score(mean_graphs, i, max_difficulty)
            lower, i_l = get_lower_score(mean_graphs, i)

            if higher and lower:
                new_graph = (higher + lower) / 2
            elif higher:
                new_graph = higher/2
            else:
                new_graph = lower * (1/abs(i - i_l))**2
        #thinking: get first higher and first lower graph use the mean
        # if higher is missing user a low value
        # if lower is missing use the high graph with worse value

        all_graphs.append((new_graph, i))


    rank = 0
    for score, graph in zip(all_scores, all_graphs):
        rank += score[0] * graph[0] * score[1]
        # print(f"Difficulty {score[1]}: {score[0]}, {graph[0]}")
    
    return rank



# get_rank("micha")
# get_rank("dogfetus")

scores = get_scores("dogfetus", 1)
print(scores)

t_1 = last_x_scores_per_difficulty("dogfetus", 10, before_time="2024-10-17T23:59:59Z")
t_2 = last_x_scores_per_difficulty("dogfetus", 10)

for i in t_1:
    print(i[0])

for i in t_2:
    print(i[0])

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
    "peter",
    "jokko",
    "butterbutt",
    "jewel",
    "beatao",
    "maverick",
]

# all_ranks = []
# for player in players:
#     rank = get_rank(player)
#     print(f"{player}: {rank}")
#     all_ranks.append((player, rank))
#
# for player, rank in sorted(all_ranks, key=lambda x: x[1]):
#     print(f"{player}: {rank}")
