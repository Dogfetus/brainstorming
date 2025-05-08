from scipy.stats import linregress
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
import sqlite3


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


def get_average_score_for_chart(chart_id):
    con = sqlite3.connect("new_data.db")
    cur = con.cursor()

    dogfetus = cur.execute("""
        SELECT AVG(score)
        FROM scores
        WHERE chart_id = ?
    """, (chart_id,)).fetchone()

    con.close()

    return dogfetus[0]



def get_average_score_for_difficulty(difficulty, display = None, name = None):
    conn = sqlite3.connect("new_data.db")
    cursor = conn.cursor()


    if display:
        query = """
            SELECT AVG(scores.score)
            FROM scores
            JOIN charts ON scores.chart_id = charts._id
            WHERE charts.difficulty = ? AND charts.difficulty_display LIKE ?;
        """
        cursor.execute(query, (difficulty, display))
        
    elif name:
        query = """
            SELECT AVG(scores.score)
            FROM scores
            JOIN charts ON scores.chart_id = charts._id
            WHERE charts.difficulty = ? AND charts.difficulty_name LIKE ?;
        """
        cursor.execute(query, (difficulty, name))

    else:
        query = """
            SELECT AVG(scores.score)
            FROM scores
            JOIN charts ON scores.chart_id = charts._id
            WHERE charts.difficulty = ?;
        """
        cursor.execute(query, (difficulty,))

    result = cursor.fetchone()

    conn.close()

    return result[0] if result and result[0] is not None else 0


def difficulties_for_display(display, name = None):
    conn = sqlite3.connect("new_data.db")
    cursor = conn.cursor()

    query = """
        SELECT DISTINCT difficulty
        FROM charts
        WHERE difficulty_display LIKE ?;
    """
    cursor.execute(query, (display,))

    if name:
        query = """
            SELECT DISTINCT difficulty
            FROM charts
            WHERE difficulty_name LIKE ?;
        """
        cursor.execute(query, (name,))

    result = cursor.fetchall()

    conn.close()
    return sorted([x[0] for x in result])


def get_difficulty_displays(singles = False):
    conn = sqlite3.connect("new_data.db")
    cursor = conn.cursor()

    query = """
        SELECT DISTINCT difficulty_display
        FROM charts;
    """

    if singles:
        query = """
            SELECT DISTINCT difficulty_display
            FROM charts
            WHERE difficulty_display NOT in ('dual', 'dual+', 'full+', 'full', 'edit');
        """

    cursor.execute(query)
    result = cursor.fetchall()

    conn.close()

    return sorted([x[0] for x in result])


def get_difficulty_names(singles = False):
    conn = sqlite3.connect("new_data.db")
    cursor = conn.cursor()

    query = """
        SELECT DISTINCT difficulty_name
        FROM charts;
    """

    if singles:
        query = """
            SELECT DISTINCT difficulty_name
            FROM charts
            WHERE difficulty_name NOT in ('dual', 'dual2', 'full', 'full2', 'edit');
        """

    cursor.execute(query)
    result = cursor.fetchall()

    conn.close()

    return [x[0] for x in result]






































avg_display = []
displays = get_difficulty_displays(singles=True)

for display in displays:
    avg_temp = {}

    difficulties = difficulties_for_display(display)
    for difficulty in difficulties:
        avg_temp[difficulty] = get_average_score_for_difficulty(difficulty=difficulty, display=display)

    avg_display.append({display: avg_temp})


for item in avg_display:
    for display, difficulties in item.items():
        print(f"Display: {display}")
        for difficulty, avg_score in difficulties.items():
            print(f"  Difficulty: {difficulty}, Average Score: {avg_score}")
        print("-" * 30)


# get average score for difficulty and display
def gasfdad(difficulty, display):
    for item in avg_display:
        if display in item:
            difficulties = item[display]
            return difficulties.get(difficulty, None)  
    return None 


def difference(your_score, avg_score):
    return (your_score - avg_score)



def value_of_score(score, difficulty, display):
    average_score = gasfdad(difficulty, display)
    if average_score is None or average_score == 0:
        return 0  # or float('-inf') or some fallback

    value = difference(score, average_score)

    return value 


# value_of_score(80999, 27, "wild")









def last_scr_for_dif_and_dis(player, x, dif, dis, date, cleared=1, ignore_cleared=False):
    conn = sqlite3.connect("new_data.db")
    cursor = conn.cursor()

    one_year_before = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")

    query = """
        SELECT scores.score, charts.difficulty, charts.difficulty_display
        FROM scores
        JOIN charts ON scores.chart_id = charts._id
        JOIN gamers ON scores.gamer_id = gamers._id
        WHERE gamers.username = ? COLLATE NOCASE
        AND charts.difficulty = ?
        AND charts.difficulty_display = ?
        AND scores.created_at < ?
    """

    params = [player, dif, dis, date]

    if not ignore_cleared:
        query += " AND scores.cleared = ?"
        params.append(cleared)

    query += " ORDER BY scores.created_at DESC LIMIT ?"
    params.append(x)

    cursor.execute(query, params)
    results = cursor.fetchall()

    conn.close()
    
    return results

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















#
def detect_abnormal_scores(player_scores):
    """
    Detects abnormally low scores for a player using adaptive anomaly detection.
    Optimized for performance and readability.

    :param player_scores: List of (chart_difficulty, score) tuples for a player.
    :return: List of flagged scores that are unusually low.
    """
    # Early return if no scores
    if not player_scores or len(player_scores) < 2:
        return []

    num_scores = len(player_scores)

    # Pre-compute arrays once - avoid repeated list comprehensions
    difficulties, scores = zip(*player_scores)
    difficulties = np.array(difficulties)
    scores = np.array(scores)

    # Adaptive parameters - calculate once outside the loop
    percentile_threshold = max(1, 10 - (num_scores * 0.05))
    deviation_factor = max(0.05, 0.2 - (num_scores * 0.001))
    std_multiplier = min(2.5, 1.5 + (num_scores * 0.005))

    # Calculate statistics once
    median_score = np.median(scores)
    mean_score = np.mean(scores)
    std_dev = np.std(scores)
    percentile_cutoff = np.percentile(scores, percentile_threshold)

    # Optimization: Use vectorized operations with numpy
    # Only compute linear regression if we have enough data points
    if len(scores) > 2:
        # Compute the linear regression once
        slope, intercept, _, _, _ = linregress(difficulties, scores)
        # Calculate expected scores for all difficulties at once
        expected_scores = slope * difficulties + intercept
    else:
        # If not enough data, use median as expected score
        expected_scores = np.full_like(difficulties, median_score, dtype=float)

    # Calculate std_dev_cutoff once
    std_dev_cutoff = median_score - (std_multiplier * std_dev)

    # Compute z-scores for all at once
    z_scores = np.zeros_like(scores, dtype=float)
    if std_dev > 0:
        z_scores = (scores - mean_score) / std_dev

    # Create arrays for each check
    below_percentile = scores < percentile_cutoff
    below_std_dev = scores < std_dev_cutoff
    below_z_score = np.abs(z_scores) > 2.5

    # Calculate dynamic deviation thresholds
    dynamic_thresholds = np.maximum(5000, expected_scores * deviation_factor)
    below_expected = (expected_scores - scores) > dynamic_thresholds

    # Count conditions that are met for each score
    condition_counts = below_percentile.astype(int) + below_expected.astype(int) + \
                      below_std_dev.astype(int) + below_z_score.astype(int)

    # Find indices where at least 2 conditions are met
    flagged_indices = np.where(condition_counts >= 2)[0]

    # Create the result list
    flagged_scores = [(difficulties[i], scores[i], expected_scores[i]) 
                      for i in flagged_indices]

    return flagged_scores















def detect_abnormal_scores2(player_scores):
    """
    Detects low outlier scores using only percentile-based thresholding.

    :param player_scores: List of (chart_difficulty, score) tuples for a player.
    :return: List of (difficulty, score) tuples flagged as abnormally low.
    """
    if not player_scores or len(player_scores) < 2:
        return []

    difficulties, scores = zip(*player_scores)
    scores = np.array(scores)
    difficulties = np.array(difficulties)

    percentile_threshold = max(1, 10 - 0.05 * len(scores))  # adaptive
    print("percentile_threshold: ", percentile_threshold)
    cutoff = np.percentile(scores, percentile_threshold)
    print("cutoff: ", cutoff)

    # Flag scores below the cutoff
    flagged_indices = np.where(scores < cutoff)[0]

    # Return list of (difficulty, score) tuples
    return [(difficulties[i], scores[i], "b") for i in flagged_indices]


def detect_abnormal_scores3(player_scores):
    """
    Detects low outlier scores using a more robust method that identifies clusters of low scores.
    
    :param player_scores: List of (chart_difficulty, score) tuples for a player.
    :return: List of (difficulty, score, reason) tuples flagged as abnormally low.
    """
    if not player_scores or len(player_scores) < 3:  # Need at least 3 points to detect outliers
        return []
    
    # Extract difficulties and scores
    difficulties, scores = zip(*player_scores)
    difficulties = np.array(difficulties)
    scores = np.array(scores)
    
    # Method 1: Use IQR (Interquartile Range) to detect outliers
    # This is a standard statistical method for outlier detection
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    
    # Define lower bound - anything below this is considered an outlier
    # The 1.5 multiplier is a common threshold in statistics
    lower_bound = q1 - 1.5 * iqr
    
    # Method 2: Look for large gaps in sorted scores that might indicate outliers
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    
    # Calculate differences between adjacent scores
    score_diffs = np.diff(sorted_scores)
    
    # Find large gaps (e.g., where the difference is more than X% of the mean score)
    mean_score = np.mean(scores)
    large_gap_threshold = 0.3 * mean_score  # 30% of the mean score is a significant gap
    
    large_gap_indices = np.where(score_diffs > large_gap_threshold)[0]
    
    # If we find large gaps, anything below the lowest large gap is an outlier
    gap_flagged = set()
    if len(large_gap_indices) > 0:
        lowest_good_score_index = sorted_indices[large_gap_indices[0] + 1]
        lowest_good_score = scores[lowest_good_score_index]
        
        # Flag all scores below this threshold
        for i, score in enumerate(scores):
            if score < lowest_good_score:
                gap_flagged.add(i)
    
    # Combine both methods
    # Flag scores that are either below the IQR lower bound or below a large gap
    flagged_indices = set(np.where(scores < lower_bound)[0]) | gap_flagged
    
    # Return the results as tuples with a reason code
    result = []
    for i in flagged_indices:
        if i in np.where(scores < lower_bound)[0] and i in gap_flagged:
            reason = "both"  # Flagged by both methods
        elif i in np.where(scores < lower_bound)[0]:
            reason = "iqr"   # Flagged by IQR method
        else:
            reason = "gap"   # Flagged by gap method
        
        result.append((difficulties[i], scores[i], reason))
    
    return result


def detect_abnormal_scores4(player_scores):
    """
    Detects low outlier scores using the IQR (Interquartile Range) method.
    
    :param player_scores: List of (chart_difficulty, score) tuples for a player.
    :return: List of (difficulty, score, reason) tuples flagged as abnormally low.
    """
    if not player_scores or len(player_scores) < 3:  # Need enough points for quartiles
        return []
    
    # Extract difficulties and scores
    difficulties, scores = zip(*player_scores)
    difficulties = np.array(difficulties)
    scores = np.array(scores)
    
    # Calculate the first quartile (25th percentile)
    q1 = np.percentile(scores, 22)
    
    # Calculate the third quartile (75th percentile)
    q3 = np.percentile(scores, 75)
    
    # Calculate the IQR
    iqr = q3 - q1
    
    # Define the lower bound - standard practice is 1.5 * IQR below Q1
    # You can adjust the multiplier (1.5) to make it more or less strict
    lower_bound = q1 - 0.5 * iqr
    print("lower_bound: ", lower_bound)
    
    # Find all scores below the lower bound
    flagged_indices = np.where(scores < lower_bound)[0]
    
    # Return the results
    return [(difficulties[i], scores[i], "iqr") for i in flagged_indices]

# Test with the example
test_scores = [
    (1, 76000),
    (2, 98000),
    (3, 37000),
    (4, 40000),
    (5, 99000),
    (6, 97000),
    (7, 99000),
    (8, 99000),
    (9, 99000),
    (10, 99000),
    (10, 99000),
    (10, 99000),
    (10, 99000),
    (10, 99000),
    (10, 99000),
    (10, 99000),
    (10, 99000),
]

print("Abnormal Scores:", detect_abnormal_scores2(test_scores))







def rank_players_by_categories(players, categories, get_rank_function, date="2026-10-10"):
    """
    Rank players across different difficulty categories and split them evenly.

    Args:
        players (list): List of players to rank
        categories (list): List of categories to use for ranking, in order of preference
                          e.g. ['wild', 'hard+', 'hard'] or ['basic', 'easy', 'hard']
        get_rank_function (callable): Function that accepts a player and a category list 
                                     and returns a numerical rank

    Returns:
        tuple: (grouped_player_ranks, grouped_player_and_ranks)
               - grouped_player_ranks: List of players in ranked order
               - grouped_player_and_ranks: List of (player, rank) tuples in ranked order
    """
    if not players or not categories:
        return [], []

    # Initialize result lists
    grouped_player_ranks = []
    grouped_player_and_ranks = []

    # Calculate how many categories to use
    num_categories = len(categories)
    # Calculate approx how many players to include per category
    players_per_category = len(players) // num_categories
    # Handle the remainder
    remainder = len(players) % num_categories

    # For each category
    for i, category in enumerate(categories):
        temp = []

        # Get players that haven't been ranked yet
        missing_players = [p for p in players if p not in grouped_player_ranks]
        if not missing_players:
            break

        # Calculate ranks for the current category
        for player in missing_players:
            rank = get_rank_function(player, dis=[category], date=date)
            temp.append((player, rank))

        # Determine how many players to take from this category
        players_to_take = players_per_category
        if i < remainder:  # Distribute remainder players among first categories
            players_to_take += 1

        # If we're at the last category, take all remaining players
        if i == num_categories - 1:
            players_to_take = len(missing_players)

        # Sort by rank and take the appropriate number of players
        for player, rank in sorted(temp, key=lambda x: x[1], reverse=True)[:players_to_take]:
            grouped_player_ranks.append(player)
            grouped_player_and_ranks.append((player, rank))

    return grouped_player_ranks, grouped_player_and_ranks





























grouped_ranks = []
date_when_grouped = None

def get_rank(player, x_scores=10, dif = [], dis = ['hard', 'wild', 'hard+'], date="2026-10-10"):
    global grouped_ranks, date_when_grouped
    dif_displayes = dis    
    scores = []

    # retrieve the last x scores for a player 
    for display in dif_displayes:
        difficulties = difficulties_for_display(display)
        for difficulty in difficulties:
            print("display: ", display, " difficulty: ", difficulty, "score: ")
            last = last_scr_for_dif_and_dis(player, x_scores, difficulty, display, date) 
            print(last)
            scores.append(last)



    # get the average of those scores per difficulty 
    difficulty_scores = defaultdict(lambda: {"total": 0, "count": 0})
    for score_set in scores:
        for score in score_set:
            print("single: ", score)
            # Extract relevant fields
            score_value, difficulty, display = score 
            key = (difficulty, display)  # Group by (difficulty, display)

            # Aggregate scores
            difficulty_scores[key]["total"] += score_value
            difficulty_scores[key]["count"] += 1
    avg_scores = {key: difficulty_scores[key]["total"] / difficulty_scores[key]["count"]
                for key in difficulty_scores}


    # remove abnormailites (scores that are too low compared to the one above and below)
    ok = []
    for (difficulty, display), avg in avg_scores.items():
        ok.append((difficulty, avg))
    flagged_scores = detect_abnormal_scores2(ok)
    print("Flagged scores: ", flagged_scores)
    flagged_set = {(d, s) for (d, s, _) in flagged_scores}


    # calculate the value of the average score
    values = []
    for (difficulty, display), avg in avg_scores.items():
        print(f"Difficulty: {difficulty}, Display: {display}, Average Score: {avg:.2f}")
        is_flagged = (difficulty, avg) in flagged_set
        if not is_flagged:
            values.append(value_of_score(avg, difficulty, display))



    # return the average value of the scores
    return sum(values) / (len(values) or 1)



# player=""
# x_scores = 10
# date = "2024-10-17"
# dis = ['wild']
# dif_displayes = dis    
# scores = []
#
#
#
#
# # retrieve the last x scores for a player 
# for display in dif_displayes:
#     difficulties = difficulties_for_display(display)
#     for difficulty in difficulties:
#         print("display: ", display, " difficulty: ", difficulty, "score: ")
#         last = last_scr_for_dif_and_dis(player, x_scores, difficulty, display, date) 
#         print(last)
#         scores.append(last)
#
#
#
#
# # get the average of those scores per difficulty 
# difficulty_scores = defaultdict(lambda: {"total": 0, "count": 0})
# for score_set in scores:
#     for score in score_set:
#         print("single: ", score)
#         # Extract relevant fields
#         score_value, difficulty, display = score 
#         key = (difficulty, display)  # Group by (difficulty, display)
#
#         # Aggregate scores
#         difficulty_scores[key]["total"] += score_value
#         difficulty_scores[key]["count"] += 1
# avg_scores = {key: difficulty_scores[key]["total"] / difficulty_scores[key]["count"]
#             for key in difficulty_scores}
#
#
#
#
# # remove abnormailites (scores that are too low compared to the one above and below)
# ok = []
# for (difficulty, display), avg in avg_scores.items():
#     ok.append((difficulty, avg))
# flagged_scores = detect_abnormal_scores(ok)
# print("Flagged scores: ", flagged_scores)
# flagged_set = {(d, s) for (d, s, _) in flagged_scores}
#
#
#
#
#
# # Safe version for direct integration into existing code
# try:
#     # Try the efficient approach first
#     max_item = max(avg_scores.items(), key=lambda item: item[1])
#     highest_score = max_item[1]
#     highest_score_difficulty, highest_score_display = max_item[0]
# except ValueError:
#     # Handle the case where avg_scores is empty
#     highest_score = 0
#     highest_score_difficulty = 27 
#
# # calculate the value of the average score
# values = []
# # print the average score for each difficulty and display
# for (difficulty, display), avg in avg_scores.items():
#     print(f"Difficulty: {difficulty}, Display: {display}, Average Score: {avg:.2f}")
#     is_flagged = (difficulty, avg) in flagged_set and difficulty < highest_score_difficulty 
#     if not is_flagged:
#         values.append(value_of_score(avg, difficulty, display))
#
#
# # return the average value of the scores
# sum(values) / (len(values) or 1)







# Example call:
# players = ["player1", "player2", "player3", ...]
# categories = ["wild", "hard+", "hard"]
# ranked_players, ranked_players_with_scores = rank_players_by_categories(players, categories, get_rank, date="2024-10-17")





# dogfetus = get_rank("dogfetus")
























