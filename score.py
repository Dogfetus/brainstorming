import last_scores
from scipy.stats import linregress
import numpy as np
import score_evaluater
from collections import defaultdict


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


# Example usage
player_scores = [
    (10, 100000), (12, 98000), (11, 97000), (9, 96000), (8, 90000),
    (10, 100000), (12, 98000), (11, 97000), (9, 96000), (8, 90000),
    (10, 100000), (12, 98000), (11, 97000), (9, 96000), (8, 90000),
    (7, 77000), (6, 49000)  # Last one should be flagged, 89k should not
]

abnormal_scores = detect_abnormal_scores(player_scores)

print("Abnormal scores:", abnormal_scores)




















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

def get_rank(player, dif = [], dis = ['hard', 'wild', 'hard+'], date="2026-10-10", grouped=False):
    global grouped_ranks, date_when_grouped
    dif_displayes = dis    
    x_scores = 10
    scores = []

    if grouped:
        if not date_when_grouped or date_when_grouped != date:
            _, groupd_player_and_ranks = rank_players_by_categories(players, dis, get_rank, date=date)
            date_when_grouped = date
            grouped_ranks = groupd_player_and_ranks

        return next((rank for p, rank in grouped_ranks if p == player), None)

    # retrieve the last x scores for a player 
    for display in dif_displayes:

        difficulties = score_evaluater.difficulties_for_display(display)
        for difficulty in difficulties:
            print("display: ", display, " difficulty: ", difficulty, "score: ")
            last = last_scores.last_scr_for_dif_and_dis(player, x_scores, difficulty, display, date) 
            print(last)
            scores.append(last)



    # get the average of those scores per difficulty 
    difficulty_scores = defaultdict(lambda: {"total": 0, "count": 0})
    for score in scores:
        for single in score:
            print("single: ", single)
            # Extract relevant fields
            _, score_value, _, difficulty, display, _, _, _, _, _, _, _ = single
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
    flagged_scores = detect_abnormal_scores(ok)
    print("Flagged scores: ", flagged_scores)
    flagged_set = {(d, s) for (d, s, _) in flagged_scores}




    # calculate the value of the average score
    values = []
    for (difficulty, display), avg in avg_scores.items():
        print(f"Difficulty: {difficulty}, Display: {display}, Average Score: {avg:.2f}")
        is_flagged = (difficulty, avg) in flagged_set
        if not is_flagged:
            values.append(score_evaluater.value_of_score(avg, difficulty, display))



    # return the average value of the scores
    return sum(values) / (len(values) or 1)










# Example call:
# players = ["player1", "player2", "player3", ...]
# categories = ["wild", "hard+", "hard"]
# ranked_players, ranked_players_with_scores = rank_players_by_categories(players, categories, get_rank, date="2024-10-17")





# dogfetus = get_rank("dogfetus")
























