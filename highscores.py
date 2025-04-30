#
#
#er vanseklig siden ikke alle har spilt alle sangene:
# noan har spilt på en side av chartene og and har spilt på andre siden av chartene
#
#
# import sqlite3
# from collections import defaultdict
#
#
# players = [
#     "JimboSMX",
#     "hintz",
#     "spencer",
#     "swagman",
#     "paranoiaboi",
#     "masongos",
#     "grady",
#     "jjk.",
#     "chezmix",
#     "inglomi",
#     "jellyslosh",
#     "wdrm",
#     "eesa",
#     "senpi",
#     "Janus5k",
#     "tayman",
#     "emcat",
#     "pilot",
#     "mxl100",
#     "snowstorm",
#     "datcoreedoe",
#     "big matt",
#     "werdwerdus",
#     "cathadan",
#     "shinobee",
#     "mesyr",
#     "zom585",
#     "ctemi",
#     "zephyrnobar",
#     "sydia",
#     "kren",
#     "xosen",
#     "sweetjohnnycage",
#     "cheesecake",
#     "ditz",
#     "enderstevemc",
#     "jiajiabinks",
#     "meeko",
#     "momosapien",
#     "auby",
#     "arual",
#     "dogfetus",
#     "noir",
#     "dali",
#     " peter",
#     "jokko",
#     "butterbutt",
#     "jewel",
#     "beatao",
#     "maverick",
# ]
#
#
# # get all players from database
# def get_players():
#     conn = sqlite3.connect("new_data.db")
#     cursor = conn.cursor()
#     query = """  
#         SELECT _id, username 
#         FROM gamers;
#     """
#     cursor.execute(query)
#     results = cursor.fetchall()
#     conn.close()
#     return results
#
#
# # get player id from database
# def get_player_id(player):
#     conn = sqlite3.connect("new_data.db")
#     cursor = conn.cursor()
#     query = """  
#         SELECT _id 
#         FROM gamers
#         WHERE username = ? COLLATE NOCASE;
#     """
#     cursor.execute(query, (player,))
#     result = cursor.fetchone()
#     conn.close()
#     return result[0] if result else None
#
#
#
# def get_most_highscore_player(player_list):
#     player_ids = [pid for pid, _ in player_list]
#     conn = sqlite3.connect("new_data.db")
#     cursor = conn.cursor()
#
#     # Step 1: Get chart_ids where ALL players have scores
#     placeholders = ','.join('?' for _ in player_ids)
#     query_common_charts = f"""
#         SELECT chart_id
#         FROM scores
#         WHERE gamer_id IN ({placeholders})
#         GROUP BY chart_id
#         HAVING COUNT(DISTINCT gamer_id) = ?
#     """
#     cursor.execute(query_common_charts, (*player_ids, len(player_ids)))
#     common_chart_ids = [row[0] for row in cursor.fetchall()]
#
#     print(f"Common chart IDs: {common_chart_ids}")
#
#     if not common_chart_ids:
#         return None  # no common charts
#
#     # Step 2: Find the highest scorer per chart
#     highscore_counts = defaultdict(int)
#
#     for chart_id in common_chart_ids:
#         cursor.execute("""
#             SELECT gamer_id, MAX(score)
#             FROM scores
#             WHERE chart_id = ?
#               AND gamer_id IN ({})
#         """.format(placeholders), (chart_id, *player_ids))
#
#         top_result = cursor.fetchone()
#         if top_result:
#             top_gamer_id = top_result[0]
#             highscore_counts[top_gamer_id] += 1
#
#     conn.close()
#
#     # Step 3: Find the player(s) with the most highscores
#     most_highscores = max(highscore_counts.values(), default=0)
#     top_players = [
#         (pid, name) for pid, name in player_list
#         if highscore_counts.get(pid, 0) == most_highscores
#     ]
#
#     return top_players, highscore_counts
#
# get_most_highscore_player(player_list[:14])
# player_list[:14]
#
# all_players = get_players()
# player_list = []
# for p in players:
#     player_id = get_player_id(p)
#     player_list.append((player_id, p))
#     print(f"Player: {p}, ID: {player_id}")
#
# while len(player_list):
#    get_most_highscore_player(player_list)
#
# get_most_highscore_player(player_list)
