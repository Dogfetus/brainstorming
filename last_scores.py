import sqlite3
from datetime import datetime, timedelta

DATE = "2024-10-17"

def player_exists(gamer_name):
    con = sqlite3.connect("new_data.db")
    cur = con.cursor()

    dogfetus = cur.execute("""
        SELECT _id
        FROM gamers
        WHERE username = ? COLLATE NOCASE;
    """, (gamer_name,)).fetchone()

    con.close()

    return dogfetus


def last_scr_for_dif_and_dis(player, x, dif, dis, date, cleared=1, ignore_cleared=False):
    conn = sqlite3.connect("new_data.db")
    cursor = conn.cursor()

    one_year_before = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")

    query = """
        SELECT scores._id, scores.score, scores.created_at, charts.difficulty, charts.difficulty_display, charts._id, charts.I_DIFF_1, charts.I_DIFF_2, charts.I_DIFF_3, charts.I_DIFF_4, charts.I_DIFF_5, charts.I_DIFF_6
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




def best_x_scr_for_chart(player, chart_id, x, date=DATE, cleared=1, ignore_cleared=False):
    """
    Get the best x scores for a player on a specific chart
    Returns:
        list: A list of tuples containing the score ID, score, created_at timestamp, difficulty, difficulty_display, chart ID.
        list goes like this:
        0: score id
        1: score
        2: created at
        3: difficulty
        4: difficulty display
        5: chart id
    """

    conn = sqlite3.connect("new_data.db")
    cursor = conn.cursor()

    query = """
        SELECT scores._id, scores.score, scores.created_at, charts.difficulty, charts.difficulty_display, charts._id
        FROM scores
        JOIN charts ON scores.chart_id = charts._id
        JOIN gamers ON scores.gamer_id = gamers._id
        WHERE gamers.username = ? COLLATE NOCASE
        AND scores.created_at < ?
        AND charts._id = ?
    """

    params = [player, date, chart_id]

    if not ignore_cleared:
        query += " AND scores.cleared = ?"
        params.append(cleared)

    query += " ORDER BY scores.score DESC LIMIT ?"
    params.append(x)

    cursor.execute(query, params)
    results = cursor.fetchall()

    conn.close()

    return results


# def best_x_scr(player, x, date=DATE, cleared=1, ignore_cleared=False):
#     """
#     Get the best x scores for a player
#
#     Returns:
#         list: A list of tuples containing the score ID, score, created_at timestamp, difficulty, difficulty_display, chart ID, and I_DIFF values.
#         list goes like this:
#         0: score id
#         1: score
#         2: created at
#         3: difficulty
#         4: difficulty display
#         5: chart id
#         6: I_DIFF_1
#         7: I_DIFF_2
#         8: I_DIFF_3
#         9: I_DIFF_4
#         10: I_DIFF_5
#         11: I_DIFF_6
#     """
#
#     conn = sqlite3.connect("new_data.db")
#     cursor = conn.cursor()
#
#     query = """
#         SELECT scores._id, scores.score, scores.created_at, charts.difficulty, charts.difficulty_display, charts._id, charts.I_DIFF_1, charts.I_DIFF_2, charts.I_DIFF_3, charts.I_DIFF_4, charts.I_DIFF_5, charts.I_DIFF_6
#         FROM scores
#         JOIN charts ON scores.chart_id = charts._id
#         JOIN gamers ON scores.gamer_id = gamers._id
#         WHERE gamers.username = ? COLLATE NOCASE
#         AND scores.created_at < ?
#     """
#
#     params = [player, date]
#
#     if not ignore_cleared:
#         query += " AND scores.cleared = ?"
#         params.append(cleared)
#
#     query += " ORDER BY scores.score DESC LIMIT ?"
#     params.append(x)
#
#     cursor.execute(query, params)
#     results = cursor.fetchall()
#
#     conn.close()
#
#     return results
#
#
#
#
#
#
# def get_played_charts(player, date="2026-10-10", cleared=1, force_cleared=False):
#     """
#     Get the charts played by a player
#     Returns:
#         list of charts
#     """
#
#     conn = sqlite3.connect("new_data.db")
#     cursor = conn.cursor()
#
#     query = """
#         SELECT charts.* 
#         FROM charts
#         JOIN scores ON scores.chart_id = charts._id
#         JOIN gamers ON scores.gamer_id = gamers._id
#         WHERE gamers.username = ? COLLATE NOCASE
#         AND scores.created_at < ?
#     """
#
#     params = [player, date]
#
#     if force_cleared:
#         query += " AND scores.cleared = ?"
#         params.append(cleared)
#
#     cursor.execute(query, params)
#     results = cursor.fetchall()
#
#     conn.close()
#
#     return results
#
#



def best_x_scr(player, x, date=DATE, cleared=1, ignore_cleared=False):
    """
    Get the best x scores for a player
    Returns:
        list: A list of tuples containing the score ID, score, created_at timestamp, difficulty, difficulty_display, chart ID, and I_DIFF values.
        list goes like this:
        0: score id
        1: score
        2: created at
        3: difficulty
        4: difficulty display
        5: chart id
        6: I_DIFF_1
        7: I_DIFF_2
        8: I_DIFF_3
        9: I_DIFF_4
        10: I_DIFF_5
        11: I_DIFF_6
    """
    conn = sqlite3.connect("new_data.db")
    cursor = conn.cursor()
    query = """
        SELECT scores._id, scores.score, scores.created_at, charts.difficulty, charts.difficulty_display, charts._id, charts.I_DIFF_1, charts.I_DIFF_2, charts.I_DIFF_3, charts.I_DIFF_4, charts.I_DIFF_5, charts.I_DIFF_6
        FROM scores
        JOIN charts ON scores.chart_id = charts._id
        JOIN gamers ON scores.gamer_id = gamers._id
        WHERE gamers.username = ? COLLATE NOCASE
        AND scores.created_at < ?
        AND charts.difficulty_display NOT IN ('dual', 'dual+', 'full', 'full+', 'edit')
    """
    params = [player, date]
    if not ignore_cleared:
        query += " AND scores.cleared = ?"
        params.append(cleared)
    query += " ORDER BY scores.score DESC LIMIT ?"
    params.append(x)
    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()
    return results

def get_played_charts(player, date="2026-10-10", cleared=1, force_cleared=False):
    """
    Get the charts played by a player
    Returns:
        list of charts
    """
    conn = sqlite3.connect("new_data.db")
    cursor = conn.cursor()
    query = """
        SELECT charts.* 
        FROM charts
        JOIN scores ON scores.chart_id = charts._id
        JOIN gamers ON scores.gamer_id = gamers._id
        WHERE gamers.username = ? COLLATE NOCASE
        AND scores.created_at < ?
        AND charts.difficulty_display NOT IN ('dual', 'dual+', 'full', 'full+', 'edit')
    """
    params = [player, date]
    if force_cleared:
        query += " AND scores.cleared = ?"
        params.append(cleared)
    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()
    return results



































































# last_scr_for_dif_and_dis("dogfetus", 10, 19, "wild", "2024-10-17")
#
# # example usage:
# player_name = "dogfetus"
# x = 10
# difficulty = 19 
# difficulty_display = "wild"
# date = "2024-10-17"
# # date = "2025-03-03"
#
# last_scores = last_scr_for_dif_and_dis(player_name, x, difficulty, difficulty_display, date)
#
# for id, score, created_at, dif, dis in last_scores:
#     print(f"id: {id}, score: {score}, date: {created_at}, difficulty: {dif}, display: {dis}, value: {value_of_score(score, dif, dis)}")
#











































#
# import sqlite3
# from datetime import datetime
#
# # Connect to the old and new databases
# old_db = "new_data.db"
# new_db = "new_data_migrated.db"
#
# conn_old = sqlite3.connect(old_db)
# cursor_old = conn_old.cursor()
#
# conn_new = sqlite3.connect(new_db)
# cursor_new = conn_new.cursor()
#
# # Create new tables with DATETIME fields
# cursor_new.executescript("""
# CREATE TABLE songs (
#     _id INTEGER PRIMARY KEY,
#     allow_edits INTEGER,
#     artist TEXT,
#     bpm TEXT,
#     cover TEXT,
#     cover_path TEXT,
#     cover_thumb TEXT,
#     created_at DATETIME,
#     extra TEXT,
#     first_beat INTEGER,
#     first_ms INTEGER,
#     game_song_id INTEGER,
#     genre TEXT,
#     id INTEGER,
#     is_enabled INTEGER,
#     label TEXT,
#     last_beat INTEGER,
#     last_ms INTEGER,
#     release_date TEXT,
#     subtitle TEXT,
#     timing_bpms TEXT,
#     timing_offset_ms INTEGER,
#     timing_stops TEXT,
#     title TEXT,
#     updated_at DATETIME,
#     website TEXT
# );
#
# CREATE TABLE charts (
#     _id INTEGER PRIMARY KEY,
#     created_at DATETIME,
#     difficulty INTEGER,
#     difficulty_display TEXT,
#     difficulty_id INTEGER,
#     difficulty_name TEXT,
#     game_difficulty_id INTEGER,
#     graph TEXT,
#     id INTEGER,
#     is_enabled INTEGER,
#     meter INTEGER,
#     pass_count INTEGER,
#     play_count INTEGER,
#     song_id INTEGER,
#     steps_author TEXT,
#     steps_index INTEGER,
#     updated_at DATETIME,
#     FOREIGN KEY (song_id) REFERENCES songs(_id) ON DELETE CASCADE
# );
#
# CREATE TABLE gamers (
#     _id INTEGER PRIMARY KEY,
#     country TEXT,
#     description TEXT,
#     hex_color TEXT,
#     id INTEGER,
#     picture_path TEXT,
#     private BOOLEAN,
#     published_edits INTEGER,
#     rival INTEGER,
#     username TEXT
# );
#
# CREATE TABLE scores (
#     _id INTEGER PRIMARY KEY,
#     calories INTEGER,
#     chart_id INTEGER,
#     cleared BOOLEAN,
#     created_at DATETIME,
#     early INTEGER,
#     flags INTEGER,
#     full_combo BOOLEAN,
#     gamer_id INTEGER,
#     global_flags INTEGER,
#     grade INTEGER,
#     green INTEGER,
#     late INTEGER,
#     max_combo INTEGER,
#     misses INTEGER,
#     music_speed INTEGER,
#     perfect1 INTEGER,
#     perfect2 INTEGER,
#     red INTEGER,
#     score INTEGER,
#     side TEXT,
#     song_chart_id INTEGER,
#     steps INTEGER,
#     updated_at DATETIME,
#     uuid TEXT,
#     yellow INTEGER,
#     FOREIGN KEY (gamer_id) REFERENCES gamers(_id) ON DELETE CASCADE,
#     FOREIGN KEY (song_chart_id) REFERENCES charts(_id) ON DELETE CASCADE
# );
# """)
#
# # Function to convert timestamps safely
# def convert_timestamp(timestamp):
#     if timestamp:
#         try:
#             dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
#             return dt.strftime("%Y-%m-%d %H:%M:%S")  # Convert to SQLite-compatible format
#         except ValueError:
#             return None
#     return None
#
# # List of tables that contain timestamps
# tables_with_timestamps = ["songs", "charts", "scores", "gamers"]
#
# for table in tables_with_timestamps:
#     cursor_old.execute(f"SELECT * FROM {table}")
#     columns = [desc[0] for desc in cursor_old.description]
#
#     # Convert timestamps
#     rows = cursor_old.fetchall()
#     new_rows = []
#
#     for row in rows:
#         row_dict = dict(zip(columns, row))
#
#         # Convert timestamps if they exist
#         if "created_at" in row_dict and row_dict["created_at"]:
#             row_dict["created_at"] = convert_timestamp(row_dict["created_at"])
#         if "updated_at" in row_dict and row_dict["updated_at"]:
#             row_dict["updated_at"] = convert_timestamp(row_dict["updated_at"])
#
#         new_rows.append(tuple(row_dict[col] for col in columns))
#
#     if new_rows:
#         placeholders = ", ".join("?" * len(columns))
#         insert_query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
#         cursor_new.executemany(insert_query, new_rows)
#
# # Commit and close connections
# conn_new.commit()
# conn_old.close()
# conn_new.close()
#
# print("Database migration completed successfully.")
#
