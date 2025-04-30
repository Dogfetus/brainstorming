import sqlite3

from requests import get


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


def percentage_difference(your_score, avg_score):
    return ((your_score - avg_score) / avg_score) * 100





def value_of_score(score, difficulty, display):
    average_score = gasfdad(difficulty, display)
    if average_score is None or average_score == 0:
        return 0  # or float('-inf') or some fallback

    value = score * percentage_difference(score, average_score)
    perfect_value = 100_000 * percentage_difference(100_000, average_score)

    if perfect_value == 0:
        return 0

    return value / perfect_value


# value_of_score(80999, 27, "wild")






