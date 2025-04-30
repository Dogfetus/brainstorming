import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set seaborn style for better plots
sns.set(style="whitegrid")

# Connect to the database
conn = sqlite3.connect("new_data.db")

print("Reading filtered tables...")
songs_df = pd.read_sql_query("SELECT * FROM songs", conn)

charts_df = pd.read_sql_query("""
SELECT * FROM charts
WHERE difficulty_display NOT IN ('full', 'full+', 'dual', 'dual+', 'edit')
""", conn)

gamers_df = pd.read_sql_query("SELECT * FROM gamers", conn)

# -----------------------
# 1. Score Progression Over Time (filtered)
# -----------------------
df_time = pd.read_sql_query("""
SELECT s.created_at, s.score
FROM scores s
JOIN charts c ON s.song_chart_id = c._id
WHERE c.difficulty_display NOT IN ('full', 'full+', 'dual', 'dual+', 'edit')
""", conn)

df_time['created_at'] = pd.to_datetime(df_time['created_at'])
df_time = df_time.sort_values('created_at')

plt.figure()
plt.plot(df_time['created_at'], df_time['score'], alpha=0.3)
plt.title("Score Progression Over Time")
plt.xlabel("Date")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig("eda_score_over_time_filtered.png")

# -----------------------
# 2. Top Gamers by Average Score (filtered)
# -----------------------
query_top_players = """
SELECT g.username, AVG(s.score) as avg_score
FROM scores s
JOIN gamers g ON s.gamer_id = g._id
JOIN charts c ON s.song_chart_id = c._id
WHERE c.difficulty_display NOT IN ('full', 'full+', 'dual', 'dual+', 'edit')
GROUP BY g.username
ORDER BY avg_score DESC
LIMIT 20;
"""
df_top = pd.read_sql_query(query_top_players, conn)

plt.figure()
plt.barh(df_top['username'], df_top['avg_score'])
plt.xlabel("Average Score")
plt.title("Top 20 Players by Average Score")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("eda_top_players_avg_score_filtered.png")

# -----------------------
# 3. Radar Chart: Accuracy Breakdown (all data)
# -----------------------
query_accuracy = """
SELECT AVG(perfect1) as perfect1, AVG(perfect2) as perfect2,
       AVG(green) as green, AVG(red) as red,
       AVG(yellow) as yellow, AVG(misses) as misses
FROM scores s
JOIN charts c ON s.song_chart_id = c._id
WHERE c.difficulty_display NOT IN ('full', 'full+', 'dual', 'dual+', 'edit')
"""
df_acc = pd.read_sql_query(query_accuracy, conn).iloc[0]

labels = df_acc.index.tolist()
values = df_acc.values.tolist()
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
values += values[:1]
angles += angles[:1]

fig, ax = plt.subplots(subplot_kw={'polar': True})
ax.plot(angles, values, marker='o')
ax.fill(angles, values, alpha=0.25)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
plt.title("Average Accuracy Breakdown")
plt.tight_layout()
plt.savefig("eda_accuracy_radar_filtered.png")

# -----------------------
# 4. Heatmap: Player vs Song Play Count (filtered)
# -----------------------
query_heatmap = """
SELECT g.username, sng.title, COUNT(*) as play_count
FROM scores sc
JOIN charts c ON sc.song_chart_id = c._id
JOIN songs sng ON c.song_id = sng._id
JOIN gamers g ON sc.gamer_id = g._id
WHERE c.difficulty_display NOT IN ('full', 'full+', 'dual', 'dual+', 'edit')
GROUP BY g.username, sng.title
"""
df_heat = pd.read_sql_query(query_heatmap, conn)
pivot = df_heat.pivot(index="username", columns="title", values="play_count").fillna(0)

plt.figure(figsize=(12, 8))
sns.heatmap(pivot, cmap="viridis")
plt.title("Heatmap of Player vs. Song Play Count")
plt.xlabel("Song Title")
plt.ylabel("Player Username")
plt.tight_layout()
plt.savefig("eda_heatmap_player_song_filtered.png")

# -----------------------
# 5. Boxplot: Score by Difficulty Level (filtered)
# -----------------------
query_boxplot = """
SELECT c.difficulty_name, s.score
FROM scores s
JOIN charts c ON s.song_chart_id = c._id
WHERE c.difficulty_display NOT IN ('full', 'full+', 'dual', 'dual+', 'edit')
"""
df_box = pd.read_sql_query(query_boxplot, conn)

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_box, x="difficulty_name", y="score")
plt.title("Score Distribution by Difficulty Level")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_score_by_difficulty_filtered.png")

# Close connection
conn.close()

print("✅ All filtered EDA plots saved!")

