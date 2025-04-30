import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Connect to the SQLite database
conn = sqlite3.connect('new_data.db')

# Load tables
print("Reading data from database...")
songs_df = pd.read_sql_query("SELECT * FROM songs", conn)
charts_df = pd.read_sql_query("""
    SELECT * FROM charts 
    WHERE difficulty_display NOT IN ('full', 'full+', 'dual', 'dual+', 'edit')
""", conn)
scores_df = pd.read_sql_query("""
    SELECT s.*
    FROM scores s
    JOIN charts c ON s.song_chart_id = c._id
    WHERE c.difficulty_display NOT IN ('full', 'full+', 'dual', 'dual+', 'edit')
""", conn)
print("Data read successfully.")

# Compute score stats per chart
chart_stats = scores_df.groupby('song_chart_id').agg({
    'score': 'mean',
    'cleared': 'mean'
}).reset_index()

# Add difficulty info from charts table
chart_stats = chart_stats.merge(
    charts_df[['_id', 'difficulty']],
    left_on='song_chart_id',
    right_on='_id',
    how='inner'
)

# Plot: Difficulty vs. Average Score
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=chart_stats,
    x='difficulty',
    y='score',
    size='cleared',               # ✅ Use cleared rate for bubble size
    sizes=(20, 200),
    alpha=0.7
)
plt.xlabel('Chart Difficulty')
plt.ylabel('Average Score')
plt.title('Difficulty vs. Performance (Bubble Size = Clear Rate)')
plt.tight_layout()
plt.show()

# Close the connection
conn.close()


