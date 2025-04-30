import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Connect to the database
conn = sqlite3.connect("new_data.db")

# Join scores and charts on song_chart_id = _id
query = """
SELECT s.calories, s.early, s.cleared, s.green, s.late, s.misses, s.perfect1, s.perfect2, s.red, s.score, s.steps, 
       c.difficulty, c.difficulty_display
FROM scores s
JOIN charts c ON s.song_chart_id = c._id
WHERE c.difficulty_display NOT IN ('full', 'full+', 'dual', 'dual+', 'edit')
"""
df = pd.read_sql_query(query, conn)
conn.close()

# Drop non-numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Compute correlation matrix
corr = numeric_df.corr()

# Plot heatmap with better readability
plt.figure(figsize=(14, 12))
sns.heatmap(
    corr, 
    annot=True, 
    fmt=".1f", 
    cmap="coolwarm", 
    center=0,
    annot_kws={"size": 10},
    cbar_kws={"shrink": 0.9, "label": "Correlation"}
)
plt.xticks(rotation=45, ha='right', fontsize=14)
plt.yticks(rotation=0, fontsize=14)
plt.title("Correlation Heatmap: Combined Scores and Charts Data", fontsize=18, pad=20)
plt.tight_layout()
plt.show()

