import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Connect to the SQLite database
conn = sqlite3.connect("new_data.db")

# Optional: For prettier plots
sns.set(style="whitegrid")

# Load filtered tables
print("Reading data from database...")
songs_df = pd.read_sql_query("SELECT * FROM songs", conn)

charts_df = pd.read_sql_query("""
SELECT * FROM charts
WHERE difficulty_display NOT IN ('full', 'full+', 'dual', 'dual+', 'edit')
""", conn)

gamers_df = pd.read_sql_query("SELECT * FROM gamers", conn)






# Score Distribution by Difficulty
query = """
SELECT s.score, c.difficulty
FROM scores s
JOIN charts c ON s.song_chart_id = c._id
WHERE s.score IS NOT NULL AND c.difficulty IS NOT NULL
AND c.difficulty_display NOT IN ('full', 'full+', 'dual', 'dual+', 'edit')
"""
df = pd.read_sql_query(query, conn)

plt.figure(figsize=(10, 6))
sns.boxplot(x='difficulty', y='score', data=df)
plt.title("Score Distribution by Difficulty")
plt.xlabel("Chart Difficulty")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()









# Heatmap of Gamers vs. Charts
query = """
SELECT DISTINCT s.gamer_id, s.song_chart_id
FROM scores s
JOIN charts c ON s.song_chart_id = c._id
"""
df = pd.read_sql_query(query, conn)

heatmap_df = df.pivot_table(index='gamer_id', columns='song_chart_id',
                            aggfunc=lambda x: 1, fill_value=0)

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_df, cmap='Blues', cbar=False)
plt.title("Gamers vs. Charts Heatmap")
plt.xlabel("Chart ID")
plt.ylabel("Gamer ID")
plt.tight_layout()
plt.show()












# Distribution of Unique Charts Played by Gamers (Zoomed In)
counts = df.groupby('gamer_id')['song_chart_id'].nunique()

plt.figure(figsize=(10, 6))
sns.histplot(counts, bins=30, kde=False)
plt.title("Charts Played Per Player")
plt.xlabel("Number of Unique Charts Played")
plt.ylabel("Number of Players")
plt.xlim(0, 1200)  # Zoom to first 1500
plt.tight_layout()
plt.show()
























# Score Variance per Player
query = """
SELECT s.gamer_id, s.score
FROM scores s
JOIN charts c ON s.song_chart_id = c._id
WHERE s.score IS NOT NULL
AND c.difficulty_display NOT IN ('full', 'full+', 'dual', 'dual+', 'edit')
"""
df = pd.read_sql_query(query, conn)

variance_df = df.groupby('gamer_id')['score'].var().dropna()

plt.figure(figsize=(8, 5))
sns.histplot(variance_df, bins=30, kde=True)
plt.title("Score Variance per Player")
plt.xlabel("Score Variance")
plt.ylabel("Number of Players")
plt.tight_layout()
plt.show()
























# Max Combo vs. Score for Top Player
top_player = df['gamer_id'].value_counts().idxmax()

query = f"""
SELECT s.score, s.max_combo
FROM scores s
JOIN charts c ON s.song_chart_id = c._id
WHERE s.gamer_id = {top_player}
AND s.score IS NOT NULL AND s.max_combo IS NOT NULL
AND c.difficulty_display NOT IN ('full', 'full+', 'dual', 'dual+', 'edit')
"""
combo_df = pd.read_sql_query(query, conn)

plt.figure(figsize=(8, 5))
sns.scatterplot(x='max_combo', y='score', data=combo_df)
plt.title(f"Max Combo vs. Score for Player {top_player}")
plt.xlabel("Max Combo")
plt.ylabel("Score")
plt.tight_layout()
plt.show()

























#33#########################################################################################################################

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# Compute charts played per player
chart_counts = df.groupby('gamer_id')['song_chart_id'].nunique()

# Run PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(score_matrix)

# Create PCA DataFrame
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
pca_df['gamer_id'] = score_matrix.index
pca_df['charts_played'] = pca_df['gamer_id'].map(chart_counts)

# Plot PCA with color by charts played
plt.figure(figsize=(10, 7))
scatter = sns.scatterplot(
    x='PC1', y='PC2',
    hue='charts_played',
    palette=sns.color_palette("viridis", as_cmap=True).reversed(),
    data=pca_df,
    s=60, alpha=0.8,
    legend=False
)

# Colorbar
norm = mpl.colors.Normalize(vmin=pca_df['charts_played'].min(), vmax=pca_df['charts_played'].max())
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
plt.colorbar(sm, label='Number of Charts Played')

# Labels and title
plt.title("PCA: Player Distribution Colored by Charts Played\n(Lighter = More Active Players)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.show()





















tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_result = tsne.fit_transform(score_matrix)

# Create DataFrame
tsne_df = pd.DataFrame(tsne_result, columns=['Dim1', 'Dim2'])
tsne_df['gamer_id'] = score_matrix.index

# Compute average score per player
avg_scores = df.groupby('gamer_id')['score'].mean()

# Merge average scores into the t-SNE DataFrame
tsne_df['avg_score'] = tsne_df['gamer_id'].map(avg_scores)



plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='Dim1', y='Dim2',
    hue='avg_score',
    palette='viridis',
    data=tsne_df,
    s=60, alpha=0.8,
    legend='brief'
)
plt.title("t-SNE: Player Similarity Colored by Average Score")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.colorbar(label='Average Score')
plt.tight_layout()
plt.show()










conn.close()

