import sqlite3
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

sns.set(style="whitegrid")

# Custom inferno-like colormap ending in green
inferno_green = LinearSegmentedColormap.from_list(
    'inferno_green',
    ['#000004', '#420a68', '#932567', '#dd513a', '#f9c932', '#32cd32'],  # dark → orange → green
    N=256
)

# Connect to the database
conn = sqlite3.connect("new_data.db")

# Load and filter scores
query = """
SELECT s.gamer_id, s.song_chart_id, s.score, s.misses, s.perfect1, s.perfect2,
       s.max_combo, c.difficulty_display
FROM scores s
JOIN charts c ON s.song_chart_id = c._id
WHERE c.difficulty_display NOT IN ('full', 'full+', 'dual', 'dual+', 'edit')
AND s.cleared = 1
AND s.score IS NOT NULL;
"""
df_scores = pd.read_sql_query(query, conn)

# Compute features per player (NO grade or full_combo)
player_features = df_scores.groupby('gamer_id').agg({
    'score': 'mean',
    'misses': 'mean',
    'perfect1': 'mean',
    'perfect2': 'mean',
    'max_combo': 'mean',
    'song_chart_id': 'nunique'
}).rename(columns={'song_chart_id': 'unique_charts'}).fillna(0)

# Normalize
scaler = StandardScaler()
scaled_features = scaler.fit_transform(player_features)

# Color values (log scale for visibility)
color_values = np.log1p(player_features['unique_charts'])

# ----------------------------------------
# 🎯 PCA: 3D
# ----------------------------------------
pca_3d = PCA(n_components=3)
pca_3d_result = pca_3d.fit_transform(scaled_features)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(
    pca_3d_result[:, 0], pca_3d_result[:, 1], pca_3d_result[:, 2],
    c=color_values,
    cmap=inferno_green,
    s=50, alpha=0.8
)
plt.title("3D PCA: Total Charts Played")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.colorbar(sc, pad=0.1).set_label("Total Charts Played")
plt.tight_layout()
plt.show()

# 🎯 PCA: 2r
pca_2d = PCA(n_components=2)
pca_2d_result = pca_2d.fit_transform(scaled_features)

plt.figure(figsize=(10, 7))
plt.scatter(
    pca_2d_result[:, 0], pca_2d_result[:, 1],
    c=color_values,
    cmap=inferno_green,
    s=50, alpha=0.8
)
plt.title("2D PCA: Total Charts Played")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Total Charts Played")
plt.tight_layout()
plt.show()

print("PCA explained variance ratio:", pca_3d.explained_variance_ratio_)
print("sum of explained variance:", np.sum(pca_3d.explained_variance_ratio_[:3]))

# ----------------------------------------
# 🎯 t-SNE: 3D
# ----------------------------------------
tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30)
tsne_3d_result = tsne_3d.fit_transform(scaled_features)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(
    tsne_3d_result[:, 0], tsne_3d_result[:, 1], tsne_3d_result[:, 2],
    c=color_values,
    cmap=inferno_green,
    s=50, alpha=0.8
)
plt.title("3D t-SNE: Total Charts Played")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_zlabel("t-SNE 3")
plt.colorbar(sc, pad=0.1).set_label(" Total Charts Played")
plt.tight_layout()
plt.show()

# 🎯 t-SNE: 2D
tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_2d_result = tsne_2d.fit_transform(scaled_features)

plt.figure(figsize=(10, 7))
plt.scatter(
    tsne_2d_result[:, 0], tsne_2d_result[:, 1],
    c=color_values,
    cmap=inferno_green,
    s=50, alpha=0.8
)
plt.title("2D t-SNE: log(1 + Unique Charts Played)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar(label="log(1 + Unique Charts Played)")
plt.tight_layout()
plt.show()

