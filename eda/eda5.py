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

# Load minimal score data: gamer_id, song_chart_id, score
query = """
SELECT s.gamer_id, s.song_chart_id, s.score
FROM scores s
JOIN charts c ON s.song_chart_id = c._id
WHERE c.difficulty_display NOT IN ('full', 'full+', 'dual', 'dual+', 'edit', 'basic', 'easy', 'easy+', 'wild')
AND s.score IS NOT NULL;
"""
df_scores = pd.read_sql_query(query, conn)

# Create player × chart score matrix
score_matrix = df_scores.pivot_table(
    index='gamer_id',
    columns='song_chart_id',
    values='score',
    fill_value=0
)

# Color values: total number of scores per player
total_scores = df_scores.groupby('gamer_id')['score'].count()
color_values = np.log1p(total_scores[score_matrix.index])

# Normalize
scaler = StandardScaler()
scaled_features = scaler.fit_transform(score_matrix)

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
plt.title("3D PCA: log(1 + Total Scores)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.colorbar(sc, pad=0.1).set_label("log(1 + Total Scores)")
plt.tight_layout()
plt.show()

# 🎯 PCA: 2D
pca_2d = PCA(n_components=2)
pca_2d_result = pca_2d.fit_transform(scaled_features)

plt.figure(figsize=(10, 7))
plt.scatter(
    pca_2d_result[:, 0], pca_2d_result[:, 1],
    c=color_values,
    cmap=inferno_green,
    s=50, alpha=0.8
)
plt.title("2D PCA: log(1 + Total Scores)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="log(1 + Total Scores)")
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
plt.title("3D t-SNE: log(1 + Total Scores)")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_zlabel("t-SNE 3")
plt.colorbar(sc, pad=0.1).set_label("log(1 + Total Scores)")
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
plt.title("2D t-SNE: log(1 + Total Scores)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar(label="log(1 + Total Scores)")
plt.tight_layout()
plt.show()

