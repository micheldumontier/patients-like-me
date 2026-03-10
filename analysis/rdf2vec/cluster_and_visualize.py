"""
Cluster and visualize RDF2Vec patient embeddings from the MIMIC-IV demo graph.

Performs K-Means clustering with silhouette-based k selection,
then visualizes with t-SNE and UMAP 2D projections.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = os.path.dirname(__file__) or "."

# Load embeddings
embeddings = np.load(os.path.join(OUTPUT_DIR, "patient_embeddings.npy"))
df = pd.read_csv(os.path.join(OUTPUT_DIR, "patient_embeddings.csv"))
patient_ids = df["patient_id"].values
print(f"Loaded embeddings: {embeddings.shape}")

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(embeddings)

# --- Find optimal k via silhouette score ---
k_range = range(2, 11)
silhouette_scores = []
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"  k={k}: silhouette={score:.4f}")

best_k = list(k_range)[np.argmax(silhouette_scores)]
print(f"\nBest k={best_k} (silhouette={max(silhouette_scores):.4f})")

# Final clustering with best k
km_final = KMeans(n_clusters=best_k, n_init=10, random_state=42)
cluster_labels = km_final.fit_predict(X)

# Save cluster assignments
cluster_df = pd.DataFrame({"patient_id": patient_ids, "cluster": cluster_labels})
cluster_df.to_csv(os.path.join(OUTPUT_DIR, "patient_clusters.csv"), index=False)
print(f"Saved cluster assignments to patient_clusters.csv")
print(f"Cluster sizes: {dict(zip(*np.unique(cluster_labels, return_counts=True)))}")

# --- Silhouette score plot ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(list(k_range), silhouette_scores, "o-", linewidth=2)
ax.axvline(best_k, color="red", linestyle="--", alpha=0.7, label=f"best k={best_k}")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Silhouette Score")
ax.set_title("K-Means Silhouette Analysis")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "silhouette_scores.png"), dpi=150)
print("Saved silhouette_scores.png")

# --- t-SNE projection ---
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
X_tsne = tsne.fit_transform(X)

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap="tab10",
                     s=60, alpha=0.8, edgecolors="white", linewidth=0.5)
for i, pid in enumerate(patient_ids):
    ax.annotate(str(pid), (X_tsne[i, 0], X_tsne[i, 1]),
                fontsize=6, alpha=0.6, ha="center", va="bottom")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_title(f"Patient Embeddings (t-SNE) — {best_k} clusters")
cbar = fig.colorbar(scatter, ax=ax, label="Cluster")
cbar.set_ticks(range(best_k))
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "tsne_clusters.png"), dpi=150)
print("Saved tsne_clusters.png")

# --- UMAP projection (if available) ---
try:
    from umap import UMAP
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    X_umap = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=cluster_labels, cmap="tab10",
                         s=60, alpha=0.8, edgecolors="white", linewidth=0.5)
    for i, pid in enumerate(patient_ids):
        ax.annotate(str(pid), (X_umap[i, 0], X_umap[i, 1]),
                    fontsize=6, alpha=0.6, ha="center", va="bottom")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Patient Embeddings (UMAP) — {best_k} clusters")
    cbar = fig.colorbar(scatter, ax=ax, label="Cluster")
    cbar.set_ticks(range(best_k))
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "umap_clusters.png"), dpi=150)
    print("Saved umap_clusters.png")
except ImportError:
    print("umap-learn not installed, skipping UMAP visualization")

print("\nDone!")
