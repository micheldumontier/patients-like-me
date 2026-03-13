"""
Compare four patient embedding approaches:
  1. RDF2Vec (graph walks with reverse + ontologies)
  2. Template (structured text → text-embedding-3-large)
  3. LLM Summary (LLM-enriched text → text-embedding-3-large)
  4. GNN (GraphSAGE with Barlow Twins on patient-code-ontology graph)

Usage:
    python compare_all_embeddings.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUTPUT_DIR = os.path.dirname(__file__) or "."


def load_embeddings():
    """Load and align all four embedding sets by patient_id."""
    rdf_df = pd.read_csv(os.path.join(PROJECT_ROOT, "analysis/rdf2vec/patient_embeddings.csv"))
    tmpl_df = pd.read_csv(os.path.join(OUTPUT_DIR, "template_embeddings.csv"))
    summ_df = pd.read_csv(os.path.join(OUTPUT_DIR, "summary_embeddings.csv"))
    gnn_df = pd.read_csv(os.path.join(PROJECT_ROOT, "analysis/gnn-embeddings/gnn_embeddings.csv"))

    # Sort all by patient_id for alignment
    rdf_df = rdf_df.sort_values("patient_id").reset_index(drop=True)
    tmpl_df = tmpl_df.sort_values("patient_id").reset_index(drop=True)
    summ_df = summ_df.sort_values("patient_id").reset_index(drop=True)
    gnn_df = gnn_df.sort_values("patient_id").reset_index(drop=True)

    assert list(rdf_df.patient_id) == list(tmpl_df.patient_id) == list(summ_df.patient_id) \
        == list(gnn_df.patient_id), "Patient IDs do not match across embedding files"

    patient_ids = rdf_df["patient_id"].values
    return {
        "RDF2Vec": rdf_df.drop(columns=["patient_id"]).values,
        "Template": tmpl_df.drop(columns=["patient_id"]).values,
        "LLM Summary": summ_df.drop(columns=["patient_id"]).values,
        "GNN": gnn_df.drop(columns=["patient_id"]).values,
    }, patient_ids


def compute_metrics(sim_matrices, patient_ids):
    """Compute all pairwise rank correlations and nearest-neighbor overlap."""
    names = list(sim_matrices.keys())
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    triu_idx = np.triu_indices(len(patient_ids), k=1)

    print("\nSpearman rank correlation of pairwise patient similarities:")
    rho_values = {}
    for a, b in pairs:
        vec_a = sim_matrices[a][triu_idx]
        vec_b = sim_matrices[b][triu_idx]
        rho, pval = spearmanr(vec_a, vec_b)
        rho_values[(a, b)] = rho
        print(f"  {a:12s} vs {b:12s}: rho={rho:.4f}  (p={pval:.2e})")

    print("\nNearest-neighbor agreement (top-5):")
    for a, b in pairs:
        sim_a = sim_matrices[a].copy()
        sim_b = sim_matrices[b].copy()
        np.fill_diagonal(sim_a, -1)
        np.fill_diagonal(sim_b, -1)
        overlaps = []
        for j in range(len(patient_ids)):
            top_a = set(np.argsort(sim_a[j])[-5:])
            top_b = set(np.argsort(sim_b[j])[-5:])
            overlaps.append(len(top_a & top_b) / 5)
        print(f"  {a:12s} vs {b:12s}: {np.mean(overlaps):.1%} avg overlap")

    return rho_values


def plot_comparison(embeddings, sim_matrices, rho_values, patient_ids):
    """Generate the four-way comparison figure (4x4 grid)."""
    names = list(embeddings.keys())
    n = len(names)
    triu_idx = np.triu_indices(len(patient_ids), k=1)
    colors = {"RDF2Vec": "steelblue", "Template": "coral", "LLM Summary": "seagreen", "GNN": "darkorange"}

    fig, axes = plt.subplots(3, n, figsize=(6 * n, 16))

    # Row 1: Similarity matrices
    for i, name in enumerate(names):
        ax = axes[0, i]
        im = ax.imshow(sim_matrices[name], cmap="RdBu_r", vmin=-0.2, vmax=1)
        ax.set_title(f"{name}\nPairwise Similarity", fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.8)

    # Row 2: t-SNE projections
    perp = min(30, len(patient_ids) - 1)
    for i, name in enumerate(names):
        ax = axes[1, i]
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(embeddings[name])
        ax.scatter(tsne[:, 0], tsne[:, 1], s=25, alpha=0.7, c=colors[name],
                   edgecolors="white", linewidth=0.3)
        for j, pid in enumerate(patient_ids):
            ax.annotate(str(pid), (tsne[j, 0], tsne[j, 1]), fontsize=4, alpha=0.4)
        ax.set_title(f"{name} (t-SNE)", fontsize=11)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

    # Row 3: Correlation heatmap + key scatter plots
    # Panel 1: Spearman rho heatmap
    ax = axes[2, 0]
    rho_matrix = np.ones((n, n))
    for (a, b), rho in rho_values.items():
        i, j = names.index(a), names.index(b)
        rho_matrix[i, j] = rho
        rho_matrix[j, i] = rho
    im = ax.imshow(rho_matrix, cmap="RdYlGn", vmin=-0.1, vmax=0.5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, fontsize=8, rotation=30, ha="right")
    ax.set_yticklabels(names, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{rho_matrix[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if rho_matrix[i, j] > 0.3 else "black")
    ax.set_title("Spearman Correlation\n(Similarity Rank)", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Panels 2-4: Selected scatter plots (most interesting pairs)
    scatter_pairs = [("Template", "LLM Summary"), ("Template", "GNN"), ("RDF2Vec", "GNN")]
    for idx, (a, b) in enumerate(scatter_pairs):
        ax = axes[2, idx + 1]
        vec_a = sim_matrices[a][triu_idx]
        vec_b = sim_matrices[b][triu_idx]
        ax.scatter(vec_a, vec_b, alpha=0.1, s=5, c="slategray")
        lo = min(vec_a.min(), vec_b.min())
        hi = max(vec_a.max(), vec_b.max())
        ax.plot([lo, hi], [lo, hi], "r--", alpha=0.4)
        rho = rho_values.get((a, b), rho_values.get((b, a), 0))
        ax.set_xlabel(f"{a} Similarity", fontsize=9)
        ax.set_ylabel(f"{b} Similarity", fontsize=9)
        ax.set_title(f"{a} vs {b}\n(rho={rho:.3f})", fontsize=10)

    fig.suptitle("Four-Way Comparison of Patient Embedding Approaches", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(OUTPUT_DIR, "four_way_comparison.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")


def main():
    print("Loading embeddings...")
    embeddings, patient_ids = load_embeddings()
    for name, emb in embeddings.items():
        print(f"  {name:12s}: {emb.shape}, norm={np.linalg.norm(emb, axis=1).mean():.4f}")

    print("\nComputing similarity matrices...")
    sim_matrices = {name: cosine_similarity(emb) for name, emb in embeddings.items()}

    rho_values = compute_metrics(sim_matrices, patient_ids)
    plot_comparison(embeddings, sim_matrices, rho_values, patient_ids)
    print("\nDone!")


if __name__ == "__main__":
    main()
