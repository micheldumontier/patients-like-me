"""
Fused patient embeddings via learned projection.

Projects RDF2Vec (200d), Template (3072d), LLM Summary (3072d), and GNN (128d)
embeddings into a shared space using a multi-view contrastive learning approach:

1. Four separate projection heads map each view to a shared dimension
2. Contrastive loss: for each patient, all projected views should be close,
   while views from different patients should be distant
3. The fused embedding is the mean of the four projected views

Usage:
    python train_fused_embeddings.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUTPUT_DIR = os.path.dirname(__file__) or "."

# Hyperparameters
FUSED_DIM = 128
HIDDEN_DIM = 256
LR = 1e-3
EPOCHS = 500
TEMPERATURE = 0.07
SEED = 42


def load_embeddings():
    rdf_df = pd.read_csv(os.path.join(PROJECT_ROOT, "analysis/rdf2vec/patient_embeddings.csv"))
    tmpl_df = pd.read_csv(os.path.join(PROJECT_ROOT, "analysis/text-embeddings/template_embeddings.csv"))
    summ_df = pd.read_csv(os.path.join(PROJECT_ROOT, "analysis/text-embeddings/summary_embeddings.csv"))
    gnn_df = pd.read_csv(os.path.join(PROJECT_ROOT, "analysis/gnn-embeddings/gnn_embeddings.csv"))

    rdf_df = rdf_df.sort_values("patient_id").reset_index(drop=True)
    tmpl_df = tmpl_df.sort_values("patient_id").reset_index(drop=True)
    summ_df = summ_df.sort_values("patient_id").reset_index(drop=True)
    gnn_df = gnn_df.sort_values("patient_id").reset_index(drop=True)

    patient_ids = rdf_df["patient_id"].values
    return {
        "rdf2vec": rdf_df.drop(columns=["patient_id"]).values.astype(np.float32),
        "template": tmpl_df.drop(columns=["patient_id"]).values.astype(np.float32),
        "summary": summ_df.drop(columns=["patient_id"]).values.astype(np.float32),
        "gnn": gnn_df.drop(columns=["patient_id"]).values.astype(np.float32),
    }, patient_ids


class ProjectionHead(nn.Module):
    """MLP projection head: input_dim -> hidden -> fused_dim."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class MultiViewContrastiveModel(nn.Module):
    """Four projection heads mapping each view to a shared space."""
    def __init__(self, dims, hidden_dim, fused_dim):
        super().__init__()
        self.projections = nn.ModuleDict({
            name: ProjectionHead(dim, hidden_dim, fused_dim)
            for name, dim in dims.items()
        })
        self.view_names = list(dims.keys())

    def forward(self, data_dict):
        return {name: self.projections[name](data_dict[name])
                for name in self.view_names}

    def fuse(self, data_dict):
        """Produce fused embedding as mean of all projected views."""
        z_dict = self.forward(data_dict)
        fused = sum(z_dict.values()) / len(z_dict)
        return F.normalize(fused, dim=-1)


def multi_view_contrastive_loss(z_dict, temperature):
    """
    Multi-view NT-Xent loss over all pairs of views.

    For each pair of views, the positive pair is the same patient across views,
    negatives are all other patients.
    """
    names = list(z_dict.keys())
    losses = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            z_a = z_dict[names[i]]
            z_b = z_dict[names[j]]
            batch_size = z_a.shape[0]
            sim = torch.mm(z_a, z_b.t()) / temperature
            labels = torch.arange(batch_size, device=sim.device)
            loss_ab = F.cross_entropy(sim, labels)
            loss_ba = F.cross_entropy(sim.t(), labels)
            losses.append((loss_ab + loss_ba) / 2)

    return sum(losses) / len(losses)


def train(model, data, epochs, lr, temperature):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    tensors = {k: torch.tensor(v) for k, v in data.items()}

    losses = []
    for epoch in range(epochs):
        model.train()
        z_dict = model(tensors)
        loss = multi_view_contrastive_loss(z_dict, temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={loss.item():.4f}")

    return losses


def evaluate(fused_emb, individual_embs, patient_ids):
    """Compare fused embedding against each individual approach."""
    sim_fused = cosine_similarity(fused_emb)
    triu_idx = np.triu_indices(len(patient_ids), k=1)

    print("\nSpearman correlation with fused embedding:")
    for name, emb in individual_embs.items():
        sim_ind = cosine_similarity(emb)
        rho, _ = spearmanr(sim_fused[triu_idx], sim_ind[triu_idx])
        print(f"  Fused vs {name:12s}: rho={rho:.4f}")

    # Cross-method agreement: for each pair of individual methods,
    # measure top-5 NN overlap using fused vs each individual
    print("\nNearest-neighbor agreement (top-5) — fused vs individual:")
    sim_fused_nn = sim_fused.copy()
    np.fill_diagonal(sim_fused_nn, -1)
    for name, emb in individual_embs.items():
        sim_ind = cosine_similarity(emb)
        np.fill_diagonal(sim_ind, -1)
        overlaps = []
        for j in range(len(patient_ids)):
            top_fused = set(np.argsort(sim_fused_nn[j])[-5:])
            top_ind = set(np.argsort(sim_ind[j])[-5:])
            overlaps.append(len(top_fused & top_ind) / 5)
        print(f"  Fused vs {name:12s}: {np.mean(overlaps):.1%} avg overlap")

    # Agreement between individual methods via fused
    print("\nPairwise individual agreement (top-5) for reference:")
    names = list(individual_embs.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sim_a = cosine_similarity(individual_embs[names[i]])
            sim_b = cosine_similarity(individual_embs[names[j]])
            np.fill_diagonal(sim_a, -1)
            np.fill_diagonal(sim_b, -1)
            overlaps = []
            for k in range(len(patient_ids)):
                top_a = set(np.argsort(sim_a[k])[-5:])
                top_b = set(np.argsort(sim_b[k])[-5:])
                overlaps.append(len(top_a & top_b) / 5)
            print(f"  {names[i]:12s} vs {names[j]:12s}: {np.mean(overlaps):.1%}")


def plot_results(fused_emb, individual_embs, patient_ids, losses):
    """Visualize fused embeddings and training."""
    n_views = len(individual_embs)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Training loss
    ax = axes[0, 0]
    ax.plot(losses, color="steelblue", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Contrastive Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)

    # 2. Fused similarity matrix
    ax = axes[0, 1]
    sim = cosine_similarity(fused_emb)
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-0.2, vmax=1)
    ax.set_title("Fused Pairwise Similarity")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # 3. t-SNE of fused embeddings
    ax = axes[0, 2]
    perp = min(30, len(patient_ids) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(fused_emb)
    ax.scatter(tsne[:, 0], tsne[:, 1], s=30, alpha=0.7, c="purple", edgecolors="white", linewidth=0.3)
    for j, pid in enumerate(patient_ids):
        ax.annotate(str(pid), (tsne[j, 0], tsne[j, 1]), fontsize=4, alpha=0.5)
    ax.set_title("Fused Embeddings (t-SNE)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # 4. Scatter: fused similarity vs each individual
    ax = axes[1, 0]
    sim_fused = cosine_similarity(fused_emb)
    triu_idx = np.triu_indices(len(patient_ids), k=1)
    fused_pairs = sim_fused[triu_idx]
    view_colors = {"rdf2vec": "steelblue", "template": "coral", "summary": "seagreen", "gnn": "darkorange"}
    for name, emb in individual_embs.items():
        sim_ind = cosine_similarity(emb)
        ind_pairs = sim_ind[triu_idx]
        rho, _ = spearmanr(fused_pairs, ind_pairs)
        ax.scatter(fused_pairs, ind_pairs, alpha=0.08, s=5,
                   c=view_colors.get(name, "gray"), label=f"{name} (rho={rho:.3f})")
    ax.set_xlabel("Fused Similarity")
    ax.set_ylabel("Individual Similarity")
    ax.set_title("Fused vs Individual Similarities")
    ax.legend(fontsize=8)

    # 5. Spearman correlation bar chart (fused vs each)
    ax = axes[1, 1]
    rho_vals = {}
    for name, emb in individual_embs.items():
        sim_ind = cosine_similarity(emb)
        rho, _ = spearmanr(sim_fused[triu_idx], sim_ind[triu_idx])
        rho_vals[name] = rho
    bars = ax.bar(rho_vals.keys(), rho_vals.values(),
                  color=[view_colors.get(n, "gray") for n in rho_vals.keys()])
    ax.set_ylabel("Spearman rho")
    ax.set_title("Fused vs Individual (Spearman)")
    ax.set_ylim(0, 1)
    for bar, v in zip(bars, rho_vals.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=9)

    # 6. NN overlap bar chart (fused vs each)
    ax = axes[1, 2]
    sim_fused_nn = sim_fused.copy()
    np.fill_diagonal(sim_fused_nn, -1)
    nn_vals = {}
    for name, emb in individual_embs.items():
        sim_ind = cosine_similarity(emb)
        np.fill_diagonal(sim_ind, -1)
        overlaps = []
        for j in range(len(patient_ids)):
            top_fused = set(np.argsort(sim_fused_nn[j])[-5:])
            top_ind = set(np.argsort(sim_ind[j])[-5:])
            overlaps.append(len(top_fused & top_ind) / 5)
        nn_vals[name] = np.mean(overlaps)
    bars = ax.bar(nn_vals.keys(), nn_vals.values(),
                  color=[view_colors.get(n, "gray") for n in nn_vals.keys()])
    ax.set_ylabel("Avg Top-5 Overlap")
    ax.set_title("Fused vs Individual (NN Overlap)")
    ax.set_ylim(0, 1)
    for bar, v in zip(bars, nn_vals.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.1%}",
                ha="center", fontsize=9)

    fig.suptitle(f"Fused Patient Embeddings ({FUSED_DIM}d, {n_views}-View Contrastive)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(OUTPUT_DIR, "fused_embeddings.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading embeddings...")
    data, patient_ids = load_embeddings()
    dims = {k: v.shape[1] for k, v in data.items()}
    for k, v in data.items():
        print(f"  {k:12s}: {v.shape}")

    print(f"\nTraining {len(dims)}-view contrastive model (fused_dim={FUSED_DIM}, epochs={EPOCHS})...")
    model = MultiViewContrastiveModel(dims, HIDDEN_DIM, FUSED_DIM)
    losses = train(model, data, EPOCHS, LR, TEMPERATURE)

    print("\nGenerating fused embeddings...")
    model.eval()
    with torch.no_grad():
        tensors = {k: torch.tensor(v) for k, v in data.items()}
        fused = model.fuse(tensors).numpy()
    print(f"  Fused shape: {fused.shape}")

    # Save fused embeddings
    np.save(os.path.join(OUTPUT_DIR, "fused_embeddings.npy"), fused)
    cols = [f"dim_{i}" for i in range(FUSED_DIM)]
    df = pd.DataFrame(fused, columns=cols)
    df.insert(0, "patient_id", patient_ids)
    df.to_csv(os.path.join(OUTPUT_DIR, "fused_embeddings.csv"), index=False)
    print("  Saved fused_embeddings.npy/csv")

    # Save model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "fusion_model.pt"))
    print("  Saved fusion_model.pt")

    # Evaluate
    evaluate(fused, data, patient_ids)
    plot_results(fused, data, patient_ids, losses)

    print("\nDone!")


if __name__ == "__main__":
    main()
