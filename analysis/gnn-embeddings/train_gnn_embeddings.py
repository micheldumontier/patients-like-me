"""
GNN-based patient embeddings using PyTorch Geometric.

Builds a heterogeneous graph from MIMIC-IV RDF data:
  - Patient nodes connected to code nodes (diagnoses, labs, meds, procedures)
  - Code nodes connected via ontology hierarchy (rdfs:subClassOf)
  - Discretized lab value bins as additional nodes

Trains a GraphSAGE encoder with unsupervised contrastive loss (positive pairs
are connected nodes, negatives are random nodes). The learned patient node
embeddings capture both graph topology and clinical semantics.

Usage:
    python train_gnn_embeddings.py
"""
import json
import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdflib
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MIMIC_TTL = os.path.join(DATA_DIR, "mimic-iv-demo.ttl")
OUTPUT_DIR = os.path.dirname(__file__) or "."

# Ontology files for hierarchy edges
ONTOLOGY_FILES = [
    ("ICD9CM", os.path.join(DATA_DIR, "ICD9CM.ttl")),
    ("ICD10CM", os.path.join(DATA_DIR, "ICD10CM.ttl")),
    ("RXNORM", os.path.join(DATA_DIR, "RXNORM.ttl")),
    ("LNC", os.path.join(DATA_DIR, "LNC.ttl")),
    ("SNOMEDCT", os.path.join(DATA_DIR, "SNOMEDCT.ttl")),
]

# GNN hyperparameters
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
LR = 1e-3
EPOCHS = 500
NEG_SAMPLES = 10
SEED = 42

# Lab discretization
N_BINS = 4
MIN_VALUES_PER_CODE = 10

# MEDS namespace
MEDS_NS = "https://teamheka.github.io/meds-ontology#"
MEDS_DATA_NS = "https://teamheka.github.io/meds-data/"
RDFS_SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"


def extract_graph_from_mimic(mimic_path):
    """Extract patient-code edges and lab values from MIMIC TTL.

    Instead of loading the full graph into rdflib and then converting,
    we parse once and extract:
      - patient_uri -> set of code_uris (via event -> hasCode -> parentCode)
      - event lab values for discretization
    """
    print("Loading MIMIC-IV RDF...")
    t0 = time.time()
    g = rdflib.Graph()
    g.parse(os.path.abspath(mimic_path), format="turtle")
    print(f"  {len(g)} triples in {time.time() - t0:.1f}s")

    meds = rdflib.Namespace(MEDS_NS)

    # Map event -> patient
    event_to_patient = {}
    for event, _, patient in g.triples((None, meds.hasSubject, None)):
        event_to_patient[str(event)] = str(patient)

    # Map event -> code nodes -> parentCode IRIs
    event_to_codes = defaultdict(set)
    event_to_codestring = {}
    for event, _, code_node in g.triples((None, meds.hasCode, None)):
        event_str = str(event)
        # Get parentCode (IRI to ontology)
        for _, _, parent in g.triples((code_node, meds.parentCode, None)):
            if not isinstance(parent, rdflib.Literal):
                event_to_codes[event_str].add(str(parent))
        # Get codeString for lab discretization
        for _, _, cs in g.triples((code_node, meds.codeString, None)):
            event_to_codestring[event_str] = str(cs)

    # Extract numeric values for lab discretization
    lab_values = []  # (event_uri, code_string, value)
    for event, _, val in g.triples((None, meds.numericValue, None)):
        event_str = str(event)
        if event_str in event_to_codestring:
            try:
                lab_values.append((event_str, event_to_codestring[event_str], float(val)))
            except (ValueError, TypeError):
                pass

    # Build patient -> codes
    patient_codes = defaultdict(set)
    for event_str, codes in event_to_codes.items():
        if event_str in event_to_patient:
            patient_uri = event_to_patient[event_str]
            patient_codes[patient_uri].update(codes)

    # Build patient -> events (for lab bin edges)
    patient_events = defaultdict(list)
    for event_str, patient_uri in event_to_patient.items():
        patient_events[patient_uri].append(event_str)

    patients = sorted(patient_codes.keys())
    all_codes = set()
    for codes in patient_codes.values():
        all_codes.update(codes)

    print(f"  {len(patients)} patients, {len(all_codes)} unique codes")
    print(f"  {sum(len(c) for c in patient_codes.values())} patient-code edges")
    print(f"  {len(lab_values)} lab value observations")

    del g  # free memory

    return patients, patient_codes, all_codes, lab_values, event_to_patient, event_to_codestring


def discretize_labs(lab_values):
    """Bin lab values into quartiles per code, return (event_uri, bin_id) pairs."""
    code_values = defaultdict(list)
    code_events = defaultdict(list)
    for event_uri, code_str, val in lab_values:
        code_values[code_str].append(val)
        code_events[code_str].append((event_uri, val))

    bin_edges = []  # (event_uri, bin_label)
    codes_binned = 0
    for code_str, values in code_values.items():
        if len(values) < MIN_VALUES_PER_CODE:
            continue
        arr = np.array(values)
        boundaries = np.percentile(arr, [25, 50, 75])
        codes_binned += 1

        code_key = code_str.replace("//", "_").replace(" ", "_")
        for event_uri, val in code_events[code_str]:
            if val <= boundaries[0]:
                q = "Q1"
            elif val <= boundaries[1]:
                q = "Q2"
            elif val <= boundaries[2]:
                q = "Q3"
            else:
                q = "Q4"
            bin_edges.append((event_uri, f"lab_bin:{code_key}/{q}"))

    print(f"  Discretized {codes_binned} codes into {N_BINS} bins, {len(bin_edges)} bin edges")
    return bin_edges


def extract_ontology_hierarchy(ontology_files, relevant_codes):
    """Load ontology files and extract rdfs:subClassOf edges for relevant codes.

    Only keeps hierarchy edges where at least one endpoint is in relevant_codes
    or is an ancestor reachable within 3 hops.
    """
    print("Loading ontology hierarchies...")
    hierarchy_edges = []  # (child, parent)
    all_hierarchy = defaultdict(set)  # child -> parents

    for onto_name, onto_path in ontology_files:
        if not os.path.exists(onto_path):
            print(f"  WARNING: {onto_name} not found, skipping")
            continue
        t0 = time.time()
        g = rdflib.Graph()
        g.parse(os.path.abspath(onto_path), format="turtle")
        count = 0
        for s, p, o in g.triples((None, rdflib.URIRef(RDFS_SUBCLASS), None)):
            if not isinstance(o, rdflib.Literal):
                all_hierarchy[str(s)].add(str(o))
                count += 1
        print(f"  {onto_name}: {count} subClassOf edges in {time.time() - t0:.1f}s")
        del g

    # BFS from relevant_codes to collect ancestors up to 3 hops
    reachable = set(relevant_codes)
    frontier = set(relevant_codes)
    for hop in range(3):
        next_frontier = set()
        for node in frontier:
            if node in all_hierarchy:
                for parent in all_hierarchy[node]:
                    if parent not in reachable:
                        next_frontier.add(parent)
                        reachable.add(parent)
        frontier = next_frontier
        if not frontier:
            break

    # Keep only edges where both endpoints are reachable
    for child, parents in all_hierarchy.items():
        if child in reachable:
            for parent in parents:
                if parent in reachable:
                    hierarchy_edges.append((child, parent))

    print(f"  Kept {len(hierarchy_edges)} hierarchy edges ({len(reachable)} reachable nodes)")
    return hierarchy_edges, reachable


def build_pyg_graph(patients, patient_codes, all_codes, hierarchy_edges,
                    hierarchy_nodes, lab_bin_edges, event_to_patient):
    """Build a PyG Data object from the extracted graph components.

    Node types (all in a single homogeneous graph with type tracked separately):
      - patient nodes (index 0..N_patients-1)
      - code nodes (index N_patients..N_patients+N_codes-1)
      - lab bin nodes (index after codes)

    Edge types (all treated as undirected):
      - patient <-> code (from patient_codes)
      - code <-> code (from hierarchy_edges)
      - patient <-> lab_bin (aggregated from event-level bin assignments)
    """
    # Assign node indices
    node_id = {}
    node_type = []  # 0=patient, 1=code, 2=lab_bin

    for p in patients:
        node_id[p] = len(node_id)
        node_type.append(0)

    # Add code nodes (from patient-code edges + hierarchy)
    all_code_nodes = set(all_codes) | hierarchy_nodes
    for c in sorted(all_code_nodes):
        if c not in node_id:
            node_id[c] = len(node_id)
            node_type.append(1)

    # Aggregate lab bins to patient level
    patient_bins = defaultdict(set)
    for event_uri, bin_label in lab_bin_edges:
        if event_uri in event_to_patient:
            patient_uri = event_to_patient[event_uri]
            patient_bins[patient_uri].add(bin_label)

    all_bin_labels = set()
    for bins in patient_bins.values():
        all_bin_labels.update(bins)

    for b in sorted(all_bin_labels):
        node_id[b] = len(node_id)
        node_type.append(2)

    num_nodes = len(node_id)
    print(f"  Graph nodes: {sum(1 for t in node_type if t == 0)} patients, "
          f"{sum(1 for t in node_type if t == 1)} codes, "
          f"{sum(1 for t in node_type if t == 2)} lab bins, "
          f"{num_nodes} total")

    # Build edge index (undirected)
    src, dst = [], []

    # Patient <-> code edges
    pc_count = 0
    for patient_uri, codes in patient_codes.items():
        pi = node_id[patient_uri]
        for code_uri in codes:
            if code_uri in node_id:
                ci = node_id[code_uri]
                src.extend([pi, ci])
                dst.extend([ci, pi])
                pc_count += 1

    # Hierarchy edges (code <-> parent code)
    hier_count = 0
    for child, parent in hierarchy_edges:
        if child in node_id and parent in node_id:
            ci = node_id[child]
            pi = node_id[parent]
            src.extend([ci, pi])
            dst.extend([pi, ci])
            hier_count += 1

    # Patient <-> lab bin edges
    lb_count = 0
    for patient_uri, bins in patient_bins.items():
        if patient_uri in node_id:
            pi = node_id[patient_uri]
            for bin_label in bins:
                if bin_label in node_id:
                    bi = node_id[bin_label]
                    src.extend([pi, bi])
                    dst.extend([bi, pi])
                    lb_count += 1

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    node_type_tensor = torch.tensor(node_type, dtype=torch.long)

    print(f"  Edges: {pc_count} patient-code, {hier_count} hierarchy, {lb_count} patient-labbin, "
          f"{edge_index.shape[1]} total (undirected)")

    # Node features: learned embeddings (initialized randomly)
    # We use a learnable embedding table rather than fixed features
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    data.node_type = node_type_tensor

    return data, node_id, patients


class GraphSAGEEncoder(nn.Module):
    """Multi-layer GraphSAGE encoder producing node embeddings."""
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, out_dim))

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class UnsupervisedGNN(nn.Module):
    """GNN with learnable node embeddings and edge-dropout contrastive loss.

    Uses two augmented views of the graph (via edge dropout) and a
    Barlow Twins-style loss to prevent embedding collapse.
    """
    def __init__(self, num_nodes, num_node_types, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, in_dim)
        self.type_emb = nn.Embedding(num_node_types, in_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)
        nn.init.xavier_uniform_(self.type_emb.weight)
        self.encoder = GraphSAGEEncoder(in_dim, hidden_dim, out_dim, num_layers)
        # Projection head for contrastive loss
        self.projector = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.out_dim = out_dim

    def encode(self, edge_index, node_types):
        x = self.node_emb.weight + self.type_emb(node_types)
        z = self.encoder(x, edge_index)
        return z

    def forward(self, edge_index, node_types):
        return self.encode(edge_index, node_types)

    def drop_edges(self, edge_index, drop_rate=0.2):
        """Randomly drop edges for augmentation."""
        num_edges = edge_index.shape[1]
        mask = torch.rand(num_edges) > drop_rate
        return edge_index[:, mask]

    def barlow_twins_loss(self, z1, z2, patient_mask, lambd=0.05):
        """Barlow Twins loss on patient node embeddings.

        Encourages the two augmented views to produce identical embeddings
        while decorrelating the embedding dimensions (preventing collapse).
        """
        # Project and normalize
        p1 = self.projector(z1[patient_mask])
        p2 = self.projector(z2[patient_mask])

        # Normalize along batch dimension
        p1 = (p1 - p1.mean(0)) / (p1.std(0) + 1e-5)
        p2 = (p2 - p2.mean(0)) / (p2.std(0) + 1e-5)

        N = p1.shape[0]
        # Cross-correlation matrix
        c = (p1.T @ p2) / N

        # Loss: diagonal should be 1, off-diagonal should be 0
        on_diag = (torch.diagonal(c) - 1).pow(2).sum()
        off_diag = c.flatten()[1:].view(self.out_dim - 1, self.out_dim + 1)[:, :-1].pow(2).sum()

        return on_diag + lambd * off_diag

    def contrastive_loss(self, z, edge_index, neg_samples=10):
        """Edge-level contrastive loss with margin."""
        num_edges = edge_index.shape[1]
        max_pos = min(num_edges, 50000)
        perm = torch.randperm(num_edges)[:max_pos]
        pos_src = edge_index[0, perm]
        pos_dst = edge_index[1, perm]

        # L2 normalize embeddings
        z_norm = F.normalize(z, dim=-1)

        # Positive scores
        pos_scores = (z_norm[pos_src] * z_norm[pos_dst]).sum(dim=-1)

        # Negative sampling
        neg_dst = torch.randint(0, z.shape[0], (max_pos * neg_samples,))
        neg_src = pos_src.repeat(neg_samples)
        neg_scores = (z_norm[neg_src] * z_norm[neg_dst]).sum(dim=-1)

        pos_loss = -F.logsigmoid(pos_scores).mean()
        neg_loss = -F.logsigmoid(-neg_scores).mean()

        return pos_loss + neg_loss


def train(model, data, epochs, lr):
    """Train the GNN model with combined Barlow Twins + contrastive loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    edge_index = data.edge_index
    node_types = data.node_type
    patient_mask = (node_types == 0)

    losses = []
    for epoch in range(epochs):
        model.train()

        # Two augmented views via edge dropout
        ei1 = model.drop_edges(edge_index, drop_rate=0.2)
        ei2 = model.drop_edges(edge_index, drop_rate=0.2)
        z1 = model.encode(ei1, node_types)
        z2 = model.encode(ei2, node_types)

        # Combined loss
        bt_loss = model.barlow_twins_loss(z1, z2, patient_mask)
        cl_loss = model.contrastive_loss(z1, edge_index, neg_samples=NEG_SAMPLES)
        loss = bt_loss + cl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={loss.item():.4f} "
                  f"(BT={bt_loss.item():.4f}, CL={cl_loss.item():.4f})")

    return losses


def evaluate(gnn_emb, patient_ids):
    """Print basic embedding statistics."""
    sim = cosine_similarity(gnn_emb)
    triu_idx = np.triu_indices(len(patient_ids), k=1)
    pairs = sim[triu_idx]
    print(f"\n  GNN embedding similarity stats:")
    print(f"    Mean: {pairs.mean():.4f}, Std: {pairs.std():.4f}")
    print(f"    Range: [{pairs.min():.4f}, {pairs.max():.4f}]")

    # Compare with RDF2Vec if available
    rdf2vec_path = os.path.join(PROJECT_ROOT, "analysis/rdf2vec/patient_embeddings.csv")
    if os.path.exists(rdf2vec_path):
        rdf_df = pd.read_csv(rdf2vec_path).sort_values("patient_id").reset_index(drop=True)
        rdf_emb = rdf_df.drop(columns=["patient_id"]).values
        sim_rdf = cosine_similarity(rdf_emb)
        rho, _ = spearmanr(sim[triu_idx], sim_rdf[triu_idx])
        print(f"\n  Spearman correlation with RDF2Vec: rho={rho:.4f}")

    # Compare with text embeddings if available
    for name, fname in [("Template", "template_embeddings.csv"),
                        ("LLM Summary", "summary_embeddings.csv")]:
        path = os.path.join(PROJECT_ROOT, f"analysis/text-embeddings/{fname}")
        if os.path.exists(path):
            df = pd.read_csv(path).sort_values("patient_id").reset_index(drop=True)
            emb = df.drop(columns=["patient_id"]).values
            sim_other = cosine_similarity(emb)
            rho, _ = spearmanr(sim[triu_idx], sim_other[triu_idx])
            print(f"  Spearman correlation with {name}: rho={rho:.4f}")

    # NN overlap with other methods
    print(f"\n  Nearest-neighbor agreement (top-5) — GNN vs others:")
    sim_gnn = sim.copy()
    np.fill_diagonal(sim_gnn, -1)
    for name, path in [("RDF2Vec", rdf2vec_path),
                       ("Template", os.path.join(PROJECT_ROOT, "analysis/text-embeddings/template_embeddings.csv")),
                       ("LLM Summary", os.path.join(PROJECT_ROOT, "analysis/text-embeddings/summary_embeddings.csv"))]:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path).sort_values("patient_id").reset_index(drop=True)
        emb = df.drop(columns=["patient_id"]).values
        sim_other = cosine_similarity(emb)
        np.fill_diagonal(sim_other, -1)
        overlaps = []
        for j in range(len(patient_ids)):
            top_gnn = set(np.argsort(sim_gnn[j])[-5:])
            top_other = set(np.argsort(sim_other[j])[-5:])
            overlaps.append(len(top_gnn & top_other) / 5)
        print(f"    GNN vs {name:12s}: {np.mean(overlaps):.1%} avg overlap")


def plot_results(gnn_emb, patient_ids, losses):
    """Visualize GNN embeddings."""
    fig = plt.figure(figsize=(18, 5))

    # 1. Training loss
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(losses, color="steelblue", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Contrastive Loss")
    ax.set_title("GNN Training Loss")
    ax.grid(True, alpha=0.3)

    # 2. Similarity matrix
    ax = fig.add_subplot(1, 3, 2)
    sim = cosine_similarity(gnn_emb)
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-0.2, vmax=1)
    ax.set_title("GNN Pairwise Similarity")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # 3. t-SNE
    ax = fig.add_subplot(1, 3, 3)
    perp = min(30, len(patient_ids) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(gnn_emb)
    ax.scatter(tsne[:, 0], tsne[:, 1], s=30, alpha=0.7, c="darkorange",
               edgecolors="white", linewidth=0.3)
    for j, pid in enumerate(patient_ids):
        ax.annotate(str(pid), (tsne[j, 0], tsne[j, 1]), fontsize=4, alpha=0.5)
    ax.set_title("GNN Embeddings (t-SNE)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    fig.suptitle(f"GNN Patient Embeddings ({EMBEDDING_DIM}d, GraphSAGE)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(OUTPUT_DIR, "gnn_embeddings.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 1. Extract graph from MIMIC
    patients, patient_codes, all_codes, lab_values, event_to_patient, _ = \
        extract_graph_from_mimic(MIMIC_TTL)

    # 2. Discretize lab values
    print("\nDiscretizing lab values...")
    lab_bin_edges = discretize_labs(lab_values)

    # 3. Extract ontology hierarchy (only for codes we actually use)
    hierarchy_edges, hierarchy_nodes = extract_ontology_hierarchy(ONTOLOGY_FILES, all_codes)

    # 4. Build PyG graph
    print("\nBuilding PyG graph...")
    data, node_id, patient_list = build_pyg_graph(
        patients, patient_codes, all_codes,
        hierarchy_edges, hierarchy_nodes,
        lab_bin_edges, event_to_patient
    )

    # 5. Train GNN
    print(f"\nTraining GraphSAGE (dim={EMBEDDING_DIM}, layers={NUM_LAYERS}, epochs={EPOCHS})...")
    num_node_types = 3  # patient, code, lab_bin
    model = UnsupervisedGNN(
        data.num_nodes, num_node_types, EMBEDDING_DIM, HIDDEN_DIM, EMBEDDING_DIM, NUM_LAYERS
    )
    t0 = time.time()
    losses = train(model, data, EPOCHS, LR)
    print(f"  Training time: {time.time() - t0:.1f}s")

    # 6. Extract patient embeddings
    print("\nExtracting patient embeddings...")
    model.eval()
    with torch.no_grad():
        all_z = model(data.edge_index, data.node_type)
    patient_indices = [node_id[p] for p in patient_list]
    gnn_emb = all_z[patient_indices].numpy()
    # L2 normalize
    norms = np.linalg.norm(gnn_emb, axis=1, keepdims=True)
    gnn_emb = gnn_emb / np.maximum(norms, 1e-8)
    print(f"  GNN embeddings shape: {gnn_emb.shape}")

    # 7. Extract patient IDs
    patient_ids = [uri.split("/")[-1] for uri in patient_list]

    # 8. Save
    np.save(os.path.join(OUTPUT_DIR, "gnn_embeddings.npy"), gnn_emb)
    cols = [f"dim_{i}" for i in range(EMBEDDING_DIM)]
    df = pd.DataFrame(gnn_emb, columns=cols)
    df.insert(0, "patient_id", patient_ids)
    df.to_csv(os.path.join(OUTPUT_DIR, "gnn_embeddings.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "gnn_model.pt"))
    print(f"  Saved gnn_embeddings.npy/csv and gnn_model.pt")

    # 9. Evaluate and plot
    evaluate(gnn_emb, patient_ids)
    plot_results(gnn_emb, patient_ids, losses)

    # Save graph stats
    stats = {
        "num_patients": len(patient_list),
        "num_code_nodes": sum(1 for t in data.node_type.tolist() if t == 1),
        "num_lab_bin_nodes": sum(1 for t in data.node_type.tolist() if t == 2),
        "num_edges": data.edge_index.shape[1],
        "embedding_dim": EMBEDDING_DIM,
        "num_layers": NUM_LAYERS,
        "epochs": EPOCHS,
        "final_loss": losses[-1],
    }
    with open(os.path.join(OUTPUT_DIR, "graph_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
