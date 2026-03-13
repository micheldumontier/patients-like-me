"""
Run RDF2Vec on the MIMIC-IV demo + medical ontology graph to generate patient embeddings.

Custom implementation using rdflib for graph loading and gensim Word2Vec,
with in-memory adjacency lists for fast random walks (including reverse).

Loads MIMIC-IV demo data alongside ICD9CM, ICD10CM, RXNORM, LNC (LOINC),
and SNOMEDCT ontologies for richer graph structure.
"""
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import rdflib
from gensim.models import Word2Vec

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
MIMIC_TTL = os.path.join(DATA_DIR, "mimic-iv-demo.ttl")
ONTOLOGY_FILES = [
    ("ICD9CM", os.path.join(DATA_DIR, "ICD9CM.ttl")),
    ("ICD10CM", os.path.join(DATA_DIR, "ICD10CM.ttl")),
    ("RXNORM", os.path.join(DATA_DIR, "RXNORM.ttl")),
    ("LNC", os.path.join(DATA_DIR, "LNC.ttl")),
    ("SNOMEDCT", os.path.join(DATA_DIR, "SNOMEDCT.ttl")),
]

# RDF2Vec parameters
VECTOR_SIZE = 200
N_WALKS = 100       # random walks per entity
MAX_DEPTH = 4       # max walk depth (hops)
SEED = 42
WITH_REVERSE = True

# Predicates to skip during walks (literal-heavy, not structurally useful)
SKIP_PREDICATES = {
    "https://teamheka.github.io/meds-ontology#numericValue",
    "https://teamheka.github.io/meds-ontology#textValue",
    "https://teamheka.github.io/meds-ontology#time",
    "https://teamheka.github.io/meds-ontology#codeString",
    "https://teamheka.github.io/meds-ontology#codeDescription",
    "http://www.w3.org/2004/02/skos/core#prefLabel",
    "http://www.w3.org/2004/02/skos/core#notation",
    "http://bioportal.bioontology.org/ontologies/umls/cui",
    "http://bioportal.bioontology.org/ontologies/umls/tui",
    "http://www.w3.org/ns/prov#wasDerivedFrom",
}

# Lab value discretization
N_BINS = 4  # quartiles
LAB_BIN_PREDICATE = "urn:mimic-miner#hasLabBin"
LAB_BIN_PREFIX = "urn:mimic-miner/lab-bin/"
MIN_VALUES_PER_CODE = 10  # minimum values to create bins for a code

OUTPUT_DIR = os.path.dirname(__file__) or "."


def discretize_lab_values(graph):
    """Extract numeric lab values and create discretized bin nodes.

    Returns:
        edges: list of (event_uri, bin_uri) tuples to add to the adjacency graph
    """
    meds = rdflib.Namespace("https://teamheka.github.io/meds-ontology#")

    # Collect (event_uri, code_string, numeric_value)
    print("  Extracting numeric values from events...")
    event_values = []
    for event, _, val in graph.triples((None, meds.numericValue, None)):
        try:
            num = float(val)
        except (ValueError, TypeError):
            continue
        for _, _, code in graph.triples((event, meds.hasCode, None)):
            for _, _, cs in graph.triples((code, meds.codeString, None)):
                event_values.append((str(event), str(cs), num))
                break
            break

    print(f"  Found {len(event_values)} events with numeric values")

    # Group by code string
    code_values = defaultdict(list)
    code_events = defaultdict(list)
    for event_uri, code_str, num in event_values:
        code_values[code_str].append(num)
        code_events[code_str].append((event_uri, num))

    # Compute quantile boundaries and assign bins
    edges = []
    codes_binned = 0
    for code_str, values in code_values.items():
        if len(values) < MIN_VALUES_PER_CODE:
            continue
        arr = np.array(values)
        boundaries = np.percentile(arr, [25, 50, 75])
        codes_binned += 1

        # Sanitize code string for use in URIs (replace //, spaces)
        code_key = code_str.replace("//", "_").replace(" ", "_")

        for event_uri, num in code_events[code_str]:
            if num <= boundaries[0]:
                bin_label = "Q1"
            elif num <= boundaries[1]:
                bin_label = "Q2"
            elif num <= boundaries[2]:
                bin_label = "Q3"
            else:
                bin_label = "Q4"
            bin_uri = f"{LAB_BIN_PREFIX}{code_key}/{bin_label}"
            edges.append((event_uri, bin_uri))

    print(f"  Created bins for {codes_binned} codes, {len(edges)} bin edges")
    return edges


def build_adjacency(graph, skip_predicates, lab_edges=None):
    """Build forward and reverse adjacency lists from an rdflib graph.

    Args:
        graph: rdflib Graph
        skip_predicates: set of predicate URIs to skip
        lab_edges: optional list of (event_uri, bin_uri) from discretize_lab_values

    Returns:
        fwd: dict mapping subject -> list of (predicate, object)
        rev: dict mapping object -> list of (predicate, subject)
    """
    fwd = defaultdict(list)
    rev = defaultdict(list)
    skipped = 0
    for s, p, o in graph:
        ps = str(p)
        if ps in skip_predicates:
            skipped += 1
            continue
        # Skip literals as walk targets
        if isinstance(o, rdflib.Literal):
            skipped += 1
            continue
        ss, os_ = str(s), str(o)
        fwd[ss].append((ps, os_))
        rev[os_].append((ps, ss))

    # Add discretized lab value edges
    if lab_edges:
        for event_uri, bin_uri in lab_edges:
            fwd[event_uri].append((LAB_BIN_PREDICATE, bin_uri))
            rev[bin_uri].append((LAB_BIN_PREDICATE, event_uri))
        print(f"  Added {len(lab_edges)} lab bin edges")

    print(f"  Adjacency: {len(fwd)} forward nodes, {len(rev)} reverse nodes, {skipped} edges skipped")
    return fwd, rev


def random_walks(entity, fwd, rev, n_walks, max_depth, with_reverse, rng):
    """Generate random walks starting from entity."""
    walks = []
    for _ in range(n_walks):
        walk = [entity]
        current = entity
        for _ in range(max_depth):
            neighbors = []
            if current in fwd:
                neighbors.extend(("fwd", p, o) for p, o in fwd[current])
            if with_reverse and current in rev:
                neighbors.extend(("rev", p, s) for p, s in rev[current])
            if not neighbors:
                break
            direction, pred, next_node = rng.choice(neighbors)
            # Encode direction in predicate to distinguish forward/reverse
            walk_pred = pred if direction == "fwd" else f"INV_{pred}"
            walk.append(walk_pred)
            walk.append(next_node)
            current = next_node
        walks.append(walk)
    return walks


def main():
    rng = random.Random(SEED)

    # Load graph
    print("Loading RDF graph...")
    g = rdflib.Graph()
    t0 = time.time()
    g.parse(os.path.abspath(MIMIC_TTL), format="turtle")
    print(f"  MIMIC: {len(g)} triples in {time.time() - t0:.1f}s")
    for onto_name, onto_path in ONTOLOGY_FILES:
        if not os.path.exists(onto_path):
            print(f"  WARNING: {onto_name} not found at {onto_path}, skipping")
            continue
        t1 = time.time()
        g.parse(os.path.abspath(onto_path), format="turtle")
        print(f"  +{onto_name}: {len(g)} triples total in {time.time() - t1:.1f}s")

    # Extract patient entities
    meds = rdflib.Namespace("https://teamheka.github.io/meds-ontology#")
    entities = sorted(set(str(s) for s in g.objects(predicate=meds.hasSubject)))
    print(f"  Found {len(entities)} patient entities")

    # Discretize lab values into bin nodes
    print("Discretizing lab values...")
    t2 = time.time()
    lab_edges = discretize_lab_values(g)
    print(f"  Done in {time.time() - t2:.1f}s")

    # Build adjacency
    print("Building adjacency lists...")
    t3 = time.time()
    fwd, rev = build_adjacency(g, SKIP_PREDICATES, lab_edges)
    print(f"  Built in {time.time() - t3:.1f}s")
    del g, lab_edges  # free rdflib graph memory

    # Generate walks
    print(f"Generating random walks (walks={N_WALKS}, depth={MAX_DEPTH}, reverse={WITH_REVERSE})...")
    t3 = time.time()
    all_walks = []
    for i, entity in enumerate(entities):
        walks = random_walks(entity, fwd, rev, N_WALKS, MAX_DEPTH, WITH_REVERSE, rng)
        all_walks.extend(walks)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(entities)} entities done")
    print(f"  Generated {len(all_walks)} walks in {time.time() - t3:.1f}s")
    avg_len = np.mean([len(w) for w in all_walks])
    print(f"  Average walk length: {avg_len:.1f} tokens")

    # Train Word2Vec
    print(f"Training Word2Vec (vector_size={VECTOR_SIZE}, sg=1)...")
    t4 = time.time()
    model = Word2Vec(
        sentences=all_walks,
        vector_size=VECTOR_SIZE,
        sg=1,
        min_count=1,
        workers=1,
        seed=SEED,
        epochs=10,
    )
    print(f"  Trained in {time.time() - t4:.1f}s, vocabulary size: {len(model.wv)}")

    # Extract embeddings for patient entities
    embeddings = []
    missing = 0
    for entity in entities:
        if entity in model.wv:
            embeddings.append(model.wv[entity])
        else:
            embeddings.append(np.zeros(VECTOR_SIZE))
            missing += 1
    if missing:
        print(f"  WARNING: {missing} entities not in vocabulary")

    embeddings_array = np.array(embeddings)
    print(f"Embeddings shape: {embeddings_array.shape}")

    # Extract patient IDs
    patient_ids = [uri.split("/")[-1] for uri in entities]

    # Save
    np.save(os.path.join(OUTPUT_DIR, "patient_embeddings.npy"), embeddings_array)

    cols = [f"dim_{i}" for i in range(VECTOR_SIZE)]
    df = pd.DataFrame(embeddings_array, columns=cols)
    df.insert(0, "patient_id", patient_ids)
    df.to_csv(os.path.join(OUTPUT_DIR, "patient_embeddings.csv"), index=False)

    entity_map = {uri: idx for idx, uri in enumerate(entities)}
    with open(os.path.join(OUTPUT_DIR, "entity_map.json"), "w") as f:
        json.dump(entity_map, f, indent=2)

    print(f"\nSaved embeddings to {OUTPUT_DIR}/patient_embeddings.npy")
    print(f"Saved CSV to {OUTPUT_DIR}/patient_embeddings.csv")
    print(f"Saved entity map to {OUTPUT_DIR}/entity_map.json")
    print(f"Total time: {time.time() - t0:.1f}s")
    print("Done!")


if __name__ == "__main__":
    main()
