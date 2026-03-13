"""
Generate patient embeddings from verbalized clinical records.

1. Template-based: SPARQL queries → structured text per patient
2. LLM-enriched: Template text → Azure OpenAI clinical summary
3. Embed both with text-embedding-3-large
4. Compare the two embedding spaces
"""
import json
import os
import sys
import time
from urllib import parse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from openai import AzureOpenAI
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
OUTPUT_DIR = os.path.dirname(__file__) or "."

# QLever
QLEVER_ENDPOINT = "http://localhost:6335"
ACCESS_TOKEN = "mimic_demo_token"

# Azure OpenAI
EMBEDDING_MODEL = "text-embedding-3-large"
CHAT_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2")

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)


def sparql_query(query):
    """Execute a SPARQL query against QLever."""
    r = requests.post(
        QLEVER_ENDPOINT,
        data={"query": query, "access-token": ACCESS_TOKEN},
    )
    r.raise_for_status()
    return r.json()["results"]["bindings"]


def get_patients():
    results = sparql_query("""
        PREFIX meds: <https://teamheka.github.io/meds-ontology#>
        SELECT DISTINCT ?patient WHERE {
            ?event meds:hasSubject ?patient .
        } ORDER BY ?patient
    """)
    return [r["patient"]["value"] for r in results]


def get_patient_data(patient_uri):
    """Fetch diagnoses, medications, labs, and procedures for a patient."""
    data = {}

    # Diagnoses with ICD descriptions
    results = sparql_query(f"""
        PREFIX meds: <https://teamheka.github.io/meds-ontology#>
        SELECT DISTINCT ?codeStr ?desc WHERE {{
            ?event meds:hasSubject <{patient_uri}> ;
                   meds:hasCode ?code .
            ?code meds:codeString ?codeStr .
            OPTIONAL {{ ?code meds:codeDescription ?desc }}
            FILTER(STRSTARTS(?codeStr, "DIAGNOSIS"))
        }}
        ORDER BY ?codeStr
    """)
    data["diagnoses"] = [
        {"code": r["codeStr"]["value"], "desc": r.get("desc", {}).get("value", "")}
        for r in results
    ]

    # Medications
    results = sparql_query(f"""
        PREFIX meds: <https://teamheka.github.io/meds-ontology#>
        SELECT DISTINCT ?codeStr ?desc WHERE {{
            ?event meds:hasSubject <{patient_uri}> ;
                   meds:hasCode ?code .
            ?code meds:codeString ?codeStr .
            OPTIONAL {{ ?code meds:codeDescription ?desc }}
            FILTER(STRSTARTS(?codeStr, "MEDICATION"))
        }}
        ORDER BY ?codeStr
    """)
    data["medications"] = [
        {"code": r["codeStr"]["value"], "desc": r.get("desc", {}).get("value", "")}
        for r in results
    ]

    # Labs (top 20 most frequent)
    results = sparql_query(f"""
        PREFIX meds: <https://teamheka.github.io/meds-ontology#>
        SELECT ?codeStr ?desc (COUNT(?event) AS ?count) WHERE {{
            ?event meds:hasSubject <{patient_uri}> ;
                   meds:hasCode ?code .
            ?code meds:codeString ?codeStr .
            OPTIONAL {{ ?code meds:codeDescription ?desc }}
            FILTER(STRSTARTS(?codeStr, "LAB"))
        }}
        GROUP BY ?codeStr ?desc
        ORDER BY DESC(?count)
        LIMIT 20
    """)
    data["labs"] = [
        {"code": r["codeStr"]["value"], "desc": r.get("desc", {}).get("value", ""), "count": r["count"]["value"]}
        for r in results
    ]

    # Procedures
    results = sparql_query(f"""
        PREFIX meds: <https://teamheka.github.io/meds-ontology#>
        SELECT DISTINCT ?codeStr ?desc WHERE {{
            ?event meds:hasSubject <{patient_uri}> ;
                   meds:hasCode ?code .
            ?code meds:codeString ?codeStr .
            OPTIONAL {{ ?code meds:codeDescription ?desc }}
            FILTER(STRSTARTS(?codeStr, "PROCEDURE"))
        }}
        ORDER BY ?codeStr
    """)
    data["procedures"] = [
        {"code": r["codeStr"]["value"], "desc": r.get("desc", {}).get("value", "")}
        for r in results
    ]

    # Event time range
    results = sparql_query(f"""
        PREFIX meds: <https://teamheka.github.io/meds-ontology#>
        SELECT (MIN(?t) AS ?first) (MAX(?t) AS ?last) (COUNT(?event) AS ?total) WHERE {{
            ?event meds:hasSubject <{patient_uri}> ;
                   meds:time ?t .
        }}
    """)
    if results:
        data["first_event"] = results[0].get("first", {}).get("value", "")
        data["last_event"] = results[0].get("last", {}).get("value", "")
        data["total_events"] = results[0].get("total", {}).get("value", "0")

    return data


def verbalize_template(patient_id, data):
    """Create a structured text verbalization of a patient record."""
    lines = [f"Patient {patient_id}."]
    lines.append(f"Total clinical events: {data.get('total_events', 'unknown')}.")
    if data.get("first_event"):
        lines.append(f"Record spans from {data['first_event'][:10]} to {data['last_event'][:10]}.")

    if data["diagnoses"]:
        dx_items = []
        for d in data["diagnoses"]:
            if d["desc"]:
                dx_items.append(f"{d['desc']} ({d['code']})")
            else:
                dx_items.append(d["code"])
        lines.append(f"Diagnoses: {'; '.join(dx_items)}.")

    if data["medications"]:
        med_items = []
        for m in data["medications"]:
            # Extract drug name from code string like "MEDICATION//START//Aspirin"
            parts = m["code"].split("//")
            name = parts[-1] if len(parts) > 1 else m["code"]
            if name not in [mi.split(" (")[0] for mi in med_items]:
                med_items.append(name)
        lines.append(f"Medications: {'; '.join(med_items)}.")

    if data["labs"]:
        lab_items = []
        for lab in data["labs"]:
            if lab["desc"]:
                lab_items.append(f"{lab['desc']} (×{lab['count']})")
            else:
                lab_items.append(f"{lab['code']} (×{lab['count']})")
        lines.append(f"Top labs: {'; '.join(lab_items)}.")

    if data["procedures"]:
        proc_items = []
        for p in data["procedures"]:
            if p["desc"]:
                proc_items.append(p["desc"])
            else:
                proc_items.append(p["code"])
        lines.append(f"Procedures: {'; '.join(proc_items)}.")

    return "\n".join(lines)


def enrich_with_llm(template_text):
    """Use Azure OpenAI to generate a clinical narrative from the template."""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical informatics specialist. Given structured patient data, "
                    "write a concise clinical summary paragraph (3-5 sentences) that captures "
                    "the key clinical picture, likely condition clusters, and notable patterns. "
                    "Use standard medical terminology. Do not invent information not present in the data."
                ),
            },
            {"role": "user", "content": template_text},
        ],
        max_completion_tokens=300,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def embed_texts(texts, batch_size=20):
    """Embed a list of texts using Azure OpenAI."""
    # Sanitize: replace empty/null with placeholder, strip null bytes
    sanitized = []
    for t in texts:
        t = (t or "No data available.").replace("\x00", "").strip()
        if not t:
            t = "No data available."
        sanitized.append(t)

    all_embeddings = []
    for i in range(0, len(sanitized), batch_size):
        batch = sanitized[i : i + batch_size]
        try:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            all_embeddings.extend([d.embedding for d in response.data])
        except Exception as e:
            # Fall back to one-by-one embedding
            print(f"    Batch {i} failed ({e}), embedding individually...")
            for j, text in enumerate(batch):
                try:
                    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
                    all_embeddings.append(response.data[0].embedding)
                except Exception as e2:
                    print(f"    Item {i+j} failed: {e2}, using zeros")
                    all_embeddings.append([0.0] * 3072)
                time.sleep(0.2)
        if i + batch_size < len(sanitized):
            time.sleep(0.5)
    return np.array(all_embeddings)


def main():
    print("=" * 60)
    print("Text-based Patient Embedding Pipeline")
    print("=" * 60)

    # Step 1: Get patients
    print("\n[1/5] Fetching patient list...")
    patients = get_patients()
    patient_ids = [uri.split("/")[-1] for uri in patients]
    print(f"  Found {len(patients)} patients")

    # Step 2: Build template verbalizations
    templates_path = os.path.join(OUTPUT_DIR, "patient_templates.json")
    if os.path.exists(templates_path):
        print("\n[2/5] Loading cached templates...")
        with open(templates_path) as f:
            templates = json.load(f)
        print(f"  Loaded {len(templates)} templates (avg {np.mean([len(t) for t in templates.values()]):.0f} chars)")
    else:
        print("\n[2/5] Building template verbalizations via SPARQL...")
        templates = {}
        for i, (uri, pid) in enumerate(zip(patients, patient_ids)):
            data = get_patient_data(uri)
            templates[pid] = verbalize_template(pid, data)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(patients)} patients verbalized")
        print(f"  Done. Avg template length: {np.mean([len(t) for t in templates.values()]):.0f} chars")
        with open(templates_path, "w") as f:
            json.dump(templates, f, indent=2)
        print("  Saved patient_templates.json")

    # Step 3: Enrich with LLM
    summaries_path = os.path.join(OUTPUT_DIR, "patient_summaries.json")
    if os.path.exists(summaries_path):
        print(f"\n[3/5] Loading cached LLM summaries...")
        with open(summaries_path) as f:
            summaries = json.load(f)
        print(f"  Loaded {len(summaries)} summaries (avg {np.mean([len(s) for s in summaries.values()]):.0f} chars)")
    else:
        print(f"\n[3/5] Generating LLM summaries ({CHAT_MODEL})...")
        summaries = {}
        for i, pid in enumerate(patient_ids):
            summaries[pid] = enrich_with_llm(templates[pid])
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(patients)} summaries generated")
            time.sleep(0.3)
        print(f"  Done. Avg summary length: {np.mean([len(s) for s in summaries.values()]):.0f} chars")
        with open(summaries_path, "w") as f:
            json.dump(summaries, f, indent=2)
        print("  Saved patient_summaries.json")

    # Step 4: Embed both
    print(f"\n[4/5] Generating embeddings ({EMBEDDING_MODEL})...")
    template_texts = [templates[pid] for pid in patient_ids]
    summary_texts = [summaries[pid] for pid in patient_ids]

    t0 = time.time()
    template_embeddings = embed_texts(template_texts)
    print(f"  Template embeddings: {template_embeddings.shape} in {time.time() - t0:.1f}s")

    t0 = time.time()
    summary_embeddings = embed_texts(summary_texts)
    print(f"  Summary embeddings:  {summary_embeddings.shape} in {time.time() - t0:.1f}s")

    # Save embeddings
    np.save(os.path.join(OUTPUT_DIR, "template_embeddings.npy"), template_embeddings)
    np.save(os.path.join(OUTPUT_DIR, "summary_embeddings.npy"), summary_embeddings)

    # Save CSV versions
    for name, emb in [("template", template_embeddings), ("summary", summary_embeddings)]:
        cols = [f"dim_{i}" for i in range(emb.shape[1])]
        df = pd.DataFrame(emb, columns=cols)
        df.insert(0, "patient_id", patient_ids)
        df.to_csv(os.path.join(OUTPUT_DIR, f"{name}_embeddings.csv"), index=False)

    print("  Saved template_embeddings.npy/csv and summary_embeddings.npy/csv")

    # Step 5: Compare
    print("\n[5/5] Comparing embedding spaces...")
    compare_embeddings(patient_ids, template_embeddings, summary_embeddings)

    print("\nDone!")


def compare_embeddings(patient_ids, template_emb, summary_emb):
    """Compare the two embedding spaces and generate visualizations."""

    # Cosine similarity matrices
    template_sim = cosine_similarity(template_emb)
    summary_sim = cosine_similarity(summary_emb)

    # Cross-space: how similar is each patient's template vs summary embedding?
    cross_sim = np.array([
        cosine_similarity(template_emb[i:i+1], summary_emb[i:i+1])[0, 0]
        for i in range(len(patient_ids))
    ])
    print(f"  Cross-space similarity (template vs summary per patient):")
    print(f"    Mean: {cross_sim.mean():.4f}, Std: {cross_sim.std():.4f}")
    print(f"    Min:  {cross_sim.min():.4f}, Max: {cross_sim.max():.4f}")

    # Rank correlation of pairwise similarities
    triu_idx = np.triu_indices(len(patient_ids), k=1)
    template_pairs = template_sim[triu_idx]
    summary_pairs = summary_sim[triu_idx]
    from scipy.stats import spearmanr
    rho, pval = spearmanr(template_pairs, summary_pairs)
    print(f"  Spearman rank correlation of pairwise similarities: rho={rho:.4f}, p={pval:.2e}")

    # --- Visualization ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # 1. Similarity matrices side by side
    im0 = axes[0, 0].imshow(template_sim, cmap="RdBu_r", vmin=-0.2, vmax=1)
    axes[0, 0].set_title("Template Pairwise Similarity")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    im1 = axes[0, 1].imshow(summary_sim, cmap="RdBu_r", vmin=-0.2, vmax=1)
    axes[0, 1].set_title("LLM Summary Pairwise Similarity")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.8)

    # 2. Scatter: template vs summary pairwise similarity
    axes[0, 2].scatter(template_pairs, summary_pairs, alpha=0.15, s=8)
    axes[0, 2].plot([0, 1], [0, 1], "r--", alpha=0.5)
    axes[0, 2].set_xlabel("Template Similarity")
    axes[0, 2].set_ylabel("Summary Similarity")
    axes[0, 2].set_title(f"Pairwise Similarity Correlation (rho={rho:.3f})")
    axes[0, 2].set_xlim(0, 1)
    axes[0, 2].set_ylim(0, 1)

    # 3. t-SNE for template embeddings
    perp = min(30, len(patient_ids) - 1)
    tsne_template = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(template_emb)
    axes[1, 0].scatter(tsne_template[:, 0], tsne_template[:, 1], s=30, alpha=0.7, c="steelblue")
    for i, pid in enumerate(patient_ids):
        axes[1, 0].annotate(pid, (tsne_template[i, 0], tsne_template[i, 1]), fontsize=5, alpha=0.5)
    axes[1, 0].set_title("Template Embeddings (t-SNE)")
    axes[1, 0].set_xlabel("t-SNE 1")
    axes[1, 0].set_ylabel("t-SNE 2")

    # 4. t-SNE for summary embeddings
    tsne_summary = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(summary_emb)
    axes[1, 1].scatter(tsne_summary[:, 0], tsne_summary[:, 1], s=30, alpha=0.7, c="coral")
    for i, pid in enumerate(patient_ids):
        axes[1, 1].annotate(pid, (tsne_summary[i, 0], tsne_summary[i, 1]), fontsize=5, alpha=0.5)
    axes[1, 1].set_title("LLM Summary Embeddings (t-SNE)")
    axes[1, 1].set_xlabel("t-SNE 1")
    axes[1, 1].set_ylabel("t-SNE 2")

    # 5. Cross-space similarity histogram
    axes[1, 2].hist(cross_sim, bins=20, color="mediumpurple", edgecolor="white")
    axes[1, 2].axvline(cross_sim.mean(), color="red", linestyle="--", label=f"mean={cross_sim.mean():.3f}")
    axes[1, 2].set_xlabel("Cosine Similarity (Template vs Summary)")
    axes[1, 2].set_ylabel("Count")
    axes[1, 2].set_title("Per-Patient Cross-Space Similarity")
    axes[1, 2].legend()

    fig.suptitle("Template vs LLM Summary Patient Embeddings", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "embedding_comparison.png"), dpi=150)
    print("  Saved embedding_comparison.png")


if __name__ == "__main__":
    main()
