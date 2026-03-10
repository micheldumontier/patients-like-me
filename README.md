# MIMIC-Miner

A clinical data analysis platform that converts MIMIC-IV demo data to RDF, loads it alongside medical ontologies into a QLever triplestore, and provides interactive visualizations including patient timelines, RDF2Vec-based patient similarity, and AI-powered clinical summaries.

## Architecture

```
MIMIC-IV Demo (MEDS format)
        │
        ▼
    meds2rdf ──► RDF (Turtle)
        │
        ▼
┌──────────────────────────┐     ┌──────────────────┐
│  QLever Triplestore      │     │  Qdrant Vector DB │
│  (port 7001)             │     │  (port 6333)      │
│  - Patient events        │     │  - RDF2Vec         │
│  - ICD9CM, ICD10CM       │     │    embeddings      │
│  - RXNORM, LOINC         │     │                    │
│  - SNOMED CT             │     │                    │
└──────────┬───────────────┘     └────────┬───────────┘
           │                              │
           ▼                              ▼
   ┌───────────────────────────────────────────┐
   │         Analysis & Visualization          │
   │  (served on port 8024)                    │
   │  - Patient timelines with AI summaries    │
   │  - Patient similarity comparisons         │
   │  - RDF2Vec clustering                     │
   └───────────────────────────────────────────┘
```

## Prerequisites

- Python 3.12+
- Docker
- [QLever](https://github.com/ad-freiburg/qlever) (`pip install qlever`)

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install qdrant-client
```

## Data Pipeline

### 1. Convert MIMIC-IV Demo to RDF

Uses [meds2rdf](https://github.com/TeamHeKA/meds2rdf) to convert MEDS-format data to RDF with BioPortal-canonical IRIs.

```bash
meds_to_rdf --input ~/data/physionet.org/files/mimic-iv-demo-meds/2.0.3 --output data/mimic-iv-demo.ttl
```

### 2. Prepare Medical Ontologies

The following ontologies are loaded alongside patient data for semantic enrichment:

| Ontology  | Source    | File            | Triples |
|-----------|-----------|-----------------|---------|
| ICD9CM    | BioPortal | `ICD9CM.ttl`    | ~168K   |
| ICD10CM   | BioPortal | `ICD10CM.ttl`   | ~915K   |
| RXNORM    | BioPortal | `RXNORM.ttl`    | ~2.7M   |
| LOINC     | BioPortal | `LNC.ttl`       | ~8.2M   |
| SNOMED CT | NLM RF2   | `SNOMEDCT.ttl`  | ~537K concepts |

SNOMED CT is converted from RF2 format using the included script:

```bash
python data_scripts/rf2_to_ttl.py data/snomed-ct-us.zip data/SNOMEDCT.ttl
```

All ontologies use BioPortal PURL IRIs (e.g., `http://purl.bioontology.org/ontology/SNOMEDCT/`).

### 3. Load into QLever

```bash
cd qlever
# Copy/symlink all .ttl files into this directory
qlever index --parse-parallel false
qlever start
```

The `--parse-parallel false` flag is required due to multiline string literals in the data.

- SPARQL endpoint: `http://localhost:7001`
- Access token: `mimic_demo_token`

### 4. Start Qdrant Vector Database

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

- REST API: `http://localhost:6333`
- Dashboard: `http://localhost:6333/dashboard`

## Analysis

Start the analysis web server:

```bash
cd analysis
python -m http.server 8024
```

Browse all analyses at `http://localhost:8024/`.

### Patient Timeline

Interactive timeline visualization with overview+detail navigation, category filtering, and AI-powered clinical summaries (via Azure OpenAI GPT-5.2).

```bash
cd analysis/patient-timeline
python patient_timeline.py <patient_id>
# View at http://localhost:8024/patient-timeline/timeline_<patient_id>.html
```

### Patient Similarity

Finds the most similar patients using RDF2Vec embeddings stored in Qdrant, then generates a comparison visualization with:
- Similarity score bars
- Event category heatmap
- Radar chart comparing clinical profiles
- Shared vs unique clinical codes

```bash
cd analysis/patient-similarity
python find_similar_patients.py <patient_id> --top_n 5
# View at http://localhost:8024/patient-similarity/similarity_<patient_id>.html
```

### RDF2Vec Embeddings

Computes graph embeddings for patients and visualizes clusters.

```bash
cd analysis/rdf2vec
python run_rdf2vec.py              # Generate embeddings
python cluster_and_visualize.py    # t-SNE/UMAP clustering
```

### Events per Patient

Histogram of event counts across the patient population.

```bash
cd analysis/events-per-patient
python plot_events_histogram.py
```

## Project Structure

```
mimic-miner/
├── data/                          # RDF data files
│   ├── mimic-iv-demo.ttl          # Patient events
│   ├── ICD9CM.ttl                 # Ontologies
│   ├── ICD10CM.ttl
│   ├── RXNORM.ttl
│   ├── LNC.ttl
│   ├── SNOMEDCT.ttl
│   └── snomed-ct-us.zip           # SNOMED CT RF2 source
├── data_scripts/
│   └── rf2_to_ttl.py              # SNOMED CT RF2 to Turtle converter
├── qlever/                        # QLever triplestore config + index
│   └── Qleverfile
├── analysis/                      # All analyses (served on port 8024)
│   ├── patient-timeline/          # Interactive patient timelines
│   ├── patient-similarity/        # Vector similarity comparisons
│   ├── rdf2vec/                   # Graph embedding generation
│   └── events-per-patient/        # Population-level statistics
└── requirements.txt
```

## Services

| Service  | Port | URL                              |
|----------|------|----------------------------------|
| QLever   | 7001 | `http://localhost:7001`           |
| Qdrant   | 6333 | `http://localhost:6333/dashboard` |
| Analysis | 8024 | `http://localhost:8024`           |
