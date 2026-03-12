# Patients Like Me

A clinical data analysis platform that converts MIMIC-IV demo data to RDF, loads it alongside medical ontologies into a QLever triplestore, and provides interactive visualizations including patient timelines, RDF2Vec-based patient similarity, and AI-powered clinical summaries.

## Architecture

```
MIMIC-IV Demo (MEDS format)
        |
        v
    meds2rdf --> RDF (Turtle)
        |
        v
+---------------------------+     +-------------------+
|  QLever Triplestore       |     |  Qdrant Vector DB  |
|  (port 7001)              |     |  (port 6333)       |
|  - Patient events         |     |  - RDF2Vec          |
|  - ICD9CM, ICD10CM        |     |    embeddings       |
|  - RXNORM, LOINC          |     |                     |
|  - SNOMED CT              |     |                     |
+-----------+---------------+     +---------+-----------+
            |                               |
            v                               v
   +--------------------------------------------+
   |       website/ (port 8025/mimic-miner/)     |
   |  - Landing page with service links          |
   |  - Patient timelines with AI summaries      |
   |  - Patient similarity comparisons           |
   +--------------------------------------------+
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

## Configuration

Copy the environment template and fill in your credentials:

```bash
cp .env.example .env
```

| Variable | Description |
|----------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint |
| `AZURE_OPENAI_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_DEPLOYMENT` | Model deployment name (e.g., `gpt-5.2`) |
| `AZURE_OPENAI_API_VERSION` | API version (e.g., `2025-01-01-preview`) |

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

- SPARQL endpoint: `http://localhost:6335`
- Access token: `mimic_demo_token`

#### QLever UI

Start the web-based SPARQL query interface:

```bash
cd qlever
qlever ui
```

- UI: `http://localhost:6336/default`
- Includes autocomplete for prefixes (meds, ICD9CM, ICD10CM, RXNORM, SNOMEDCT, LOINC, etc.) and example queries
- Configuration is in `qlever/Qleverfile-ui.yml`

### 4. Start Qdrant Vector Database

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

- REST API: `http://localhost:6333`
- Dashboard: `http://localhost:6333/dashboard`

## Website

All generated visualizations are served from the `website/` directory under the `/mimic-miner/` path prefix. Start the web server:

```bash
python website/serve.py
```

Browse the platform at `http://localhost:9000/mimic-miner/`. The landing page provides links to all analyses and data services.

### Generating Content

#### Patient Timelines

Generate interactive timeline visualizations with category filtering, overview+detail navigation, and AI-powered clinical summaries (via Azure OpenAI GPT-5.2).

```bash
# Generate a single patient timeline
python analysis/patient-timeline/patient_timeline.py <patient_id>

# Generate all patient timelines + index page
python analysis/patient-timeline/patient_timeline.py
```

Output goes to `website/patient-timeline/`. View at `http://localhost:9000/mimic-miner/patient-timeline/`.

#### Patient Similarity

Find the most similar patients using RDF2Vec embeddings stored in Qdrant, then generate a comparison visualization with similarity scores, event category heatmap, radar chart, and shared/unique clinical codes.

```bash
python analysis/patient-similarity/find_similar_patients.py <patient_id> --top_n 5
```

Output goes to `website/patient-similarity/`. View at `http://localhost:9000/mimic-miner/patient-similarity/`.

#### RDF2Vec Embeddings

Compute graph embeddings for patients and visualize clusters.

```bash
cd analysis/rdf2vec
python run_rdf2vec.py              # Generate embeddings
python cluster_and_visualize.py    # t-SNE/UMAP clustering
python update_qdrant.py            # Load into Qdrant
```

#### Events per Patient

Histogram of event counts across the patient population.

```bash
cd analysis/events-per-patient
python plot_events_histogram.py
```

## Project Structure

```
patients-like-me/
├── .env.example                   # Environment variable template
├── website/                       # Generated website (served on port 8080)
│   ├── index.html                 # Landing page with links to all services
│   ├── patient-timeline/          # Generated patient timeline HTML files
│   │   ├── index.html             # Patient listing page
│   │   └── data/                  # Individual timeline files
│   └── patient-similarity/        # Generated similarity comparison files
│       └── index.html             # Similarity listing page
├── analysis/                      # Analysis scripts
│   ├── patient-timeline/          # Timeline generation script
│   ├── patient-similarity/        # Similarity generation script
│   ├── rdf2vec/                   # Graph embedding generation
│   └── events-per-patient/        # Population-level statistics
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
│   ├── Qleverfile
│   └── Qleverfile-ui.yml          # QLever UI configuration
├── iri_redirect.py                # IRI -> QLever UI redirect service
└── requirements.txt
```

## Services

| Service       | Port | URL                               |
|---------------|------|-----------------------------------|
| Website       | 9000  | `http://localhost:9000/mimic-miner/` |
| QLever        | 6335 | `http://localhost:6335`            |
| QLever UI     | 6336 | `http://localhost:6336/default`    |
| IRI Redirect  | 6337 | `http://localhost:6337`            |
| Qdrant        | 6333 | `http://localhost:6333/dashboard`  |

### IRI Redirect Service

Resolves RDF IRIs to QLever UI entity views. Useful for making IRIs in results browsable.

```bash
python iri_redirect.py
```

Example redirects:

| Local URL | Resolves IRI |
|-----------|-------------|
| `http://localhost:6337/meds-data/subject/10000032` | `https://teamheka.github.io/meds-data/subject/10000032` |
| `http://localhost:6337/ICD9CM/038.0` | `http://purl.bioontology.org/ontology/ICD9CM/038.0` |
| `http://localhost:6337/RXNORM/161` | `http://purl.bioontology.org/ontology/RXNORM/161` |
| `http://localhost:6337/?uri=<any_iri>` | Any IRI |
