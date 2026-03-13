# Patients Like Me

A clinical data analysis platform that converts MIMIC-IV demo data to RDF, loads it alongside medical ontologies into a QLever triplestore, and computes patient embeddings using four complementary approaches (RDF2Vec, text embeddings, GNN, and multi-view fusion). Includes interactive visualizations for patient timelines, similarity comparisons, and AI-powered clinical summaries.

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

### 1. Obtain MIMIC-IV Demo Data

The source data is [MIMIC-IV Clinical Database Demo](https://physionet.org/content/mimic-iv-demo/), a freely available subset of 100 de-identified patients from the Beth Israel Deaconess Medical Center. It includes hospital admissions, ICU stays, diagnoses, procedures, lab results, medications, and vitals.

```bash
# Download via PhysioNet (no credentialing required for the demo)
wget -r -N -c -np https://physionet.org/files/mimiciv/3.1/
```

### 2. Convert MIMIC-IV to MEDS Format

[MEDS](https://github.com/Medical-Event-Data-Standard/meds) (Medical Event Data Standard) is a standardized format for representing patient event streams as flat Parquet files. Each row is an event with a `subject_id`, `time`, `code`, and optional `numeric_value`/`text_value` fields. Code strings follow the pattern `CATEGORY//VOCABULARY//CODE` (e.g., `DIAGNOSIS//ICD//9//4280`, `LAB//51301//mL`).

The conversion uses [MIMIC-IV-MEDS](https://pypi.org/project/MIMIC-IV-MEDS/), an ETL package that extracts events from the relational MIMIC-IV tables, maps diagnoses to ICD codes, labs to LOINC, and produces a MEDS-compliant Parquet dataset.

```bash
pip install MIMIC-IV-MEDS

export N_WORKERS=10
# Convert MIMIC-IV Demo to MEDS format
MEDS_extract-MIMIC_IV \
  input_dir=~/data/physionet.org/files/mimiciv/3.1/ \
  output_dir=~/data/mimic-iv-meds-dem
```

Alternatively, a pre-built MEDS version of the demo is available at [mimic-iv-demo-meds](https://physionet.org/content/mimic-iv-demo-meds/):

```bash
wget -r -N -c -np https://physionet.org/files/mimic-iv-demo-meds/0.0.1/
```

The resulting dataset structure:

```
mimic-iv-demo-meds/
├── metadata/
│   ├── dataset.json          # Dataset metadata (MEDS version, ETL info)
│   ├── codes.parquet         # 2,661 unique codes with descriptions and parent mappings
│   └── subject_splits.parquet
└── data/
    ├── train/                # 80 patients (~804K events)
    ├── tuning/               # 10 patients
    └── held_out/             # 10 patients
```

Each event Parquet file contains columns: `subject_id`, `time`, `code`, `numeric_value`, `text_value`, plus MIMIC-specific fields (`hadm_id`, `icustay_id`, `route`, `unit`, etc.).

### 3. Convert MEDS to RDF

[meds2rdf](https://github.com/TeamHeKA/meds2rdf) converts the MEDS Parquet dataset into RDF (Turtle format) using the [MEDS Ontology](https://teamheka.github.io/meds-ontology). Each patient event becomes an `meds:Event` instance linked to a subject, timestamp, code string, and optional numeric/text values. Code strings are mapped to ontology IRIs using BioPortal PURLs.

```python
from meds2rdf import MedsRDFConverter

converter = MedsRDFConverter("~/data/physionet.org/files/mimic-iv-demo-meds/0.0.1")
converter.convert(include_dataset_metadata=True, include_codes=True)
converter.to_turtle("data/mimic-iv-demo.ttl")
```

Key RDF namespaces produced:

| Prefix | Namespace | Content |
|--------|-----------|---------|
| `meds:` | `https://teamheka.github.io/meds-ontology#` | Ontology classes/properties (Event, hasSubject, codeString, etc.) |
| `meds-data:` | `https://teamheka.github.io/meds-data/` | Patient and event instances |
| `icd9cm:` | `http://purl.bioontology.org/ontology/ICD9CM/` | ICD-9-CM diagnosis codes |
| `icd10cm:` | `http://purl.bioontology.org/ontology/ICD10CM/` | ICD-10-CM diagnosis codes |
| `rxnorm:` | `http://purl.bioontology.org/ontology/RXNORM/` | Medication codes |
| `lnc:` | `http://purl.bioontology.org/ontology/LNC/` | LOINC lab codes |
| `snomedct:` | `http://purl.bioontology.org/ontology/SNOMEDCT/` | SNOMED CT concepts |

Example RDF for a single event:

```turtle
meds-data:event/abc123 a meds:Event ;
    meds:hasSubject meds-data:subject/10000032 ;
    meds:time "2180-03-23T11:51:00"^^xsd:dateTime ;
    meds:codeString "DIAGNOSIS//ICD//9//4280" ;
    meds:hasCode [ meds:parentCode icd9cm:428.0 ] .
```

### 4. Prepare Medical Ontologies

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

### 5. Load into QLever

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

### 6. Start Qdrant Vector Database

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

### Patient Embeddings

Four complementary embedding approaches capture different aspects of patient similarity, plus a fusion model that combines all four into a shared space.

#### 1. RDF2Vec (200 dimensions)

Graph-based embeddings using random walks over the MIMIC-IV RDF graph with all five medical ontologies. Custom implementation with in-memory adjacency lists for fast walks.

- **Reverse walks** are essential: patient URIs have only 2 forward edges but ~1,500 reverse edges (events via `meds:hasSubject`)
- Loads MIMIC-IV data + ICD9CM, ICD10CM, RXNORM, LOINC, and SNOMED CT ontologies
- **Lab value discretization**: 405K events with numeric values binned into quartiles per code (817 codes), creating shared graph nodes that connect patients with similar lab profiles
- 100 walks per patient, depth 4, Skip-gram Word2Vec

```bash
cd analysis/rdf2vec
python run_rdf2vec.py              # Generate embeddings
python cluster_and_visualize.py    # t-SNE/UMAP clustering
python update_qdrant.py            # Load into Qdrant
```

#### 2. Template Text Embeddings (3,072 dimensions)

SPARQL queries extract diagnoses, medications, labs, and procedures per patient, then verbalize them into structured text templates (avg 5,230 chars). Embedded with Azure OpenAI `text-embedding-3-large`.

#### 3. LLM Summary Embeddings (3,072 dimensions)

Template text enriched by GPT-5.2 into concise clinical narrative summaries (avg 1,020 chars), then embedded with `text-embedding-3-large`. Produces tighter and more distinct clusters than raw templates.

```bash
cd analysis/text-embeddings
python generate_embeddings.py      # Generate template + summary embeddings
```

#### 4. GNN / GraphSAGE Embeddings (128 dimensions)

A Graph Neural Network trained on a heterogeneous patient-code graph using PyTorch Geometric. Uses a GraphSAGE encoder with Barlow Twins + contrastive loss to learn embeddings through message passing.

- **Graph**: 100 patients, 4,916 code nodes, 2,571 lab bin nodes, 130K edges (undirected)
- **Edge types**: patient↔code (13.7K), code↔code hierarchy via `rdfs:subClassOf` (5K), patient↔lab bin (46K)
- Ontology hierarchy edges collected via BFS 3 hops from referenced codes
- Type embeddings + edge dropout augmentation for two views

```bash
cd analysis/gnn-embeddings
python train_gnn_embeddings.py     # Train GNN and generate embeddings
```

#### 5. Fused Embeddings (128 dimensions)

Multi-view contrastive learning (NT-Xent loss) across all pairs of the four approaches. Four projection heads map each view into a shared 128-dimensional space; the fused embedding is the normalized mean.

```bash
cd analysis/fused-embeddings
python train_fused_embeddings.py   # Train fusion model
```

#### Embedding Comparison

Compare all four approaches side-by-side with Spearman rank correlations, top-5 nearest-neighbor overlap, similarity matrices, and t-SNE projections.

```bash
cd analysis/text-embeddings
python compare_all_embeddings.py   # Generate four-way comparison
```

| Pair                    | Spearman rho | Top-5 NN Overlap |
|-------------------------|-------------|-----------------|
| RDF2Vec vs Template     | 0.053       | 7.2%            |
| RDF2Vec vs LLM Summary  | -0.021      | 5.0%            |
| RDF2Vec vs GNN          | 0.140       | 5.2%            |
| Template vs LLM Summary | **0.353**   | **28.2%**       |
| Template vs GNN         | **0.334**   | **18.4%**       |
| LLM Summary vs GNN     | 0.173       | 15.2%           |

The four methods are largely complementary: RDF2Vec captures graph topology, GNN captures message-passing structure, Template encodes the full clinical profile, and LLM Summary distills high-level clinical phenotypes. The fused embedding successfully finds a shared space that respects all four perspectives, with higher agreement to each individual method than any pair of individual methods has with each other.

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
│   ├── rdf2vec/                   # RDF2Vec graph embeddings (200d)
│   ├── text-embeddings/           # Template + LLM summary embeddings (3072d)
│   ├── gnn-embeddings/            # GraphSAGE GNN embeddings (128d)
│   ├── fused-embeddings/          # Multi-view fused embeddings (128d)
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
