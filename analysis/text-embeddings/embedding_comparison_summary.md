# Patient Embedding Comparison Summary

## Approaches

### 1. RDF2Vec (200 dimensions)
- Custom implementation with in-memory adjacency lists
- Random walks with **reverse traversal** over MIMIC-IV + ICD9CM graph
- 100 walks per patient, depth 4, Skip-gram Word2Vec
- Skips literal predicates (numericValue, textValue, time, codeString, etc.) and prov:wasDerivedFrom
- Reverse walks are essential: patient URIs have only 2 forward edges (rdf:type, subjectId) but ~1500 reverse edges (events via hasSubject)

### 2. Template Text Embedding (3072 dimensions)
- SPARQL queries extract diagnoses, medications, labs, procedures per patient
- Structured template verbalization (avg 5230 chars per patient)
- Embedded with Azure OpenAI text-embedding-3-large

### 3. LLM Summary Embedding (3072 dimensions)
- Template text enriched by GPT-5.2 into a clinical narrative (avg 1020 chars)
- Prompt: "Write a concise clinical summary paragraph capturing key clinical picture, condition clusters, and notable patterns"
- Embedded with Azure OpenAI text-embedding-3-large

## Results

### Pairwise Similarity Rank Correlation (Spearman)

| Comparison              | rho    | p-value   |
|-------------------------|--------|-----------|
| RDF2Vec vs Template     | 0.061  | 2.0e-05   |
| RDF2Vec vs LLM Summary  | -0.004 | 7.8e-01   |
| Template vs LLM Summary | 0.353  | 9.2e-145  |

### Nearest-Neighbor Agreement (Top-5)

| Comparison              | Avg Overlap |
|-------------------------|-------------|
| RDF2Vec vs Template     | 9.4%        |
| RDF2Vec vs LLM Summary  | 7.2%        |
| Template vs LLM Summary | 28.2%       |

### Cross-Space Similarity (Template vs LLM Summary, per patient)

- Mean: 0.734, Std: 0.123
- Range: 0.235 - 0.878

## Interpretation

- **RDF2Vec** captures graph topology: which patients share structural neighborhoods through common codes and the ICD9CM hierarchy. The similarity matrix shows visible block structure with reverse walks enabled.
- **Template embeddings** encode the full clinical profile as structured text. The text encoder benefits from understanding medical terminology but processes it as a flat list.
- **LLM Summary embeddings** capture high-level clinical phenotypes. The LLM distills raw data into the most salient clinical patterns, producing tighter and more distinct clusters in t-SNE.
- The two text approaches agree moderately (rho=0.35, 28% NN overlap). Neither correlates with RDF2Vec (~0 rho, ~5-9% NN overlap at chance level).
- The three approaches are **complementary**: graph structure, detailed clinical profile, and clinical phenotype summarization each capture different aspects of patient similarity.

## Files

| File | Description |
|------|-------------|
| `template_embeddings.npy/csv` | Template text embeddings (100 x 3072) |
| `summary_embeddings.npy/csv` | LLM summary text embeddings (100 x 3072) |
| `patient_templates.json` | Structured text verbalizations per patient |
| `patient_summaries.json` | LLM-generated clinical summaries per patient |
| `embedding_comparison.png` | Template vs LLM Summary comparison (6 panels) |
| `three_way_comparison.png` | Three-way comparison (9 panels) |
| `../rdf2vec/patient_embeddings.csv` | RDF2Vec embeddings (100 x 200) |
