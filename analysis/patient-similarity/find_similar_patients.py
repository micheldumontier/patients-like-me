"""Find similar patients using RDF2Vec embeddings stored in Qdrant vector DB.

Usage:
    python find_similar_patients.py <patient_id> [--top_n 5]

Loads patient embeddings into Qdrant (if not already loaded), queries for
the top-n most similar patients, fetches their clinical events from QLever,
and generates an interactive HTML comparison visualization.
"""
import csv
import json
import sys
import argparse
import urllib.request
import urllib.parse
from collections import Counter, defaultdict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

QDRANT_URL = "http://localhost:6333"
QLEVER_URL = "http://localhost:7001"
COLLECTION = "patient_embeddings"
EMBEDDING_DIM = 200
EMBEDDINGS_CSV = "../rdf2vec/patient_embeddings.csv"

# ── Qdrant helpers ──────────────────────────────────────────────────────────

def load_embeddings_into_qdrant(client):
    """Load patient embeddings from CSV into Qdrant collection."""
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION in collections:
        info = client.get_collection(COLLECTION)
        if info.points_count > 0:
            print(f"Collection '{COLLECTION}' already loaded ({info.points_count} points).")
            return

    # Create collection
    if COLLECTION not in collections:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

    # Read CSV
    points = []
    with open(EMBEDDINGS_CSV) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            patient_id = int(row["patient_id"])
            vector = [float(row[f"dim_{d}"]) for d in range(EMBEDDING_DIM)]
            points.append(PointStruct(
                id=i,
                vector=vector,
                payload={"patient_id": patient_id},
            ))

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Loaded {len(points)} patient embeddings into Qdrant.")


def find_similar(client, patient_id, top_n=5):
    """Find top-n most similar patients to the given patient."""
    # Get the query patient's vector
    results = client.scroll(
        collection_name=COLLECTION,
        scroll_filter={"must": [{"key": "patient_id", "match": {"value": patient_id}}]},
        with_vectors=True,
        limit=1,
    )
    points = results[0]
    if not points:
        print(f"Patient {patient_id} not found in embeddings.")
        sys.exit(1)

    query_vector = points[0].vector
    # Search (top_n + 1 because the patient itself will be in results)
    hits = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=top_n + 1,
        with_payload=True,
    )

    similar = []
    for hit in hits.points:
        pid = hit.payload["patient_id"]
        if pid != patient_id:
            similar.append({"patient_id": pid, "score": hit.score})
        if len(similar) == top_n:
            break

    return similar


# ── QLever SPARQL helpers ───────────────────────────────────────────────────

def sparql(query):
    url = QLEVER_URL + "?" + urllib.parse.urlencode({"query": query})
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())["results"]["bindings"]


CATEGORY_MAP = {
    "HOSPITAL_ADMISSION": "ADMIN", "HOSPITAL_DISCHARGE": "ADMIN",
    "ICU_ADMISSION": "ADMIN", "ICU_DISCHARGE": "ADMIN",
    "TRANSFER_TO": "ADMIN", "DRG": "ADMIN", "GENDER": "ADMIN",
    "ED_REGISTRATION": "ADMIN", "ED_OUT": "ADMIN", "MEDS_BIRTH": "ADMIN",
    "INFUSION_START": "INFUSION", "INFUSION_END": "INFUSION",
    "SUBJECT_WEIGHT_AT_INFUSION": "INFUSION",
    "Blood Pressure": "VITALS", "Weight (Lbs)": "VITALS",
    "SUBJECT_FLUID_OUTPUT": "VITALS",
}

VOCAB_CATEGORY = {
    "ICD9CM": "DIAGNOSIS", "ICD10CM": "DIAGNOSIS", "ICD10PCS": "PROCEDURE",
    "RXNORM": "MEDICATION", "LNC": "LAB", "SNOMEDCT": "DIAGNOSIS",
}


def classify_event(code_string):
    """Return (category, detail) for a code string."""
    if "//" in code_string:
        parts = code_string.split("//")
        vocab = parts[0]
        detail = parts[1] if len(parts) > 1 else code_string
        category = VOCAB_CATEGORY.get(vocab, vocab)
        return category, detail
    if code_string in CATEGORY_MAP:
        return CATEGORY_MAP[code_string], code_string
    return code_string, code_string


def get_patient_profile(subject_id):
    """Fetch events and build a clinical profile summary."""
    query = f"""
    PREFIX meds: <https://teamheka.github.io/meds-ontology#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?time ?codeString ?numVal ?textVal ?parentLabel WHERE {{
      ?e a meds:Event ;
         meds:hasSubject <https://teamheka.github.io/meds-data/subject/{subject_id}> ;
         meds:codeString ?codeString ;
         meds:time ?time .
      OPTIONAL {{ ?e meds:numericValue ?numVal }}
      OPTIONAL {{ ?e meds:textValue ?textVal }}
      OPTIONAL {{
        ?e meds:hasCode ?code .
        ?code meds:parentCode ?parent .
        ?parent skos:prefLabel ?parentLabel .
      }}
    }}
    ORDER BY ?time
    """
    rows = sparql(query)

    categories = Counter()
    codes = Counter()
    labels = {}
    times = []
    for r in rows:
        cs = r["codeString"]["value"]
        cat, detail = classify_event(cs)
        categories[cat] += 1
        codes[cs] += 1
        lbl = r.get("parentLabel", {}).get("value", "")
        if lbl:
            labels[cs] = lbl
        times.append(r["time"]["value"])

    date_range = (times[0][:10], times[-1][:10]) if times else ("", "")

    return {
        "patient_id": subject_id,
        "total_events": len(rows),
        "date_range": date_range,
        "categories": dict(categories),
        "top_codes": codes.most_common(30),
        "labels": labels,
    }


# ── HTML visualization ─────────────────────────────────────────────────────

def generate_comparison_html(query_patient, similar_patients, profiles, output_path):
    """Generate interactive HTML comparing similar patients."""

    # Build data for the visualization
    all_patient_ids = [query_patient["patient_id"]] + [s["patient_id"] for s in similar_patients]
    scores = {query_patient["patient_id"]: 1.0}
    for s in similar_patients:
        scores[s["patient_id"]] = s["score"]

    # Collect all categories across patients
    all_categories = set()
    for pid in all_patient_ids:
        if pid in profiles:
            all_categories.update(profiles[pid]["categories"].keys())
    all_categories = sorted(all_categories)

    # Build shared/unique code analysis
    # Codes present in query patient
    query_codes = set(c for c, _ in profiles[query_patient["patient_id"]]["top_codes"])
    similar_code_sets = {}
    for s in similar_patients:
        pid = s["patient_id"]
        if pid in profiles:
            similar_code_sets[pid] = set(c for c, _ in profiles[pid]["top_codes"])

    profiles_json = json.dumps({
        str(pid): profiles[pid] for pid in all_patient_ids if pid in profiles
    }, default=str)
    scores_json = json.dumps({str(k): v for k, v in scores.items()})
    categories_json = json.dumps(all_categories)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Patient Similarity: {query_patient['patient_id']}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f172a; color: #e2e8f0; padding: 24px; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 8px; color: #38bdf8; }}
  h2 {{ font-size: 1.1rem; margin: 16px 0 8px; color: #7dd3fc; }}
  .subtitle {{ color: #94a3b8; font-size: 0.9rem; margin-bottom: 20px; }}

  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }}
  @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}

  .card {{ background: #1e293b; border-radius: 12px; padding: 18px; border: 1px solid #334155; }}
  .card.query {{ border-color: #38bdf8; }}
  .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
  .card-header h3 {{ font-size: 1rem; }}
  .badge {{ background: #0ea5e9; color: #fff; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; }}
  .badge.sim {{ background: #8b5cf6; }}
  .stat {{ display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #334155; font-size: 0.85rem; }}
  .stat:last-child {{ border-bottom: none; }}
  .stat-label {{ color: #94a3b8; }}

  /* Radar chart */
  .radar-container {{ display: flex; justify-content: center; margin: 20px 0; }}
  canvas {{ background: #1e293b; border-radius: 12px; }}

  /* Category heatmap */
  .heatmap {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
  .heatmap th {{ text-align: left; padding: 6px 10px; color: #94a3b8; border-bottom: 1px solid #334155; }}
  .heatmap td {{ padding: 6px 10px; text-align: center; border-bottom: 1px solid #1e293b; }}
  .heatmap tr:hover {{ background: #1e293b; }}
  .heatmap .pid-col {{ text-align: left; color: #7dd3fc; font-weight: 600; }}

  /* Shared codes */
  .code-chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }}
  .chip {{ background: #334155; padding: 3px 10px; border-radius: 8px; font-size: 0.75rem; }}
  .chip.shared {{ background: #166534; color: #bbf7d0; }}
  .chip.unique {{ background: #7c2d12; color: #fed7aa; }}

  /* Bar chart */
  .bar-row {{ display: flex; align-items: center; margin: 3px 0; font-size: 0.8rem; }}
  .bar-label {{ width: 80px; text-align: right; padding-right: 8px; color: #94a3b8; }}
  .bar-track {{ flex: 1; height: 18px; background: #334155; border-radius: 4px; overflow: hidden; position: relative; }}
  .bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.5s; }}
  .bar-value {{ position: absolute; right: 6px; top: 1px; font-size: 0.7rem; color: #e2e8f0; }}

  .section {{ margin: 24px 0; }}
  .legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 8px 0; font-size: 0.8rem; }}
  .legend-item {{ display: flex; align-items: center; gap: 4px; }}
  .legend-swatch {{ width: 12px; height: 12px; border-radius: 3px; }}
</style>
</head>
<body>

<h1>Patient Similarity Analysis</h1>
<p class="subtitle">Query patient: <strong>{query_patient['patient_id']}</strong> &mdash;
   Top {len(similar_patients)} similar patients by RDF2Vec cosine similarity</p>

<div class="grid" id="patient-cards"></div>

<div class="section">
  <h2>Similarity Scores</h2>
  <div id="sim-bars"></div>
</div>

<div class="section">
  <h2>Event Category Comparison</h2>
  <div style="overflow-x:auto;">
    <table class="heatmap" id="heatmap"></table>
  </div>
</div>

<div class="section">
  <h2>Category Distribution (Radar)</h2>
  <div class="radar-container">
    <canvas id="radar" width="500" height="500"></canvas>
  </div>
  <div class="legend" id="radar-legend"></div>
</div>

<div class="section">
  <h2>Shared vs Unique Clinical Codes</h2>
  <p style="font-size:0.85rem;color:#94a3b8;margin-bottom:8px;">
    Comparing top codes of each similar patient against the query patient.</p>
  <div id="code-comparison"></div>
</div>

<script>
const profiles = {profiles_json};
const scores = {scores_json};
const categories = {categories_json};
const queryPid = "{query_patient['patient_id']}";
const COLORS = ['#38bdf8','#a78bfa','#f472b6','#34d399','#fb923c','#facc15','#f87171','#22d3ee'];

function esc(s) {{ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }}
function hexToRgba(hex, alpha) {{
  const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
  return `rgba(${{r}},${{g}},${{b}},${{alpha}})`;
}}

// ── Patient cards ──
const cardsDiv = document.getElementById('patient-cards');
const allPids = [queryPid, ...Object.keys(scores).filter(p => p !== queryPid)];
allPids.forEach((pid, i) => {{
  const p = profiles[pid];
  if (!p) return;
  const isQuery = pid === queryPid;
  const card = document.createElement('div');
  card.className = 'card' + (isQuery ? ' query' : '');
  const score = isQuery ? '' : `<span class="badge sim">Similarity: ${{(scores[pid]*100).toFixed(1)}}%</span>`;
  card.innerHTML = `
    <div class="card-header">
      <h3>Patient ${{pid}}</h3>
      ${{isQuery ? '<span class="badge">Query Patient</span>' : score}}
    </div>
    <div class="stat"><span class="stat-label">Total Events</span><span>${{p.total_events}}</span></div>
    <div class="stat"><span class="stat-label">Date Range</span><span>${{p.date_range[0]}} to ${{p.date_range[1]}}</span></div>
    <div class="stat"><span class="stat-label">Categories</span><span>${{Object.keys(p.categories).length}}</span></div>
    <div class="stat"><span class="stat-label">Unique Codes</span><span>${{p.top_codes.length}}</span></div>
  `;
  cardsDiv.appendChild(card);
}});

// ── Similarity bars ──
const barsDiv = document.getElementById('sim-bars');
allPids.filter(p => p !== queryPid).forEach((pid, i) => {{
  const pct = (scores[pid] * 100).toFixed(1);
  const row = document.createElement('div');
  row.className = 'bar-row';
  row.innerHTML = `
    <span class="bar-label">${{pid}}</span>
    <div class="bar-track">
      <div class="bar-fill" style="width:${{pct}}%;background:${{COLORS[(i+1)%COLORS.length]}}"></div>
      <span class="bar-value">${{pct}}%</span>
    </div>
  `;
  barsDiv.appendChild(row);
}});

// ── Heatmap table ──
const heatmap = document.getElementById('heatmap');
let hdr = '<tr><th>Patient</th>';
categories.forEach(c => hdr += `<th>${{c}}</th>`);
hdr += '<th>Total</th></tr>';
heatmap.innerHTML = hdr;

// Find max per category for color scaling
const maxPerCat = {{}};
categories.forEach(c => {{
  maxPerCat[c] = Math.max(...allPids.map(pid => (profiles[pid]?.categories[c]) || 0));
}});

allPids.forEach((pid, i) => {{
  const p = profiles[pid];
  if (!p) return;
  let row = `<tr><td class="pid-col">${{pid}}${{pid===queryPid?' *':''}}</td>`;
  categories.forEach(c => {{
    const val = p.categories[c] || 0;
    const intensity = maxPerCat[c] > 0 ? val / maxPerCat[c] : 0;
    const bg = `rgba(56,189,248,${{intensity * 0.6}})`;
    row += `<td style="background:${{bg}}">${{val || ''}}</td>`;
  }});
  row += `<td style="font-weight:600">${{p.total_events}}</td></tr>`;
  heatmap.innerHTML += row;
}});

// ── Radar chart ──
function drawRadar() {{
  const canvas = document.getElementById('radar');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const cx = W/2, cy = H/2, R = Math.min(W,H)/2 - 60;
  const n = categories.length;
  if (n < 3) return;

  // Normalize: per-category max across patients
  const maxVals = {{}};
  categories.forEach(c => {{
    maxVals[c] = Math.max(1, ...allPids.map(pid => profiles[pid]?.categories[c] || 0));
  }});

  ctx.clearRect(0, 0, W, H);

  // Draw grid
  for (let ring = 1; ring <= 4; ring++) {{
    ctx.beginPath();
    const r = R * ring / 4;
    for (let i = 0; i <= n; i++) {{
      const angle = (2 * Math.PI * i / n) - Math.PI/2;
      const x = cx + r * Math.cos(angle);
      const y = cy + r * Math.sin(angle);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }}
    ctx.strokeStyle = '#334155';
    ctx.stroke();
  }}

  // Draw axes & labels
  categories.forEach((c, i) => {{
    const angle = (2 * Math.PI * i / n) - Math.PI/2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + R * Math.cos(angle), cy + R * Math.sin(angle));
    ctx.strokeStyle = '#475569';
    ctx.stroke();
    const lx = cx + (R + 30) * Math.cos(angle);
    const ly = cy + (R + 30) * Math.sin(angle);
    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(c, lx, ly);
  }});

  // Draw patient polygons
  allPids.forEach((pid, pi) => {{
    const p = profiles[pid];
    if (!p) return;
    ctx.beginPath();
    categories.forEach((c, i) => {{
      const val = (p.categories[c] || 0) / maxVals[c];
      const angle = (2 * Math.PI * i / n) - Math.PI/2;
      const x = cx + R * val * Math.cos(angle);
      const y = cy + R * val * Math.sin(angle);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }});
    ctx.closePath();
    const color = COLORS[pi % COLORS.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = pid === queryPid ? 3 : 1.5;
    ctx.stroke();
    ctx.fillStyle = hexToRgba(color, 0.1);
    ctx.fill();
  }});

  // Legend
  const legendDiv = document.getElementById('radar-legend');
  allPids.forEach((pid, i) => {{
    const item = document.createElement('span');
    item.className = 'legend-item';
    const swatch = `<span class="legend-swatch" style="background:${{COLORS[i%COLORS.length]}}"></span>`;
    item.innerHTML = `${{swatch}} ${{pid}}${{pid===queryPid?' (query)':''}}`;
    legendDiv.appendChild(item);
  }});
}}
drawRadar();

// ── Code comparison ──
const compDiv = document.getElementById('code-comparison');
const queryCodes = new Set(profiles[queryPid]?.top_codes.map(c => c[0]) || []);

allPids.filter(p => p !== queryPid).forEach((pid, i) => {{
  const p = profiles[pid];
  if (!p) return;
  const simCodes = new Set(p.top_codes.map(c => c[0]));
  const shared = [...queryCodes].filter(c => simCodes.has(c));
  const uniqueToQuery = [...queryCodes].filter(c => !simCodes.has(c));
  const uniqueToSim = [...simCodes].filter(c => !queryCodes.has(c));

  const section = document.createElement('div');
  section.className = 'card';
  section.style.marginBottom = '12px';

  function codeLabel(code) {{
    const lbl = profiles[queryPid]?.labels[code] || profiles[pid]?.labels[code] || '';
    return lbl ? `${{esc(code)}} (${{esc(lbl)}})` : esc(code);
  }}

  section.innerHTML = `
    <div class="card-header">
      <h3>vs Patient ${{pid}}</h3>
      <span class="badge sim">Similarity: ${{(scores[pid]*100).toFixed(1)}}%</span>
    </div>
    <p style="font-size:0.8rem;color:#94a3b8;margin-bottom:6px;">
      Shared: ${{shared.length}} &bull; Unique to query: ${{uniqueToQuery.length}} &bull; Unique to ${{pid}}: ${{uniqueToSim.length}}</p>
    <div style="margin-bottom:6px"><strong style="font-size:0.8rem;color:#34d399;">Shared codes:</strong>
      <div class="code-chips">${{shared.slice(0,15).map(c => `<span class="chip shared">${{codeLabel(c)}}</span>`).join('')}}</div>
    </div>
    <div style="margin-bottom:6px"><strong style="font-size:0.8rem;color:#fb923c;">Only in query patient:</strong>
      <div class="code-chips">${{uniqueToQuery.slice(0,10).map(c => `<span class="chip unique">${{codeLabel(c)}}</span>`).join('')}}</div>
    </div>
    <div><strong style="font-size:0.8rem;color:#a78bfa;">Only in ${{pid}}:</strong>
      <div class="code-chips">${{uniqueToSim.slice(0,10).map(c => `<span class="chip unique">${{codeLabel(c)}}</span>`).join('')}}</div>
    </div>
  `;
  compDiv.appendChild(section);
}});
</script>
</body>
</html>"""
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Visualization saved to {output_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Find similar patients via RDF2Vec embeddings")
    parser.add_argument("patient_id", type=int, help="Query patient ID")
    parser.add_argument("--top_n", type=int, default=5, help="Number of similar patients to find")
    args = parser.parse_args()

    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL)

    print("Loading embeddings into Qdrant...")
    load_embeddings_into_qdrant(client)

    print(f"\nFinding top {args.top_n} similar patients to {args.patient_id}...")
    similar = find_similar(client, args.patient_id, args.top_n)

    print("\nSimilar patients:")
    for s in similar:
        print(f"  Patient {s['patient_id']:>10}  cosine similarity: {s['score']:.4f}")

    # Fetch clinical profiles
    all_pids = [args.patient_id] + [s["patient_id"] for s in similar]
    profiles = {}
    for pid in all_pids:
        print(f"Fetching clinical profile for patient {pid}...")
        profiles[pid] = get_patient_profile(pid)

    query_patient = {"patient_id": args.patient_id}
    output_path = f"similarity_{args.patient_id}.html"
    generate_comparison_html(query_patient, similar, profiles, output_path)


if __name__ == "__main__":
    main()
