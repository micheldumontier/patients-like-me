"""Generate an interactive patient event timeline from the QLever SPARQL endpoint."""
import json
import urllib.request
import urllib.parse
import sys
import os

# Load .env from project root
ENV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
if os.path.exists(ENV_PATH):
    with open(ENV_PATH) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

ENDPOINT = "http://localhost:6335"


def sparql(query):
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query})
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())["results"]["bindings"]


def get_patient_events(subject_id):
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
    return sparql(query)


CATEGORY_MAP = {
    # Administrative / encounter events
    "HOSPITAL_ADMISSION": "ADMIN",
    "HOSPITAL_DISCHARGE": "ADMIN",
    "ICU_ADMISSION": "ADMIN",
    "ICU_DISCHARGE": "ADMIN",
    "TRANSFER_TO": "ADMIN",
    "DRG": "ADMIN",
    "GENDER": "ADMIN",
    "ED_REGISTRATION": "ADMIN",
    "ED_OUT": "ADMIN",
    "MEDS_BIRTH": "ADMIN",
    # Infusions grouped together
    "INFUSION_START": "INFUSION",
    "INFUSION_END": "INFUSION",
    "SUBJECT_WEIGHT_AT_INFUSION": "INFUSION",
    # Observations / vitals (codes without // delimiter)
    "Blood Pressure": "VITALS",
    "Weight (Lbs)": "VITALS",
    "SUBJECT_FLUID_OUTPUT": "VITALS",
}


def classify_event(code_str):
    """Return (major_category, detail) for a code string."""
    # Check if the whole code string maps to a known category
    if code_str in CATEGORY_MAP:
        return CATEGORY_MAP[code_str], code_str

    parts = code_str.split("//")
    raw_cat = parts[0] if parts else ""
    detail = "//".join(parts[1:]) if len(parts) > 1 else code_str

    # Map raw prefix to major category
    if raw_cat in CATEGORY_MAP:
        return CATEGORY_MAP[raw_cat], detail

    # No // delimiter and not in the map → likely a vital/observation
    if "//" not in code_str and raw_cat:
        return "VITALS", code_str

    return raw_cat if raw_cat else "OTHER", detail


def parse_events(bindings):
    events = []
    for b in bindings:
        code_str = b["codeString"]["value"]
        category, detail = classify_event(code_str)

        events.append({
            "time": b["time"]["value"],
            "category": category,
            "code": code_str,
            "detail": detail,
            "label": b.get("parentLabel", {}).get("value", ""),
            "numVal": b.get("numVal", {}).get("value", ""),
            "textVal": b.get("textVal", {}).get("value", ""),
        })
    return events


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Patient Timeline — Subject {subject_id}</title>
<style>
:root {{
  --bg: #f7f8fc; --surface: #fff; --text: #1a1a2e; --muted: #7f8c8d;
  --border: #e0e0e0; --accent: #3498db;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); overflow-x: hidden; }}

/* Header */
header {{ background: linear-gradient(135deg, #2c3e50, #34495e); color: #fff; padding: 14px 28px; display: flex; justify-content: space-between; align-items: center; }}
header h1 {{ font-size: 18px; font-weight: 600; }}
header .stats {{ font-size: 12px; opacity: .75; }}

/* Category filter bar */
.filter-bar {{
  padding: 10px 28px; background: var(--surface); border-bottom: 1px solid var(--border);
  display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
}}
.filter-bar .label {{ font-size: 12px; font-weight: 600; margin-right: 4px; }}
.cat-btn {{
  display: inline-flex; align-items: center; gap: 4px;
  padding: 4px 10px; border-radius: 16px; border: 2px solid var(--border);
  font-size: 11px; font-weight: 500; cursor: pointer; background: var(--surface);
  transition: all .15s;
}}
.cat-btn .swatch {{ width: 8px; height: 8px; border-radius: 50%; }}
.cat-btn .cnt {{ color: var(--muted); font-size: 10px; }}
.cat-btn.active {{ border-color: currentColor; background: color-mix(in srgb, currentColor 8%, white); }}
.cat-btn.inactive {{ opacity: .3; }}

/* ===== OVERVIEW MINIMAP ===== */
.overview-section {{
  padding: 8px 28px 4px; background: var(--surface); border-bottom: 1px solid var(--border);
}}
.overview-label {{
  font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase;
  letter-spacing: .5px; margin-bottom: 4px;
}}
#overview {{
  position: relative; height: 60px; background: #f0f2f5; border-radius: 6px;
  cursor: crosshair; user-select: none;
}}
#overview .ov-mark {{
  position: absolute; border-radius: 1px; pointer-events: none;
}}
/* brush selection */
#brush {{
  position: absolute; top: 0; height: 100%; background: rgba(52,152,219,.12);
  border-left: 2px solid var(--accent); border-right: 2px solid var(--accent);
  cursor: grab; z-index: 2;
}}
#brush:active {{ cursor: grabbing; }}
#brush .handle {{
  position: absolute; top: 0; width: 8px; height: 100%; cursor: ew-resize; z-index: 3;
}}
#brush .handle-l {{ left: -5px; }}
#brush .handle-r {{ right: -5px; }}
#brush .handle::after {{
  content: ''; position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
  width: 4px; height: 20px; background: var(--accent); border-radius: 2px; opacity: .6;
}}
.ov-axis {{
  position: relative; height: 18px;
}}
.ov-axis span {{
  position: absolute; font-size: 10px; color: var(--muted); top: 2px;
  transform: translateX(-50%); white-space: nowrap;
}}

/* Detail range label */
.detail-header {{
  padding: 8px 28px 4px; display: flex; align-items: center; gap: 12px;
}}
.detail-header .range-label {{ font-size: 13px; font-weight: 600; }}
.detail-header .range-hint {{ font-size: 11px; color: var(--muted); }}
.detail-header button {{
  padding: 3px 10px; border-radius: 4px; border: 1px solid var(--border);
  background: var(--surface); cursor: pointer; font-size: 12px;
}}
.detail-header button:hover {{ background: #eee; }}

/* ===== DETAIL SWIMLANES ===== */
#timeline {{ padding: 0 28px 20px; }}

.lane {{ margin-bottom: 2px; }}
.lane-head {{
  display: flex; align-items: center; gap: 8px; padding: 5px 0;
  cursor: pointer; user-select: none;
}}
.lane-head .swatch {{ width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; }}
.lane-head .name {{ font-size: 13px; font-weight: 600; }}
.lane-head .cnt {{ font-size: 11px; color: var(--muted); }}
.lane-head .arrow {{ font-size: 10px; color: var(--muted); transition: transform .2s; }}
.lane-head .arrow.open {{ transform: rotate(90deg); }}

.lane-track {{
  position: relative; height: 36px; background: #f0f2f5; border-radius: 6px;
  margin: 0 0 2px 20px; overflow: hidden;
}}
.mark {{
  position: absolute; top: 3px; border-radius: 3px; cursor: pointer; min-width: 3px;
}}
.mark:hover {{ outline: 2px solid var(--text); z-index: 5; }}

.detail-axis {{
  position: relative; height: 20px; margin: 0 0 0 20px;
}}
.detail-axis span {{
  position: absolute; font-size: 10px; color: var(--muted); top: 0;
  transform: translateX(-50%); white-space: nowrap;
}}

/* Detail table */
.lane-details {{ display: none; margin: 4px 0 8px 20px; max-height: 280px; overflow-y: auto; border-radius: 6px; border: 1px solid var(--border); }}
.lane-details.open {{ display: block; }}
.lane-details table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
.lane-details th {{
  position: sticky; top: 0; background: #ecf0f1; text-align: left;
  padding: 5px 10px; font-weight: 600;
}}
.lane-details td {{ padding: 4px 10px; border-bottom: 1px solid #f5f5f5; }}
.lane-details tr:hover td {{ background: #f0f8ff; }}
.lane-details .val {{ font-family: 'SF Mono', Menlo, monospace; font-size: 11px; }}

/* ===== AI SECTION ===== */
.ai-section {{
  padding: 16px 28px; background: var(--bg);
}}
.ai-section-title {{
  font-size: 14px; font-weight: 700; color: var(--text); margin-bottom: 10px;
  text-transform: uppercase; letter-spacing: .5px;
}}
.ai-row {{
  display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
}}
@media (max-width: 900px) {{ .ai-row {{ grid-template-columns: 1fr; }} }}
.ai-panel {{
  background: var(--surface); border-radius: 10px; border: 1px solid var(--border);
  padding: 14px 18px; display: flex; flex-direction: column;
}}
.ai-panel-head {{
  display: flex; align-items: center; gap: 8px; margin-bottom: 10px;
}}
.ai-panel-icon {{ font-size: 16px; }}
.ai-panel-label {{ font-size: 13px; font-weight: 700; }}
.ai-panel-meta {{ font-size: 11px; color: var(--muted); }}
.summary-bar {{
  display: flex; align-items: center; gap: 10px; padding: 4px 0;
}}
.summary-bar .gen-btn {{
  padding: 7px 18px; border-radius: 8px; border: none;
  background: linear-gradient(135deg, #6c5ce7, #a29bfe); color: #fff;
  font-size: 12px; font-weight: 600; cursor: pointer; transition: all .2s;
  box-shadow: 0 2px 8px rgba(108,92,231,.25);
}}
.summary-bar .gen-btn:hover {{ transform: translateY(-1px); box-shadow: 0 4px 12px rgba(108,92,231,.35); }}
.summary-bar .gen-btn:disabled {{ opacity: .5; cursor: wait; transform: none; box-shadow: none; }}
.summary-bar .gen-btn .spinner {{
  display: inline-block; width: 12px; height: 12px; border: 2px solid rgba(255,255,255,.3);
  border-top-color: #fff; border-radius: 50%; animation: spin .6s linear infinite;
  vertical-align: middle; margin-right: 6px;
}}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
.summary-bar .status {{ font-size: 11px; color: var(--muted); }}

#summaryCard {{
  display: none; margin: 8px 0; border-radius: 10px; overflow: hidden;
  border: 1px solid #d4c5f9; background: var(--surface);
  box-shadow: 0 2px 12px rgba(108,92,231,.08);
}}
#summaryCard.visible {{ display: block; }}

.summary-header {{
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 16px; background: linear-gradient(135deg, #f8f6ff, #ede8ff);
  cursor: pointer; user-select: none;
}}
.summary-header .left {{ display: flex; align-items: center; gap: 8px; }}
.summary-header .icon {{ font-size: 16px; }}
.summary-header .title {{ font-size: 12px; font-weight: 700; color: #6c5ce7; text-transform: uppercase; letter-spacing: .5px; }}
.summary-header .meta {{ font-size: 11px; color: var(--muted); }}
.summary-header .chevron {{
  font-size: 14px; color: #6c5ce7; transition: transform .2s;
}}
.summary-header .chevron.collapsed {{ transform: rotate(-90deg); }}

.summary-body {{
  padding: 16px 20px; font-size: 13px; line-height: 1.8; color: var(--text);
  max-height: 400px; overflow-y: auto;
}}
.summary-body.collapsed {{ display: none; }}
.summary-body p {{ margin-bottom: 10px; }}
.summary-body p:last-child {{ margin-bottom: 0; }}
.summary-body strong {{ color: #2d3436; }}
.summary-body em {{ color: #6c5ce7; font-style: normal; font-weight: 500; }}

/* ===== CHAT WIDGET ===== */
.chat-messages {{
  min-height: 60px; max-height: 350px; overflow-y: auto;
  border: 1px solid var(--border); border-radius: 8px; background: #f9fafb;
  padding: 10px; margin-bottom: 8px; display: flex; flex-direction: column; gap: 8px;
}}
.chat-messages:empty::before {{
  content: 'No messages yet. Ask a question about this patient\\'s record.';
  color: var(--muted); font-size: 12px; font-style: italic;
}}
.chat-msg {{
  padding: 8px 12px; border-radius: 8px; font-size: 13px; line-height: 1.6;
  max-width: 90%; word-wrap: break-word;
}}
.chat-msg.user {{
  background: var(--accent); color: #fff; align-self: flex-end; border-bottom-right-radius: 2px;
}}
.chat-msg.assistant {{
  background: var(--surface); border: 1px solid var(--border); align-self: flex-start; border-bottom-left-radius: 2px;
}}
.chat-msg.assistant p {{ margin-bottom: 6px; }}
.chat-msg.assistant p:last-child {{ margin-bottom: 0; }}
.chat-msg.assistant strong {{ color: #2d3436; }}
.chat-msg.thinking {{
  background: var(--surface); border: 1px solid var(--border); align-self: flex-start;
  color: var(--muted); font-style: italic;
}}
.chat-input-row {{
  display: flex; gap: 8px;
}}
.chat-input-row input {{
  flex: 1; padding: 8px 12px; border: 1px solid var(--border); border-radius: 8px;
  font-size: 13px; outline: none;
}}
.chat-input-row input:focus {{ border-color: var(--accent); }}
.chat-hint {{
  font-size: 11px; color: var(--muted); margin-top: 6px;
}}

/* Tooltip */
#tip {{
  position: fixed; background: rgba(30,30,46,.95); color: #fff; padding: 10px 14px;
  border-radius: 8px; font-size: 12px; pointer-events: none; display: none;
  z-index: 200; max-width: 420px; line-height: 1.6; backdrop-filter: blur(4px);
  box-shadow: 0 4px 16px rgba(0,0,0,.25);
}}
#tip b {{ color: #74b9ff; }}
</style>
</head>
<body>
<header>
  <h1>Patient Timeline &mdash; Subject {subject_id}</h1>
  <div class="stats" id="summary"></div>
</header>

<div class="filter-bar">
  <span class="label">Categories:</span>
  <div id="filters"></div>
</div>

<!-- Overview minimap -->
<div class="overview-section">
  <div class="overview-label">Overview — drag to select time range, drag edges to resize</div>
  <div id="overview">
    <div id="brush">
      <div class="handle handle-l"></div>
      <div class="handle handle-r"></div>
    </div>
  </div>
  <div class="ov-axis" id="ovAxis"></div>
</div>

<!-- Detail view -->
<div class="detail-header">
  <span class="range-label" id="rangeLabel"></span>
  <span class="range-hint">Scroll to zoom &middot; Drag tracks to pan &middot; Double-click overview to reset</span>
  <button onclick="resetZoom()">Reset view</button>
</div>
<div id="timeline"></div>

<!-- AI Summary (scoped to detail view) -->
<div class="ai-section">
  <div class="ai-section-title">AI Tools</div>
  <div class="ai-row">
    <div class="ai-panel">
      <div class="ai-panel-head">
        <span class="ai-panel-icon">&#10024;</span>
        <span class="ai-panel-label">Summarize View</span>
        <span class="ai-panel-meta" id="summaryEvtCount"></span>
      </div>
      <div class="summary-bar">
        <button class="gen-btn" id="summarizeBtn" onclick="summarizeView()">
          <span id="btnLabel">&#10024; Summarize selected range</span>
        </button>
        <span class="status" id="summaryStatus"></span>
      </div>
      <div id="summaryCard">
        <div class="summary-header" onclick="toggleSummary()">
          <div class="left">
            <span class="icon">&#129302;</span>
            <span class="title">AI Clinical Summary</span>
            <span class="meta" id="summaryMeta"></span>
          </div>
          <span class="chevron" id="summaryChevron">&#9660;</span>
        </div>
        <div class="summary-body" id="summaryBody"></div>
      </div>
    </div>
    <div class="ai-panel">
      <div class="ai-panel-head">
        <span class="ai-panel-icon">&#128172;</span>
        <span class="ai-panel-label">Ask About This Patient</span>
      </div>
      <div id="chatMessages" class="chat-messages"></div>
      <div class="chat-input-row">
        <input type="text" id="chatInput" placeholder="e.g. What medications were given on 2180-07-23?" onkeydown="if(event.key==='Enter')sendChat()">
        <button class="gen-btn" id="chatSendBtn" onclick="sendChat()">Send</button>
      </div>
      <div class="chat-hint">Ask about specific dates, conditions, medications, lab trends, etc. Uses events in the current view.</div>
    </div>
  </div>
</div>

<div id="tip"></div>

<script>
const ALL = {events_json};

const PAL = {{
  LAB:'#3498db', MEDICATION:'#e67e22', DIAGNOSIS:'#e74c3c', PROCEDURE:'#9b59b6',
  INFUSION:'#2ecc71', ADMIN:'#8e44ad', VITALS:'#e84393', OTHER:'#95a5a6'
}};
function color(c) {{ return PAL[c] || PAL.OTHER; }}

/* ---- state ---- */
const cats = [...new Set(ALL.map(e => e.category))];
let activeCats = new Set(cats);
const allTimes = ALL.map(e => +new Date(e.time));
const gMin0 = Math.min(...allTimes), gMax0 = Math.max(...allTimes);
const pad = (gMax0 - gMin0) * .02 || 36e5;
const gMin = gMin0 - pad, gMax = gMax0 + pad;
let vMin = gMin, vMax = gMax;
let openLanes = new Set();

const catCounts = {{}};
ALL.forEach(e => catCounts[e.category] = (catCounts[e.category]||0) + 1);
const sortedCats = cats.sort((a,b) => (catCounts[b]||0) - (catCounts[a]||0));

/* ========== OVERVIEW MINIMAP ========== */
function renderOverview() {{
  const ov = document.getElementById('overview');
  const w = ov.offsetWidth;
  const gSpan = gMax - gMin;

  /* draw all event marks (tiny) */
  let marks = '';
  const filtered = ALL.filter(e => activeCats.has(e.category));
  /* bin into pixels */
  const bins = {{}};
  filtered.forEach(e => {{
    const t = +new Date(e.time);
    const px = Math.floor(((t - gMin) / gSpan) * w);
    const key = e.category + '|' + px;
    bins[key] = bins[key] || {{ cat: e.category, px, n: 0 }};
    bins[key].n++;
  }});

  /* render per-category rows: stack categories vertically */
  const activeSorted = sortedCats.filter(c => activeCats.has(c));
  const rowH = Math.max(4, Math.min(10, Math.floor(56 / Math.max(activeSorted.length, 1))));

  Object.values(bins).forEach(b => {{
    const catIdx = activeSorted.indexOf(b.cat);
    if (catIdx < 0) return;
    const y = 2 + catIdx * rowH;
    const h = Math.max(rowH - 1, 2);
    const op = Math.min(1, .3 + b.n * .1);
    marks += `<div class="ov-mark" style="left:${{b.px}}px;top:${{y}}px;width:2px;height:${{h}}px;background:${{color(b.cat)}};opacity:${{op}}"></div>`;
  }});

  /* keep only marks, preserve brush */
  ov.querySelectorAll('.ov-mark').forEach(el => el.remove());
  ov.insertAdjacentHTML('afterbegin', marks);

  updateBrush();
  renderOvAxis();
}}

function updateBrush() {{
  const ov = document.getElementById('overview');
  const brush = document.getElementById('brush');
  const w = ov.offsetWidth;
  const gSpan = gMax - gMin;
  const left = ((vMin - gMin) / gSpan) * w;
  const right = ((vMax - gMin) / gSpan) * w;
  brush.style.left = left + 'px';
  brush.style.width = Math.max(6, right - left) + 'px';
}}

function renderOvAxis() {{
  document.getElementById('ovAxis').innerHTML = generateSmartTicks(gMin, gMax);
}}

/* ---- brush interaction ---- */
(function() {{
  const ov = document.getElementById('overview');
  const brush = document.getElementById('brush');
  let mode = null; // 'move', 'resize-l', 'resize-r', 'create'
  let startX = 0, startVMin = 0, startVMax = 0;

  function xToTime(x) {{
    const rect = ov.getBoundingClientRect();
    const frac = (x - rect.left) / rect.width;
    return gMin + frac * (gMax - gMin);
  }}

  brush.querySelector('.handle-l').addEventListener('mousedown', e => {{
    e.stopPropagation(); mode = 'resize-l'; startX = e.clientX; startVMin = vMin; startVMax = vMax;
  }});
  brush.querySelector('.handle-r').addEventListener('mousedown', e => {{
    e.stopPropagation(); mode = 'resize-r'; startX = e.clientX; startVMin = vMin; startVMax = vMax;
  }});
  brush.addEventListener('mousedown', e => {{
    if (mode) return;
    mode = 'move'; startX = e.clientX; startVMin = vMin; startVMax = vMax; e.preventDefault();
  }});
  ov.addEventListener('mousedown', e => {{
    if (mode) return;
    if (e.target !== ov && !e.target.classList.contains('ov-mark')) return;
    mode = 'create'; startX = e.clientX;
    const t = xToTime(e.clientX);
    vMin = t; vMax = t;
    updateBrush();
  }});
  ov.addEventListener('dblclick', () => {{ resetZoom(); }});

  document.addEventListener('mousemove', e => {{
    if (!mode) return;
    const rect = ov.getBoundingClientRect();
    const gSpan = gMax - gMin;
    const dx = e.clientX - startX;
    const dt = (dx / rect.width) * gSpan;

    if (mode === 'move') {{
      let newMin = startVMin + dt, newMax = startVMax + dt;
      const span = newMax - newMin;
      if (newMin < gMin) {{ newMin = gMin; newMax = gMin + span; }}
      if (newMax > gMax) {{ newMax = gMax; newMin = gMax - span; }}
      vMin = newMin; vMax = newMax;
    }} else if (mode === 'resize-l') {{
      vMin = Math.max(gMin, Math.min(startVMin + dt, vMax - gSpan * .005));
    }} else if (mode === 'resize-r') {{
      vMax = Math.min(gMax, Math.max(startVMax + dt, vMin + gSpan * .005));
    }} else if (mode === 'create') {{
      const t = xToTime(e.clientX);
      const t0 = xToTime(startX);
      vMin = Math.max(gMin, Math.min(t, t0));
      vMax = Math.min(gMax, Math.max(t, t0));
    }}
    updateBrush();
    renderDetail();
  }});

  document.addEventListener('mouseup', () => {{
    if (mode === 'create' && (vMax - vMin) < (gMax - gMin) * .003) {{
      vMin = gMin; vMax = gMax; /* click without drag = reset */
    }}
    mode = null;
    renderDetail();
    updateBrush();
  }});
}})();

/* ========== FILTERS ========== */
function buildFilters() {{
  const box = document.getElementById('filters');
  box.innerHTML = '';
  sortedCats.forEach(cat => {{
    const btn = document.createElement('div');
    btn.className = 'cat-btn' + (activeCats.has(cat) ? ' active' : ' inactive');
    btn.style.color = color(cat);
    btn.innerHTML = `<span class="swatch" style="background:${{color(cat)}}"></span>${{cat}} <span class="cnt">${{catCounts[cat]}}</span>`;
    btn.onclick = () => {{
      if (activeCats.has(cat)) activeCats.delete(cat); else activeCats.add(cat);
      buildFilters(); renderOverview(); renderDetail();
    }};
    box.appendChild(btn);
  }});
}}

/* ========== DETAIL VIEW ========== */
function renderDetail() {{
  const span = vMax - vMin;
  const filtered = ALL.filter(e => activeCats.has(e.category));

  const groups = {{}};
  filtered.forEach(e => {{ (groups[e.category] = groups[e.category]||[]).push(e); }});

  const order = Object.keys(groups).sort((a,b) => {{
    const cO = ['ADMIN','LAB','MEDICATION','INFUSION','DIAGNOSIS','PROCEDURE','VITALS','OTHER'];
    return cO.indexOf(a) - cO.indexOf(b);
  }});

  let html = '';
  order.forEach(cat => {{
    const allEvts = groups[cat];
    const visEvts = allEvts.filter(e => {{ const t=+new Date(e.time); return t>=vMin && t<=vMax; }});
    const c = color(cat);
    const isOpen = openLanes.has(cat);

    html += `<div class="lane">`;
    html += `<div class="lane-head" onclick="toggleLane('${{cat}}')">`;
    html += `<span class="swatch" style="background:${{c}}"></span>`;
    html += `<span class="name">${{cat}}</span>`;
    html += `<span class="cnt">(${{visEvts.length}} in view / ${{allEvts.length}} total)</span>`;
    html += `<span class="arrow${{isOpen?' open':''}}">&rsaquo;</span>`;
    html += `</div>`;

    html += `<div class="lane-track">`;
    const pxW = (document.getElementById('timeline')?.offsetWidth || 900) - 40;
    const binSize = span / Math.max(pxW, 1);
    const bins = {{}};
    visEvts.forEach(e => {{
      const t = +new Date(e.time);
      const bk = Math.floor((t - vMin) / binSize);
      (bins[bk] = bins[bk] || []).push(e);
    }});

    Object.entries(bins).forEach(([bk, be]) => {{
      const pct = (parseInt(bk) * binSize) / span * 100;
      if (pct < -1 || pct > 101) return;
      const n = be.length;
      const w = Math.max(3, Math.min(28, 3 + n * 2));
      const h = Math.min(30, 10 + n * 3);
      const top = (30 - h) / 2 + 3;
      const op = Math.min(1, .5 + n * .1);
      const tipText = be.slice(0, 8).map(e => {{
        let s = e.time.slice(0,16).replace('T',' ') + '  ' + e.detail;
        if (e.label) s += ' (' + e.label + ')';
        if (e.numVal) s += ' = ' + e.numVal;
        else if (e.textVal) s += ' = ' + e.textVal;
        return s;
      }}).join('\\n') + (n > 8 ? '\\n... +' + (n-8) + ' more' : '');
      html += `<div class="mark" style="left:${{pct}}%;width:${{w}}px;height:${{h}}px;top:${{top}}px;background:${{c}};opacity:${{op}}" data-tip="${{tipText.replace(/"/g,'&quot;').replace(/</g,'&lt;')}}" onmouseenter="showTip(event)" onmouseleave="hideTip()"></div>`;
    }});
    html += `</div>`;

    html += `<div class="lane-details${{isOpen?' open':''}}" id="det-${{cat}}">`;
    html += `<table><tr><th>Time</th><th>Code</th><th>Label</th><th>Value</th></tr>`;
    visEvts.slice(0, 500).forEach(e => {{
      const v = e.numVal ? `<span class="val">${{e.numVal}}</span>` : e.textVal ? e.textVal : '';
      html += `<tr><td>${{e.time.slice(0,16).replace('T',' ')}}</td><td>${{e.detail}}</td><td>${{e.label}}</td><td>${{v}}</td></tr>`;
    }});
    if (visEvts.length > 500) html += `<tr><td colspan="4" style="color:var(--muted)">… ${{visEvts.length-500}} more</td></tr>`;
    html += `</table></div></div>`;
  }});

  /* detail time axis */
  html += `<div class="detail-axis">${{generateSmartTicks(vMin, vMax, 20)}}</div>`;

  document.getElementById('timeline').innerHTML = html;

  /* count visible events for the header and AI panel */
  const allVisible = ALL.filter(e => {{
    if (!activeCats.has(e.category)) return false;
    const t = +new Date(e.time);
    return t >= vMin && t <= vMax;
  }});
  document.getElementById('rangeLabel').textContent =
    fmtRange(vMin) + ' — ' + fmtRange(vMax) + `  (${{allVisible.length.toLocaleString()}} events in view)`;
  document.getElementById('summary').textContent =
    `${{ALL.length}} events · ${{cats.length}} categories · ${{fmtRange(gMin0)}} to ${{fmtRange(gMax0)}}`;
  const seLabel = document.getElementById('summaryEvtCount');
  if (seLabel) seLabel.textContent = `${{allVisible.length.toLocaleString()}} events in view`;
}}

function toggleLane(cat) {{
  if (openLanes.has(cat)) openLanes.delete(cat); else openLanes.add(cat);
  renderDetail();
}}

/* ---- time formatting ---- */
function fmtRange(ms) {{
  const d = new Date(ms);
  return d.toLocaleDateString('en-US', {{month:'short', day:'numeric', year:'numeric'}});
}}

function generateSmartTicks(tMin, tMax, marginPx) {{
  /* Produce well-spaced, human-readable time axis labels.
     Chooses year / month / day / hour granularity based on span. */
  const span = tMax - tMin;
  const dMin = new Date(tMin), dMax = new Date(tMax);
  let ticks = [];

  if (span > 86400e3 * 365 * 1.5) {{
    /* Multi-year: show years */
    const y0 = dMin.getFullYear(), y1 = dMax.getFullYear();
    const nYears = y1 - y0 + 1;
    const step = Math.max(1, Math.ceil(nYears / 10));
    for (let y = y0; y <= y1 + 1; y += step) {{
      const t = +new Date(y, 0, 1);
      if (t >= tMin && t <= tMax) ticks.push({{ t, label: String(y) }});
    }}
  }} else if (span > 86400e3 * 60) {{
    /* Months-to-~year: show "Mon 'YY" */
    const step = span > 86400e3 * 300 ? 3 : span > 86400e3 * 120 ? 2 : 1;
    let cur = new Date(dMin.getFullYear(), dMin.getMonth(), 1);
    while (+cur <= tMax) {{
      const t = +cur;
      if (t >= tMin) ticks.push({{ t, label: cur.toLocaleDateString('en-US', {{month:'short', year:'2-digit'}}) }});
      cur.setMonth(cur.getMonth() + step);
    }}
  }} else if (span > 86400e3 * 2) {{
    /* Days-to-weeks: show "Mon D" */
    const stepDays = Math.max(1, Math.ceil(span / 86400e3 / 10));
    let cur = new Date(dMin.getFullYear(), dMin.getMonth(), dMin.getDate());
    while (+cur <= tMax + 86400e3) {{
      const t = +cur;
      if (t >= tMin && t <= tMax) ticks.push({{ t, label: cur.toLocaleDateString('en-US', {{month:'short', day:'numeric'}}) }});
      cur.setDate(cur.getDate() + stepDays);
    }}
  }} else if (span > 36e5 * 6) {{
    /* Hours-to-days */
    const stepH = Math.max(1, Math.ceil(span / 36e5 / 10));
    let cur = new Date(dMin.getFullYear(), dMin.getMonth(), dMin.getDate(), dMin.getHours());
    while (+cur <= tMax + 36e5) {{
      const t = +cur;
      if (t >= tMin && t <= tMax) {{
        const lbl = cur.toLocaleDateString('en-US', {{month:'short', day:'numeric'}}) + ' ' +
          cur.toLocaleTimeString('en-US', {{hour:'2-digit', minute:'2-digit'}});
        ticks.push({{ t, label: lbl }});
      }}
      cur.setHours(cur.getHours() + stepH);
    }}
  }} else {{
    /* Sub-6h: minutes */
    const stepM = Math.max(1, Math.ceil(span / 60e3 / 10));
    let cur = new Date(Math.ceil(tMin / (stepM*60e3)) * (stepM*60e3));
    while (+cur <= tMax) {{
      ticks.push({{ t: +cur, label: cur.toLocaleTimeString('en-US', {{hour:'2-digit', minute:'2-digit', second:'2-digit'}}) }});
      cur = new Date(+cur + stepM * 60e3);
    }}
  }}

  /* Remove ticks that would overlap (< 60px apart) */
  const totalSpan = tMax - tMin;
  let filtered = [];
  let lastPx = -999;
  ticks.forEach(tk => {{
    const pct = (tk.t - tMin) / totalSpan;
    const px = pct * (window.innerWidth || 900);
    if (px - lastPx > 60) {{ filtered.push(tk); lastPx = px; }}
  }});

  const mg = marginPx || 0;
  return filtered.map(tk => {{
    const pct = ((tk.t - tMin) / totalSpan) * 100;
    return `<span style="left:calc(${{mg}}px + ${{pct}}% * (1 - ${{mg*2}}px / 100%))">${{tk.label}}</span>`;
  }}).join('');
}}

/* ---- tooltip ---- */
function showTip(ev) {{
  const tip = document.getElementById('tip');
  const raw = ev.target.dataset.tip;
  tip.innerHTML = raw.split('\\n').map(line => {{
    const m = line.match(/^(\\S+ \\S+)\\s+(.+?)(?:\\s+\\((.+?)\\))?(?:\\s+=\\s+(.+))?$/);
    if (!m) return line;
    let s = `<b>${{m[1]}}</b> ${{m[2]}}`;
    if (m[3]) s += ` <span style="color:#dfe6e9">(${{m[3]}})</span>`;
    if (m[4]) s += ` <span style="color:#55efc4">= ${{m[4]}}</span>`;
    return s;
  }}).join('<br>');
  tip.style.display = 'block';
  const r = ev.target.getBoundingClientRect();
  tip.style.left = Math.min(r.left, window.innerWidth - 440) + 'px';
  tip.style.top = (r.top - tip.offsetHeight - 8) + 'px';
  if (parseInt(tip.style.top) < 0) tip.style.top = (r.bottom + 8) + 'px';
}}
function hideTip() {{ document.getElementById('tip').style.display = 'none'; }}

/* ---- zoom ---- */
function resetZoom() {{ vMin = gMin; vMax = gMax; updateBrush(); renderDetail(); }}

/* scroll-to-zoom on detail tracks */
document.addEventListener('wheel', ev => {{
  if (!ev.target.closest('.lane-track')) return;
  ev.preventDefault();
  const rect = document.getElementById('timeline').getBoundingClientRect();
  const frac = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
  const anchor = vMin + (vMax - vMin) * frac;
  const factor = ev.deltaY > 0 ? 1.25 : 0.75;
  vMin = Math.max(gMin, anchor - (anchor - vMin) * factor);
  vMax = Math.min(gMax, anchor + (vMax - anchor) * factor);
  updateBrush(); renderDetail();
}}, {{passive: false}});

/* drag-to-pan on detail tracks */
let panMode = false, panX = 0;
document.addEventListener('mousedown', ev => {{
  if (!ev.target.closest('.lane-track')) return;
  panMode = true; panX = ev.clientX; ev.preventDefault();
}});
document.addEventListener('mousemove', ev => {{
  if (!panMode) return;
  const dx = ev.clientX - panX; panX = ev.clientX;
  const rect = document.getElementById('timeline').getBoundingClientRect();
  const shift = -(dx / rect.width) * (vMax - vMin);
  const s = vMax - vMin;
  let newMin = vMin + shift, newMax = vMax + shift;
  if (newMin < gMin) {{ newMin = gMin; newMax = gMin + s; }}
  if (newMax > gMax) {{ newMax = gMax; newMin = gMax - s; }}
  vMin = newMin; vMax = newMax;
  updateBrush(); renderDetail();
}});
document.addEventListener('mouseup', () => {{ panMode = false; }});

/* ========== AI SUMMARY ========== */
const AZURE_ENDPOINT = '{azure_endpoint}';
const AZURE_DEPLOYMENT = '{azure_deployment}';
const AZURE_API_VERSION = '{azure_api_version}';
const AZURE_API_KEY = '{azure_key}';

let summaryCollapsed = false;

function toggleSummary() {{
  summaryCollapsed = !summaryCollapsed;
  document.getElementById('summaryBody').classList.toggle('collapsed', summaryCollapsed);
  document.getElementById('summaryChevron').classList.toggle('collapsed', summaryCollapsed);
}}

function renderMarkdown(text) {{
  /* lightweight markdown: bold, headers, bullets, paragraphs */
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/^### (.+)$/gm, '<p style="font-size:14px;font-weight:700;color:#2d3436;margin:12px 0 4px">$1</p>')
    .replace(/^## (.+)$/gm, '<p style="font-size:15px;font-weight:700;color:#2d3436;margin:14px 0 6px">$1</p>')
    .replace(/^[-–•] (.+)$/gm, '<div style="padding-left:16px;margin:2px 0">&#8226; $1</div>')
    .replace(/^(\d+)\. (.+)$/gm, '<div style="padding-left:16px;margin:2px 0">$1. $2</div>')
    .split(/\\n\\n+/).map(p => {{
      if (p.trim().startsWith('<')) return p;
      return `<p>${{p.replace(/\\n/g, '<br>')}}</p>`;
    }}).join('');
}}

async function summarizeView() {{
  const btn = document.getElementById('summarizeBtn');
  const btnLabel = document.getElementById('btnLabel');
  const status = document.getElementById('summaryStatus');
  const card = document.getElementById('summaryCard');
  const body = document.getElementById('summaryBody');
  const meta = document.getElementById('summaryMeta');

  const filtered = ALL.filter(e => activeCats.has(e.category));
  const visible = filtered.filter(e => {{
    const t = +new Date(e.time);
    return t >= vMin && t <= vMax;
  }});

  if (visible.length === 0) {{
    status.textContent = 'No events in the selected range.';
    return;
  }}

  btn.disabled = true;
  btnLabel.innerHTML = '<span class="spinner"></span>Generating summary...';
  status.textContent = '';
  card.classList.remove('visible');

  const groups = {{}};
  visible.forEach(e => (groups[e.category] = groups[e.category] || []).push(e));

  const span = vMax - vMin;
  let prompt = `You are a clinical informatics assistant. Below are medical events for a patient within a selected time window (${{fmtRange(vMin)}} to ${{fmtRange(vMax)}}). Provide a concise clinical summary of this period. Structure your response with these sections using markdown headers (##):\n\n## Overview\nBrief 1-2 sentence summary of the encounter.\n\n## Key Findings\nImportant lab results, vital signs, or observations.\n\n## Medications & Treatments\nMedications administered, infusions, procedures.\n\n## Diagnoses\nDiagnoses recorded.\n\n## Notable Patterns\nAny trends, concerns, or clinically significant patterns.\n\nUse **bold** for important values. Be concise but thorough.\n\n`;

  for (const [cat, evts] of Object.entries(groups)) {{
    prompt += `=== ${{cat}} (${{evts.length}} events) ===\n`;
    if (evts.length > 80) {{
      const detailCounts = {{}};
      evts.forEach(e => {{
        const key = e.label || e.detail;
        detailCounts[key] = (detailCounts[key] || 0) + 1;
      }});
      const sorted = Object.entries(detailCounts).sort((a,b) => b[1]-a[1]);
      prompt += `Top items: ${{sorted.slice(0,20).map(([k,v]) => `${{k}} (x${{v}})`).join(', ')}}\n`;
      const withVal = evts.filter(e => e.numVal || e.textVal).slice(0, 15);
      if (withVal.length) {{
        prompt += `Sample values:\n`;
        withVal.forEach(e => {{
          prompt += `  ${{e.time.slice(0,16)}} ${{e.label || e.detail}}: ${{e.numVal || e.textVal}}\n`;
        }});
      }}
    }} else {{
      evts.forEach(e => {{
        let line = `  ${{e.time.slice(0,16)}} ${{e.detail}}`;
        if (e.label) line += ` (${{e.label}})`;
        if (e.numVal) line += ` = ${{e.numVal}}`;
        else if (e.textVal) line += ` = ${{e.textVal}}`;
        prompt += line + `\n`;
      }});
    }}
    prompt += `\n`;
  }}

  try {{
    const url = `${{AZURE_ENDPOINT}}/openai/deployments/${{AZURE_DEPLOYMENT}}/chat/completions?api-version=${{AZURE_API_VERSION}}`;
    const resp = await fetch(url, {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json', 'api-key': AZURE_API_KEY }},
      body: JSON.stringify({{
        messages: [{{ role: 'user', content: prompt }}],
        max_completion_tokens: 1500,
        temperature: 0.3,
      }})
    }});
    const data = await resp.json();
    if (data.error) throw new Error(data.error.message);
    const summary = data.choices[0].message.content;
    body.innerHTML = renderMarkdown(summary);
    meta.textContent = `${{visible.length}} events · ${{fmtRange(vMin)}} to ${{fmtRange(vMax)}}`;
    summaryCollapsed = false;
    document.getElementById('summaryBody').classList.remove('collapsed');
    document.getElementById('summaryChevron').classList.remove('collapsed');
    card.classList.add('visible');
    status.textContent = '';
  }} catch (err) {{
    status.textContent = `Error: ${{err.message}}`;
    console.error(err);
  }} finally {{
    btn.disabled = false;
    btnLabel.innerHTML = '&#10024; Summarize selected range';
  }}
}}

/* ========== CHAT Q&A ========== */
let chatHistory = [];

function getVisibleEvents() {{
  return ALL.filter(e => {{
    if (!activeCats.has(e.category)) return false;
    const t = +new Date(e.time);
    return t >= vMin && t <= vMax;
  }});
}}

function buildEventContext(events) {{
  /* Build a concise text representation of events for the LLM */
  const groups = {{}};
  events.forEach(e => (groups[e.category] = groups[e.category] || []).push(e));
  let ctx = `Patient record: ${{events.length}} events from ${{fmtRange(vMin)}} to ${{fmtRange(vMax)}}.\n\n`;
  for (const [cat, evts] of Object.entries(groups)) {{
    ctx += `=== ${{cat}} (${{evts.length}} events) ===\n`;
    if (evts.length > 100) {{
      const counts = {{}};
      evts.forEach(e => {{ const k = e.label || e.detail; counts[k] = (counts[k]||0)+1; }});
      const sorted = Object.entries(counts).sort((a,b) => b[1]-a[1]);
      ctx += `Top items: ${{sorted.slice(0,25).map(([k,v]) => `${{k}} (x${{v}})`).join(', ')}}\n`;
      const withVal = evts.filter(e => e.numVal || e.textVal).slice(0, 20);
      if (withVal.length) {{
        ctx += `Sample values:\n`;
        withVal.forEach(e => ctx += `  ${{e.time.slice(0,16)}} ${{e.label || e.detail}}: ${{e.numVal || e.textVal}}\n`);
      }}
    }} else {{
      evts.forEach(e => {{
        let line = `  ${{e.time.slice(0,16)}} ${{e.detail}}`;
        if (e.label) line += ` (${{e.label}})`;
        if (e.numVal) line += ` = ${{e.numVal}}`;
        else if (e.textVal) line += ` = ${{e.textVal}}`;
        ctx += line + `\n`;
      }});
    }}
    ctx += `\n`;
  }}
  return ctx;
}}

function appendChatMsg(role, html) {{
  const box = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.className = 'chat-msg ' + role;
  div.innerHTML = html;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  return div;
}}

async function sendChat() {{
  const input = document.getElementById('chatInput');
  const question = input.value.trim();
  if (!question) return;
  input.value = '';

  appendChatMsg('user', question.replace(/</g, '&lt;'));

  const visible = getVisibleEvents();
  if (visible.length === 0) {{
    appendChatMsg('assistant', 'No events in the current view. Adjust the time range and try again.');
    return;
  }}

  const thinkingEl = appendChatMsg('thinking', 'Thinking...');
  const btn = document.getElementById('chatSendBtn');
  btn.disabled = true;

  /* Build messages: system context + chat history + new question */
  const eventCtx = buildEventContext(visible);
  const sysMsg = `You are a clinical informatics assistant helping a clinician review a patient record. You have access to the following patient events currently displayed in the timeline view. Answer the user's question based on this data. Be specific, cite dates and values when relevant. Use markdown formatting.\n\n${{eventCtx}}`;

  chatHistory.push({{ role: 'user', content: question }});

  const messages = [
    {{ role: 'system', content: sysMsg }},
    ...chatHistory.slice(-10) /* keep last 10 turns for context */
  ];

  try {{
    const url = `${{AZURE_ENDPOINT}}/openai/deployments/${{AZURE_DEPLOYMENT}}/chat/completions?api-version=${{AZURE_API_VERSION}}`;
    const resp = await fetch(url, {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json', 'api-key': AZURE_API_KEY }},
      body: JSON.stringify({{
        messages,
        max_completion_tokens: 1500,
        temperature: 0.3,
      }})
    }});
    const data = await resp.json();
    if (data.error) throw new Error(data.error.message);
    const answer = data.choices[0].message.content;
    chatHistory.push({{ role: 'assistant', content: answer }});
    thinkingEl.remove();
    appendChatMsg('assistant', renderMarkdown(answer));
  }} catch (err) {{
    thinkingEl.remove();
    appendChatMsg('assistant', `<span style="color:#e74c3c">Error: ${{err.message}}</span>`);
  }} finally {{
    btn.disabled = false;
    document.getElementById('chatInput').focus();
  }}
}}

/* ---- boot ---- */
buildFilters();
renderOverview();
renderDetail();
window.addEventListener('resize', () => {{ renderOverview(); renderDetail(); }});
</script>
</body>
</html>"""


def generate_html(subject_id, events):
    return TEMPLATE.format(
        subject_id=subject_id,
        events_json=json.dumps(events),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", ""),
        azure_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", ""),
        azure_key=os.environ.get("AZURE_OPENAI_KEY", ""),
    )


def get_all_subject_ids():
    query = """
    PREFIX meds: <https://teamheka.github.io/meds-ontology#>
    SELECT DISTINCT ?sid WHERE {
      ?s a meds:Subject ;
         meds:subjectId ?sid .
    }
    ORDER BY ?sid
    """
    return [b["sid"]["value"] for b in sparql(query)]


def get_patient_summary(subject_ids):
    """Fetch event count, date range, and category counts per patient in bulk."""
    ids_values = " ".join(f'("{sid}")' for sid in subject_ids)
    query = f"""
    PREFIX meds: <https://teamheka.github.io/meds-ontology#>
    SELECT ?sid (COUNT(?e) AS ?cnt) (MIN(?time) AS ?minTime) (MAX(?time) AS ?maxTime) WHERE {{
      VALUES (?sid) {{ {ids_values} }}
      ?e a meds:Event ;
         meds:hasSubject ?s ;
         meds:time ?time .
      ?s meds:subjectId ?sid .
    }}
    GROUP BY ?sid
    ORDER BY ?sid
    """
    results = {}
    for b in sparql(query):
        results[b["sid"]["value"]] = {
            "count": int(b["cnt"]["value"]),
            "min_time": b["minTime"]["value"][:10],
            "max_time": b["maxTime"]["value"][:10],
        }
    return results


INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Patient Timelines — MIMIC-IV Demo</title>
<style>
:root {{
  --bg: #f7f8fc; --surface: #fff; --text: #1a1a2e; --muted: #7f8c8d;
  --border: #e0e0e0; --accent: #3498db;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); }}
header {{
  background: linear-gradient(135deg, #2c3e50, #34495e); color: #fff;
  padding: 24px 32px;
}}
header h1 {{ font-size: 22px; font-weight: 600; }}
header p {{ font-size: 13px; opacity: .75; margin-top: 4px; }}
.controls {{
  padding: 14px 32px; background: var(--surface); border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
}}
.controls input {{
  padding: 7px 12px; border: 1px solid var(--border); border-radius: 6px;
  font-size: 13px; width: 240px; outline: none;
}}
.controls input:focus {{ border-color: var(--accent); }}
.controls .sort-btn {{
  padding: 5px 12px; border-radius: 6px; border: 1px solid var(--border);
  background: var(--surface); font-size: 12px; cursor: pointer;
}}
.controls .sort-btn.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
.controls .count {{ font-size: 12px; color: var(--muted); margin-left: auto; }}
.grid {{
  display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 12px; padding: 20px 32px;
}}
.card {{
  background: var(--surface); border-radius: 10px; border: 1px solid var(--border);
  padding: 16px 20px; text-decoration: none; color: var(--text);
  transition: all .15s; display: block;
}}
.card:hover {{ border-color: var(--accent); box-shadow: 0 4px 16px rgba(52,152,219,.12); transform: translateY(-2px); }}
.card .sid {{ font-size: 16px; font-weight: 700; color: var(--accent); }}
.card .meta {{ font-size: 12px; color: var(--muted); margin-top: 6px; line-height: 1.7; }}
.card .meta b {{ color: var(--text); font-weight: 600; }}
.card .bar {{ display: flex; gap: 2px; margin-top: 10px; height: 6px; border-radius: 3px; overflow: hidden; }}
.card .bar span {{ height: 100%; border-radius: 3px; }}
.hidden {{ display: none !important; }}
</style>
</head>
<body>
<header>
  <h1>Patient Timelines — MIMIC-IV Demo</h1>
  <p>{num_patients} patients &middot; {total_events:,} total events</p>
</header>
<div class="controls">
  <input type="text" id="search" placeholder="Search by subject ID..." oninput="filterCards()">
  <button class="sort-btn active" data-sort="id" onclick="sortCards('id', this)">Sort by ID</button>
  <button class="sort-btn" data-sort="events" onclick="sortCards('events', this)">Sort by Events</button>
  <button class="sort-btn" data-sort="date" onclick="sortCards('date', this)">Sort by Date</button>
  <span class="count" id="countLabel">{num_patients} patients</span>
</div>
<div class="grid" id="grid">
{cards}
</div>
<script>
function filterCards() {{
  const q = document.getElementById('search').value.trim().toLowerCase();
  let shown = 0;
  document.querySelectorAll('.card').forEach(c => {{
    const match = !q || c.dataset.sid.includes(q);
    c.classList.toggle('hidden', !match);
    if (match) shown++;
  }});
  document.getElementById('countLabel').textContent = shown + ' patients';
}}
function sortCards(key, btn) {{
  document.querySelectorAll('.sort-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const grid = document.getElementById('grid');
  const cards = [...grid.children];
  cards.sort((a, b) => {{
    if (key === 'id') return a.dataset.sid.localeCompare(b.dataset.sid);
    if (key === 'events') return parseInt(b.dataset.events) - parseInt(a.dataset.events);
    if (key === 'date') return a.dataset.mintime.localeCompare(b.dataset.mintime);
  }});
  cards.forEach(c => grid.appendChild(c));
}}
</script>
</body>
</html>"""


def generate_index(patient_info, out_dir):
    """Generate an index.html linking to all patient timelines."""
    total_events = sum(info["count"] for info in patient_info.values())

    cards = ""
    for sid in sorted(patient_info.keys()):
        info = patient_info[sid]
        cards += (
            f'<a class="card" href="data/timeline_{sid}.html" '
            f'data-sid="{sid}" data-events="{info["count"]}" data-mintime="{info["min_time"]}">\n'
            f'  <div class="sid">Subject {sid}</div>\n'
            f'  <div class="meta">'
            f'<b>{info["count"]:,}</b> events &middot; '
            f'{info["min_time"]} to {info["max_time"]}'
            f'</div>\n'
            f'</a>\n'
        )

    html = INDEX_TEMPLATE.format(
        num_patients=len(patient_info),
        total_events=total_events,
        cards=cards,
    )
    out_path = os.path.join(out_dir, "index.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Index saved to {out_path}")


def generate_for_patient(subject_id, out_dir):
    print(f"  [{subject_id}] Fetching events...")
    bindings = get_patient_events(subject_id)
    if not bindings:
        print(f"  [{subject_id}] No events found, skipping.")
        return False
    events = parse_events(bindings)
    print(f"  [{subject_id}] {len(events)} events parsed")

    html = generate_html(subject_id, events)
    out_path = os.path.join(out_dir, f"timeline_{subject_id}.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"  [{subject_id}] Saved to {out_path}")
    return True


def main():
    # Output to website/patient-timeline/ at project root
    project_root = os.path.join(os.path.dirname(__file__), "..", "..")
    website_dir = os.path.join(project_root, "website", "patient-timeline")
    out_dir = os.path.join(website_dir, "data")
    os.makedirs(out_dir, exist_ok=True)

    if len(sys.argv) > 1:
        # Single patient mode
        subject_id = sys.argv[1]
        generate_for_patient(subject_id, out_dir)
    else:
        # All patients mode
        print("Querying all patient subject IDs...")
        subject_ids = get_all_subject_ids()
        print(f"Found {len(subject_ids)} patients.")
        success = 0
        for i, sid in enumerate(subject_ids, 1):
            print(f"[{i}/{len(subject_ids)}]")
            if generate_for_patient(sid, out_dir):
                success += 1
        print(f"\nDone. Generated {success}/{len(subject_ids)} timelines in {out_dir}")

        # Generate index page
        print("Fetching patient summaries for index...")
        patient_info = get_patient_summary(subject_ids)
        generate_index(patient_info, website_dir)


if __name__ == "__main__":
    main()
