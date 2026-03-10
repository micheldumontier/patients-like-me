"""Generate an interactive patient event timeline from the QLever SPARQL endpoint."""
import json
import urllib.request
import urllib.parse
import sys
import os

ENDPOINT = "http://localhost:7001"


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

/* ===== AI SUMMARY ===== */
.summary-section {{
  padding: 0 28px; background: var(--bg);
}}
.summary-bar {{
  display: flex; align-items: center; gap: 10px; padding: 8px 0 4px;
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

<!-- AI Summary -->
<div class="summary-section">
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
  const gSpan = gMax - gMin;
  const n = 10; let h = '';
  for (let i = 0; i <= n; i++) {{
    const t = gMin + gSpan * i / n;
    h += `<span style="left:${{(i/n)*100}}%">${{fmtTick(t, gSpan)}}</span>`;
  }}
  document.getElementById('ovAxis').innerHTML = h;
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
  html += `<div class="detail-axis">`;
  const nTicks = 8;
  for (let i = 0; i <= nTicks; i++) {{
    const t = vMin + span * i / nTicks;
    html += `<span style="left:calc(20px + ${{i/nTicks*100}}% * (1 - 40px/100%))">${{fmtTick(t, span)}}</span>`;
  }}
  html += `</div>`;

  document.getElementById('timeline').innerHTML = html;
  document.getElementById('rangeLabel').textContent = fmtTick(vMin, span) + ' — ' + fmtTick(vMax, span);
  document.getElementById('summary').textContent =
    `${{ALL.length}} events · ${{cats.length}} categories · ${{fmtDate(gMin0)}} to ${{fmtDate(gMax0)}}`;
}}

function toggleLane(cat) {{
  if (openLanes.has(cat)) openLanes.delete(cat); else openLanes.add(cat);
  renderDetail();
}}

/* ---- time formatting ---- */
function fmtDate(ms) {{
  return new Date(ms).toLocaleDateString('en-US', {{month:'short',day:'numeric',year:'numeric'}});
}}
function fmtTick(ms, span) {{
  const d = new Date(ms);
  if (span > 86400e3*60) return d.toLocaleDateString('en-US',{{month:'short',year:'2-digit'}});
  if (span > 86400e3*2) return d.toLocaleDateString('en-US',{{month:'short',day:'numeric'}});
  if (span > 36e5*6) return d.toLocaleDateString('en-US',{{month:'short',day:'numeric'}}) + ' ' +
    d.toLocaleTimeString('en-US',{{hour:'2-digit',minute:'2-digit'}});
  return d.toLocaleTimeString('en-US',{{hour:'2-digit',minute:'2-digit',second:'2-digit'}});
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
const AZURE_ENDPOINT = 'https://mmigh-m8gpxf72-eastus2.cognitiveservices.azure.com';
const AZURE_DEPLOYMENT = 'gpt-5.2';
const AZURE_API_VERSION = '2025-01-01-preview';
const AZURE_API_KEY = prompt('Enter your Azure OpenAI API key:') || '';

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
  let prompt = `You are a clinical informatics assistant. Below are medical events for a patient within a selected time window (${{fmtTick(vMin, span)}} to ${{fmtTick(vMax, span)}}). Provide a concise clinical summary of this period. Structure your response with these sections using markdown headers (##):\n\n## Overview\nBrief 1-2 sentence summary of the encounter.\n\n## Key Findings\nImportant lab results, vital signs, or observations.\n\n## Medications & Treatments\nMedications administered, infusions, procedures.\n\n## Diagnoses\nDiagnoses recorded.\n\n## Notable Patterns\nAny trends, concerns, or clinically significant patterns.\n\nUse **bold** for important values. Be concise but thorough.\n\n`;

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
    meta.textContent = `${{visible.length}} events · ${{fmtTick(vMin, span)}} to ${{fmtTick(vMax, span)}}`;
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
    )


def main():
    subject_id = sys.argv[1] if len(sys.argv) > 1 else "10029484"
    print(f"Fetching events for subject {subject_id}...")
    bindings = get_patient_events(subject_id)
    print(f"  {len(bindings)} event rows returned")
    events = parse_events(bindings)
    print(f"  {len(events)} events parsed")

    out_path = os.path.join(os.path.dirname(__file__), f"timeline_{subject_id}.html")
    html = generate_html(subject_id, events)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Timeline saved to {out_path}")


if __name__ == "__main__":
    main()
