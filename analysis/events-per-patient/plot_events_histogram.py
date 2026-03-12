import json
import urllib.request
import urllib.parse
import matplotlib.pyplot as plt
import numpy as np

query = """
PREFIX meds: <https://teamheka.github.io/meds-ontology#>
SELECT ?subject (COUNT(?event) AS ?eventCount)
WHERE {
    ?event a meds:Event ;
           meds:hasSubject ?subject .
}
GROUP BY ?subject
ORDER BY ?eventCount
"""

url = "http://localhost:6335?" + urllib.parse.urlencode({"query": query})
with urllib.request.urlopen(url) as resp:
    data = json.loads(resp.read())

counts = [int(b["eventCount"]["value"]) for b in data["results"]["bindings"]]

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(counts, bins=20, edgecolor="black", color="#4C72B0", alpha=0.85)
ax.set_xlabel("Number of Events", fontsize=13)
ax.set_ylabel("Number of Patients", fontsize=13)
ax.set_title("Distribution of Events per Patient (MIMIC-IV Demo, n=100)", fontsize=14)

mean_val = np.mean(counts)
median_val = np.median(counts)
ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:,.0f}")
ax.axvline(median_val, color="orange", linestyle="--", linewidth=1.5, label=f"Median: {median_val:,.0f}")
ax.legend(fontsize=11)

ax.text(0.97, 0.95, f"Min: {min(counts):,}\nMax: {max(counts):,}\nStd: {np.std(counts):,.0f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.tight_layout()
plt.savefig("events_per_patient_histogram.png", dpi=150)
print("Saved to events_per_patient_histogram.png")
