"""Convert SNOMED CT RF2 to Turtle matching BioPortal IRI pattern."""
import csv
import io
import zipfile
import sys

SNOMED_NS = "http://purl.bioontology.org/ontology/SNOMEDCT/"
ISA_TYPE_ID = "116680003"
# FSN typeId=900000000000003001, Synonym typeId=900000000000013009
FSN_TYPE_ID = "900000000000003001"
PREFERRED_TYPE_ID = "900000000000013009"

csv.field_size_limit(10_000_000)

def main():
    zip_path = sys.argv[1]
    out_path = sys.argv[2]

    zf = zipfile.ZipFile(zip_path)

    # Find the relevant files in the zip
    desc_file = [n for n in zf.namelist() if "Snapshot/Terminology/sct2_Description_Snapshot" in n and n.endswith(".txt")][0]
    rel_file = [n for n in zf.namelist() if "Snapshot/Terminology/sct2_Relationship_Snapshot_" in n and n.endswith(".txt")][0]

    print(f"Reading descriptions from {desc_file}...")
    # Extract preferred terms (synonyms) for active concepts
    # We prefer FSN first, then fall back to preferred synonym
    fsn = {}
    synonym = {}
    with zf.open(desc_file) as f:
        reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"), delimiter="\t")
        for row in reader:
            if row["active"] != "1":
                continue
            cid = row["conceptId"]
            if row["typeId"] == FSN_TYPE_ID:
                fsn[cid] = row["term"]
            elif row["typeId"] == PREFERRED_TYPE_ID:
                if cid not in synonym:
                    synonym[cid] = row["term"]

    # Merge: use synonym (shorter) as prefLabel, FSN as notation
    labels = {}
    for cid in set(list(fsn.keys()) + list(synonym.keys())):
        labels[cid] = synonym.get(cid, fsn.get(cid))

    print(f"  {len(labels)} active concepts with labels")

    print(f"Reading relationships from {rel_file}...")
    parents = {}  # conceptId -> set of parent conceptIds
    with zf.open(rel_file) as f:
        reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"), delimiter="\t")
        for row in reader:
            if row["active"] != "1" or row["typeId"] != ISA_TYPE_ID:
                continue
            src = row["sourceId"]
            dst = row["destinationId"]
            parents.setdefault(src, set()).add(dst)

    print(f"  {sum(len(v) for v in parents.values())} active is-a relationships")

    print(f"Writing Turtle to {out_path}...")
    with open(out_path, "w") as out:
        out.write("@prefix skos: <http://www.w3.org/2004/02/skos/core#> .\n")
        out.write("@prefix owl: <http://www.w3.org/2002/07/owl#> .\n")
        out.write("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n")
        out.write("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n")
        out.write(f"@prefix sct: <{SNOMED_NS}> .\n")
        out.write("@prefix umls: <http://bioportal.bioontology.org/ontologies/umls/> .\n\n")

        out.write(f"<{SNOMED_NS}> a owl:Ontology ;\n")
        out.write('    rdfs:label "SNOMEDCT" ;\n')
        out.write('    owl:versionInfo "US Edition 2026-03-01" .\n\n')

        for cid, label in labels.items():
            escaped = label.replace("\\", "\\\\").replace('"', '\\"')
            out.write(f"sct:{cid} a owl:Class ;\n")
            out.write(f'    skos:prefLabel """{escaped}"""@en ;\n')
            out.write(f'    skos:notation """{cid}"""^^xsd:string')
            if cid in parents:
                for pid in parents[cid]:
                    out.write(f" ;\n    rdfs:subClassOf sct:{pid}")
            out.write(" .\n\n")

    print("Done.")

if __name__ == "__main__":
    main()
