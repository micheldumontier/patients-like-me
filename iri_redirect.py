"""
IRI redirect service for MIMIC-IV RDF data.

Intercepts requests matching known RDF namespace prefixes and redirects
to the QLever UI with a DESCRIBE-style query for that entity.

Run:  python iri_redirect.py
Port: 9003 (configurable below)
"""
import http.server
import socketserver
import urllib.parse

PORT = 6337
QLEVER_UI = "http://localhost:6336/default"

# Map local path prefixes to full IRI base URIs
PREFIX_MAP = {
    "/meds-data/":      "https://teamheka.github.io/meds-data/",
    "/meds-ontology":   "https://teamheka.github.io/meds-ontology",
    "/ICD9CM/":         "http://purl.bioontology.org/ontology/ICD9CM/",
    "/ICD10CM/":        "http://purl.bioontology.org/ontology/ICD10CM/",
    "/ICD10PCS/":       "http://purl.bioontology.org/ontology/ICD10PCS/",
    "/RXNORM/":         "http://purl.bioontology.org/ontology/RXNORM/",
    "/SNOMEDCT/":       "http://purl.bioontology.org/ontology/SNOMEDCT/",
    "/LNC/":            "http://purl.bioontology.org/ontology/LNC/",
    "/STY/":            "http://purl.bioontology.org/ontology/STY/",
}


def build_query(iri):
    return (
        f"SELECT ?p ?o WHERE {{\n"
        f"  <{iri}> ?p ?o .\n"
        f"}}"
    )


class RedirectHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        # Try to match the request path to a known prefix
        for prefix, base_iri in PREFIX_MAP.items():
            if self.path.startswith(prefix):
                full_iri = base_iri + self.path[len(prefix):]
                query = build_query(full_iri)
                redirect_url = f"{QLEVER_UI}/?query={urllib.parse.quote(query)}"
                self.send_response(302)
                self.send_header("Location", redirect_url)
                self.end_headers()
                return

        # If path has a ?uri= parameter, use that directly
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        if "uri" in params:
            full_iri = params["uri"][0]
            query = build_query(full_iri)
            redirect_url = f"{QLEVER_UI}/?query={urllib.parse.quote(query)}"
            self.send_response(302)
            self.send_header("Location", redirect_url)
            self.end_headers()
            return

        # No match — show usage
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        usage = (
            "<h2>IRI Redirect Service</h2>"
            "<p>Redirects RDF IRIs to the QLever UI.</p>"
            "<h3>Usage</h3>"
            "<ul>"
            "<li><a href='/meds-data/subject/10000032'>/meds-data/subject/10000032</a></li>"
            "<li><a href='/meds-data/code/DIAGNOSIS_ICD_9_038.0'>/meds-data/code/DIAGNOSIS_ICD_9_038.0</a></li>"
            "<li><a href='/ICD9CM/038.0'>/ICD9CM/038.0</a></li>"
            "<li><a href='/ICD10CM/A41.9'>/ICD10CM/A41.9</a></li>"
            "<li><a href='/RXNORM/161'>/RXNORM/161</a></li>"
            "<li><a href='/SNOMEDCT/38341003'>/SNOMEDCT/38341003</a></li>"
            "<li><code>?uri=&lt;full_iri&gt;</code> for any IRI</li>"
            "</ul>"
        )
        self.wfile.write(usage.encode())

    def log_message(self, format, *args):
        print(f"[redirect] {args[0]}")


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    server = ThreadedHTTPServer(("127.0.0.1", PORT), RedirectHandler)
    print(f"IRI redirect service running on http://localhost:{PORT}")
    print(f"Redirecting to QLever UI at {QLEVER_UI}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
