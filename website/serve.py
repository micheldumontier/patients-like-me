"""Simple HTTP server that serves website/ under the /mimic-miner/ path prefix."""
import http.server
import socketserver
import os

PORT = 9000
PREFIX = "/mimic-miner/"
DIRECTORY = os.path.dirname(os.path.abspath(__file__))


class PrefixHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        if self.path == PREFIX.rstrip("/"):
            self.send_response(301)
            self.send_header("Location", PREFIX)
            self.end_headers()
            return
        if self.path.startswith(PREFIX):
            self.path = self.path[len(PREFIX) - 1:]
        elif self.path == "/":
            self.send_response(301)
            self.send_header("Location", PREFIX)
            self.end_headers()
            return
        else:
            self.send_error(404)
            return
        super().do_GET()


class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    with ThreadedServer(("0.0.0.0", PORT), PrefixHandler) as httpd:
        print(f"Serving website at http://localhost:{PORT}{PREFIX}")
        httpd.serve_forever()
