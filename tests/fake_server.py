"""Fake llama-server / router for lifecycle tests (stdlib only).

    python fake_server.py --port N [--children K] [--ready-after S] [--exit-after S] [--models a,b]

/health answers 503 until --ready-after seconds have passed, then 200. --children starts K child
processes that just sleep (like router model instances), so tree kill can be checked. With --models
it also answers the router API: GET /models, POST /models/load, POST /models/unload.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--children", type=int, default=0)
    ap.add_argument("--ready-after", type=float, default=0.0)
    ap.add_argument("--exit-after", type=float, default=None)
    ap.add_argument("--models", default="")
    a = ap.parse_args()
    t0 = time.monotonic()
    models = {m: "unloaded" for m in a.models.split(",") if m}
    kids = [subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"]) for _ in range(a.children)]

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):   # noqa: N802
            print("http:", fmt % args, flush=True)

        def _send(self, code: int, body: object) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):   # noqa: N802
            if self.path == "/health":
                if time.monotonic() - t0 < a.ready_after:
                    self._send(503, {"error": {"message": "Loading model"}})
                else:
                    self._send(200, {"status": "ok"})
            elif self.path.startswith("/models"):
                self._send(200, {"data": [{"id": k, "status": {"value": v}} for k, v in models.items()]})
            else:
                self._send(404, {})

        def do_POST(self):   # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            name = body.get("model")
            if name not in models:
                self._send(400, {"error": {"message": f"model {name} not found"}})
                return
            if self.path == "/models/load":
                for k in models:            # models-max 1
                    models[k] = "unloaded"
                models[name] = "loading"
                threading.Timer(0.3, lambda: models.__setitem__(name, "loaded")).start()
            elif self.path == "/models/unload":
                models[name] = "unloaded"
            self._send(200, {"success": True})

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"main: server is listening on http://127.0.0.1:{a.port}", flush=True)
    print("llama_context: n_ctx         = 4096", flush=True)
    print("load_tensors: offloaded 10/10 layers to GPU", flush=True)
    print("some noise line", flush=True)
    if a.exit_after is not None:
        threading.Timer(a.exit_after, lambda: (print("exiting on purpose", flush=True), srv.shutdown())).start()
    try:
        srv.serve_forever()
    finally:
        for k in kids:
            k.kill()
    sys.exit(3)


if __name__ == "__main__":
    main()
