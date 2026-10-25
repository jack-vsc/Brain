"""Painel do projeto.

  python -m brain.dashboard build [--out dashboard/painel.html]
      grava um HTML único, sem dependências, com o estado atual embutido;
  python -m brain.dashboard serve [--host 127.0.0.1] [--port 8765]
      servidor local que relê runs/ e as filas a cada 20 s. Com --host 0.0.0.0 ele
      fica acessível a outros aparelhos da mesma rede Wi-Fi.
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from brain.dashboard.collect import collect_state
from brain.training.control import pause, resume

PAGE = Path(__file__).with_name("page.html")
CHECKPOINTS = {"transformer": "runs/b_transformer_lr5e-4-s1/best.pt", "ours": "runs/b_fastweight_hops-s1/best.pt"}


def traces(root: Path) -> dict | None:
    """Traços de atividade, recalculados só quando um checkpoint muda."""
    cache = root / "runs" / "_dashboard" / "traces.json"
    paths = {k: root / v for k, v in CHECKPOINTS.items()}
    if not all(p.exists() for p in paths.values()):
        return None
    key = {k: p.stat().st_mtime for k, p in paths.items()}
    if cache.exists():
        old = json.loads(cache.read_text(encoding="utf-8"))
        if old.get("key") == key:
            return old["traces"]
    from brain.dashboard.traces import record_traces  # importa torch só aqui
    t = record_traces({k: Path(v) for k, v in CHECKPOINTS.items()})
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"key": key, "traces": t}, separators=(",", ":")), encoding="utf-8")
    return t


def render(state: dict, tr: dict | None, live: bool) -> str:
    blob = json.dumps({"state": state, "traces": tr, "live": live}, ensure_ascii=False, separators=(",", ":"))
    page = PAGE.read_text(encoding="utf-8").replace("__BRAIN_DATA__", blob.replace("</", "<\\/"))
    return ('<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" '
            'content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body>'
            + page + "</body></html>")


def build(root: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(collect_state(root), traces(root), live=False), encoding="utf-8")
    return out


def serve(root: Path, host: str, port: int) -> None:
    tr = traces(root)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: str, ctype: str):
            data = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/api/state":
                self._send(json.dumps(collect_state(root), ensure_ascii=False), "application/json")
            elif path in ("/", "/index.html"):
                self._send(render(collect_state(root), tr, live=True), "text/html")
            else:
                self.send_error(404)

        def do_POST(self):
            # Pausar/continuar a fila. O cabeçalho próprio barra formulários de outros sites
            # (um pedido com cabeçalho customizado exige preflight, que este servidor não responde).
            path = self.path.split("?", 1)[0]
            if self.headers.get("X-Brain") != "1" or path not in ("/api/pause", "/api/resume"):
                self.send_error(404)
                return
            (pause if path == "/api/pause" else resume)(root / "runs")
            self._send(json.dumps(collect_state(root), ensure_ascii=False), "application/json")

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"painel em http://{host}:{port}  (Ctrl+C para sair)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m brain.dashboard", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, default=Path("dashboard/painel.html"))
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    for p in (b, s):
        p.add_argument("--root", type=Path, default=Path("."))
    args = ap.parse_args()
    if args.cmd == "build":
        t = time.perf_counter()
        out = build(args.root, args.out)
        print(f"painel gravado em {out} ({out.stat().st_size / 1024:.0f} KB, {time.perf_counter() - t:.1f} s)")
    else:
        serve(args.root, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
