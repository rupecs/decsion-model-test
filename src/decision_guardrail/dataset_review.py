"""Dependency-free visual review; annotations are separate from frozen source data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import secrets
from socketserver import TCPServer
import tempfile
import threading
from typing import Any
from urllib.parse import urlsplit

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
CLASSES = {"factual_information", "general_advice", "personal_advice", "other"}
STATUSES = {"unreviewed", "checked", "needs_review"}


class ReviewConflict(ValueError):
    pass


def load_dataset(project: Path = ROOT) -> dict[str, Any]:
    pointer = json.loads((project / "data/FINAL_DATASET.json").read_text())
    master = project / pointer["master"]
    raw = master.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != pointer["master_sha256"]:
        raise ValueError("Master dataset hash differs from FINAL_DATASET.json")
    records = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        row = json.loads(line)
        records.append({**row, "master_line": line_number})
    if len(records) != pointer["records"] or len({r["id"] for r in records}) != len(records):
        raise ValueError("Dataset count or record IDs are invalid")
    return {
        "version": pointer["version"], "sha256": digest,
        "source": pointer["master"], "label_status": pointer["label_status"],
        "questions": json.loads((project / pointer["questions"]).read_text()),
        "records": records,
    }


def render_html(dataset: dict[str, Any], live: dict[str, Any] | None = None) -> bytes:
    payload = json.dumps({"dataset": dataset, "live": live}, ensure_ascii=False)
    # A record/comment can never terminate its JSON script element.
    payload = payload.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = Path(__file__).with_name("reviewer.html").read_text()
    return template.replace("__REVIEW_PAYLOAD__", payload).encode("utf-8")


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".review-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class ReviewStore:
    """One JSONL row per record, atomic saves and optimistic concurrency control."""

    def __init__(self, path: Path, dataset: dict[str, Any]):
        self.path = path.resolve()
        source = (ROOT / dataset["source"]).resolve()
        if self.path == source or (ROOT / "data/generated").resolve() in self.path.parents:
            raise ValueError("Review notes must be outside generated datasets")
        self.digest = dataset["sha256"]
        self.version = dataset["version"]
        self.records = {r["id"]: r for r in dataset["records"]}
        self.lock = threading.Lock()
        self.reviews: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                item = json.loads(line)
                if item["dataset_sha256"] != self.digest or item["record_id"] not in self.records:
                    raise ValueError("Existing comments refer to a different dataset")
                if item["record_id"] in self.reviews:
                    raise ValueError("Duplicate record ID in existing review file")
                self.reviews[item["record_id"]] = item

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {"reviews": list(self.reviews.values()), "path": str(self.path)}

    def save(self, body: dict[str, Any]) -> dict[str, Any]:
        record_id = body.get("record_id")
        if not isinstance(record_id, str) or record_id not in self.records:
            raise ValueError("Unknown record ID")
        if body.get("dataset_sha256") != self.digest:
            raise ValueError("Dataset hash does not match")
        comment = body.get("comment", "")
        reviewer = body.get("reviewer", "")
        status = body.get("status", "unreviewed")
        proposed = body.get("proposed_labels", {})
        if not isinstance(comment, str) or len(comment) > 20000:
            raise ValueError("Comments must be text, up to 20,000 characters")
        if not isinstance(reviewer, str) or len(reviewer) > 120:
            raise ValueError("Reviewer name must be text, up to 120 characters")
        if not isinstance(status, str) or status not in STATUSES:
            raise ValueError("Invalid review status")
        if not isinstance(proposed, dict):
            raise ValueError("Invalid suggested labels")
        for key, value in proposed.items():
            if key == "D6":
                if not isinstance(value, str) or value not in CLASSES:
                    raise ValueError("Invalid suggested classification")
            elif key not in {"D1", "D2", "D3", "D4", "D5"} or type(value) is not bool:
                raise ValueError("Invalid suggested decision")
        with self.lock:
            previous = self.reviews.get(record_id, {})
            expected = body.get("revision", 0)
            if type(expected) is not int or expected != previous.get("revision", 0):
                raise ReviewConflict("This note changed in another tab. Export your draft before reloading.")
            row = self.records[record_id]
            now = datetime.now(timezone.utc).isoformat()
            item = {
                "schema_version": "dataset_review_v1", "dataset_version": self.version,
                "dataset_sha256": self.digest, "record_id": record_id,
                "master_line": row["master_line"], "split": row["metadata"]["split"],
                "original_labels": row["labels"], "status": status,
                "comment": comment, "proposed_labels": proposed, "reviewer": reviewer,
                "revision": expected + 1, "created_at": previous.get("created_at", now),
                "updated_at": now,
            }
            updated = {**self.reviews, record_id: item}
            contents = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in
                               sorted(updated.values(), key=lambda r: r["master_line"]))
            atomic_write(self.path, contents)
            self.reviews = updated
            return item


def make_server(dataset: dict[str, Any], notes: Path, port: int = 8765) -> ThreadingHTTPServer:
    store = ReviewStore(notes, dataset)
    token = secrets.token_urlsafe(32)
    html = render_html(dataset, {"token": token, "path": str(store.path)})

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            # Request bodies and dataset text are never logged.
            LOG.debug(fmt, *args)

        def respond(self, code: int, data: Any, kind: str = "application/json; charset=utf-8") -> None:
            raw = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src data:; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(raw)

        def valid_host(self) -> bool:
            return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

        def do_GET(self) -> None:
            if not self.valid_host():
                self.respond(403, {"error": "Use the displayed 127.0.0.1 URL"})
                return
            path = urlsplit(self.path).path
            if path in {"/", "/index.html"}:
                self.respond(200, html, "text/html; charset=utf-8")
            elif path == "/api/reviews":
                self.respond(200, store.snapshot())
            else:
                self.respond(404, {"error": "Not found"})

        def do_POST(self) -> None:
            if not self.valid_host() or self.headers.get("X-Review-Token") != token:
                self.respond(403, {"error": "Invalid review session"})
                return
            origin = self.headers.get("Origin")
            if origin and origin != f"http://127.0.0.1:{self.server.server_port}":
                self.respond(403, {"error": "Invalid origin"})
                return
            if urlsplit(self.path).path != "/api/reviews":
                self.respond(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 200000:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Expected a review object")
                self.respond(200, store.save(body))
            except ReviewConflict as error:
                self.respond(409, {"error": str(error)})
            except (ValueError, TypeError) as error:
                self.respond(400, {"error": str(error)})
            except OSError:
                LOG.exception("Could not save review notes")
                self.respond(500, {"error": "Could not write the review file. Your draft is still on screen."})

    class LocalReviewServer(ThreadingHTTPServer):
        def server_bind(self) -> None:
            # HTTPServer's reverse DNS lookup can stall for a minute on an offline Mac.
            TCPServer.server_bind(self)
            self.server_name, self.server_port = self.server_address[:2]

    return LocalReviewServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765, help="127.0.0.1 port; 0 selects a free port")
    parser.add_argument("--comments", type=Path, default=ROOT / "data/reviews/manual_v6/comments.jsonl")
    parser.add_argument("--html", type=Path, default=ROOT / "data/review_ui/pilot_v6_review.html")
    parser.add_argument("--build-only", action="store_true", help="Write the standalone HTML and exit")
    args = parser.parse_args()
    dataset = load_dataset()
    offline = render_html(dataset)
    args.html.parent.mkdir(parents=True, exist_ok=True)
    if args.html.exists():
        if args.html.read_bytes() != offline:
            parser.error("HTML output already exists with different contents; choose a new --html path")
    else:
        with args.html.open("xb") as stream:
            stream.write(offline)
    print(f"Standalone HTML: {args.html.resolve()}", flush=True)
    if args.build_only:
        return
    server = make_server(dataset, args.comments, args.port)
    print(f"Reviewer: http://127.0.0.1:{server.server_port}", flush=True)
    print(f"Comments: {args.comments.resolve()}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
