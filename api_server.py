import argparse
import json
import sqlite3
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fraud_monitor.constants import risk_level
from fraud_monitor.models import feature_evidence, predict_probability


class FraudAPI(BaseHTTPRequestHandler):
    database: Path
    artifact: dict

    def _json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def _body(self):
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            return self._json({"status": "ok", "model_version": self.artifact["model_version"]})
        if parsed.path == "/alerts":
            query = parse_qs(parsed.query); limit = min(int(query.get("limit", [100])[0]), 1000)
            risk = query.get("risk", [None])[0]
            sql = "SELECT * FROM alerts"; params = []
            if risk: sql += " WHERE risk_level = ?"; params.append(risk)
            sql += " ORDER BY fraud_probability DESC LIMIT ?"; params.append(limit)
            with sqlite3.connect(self.database) as connection:
                connection.row_factory = sqlite3.Row
                return self._json([dict(row) for row in connection.execute(sql, params)])
        if parsed.path.startswith("/alerts/"):
            transaction_id = parsed.path.rsplit("/", 1)[-1]
            with sqlite3.connect(self.database) as connection:
                connection.row_factory = sqlite3.Row
                row = connection.execute("SELECT * FROM alerts WHERE transaction_id = ?", (transaction_id,)).fetchone()
            return self._json(dict(row) if row else {"error": "alert not found"}, HTTPStatus.OK if row else HTTPStatus.NOT_FOUND)
        return self._json({"error": "route not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if self.path != "/transactions/analyze":
            return self._json({"error": "route not found"}, HTTPStatus.NOT_FOUND)
        try:
            payload = self._body(); model = self.artifact["model"]
            missing = [name for name in model["features"] if name not in payload]
            if missing: return self._json({"error": "missing features", "fields": missing}, HTTPStatus.BAD_REQUEST)
            frame = pd.DataFrame([{name: payload[name] for name in model["features"]}])
            probability = float(predict_probability(model, frame)[0])
            return self._json({"fraud_probability": probability, "alert": probability >= self.artifact["threshold"], "risk_level": risk_level(probability, self.artifact["threshold"]), "evidence": feature_evidence(model, frame.iloc[0]), "model_version": self.artifact["model_version"]})
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            return self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def do_PATCH(self):
        if not self.path.startswith("/alerts/"):
            return self._json({"error": "route not found"}, HTTPStatus.NOT_FOUND)
        status = self._body().get("review_status")
        if status not in {"pending", "confirmed", "dismissed", "investigating"}:
            return self._json({"error": "invalid review_status"}, HTTPStatus.BAD_REQUEST)
        transaction_id = self.path.rsplit("/", 1)[-1]
        with sqlite3.connect(self.database) as connection:
            cursor = connection.execute("UPDATE alerts SET review_status = ? WHERE transaction_id = ?", (status, transaction_id))
        return self._json({"transaction_id": int(transaction_id), "review_status": status}, HTTPStatus.OK if cursor.rowcount else HTTPStatus.NOT_FOUND)

    def log_message(self, message, *args):
        print(f"[api] {message % args}")


def main():
    parser = argparse.ArgumentParser(description="Fraud Transaction Monitor API")
    parser.add_argument("--db", type=Path, default=Path("outputs/fraud_detection.db"))
    parser.add_argument("--model", type=Path, default=Path("outputs/fraud_model.json"))
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    FraudAPI.database = args.db; FraudAPI.artifact = json.loads(args.model.read_text(encoding="utf-8"))
    print(f"API disponível em http://localhost:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), FraudAPI).serve_forever()


if __name__ == "__main__":
    main()
