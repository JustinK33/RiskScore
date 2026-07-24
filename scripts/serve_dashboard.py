"""Serve the local credit risk dashboard."""

from __future__ import annotations

import argparse
import csv
import io
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from risk_score.config import load_yaml_config
from risk_score.evaluation import CostMatrix
from risk_score.pipeline import run_baseline_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_ROOT = PROJECT_ROOT / "dashboard"
REPORTS_ROOT = PROJECT_ROOT / "reports"
UPLOAD_ROOT = PROJECT_ROOT / "data" / "raw" / "uploads"


class DashboardHandler(SimpleHTTPRequestHandler):
    """HTTP handler for dashboard assets and report APIs."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, directory=str(DASHBOARD_ROOT), **kwargs)

    def log_message(self, format: str, *args: object) -> None:
        """Keep local dashboard logs compact."""
        print(f"{self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        """Route API and artifact requests before static file handling."""
        path = unquote(self.path.split("?", 1)[0])
        if path == "/api/metrics":
            self._send_json_file(REPORTS_ROOT / "metrics" / "logistic_regression_metrics.json")
            return
        if path == "/api/calibration":
            self._send_calibration_csv(
                REPORTS_ROOT / "metrics" / "logistic_regression_calibration.csv"
            )
            return
        if path.startswith("/artifacts/"):
            self._send_artifact(path.removeprefix("/artifacts/"))
            return
        super().do_GET()

    def do_POST(self) -> None:
        """Handle local CSV uploads and pipeline runs."""
        path = unquote(self.path.split("?", 1)[0])
        if path == "/api/run":
            self._run_uploaded_dataset()
            return
        self.send_error(404, "Route not found")

    def _send_json_file(self, path: Path) -> None:
        if not path.exists():
            self.send_error(404, f"Missing report file: {path.relative_to(PROJECT_ROOT)}")
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._send_json(payload)

    def _send_calibration_csv(self, path: Path) -> None:
        if not path.exists():
            self.send_error(404, f"Missing report file: {path.relative_to(PROJECT_ROOT)}")
            return
        with path.open(newline="", encoding="utf-8") as file:
            payload = list(csv.DictReader(file))
        self._send_json(payload)

    def _send_artifact(self, relative_path: str) -> None:
        target = (REPORTS_ROOT / relative_path).resolve()
        if not target.is_relative_to(REPORTS_ROOT.resolve()) or not target.exists():
            self.send_error(404, "Artifact not found")
            return
        self.path = f"/{target.relative_to(REPORTS_ROOT)}"
        self.directory = str(REPORTS_ROOT)
        super().do_GET()

    def _run_uploaded_dataset(self) -> None:
        try:
            payload = self._read_json_body()
            csv_text = str(payload["csv_text"])
            filename = str(payload.get("filename") or "uploaded_loans.csv")
            train_end_date = str(payload["train_end_date"])
            test_start_date = str(payload["test_start_date"])
            raw_path = self._write_upload(filename, csv_text)
            config = load_yaml_config(PROJECT_ROOT / "configs" / "model_config.yaml")
            schema_config = load_yaml_config(PROJECT_ROOT / "configs" / "dataset_schema.yaml")
            selected_model_config = config.get("logistic_regression", {})
            metrics = run_baseline_pipeline(
                raw_path,
                train_end_date=train_end_date,
                test_start_date=test_start_date,
                output_dir=REPORTS_ROOT,
                model_type="logistic_regression",
                model_config=selected_model_config,
                schema_config=schema_config,
                cost_matrix=CostMatrix(
                    false_negative_cost=5.0,
                    false_positive_cost=1.0,
                ),
            )
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)
            return

        self._send_json(
            {
                "ok": True,
                "uploaded_path": str(raw_path.relative_to(PROJECT_ROOT)),
                "metrics": {
                    "auc_roc": metrics.auc_roc,
                    "average_precision": metrics.average_precision,
                    "ks_statistic": metrics.ks_statistic,
                },
            }
        )

    def _read_json_body(self) -> dict[str, object]:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def _write_upload(self, filename: str, csv_text: str) -> Path:
        upload_name = Path(filename).name
        if not upload_name.lower().endswith(".csv"):
            upload_name = f"{upload_name}.csv"
        UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
        raw_path = UPLOAD_ROOT / upload_name
        safe_text = csv_text.replace("\r\n", "\n")
        with io.StringIO(safe_text) as buffer:
            sample = buffer.readline()
        if "," not in sample:
            raise ValueError("Uploaded file does not look like a CSV.")
        raw_path.write_text(safe_text, encoding="utf-8")
        return raw_path

    def _send_json(self, payload: object, *, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    """Parse dashboard server arguments."""
    parser = argparse.ArgumentParser(description="Serve the credit risk dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    """Start the local dashboard server."""
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
