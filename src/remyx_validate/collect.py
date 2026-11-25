"""Read what a workload reported, treating every byte as untrusted input.

Accepted, in order:
  1. ``$VALIDATE_OUTPUT_DIR/results.json``
  2. the last ``METRICS_JSON={...}`` line on stdout
  3. the last stdout line that is a JSON object

Shapes:
  * ``{"metrics": {name: number}, "observations": [{"unit": id, "slice": s,
    "metrics": {name: number}}]}``
  * a flat ``{name: number}`` map, read as run-level metrics.

Only finite numbers are kept. Booleans become 0/1. Anything else is listed in
``ignored`` so a dropped value is visible.

A top-level ``"fingerprints": {name: text}`` names what an attempt worked on:
a hash of its inputs, a dataset revision, a prompt set id. Each becomes the
run-level metric ``fingerprint.<name>``, a 48-bit number derived from the
text, so it can be listed under ``validity.held_constant`` and compared
between arms and between repeated runs like any other control.
"""
from __future__ import annotations

import hashlib
import json
import math
import os

MAX_RESULTS_BYTES = 64 * 1024 * 1024
MAX_OBSERVATIONS = 2_000_000
MARKER = "METRICS_JSON="


class ResultsError(Exception):
    pass


def _reject_constant(name):
    raise ResultsError(f"non-finite number {name} in results")


def _loads(text: str):
    return json.loads(text, parse_constant=_reject_constant)


def _number(value):
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            raise ResultsError("non-finite number in results")
        return f
    return None


def _metrics(raw, ignored: list, where: str) -> dict:
    out = {}
    if not isinstance(raw, dict):
        raise ResultsError(f"{where}: metrics must be an object")
    for key, value in raw.items():
        num = _number(value)
        if num is None:
            ignored.append(f"{where}.{key}")
        else:
            out[str(key)] = num
    return out


def _parse_doc(doc, source: str) -> dict:
    if not isinstance(doc, dict):
        raise ResultsError("results must be a JSON object")
    ignored: list = []
    if "observations" in doc or "metrics" in doc or "fingerprints" in doc:
        run_metrics = _metrics(doc.get("metrics") or {}, ignored, "metrics")
        prints = doc.get("fingerprints") or {}
        if not isinstance(prints, dict):
            raise ResultsError("fingerprints must be an object")
        for key, value in prints.items():
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                raise ResultsError(f"fingerprints.{key}: must be text or an integer")
            name = f"fingerprint.{key}"
            if name in run_metrics:
                raise ResultsError(f"fingerprints.{key}: clashes with a metric of the same name")
            run_metrics[name] = float(int(hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12], 16))
        observations = []
        rows = doc.get("observations") or []
        if not isinstance(rows, list):
            raise ResultsError("observations must be a list")
        if len(rows) > MAX_OBSERVATIONS:
            raise ResultsError("too many observations")
        seen = set()
        for i, row in enumerate(rows):
            if not isinstance(row, dict) or "unit" not in row:
                raise ResultsError(f"observations[{i}]: needs a `unit`")
            unit = str(row["unit"])
            if unit in seen:
                raise ResultsError(f"observations[{i}]: duplicate unit {unit!r}")
            seen.add(unit)
            slice_ = row.get("slice")
            observations.append({
                "unit": unit,
                "slice": None if slice_ is None else str(slice_),
                "metrics": _metrics(row.get("metrics") or {}, ignored, f"observations[{i}]"),
            })
    else:
        run_metrics = _metrics(doc, ignored, "metrics")
        observations = []
    return {"metrics": run_metrics, "observations": observations,
            "source": source, "ignored": ignored[:50]}


def read(output_dir: str, stdout_text: str):
    """Returns the parsed results or None when the workload reported nothing."""
    path = os.path.join(output_dir, "results.json")
    if os.path.lexists(path):
        if os.path.islink(path) or not os.path.isfile(path):
            raise ResultsError("results.json must be a regular file, not a link")
        if os.path.getsize(path) > MAX_RESULTS_BYTES:
            raise ResultsError("results.json is larger than the 64 MB limit")
        with open(path, "r", encoding="utf-8") as fh:
            try:
                doc = _loads(fh.read())
            except ValueError as exc:
                raise ResultsError(f"results.json is not valid JSON: {exc}")
        return _parse_doc(doc, "results.json")
    lines = stdout_text.splitlines()
    for line in reversed(lines):
        if line.startswith(MARKER):
            try:
                return _parse_doc(_loads(line[len(MARKER):]), "METRICS_JSON")
            except ValueError:
                continue
    for line in reversed(lines[-200:]):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return _parse_doc(_loads(line), "stdout_json")
            except ValueError:
                continue
    return None


def flatten(parsed: dict, arm: str, seed: int, attempt_id: str) -> list:
    """Turn one attempt's results into observation records."""
    rows = []
    base = {"arm": arm, "seed": seed, "attempt_id": attempt_id, "source": parsed["source"]}
    for metric, value in sorted(parsed["metrics"].items()):
        rows.append({**base, "unit": None, "slice": None, "metric": metric, "value": value})
    for obs in parsed["observations"]:
        for metric, value in sorted(obs["metrics"].items()):
            rows.append({**base, "unit": obs["unit"], "slice": obs["slice"],
                         "metric": metric, "value": value})
    return rows
