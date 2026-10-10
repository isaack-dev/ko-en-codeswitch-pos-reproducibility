"""
clean_run_compare.py - Reusable tolerance-comparison helpers for the final
clean reproducibility pipeline's Phase 8 (compare clean-run artifacts against
already-committed authoritative outputs).

This module contains NO orchestration logic (no training, no prediction, no
knowledge of specific file paths). It is deliberately small and pure so it
can be unit-tested directly with synthetic fixtures (see
scripts/tests/test_final_clean_reproducibility_pipeline.py), and reused from
scripts/run_final_clean_reproducibility_pipeline.py's Phase 8 step.

Row schema (matches the task specification exactly):
  artifact, field, authoritative_value, clean_run_value, absolute_difference,
  tolerance, status, explanation

status is one of:
  exact_match, within_tolerance, expected_serialization_difference,
  unexplained_discrepancy, missing_artifact
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

STATUS_EXACT = "exact_match"
STATUS_TOLERANCE = "within_tolerance"
STATUS_SERIALIZATION = "expected_serialization_difference"
STATUS_UNEXPLAINED = "unexplained_discrepancy"
STATUS_MISSING = "missing_artifact"

BLOCKING_STATUSES = {STATUS_UNEXPLAINED, STATUS_MISSING}

ROW_FIELDS = [
    "artifact", "field", "authoritative_value", "clean_run_value",
    "absolute_difference", "tolerance", "status", "explanation",
]


def _try_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def compare_row(
    artifact: str,
    field: str,
    authoritative_value: Any,
    clean_run_value: Any,
    tolerance: float = 0.0,
    explanation: str = "",
    force_status: str | None = None,
) -> dict:
    """Build one comparison row applying the project's comparison policy.

    - clean_run_value is None (or explicitly marked absent) -> missing_artifact
    - both values parse as float -> compare abs(diff) against tolerance
    - otherwise -> exact string/repr equality required
    - force_status lets a caller record a pre-classified status (e.g.
      expected_serialization_difference for a known-benign difference) while
      still going through this single row-construction path.
    """
    if clean_run_value is None:
        return {
            "artifact": artifact, "field": field,
            "authoritative_value": authoritative_value, "clean_run_value": None,
            "absolute_difference": None, "tolerance": tolerance,
            "status": STATUS_MISSING,
            "explanation": explanation or "clean-run value not found",
        }

    if force_status is not None:
        a_f = _try_float(authoritative_value)
        c_f = _try_float(clean_run_value)
        diff = abs(a_f - c_f) if (a_f is not None and c_f is not None) else None
        return {
            "artifact": artifact, "field": field,
            "authoritative_value": authoritative_value, "clean_run_value": clean_run_value,
            "absolute_difference": diff, "tolerance": tolerance,
            "status": force_status, "explanation": explanation,
        }

    a_f = _try_float(authoritative_value)
    c_f = _try_float(clean_run_value)

    if a_f is not None and c_f is not None:
        diff = abs(a_f - c_f)
        if diff == 0.0:
            status = STATUS_EXACT
        elif diff <= tolerance:
            status = STATUS_TOLERANCE
        else:
            status = STATUS_UNEXPLAINED
        return {
            "artifact": artifact, "field": field,
            "authoritative_value": authoritative_value, "clean_run_value": clean_run_value,
            "absolute_difference": diff, "tolerance": tolerance,
            "status": status, "explanation": explanation,
        }

    # Non-numeric: require exact equality (strings, bools, integer counts
    # already handled by the float path above, tuples/lists as their repr).
    equal = authoritative_value == clean_run_value
    return {
        "artifact": artifact, "field": field,
        "authoritative_value": authoritative_value, "clean_run_value": clean_run_value,
        "absolute_difference": 0 if equal else None, "tolerance": tolerance,
        "status": STATUS_EXACT if equal else STATUS_UNEXPLAINED,
        "explanation": explanation,
    }


def compare_dicts_exact(
    artifact: str,
    authoritative: dict,
    clean_run: dict,
    fields: Iterable[str],
    explanation: str = "",
) -> list[dict]:
    """Exact-equality comparison of named fields between two dict-like rows."""
    rows = []
    for f in fields:
        a_val = authoritative.get(f) if authoritative is not None else None
        c_val = clean_run.get(f) if clean_run is not None else None
        rows.append(compare_row(artifact, f, a_val, c_val, tolerance=0.0, explanation=explanation))
    return rows


def has_blocking_discrepancies(rows: Iterable[dict]) -> bool:
    return any(r.get("status") in BLOCKING_STATUSES for r in rows)


def summarize(rows: Iterable[dict]) -> dict:
    counts = {STATUS_EXACT: 0, STATUS_TOLERANCE: 0, STATUS_SERIALIZATION: 0,
              STATUS_UNEXPLAINED: 0, STATUS_MISSING: 0}
    for r in rows:
        counts[r.get("status", STATUS_UNEXPLAINED)] = counts.get(r.get("status"), 0) + 1
    return counts


def write_comparison_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ROW_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in ROW_FIELDS})


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def index_csv_rows(rows: list[dict], key_fields: list[str]) -> dict[tuple, dict]:
    return {tuple(r.get(k) for k in key_fields): r for r in rows}


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
