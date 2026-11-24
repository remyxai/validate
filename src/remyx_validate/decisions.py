"""Human decisions, kept apart from verdicts and from merges.

Append-only ``.validate/decisions.jsonl``. A later decision may name the one
it supersedes; nothing is ever rewritten.
"""
from __future__ import annotations

import json
import os

from . import SPEC_VERSION, bundle, declaration as decl
from .canonical import digest, now_iso

CHOICES = ("ship", "revise", "hold", "drop")


def path(project: str) -> str:
    return os.path.join(project, ".validate", "decisions.jsonl")


def load(project: str) -> list:
    p = path(project)
    if not os.path.isfile(p):
        return []
    with open(p, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def record(project: str, run_dir: str, decision: str, reason: str, author: str,
           supersedes: str = "") -> dict:
    if decision not in CHOICES:
        raise ValueError(f"decision must be one of {', '.join(CHOICES)}")
    if not reason.strip():
        raise ValueError("a decision needs a reason")
    existing = load(project)
    if supersedes and not any(d["decision_id"] == supersedes for d in existing):
        raise ValueError(f"no decision {supersedes} to supersede")
    evidence = bundle.load_evidence(run_dir)
    doc = {
        "spec_version": SPEC_VERSION,
        "kind": "decision",
        "decision": decision,
        "reason": reason.strip(),
        "author": author,
        "recorded_at": now_iso(),
        "supersedes": supersedes or None,
        "evidence": [{
            "run_id": evidence["execution"]["run_id"],
            "bundle_digest": bundle.bundle_digest(run_dir),
            "change_id": evidence["change"]["change_id"],
            "declaration_hash": evidence["declaration"]["declaration_hash"],
            "verdict": evidence["comparison"]["verdict"],
        }],
    }
    doc["decision_id"] = "dec_" + digest(doc)[:12]
    problems = decl.schema_problems(doc, "decision")
    if problems:
        raise ValueError("; ".join(problems))
    os.makedirs(os.path.dirname(path(project)), exist_ok=True)
    with open(path(project), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(doc, sort_keys=True) + "\n")
    return doc


def current(project: str) -> list:
    """Decisions that no later decision supersedes."""
    all_ = load(project)
    replaced = {d["supersedes"] for d in all_ if d.get("supersedes")}
    return [d for d in all_ if d["decision_id"] not in replaced]
