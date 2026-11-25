"""The portable evidence bundle: write, verify, redact, export.

A bundle is a directory:

    evidence.json        the record (change, declaration, execution, comparison)
    manifest.json        the exact approved plan plus the approval
    declaration.json     the resolved declaration that was judged
    observations.jsonl   one observation per line
    report.md            the same result for a person
    attempts/…           logs and raw outputs of every attempt
    checksums.json       sha256 of every file above, and a bundle digest

``verify`` needs nothing but the bundle. It rechecks the files, the schemas
and the identities that tie the files together, and recomputes the whole
comparison from the observations. It establishes that a bundle is internally
consistent. It cannot establish who ran it, on what, or that the workload
measured honestly.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import sys
import tarfile
import tempfile

from . import SPEC_VERSION, __version__, compare, declaration as decl, report, schedule
from .canonical import canonical_json, digest, sha256_file, tree_digest

EXCLUDED = ("status.json", "status.json.tmp", "checksums.json", "decisions.jsonl")

SECRET_PATTERNS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"rmx[a-z]_[A-Za-z0-9]{16,}"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password|authorization)(\"?\s*[:=]\s*\"?)[^\s\"',]{8,}"),
]


EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")


def _dump(path: str, doc) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _files(root: str) -> dict:
    out = {}
    for base, dirs, names in os.walk(root, followlinks=False):
        dirs.sort()
        for name in sorted(names):
            full = os.path.join(base, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if rel in EXCLUDED:
                continue
            if os.path.islink(full):
                out[rel] = "symlink:" + os.readlink(full)
            elif os.path.isfile(full):
                out[rel] = sha256_file(full)
    return out


def write_checksums(run_dir: str, extra: dict = None) -> dict:
    files = _files(run_dir)
    doc = {"algorithm": "sha256", "files": files, "bundle_digest": digest(files)}
    doc.update(extra or {})
    _dump(os.path.join(run_dir, "checksums.json"), doc)
    return doc


def write(run_dir, plan, run_id, approval, execution, observations, comparison, cost) -> dict:
    d = plan["declaration"]
    obs_path = os.path.join(run_dir, "observations.jsonl")
    with open(obs_path, "w", encoding="utf-8") as fh:
        for row in observations:
            fh.write(canonical_json(row) + "\n")
    _dump(os.path.join(run_dir, "manifest.json"), {"plan": plan, "approval": approval, "run_id": run_id})
    _dump(os.path.join(run_dir, "declaration.json"), d)
    evidence = {
        "spec_version": SPEC_VERSION,
        "kind": "evidence",
        "change": {**plan["change"],
                   "family": d["change"].get("family", ""),
                   "hypothesis": d["change"].get("hypothesis", ""),
                   "pr": d["change"].get("pr", "")},
        "declaration": {"declaration_hash": plan["declaration_hash"], "resolved": d},
        "execution": {
            "run_id": run_id,
            "spec_hash": plan["spec_hash"],
            "plan_digest": plan["plan_digest"],
            "env_keys": plan["env_keys"],
            "harness": plan["harness"],
            **execution,
        },
        "observations": {"path": "observations.jsonl", "count": len(observations),
                         "sha256": sha256_file(obs_path)},
        "comparison": comparison,
        "cost": cost,
        "provenance": {
            "producer": {"name": "remyx-validate", "version": __version__,
                         "python": platform.python_version(), "platform": sys.platform},
            "trust_level": "self_reported_local",
            "approved_by": approval.get("approved_by", ""),
            "approved_at": approval.get("approved_at", ""),
        },
    }
    _dump(os.path.join(run_dir, "evidence.json"), evidence)
    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(report.render(evidence))
    sums = write_checksums(run_dir)
    return {"bundle_digest": sums["bundle_digest"], "evidence": evidence}


def _strict_json(text: str):
    def reject(name):
        raise ValueError(f"non-finite number {name}")
    return json.loads(text, parse_constant=reject)


def load_evidence(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "evidence.json"), encoding="utf-8") as fh:
        return _strict_json(fh.read())


def load_observations(run_dir: str) -> list:
    rows = []
    with open(os.path.join(run_dir, "observations.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(_strict_json(line))
    return rows


def bundle_digest(run_dir: str) -> str:
    with open(os.path.join(run_dir, "checksums.json"), encoding="utf-8") as fh:
        return json.load(fh)["bundle_digest"]


# ── verify ────────────────────────────────────────────────────────────────

REQUIRED = ("evidence.json", "manifest.json", "declaration.json", "observations.jsonl", "report.md")
MAX_MEMBERS = 20000
MAX_MEMBER_BYTES = 512 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024


class BundleError(Exception):
    pass


def verify(path: str) -> dict:
    """Check a bundle directory or .tar.gz. Returns {ok, problems, …}. A
    malformed bundle is a failed result, never an exception."""
    tmp = None
    try:
        if os.path.isfile(path):
            tmp = tempfile.mkdtemp(prefix="validate-verify-")
            try:
                with tarfile.open(path, "r:gz") as tar:
                    _safe_extract(tar, tmp)
            except (tarfile.TarError, OSError, EOFError) as exc:
                raise BundleError(f"not a readable .tar.gz bundle: {exc}")
            entries = os.listdir(tmp)
            path = os.path.join(tmp, entries[0]) if len(entries) == 1 else tmp
        elif not os.path.isdir(path):
            raise BundleError(f"no bundle at {path}")
        return _verify_dir(path)
    except BundleError as exc:
        return {"ok": False, "problems": [str(exc)], "path": path}
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def _safe_extract(tar: tarfile.TarFile, dest: str) -> None:
    root = os.path.realpath(dest)
    seen, total = set(), 0
    members = tar.getmembers()
    if len(members) > MAX_MEMBERS:
        raise BundleError("archive has too many members")
    for member in members:
        target = os.path.realpath(os.path.join(dest, member.name))
        if not (target == root or target.startswith(root + os.sep)):
            raise BundleError(f"archive member escapes the destination: {member.name}")
        if not (member.isreg() or member.isdir()):
            raise BundleError(f"archive member is not a regular file or directory: {member.name}")
        if target in seen:
            raise BundleError(f"archive lists a path twice: {member.name}")
        seen.add(target)
        if member.size > MAX_MEMBER_BYTES:
            raise BundleError(f"archive member is too large: {member.name}")
        total += member.size
        if total > MAX_TOTAL_BYTES:
            raise BundleError("archive is larger than the extraction limit")
    for member in members:
        member.mode = 0o755 if member.isdir() else 0o644
        try:
            tar.extract(member, dest, set_attrs=False, filter="data")
        except TypeError:   # Python builds that predate extraction filters
            tar.extract(member, dest, set_attrs=False)


def _differences(recorded, recomputed, path="comparison", out=None, tol=1e-9):
    """Every place two JSON values disagree, with a relative tolerance on numbers."""
    out = [] if out is None else out
    if len(out) >= 12:
        return out
    numeric = (int, float)
    if isinstance(recorded, bool) or isinstance(recomputed, bool):
        if recorded is not recomputed:
            out.append(f"{path}: recorded {recorded!r}, recomputed {recomputed!r}")
    elif isinstance(recorded, numeric) and isinstance(recomputed, numeric):
        if abs(recorded - recomputed) > tol * max(1.0, abs(recorded), abs(recomputed)):
            out.append(f"{path}: recorded {recorded!r}, recomputed {recomputed!r}")
    elif isinstance(recorded, dict) and isinstance(recomputed, dict):
        for key in sorted(set(recorded) | set(recomputed)):
            if key not in recorded:
                out.append(f"{path}.{key}: absent from the record")
            elif key not in recomputed:
                out.append(f"{path}.{key}: not something the observations produce")
            else:
                _differences(recorded[key], recomputed[key], f"{path}.{key}", out, tol)
    elif isinstance(recorded, list) and isinstance(recomputed, list):
        if len(recorded) != len(recomputed):
            out.append(f"{path}: recorded {len(recorded)} item(s), recomputed {len(recomputed)}")
        else:
            for i, (a, b) in enumerate(zip(recorded, recomputed)):
                label = a.get("id", i) if isinstance(a, dict) else i
                _differences(a, b, f"{path}[{label}]", out, tol)
    elif recorded != recomputed:
        out.append(f"{path}: recorded {recorded!r}, recomputed {recomputed!r}")
    return out


def _verify_dir(run_dir: str) -> dict:
    problems = []
    result = {"ok": False, "problems": problems, "path": run_dir,
              "scope": "internal consistency only: files, identities and recomputed statistics. "
                       "Integrity findings are taken as recorded. This does not authenticate "
                       "the run, its author or its hardware."}
    sums_path = os.path.join(run_dir, "checksums.json")
    if not os.path.isfile(sums_path):
        problems.append("checksums.json is missing")
        return result
    try:
        with open(sums_path, encoding="utf-8") as fh:
            sums = json.load(fh)
        listed = sums["files"]
        assert isinstance(listed, dict)
    except (ValueError, KeyError, AssertionError, OSError):
        problems.append("checksums.json is not readable")
        return result
    actual = _files(run_dir)
    for rel, sha in listed.items():
        if rel not in actual:
            problems.append(f"missing file: {rel}")
        elif actual[rel] != sha:
            problems.append(f"modified file: {rel}")
    for rel in actual:
        if rel not in listed:
            problems.append(f"file not in checksums: {rel}")
        if str(actual[rel]).startswith("symlink:"):
            problems.append(f"bundle contains a link: {rel}")
    for rel in REQUIRED:
        if rel not in actual:
            problems.append(f"required file absent: {rel}")
    if digest(listed) != sums.get("bundle_digest"):
        problems.append("bundle digest does not match the file list")
    result["bundle_digest"] = sums.get("bundle_digest", "")
    result["redacted"] = bool(sums.get("redacted"))
    if problems:
        return result
    try:
        evidence = load_evidence(run_dir)
        observations = load_observations(run_dir)
        with open(os.path.join(run_dir, "manifest.json"), encoding="utf-8") as fh:
            manifest = _strict_json(fh.read())
        with open(os.path.join(run_dir, "declaration.json"), encoding="utf-8") as fh:
            declared = _strict_json(fh.read())
        with open(os.path.join(run_dir, "report.md"), encoding="utf-8") as fh:
            report_text = fh.read()
    except (OSError, ValueError) as exc:
        problems.append(f"cannot read the bundle: {exc}")
        return result
    problems += [f"evidence.json: {p}" for p in decl.schema_problems(evidence, "evidence")]
    resolved = evidence.get("declaration", {}).get("resolved", {})
    problems += [f"declaration: {p}" for p in decl.schema_problems(resolved, "declaration")]
    for i, row in enumerate(observations):
        bad = decl.schema_problems(row, "observation")
        if bad:
            problems.append(f"observations.jsonl line {i + 1}: {bad[0]}")
            break
    plan = manifest.get("plan") if isinstance(manifest, dict) else None
    if not isinstance(plan, dict) or not isinstance(manifest.get("approval"), dict):
        problems.append("manifest.json has no plan or no approval")
    if problems:
        return result

    try:
        _check_identities(problems, evidence, manifest, declared, resolved, observations, run_dir)
        recorded = evidence["comparison"]
        inputs = recorded.get("inputs", {})
        redo = compare.compare(resolved, observations, inputs.get("execution_status"),
                               inputs.get("integrity", []))
        result["verdict"] = recorded["verdict"]
        protocol = evidence["execution"].get("protocol")
        result["order_control"] = (
            "none recorded: produced before measurement protocols existed, so run order and "
            "cache state were not controlled" if protocol is None else
            f"{protocol['design']}, order controlled" if protocol.get("order_controlled") else
            f"{protocol['design']}, order not controlled")
        result["recomputed_verdict"] = redo["verdict"]
        if redo["verdict"] != recorded["verdict"]:
            problems.append(f"recorded verdict {recorded['verdict']} but the observations "
                            f"give {redo['verdict']}")
        problems += _differences(recorded, redo)
        if inputs.get("execution_status") != evidence["execution"]["status"]:
            problems.append("execution status disagrees between execution and comparison")
        if report_text != report.render(evidence):
            problems.append("report.md does not match what evidence.json renders to")
        problems += _check_decisions(run_dir, evidence)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        problems.append(f"malformed bundle: {type(exc).__name__}: {exc}")
    result["run_id"] = evidence["execution"].get("run_id", "")
    result["ok"] = not problems
    return result


def _check_identities(problems, evidence, manifest, declared, resolved, observations, run_dir):
    plan, ex, change = manifest["plan"], evidence["execution"], evidence["change"]
    say = problems.append
    plan_digest = digest({k: v for k, v in plan.items() if k != "plan_digest"})
    if not (plan_digest == plan.get("plan_digest") == ex["plan_digest"]):
        say("plan_digest does not match the plan in manifest.json")
    if manifest["approval"].get("plan_digest") != ex["plan_digest"]:
        say("the approval in manifest.json is for a different plan")
    if manifest.get("run_id") != ex["run_id"]:
        say("run_id disagrees between manifest.json and evidence.json")
    same = lambda a, b: canonical_json(a, False) == canonical_json(b, False)  # noqa: E731
    if not (same(plan["declaration"], declared) and same(declared, resolved)):
        say("the declaration differs between manifest.json, declaration.json and evidence.json")
    dhash = decl.declaration_hash(resolved)
    if not (dhash == evidence["declaration"]["declaration_hash"] == plan["declaration_hash"]):
        say("declaration_hash does not match the resolved declaration")
    shas = {arm: plan["change"][arm]["sha"] for arm in ("baseline", "candidate")}
    harness = plan.get("harness")
    if not (decl.spec_hash(resolved, shas, harness and harness["digest"]) == plan["spec_hash"] == ex["spec_hash"]):
        say("spec_hash does not match the plan")
    cid = decl.change_id(plan["change"]["repo"], shas, resolved["arms"])
    if not (cid == plan["change"]["change_id"] == change["change_id"]):
        say("change_id does not match the repo, commits and arm settings")
    for arm in shas:
        if change[arm]["sha"] != shas[arm]:
            say(f"{arm} commit disagrees between manifest.json and evidence.json")
        if decl.env_key(resolved, arm, shas[arm], plan.get("image_id")) != plan["env_keys"][arm]:
            say(f"env_key for {arm} does not match the plan")
    snapshot = os.path.join(run_dir, "harness")
    if harness and os.path.isdir(snapshot):
        recorded_bad = any(f.get("code") == "harness_modified" and not f.get("passed")
                           for f in evidence["comparison"].get("inputs", {}).get("integrity", []))
        if tree_digest(snapshot)["digest"] != harness["digest"] and not recorded_bad:
            say("the harness snapshot does not match the approved harness digest")
    if len(observations) != evidence["observations"]["count"]:
        say("observation count does not match evidence.json")
    if sha256_file(os.path.join(run_dir, "observations.jsonl")) != evidence["observations"]["sha256"]:
        say("observations.jsonl does not match the digest in evidence.json")
    attempts = {}
    for a in ex["attempts"]:
        if a["attempt_id"] in attempts:
            say(f"attempt id listed twice: {a['attempt_id']}")
        attempts[a["attempt_id"]] = a
    counted = [a for a in attempts.values() if a["status"] == "ok" and not a.get("discarded")]
    ok_slots = [schedule.slot_key(a) for a in counted]
    if len(ok_slots) != len(set(ok_slots)):
        say("more than one successful attempt for the same arm and seed")
    for i, row in enumerate(observations):
        a = attempts.get(row["attempt_id"])
        if (a is None or a["status"] != "ok" or a.get("discarded")
                or schedule.slot_key(a) != schedule.slot_key(row)
                or a.get("period") != row.get("period")):
            say(f"observations.jsonl line {i + 1} does not belong to a counted, successful "
                "attempt with that arm, seed and run order")
            break
    _check_schedule(say, plan, ex, resolved, counted)


def _check_schedule(say, plan, ex, resolved, counted):
    """Under a measurement protocol the schedule is part of what was approved.
    The record must carry that schedule, it must be the one the declaration
    derives, and every counted attempt must be one of its slots."""
    m = resolved.get("measurement")
    if not m:
        if plan.get("schedule") or ex.get("schedule"):
            say("a run schedule is recorded but the declaration has no measurement protocol")
        return
    planned = decl.planned_schedule(resolved)
    if plan.get("schedule") != planned or ex.get("schedule") != planned:
        say("the run schedule does not match the one the declaration and scheduler seed give")
        return
    for problem in schedule.problems(planned, resolved["method"]["seeds"], m):
        say(f"schedule: {problem}")
    slots = {schedule.slot_key(s): s for s in planned}
    for a in counted:
        slot = slots.get(schedule.slot_key(a))
        if slot is None or slot["period"] != a.get("period") or slot["ordinal"] != a.get("planned_ordinal"):
            say(f"attempt {a['attempt_id']} is not a slot of the approved schedule")
            return
        iso = a.get("isolation") or {}
        if not iso.get("fresh"):
            say(f"attempt {a['attempt_id']} is counted but did not start from a clean cache")
            return
    if ex["status"] == "COMPLETED" and len(counted) != len(planned):
        say("the run is recorded as completed but not every scheduled attempt was counted")
    # A kept sequence ran its two arms back to back, in the planned order.
    order = {a["attempt_id"]: i for i, a in enumerate(ex["attempts"])}
    pairs = {}
    for a in counted:
        pairs.setdefault((a["seed"], a["replicate"], a["sequence"]), []).append(a)
    for members in pairs.values():
        members.sort(key=lambda a: a["period"])
        if len(members) == 2 and order[members[1]["attempt_id"]] - order[members[0]["attempt_id"]] != 1:
            say(f"the two arms of {members[0]['attempt_id'][4:]} did not run back to back")
            return


def _check_decisions(run_dir: str, evidence: dict) -> list:
    path = os.path.join(run_dir, "decisions.jsonl")
    if not os.path.isfile(path):
        return []
    out, ids = [], set()
    with open(path, encoding="utf-8") as fh:
        rows = [_strict_json(line) for line in fh if line.strip()]
    for i, doc in enumerate(rows):
        where = f"decisions.jsonl line {i + 1}"
        bad = decl.schema_problems(doc, "decision")
        if bad:
            out.append(f"{where}: {bad[0]}")
            continue
        body = {k: v for k, v in doc.items() if k != "decision_id"}
        if "dec_" + digest(body)[:12] != doc["decision_id"]:
            out.append(f"{where}: decision_id does not match its content")
        ref = doc["evidence"][0]
        if ref["run_id"] != evidence["execution"]["run_id"]:
            out.append(f"{where}: refers to a different run")
        if ref["verdict"] != evidence["comparison"]["verdict"] or \
                ref["declaration_hash"] != evidence["declaration"]["declaration_hash"]:
            out.append(f"{where}: its evidence reference disagrees with this bundle")
        if doc.get("supersedes") and doc["supersedes"] not in ids:
            out.append(f"{where}: supersedes a decision that is not in this file")
        ids.add(doc["decision_id"])
    return out


# ── export ────────────────────────────────────────────────────────────────

PUBLIC_FILES = ("evidence.json", "manifest.json", "declaration.json", "observations.jsonl")
REDACTION_POLICY = "public-v2: allowlisted files; local paths, email addresses and credential-shaped strings replaced; logs and harness dropped unless asked"


class ExportRefused(Exception):
    pass


def _redactor(replacements):
    def scrub(text: str) -> str:
        for needle, label in replacements:
            if needle:
                text = text.replace(needle, label)
        text = EMAIL_PATTERN.sub("<EMAIL>", text)
        for pattern in SECRET_PATTERNS:
            if pattern.groups >= 2:
                text = pattern.sub(lambda m: m.group(1) + m.group(2) + "<REDACTED>", text)
            else:
                text = pattern.sub("<REDACTED>", text)
        return text
    return scrub


def _copy_decisions(run_dir: str, stage: str, project: str, scrub=None) -> None:
    run_id = os.path.basename(run_dir.rstrip(os.sep))
    source = os.path.join(project, ".validate", "decisions.jsonl")
    if not os.path.isfile(source):
        return
    with open(source, encoding="utf-8") as fh:
        mine = [ln for ln in fh if ln.strip() and
                json.loads(ln)["evidence"][0]["run_id"] == run_id]
    if mine and scrub is None:
        with open(os.path.join(stage, "decisions.jsonl"), "w", encoding="utf-8") as fh:
            fh.writelines(mine)


def export(run_dir: str, dest: str, project: str, public: bool = False,
           include_logs: bool = False) -> dict:
    """Write a .tar.gz of the bundle. The source bundle is never modified.

    A private export is a copy. A public export is a derivative: only
    allowlisted files, with local paths, email addresses and credential-shaped
    strings replaced, and the report rendered again from the redacted record.
    If redaction would alter anything an identity is computed from (the
    declaration, the plan, the observations) the export is refused, because
    the result would no longer be the evidence that was approved and run.
    Redaction is pattern matching. Read every exported file before sharing.
    """
    run_id = os.path.basename(run_dir.rstrip(os.sep))
    stage_root = tempfile.mkdtemp(prefix="validate-export-")
    stage = os.path.join(stage_root, run_id)
    try:
        extra, dropped = {}, []
        if not public:
            shutil.copytree(run_dir, stage, symlinks=True, ignore=shutil.ignore_patterns(
                "status.json", "status.json.tmp", "setup", "preflight"))
            _copy_decisions(run_dir, stage, project)
        else:
            os.makedirs(stage)
            scrub = _redactor([(project, "<PROJECT>"), (os.path.realpath(project), "<PROJECT>"),
                               (os.path.expanduser("~"), "<HOME>")])
            texts = {}
            for name in PUBLIC_FILES:
                with open(os.path.join(run_dir, name), encoding="utf-8") as fh:
                    texts[name] = fh.read()
            cleaned = {name: scrub(text) for name, text in texts.items()}
            hashed = ["declaration.json", "observations.jsonl"]
            plan_before = json.loads(texts["manifest.json"])["plan"]
            plan_after = json.loads(cleaned["manifest.json"])["plan"]
            changed = [n for n in hashed if cleaned[n] != texts[n]]
            if plan_before != plan_after:
                changed.append("manifest.json (the approved plan)")
            if changed:
                raise ExportRefused(
                    "public export refused: redaction would change " + ", ".join(changed) +
                    ". These carry the identities the evidence was approved and run under, so a "
                    "redacted copy would not be the same evidence. Remove local paths, email "
                    "addresses or credential-shaped strings from the declaration and unit ids "
                    "and run again, or share the bundle privately.")
            for name, text in cleaned.items():
                with open(os.path.join(stage, name), "w", encoding="utf-8") as fh:
                    fh.write(text)
            evidence = load_evidence(stage)
            with open(os.path.join(stage, "report.md"), "w", encoding="utf-8") as fh:
                fh.write(report.render(evidence))
            for base, dirs, names in os.walk(run_dir):
                for name in names:
                    rel = os.path.relpath(os.path.join(base, name), run_dir).replace(os.sep, "/")
                    if rel in PUBLIC_FILES or rel in EXCLUDED or rel == "report.md":
                        continue
                    is_log = rel.startswith("attempts/") and rel.endswith("/log.txt")
                    if not (include_logs and is_log):
                        dropped.append(rel)
                        continue
                    try:
                        with open(os.path.join(base, name), encoding="utf-8") as fh:
                            text = scrub(fh.read())
                    except (UnicodeDecodeError, OSError):
                        dropped.append(rel)
                        continue
                    os.makedirs(os.path.dirname(os.path.join(stage, rel)), exist_ok=True)
                    with open(os.path.join(stage, rel), "w", encoding="utf-8") as fh:
                        fh.write(text)
            extra = {"redacted": True, "derived_from_bundle_digest": bundle_digest(run_dir),
                     "redaction_policy": REDACTION_POLICY, "logs_included": include_logs}
        sums = write_checksums(stage, extra)
        check = _verify_dir(stage)
        if not check["ok"]:
            raise ExportRefused("export refused: the staged bundle does not verify: "
                                + "; ".join(check["problems"][:3]))
        with tarfile.open(dest, "w:gz") as tar:
            tar.add(stage, arcname=run_id)
        return {"path": dest, "bundle_digest": sums["bundle_digest"], "redacted": public,
                "files": sorted(sums["files"]), "dropped": len(dropped),
                "review": "Redaction is pattern matching, not a guarantee. Read every file in "
                          "the archive before sharing it." if public else ""}
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)
