"""Local, redacted publication audit. Does not stage, alter history, or publish.

Findings are leads for human review, never a proof that corpus text is PII-free.
Vendor toolchains and caches are inventoried but not content-scanned. Private
archives are scanned in memory (including OOXML metadata); no extraction.
"""
from __future__ import annotations

import collections
import hashlib
import io
import json
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/public_release_audit"
PATTERNS = {
    "credential_token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16}|sk-(?:proj-)?[A-Za-z0-9_-]{35,})"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential_assignment_review": re.compile(r'''(?i)(?:api[_-]?key|access[_-]?token|password|secret[_-]?key)\s*[=:]\s*["']([^"'\s]{8,})["']'''),
    "email_review": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "machine_path_review": re.compile(r"(?:[A-Za-z]:[\\/](?:Users|Python)[\\/][^\s\"<>]+|/home/[^\s\"<>]+)"),
    "phone_or_tax_id_review": re.compile(r"(?<!\d)(?:\d{3}-\d{2}-\d{4}|(?:\+1[ -]?)?\(?\d{3}\)?[ -]\d{3}[ -]\d{4})(?!\d)"),
    "tax_or_client_context_review": re.compile(r"(?i)\b(?:taxpayer|social security|ssn|client address|bank account|routing number)\b"),
}
TEXT = {".py", ".md", ".txt", ".json", ".jsonl", ".csv", ".tsv", ".tex", ".bib", ".yaml", ".yml", ".log", ".conllu", ".ipynb", ".xml", ".rels", ".js", ".mjs", ".html", ".ps1", ".env"}
ZIP = {".zip", ".xlsx", ".docx", ".pptx"}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def excluded(path):
    parts = path.parts
    return any(p in {".git", "__pycache__", "isolated_pycache", "texlive2025_portable", ".venv", "venv", "node_modules"} for p in parts) or path.is_relative_to(OUT)


def scan(data, name, findings, coverage, depth=0):
    ext = Path(name).suffix.lower()
    if ext in ZIP:
        if depth >= 5:
            coverage["archive_depth_skipped"] += 1
            return
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                coverage["archives_opened"] += 1
                for info in z.infolist():
                    if not info.is_dir():
                        if info.file_size > 100_000_000:
                            coverage["oversize_members_skipped"] += 1
                            continue
                        scan(z.read(info), name + "!" + info.filename, findings, coverage, depth+1)
        except (zipfile.BadZipFile, RuntimeError) as exc:
            findings.append({"path": name, "kind": "archive_unreadable", "exception": type(exc).__name__})
    elif ext == ".pdf":
        try:
            import fitz
            with fitz.open(stream=data, filetype="pdf") as pdf:
                scan((json.dumps(pdf.metadata) + "\n" + "\n".join(p.get_text() for p in pdf)).encode(), name+"!extracted.txt", findings, coverage, depth)
                coverage["pdfs_extracted"] += 1
        except (ImportError, RuntimeError, ValueError):
            coverage["pdfs_unreadable"] += 1
    elif ext in TEXT or Path(name).name in {"LICENSE", ".gitignore", ".gitattributes", ".env", "SHA256SUMS"}:
        text = data.decode("utf-8", errors="replace")
        coverage["text_members_scanned"] += 1
        for kind, pattern in PATTERNS.items():
            matches = list(pattern.finditer(text))
            if matches:
                findings.append({"path": name, "kind": kind, "count": len(matches),
                                 "first_line": text.count("\n", 0, matches[0].start()) + 1})
    else:
        coverage["binary_not_text_scanned"] += 1


def files_snapshot():
    return {p.relative_to(ROOT).as_posix(): sha(p.read_bytes())
            for p in sorted(ROOT.rglob("*")) if p.is_file() and not excluded(p)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tracked = git("ls-files", "-z").decode().split("\0")[:-1]
    (OUT/"tracked_files.txt").write_text("\n".join(tracked)+"\n", encoding="utf-8")
    snapshot = files_snapshot()
    (OUT/"before_sha256.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    findings, coverage = [], collections.Counter()
    for name in snapshot:
        scan((ROOT/name).read_bytes(), name, findings, coverage)
    # Every unique reachable blob, including deleted versions and local refs.
    entries = git("rev-list", "--objects", "--all").decode("utf-8", errors="replace").splitlines()
    objects = [line.split(" ", 1) for line in entries]
    process = subprocess.Popen(["git", "cat-file", "--batch"], cwd=ROOT,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    history_findings, history_coverage = [], collections.Counter()
    for item in objects:
        oid, name = item[0], item[1] if len(item)>1 else ""
        process.stdin.write((oid+"\n").encode()); process.stdin.flush()
        header = process.stdout.readline().decode().split()
        size = int(header[2]); data = process.stdout.read(size)
        process.stdout.read(1)
        if header[1] == "blob":
            history_coverage["unique_blobs"] += 1
            start = len(history_findings)
            scan(data, name, history_findings, history_coverage)
            for finding in history_findings[start:]:
                finding["git_blob"] = oid
    process.stdin.close(); process.wait()
    # Ignore checks deliberately include tracked files, unlike check-ignore's default.
    ignored = subprocess.run(["git", "check-ignore", "--no-index", "-z", "--stdin"], cwd=ROOT,
                             input=("\0".join(tracked)+"\0").encode(), capture_output=True).stdout.decode().split("\0")[:-1]
    (OUT/"tracked_private_paths.txt").write_text("\n".join(ignored)+"\n", encoding="utf-8")
    remote = {}
    for ref in git("for-each-ref", "--format=%(refname)", "refs/remotes/").decode().splitlines():
        remote[ref] = {"tip": git("rev-parse", ref).decode().strip(),
                       "tracked_paths": git("ls-tree", "-r", "--name-only", ref).decode().splitlines()}
    head = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True)
    result = {"head":head.stdout.strip() if head.returncode == 0 else None, "tracked_count":len(tracked),
              "worktree_coverage":dict(coverage), "history_coverage":dict(history_coverage),
              "tracked_paths_now_ignored_count":len(ignored), "remote_tracking_inventory":remote,
              "worktree_findings":findings, "history_findings":history_findings,
              "limitations":["Pattern hits require review; values intentionally omitted.",
                 "No automatic person-name detection or proof of text redistribution rights.",
                 "Vendor TeX distribution, Git internals and caches excluded from working-tree scan.",
                 "Binary/image contents not OCR-scanned; archive size/depth skips reported.",
                 "Remote tips verified separately; server caches, forks and unreachable remote objects not audited."]}
    (OUT/"privacy_audit.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"tracked":len(tracked), "tracked_now_ignored":len(ignored),
        "worktree_coverage":dict(coverage), "history_coverage":dict(history_coverage),
        "worktree_hit_types":dict(collections.Counter(f["kind"] for f in findings)),
        "history_hit_types":dict(collections.Counter(f["kind"] for f in history_findings))}))


if __name__ == "__main__":
    main()
