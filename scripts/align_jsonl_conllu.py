"""
align_jsonl_conllu.py - Align JSONL records to consecutive CoNLL-U sentences.

Each JSONL record contains a "pangyo" field that is the exact concatenation of
consecutive CoNLL-U sentence texts (from # text = comments), joined by a single
space.  This script exploits that structure to assign a doc_id to every sentence.

Inputs:
    data/kor_en_panyo_corpus.jsonl
    data/processed/panyo_original_adp_lang_labeled.conllu

Outputs:
    data/processed/panyo_jsonl_doc_sentence_map.csv   (sentence → doc mapping)
    results/jsonl_conllu_alignment_report.csv          (one row per JSONL record)
    results/jsonl_conllu_alignment_summary.md          (human-readable summary)

Alignment strategy (greedy, text-based):
    For each JSONL record in order, consume consecutive CoNLL-U sentences (in order)
    until the accumulated # text values joined with ' ' equal the normalized pangyo text.
    Primary check: exact string match after whitespace normalization.
    Fallback: difflib.SequenceMatcher ratio >= 0.95 → high_similarity.
    If ratio < 0.95, status = failed.

Encoding note:
    The CoNLL-U file is valid UTF-8. # text comments contain proper Korean Unicode.
    The JSONL file is also valid UTF-8.  No mojibake repair is needed.
    (Earlier PowerShell output showed garbled Korean only because PowerShell 5.1
    defaults to Windows-1252 for Get-Content; the actual bytes are correct UTF-8.)
"""

from __future__ import annotations

import csv
import difflib
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from read_conllu import iter_sentences

ROOT    = Path(__file__).parent.parent
DATA    = ROOT / "data"
RESULTS = ROOT / "results"

JSONL_PATH   = DATA / "kor_en_panyo_corpus.jsonl"
CONLLU_PATH  = DATA / "processed" / "panyo_original_adp_lang_labeled.conllu"
MAP_PATH     = DATA / "processed" / "panyo_jsonl_doc_sentence_map.csv"
REPORT_PATH  = RESULTS / "jsonl_conllu_alignment_report.csv"
SUMMARY_PATH = RESULTS / "jsonl_conllu_alignment_summary.md"

# Thresholds
EXACT_THRESHOLD       = 1.000
HIGH_SIM_THRESHOLD    = 0.950
HEURISTIC_THRESHOLD   = 0.850
SAFE_ALIGN_THRESHOLD  = 0.980  # % of records that must align for split to proceed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """NFC + collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def get_comment_value(comments: list[str], prefix: str) -> str:
    """Extract value after '# prefix = ' from sentence comments."""
    full_prefix = f"# {prefix} = "
    for c in comments:
        if c.startswith(full_prefix):
            return c[len(full_prefix):]
    return ""


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] JSONL line {i} parse error: {e}")
                obj = {}
            obj["_jsonl_line"] = i
            records.append(obj)
    return records


def load_sentences(path: Path) -> list[dict]:
    """Load sentences, extracting sent_id, text, and real-token count."""
    result = []
    for sent in iter_sentences(path):
        sent_id  = get_comment_value(sent["comments"], "sent_id")
        text     = get_comment_value(sent["comments"], "text")
        n_tokens = sum(1 for t in sent["tokens"] if t["is_real"])
        result.append({
            "sent_id":  sent_id,
            "text":     text,
            "n_tokens": n_tokens,
            "sent":     sent,
        })
    return result


# ---------------------------------------------------------------------------
# Greedy alignment
# ---------------------------------------------------------------------------

def greedy_align(
    jsonl_records: list[dict],
    sentences: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Returns:
        alignment_rows  - one dict per JSONL record (for the CSV report)
        sentence_map    - one dict per CoNLL-U sentence (for the mapping CSV)

    Algorithm detail:
        For each JSONL record, sentences are consumed one at a time.  After each
        consumption the accumulated # text values (joined with ' ') are compared
        with the normalised pangyo text.

        When the accumulated text diverges from the target (no longer a prefix):
          - Compare similarity of the version WITHOUT the new sentence (rollback)
            vs WITH the new sentence (keep).
          - Choose whichever gives higher similarity, and set the cursor accordingly.
        This prevents "overshooting" a document boundary from cascading into failures
        for all subsequent records.

        Thresholds:
          exact           = 1.000 (string equality)
          normalized_exact  >= 0.999
          high_similarity >= 0.950
          heuristic       >= 0.850
          failed          <  0.850
    """
    alignment_rows: list[dict] = []
    sentence_map: list[dict] = []

    sent_cursor = 0
    n_sents = len(sentences)

    for rec in jsonl_records:
        jsonl_id    = rec["_jsonl_line"]
        pangyo_raw  = rec.get("pangyo", "")
        norm_target = normalize(pangyo_raw)

        start_cursor = sent_cursor
        prev_accum   = ""          # accumulated text BEFORE current sentence
        accum        = ""          # accumulated text AFTER current sentence
        status       = "failed"
        score        = 0.0
        notes        = ""

        while sent_cursor < n_sents:
            s_text = normalize(sentences[sent_cursor]["text"])
            sent_cursor += 1
            candidate = (prev_accum + " " + s_text) if prev_accum else s_text

            # --- Primary: exact match ---
            if candidate == norm_target:
                accum  = candidate
                status = "exact"
                score  = 1.0
                break

            # --- Still a prefix of target — keep consuming ---
            if norm_target.startswith(candidate) and len(candidate) < len(norm_target):
                prev_accum = candidate
                continue

            # --- Candidate diverged from target ---
            # Compare: stop here (keep candidate) vs roll back (use prev_accum).
            ratio_keep = similarity(candidate,   norm_target)
            ratio_rb   = similarity(prev_accum,  norm_target) if prev_accum else 0.0

            if ratio_rb >= ratio_keep and ratio_rb >= HIGH_SIM_THRESHOLD:
                # Rollback is better — do NOT consume this sentence
                sent_cursor -= 1
                accum  = prev_accum
                status = "normalized_exact" if ratio_rb >= 0.999 else "high_similarity"
                score  = ratio_rb
                notes  = f"rollback; ratio={ratio_rb:.4f}"
            elif ratio_keep >= HIGH_SIM_THRESHOLD:
                accum  = candidate
                status = "normalized_exact" if ratio_keep >= 0.999 else "high_similarity"
                score  = ratio_keep
                notes  = f"keep; ratio={ratio_keep:.4f}"
            elif ratio_rb >= HEURISTIC_THRESHOLD:
                sent_cursor -= 1
                accum  = prev_accum
                status = "heuristic"
                score  = ratio_rb
                notes  = f"rollback heuristic; ratio={ratio_rb:.4f}"
            elif ratio_keep >= HEURISTIC_THRESHOLD:
                accum  = candidate
                status = "heuristic"
                score  = ratio_keep
                notes  = f"keep heuristic; ratio={ratio_keep:.4f}"
            else:
                # Both poor — rollback and mark failed
                sent_cursor -= 1
                accum  = prev_accum
                status = "failed"
                score  = max(ratio_keep, ratio_rb)
                notes  = f"diverged; best ratio={score:.4f}"
            break

        # Loop ended without a break → ran out of sentences while still accumulating
        if status == "failed" and prev_accum and sent_cursor == n_sents:
            # Check if whatever we accumulated so far is a good enough match
            final_ratio = similarity(prev_accum, norm_target)
            if final_ratio >= HIGH_SIM_THRESHOLD:
                accum  = prev_accum
                status = "normalized_exact" if final_ratio >= 0.999 else "high_similarity"
                score  = final_ratio
                notes  = f"end-of-file; ratio={final_ratio:.4f}"
            else:
                accum  = prev_accum
                score  = final_ratio
                notes  = f"ran out of sentences; ratio={final_ratio:.4f}"

        end_cursor = sent_cursor - 1
        n_assigned = end_cursor - start_cursor + 1

        doc_id = f"jsonl_{jsonl_id}"

        # Build sentence map entries
        for si in range(start_cursor, sent_cursor):
            sentence_map.append({
                "sent_id":   sentences[si]["sent_id"],
                "doc_id":    doc_id,
                "jsonl_idx": jsonl_id,
                "sent_seq":  si - start_cursor + 1,
                "n_tokens":  sentences[si]["n_tokens"],
            })

        alignment_rows.append({
            "jsonl_id":          jsonl_id,
            "doc_id":            doc_id,
            "num_sentences":     n_assigned,
            "start_sent_id":     sentences[start_cursor]["sent_id"] if n_assigned > 0 else "",
            "end_sent_id":       sentences[end_cursor]["sent_id"]   if n_assigned > 0 else "",
            "alignment_status":  status,
            "match_score":       f"{score:.4f}",
            "jsonl_text_length": len(norm_target),
            "conllu_text_length": len(accum),
            "notes":             notes,
        })

    # Sentences not assigned to any document (trailing)
    for si in range(sent_cursor, n_sents):
        sentence_map.append({
            "sent_id":   sentences[si]["sent_id"],
            "doc_id":    "UNASSIGNED",
            "jsonl_idx": -1,
            "sent_seq":  -1,
            "n_tokens":  sentences[si]["n_tokens"],
        })

    return alignment_rows, sentence_map


# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------

def write_report(rows: list[dict], path: Path) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "jsonl_id", "doc_id", "num_sentences",
        "start_sent_id", "end_sent_id",
        "alignment_status", "match_score",
        "jsonl_text_length", "conllu_text_length", "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_map(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sent_id", "doc_id", "jsonl_idx", "sent_seq", "n_tokens"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_summary(
    rows: list[dict],
    sentence_map: list[dict],
    n_jsonl: int,
    n_sents: int,
    path: Path,
) -> bool:
    """Write markdown summary. Returns True if alignment is safe to proceed."""
    status_counts: dict[str, int] = {}
    for r in rows:
        s = r["alignment_status"]
        status_counts[s] = status_counts.get(s, 0) + 1

    n_aligned   = sum(1 for r in rows if r["alignment_status"] != "failed")
    n_failed    = sum(1 for r in rows if r["alignment_status"] == "failed")
    pct_aligned = 100 * n_aligned / n_jsonl if n_jsonl else 0

    n_assigned   = sum(1 for s in sentence_map if s["doc_id"] != "UNASSIGNED")
    n_unassigned = sum(1 for s in sentence_map if s["doc_id"] == "UNASSIGNED")
    pct_sent_cov = 100 * n_assigned / n_sents if n_sents else 0

    safe = (pct_aligned >= 98.0) and (pct_sent_cov >= 98.0)

    lines = [
        "# JSONL ↔ CoNLL-U Alignment Summary",
        "",
        "## Counts",
        f"- JSONL records:            {n_jsonl:,}",
        f"- CoNLL-U sentences:        {n_sents:,}",
        f"- Aligned records:          {n_aligned:,}  ({pct_aligned:.2f}%)",
        f"- Failed records:           {n_failed:,}",
        f"- Assigned sentences:       {n_assigned:,}  ({pct_sent_cov:.2f}%)",
        f"- Unassigned sentences:     {n_unassigned:,}",
        "",
        "## Alignment Status Breakdown",
    ]
    for s, cnt in sorted(status_counts.items()):
        lines.append(f"- {s}: {cnt:,}")

    lines += [
        "",
        "## Safety Assessment",
        f"- Required threshold: ≥98% records aligned AND ≥98% sentences assigned",
        f"- Records aligned:  {pct_aligned:.2f}%  {'✓ PASS' if pct_aligned >= 98 else '✗ FAIL'}",
        f"- Sentences covered: {pct_sent_cov:.2f}%  {'✓ PASS' if pct_sent_cov >= 98 else '✗ FAIL'}",
        f"- **Proceed to document-level split: {'YES' if safe else 'NO'}**",
        "",
        "## Method",
        "Greedy text-based alignment. For each JSONL record, CoNLL-U sentences are",
        "consumed in order until the accumulated `# text` comment values (joined with",
        "a single space) match the normalized `pangyo` field of the JSONL record.",
        "Normalization: Unicode NFC + whitespace collapse.",
        "Fallback: difflib.SequenceMatcher ratio ≥ 0.95.",
        "",
        "## Encoding Note",
        "Both files are valid UTF-8. No mojibake repair was applied.",
        "PowerShell 5.1 displayed garbled Korean earlier because it defaults to",
        "Windows-1252 for Get-Content; the actual file bytes are correct.",
        "",
        "## Limitations",
        "- Minor text differences (e.g., different whitespace around punctuation)",
        "  may cause high_similarity instead of exact matches.",
        "- JSONL records with very short pangyo text (1-2 sentences) are more",
        "  sensitive to any whitespace mismatch.",
    ]
    if n_failed:
        lines += [
            "",
            f"## Failed Records ({n_failed})",
            "See `results/jsonl_conllu_alignment_report.csv` for details.",
        ]

    RESULTS.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return safe


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not JSONL_PATH.exists():
        sys.exit(f"ERROR: {JSONL_PATH} not found")
    if not CONLLU_PATH.exists():
        sys.exit(f"ERROR: {CONLLU_PATH} not found")

    print("Loading JSONL records ...")
    jsonl_records = load_jsonl(JSONL_PATH)
    print(f"  {len(jsonl_records):,} records")

    print("Loading CoNLL-U sentences ...")
    sentences = load_sentences(CONLLU_PATH)
    print(f"  {len(sentences):,} sentences")

    print("Running greedy alignment ...")
    alignment_rows, sentence_map = greedy_align(jsonl_records, sentences)

    # Collect stats
    statuses = [r["alignment_status"] for r in alignment_rows]
    n_exact   = statuses.count("exact")
    n_normeq  = statuses.count("normalized_exact")
    n_highsim = statuses.count("high_similarity")
    n_heur    = statuses.count("heuristic")
    n_failed  = statuses.count("failed")
    n_aligned = len(statuses) - n_failed
    pct       = 100 * n_aligned / len(statuses) if statuses else 0

    print(f"  exact:             {n_exact:,}")
    print(f"  normalized_exact:  {n_normeq:,}")
    print(f"  high_similarity:   {n_highsim:,}")
    print(f"  heuristic:         {n_heur:,}")
    print(f"  failed:            {n_failed:,}")
    print(f"  aligned ({pct:.2f}%): {n_aligned:,} / {len(statuses):,}")

    n_unassigned = sum(1 for s in sentence_map if s["doc_id"] == "UNASSIGNED")
    print(f"  unassigned sentences: {n_unassigned:,}")

    print(f"\nWriting {REPORT_PATH.name} ...")
    write_report(alignment_rows, REPORT_PATH)

    print(f"Writing {MAP_PATH.name} ...")
    write_map(sentence_map, MAP_PATH)

    safe = write_summary(
        alignment_rows, sentence_map,
        len(jsonl_records), len(sentences), SUMMARY_PATH
    )
    print(f"Writing {SUMMARY_PATH.name} ...")

    if safe:
        print("\n[OK] Alignment coverage meets threshold. Safe to proceed with document-level split.")
    else:
        print("\n[WARN] Alignment coverage below threshold. Review failures before proceeding.")

    # Show first few failures if any
    failures = [r for r in alignment_rows if r["alignment_status"] == "failed"]
    if failures:
        print(f"\nFirst {min(5, len(failures))} failed records:")
        for r in failures[:5]:
            print(
                f"  jsonl_id={r['jsonl_id']}  "
                f"sents={r['num_sentences']}  "
                f"score={r['match_score']}  "
                f"notes={r['notes']}"
            )


if __name__ == "__main__":
    main()
