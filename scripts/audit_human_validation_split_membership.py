"""
audit_human_validation_split_membership.py - Audit train/dev/test split
membership of the 60-sentence finalized human-validation sample (40
Representative + 20 Diagnostic, 932 tokens), WITHOUT modifying the canonical
corpus, the split files, the model outputs, or any human-validation input.

Inputs (all read-only):
  - outputs/human_validation/annotations/Human_POS_Tagging_v3_P1.xlsx
  - outputs/human_validation/annotations/Human_POS_Tagging_v3_P2.xlsx
  - outputs/human_validation/annotations/Human_POS_Tagging_v3_Blind_Adjudication_FINAL.xlsx
  - outputs/human_validation/results/Human_POS_Tagging_v3_Final_Adjudicated_Results.xlsx
    (read only for an independent cross-check of the reconstruction; not
    itself required to reconstruct the 932 tokens)
  - data/processed/panyo_original_adp_docsplit_{train,dev,test}.conllu
    (the authoritative document-level split; see guidelines/data_split_decisions.md)
  - data/processed/panyo_jsonl_doc_sentence_map.csv (doc_id cross-check only)

Outputs (all new, under outputs/human_validation/split_audit/):
  - sentence_split_membership.csv
  - token_split_membership.csv
  - summary.json / summary.txt
  - README.md / command_log.txt

Matching priority (strict, as required by the audit):
  1. persistent identifier: the canonical `sent_id` (e.g. "pangyo829")
     recorded as Source_Sent_ID in the P1/P2 workbooks, looked up directly
     in the authoritative docsplit files.
  2. exact ordered token-form sequence, searched across the full docsplit
     corpus, used only when (1) fails or fails token verification.
  3. exact sentence text (the literal `# text` comment / "Sentence" column,
     no normalization), used only when (1) and (2) both fail. Documented as
     the weakest fallback; refuses to guess when more than one candidate
     shares the same text.

A sentence is never accepted on a weaker signal when a stronger one is
available, and any stage that yields more than one candidate is reported as
ambiguous rather than guessed.

Run from repo root:
  python scripts/audit_human_validation_split_membership.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import openpyxl

sys.path.insert(0, str(Path(__file__).parent))
from read_conllu import F_FORM, iter_sentences  # noqa: E402

ROOT = Path(__file__).parent.parent

P1_XLSX = ROOT / "outputs" / "human_validation" / "annotations" / "Human_POS_Tagging_v3_P1.xlsx"
P2_XLSX = ROOT / "outputs" / "human_validation" / "annotations" / "Human_POS_Tagging_v3_P2.xlsx"
FINAL_ADJ_XLSX = ROOT / "outputs" / "human_validation" / "annotations" / "Human_POS_Tagging_v3_Blind_Adjudication_FINAL.xlsx"
RESULTS_XLSX = ROOT / "outputs" / "human_validation" / "results" / "Human_POS_Tagging_v3_Final_Adjudicated_Results.xlsx"

DOCSPLIT_PATHS = {
    "train": ROOT / "data" / "processed" / "panyo_original_adp_docsplit_train.conllu",
    "dev": ROOT / "data" / "processed" / "panyo_original_adp_docsplit_dev.conllu",
    "test": ROOT / "data" / "processed" / "panyo_original_adp_docsplit_test.conllu",
}
DOC_MAP_CSV = ROOT / "data" / "processed" / "panyo_jsonl_doc_sentence_map.csv"

OUT_DIR = ROOT / "outputs" / "human_validation" / "split_audit"

P1_P2_HEADER = (
    "Set", "Sample_ID", "Source_Sent_ID", "Sentence", "Token_ID", "Token",
    "Human_UPOS", "Confidence", "Tokenization_Issue", "Note",
)
ADJUDICATION_HEADER = (
    "Set", "Sample_ID", "Sent_ID", "Sentence", "Token_ID", "Token",
    "P1_UPOS", "P1_Confidence", "P1_Tokenization", "P1_Note",
    "P2_UPOS", "P2_Confidence", "P2_Tokenization", "P2_Note",
    "Adjudicated_UPOS", "Final_Tokenization_Issue", "Adjudication_Reason",
)
RESULTS_HEADER = (
    "Set", "Sample_ID", "Sent_ID", "Sentence", "Token_ID", "Token",
    "P1_UPOS", "P2_UPOS", "Final_Human_UPOS", "Decision_Source",
    "P1_Confidence", "P2_Confidence", "Final_Tokenization_Issue", "Adjudication_Reason",
    "Language", "Diagnostic_Category", "Canonical_UPOS", "Human_Canonical_Match",
    "P1_Tokenization", "P2_Tokenization", "Canonical_Token",
)

SHEET_TO_SAMPLE_SET = {"Representative_40": "Representative", "Diagnostic_20": "Diagnostic"}


class AuditError(Exception):
    """Raised for a fatal precondition failure (not a per-sentence audit finding)."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HumanToken:
    token_id: int
    form: str
    final_upos: str
    decision_source: str  # "agreement" | "adjudicated"


@dataclass(frozen=True)
class HumanSentence:
    sample_set: str
    sample_id: str
    source_sent_id: str
    sentence_text: str
    tokens: tuple[HumanToken, ...]

    @property
    def token_forms(self) -> tuple[str, ...]:
        return tuple(t.form for t in self.tokens)


@dataclass(frozen=True)
class SplitSentence:
    sent_id: str
    doc_id: str
    split: str
    text: str
    token_forms: tuple[str, ...]


@dataclass(frozen=True)
class MatchResult:
    method: str
    status: str  # "matched" | "ambiguous" | "unmatched"
    split: Optional[str]
    doc_id: Optional[str]
    notes: str


# ---------------------------------------------------------------------------
# XLSX loading (I/O only -- no matching/merge logic here)
# ---------------------------------------------------------------------------

def find_header_row(rows: list[tuple], expected_header: tuple) -> int:
    for i, row in enumerate(rows):
        if row == expected_header:
            return i
    raise AuditError(f"header row {expected_header!r} not found in sheet")


def load_p1_or_p2(path: Path) -> dict[tuple[str, int], dict]:
    """Return {(sample_id, token_id): {sample_set, source_sent_id, sentence_text, token_form, upos}}."""
    if not path.exists():
        raise AuditError(f"Missing required annotation workbook: {path}")
    wb = openpyxl.load_workbook(path, data_only=True)
    out: dict[tuple[str, int], dict] = {}
    for sheet_name, sample_set in SHEET_TO_SAMPLE_SET.items():
        if sheet_name not in wb.sheetnames:
            raise AuditError(f"{path.name}: missing expected sheet {sheet_name!r}")
        rows = list(wb[sheet_name].iter_rows(values_only=True))
        header_idx = find_header_row(rows, P1_P2_HEADER)
        for row in rows[header_idx + 1:]:
            if row[1] is None:
                continue
            _, sample_id, source_sent_id, sentence, token_id, token, upos = row[:7]
            key = (sample_id, int(token_id))
            if key in out:
                raise AuditError(f"{path.name}:{sheet_name}: duplicate row for {key}")
            out[key] = {
                "sample_set": sample_set,
                "source_sent_id": source_sent_id,
                "sentence_text": sentence,
                "token_form": token,
                "upos": upos,
            }
    return out


def load_final_adjudication(path: Path) -> dict[tuple[str, int], str]:
    """Return {(sample_id, token_id): adjudicated_upos} for the 51 disagreements."""
    if not path.exists():
        raise AuditError(f"Missing required adjudication workbook: {path}")
    wb = openpyxl.load_workbook(path, data_only=True)
    if "Adjudication_51" not in wb.sheetnames:
        raise AuditError(f"{path.name}: missing expected sheet 'Adjudication_51'")
    rows = list(wb["Adjudication_51"].iter_rows(values_only=True))
    header_idx = find_header_row(rows, ADJUDICATION_HEADER)
    out: dict[tuple[str, int], str] = {}
    for row in rows[header_idx + 1:]:
        if row[1] is None:
            continue
        sample_id, token_id, adjudicated_upos = row[1], int(row[4]), row[14]
        key = (sample_id, token_id)
        if key in out:
            raise AuditError(f"{path.name}: duplicate adjudication row for {key}")
        out[key] = adjudicated_upos
    return out


def load_results_workbook_reference(path: Path) -> dict[tuple[str, int], dict]:
    """Return {(sample_id, token_id): {form, final_upos}} from Final_Reference_932 (cross-check only)."""
    if not path.exists():
        raise AuditError(f"Missing required results workbook: {path}")
    wb = openpyxl.load_workbook(path, data_only=True)
    if "Final_Reference_932" not in wb.sheetnames:
        raise AuditError(f"{path.name}: missing expected sheet 'Final_Reference_932'")
    rows = list(wb["Final_Reference_932"].iter_rows(values_only=True))
    header_idx = find_header_row(rows, RESULTS_HEADER)
    out: dict[tuple[str, int], dict] = {}
    for row in rows[header_idx + 1:]:
        if row[1] is None:
            continue
        sample_id, token_id, token, final_upos = row[1], int(row[4]), row[5], row[8]
        out[(sample_id, token_id)] = {"form": token, "final_upos": final_upos}
    return out


# ---------------------------------------------------------------------------
# Reconstruction (pure logic, testable without any file I/O)
# ---------------------------------------------------------------------------

def reconstruct_human_sample_from_rows(
    p1: dict[tuple[str, int], dict],
    p2: dict[tuple[str, int], dict],
    adjudicated: dict[tuple[str, int], str],
) -> list[HumanSentence]:
    if set(p1) != set(p2):
        only_p1 = set(p1) - set(p2)
        only_p2 = set(p2) - set(p1)
        raise AuditError(f"P1/P2 token key sets differ: only_in_p1={only_p1} only_in_p2={only_p2}")

    by_sentence: dict[tuple[str, str], dict] = {}
    for key, p1row in p1.items():
        p2row = p2[key]
        if p1row["token_form"] != p2row["token_form"]:
            raise AuditError(f"P1/P2 token form mismatch at {key}: {p1row['token_form']!r} vs {p2row['token_form']!r}")
        if p1row["source_sent_id"] != p2row["source_sent_id"]:
            raise AuditError(f"P1/P2 source_sent_id mismatch at {key}")
        if p1row["sample_set"] != p2row["sample_set"]:
            raise AuditError(f"P1/P2 sample_set mismatch at {key}")

        if p1row["upos"] == p2row["upos"]:
            final_upos, decision_source = p1row["upos"], "agreement"
        else:
            if key not in adjudicated:
                raise AuditError(
                    f"Disagreement at {key} (P1={p1row['upos']!r} P2={p2row['upos']!r}) has no adjudication record"
                )
            final_upos, decision_source = adjudicated[key], "adjudicated"

        sample_id, token_id = key
        sent_key = (p1row["sample_set"], sample_id)
        entry = by_sentence.setdefault(sent_key, {
            "source_sent_id": p1row["source_sent_id"],
            "sentence_text": p1row["sentence_text"],
            "tokens": {},
        })
        if entry["source_sent_id"] != p1row["source_sent_id"] or entry["sentence_text"] != p1row["sentence_text"]:
            raise AuditError(f"inconsistent sentence metadata within sample {sent_key}")
        entry["tokens"][token_id] = HumanToken(
            token_id=token_id, form=p1row["token_form"],
            final_upos=final_upos, decision_source=decision_source,
        )

    sentences: list[HumanSentence] = []
    for (sample_set, sample_id), entry in by_sentence.items():
        token_ids = sorted(entry["tokens"])
        expected_ids = list(range(1, len(token_ids) + 1))
        if token_ids != expected_ids:
            raise AuditError(f"non-contiguous token ids for {(sample_set, sample_id)}: {token_ids}")
        tokens = tuple(entry["tokens"][i] for i in expected_ids)
        sentences.append(HumanSentence(
            sample_set=sample_set, sample_id=sample_id,
            source_sent_id=entry["source_sent_id"], sentence_text=entry["sentence_text"],
            tokens=tokens,
        ))

    sentences.sort(key=lambda s: (s.sample_set, s.sample_id))
    return sentences


def reconstruct_human_sample(p1_path: Path, p2_path: Path, final_adj_path: Path) -> list[HumanSentence]:
    return reconstruct_human_sample_from_rows(
        load_p1_or_p2(p1_path), load_p1_or_p2(p2_path), load_final_adjudication(final_adj_path),
    )


def cross_check_against_results_workbook(sentences: list[HumanSentence], results_path: Path) -> dict:
    reference = load_results_workbook_reference(results_path)
    compared = form_mismatches = upos_mismatches = missing = 0
    mismatch_examples: list[dict] = []
    for sent in sentences:
        for tok in sent.tokens:
            key = (sent.sample_id, tok.token_id)
            ref = reference.get(key)
            if ref is None:
                missing += 1
                continue
            compared += 1
            if ref["form"] != tok.form:
                form_mismatches += 1
                mismatch_examples.append({"sample_id": sent.sample_id, "token_id": tok.token_id, "field": "form"})
            if ref["final_upos"] != tok.final_upos:
                upos_mismatches += 1
                mismatch_examples.append({"sample_id": sent.sample_id, "token_id": tok.token_id, "field": "final_upos"})
    return {
        "tokens_compared": compared,
        "tokens_missing_from_results_workbook": missing,
        "token_form_mismatches": form_mismatches,
        "final_upos_mismatches": upos_mismatches,
        "mismatch_examples": mismatch_examples[:20],
    }


# ---------------------------------------------------------------------------
# Authoritative split index (from the docsplit CoNLL-U files)
# ---------------------------------------------------------------------------

def get_comment(comments: list[str], prefix: str) -> str:
    full = f"# {prefix} = "
    for c in comments:
        if c.startswith(full):
            return c[len(full):]
    return ""


def load_split_sentences(paths: dict[str, Path]) -> list[SplitSentence]:
    all_sents: list[SplitSentence] = []
    for split, path in paths.items():
        if not path.exists():
            raise AuditError(f"Missing required split file: {path}")
        for sent in iter_sentences(path):
            sid = get_comment(sent["comments"], "sent_id")
            doc_id = get_comment(sent["comments"], "doc_id")
            text = get_comment(sent["comments"], "text")
            real = [t for t in sent["tokens"] if t["is_real"]]
            forms = tuple(t["fields"][F_FORM] for t in real)
            all_sents.append(SplitSentence(sent_id=sid, doc_id=doc_id, split=split, text=text, token_forms=forms))
    return all_sents


def build_sentid_index(all_sents: list[SplitSentence]) -> dict[str, list[SplitSentence]]:
    idx: dict[str, list[SplitSentence]] = defaultdict(list)
    for s in all_sents:
        idx[s.sent_id].append(s)
    return idx


def build_token_seq_index(all_sents: list[SplitSentence]) -> dict[tuple[str, ...], list[SplitSentence]]:
    idx: dict[tuple[str, ...], list[SplitSentence]] = defaultdict(list)
    for s in all_sents:
        idx[s.token_forms].append(s)
    return idx


def build_text_index(all_sents: list[SplitSentence]) -> dict[str, list[SplitSentence]]:
    idx: dict[str, list[SplitSentence]] = defaultdict(list)
    for s in all_sents:
        idx[s.text].append(s)
    return idx


def build_doc_to_splits(all_sents: list[SplitSentence]) -> dict[str, set[str]]:
    idx: dict[str, set[str]] = defaultdict(set)
    for s in all_sents:
        if s.doc_id:
            idx[s.doc_id].add(s.split)
    return idx


def find_doc_id_map_mismatches(all_sents: list[SplitSentence], doc_map: dict[str, str]) -> list[tuple[str, str, str]]:
    mismatches = []
    for s in all_sents:
        mapped = doc_map.get(s.sent_id)
        if mapped is not None and mapped != s.doc_id:
            mismatches.append((s.sent_id, s.doc_id, mapped))
    return mismatches


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_human_sentence(
    hs: HumanSentence,
    sentid_index: dict[str, list[SplitSentence]],
    token_seq_index: dict[tuple[str, ...], list[SplitSentence]],
    text_index: dict[str, list[SplitSentence]],
) -> MatchResult:
    notes: list[str] = []

    candidates = sentid_index.get(hs.source_sent_id, [])
    if len(candidates) == 1:
        cand = candidates[0]
        if cand.token_forms == hs.token_forms:
            return MatchResult(method="sent_id_exact", status="matched", split=cand.split, doc_id=cand.doc_id, notes="")
        notes.append(
            f"sent_id {hs.source_sent_id!r} found in split={cand.split} but token sequence differs "
            f"(human={len(hs.token_forms)} tokens, split={len(cand.token_forms)} tokens); "
            "falling back to token-sequence search"
        )
    elif len(candidates) > 1:
        notes.append(f"sent_id {hs.source_sent_id!r} ambiguous: appears in {len(candidates)} split entries")
        return MatchResult(method="sent_id_ambiguous", status="ambiguous", split=None, doc_id=None, notes="; ".join(notes))

    seq_candidates = token_seq_index.get(hs.token_forms, [])
    if len(seq_candidates) == 1:
        cand = seq_candidates[0]
        notes.append("matched via token-sequence fallback, not the persistent identifier")
        return MatchResult(method="token_sequence_exact", status="matched", split=cand.split, doc_id=cand.doc_id, notes="; ".join(notes))
    elif len(seq_candidates) > 1:
        notes.append(f"token sequence ambiguous: {len(seq_candidates)} candidate sentences share this exact token sequence")
        return MatchResult(method="token_sequence_ambiguous", status="ambiguous", split=None, doc_id=None, notes="; ".join(notes))

    text_candidates = text_index.get(hs.sentence_text, [])
    if len(text_candidates) == 1:
        cand = text_candidates[0]
        notes.append("matched via documented sentence-text fallback only; no identifier or token-sequence match found")
        return MatchResult(method="sentence_text_exact_fallback", status="matched", split=cand.split, doc_id=cand.doc_id, notes="; ".join(notes))
    elif len(text_candidates) > 1:
        notes.append(f"sentence text ambiguous across {len(text_candidates)} candidates; refusing to match by text alone")
        return MatchResult(method="sentence_text_ambiguous", status="ambiguous", split=None, doc_id=None, notes="; ".join(notes))

    notes.append("no match found via identifier, token sequence, or sentence text")
    return MatchResult(method="none", status="unmatched", split=None, doc_id=None, notes="; ".join(notes))


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sentence_csv(sentences: list[HumanSentence], results: dict[str, MatchResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "sample_set", "sample_id", "source_sent_id", "document_id", "split",
            "sentence_token_count", "matching_method", "match_status", "notes",
        ])
        for sent in sentences:
            r = results[sent.sample_id]
            w.writerow([
                sent.sample_set, sent.sample_id, sent.source_sent_id,
                r.doc_id or "", r.split or "", len(sent.tokens),
                r.method, r.status, r.notes,
            ])


def write_token_csv(sentences: list[HumanSentence], results: dict[str, MatchResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "sample_set", "sample_id", "source_sent_id", "document_id", "split",
            "token_id", "token_form", "final_human_upos",
        ])
        for sent in sentences:
            r = results[sent.sample_id]
            for tok in sent.tokens:
                w.writerow([
                    sent.sample_set, sent.sample_id, sent.source_sent_id,
                    r.doc_id or "", r.split or "",
                    tok.token_id, tok.form, tok.final_upos,
                ])


def build_summary(
    sentences: list[HumanSentence],
    results: dict[str, MatchResult],
    doc_to_splits: dict[str, set[str]],
    doc_id_map_mismatches: list[tuple[str, str, str]],
    cross_check: dict,
    checksums: dict[str, str],
) -> dict:
    sent_split_counts: dict[str, int] = defaultdict(int)
    tok_split_counts: dict[str, int] = defaultdict(int)
    sent_split_by_set: dict[str, dict[str, int]] = {
        "Representative": defaultdict(int), "Diagnostic": defaultdict(int),
    }
    tok_split_by_set: dict[str, dict[str, int]] = {
        "Representative": defaultdict(int), "Diagnostic": defaultdict(int),
    }
    method_counts: dict[str, int] = defaultdict(int)
    unmatched, ambiguous = [], []
    sample_docs_multi_split = []

    for sent in sentences:
        r = results[sent.sample_id]
        bucket = r.split if r.status == "matched" else r.status
        sent_split_counts[bucket] += 1
        tok_split_counts[bucket] += len(sent.tokens)
        sent_split_by_set[sent.sample_set][bucket] += 1
        tok_split_by_set[sent.sample_set][bucket] += len(sent.tokens)
        method_counts[r.method] += 1

        if r.status == "unmatched":
            unmatched.append({"sample_set": sent.sample_set, "sample_id": sent.sample_id, "source_sent_id": sent.source_sent_id})
        elif r.status == "ambiguous":
            ambiguous.append({"sample_set": sent.sample_set, "sample_id": sent.sample_id, "source_sent_id": sent.source_sent_id, "notes": r.notes})
        elif r.doc_id and len(doc_to_splits.get(r.doc_id, set())) > 1:
            sample_docs_multi_split.append({"sample_id": sent.sample_id, "doc_id": r.doc_id, "splits": sorted(doc_to_splits[r.doc_id])})

    docs_multi_split_corpus_wide = {doc: sorted(splits) for doc, splits in doc_to_splits.items() if len(splits) > 1}

    n_tokens_total = sum(len(s.tokens) for s in sentences)
    n_rep = sum(1 for s in sentences if s.sample_set == "Representative")
    n_diag = sum(1 for s in sentences if s.sample_set == "Diagnostic")
    tok_rep = sum(len(s.tokens) for s in sentences if s.sample_set == "Representative")
    tok_diag = sum(len(s.tokens) for s in sentences if s.sample_set == "Diagnostic")

    train_sent = sent_split_counts.get("train", 0)
    valid_held_out = train_sent == 0 and not unmatched and not ambiguous

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checksums_sha256": checksums,
        "reconstruction": {
            "sentences_total": len(sentences),
            "sentences_representative": n_rep,
            "sentences_diagnostic": n_diag,
            "tokens_total": n_tokens_total,
            "tokens_representative": tok_rep,
            "tokens_diagnostic": tok_diag,
            "all_60_sentences_recovered_exactly": len(sentences) == 60 and n_rep == 40 and n_diag == 20,
            "all_932_tokens_recovered_exactly": n_tokens_total == 932 and tok_rep == 653 and tok_diag == 279,
            "disagreements_adjudicated": sum(1 for s in sentences for t in s.tokens if t.decision_source == "adjudicated"),
            "results_workbook_cross_check": cross_check,
        },
        "split_membership": {
            "sentence_counts_by_bucket": dict(sent_split_counts),
            "token_counts_by_bucket": dict(tok_split_counts),
            "sentence_counts_by_bucket_and_sample_set": {k: dict(v) for k, v in sent_split_by_set.items()},
            "token_counts_by_bucket_and_sample_set": {k: dict(v) for k, v in tok_split_by_set.items()},
            "matching_method_counts": dict(method_counts),
            "unmatched_sentences": unmatched,
            "ambiguous_sentences": ambiguous,
            "documents_in_multiple_splits_corpus_wide": {
                "count": len(docs_multi_split_corpus_wide),
                "examples": dict(list(docs_multi_split_corpus_wide.items())[:20]),
            },
            "documents_in_multiple_splits_among_sample": {
                "count": len(sample_docs_multi_split),
                "examples": sample_docs_multi_split,
            },
            "doc_id_map_cross_check_mismatches": len(doc_id_map_mismatches),
        },
        "evaluation_validity": {
            "can_be_used_as_held_out_test_set_as_is": valid_held_out,
            "reason": (
                "Valid: no sample sentence falls in the train split and every sentence resolved "
                "without ambiguity."
                if valid_held_out else
                f"Invalid as-is: {train_sent} of {len(sentences)} sample sentences belong to the "
                f"train split (train sentences cannot be reported as held-out generalization "
                f"performance), plus {len(unmatched)} unmatched and {len(ambiguous)} ambiguous "
                "sentences. Restrict analysis to the dev+test subset, or construct a new "
                "test-only human evaluation sample, per outputs/human_validation/Human_Validation_README.md."
            ),
        },
    }


def write_summary_txt(summary: dict, path: Path) -> None:
    lines = ["Human Validation Sample -- Split Membership Audit", "=" * 70, ""]
    lines.append(f"generated_at_utc = {summary['generated_at_utc']}")
    lines.append("")
    lines.append("## Checksums (SHA-256)")
    for name, h in summary["checksums_sha256"].items():
        lines.append(f"  {name}: {h}")
    lines.append("")
    lines.append("## Reconstruction")
    for k, v in summary["reconstruction"].items():
        if k == "results_workbook_cross_check":
            continue
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("## Split membership")
    sm = summary["split_membership"]
    lines.append(f"  sentence_counts_by_bucket: {sm['sentence_counts_by_bucket']}")
    lines.append(f"  token_counts_by_bucket: {sm['token_counts_by_bucket']}")
    lines.append(f"  sentence_counts_by_bucket_and_sample_set: {sm['sentence_counts_by_bucket_and_sample_set']}")
    lines.append(f"  matching_method_counts: {sm['matching_method_counts']}")
    lines.append(f"  unmatched_sentences: {len(sm['unmatched_sentences'])}")
    lines.append(f"  ambiguous_sentences: {len(sm['ambiguous_sentences'])}")
    lines.append(f"  documents_in_multiple_splits_corpus_wide: {sm['documents_in_multiple_splits_corpus_wide']['count']}")
    lines.append(f"  documents_in_multiple_splits_among_sample: {sm['documents_in_multiple_splits_among_sample']['count']}")
    lines.append(f"  doc_id_map_cross_check_mismatches: {sm['doc_id_map_cross_check_mismatches']}")
    lines.append("")
    lines.append("## Evaluation validity")
    ev = summary["evaluation_validity"]
    lines.append(f"  can_be_used_as_held_out_test_set_as_is: {ev['can_be_used_as_held_out_test_set_as_is']}")
    lines.append(f"  reason: {ev['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(summary: dict, path: Path) -> None:
    sm = summary["split_membership"]
    ev = summary["evaluation_validity"]
    text = f"""# Human Validation Sample -- Split Membership Audit

Audits the train/dev/test split membership of the 60-sentence finalized
human-validation sample (40 Representative + 20 Diagnostic, 932 tokens
total). This directory answers the "next required step" noted in
`outputs/human_validation/Human_Validation_README.md`: whether the human
reference can be used as a held-out generalization benchmark.

The canonical corpus, the split files, model outputs, and the existing
human-validation workbooks are read-only inputs. Nothing outside this
directory is modified.

## Method

1. Reconstruct all 932 tokens across the 60 sentences from the P1 and P2
   independent annotation workbooks (`Human_POS_Tagging_v3_P1.xlsx`,
   `Human_POS_Tagging_v3_P2.xlsx`), taking the shared label where P1 and P2
   agree and the blind-adjudicated label
   (`Human_POS_Tagging_v3_Blind_Adjudication_FINAL.xlsx`, sheet
   `Adjudication_51`) for the 51 disagreements. Cross-checked (read-only)
   against the independently pre-merged `Final_Reference_932` sheet in
   `Human_POS_Tagging_v3_Final_Adjudicated_Results.xlsx`.
2. Match each of the 60 sentences to the authoritative document-level split
   (`data/processed/panyo_original_adp_docsplit_{{train,dev,test}}.conllu`,
   per `guidelines/data_split_decisions.md`), in strict priority order:
   1. the persistent `sent_id` (e.g. `pangyo829`),
   2. the exact ordered token-form sequence (used only if the identifier
      lookup fails or its tokens don't match the human sample's tokens),
   3. the exact sentence text (used only if both of the above fail; a
      documented fallback, not a normalized/fuzzy match).
   Any stage that returns more than one candidate is reported as
   **ambiguous**, never guessed from a weaker signal.

## Files

- `sentence_split_membership.csv` -- one row per sentence: split, matching
  method, match status, notes.
- `token_split_membership.csv` -- one row per token (932 rows): split,
  token form, final adjudicated human UPOS.
- `summary.json` / `summary.txt` -- counts by split and sample set,
  matching-method counts, unmatched/ambiguous cases, document/split
  integrity checks, checksums.
- `command_log.txt` -- every command executed.

## Headline result

Sentence counts by split: {sm['sentence_counts_by_bucket']}
Token counts by split: {sm['token_counts_by_bucket']}

**Can the current 60-sentence sample be used as-is for held-out model
evaluation? {ev['can_be_used_as_held_out_test_set_as_is']}**

{ev['reason']}

## How to reproduce

```bash
python scripts/tests/test_audit_human_validation_split_membership.py
python scripts/audit_human_validation_split_membership.py
```
"""
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reconstructing 932-token human-validation reference from P1/P2/adjudication ...")
    sentences = reconstruct_human_sample(P1_XLSX, P2_XLSX, FINAL_ADJ_XLSX)
    print(f"  {len(sentences)} sentences, {sum(len(s.tokens) for s in sentences)} tokens")

    print("Cross-checking against Final_Reference_932 (read-only) ...")
    cross_check = cross_check_against_results_workbook(sentences, RESULTS_XLSX)
    print(f"  {cross_check}")

    print("Loading authoritative document-level split (docsplit train/dev/test) ...")
    all_split_sents = load_split_sentences(DOCSPLIT_PATHS)
    print(f"  {len(all_split_sents)} split sentences indexed")
    sentid_index = build_sentid_index(all_split_sents)
    token_seq_index = build_token_seq_index(all_split_sents)
    text_index = build_text_index(all_split_sents)
    doc_to_splits = build_doc_to_splits(all_split_sents)

    doc_map: dict[str, str] = {}
    if DOC_MAP_CSV.exists():
        with DOC_MAP_CSV.open(encoding="utf-8") as f:
            doc_map = {row["sent_id"]: row["doc_id"] for row in csv.DictReader(f)}
    doc_id_map_mismatches = find_doc_id_map_mismatches(all_split_sents, doc_map)
    print(f"  doc_id map cross-check mismatches: {len(doc_id_map_mismatches)}")

    print("Matching sentences ...")
    results: dict[str, MatchResult] = {}
    for sent in sentences:
        results[sent.sample_id] = match_human_sentence(sent, sentid_index, token_seq_index, text_index)

    checksums = {
        str(p.relative_to(ROOT)).replace("\\", "/"): sha256_of(p)
        for p in [P1_XLSX, P2_XLSX, FINAL_ADJ_XLSX, RESULTS_XLSX, DOC_MAP_CSV, *DOCSPLIT_PATHS.values()]
    }

    summary = build_summary(sentences, results, doc_to_splits, doc_id_map_mismatches, cross_check, checksums)

    print("Writing outputs ...")
    write_sentence_csv(sentences, results, OUT_DIR / "sentence_split_membership.csv")
    write_token_csv(sentences, results, OUT_DIR / "token_split_membership.csv")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_txt(summary, OUT_DIR / "summary.txt")
    write_readme(summary, OUT_DIR / "README.md")

    print(f"\nSentence counts by split: {summary['split_membership']['sentence_counts_by_bucket']}")
    print(f"Token counts by split: {summary['split_membership']['token_counts_by_bucket']}")
    print(f"Held-out-valid-as-is: {summary['evaluation_validity']['can_be_used_as_held_out_test_set_as_is']}")
    print("\nDone.")


if __name__ == "__main__":
    main()
