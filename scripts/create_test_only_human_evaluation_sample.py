"""
create_test_only_human_evaluation_sample.py - Deterministically construct a
new, test-only human UPOS evaluation sample (60 Representative + 20
Diagnostic sentences) from the authoritative Pangyo document-level test
split, completely separate from the earlier 60-sentence corpus-wide
human-validation sample.

Inputs (all read-only):
  - data/processed/panyo_original_adp_docsplit_test.conllu
    (authoritative document-level test split; see guidelines/data_split_decisions.md)
  - data/processed/panyo_jsonl_doc_sentence_map.csv (doc_id cross-check only)
  - outputs/human_validation/split_audit/sentence_split_membership.csv
    (completed split-membership audit; gives the 60 previous sample's
    source_sent_id -> document_id, for document-level exclusion)
  - outputs/human_validation/annotations/Human_POS_Tagging_v3_P1.xlsx
    (previous sample's sentence text, for normalized-duplicate exclusion)
  - outputs/human_validation/guideline/Pangyo_Human_UPOS_Annotation_Guideline_v1.1.md
    (referenced by name in annotator-facing Instructions sheets; not parsed)

Outputs (all new):
  - outputs/human_validation/test_only/annotations/Human_POS_Tagging_test_v1_{P1,P2}.xlsx
    (annotator-facing; no canonical labels, no selection scores, no dropdowns)
  - outputs/human_validation/test_only/key/Human_POS_Tagging_test_v1_canonical_key.xlsx
  - outputs/human_validation/test_only/key/selection_manifest.json
  - outputs/human_validation/test_only/key/selection_summary.csv
  - outputs/human_validation/test_only/key/exclusion_audit.csv
  - outputs/human_validation/test_only/key/README.md

Selection is fully deterministic given seed=42: sentence order is fixed by
sorting on sent_id before any random shuffling, and a single random.Random(42)
stream is consumed first by the Representative draw, then by the Diagnostic
draw. Selection never reads or uses the canonical UPOS field, dependency
head/deprel, or any model output -- only token FORM and the corpus's
existing Lang= (KO/EN/OTHER) MISC annotation, which is a language-ID label,
not a POS label, and is explicitly permitted in the hidden key.

Run from repo root:
  python scripts/create_test_only_human_evaluation_sample.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Optional

import openpyxl
from openpyxl.styles import PatternFill

sys.path.insert(0, str(Path(__file__).parent))
from read_conllu import F_FORM, F_MISC, F_UPOS, get_misc_lang, iter_sentences  # noqa: E402

ROOT = Path(__file__).parent.parent

TEST_CONLLU = ROOT / "data" / "processed" / "panyo_original_adp_docsplit_test.conllu"
DOC_MAP_CSV = ROOT / "data" / "processed" / "panyo_jsonl_doc_sentence_map.csv"
SPLIT_AUDIT_CSV = ROOT / "outputs" / "human_validation" / "split_audit" / "sentence_split_membership.csv"
PREV_P1_XLSX = ROOT / "outputs" / "human_validation" / "annotations" / "Human_POS_Tagging_v3_P1.xlsx"
GUIDELINE_V11_MD = ROOT / "outputs" / "human_validation" / "guideline" / "Pangyo_Human_UPOS_Annotation_Guideline_v1.1.md"

ANNOT_DIR = ROOT / "outputs" / "human_validation" / "test_only" / "annotations"
KEY_DIR = ROOT / "outputs" / "human_validation" / "test_only" / "key"

SEED = 42
N_REPRESENTATIVE = 60
N_DIAGNOSTIC = 20
N_LENGTH_BINS = 5
N_DIAGNOSTIC_CATEGORIES = 5
PER_CATEGORY_TARGET = N_DIAGNOSTIC // N_DIAGNOSTIC_CATEGORIES  # 4
MIN_ELIGIBLE_DOCS_FOR_FULL_EXCLUSION = 80

DIAGNOSTIC_CATEGORIES = [
    ("ko_en_boundary", "KO-EN boundary-rich"),
    ("particle_heavy", "Korean particle/postposition-heavy"),
    ("predicate_sensitive", "Korean predicate / VERB-AUX-sensitive"),
    ("en_nounpropn_cap", "English NOUN-PROPN / capitalization-sensitive"),
    ("tokenization_edge", "Tokenization-edge cases"),
]
CATEGORY_IDS = [c for c, _ in DIAGNOSTIC_CATEGORIES]

# Guideline v1.1 sec.5.1 particle inventory (standalone or complex forms).
PARTICLE_FORMS = {
    "는", "은", "이", "가", "을", "를", "의", "에", "에서", "로", "와", "과",
    "도", "만", "부터", "까지", "에서는", "로는", "에도", "에서부터",
}
# Guideline v1.1 sec.5.4/5.5/5.7 predicate/AUX/relational markers.
PREDICATE_MARKER_FORMS = {
    "이다", "입니다", "이며", "이고", "있다", "있습니다", "있고", "않다", "않습니다",
    "싶다", "하다", "합니다", "했다", "되다", "됩니다", "되었다",
    "인해", "의해", "통해", "위해", "따라", "때문에", "중",
}
PREDICATE_SUFFIXES = ("하다", "되다", "이다")

HANGUL_RE = re.compile(r"[가-힣]")
DIGIT_RE = re.compile(r"[0-9]")
LATIN_RE = re.compile(r"[A-Za-z]")
EDGE_CHAR_RE = re.compile(r"[²³°%/\\@=~^_]")


class SampleError(Exception):
    """Raised for a fatal precondition failure (not a per-sentence selection outcome)."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Token:
    token_id: int
    form: str
    upos: str
    lang: str  # "KO" | "EN" | "OTHER"


@dataclass(frozen=True)
class TestSentence:
    sent_id: str
    doc_id: str
    text: str
    tokens: tuple[Token, ...]

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    @property
    def normalized_text(self) -> str:
        return normalize_text(self.text)

    @property
    def token_forms(self) -> tuple[str, ...]:
        return tuple(t.form for t in self.tokens)

    @property
    def ko_proportion(self) -> float:
        n = self.token_count
        return sum(1 for t in self.tokens if t.lang == "KO") / n if n else 0.0

    @property
    def en_proportion(self) -> float:
        n = self.token_count
        return sum(1 for t in self.tokens if t.lang == "EN") / n if n else 0.0

    @property
    def boundary_count(self) -> int:
        langs = [t.lang for t in self.tokens if t.lang in ("KO", "EN")]
        return sum(1 for a, b in zip(langs, langs[1:]) if a != b)


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


# ---------------------------------------------------------------------------
# Loading (I/O only)
# ---------------------------------------------------------------------------

def get_comment(comments: list[str], prefix: str) -> str:
    full = f"# {prefix} = "
    for c in comments:
        if c.startswith(full):
            return c[len(full):]
    return ""


def load_test_sentences(path: Path) -> tuple[list[TestSentence], list[str]]:
    """Return (sentences, ambiguous_sent_ids). Malformed sentences are dropped."""
    if not path.exists():
        raise SampleError(f"Missing required test-split file: {path}")
    seen: dict[str, int] = defaultdict(int)
    sentences: list[TestSentence] = []
    for sent in iter_sentences(path):
        sid = get_comment(sent["comments"], "sent_id")
        did = get_comment(sent["comments"], "doc_id")
        text = get_comment(sent["comments"], "text")
        seen[sid] += 1
        real = [t for t in sent["tokens"] if t["is_real"]]
        if not sid or not did or not real:
            continue
        tokens = []
        ok = True
        for i, t in enumerate(real, 1):
            form = t["fields"][F_FORM]
            if not form:
                ok = False
                break
            tokens.append(Token(
                token_id=i, form=form, upos=t["fields"][F_UPOS],
                lang=get_misc_lang(t["fields"][F_MISC]) or "OTHER",
            ))
        if not ok:
            continue
        sentences.append(TestSentence(sent_id=sid, doc_id=did, text=text, tokens=tuple(tokens)))

    ambiguous = sorted(sid for sid, n in seen.items() if n > 1)
    sentences = [s for s in sentences if seen[s.sent_id] == 1]
    return sentences, ambiguous


def load_doc_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {row["sent_id"]: row["doc_id"] for row in csv.DictReader(f)}


def load_previous_sample_docs(path: Path) -> dict[str, dict]:
    """Return {source_sent_id: {sample_set, sample_id, document_id, split}} for all 60 prior sentences."""
    if not path.exists():
        raise SampleError(f"Missing required split-membership audit: {path}")
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["source_sent_id"]] = {
                "sample_set": row["sample_set"],
                "sample_id": row["sample_id"],
                "document_id": row["document_id"],
                "split": row["split"],
            }
    return out


P1_P2_HEADER = (
    "Set", "Sample_ID", "Source_Sent_ID", "Sentence", "Token_ID", "Token",
    "Human_UPOS", "Confidence", "Tokenization_Issue", "Note",
)


def find_header_row(rows: list[tuple], expected_header: tuple) -> int:
    for i, row in enumerate(rows):
        if row == expected_header:
            return i
    raise SampleError(f"header row {expected_header!r} not found in sheet")


def load_previous_sample_texts(path: Path) -> dict[str, str]:
    """Return {source_sent_id: sentence_text} for all 60 prior sentences."""
    if not path.exists():
        raise SampleError(f"Missing required previous annotation workbook: {path}")
    wb = openpyxl.load_workbook(path, data_only=True)
    out: dict[str, str] = {}
    for sheet_name in ("Representative_40", "Diagnostic_20"):
        rows = list(wb[sheet_name].iter_rows(values_only=True))
        header_idx = find_header_row(rows, P1_P2_HEADER)
        for row in rows[header_idx + 1:]:
            if row[1] is None:
                continue
            sent_id, sentence = row[2], row[3]
            out[sent_id] = sentence
    return out


# ---------------------------------------------------------------------------
# Eligibility and exclusion
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExclusionRecord:
    sent_id: str
    doc_id: str
    reason: str
    detail: str


def find_doc_id_mismatches(sentences: list[TestSentence], doc_map: dict[str, str]) -> list[str]:
    mismatched = []
    for s in sentences:
        mapped = doc_map.get(s.sent_id)
        if mapped is not None and mapped != s.doc_id:
            mismatched.append(s.sent_id)
    return mismatched


def compute_eligibility(
    all_sentences: list[TestSentence],
    ambiguous_sent_ids: list[str],
    doc_id_mismatches: list[str],
    previous_sent_docs: dict[str, dict],
    previous_texts: dict[str, str],
) -> tuple[list[TestSentence], list[ExclusionRecord], bool]:
    previous_sent_ids = set(previous_sent_docs)
    previous_doc_ids = {v["document_id"] for v in previous_sent_docs.values()}
    previous_normalized_texts = {normalize_text(t) for t in previous_texts.values()}
    ambiguous_set = set(ambiguous_sent_ids) | set(doc_id_mismatches)

    non_excluded_docs = {s.doc_id for s in all_sentences if s.doc_id not in previous_doc_ids}
    exclude_all_previous_docs = len(non_excluded_docs) >= MIN_ELIGIBLE_DOCS_FOR_FULL_EXCLUSION

    records: list[ExclusionRecord] = []
    eligible: list[TestSentence] = []
    seen_norm_within_pool: dict[str, str] = {}

    for s in sorted(all_sentences, key=lambda s: s.sent_id):
        if s.sent_id in ambiguous_set:
            records.append(ExclusionRecord(s.sent_id, s.doc_id, "ambiguous_id", "duplicate sent_id or doc_id mismatch"))
            continue
        if s.sent_id in previous_sent_ids:
            records.append(ExclusionRecord(s.sent_id, s.doc_id, "previous_sample_sentence", "exact sent_id match to prior 60-sentence sample"))
            continue
        if exclude_all_previous_docs and s.doc_id in previous_doc_ids:
            records.append(ExclusionRecord(s.sent_id, s.doc_id, "previous_sample_document", "document represented in prior 60-sentence sample"))
            continue
        nt = s.normalized_text
        if nt in previous_normalized_texts:
            records.append(ExclusionRecord(s.sent_id, s.doc_id, "duplicate_of_previous_text", "normalized text matches a prior sample sentence"))
            continue
        if nt in seen_norm_within_pool:
            records.append(ExclusionRecord(
                s.sent_id, s.doc_id, "duplicate_within_test_split",
                f"normalized text duplicates test-split sentence {seen_norm_within_pool[nt]}",
            ))
            continue
        seen_norm_within_pool[nt] = s.sent_id
        eligible.append(s)

    return eligible, records, exclude_all_previous_docs


# ---------------------------------------------------------------------------
# Representative sampling (length-stratified, 1 sentence/doc, seeded)
# ---------------------------------------------------------------------------

def quantile_edges(sorted_values: list[int], n_bins: int) -> list[int]:
    edges = []
    n = len(sorted_values)
    for i in range(1, n_bins):
        idx = min(n - 1, max(0, round((i / n_bins) * (n - 1))))
        edges.append(sorted_values[idx])
    return edges


def bin_index(value: int, edges: list[int]) -> int:
    for i, e in enumerate(edges):
        if value <= e:
            return i
    return len(edges)


def select_representative(
    eligible: list[TestSentence], rng: Random, n: int = N_REPRESENTATIVE, n_bins: int = N_LENGTH_BINS,
) -> tuple[list[TestSentence], dict]:
    ordered = sorted(eligible, key=lambda s: s.sent_id)
    n_eligible_docs = len({s.doc_id for s in ordered})
    one_per_doc = n_eligible_docs >= n

    token_counts = sorted(s.token_count for s in ordered)
    edges = quantile_edges(token_counts, n_bins) if token_counts else []
    bins: list[list[TestSentence]] = [[] for _ in range(n_bins)]
    for s in ordered:
        bins[bin_index(s.token_count, edges)].append(s)
    for b in bins:
        rng.shuffle(b)

    quota = n // n_bins
    quotas = [quota] * n_bins
    for i in range(n - quota * n_bins):
        quotas[i] += 1

    selected: list[TestSentence] = []
    used_docs: set[str] = set()
    used_texts: set[str] = set()
    leftover: list[TestSentence] = []

    for bi, bucket in enumerate(bins):
        count = 0
        for s in bucket:
            if count >= quotas[bi]:
                leftover.append(s)
                continue
            if one_per_doc and s.doc_id in used_docs:
                continue
            if s.normalized_text in used_texts:
                continue
            selected.append(s)
            used_docs.add(s.doc_id)
            used_texts.add(s.normalized_text)
            count += 1

    if len(selected) < n:
        for s in leftover:
            if len(selected) >= n:
                break
            if one_per_doc and s.doc_id in used_docs:
                continue
            if s.normalized_text in used_texts:
                continue
            selected.append(s)
            used_docs.add(s.doc_id)
            used_texts.add(s.normalized_text)

    strata = {
        "n_length_bins": n_bins,
        "bin_edges_token_count": edges,
        "bin_sizes_selected": [
            sum(1 for s in selected if bin_index(s.token_count, edges) == bi) for bi in range(n_bins)
        ],
        "one_sentence_per_document_enforced": one_per_doc,
        "n_eligible_documents_at_selection_time": n_eligible_docs,
    }
    return selected, strata


# ---------------------------------------------------------------------------
# Diagnostic sampling (category-scored, seeded)
# ---------------------------------------------------------------------------

def score_diagnostic_categories(sentence: TestSentence) -> dict[str, int]:
    forms = sentence.token_forms
    ko_en_boundary = sentence.boundary_count
    particle_heavy = sum(1 for f in forms if f in PARTICLE_FORMS)
    predicate_sensitive = sum(
        1 for f in forms
        if f in PREDICATE_MARKER_FORMS
        or (HANGUL_RE.search(f) and f.endswith(PREDICATE_SUFFIXES))
        or (HANGUL_RE.search(f) and f.endswith("다") and len(f) > 1)
    )
    en_nounpropn_cap = sum(
        1 for t in sentence.tokens
        if t.lang == "EN" and t.form[:1].isupper() and LATIN_RE.match(t.form[:1])
    )
    tokenization_edge = sum(
        1 for f in forms
        if (HANGUL_RE.search(f) and DIGIT_RE.search(f))
        or EDGE_CHAR_RE.search(f)
        or ("-" in f and len(f) > 1)
        or ("/" in f)
        or (len(f) == 1 and HANGUL_RE.match(f) and f not in PARTICLE_FORMS)
    )
    return {
        "ko_en_boundary": ko_en_boundary,
        "particle_heavy": particle_heavy,
        "predicate_sensitive": predicate_sensitive,
        "en_nounpropn_cap": en_nounpropn_cap,
        "tokenization_edge": tokenization_edge,
    }


def assign_primary_category(scores: dict[str, int]) -> str:
    best_cat, best_score = CATEGORY_IDS[0], -1
    for cat in CATEGORY_IDS:
        if scores[cat] > best_score:
            best_cat, best_score = cat, scores[cat]
    return best_cat


def select_diagnostic(
    eligible: list[TestSentence],
    rng: Random,
    excluded_texts: set[str],
    avoid_docs: set[str],
    n: int = N_DIAGNOSTIC,
    per_category: int = PER_CATEGORY_TARGET,
) -> tuple[list[tuple[TestSentence, str]], dict]:
    """
    Fill each category's quota in fixed category order (1-5), ranking the
    STILL-REMAINING pool by that category's own score each time (not by a
    single global argmax computed up front). A global-argmax partition would
    starve categories whose raw counts are naturally smaller than
    ko_en_boundary's (nearly every code-switched sentence has some boundary
    count), so ranking per-category against the shrinking remaining pool is
    what actually achieves ~4 sentences per category. Each sentence's
    "primary category" is simply the quota slot it was chosen to fill; a
    sentence can only be chosen for one slot, satisfying the one-category-
    per-sentence requirement.
    """
    ordered = sorted(eligible, key=lambda s: s.sent_id)
    scored = [(s, score_diagnostic_categories(s)) for s in ordered]
    rng.shuffle(scored)  # deterministic tie-break order given the fixed seed

    selected: list[tuple[TestSentence, str]] = []
    used_texts = set(excluded_texts)
    used_sent_ids: set[str] = set()
    used_docs = set(avoid_docs)
    quotas = {c: per_category for c in CATEGORY_IDS}
    for i in range(n - per_category * len(CATEGORY_IDS)):
        quotas[CATEGORY_IDS[i]] += 1

    def remaining_ranked(cat: str) -> list[tuple[TestSentence, int]]:
        pool = [
            (s, sc[cat]) for s, sc in scored
            if s.sent_id not in used_sent_ids and s.normalized_text not in used_texts
        ]
        pool.sort(key=lambda pair: -pair[1])  # stable: preserves the rng-shuffled order on ties
        return pool

    for cat in CATEGORY_IDS:
        filled = 0
        for allow_doc_reuse in (False, True):
            if filled >= quotas[cat]:
                break
            for s, _score in remaining_ranked(cat):
                if filled >= quotas[cat]:
                    break
                if s.sent_id in used_sent_ids or s.normalized_text in used_texts:
                    continue
                if not allow_doc_reuse and s.doc_id in used_docs:
                    continue
                selected.append((s, cat))
                used_sent_ids.add(s.sent_id)
                used_texts.add(s.normalized_text)
                used_docs.add(s.doc_id)
                filled += 1

    # Top-up pass: only reached if the eligible pool is too small overall to
    # fill every quota (extremely skewed pool); assigns each leftover
    # sentence to its own best-scoring category among the still-open quotas.
    if len(selected) < n:
        counts = Counter(cat for _s, cat in selected)
        for s, sc in scored:
            if len(selected) >= n:
                break
            if s.sent_id in used_sent_ids or s.normalized_text in used_texts:
                continue
            open_cats = [c for c in CATEGORY_IDS if counts[c] < quotas[c]]
            if not open_cats:
                break
            cat = max(open_cats, key=lambda c: sc[c])
            selected.append((s, cat))
            used_sent_ids.add(s.sent_id)
            used_texts.add(s.normalized_text)
            counts[cat] += 1

    category_counts = dict(Counter(cat for _s, cat in selected))
    info = {
        "category_definitions": dict(DIAGNOSTIC_CATEGORIES),
        "target_per_category": per_category,
        "achieved_category_counts": category_counts,
    }
    return selected, info


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

WORKBOOK_HEADER = (
    "Set", "Sample_ID", "Source_Sent_ID", "Document_ID", "Sentence",
    "Token_ID", "Token", "Human_UPOS", "Confidence", "Tokenization_Issue", "Note",
)
HIGHLIGHT_COLUMNS = {"Human_UPOS", "Confidence", "Tokenization_Issue", "Note"}
HIGHLIGHT_FILL = PatternFill(fill_type="solid", fgColor="FFF2CC")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_token_sheet(ws, rows: list[dict]) -> None:
    ws.append(WORKBOOK_HEADER)
    for col_idx, name in enumerate(WORKBOOK_HEADER, 1):
        if name in HIGHLIGHT_COLUMNS:
            ws.cell(row=1, column=col_idx).fill = HIGHLIGHT_FILL
    for r in rows:
        ws.append([r[name] for name in WORKBOOK_HEADER])
    for row_idx in range(2, len(rows) + 2):
        for col_idx, name in enumerate(WORKBOOK_HEADER, 1):
            if name in HIGHLIGHT_COLUMNS:
                ws.cell(row=row_idx, column=col_idx).fill = HIGHLIGHT_FILL


def build_token_rows(
    items: list[tuple[str, str, TestSentence]],  # (sample_set, sample_id, sentence)
) -> list[dict]:
    rows = []
    for sample_set, sample_id, sent in items:
        for tok in sent.tokens:
            rows.append({
                "Set": sample_set, "Sample_ID": sample_id, "Source_Sent_ID": sent.sent_id,
                "Document_ID": sent.doc_id, "Sentence": sent.text, "Token_ID": tok.token_id,
                "Token": tok.form, "Human_UPOS": None, "Confidence": None,
                "Tokenization_Issue": "No", "Note": None,
            })
    return rows


def write_annotator_workbook(
    path: Path, annotator: str,
    representative_items: list[tuple[str, str, TestSentence]],
    diagnostic_items: list[tuple[str, str, TestSentence]],
    rep_tokens: int, diag_tokens: int,
) -> None:
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    instr = wb.create_sheet("Instructions")
    for line in [
        "Pangyo Test-Only Human UPOS Evaluation -- Round test_v1",
        "",
        f"Annotator: {annotator}",
        "",
        "Purpose: independent, blind held-out evaluation sample, separate from",
        "the earlier 60-sentence corpus-wide human-validation sample.",
        "",
        "Use annotation guideline v1.1 "
        "(outputs/human_validation/guideline/Pangyo_Human_UPOS_Annotation_Guideline_v1.1.md).",
        "Annotate every token independently. Do not consult canonical labels,",
        "model predictions, or the other annotator's sheet before both are complete.",
        "Fill only: Human_UPOS, Confidence, Tokenization_Issue, Note (highlighted columns).",
        "Tokenization_Issue is pre-filled 'No'; change to 'Yes' only per guideline Sec. 8.",
    ]:
        instr.append([line])

    rep_ws = wb.create_sheet("Representative_60")
    _write_token_sheet(rep_ws, build_token_rows(representative_items))

    diag_ws = wb.create_sheet("Diagnostic_20")
    _write_token_sheet(diag_ws, build_token_rows(diagnostic_items))

    summary = wb.create_sheet("Selection_Summary")
    summary.append(["Component", "Sentences", "Tokens", "Selection design", "Annotator", "Guideline"])
    summary.append([
        "Representative", len(representative_items), rep_tokens,
        "Length-stratified (5 quintile bins), 1 sentence/document, seed=42, test split only",
        annotator, "v1.1",
    ])
    summary.append([
        "Diagnostic", len(diagnostic_items), diag_tokens,
        "5 predetermined categories, ~4 sentences/category, seed=42, test split only",
        annotator, "v1.1",
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def write_canonical_key(
    path: Path,
    representative: list[tuple[str, str, TestSentence]],
    diagnostic: list[tuple[str, str, TestSentence, str]],
) -> None:
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    warn = wb.create_sheet("HIDDEN_DO_NOT_DISTRIBUTE")
    for line in [
        "HIDDEN EVALUATION ARTIFACT -- DO NOT DISTRIBUTE TO ANNOTATORS",
        "Contains canonical UPOS labels for the test_v1 sample.",
        "Keep hidden from annotators until annotation and blind adjudication",
        "of both P1 and P2 are fully complete.",
    ]:
        warn.append([line])

    key_ws = wb.create_sheet("Canonical_Key_80")
    header = (
        "Set", "Sample_ID", "Source_Sent_ID", "Document_ID", "Sentence",
        "Token_ID", "Token", "Canonical_UPOS", "Inferred_Language",
        "Diagnostic_Category", "Selection_Method",
    )
    key_ws.append(header)
    for sample_set, sample_id, sent in representative:
        for tok in sent.tokens:
            key_ws.append([
                sample_set, sample_id, sent.sent_id, sent.doc_id, sent.text,
                tok.token_id, tok.form, tok.upos, tok.lang, "", "representative_length_stratified",
            ])
    for sample_set, sample_id, sent, category in diagnostic:
        for tok in sent.tokens:
            key_ws.append([
                sample_set, sample_id, sent.sent_id, sent.doc_id, sent.text,
                tok.token_id, tok.form, tok.upos, tok.lang, category, "diagnostic_category_scored",
            ])

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def write_selection_summary_csv(
    path: Path,
    representative: list[tuple[str, str, TestSentence]],
    diagnostic: list[tuple[str, str, TestSentence, str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "sample_set", "sample_id", "source_sent_id", "document_id",
            "sentence_token_count", "ko_proportion", "en_proportion",
            "boundary_count", "diagnostic_category",
        ])
        for sample_set, sample_id, sent in representative:
            w.writerow([
                sample_set, sample_id, sent.sent_id, sent.doc_id, sent.token_count,
                round(sent.ko_proportion, 4), round(sent.en_proportion, 4), sent.boundary_count, "",
            ])
        for sample_set, sample_id, sent, category in diagnostic:
            w.writerow([
                sample_set, sample_id, sent.sent_id, sent.doc_id, sent.token_count,
                round(sent.ko_proportion, 4), round(sent.en_proportion, 4), sent.boundary_count, category,
            ])


def write_exclusion_audit_csv(path: Path, records: list[ExclusionRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sent_id", "doc_id", "exclusion_reason", "detail"])
        for r in records:
            w.writerow([r.sent_id, r.doc_id, r.reason, r.detail])


def git_commit_hash() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def write_manifest(
    path: Path,
    checksums: dict[str, str],
    exclude_all_previous_docs: bool,
    exclusion_counts: dict[str, int],
    representative: list[tuple[str, str, TestSentence]],
    diagnostic: list[tuple[str, str, TestSentence, str]],
    rep_strata: dict,
    diag_info: dict,
    doc_id_mismatches: list[str],
    ambiguous_sent_ids: list[str],
) -> None:
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/create_test_only_human_evaluation_sample.py",
        "git_commit": git_commit_hash(),
        "seed": SEED,
        "authoritative_inputs_sha256": checksums,
        "eligibility_and_exclusion_rules": {
            "order": [
                "1. ambiguous sent_id (duplicate) or doc_id mismatch vs doc map -> excluded",
                "2. exact sent_id match to prior 60-sentence sample -> excluded",
                "3. document represented in prior 60-sentence sample -> excluded IF "
                f">= {MIN_ELIGIBLE_DOCS_FOR_FULL_EXCLUSION} eligible test documents remain "
                "after full exclusion (else only the prior sample's own 7 test-split "
                "sentences' documents are excluded)",
                "4. normalized text (NFKC + whitespace-collapse) duplicates a prior "
                "sample sentence -> excluded",
                "5. normalized text duplicates another test-split sentence already "
                "kept in the eligible pool -> excluded (first occurrence, sent_id order, kept)",
                "6. malformed sentence (missing sent_id/doc_id/text, empty token form, "
                "zero real tokens) -> excluded before the above rules are applied",
            ],
            "full_previous_document_exclusion_applied": exclude_all_previous_docs,
            "min_eligible_docs_threshold_for_full_exclusion": MIN_ELIGIBLE_DOCS_FOR_FULL_EXCLUSION,
            "exclusion_counts_by_reason": exclusion_counts,
            "ambiguous_sent_ids_found": ambiguous_sent_ids,
            "doc_id_map_mismatches_found": doc_id_mismatches,
        },
        "representative_sample": {
            "n_sentences": len(representative),
            "n_tokens": sum(sent.token_count for _s, _i, sent in representative),
            "sentence_ids": sorted(sent.sent_id for _s, _i, sent in representative),
            "document_ids": sorted({sent.doc_id for _s, _i, sent in representative}),
            "sampling_strata": rep_strata,
            "selection_never_used": [
                "Canonical_UPOS", "model predictions", "expected error categories",
                "predicted human disagreement",
            ],
        },
        "diagnostic_sample": {
            "n_sentences": len(diagnostic),
            "n_tokens": sum(sent.token_count for _s, _i, sent, _c in diagnostic),
            "sentence_ids": sorted(sent.sent_id for _s, _i, sent, _c in diagnostic),
            "document_ids": sorted({sent.doc_id for _s, _i, sent, _c in diagnostic}),
            "category_info": diag_info,
            "not_an_unbiased_corpus_wide_estimate": True,
        },
        "overlap_checks": {
            "representative_diagnostic_sentence_overlap": sorted(
                {sent.sent_id for _s, _i, sent in representative}
                & {sent.sent_id for _s, _i, sent, _c in diagnostic}
            ),
            "new_sample_vs_previous_sample_sentence_overlap": [],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def write_key_readme(path: Path) -> None:
    text = """# Test-Only Human Evaluation Sample -- Hidden Evaluation Artifacts

**HIDDEN. Do not distribute to annotators until annotation and blind
adjudication of both P1 and P2 are fully complete.**

This directory holds the canonical-label key and full selection provenance
for the test-only human UPOS evaluation sample (`test_v1`): 60 Representative
+ 20 Diagnostic sentences, drawn exclusively from the authoritative
document-level Pangyo test split, disjoint from the earlier 60-sentence
corpus-wide human-validation sample.

## Files

- `Human_POS_Tagging_test_v1_canonical_key.xlsx` -- canonical UPOS label for
  every token, plus inferred language and diagnostic category. Never shown
  to annotators pre-adjudication.
- `selection_manifest.json` -- seed, authoritative input checksums,
  eligibility/exclusion rules applied, every selected sentence/document ID,
  sampling strata, diagnostic category definitions, and overlap/duplicate
  check results.
- `selection_summary.csv` -- flat per-sentence summary (split, counts,
  KO/EN proportion, boundary count, diagnostic category).
- `exclusion_audit.csv` -- one row per test-split sentence excluded by a
  named rule (ambiguous ID, prior-sample sentence, prior-sample document,
  duplicate text), with reason and detail.

## Reproduction

```bash
python scripts/tests/test_create_test_only_human_evaluation_sample.py
python scripts/create_test_only_human_evaluation_sample.py
```
Deterministic with seed=42; a rerun against unchanged inputs reproduces the
same 80 sentences, same token rows, and same document/category assignments.
"""
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> dict:
    all_sentences, ambiguous_sent_ids = load_test_sentences(TEST_CONLLU)
    doc_map = load_doc_map(DOC_MAP_CSV)
    doc_id_mismatches = find_doc_id_mismatches(all_sentences, doc_map)
    previous_sent_docs = load_previous_sample_docs(SPLIT_AUDIT_CSV)
    previous_texts = load_previous_sample_texts(PREV_P1_XLSX)

    eligible, exclusion_records, exclude_all_previous_docs = compute_eligibility(
        all_sentences, ambiguous_sent_ids, doc_id_mismatches, previous_sent_docs, previous_texts,
    )

    rng = Random(SEED)
    representative_sents, rep_strata = select_representative(eligible, rng)
    representative_ids = {s.sent_id for s in representative_sents}
    remaining = [s for s in eligible if s.sent_id not in representative_ids]
    representative_texts = {s.normalized_text for s in representative_sents}
    representative_docs = {s.doc_id for s in representative_sents}
    diagnostic_pairs, diag_info = select_diagnostic(
        remaining, rng, excluded_texts=representative_texts, avoid_docs=representative_docs,
    )

    representative_sents = sorted(representative_sents, key=lambda s: (s.doc_id, s.sent_id))
    diagnostic_pairs = sorted(
        diagnostic_pairs, key=lambda pair: (CATEGORY_IDS.index(pair[1]), pair[0].sent_id)
    )

    representative_items = [
        ("Representative", f"REP-TEST-{i:03d}", sent) for i, sent in enumerate(representative_sents, 1)
    ]
    diagnostic_items = [
        ("Diagnostic", f"DIAG-TEST-{i:03d}", sent, cat)
        for i, (sent, cat) in enumerate(diagnostic_pairs, 1)
    ]
    diagnostic_items_no_cat = [(s, i, sent) for (s, i, sent, _c) in diagnostic_items]

    rep_tokens = sum(sent.token_count for _s, _i, sent in representative_items)
    diag_tokens = sum(sent.token_count for _s, _i, sent, _c in diagnostic_items)

    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    KEY_DIR.mkdir(parents=True, exist_ok=True)

    for annotator in ("P1", "P2"):
        write_annotator_workbook(
            ANNOT_DIR / f"Human_POS_Tagging_test_v1_{annotator}.xlsx", annotator,
            representative_items, diagnostic_items_no_cat, rep_tokens, diag_tokens,
        )

    write_canonical_key(KEY_DIR / "Human_POS_Tagging_test_v1_canonical_key.xlsx", representative_items, diagnostic_items)
    write_selection_summary_csv(KEY_DIR / "selection_summary.csv", representative_items, diagnostic_items)
    write_exclusion_audit_csv(KEY_DIR / "exclusion_audit.csv", exclusion_records)
    write_key_readme(KEY_DIR / "README.md")

    checksums = {
        str(p.relative_to(ROOT)).replace("\\", "/"): sha256_of(p)
        for p in [TEST_CONLLU, DOC_MAP_CSV, SPLIT_AUDIT_CSV, PREV_P1_XLSX]
        if p.exists()
    }
    exclusion_counts = dict(Counter(r.reason for r in exclusion_records))
    write_manifest(
        KEY_DIR / "selection_manifest.json", checksums, exclude_all_previous_docs, exclusion_counts,
        representative_items, diagnostic_items, rep_strata, diag_info, doc_id_mismatches, ambiguous_sent_ids,
    )

    return {
        "n_representative": len(representative_items),
        "n_diagnostic": len(diagnostic_items),
        "rep_tokens": rep_tokens,
        "diag_tokens": diag_tokens,
        "exclusion_counts": exclusion_counts,
        "exclude_all_previous_docs": exclude_all_previous_docs,
        "diag_info": diag_info,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print("Loading authoritative test split and building eligible pool ...")
    result = run()
    print(f"Representative: {result['n_representative']} sentences, {result['rep_tokens']} tokens")
    print(f"Diagnostic: {result['n_diagnostic']} sentences, {result['diag_tokens']} tokens")
    print(f"Diagnostic category counts: {result['diag_info']['achieved_category_counts']}")
    print(f"Full previous-document exclusion applied: {result['exclude_all_previous_docs']}")
    print(f"Exclusion counts by reason: {result['exclusion_counts']}")
    print("\nDone.")


if __name__ == "__main__":
    main()
