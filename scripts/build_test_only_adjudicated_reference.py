"""
build_test_only_adjudicated_reference.py - Deterministically reconstruct the
final test-only adjudicated human UPOS reference (1,879 tokens: 1,063
Representative + 816 Diagnostic) from source annotation artifacts, and
compare it against the canonical silver labels.

Final-label logic per token:
  - P1 and P2 agree, and the token was not later changed by the
    post-adjudication consistency audit -> the agreed label.
  - P1 and P2 disagree (one of the 69 Adjudication_69 rows) -> Adjudicated_UPOS.
  - A token on which P1 and P2 originally agreed, but which the documented
    post-adjudication audit later corrected -> the post-audit Final_UPOS.

Authoritative label sources (this script never reads Human_UPOS-equivalent
values from anywhere else):
  1. Completed P1 workbook
  2. Completed P2 workbook
  3. Final blind adjudication workbook (Adjudication_69 sheet)
  4. Post-adjudication consistency-audit sheet (same workbook)
  5. Canonical key + authoritative test CoNLL-U -- used ONLY for token-ID/
     token-form alignment checks and canonical-silver comparison, never as a
     source of the human final label.

outputs/human_validation/test_only/results/Human_POS_Tagging_test_v1_Final_Adjudicated_Results.xlsx
was provided as an independent audit target. This script reads a snapshot of
it taken BEFORE this script's first run (see scripts/_phase2_capture_audit_target.py)
purely to report agreement, then writes its own reconstruction to that same
path -- it never reads Final_Human_UPOS out of that file to build the
reference itself.

Run from repo root:
  python scripts/build_test_only_adjudicated_reference.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill

sys.path.insert(0, str(Path(__file__).parent))
from read_conllu import F_FORM, F_LEMMA, F_UPOS, F_MISC, iter_sentences  # noqa: E402

ROOT = Path(__file__).parent.parent

ANNOT_COMPLETED_DIR = ROOT / "outputs/human_validation/test_only/annotations/completed"
P1_XLSX = ANNOT_COMPLETED_DIR / "Human_POS_Tagging_test_v1_P1_COMPLETED.xlsx"
P2_XLSX = ANNOT_COMPLETED_DIR / "Human_POS_Tagging_test_v1_P2_COMPLETED.xlsx"
ADJUDICATION_XLSX = ROOT / "outputs/human_validation/test_only/annotations/Human_POS_Tagging_test_v1_Blind_Adjudication_FINAL_v1.2.xlsx"
CANONICAL_KEY_XLSX = ROOT / "outputs/human_validation/test_only/key/Human_POS_Tagging_test_v1_canonical_key.xlsx"
TEST_CONLLU = ROOT / "data/processed/panyo_original_adp_docsplit_test.conllu"
AUDIT_TARGET_XLSX = ROOT / "outputs/human_validation/test_only/results/Human_POS_Tagging_test_v1_Final_Adjudicated_Results.xlsx"
AUDIT_TARGET_SNAPSHOT = ROOT / "outputs/human_validation/test_only/results/_audit_target_snapshot.json"

OUT_DIR = ROOT / "outputs/human_validation/test_only/results"

SEED = 42
N_BOOTSTRAP = 10_000
UPOS17 = {
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART",
    "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
}

N_REPRESENTATIVE_SENT = 60
N_DIAGNOSTIC_SENT = 20
N_REPRESENTATIVE_TOK = 1063
N_DIAGNOSTIC_TOK = 816
N_COMBINED_TOK = N_REPRESENTATIVE_TOK + N_DIAGNOSTIC_TOK

# Expected audit values from the task specification -- asserted with tolerance,
# never hard-coded into the outputs themselves. Any mismatch is reported, not
# silently overwritten.
EXPECTED = {
    "pre_adjudication": {
        "Representative": {"n_tokens": 1063, "disagreements": 26, "agreement": 0.9755409219, "kappa": 0.9712427985},
        "Diagnostic": {"n_tokens": 816, "disagreements": 43, "agreement": 0.9473039216, "kappa": 0.9378833826},
        "Combined": {"n_tokens": 1879, "disagreements": 69, "agreement": 0.9632783395, "kappa": 0.9568569185},
    },
    "adjudication_summary": {
        "p1_selected": 44, "p2_selected": 25, "third_label": 0,
        "post_audit_correction": 1, "final_tokenization_issue_yes_total": 22,
    },
    "human_canonical": {
        "Representative": {
            "correct": 655, "n_tokens": 1063, "agreement": 0.6161806209,
            "non_punct_correct": 516, "non_punct_tokens": 924, "non_punct_agreement": 0.5584415584,
            "macro_f1": 0.2883470926, "weighted_f1": 0.5470726965, "kappa": 0.5143129108,
        },
        "Diagnostic": {
            "correct": 556, "n_tokens": 816, "agreement": 0.6813725490,
            "non_punct_correct": 419, "non_punct_tokens": 679, "non_punct_agreement": 0.6170839470,
        },
    },
}
TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RawToken:
    set_name: str
    sample_id: str
    source_sent_id: str
    document_id: str
    sentence: str
    token_id: int
    token: str
    upos: str
    confidence: str
    tokenization_issue: str
    note: Optional[str]


@dataclass(frozen=True)
class AdjudicationDecision:
    sample_id: str
    token_id: int
    final_upos: str
    final_tokenization_issue: str
    reason: str
    decision_status: str


@dataclass
class FinalToken:
    set_name: str
    sample_id: str
    source_sent_id: str
    document_id: str
    sentence: str
    token_id: int
    token: str
    p1_upos: str
    p1_confidence: str
    p1_tokenization_issue: str
    p1_note: Optional[str]
    p2_upos: str
    p2_confidence: str
    p2_tokenization_issue: str
    p2_note: Optional[str]
    final_upos: str
    final_tokenization_issue: str
    decision_source: str
    decision_reason: str
    canonical_upos: str
    canonical_lemma: str


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit_hash() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


def load_p_workbook(path: Path) -> dict[tuple[str, int], RawToken]:
    wb = openpyxl.load_workbook(path, data_only=True)
    out: dict[tuple[str, int], RawToken] = {}
    for sheet in ("Representative_60", "Diagnostic_20"):
        ws = wb[sheet]
        header = [c.value for c in ws[1]]
        idx = {name: i for i, name in enumerate(header)}
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[idx["Sample_ID"]] is None:
                continue
            key = (row[idx["Sample_ID"]], int(row[idx["Token_ID"]]))
            if key in out:
                raise ValueError(f"{path}: duplicate Sample_ID+Token_ID {key}")
            upos = row[idx["Human_UPOS"]]
            if not upos:
                raise ValueError(f"{path}: missing Human_UPOS at {key}")
            if upos not in UPOS17:
                raise ValueError(f"{path}: invalid Human_UPOS {upos!r} at {key}")
            tok_issue = row[idx["Tokenization_Issue"]]
            if tok_issue not in ("Yes", "No"):
                raise ValueError(f"{path}: invalid Tokenization_Issue {tok_issue!r} at {key}")
            out[key] = RawToken(
                set_name=row[idx["Set"]],
                sample_id=row[idx["Sample_ID"]],
                source_sent_id=row[idx["Source_Sent_ID"]],
                document_id=row[idx["Document_ID"]],
                sentence=row[idx["Sentence"]],
                token_id=int(row[idx["Token_ID"]]),
                token=row[idx["Token"]],
                upos=upos,
                confidence=row[idx["Confidence"]],
                tokenization_issue=tok_issue,
                note=row[idx["Note"]],
            )
    return out


def _load_decision_sheet(
    path: Path, sheet: str, header_row: int, upos_col: str, audit_no_gate: bool
) -> dict[tuple[str, int], AdjudicationDecision]:
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet]
    header = [c.value for c in ws[header_row]]
    idx = {name: i for i, name in enumerate(header)}
    out: dict[tuple[str, int], AdjudicationDecision] = {}
    for row in ws.iter_rows(min_row=header_row + 1, values_only=True):
        if audit_no_gate:
            if not isinstance(row[idx["Audit_No"]], int):
                continue
        else:
            if row[idx["Sample_ID"]] is None:
                continue
        key = (row[idx["Sample_ID"]], int(row[idx["Token_ID"]]))
        upos = row[idx[upos_col]]
        if upos not in UPOS17:
            raise ValueError(f"{path}:{sheet}: invalid {upos_col} {upos!r} at {key}")
        status = row[idx["Decision_Status"]]
        if status != "Resolved":
            raise ValueError(f"{path}:{sheet}: unresolved decision at {key} (status={status!r})")
        reason = row[idx["Adjudication_Reason"]]
        if not reason:
            raise ValueError(f"{path}:{sheet}: empty adjudication reason at {key}")
        out[key] = AdjudicationDecision(
            sample_id=row[idx["Sample_ID"]],
            token_id=int(row[idx["Token_ID"]]),
            final_upos=upos,
            final_tokenization_issue=row[idx["Final_Tokenization_Issue"]],
            reason=reason,
            decision_status=status,
        )
    return out


def load_adjudication_decisions(path: Path) -> dict[tuple[str, int], AdjudicationDecision]:
    out = _load_decision_sheet(path, "Adjudication_69", 3, "Adjudicated_UPOS", audit_no_gate=False)
    if len(out) != 69:
        raise ValueError(f"{path}: expected 69 adjudication decisions, found {len(out)}")
    return out


def load_post_audit_corrections(path: Path) -> dict[tuple[str, int], AdjudicationDecision]:
    return _load_decision_sheet(path, "Post_Adjudication_Audit", 3, "Final_UPOS", audit_no_gate=True)


def load_canonical_key(path: Path) -> dict[tuple[str, int], dict]:
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb["Canonical_Key_80"]
    header = [c.value for c in ws[1]]
    idx = {name: i for i, name in enumerate(header)}
    out: dict[tuple[str, int], dict] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[idx["Sample_ID"]] is None:
            continue
        key = (row[idx["Sample_ID"]], int(row[idx["Token_ID"]]))
        out[key] = {
            "canonical_upos": row[idx["Canonical_UPOS"]],
            "token": row[idx["Token"]],
        }
    return out


def load_conllu_index(path: Path) -> dict[str, dict[int, tuple[str, str, str, str]]]:
    """sent_id -> {token_id(1-based real-token index): (form, lemma, upos, misc)}"""
    out: dict[str, dict[int, tuple[str, str, str, str]]] = {}
    for sent in iter_sentences(path):
        sid = None
        for c in sent["comments"]:
            if c.startswith("# sent_id = "):
                sid = c[len("# sent_id = "):]
        if sid is None:
            continue
        real = [t for t in sent["tokens"] if t["is_real"]]
        out[sid] = {
            i + 1: (t["fields"][F_FORM], t["fields"][F_LEMMA], t["fields"][F_UPOS], t["fields"][F_MISC])
            for i, t in enumerate(real)
        }
    return out


# ---------------------------------------------------------------------------
# Final-reference construction
# ---------------------------------------------------------------------------

def classify_decision_source(final_upos: str, p1_upos: str, p2_upos: str) -> str:
    if final_upos == p1_upos and final_upos == p2_upos:
        return "agreed"
    if final_upos == p1_upos:
        return "P1_selected"
    if final_upos == p2_upos:
        return "P2_selected"
    return "third_label"


def build_final_reference() -> tuple[list[FinalToken], dict]:
    p1 = load_p_workbook(P1_XLSX)
    p2 = load_p_workbook(P2_XLSX)
    adjudications = load_adjudication_decisions(ADJUDICATION_XLSX)
    post_audit = load_post_audit_corrections(ADJUDICATION_XLSX)
    canonical = load_canonical_key(CANONICAL_KEY_XLSX)
    conllu_index = load_conllu_index(TEST_CONLLU)

    if set(p1) != set(p2):
        missing_p2 = sorted(set(p1) - set(p2))
        missing_p1 = sorted(set(p2) - set(p1))
        raise ValueError(
            f"P1/P2 key mismatch: missing_in_p2={missing_p2[:5]} missing_in_p1={missing_p1[:5]}"
        )

    keys = sorted(p1, key=lambda k: (k[0], k[1]))
    if len(keys) != N_COMBINED_TOK:
        raise ValueError(f"Expected {N_COMBINED_TOK} tokens, found {len(keys)}")

    tokenization_discrepancies: list[tuple[str, int]] = []
    alignment_errors: list[tuple] = []
    tokens: list[FinalToken] = []

    for key in keys:
        r1, r2 = p1[key], p2[key]
        sample_id, token_id = key

        if r1.token != r2.token:
            alignment_errors.append((key, "P1/P2 token form mismatch", r1.token, r2.token))
        if r1.source_sent_id != r2.source_sent_id or r1.set_name != r2.set_name:
            alignment_errors.append((key, "P1/P2 structural field mismatch", r1.source_sent_id, r2.source_sent_id))

        if key in post_audit:
            dec = post_audit[key]
            if r1.upos != r2.upos:
                alignment_errors.append((key, "post-audit token was NOT an original P1/P2 agreement", r1.upos, r2.upos))
            final_upos = dec.final_upos
            final_tok = dec.final_tokenization_issue
            source = "post_audit_correction"
            reason = dec.reason
        elif key in adjudications:
            if r1.upos == r2.upos:
                alignment_errors.append((key, "adjudicated token was NOT actually a P1/P2 disagreement", r1.upos, r2.upos))
            dec = adjudications[key]
            final_upos = dec.final_upos
            final_tok = dec.final_tokenization_issue
            source = classify_decision_source(final_upos, r1.upos, r2.upos)
            reason = dec.reason
        else:
            if r1.upos != r2.upos:
                raise ValueError(
                    f"Data inconsistency: {key} has P1/P2 disagreement "
                    f"(P1={r1.upos!r}, P2={r2.upos!r}) but is absent from the "
                    f"69-row Adjudication_69 sheet."
                )
            final_upos = r1.upos
            source = "agreed"
            reason = "P1 and P2 independently agreed"
            if r1.tokenization_issue == r2.tokenization_issue:
                final_tok = r1.tokenization_issue
            else:
                tokenization_discrepancies.append(key)
                final_tok = "Yes" if "Yes" in (r1.tokenization_issue, r2.tokenization_issue) else "No"

        canon = canonical.get(key)
        if canon is None:
            alignment_errors.append((key, "missing from canonical key", None, None))
            canonical_upos = ""
        else:
            canonical_upos = canon["canonical_upos"]
            if canon["token"] != r1.token:
                alignment_errors.append((key, "canonical key token mismatch", canon["token"], r1.token))

        conllu_lemma = ""
        conllu_toks = conllu_index.get(r1.source_sent_id)
        if conllu_toks is None:
            alignment_errors.append((key, "sent_id not found in authoritative test CoNLL-U", r1.source_sent_id, None))
        else:
            match = conllu_toks.get(token_id)
            if match is None:
                alignment_errors.append((key, "token_id not found in CoNLL-U sentence", token_id, None))
            elif match[0] != r1.token:
                alignment_errors.append((key, "CoNLL-U token form mismatch", match[0], r1.token))
            else:
                conllu_lemma = match[1]
                if match[2] != canonical_upos:
                    alignment_errors.append((key, "canonical key vs CoNLL-U UPOS mismatch", canonical_upos, match[2]))

        tokens.append(FinalToken(
            set_name=r1.set_name, sample_id=sample_id, source_sent_id=r1.source_sent_id,
            document_id=r1.document_id, sentence=r1.sentence, token_id=token_id, token=r1.token,
            p1_upos=r1.upos, p1_confidence=r1.confidence, p1_tokenization_issue=r1.tokenization_issue, p1_note=r1.note,
            p2_upos=r2.upos, p2_confidence=r2.confidence, p2_tokenization_issue=r2.tokenization_issue, p2_note=r2.note,
            final_upos=final_upos, final_tokenization_issue=final_tok,
            decision_source=source, decision_reason=reason,
            canonical_upos=canonical_upos, canonical_lemma=conllu_lemma,
        ))

    diagnostics = {
        "tokenization_discrepancies": [f"{k[0]}#{k[1]}" for k in tokenization_discrepancies],
        "alignment_errors": [
            {"key": f"{k[0]}#{k[1]}", "issue": issue, "a": str(a), "b": str(b)}
            for k, issue, a, b in alignment_errors
        ],
    }
    if alignment_errors:
        details = "\n".join(f"  {k[0]}#{k[1]}: {issue} ({a!r} vs {b!r})" for k, issue, a, b in alignment_errors[:20])
        raise ValueError(f"{len(alignment_errors)} alignment error(s) found; refusing to proceed:\n{details}")

    return tokens, diagnostics


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cohens_kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    if n == 0:
        return 0.0
    po = sum(1 for a, b in pairs if a == b) / n
    row_marg = Counter(a for a, b in pairs)
    col_marg = Counter(b for a, b in pairs)
    labels = set(row_marg) | set(col_marg)
    pe = sum((row_marg[l] / n) * (col_marg[l] / n) for l in labels)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def compute_per_class(pairs: list[tuple[str, str]]) -> dict[str, dict]:
    """pairs = (gold, pred). Here gold=Final_Human_UPOS, pred=Canonical_UPOS
    (i.e. treats the human reference as ground truth and scores canonical
    silver against it, consistent with this project's stated purpose of
    assessing canonical-silver quality)."""
    tp, fp, fn, sup = Counter(), Counter(), Counter(), Counter()
    for gold, pred in pairs:
        sup[gold] += 1
        if gold == pred:
            tp[gold] += 1
        else:
            fn[gold] += 1
            fp[pred] += 1
    tags = sorted(set(sup) | set(fp))
    result = {}
    for tag in tags:
        prec = tp[tag] / (tp[tag] + fp[tag]) if (tp[tag] + fp[tag]) else 0.0
        rec = tp[tag] / (tp[tag] + fn[tag]) if (tp[tag] + fn[tag]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        result[tag] = {"precision": prec, "recall": rec, "f1": f1, "support": sup[tag]}
    return result


def macro_f1(per_class: dict[str, dict]) -> float:
    vals = [v["f1"] for v in per_class.values() if v["support"] > 0]
    return sum(vals) / len(vals) if vals else 0.0


def weighted_f1(per_class: dict[str, dict]) -> float:
    total = sum(v["support"] for v in per_class.values())
    return sum(v["f1"] * v["support"] for v in per_class.values()) / total if total else 0.0


def sentence_bootstrap_ci(
    correct_by_sentence: list[int], total_by_sentence: list[int], seed: int = SEED, n_boot: int = N_BOOTSTRAP
) -> tuple[float, float, float]:
    n = len(correct_by_sentence)
    correct = np.array(correct_by_sentence, dtype=np.int64)
    total = np.array(total_by_sentence, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_correct = correct[idx].sum(axis=1)
    boot_total = total[idx].sum(axis=1)
    accs = boot_correct / boot_total
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return float(lo), float(hi), float(accs.mean())


def compute_pre_adjudication_metrics(tokens: list[FinalToken]) -> dict:
    pairs = [(t.p1_upos, t.p2_upos) for t in tokens]
    n = len(pairs)
    disagreements = sum(1 for a, b in pairs if a != b)
    return {
        "n_tokens": n,
        "disagreements": disagreements,
        "agreement": 1 - disagreements / n if n else 0.0,
        "kappa": cohens_kappa(pairs),
    }


def compute_human_canonical_metrics(tokens: list[FinalToken]) -> dict:
    n = len(tokens)
    correct = sum(1 for t in tokens if t.final_upos == t.canonical_upos)
    non_punct = [t for t in tokens if t.final_upos != "PUNCT"]
    non_punct_correct = sum(1 for t in non_punct if t.final_upos == t.canonical_upos)
    pairs = [(t.final_upos, t.canonical_upos) for t in tokens]
    per_class = compute_per_class(pairs)

    by_sentence: dict[str, list[bool]] = defaultdict(list)
    for t in tokens:
        by_sentence[t.sample_id].append(t.final_upos == t.canonical_upos)
    sample_ids = sorted(by_sentence)
    correct_by_sent = [sum(by_sentence[s]) for s in sample_ids]
    total_by_sent = [len(by_sentence[s]) for s in sample_ids]
    lo, hi, boot_mean = sentence_bootstrap_ci(correct_by_sent, total_by_sent)

    confusions = Counter((t.final_upos, t.canonical_upos) for t in tokens if t.final_upos != t.canonical_upos)

    tok_issue_yes = [t for t in tokens if t.final_tokenization_issue == "Yes"]
    tok_issue_no = [t for t in tokens if t.final_tokenization_issue == "No"]

    def strat_acc(subset: list[FinalToken]) -> tuple[int, int, float]:
        c = sum(1 for t in subset if t.final_upos == t.canonical_upos)
        n_sub = len(subset)
        return c, n_sub, (c / n_sub if n_sub else 0.0)

    return {
        "n_sentences": len(sample_ids),
        "n_tokens": n,
        "correct": correct,
        "agreement": correct / n if n else 0.0,
        "non_punct_tokens": len(non_punct),
        "non_punct_correct": non_punct_correct,
        "non_punct_agreement": non_punct_correct / len(non_punct) if non_punct else 0.0,
        "macro_f1": macro_f1(per_class),
        "weighted_f1": weighted_f1(per_class),
        "kappa": cohens_kappa(pairs),
        "per_class": per_class,
        "bootstrap_mean": boot_mean,
        "bootstrap_ci_low": lo,
        "bootstrap_ci_high": hi,
        "confusions": confusions,
        "tokenization_issue_yes": strat_acc(tok_issue_yes),
        "tokenization_issue_no": strat_acc(tok_issue_no),
    }


def compute_adjudication_summary(tokens: list[FinalToken]) -> dict:
    by_source = Counter(t.decision_source for t in tokens)
    total_tok_yes = sum(1 for t in tokens if t.final_tokenization_issue == "Yes")
    return {
        "p1_selected": by_source.get("P1_selected", 0),
        "p2_selected": by_source.get("P2_selected", 0),
        "third_label": by_source.get("third_label", 0),
        "post_audit_correction": by_source.get("post_audit_correction", 0),
        "agreed": by_source.get("agreed", 0),
        "final_tokenization_issue_yes_total": total_tok_yes,
    }


def compare_against_audit_target(tokens: list[FinalToken]) -> dict:
    if not AUDIT_TARGET_SNAPSHOT.exists():
        return {"available": False, "reason": f"snapshot not found at {AUDIT_TARGET_SNAPSHOT}"}
    snapshot = json.loads(AUDIT_TARGET_SNAPSHOT.read_text(encoding="utf-8"))
    audit_rows = {(r["Sample_ID"], int(r["Token_ID"])): r for r in snapshot["rows"]}
    my_by_key = {(t.sample_id, t.token_id): t for t in tokens}

    missing_in_audit = sorted(set(my_by_key) - set(audit_rows))
    missing_in_mine = sorted(set(audit_rows) - set(my_by_key))

    upos_mismatches = []
    tok_mismatches = []
    for key, t in my_by_key.items():
        a = audit_rows.get(key)
        if a is None:
            continue
        if a["Final_Human_UPOS"] != t.final_upos:
            upos_mismatches.append({
                "key": f"{key[0]}#{key[1]}", "token": t.token,
                "mine": t.final_upos, "audit_target": a["Final_Human_UPOS"],
            })
        if a["Final_Tokenization_Issue"] != t.final_tokenization_issue:
            tok_mismatches.append({
                "key": f"{key[0]}#{key[1]}", "token": t.token,
                "mine": t.final_tokenization_issue, "audit_target": a["Final_Tokenization_Issue"],
            })

    n = len(my_by_key)
    n_matched_upos = n - len(upos_mismatches)
    return {
        "available": True,
        "n_tokens_compared": n,
        "missing_in_audit_target": [f"{k[0]}#{k[1]}" for k in missing_in_audit],
        "missing_in_reconstruction": [f"{k[0]}#{k[1]}" for k in missing_in_mine],
        "final_upos_matches": n_matched_upos,
        "final_upos_agreement_rate": n_matched_upos / n if n else 0.0,
        "final_upos_mismatches": upos_mismatches,
        "final_tokenization_issue_mismatches": tok_mismatches,
    }


def check_expected(actual: dict, expected: dict, path: str, report: list[str]) -> None:
    for k, exp_v in expected.items():
        if isinstance(exp_v, dict):
            check_expected(actual.get(k, {}), exp_v, f"{path}.{k}", report)
            continue
        act_v = actual.get(k)
        if isinstance(exp_v, float):
            ok = act_v is not None and abs(act_v - exp_v) <= max(TOLERANCE, abs(exp_v) * 1e-4)
        else:
            ok = act_v == exp_v
        if not ok:
            report.append(f"MISMATCH {path}.{k}: expected {exp_v!r}, got {act_v!r}")


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

TOKEN_CSV_FIELDS = [
    "Set", "Sample_ID", "Source_Sent_ID", "Document_ID", "Sentence", "Token_ID", "Token",
    "P1_UPOS", "P1_Confidence", "P1_Tokenization_Issue", "P1_Note",
    "P2_UPOS", "P2_Confidence", "P2_Tokenization_Issue", "P2_Note",
    "Final_Human_UPOS", "Final_Tokenization_Issue", "Decision_Source", "Decision_Reason",
    "Canonical_UPOS", "Canonical_Lemma",
]


def token_to_row(t: FinalToken) -> dict:
    return {
        "Set": t.set_name, "Sample_ID": t.sample_id, "Source_Sent_ID": t.source_sent_id,
        "Document_ID": t.document_id, "Sentence": t.sentence, "Token_ID": t.token_id, "Token": t.token,
        "P1_UPOS": t.p1_upos, "P1_Confidence": t.p1_confidence,
        "P1_Tokenization_Issue": t.p1_tokenization_issue, "P1_Note": t.p1_note,
        "P2_UPOS": t.p2_upos, "P2_Confidence": t.p2_confidence,
        "P2_Tokenization_Issue": t.p2_tokenization_issue, "P2_Note": t.p2_note,
        "Final_Human_UPOS": t.final_upos, "Final_Tokenization_Issue": t.final_tokenization_issue,
        "Decision_Source": t.decision_source, "Decision_Reason": t.decision_reason,
        "Canonical_UPOS": t.canonical_upos, "Canonical_Lemma": t.canonical_lemma,
    }


def write_tokens_csv(tokens: list[FinalToken], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=TOKEN_CSV_FIELDS)
        w.writeheader()
        for t in tokens:
            w.writerow(token_to_row(t))


def write_sentences_jsonl(tokens: list[FinalToken], path: Path) -> None:
    by_sentence: dict[str, list[FinalToken]] = defaultdict(list)
    for t in tokens:
        by_sentence[t.sample_id].append(t)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for sample_id in sorted(by_sentence):
            toks = sorted(by_sentence[sample_id], key=lambda t: t.token_id)
            rec = {
                "set": toks[0].set_name,
                "sample_id": sample_id,
                "source_sent_id": toks[0].source_sent_id,
                "document_id": toks[0].document_id,
                "sentence": toks[0].sentence,
                "tokens": [
                    {
                        "token_id": t.token_id, "token": t.token, "final_upos": t.final_upos,
                        "final_tokenization_issue": t.final_tokenization_issue,
                        "decision_source": t.decision_source, "canonical_upos": t.canonical_upos,
                    }
                    for t in toks
                ],
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_final_conllu(tokens: list[FinalToken], path: Path) -> None:
    """Preserve original sentence IDs, document IDs, token IDs, forms, lemmas,
    and non-UPOS fields from the authoritative test CoNLL-U; replace only the
    UPOS field with Final_Human_UPOS. Marked in comments as an adjudicated
    human-reference subset (not a replacement for the full canonical corpus),
    with Representative/Diagnostic membership recorded in MISC."""
    conllu_index_full: dict[str, list[dict]] = defaultdict(list)
    for sent in iter_sentences(TEST_CONLLU):
        sid = None
        for c in sent["comments"]:
            if c.startswith("# sent_id = "):
                sid = c[len("# sent_id = "):]
        if sid is not None:
            conllu_index_full[sid] = sent["tokens"]

    by_sentence: dict[str, list[FinalToken]] = defaultdict(list)
    for t in tokens:
        by_sentence[t.sample_id].append(t)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for sample_id in sorted(by_sentence):
            toks = sorted(by_sentence[sample_id], key=lambda t: t.token_id)
            sent_id = toks[0].source_sent_id
            src_real_tokens = [t for t in conllu_index_full[sent_id] if t["is_real"]]
            by_tid = {i + 1: t for i, t in enumerate(src_real_tokens)}
            fh.write(f"# adjudicated_human_reference_subset = true (test-only round test_v1; "
                     f"NOT a replacement for the full canonical corpus)\n")
            fh.write(f"# set = {toks[0].set_name}\n")
            fh.write(f"# sample_id = {sample_id}\n")
            fh.write(f"# sent_id = {sent_id}\n")
            fh.write(f"# doc_id = {toks[0].document_id}\n")
            fh.write(f"# text = {toks[0].sentence}\n")
            for t in toks:
                src = by_tid[t.token_id]
                fields = list(src["fields"])
                fields[F_UPOS] = t.final_upos
                misc = fields[F_MISC]
                extra = f"Set={t.set_name}|FinalTokenizationIssue={t.final_tokenization_issue}|DecisionSource={t.decision_source}"
                fields[F_MISC] = extra if misc == "_" else f"{misc}|{extra}"
                fh.write("\t".join(fields) + "\n")
            fh.write("\n")


def _json_safe_human_canon(human_canon: dict) -> dict:
    out = {}
    for set_name, m in human_canon.items():
        m2 = dict(m)
        m2["confusions"] = [
            {"final_human_upos": h, "canonical_upos": c, "count": n}
            for (h, c), n in sorted(m["confusions"].items(), key=lambda kv: -kv[1])
        ]
        out[set_name] = m2
    return out


def write_metrics_json(payload: dict, path: Path) -> None:
    payload = dict(payload)
    if "human_vs_canonical" in payload:
        payload["human_vs_canonical"] = _json_safe_human_canon(payload["human_vs_canonical"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_metrics_txt(
    pre_adj: dict, adj_summary: dict, human_canon: dict, audit_cmp: dict, discrepancies: list[str], path: Path
) -> None:
    lines = []
    lines.append("Test-only Final Adjudicated Human Reference -- Metrics Summary")
    lines.append("=" * 88)
    lines.append("")
    lines.append("## Pre-adjudication human-human agreement")
    for set_name in ("Representative", "Diagnostic", "Combined"):
        m = pre_adj[set_name]
        lines.append(
            f"  {set_name}: n={m['n_tokens']} disagreements={m['disagreements']} "
            f"agreement={m['agreement']:.10f} kappa={m['kappa']:.10f}"
        )
    lines.append("")
    lines.append("## Adjudication summary")
    for k, v in adj_summary.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("## Human vs canonical silver")
    for set_name in ("Representative", "Diagnostic", "Combined"):
        m = human_canon[set_name]
        lines.append(f"  [{set_name}] (primary)" if set_name == "Representative" else
                     f"  [{set_name}] (targeted, not unbiased)" if set_name == "Diagnostic" else
                     f"  [{set_name}] (descriptive only)")
        lines.append(
            f"    tokens={m['n_tokens']} correct={m['correct']} agreement={m['agreement']:.10f}"
        )
        lines.append(
            f"    non_punct_tokens={m['non_punct_tokens']} non_punct_correct={m['non_punct_correct']} "
            f"non_punct_agreement={m['non_punct_agreement']:.10f}"
        )
        lines.append(
            f"    macro_f1={m['macro_f1']:.10f} weighted_f1={m['weighted_f1']:.10f} kappa={m['kappa']:.10f}"
        )
        lines.append(
            f"    sentence_bootstrap_95%_CI=[{m['bootstrap_ci_low']:.6f}, {m['bootstrap_ci_high']:.6f}] "
            f"(mean={m['bootstrap_mean']:.6f}, seed={SEED}, n_boot={N_BOOTSTRAP})"
        )
        lines.append(
            f"    tokenization_issue=Yes: {m['tokenization_issue_yes'][0]}/{m['tokenization_issue_yes'][1]} "
            f"({m['tokenization_issue_yes'][2]:.4f})  "
            f"tokenization_issue=No: {m['tokenization_issue_no'][0]}/{m['tokenization_issue_no'][1]} "
            f"({m['tokenization_issue_no'][2]:.4f})"
        )
    lines.append("")
    lines.append("## Agreement with independent audit-target workbook (pre-existing "
                  "Human_POS_Tagging_test_v1_Final_Adjudicated_Results.xlsx, snapshotted "
                  "before this script overwrote it)")
    if audit_cmp.get("available"):
        lines.append(
            f"  {audit_cmp['final_upos_matches']}/{audit_cmp['n_tokens_compared']} tokens match "
            f"({audit_cmp['final_upos_agreement_rate']:.10f})"
        )
        if audit_cmp["final_upos_mismatches"]:
            lines.append(f"  {len(audit_cmp['final_upos_mismatches'])} Final_UPOS mismatch(es):")
            for m in audit_cmp["final_upos_mismatches"][:20]:
                lines.append(f"    {m['key']} ({m['token']!r}): mine={m['mine']} audit_target={m['audit_target']}")
    else:
        lines.append(f"  NOT AVAILABLE: {audit_cmp.get('reason')}")
    lines.append("")
    if discrepancies:
        lines.append("## DISCREPANCIES vs. expected audit values (investigated, not overwritten)")
        for d in discrepancies:
            lines.append(f"  {d}")
    else:
        lines.append("## All values matched the expected audit values within tolerance.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_per_upos_csv(per_class: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["UPOS", "precision", "recall", "f1", "support"])
        for tag in sorted(per_class):
            v = per_class[tag]
            w.writerow([tag, f"{v['precision']:.6f}", f"{v['recall']:.6f}", f"{v['f1']:.6f}", v["support"]])


def write_confusions_csv(confusions: Counter, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Final_Human_UPOS", "Canonical_UPOS", "count"])
        for (h, c), n in sorted(confusions.items(), key=lambda kv: -kv[1]):
            w.writerow([h, c, n])


def write_label_distribution_csv(tokens: list[FinalToken], path: Path) -> None:
    human = Counter(t.final_upos for t in tokens)
    canon = Counter(t.canonical_upos for t in tokens)
    tags = sorted(set(human) | set(canon))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["UPOS", "final_human_count", "canonical_count"])
        for tag in tags:
            w.writerow([tag, human.get(tag, 0), canon.get(tag, 0)])


def write_adjudication_decision_summary_csv(tokens: list[FinalToken], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    decided = [t for t in tokens if t.decision_source != "agreed"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "Set", "Sample_ID", "Token_ID", "Token", "P1_UPOS", "P2_UPOS", "Final_Human_UPOS",
            "Decision_Source", "Final_Tokenization_Issue", "Reason",
        ])
        for t in decided:
            w.writerow([
                t.set_name, t.sample_id, t.token_id, t.token, t.p1_upos, t.p2_upos, t.final_upos,
                t.decision_source, t.final_tokenization_issue, t.decision_reason,
            ])


def _write_sheet_rows(ws, rows: list[list], bold_header_rows: set[int] = frozenset()) -> None:
    for r_idx, row in enumerate(rows, start=1):
        for c_idx, val in enumerate(row, start=1):
            cell = ws.cell(row=r_idx, column=c_idx, value=val)
            if r_idx in bold_header_rows:
                cell.font = Font(bold=True)


def write_results_workbook(
    tokens: list[FinalToken], pre_adj: dict, human_canon: dict, path: Path
) -> None:
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    ws = wb.create_sheet("Summary")
    rows = [
        ["Test-only Final Adjudicated Human Reference -- Canonical Silver Comparison"],
        [],
        ["Set", "Sentences", "Tokens", "Correct", "Agreement", "Non-PUNCT Tokens",
         "Non-PUNCT Correct", "Non-PUNCT Agreement", "Macro F1", "Weighted F1",
         "Cohen's kappa vs Canonical", "Bootstrap Mean", "Sentence-bootstrap 95% CI"],
    ]
    for set_name in ("Representative", "Diagnostic", "Combined"):
        m = human_canon[set_name]
        rows.append([
            set_name, m["n_sentences"], m["n_tokens"], m["correct"], m["agreement"],
            m["non_punct_tokens"], m["non_punct_correct"], m["non_punct_agreement"],
            m["macro_f1"], m["weighted_f1"], m["kappa"], m["bootstrap_mean"],
            f"{m['bootstrap_ci_low']:.4f}-{m['bootstrap_ci_high']:.4f}",
        ])
    _write_sheet_rows(ws, rows, bold_header_rows={1, 3})

    for set_name, sheet_name in (("Representative", "Representative_UPOS"), ("Diagnostic", "Diagnostic_UPOS")):
        ws = wb.create_sheet(sheet_name)
        rows = [[f"{set_name} -- Final Human vs Canonical Per-UPOS Metrics"], [],
                ["UPOS", "precision", "recall", "f1", "support"]]
        for tag in sorted(human_canon[set_name]["per_class"]):
            v = human_canon[set_name]["per_class"][tag]
            rows.append([tag, v["precision"], v["recall"], v["f1"], v["support"]])
        _write_sheet_rows(ws, rows, bold_header_rows={1, 3})

    for set_name, sheet_name in (("Representative", "Representative_Confusions"), ("Diagnostic", "Diagnostic_Confusions")):
        ws = wb.create_sheet(sheet_name)
        rows = [[f"{set_name} -- Final Human -> Canonical Confusions"], [],
                ["Final_Human_UPOS", "Canonical_UPOS", "Count"]]
        for (h, c), n in sorted(human_canon[set_name]["confusions"].items(), key=lambda kv: -kv[1]):
            rows.append([h, c, n])
        _write_sheet_rows(ws, rows, bold_header_rows={1, 3})

    ws = wb.create_sheet("Label_Distributions")
    human_all = Counter(t.final_upos for t in tokens)
    canon_all = Counter(t.canonical_upos for t in tokens)
    tags = sorted(set(human_all) | set(canon_all))
    rows = [["Final Human and Canonical Label Distributions"], [], ["UPOS", "Final_Human_Count", "Canonical_Count"]]
    for tag in tags:
        rows.append([tag, human_all.get(tag, 0), canon_all.get(tag, 0)])
    _write_sheet_rows(ws, rows, bold_header_rows={1, 3})

    ws = wb.create_sheet(f"Final_Reference_{len(tokens)}")
    rows = [[f"Final Test-only Adjudicated Human Reference -- {len(tokens)} Tokens"], [], TOKEN_CSV_FIELDS]
    for t in tokens:
        rows.append([token_to_row(t)[f] for f in TOKEN_CSV_FIELDS])
    _write_sheet_rows(ws, rows, bold_header_rows={1, 3})

    ws = wb.create_sheet("Adjudicated_69")
    decided_69 = [t for t in tokens if t.decision_source in ("P1_selected", "P2_selected", "third_label")]
    rows = [["Final Blind Decisions for the 69 Initial Disagreements"], [],
            ["Set", "Sample_ID", "Source_Sent_ID", "Document_ID", "Sentence", "Token_ID", "Token",
             "P1_UPOS", "P2_UPOS", "Final_UPOS", "Decision_Source", "Final_Tokenization_Issue", "Reason"]]
    for t in decided_69:
        rows.append([t.set_name, t.sample_id, t.source_sent_id, t.document_id, t.sentence, t.token_id,
                     t.token, t.p1_upos, t.p2_upos, t.final_upos, t.decision_source,
                     t.final_tokenization_issue, t.decision_reason])
    _write_sheet_rows(ws, rows, bold_header_rows={1, 3})

    ws = wb.create_sheet("Post_Adjudication_Audit")
    post = [t for t in tokens if t.decision_source == "post_audit_correction"]
    rows = [["Post-Adjudication Full-Expression Consistency Audit"], [],
            ["Set", "Sample_ID", "Source_Sent_ID", "Document_ID", "Sentence", "Token_ID", "Token",
             "P1_UPOS", "P2_UPOS", "Final_UPOS", "Final_Tokenization_Issue", "Reason"]]
    for t in post:
        rows.append([t.set_name, t.sample_id, t.source_sent_id, t.document_id, t.sentence, t.token_id,
                     t.token, t.p1_upos, t.p2_upos, t.final_upos, t.final_tokenization_issue, t.decision_reason])
    _write_sheet_rows(ws, rows, bold_header_rows={1, 3})

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def write_manifest(
    pre_adj: dict, adj_summary: dict, human_canon: dict, audit_cmp: dict,
    discrepancies: list[str], diagnostics: dict, path: Path
) -> None:
    inputs = [P1_XLSX, P2_XLSX, ADJUDICATION_XLSX, CANONICAL_KEY_XLSX, TEST_CONLLU]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/build_test_only_adjudicated_reference.py",
        "git_commit": git_commit_hash(),
        "python_version": platform.python_version(),
        "seed": SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "authoritative_inputs_sha256": {str(p.relative_to(ROOT)): sha256_of(p) for p in inputs if p.exists()},
        "audit_target_used_as_source_of_labels": False,
        "pre_adjudication_human_human": pre_adj,
        "adjudication_summary": adj_summary,
        "human_vs_canonical": {
            k: {kk: vv for kk, vv in v.items() if kk not in ("per_class", "confusions")}
            for k, v in human_canon.items()
        },
        "audit_target_comparison": {
            k: v for k, v in audit_cmp.items()
            if k not in ("final_upos_mismatches", "final_tokenization_issue_mismatches")
        },
        "audit_target_comparison_mismatch_examples": {
            "final_upos_mismatches": audit_cmp.get("final_upos_mismatches", [])[:20],
            "final_tokenization_issue_mismatches": audit_cmp.get("final_tokenization_issue_mismatches", [])[:20],
        },
        "expected_value_discrepancies": discrepancies,
        "build_diagnostics": diagnostics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_command_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat()} python scripts/build_test_only_adjudicated_reference.py\n")


def write_results_readme(path: Path) -> None:
    text = """# Test-only Final Adjudicated Human Reference -- Results

Deterministically reconstructed by `scripts/build_test_only_adjudicated_reference.py`
from the completed P1/P2 annotation workbooks, the final blind adjudication
workbook (69 disagreements), and the post-adjudication consistency-audit
sheet (1 correction). The canonical key and authoritative test CoNLL-U are
used only for alignment and canonical-silver comparison, never as a source
of the human final label.

## Files

- `final_human_reference_tokens.csv` -- all 1,879 tokens, one row per token,
  with P1/P2 evidence preserved alongside the final label.
- `final_human_reference_sentences.jsonl` -- one JSON object per sentence.
- `final_human_reference.conllu` -- CoNLL-U subset with UPOS replaced by the
  final human label; all other fields preserved from the authoritative test
  split. Not a replacement for the full canonical corpus.
- `Human_POS_Tagging_test_v1_Final_Adjudicated_Results.xlsx` -- this
  script's own reconstruction, replacing the pre-existing file that was
  supplied only as an independent audit target (see
  `human_silver_metrics.txt` for the token-level agreement between this
  reconstruction and that prior file).
- `human_silver_metrics.{json,txt}`, `representative_per_upos_metrics.csv`,
  `diagnostic_per_upos_metrics.csv`, `representative_human_silver_confusions.csv`,
  `diagnostic_human_silver_confusions.csv`, `final_label_distribution.csv`,
  `adjudication_decision_summary.csv` -- metrics and supporting tables.
- `reference_build_manifest.json` -- seed, input checksums, full metrics,
  discrepancy report against the task's expected audit values, and the
  comparison against the independent audit-target snapshot.
- `reference_build_command_log.txt` -- append-only run log.

## Reporting language

Representative_60 is the primary held-out estimate. Diagnostic_20 is a
targeted, non-representative analysis reported separately. Combined figures
are descriptive only.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(out_dir: Path = OUT_DIR) -> dict:
    print("Comparing against pre-existing independent audit-target snapshot (read-only, pre-overwrite)...")
    print("Building final human reference from P1/P2 + adjudication + post-audit sources...")
    tokens, diagnostics = build_final_reference()

    rep = [t for t in tokens if t.set_name == "Representative"]
    diag = [t for t in tokens if t.set_name == "Diagnostic"]
    if len(rep) != N_REPRESENTATIVE_TOK:
        raise ValueError(f"Expected {N_REPRESENTATIVE_TOK} Representative tokens, got {len(rep)}")
    if len(diag) != N_DIAGNOSTIC_TOK:
        raise ValueError(f"Expected {N_DIAGNOSTIC_TOK} Diagnostic tokens, got {len(diag)}")

    pre_adj = {
        "Representative": compute_pre_adjudication_metrics(rep),
        "Diagnostic": compute_pre_adjudication_metrics(diag),
        "Combined": compute_pre_adjudication_metrics(tokens),
    }
    adj_summary = compute_adjudication_summary(tokens)
    human_canon = {
        "Representative": compute_human_canonical_metrics(rep),
        "Diagnostic": compute_human_canonical_metrics(diag),
        "Combined": compute_human_canonical_metrics(tokens),
    }
    audit_cmp = compare_against_audit_target(tokens)

    discrepancies: list[str] = []
    check_expected(pre_adj, EXPECTED["pre_adjudication"], "pre_adjudication", discrepancies)
    check_expected(adj_summary, EXPECTED["adjudication_summary"], "adjudication_summary", discrepancies)
    check_expected(human_canon, EXPECTED["human_canonical"], "human_canonical", discrepancies)

    print(f"Representative: {len(rep)} tokens, pre-adj agreement={pre_adj['Representative']['agreement']:.6f}, "
          f"human-canon agreement={human_canon['Representative']['agreement']:.6f}")
    print(f"Diagnostic:     {len(diag)} tokens, pre-adj agreement={pre_adj['Diagnostic']['agreement']:.6f}, "
          f"human-canon agreement={human_canon['Diagnostic']['agreement']:.6f}")
    print(f"Adjudication summary: {adj_summary}")
    if audit_cmp.get("available"):
        print(f"Audit-target agreement: {audit_cmp['final_upos_matches']}/{audit_cmp['n_tokens_compared']} "
              f"({audit_cmp['final_upos_agreement_rate']:.6f})")
    if discrepancies:
        print(f"\n{len(discrepancies)} DISCREPANCY(IES) vs. expected audit values:")
        for d in discrepancies:
            print(f"  {d}")
    else:
        print("\nAll computed values matched the expected audit values within tolerance.")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_tokens_csv(tokens, out_dir / "final_human_reference_tokens.csv")
    write_sentences_jsonl(tokens, out_dir / "final_human_reference_sentences.jsonl")
    write_final_conllu(tokens, out_dir / "final_human_reference.conllu")
    write_results_workbook(tokens, pre_adj, human_canon, out_dir / "Human_POS_Tagging_test_v1_Final_Adjudicated_Results.xlsx")
    write_per_upos_csv(human_canon["Representative"]["per_class"], out_dir / "representative_per_upos_metrics.csv")
    write_per_upos_csv(human_canon["Diagnostic"]["per_class"], out_dir / "diagnostic_per_upos_metrics.csv")
    write_confusions_csv(human_canon["Representative"]["confusions"], out_dir / "representative_human_silver_confusions.csv")
    write_confusions_csv(human_canon["Diagnostic"]["confusions"], out_dir / "diagnostic_human_silver_confusions.csv")
    write_label_distribution_csv(tokens, out_dir / "final_label_distribution.csv")
    write_adjudication_decision_summary_csv(tokens, out_dir / "adjudication_decision_summary.csv")
    write_metrics_json(
        {"pre_adjudication": pre_adj, "adjudication_summary": adj_summary,
         "human_vs_canonical": human_canon, "audit_target_comparison": audit_cmp,
         "discrepancies": discrepancies},
        out_dir / "human_silver_metrics.json",
    )
    write_metrics_txt(pre_adj, adj_summary, human_canon, audit_cmp, discrepancies, out_dir / "human_silver_metrics.txt")
    write_manifest(pre_adj, adj_summary, human_canon, audit_cmp, discrepancies, diagnostics, out_dir / "reference_build_manifest.json")
    write_command_log(out_dir / "reference_build_command_log.txt")
    write_results_readme(out_dir / "README.md")

    print(f"\nWrote outputs under {out_dir}")
    return {
        "tokens": tokens, "pre_adjudication": pre_adj, "adjudication_summary": adj_summary,
        "human_vs_canonical": human_canon, "audit_target_comparison": audit_cmp,
        "discrepancies": discrepancies, "diagnostics": diagnostics,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    run()


if __name__ == "__main__":
    main()
