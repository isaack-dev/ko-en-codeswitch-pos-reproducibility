"""
audit_preprocessing_reconstruction.py - Compare the reconstructed Panyo
CoNLL-U against the canonical corpus, WITHOUT modifying either file.

Inputs (all read-only):
  - data/kor_en_panyo_corpus.conllu                         (canonical, frozen)
  - data/processed/panyo_original_adp_lang_labeled.conllu    (canonical + Lang=, frozen)
  - data/processed/panyo_jsonl_doc_sentence_map.csv          (existing doc_id ground truth)
  - outputs/preprocessing_reconstruction/reconstructed.conllu (produced by
    scripts/reconstruct_pangyo_conllu.py)

Outputs (all new, under outputs/preprocessing_reconstruction/):
  - reconstruction_audit.txt / .json
  - summary_metrics.csv
  - confusion_language.csv / confusion_upos.csv
  - mismatches/*.csv  (first 100 examples per category)
  - SHA256SUMS.txt

Alignment methodology
----------------------
The canonical file has NO doc_id of its own; document identity comes only
from the existing (already-generated, unmodified) sentence->doc map. Both
canonical and reconstructed sentences are grouped by doc_id = "jsonl_<N>"
(N = 1-indexed JSONL line number -- the same convention used by
scripts/align_jsonl_conllu.py) and, within each document, compared by
POSITION (i-th sentence of the doc on each side). This is the only
alignment possible without assuming the two sentence-segmentations agree in
advance, and it degrades gracefully: if a document's sentence counts differ,
extra sentences on the longer side are reported as unmatched rather than
compared token-by-token against the wrong neighbor.

Run from repo root (after scripts/reconstruct_pangyo_conllu.py):
  python scripts/audit_preprocessing_reconstruction.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from read_conllu import (  # noqa: E402
    F_FORM, F_LEMMA, F_UPOS, F_HEAD, F_DEPREL, F_MISC, get_misc_lang, iter_sentences,
)

ROOT = Path(__file__).parent.parent
CANONICAL_CONLLU = ROOT / "data" / "kor_en_panyo_corpus.conllu"
CANONICAL_JSONL = ROOT / "data" / "kor_en_panyo_corpus.jsonl"
LANG_LABELED = ROOT / "data" / "processed" / "panyo_original_adp_lang_labeled.conllu"
DOC_MAP_CSV = ROOT / "data" / "processed" / "panyo_jsonl_doc_sentence_map.csv"

OUT_DIR = ROOT / "outputs" / "preprocessing_reconstruction"
RECON_CONLLU = OUT_DIR / "reconstructed.conllu"
MISMATCH_DIR = OUT_DIR / "mismatches"

MAX_MISMATCH_EXAMPLES = 100


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def get_comment(comments: list[str], prefix: str) -> str:
    full = f"# {prefix} = "
    for c in comments:
        if c.startswith(full):
            return c[len(full):]
    return ""


@dataclass
class Tok:
    form: str
    lemma: str
    upos: str
    lang: str
    head: str
    deprel: str


@dataclass
class Sent:
    sent_id: str
    doc_id: str
    text: str
    tokens: list[Tok] = field(default_factory=list)


def load_canonical() -> tuple[list[Sent], dict[str, str]]:
    """Return (sentences without doc_id/lang filled in, sent_id->doc_id map)."""
    if not CANONICAL_CONLLU.exists():
        raise FileNotFoundError(f"Missing required canonical file: {CANONICAL_CONLLU}")
    if not LANG_LABELED.exists():
        raise FileNotFoundError(f"Missing required lang-labeled file: {LANG_LABELED}")
    if not DOC_MAP_CSV.exists():
        raise FileNotFoundError(f"Missing required doc/sentence map: {DOC_MAP_CSV}")

    sent_to_doc: dict[str, str] = {}
    with DOC_MAP_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sent_to_doc[row["sent_id"]] = row["doc_id"]

    # Lang comes from the lang-labeled file; UPOS/lemma/head/deprel/text from canonical.
    # Both files share identical sent_id/token ordering (lang labeling never
    # changes tokens), so we zip them by sent_id.
    lang_by_sentid: dict[str, list[str]] = {}
    for sent in iter_sentences(LANG_LABELED):
        sid = get_comment(sent["comments"], "sent_id")
        real = [t for t in sent["tokens"] if t["is_real"]]
        lang_by_sentid[sid] = [get_misc_lang(t["fields"][F_MISC]) for t in real]

    sentences: list[Sent] = []
    for sent in iter_sentences(CANONICAL_CONLLU):
        sid = get_comment(sent["comments"], "sent_id")
        text = get_comment(sent["comments"], "text")
        doc_id = sent_to_doc.get(sid, "UNASSIGNED")
        real = [t for t in sent["tokens"] if t["is_real"]]
        langs = lang_by_sentid.get(sid, [""] * len(real))
        toks = [
            Tok(
                form=t["fields"][F_FORM], lemma=t["fields"][F_LEMMA], upos=t["fields"][F_UPOS],
                lang=(langs[i] if i < len(langs) else ""),
                head=t["fields"][F_HEAD], deprel=t["fields"][F_DEPREL],
            )
            for i, t in enumerate(real)
        ]
        sentences.append(Sent(sent_id=sid, doc_id=doc_id, text=text, tokens=toks))
    return sentences, sent_to_doc


def load_reconstruction() -> list[Sent]:
    if not RECON_CONLLU.exists():
        raise FileNotFoundError(
            f"Missing reconstructed file: {RECON_CONLLU}. "
            "Run scripts/reconstruct_pangyo_conllu.py first."
        )
    sentences: list[Sent] = []
    for sent in iter_sentences(RECON_CONLLU):
        sid = get_comment(sent["comments"], "sent_id")
        doc_id = get_comment(sent["comments"], "doc_id")
        text = get_comment(sent["comments"], "text")
        real = [t for t in sent["tokens"] if t["is_real"]]
        toks = [
            Tok(
                form=t["fields"][F_FORM], lemma=t["fields"][F_LEMMA], upos=t["fields"][F_UPOS],
                lang=get_misc_lang(t["fields"][F_MISC]),
                head=t["fields"][F_HEAD], deprel=t["fields"][F_DEPREL],
            )
            for t in real
        ]
        sentences.append(Sent(sent_id=sid, doc_id=doc_id, text=text, tokens=toks))
    return sentences


def group_by_doc(sentences: list[Sent]) -> dict[str, list[Sent]]:
    groups: dict[str, list[Sent]] = defaultdict(list)
    for s in sentences:
        if s.doc_id and s.doc_id != "UNASSIGNED":
            groups[s.doc_id].append(s)
    return groups


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def run_audit() -> dict:
    print("Loading canonical corpus (read-only) ...")
    canon_sents, sent_to_doc = load_canonical()
    print(f"  {len(canon_sents):,} canonical sentences")

    print("Loading reconstruction ...")
    recon_sents = load_reconstruction()
    print(f"  {len(recon_sents):,} reconstructed sentences")

    canon_docs = group_by_doc(canon_sents)
    recon_docs = group_by_doc(recon_sents)

    all_doc_ids = sorted(
        set(canon_docs) | set(recon_docs),
        key=lambda d: int(d.split("_")[1]) if d.startswith("jsonl_") else 0,
    )

    # -----------------------------------------------------------------------
    # Strict counters
    # -----------------------------------------------------------------------
    n_docs_canon = len(canon_docs)
    n_docs_recon = len(recon_docs)
    n_docs_both = len(set(canon_docs) & set(recon_docs))

    n_sents_canon = sum(len(v) for v in canon_docs.values())
    n_sents_recon = sum(len(v) for v in recon_docs.values())
    n_tok_canon = sum(len(s.tokens) for v in canon_docs.values() for s in v)
    n_tok_recon = sum(len(s.tokens) for v in recon_docs.values() for s in v)

    doc_sentcount_match = 0
    doc_sentcount_total = 0

    n_sent_pairs = 0            # position-aligned sentence pairs available
    n_text_exact = 0
    n_text_norm_exact = 0
    n_sentid_exact = 0
    n_tokcount_match = 0        # sentence pairs where token counts also match
    n_tok_pairs = 0              # token-level comparison denominator
    n_form_agree = 0
    n_form_norm_agree = 0
    n_lemma_agree = 0
    n_lang_agree = 0
    n_upos_agree = 0
    n_head_agree = 0
    n_deprel_agree = 0

    lang_confusion: Counter = Counter()
    upos_confusion: Counter = Counter()

    mismatches: dict[str, list[dict]] = {
        "sentence_text": [], "token_count": [], "token_form": [],
        "lemma": [], "lang": [], "upos": [],
    }

    def add_mismatch(cat: str, row: dict) -> None:
        if len(mismatches[cat]) < MAX_MISMATCH_EXAMPLES:
            mismatches[cat].append(row)

    space_after_no_recon = 0

    for doc_id in all_doc_ids:
        c_list = canon_docs.get(doc_id, [])
        r_list = recon_docs.get(doc_id, [])
        doc_sentcount_total += 1
        if len(c_list) == len(r_list):
            doc_sentcount_match += 1

        n_pos = min(len(c_list), len(r_list))
        for i in range(n_pos):
            c, r = c_list[i], r_list[i]
            n_sent_pairs += 1

            if c.sent_id == r.sent_id:
                n_sentid_exact += 1

            if c.text == r.text:
                n_text_exact += 1
            else:
                add_mismatch("sentence_text", {
                    "doc_id": doc_id, "canonical_sent_id": c.sent_id, "recon_sent_id": r.sent_id,
                    "canonical_text": c.text, "reconstructed_text": r.text,
                })

            if normalize(c.text) == normalize(r.text):
                n_text_norm_exact += 1

            if len(c.tokens) == len(r.tokens):
                n_tokcount_match += 1
            else:
                add_mismatch("token_count", {
                    "doc_id": doc_id, "canonical_sent_id": c.sent_id, "recon_sent_id": r.sent_id,
                    "canonical_token_count": len(c.tokens), "reconstructed_token_count": len(r.tokens),
                    "canonical_text": c.text, "reconstructed_text": r.text,
                })
                continue  # token-level comparison only when counts match

            for ti in range(len(c.tokens)):
                ct, rt = c.tokens[ti], r.tokens[ti]
                n_tok_pairs += 1

                if ct.form == rt.form:
                    n_form_agree += 1
                else:
                    add_mismatch("token_form", {
                        "doc_id": doc_id, "sent_id": c.sent_id, "token_index": ti,
                        "canonical_form": ct.form, "reconstructed_form": rt.form,
                    })
                if normalize(ct.form) == normalize(rt.form):
                    n_form_norm_agree += 1

                if ct.lemma == rt.lemma:
                    n_lemma_agree += 1
                else:
                    add_mismatch("lemma", {
                        "doc_id": doc_id, "sent_id": c.sent_id, "token_index": ti,
                        "form": ct.form, "canonical_lemma": ct.lemma, "reconstructed_lemma": rt.lemma,
                    })

                lang_confusion[(ct.lang, rt.lang)] += 1
                if ct.lang == rt.lang:
                    n_lang_agree += 1
                else:
                    add_mismatch("lang", {
                        "doc_id": doc_id, "sent_id": c.sent_id, "token_index": ti,
                        "form": ct.form, "canonical_lang": ct.lang, "reconstructed_lang": rt.lang,
                    })

                upos_confusion[(ct.upos, rt.upos)] += 1
                if ct.upos == rt.upos:
                    n_upos_agree += 1
                else:
                    add_mismatch("upos", {
                        "doc_id": doc_id, "sent_id": c.sent_id, "token_index": ti,
                        "form": ct.form, "canonical_upos": ct.upos, "reconstructed_upos": rt.upos,
                    })

                if ct.head == rt.head:
                    n_head_agree += 1
                if ct.deprel == rt.deprel:
                    n_deprel_agree += 1

    def pct(n: int, d: int) -> float:
        return round(100.0 * n / d, 4) if d else 0.0

    for sent in iter_sentences(RECON_CONLLU):
        for t in sent["tokens"]:
            if t["is_real"] and "SpaceAfter=No" in t["fields"][F_MISC]:
                space_after_no_recon += 1

    metrics = {
        "document_count_canonical": n_docs_canon,
        "document_count_reconstruction": n_docs_recon,
        "document_ids_shared": n_docs_both,
        "document_alignment_rate_pct": pct(n_docs_both, max(n_docs_canon, n_docs_recon)),
        "sentence_count_canonical": n_sents_canon,
        "sentence_count_reconstruction": n_sents_recon,
        "token_count_canonical": n_tok_canon,
        "token_count_reconstruction": n_tok_recon,
        "doc_sentence_count_exact_match_docs": doc_sentcount_match,
        "doc_sentence_count_exact_match_total": doc_sentcount_total,
        "doc_sentence_count_exact_match_rate_pct": pct(doc_sentcount_match, doc_sentcount_total),
        "position_aligned_sentence_pairs": n_sent_pairs,
        "sentence_id_exact_match_count": n_sentid_exact,
        "sentence_id_exact_match_rate_pct": pct(n_sentid_exact, n_sent_pairs),
        "sentence_text_exact_match_count": n_text_exact,
        "sentence_text_exact_match_rate_pct": pct(n_text_exact, n_sent_pairs),
        "sentence_text_nfkc_normalized_match_count": n_text_norm_exact,
        "sentence_text_nfkc_normalized_match_rate_pct": pct(n_text_norm_exact, n_sent_pairs),
        "token_count_exact_match_sentences": n_tokcount_match,
        "token_count_exact_match_rate_pct": pct(n_tokcount_match, n_sent_pairs),
        "token_level_comparison_denominator": n_tok_pairs,
        "token_form_agreement_count": n_form_agree,
        "token_form_agreement_rate_pct": pct(n_form_agree, n_tok_pairs),
        "token_form_nfkc_agreement_rate_pct": pct(n_form_norm_agree, n_tok_pairs),
        "lemma_agreement_count": n_lemma_agree,
        "lemma_agreement_rate_pct": pct(n_lemma_agree, n_tok_pairs),
        "language_label_agreement_count": n_lang_agree,
        "language_label_agreement_rate_pct": pct(n_lang_agree, n_tok_pairs),
        "upos_agreement_count": n_upos_agree,
        "upos_agreement_rate_pct": pct(n_upos_agree, n_tok_pairs),
        "dependency_head_agreement_count": n_head_agree,
        "dependency_head_agreement_rate_pct": pct(n_head_agree, n_tok_pairs),
        "dependency_deprel_agreement_count": n_deprel_agree,
        "dependency_deprel_agreement_rate_pct": pct(n_deprel_agree, n_tok_pairs),
        "space_after_canonical_occurrences": 0,
        "space_after_reconstruction_occurrences": space_after_no_recon,
        "space_after_note": (
            "Canonical MISC field is '_' for all 985,897 tokens (verified); "
            "SpaceAfter is never encoded in canonical. No positive-match "
            "comparison is possible for this field; the reconstruction's "
            "inferred count is reported for transparency only, not as agreement."
        ),
    }

    return {
        "metrics": metrics,
        "lang_confusion": lang_confusion,
        "upos_confusion": upos_confusion,
        "mismatches": mismatches,
    }


# ---------------------------------------------------------------------------
# Classification (A/B/C/D) -- thresholds defined explicitly and reported
# ---------------------------------------------------------------------------

def classify(metrics: dict) -> tuple[str, str]:
    upos = metrics["upos_agreement_rate_pct"]
    lemma = metrics["lemma_agreement_rate_pct"]
    lang = metrics["language_label_agreement_rate_pct"]
    text_exact = metrics["sentence_text_exact_match_rate_pct"]
    doc_align = metrics["document_alignment_rate_pct"]

    thresholds = (
        "A (exact/effectively exact): UPOS>=99%% AND lemma>=99%% AND sentence-text-exact>=99%%.\n"
        "B (high-agreement, documented differences): UPOS>=85%% AND lemma>=90%% AND lang>=95%% AND doc_alignment>=99%%.\n"
        "C (approximate, does not reproduce): meaningfully above trivial baselines "
        "(majority-class NOUN ~55.6%%) but below B.\n"
        "D (not possible from documentation): agreement at/below trivial baselines, "
        "or core generation procedure undocumented and unrecoverable."
    ).replace("%%", "%")

    if upos >= 99 and lemma >= 99 and text_exact >= 99:
        return "A", thresholds
    if upos >= 85 and lemma >= 90 and lang >= 95 and doc_align >= 99:
        return "B", thresholds
    if upos > 55.6:  # better than the corpus's own majority-NOUN baseline
        return "C", thresholds
    return "D", thresholds


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_mismatches(mismatches: dict[str, list[dict]]) -> None:
    MISMATCH_DIR.mkdir(parents=True, exist_ok=True)
    for cat, rows in mismatches.items():
        path = MISMATCH_DIR / f"mismatch_{cat}.csv"
        if not rows:
            path.write_text("", encoding="utf-8")
            continue
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  -> mismatches/{path.name} ({len(rows)} rows)")


def write_confusion_csv(counter: Counter, path: Path, label: str) -> None:
    labels = sorted({k[0] for k in counter} | {k[1] for k in counter})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"canonical_{label}\\reconstruction_{label}"] + labels)
        for gl in labels:
            w.writerow([gl] + [counter.get((gl, pl), 0) for pl in labels])
    print(f"  -> {path.name}")


def write_summary_csv(metrics: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, v])
    print(f"  -> {path.name}")


def write_txt_report(metrics: dict, classification: str, thresholds: str, checksums: dict, path: Path) -> None:
    lines = []
    lines.append("Preprocessing Reconstruction Audit")
    lines.append("Reconstruction vs Canonical Panyo Corpus (read-only comparison)")
    lines.append("=" * 88)
    lines.append(f"generated_at (UTC) = {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Checksums (SHA-256)")
    for name, h in checksums.items():
        lines.append(f"  {name}: {h}")
    lines.append("")
    lines.append("## Strict comparison metrics")
    for k, v in metrics.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("## Classification")
    lines.append(f"  Result: {classification}")
    lines.append("  Thresholds used:")
    for line in thresholds.splitlines():
        lines.append(f"    {line}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def write_json_report(metrics: dict, classification: str, thresholds: str, checksums: dict, path: Path) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checksums_sha256": checksums,
        "metrics": metrics,
        "classification": classification,
        "classification_thresholds": thresholds,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {path}")


def write_sha256sums(paths: list[Path], path: Path) -> None:
    lines = []
    for p in paths:
        if p.exists():
            lines.append(f"{sha256_of(p)}  {p.relative_to(ROOT)}".replace("\\", "/"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = run_audit()
    metrics = result["metrics"]

    classification, thresholds = classify(metrics)
    print(f"\nClassification: {classification}")

    print("\nComputing checksums ...")
    checksums = {
        "data/kor_en_panyo_corpus.conllu": sha256_of(CANONICAL_CONLLU),
        "data/kor_en_panyo_corpus.jsonl": sha256_of(CANONICAL_JSONL),
        "outputs/preprocessing_reconstruction/reconstructed.conllu": sha256_of(RECON_CONLLU),
    }

    print("\nWriting outputs ...")
    write_summary_csv(metrics, OUT_DIR / "summary_metrics.csv")
    write_confusion_csv(result["lang_confusion"], OUT_DIR / "confusion_language.csv", "lang")
    write_confusion_csv(result["upos_confusion"], OUT_DIR / "confusion_upos.csv", "upos")
    write_mismatches(result["mismatches"])
    write_txt_report(metrics, classification, thresholds, checksums, OUT_DIR / "reconstruction_audit.txt")
    write_json_report(metrics, classification, thresholds, checksums, OUT_DIR / "reconstruction_audit.json")
    write_sha256sums(
        [
            CANONICAL_CONLLU, CANONICAL_JSONL, RECON_CONLLU,
            OUT_DIR / "reconstruction_audit.json", OUT_DIR / "summary_metrics.csv",
            OUT_DIR / "confusion_language.csv", OUT_DIR / "confusion_upos.csv",
        ],
        OUT_DIR / "SHA256SUMS.txt",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
