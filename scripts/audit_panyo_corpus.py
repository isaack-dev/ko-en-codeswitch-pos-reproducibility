"""
audit_panyo_corpus.py - Corpus audit for the Panyo code-switching dataset.

Investigates anomalies found in Step 1:
  A. MIXED token audit   (expected count = 0; explains morpheme-level tokenization)
  B. Boundary sample     (50 KO↔EN adjacent transitions)
  C. ADP particle audit  (Korean particles tagged ADP)
  D. Potential adjective audit  (0 ADJ in corpus — what was annotated instead?)
  E. High-NOUN sentence audit   (suspicious NOUN over-annotation)

Input (preferred): data/processed/panyo_original_adp_docsplit_full.conllu
Fallback:          data/processed/panyo_original_adp_lang_labeled.conllu

Outputs:
    results/data_audit_report.md
    results/sample_boundary_cases.csv
    results/sample_adp_particle_cases.csv
    results/sample_potential_adjective_cases.csv
    results/sample_high_noun_sentences.csv
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from read_conllu import F_FORM, F_MISC, F_UPOS, get_misc_lang, iter_sentences

ROOT    = Path(__file__).parent.parent
DATA    = ROOT / "data"
RESULTS = ROOT / "results"
PROC    = DATA / "processed"

DOCSPLIT_PATH = PROC / "panyo_original_adp_docsplit_full.conllu"
LABELED_PATH  = PROC / "panyo_original_adp_lang_labeled.conllu"

SAMPLE_BOUNDARIES   = 50
SAMPLE_ADP          = 100
SAMPLE_ADJ          = 100
SAMPLE_HIGH_NOUN    = 50
MIN_TOKENS_NOUN     = 8   # minimum real tokens for high-NOUN sentence check

# Korean particle surface forms to look for in ADP audit
KO_PARTICLES = {
    "은", "는", "이", "가", "을", "를", "의", "에", "에서",
    "로", "으로", "와", "과", "도", "만", "까지", "부터",
    "보다", "처럼", "하고",
}

# English adjective suffixes for potential-adjective heuristic
EN_ADJ_SUFFIXES = (
    "ive", "al", "ous", "ic", "able", "ible", "ary", "ful", "less",
    "ent", "ant", "ish", "ive", "ory",
)

# Korean form endings that suggest adjective-like predicates
KO_ADJ_ENDINGS = (
    "한", "적인", "스러운", "있는", "없는", "높은", "낮은",
    "큰", "작은", "많은", "적은", "좋은", "나쁜", "새로운",
    "중요한", "다양한", "특별한",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_comment_value(comments: list[str], key: str) -> str:
    prefix = f"# {key} = "
    for c in comments:
        if c.startswith(prefix):
            return c[len(prefix):]
    return ""


def sent_context(sent: dict, max_tokens: int = 12) -> str:
    """Return a short string of the first N token forms."""
    forms = [
        t["fields"][F_FORM] for t in sent["tokens"]
        if t["is_real"]
    ][:max_tokens]
    return " ".join(forms)


def has_hangul(s: str) -> bool:
    for c in s:
        cp = ord(c)
        if (0xAC00 <= cp <= 0xD7A3 or 0x1100 <= cp <= 0x11FF or
                0x3130 <= cp <= 0x318F):
            return True
    return False


def has_latin(s: str) -> bool:
    return any("a" <= c <= "z" or "A" <= c <= "Z" for c in s)


# ---------------------------------------------------------------------------
# Audit A: MIXED token count
# ---------------------------------------------------------------------------

def audit_mixed(path: Path) -> dict:
    mixed_count = 0
    total_tokens = 0
    for sent in iter_sentences(path):
        for tok in sent["tokens"]:
            if not tok["is_real"]:
                continue
            total_tokens += 1
            if get_misc_lang(tok["fields"][F_MISC]) == "MIXED":
                mixed_count += 1
    return {"mixed_count": mixed_count, "total_tokens": total_tokens}


# ---------------------------------------------------------------------------
# Audit B: Boundary samples (KO↔EN adjacent transitions)
# ---------------------------------------------------------------------------

def audit_boundaries(path: Path, n_samples: int = SAMPLE_BOUNDARIES) -> list[dict]:
    samples = []
    for sent in iter_sentences(path):
        if len(samples) >= n_samples:
            break
        sent_id  = get_comment_value(sent["comments"], "sent_id")
        doc_id   = get_comment_value(sent["comments"], "doc_id")
        split    = get_comment_value(sent["comments"], "split")
        context  = sent_context(sent)

        real_toks = [t for t in sent["tokens"] if t["is_real"]]
        for i in range(len(real_toks) - 1):
            if len(samples) >= n_samples:
                break
            a, b = real_toks[i], real_toks[i + 1]
            la = get_misc_lang(a["fields"][F_MISC])
            lb = get_misc_lang(b["fields"][F_MISC])
            if {la, lb} == {"KO", "EN"}:
                samples.append({
                    "sent_id":      sent_id,
                    "doc_id":       doc_id,
                    "split":        split,
                    "prev_form":    a["fields"][F_FORM],
                    "prev_lang":    la,
                    "prev_upos":    a["fields"][F_UPOS],
                    "curr_form":    b["fields"][F_FORM],
                    "curr_lang":    lb,
                    "curr_upos":    b["fields"][F_UPOS],
                    "direction":    f"{la}→{lb}",
                    "context":      context,
                })
    return samples


# ---------------------------------------------------------------------------
# Audit C: ADP particle samples
# ---------------------------------------------------------------------------

def audit_adp_particles(path: Path, n_samples: int = SAMPLE_ADP) -> list[dict]:
    samples = []
    for sent in iter_sentences(path):
        if len(samples) >= n_samples:
            break
        sent_id = get_comment_value(sent["comments"], "sent_id")
        doc_id  = get_comment_value(sent["comments"], "doc_id")
        split   = get_comment_value(sent["comments"], "split")
        context = sent_context(sent)

        real_toks = [t for t in sent["tokens"] if t["is_real"]]
        for i, tok in enumerate(real_toks):
            if len(samples) >= n_samples:
                break
            form = tok["fields"][F_FORM]
            upos = tok["fields"][F_UPOS]
            lang = get_misc_lang(tok["fields"][F_MISC])

            if upos == "ADP" and lang == "KO" and form in KO_PARTICLES:
                prev_form = real_toks[i - 1]["fields"][F_FORM] if i > 0 else ""
                next_form = real_toks[i + 1]["fields"][F_FORM] if i < len(real_toks) - 1 else ""
                samples.append({
                    "sent_id":   sent_id,
                    "doc_id":    doc_id,
                    "split":     split,
                    "token":     form,
                    "upos":      upos,
                    "lang":      lang,
                    "prev_token": prev_form,
                    "next_token": next_form,
                    "context":   context,
                })
    return samples


# ---------------------------------------------------------------------------
# Audit D: Potential adjective cases
# ---------------------------------------------------------------------------

def _looks_like_adj_en(form: str) -> bool:
    fl = form.lower()
    return any(fl.endswith(sfx) for sfx in EN_ADJ_SUFFIXES) and len(form) > 4


def _looks_like_adj_ko(form: str) -> bool:
    return any(form.endswith(sfx) for sfx in KO_ADJ_ENDINGS)


def audit_potential_adj(path: Path, n_samples: int = SAMPLE_ADJ) -> list[dict]:
    samples = []
    for sent in iter_sentences(path):
        if len(samples) >= n_samples:
            break
        sent_id = get_comment_value(sent["comments"], "sent_id")
        doc_id  = get_comment_value(sent["comments"], "doc_id")
        split   = get_comment_value(sent["comments"], "split")
        context = sent_context(sent)

        for tok in sent["tokens"]:
            if len(samples) >= n_samples:
                break
            if not tok["is_real"]:
                continue
            form = tok["fields"][F_FORM]
            upos = tok["fields"][F_UPOS]
            lang = get_misc_lang(tok["fields"][F_MISC])

            if upos not in ("NOUN", "VERB"):
                continue

            reason = ""
            if lang == "EN" and _looks_like_adj_en(form):
                reason = f"EN form ending suggests adjective (upos={upos})"
            elif lang == "KO" and _looks_like_adj_ko(form):
                reason = f"KO form ending suggests predicative adjective (upos={upos})"

            if reason:
                samples.append({
                    "sent_id": sent_id,
                    "doc_id":  doc_id,
                    "split":   split,
                    "token":   form,
                    "upos":    upos,
                    "lang":    lang,
                    "reason":  reason,
                    "context": context,
                })
    return samples


# ---------------------------------------------------------------------------
# Audit E: High-NOUN sentences
# ---------------------------------------------------------------------------

def audit_high_noun(path: Path, n_samples: int = SAMPLE_HIGH_NOUN) -> list[dict]:
    candidates = []
    for sent in iter_sentences(path):
        sent_id = get_comment_value(sent["comments"], "sent_id")
        doc_id  = get_comment_value(sent["comments"], "doc_id")
        split   = get_comment_value(sent["comments"], "split")

        real_toks = [t for t in sent["tokens"] if t["is_real"]]
        n = len(real_toks)
        if n < MIN_TOKENS_NOUN:
            continue
        n_noun = sum(1 for t in real_toks if t["fields"][F_UPOS] == "NOUN")
        noun_pct = n_noun / n

        forms = [t["fields"][F_FORM] for t in real_toks[:15]]
        candidates.append({
            "sent_id":   sent_id,
            "doc_id":    doc_id,
            "split":     split,
            "n_tokens":  n,
            "n_noun":    n_noun,
            "noun_pct":  f"{100 * noun_pct:.1f}",
            "context":   " ".join(forms),
        })

    # Sort by noun_pct descending, take top N
    candidates.sort(key=lambda x: float(x["noun_pct"]), reverse=True)
    return candidates[:n_samples]


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], fieldnames: list[str], path: Path) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Choose input file
    if DOCSPLIT_PATH.exists():
        input_path = DOCSPLIT_PATH
        print(f"Using docsplit full: {DOCSPLIT_PATH.name}")
    elif LABELED_PATH.exists():
        input_path = LABELED_PATH
        print(f"Docsplit full not found; using labeled: {LABELED_PATH.name}")
    else:
        sys.exit("ERROR: No suitable input file found.")

    # ---- A: MIXED ----
    print("A: MIXED token audit ...")
    mixed_result = audit_mixed(input_path)

    # ---- B: Boundary samples ----
    print("B: Boundary sample audit ...")
    boundary_samples = audit_boundaries(input_path)

    # ---- C: ADP particles ----
    print("C: ADP particle audit ...")
    adp_samples = audit_adp_particles(input_path)

    # ---- D: Potential adjectives ----
    print("D: Potential adjective audit ...")
    adj_samples = audit_potential_adj(input_path)

    # ---- E: High-NOUN sentences ----
    print("E: High-NOUN sentence audit ...")
    high_noun_samples = audit_high_noun(input_path)

    # ---- Write CSVs ----
    write_csv(
        boundary_samples,
        ["sent_id", "doc_id", "split", "prev_form", "prev_lang", "prev_upos",
         "curr_form", "curr_lang", "curr_upos", "direction", "context"],
        RESULTS / "sample_boundary_cases.csv",
    )
    print(f"  Wrote sample_boundary_cases.csv ({len(boundary_samples)} rows)")

    write_csv(
        adp_samples,
        ["sent_id", "doc_id", "split", "token", "upos", "lang",
         "prev_token", "next_token", "context"],
        RESULTS / "sample_adp_particle_cases.csv",
    )
    print(f"  Wrote sample_adp_particle_cases.csv ({len(adp_samples)} rows)")

    write_csv(
        adj_samples,
        ["sent_id", "doc_id", "split", "token", "upos", "lang", "reason", "context"],
        RESULTS / "sample_potential_adjective_cases.csv",
    )
    print(f"  Wrote sample_potential_adjective_cases.csv ({len(adj_samples)} rows)")

    write_csv(
        high_noun_samples,
        ["sent_id", "doc_id", "split", "n_tokens", "n_noun", "noun_pct", "context"],
        RESULTS / "sample_high_noun_sentences.csv",
    )
    print(f"  Wrote sample_high_noun_sentences.csv ({len(high_noun_samples)} rows)")

    # ---- Write audit report ----
    report_lines = _build_report(
        input_path,
        mixed_result,
        boundary_samples,
        adp_samples,
        adj_samples,
        high_noun_samples,
    )
    report_path = RESULTS / "data_audit_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"  Wrote data_audit_report.md")


def _build_report(
    input_path: Path,
    mixed: dict,
    boundaries: list[dict],
    adp: list[dict],
    adj: list[dict],
    high_noun: list[dict],
) -> list[str]:
    lines = [
        "# Panyo Corpus Data Audit Report",
        "",
        f"Input file: `{input_path.name}`",
        "",
        "---",
        "",
        "## A. MIXED Token Audit",
        "",
        f"- **MIXED token count: {mixed['mixed_count']:,}** (out of {mixed['total_tokens']:,} real tokens)",
        "",
        "### Explanation",
        "The Panyo corpus tokenizes at morpheme boundaries. Code-switched words that",
        "combine English and Korean morphemes (e.g., `Python으로`, `debug하는`) are",
        "split into separate tokens:",
        "  - `Python` (Latin → Lang=EN)",
        "  - `으로` (Hangul → Lang=KO)",
        "",
        "Therefore no single token ever contains both scripts, and `Lang=MIXED` is never",
        "assigned. This is a property of the **tokenization scheme**, not an error.",
        "Mixed-script expressions appear as EN→KO or KO→EN boundary pairs at the token level.",
        "",
        "**Implication:** The `MIXED` label is not useful for this corpus.",
        "Code-switching density is better measured via boundary counts (KO↔EN transitions).",
        "",
        "---",
        "",
        "## B. Code-Switching Boundary Samples",
        "",
        f"Collected {len(boundaries)} sample KO↔EN boundary transitions.",
        "See `results/sample_boundary_cases.csv` for full list.",
        "",
        "### Example boundaries (first 5)",
    ]

    for i, b in enumerate(boundaries[:5], 1):
        lines.append(
            f"{i}. sent_id={b['sent_id']}  "
            f"`{b['prev_form']}`({b['prev_lang']}/{b['prev_upos']}) → "
            f"`{b['curr_form']}`({b['curr_lang']}/{b['curr_upos']})  "
            f"context: *{b['context']}*"
        )

    lines += [
        "",
        "### Pattern",
        "The most common boundary pattern is a Korean particle (ADP/KO) adjacent to an",
        "English noun (NOUN/EN), e.g.: `countries` → `를` or `를` → `list`.",
        "This reflects the head-final morphosyntax of Korean, where particles follow",
        "the nouns they case-mark, creating dense EN→KO→EN or KO→EN→KO sequences.",
        "",
        "---",
        "",
        "## C. ADP Particle Audit",
        "",
        f"Collected {len(adp)} sample Korean particles tagged as ADP.",
        "See `results/sample_adp_particle_cases.csv` for full list.",
        "",
        "### Key particles found",
    ]

    particle_freq = Counter(r["token"] for r in adp)
    for particle, cnt in particle_freq.most_common(10):
        lines.append(f"- `{particle}`: {cnt} occurrences in sample")

    lines += [
        "",
        "### Decision",
        "Korean postpositions/particles are **preserved as ADP** in the main experiment",
        "(see `guidelines/annotation_decisions.md`).  The KAIST-UD treebank uses a",
        "different convention for some of these; cross-treebank normalisation is",
        "deferred to a later step.",
        "",
        "**Implication for evaluation:** A model trained on English GUM (where ADP means",
        "preposition) and Korean KAIST may struggle with ADP-tagged Korean particles,",
        "because the syntactic distribution is very different. Error analysis should",
        "examine ADP precision/recall separately for Korean-origin tokens.",
        "",
        "---",
        "",
        "## D. Potential Adjective Audit (ADJ = 0 issue)",
        "",
        "The Panyo corpus has **0 ADJ tokens**.  For reference, KAIST-UD has 12,088 ADJ",
        "in the training split, and English GUM has 11,494.",
        "",
        f"Collected {len(adj)} candidate tokens with adjective-like surface forms",
        "annotated as NOUN or VERB.",
        "See `results/sample_potential_adjective_cases.csv` for full list.",
        "",
        "### Possible explanations",
        "1. **Korean descriptive verbs** (형용사): In Korean, descriptive predicates",
        "   that function as adjectives inflect like verbs. The Panyo corpus may",
        "   annotate them as VERB rather than ADJ.",
        "2. **Pre-nominal modifiers** annotated as NOUN: e.g., `많은` (many/numerous),",
        "   which modifies a following noun, may be tagged NOUN in this corpus.",
        "3. **English adjectives in attributive position** (e.g., `diplomatic`) tagged",
        "   as NOUN because the corpus uses a coarser tagset.",
        "4. **Intentional design**: The annotators may have decided to collapse ADJ into",
        "   NOUN/VERB to simplify annotation for code-switching text.",
        "",
        "**Implication:** A model evaluated on this corpus will never need to predict",
        "ADJ.  Cross-corpora evaluation (against KAIST or GUM) will require label",
        "mapping for ADJ↔VERB/NOUN comparison.",
        "",
        "---",
        "",
        "## E. High-NOUN Sentence Audit",
        "",
        "NOUN rate in the full corpus: **~55.6%** of all real tokens.",
        "For comparison: KAIST-UD ~30%, English GUM ~16.5%.",
        "",
        f"Top {len(high_noun)} sentences by NOUN proportion (≥{MIN_TOKENS_NOUN} tokens):",
    ]

    for i, s in enumerate(high_noun[:5], 1):
        lines.append(
            f"{i}. sent_id={s['sent_id']}  "
            f"NOUN={s['n_noun']}/{s['n_tokens']} ({s['noun_pct']}%)  "
            f"*{s['context']}*"
        )

    lines += [
        "",
        "### Interpretation",
        "The very high NOUN rate likely reflects several converging factors:",
        "- Korean nominalised verb forms (동명사) annotated as NOUN",
        "- English nouns (inherently frequent in CS text) annotated as NOUN",
        "- Some Korean adjective/verb forms annotated as NOUN (see section D)",
        "- A deliberate simplified annotation scheme for CS text",
        "",
        "**Implication for pipeline design:** A sequence model will find it hard to",
        "discriminate NOUN from VERB/ADJ since the boundary is blurred.  Joint",
        "language-POS models may benefit from the lang tag as an auxiliary signal.",
        "",
        "---",
        "",
        "## Summary of Implications for Experiments",
        "",
        "| Issue | Status | Action |",
        "|---|---|---|",
        "| MIXED=0 | Expected (morpheme tokenization) | Report in paper; use boundary count |",
        "| ADP particles | Preserved by design | Analyze ADP errors separately |",
        "| ADJ=0 | Corpus-specific; label collapse | Map labels for cross-corpus eval |",
        "| High NOUN rate | Corpus-specific annotation | Acknowledge in corpus description |",
        "",
        "See `guidelines/annotation_decisions.md` for the formal decision record.",
    ]

    return lines


if __name__ == "__main__":
    main()
