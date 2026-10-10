"""
reconstruct_pangyo_conllu.py - Transparent reconstruction of the Panyo
JSONL -> CoNLL-U conversion, for reproducibility auditing ONLY.

This script reads ONLY:
  - data/kor_en_panyo_corpus.jsonl  (field "pangyo" -- the code-switched text)
  - the explicit rule tables defined in this file (RULES, below)

It NEVER reads data/kor_en_panyo_corpus.conllu (or any derived annotation
file) as an input to the reconstruction. No canonical UPOS, lemma, Lang=,
HEAD, or DEPREL value is ever copied into the output. Output is written
only under outputs/preprocessing_reconstruction/ and never overwrites the
canonical corpus, split files, or any existing model/prediction output.

WHY THIS EXISTS
----------------
The canonical CoNLL-U file (data/kor_en_panyo_corpus.conllu) has no recorded
generation script, prompt, or rule documentation in this repository (verified:
no "GPT-5"/"GPT5" mention anywhere in the repo, and reports/internal_research_report_ko.md
describes research framing/results but not the tokenization/tagging procedure
itself). This script is a best-effort, fully transparent RECONSTRUCTION built
only from what IS documented (guidelines/*.md) plus deterministic patterns
independently observable in the JSONL text alone. It does not, and does not
claim to, reproduce the canonical file's exact generation process. See
RULES below and outputs/preprocessing_reconstruction/README.md for the
documented-vs-inferred provenance of every rule.

RULE PROVENANCE (see RULES dict)
---------------------------------
DOCUMENTED (verified against a guideline or decisions file in this repo):
  - lang_classify         guidelines/language_labeling_guideline.md (exact heuristic)
  - ko_particle_adp       guidelines/annotation_decisions.md, Decision 2 (explicit particle list)
  - tokenize_script_split guidelines/annotation_decisions.md, Step-2 addendum #3
                          ("debug하는" tokenizes as "debug"(EN)+"하는"(KO))

INFERRED (no guideline documents these; chosen as the smallest reasonable
rule from patterns independently observed in the reconstruction's OWN
tokenization output, never by reading canonical annotations):
  - punct_is_punct        every non-alnum, non-script character is its own PUNCT token
  - digit_is_num          a digit run (with embedded , or . separators) is NUM
  - da_final_verb         a Hangul-only token ending in the syllable "다" is VERB
  - en_function_words     a small closed-class dictionary of English function words
  - propn_capitalized     a capitalized, non-sentence-initial Latin token is PROPN
  - noun_fallback         anything not otherwise classified is NOUN
  - en_lemma_lowercase    a capitalized Latin token's lemma is its lowercased form
  - ko_lemma_identity     a Hangul-only token's lemma equals its form
  - flat_root_first       dependency tree is flat; root = first real token
  - space_after_no        SpaceAfter=No is inferred wherever two sub-tokens came from
                          the same whitespace-delimited chunk of the source text
  - sentence_segmentation split "pangyo" text on sentence-final .!? followed by
                          whitespace/EOS, protecting decimal numbers

None of the inferred rules are claimed to exactly reproduce the original
conversion. Each is independently configurable via ReconstructionConfig.

Run from repo root:
  python scripts/reconstruct_pangyo_conllu.py                 # full corpus
  python scripts/reconstruct_pangyo_conllu.py --dry-run        # first 5 records, print only
  python scripts/reconstruct_pangyo_conllu.py --limit 100      # first 100 records, write output
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent.parent
JSONL_PATH = ROOT / "data" / "kor_en_panyo_corpus.jsonl"
OUT_DIR = ROOT / "outputs" / "preprocessing_reconstruction"
DEFAULT_OUTPUT = OUT_DIR / "reconstructed.conllu"

# ---------------------------------------------------------------------------
# Rule provenance table (single source of truth for the README/report)
# ---------------------------------------------------------------------------

RULES: dict[str, dict] = {
    "lang_classify": {
        "documented": True,
        "source": "guidelines/language_labeling_guideline.md",
        "description": "Hangul-only -> KO, Latin-only -> EN, both -> MIXED, neither -> OTHER.",
    },
    "ko_particle_adp": {
        "documented": True,
        "source": "guidelines/annotation_decisions.md (Decision 2)",
        "description": "Explicit list of Korean particles/postpositions -> ADP.",
    },
    "tokenize_script_split": {
        "documented": True,
        "source": "guidelines/annotation_decisions.md (Step 2 addendum, item 3)",
        "description": "Tokenize at Hangul/Latin/digit script-class boundaries "
                        "(e.g. 'debug하는' -> 'debug' + '하는').",
    },
    "punct_is_punct": {
        "documented": False,
        "source": "inferred: 100% consistent in canonical data, not documented",
        "description": "Each punctuation character is its own PUNCT token.",
    },
    "digit_is_num": {
        "documented": False,
        "source": "inferred: 100% consistent in canonical data, not documented",
        "description": "A digit run (optionally with embedded , or . separators) -> NUM.",
    },
    "da_final_verb": {
        "documented": False,
        "source": "inferred: not documented anywhere in this repo",
        "description": "A Hangul-only token ending in the syllable '다' -> VERB.",
    },
    "en_function_words": {
        "documented": False,
        "source": "inferred: not documented; small closed-class dictionary",
        "description": "Closed-class English function words mapped to DET/CCONJ/AUX/ADP/PRON/SCONJ.",
    },
    "propn_capitalized": {
        "documented": False,
        "source": "inferred: not documented, and known imperfect even against canonical",
        "description": "A capitalized, non-sentence-initial Latin token -> PROPN.",
    },
    "noun_fallback": {
        "documented": False,
        "source": "inferred: consistent with documented 'high NOUN rate' observation",
        "description": "Anything not otherwise classified -> NOUN.",
    },
    "en_lemma_lowercase": {
        "documented": False,
        "source": "inferred: 100% consistent pattern observed independently in tokenized text",
        "description": "A capitalized Latin token's lemma is its lowercased form.",
    },
    "ko_lemma_identity": {
        "documented": False,
        "source": "inferred: Hangul tokens are not morphologically lemmatized",
        "description": "A Hangul-only token's lemma equals its surface form.",
    },
    "flat_root_first": {
        "documented": False,
        "source": "inferred structural choice",
        "description": "Flat dependency tree; first real token is root, others attach to it.",
    },
    "space_after_no": {
        "documented": False,
        "source": "inferred from source text spacing; canonical never encodes SpaceAfter",
        "description": "SpaceAfter=No between sub-tokens split out of one whitespace-delimited chunk.",
    },
    "sentence_segmentation": {
        "documented": False,
        "source": "inferred: no segmentation algorithm is documented anywhere",
        "description": "Split on sentence-final .!? followed by whitespace/EOS, protecting decimals.",
    },
}


# ---------------------------------------------------------------------------
# Configuration (every inferred rule is toggleable)
# ---------------------------------------------------------------------------

KO_PARTICLES = (
    "은", "는", "이", "가", "을", "를", "의", "에",
    "에서", "으로", "로", "와", "과", "도", "만",
    "까지", "부터",
)

EN_FUNCTION_WORDS: dict[str, str] = {
    "the": "DET", "a": "DET", "an": "DET", "this": "DET", "that": "DET",
    "these": "DET", "those": "DET",
    "and": "CCONJ", "or": "CCONJ", "but": "CCONJ",
    "because": "SCONJ", "although": "SCONJ", "if": "SCONJ", "since": "SCONJ", "while": "SCONJ",
    "is": "AUX", "are": "AUX", "was": "AUX", "were": "AUX", "be": "AUX",
    "been": "AUX", "being": "AUX", "am": "AUX",
    "of": "ADP", "in": "ADP", "on": "ADP", "at": "ADP", "to": "ADP", "for": "ADP",
    "with": "ADP", "by": "ADP", "from": "ADP", "as": "ADP", "into": "ADP", "about": "ADP",
    "he": "PRON", "she": "PRON", "it": "PRON", "they": "PRON", "i": "PRON",
    "you": "PRON", "we": "PRON",
}


@dataclass(frozen=True)
class ReconstructionConfig:
    apply_da_final_verb: bool = True
    apply_en_function_words: bool = True
    apply_propn_capitalization: bool = True
    apply_space_after_inference: bool = True
    ko_particles: tuple[str, ...] = KO_PARTICLES
    en_function_words: dict[str, str] = field(default_factory=lambda: dict(EN_FUNCTION_WORDS))


DEFAULT_CONFIG = ReconstructionConfig()

HANGUL_RANGES = ((0xAC00, 0xD7A3), (0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F), (0xD7B0, 0xD7FF))


def has_hangul(s: str) -> bool:
    return any(any(lo <= ord(c) <= hi for lo, hi in HANGUL_RANGES) for c in s)


def has_latin(s: str) -> bool:
    return any("a" <= c <= "z" or "A" <= c <= "Z" for c in s)


def classify_lang(form: str) -> str:
    """rule: lang_classify (DOCUMENTED, guidelines/language_labeling_guideline.md)."""
    h, la = has_hangul(form), has_latin(form)
    if h and la:
        return "MIXED"
    if h:
        return "KO"
    if la:
        return "EN"
    return "OTHER"


# ---------------------------------------------------------------------------
# JSONL loading (fails clearly on malformed records)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JsonlRecord:
    line_no: int
    pangyo: str


def load_jsonl(path: Path) -> list[JsonlRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Required source file not found: {path}")
    records: list[JsonlRecord] = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: malformed JSON: {e}") from e
            if not isinstance(obj, dict) or "pangyo" not in obj:
                raise ValueError(f"{path}:{i}: record missing required 'pangyo' field: {obj!r}")
            pangyo = obj["pangyo"]
            if not isinstance(pangyo, str) or not pangyo.strip():
                raise ValueError(f"{path}:{i}: 'pangyo' field is empty or not a string")
            records.append(JsonlRecord(line_no=i, pangyo=pangyo))
    return records


# ---------------------------------------------------------------------------
# Sentence segmentation (rule: sentence_segmentation, INFERRED)
# ---------------------------------------------------------------------------

_SENT_BOUNDARY = re.compile(
    r"(?<!\d)(?<=[.!?。！？])(?!\d)\s+"
)


def segment_sentences(text: str) -> list[str]:
    """Split on sentence-final .!? (incl. fullwidth) followed by whitespace,
    protecting decimal numbers (a digit immediately before/after the boundary
    blocks the split, e.g. '7.0.2' or '1.4')."""
    text = " ".join(text.split())
    if not text:
        return []
    parts = _SENT_BOUNDARY.split(text)
    return [p for p in (s.strip() for s in parts) if p]


# ---------------------------------------------------------------------------
# Tokenizer (rule: tokenize_script_split, DOCUMENTED pattern; boundaries chosen
# by script class -- Hangul run / Latin run / digit run / single punctuation char)
# ---------------------------------------------------------------------------

_TOKEN_PATTERN = re.compile(
    r"\d+(?:[.,]\d+)*"                      # digit run w/ embedded separators
    r"|[A-Za-z]+(?:[-'][A-Za-z]+)*"          # latin run w/ internal hyphen/apostrophe
    r"|[가-힣ᄀ-ᇿ㄰-㆏ꥠ-꥿ힰ-퟿]+"  # hangul run
    r"|."                                    # fallback: single non-whitespace char
)


@dataclass(frozen=True)
class RawToken:
    form: str
    space_after: bool  # True if a real space followed this token in the source text


def tokenize_sentence(text: str) -> list[RawToken]:
    """Whitespace-delimited chunks, each further split at script-class
    boundaries. space_after=False is set (rule: space_after_no) between
    sub-tokens that came from the same whitespace-delimited chunk."""
    words = text.split(" ")
    tokens: list[RawToken] = []
    for wi, word in enumerate(words):
        if not word:
            continue
        sub = [m.group(0) for m in _TOKEN_PATTERN.finditer(word)]
        for si, s in enumerate(sub):
            is_last_sub = si == len(sub) - 1
            tokens.append(RawToken(form=s, space_after=is_last_sub))
    return tokens


# ---------------------------------------------------------------------------
# UPOS assignment (documented + inferred rules, in priority order)
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"^[^\w가-힣ᄀ-ᇿ㄰-㆏]+$", re.UNICODE)
_DIGIT_RE = re.compile(r"^\d+(?:[.,]\d+)*$")


def assign_upos(form: str, is_sentence_initial: bool, config: ReconstructionConfig) -> tuple[str, str]:
    """Return (upos, rule_name_applied)."""
    if _PUNCT_RE.match(form) and not has_latin(form) and not has_hangul(form):
        return "PUNCT", "punct_is_punct"
    if _DIGIT_RE.match(form):
        return "NUM", "digit_is_num"

    is_ko = has_hangul(form) and not has_latin(form)
    is_en = has_latin(form) and not has_hangul(form)

    if is_ko:
        if form in config.ko_particles:
            return "ADP", "ko_particle_adp"
        if config.apply_da_final_verb and form.endswith("다"):
            return "VERB", "da_final_verb"
        return "NOUN", "noun_fallback"

    if is_en:
        low = form.lower()
        if config.apply_en_function_words and low in config.en_function_words:
            return config.en_function_words[low], "en_function_words"
        if config.apply_propn_capitalization and not is_sentence_initial and form[:1].isupper():
            return "PROPN", "propn_capitalized"
        return "NOUN", "noun_fallback"

    return "NOUN", "noun_fallback"


def assign_lemma(form: str, upos: str) -> str:
    is_en = has_latin(form) and not has_hangul(form)
    if is_en and form[:1].isupper():
        return form.lower()  # rule: en_lemma_lowercase
    return form  # rule: ko_lemma_identity (and default identity elsewhere)


# ---------------------------------------------------------------------------
# Sentence assembly
# ---------------------------------------------------------------------------

@dataclass
class ReconToken:
    form: str
    lemma: str
    upos: str
    lang: str
    space_after: bool
    upos_rule: str


@dataclass
class ReconSentence:
    doc_id: str
    sent_id: str
    text: str
    tokens: list[ReconToken]


def build_sentence(text: str, doc_id: str, sent_id: str, config: ReconstructionConfig) -> ReconSentence:
    raw_tokens = tokenize_sentence(text)
    tokens: list[ReconToken] = []
    for i, rt in enumerate(raw_tokens):
        upos, rule = assign_upos(rt.form, is_sentence_initial=(i == 0), config=config)
        lemma = assign_lemma(rt.form, upos)
        lang = classify_lang(rt.form)
        tokens.append(ReconToken(
            form=rt.form, lemma=lemma, upos=upos, lang=lang,
            space_after=rt.space_after, upos_rule=rule,
        ))
    return ReconSentence(doc_id=doc_id, sent_id=sent_id, text=text, tokens=tokens)


def serialize_sentence(sent: ReconSentence, config: ReconstructionConfig) -> str:
    """Serialize to 10-column CoNLL-U with a flat dependency tree
    (rule: flat_root_first)."""
    real = sent.tokens
    n = len(real)
    root_idx = 1 if n else 0  # 1-indexed; rule: flat_root_first

    lines = [
        f"# doc_id = {sent.doc_id}",
        f"# sent_id = {sent.sent_id}",
        f"# text = {sent.text}",
    ]
    for i, tok in enumerate(real, start=1):
        head = 0 if i == root_idx else root_idx
        deprel = "root" if i == root_idx else ("punct" if tok.upos == "PUNCT" else "dep")
        misc_parts = [f"Lang={tok.lang}"]
        if config.apply_space_after_inference and not tok.space_after:
            misc_parts.insert(0, "SpaceAfter=No")
        misc = "|".join(misc_parts) if misc_parts else "_"
        lines.append(
            "\t".join([str(i), tok.form, tok.lemma, tok.upos, "_", "_", str(head), deprel, "_", misc])
        )
    return "\n".join(lines) + "\n\n"


# ---------------------------------------------------------------------------
# Top-level reconstruction over the whole JSONL
# ---------------------------------------------------------------------------

def reconstruct_corpus(
    records: list[JsonlRecord],
    config: ReconstructionConfig = DEFAULT_CONFIG,
) -> list[ReconSentence]:
    sentences: list[ReconSentence] = []
    global_counter = 0
    for rec in records:
        doc_id = f"jsonl_{rec.line_no}"
        for raw_text in segment_sentences(rec.pangyo):
            global_counter += 1
            sent_id = f"pangyo{global_counter}"
            sentences.append(build_sentence(raw_text, doc_id, sent_id, config))
    return sentences


def write_conllu(sentences: list[ReconSentence], path: Path, config: ReconstructionConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for sent in sentences:
            fh.write(serialize_sentence(sent, config))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N JSONL records.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Process only the first 5 records and print to stdout; write nothing.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-da-verb", action="store_true", help="Disable the da_final_verb rule.")
    parser.add_argument("--no-en-function-words", action="store_true")
    parser.add_argument("--no-propn-capitalization", action="store_true")
    parser.add_argument("--no-space-after", action="store_true")
    args = parser.parse_args()

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    config = ReconstructionConfig(
        apply_da_final_verb=not args.no_da_verb,
        apply_en_function_words=not args.no_en_function_words,
        apply_propn_capitalization=not args.no_propn_capitalization,
        apply_space_after_inference=not args.no_space_after,
    )

    print(f"Loading {JSONL_PATH} ...")
    records = load_jsonl(JSONL_PATH)
    print(f"  {len(records):,} JSONL records loaded")

    if args.dry_run:
        records = records[:5]
        print(f"[dry-run] Using first {len(records)} records; nothing will be written.")
    elif args.limit is not None:
        records = records[: args.limit]
        print(f"[--limit] Using first {len(records)} records.")

    sentences = reconstruct_corpus(records, config)
    n_tokens = sum(len(s.tokens) for s in sentences)
    print(f"Reconstructed {len(sentences):,} sentences, {n_tokens:,} tokens "
          f"from {len(records):,} documents.")

    if args.dry_run:
        for s in sentences[:10]:
            sys.stdout.write(serialize_sentence(s, config))
        return

    write_conllu(sentences, args.output, config)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
