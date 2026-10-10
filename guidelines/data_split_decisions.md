# Data Split Decisions

## Overview

The Panyo code-switching corpus is split into train/dev/test sets for model development
and evaluation. Two split strategies have been implemented:

| Strategy | Files | Step |
|---|---|---|
| Sentence-level | `panyo_original_adp_{train,dev,test}.conllu` | Step 1 |
| Document-level | `panyo_original_adp_docsplit_{train,dev,test}.conllu` | Step 2 |

**Recommendation:** Use the **document-level split** for main experiments, provided that
alignment validation passes (≥98% alignment coverage; see below).

---

## Step 1: Sentence-Level Split (Created for Baseline)

### Rationale
Step 1 was created before document-level alignment was implemented, to have a working
split immediately available for initial experiments and debugging.

### Method
- Sentences shuffled with `random.Random(42)` at the sentence level.
- Ratio: 80% train / 10% dev / 10% test (by sentence count).
- Split comment `# split = train/dev/test` prepended to each sentence.

### Limitation
Sentence-level shuffling may place sentences from the same source paragraph (JSONL record)
into different splits (train, dev, or test). This creates **potential lexical leakage**:
sentences describing the same topic, using the same vocabulary, can appear in both the
training set and the evaluation sets. For robust generalization evaluation, this is
undesirable.

---

## Step 2: Document-Level Split (Recommended for Main Experiments)

### JSONL Record = Document Assumption
The `kor_en_panyo_corpus.jsonl` file contains 8,233 records. Each record has two fields:
- `"korean"`: the original Korean paragraph text
- `"pangyo"`: the code-switched version (Korean-English mixed paragraph)

Each JSONL record is treated as one **document**. The `"pangyo"` text of each record is
the exact concatenation of consecutive CoNLL-U `# text` comments joined by a single space.

This was verified empirically: the `# text` values of CoNLL-U sentences
`pangyo1`–`pangyo8` joined with `" "` match the `"pangyo"` field of JSONL record 1 exactly.
The same pattern holds throughout the corpus.

### Alignment Method
**Script:** `scripts/align_jsonl_conllu.py`

Algorithm (greedy, text-based):
1. Read JSONL records sequentially.
2. For each record, normalize the `"pangyo"` text (NFC + whitespace collapse).
3. Read CoNLL-U sentences sequentially. For each sentence, normalize its `# text` comment.
4. Greedily accumulate sentences: join their normalized texts with `" "` until the
   accumulated string equals the normalized `"pangyo"` text.
5. On exact match: status = `exact`. On close match (SequenceMatcher ratio ≥ 0.95):
   status = `high_similarity`. If ratio < 0.85: status = `failed`.
6. Proceed to next JSONL record.

The alignment map is saved to `data/processed/panyo_jsonl_doc_sentence_map.csv`.

### Safety Threshold
Document-level split is only generated if:
- **≥98%** of JSONL records are aligned successfully (`exact` or `high_similarity`)
- **≥98%** of CoNLL-U sentences are assigned to a document

Results are reported in:
- `results/jsonl_conllu_alignment_report.csv` (per-record alignment quality)
- `results/jsonl_conllu_alignment_summary.md` (overall coverage and safety decision)

### Split Ratios and Seed
- Split target: **80% train / 10% dev / 10% test by token count** (not document count).
- Random seed: **42** (same as Step 1 for reproducibility).
- Documents are shuffled at the document level; sentences within each document remain
  in their original order.

### Token-Based Split Assignment
Documents are assigned to splits greedily based on cumulative token count:
1. Shuffle all documents with seed 42.
2. Assign documents to `train` until 80% of total tokens are allocated.
3. Assign documents to `dev` until 90% of total tokens are allocated.
4. Remaining documents go to `test`.

This approximates a 80/10/10 token split; the actual ratios may vary slightly depending
on document size distribution.

### Output Comments
Each sentence in the docsplit files has two new comment lines prepended:
```
# doc_id = jsonl_<N>
# split  = train|dev|test
```
The original `# sent_id` and `# text` comments are preserved immediately after.

The `docsplit_full.conllu` file has only `# doc_id = jsonl_<N>` prepended (no `# split`).

### Limitation
- Documents (JSONL records) are independent paragraphs. They were originally sourced
  from Korean Wikipedia articles and code-switched algorithmically. Documents from the
  same Wikipedia article may therefore appear in different splits.
- The alignment assumes that sentences within a document appear in exactly the same
  order in the CoNLL-U file as they do in the JSONL `"pangyo"` text. Any reordering
  during corpus construction would break the alignment.
- A small number of records may fail alignment if there are whitespace or punctuation
  differences between the JSONL text and the CoNLL-U `# text` comments. These are
  reported in `results/jsonl_conllu_alignment_report.csv`.

---

## Comparison of Split Strategies

| Property | Sentence-level | Document-level |
|---|---|---|
| Unit of shuffling | Sentence | Document (JSONL record) |
| Within-paragraph leakage | Possible | Prevented |
| Reproducible | Yes (seed 42) | Yes (seed 42) |
| Requires alignment | No | Yes |
| Token ratio | ~80/10/10 | ~80/10/10 (by tokens) |
| Split files | `panyo_original_adp_*.conllu` | `panyo_original_adp_docsplit_*.conllu` |

---

## Final Recommendation

Use `panyo_original_adp_docsplit_{train,dev,test}.conllu` for all main experiments,
provided that:
1. `results/jsonl_conllu_alignment_summary.md` reports "Proceed to document-level split: YES".
2. `scripts/validate_docsplit.py` exits with code 0 (all checks pass).

The sentence-level split files are retained for:
- Comparison experiments
- Ablation studies on split strategy effects
- Sanity-checking model behavior
