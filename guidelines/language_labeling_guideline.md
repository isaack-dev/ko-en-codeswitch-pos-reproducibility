# Language Labeling Guideline

## Overview

Each real token in the Korean-English Panyo corpus is labeled with a token-level language tag
using automatic Unicode-based heuristics. Labels are stored in the CoNLL-U MISC column as
`Lang=XX` and are added by `scripts/label_language.py`.

> **Note**: This is automatic heuristic labeling. It is based solely on which Unicode scripts
> appear in the token surface form. A subset of the corpus may be manually audited in later
> stages to correct systematic errors.

## Label Set

| Label | Meaning |
|---|---|
| `Lang=KO` | Token contains Hangul characters and no Latin letters |
| `Lang=EN` | Token contains Latin letters and no Hangul characters |
| `Lang=MIXED` | Token contains both Hangul and Latin characters |
| `Lang=OTHER` | Token contains neither Hangul nor Latin (punctuation, digits, symbols) |

## Heuristic Rules

A token's form is scanned character-by-character:

### Hangul detection — any character in these Unicode ranges:

| Range | Block name |
|---|---|
| AC00–D7A3 | Hangul Syllables (precomposed) |
| 1100–11FF | Hangul Jamo |
| 3130–318F | Hangul Compatibility Jamo |
| A960–A97F | Hangul Jamo Extended-A |
| D7B0–D7FF | Hangul Jamo Extended-B |

### Latin detection — any character in [A-Z] or [a-z]

### Classification logic:

```
has_hangul AND has_latin  →  MIXED
has_hangul only           →  KO
has_latin only            →  EN
neither                   →  OTHER
```

## Examples

| Token | Label | Reason |
|---|---|---|
| `는` | `KO` | Hangul only |
| `전세계의` | `KO` | Hangul only |
| `206개` | `KO` | `개` is Hangul; `206` digits are neither Hangul nor Latin, but Hangul is present |
| `하고` | `KO` | Hangul only |
| `있습니다` | `KO` | Hangul only |
| `This` | `EN` | Latin only |
| `document` | `EN` | Latin only |
| `countries` | `EN` | Latin only |
| `Python으로` | `MIXED` | `Python` = Latin, `으로` = Hangul |
| `debug하는` | `MIXED` | `debug` = Latin, `하는` = Hangul |
| `push했다` | `MIXED` | `push` = Latin, `했다` = Hangul |
| `commit을` | `MIXED` | `commit` = Latin, `을` = Hangul |
| `.` | `OTHER` | Punctuation; neither Hangul nor Latin |
| `,` | `OTHER` | Punctuation |
| `(` | `OTHER` | Punctuation |
| `206` | `OTHER` | Pure digits |
| `3.14` | `OTHER` | Digits and period |
| `'` | `OTHER` | Quotation mark |

## Edge Cases

- **Numbers with Korean counter words** (`206개`, `5개월`): labeled `KO` because the Hangul
  counter word is present. The digit portion does not change the classification.
- **Pure numbers** (`206`, `2023`): labeled `OTHER` (no Hangul, no Latin).
- **Punctuation**: always `OTHER` unless the punctuation character itself is in a Hangul or
  Latin code point range (which standard punctuation is not).
- **Abbreviations and acronyms in Korean text** (`DSP`, `URL`, `AI`): labeled `EN` because
  they are pure Latin. Semantically they may originate in English or be borrowed, but the
  heuristic is script-based.
- **Japanese or Chinese characters** (no Hangul, no Latin): labeled `OTHER`. If such
  characters appear in the corpus they are treated as script-neutral tokens.
- **Emoji and special symbols**: labeled `OTHER`.

## MISC Column Behavior

The script modifies the CoNLL-U MISC field (column 10, 0-indexed as field 9):

| Original MISC | Result |
|---|---|
| `_` | `Lang=XX` |
| `SpaceAfter=No` | `SpaceAfter=No\|Lang=XX` |
| `SpaceAfter=No\|Translit=...` | `SpaceAfter=No\|Translit=...\|Lang=XX` |

Any existing `Lang=` value is replaced (not duplicated) if the script is run again.

## Code-Switching Boundary Definition (Step 1)

A **code-switching boundary** is counted when two directly adjacent real tokens in the same
sentence have labels `KO` and `EN` (in either order).

### Counted as boundary:
- `KO → EN` ✓
- `EN → KO` ✓

### Not counted as boundary in Step 1:
- `KO → OTHER` — OTHER does not trigger a boundary
- `OTHER → EN` — OTHER does not trigger a boundary
- `KO → MIXED` — MIXED is not KO or EN
- `MIXED → EN` — MIXED is not KO or EN
- `KO → OTHER → EN` — not counted because OTHER breaks the adjacency

### MIXED tokens:
MIXED tokens are counted separately and reported in `results/boundary_statistics.csv`.
They are not included in the boundary count because it is ambiguous whether they represent
a switch point or a within-token mixing event.

### Limitation:
The simple adjacent-only boundary definition undercounts boundaries when `OTHER` tokens
(punctuation, digits) appear between language-switching segments. A version that skips OTHER
tokens when tracing language context may be implemented in a later step. This limitation is
documented in `guidelines/annotation_decisions.md`.
