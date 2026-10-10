# Pangyo Human UPOS Evaluation Extension
## Frozen Analysis Plan — Legacy v1.1 vs. Prospective v1.2 and Length-Stratified Reporting

**Protocol ID:** PANGYO-CR-HUMAN-V12-ANALYSIS-001  
**Version/status:** 1.0 — FROZEN (researcher-confirmed analysis choices)  
**Freeze date:** 2026-10-01 (America/Chicago)  
**Study:** Korean–English code-switching UPOS tagging, AACL-IJCNLP 2026 SRW camera-ready human-reference extension  
**Applicability:** Frozen main TEST Batch A and Batch B; calibration is a separate DEV procedure.  
**Timing restriction:** This plan must remain unchanged after access to main-extension human labels, corresponding silver labels, or model correctness/performance for the selected 180 TEST sentences. This document records the two choices confirmed by the researcher on the freeze date; it does not establish a third-party preregistration timestamp.

### 1. Researcher-confirmed, binding decisions

**D1 — Do not pool annotation guideline versions.** Keep the previously collected *v1.1* independent annotations of 60 historical representative TEST sentences as a historical analysis. Analyze the new *v1.2* main extension of 180 TEST sentences separately. Never merge them into a single human reference, silently harmonize labels across versions, or treat numerical differences between them as a causal effect of the guideline revision. Preserve any historical diagnostic subset separately from both representative analyses.

**D2 — Primary estimand and reporting.** For the new v1.2 extension, the primary performance measure is **token-level micro accuracy over all scored tokens from the combined 180-sentence A+B sample**, including PUNCT. Provide the five pre-specified sentence-length-stratum accuracies and their **equal-weight (20% each) mean** as secondary analyses, along with non-PUNCT and UPOS-specific diagnostics. The primary micro result describes the **sample's observed token mix**; it does **not** claim to be a design-unbiased or population-weighted estimate for the full Pangyo TEST corpus.

These choices apply consistently to each prespecified comparison with the same available reference, including frozen silver vs. adjudicated human reference. Comparisons involving previously frozen model predictions are eligible only if all sentence/token IDs, token order, and tag inventory can be reconciled without altering or fitting the models using these human references.

### 2. Frozen samples, selection provenance and partition boundaries

The prospective v1.2 main sample comprises **180 distinct TEST sentences from 180 distinct source documents**, selected without using silver/human UPOS labels, predictions, correctness, error patterns or language proportions. Selection used a seeded constrained document-matching algorithm (seed **20261001**), fixed exclusion criteria, and equal *sentence* allocation across five length strata. The existing v1.1 60-sentence representative sample has 12 sentences per stratum; it is historical, not part of the new v1.2 estimand.

| Original tokens per sentence | Batch A sentences / tokens | Batch B sentences / tokens | Combined sentences / tokens |
|:--|--:|--:|--:|
| Stratum 1: 1–12 | 12 / 123 | 24 / 234 | 36 / 357 |
| Stratum 2: 13–15 | 12 / 163 | 24 / 340 | 36 / 503 |
| Stratum 3: 16–18 | 12 / 203 | 24 / 412 | 36 / 615 |
| Stratum 4: 19–22 | 12 / 243 | 24 / 488 | 36 / 731 |
| Stratum 5: 23+ | 12 / 314 | 24 / 666 | 36 / 980 |
| **Total** | **60 / 1,046** | **120 / 2,140** | **180 / 3,186** |

Use the fixed source `sampling_manifest.csv` for sample identity, `sampling_design.json` for methodology, and `selection_validation.json` for sample/workbook audit. Neither dev-calibration Pilot (12 DEV sentences / 218 tokens) nor Confirmation (16 DEV sentences / 315 tokens) enters any main TEST accuracy numerator or denominator. Batch A and Batch B are an operational annotation split, **not** independently selected statistical populations; the prespecified primary result pools A+B *within v1.2 only*. Optional A-only/B-only descriptives must be labeled diagnostics and must not supersede the combined result.

### 3. Human-reference construction and blinding

Two annotators independently label the *same original frozen tokens*, including punctuation, using guideline **v1.2**. They receive annotator-specific A and B workbooks and may not discuss labels or inspect the silver key/model predictions until both have completed **both** batches and the four independent original workbooks have been archived with checksums.

1. Verify every completed original against the manifest by `sample_id`, source document/sentence IDs, ordinal, original token ID/form and order. Flag missing responses, invalid UPOS, incomplete rationale requirements, and source alteration before computing reliability.
2. Compute and archive **pre-adjudication** token-level raw agreement, nominal unweighted Cohen's kappa over the frozen 17-tag inventory, disagreement/confusion counts, and non-PUNCT agreement. For this *human–human* non-PUNCT diagnostic, exclude any token labeled PUNCT by **either** annotator; report its denominator explicitly.
3. Only after those files and statistics are fixed, hold *blind* disagreement adjudication using v1.2, without silver labels or model predictions. Record the original two labels, discussion, final consensus, rationale and applicable guideline clauses. Audit any agreed adjacent tokens in the full neighboring multiword expression. If no defensible consensus is reached, record **UNRESOLVED**; never silently force agreement.
4. Freeze a distinct adjudicated v1.2 human-reference file and decision log. Never overwrite the two independent submissions or replace pre-adjudication IAA with post-consensus agreement.

Pilot was concluded after documented joint review. On an *independent* 16-sentence DEV Confirmation, pre-adjudication agreement was 311/315 = 98.73% and kappa = 0.9852, meeting the frozen guideline's operational release thresholds of 90% and 0.80; the DEV observations are not part of the TEST extension. Calibration figures serve only as readiness/procedure context.

### 4. Primary v1.2 outcome and fixed metric definitions

**Primary comparison:** evaluate each available frozen UPOS label source of interest against the *adjudicated v1.2 human reference* at the exact original token level. In particular, compute canonical/frozen **silver vs. adjudicated-human token accuracy**; compare preexisting model outputs using this same reference only where independently saved, provenance-verified and token-aligned outputs exist. No new training, threshold tuning, model selection, rule changes, or reference-based correction is permitted on these TEST sentences for this analysis.

For source `m` and a set `E` of *scorable* v1.2 original TEST tokens after blinded adjudication, define:

- **Primary token-level micro accuracy:** `Acc_micro(m) = (# tokens i in E with prediction_m(i) = human_v1.2(i)) / |E|`. Include PUNCT. When every item is resolved/aligned, `|E| = 3,186`.
- **Stratum accuracy:** for each fixed original-length stratum `s = 1,...,5`, `Acc_s(m) = (# correct scorable tokens in s) / (# scorable tokens in s)`.
- **Equal-stratum secondary mean:** `Acc_equal_strata(m) = (Acc_1 + Acc_2 + Acc_3 + Acc_4 + Acc_5) / 5` (each stratum contributes 20%, irrespective of token count).
- **Non-PUNCT diagnostic accuracy:** score only original tokens whose **adjudicated human reference** is not PUNCT; include all remaining predictions, including a mistaken PUNCT prediction. Report correct/eligible counts and the accuracy.
- **UPOS diagnostics:** per-class support, precision, recall and F1 on the adjudicated reference; confusion matrix; macro F1 as the unweighted mean over labels with positive reference support, and support-weighted F1 over those labels. Use F1 = 0 for a supported label with zero predicted instances, and disclose labels absent from the reference. Label-specific metrics supplement, rather than replace, accuracy.

The *same scorable-token set and original fixed strata* must be used for paired comparisons of two systems; if one system lacks aligned predictions for a token, resolve the alignment problem or declare the comparison unavailable rather than quietly changing its denominator. Record exact numerators/denominators, not just percentages. Any language-specific (KO/EN/OTHER), boundary or tokenization-flag analysis is supplemental and conditional on a **preexisting, independently defined, provenance-checked grouping variable**; do not infer groups from observed errors after unblinding.

**Unresolved cases:** Exclude any UNRESOLVED human-reference token from the primary scored denominator and every system being compared, disclose the count overall and in each stratum, and report reference coverage `|E| / 3,186`. If there are unresolved tokens, provide a transparent full-sample sensitivity range for accuracy, keeping denominator 3,186 and treating every unresolved token first as incorrect and then as correct (an algebraic bound, not a confidence interval). Missing model output is an alignment/provenance failure, **not** an UNRESOLVED annotation.

### 5. Interpretation and analysis exclusions

The design fixes **36 sentences in each length stratum**, not an equal number of tokens; the micro result therefore weights longer sentences more heavily because they contribute more original tokens. The equal-stratum mean answers a distinct descriptive question in which each length band receives exactly one-fifth of the weight. Always report them side by side and never relabel either as a representative estimate of Pangyo TEST population accuracy. The extraction frame counts and one-sentence-per-document constraint do not establish simple token-level population inclusion probabilities; **no post-stratification or inverse-probability population weighting** is authorized by this freeze.

The legacy v1.1 60-sentence representative results remain reproducible under their *original* annotation version, reference construction, scoring, exclusions and provenance as previously documented. Do not rerun them under v1.2 without a separate, prospectively approved study. A clearly separated historical contextual comparison table may display the two version-specific results, with sample sizes, datasets, guidelines and denominators visible. It must not present their difference as a like-for-like benchmark improvement, a treatment effect or a statistical test of guideline change.

No new confirmatory significance test, bootstrap confidence interval, population-generalization claim, or model-winner claim is specified by these two researcher selections. If such an analysis is wanted, document its comparison family, resampling unit (sentences/documents, not independent tokens), seed, replication count and multiplicity approach in a *dated addendum before examining the relevant main TEST outcomes*. Absent that addendum, results are descriptive; use existing frozen paper-level statistical analyses only under their original stated populations and hypotheses.

### 6. Reporting outputs and decision-gate checklist

**Required outputs:** (a) source/workbook integrity and freeze manifests; (b) both untouched original annotator files for both batches; (c) pre-adjudication overall/non-PUNCT IAA, Cohen's kappa, confusion list and confidence/issue diagnostics; (d) blind adjudication log and post-adjudication adjacent-token audit; (e) v1.2 reference with unresolved/coverage reporting; (f) silver-vs-human primary token micro accuracy with exact counts; (g) fixed five-stratum table and equal-stratum mean; (h) non-PUNCT and UPOS-specific diagnostics; (i) strictly separate v1.1 historical results with their original provenance; and (j) source code, calculation command, tool versions and reproducibility checksums.

**Before accessing MAIN TEST correctness:** archive this file, preserve its SHA-256 in the repository or research log, and retain the sampling and blank-workbook source checksums. Verify that neither annotator has seen silver labels or model outputs. **After both complete A+B:** source-validate, preserve raw submissions and calculate raw IAA *before* blind adjudication. **Only after blind consensus and reference freeze:** open silver/model keys for the fixed comparisons and record the numerical outcomes. Any deviation requires a dated, separate amendment stating the reason, original plan, revised plan, and whether any relevant outcomes had already been observed; retain this frozen original unchanged.

### 7. Frozen source register

| Artifact | SHA-256 |
|:--|:--|
| `sampling_manifest.csv` | `831e288d9bcf47c5073b902271f3f0a21c5e9a8a7604d6aa4a0b653c6d143b4d` |
| `sampling_design.json` | `264a4e61fc18a871911ac97d1508538ce9c80d95d600a355af81508e71148de2` |
| `selection_validation.json` | `51e81ff6d0ad154b784e1449d2a84c7664fc11b9ff2a1bd97b15334adf59fc94` |
| `Pangyo_Human_UPOS_Annotation_Guideline_v1.2.pdf` | `4eba40f9e776adad8af00f0cd1bd44713171a0fcddc9513f43b481d4c478c131` |
| `pilot_pre_adjudication_metrics.json` | `2fe7121b29ee6c1e5eaadd6560ccf809c8c153799dd569c7d0d8e262933470bf` |
| `confirmation_pre_adjudication_metrics.json` | `4bad071ab62001cf69d107c6a618557275232fe253ee820549558ab155309508` |

**Status boundary:** This is a *frozen analysis plan*, not a claim that the 180 TEST annotations have been returned, that blind adjudication has taken place, or that any model–human comparison has been observed. **Researcher-approved choices:** D1 = separate v1.1 from v1.2; D2 = token-level micro primary and length-stratum/equal-stratum secondary. All other implementation details above operationalize those decisions and existing v1.2 workflow, and deviations must be logged prospectively.
