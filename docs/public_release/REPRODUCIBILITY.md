# Reproduction boundaries and authoritative artifacts

## Lineage and available aggregates

`results/public_camera_ready/provenance.json` records original source paths and hashes. Copies are byte-identical except the explicitly reduced aggregate agreement summary. Source artifacts remain in the unchanged private project; only reviewed byte copies enter this independently initialized release. `.gitattributes` preserves exact reviewed-source/result bytes rather than applying platform-dependent newline conversion. Candidate SHA-256 records are in `docs/public_release/SHA256SUMS`; the verifier checks them as well as aggregate provenance. These hashes are integrity records, not signed attestations.

| Evidence | Authoritative local source | Reviewed public copy |
|---|---|---|
| Corrected clean silver baselines | `outputs/final_clean_reproducibility_run/baselines/` | `results/public_camera_ready/silver/overall_results.csv` |
| Historical v1.1 representative comparison | `outputs/final_clean_reproducibility_run/human_model_evaluation/model_metrics_representative.csv` | `results/public_camera_ready/v11/model_metrics_representative.csv` |
| Primary v1.2 evaluation | `outputs/camera_ready/v12_integration/` | `results/public_camera_ready/v12/` |
| Frozen primary protocol | `reports/analysis_plans/pangyo_v1.2/PANGYO_V12_ANALYSIS_PLAN_FREEZE_2026-10-01.md` | Same immutable Markdown path |
| Post-review discrepancy/mapping/ablation | `outputs/camera_ready/additional_diagnostics/` | `results/public_camera_ready/postreview/` |
| Frozen split payloads | `data/processed/panyo_original_adp_docsplit_{train,dev,test}.conllu` | Hashes only: `results/public_camera_ready/split_checksums.json` |

TRAIN: 6,579 documents / 45,450 sentences / 788,802 real tokens. DEV: 819 / 5,690 / 98,566. TEST: 835 / 5,701 / 98,529. Never substitute older sentence-level splits. `data/processed/panyo_jsonl_doc_sentence_map.csv` links paragraph-derived documents and sentences; article-level independence is not established. Original token IDs/order are preserved; multiword ranges/empty nodes are not words.

## Implementations and dependencies

| Component | Source | Required resources |
|---|---|---|
| Reconstruction audit | `scripts/reconstruct_pangyo_conllu.py` | Original JSONL and preserved CoNLL-U; does not reproduce missing hosted silver-generation sessions |
| Language labeling / split provenance | `scripts/label_language.py`, `guidelines/data_split_decisions.md` | Original sources and original splitting implementation; use frozen bytes rather than regenerate partitions |
| Real-token I/O | `scripts/read_conllu.py` | CoNLL-U text and MISC language IDs |
| Majority/joint/routed/UD training | `scripts/train_baselines.py`, `scripts/perceptron_tagger.py` | TRAIN, GUM/KAIST; eight checkpoints |
| Prediction / silver metrics | `scripts/predict_baselines.py`, `scripts/evaluate_baselines.py` | DEV/TEST, checkpoints; predictions retain silver UPOS and add MISC PredUPOS |
| Frozen mappings | `scripts/diagnostic_ud_mapping.py::apply_map` | Corrected UD predictions; frozen DEV decision |
| Historical human build/evaluation | `scripts/build_test_only_adjudicated_reference.py`, `scripts/evaluate_models_on_test_only_human_reference.py` | Private independent/adjudication workbooks, keys and predictions |
| Corrected clean orchestration | `scripts/run_final_clean_reproducibility_pipeline.py` | Full original layout and private human sources; default output is authoritative |
| v1.2 audit/evaluation | `scripts/camera_ready/audit_v12_reference.py`, `scripts/camera_ready/evaluate_v12_reference.py` | Private sampling manifest, operative guideline PDF, references and frozen predictions |
| Post-review ablations | `scripts/camera_ready/additional_diagnostics.py`, `scripts/camera_ready/run_lexical_ablation.py`, `scripts/camera_ready/resume_lexical_ablation.py` | Frozen source/config/splits, A/B/C models and predictions |

The original code resolves paths relative to the repo but assumes `data/processed/` and specific private workbook locations. `scripts/public_release/run_baselines.py` supplies explicit input/output paths for the five silver baselines, including staged `data/splits/`, without editing frozen modules. It is not a full human/statistical orchestration replacement.

Baselines/lexical diagnostics use the standard library; statistics require NumPy; workbook audits require openpyxl. Recorded narrow versions are in `requirements-reproduction.txt`. Legacy CLI/configuration scaffolds and broad optional environment files are omitted. The retained narrow requirements cover the preserved statistical/workbook modules; optional SciPy cross-checks and PDF scanning are not dependencies of baseline training.

GUM TRAIN belongs at `data/en_gum-ud-train.conllu`, KAIST at `data/ko_kaist-ud-train.conllu`. Corrected loading assigns constant EN/KO language features. Do not recreate the earlier blank-language variant. The exact external treebank release must be identified by matching bytes; filenames alone are insufficient. No external transformer checkpoint is needed by the implemented baselines.

Eight original checkpoints exist locally in `outputs/final_clean_reproducibility_run/baselines/models/`; twelve ablation checkpoints are in `outputs/camera_ready/additional_diagnostics/models/{A,B,C}/`. No verified public checkpoint download is supplied. Lexical weights expose corpus vocabulary and require separate release review.

The minimal release withholds raw/derived text, canonical annotations, predictions, sample keys and independent human labels. Upstream JSONL does not recreate exact hosted silver annotations. Aggregate CSVs therefore support inspection, not end-to-end reproduction from a fresh public checkout.

The frozen plan Markdown is retained; its researcher-only ZIP is withheld. Operative guideline PDF: `outputs/camera_ready/human_annotation_extension/guidelines/Pangyo_Human_UPOS_Annotation_Guideline_v1.2.pdf`, SHA-256 `4eba40f9e776adad8af00f0cd1bd44713171a0fcddc9513f43b481d4c478c131`. It is not replaced by generic UPOS instructions. Consent/rights clearance is needed before publicly distributing human source records or guideline bundles with examples/metadata.

## Exact mappings and lexical configurations

Conservative: PART→ADP, SYM→PUNCT, X→NOUN; ADJ unchanged. Coarse additionally maps ADJ/INTJ→NOUN. Korean-oriented: coarse for KO; EN X/INTJ→NOUN and SYM→PUNCT, plus ADJ→NOUN with frozen DEV flag true; EN PART unchanged. OTHER X→NOUN and SYM→PUNCT. These prediction-side rules are frozen; do not fit them on human labels. Per-rule gains/losses and interactions are descriptive.

A: original features. B: remove `w=`, `p1=`, `n1=` identities. C: also remove `pre1=`, `pre2=`, `pre3=`, `suf1=`, `suf2=`, `suf3=`. Original updates, averaging, rounding, greedy history, five epochs and seed 42 remain. Neighbor identity removal also removes BOS/EOS values in those features; no replacement boundary feature was introduced. Seen = exact Python Unicode `FORM.lower()` occurs anywhere in TRAIN, across languages; unseen = complement. Non-PUNCT is determined by the evaluation reference, not predictions. B retains character lexical information; ablation does not prove deterministic annotation-rule reproduction. These new models have no frozen v1.2 human scores.

## Safe execution and supplementary limitations

Run README training/prediction commands only with authorized exact inputs and fresh output directories. Verify split hashes first. Adapter smoke tests train only tiny synthetic fixtures, not the research corpus. Full integration/ablation entry points write to authoritative default directories: fresh reproduction needs a separately approved isolated copy with private inputs, frozen code and original layout. Never clear existing outputs to force a run.

Historical ablation validation requires the recorded numeric CSV-count adapter. Its source embeds a local-machine command and is excluded from minimal staging; inspect privately and create a separately documented portable full-validation wrapper before claiming public full replay. Unit/fixture checks are available without it.

The anonymous supplementary ZIP is not an approved public package: split paths conflict with hardcoded paths, checkpoints are omitted, fresh `--verify-only` is not a training substitute, anonymization damaged one test fixture, and v1.2/ablation materials are absent. It was not modified/repacked in this task.

Checksums establish integrity, not experiment reproduction. Bounded replay establishes only the tested subset. Prior clean-run/full-replay reports are historical records, not new full experiments executed here.


## Release boundary and omitted scaffolds

Only the exact release inventory is retained. Generic annotation/mapping guidance, CRF placeholders, sklearn scaffold models, generic split configurations and obsolete manuscript generators are omitted. See EXCLUDED_CANDIDATES.csv. Historical source docstrings may cite omitted private working notes; those references are provenance, not publicly executable inputs or operative annotation instructions. The retained script-level language and document-split notes are historical provenance; supplied frozen split bytes are authoritative. No full fresh-checkout reproduction is claimed. Model/scientific code remains byte-identical to the audited source; only publication documentation and the standalone audit utility are adapted.
