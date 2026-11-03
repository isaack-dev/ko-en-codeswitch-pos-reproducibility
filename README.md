# Korean-English code-switching POS tagging

Code and aggregate results for **A Baseline and Error Analysis Study of POS Tagging for Korean-English Code-Switching Text**, accepted to AACL-IJCNLP 2026 Student Research Workshop.

This is a minimal code-and-aggregate release prepared independently of the private development repository. It contains no corpus payloads, private annotations, predictions or lexical checkpoints. The reviewed camera-ready code-and-aggregate release is publicly available at https://github.com/isaack-dev/ko-en-codeswitch-pos-reproducibility. [Exact inventory](docs/public_release/PUBLISH_FILES.txt), [reproduction boundaries](docs/public_release/REPRODUCIBILITY.md), [executed checks](docs/public_release/VALIDATION.md) and [release readiness](docs/public_release/READINESS.md).

## Evaluation settings

The preserved models are global and language-aware majority baselines, joint and language-routed averaged perceptrons, and corrected Korean KAIST / English GUM UD transfer. Training uses five epochs, seed **42**, supplied language IDs and greedy decoding on immutable document-level splits.

- **Silver benchmark:** agreement with canonical automatically generated annotations; near-perfect scores do not establish independent linguistic correctness.
- **Historical v1.1:** 60 representative TEST sentences / 1,063 tokens, kept separate from 20 diagnostic sentences. Independent annotation used v1.1; a provisional v1.2 draft was used for historical adjudication/consistency review. These are contextual results, not the new prospective reference.
- **Primary v1.2:** 180 TEST sentences from distinct documents / 3,186 resolved tokens, under the [frozen analysis plan](reports/analysis_plans/pangyo_v1.2/PANGYO_V12_ANALYSIS_PLAN_FREEZE_2026-10-01.md). Equal quotas in five length strata are not a self-weighting sample of all test tokens. Paragraph-derived documents do not establish article independence. Local freeze records do not independently authenticate timing or blinding.
- **A/B/C lexical ablations:** additional post-review descriptive silver-reference diagnostics. They are outside the frozen v1.2 human comparisons; no human scores are supplied for the new models. Frozen mappings were not optimized on the new human labels.

Results and source hashes are in `results/public_camera_ready/`. The 21 reviewed files contain aggregate counts, metrics and configurations, without token text, source identities or annotator notes. Historical confidence intervals are preserved only in their v1.1 context; no new significance or population-generalization claims are made.

## Installation and lightweight validation

Python **3.12.10** was tested. Baseline training, prediction and integrity checks use the standard library. Human/statistical tools additionally use NumPy and openpyxl. No dependency installation or full research retraining was performed during this release preparation.

```bash
python -m venv .venv
# Activate .venv with your shell's usual command.
python -m pip install -r requirements-reproduction.txt
python -B scripts/public_release/test_release.py
python -B scripts/public_release/fixture_checks.py
python -B scripts/public_release/verify_release.py
```

Fixture tests operate on synthetic text in temporary directories. Passing them and verifying hashes is not full experiment reproduction. Do not run historical integration suites without their private inputs and isolation; several assume the original project layout or write to default experiment directories.

## Training and silver evaluation with separately authorized inputs

Obtain the **exact frozen split bytes** and the matching external UD training files. A newly downloaded upstream JSONL or newly generated split cannot substitute for them. Hosted ChatGPT web silver generation cannot be reproduced exactly from the preserved materials.

```bash
python -B scripts/public_release/verify_release.py --data-root data/processed
python -B scripts/public_release/run_baselines.py train --train data/processed/panyo_original_adp_docsplit_train.conllu --gum data/en_gum-ud-train.conllu --kaist data/ko_kaist-ud-train.conllu --output outputs/local_reproduction/training
python -B scripts/public_release/run_baselines.py predict --dev data/processed/panyo_original_adp_docsplit_dev.conllu --test data/processed/panyo_original_adp_docsplit_test.conllu --models outputs/local_reproduction/training/models --output outputs/local_reproduction/predictions
```

These commands require new/empty output directories. For staged `data/splits/`, pass those actual paths and `--data-root data/splits`; the explicit-path adapter preserves the original models. Training creates eight checkpoints for five baseline systems. Prediction also runs silver evaluation. Full research training was not rerun here. Human evaluation requires private aligned references, operative guidelines and frozen predictions; public aggregates alone cannot reproduce it. See [data access](data/README.md).

## Licensing and citation

The existing [MIT license](LICENSE) covers project software; it does not license third-party corpora or confer annotation consent. Upstream [PangyoCorpora](https://github.com/MrBananaHuman/PangyoCorpora#license) states MIT licensing, without resolving privacy/rights for every passage or derived record. UD r2.16 [English GUM](https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/r2.16/README.md) lists CC BY-NC-SA 4.0; [Korean KAIST](https://raw.githubusercontent.com/UniversalDependencies/UD_Korean-Kaist/r2.16/README.md) lists CC BY-SA 4.0. These checked versions are not asserted to be the exact original training release. External resources remain subject to their own terms and attribution.

No corpus, checkpoints, workbooks, calibration/adjudication records, manuscripts, reviewer communications or private audit logs are distributed. `.gitignore` uses an exact release boundary; new files require review rather than broad CSV/JSON exclusion. Use the paper title and venue/year when citing this work; a verified proceedings identifier will be added when available.
