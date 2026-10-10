# Final publication readiness report (2026-10-10)

## Decision

The minimal reviewed payload is prepared for maintainer approval; it is not yet committed or published. An initial root commit is deliberately pending the explicit instruction not to commit without approval. Proposed URL: https://github.com/isaack-dev/ko-en-codeswitch-pos-reproducibility . Name availability remains unverified, not reserved.

## Exact inventory and changes

PUBLISH_FILES.txt lists every included file (96). SHA256SUMS covers 95 files, excluding only itself. Starting inventory: 116 candidates. Retained: 83 unchanged byte copies and nine updated release-specific files. Omitted: 24 candidates, individually enumerated with reasons in EXCLUDED_CANDIDATES.csv. Added: this report, EXCLUDED_CANDIDATES.csv, AGGREGATE_REVIEW.json and scripts/public_release/fixture_checks.py.

The nine updated files are `.gitignore`, `README.md`, `data/README.md`, `docs/public_release/PUBLICATION.md`, `docs/public_release/PUBLISH_FILES.txt`, `docs/public_release/REPRODUCIBILITY.md`, `docs/public_release/SHA256SUMS`, `docs/public_release/VALIDATION.md`, and `scripts/public_release/audit_release.py`. The only utility behavior adjustment lets the privacy scanner handle unborn HEAD and enumerate actual remote refs instead of assuming historical branch names. All model, mapping, evaluation, statistical, ablation and frozen-plan source bytes are unchanged.

The independently reviewed 21 aggregate files and lineage are itemized in AGGREGATE_REVIEW.json and results/public_camera_ready/provenance.json. v1.1, v1.2, silver and post-review results remain separate. The frozen v1.2 plan and camera-ready manuscript's core numbers agree with these authoritative records; no new inference or result rewriting was introduced.

## Excluded categories

- Every original Git object, commit, reference, branch, index, hook, LFS artifact and historical worktree connection.
- Raw JSONL/CoNLL-U corpora, all frozen split payloads and source/document maps; token-bearing predictions, trace examples and per-token error outputs.
- All human annotation/calibration/adjudication workbooks, independent labels, keys, manifests, source IDs, notes and researcher-only archives, including all v3, v1.1, v1.2 and camera-ready sample records.
- Lexical model checkpoints/weights, local backups, archives, notebook outputs, temporary files, logs, credentials, local configuration, machine-specific helper scripts and private audit snapshots.
- Manuscripts/accepted PDFs, reviewer communications, internal correspondence, supplementary ZIPs and unrelated project files.
- The 24 stale/conflicting allowlist candidates: generic annotation/POS guidance, broad legacy environment and scaffold configs/models, sentence-split alternatives and superseded manuscript/diagnostic generators. These were omitted rather than repaired or relabeled as operative science.

The positive `.gitignore` permits exact reviewed paths, rather than blanket research-extension ignores. Future unreviewed CSV/JSON or source files are ignored until their paths and content are audited. There are no privately tracked files in this new repository because nothing is tracked yet; no ignored source files were copied.

## Security and licensing findings

Each candidate was opened, decoded, checksum-recorded and checked for secrets/contact/local-path patterns, corpus sentences, embedded string literals and structured content. All final files are plain text; no binary documents, embedded archive metadata or hidden sheets are present. No credentials, actual personal contact details, source-sentence records or annotator-identifying material were detected. Scanner-pattern self-hits were reviewed. Pattern/manual screening is not proof that every possible indirect identifier is absent.

The 21 aggregates contain counts, metrics, POS/model categories and configurations, without individual annotator distributions, private notes, document IDs or free corpus text. Sparse cells cannot be linked to an individual from retained material; aggregate interpretations still respect small supports. No artifact-specific redistribution restriction was identified for these research summaries. This is a technical disclosure review, not a blanket legal determination about underlying data.

Existing project software licensing is MIT; third-party and human materials are withheld. Upstream Pangyo's README states MIT, while checked UD r2.16 READMEs state GUM CC BY-NC-SA 4.0 and KAIST CC BY-SA 4.0. These are source checks, not proof of every passage's rights or the exact local training version. No third-party text/license is silently reassigned to project MIT. Operative human guideline bundles and source annotations need separate rights/consent review before any broader release.

## Actual validation and remaining limitations

See VALIDATION.md: 23/23 synthetic tests, 56 syntax checks, all 21 aggregate provenance checks, 95 release checksum checks, exact inventory/ignore probes, original preservation and independent empty-Git checks passed. No full research retraining, full ablation replay, human re-evaluation or fresh committed clone reproduction was run here. Checksum and fixture success do not establish end-to-end scientific reproduction.

Missing public resources: exact canonical silver and document splits/map; matching GUM/KAIST training-version pins; eight baseline and twelve ablation checkpoints; frozen predictions; historical/new human workbooks, references, sampling/exclusion sources and operative guideline bundle; private freeze/preservation manifests. No verified checkpoint download or data access promise is fabricated. Upstream JSONL cannot recover missing hosted silver sessions. Historical full pipeline and ablation validators require these resources, so they are inspected implementations, not public one-command reproduction.

## Remaining gates and publication commands

1. Maintainer approves this exact minimal inventory, public author identity and initial root commit.
2. Authenticate as the owner and verify the proposed name is available; authentication/network failure is not evidence of availability.
3. Execute the explicit, inventory-only staging/commit/remote/push/tag sequence in PUBLICATION.md after approval. Never import the old history.
4. After an approved push, clone the new remote independently and rerun fixture/integrity checks. Recommended tag: `aacl-srw-2026-camera-ready`.

The previous repository's exposed history is a separate unresolved issue. Creating this new repository does not remove its private records from the previous public remote or downstream caches/forks. The original project is preserved unchanged; no history remediation, visibility change or destructive cleanup was attempted here.
