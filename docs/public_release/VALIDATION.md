# Executed standalone-release checks (2026-10-10)

Executed against this independent directory, using Python 3.12.10 on Windows, NumPy 2.4.4 and openpyxl 3.1.5 already installed. No installations, full research retraining or human model evaluation were performed.

| Actual check | Result and scope |
|---|---|
| `python -B scripts/public_release/test_release.py` | 3/3 passed. Synthetic two-token end-to-end training of eight checkpoints, prediction/evaluation for five systems on two fixture partitions; explicit staged paths; existing-output guard; missing/corrupt input rejection |
| `python -B scripts/public_release/fixture_checks.py` | 20/20 passed. Ten v1.2 metric/algebraic-coverage fixtures, nine lexical/mapping/determinism/source-mutation fixtures and one real-token prediction-reader fixture; no private corpus required |
| In-memory parse and compilation of every retained Python module | 56/56 passed; syntax only |
| Public aggregate provenance | 21/21 current source hashes matched. Twenty exact source copies; the pre-adjudication summary retains verified aggregate values only, as explicitly recorded in provenance |
| Public release integrity verifier | 21 aggregate hashes and 95 release-file hashes matched; integrity only. This copy does not contain the three private split payloads |
| Per-file content and structured aggregate review | All 116 starting candidates reviewed; 96 final files rescanned. No recognized secret, personal contact or tax identifier matches. Two pattern findings are scanner regex vocabulary itself, not real machine paths or client data |
| Frozen corpus sentence screening | No exact source-sentence matches in reviewed literal/line strings at 12+ characters or full final text at 20+ characters. Short generic lexical rules/fixtures remain; no corpus records, token-trace output or source identities are distributed |
| Positive inventory / ignore boundary | Exactly 96 unstaged, nonignored candidate files; all 11 prohibited/unreviewed path probes ignored. No symlinks or other payload files |
| Git-filter integrity | All 96 filtered versus raw object hashes equal; `.gitattributes` preserves exact file bytes |
| Independent Git initialization | Unborn `main`; zero commits, references to old ancestry, objects, index entries or remotes; no LFS, hooks, alternates or shared-worktree metadata |
| Original-project preservation | 1,245 original working/Git administrative files verified unchanged; no additions or removals in the snapshot scope. Vendor/toolchain/cache directories and Git object storage were not copied or operated on |

The initial sandbox-user Git scan encountered Windows ownership rejection. It was rerun successfully under the owning user; no global `safe.directory` or Git identity setting was changed. Raw execution logs and original snapshots are retained outside this payload as private maintainer records.

## Not executed or unavailable

Full frozen-corpus retraining, full ablation retraining, fresh human integration, exact hosted silver generation, full-checkpoint replay, a post-commit clone test and remote publication were not performed here. The 20-fixture runner explicitly excludes private-data partition/preservation tests and entire integration suites; their omission is not a pass. The previous original-project audit included broader and bounded real-data checks; those historical checks are not presented as executions of this new public checkout.

Retained historical entry points assume private inputs and original default output paths. Do not run them into authoritative directories. Optional SciPy cross-checks and PDF scanner branches were not exercised. Hash verification is not full reproduction, semantic annotation validation, independent timestamp authentication or legal clearance.
