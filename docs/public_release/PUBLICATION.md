# Publication controls

Prepared as an independent directory with `git init`, an explicit empty template and no imported history. No original Git directory, objects, references, LFS payloads, alternates, remotes, hooks, worktrees or commit identity are copied. Preparation creates no commit, tag, index entries or remote. The requested initial root commit remains pending explicit approval, because the task also explicitly forbids committing without approval.

The exact included files are in PUBLISH_FILES.txt and covered by SHA256SUMS (except the checksum file itself). Unknown new files are ignored by default. `.gitignore` does not replace content review: never force-add data, models, human records, private output logs or archives. Do not import any prior repository history through fetch, merge, cherry-pick, bundles, alternates, mirroring or LFS.

Proposed repository: https://github.com/isaack-dev/ko-en-codeswitch-pos-reproducibility . Availability is not confirmed or reserved; an authenticated owner check is required. No remote repository was created or queried with privileged write access.

## Commands AFTER explicit publication/commit approval

Run from this clean release directory only. Configure your verified public commit identity locally first; do not reuse a private global email automatically.

```powershell
git config --local user.name "<approved public author name>"
git config --local user.email "<your GitHub-provided noreply address>"
git status --short
python -B scripts/public_release/test_release.py
python -B scripts/public_release/fixture_checks.py
python -B scripts/public_release/verify_release.py
git --literal-pathspecs add --pathspec-from-file=docs/public_release/PUBLISH_FILES.txt
git diff --cached --check
git diff --cached --stat
git ls-files
# Review exactly these staged paths and contents before continuing.
git commit -m "Prepare camera-ready reproducibility release"
git rev-list --count --all
git rev-list --max-parents=0 --all
git show -s --format=%P HEAD
# Expect one commit, one root and an empty parent line.
gh auth status
gh repo view isaack-dev/ko-en-codeswitch-pos-reproducibility
# Check that any failure means name absence, not authentication/network failure.
# Only after that check and explicit remote-publication approval:
gh repo create isaack-dev/ko-en-codeswitch-pos-reproducibility --public --source . --remote origin
git push -u origin main
git tag -a aacl-srw-2026-camera-ready -m "AACL-IJCNLP 2026 SRW camera-ready code and aggregate results"
git push origin aacl-srw-2026-camera-ready
```

No release package or GitHub release is automatically published. After push, clone the new repository into another empty directory and repeat fixture/integrity checks. Do not replace a permissions error with an automatic alternative repository name.

Historical exposure in the previous remote is a separate maintainer remediation issue. A new repository cannot retract files from earlier commits, caches or forks. No old repository changes or history rewriting are authorized here.
