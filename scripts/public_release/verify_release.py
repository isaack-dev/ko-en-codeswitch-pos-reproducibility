"""Read-only verification of public aggregate copies and optional frozen inputs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def verify(manifest, data_root=None):
    failures = []
    for entry in manifest["artifacts"]:
        path = ROOT/entry["path"]
        if not path.is_file() or sha(path) != entry["sha256"]:
            failures.append(entry["path"])
    inputs = 0
    if data_root is not None:
        frozen = json.loads((ROOT/"results/public_camera_ready/split_checksums.json").read_text())
        for partition in frozen["partitions"].values():
            path = data_root / Path(partition["path"]).name
            inputs += 1
            if not path.is_file() or sha(path) != partition["sha256"]:
                failures.append(str(path))
    if failures:
        raise ValueError("Missing or checksum-mismatched files: " + ", ".join(failures))
    return {"aggregate_files_verified": len(manifest["artifacts"]),
            "frozen_split_files_verified": inputs,
            "scope": "integrity only; not model or annotation reproduction"}


def verify_candidate_files():
    entries = (ROOT/"docs/public_release/SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for line in entries:
        expected, name = line.split("  ", 1)
        path = ROOT/name
        if not path.is_file() or sha(path) != expected:
            raise ValueError("Candidate file checksum mismatch: " + name)
    return len(entries)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, help="Directory containing the three frozen docsplit files (processed/ or splits/)")
    a = p.parse_args()
    manifest = json.loads((ROOT/"results/public_camera_ready/provenance.json").read_text())
    result = verify(manifest, a.data_root)
    result["candidate_files_verified"] = verify_candidate_files()
    print(json.dumps(result, sort_keys=True))
