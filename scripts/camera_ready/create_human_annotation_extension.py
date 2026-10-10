"""Freeze a label-blind, document-disjoint human annotation extension.

Run with Python 3.12 and openpyxl (read-only validation); workbook authoring
uses the bundled @oai/artifact-tool through build_annotation_workbooks.mjs.
--verify-only never rewrites the frozen outputs. Never run this verification
on completed annotations: it deliberately requires initially blank fields.
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import hashlib
import io
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import openpyxl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/camera_ready/human_annotation_extension"
SEED = 20261001
VERSION = "cr-human-extension-v1"
EDGES = (12, 15, 18, 22)
GUIDELINE = "v1.2"
HEADER = ("Set", "Sample_ID", "Source_Sent_ID", "Document_ID", "Sentence",
          "Token_ID", "Token", "Human_UPOS", "Confidence", "Tokenization_Issue", "Note")
UPOS = "ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X".split()
MANIFEST_FIELDS = ("batch", "stratum", "sample_id", "source_document_id", "source_sentence_id",
                   "sentence", "original_token_count", "token_ordinal", "original_token_id", "token",
                   "normalized_text_sha256", "token_sequence_sha256", "selection_seed",
                   "algorithm_version", "guideline_version", "test_file_sha256",
                   "original_sampling_manifest_sha256")
SPLITS = {s: ROOT / f"data/processed/panyo_original_adp_docsplit_{s}.conllu"
          for s in ("train", "dev", "test")}
DOC_MAP = ROOT / "data/processed/panyo_jsonl_doc_sentence_map.csv"
OLD_MANIFEST = ROOT / "outputs/human_validation/test_only/key/selection_manifest.json"
OLD_AUDIT = ROOT / "outputs/human_validation/split_audit/sentence_split_membership.csv"
ANNOT = ROOT / "outputs/human_validation/test_only/annotations/completed"
PREV = ROOT / "outputs/human_validation/annotations"
GUIDES = ROOT / "outputs/human_validation/guideline"
XMLNS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def stratum(length):
    return next((i for i, edge in enumerate(EDGES, 1) if length <= edge), 5)


def serialized(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def csv_bytes(rows, fields):
    f = io.StringIO(newline="")
    writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return f.getvalue().encode("utf-8")


@dataclass(frozen=True)
class Sentence:
    sid: str
    doc: str
    text: str
    tokens: tuple[tuple[str, str], ...]  # original ID, original FORM; no labels
    errors: tuple[str, ...] = ()

    @property
    def norm(self):
        return normalize(self.text)


def load_split(path):
    """Retain ONLY comment identity/text and columns 1/2. POS/MISC are not used."""
    out, comments, tokens, errors = [], {}, [], []

    def finish():
        if comments or tokens or errors:
            sid, doc, text = (comments.get(k, "") for k in ("sent_id", "doc_id", "text"))
            issues = list(errors)
            if not sid or not doc or not text.strip() or not tokens:
                issues.append("missing_identity_text_or_tokens")
            if any(not form.strip() for _, form in tokens):
                issues.append("empty_token")
            if [i for i, _ in tokens] != [str(i) for i in range(1, len(tokens) + 1)]:
                issues.append("noncontiguous_original_token_ids")
            out.append(Sentence(sid, doc, text, tuple(tokens), tuple(sorted(set(issues)))))

    with path.open(encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            if not line:
                finish()
                comments, tokens, errors = {}, [], []
            elif line.startswith("#"):
                if " = " in line:
                    key, value = line[2:].split(" = ", 1)
                    if key in ("sent_id", "doc_id", "text"):
                        if key in comments:
                            errors.append("duplicate_identity_comment")
                        comments[key] = value
            else:
                fields = line.split("\t")
                if len(fields) != 10:
                    errors.append("malformed_conllu_row")
                elif fields[0].isdigit():
                    tokens.append((fields[0], fields[1]))
        finish()
    return out


def annotation_sources(path, sheets):
    """Read source columns only, never Human_UPOS/confidence/notes."""
    wb = openpyxl.load_workbook(path, data_only=False, read_only=True)
    out = {}
    try:
        for sheet in sheets:
            ws = wb[sheet]
            ws.reset_dimensions()
            it = ws.iter_rows(max_col=7 if path.parent == ANNOT else 6, values_only=True)
            header = next(row for row in it if "Source_Sent_ID" in row and "Token_ID" in row)
            idx = {name: i for i, name in enumerate(header)}
            for row in it:
                if not row[idx["Sample_ID"]]:
                    continue
                sid = row[idx["Source_Sent_ID"]]
                item = out.setdefault(sid, {"sample_id": row[idx["Sample_ID"]],
                    "set": row[idx["Set"]], "text": row[idx["Sentence"]],
                    "doc": row[idx["Document_ID"]] if "Document_ID" in idx else None, "tokens": []})
                require(item["text"] == row[idx["Sentence"]], f"Inconsistent workbook text: {sid}")
                item["tokens"].append((int(row[idx["Token_ID"]]), row[idx["Token"]]))
    finally:
        wb.close()
    return out


def audit_inputs(splits):
    original = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))
    for name, expected in original["authoritative_inputs_sha256"].items():
        require(sha(ROOT / name) == expected, f"Original frozen input changed: {name}")
    current = annotation_sources(ANNOT / "Human_POS_Tagging_test_v1_P1_COMPLETED.xlsx",
                                 ("Representative_60", "Diagnostic_20"))
    require(current == annotation_sources(ANNOT / "Human_POS_Tagging_test_v1_P2_COMPLETED.xlsx",
                 ("Representative_60", "Diagnostic_20")), "P1/P2 source rows differ")
    previous = annotation_sources(PREV / "Human_POS_Tagging_v3_P1.xlsx", ("Representative_40", "Diagnostic_20"))
    require(previous == annotation_sources(PREV / "Human_POS_Tagging_v3_P2.xlsx",
                 ("Representative_40", "Diagnostic_20")), "Earlier P1/P2 source rows differ")
    with OLD_AUDIT.open(encoding="utf-8", newline="") as f:
        membership = {r["source_sent_id"]: r for r in csv.DictReader(f)}
    with DOC_MAP.open(encoding="utf-8", newline="") as f:
        maps = list(csv.DictReader(f))
    require(len({r["sent_id"] for r in maps}) == len(maps), "Ambiguous document map")
    docmap = {r["sent_id"]: (r["doc_id"], int(r["n_tokens"])) for r in maps}
    index = {}
    for split, sentences in splits.items():
        for s in sentences:
            require(s.sid not in index, f"Sentence ID crosses splits: {s.sid}")
            index[s.sid] = (split, s)
    for source in (current, previous):
        for sid, entry in source.items():
            split, s = index[sid]
            require(entry["tokens"] == [(i, form) for i, (_, form) in enumerate(s.tokens, 1)],
                    f"Prior tokens cannot align: {sid}")
            require(entry["text"] == s.text, f"Prior context cannot align: {sid}")
            require(entry["doc"] in (None, s.doc), f"Prior document mismatch: {sid}")
            entry["doc"] = s.doc
            if source is current:
                require(split == "test", "Existing test-only annotation outside test")
            else:
                require(membership[sid]["split"] == split and membership[sid]["document_id"] == s.doc,
                        f"Earlier membership mismatch: {sid}")
    for kind, count, tokcount in (("representative", 60, 1063), ("diagnostic", 20, 816)):
        ids = set(original[f"{kind}_sample"]["sentence_ids"])
        actual = {sid for sid, v in current.items() if v["set"].lower() == kind}
        require(ids == actual and len(ids) == count, f"{kind} manifest mismatch")
        require(sum(len(current[s]["tokens"]) for s in ids) == tokcount, "Original token total differs")
        require({current[s]["doc"] for s in ids} == set(original[f"{kind}_sample"]["document_ids"]),
                "Original document manifest mismatch")
    require(len(previous) == 60 and sum(len(v["tokens"]) for v in previous.values()) == 932,
            "Earlier sample count mismatch")
    # Audit/replay the old code only. Its label-bearing objects do not enter
    # the extension frame or selector. No old run()/writer is ever invoked.
    sys.path.insert(0, str(ROOT / "scripts"))
    import create_test_only_human_evaluation_sample as old
    old_sents, ambiguous = old.load_test_sentences(old.TEST_CONLLU)
    eligible, exclusion, full = old.compute_eligibility(old_sents, ambiguous,
        old.find_doc_id_mismatches(old_sents, old.load_doc_map(old.DOC_MAP_CSV)),
        old.load_previous_sample_docs(old.SPLIT_AUDIT_CSV), old.load_previous_sample_texts(old.PREV_P1_XLSX))
    rng = random.Random(42)
    rep, strata = old.select_representative(eligible, rng)
    diag, info = old.select_diagnostic([s for s in eligible if s.sent_id not in {a.sent_id for a in rep}],
          rng, {s.normalized_text for s in rep}, {s.doc_id for s in rep})
    require(sorted(s.sent_id for s in rep) == original["representative_sample"]["sentence_ids"], "Rep replay failed")
    require(sorted(s.sent_id for s, _ in diag) == original["diagnostic_sample"]["sentence_ids"], "Diag replay failed")
    require(strata == original["representative_sample"]["sampling_strata"], "Original strata differ")
    require(tuple(strata["bin_edges_token_count"]) == EDGES, "Original boundaries differ")
    require(info == original["diagnostic_sample"]["category_info"], "Original categories differ")
    require(dict(collections.Counter(x.reason for x in exclusion)) ==
            original["eligibility_and_exclusion_rules"]["exclusion_counts_by_reason"], "Original exclusions differ")
    doc_splits = collections.defaultdict(set)
    for split, sentences in splits.items():
        for s in sentences:
            doc_splits[s.doc].add(split)
    require(all(len(v) == 1 for v in doc_splits.values()), "Document crosses frozen splits")
    annotated = {**previous, **current}
    require(len(annotated) == len(previous) + len(current), "Annotation rounds overlap")
    prior_docs = collections.Counter(v["doc"] for v in annotated.values())
    require(all(n == 1 for n in prior_docs.values()), "Existing annotation has repeated source documents")
    jsonl_fields = collections.Counter()
    with (ROOT / "data/kor_en_panyo_corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            jsonl_fields.update(json.loads(line).keys())
    train_texts = {s.norm for s in splits["train"]}
    dev_texts = {s.norm for s in splits["dev"]}
    audit = {"original_seed": 42, "original_frame_sentences": len(eligible),
        "original_frame_documents": len({s.doc_id for s in eligible}), "original_full_prior_document_exclusion": full,
        "original_exclusions": dict(collections.Counter(x.reason for x in exclusion)),
        "original_eligible_stratum_counts": dict(collections.Counter(stratum(s.token_count) for s in eligible)),
        "original_document_exclusion_threshold": old.MIN_ELIGIBLE_DOCS_FOR_FULL_EXCLUSION,
        "original_frame_exact_train_text_overlap": sum(s.normalized_text in train_texts for s in eligible),
        "original_frame_exact_dev_text_overlap": sum(s.normalized_text in dev_texts for s in eligible),
        "original_strata": strata, "representative_replay_matches": True, "diagnostic_replay_matches": True,
        "completed_annotator_source_rows_match": True, "earlier_sample_source_rows_match": True,
        "earlier_sample_sentences_by_split": dict(collections.Counter(index[s][0] for s in previous)),
        "earlier_test_documents": sorted(previous[s]["doc"] for s in previous if index[s][0] == "test"),
        "existing_representative_stratum_counts": dict(collections.Counter(stratum(len(v["tokens"]))
            for v in current.values() if v["set"] == "Representative")),
        "historical_unique_annotated_documents": len(prior_docs),
        "frozen_splits": {split: {"sentences": len(ss), "documents": len({s.doc for s in ss}),
            "tokens": sum(len(s.tokens) for s in ss)} for split, ss in splits.items()},
        "article_identifier_audit": {"jsonl_fields": dict(jsonl_fields), "verifiable_article_ids_found": False,
          "limitation": "JSONL has korean/pangyo only. jsonl_N is a paragraph ID, not an article identifier."},
        "discrepancies": ["Original missing-text exclusion is described but load_test_sentences does not reject empty text.",
          "Original representative selector permits quota top-up; diagnostic selector permits document reuse. Neither affected this stored sample.",
          "Original blank templates default Tokenization_Issue to No; extension leaves all four annotation fields blank as requested."]}
    require(audit["frozen_splits"]["test"] == {"sentences": 5701, "documents": 835, "tokens": 98529},
            "Authoritative test counts differ")
    return annotated, docmap, audit


def candidate_frame(splits, annotated, docmap):
    previous_ids, previous_docs = set(annotated), {v["doc"] for v in annotated.values()}
    previous_texts = {normalize(v["text"]) for v in annotated.values()}
    train = {s.norm for s in splits["train"] if s.norm}
    dev = {s.norm for s in splits["dev"] if s.norm}
    ids = collections.Counter(s.sid for s in splits["test"])
    frame, excluded, kept_text = [], [], {}
    for s in sorted(splits["test"], key=lambda s: (s.sid, s.doc)):
        reasons = []
        if s.errors:
            reasons.extend(s.errors)
        if ids[s.sid] != 1:
            reasons.append("ambiguous_sentence_id")
        if docmap.get(s.sid) != (s.doc, len(s.tokens)):
            reasons.append("unalignable_document_map")
        if s.sid in previous_ids:
            reasons.append("previous_annotated_sentence")
        if s.doc in previous_docs:
            reasons.append("previous_annotated_document")
        if s.norm in previous_texts:
            reasons.append("normalized_previous_annotation_duplicate")
        if s.norm in train:
            reasons.append("normalized_train_duplicate")
        if s.norm in dev:
            reasons.append("normalized_dev_duplicate")
        if s.norm in kept_text:
            reasons.append("normalized_test_frame_duplicate")
        if reasons:
            excluded.append({"source_sentence_id": s.sid, "source_document_id": s.doc,
                "original_token_count": len(s.tokens), "stratum": stratum(len(s.tokens)),
                "primary_reason": reasons[0], "all_reasons": "|".join(reasons),
                "normalized_text_sha256": digest(s.norm.encode("utf-8"))})
        else:
            kept_text[s.norm] = s.sid
            frame.append(s)
    bins = [[s for s in frame if stratum(len(s.tokens)) == b] for b in range(1, 6)]
    audit = [{"stratum": b, "minimum_tokens": 1 if b == 1 else EDGES[b-2] + 1,
        "maximum_tokens": EDGES[b-1] if b < 5 else "unbounded",
        "frozen_test_sentences": sum(stratum(len(s.tokens)) == b for s in splits["test"]),
        "eligible_sentences": len(bins[b-1]), "eligible_documents": len({s.doc for s in bins[b-1]}),
        "existing_representative_sentences": 12, "batch_a_requested": 12, "batch_b_requested": 24}
        for b in range(1, 6)]
    # Hall's condition for five stratum groups proves distinct-document
    # quota feasibility; per-bin counts alone would not establish this.
    hall = []
    for mask in range(1, 32):
        bs = [b for b in range(1, 6) if mask & (1 << (b-1))]
        n = len({s.doc for b in bs for s in bins[b-1]})
        hall.append({"strata": bs, "available_documents": n, "required": 36*len(bs), "passed": n >= 36*len(bs)})
    return frame, excluded, audit, hall


def select_extension(frame):
    """One seeded bipartite matching, followed by seeded batch allocation.

    Candidate order: lexicographic sentence ID then document ID. Shuffle
    once per ascending bin. Slots: 36 per bin, ascending bin/slot. An
    augmenting path relocates earlier slots when needed, never relaxing a
    quota or retrying a draw. Batch shuffle follows matching on the same RNG.
    """
    rng = random.Random(SEED)
    bins = {b: sorted((s for s in frame if stratum(len(s.tokens)) == b), key=lambda s: (s.sid, s.doc))
            for b in range(1, 6)}
    for b in range(1, 6):
        rng.shuffle(bins[b])
    owner, assigned = {}, {}

    def augment(slot, seen):
        for s in bins[slot[0]]:
            if s.doc in seen:
                continue
            seen.add(s.doc)
            prior = owner.get(s.doc)
            if prior is None or augment(prior, seen):
                owner[s.doc] = slot
                assigned[slot] = s
                return True
        return False

    for b in range(1, 6):
        for k in range(36):
            require(augment((b, k), set()), f"Cannot satisfy stratum {b}; no quotas changed")
    batches = {"A": [], "B": []}
    for b in range(1, 6):
        chosen = [assigned[(b, k)] for k in range(36)]
        rng.shuffle(chosen)
        batches["A"].extend(chosen[:12])
        batches["B"].extend(chosen[12:])
    return [(batch, f"CR-REP-{batch}-{i:03d}", s)
        for batch in ("A", "B")
        for i, s in enumerate(sorted(batches[batch], key=lambda s: (stratum(len(s.tokens)), s.doc, s.sid)), 1)]


def manifest_rows(selected, hashes):
    rows = []
    for batch, sample_id, s in selected:
        seq_sha = digest(serialized([form for _, form in s.tokens]))
        for ordinal, (tid, form) in enumerate(s.tokens, 1):
            rows.append(dict(zip(MANIFEST_FIELDS, (batch, stratum(len(s.tokens)), sample_id, s.doc, s.sid,
                s.text, len(s.tokens), ordinal, tid, form, digest(s.norm.encode("utf-8")), seq_sha,
                SEED, VERSION, GUIDELINE, hashes[SPLITS["test"].relative_to(ROOT).as_posix()],
                hashes[OLD_MANIFEST.relative_to(ROOT).as_posix()]))))
    return rows


def validate_selection(selected, splits, annotated, frame):
    test = {s.sid: s for s in splits["test"]}
    old_docs = {v["doc"] for v in annotated.values()}
    old_text = {normalize(v["text"]) for v in annotated.values()}
    other_text = {s.norm for split in ("train", "dev") for s in splits[split]}
    require(len(selected) == 180 and len({sid for _, sid, _ in selected}) == 180, "Sample IDs/count invalid")
    ss = [s for _, _, s in selected]
    require(len({s.sid for s in ss}) == 180 and len({s.doc for s in ss}) == 180, "Repeated sentence/document")
    require(len({s.norm for s in ss}) == 180, "New normalized duplicates")
    require(all(s.doc not in old_docs and s.norm not in old_text and s.norm not in other_text for s in ss),
            "Selection violates exclusions")
    require(all(test[s.sid] == s and s in frame for s in ss), "Selection is not unchanged eligible test source")
    require(len({v["doc"] for v in annotated.values()} | {s.doc for s in ss}) == len(annotated)+180,
            "Document reused across historical and extension samples")
    stats = []
    for batch, quota in (("A", 12), ("B", 24)):
        for b in range(1, 6):
            items = [s for x, _, s in selected if x == batch and stratum(len(s.tokens)) == b]
            require(len(items) == quota, "Batch/stratum quota mismatch")
            stats.append({"batch": batch, "stratum": b, "sentences": len(items),
                          "tokens": sum(len(s.tokens) for s in items)})
    return stats


def protected_snapshot():
    excluded = {".git", ".claude", "__pycache__", ".pytest_cache"}
    files = {}
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in excluded and
            not (Path(base)/d).resolve().is_relative_to(OUT.resolve()) and
            not (Path(base)/d).resolve().is_relative_to((ROOT/"scripts/camera_ready").resolve()))
        for name in sorted(names):
            p = Path(base)/name
            files[p.relative_to(ROOT).as_posix()] = sha(p)
    return files


def finalize_xlsx(path):
    """Add standard OOXML accidental-edit protection and strict list errors.

    Workbook values/layout originate in artifact-tool. No reference workbook
    is copied. This small XML pass adds export controls unsupported by the
    documented quick API; it does not add source or reference-label data.
    """
    with zipfile.ZipFile(path) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    q = lambda name: f"{{{XMLNS}}}{name}"
    styles = ET.fromstring(parts["xl/styles.xml"])
    xfs = styles.find(q("cellXfs"))
    unlocked = {}
    for name in sorted(n for n in parts if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")):
        sheet = ET.fromstring(parts[name])
        for c in sheet.iter(q("c")):
            ref = c.attrib["r"]
            col = "".join(x for x in ref if x.isalpha())
            row = int("".join(x for x in ref if x.isdigit()))
            if name.endswith("sheet2.xml") and col in "HIJK" and row > 1:
                style_id = int(c.get("s", "0"))
                if style_id not in unlocked:
                    style = copy.deepcopy(xfs[style_id])
                    existing = style.find(q("protection"))
                    if existing is not None:
                        style.remove(existing)
                    ET.SubElement(style, q("protection"), locked="0", hidden="0")
                    style.set("applyProtection", "1")
                    unlocked[style_id] = len(xfs)
                    xfs.append(style)
                c.set("s", str(unlocked[style_id]))
        for dv in sheet.iter(q("dataValidation")):
            dv.set("allowBlank", "1")
            dv.set("showErrorMessage", "1")
            dv.set("errorStyle", "stop")
            dv.set("errorTitle", "Use the dropdown values")
            dv.set("error", "Choose one of the listed values; do not change source rows.")
            dv.set("showDropDown", "0")
        protection = ET.Element(q("sheetProtection"), sheet="1", objects="1", scenarios="1",
            selectLockedCells="0", selectUnlockedCells="0", formatCells="1", formatRows="1",
            formatColumns="1", insertRows="1", deleteRows="1", insertColumns="1", deleteColumns="1")
        data = sheet.find(q("sheetData"))
        sheet.insert(list(sheet).index(data)+1, protection)
        parts[name] = ET.tostring(sheet, encoding="utf-8", xml_declaration=True)
    xfs.set("count", str(len(xfs)))
    parts["xl/styles.xml"] = ET.tostring(styles, encoding="utf-8", xml_declaration=True)
    # Keep document properties generic, never identity from the old workbooks.
    if "docProps/core.xml" in parts:
        core = ET.fromstring(parts["docProps/core.xml"])
        for item in core:
            tag = item.tag.split("}")[-1]
            if tag in ("creator", "lastModifiedBy"):
                item.text = "Human annotation coordinator"
            elif tag in ("created", "modified"):
                item.text = "2026-10-01T00:00:00Z"
        parts["docProps/core.xml"] = ET.tostring(core, encoding="utf-8", xml_declaration=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(parts):
            info = zipfile.ZipInfo(name, (2026, 10, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, parts[name])


def validate_workbooks(output, rows):
    result = {}
    for batch in ("A", "B"):
        expected = [("Representative", r["sample_id"], r["source_sentence_id"], r["source_document_id"],
                     r["sentence"], int(r["token_ordinal"]), r["token"], None, None, None, None)
                    for r in rows if r["batch"] == batch]
        sources = []
        for person in ("P1", "P2"):
            path = output / f"Human_POS_CR_Batch{batch}_{person}.xlsx"
            wb = openpyxl.load_workbook(path, data_only=False, keep_links=True)
            sheet_name = "Representative_60" if batch == "A" else "Representative_120"
            require(wb.sheetnames == ["Instructions", sheet_name], "Unexpected/hidden reference sheet")
            require(all(s.sheet_state == "visible" for s in wb), "Hidden sheet")
            require(not wb._external_links and not wb.defined_names, "External links/named references")
            ws = wb[sheet_name]
            actual = list(ws.iter_rows(values_only=True))
            require(actual[0] == HEADER and actual[1:] == expected, f"Workbook source/blank mismatch: {path.name}")
            require(ws.protection.sheet, "Source protection missing")
            require(ws.freeze_panes == "C2", "Source identifiers/header freeze missing")
            for row in ws.iter_rows(min_row=2):
                require(all(c.protection.locked for c in row[:7]), "Editable source metadata")
                require(all(not c.protection.locked for c in row[7:]), "Annotation field locked")
            require(len(ws.data_validations.dataValidation) == 3, "Dropdown validations missing")
            for dv in ws.data_validations.dataValidation:
                require(dv.showErrorMessage and dv.errorStyle == "stop", "Nonblocking dropdown")
            allowed = set(UPOS) | {"High", "Medium", "Low", "Yes", "No"}
            for dv in ws.data_validations.dataValidation:
                require(set(dv.formula1.strip('"').split(',')) <= allowed, "Answer data in validation")
            for sheet in wb:
                for row in sheet:
                    for cell in row:
                        require(cell.data_type != "f" and cell.comment is None and cell.hyperlink is None,
                                "Formula/comment/link embedded in workbook")
            sources.append(actual)
            wb.close()
            with zipfile.ZipFile(path) as z:
                members = z.namelist()
                require(not any(any(x in n.lower() for x in
                    ("externallink", "embedding", "vbaproject", "comment", "customxml", "drawing")) for n in members),
                    "Unexpected embedded workbook content")
                for name in members:
                    if name.endswith(".rels"):
                        require(b'TargetMode="External"' not in z.read(name), "External package relationship")
                    if name.startswith("xl/worksheets/") and name.endswith(".xml"):
                        xml = ET.fromstring(z.read(name))
                        require(not list(xml.iter(f"{{{XMLNS}}}f")), "Formula in XML")
            result[path.name] = {"passed": True, "source_token_rows": len(expected),
                "sentences": len({r[1] for r in expected}), "blank_annotation_fields": True,
                "exact_manifest_source_match": True, "protected_source_editable_inputs": True,
                "no_hidden_sheets_formulas_comments_links_embedded_data": True, "sha256": sha(path)}
        require(sources[0] == sources[1], "P1/P2 source rows differ")
    return result


def report_text(design, stats, excluded, validation):
    orig = design["original_sampling_audit"]
    lines = ["# Prospective Human Annotation Extension", "", "Selection frozen before any new annotation; no human labels generated or model evaluation performed.",
        "", "## Original sampling audit", "",
        "The original implementation and manifest replay agree on all 60 representative and 20 diagnostic sentence IDs, categories, and strata. Completed P1/P2 source rows agree and align exactly to the frozen corpus.",
        "", f"Original eligible frame: {orig['original_frame_sentences']} sentences / {orig['original_frame_documents']} documents. Seed 42; NFKC plus whitespace collapse; case preserved.",
        "Original exclusions: " + json.dumps(orig["original_exclusions"], sort_keys=True) + ".",
        "Original eligible sentences per bin: " + json.dumps(orig["original_eligible_stratum_counts"], sort_keys=True) + ".",
        "Original loader retains real tokens with nonempty forms and sentence/document IDs, excludes repeated sentence IDs and document-map mismatches, excludes earlier sentence IDs and normalized texts, and retains the first eligible normalized text within test. Full prior-document exclusion is conditional on at least 80 remaining documents; it was active. Empty text was not actually rejected. Train/dev exact-text overlap was not checked by the original sampler: its eligible frame contains " + str(orig["original_frame_exact_train_text_overlap"]) + " train matches and " + str(orig["original_frame_exact_dev_text_overlap"]) + " dev matches. The extension adds that exclusion.",
        "Earlier v3: 60 sentences / 932 tokens; train 49, dev 4, test 7. Its labels are not pooled.",
        "", "Length edges were quantiles of the original eligible sentence pool: index round((i/5)*(n-1)), capped to range. Assignment uses length <= edge. Exact preserved bins: 1–12, 13–15, 16–18, 19–22, and 23+ real tokens.",
        "The original representative selector sorts lexicographic sentence IDs, shuffles each bin with one Random(42), and visits bins in order, enforcing document/text uniqueness. It has a quota top-up fallback. Diagnostic selection then uses the same RNG, removes representative sentences, initially avoids their documents, and ranks surface/script heuristics in five fixed category slots. It has document-reuse/top-up fallbacks. No fallback document reuse occurred in the stored sample.",
        "The original boundary diagnostic score skips OTHER tokens before counting KO/EN changes. Diagnostic selection is targeted and remains separate.",
        "", "Discrepancies/qualifications:", *["- " + x for x in orig["discrepancies"]],
        "", "Source linkage: Source_Sent_ID plus Token_ID (1-based real-token ordinal), exact FORM, and Document_ID. Current source IDs are contiguous integers with no multiword/empty-node rows. Predictions can eventually be joined by the same sentence/ordinal/form triple; predictions were not read for extension selection.",
        "", "## Extension frame and exclusions", "",
        f"Frozen test: 5,701 sentences / 835 documents / 98,529 tokens. Eligible extension frame: {design['candidate_frame']['sentences']} sentences / {design['candidate_frame']['documents']} documents.",
        "Previously annotated test documents excluded: 87 (60 representative + 20 diagnostic + 7 earlier test). All 140 previous annotated sentence texts, including earlier train/dev annotations, are checked for normalized duplicates.",
        "Primary exclusion counts: " + json.dumps(dict(collections.Counter(r["primary_reason"] for r in excluded)), sort_keys=True) + ".",
        "Primary reasons are mutually exclusive; exclusion_audit.csv also records every applicable reason. Near duplicates are not removed by a subjective similarity threshold. Exact normalized duplicates only are excluded. No article identifiers are available: JSONL contains korean and pangyo, and document IDs denote paragraphs.",
        "", "| Stratum | Token range | Frozen test sentences | Eligible sentences | Eligible documents |", "|---|---|---:|---:|---:|"]
    for a in design["candidate_frame"]["strata"]:
        lines.append(f"| {a['stratum']} | {a['minimum_tokens']}–{a['maximum_tokens']} | {a['frozen_test_sentences']} | {a['eligible_sentences']} | {a['eligible_documents']} |")
    lines += ["", "Eligible document counts overlap across bins and must not be added. All 31 nonempty subsets of strata pass Hall's distinct-document feasibility condition (36 new slots per bin).", "", "## Frozen selection", "",
        "Algorithm cr-human-extension-v1: label-free candidate records are sorted by sentence ID/document ID; bins 1–5 are shuffled once with Random(20261001). A deterministic augmenting-path document-to-stratum matching fills 36 slots per bin without retrying a random draw or relaxing quotas. After matching, each bin's 36 selected sentences is shuffled with the same RNG; first 12 go to Batch A, next 24 to Batch B. IDs are assigned after sorting within batches by stratum/document/sentence. No length edges are recomputed.",
        "", "| Batch | Stratum | Sentences | Tokens |", "|---|---:|---:|---:|"]
    for a in stats:
        lines.append(f"| {a['batch']} | {a['stratum']} | {a['sentences']} | {a['tokens']} |")
    for b in ("A", "B"):
        lines.append(f"| {b} total | | {sum(a['sentences'] for a in stats if a['batch']==b)} | {sum(a['tokens'] for a in stats if a['batch']==b)} |")
    lines += ["", "## Guideline and annotation workbooks", "",
        "Frozen v1.2 explicitly applies to future rounds (guideline version-history table and v1.1→v1.2 change log). This is the approved prospective revision permitted by the task. Original independent annotations used v1.1; their final adjudication used provisional v1.2. The extension uses v1.2 from the start; do not describe the original independent round retroactively as v1.2.",
        "The established eleven source/input columns are preserved. All Human_UPOS, Confidence, Tokenization_Issue, and Note cells start blank, including the tokenization flag. Seventeen UPOS choices, High/Medium/Low confidence, and Yes/No flags have blocking dropdown validation. Source columns are locked; four annotation columns are editable. Protection prevents accidents, not intentional tampering.",
        "", "## Validation and repeatability", "", "All checks in selection_validation.json passed. Source token identities, historical document exclusions, train/dev/prior exact-text exclusions, quotas, batch separation, pair equality, blank fields, OOXML contents, and source-file integrity are validated. The seed/procedure is repeated and stable serialized token records compared. Each output workbook is reopened and compared to the frozen manifest.",
        "", f"Protected pre-existing files checked: {validation['protected_files']['count']}; changed/missing files: 0.",
        "", "## Statistical interpretation and prospective confirmation", "",
        "The target is the remaining eligible frozen-test sentences under exact duplicate exclusions, conditional on the historical annotation selections. Five equal length-bin quotas and document-constrained matching are not self-weighting sampling of all test tokens. No inclusion probabilities or population weights are calculated. Matching and sequential assignment induce unequal selection probabilities; exclusion of previously sampled documents also changes the frame. Treat the extension as a prospective, stratified confirmation sample, not an unbiased full-corpus token estimate.",
        "Freeze intended model configurations, mappings, preprocessing, outcomes, comparisons, punctuation rules, and uncertainty protocol before new labels are unblinded. No such model/evaluation freeze is claimed by this sampling task. A and B are allocated before annotation and cannot be selected later based on favorable outcomes. Report new-sample results first; combined existing/new analysis is secondary with design and guideline-version qualifications. Do not develop mappings on these labels and then claim independent confirmation on the same labels.",
        "", "## Safe distribution", "",
        "Send P1 only Human_POS_CR_BatchA_P1.xlsx and Human_POS_CR_BatchB_P1.xlsx; send P2 only the corresponding P2 files. Supply the unchanged frozen v1.2 PDF from guidelines/. Annotators work independently and do not consult previous completed workbooks, silver/model labels, each other, or internal manifests/audits. Keep all internal JSON/CSV/report/preservation/build files with the coordinator. Save completed files as new copies; retain these blank originals. Do not distribute the whole output directory.",
        "", "## Reproduce or verify", "", "`python -B scripts/camera_ready/create_human_annotation_extension.py` creates a fresh output directory and refuses overwrite. `python -B scripts/camera_ready/create_human_annotation_extension.py --verify-only` recomputes the label-blind selection and validates existing blank workbooks without writing. Python 3.12, openpyxl, Node, and @oai/artifact-tool are required for creation; verification does not require Node. Runtime versions and all input/script checksums are in sampling_design.json.",
        "", "## Input checksums", "", "| Source | SHA-256 |", "|---|---|"]
    lines += [f"| `{p}` | `{h}` |" for p,h in design["source_checksums_sha256"].items()]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        require(not OUT.exists(), "Output directory already exists; refusing overwrite. Use --verify-only.")
    preserved = (json.loads((OUT/"protected_file_checksums.json").read_text()) if args.verify_only
                 else protected_snapshot())
    inputs = [*SPLITS.values(), DOC_MAP, OLD_MANIFEST, OLD_AUDIT,
        ANNOT/"Human_POS_Tagging_test_v1_P1_COMPLETED.xlsx", ANNOT/"Human_POS_Tagging_test_v1_P2_COMPLETED.xlsx",
        PREV/"Human_POS_Tagging_v3_P1.xlsx", PREV/"Human_POS_Tagging_v3_P2.xlsx",
        GUIDES/"Pangyo_Human_UPOS_Annotation_Guideline_v1.1.md",
        GUIDES/"Pangyo_Human_UPOS_Annotation_Guideline_v1.2.md",
        GUIDES/"Pangyo_Human_UPOS_Annotation_Guideline_v1.2.pdf",
        GUIDES/"Pangyo_Human_UPOS_Annotation_Guideline_v1.1_to_v1.2_ChangeLog.md",
        ROOT/"data/kor_en_panyo_corpus.jsonl", ROOT/"scripts/create_test_only_human_evaluation_sample.py",
        ROOT/"guidelines/data_split_decisions.md", Path(__file__),
        Path(__file__).with_name("build_annotation_workbooks.mjs"),
        Path(__file__).with_name("test_human_annotation_extension.py")]
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs}
    splits = {split: load_split(path) for split,path in SPLITS.items()}
    annotated, docmap, original = audit_inputs(splits)
    frame, excluded, audit, hall = candidate_frame(splits, annotated, docmap)
    if not all(h["passed"] for h in hall):
        print(json.dumps({"feasible": False, "limiting_strata": [h for h in hall if not h['passed']],
                          "frame_audit": audit}, indent=2))
        raise SystemExit("Infeasible: no selection or annotation workbooks generated; quotas unchanged.")
    selected = select_extension(frame)
    stats = validate_selection(selected, splits, annotated, frame)
    rows = manifest_rows(selected, hashes)
    repeated = manifest_rows(select_extension(list(reversed(frame))), hashes)
    require(csv_bytes(rows, MANIFEST_FIELDS) == csv_bytes(repeated, MANIFEST_FIELDS), "Repeatability/order test failed")
    require(not any(s.errors for ss in splits.values() for s in ss), "Frozen corpus contains malformed input")
    safeguards = subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("test_human_annotation_extension.py"))],
        capture_output=True, text=True, encoding="utf-8")
    require(safeguards.returncode == 0, "Sampling safeguard tests failed: " + safeguards.stderr)
    validation = {"all_passed": True, "seed": SEED, "algorithm_version": VERSION,
        "selection_checks": {k: True for k in ("frozen_test_only", "exact_original_tokens_and_ordinals",
            "excluded_documents_not_reused", "no_previous_normalized_duplicates", "no_train_dev_normalized_duplicates",
            "one_sentence_per_document_across_all_annotations", "exact_stratum_quotas", "batch_a_60_batch_b_120",
            "unique_stable_sample_ids", "no_labels_or_predictions_used_by_extension_selector")},
        "repeatability": {"same_seed_identical_manifest_bytes": True, "reversed_frame_order_identical": True,
            "manifest_sha256": digest(csv_bytes(rows, MANIFEST_FIELDS))}, "batch_stratum_counts": stats,
        "feasibility_checks": hall, "sampling_safeguard_tests": {"passed": True,
            "test_file": "scripts/camera_ready/test_human_annotation_extension.py", "output": safeguards.stderr}}
    if args.verify_only:
        design = json.loads((OUT/"sampling_design.json").read_text())
        require(design["source_checksums_sha256"] == hashes, "Frozen sampling sources/scripts changed")
        require((OUT/"sampling_manifest.csv").read_bytes() == csv_bytes(rows, MANIFEST_FIELDS), "Frozen manifest differs")
    else:
        OUT.mkdir(parents=True)
        (OUT/"protected_file_checksums.json").write_bytes(serialized(preserved))
        frame_digest = digest(serialized([{"sid": s.sid, "doc": s.doc, "text": s.text, "tokens": s.tokens} for s in frame]))
        design = {"algorithm_version": VERSION, "selection_seed": SEED, "guideline_version": GUIDELINE,
            "guideline_basis": "Frozen v1.2 explicitly applies prospectively to future rounds; original independent round used v1.1.",
            "normalization": "Unicode NFKC, collapse whitespace, preserve case", "stratum_edges_inclusive": list(EDGES),
            "quotas": {"A": [12]*5, "B": [24]*5}, "original_sampling_audit": original,
            "candidate_frame": {"sentences": len(frame), "documents": len({s.doc for s in frame}),
                "sha256": frame_digest, "strata": audit}, "selection_algorithm": select_extension.__doc__,
            "exclusion_criteria": ["invalid/empty/ambiguous/unmapped", "all historical annotated sentences/documents",
                "NFKC exact historical text duplicate", "NFKC exact train/dev text duplicate",
                "NFKC exact duplicate in candidate frame (lexicographically first eligible occurrence retained)"],
            "selection_uses": ["sentence length", "document membership", "identity/text exclusions", "seeded randomization"],
            "selection_never_uses": ["silver POS", "human POS", "predictions", "correctness", "errors/confusions", "language proportions"],
            "inclusion_probabilities_or_population_weights": None, "prospective_protocol_freeze_required_before_unblinding": True,
            "source_checksums_sha256": hashes, "environment": {"python": platform.python_version(),
                "openpyxl": openpyxl.__version__}, "exclusion_primary_counts": dict(collections.Counter(x["primary_reason"] for x in excluded)),
            "exclusion_all_reason_counts": dict(collections.Counter(reason for x in excluded for reason in x["all_reasons"].split("|")))}
        (OUT/"sampling_manifest.csv").write_bytes(csv_bytes(rows, MANIFEST_FIELDS))
        (OUT/"candidate_frame_audit.csv").write_bytes(csv_bytes(audit, list(audit[0])))
        (OUT/"exclusion_audit.csv").write_bytes(csv_bytes(excluded, list(excluded[0])))
        (OUT/"guidelines").mkdir()
        shutil.copyfile(GUIDES/"Pangyo_Human_UPOS_Annotation_Guideline_v1.2.pdf",
                        OUT/"guidelines/Pangyo_Human_UPOS_Annotation_Guideline_v1.2.pdf")
        payload = {"header": HEADER, "upos": UPOS, "guideline_version": GUIDELINE, "batches": {}}
        for b in ("A", "B"):
            token_rows = [["Representative", r["sample_id"], r["source_sentence_id"], r["source_document_id"],
                           r["sentence"], r["token_ordinal"], r["token"], None, None, None, None]
                          for r in rows if r["batch"] == b]
            payload["batches"][b] = {"rows": token_rows, "sentences": 60 if b == "A" else 120}
        (OUT/"workbook_payload.json").write_bytes(serialized(payload))
        deps = Path(os.environ.get("CODEX_WORKSPACE_NODE_MODULES",
            str(Path.home()/".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules")))
        require((deps/"@oai/artifact-tool/package.json").exists(), "Bundled artifact-tool unavailable")
        build = OUT/".build"
        build.mkdir()
        link = build/"node_modules"
        if os.name == "nt":
            subprocess.run(["powershell", "-NoProfile", "-Command",
                "New-Item -ItemType Junction -Path '"+str(link).replace("'", "''")+"' -Target '"+
                str(deps).replace("'", "''")+"' | Out-Null"], check=True)
        else:
            link.symlink_to(deps, target_is_directory=True)
        marker = Path.home()/".cache/codex-runtimes/codex-primary-runtime/plugins/openai-primary-runtime/plugins/spreadsheets/skills/spreadsheets/container_tools/mark_artifact_operation_started.mjs"
        subprocess.run(["node", str(marker), "--operation-kind", "create", "--expected-output-count", "4", "--output-format", "xlsx"], check=True)
        subprocess.run(["node", str(Path(__file__).with_name("build_annotation_workbooks.mjs")), str(OUT)], check=True)
        design["environment"].update({"node": subprocess.check_output(["node", "--version"], text=True).strip(),
             "artifact_tool": json.loads((deps/"@oai/artifact-tool/package.json").read_text())["version"]})
        for p in sorted(OUT.glob("*.xlsx")):
            finalize_xlsx(p)
        (OUT/"sampling_design.json").write_bytes(serialized(design))
    validation["workbooks"] = validate_workbooks(OUT, rows)
    changed = [p for p,h in preserved.items() if not (ROOT/p).exists() or sha(ROOT/p) != h]
    require(not changed, f"Protected original files changed: {changed}")
    validation["protected_files"] = {"count": len(preserved), "changed_or_missing": changed,
        "snapshot_file": "protected_file_checksums.json", "all_original_files_unchanged": True}
    require(sha(OUT/"guidelines/Pangyo_Human_UPOS_Annotation_Guideline_v1.2.pdf") ==
        hashes["outputs/human_validation/guideline/Pangyo_Human_UPOS_Annotation_Guideline_v1.2.pdf"], "Guideline copy changed")
    if not args.verify_only:
        (OUT/"selection_validation.json").write_bytes(serialized(validation))
        (OUT/"sampling_report.md").write_text(report_text(design, stats, excluded, validation), encoding="utf-8", newline="\n")
    print(json.dumps({"all_passed": True, "eligible_sentences": len(frame), "eligible_documents": len({s.doc for s in frame}),
        "batch_stratum_counts": stats, "original_files_unchanged": len(preserved), "verify_only": args.verify_only}, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
