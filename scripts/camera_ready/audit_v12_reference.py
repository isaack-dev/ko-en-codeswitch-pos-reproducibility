"""Read-only source audit preceding any v1.2 model evaluation.

Archives are inspected in memory, never executed/extracted over source files.
Generated reports are researcher-only. Material failures stop the pipeline.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/camera_ready/v12_integration"
INPUT = OUT / "inputs"
SAMPLE = ROOT / "outputs/camera_ready/human_annotation_extension"
PLAN = ROOT / "reports/analysis_plans/pangyo_v1.2/PANGYO_V12_ANALYSIS_PLAN_FREEZE_2026-10-01.md"
PROTOCOL = PLAN.with_name("PANGYO_V12_FROZEN_ANALYSIS_PROTOCOL_2026-10-01.zip")
REFERENCE = INPUT / "PANGYO_V12_FINAL_HUMAN_REFERENCE_3186_TOKENS.csv"
FINAL_ZIP = INPUT / "final_archive/Pangyo_MAIN_v12_FINAL_HUMAN_REFERENCE_RESEARCHER_ONLY.zip"
PRE_ZIP = INPUT / "pre_adjudication/Pangyo_MAIN_v12_PRE_ADJUDICATION_RESEARCHER_ONLY.zip"
GUIDE = SAMPLE / "guidelines/Pangyo_Human_UPOS_Annotation_Guideline_v1.2.pdf"
sys.path.insert(0, str(ROOT / "scripts/camera_ready"))
import create_human_annotation_extension as sampling
UPOS = tuple(sampling.UPOS)
SOURCE_FIELDS = ("batch", "stratum", "sample_id", "source_document_id", "source_sentence_id",
                 "sentence", "token_ordinal", "original_token_id", "token")
EXPECTED = {PLAN: "13ad84858bca6479ebb15b3851856d0713d887eedcda172308b7df3fd32a9f4a",
    REFERENCE: "aca535a22f46a56a2083635ade0cca854de6066d4a62c9fa84ceda3ce70e01ff",
    SAMPLE/"sampling_manifest.csv": "831e288d9bcf47c5073b902271f3f0a21c5e9a8a7604d6aa4a0b653c6d143b4d",
    SAMPLE/"sampling_design.json": "264a4e61fc18a871911ac97d1508538ce9c80d95d600a355af81508e71148de2",
    SAMPLE/"selection_validation.json": "51e81ff6d0ad154b784e1449d2a84c7664fc11b9ff2a1bd97b15334adf59fc94",
    GUIDE: "4eba40f9e776adad8af00f0cd1bd44713171a0fcddc9513f43b481d4c478c131"}


class ValidationFailure(ValueError):
    pass


def check(condition, message):
    if not condition:
        raise ValidationFailure(message)


def sha(path):
    return sampling.sha(path)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(value):
    return sampling.serialized(value)


def csv_rows(data):
    if isinstance(data, Path):
        data = data.read_bytes()
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline="")))


def write_csv(name, rows, fields=None):
    fields = fields or list(rows[0])
    data = sampling.csv_bytes(rows, fields)
    (OUT/name).write_bytes(data)


def key(row):
    return row["sample_id"], int(row["token_ordinal"])


def unique(rows, keyfunc=key):
    result = {}
    for row in rows:
        k = keyfunc(row)
        check(k not in result, f"Duplicate identity: {k}")
        result[k] = row
    return result


def workbook(data, data_only=False):
    return openpyxl.load_workbook(io.BytesIO(data) if isinstance(data, bytes) else data,
                                 read_only=True, data_only=data_only)


def table(data, sheet, header_token="Sample_ID"):
    wb = workbook(data)
    try:
        ws = wb[sheet]
        ws.reset_dimensions()
        it = ws.iter_rows(values_only=True)
        header = next(r for r in it if header_token in r)
        idx = header.index(header_token)
        return [dict(zip(header, row)) for row in it if len(row) > idx and row[idx] is not None]
    finally:
        wb.close()


def values(data, sheet, region):
    wb = workbook(data)
    try:
        return [[c.value for c in row] for row in wb[sheet][region]]
    finally:
        wb.close()


def archived(path, name):
    with zipfile.ZipFile(path) as z:
        return z.read(name)


def preserve():
    baseline = json.loads((OUT/"preservation_checksums.json").read_text(encoding="utf-8"))
    changed = [p for p,h in baseline["files"].items() if not (ROOT/p).exists() or sha(ROOT/p) != h]
    check(not changed, f"Pre-existing source files changed/missing: {changed}")
    return baseline


def inventory():
    baseline = preserve()
    inputs = sorted(INPUT.rglob("*"))
    inputs = [p for p in inputs if p.is_file()] + [PLAN, PROTOCOL, GUIDE] + [SAMPLE/n for n in
        ("sampling_manifest.csv", "sampling_design.json", "selection_validation.json", "sampling_report.md")]
    inputs += [sampling.SPLITS[s] for s in ("train", "dev", "test")] + [sampling.DOC_MAP,
        ROOT/"data/kor_en_panyo_corpus.conllu", ROOT/"outputs/final_clean_reproducibility_run/manifest.json",
        ROOT/"outputs/final_clean_reproducibility_run/environment/environment.json",
        ROOT/"outputs/final_clean_reproducibility_run/human_model_evaluation/model_evaluation_manifest.json",
        ROOT/"outputs/final_clean_reproducibility_run/human_reference/reference_build_manifest.json"]
    for path, expected in EXPECTED.items():
        check(path.exists() and sha(path) == expected, f"Frozen checksum mismatch/missing: {path}")
    registry = {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs}
    archives, occurrences, unresolved_register_entries = {}, collections.defaultdict(list), []
    local = {p.name: (p, p.read_bytes()) for p in inputs}
    for p in sorted([PROTOCOL, *INPUT.rglob("*.zip")]):
        with zipfile.ZipFile(p) as z:
            check(z.testzip() is None, f"Archive CRC failure: {p}")
            names = z.namelist()
            check(len(names) == len(set(names)), f"Duplicate ZIP entry names: {p}")
            items = []
            for info in z.infolist():
                if info.is_dir():
                    continue
                data = z.read(info.filename)
                h = digest(data)
                location = p.relative_to(ROOT).as_posix()+"::"+info.filename
                occurrences[h].append(location)
                items.append({"member": info.filename, "bytes": len(data), "sha256": h})
                basename = Path(info.filename).name
                if basename in local:
                    check(data == local[basename][1], f"Archive/local contradiction: {location}")
                if basename in ("sampling_manifest.csv", "sampling_design.json", "selection_validation.json"):
                    check(data == (SAMPLE/basename).read_bytes(), f"Archive sampling contradiction: {location}")
                if basename.endswith("SHA256SUMS.txt") or basename == "SHA256SUMS.txt":
                    for line in data.decode("utf-8-sig").splitlines():
                        match = re.match(r"^([0-9a-f]{64})\s+(.+)$", line)
                        if not match:
                            continue
                        expected, registered = match.groups()
                        candidates = [n for n in names if n == registered or Path(n).name == Path(registered).name]
                        if candidates:
                            check(all(digest(z.read(n)) == expected for n in candidates), f"ZIP checksum register mismatch: {registered}")
                        elif Path(registered).name in local:
                            check(digest(local[Path(registered).name][1]) == expected, f"External source register mismatch: {registered}")
                        else:
                            unresolved_register_entries.append({"archive": p.relative_to(ROOT).as_posix(),
                                "registered_source": registered, "sha256": expected})
            archives[p.relative_to(ROOT).as_posix()] = items
    check(archived(PROTOCOL, PLAN.name) == PLAN.read_bytes(), "Protocol Markdown conflicts with authoritative plan")
    manifest = {"git": {k: baseline[k] for k in ("branch", "head", "initial_working_tree")},
        "file_checksums_sha256": registry, "archive_members": archives,
        "identical_archived_records": [locations for locations in occurrences.values() if len(locations) > 1],
        "registered_sources_not_supplied": unresolved_register_entries,
        "evidence_classes": {"mechanical": "Checksums, ZIP integrity, exact source/token and decision comparisons",
            "documentary": "Stage dependencies, instructions, sign-off fields and recorded rationales",
            "researcher_attestation_only": "Human authorship/identity, actual independent work, historical nonaccess to silver/models, and prospective timing"},
        "preregistration_timestamp_independently_authenticated": False,
        "environment": {"python": platform.python_version(), "openpyxl": openpyxl.__version__},
        "scripts_sha256": {"scripts/camera_ready/audit_v12_reference.py": sha(Path(__file__)),
            "scripts/camera_ready/create_human_annotation_extension.py": sha(Path(sampling.__file__))}}
    (OUT/"input_and_provenance_manifest.json").write_bytes(dump(manifest))
    return manifest


def validate_source(rows, sampled, test):
    check(len(rows) == len(sampled) == 3186, "Reference/manifest row count differs from 3186")
    unique(rows)
    unique(sampled)
    seen_samples, sentence_owner, documents = {}, {}, set()
    for row, m in zip(rows, sampled):
        check(all(row[f] == m[f] for f in SOURCE_FIELDS), f"Reference/manifest source differs: {key(m)}")
        check(m["source_sentence_id"] in test, f"Reference outside frozen TEST: {key(m)}")
        s = test[m["source_sentence_id"]]
        n = int(m["token_ordinal"])
        check(not s.errors and s.doc == m["source_document_id"] and s.text == m["sentence"] and
            1 <= n <= len(s.tokens) and s.tokens[n-1] == (m["original_token_id"], m["token"]),
            f"Original source/token/ordinal mismatch: {key(m)}")
        check(sampling.stratum(len(s.tokens)) == int(m["stratum"]), f"Length stratum mismatch: {key(m)}")
        sample_id = m["sample_id"]
        identity = (s.sid, s.doc, m["batch"], m["stratum"])
        check(sample_id not in seen_samples or seen_samples[sample_id] == identity, "Sample ID spans sources")
        check(s.sid not in sentence_owner or sentence_owner[s.sid] == sample_id, "Source sentence has multiple sample IDs")
        seen_samples[sample_id], sentence_owner[s.sid] = identity, sample_id
        documents.add(s.doc)
    check(len(seen_samples) == len(documents) == 180, "Sentence/document sample independence differs")
    for sample_id, (sid,doc,batch,stratum) in seen_samples.items():
        ordinals = [int(r["token_ordinal"]) for r in rows if r["sample_id"] == sample_id]
        check(ordinals == list(range(1,len(test[sid].tokens)+1)), "Incomplete/reordered token sequence")
    check(collections.Counter(r["stratum"] for r in rows) == {"1":357,"2":503,"3":615,"4":731,"5":980}, "Stratum token totals differ")
    check(collections.Counter((batch,stratum) for sid,doc,batch,stratum in seen_samples.values()) ==
          {(b,str(s)):n for b,n in (("A",12),("B",24)) for s in range(1,6)}, "Stratum sentence quotas differ")
    return seen_samples


def annotations(sampled):
    expected = unique(sampled)
    paired, diagnostics = {}, {}
    source_names = ("Batch","Length_Stratum","Sample_ID","Source_Document_ID","Source_Sentence_ID",
                    "Full_Sentence","Token_Ordinal","Original_Token_ID","Token")
    for batch in ("A", "B"):
        for person in ("P1", "P2"):
            filename = f"Human_POS_CR_Batch{batch}_{person}_{'COMPLETED' if person=='P1' else 'completed'}.xlsx"
            path = INPUT/"independent_annotations"/filename
            data = path.read_bytes()
            rows = table(data, "ANNOTATE")
            exp = [m for m in sampled if m["batch"] == batch]
            check(len(rows) == len(exp), f"Annotation row count mismatch: {filename}")
            # Received source-only templates establish actual distributed content.
            send = INPUT/f"process_archive/Pangyo_v1.2_MAIN_{person}_SEND.zip"
            blank = archived(send, f"Human_POS_CR_Batch{batch}_{person}_REVIEWED.xlsx")
            blank_rows = table(blank, "ANNOTATE")
            check(len(blank_rows) == len(rows), "Sent template token coverage mismatch")
            for r,b,m in zip(rows,blank_rows,exp):
                check([str(r[f]) for f in source_names] == [m[f] for f in SOURCE_FIELDS], f"Independent source altered: {filename}/{key(m)}")
                check([r[f] for f in source_names] == [b[f] for f in source_names], "Sent/submitted source differs")
                check(not b["Human_UPOS"] and not b["Confidence"] and not b["Note_Code"] and not b["Annotation_Note"], "Sent worksheet contains annotation answers")
                check(r["Human_UPOS"] in UPOS and r["Confidence"] in ("High","Medium","Low") and
                      r["Tokenization_Issue"] in ("Yes","No"), f"Missing/invalid independent response: {filename}/{key(m)}")
                check(r["Confidence"] == "High" or bool(r["Note_Code"]), f"Missing required Medium/Low note: {filename}/{key(m)}")
                check(r["Tokenization_Issue"] != "Yes" or (r["Note_Code"] == "TOK_BOUNDARY" and bool(r["Annotation_Note"])),
                      f"Missing boundary rationale: {filename}/{key(m)}")
                paired.setdefault(key(m), {})[person] = r
            wb = workbook(blank)
            try:
                check(all(s.sheet_state == "visible" for s in wb), "Hidden distributed sheet")
                check(not wb._external_links, "Distributed external workbook links")
                check(wb.sheetnames == ["START_HERE","ANNOTATE","PROGRESS","TAG_REFERENCE"], "Unexpected distributed answer/reference sheet")
            finally:
                wb.close()
            diagnostics[f"{batch}_{person}"] = {"file": filename, "sha256": sha(path), "rows": len(rows),
                "confidence": dict(collections.Counter(r["Confidence"] for r in rows)),
                "flag": dict(collections.Counter(r["Tokenization_Issue"] for r in rows)),
                "tag": dict(collections.Counter(r["Human_UPOS"] for r in rows)),
                "medium_low_note_violations": 0, "boundary_rationale_violations": 0,
                "blank_template_boundary_defaults": dict(collections.Counter(str(r["Tokenization_Issue"]) for r in blank_rows))}
    check(set(paired) == set(expected) and all(set(v) == {"P1","P2"} for v in paired.values()), "Incomplete paired annotation coverage")
    pairs = [(v["P1"]["Human_UPOS"],v["P2"]["Human_UPOS"]) for v in paired.values()]
    confusion = collections.Counter(pairs)
    ca,cb = collections.Counter(a for a,b in pairs),collections.Counter(b for a,b in pairs)
    n,agree = len(pairs),sum(a==b for a,b in pairs)
    pe = sum(ca[t]*cb[t] for t in UPOS)/(n*n)
    nonpunct = [(a,b) for a,b in pairs if "PUNCT" not in (a,b)]
    flags = {k for k,v in paired.items() if v["P1"]["Tokenization_Issue"] != v["P2"]["Tokenization_Issue"]}
    disagreements = {k for k,v in paired.items() if v["P1"]["Human_UPOS"] != v["P2"]["Human_UPOS"]}
    metrics = {"tokens":n, "raw_agree":agree, "upos_disagree":n-agree, "raw_agreement":agree/n,
        "chance_expected_agreement":pe, "cohens_kappa_nominal_unweighted_17tags":(agree/n-pe)/(1-pe),
        "nonpunct_n":len(nonpunct), "nonpunct_agree":sum(a==b for a,b in nonpunct),
        "nonpunct_agreement":sum(a==b for a,b in nonpunct)/len(nonpunct),
        "boundary_flag_disagree":len(flags), "upos_boundary_disagreement_overlap":len(flags & disagreements),
        "distinct_disputed_tokens":len(flags | disagreements),
        "confidence_disagree":sum(v["P1"]["Confidence"] != v["P2"]["Confidence"] for v in paired.values()),
        "inputs":diagnostics}
    preserved = json.loads(archived(PRE_ZIP,"researcher_record/main_pre_adjudication_metrics.json"))
    for name,value in metrics.items():
        if name in preserved and name != "inputs":
            check(abs(value-preserved[name]) < 1e-12, f"Pre-adjudication metric contradiction: {name}")
    for name, diag in diagnostics.items():
        for field in ("sha256","rows","confidence","flag","tag"):
            check(diag[field] == preserved["inputs"][name][field], f"Pre-adjudication annotation diagnostic mismatch: {name}/{field}")
    saved_rows = csv_rows(archived(PRE_ZIP,"researcher_record/MAIN_ALL_P1_P2_RESEARCHER_ONLY.csv"))
    for m,r in zip(sampled,saved_rows):
        check(all(m[f] == r[f] for f in SOURCE_FIELDS), "Preserved paired source differs")
        p = paired[key(m)]
        for person in ("P1","P2"):
            for csvfield,wbfield in (("upos","Human_UPOS"),("confidence","Confidence"),("note_code","Note_Code"),
                                   ("tokenization_issue","Tokenization_Issue"),("explanation","Annotation_Note")):
                check(r[f"{person.lower()}_{csvfield}"] == (p[person][wbfield] or ""), f"Preserved independent answer differs: {key(m)}/{person}/{csvfield}")
    check(len(saved_rows)==n, "Preserved paired token count differs")
    saved_conf = {(r["p1"],r["p2"]):r["n"] for r in preserved["confusion_counts"]}
    check(saved_conf == {k:v for k,v in confusion.items() if k[0]!=k[1]}, "Pre-adjudication confusion table differs")
    (OUT/"v12_pre_adjudication_metrics.json").write_bytes(dump(metrics))
    write_csv("v12_annotator_confusion.csv", [{"p1_upos":a,"p2_upos":b,"count":confusion[a,b]} for a in UPOS for b in UPOS])
    return paired, metrics


def adjudication(rows, paired):
    ledger = csv_rows(archived(FINAL_ZIP,"PANGYO_V12_FINAL_ADJUDICATION_LEDGER_158.csv"))
    adj = csv_rows(archived(FINAL_ZIP,"PANGYO_V12_ADJACENT_TOKEN_AUDIT_3.csv"))
    unique(ledger, lambda r:r["case_id"])
    check(len(ledger)==158 and len(adj)==3, "Decision ledger/adjacent count differs")
    recon = INPUT/"reconciliation/Pangyo_MAIN_v12_P1_RECONCILIATION_Completed.xlsx"
    p2 = INPUT/"reconciliation/Pangyo_MAIN_v12_P2_FINAL_10_REVIEW_completed.xlsx"
    four = INPUT/"reconciliation/Pangyo_MAIN_v12_P1_FINAL_FOUR_Completed.xlsx"
    p2first = INPUT/"blind_adjudication/Pangyo_MAIN_v12_BLIND_JOINT_ADJUDICATION_P2_entered.xlsx"
    case_rows = table(p2first,"UPOS_CASES","Case_ID")+table(p2first,"BOUNDARY_CASES","Case_ID")
    original_cases = unique(case_rows,lambda r:r["Case_ID"])
    check(set(original_cases)=={r["case_id"] for r in ledger}, "Ledger does not cover original cases")
    blind_send = INPUT/"blind_adjudication/Pangyo_MAIN_v12_BLIND_JOINT_ADJUDICATION_SEND_TO_BOTH.zip"
    joint_blank = archived(blind_send,"Pangyo_MAIN_v12_BLIND_JOINT_ADJUDICATION.xlsx")
    p1first = INPUT/"blind_adjudication/Pangyo_MAIN_v12_P1_BLIND_ADJUDICATION_PROPOSALS.xlsx"
    for material in (joint_blank,p1first,p2first):
        source_rows = table(material,"SOURCE_TOKENS")
        names = ("Batch","Stratum","Sample_ID","Source_Document_ID","Source_Sentence_ID",
                 "Full_Sentence","Token_Ordinal","Original_Token_ID","Original_Token")
        check(len(source_rows)==len(rows), "Blind handoff source coverage differs")
        for r,fr in zip(source_rows,rows):
            check([str(r[n]) for n in names]==[fr[f] for f in SOURCE_FIELDS], "Blind handoff source/context changed")
        wb = workbook(material)
        try:
            check(all(s.sheet_state=="visible" for s in wb) and not wb._external_links, "Hidden/external blind handoff data")
            check(wb.sheetnames==["READ_FIRST","UPOS_CASES","BOUNDARY_CASES","SENTENCE_AUDIT",
                "SOURCE_TOKENS","GROUP_COUNTS","PROGRESS","PROVENANCE"], "Unexpected blind handoff sheet")
        finally:
            wb.close()
    for sheet,field in (("UPOS_CASES","Final_UPOS"),("BOUNDARY_CASES","Final_Boundary_Issue")):
        for row in table(joint_blank,sheet,"Case_ID"):
            check(not row[field] and not row["Joint_Rationale"] and not row["Status"], "Joint send package contains prefilled consensus answers")
    recon_open = table(recon,"OPEN_48")
    recon_matched = table(recon,"MATCHED_110")
    recon_map = unique(recon_open+recon_matched,lambda r:r["Case_ID"])
    six = csv_rows(archived(FINAL_ZIP,"P2_NEWLY_ACCEPTED_6_RESEARCHER_ONLY.csv"))
    settled = csv_rows(archived(FINAL_ZIP,"SETTLED_150_RESEARCHER_ONLY.csv"))
    six_map,settled_map = unique(six,lambda r:r["case_id"]), unique(settled,lambda r:r["case_id"])
    p2_map = unique([r for sn in ("UPOS_5","BOUNDARY_3","ADJACENT_2") for r in table(p2,sn)],lambda r:r["Case_ID"])
    four_map = unique(table(four,"FINAL_FOUR"),lambda r:r["Case_ID"])
    check(len(recon_open)==48 and len(recon_matched)==110 and len(settled)==150 and len(six)==6,
          "Reconciliation stage counts differ")
    # Verify immutable fields against stage-specific delivered templates.
    templates = [(p2, INPUT/"process_archive/Pangyo_MAIN_v12_P2_FINAL_10_REVIEW_SEND.zip",
                  "Pangyo_MAIN_v12_P2_FINAL_10_REVIEW_REQUIRED.xlsx", [("UPOS_5","A4:O8"),("BOUNDARY_3","A4:O6"),("ADJACENT_2","A4:O5")]),
                 (four, INPUT/"process_archive/Pangyo_MAIN_v12_P1_FINAL_FOUR_SEND.zip",
                  "Pangyo_MAIN_v12_P1_FINAL_FOUR_REVIEW.xlsx", [("FINAL_FOUR","A4:L7"),("SOURCE_CONTEXT","A4:G76"),("FROZEN_PROVENANCE","A1:B8")]),
                 (recon, INPUT/"process_archive/Pangyo_MAIN_v12_P1_RECONCILIATION_SEND.zip",
                  "Pangyo_MAIN_v12_P1_RECONCILIATION_REQUIRED.xlsx", [("OPEN_48","A4:R51"),("MATCHED_110","A4:R113"),("ADJACENT_3","A4:M6")])]
    for submitted,package,member,ranges in templates:
        blank = archived(package,member)
        for sheet,region in ranges:
            check(values(submitted,sheet,region)==values(blank,sheet,region), f"Adjudication immutable content differs: {sheet}/{region}")
    check(values(recon,"READ_FIRST","B13:B14")==[["Yes"],["Yes"]], "P1 reconciliation sign-off absent")
    check(values(p2,"READ_FIRST","B14:B16")==[["Yes"],["P2"],["2026-10-02"]], "P2 final10 sign-off absent")
    check(values(four,"READ_FIRST","B14:B16")==[["Yes"],["P1"],["2026-10-02"]], "P1 final-four sign-off absent")
    check(collections.Counter(r["P2_RESPONSE"] for r in p2_map.values())=={"ACCEPT_P1_LATEST":6,"RETAIN_P2":4}, "P2 response counts differ")
    check(set(four_map)=={cid for cid,r in p2_map.items() if r["P2_RESPONSE"]=="RETAIN_P2"}, "Final-four lineage differs")
    final_map = unique(rows)
    decisions, traced = {}, []
    for r in ledger:
        cid,typ = r["case_id"],r["decision_type"]
        token_key = key(r)
        check(token_key in paired and typ in ("UPOS","BOUNDARY") and r["status"]=="RESOLVED", "Unresolved/unknown ledger decision")
        p = paired[token_key]
        field = "Human_UPOS" if typ=="UPOS" else "Tokenization_Issue"
        check(r["original_p1"]==p["P1"][field] and r["original_p2"]==p["P2"][field] and r["original_p1"]!=r["original_p2"],
              f"Original disagreement mismatch: {cid}")
        fr = final_map[token_key]
        check(r["token"]==fr["token"] and r["original_token_id"]==fr["original_token_id"] and r["batch"]==fr["batch"], f"Decision source mismatch: {cid}")
        case = original_cases[cid]
        check(case["Sample_ID"]==r["sample_id"] and int(case["Ordinal"])==int(r["token_ordinal"]) and
              case["Original_Token"]==r["token"] and case["Full_Sentence"]==fr["sentence"], f"Case context differs: {cid}")
        fields = ("P1_UPOS","P2_UPOS") if typ=="UPOS" else ("P1_Issue","P2_Issue")
        check(case[fields[0]]==r["original_p1"] and case[fields[1]]==r["original_p2"], "Blind case original annotator decisions differ")
        recon_r = recon_map[cid]
        check(recon_r["P1_original"]==r["original_p1"] and recon_r["P2_original"]==r["original_p2"], "Reconciliation original labels differ")
        if cid in settled_map:
            s = settled_map[cid]
            check(s["agreed_decision"]==r["final_decision"]==recon_r["P2_final_proposal"], "Settled final label differs")
            if cid in {x["Case_ID"] for x in recon_open}:
                check(recon_r["P1_REVIEW_ACTION"]=="ACCEPT_P2", f"Missing P1 acceptance: {cid}")
            else:
                check(recon_r["P1_LATEST_proposal"]==recon_r["P2_final_proposal"] and
                      not recon_r["P1_EXCEPTION_IF_ANY"] and not recon_r["P1_EXCEPTION_REASON"], "Matched proposal has unresolved exception")
            stage = "settled_150"
        elif cid in six_map:
            check(p2_map[cid]["P2_RESPONSE"]=="ACCEPT_P1_LATEST" and
                r["final_decision"]==six_map[cid]["agreed_decision"]==p2_map[cid]["P1_latest_position"], "Final10 acceptance differs")
            check(r["p1_latest_rationale"]==six_map[cid]["p1_latest_rationale"] and
                r["p2_latest_rationale"]==p2_map[cid]["P2_final_rationale_and_guideline"], "Final10 rationale differs")
            stage = "accepted_6"
        else:
            f = four_map.get(cid)
            check(f and typ=="BOUNDARY" and f["P1_FINAL_ACTION"]=="ACCEPT_P2_NOW" and
                  r["final_decision"]==f["P2_CURRENT_position"] and bool(f["P1_FINAL_rationale"]), "Final-four acceptance missing")
            check(r["p1_latest_rationale"]==f["P1_FINAL_rationale"] and r["p2_latest_rationale"]==f["P2_CURRENT_rationale"], "Final-four rationale differs")
            stage = "last_2_boundary"
        check(r["p1_latest_rationale"] and r["p2_latest_rationale"] and bool(r["decision_basis"]), "Decision rationale missing")
        decisions[typ,token_key] = r
        traced.append({"case_id":cid,"decision_type":typ,"sample_id":r["sample_id"],"token_ordinal":r["token_ordinal"],
            "final_decision":r["final_decision"],"documented_stage":stage,"source_verified":True})
    upos_keys = {k for k,p in paired.items() if p["P1"]["Human_UPOS"] != p["P2"]["Human_UPOS"]}
    flag_keys = {k for k,p in paired.items() if p["P1"]["Tokenization_Issue"] != p["P2"]["Tokenization_Issue"]}
    check({k for typ,k in decisions if typ=="UPOS"}==upos_keys and {k for typ,k in decisions if typ=="BOUNDARY"}==flag_keys,
          "Ledger does not match every original disagreement exactly")
    for r in rows:
        k,p = key(r),paired[key(r)]
        check(r["reference_status"]=="RESOLVED" and r["human_upos_v12"] in UPOS and r["tokenization_issue_v12"] in ("Yes","No"), "Final reference invalid/unresolved")
        for typ,field,provenance,case_id,wbfield in (("UPOS","human_upos_v12","upos_provenance","upos_case_id","Human_UPOS"),
                  ("BOUNDARY","tokenization_issue_v12","boundary_provenance","boundary_case_id","Tokenization_Issue")):
            d = decisions.get((typ,k))
            check(r[field] == (d["final_decision"] if d else p["P1"][wbfield]), f"Final reference not supported by original/decision: {k}/{typ}")
            check(r[provenance] == ("ADJUDICATED" if d else "ORIGINAL_P1_P2_AGREEMENT") and
                  r[case_id] == (d["case_id"] if d else ""), f"Final provenance/case ID differs: {k}/{typ}")
    adj_submissions = table(recon,"ADJACENT_3")
    for r in adj:
        k,p = key(r),paired[key(r)]
        check(r["final_upos"]==r["p1_original"]==r["p2_original"]==p["P1"]["Human_UPOS"]==p["P2"]["Human_UPOS"]==final_map[k]["human_upos_v12"],
              "Adjacent final correction not supported")
        proposal = next(x for x in adj_submissions if x["Sample_ID"]==r["sample_id"])
        check(proposal["P1_proposed_new_UPOS"]==r["proposed_p1"], "Adjacent original proposal differs")
        if proposal["P1_FINAL_STANCE"]=="ACCEPT_ORIGINAL":
            check(bool(proposal["P1_final_reason_and_guideline"]), "Adjacent withdrawal rationale missing")
        else:
            response = next(x for x in four_map.values() if x["Sample_ID"]==r["sample_id"])
            check(response["P1_FINAL_ACTION"]=="ACCEPT_P2_NOW" and response["P2_CURRENT_position"]==r["final_upos"], "Adjacent final review missing")
    write_csv("v12_decision_trace.csv",traced)
    return {"original_upos_disagreements_resolved":len(upos_keys),"boundary_disagreements_resolved":len(flag_keys),
        "stages":dict(collections.Counter(r["documented_stage"] for r in traced)),"adjacent_proposals_reviewed":len(adj),
        "adjacent_final_changes":0,"resolved_tokens":len(rows),"final_boundary_yes":sum(r["tokenization_issue_v12"]=="Yes" for r in rows)}


def run():
    provenance = inventory()
    sampled,rows = csv_rows(SAMPLE/"sampling_manifest.csv"),csv_rows(REFERENCE)
    check(archived(FINAL_ZIP,REFERENCE.name)==REFERENCE.read_bytes(), "Final archive CSV differs byte-for-byte")
    test = {s.sid:s for s in sampling.load_split(sampling.SPLITS["test"])}
    samples = validate_source(rows,sampled,test)
    # Existing read-only selection verifier checks the original historical exclusions.
    result = subprocess.run([sys.executable,"-B",str(Path(sampling.__file__)),"--verify-only"],capture_output=True,text=True,encoding="utf-8")
    check(result.returncode==0, "Frozen sampling read-only verification failed: "+result.stdout+result.stderr)
    (OUT/"sampling_verification_log.txt").write_text(result.stdout+result.stderr,encoding="utf-8")
    paired,metrics = annotations(sampled)
    decisions = adjudication(rows,paired)
    preserve()
    validation = {"gate_0_passed":True,"gate_1_passed":True,"reference_file_sha256":sha(REFERENCE),
        "source_rows":len(rows),"sentences":len(samples),"documents":len({v[1] for v in samples.values()}),
        "schema":list(rows[0]),"reference_status_counts":dict(collections.Counter(r["reference_status"] for r in rows)),
        "pre_adjudication":metrics,"adjudication":decisions,"historical_procedure_authenticated":False}
    (OUT/"reference_validation.json").write_bytes(dump(validation))
    (OUT/"reference_validation_report.md").write_text(
        "# v1.2 source/reference validation\n\nGates 0 and 1 passed.\n\n"
        f"- Final CSV SHA-256: `{sha(REFERENCE)}`; byte-identical to final researcher archive.\n"
        "- 180 original TEST sentences / 180 documents / 3,186 exact original token rows.\n"
        "- Batch A: 60 / 1,046; Batch B: 120 / 2,140. All five strata retain 36 sentences and 357/503/615/731/980 tokens.\n"
        "- Manifest/source identities, ordinals, forms, context, quotas and frozen input hashes match. Original sampling verifier passed without reselection.\n"
        "- All 3,186 labels are valid and RESOLVED; tokenization was not changed. All case IDs and provenance reconcile to independent submissions and documented decisions.\n"
        "- Original P1/P2 source sequences and responses reproduce the preserved pre-adjudication paired file.\n"
        "- Details: `reference_validation.json`, `v12_decision_trace.csv`, `v12_pre_adjudication_metrics.json`, `v12_annotator_confusion.csv`, `input_and_provenance_manifest.json`.\n"
        "- No calibration or historical reference tokens enter this analysis. Frozen files and prior user edits remain unchanged.\n",encoding="utf-8")
    (OUT/"annotation_process_audit.md").write_text(
        "# Annotation-process audit (researcher-only)\n\n"
        "## Mechanically verified\n\n"
        "The four completed submissions match both frozen sampling tokens and the reviewed workbooks in the preserved P1/P2 send archives. The reviewed ANNOTATE sheets have no UPOS/confidence/note answers, no external workbook links or hidden sheets. TAG_REFERENCE contains generic guideline instructions; PROGRESS contains response-check formulas, not reference answers.\n\n"
        f"Original agreement: {metrics['raw_agree']}/{metrics['tokens']} = {metrics['raw_agreement']:.12f}; nominal unweighted 17-tag kappa = {metrics['cohens_kappa_nominal_unweighted_17tags']:.12f}. Human-human non-PUNCT (exclude PUNCT by either): {metrics['nonpunct_agree']}/{metrics['nonpunct_n']} = {metrics['nonpunct_agreement']:.12f}.\n\n"
        f"UPOS disagreements: {metrics['upos_disagree']}; boundary disagreements: {metrics['boundary_flag_disagree']}; overlapping tokens: {metrics['upos_boundary_disagreement_overlap']}; distinct disputed tokens: {metrics['distinct_disputed_tokens']}; confidence differences: {metrics['confidence_disagree']}. All Medium/Low note and flagged-boundary explanation checks passed. Archived statistics/confusions and independent original file hashes match.\n\n"
        "158 original judgments comprise 136 UPOS and 22 boundary judgments, not 158 UPOS disagreements. Reconstructed closure: 150 initial settled judgments + 6 final10 acceptances + 2 final-four boundary acceptances. Three originally agreed adjacent-token proposals were reviewed and withdrawn; all three final labels retain original agreement. Immutable stage inputs and written acceptance/sign-off/rationale fields match delivered templates.\n\n"
        "## Documentary sequence\n\n"
        "Identical hashes connect MAIN send templates → archived independent submissions/pre-adjudication paired data → blank blind joint handoff → P1 proposals/P2 entered responses → P1 reconciliation → P2 final10 → P1 final-four → final ledger/CSV. Checkpoint archives preserve the preceding stage's inputs and pending cases. This is a dependency sequence supported by content and written sign-offs; filesystem/ZIP timestamps were not used to infer chronology. No received submitted workbooks or original pre-adjudication annotations were overwritten.\n\n"
        "## Researcher attestations and limits\n\n"
        "Actual annotator identity/human authorship, independent completion without consultation, actual nonaccess to silver/model outputs, and the timing of protocol freeze relative to outcome access cannot be authenticated from these files. No third-party preregistration timestamp is claimed. The plan asserts DEV readiness statistics, but the registered original pilot/confirmation metric files are not supplied in this input set; those contextual statistics are not independently verified here and do not enter TEST outcomes.\n\n"
        "Agreement measures scheme consistency, not correctness. Preserve the original submissions; report pre-adjudication IAA rather than post-consensus agreement. Raw notes/paired responses and all integration outputs are researcher-only.\n",encoding="utf-8")
    return rows,provenance,validation


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        _,_,result = run()
        print(json.dumps({"gates_0_1_passed":True,"tokens":result["source_rows"],"pre_adjudication":
            {k:v for k,v in result["pre_adjudication"].items() if k!="inputs"},"adjudication":result["adjudication"]},indent=2))
    except (ValidationFailure,FileNotFoundError,KeyError) as error:
        (OUT/"validation_blocker.json").write_bytes(dump({"gates_0_1_passed":False,"error":str(error),"model_evaluation_performed":False}))
        print("STOP BEFORE MODEL EVALUATION:",error)
        raise SystemExit(1)
