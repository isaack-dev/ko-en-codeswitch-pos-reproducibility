"""Supplementary source-only checks; does not alter the frozen selection."""
import argparse
import collections
import csv
import json
from pathlib import Path

import openpyxl

import create_human_annotation_extension as sampling


def restricted_ids(path, sheet, max_col, header_name):
    """Only identity columns; never import adjudication/consensus labels."""
    wb = openpyxl.load_workbook(path, read_only=True)
    try:
        ws = wb[sheet]
        ws.reset_dimensions()
        it = ws.iter_rows(max_col=max_col, values_only=True)
        header = next(r for r in it if header_name in r)
        column = header.index(header_name)
        return {str(r[column]) for r in it if r[column]}
    finally:
        wb.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    out = sampling.OUT / "additional_provenance_validation.json"
    if not args.verify_only:
        sampling.require(not out.exists(), "Refusing to overwrite provenance validation")
    current = sampling.annotation_sources(sampling.ANNOT / "Human_POS_Tagging_test_v1_P1_COMPLETED.xlsx",
        ("Representative_60", "Diagnostic_20"))
    previous = sampling.annotation_sources(sampling.PREV / "Human_POS_Tagging_v3_P1.xlsx",
        ("Representative_40", "Diagnostic_20"))
    final = sampling.ROOT / "outputs/human_validation/test_only/annotations/Human_POS_Tagging_test_v1_Blind_Adjudication_FINAL_v1.2.xlsx"
    specs = [(sampling.PREV / "Human_POS_Tagging_v3_Blind_Adjudication_FINAL.xlsx",
              "Adjudication_51", 3, "Sent_ID", set(previous)),
             (final, "Adjudication_69", 4, "Source_Sent_ID", set(current)),
             (final, "Post_Adjudication_Audit", 3, "Sample_ID", {v["sample_id"] for v in current.values()})]
    result = {"all_passed": True, "adjudication_source_audit": {},
              "script_sha256": sampling.sha(Path(__file__))}
    for path, sheet, cols, header, allowed in specs:
        ids = restricted_ids(path, sheet, cols, header)
        sampling.require(not ids - allowed, f"Additional annotated sources in {sheet}: {ids - allowed}")
        result["adjudication_source_audit"][sheet] = {
            "all_within_existing_annotation_pool": True, "unique_source_identifiers": len(ids),
            "source_file": path.relative_to(sampling.ROOT).as_posix(), "source_file_sha256": sampling.sha(path)}
    alignment = sampling.ROOT / "results/jsonl_conllu_alignment_report.csv"
    with alignment.open(encoding="utf-8", newline="") as f:
        records = list(csv.DictReader(f))
    with (sampling.OUT / "sampling_manifest.csv").open(encoding="utf-8", newline="") as f:
        docs = {r["source_document_id"] for r in csv.DictReader(f)}
    selected = [r for r in records if r["doc_id"] in docs]
    sampling.require(len(selected) == len(docs) == 180, "Selected document alignment missing/ambiguous")
    sampling.require(all(r["alignment_status"] == "exact" and r["match_score"] == "1.0000" for r in selected),
                     "Selected paragraph alignment is not exact")
    result["paragraph_alignment_audit"] = {
        "source_file": alignment.relative_to(sampling.ROOT).as_posix(), "source_file_sha256": sampling.sha(alignment),
        "all_record_statuses": dict(collections.Counter(r["alignment_status"] for r in records)),
        "selected_document_count": len(selected), "all_selected_paragraph_alignments_exact": True,
        "article_identifiers_available": False}
    previews = json.loads((sampling.OUT / ".build/visual_qa.json").read_text(encoding="utf-8"))
    sampling.require(all(x["preview"] == "rendered first eight source rows" for x in previews), "Preview render failed")
    result["preview_render_checks"] = previews
    result["visual_review_scope"] = "First eight source rows in each batch; full row/source and formatting controls validated separately."
    if args.verify_only:
        sampling.require(out.read_bytes() == sampling.serialized(result), "Provenance validation differs")
    else:
        out.write_bytes(sampling.serialized(result))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
