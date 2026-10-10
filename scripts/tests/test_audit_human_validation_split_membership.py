"""
test_audit_human_validation_split_membership.py - Unit tests for
scripts/audit_human_validation_split_membership.py.

Stdlib unittest only, no xlsx/CoNLL-U fixtures on disk: reconstruction is
tested against synthetic in-memory row dicts, and matching is tested
against synthetic in-memory SplitSentence indices. Run from repo root:
  python scripts/tests/test_audit_human_validation_split_membership.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import audit_human_validation_split_membership as audit  # noqa: E402


def p1p2_row(sample_set, source_sent_id, sentence, token_form, upos):
    return {
        "sample_set": sample_set,
        "source_sent_id": source_sent_id,
        "sentence_text": sentence,
        "token_form": token_form,
        "upos": upos,
    }


class TestReconstruction(unittest.TestCase):
    def test_agreement_tokens_keep_p1_upos_and_token_order(self):
        p1 = {
            ("SAMPLE-1", 1): p1p2_row("Representative", "pangyo1", "A B", "A", "DET"),
            ("SAMPLE-1", 2): p1p2_row("Representative", "pangyo1", "A B", "B", "NOUN"),
        }
        p2 = {
            ("SAMPLE-1", 1): p1p2_row("Representative", "pangyo1", "A B", "A", "DET"),
            ("SAMPLE-1", 2): p1p2_row("Representative", "pangyo1", "A B", "B", "NOUN"),
        }
        sentences = audit.reconstruct_human_sample_from_rows(p1, p2, {})
        self.assertEqual(len(sentences), 1)
        sent = sentences[0]
        self.assertEqual([t.token_id for t in sent.tokens], [1, 2])
        self.assertEqual([t.form for t in sent.tokens], ["A", "B"])
        self.assertEqual([t.final_upos for t in sent.tokens], ["DET", "NOUN"])
        self.assertEqual([t.decision_source for t in sent.tokens], ["agreement", "agreement"])

    def test_disagreement_uses_adjudicated_label_not_p1_or_p2(self):
        p1 = {("SAMPLE-1", 1): p1p2_row("Diagnostic", "pangyo2", "X", "X", "VERB")}
        p2 = {("SAMPLE-1", 1): p1p2_row("Diagnostic", "pangyo2", "X", "X", "ADP")}
        adjudicated = {("SAMPLE-1", 1): "ADP"}
        sentences = audit.reconstruct_human_sample_from_rows(p1, p2, adjudicated)
        tok = sentences[0].tokens[0]
        self.assertEqual(tok.final_upos, "ADP")
        self.assertEqual(tok.decision_source, "adjudicated")

    def test_disagreement_without_adjudication_record_raises(self):
        p1 = {("SAMPLE-1", 1): p1p2_row("Diagnostic", "pangyo2", "X", "X", "VERB")}
        p2 = {("SAMPLE-1", 1): p1p2_row("Diagnostic", "pangyo2", "X", "X", "ADP")}
        with self.assertRaises(audit.AuditError):
            audit.reconstruct_human_sample_from_rows(p1, p2, {})

    def test_p1_p2_token_form_mismatch_raises(self):
        p1 = {("SAMPLE-1", 1): p1p2_row("Diagnostic", "pangyo2", "X", "cat", "NOUN")}
        p2 = {("SAMPLE-1", 1): p1p2_row("Diagnostic", "pangyo2", "X", "dog", "NOUN")}
        with self.assertRaises(audit.AuditError):
            audit.reconstruct_human_sample_from_rows(p1, p2, {})

    def test_p1_p2_key_set_mismatch_raises(self):
        p1 = {("SAMPLE-1", 1): p1p2_row("Diagnostic", "pangyo2", "X", "cat", "NOUN")}
        p2 = {("SAMPLE-1", 2): p1p2_row("Diagnostic", "pangyo2", "X", "cat", "NOUN")}
        with self.assertRaises(audit.AuditError):
            audit.reconstruct_human_sample_from_rows(p1, p2, {})

    def test_non_contiguous_token_ids_raise(self):
        p1 = {
            ("SAMPLE-1", 1): p1p2_row("Diagnostic", "pangyo2", "X", "a", "NOUN"),
            ("SAMPLE-1", 3): p1p2_row("Diagnostic", "pangyo2", "X", "c", "NOUN"),
        }
        p2 = {
            ("SAMPLE-1", 1): p1p2_row("Diagnostic", "pangyo2", "X", "a", "NOUN"),
            ("SAMPLE-1", 3): p1p2_row("Diagnostic", "pangyo2", "X", "c", "NOUN"),
        }
        with self.assertRaises(audit.AuditError):
            audit.reconstruct_human_sample_from_rows(p1, p2, {})


def make_human_sentence(sample_id, sent_id, text, forms, sample_set="Representative"):
    tokens = tuple(
        audit.HumanToken(token_id=i + 1, form=f, final_upos="NOUN", decision_source="agreement")
        for i, f in enumerate(forms)
    )
    return audit.HumanSentence(
        sample_set=sample_set, sample_id=sample_id,
        source_sent_id=sent_id, sentence_text=text, tokens=tokens,
    )


def make_split_sentence(sent_id, doc_id, split, text, forms):
    return audit.SplitSentence(sent_id=sent_id, doc_id=doc_id, split=split, text=text, token_forms=tuple(forms))


class TestMatching(unittest.TestCase):
    def test_exact_identifier_match(self):
        hs = make_human_sentence("S-1", "pangyo1", "A B", ["A", "B"])
        split_sent = make_split_sentence("pangyo1", "jsonl_1", "dev", "A B", ["A", "B"])
        sentid_index = {"pangyo1": [split_sent]}
        result = audit.match_human_sentence(hs, sentid_index, {}, {})
        self.assertEqual(result.method, "sent_id_exact")
        self.assertEqual(result.status, "matched")
        self.assertEqual(result.split, "dev")
        self.assertEqual(result.doc_id, "jsonl_1")

    def test_token_sequence_fallback_when_identifier_missing(self):
        hs = make_human_sentence("S-1", "pangyo999", "A B", ["A", "B"])
        split_sent = make_split_sentence("pangyo1", "jsonl_1", "test", "A B", ["A", "B"])
        token_seq_index = {("A", "B"): [split_sent]}
        result = audit.match_human_sentence(hs, {}, token_seq_index, {})
        self.assertEqual(result.method, "token_sequence_exact")
        self.assertEqual(result.status, "matched")
        self.assertEqual(result.split, "test")

    def test_token_sequence_fallback_when_identifier_token_mismatch(self):
        hs = make_human_sentence("S-1", "pangyo1", "A B", ["A", "B"])
        stale = make_split_sentence("pangyo1", "jsonl_1", "train", "A B C", ["A", "B", "C"])
        correct = make_split_sentence("pangyo2", "jsonl_2", "test", "A B", ["A", "B"])
        sentid_index = {"pangyo1": [stale]}
        token_seq_index = {("A", "B"): [correct]}
        result = audit.match_human_sentence(hs, sentid_index, token_seq_index, {})
        self.assertEqual(result.method, "token_sequence_exact")
        self.assertEqual(result.split, "test")
        self.assertIn("token sequence differs", result.notes)

    def test_sentence_text_fallback_used_only_when_no_stronger_match(self):
        hs = make_human_sentence("S-1", "pangyo999", "same text", ["A", "B"])
        cand = make_split_sentence("pangyo1", "jsonl_1", "dev", "same text", ["X", "Y", "Z"])
        text_index = {"same text": [cand]}
        result = audit.match_human_sentence(hs, {}, {}, text_index)
        self.assertEqual(result.method, "sentence_text_exact_fallback")
        self.assertEqual(result.status, "matched")
        self.assertEqual(result.split, "dev")

    def test_duplicate_sentence_text_across_documents_is_ambiguous_when_tokens_dont_disambiguate(self):
        hs = make_human_sentence("S-1", "pangyo999", "same text", ["A", "B"])
        cand1 = make_split_sentence("pangyo1", "jsonl_1", "train", "same text", ["X", "Y"])
        cand2 = make_split_sentence("pangyo2", "jsonl_2", "test", "same text", ["P", "Q"])
        text_index = {"same text": [cand1, cand2]}
        result = audit.match_human_sentence(hs, {}, {}, text_index)
        self.assertEqual(result.method, "sentence_text_ambiguous")
        self.assertEqual(result.status, "ambiguous")
        self.assertIsNone(result.split)

    def test_duplicate_sentence_text_across_documents_resolved_by_token_sequence(self):
        # Two docs share the exact same sentence text, but only one shares the exact token sequence.
        hs = make_human_sentence("S-1", "pangyo999", "same text", ["A", "B"])
        cand1 = make_split_sentence("pangyo1", "jsonl_1", "train", "same text", ["X", "Y"])
        cand2 = make_split_sentence("pangyo2", "jsonl_2", "test", "same text", ["A", "B"])
        token_seq_index = {("A", "B"): [cand2]}
        text_index = {"same text": [cand1, cand2]}
        result = audit.match_human_sentence(hs, {}, token_seq_index, text_index)
        self.assertEqual(result.method, "token_sequence_exact")
        self.assertEqual(result.status, "matched")
        self.assertEqual(result.split, "test")

    def test_ambiguous_token_sequence_match_fails_rather_than_guesses(self):
        hs = make_human_sentence("S-1", "pangyo999", "text a", ["A", "B"])
        cand1 = make_split_sentence("pangyo1", "jsonl_1", "train", "text a", ["A", "B"])
        cand2 = make_split_sentence("pangyo2", "jsonl_2", "dev", "text b", ["A", "B"])
        token_seq_index = {("A", "B"): [cand1, cand2]}
        result = audit.match_human_sentence(hs, {}, token_seq_index, {})
        self.assertEqual(result.method, "token_sequence_ambiguous")
        self.assertEqual(result.status, "ambiguous")
        self.assertIsNone(result.split)
        self.assertIsNone(result.doc_id)

    def test_ambiguous_identifier_match_fails_rather_than_guesses(self):
        hs = make_human_sentence("S-1", "pangyo1", "text a", ["A", "B"])
        cand1 = make_split_sentence("pangyo1", "jsonl_1", "train", "text a", ["A", "B"])
        cand2 = make_split_sentence("pangyo1", "jsonl_2", "dev", "text a", ["A", "B"])
        sentid_index = {"pangyo1": [cand1, cand2]}
        result = audit.match_human_sentence(hs, sentid_index, {}, {})
        self.assertEqual(result.method, "sent_id_ambiguous")
        self.assertEqual(result.status, "ambiguous")

    def test_unmatched_sentence_when_no_method_finds_a_candidate(self):
        hs = make_human_sentence("S-1", "pangyo999", "nowhere to be found", ["A", "B"])
        result = audit.match_human_sentence(hs, {}, {}, {})
        self.assertEqual(result.method, "none")
        self.assertEqual(result.status, "unmatched")
        self.assertIsNone(result.split)
        self.assertIsNone(result.doc_id)


class TestDocToSplitsAndDocMap(unittest.TestCase):
    def test_build_doc_to_splits_flags_document_spanning_two_splits(self):
        sents = [
            make_split_sentence("pangyo1", "jsonl_1", "train", "t1", ["A"]),
            make_split_sentence("pangyo2", "jsonl_1", "dev", "t2", ["B"]),
            make_split_sentence("pangyo3", "jsonl_2", "test", "t3", ["C"]),
        ]
        doc_to_splits = audit.build_doc_to_splits(sents)
        self.assertEqual(doc_to_splits["jsonl_1"], {"train", "dev"})
        self.assertEqual(doc_to_splits["jsonl_2"], {"test"})

    def test_find_doc_id_map_mismatches(self):
        sents = [make_split_sentence("pangyo1", "jsonl_1", "train", "t1", ["A"])]
        doc_map = {"pangyo1": "jsonl_2"}
        mismatches = audit.find_doc_id_map_mismatches(sents, doc_map)
        self.assertEqual(mismatches, [("pangyo1", "jsonl_1", "jsonl_2")])

    def test_find_doc_id_map_mismatches_empty_when_consistent(self):
        sents = [make_split_sentence("pangyo1", "jsonl_1", "train", "t1", ["A"])]
        doc_map = {"pangyo1": "jsonl_1"}
        self.assertEqual(audit.find_doc_id_map_mismatches(sents, doc_map), [])


class TestIndexBuilders(unittest.TestCase):
    def test_build_sentid_index_groups_by_sent_id(self):
        sents = [
            make_split_sentence("pangyo1", "jsonl_1", "train", "t1", ["A"]),
            make_split_sentence("pangyo1", "jsonl_2", "dev", "t1", ["A"]),
        ]
        idx = audit.build_sentid_index(sents)
        self.assertEqual(len(idx["pangyo1"]), 2)

    def test_build_token_seq_index_groups_by_forms(self):
        sents = [
            make_split_sentence("pangyo1", "jsonl_1", "train", "t1", ["A", "B"]),
            make_split_sentence("pangyo2", "jsonl_2", "dev", "t2", ["A", "B"]),
        ]
        idx = audit.build_token_seq_index(sents)
        self.assertEqual(len(idx[("A", "B")]), 2)


class TestSentenceCsvWriter(unittest.TestCase):
    def test_sentence_csv_has_required_columns_and_rows(self):
        import csv
        import tempfile

        hs = make_human_sentence("S-1", "pangyo1", "A B", ["A", "B"])
        result = audit.MatchResult(method="sent_id_exact", status="matched", split="dev", doc_id="jsonl_1", notes="")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.csv"
            audit.write_sentence_csv([hs], {"S-1": result}, path)
            with path.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        for col in (
            "sample_set", "sample_id", "source_sent_id", "document_id", "split",
            "sentence_token_count", "matching_method", "match_status", "notes",
        ):
            self.assertIn(col, row)
        self.assertEqual(row["split"], "dev")
        self.assertEqual(row["sentence_token_count"], "2")
        self.assertEqual(row["match_status"], "matched")


if __name__ == "__main__":
    unittest.main()
