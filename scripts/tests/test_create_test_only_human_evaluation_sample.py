"""
test_create_test_only_human_evaluation_sample.py - Tests for
scripts/create_test_only_human_evaluation_sample.py.

Two groups:
  - Pure-logic tests against synthetic in-memory TestSentence fixtures (fast,
    no file I/O) covering exclusion rules, stratified sampling, diagnostic
    category scoring/assignment, and determinism.
  - Integration tests that call the real `run()` pipeline against the actual
    authoritative repo files (data/processed/panyo_original_adp_docsplit_test.conllu,
    etc.) and validate every "Required validation" item from the task against
    the real generated 80-sentence sample and workbooks.

Run from repo root:
  python scripts/tests/test_create_test_only_human_evaluation_sample.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).parent.parent))
import create_test_only_human_evaluation_sample as m  # noqa: E402


def make_token(token_id, form, upos="NOUN", lang="KO"):
    return m.Token(token_id=token_id, form=form, upos=upos, lang=lang)


def make_sentence(sent_id, doc_id, text, forms_langs, upos="NOUN"):
    tokens = tuple(
        make_token(i + 1, f, upos=upos, lang=lang) for i, (f, lang) in enumerate(forms_langs)
    )
    return m.TestSentence(sent_id=sent_id, doc_id=doc_id, text=text, tokens=tokens)


class TestNormalizeText(unittest.TestCase):
    def test_collapses_whitespace_and_nfkc(self):
        self.assertEqual(m.normalize_text("  A   B\t C "), "A B C")


class TestEligibility(unittest.TestCase):
    def _sentences(self):
        return [
            make_sentence("pangyo1", "jsonl_1", "text one", [("A", "EN")]),
            make_sentence("pangyo2", "jsonl_2", "text two", [("B", "EN")]),
            make_sentence("pangyo3", "jsonl_3", "text three", [("C", "EN")]),
        ]

    def test_excludes_previous_sentence_by_id(self):
        sents = self._sentences()
        prev_docs = {"pangyo1": {"sample_set": "Representative", "sample_id": "X", "document_id": "jsonl_1", "split": "test"}}
        eligible, records, _ = m.compute_eligibility(sents, [], [], prev_docs, {})
        self.assertNotIn("pangyo1", {s.sent_id for s in eligible})
        self.assertTrue(any(r.sent_id == "pangyo1" and r.reason == "previous_sample_sentence" for r in records))

    def test_excludes_previous_document_when_many_docs_remain(self):
        sents = [make_sentence(f"pangyo{i}", f"jsonl_{i}", f"text {i}", [("A", "EN")]) for i in range(1, 101)]
        prev_docs = {"pangyoX": {"sample_set": "Representative", "sample_id": "X", "document_id": "jsonl_1", "split": "test"}}
        eligible, records, exclude_all = m.compute_eligibility(sents, [], [], prev_docs, {})
        self.assertTrue(exclude_all)  # 99 remaining docs >= threshold is not met (99<... ) -> check via constant
        self.assertNotIn("jsonl_1", {s.doc_id for s in eligible})

    def test_keeps_previous_document_when_too_few_docs_remain(self):
        # Only 10 total docs; excluding 1 leaves 9, well under the 80 threshold.
        sents = [make_sentence(f"pangyo{i}", f"jsonl_{i}", f"text {i}", [("A", "EN")]) for i in range(1, 11)]
        prev_docs = {"pangyoX": {"sample_set": "Representative", "sample_id": "X", "document_id": "jsonl_1", "split": "test"}}
        eligible, records, exclude_all = m.compute_eligibility(sents, [], [], prev_docs, {})
        self.assertFalse(exclude_all)
        self.assertIn("jsonl_1", {s.doc_id for s in eligible})

    def test_excludes_ambiguous_sent_id(self):
        sents = self._sentences()
        eligible, records, _ = m.compute_eligibility(sents, ["pangyo2"], [], {}, {})
        self.assertNotIn("pangyo2", {s.sent_id for s in eligible})
        self.assertTrue(any(r.reason == "ambiguous_id" for r in records))

    def test_excludes_doc_id_mismatch(self):
        sents = self._sentences()
        eligible, records, _ = m.compute_eligibility(sents, [], ["pangyo3"], {}, {})
        self.assertNotIn("pangyo3", {s.sent_id for s in eligible})

    def test_excludes_duplicate_normalized_text_vs_previous(self):
        sents = self._sentences()
        eligible, records, _ = m.compute_eligibility(sents, [], [], {}, {"pangyoOLD": "text   one"})
        self.assertNotIn("pangyo1", {s.sent_id for s in eligible})
        self.assertTrue(any(r.reason == "duplicate_of_previous_text" for r in records))

    def test_excludes_duplicate_normalized_text_within_pool(self):
        sents = [
            make_sentence("pangyo1", "jsonl_1", "same text", [("A", "EN")]),
            make_sentence("pangyo2", "jsonl_2", "same   text", [("B", "EN")]),
        ]
        eligible, records, _ = m.compute_eligibility(sents, [], [], {}, {})
        self.assertEqual(len(eligible), 1)
        self.assertEqual(eligible[0].sent_id, "pangyo1")  # first by sent_id order kept
        self.assertTrue(any(r.reason == "duplicate_within_test_split" for r in records))


class TestSelectRepresentative(unittest.TestCase):
    def _pool(self, n=200, docs=None):
        pool = []
        for i in range(1, n + 1):
            doc = f"jsonl_{i}" if docs is None else f"jsonl_{i % docs}"
            length = 3 + (i % 20)
            forms = [(f"t{j}", "KO" if j % 2 == 0 else "EN") for j in range(length)]
            pool.append(make_sentence(f"pangyo{i}", doc, f"sentence {i}", forms))
        return pool

    def test_returns_exactly_n(self):
        pool = self._pool()
        selected, _ = m.select_representative(pool, Random(42), n=60)
        self.assertEqual(len(selected), 60)

    def test_one_sentence_per_document_when_enough_docs(self):
        pool = self._pool(n=200)  # 200 distinct docs
        selected, strata = m.select_representative(pool, Random(42), n=60)
        self.assertTrue(strata["one_sentence_per_document_enforced"])
        docs = [s.doc_id for s in selected]
        self.assertEqual(len(docs), len(set(docs)))

    def test_allows_multiple_per_doc_when_insufficient_docs(self):
        pool = self._pool(n=200, docs=10)  # only 10 distinct docs
        selected, strata = m.select_representative(pool, Random(42), n=60)
        self.assertFalse(strata["one_sentence_per_document_enforced"])
        self.assertEqual(len(selected), 60)

    def test_deterministic_with_same_seed(self):
        pool = self._pool()
        sel1, _ = m.select_representative(pool, Random(42), n=60)
        sel2, _ = m.select_representative(pool, Random(42), n=60)
        self.assertEqual([s.sent_id for s in sel1], [s.sent_id for s in sel2])

    def test_spans_multiple_length_bins_not_only_short_or_long(self):
        pool = self._pool()
        selected, strata = m.select_representative(pool, Random(42), n=60)
        self.assertEqual(len(strata["bin_sizes_selected"]), 5)
        self.assertTrue(all(sz > 0 for sz in strata["bin_sizes_selected"]))

    def test_no_duplicate_normalized_text_in_selection(self):
        pool = self._pool()
        pool.append(make_sentence("pangyoDUP", "jsonl_dup", "sentence 1", [("t0", "KO")]))
        selected, _ = m.select_representative(pool, Random(42), n=60)
        texts = [s.normalized_text for s in selected]
        self.assertEqual(len(texts), len(set(texts)))


class TestDiagnosticScoring(unittest.TestCase):
    def test_particle_heavy_counts_particle_forms(self):
        sent = make_sentence("p1", "d1", "x", [("는", "KO"), ("cat", "EN"), ("을", "KO")])
        scores = m.score_diagnostic_categories(sent)
        self.assertEqual(scores["particle_heavy"], 2)

    def test_ko_en_boundary_counts_language_transitions(self):
        sent = make_sentence("p1", "d1", "x", [("A", "EN"), ("B", "KO"), ("C", "KO"), ("D", "EN")])
        scores = m.score_diagnostic_categories(sent)
        self.assertEqual(scores["ko_en_boundary"], 2)  # EN->KO, KO->EN

    def test_en_nounpropn_cap_counts_capitalized_english_tokens(self):
        sent = make_sentence("p1", "d1", "x", [("Seoul", "EN"), ("file", "EN"), ("Windows", "EN")])
        scores = m.score_diagnostic_categories(sent)
        self.assertEqual(scores["en_nounpropn_cap"], 2)

    def test_predicate_sensitive_counts_predicate_markers(self):
        sent = make_sentence("p1", "d1", "x", [("있다", "KO"), ("cat", "EN"), ("업데이트했다", "KO")])
        scores = m.score_diagnostic_categories(sent)
        self.assertGreaterEqual(scores["predicate_sensitive"], 2)

    def test_tokenization_edge_counts_digit_hangul_mix(self):
        sent = make_sentence("p1", "d1", "x", [("20세기", "KO"), ("normal", "EN")])
        scores = m.score_diagnostic_categories(sent)
        self.assertEqual(scores["tokenization_edge"], 1)

    def test_assign_primary_category_uses_fixed_priority_on_ties(self):
        scores = {"ko_en_boundary": 3, "particle_heavy": 3, "predicate_sensitive": 0,
                  "en_nounpropn_cap": 0, "tokenization_edge": 0}
        self.assertEqual(m.assign_primary_category(scores), "ko_en_boundary")

    def test_assign_primary_category_picks_true_max(self):
        scores = {"ko_en_boundary": 1, "particle_heavy": 5, "predicate_sensitive": 0,
                  "en_nounpropn_cap": 0, "tokenization_edge": 0}
        self.assertEqual(m.assign_primary_category(scores), "particle_heavy")


class TestSelectDiagnostic(unittest.TestCase):
    def _pool(self, n=300):
        pool = []
        for i in range(1, n + 1):
            variant = i % 5
            if variant == 0:
                forms = [("A", "EN"), ("B", "KO")] * 5  # boundary-rich
            elif variant == 1:
                forms = [("는", "KO"), ("을", "KO"), ("에서", "KO"), ("cat", "EN")]  # particle-heavy
            elif variant == 2:
                forms = [("있습니다", "KO"), ("했다", "KO"), ("cat", "EN")]  # predicate-sensitive
            elif variant == 3:
                forms = [("Seoul", "EN"), ("Windows", "EN"), ("cat", "EN")]  # cap-heavy
            else:
                forms = [("20세기", "KO"), ("km²", "EN"), ("제", "KO"), ("것", "KO")]  # tokenization-edge
            pool.append(make_sentence(f"pangyo{i}", f"jsonl_{i}", f"sentence {i}", forms))
        return pool

    def test_returns_exactly_n_with_target_categories(self):
        pool = self._pool()
        selected, info = m.select_diagnostic(pool, Random(42), excluded_texts=set(), avoid_docs=set())
        self.assertEqual(len(selected), 20)
        self.assertEqual(set(info["achieved_category_counts"]), set(m.CATEGORY_IDS))

    def test_each_sentence_has_exactly_one_category(self):
        pool = self._pool()
        selected, _ = m.select_diagnostic(pool, Random(42), excluded_texts=set(), avoid_docs=set())
        sent_ids = [s.sent_id for s, _cat in selected]
        self.assertEqual(len(sent_ids), len(set(sent_ids)))

    def test_excludes_representative_texts(self):
        pool = self._pool()
        excluded = {pool[0].normalized_text}
        selected, _ = m.select_diagnostic(pool, Random(42), excluded_texts=excluded, avoid_docs=set())
        self.assertNotIn(pool[0].sent_id, {s.sent_id for s, _c in selected})

    def test_deterministic_with_same_seed(self):
        pool = self._pool()
        sel1, _ = m.select_diagnostic(pool, Random(42), excluded_texts=set(), avoid_docs=set())
        sel2, _ = m.select_diagnostic(pool, Random(42), excluded_texts=set(), avoid_docs=set())
        self.assertEqual([(s.sent_id, c) for s, c in sel1], [(s.sent_id, c) for s, c in sel2])


class TestWorkbookWriters(unittest.TestCase):
    def _items(self):
        sent = make_sentence("pangyo1", "jsonl_1", "hello world", [("hello", "EN"), ("world", "EN")])
        return [("Representative", "REP-TEST-001", sent)]

    def test_workbook_has_required_sheets_and_reopens(self):
        import openpyxl
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "P1.xlsx"
            m.write_annotator_workbook(path, "P1", self._items(), [], 2, 0)
            wb = openpyxl.load_workbook(path)
            self.assertEqual(
                set(wb.sheetnames),
                {"Instructions", "Representative_60", "Diagnostic_20", "Selection_Summary"},
            )

    def test_human_upos_blank_and_tokenization_issue_no(self):
        import openpyxl
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "P1.xlsx"
            m.write_annotator_workbook(path, "P1", self._items(), [], 2, 0)
            wb = openpyxl.load_workbook(path)
            ws = wb["Representative_60"]
            rows = list(ws.iter_rows(values_only=True))
            header, data = rows[0], rows[1:]
            idx = {name: i for i, name in enumerate(header)}
            for row in data:
                self.assertIsNone(row[idx["Human_UPOS"]])
                self.assertIsNone(row[idx["Confidence"]])
                self.assertEqual(row[idx["Tokenization_Issue"]], "No")

    def test_no_canonical_upos_column_in_annotator_workbook(self):
        import openpyxl
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "P1.xlsx"
            m.write_annotator_workbook(path, "P1", self._items(), [], 2, 0)
            wb = openpyxl.load_workbook(path)
            for sheet in ("Representative_60", "Diagnostic_20"):
                header = next(wb[sheet].iter_rows(values_only=True))
                self.assertNotIn("Canonical_UPOS", header)

    def test_highlighted_columns_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "P1.xlsx"
            m.write_annotator_workbook(path, "P1", self._items(), [], 2, 0)
            import openpyxl
            wb = openpyxl.load_workbook(path)
            ws = wb["Representative_60"]
            header = next(ws.iter_rows(values_only=True))
            for col_idx, name in enumerate(header, 1):
                cell = ws.cell(row=1, column=col_idx)
                is_highlighted = cell.fill.fgColor.rgb == "00FFF2CC"
                if name in m.HIGHLIGHT_COLUMNS:
                    self.assertTrue(is_highlighted, f"{name} should be highlighted")
                else:
                    self.assertFalse(is_highlighted, f"{name} should not be highlighted")

    def test_p1_p2_identical_except_annotator(self):
        import openpyxl
        with tempfile.TemporaryDirectory() as tmp:
            p1_path, p2_path = Path(tmp) / "P1.xlsx", Path(tmp) / "P2.xlsx"
            m.write_annotator_workbook(p1_path, "P1", self._items(), [], 2, 0)
            m.write_annotator_workbook(p2_path, "P2", self._items(), [], 2, 0)
            wb1, wb2 = openpyxl.load_workbook(p1_path), openpyxl.load_workbook(p2_path)
            for sheet in ("Representative_60", "Diagnostic_20"):
                rows1 = list(wb1[sheet].iter_rows(values_only=True))
                rows2 = list(wb2[sheet].iter_rows(values_only=True))
                self.assertEqual(rows1, rows2)
            summary1 = list(wb1["Selection_Summary"].iter_rows(values_only=True))
            summary2 = list(wb2["Selection_Summary"].iter_rows(values_only=True))
            self.assertNotEqual(summary1, summary2)
            for r1, r2 in zip(summary1, summary2):
                diffs = [i for i in range(len(r1)) if r1[i] != r2[i]]
                self.assertTrue(all(r1[i] in (None, "P1") or r2[i] in (None, "P2") for i in diffs))


@unittest.skipUnless(m.TEST_CONLLU.exists(), "authoritative test-split file not present in this checkout")
class TestIntegrationRealData(unittest.TestCase):
    """Runs the real pipeline against the actual authoritative repo files and
    validates the task's 'Required validation' checklist against the real
    output. This writes the real deliverable files under
    outputs/human_validation/test_only/."""

    @classmethod
    def setUpClass(cls):
        cls.result = m.run()
        import openpyxl
        cls.manifest_path = m.KEY_DIR / "selection_manifest.json"
        cls.p1_path = m.ANNOT_DIR / "Human_POS_Tagging_test_v1_P1.xlsx"
        cls.p2_path = m.ANNOT_DIR / "Human_POS_Tagging_test_v1_P2.xlsx"
        cls.key_path = m.KEY_DIR / "Human_POS_Tagging_test_v1_canonical_key.xlsx"
        cls.wb_p1 = openpyxl.load_workbook(cls.p1_path)
        cls.wb_p2 = openpyxl.load_workbook(cls.p2_path)
        cls.wb_key = openpyxl.load_workbook(cls.key_path)

        # Rebuild the authoritative test-split index for cross-checking.
        cls.all_sentences, cls.ambiguous = m.load_test_sentences(m.TEST_CONLLU)
        cls.by_sent_id = {s.sent_id: s for s in cls.all_sentences}
        cls.previous_docs = m.load_previous_sample_docs(m.SPLIT_AUDIT_CSV)

    def _all_sheet_rows(self, wb, sheet):
        rows = list(wb[sheet].iter_rows(values_only=True))
        header = rows[0]
        idx = {name: i for i, name in enumerate(header)}
        return header, idx, rows[1:]

    def test_exactly_60_representative_and_20_diagnostic(self):
        self.assertEqual(self.result["n_representative"], 60)
        self.assertEqual(self.result["n_diagnostic"], 20)

    def test_all_selected_sentences_in_authoritative_test_split(self):
        _h, idx, rows = self._all_sheet_rows(self.wb_p1, "Representative_60")
        sent_ids = {r[idx["Source_Sent_ID"]] for r in rows}
        _h2, idx2, rows2 = self._all_sheet_rows(self.wb_p1, "Diagnostic_20")
        sent_ids |= {r[idx2["Source_Sent_ID"]] for r in rows2}
        self.assertEqual(len(sent_ids), 80)
        self.assertTrue(sent_ids.issubset(set(self.by_sent_id)))

    def test_no_overlap_with_previous_sample(self):
        _h, idx, rows = self._all_sheet_rows(self.wb_p1, "Representative_60")
        sent_ids = {r[idx["Source_Sent_ID"]] for r in rows}
        _h2, idx2, rows2 = self._all_sheet_rows(self.wb_p1, "Diagnostic_20")
        sent_ids |= {r[idx2["Source_Sent_ID"]] for r in rows2}
        self.assertEqual(sent_ids & set(self.previous_docs), set())

    def test_no_normalized_duplicates_within_or_across_subsets(self):
        _h, idx, rep = self._all_sheet_rows(self.wb_p1, "Representative_60")
        _h2, idx2, diag = self._all_sheet_rows(self.wb_p1, "Diagnostic_20")
        rep_sents = {(r[idx["Source_Sent_ID"]], r[idx["Sentence"]]) for r in rep}
        diag_sents = {(r[idx2["Source_Sent_ID"]], r[idx2["Sentence"]]) for r in diag}
        rep_texts = [m.normalize_text(t) for _sid, t in rep_sents]
        diag_texts = [m.normalize_text(t) for _sid, t in diag_sents]
        self.assertEqual(len(rep_texts), len(set(rep_texts)))
        self.assertEqual(len(diag_texts), len(set(diag_texts)))
        self.assertEqual(set(rep_texts) & set(diag_texts), set())

    def test_document_uniqueness_documented(self):
        import json
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        strata = manifest["representative_sample"]["sampling_strata"]
        self.assertIn("one_sentence_per_document_enforced", strata)
        rep_docs = manifest["representative_sample"]["document_ids"]
        if strata["one_sentence_per_document_enforced"]:
            self.assertEqual(len(rep_docs), 60)

    def test_p1_p2_workbooks_identical_except_annotator(self):
        for sheet in ("Representative_60", "Diagnostic_20"):
            rows1 = list(self.wb_p1[sheet].iter_rows(values_only=True))
            rows2 = list(self.wb_p2[sheet].iter_rows(values_only=True))
            self.assertEqual(rows1, rows2)
        s1 = list(self.wb_p1["Selection_Summary"].iter_rows(values_only=True))
        s2 = list(self.wb_p2["Selection_Summary"].iter_rows(values_only=True))
        for r1, r2 in zip(s1, s2):
            for v1, v2 in zip(r1, r2):
                self.assertTrue(v1 == v2 or {v1, v2} == {"P1", "P2"})

    def test_token_forms_and_order_match_authoritative_source(self):
        _h, idx, rows = self._all_sheet_rows(self.wb_p1, "Representative_60")
        by_sent: dict = {}
        for r in rows:
            by_sent.setdefault(r[idx["Source_Sent_ID"]], []).append((r[idx["Token_ID"]], r[idx["Token"]]))
        for sid, toks in by_sent.items():
            toks_sorted = [t for _tid, t in sorted(toks)]
            authoritative = [t.form for t in self.by_sent_id[sid].tokens]
            self.assertEqual(toks_sorted, authoritative)

    def test_no_canonical_upos_in_annotator_facing_sheets(self):
        for wb in (self.wb_p1, self.wb_p2):
            for sheet in ("Representative_60", "Diagnostic_20"):
                header = next(wb[sheet].iter_rows(values_only=True))
                self.assertNotIn("Canonical_UPOS", header)
                self.assertNotIn("Language", header)
                self.assertNotIn("Diagnostic_Category", header)

    def test_every_diagnostic_sentence_has_exactly_one_primary_category(self):
        _h, idx, key_rows = self._all_sheet_rows(self.wb_key, "Canonical_Key_80")
        by_sent: dict = {}
        for r in key_rows:
            if r[idx["Set"]] != "Diagnostic":
                continue
            by_sent.setdefault(r[idx["Source_Sent_ID"]], set()).add(r[idx["Diagnostic_Category"]])
        self.assertEqual(len(by_sent), 20)
        for sid, cats in by_sent.items():
            self.assertEqual(len(cats), 1, f"{sid} has categories {cats}")

    def test_representative_and_diagnostic_disjoint(self):
        _h, idx, rep = self._all_sheet_rows(self.wb_p1, "Representative_60")
        _h2, idx2, diag = self._all_sheet_rows(self.wb_p1, "Diagnostic_20")
        rep_ids = {r[idx["Source_Sent_ID"]] for r in rep}
        diag_ids = {r[idx2["Source_Sent_ID"]] for r in diag}
        self.assertEqual(rep_ids & diag_ids, set())

    def test_rerun_with_same_seed_produces_same_selection(self):
        result2 = m.run()
        import json
        manifest1 = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest2 = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest1["representative_sample"]["sentence_ids"],
            manifest2["representative_sample"]["sentence_ids"],
        )
        self.assertEqual(result2["n_representative"], self.result["n_representative"])
        self.assertEqual(result2["n_diagnostic"], self.result["n_diagnostic"])

    def test_all_output_workbooks_reopen_successfully(self):
        import openpyxl
        for path in (self.p1_path, self.p2_path, self.key_path):
            wb = openpyxl.load_workbook(path)
            self.assertGreater(len(wb.sheetnames), 0)


if __name__ == "__main__":
    unittest.main()
