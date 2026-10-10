"""
test_reconstruct_pangyo_conllu.py - Unit tests for scripts/reconstruct_pangyo_conllu.py.

Stdlib unittest only. Run from repo root:
  python scripts/tests/test_reconstruct_pangyo_conllu.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import reconstruct_pangyo_conllu as rc  # noqa: E402

ROOT = Path(__file__).parent.parent.parent
JSONL_PATH = ROOT / "data" / "kor_en_panyo_corpus.jsonl"


class TestLangClassify(unittest.TestCase):
    def test_hangul_only_is_ko(self):
        self.assertEqual(rc.classify_lang("전세계의"), "KO")

    def test_latin_only_is_en(self):
        self.assertEqual(rc.classify_lang("countries"), "EN")

    def test_mixed_script_is_mixed(self):
        self.assertEqual(rc.classify_lang("Python으로"), "MIXED")

    def test_digits_and_punct_are_other(self):
        self.assertEqual(rc.classify_lang("206"), "OTHER")
        self.assertEqual(rc.classify_lang("."), "OTHER")


class TestSentenceSegmentation(unittest.TestCase):
    def test_splits_on_period_plus_space(self):
        self.assertEqual(
            rc.segment_sentences("First sentence. Second sentence."),
            ["First sentence.", "Second sentence."],
        )

    def test_does_not_split_decimal_numbers(self):
        segs = rc.segment_sentences("The value is 1.4 meters. Next sentence.")
        self.assertEqual(len(segs), 2)
        self.assertIn("1.4", segs[0])

    def test_collapses_internal_whitespace(self):
        segs = rc.segment_sentences("A   b\tc.")
        self.assertEqual(segs, ["A b c."])

    def test_empty_text_yields_no_sentences(self):
        self.assertEqual(rc.segment_sentences("   "), [])


class TestTokenizer(unittest.TestCase):
    def test_splits_at_script_boundary_ko_suffix(self):
        toks = [t.form for t in rc.tokenize_sentence("document는")]
        self.assertEqual(toks, ["document", "는"])

    def test_splits_digit_and_hangul_counter(self):
        toks = [t.form for t in rc.tokenize_sentence("206개")]
        self.assertEqual(toks, ["206", "개"])

    def test_keeps_decimal_number_whole(self):
        toks = [t.form for t in rc.tokenize_sentence("1.4 meters")]
        self.assertEqual(toks[0], "1.4")

    def test_keeps_hyphenated_latin_whole(self):
        toks = [t.form for t in rc.tokenize_sentence("L-amino acid")]
        self.assertEqual(toks[0], "L-amino")

    def test_space_after_false_within_word_true_between_words(self):
        toks = rc.tokenize_sentence("document는 있습니다")
        # "document" is followed immediately by "는" (no space) -> space_after False
        self.assertFalse(toks[0].space_after)
        # "는" is followed by a real space before "있습니다" -> space_after True
        self.assertTrue(toks[1].space_after)

    def test_punctuation_is_single_char_tokens(self):
        toks = [t.form for t in rc.tokenize_sentence("hello, world!")]
        self.assertIn(",", toks)
        self.assertIn("!", toks)
        self.assertTrue(all(len(t) == 1 for t in toks if t in (",", "!")))


class TestUposAssignment(unittest.TestCase):
    def setUp(self):
        self.config = rc.DEFAULT_CONFIG

    def test_punctuation_is_punct(self):
        upos, rule = rc.assign_upos(".", is_sentence_initial=False, config=self.config)
        self.assertEqual(upos, "PUNCT")
        self.assertEqual(rule, "punct_is_punct")

    def test_digit_is_num(self):
        upos, rule = rc.assign_upos("206", is_sentence_initial=False, config=self.config)
        self.assertEqual(upos, "NUM")
        self.assertEqual(rule, "digit_is_num")

    def test_documented_particle_is_adp(self):
        upos, rule = rc.assign_upos("는", is_sentence_initial=False, config=self.config)
        self.assertEqual(upos, "ADP")
        self.assertEqual(rule, "ko_particle_adp")

    def test_da_final_korean_token_is_verb(self):
        upos, rule = rc.assign_upos("있습니다", is_sentence_initial=False, config=self.config)
        self.assertEqual(upos, "VERB")
        self.assertEqual(rule, "da_final_verb")

    def test_da_final_verb_rule_is_configurable(self):
        cfg = rc.ReconstructionConfig(apply_da_final_verb=False)
        upos, rule = rc.assign_upos("있습니다", is_sentence_initial=False, config=cfg)
        self.assertEqual(upos, "NOUN")
        self.assertEqual(rule, "noun_fallback")

    def test_english_function_word_the_is_det(self):
        upos, rule = rc.assign_upos("The", is_sentence_initial=True, config=self.config)
        self.assertEqual(upos, "DET")
        self.assertEqual(rule, "en_function_words")

    def test_english_preposition_of_is_adp(self):
        upos, _ = rc.assign_upos("of", is_sentence_initial=False, config=self.config)
        self.assertEqual(upos, "ADP")

    def test_capitalized_noninitial_english_is_propn(self):
        upos, rule = rc.assign_upos("Montevideo", is_sentence_initial=False, config=self.config)
        self.assertEqual(upos, "PROPN")
        self.assertEqual(rule, "propn_capitalized")

    def test_capitalized_sentence_initial_english_is_not_forced_propn(self):
        # sentence-initial capitalization must not trigger propn_capitalized
        upos, rule = rc.assign_upos("Montevideo", is_sentence_initial=True, config=self.config)
        self.assertNotEqual(rule, "propn_capitalized")

    def test_propn_rule_is_configurable(self):
        cfg = rc.ReconstructionConfig(apply_propn_capitalization=False)
        upos, rule = rc.assign_upos("Montevideo", is_sentence_initial=False, config=cfg)
        self.assertEqual(upos, "NOUN")
        self.assertEqual(rule, "noun_fallback")

    def test_lexical_korean_falls_back_to_noun(self):
        upos, rule = rc.assign_upos("국가", is_sentence_initial=False, config=self.config)
        self.assertEqual(upos, "NOUN")
        self.assertEqual(rule, "noun_fallback")


class TestLemmaAssignment(unittest.TestCase):
    def test_capitalized_english_lemma_is_lowercased(self):
        self.assertEqual(rc.assign_lemma("Montevideo", "PROPN"), "montevideo")

    def test_lowercase_english_lemma_unchanged(self):
        self.assertEqual(rc.assign_lemma("countries", "NOUN"), "countries")

    def test_korean_lemma_identity(self):
        self.assertEqual(rc.assign_lemma("전세계의", "NOUN"), "전세계의")


class TestSentenceAssemblyAndSerialization(unittest.TestCase):
    def test_flat_dependency_root_is_first_token(self):
        sent = rc.build_sentence("This document는 있습니다.", "jsonl_1", "pangyo1", rc.DEFAULT_CONFIG)
        text = rc.serialize_sentence(sent, rc.DEFAULT_CONFIG)
        lines = [l for l in text.splitlines() if l and not l.startswith("#")]
        first_fields = lines[0].split("\t")
        self.assertEqual(first_fields[6], "0")   # HEAD
        self.assertEqual(first_fields[7], "root")
        for line in lines[1:]:
            fields = line.split("\t")
            if fields[3] == "PUNCT":
                self.assertEqual(fields[7], "punct")
            else:
                self.assertEqual(fields[7], "dep")
            self.assertEqual(fields[6], "1")  # all attach to token 1

    def test_serialization_is_valid_10_column_conllu(self):
        sent = rc.build_sentence("hello world.", "jsonl_1", "pangyo1", rc.DEFAULT_CONFIG)
        text = rc.serialize_sentence(sent, rc.DEFAULT_CONFIG)
        lines = [l for l in text.splitlines() if l and not l.startswith("#")]
        for line in lines:
            self.assertEqual(len(line.split("\t")), 10)

    def test_doc_id_and_sent_id_comments_present(self):
        sent = rc.build_sentence("hello.", "jsonl_42", "pangyo7", rc.DEFAULT_CONFIG)
        text = rc.serialize_sentence(sent, rc.DEFAULT_CONFIG)
        self.assertIn("# doc_id = jsonl_42", text)
        self.assertIn("# sent_id = pangyo7", text)


class TestJsonlLoadingFailsClearly(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            rc.load_jsonl(self.tmp_path / "does_not_exist.jsonl")

    def test_malformed_json_raises_with_line_number(self):
        p = self.tmp_path / "bad.jsonl"
        p.write_text('{"pangyo": "ok"}\n{not valid json\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, r":2:"):
            rc.load_jsonl(p)

    def test_missing_pangyo_field_raises(self):
        p = self.tmp_path / "missing_field.jsonl"
        p.write_text('{"korean": "only korean field"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "pangyo"):
            rc.load_jsonl(p)

    def test_empty_pangyo_field_raises(self):
        p = self.tmp_path / "empty_field.jsonl"
        p.write_text('{"pangyo": "   "}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            rc.load_jsonl(p)

    def test_valid_record_loads(self):
        p = self.tmp_path / "good.jsonl"
        p.write_text('{"korean": "k", "pangyo": "Hello world."}\n', encoding="utf-8")
        records = rc.load_jsonl(p)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].line_no, 1)
        self.assertEqual(records[0].pangyo, "Hello world.")


class TestReconstructCorpusDocIdScheme(unittest.TestCase):
    def test_doc_id_uses_jsonl_line_number(self):
        records = [
            rc.JsonlRecord(line_no=1, pangyo="First doc sentence one. Sentence two."),
            rc.JsonlRecord(line_no=2, pangyo="Second doc sentence."),
        ]
        sentences = rc.reconstruct_corpus(records)
        doc_ids = {s.doc_id for s in sentences}
        self.assertEqual(doc_ids, {"jsonl_1", "jsonl_2"})

    def test_sent_id_is_sequential_across_corpus(self):
        records = [
            rc.JsonlRecord(line_no=5, pangyo="One. Two. Three."),
        ]
        sentences = rc.reconstruct_corpus(records)
        self.assertEqual([s.sent_id for s in sentences], ["pangyo1", "pangyo2", "pangyo3"])


@unittest.skipUnless(JSONL_PATH.exists(), "real Panyo JSONL not found")
class TestRealJsonlIntegration(unittest.TestCase):
    """Lightweight integration check against the real corpus (read-only)."""

    def test_loads_and_reconstructs_first_record_without_error(self):
        records = rc.load_jsonl(JSONL_PATH)
        self.assertGreater(len(records), 8000)
        sentences = rc.reconstruct_corpus(records[:1])
        self.assertGreater(len(sentences), 0)
        self.assertTrue(all(s.doc_id == "jsonl_1" for s in sentences))


if __name__ == "__main__":
    unittest.main(verbosity=2)
