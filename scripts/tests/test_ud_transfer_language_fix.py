"""
test_ud_transfer_language_fix.py - Validate the UD-transfer language-feature fix.

Proves, per the scientific requirement:
  - GUM training examples expose language = EN,
  - KAIST training examples expose language = KO,
  - Panyo inference uses EN and KO in exactly the same string representation
    as UD training (same feature name, capitalization, encoding),
  - no blank ("") language feature remains in UD training data.

Uses only the standard library (unittest). Run from repo root:
  python scripts/tests/test_ud_transfer_language_fix.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import train_baselines as tb  # noqa: E402
from read_conllu import get_misc_lang  # noqa: E402
from label_language import classify_lang  # noqa: E402
from perceptron_tagger import get_features  # noqa: E402

ROOT = Path(__file__).parent.parent.parent
GUM_TRAIN = ROOT / "data" / "en_gum-ud-train.conllu"
KAIST_TRAIN = ROOT / "data" / "ko_kaist-ud-train.conllu"
PANYO_TEST = ROOT / "data" / "processed" / "panyo_original_adp_docsplit_test.conllu"


def _write_conllu(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


# A minimal 2-sentence monolingual fixture standing in for GUM/KAIST: no
# Lang= in MISC at all (mirrors the real UD files, which never carry Lang=).
MONOLINGUAL_FIXTURE = (
    "# sent_id = s1\n"
    "1\tThe\tthe\tDET\t_\t_\t2\tdet\t_\t_\n"
    "2\tdog\tdog\tNOUN\t_\t_\t0\troot\t_\t_\n"
    "\n"
    "# sent_id = s2\n"
    "1\truns\trun\tVERB\t_\t_\t0\troot\t_\t_\n"
)


class TestLoadTaggedSentsLangOverride(unittest.TestCase):
    """Unit-level proof that the fixed data loader injects the override."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.fixture_path = Path(self._tmpdir.name) / "mono.conllu"
        _write_conllu(self.fixture_path, MONOLINGUAL_FIXTURE)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_no_override_reads_blank_lang_from_monolingual_misc(self):
        # Sanity check on the OLD behavior (no override = the pre-fix bug):
        # a corpus with no Lang= in MISC yields lang="" for every token.
        sents = tb.load_tagged_sents(self.fixture_path)
        langs = [lang for sent in sents for _, lang, _ in sent]
        self.assertTrue(all(l == "" for l in langs))

    def test_en_override_gives_every_token_lang_en(self):
        sents = tb.load_tagged_sents(self.fixture_path, lang_override="EN")
        langs = [lang for sent in sents for _, lang, _ in sent]
        self.assertEqual(len(langs), 3)
        self.assertTrue(all(l == "EN" for l in langs))
        self.assertNotIn("", langs)

    def test_ko_override_gives_every_token_lang_ko(self):
        sents = tb.load_tagged_sents(self.fixture_path, lang_override="KO")
        langs = [lang for sent in sents for _, lang, _ in sent]
        self.assertEqual(len(langs), 3)
        self.assertTrue(all(l == "KO" for l in langs))
        self.assertNotIn("", langs)


class TestSameRepresentationAsPanyoInference(unittest.TestCase):
    """Proves training-time and inference-time lang strings are identical:
    same feature name ('lang='), same capitalization ('EN'/'KO', not
    'en'/'ko' or 'English'/'Korean'), same type (str), same encoding (ASCII
    within the wider UTF-8 file)."""

    def test_override_values_match_classify_lang_vocabulary_exactly(self):
        # classify_lang() is the single source of truth for Panyo Lang=
        # values (label_language.py). The override constants used for GUM
        # and KAIST training must be members of that exact vocabulary.
        self.assertIn("EN", {classify_lang("hello"), classify_lang("Hello123")})
        self.assertIn("KO", {classify_lang("안녕")})
        # And must be exact string equality, not just case-insensitive match.
        self.assertEqual(classify_lang("hello"), "EN")
        self.assertEqual(classify_lang("안녕"), "KO")

    def test_get_misc_lang_roundtrip_matches_override_strings(self):
        # get_misc_lang() is what predict_baselines.py calls at inference
        # time for Panyo tokens (sent_to_real_tokens -> get_misc_lang).
        self.assertEqual(get_misc_lang("Lang=EN"), "EN")
        self.assertEqual(get_misc_lang("Lang=KO"), "KO")
        # These are exactly the strings passed via lang_override in the
        # fixed train_baselines.py E. section (UD_LANG_OVERRIDE mapping).

    def test_feature_string_identical_for_train_and_inference_lang(self):
        # The actual feature the perceptron sees is f"lang={lang}" (and the
        # p1.lang=/n1.lang=/p1.tag+lang= variants). Construct it both ways
        # and confirm byte-for-byte identical feature keys.
        feats_train = get_features(
            "dog", "EN", "the", "runs", "EN", "EN", "__START__", "__START__"
        )
        feats_infer = get_features(
            "dog", get_misc_lang("Lang=EN"), "the", "runs",
            get_misc_lang("Lang=EN"), get_misc_lang("Lang=EN"),
            "__START__", "__START__",
        )
        self.assertEqual(feats_train, feats_infer)
        self.assertIn("lang=EN", feats_train)

    @unittest.skipUnless(PANYO_TEST.exists(), "Panyo test docsplit file not found")
    def test_panyo_inference_lang_values_are_exactly_en_and_ko(self):
        # Confirms the real inference-side file only ever produces the
        # bare strings "EN"/"KO" (plus MIXED/OTHER) - never "en"/"ko" or
        # anything else - so the override must match these exactly.
        from read_conllu import iter_sentences, F_MISC
        seen = set()
        for sent in iter_sentences(PANYO_TEST):
            for tok in sent["tokens"]:
                if tok["is_real"]:
                    seen.add(get_misc_lang(tok["fields"][F_MISC]))
        self.assertTrue({"EN", "KO"}.issubset(seen))
        self.assertNotIn("en", seen)
        self.assertNotIn("ko", seen)
        self.assertNotIn("", seen)


@unittest.skipUnless(GUM_TRAIN.exists() and KAIST_TRAIN.exists(), "UD source files not found")
class TestRealCorporaNoBlankLangAfterFix(unittest.TestCase):
    """Integration check against the actual GUM/KAIST training files."""

    def test_gum_training_examples_expose_language_en(self):
        sents = tb.load_tagged_sents(GUM_TRAIN, lang_override="EN")
        self.assertGreater(len(sents), 0)
        langs = {lang for sent in sents for _, lang, _ in sent}
        self.assertEqual(langs, {"EN"})

    def test_kaist_training_examples_expose_language_ko(self):
        sents = tb.load_tagged_sents(KAIST_TRAIN, lang_override="KO")
        self.assertGreater(len(sents), 0)
        langs = {lang for sent in sents for _, lang, _ in sent}
        self.assertEqual(langs, {"KO"})

    def test_no_blank_language_feature_remains_in_ud_training(self):
        for path, override in [(GUM_TRAIN, "EN"), (KAIST_TRAIN, "KO")]:
            sents = tb.load_tagged_sents(path, lang_override=override)
            n_blank = sum(1 for sent in sents for _, lang, _ in sent if lang == "")
            self.assertEqual(n_blank, 0, f"{path.name}: found blank-lang tokens after fix")

    def test_pre_fix_behavior_would_have_produced_all_blank(self):
        # Regression guard: without the override (the old call signature's
        # effective behavior for monolingual files with no Lang= in MISC),
        # every token's lang is blank - this is the bug the fix corrects.
        sents = tb.load_tagged_sents(GUM_TRAIN)
        langs = {lang for sent in sents for _, lang, _ in sent}
        self.assertEqual(langs, {""})


if __name__ == "__main__":
    unittest.main(verbosity=2)
