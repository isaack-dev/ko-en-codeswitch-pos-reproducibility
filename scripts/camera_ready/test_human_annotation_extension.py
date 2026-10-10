"""Sampling safeguards; no models, labels, or original outputs are written."""
import collections
import dataclasses
import tempfile
import unittest
from pathlib import Path

import create_human_annotation_extension as sampling


def sentence(sid, doc, length, text=None):
    return sampling.Sentence(sid, doc, text or sid,
        tuple((str(i), f"token{i}") for i in range(1, length + 1)))


class SamplingSafeguards(unittest.TestCase):
    def test_original_edges_and_normalization(self):
        self.assertEqual([sampling.stratum(n) for n in (1, 12, 13, 15, 16, 18, 19, 22, 23)],
                         [1, 1, 2, 2, 3, 3, 4, 4, 5])
        self.assertEqual(sampling.normalize("  Ａ\u00a0한글\t\n B  "), "A 한글 B")
        self.assertNotEqual(sampling.normalize("Ab"), sampling.normalize("ab"))

    def test_parser_does_not_retain_pos_or_language(self):
        template = "# sent_id = s1\n# doc_id = d1\n# text = 한글 English\n" + \
            "1\t한글\t_\t{pos}\t_\t_\t0\t_\t_\tLang={lang}\n" + \
            "2\tEnglish\t_\t{pos}\t_\t_\t0\t_\t_\tLang={lang}\n\n"
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "source.conllu"
            p.write_text(template.format(pos="NOUN", lang="KO"), encoding="utf-8")
            a = sampling.load_split(p)
            p.write_text(template.format(pos="VERB", lang="EN"), encoding="utf-8")
            b = sampling.load_split(p)
        self.assertEqual(a, b)
        self.assertEqual([f.name for f in dataclasses.fields(a[0])],
                         ["sid", "doc", "text", "tokens", "errors"])

    def test_frame_exclusions_and_predefined_duplicate_retention(self):
        old = sentence("old", "old_doc", 2, "prior text")
        cases = [old, sentence("same_doc", "old_doc", 2),
                 sentence("old_text", "other", 2, "prior\ttext"),
                 sentence("train_copy", "train_copy_doc", 2, "ＴＲＡＩＮ"),
                 sentence("dev_copy", "dev_copy_doc", 2, "DEV"),
                 sentence("keep_a", "new_a", 2, "new text"),
                 sentence("keep_b", "new_b", 2, "new  text"),
                 sampling.Sentence("invalid", "bad_doc", "", (), ("missing_identity_text_or_tokens",))]
        splits = {"test": cases, "train": [sentence("tr", "trdoc", 1, "TRAIN")],
                  "dev": [sentence("dv", "dvdoc", 1, "DEV")]}
        annotated = {old.sid: {"doc": old.doc, "text": old.text}}
        docmap = {s.sid: (s.doc, len(s.tokens)) for s in cases}
        frame, excluded, _, _ = sampling.candidate_frame(splits, annotated, docmap)
        self.assertEqual([s.sid for s in frame], ["keep_a"])
        reasons = {r["source_sentence_id"]: r["all_reasons"].split("|") for r in excluded}
        for sid, reason in (("old", "previous_annotated_sentence"),
                ("same_doc", "previous_annotated_document"),
                ("old_text", "normalized_previous_annotation_duplicate"),
                ("train_copy", "normalized_train_duplicate"),
                ("dev_copy", "normalized_dev_duplicate"),
                ("keep_b", "normalized_test_frame_duplicate")):
            self.assertIn(reason, reasons[sid])

    def test_joint_document_feasibility_is_not_per_bin_counts(self):
        # Every bin individually has 36 documents, but the union has only 36.
        shared = [sentence(f"s{b}_{i}", f"d{i}", n)
                  for b, n in enumerate((1, 13, 16, 19, 23), 1) for i in range(36)]
        splits = {"test": shared, "train": [], "dev": []}
        docmap = {s.sid: (s.doc, len(s.tokens)) for s in shared}
        _, _, audit, hall = sampling.candidate_frame(splits, {}, docmap)
        self.assertTrue(all(a["eligible_documents"] == 36 for a in audit))
        self.assertTrue(any(not h["passed"] for h in hall))
        with self.assertRaisesRegex(ValueError, "Cannot satisfy"):
            sampling.select_extension(shared)

    def test_matching_repeatability_quotas_and_token_sequences(self):
        # Shared documents require an augmenting path, not independent draws.
        frame = [sentence(f"s{b}_{i:03}", f"d{i:03}", n)
                 for b, n in enumerate((1, 13, 16, 19, 23), 1)
                 for i in range((b-1)*36, b*36+10)]
        selected = sampling.select_extension(frame)
        self.assertEqual(selected, sampling.select_extension(list(reversed(frame))))
        self.assertEqual(len({s.doc for _, _, s in selected}), 180)
        counts = collections.Counter((batch, sampling.stratum(len(s.tokens))) for batch, _, s in selected)
        self.assertEqual(counts, {(batch, b): quota for batch, quota in (("A", 12), ("B", 24))
                                  for b in range(1, 6)})
        self.assertTrue(all(s in frame for _, _, s in selected))
        self.assertEqual(len({sample_id for _, sample_id, _ in selected}), 180)


if __name__ == "__main__":
    unittest.main(verbosity=2)
