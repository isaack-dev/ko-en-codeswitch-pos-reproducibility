"""Small explicit-path/guard tests; train only synthetic two-token fixtures."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_baselines as adapter
import verify_release as integrity
from read_conllu import read_sentences


class AdapterTests(unittest.TestCase):
    def test_checksum_mismatch_and_missing_input_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            file = root/"aggregate.csv"
            file.write_text("count\n1\n")
            manifest = {"artifacts": [{"path": file.name, "sha256": integrity.sha(file)}]}
            with patch.object(integrity, "ROOT", root):
                self.assertEqual(integrity.verify(manifest)["aggregate_files_verified"], 1)
                file.write_text("count\n2\n")
                with self.assertRaises(ValueError):
                    integrity.verify(manifest)
                with self.assertRaises(FileNotFoundError):
                    adapter.configure_prediction(root/"absent", root/"absent", root, root)

    def test_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path/"keep.txt").write_text("keep")
            with self.assertRaises(ValueError):
                adapter.fresh(path)
            self.assertEqual((path/"keep.txt").read_text(), "keep")

    def test_staged_paths_and_original_implementations_end_to_end(self):
        fixture = "# sent_id = fixture\n1\talpha\t_\tNOUN\t_\t_\t0\troot\t_\tLang=EN\n2\t.\t_\tPUNCT\t_\t_\t1\tpunct\t_\tLang=OTHER\n\n"
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            staged = root/"data/splits"
            staged.mkdir(parents=True)
            source = staged/"frozen.conllu"
            source.write_text(fixture, encoding="utf-8")
            output = adapter.fresh(root/"run")
            adapter.configure_training(source, source, source, output)
            adapter.train.main()
            self.assertEqual((adapter.train.SEED, adapter.train.N_EPOCHS), (42, 5))
            self.assertEqual(len(list((output/"models").glob("*.json"))), 8)
            predictions = adapter.fresh(root/"predictions")
            adapter.configure_prediction(source, source, output/"models", predictions)
            adapter.predict.main()
            adapter.evaluate.RESULTS = predictions
            adapter.evaluate.main()
            self.assertEqual(len(list(predictions.glob("*_predictions.conllu"))), 10)
            self.assertTrue((predictions/"overall_results.csv").is_file())
            for path in predictions.glob("*_predictions.conllu"):
                rows = read_sentences(path)
                self.assertEqual([t["fields"][:9] for t in rows[0]["tokens"]],
                                 [t["fields"][:9] for t in read_sentences(source)[0]["tokens"]])
            self.assertEqual(source.read_text(encoding="utf-8"), fixture)


if __name__ == "__main__":
    unittest.main(verbosity=2)
