"""Offline tests for transparent, correctly weighted evaluation metrics."""
import csv
import tempfile
import unittest
from pathlib import Path

from evaluate import edit_distance, error_rates, read_manifest


class MetricTests(unittest.TestCase):
    def test_character_and_word_edit_distance(self):
        self.assertEqual(edit_distance("kitten", "sitting"), 3)
        self.assertEqual(edit_distance(["hello", "world"], ["hello", "planet"]), 1)
        self.assertEqual(edit_distance("", "x"), 1)

    def test_cer_and_wer_micro_average(self):
        report = error_rates([("abc", "axc"), ("hello world", "hello world")])
        self.assertEqual(report["samples"], 2)
        self.assertEqual(report["character_errors"], 1)
        self.assertEqual(report["reference_characters"], 14)
        self.assertAlmostEqual(report["cer"], 1 / 14)
        self.assertEqual(report["word_errors"], 1)
        self.assertEqual(report["reference_words"], 3)
        self.assertAlmostEqual(report["wer"], 1 / 3)

    def test_empty_reference_and_empty_corpus_are_invalid(self):
        with self.assertRaises(ValueError):
            error_rates([])
        with self.assertRaises(ValueError):
            error_rates([("", "")])

    def test_manifest_reads_historic_unnamed_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "labels.csv"
            with csv_path.open("w", newline="") as out:
                writer = csv.writer(out)
                writer.writerow(["0", "1"])
                writer.writerow(["Datasets/IAM_Sentences/example.png", "ground truth"])
            self.assertEqual(
                list(read_manifest(csv_path)),
                [("Datasets/IAM_Sentences/example.png", "ground truth")],
            )

    def test_bad_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "labels.csv"
            csv_path.write_text("foo,bar\na,b\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                list(read_manifest(csv_path))


if __name__ == "__main__":
    unittest.main()
