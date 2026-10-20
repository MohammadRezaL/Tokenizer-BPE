"""Unit tests for the educational character-level BPE implementation.

Run from the directory containing both files:
    python -m unittest test_bpe.py -v

Only Python's standard library is required.
"""

import json
import tempfile
import unittest
from pathlib import Path

from bpe import BPE


class TestBPE(unittest.TestCase):
    """Check the main behaviors and edge cases of the BPE class."""

    def test_initial_parameters(self):
        bpe = BPE(num_merges=5, min_frequency=2)
        self.assertEqual(bpe.num_merges, 5)
        self.assertEqual(bpe.min_frequency, 2)
        self.assertFalse(bpe.trained)
        self.assertEqual(bpe.merges, [])

    def test_reject_invalid_parameters(self):
        with self.assertRaises(ValueError):
            BPE(num_merges=-1)
        with self.assertRaises(ValueError):
            BPE(min_frequency=0)

    def test_merge_pair_non_overlapping(self):
        # Four a's contain two NON-overlapping occurrences of (a, a).
        result = BPE._merge_pair(("a", "a", "a", "a"), ("a", "a"))
        self.assertEqual(result, ("aa", "aa"))

    def test_merge_pair_leaves_other_symbols_intact(self):
        self.assertEqual(
            BPE._merge_pair(("c", "a", "t"), ("a", "r")),
            ("c", "a", "t"),
        )

    def test_first_merge_is_most_frequent_pair(self):
        bpe = BPE(num_merges=1).fit("cat cat car")
        self.assertEqual(bpe.merges, [("c", "a")])
        self.assertEqual(bpe.encode("cat car"), ["ca", "t", "ca", "r"])

    def test_three_merges_training_example(self):
        bpe = BPE(num_merges=3).fit("cat cat car")
        self.assertEqual(
            bpe.merges,
            [("c", "a"), ("ca", "t"), ("cat", "</w>")],
        )
        self.assertEqual(bpe.encode("cat car"), ["cat", "ca", "r"])
        self.assertTrue(bpe.trained)

    def test_zero_merges_means_character_tokens(self):
        bpe = BPE(num_merges=0).fit("cat cat car")
        self.assertEqual(bpe.merges, [])
        self.assertEqual(bpe.encode("cat"), ["c", "a", "t"])

    def test_minimum_frequency_stops_merging(self):
        bpe = BPE(num_merges=10, min_frequency=4).fit("cat cat car")
        # No adjacent pair reaches frequency 4.
        self.assertEqual(bpe.merges, [])

    def test_training_stops_at_natural_limit(self):
        bpe = BPE(num_merges=100, min_frequency=1).fit("hi")
        self.assertLess(len(bpe.merges), 100)
        self.assertEqual(bpe.encode("hi"), ["hi"])

    def test_empty_corpus_is_rejected(self):
        with self.assertRaises(ValueError):
            BPE().fit("")
        with self.assertRaises(ValueError):
            BPE().fit(" \n\t  ")

    def test_encode_before_training_is_rejected(self):
        with self.assertRaises(RuntimeError):
            BPE().encode("hello")

    def test_unseen_word_is_tokenized(self):
        bpe = BPE(num_merges=3).fit("cat cat car")
        result = bpe.encode("cab")
        self.assertEqual("".join(result), "cab")
        self.assertNotIn("</w>", "".join(result))

    def test_persian_and_unicode(self):
        # The zero-width non-joiner in "کتاب‌ها" must not be lost.
        text = "سلام کتاب‌ها"
        bpe = BPE(num_merges=10).fit(text + " " + text)
        self.assertEqual("".join(bpe.encode(text)), "".join(text.split()))
        self.assertNotIn("</w>", "".join(bpe.encode(text)))

    def test_whitespace_is_not_preserved(self):
        bpe = BPE(num_merges=0).fit("a b")
        self.assertEqual(bpe.encode(" a  b\n"), ["a", "b"])
        self.assertEqual(bpe.encode(""), [])

    def test_fit_again_resets_merge_rules(self):
        bpe = BPE(num_merges=1).fit("cat cat car")
        self.assertEqual(bpe.merges, [("c", "a")])
        bpe.fit("dog dog dot")
        self.assertEqual(bpe.merges, [("d", "o")])
        self.assertNotIn("ca", bpe.vocab)

    def test_save_before_training_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(RuntimeError):
                BPE().save_vocab(Path(temp_dir) / "vocab.json")

    def test_saved_vocabulary_is_valid_utf8_json(self):
        bpe = BPE(num_merges=2).fit("سلام سلام")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vocab_fa.json"
            bpe.save_vocab(path)
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["num_merges_learned"], len(bpe.merges))
        self.assertEqual(saved["merges"], [list(pair) for pair in bpe.merges])
        self.assertEqual(saved["tokens"], sorted(bpe.vocab))
        self.assertIn("س", saved["tokens"])


class TestDeliveredArtifacts(unittest.TestCase):
    """Validate the files and counts submitted with the final assignment."""

    ROOT = Path(__file__).resolve().parent
    EXPECTED_GPT = {
        "EN01": 13, "EN02": 9, "EN03": 10, "EN04": 8, "EN05": 14,
        "EN06": 9, "EN07": 13, "EN08": 10, "EN09": 9,
        "FA01": 34, "FA02": 32, "FA03": 34, "FA04": 26,
        "FA05": 39, "FA06": 38, "FA07": 32, "FA08": 33, "FA09": 34,
    }

    @classmethod
    def setUpClass(cls):
        import csv
        with (cls.ROOT / "results" / "comparison_table.csv").open(
                encoding="utf-8-sig", newline="") as f:
            cls.rows = list(csv.DictReader(f))
        cls.models = {
            lang: BPE.from_vocab(cls.ROOT / "results" / f"vocab_{lang}.json")
            for lang in ("en", "fa")
        }

    def test_vocabulary_files_are_present(self):
        for lang in ("en", "fa"):
            self.assertTrue((self.ROOT / "results" / f"vocab_{lang}.json").is_file())

    def test_saved_vocab_has_200_merge_rules(self):
        for model in self.models.values():
            self.assertEqual(len(model.merges), 200)

    def test_saved_vocab_sizes(self):
        self.assertEqual(len(self.models["en"].vocab), 242)
        self.assertEqual(len(self.models["fa"].vocab), 235)

    def test_loaded_vocabulary_produces_known_tokens(self):
        self.assertEqual(len(self.models["en"].encode("The curious child opened the old wooden box.")), 20)
        self.assertEqual(len(self.models["fa"].encode("امروز یک کودک کنجکاو جعبه قدیمی را باز کرد.")), 17)

    def test_comparison_contains_all_18_sentences(self):
        self.assertEqual(len(self.rows), 18)
        self.assertEqual({x["id"] for x in self.rows}, set(self.EXPECTED_GPT))

    def test_gpt4_manual_counts_match_original_inputs(self):
        for row in self.rows:
            self.assertEqual(int(float(row["gpt4_count"])), self.EXPECTED_GPT[row["id"]])

    def test_no_claim_of_locally_verified_gpt_tokens(self):
        for row in self.rows:
            self.assertEqual(row["gpt4_measurement_source"], "user_reported_online_tokenizer")
            self.assertEqual(row["gpt4_token_ids_json"], "")

    def test_our_bpe_counts_can_be_reproduced(self):
        for row in self.rows:
            model = self.models[row["language"]]
            self.assertEqual(len(model.encode(row["text"])), int(row["our_bpe_count"]))

    def test_arithmetic_differences_correct(self):
        for row in self.rows:
            our = int(row["our_bpe_count"])
            gpt4 = self.EXPECTED_GPT[row["id"]]
            self.assertEqual(int(float(row["our_minus_gpt4"])), our-gpt4)

    def test_heldout_sample_totals(self):
        en = [r for r in self.rows if r["language"] == "en" and r["split"] == "held_out"]
        fa = [r for r in self.rows if r["language"] == "fa" and r["split"] == "held_out"]
        self.assertEqual(len(en), 8)
        self.assertEqual(len(fa), 8)
        self.assertEqual((sum(int(r["our_bpe_count"]) for r in en), sum(self.EXPECTED_GPT[r["id"]] for r in en)), (202, 82))
        self.assertEqual((sum(int(r["our_bpe_count"]) for r in fa), sum(self.EXPECTED_GPT[r["id"]] for r in fa)), (178, 268))

    def test_plots_are_valid_png_files(self):
        for name in ("vocab_growth.png", "token_distribution.png"):
            path = self.ROOT / "results" / "plots" / name
            self.assertGreater(path.stat().st_size, 1000)
            self.assertEqual(path.read_bytes()[:8], b"\\x89PNG\\r\\n\\x1a\\n".decode('unicode_escape').encode('latin1'))

    def test_report_and_readme_are_present(self):
        self.assertTrue((self.ROOT / "README.md").is_file())
        self.assertEqual((self.ROOT / "report.pdf").read_bytes()[:5], b"%PDF-")


if __name__ == "__main__":
    unittest.main(verbosity=2)
