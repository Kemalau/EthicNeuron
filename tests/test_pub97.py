"""Offline checks for frozen Pub97 membership and prompt-level partitioning."""
import csv
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pub97


class Pub97Tests(unittest.TestCase):
    def test_manifest_and_split(self):
        manifest = pub97.panel_manifest()
        self.assertEqual(len(manifest), 97)
        self.assertEqual(len({r["prompt_sha256"] for r in manifest}), 97)
        disco, evaluation = pub97.panel_ids("discovery"), pub97.panel_ids("evaluation")
        self.assertEqual((len(disco), len(evaluation)), (77, 20))
        self.assertTrue(set(disco).isdisjoint(evaluation))
        self.assertEqual(set(disco) | set(evaluation), set(pub97.panel_ids("all")))
        self.assertEqual(evaluation, pub97.panel_ids("evaluation", seed=42))
        self.assertNotEqual(evaluation, pub97.panel_ids("evaluation", seed=43))
        self.assertTrue(set(evaluation) < set(pub97.panel_ids("evaluation", eval_size=21)))
        self.assertEqual(pub97.panel_ids("discovery", eval_size=0), pub97.panel_ids("all"))
        self.assertEqual(pub97.panel_ids("evaluation", eval_size=0), [])
        for bad in [-1, 97, False, 1.5]:
            with self.assertRaises(ValueError):
                pub97.panel_ids("evaluation", eval_size=bad)

    def test_resolution_and_binding(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "synthetic.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                out = csv.writer(f)
                out.writerow(["question", "id"])
                for i in range(3):
                    out.writerow([f"Sample {i}?\n", f"item-{i}"])
            def hexdigest(text):
                return hashlib.sha256(text.encode()).hexdigest()
            manifest = [dict(question_id=f"extended_{i:03d}", source_row=i, source_id=f"item-{i}",
                             prompt_sha256=hexdigest(f"Sample {i}?\n")) for i in [0, 2]]
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                pub97.load_pub97(path)
            with patch.object(pub97, "panel_manifest", return_value=manifest), patch.object(
                    pub97, "SOURCE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()):
                rows = pub97.load_pub97(path)
                self.assertEqual([r["prompt"] for r in rows], ["Sample 0?\n", "Sample 2?\n"])
                self.assertEqual(pub97.select_split(reversed(rows), "all", eval_size=1), rows)
                rows[1]["prompt"] = rows[0]["prompt"]
                with self.assertRaisesRegex(ValueError, "binding mismatch"):
                    pub97.select_split(rows, "all", eval_size=1)

    @unittest.skipUnless(os.environ.get("PUB97_SOURCE"), "set PUB97_SOURCE for upstream round-trip")
    def test_round_trip_pinned_source(self):
        rows = pub97.load_pub97(Path(os.environ["PUB97_SOURCE"]))
        parts = [pub97.select_split(rows, part) for part in ["discovery", "evaluation"]]
        self.assertEqual([len(r) for r in parts], [77,20])
        self.assertEqual(len({r["prompt"] for part in parts for r in part}), 97)
