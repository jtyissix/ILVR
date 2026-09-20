import json
from pathlib import Path
import tempfile
import unittest

from src.interaction.audit_vsp import assess, audit, extended_actions, official_actions


class VSPAuditTest(unittest.TestCase):
    def test_official_box_contract_and_character_filter(self):
        cases = {
            r"\boxed{DLLU}": "DLLU",
            r"\boxed{L} then \boxed{\text{D L L U}}": "DLLU",
            r"\boxed{4R1U}": "RU",  # Official code ignores counts.
            r"\boxed{RIGHT DOWN LEFT UP}": "RDLU",
            "Final answer: DLLU": "",  # MathRuler returns 'None', not a fallback.
            r"\boxed{DLLU} but \boxed{DLLU": "",  # Last incomplete box wins.
            r"\boxed {DLLU}": "",  # Official extractor requires the literal marker.
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual("".join(official_actions(text)), expected)

    def test_extended_explicit_grammar_without_prose_mining(self):
        cases = {"4R1U": "RRRRU", "3Rights": "RRR", "Up 2 times, right 2 times": "UURR",
                 "move right then move down": "RD", "D then R": "DR", "DLLU": "DLLU",
                 r"\text{DOWN LEFT LEFT UP}": "DLLU", "D -> L -> L -> U": "DLLU"}
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual("".join(extended_actions(text)), expected)
        for text in ("5", "The route is DLLU", "1 gift, right", "U JUMP L", "0R", "999999R", "R then"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                extended_actions(text)

    def test_no_map_guided_answer_selection(self):
        sample = {"map_id": "a", "map_desc": [[2, -1, 1], [0, 0, 0], [0, 0, 0]]}
        # Earlier successful routes must not override a wrong final answer.
        _, scores = assess(sample, r"Try DLLU. \boxed{DLLU} Final answer: \boxed{L}")
        self.assertTrue(all(not score["success"] for score in scores.values()))
        _, scores = assess(sample, "The route is DLLU.\nFinal answer: 5")
        self.assertTrue(all(not score["success"] for score in scores.values()))

    def test_identity_completeness_duplicates_and_mixed_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = [{"map_id": str(i), "map_desc": [[1, 2], [0, 0]]} for i in range(2)]
            test = root / "test.jsonl"
            test.write_text("\n".join(json.dumps(s) for s in samples), encoding="utf-8")
            path = root / "rank.jsonl"
            row = {"index": 0, "map_id": "0", "grid_size": "2x2", "raw_output": r"\boxed{R}",
                   "segment_steps": [8], "correct": True, "prediction": "R"}
            def write(rows):
                path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            write([row])
            with self.assertRaisesRegex(ValueError, "Only 1/2"):
                audit(test, [path])
            report, _ = audit(test, [path], allow_partial=True)
            self.assertFalse(report["complete"])
            self.assertEqual(report["profiles"]["mirage_official"]["correct"], 1)
            self.assertEqual(report["stored_score_mismatch_indices"], [])
            write([row, row])
            with self.assertRaisesRegex(ValueError, "Duplicate index"):
                audit(test, [path], True)
            write([{**row, "map_id": "wrong"}])
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                audit(test, [path], True)
            write([row, {**row, "index": 1, "map_id": "1", "segment_steps": [9]}])
            with self.assertRaisesRegex(ValueError, "Mixed latent sizes"):
                audit(test, [path])
            write([row, {**row, "index": 1, "map_id": "1"}])
            report, _ = audit(test, [path])
            self.assertTrue(report["complete"])


if __name__ == "__main__":
    unittest.main()
