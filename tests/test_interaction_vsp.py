import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from PIL import Image
from src.interaction import vsp


GRID = [[2, -1, 1], [0, 0, 0], [0, 0, 0]]


class VSPTest(unittest.TestCase):
    def sample(self, identity="31"):
        return {"map_id": identity, "map_desc": GRID,
                "text_input": "Solve this map: <image> Put your answer in \\boxed{}.",
                "image_input": f"./data/vsp_spatial_planning/imgs_test/level3/img/{identity}.png",
                # Ground truth/helper fields must never be read as model input.
                "text_output": "SECRET ANSWER", "image_output": "missing_helper.png"}

    def write_data(self, root, count=1):
        path = root / "test_direct.jsonl"
        rows = [self.sample(str(i)) for i in range(count)]
        for row in rows:
            image = root / f"imgs_test/level3/img/{row['map_id']}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (4, 4), "white").save(image)
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        return path, rows

    def test_action_formats_and_last_answer(self):
        for text in (r"\boxed{DLLU}", r"\boxed{D, L, L, U}", r"\boxed{DOWN LEFT LEFT UP}",
                     r"\boxed{\text{D L L U}}", "The final answer is: DLLU.",
                     "['DOWN', 'LEFT', 'LEFT', 'UP']", "D -> L -> L -> U",
                     "Reason: try L.\nFinal answer: DLLU<|im_end|>",
                     r"\boxed{L} but the final result is \boxed{DLLU}"):
            with self.subTest(text=text):
                result = vsp.score(self.sample(), text)
                self.assertTrue(result["correct"])
                self.assertEqual(result["prediction"], "DLLU")

    def test_no_actions_from_prose_or_truncated_answer(self):
        for text in ("", "The route goes down and left.", r"\boxed{D JUMP LLU}",
                     "Reasoning: DLLU\n\\boxed{DLLU", r"\boxed{}", "DLLUX"):
            with self.subTest(text=text):
                result = vsp.score(self.sample(), text)
                self.assertFalse(result["correct"])
                self.assertTrue(result["simulation"]["invalid"])

    def test_map_simulation_boundaries_holes_final_cell_and_nonshortest_route(self):
        self.assertTrue(vsp.simulate(GRID, "RDLLU")["success"])  # right edge: no movement
        self.assertTrue(vsp.simulate(GRID, "DDLLUU")["success"])  # valid longer path
        hole = vsp.simulate(GRID, "LDLLU")
        self.assertEqual(hole["status"], "fell_into_hole")
        self.assertEqual(hole["steps_executed"], 1)
        self.assertFalse(vsp.simulate(GRID, "DLLUD")["success"])  # leaves goal again
        self.assertFalse(vsp.simulate(GRID, "")["success"])

    def test_bad_map_is_data_error(self):
        for grid in ([], [[1, 2], [0]], [[1, 2], [3, 0]], [[1, 2], [1, 0]],
                     [[0, 2], [0, 0]], [[1, 0], [0, 0]], [[True, 2], [0, 0]]):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                vsp.validate_map(grid)

    def test_raw_archive_paths_and_no_answer_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, rows = self.write_data(root)
            samples, report = vsp.load_samples(path, root, expected_samples=1)
            self.assertEqual(report["by_grid_size"], {"3x3": 1})
            paths = vsp.image_paths(samples[0], root)
            message = vsp.user_message(samples[0], paths)
            self.assertEqual([x["type"] for x in message["content"]], ["text", "image", "text"])
            prompt = json.dumps(message)
            for forbidden in ("map_desc", "SECRET ANSWER", "missing_helper", "<image>"):
                self.assertNotIn(forbidden, prompt)
            self.assertEqual(len(paths), 1)
            self.assertEqual(Path(paths[0]), (root / "imgs_test/level3/img/0.png").resolve())
            rows[0]["image_input"] = "imgs_test/level3/img/0.png"
            self.assertEqual(vsp.image_paths(rows[0], root), paths)

    def test_preflight_reports_missing_images_duplicate_ids_and_bad_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, rows = self.write_data(root)
            with self.assertRaisesRegex(ValueError, "Expected 400"):
                vsp.load_samples(path, root, expected_samples=400)
            path.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[0]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 2: Duplicate"):
                vsp.load_samples(path, root)
            rows[0]["image_input"] = "missing.png"
            path.write_text(json.dumps(rows[0]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1"):
                vsp.load_samples(path, root)

    def test_metrics_keep_invalid_and_truncated_samples_in_denominator(self):
        records = [vsp.score(self.sample(), text) for text in ("DLLU", "L", "garbage")]
        for i, record in enumerate(records):
            record["hit_token_limit"] = i == 2
        metrics = vsp.summarize(records)
        self.assertEqual(metrics["samples"], 3)
        self.assertEqual(metrics["correct"], 1)
        self.assertEqual(metrics["accuracy"], 1/3)
        self.assertEqual(metrics["invalid_answer_count"], 1)
        self.assertEqual(metrics["token_limit_count"], 1)
        self.assertEqual(metrics["status_counts"]["fell_into_hole"], 1)

    def test_evaluation_entrypoint_with_cpu_stubs_never_imports_comt_scorer(self):
        """Exercise JSONL -> generation -> scoring -> output wiring without a GPU."""
        from contextlib import ExitStack
        from src.interaction import evaluate
        from transformers import AutoProcessor

        model = mock.MagicMock()
        model.eval.return_value = model
        model.to.return_value = model
        model.config = SimpleNamespace(eos_token_id=90, latent_size=9, image_token_id=80, to_dict=lambda: {})
        model.visual.blocks = []
        processor = mock.MagicMock()
        processor.return_value = {key: mock.MagicMock() for key in ("input_ids", "pixel_values", "image_grid_thw")}
        processor.tokenizer.decode.return_value = r"\boxed{DLLU}"
        generation = {"token_ids": [1], "segment_steps": [9], "hit_token_limit": False}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            path, _ = self.write_data(root, count=2)
            output = root / "results"
            stack.enter_context(mock.patch("sys.argv", ["evaluate", "--task", "vsp", "--model_path", "unused",
                                "--test_data_path", str(path), "--image_root", str(root), "--output_dir", str(output)]))
            stack.enter_context(mock.patch.dict("sys.modules", {"eval": None}))
            stack.enter_context(mock.patch.object(evaluate.torch.cuda, "set_device"))
            stack.enter_context(mock.patch.object(evaluate.torch.cuda, "synchronize"))
            stack.enter_context(mock.patch.object(evaluate, "load_full_model", return_value=model))
            stack.enter_context(mock.patch.object(AutoProcessor, "from_pretrained", return_value=processor))
            stack.enter_context(mock.patch.object(evaluate, "token_ids", return_value={"start": 70, "pad": 71, "end": 72}))
            stack.enter_context(mock.patch.object(evaluate, "position_ids_for_images"))
            stack.enter_context(mock.patch.object(evaluate, "input_embeddings"))
            generate = stack.enter_context(mock.patch.object(evaluate, "generate_continuous", return_value=generation))
            stack.enter_context(mock.patch("builtins.print"))
            evaluate.main()
            self.assertEqual(generate.call_count, 2)
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["correct"], 2)
            records = [json.loads(line) for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["index"] for r in records], [0, 1])
            self.assertTrue(all(r["segment_steps"] == [9] for r in records))
            for call in processor.apply_chat_template.call_args_list:
                self.assertNotIn("SECRET ANSWER", str(call))


if __name__ == "__main__":
    unittest.main()
