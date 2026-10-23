"""Small CPU fixtures check data semantics without downloading a checkpoint."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from PIL import Image
import torch

from src.sft1.data import (
    WarmupCollator, WarmupDataset, _processor_identity, normalize_sample,
    prepare_dataset, prepare_sample, validate_tokenizer_capacity,
)


class FakeTokenizer:
    special = {"<|pad|>": 0, "<|im_start|>": 1, "<|im_end|>": 2,
               "<|vision_start|>": 3, "<|vision_end|>": 4, "<|image_pad|>": 5}
    pad_token_id = 0
    eos_token_id = 2
    all_special_ids = list(special.values())
    all_special_tokens = list(special)

    def __len__(self):
        return 512

    def get_vocab(self):
        return {**self.special, **{f"token_{i}": i for i in range(6, 512)}}

    def encode(self, text, add_special_tokens=False):
        result, cursor = [], 0
        while cursor < len(text):
            for token, value in self.special.items():
                if text.startswith(token, cursor):
                    result.append(value)
                    cursor += len(token)
                    break
            else:
                result.append(ord(text[cursor]) + 100)
                cursor += 1
        return result

    def decode(self, values):
        inverse = {value: token for token, value in self.special.items()}
        return "".join(inverse[value] if value in inverse else chr(value - 100) for value in values)


class FakeImageProcessor:
    merge_size = 2

    def to_dict(self):
        return {"merge_size": self.merge_size}

    def __call__(self, images, return_tensors):
        return {"image_grid_thw": torch.tensor([[1, 4, 4] for _ in images]),
                "pixel_values": torch.zeros(len(images) * 16, 3)}


class FakeProcessor:
    tokenizer = FakeTokenizer()
    image_processor = FakeImageProcessor()
    image_token = "<|image_pad|>"
    chat_template = "fake-qwen-style-template-v1"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = ""
        for message in messages:
            text += "<|im_start|>" + message["role"] + "\n"
            for part in message["content"]:
                text += part["text"] if part["type"] == "text" else "<|vision_start|><|image_pad|><|vision_end|>"
            text += "<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def __call__(self, text, images, return_tensors):
        expanded = text[0].replace(self.image_token, self.image_token * 4)
        result = {"input_ids": torch.tensor([self.tokenizer.encode(expanded)])}
        if images:
            result.update(self.image_processor(images, return_tensors))
        return result


def real_qwen_processor():
    """Real offline tokenizer/image pipeline, also reusable for model integration."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import Qwen2TokenizerFast, Qwen2VLImageProcessor, Qwen2_5_VLProcessor
    vocabulary = {"<unk>": 0, "user": 1, "assistant": 2, "question": 3, "reason": 4,
                  "answer": 5, "7": 6, "<|endoftext|>": 7, "<|im_start|>": 8, "<|im_end|>": 9,
                  "<|vision_start|>": 10, "<|vision_end|>": 11, "<|image_pad|>": 12,
                  "look": 13, "SHORT": 14}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = Qwen2TokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                   additional_special_tokens=[x for x in vocabulary if x.startswith("<|")])
    template = ("{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' }}"
                "{% for c in m['content'] %}{% if c['type'] == 'image' %}"
                "{{ '<|vision_start|><|image_pad|><|vision_end|>' }}{% else %}{{ c['text'] }}{% endif %}{% endfor %}"
                "{{ '<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")
    return Qwen2_5_VLProcessor(Qwen2VLImageProcessor(min_pixels=3136, max_pixels=6272),
                               tokenizer, chat_template=template)


class WarmupDataTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("question.png", "second.png", "helper.png"):
            Image.new("RGB", (28, 28), "red").save(self.root / name)
        self.source = {"name": "fixture", "image_root": str(self.root), "plan_field": "thought_rewritten"}
        self.processor = FakeProcessor()
        self.model = {"vocab_size": 512, "image_token_id": 5, "vision_start_token_id": 3,
                      "vision_end_token_id": 4, "vision_config": {"spatial_merge_size": 2}}
        self.row = {"text_input": "What color?", "image_input": ["question.png"],
                    "original_final_answer": "red",
                    "sequence_plan": [{"type": "text", "content": "STALE OLD PLAN"}],
                    "thought_rewritten": [
                        {"type": "text", "content": [
                            {"sub_type": "analysis", "content": "Inspect the object."},
                            {"sub_type": "action", "content": "SHORT PHRASE", "original_text": "I crop the object."}]},
                        {"type": "latent", "helper_image": "helper.png"},
                        {"type": "text", "content": "The final answer is: red", "is_final_answer": True}]}

    def record(self, row=None, **kwargs):
        return prepare_sample(row or self.row, self.source, self.processor, self.model,
                              kwargs.get("max_seq_length", 4096))

    def test_padded_model_vocabulary_is_valid_but_out_of_range_ids_are_rejected(self):
        padded = {**self.model, "vocab_size": 640}
        validate_tokenizer_capacity(self.processor, padded)
        record = prepare_sample(self.row, self.source, self.processor, padded, 4096)
        self.assertEqual(record["input_ids"], self.record()["input_ids"])
        with self.assertRaisesRegex(ValueError, "Tokenizer IDs do not fit"):
            validate_tokenizer_capacity(self.processor, {**self.model, "vocab_size": 511})
        with self.assertRaisesRegex(ValueError, "exceeds model capacity"):
            prepare_sample(self.row, self.source, self.processor, {**self.model, "vocab_size": 511}, 4096)
        with mock.patch.object(self.processor.tokenizer, "get_vocab",
                               return_value={**self.processor.tokenizer.get_vocab(), "sparse_id": 640}):
            with self.assertRaisesRegex(ValueError, "max_token_id=640"):
                validate_tokenizer_capacity(self.processor, padded)
        with self.assertRaisesRegex(ValueError, "special token mismatch"):
            validate_tokenizer_capacity(self.processor, {**padded, "image_token_id": 6})

    def test_encoded_out_of_range_id_is_rejected_even_with_short_tokenizer(self):
        row = copy.deepcopy(self.row)
        row["text_input"] += chr(600)
        with self.assertRaisesRegex(ValueError, "Encoded token ID"):
            self.record(row)

    def test_explicit_rewritten_field_and_original_action_order(self):
        record = self.record()
        supervised = self.processor.tokenizer.decode([v for v in record["labels"] if v != -100])
        self.assertIn("Inspect the object.\nI crop the object.", supervised)
        self.assertNotIn("SHORT PHRASE", supervised)
        self.assertNotIn("STALE OLD PLAN", supervised)
        self.assertIn("The final answer is: red", supervised)
        self.assertIn("<|im_end|>", supervised)
        self.assertNotIn("What color?", supervised)
        self.assertNotIn("vision", supervised)
        self.assertNotIn("image_pad", supervised)
        segment = record["segments"][0]
        helper = record["images"][segment["image_index"]]
        self.assertEqual(segment["position"] + 3, helper["boundary_start"])
        self.assertEqual(record["labels"][segment["position"]:segment["position"] + 3], [-100] * 3)
        self.assertEqual(record["input_ids"][segment["position"]], 0)
        self.assertEqual(record["labels"][segment["position"]], -100)
        local = self.processor.tokenizer.decode([record["input_ids"][i] for i in segment["text"]])
        self.assertEqual(local, "Inspect the object.\nI crop the object.\n")
        self.assertNotIn("red", local)

    def test_refocus_field_mapping_and_unmarked_authoritative_answer(self):
        row = {"question": "How many?", "image": "question.png", "answer": "16",
               "sequence_plan": [{"type": "text", "content": "The result is 16."}]}
        source = {**self.source, "plan_field": "sequence_plan", "question_field": "question",
                  "images_field": "image", "answer_field": "answer"}
        result = normalize_sample(row, source)
        self.assertEqual(result["answer"], "16")
        self.assertEqual(len(result["question_images"]), 1)
        self.assertEqual(len(result["sequence"]), 1)

    def test_json_encoded_plan_and_answer_is_not_duplicated(self):
        row = copy.deepcopy(self.row)
        row["thought_rewritten"] = json.dumps(row["thought_rewritten"])
        result = normalize_sample(row, {**self.source, "allow_append_final_answer": True})
        self.assertEqual(len(result["sequence"]), 3)
        self.assertEqual(sum("The final answer" in s.get("content", "") for s in result["sequence"]), 1)

    def test_metadata_only_answer_rejected_and_explicit_append_is_opt_in(self):
        row = copy.deepcopy(self.row)
        row["thought_rewritten"][-1] = {"type": "text", "content": "Need to inspect more."}
        with self.assertRaisesRegex(ValueError, "Final answer is missing"):
            self.record(row)
        result = normalize_sample(row, {**self.source, "allow_append_final_answer": True})
        self.assertEqual(result["sequence"][-1]["content"], "The final answer is: red")
        record = prepare_sample(row, {**self.source, "allow_append_final_answer": True},
                                self.processor, self.model, 4096)
        supervised = self.processor.tokenizer.decode([label for label in record["labels"] if label != -100])
        self.assertIn("The final answer is: red", supervised)
        self.assertIn("<|im_end|>", supervised)
        self.assertEqual(row["thought_rewritten"][-1]["content"], "Need to inspect more.")
        row["original_final_answer"] = "2"
        row["thought_rewritten"][-1]["content"] = "There are 12 items."
        with self.assertRaisesRegex(ValueError, "Final answer is missing"):
            self.record(row)

    def test_final_marker_cannot_hide_metadata_conflict_or_later_steps(self):
        row = copy.deepcopy(self.row)
        row["original_final_answer"] = "blue"
        with self.assertRaisesRegex(ValueError, "Final answer is missing"):
            self.record(row)
        with self.assertRaisesRegex(ValueError, "Final answer is missing"):
            normalize_sample(row, {**self.source, "allow_append_final_answer": True})
        row = copy.deepcopy(self.row)
        row["thought_rewritten"].append({"type": "latent", "helper_image": "helper.png"})
        with self.assertRaisesRegex(ValueError, "must end"):
            self.record(row)

    def test_missing_original_action_and_demo_data_rejected(self):
        row = copy.deepcopy(self.row)
        del row["thought_rewritten"][0]["content"][1]["original_text"]
        with self.assertRaisesRegex(ValueError, "original_text"):
            self.record(row)
        row = {**self.row, "preprocess": {"audit_demo_only": True}}
        with self.assertRaisesRegex(ValueError, "audit_demo_only"):
            self.record(row)

    def test_structured_rewrite_guard_keeps_plain_tail_and_consecutive_helpers(self):
        strict = {**self.source, "require_structured_actions": True}
        row = copy.deepcopy(self.row)
        row["thought_rewritten"].insert(2, {"type": "latent", "helper_image": "helper.png"})
        self.assertEqual(len(normalize_sample(row, strict)["sequence"]), 4)
        row["thought_rewritten"][0]["content"] = "Inspect the object. I crop the object."
        with self.assertRaisesRegex(ValueError, "rewritten structured plan"):
            normalize_sample(row, strict)
        self.assertEqual(len(normalize_sample(row, self.source)["sequence"]), 4)
        row["thought_rewritten"] = [row["thought_rewritten"][-1]]
        self.assertEqual(len(normalize_sample(row, strict)["sequence"]), 1)

    def test_no_question_images_and_consecutive_helpers_allow_empty_local_text(self):
        row = copy.deepcopy(self.row)
        row["image_input"] = []
        row["thought_rewritten"].insert(2, {"type": "latent", "helper_image": "helper.png"})
        record = self.record(row)
        self.assertEqual([i["kind"] for i in record["images"]], ["helper", "helper"])
        self.assertEqual(record["segments"][1]["text"], [])
        self.assertEqual([s["image_index"] for s in record["segments"]], [0, 1])
        row["thought_rewritten"].pop(0)
        self.assertEqual(self.record(row)["segments"][0]["text"], [])

    def test_overlength_never_truncates_final_answer(self):
        record = self.record()
        self.assertEqual(self.record(max_seq_length=record["length"]), record)
        with self.assertRaisesRegex(ValueError, "no truncation"):
            self.record(max_seq_length=record["length"] - 1)

    def test_multiple_images_batch_spans_dedup_padding_and_position_ids(self):
        row = copy.deepcopy(self.row)
        row["image_input"].append("second.png")
        row["thought_rewritten"].insert(2, {"type": "latent", "helper_image": "helper.png"})
        first, second = self.record(row), self.record()
        collator = WarmupCollator(self.processor, self.model, {"processor_identity": _processor_identity(self.processor)})
        batch = collator([first, {**second, "dummy": True}])
        self.assertEqual([i["kind"] for i in batch["images"][0]], ["question", "question", "helper", "helper"])
        self.assertEqual([(i["feature_start"], i["feature_end"]) for r in batch["images"] for i in r],
                         [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)])
        self.assertEqual(batch["visual"]["image_grid_thw"].shape[0], 3)
        self.assertEqual(batch["visual"]["feature_indices"].tolist(),
                         list(range(12)) + list(range(8, 12)) + list(range(4)) + list(range(8, 12)))
        self.assertEqual(batch["sample_count"], 1)
        self.assertTrue(batch["labels"][1].eq(-100).all())
        self.assertEqual(batch["ce_count"], sum(v != -100 for v in first["labels"][1:]))
        self.assertFalse(batch["attention_mask"][1, second["length"]:].any())
        for segment in first["segments"]:
            pos = segment["position"]
            self.assertTrue(batch["attention_mask"][0, pos])
            # Both b and vision_start are ordinary 1-D positions, so the inserted
            # b must consume a position before the image's two-dimensional grid.
            torch.testing.assert_close(batch["position_ids"][:, 0, pos + 1],
                                       batch["position_ids"][:, 0, pos] + 1)

    def test_cache_rejects_changed_source_settings_images_and_records(self):
        path = self.root / "source.jsonl"
        path.write_text(json.dumps(self.row) + "\n", encoding="utf-8")
        config = SimpleNamespace(sources=[{**self.source, "path": str(path)}], model_path="fake-model",
                                 prepared_dir=str(self.root / "prepared"), max_seq_length=4096, interaction_tokens=3,
                                 min_image_pixels=None, max_image_pixels=None, shrink_overlong_images=True,
                                 skip_overlong_samples=False,
                                 prepare_workers=1, prepare_log_seconds=1)
        inventory = {"config": self.model, "metadata": {"test": "fixed"}}
        with mock.patch("src.sft1.data.checkpoint_inventory", return_value=inventory), \
             mock.patch("src.sft1.data.load_processor", return_value=self.processor), \
             mock.patch("src.sft1.preprocessing.checkpoint_inventory", return_value=inventory), \
             mock.patch("src.sft1.preprocessing.load_processor", return_value=self.processor):
            manifest = prepare_dataset(config)
            self.assertEqual(manifest["samples"], 1)
            dataset = WarmupDataset(config.prepared_dir)
            dataset.validate_sources(config)
            config.interaction_tokens = 1
            with self.assertRaisesRegex(ValueError, "Interaction token count"):
                dataset.validate_sources(config)
            config.interaction_tokens = 3
            config.skip_overlong_samples = True
            with self.assertRaisesRegex(ValueError, "Overlength exclusion policy"):
                dataset.validate_sources(config)
            config.skip_overlong_samples = False
            self.assertTrue(dataset[-1]["dummy"])
            self.assertFalse(dataset[0]["dummy"])
            config.max_image_pixels = 4096
            with self.assertRaisesRegex(ValueError, "pixel limits"):
                dataset.validate_sources(config)
            config.max_image_pixels = None
            path.write_text(json.dumps({**self.row, "id": "changed"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sources"):
                dataset.validate_sources(config)
            path.write_text(json.dumps(self.row) + "\n", encoding="utf-8")
            Image.new("RGB", (40, 40), "blue").save(self.root / "helper.png")
            with self.assertRaisesRegex(ValueError, "image changed"):
                dataset.validate_sources(config)
        cache = Path(config.prepared_dir) / "records.jsonl"
        cache.write_text(cache.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            WarmupDataset(config.prepared_dir)

    def test_real_qwen_processor_complete_images_labels_and_native_mrope(self):
        from tests.test_interaction_qwen import tiny_qwen
        processor = real_qwen_processor()
        tokenizer = processor.tokenizer
        native = tiny_qwen()
        native.config.image_token_id = 12
        native.config.vision_start_token_id = 10
        native.config.vision_end_token_id = 11
        config = native.config.to_dict()
        config["vocab_size"] = len(tokenizer)
        Image.new("RGB", (56, 56), "red").save(self.root / "question.png")
        Image.new("RGB", (84, 56), "green").save(self.root / "second.png")
        Image.new("RGB", (112, 56), "blue").save(self.root / "helper.png")
        row = {"text_input": "question", "image_input": ["question.png", "second.png"],
               "original_final_answer": "7", "thought_rewritten": [
                   {"type": "text", "content": [{"sub_type": "analysis", "content": "reason"},
                      {"sub_type": "action", "content": "SHORT", "original_text": "look"}]},
                   {"type": "latent", "helper_image": "helper.png"},
                   {"type": "latent", "helper_image": "helper.png"},
                   {"type": "text", "content": "answer 7", "is_final_answer": True}]}
        record = prepare_sample(row, self.source, processor, config, 512)
        self.assertEqual(len(record["image_grid_thw"]), 4)
        answer_token = tokenizer.convert_tokens_to_ids("7")
        self.assertEqual(record["labels"].count(answer_token), 1)
        self.assertEqual(record["labels"].count(tokenizer.convert_tokens_to_ids("<|im_end|>")), 1)
        self.assertNotIn(tokenizer.convert_tokens_to_ids("SHORT"), record["input_ids"])
        self.assertTrue(all(v == -100 for v in record["labels"][:record["assistant_start"]]))
        self.assertEqual(record["segments"][1]["text"], [])
        for item in record["images"]:
            self.assertTrue(all(v == -100 for v in record["labels"][item["boundary_start"]:item["boundary_end"]]))
        plain = {"text_input": "question", "image_input": [], "original_final_answer": "7",
                 "thought_rewritten": [{"type": "text", "content": "answer 7", "is_final_answer": True}]}
        short = prepare_sample(plain, self.source, processor, config, 512)
        batch = WarmupCollator(processor, config, {"processor_identity": _processor_identity(processor)})([record, short])
        self.assertEqual(batch["visual"]["image_grid_thw"].shape[0], 3)
        self.assertEqual(batch["visual"]["feature_indices"].numel(), record["input_ids"].count(12))
        positions, _ = native.get_rope_index(batch["input_ids"],
                                            image_grid_thw=torch.tensor(record["image_grid_thw"]),
                                            attention_mask=batch["attention_mask"].long())
        torch.testing.assert_close(batch["position_ids"], positions, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "Final answer is missing"):
            prepare_sample({**row, "original_final_answer": "8"}, self.source, processor, config, 512)


if __name__ == "__main__":
    unittest.main()
