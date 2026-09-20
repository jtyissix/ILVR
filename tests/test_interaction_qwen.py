"""Integration against the actual vendored Qwen layers, processor and HF export.

Run after installing requirements-interaction.txt (no checkpoint download needed).
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import torch
from src.interaction.data import Collator, prepare_sample, position_ids_for_images, token_ids
from src.interaction.execution import DecoderExecutor
from src.interaction.model import assert_local_transformers, input_embeddings, load_full_model


def tiny_qwen():
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    config = Qwen2_5_VLConfig(
        vocab_size=96, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        rope_scaling={"rope_type": "default", "mrope_section": [2, 2, 4]},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 48, "num_heads": 4,
                       "out_hidden_size": 64, "patch_size": 14, "spatial_merge_size": 2,
                       "temporal_patch_size": 2, "window_size": 112, "fullatt_block_indexes": [0]},
        pad_token_id=0, eos_token_id=90, image_token_id=80, video_token_id=81,
        vision_start_token_id=82, vision_end_token_id=83, latent_start_id=70, latent_token_id=71,
        latent_end_id=72, latent_size=8, stage="stage1",
    )
    config._attn_implementation = "sdpa"
    return Qwen2_5_VLForConditionalGeneration(config)


class QwenIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert_local_transformers()
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(12)

    def test_qwen_native_forward_gradients_and_image_mrope(self):
        model = tiny_qwen()
        grid = torch.tensor([[1, 4, 4], [1, 4, 6]])
        ids = torch.tensor([[2, 82, 80, 80, 80, 80, 83, 3, 82] + [80]*6 + [83, 4, 5]])
        pos, _ = model.get_rope_index(ids, image_grid_thw=grid)
        expected_pos = position_ids_for_images(ids[0].tolist(), grid.tolist(), model.config.to_dict())
        torch.testing.assert_close(pos[:, 0], expected_pos)
        original = copy.deepcopy(model.model)
        features = torch.randn(10, 64)
        embeds = input_embeddings(original, ids, features, 80)
        mask = torch.ones_like(ids, dtype=torch.bool)
        expected = original(inputs_embeds=embeds, attention_mask=mask, position_ids=pos, use_cache=False).last_hidden_state
        expected.square().mean().backward()
        executor = DecoderExecutor(model.model, "sdpa", activation_checkpointing=True)
        embeds = input_embeddings(model.model, ids, features, 80)
        cache, pieces = None, []
        for start, end in ((0, 8), (8, 9), (9, ids.shape[1])):
            h, cache = executor.chunk(embeds[:, start:end], mask[:, :end], pos[:, :, start:end], cache)
            pieces.append(h)
        actual = torch.cat(pieces, 1)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        actual.square().mean().backward()
        for (name, p), (other, ref) in zip(model.model.named_parameters(), original.named_parameters()):
            self.assertEqual(name, other)
            torch.testing.assert_close(p.grad, ref.grad, rtol=2e-3, atol=1e-5, msg=name)
        # Cache stores exactly one copy of each token's K/V, as immutable chunks.
        self.assertEqual([piece.shape[2] for piece in cache[0][0]], [8, 1, ids.shape[1]-9])

    def test_real_vision_encoder_and_exact_checkpoint_loading(self):
        model = tiny_qwen().eval()
        pixels = torch.randn(16, 3*2*14*14)
        features = model.visual(pixels, grid_thw=torch.tensor([[1, 4, 4]]))
        self.assertEqual(features.shape, (4, 64))
        with tempfile.TemporaryDirectory() as root:
            model.save_pretrained(root, safe_serialization=True)
            loaded = load_full_model(root, "sdpa", dtype=torch.float32)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)

    def test_loading_old_export_resets_attention_backend_flags(self):
        model = tiny_qwen()
        with tempfile.TemporaryDirectory() as root:
            model.save_pretrained(root, safe_serialization=True)
            path = Path(root) / "config.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            config["_attn_implementation_autoset"] = True
            config["vision_config"]["_attn_implementation_autoset"] = True
            path.write_text(json.dumps(config), encoding="utf-8")
            original = path.read_bytes()
            # Switching SDPA -> eager makes failed propagation observable on CPU.
            loaded = load_full_model(root, "eager", dtype=torch.float32)
            self.assertEqual(type(loaded.visual.blocks[0].attn).__name__, "Qwen2_5_VLVisionAttention")
            self.assertEqual(loaded.config.vision_config._attn_implementation, "eager")
            self.assertEqual(path.read_bytes(), original)  # Existing checkpoint is never rewritten.
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)

    def test_inference_export_roundtrip_excludes_training_modules(self):
        from src.interaction.checkpoint import export_inference
        from src.interaction.model import InteractionStudent
        from src.interaction.config import TrainConfig
        model = tiny_qwen()
        config = TrainConfig(fusion_dim=16, fusion_heads=2, fusion_ffn_dim=32, attention_backend="sdpa")
        student = InteractionStudent(model.model, model.lm_head, model.config, config)
        processor = SimpleNamespace(save_pretrained=lambda path: None)
        with tempfile.TemporaryDirectory() as root:
            export_inference(SimpleNamespace(module=student), model.visual, processor, root, config, "test-source")
            loaded = load_full_model(root, "sdpa")
            self.assertEqual(loaded.config.latent_size, 9)
            self.assertEqual(loaded.config.visual_latent_size, 8)
            self.assertEqual(loaded.config.interaction_latent_size, 1)
            self.assertFalse(any("fusion" in key or "reference" in key for key in loaded.state_dict()))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[key], value.to(torch.bfloat16), rtol=0, atol=0)

    def test_real_processor_prepare_and_collate_no_helper_pixels(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import Qwen2TokenizerFast, Qwen2VLImageProcessor, Qwen2_5_VLProcessor
        from PIL import Image
        vocabulary = {"<unk>": 0, "user": 1, "assistant": 2, "question": 3, "reason": 4,
                      "answer": 5, "7": 6, "<|endoftext|>": 7, "<|im_start|>": 8, "<|im_end|>": 9,
                      "<|vision_start|>": 10, "<|vision_end|>": 11, "<|image_pad|>": 12,
                      "<|latent_start|>": 13, "<|latent_pad|>": 14, "<|latent_end|>": 15}
        backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = Qwen2TokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                       additional_special_tokens=[x for x in vocabulary if x.startswith("<|")])
        template = ("{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' }}"
                    "{% for c in m['content'] %}{% if c['type'] == 'image' %}"
                    "{{ '<|vision_start|><|image_pad|><|vision_end|>' }}{% else %}{{ c['text'] }}{% endif %}{% endfor %}"
                    "{{ '<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")
        processor = Qwen2_5_VLProcessor(Qwen2VLImageProcessor(min_pixels=3136, max_pixels=3136), tokenizer, chat_template=template)
        config = tiny_qwen().config.to_dict()
        config.update(vocab_size=len(tokenizer), latent_start_id=13, latent_token_id=14, latent_end_id=15,
                      image_token_id=12, vision_start_token_id=10, vision_end_token_id=11)
        with tempfile.TemporaryDirectory() as root:
            image = Path(root) / "images_comt" / "creation" / "input.png"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (56, 56), "red").save(image)
            helper = image.parent / "helper.png"
            Image.new("RGB", (112, 56), "blue").save(helper)
            sample = {"text_input": "question", "image_input": ["images_comt/creation/input.png"],
                      "sequence_plan": [{"type": "text", "content": "reason "},
                                        {"type": "latent", "helper_image": "images_comt/creation/helper.png"},
                                        {"type": "text", "content": " answer 7"}], "original_final_answer": "7"}
            record = prepare_sample(sample, processor, config, root, 512)
            self.assertEqual(len(record["image_grid_thw"]), 1)
            self.assertEqual(len(record["segments"]), 1)
            self.assertEqual(record["input_ids"].count(tokenizer.convert_tokens_to_ids("7")), 1)
            self.assertTrue(all(x == -100 for x in record["labels"][:record["assistant_start"]]))
            for pos in record["segments"][0]["pads"]:
                self.assertEqual(record["labels"][pos], -100)
            record["source_line"] = 1
            manifest = {"token_ids": token_ids(processor, config), "special_ids": tokenizer.all_special_ids}
            batch = Collator(processor, config, 1, manifest)([record, {**record, "dummy": True}])
            self.assertEqual(batch["visual"]["image_grid_thw"].shape[0], 1)  # shared input encoded once, never helpers
            self.assertEqual(batch["visual"]["feature_indices"].tolist(), [0, 1, 2, 3]*2)
            self.assertEqual(len(batch["student"]["segments"][0][0]["pads"]), 9)
            self.assertTrue(batch["student"]["labels"][1].eq(-100).all())
            with self.assertRaisesRegex(ValueError, "Overlength"):
                prepare_sample(sample, processor, config, root, 4)


if __name__ == "__main__":
    unittest.main()
