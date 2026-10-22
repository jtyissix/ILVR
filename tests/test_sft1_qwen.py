"""SFT1 checks using the actual vendored Qwen layers (no weight download)."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from src.interaction.distributed import to_device
from src.interaction.execution import causal_ce_sum
from src.interaction.model import assert_local_transformers, input_embeddings
from src.sft1.model import WarmupModel
from tests.test_interaction_qwen import tiny_qwen
from tests.test_sft1_model import example_batch, tiny_config


class WarmupQwenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert_local_transformers()
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(19)

    def test_native_qwen_matches_segmented_actual_history_and_gradients(self):
        full = tiny_qwen()
        actual = WarmupModel(full.model, full.lm_head, full.config, tiny_config(activation_checkpointing=False))
        expected = copy.deepcopy(actual)
        batch, features = example_batch([(2, [[3, 4], "image", [5, 6], "image", [7, 90]])], hidden=64)
        incremental = actual.trajectory(batch, features)
        # Materialize the exact same dependent b embeddings, then run the native
        # full-sequence decoder. b retains its own prefix computation graph.
        reference_b = expected.trajectory(batch, features)
        embeds = input_embeddings(expected.decoder, batch["input_ids"], features, 80)
        replacements = {s["position"] + token: value
                        for index, s in enumerate(batch["segments"][0])
                        for token, value in enumerate(reference_b.interactions[0, index])}
        embeds = torch.stack([replacements.get(p, embeds[0, p]) for p in range(embeds.shape[1])])[None]
        native = expected.decoder(inputs_embeds=embeds, attention_mask=batch["attention_mask"],
                                  position_ids=batch["position_ids"], use_cache=False).last_hidden_state
        torch.testing.assert_close(incremental.hidden, native, atol=2e-5, rtol=2e-5)
        actual_loss = causal_ce_sum(incremental.hidden, batch["labels"], actual.lm_head, 3, False)
        expected_loss = causal_ce_sum(native, batch["labels"], expected.lm_head, 3, False)
        torch.testing.assert_close(actual_loss, expected_loss)
        actual_loss.backward()
        expected_loss.backward()
        expected_parameters = dict(expected.named_parameters())
        for name, parameter in actual.named_parameters():
            other = expected_parameters[name].grad
            if parameter.grad is None:
                self.assertIsNone(other, name)
            else:
                torch.testing.assert_close(parameter.grad, other, atol=4e-5, rtol=3e-3, msg=name)

    def test_real_pixels_plan_processor_vision_and_ce_training_pipeline(self):
        from PIL import Image
        from transformers import Qwen2_5_VLForConditionalGeneration
        from src.sft1.data import WarmupCollator, _processor_identity, prepare_sample
        from src.sft1.train import encode_images
        from tests.test_sft1_data import real_qwen_processor
        processor = real_qwen_processor()
        config = tiny_qwen().config
        config.vocab_size = len(processor.tokenizer)
        config.image_token_id = 12
        config.vision_start_token_id, config.vision_end_token_id = 10, 11
        config.pad_token_id = processor.tokenizer.pad_token_id
        config.eos_token_id = 9
        full = Qwen2_5_VLForConditionalGeneration(config)
        visual = full.visual.requires_grad_(False).eval()
        model = WarmupModel(full.model, full.lm_head, config, tiny_config())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (56, 56), "red").save(root / "question.png")
            Image.new("RGB", (84, 56), "blue").save(root / "helper.png")
            sample = {"text_input": "question", "image_input": ["question.png"],
                      "original_final_answer": "7", "sequence_plan": [
                          {"type": "text", "content": [
                              {"sub_type": "analysis", "content": "reason"},
                              {"sub_type": "action", "original_text": "look", "content": "SHORT"}]},
                          {"type": "latent", "helper_image": "helper.png"},
                          {"type": "text", "content": "answer 7", "is_final_answer": True}]}
            source = dict(name="fixture", image_root=str(root), plan_field="sequence_plan",
                          require_structured_actions=True)
            record = prepare_sample(sample, source, processor, config.to_dict(), 512)
            collator = WarmupCollator(processor, config.to_dict(),
                                     {"processor_identity": _processor_identity(processor)})
            batch = collator([record])
            features = encode_images(visual, batch)
            self.assertEqual(len(features), int(batch["input_ids"].eq(12).sum()))
            self.assertFalse(features.requires_grad)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            before = model.warmup_adapter.queries.detach().clone()
            loss = model(batch, features)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
            self.assertFalse(torch.equal(before, model.warmup_adapter.queries.detach()))
            self.assertTrue(all(p.grad is None for p in visual.parameters()))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA not available")
    def test_bfloat16_cuda_forward_backward_with_checkpointing(self):
        full = tiny_qwen()
        model = WarmupModel(full.model, full.lm_head, full.config, tiny_config()).to("cuda", torch.bfloat16)
        batch, features = example_batch(hidden=64)
        batch = to_device(batch, torch.device("cuda"))
        loss = model(batch, features.to("cuda", torch.bfloat16))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(float(model.warmup_adapter.queries.grad.float().norm()), 0)
        self.assertGreater(float(model.interaction.query_mlp[0].weight.grad.float().norm()), 0)


if __name__ == "__main__":
    unittest.main()
