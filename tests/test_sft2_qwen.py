"""Offline integration with actual Qwen layers, pixels, stage-I transfer and CUDA."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from src.interaction.execution import causal_ce_sum
from src.interaction.model import assert_local_transformers, input_embeddings
from src.sft2.model import LatentModel, load_training_components
from src.sft2.train import to_device, combined_loss, loss_weights
from tests.test_interaction_qwen import tiny_qwen
from tests.test_sft2_model import example_batch, recipe


class Stage2QwenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert_local_transformers()
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(302)

    def test_native_decoder_matches_segmented_values_and_ce_gradients(self):
        full = tiny_qwen()
        actual = LatentModel(full.model, full.lm_head, full.config, recipe())
        expected = copy.deepcopy(actual)
        batch, features = example_batch([(2, [[3, 4], "action", "helper", [5, 90]])], hidden=64)
        a = actual.trajectory(batch, features)
        b = expected.trajectory(batch, features)
        question_end = batch["images"][0][1]["feature_end"]
        embeds = input_embeddings(expected.decoder, batch["input_ids"], features[:question_end], 80)
        replace = {}
        for i, action in enumerate(batch["actions"][0]):
            replace.update({position: b.actions[0, i, k] for k, position in enumerate(action["positions"])})
        for i, segment in enumerate(batch["segments"][0]):
            replace.update({segment["position"] + k: b.interactions[0, i][k] for k in range(3)})
            replace.update({position: b.visuals[0, i, k] for k, position in enumerate(segment["visual_positions"])})
        inputs = torch.stack([replace.get(p, embeds[0, p]) for p in range(embeds.shape[1])])[None]
        native = expected.decoder(inputs_embeds=inputs, attention_mask=batch["attention_mask"],
                                  position_ids=batch["position_ids"], use_cache=False).last_hidden_state
        torch.testing.assert_close(a.hidden, native, atol=2e-5, rtol=2e-5)
        actual_loss = causal_ce_sum(a.hidden, batch["labels"], actual.lm_head, 3, False)
        expected_loss = causal_ce_sum(native, batch["labels"], expected.lm_head, 3, False)
        actual_loss.backward()
        expected_loss.backward()
        reference = dict(expected.named_parameters())
        for name, parameter in actual.named_parameters():
            if parameter.grad is None:
                self.assertIsNone(reference[name].grad, name)
            else:
                torch.testing.assert_close(parameter.grad, reference[name].grad, atol=5e-5, rtol=3e-3, msg=name)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA not available")
    def test_actual_qwen_bfloat16_four_loss_backward(self):
        full = tiny_qwen()
        model = LatentModel(full.model, full.lm_head, full.config,
                            recipe(activation_checkpointing=True)).to("cuda", torch.bfloat16)
        batch, features = example_batch(hidden=64)
        batch = to_device(batch, torch.device("cuda"))
        losses = model(batch, features.to("cuda", torch.bfloat16))
        self.assertTrue(all(bool(torch.isfinite(loss)) for loss in losses.values()))
        sum(losses.values()).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
        self.assertGreater(float(model.interaction.query_mlp[0].weight.grad.float().norm()), 0)
        self.assertGreater(float(model.aux_lm_head.weight.grad.float().norm()), 0)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA not available")
    def test_full_export_prepare_teacher_cache_transfer_pixels_and_student_update(self):
        from PIL import Image
        from transformers import Qwen2_5_VLForConditionalGeneration
        from src.sft1.checkpoint import export_warmup
        from src.sft1.model import WarmupModel
        from src.sft1.train import encode_images
        from src.sft2.config import Stage2Config
        from src.sft2.data import Stage2Dataset, Stage2Collator, load_processor
        from src.sft2.prepare import prepare
        from src.sft2.teacher import prepare_teacher_cache, validate_teacher_cache
        from tests.test_sft1_data import real_qwen_processor
        from tests.test_sft1_model import tiny_config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processor = real_qwen_processor()
            mc = tiny_qwen().config
            mc.vocab_size = len(processor.tokenizer)
            mc.image_token_id, mc.vision_start_token_id, mc.vision_end_token_id = 12, 10, 11
            mc.pad_token_id, mc.eos_token_id = processor.tokenizer.pad_token_id, 9
            full = Qwen2_5_VLForConditionalGeneration(mc)
            warmup_config = tiny_config(min_image_pixels=3136, max_image_pixels=6272)
            warmup = WarmupModel(full.model, full.lm_head, mc, warmup_config)
            export_warmup(SimpleNamespace(module=warmup), full.visual, processor, root / "warmup",
                          warmup_config, "tiny-offline-fixture")
            Image.new("RGB", (56, 56), "red").save(root / "question.png")
            Image.new("RGB", (84, 56), "blue").save(root / "helper.png")
            sample = {"text_input": "question", "image_input": ["question.png"], "original_final_answer": "7",
                      "sequence_plan": [
                          {"type": "text", "content": [
                              {"sub_type": "analysis", "content": "reason"},
                              {"sub_type": "action", "content": "SHORT", "original_text": "look"}]},
                          {"type": "latent", "helper_image": "helper.png"},
                          {"type": "text", "content": "answer 7", "is_final_answer": True}]}
            source_file = root / "train.jsonl"
            source_file.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            source = dict(name="tiny", path=str(source_file), image_root=str(root), plan_field="sequence_plan",
                          require_structured_actions=True)
            config = Stage2Config(warmup_path=str(root / "warmup"), sources=[source],
                                  prepared_dir=str(root / "prepared"), output_dir=str(root / "output"),
                                  deepspeed=str(Path(__file__).resolve().parents[1] / "configs/sft2/zero2.json"),
                                  attention_backend="sdpa", fusion_dim=16, fusion_heads=2,
                                  fusion_ffn_dim=32, fusion_score_dim=8, max_seq_length=512,
                                  min_image_pixels=3136, max_image_pixels=6272, prepare_workers=1,
                                  num_workers=0, activation_checkpointing=True).validate().resolve_paths()
            prepare(config)
            teacher_manifest = prepare_teacher_cache(config)
            dataset = Stage2Dataset(config.prepared_dir)
            self.assertEqual(validate_teacher_cache(config, dataset)["identity"], teacher_manifest["identity"])
            student, visual = load_training_components(config, torch.device("cuda"))
            self.assertFalse(any("warmup_adapter" in name for name, _ in student.named_parameters()))
            for name, value in warmup.interaction.state_dict().items():
                torch.testing.assert_close(student.interaction.state_dict()[name].cpu(), value.to(torch.bfloat16))
            processor = load_processor(config)
            collator = Stage2Collator(processor, student.model_config.to_dict(), dataset.manifest, config.teacher_cache_dir)
            batch = to_device(collator([dataset[0]]), torch.device("cuda"))
            self.assertEqual(batch["teacher_targets"].shape, (1, 1, 64))
            features = encode_images(visual, batch)
            losses = student(batch, features)
            loss = combined_loss(losses, batch["loss_counts"], loss_weights(config), 1, 1)
            before = student.interaction.query_mlp[0].weight.detach().clone()
            optimizer = torch.optim.AdamW(student.parameters(), lr=1e-2)
            loss.backward()
            optimizer.step()
            self.assertFalse(torch.equal(before, student.interaction.query_mlp[0].weight))
            self.assertTrue(all(parameter.grad is None for parameter in visual.parameters()))
            self.assertTrue(all(bool(torch.isfinite(value)) for value in losses.values()))
            # The enlarged vocabulary and special tokens must survive an actual
            # stage-II component export, not just a synthetic config dictionary.
            from src.sft2.checkpoint import export_stage2
            from src.sft2.tokens import ensure_boundary_tokens
            from src.interaction.model import load_full_model
            from transformers import Qwen2_5_VLProcessor
            export = root / "stage2_export"
            export_stage2(SimpleNamespace(module=student), visual, processor, export, config,
                          "tiny-warmup", teacher_manifest["identity"])
            restored = load_full_model(str(export / "backbone"), "sdpa")
            restored_processor = Qwen2_5_VLProcessor.from_pretrained(export / "backbone")
            start_id, end_id = ensure_boundary_tokens(restored_processor)
            self.assertEqual([start_id, end_id], batch["latent_boundary_ids"])
            self.assertEqual(restored.config.vocab_size, student.model_config.vocab_size)
            self.assertEqual(restored.config.latent_start_id, start_id)
            self.assertEqual(restored.config.latent_end_id, end_id)
            self.assertEqual(restored_processor._sft2_added_token_ids, ())
            torch.testing.assert_close(restored.model.embed_tokens.weight,
                                       student.decoder.embed_tokens.weight.cpu(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
