"""Native Qwen reference unroll, independent of DecoderExecutor's latent loop."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.interaction.execution import causal_ce_sum
from src.interaction.model import assert_local_transformers, input_embeddings
from src.sft3.model import RelaxationModel
from tests.test_interaction_qwen import tiny_qwen
from tests.test_sft3_model import example, recipe


class RelaxationQwenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert_local_transformers()
        torch.set_num_threads(1)

    def test_native_full_prefix_unroll_matches_hidden_states_ce_and_gradients(self):
        torch.manual_seed(321)
        full = tiny_qwen()
        model = RelaxationModel(full.model, full.lm_head, full.config, recipe(activation_checkpointing=True))
        reference = copy.deepcopy(full)
        batch, features = example([(1, [[3], "action", "helper", [4, 90]])], hidden=64)
        actual = model.trajectory(batch, features)
        positions = set(batch["actions"][0][0]["positions"])
        segment = batch["segments"][0][0]
        positions.update(range(segment["position"], segment["position"] + 3))
        positions.update(segment["visual_positions"])
        embeds = input_embeddings(reference.model, batch["input_ids"], features, 80)
        inputs, outputs = [], []
        for index in range(embeds.shape[1]):
            inputs.append(outputs[-1] if index in positions else embeds[:, index:index + 1])
            hidden = reference.model(inputs_embeds=torch.cat(inputs, dim=1),
                                     attention_mask=batch["attention_mask"][:, :index + 1],
                                     position_ids=batch["position_ids"][:, :, :index + 1],
                                     use_cache=False).last_hidden_state
            outputs.append(hidden[:, -1:])
        expected = torch.cat(outputs, dim=1)
        torch.testing.assert_close(actual.hidden, expected, rtol=2e-5, atol=2e-5)
        a = causal_ce_sum(actual.hidden, batch["labels"], model.lm_head, 3, True)
        b = causal_ce_sum(expected, batch["labels"], reference.lm_head, 3, False)
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
        a.backward()
        b.backward()
        for (name, p), (other, q) in zip(model.decoder.named_parameters(), reference.model.named_parameters()):
            self.assertEqual(name, other)
            torch.testing.assert_close(p.grad, q.grad, rtol=3e-3, atol=1e-4, msg=name)
        torch.testing.assert_close(model.lm_head.weight.grad, reference.lm_head.weight.grad, rtol=3e-3, atol=1e-4)

    def test_real_stage2_backbone_preflight_load_pixels_update_and_export(self):
        from PIL import Image
        from src.interaction.data import file_hash, json_write
        from src.sft1.data import _processor_identity
        from src.sft1.train import encode_images
        from src.sft2.checkpoint import FORMAT
        from src.sft2.config import Stage2Config
        from src.sft2.data import CACHE_VERSION, TOKENIZATION_POLICY, prepare_sample
        from src.sft2.tokens import BOUNDARY_POLICY, initialize_student_vocabulary
        from src.sft3.checkpoint import export_stage3
        from src.sft3.config import Stage3Config
        from src.sft3.data import Stage3Collator, Stage3Dataset, load_processor
        from src.sft3.model import load_training_components
        from src.sft3.prepare import prepare, read_preflight
        from src.sft3.train import combined_loss, loss_weights, optimizer_groups
        from tests.test_sft1_data import real_qwen_processor
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processor = real_qwen_processor()
            full = tiny_qwen()
            for name, value in {"image_token_id": 12, "vision_start_token_id": 10, "vision_end_token_id": 11,
                                "pad_token_id": processor.tokenizer.pad_token_id,
                                "eos_token_id": processor.tokenizer.eos_token_id}.items():
                setattr(full.config, name, value)
            initialize_student_vocabulary(full, processor)
            bundle = root / "stage2"
            backbone = bundle / "backbone"
            full.save_pretrained(backbone)
            processor.save_pretrained(backbone)
            json_write(bundle / "stage2_recipe.json", {"format": FORMAT, "config": Stage2Config().to_dict()})
            prepared = root / "prepared"
            prepared.mkdir()
            for name in ("question", "helper"):
                Image.new("RGB", (28, 28), "red").save(root / (name + ".png"))
            source = {"name": "fixture", "image_root": str(root), "plan_field": "thought_rewritten"}
            row = {"text_input": "question", "image_input": ["question.png"], "original_final_answer": "7",
                   "thought_rewritten": [
                       {"type": "text", "content": [{"sub_type": "analysis", "content": "reason"},
                           {"sub_type": "action", "content": "SHORT", "original_text": "look"}]},
                       {"type": "latent", "helper_image": "helper.png"},
                       {"type": "text", "content": "answer 7"}]}
            row["thought_rewritten"][2:2] = copy.deepcopy(row["thought_rewritten"][:2])
            record = prepare_sample(row, source, processor, full.config.to_dict(), 4096)
            cache = prepared / "records.jsonl"
            cache.write_text(json.dumps(record) + "\n", encoding="utf-8")
            manifest = {"version": CACHE_VERSION, "stage": "sft2_latent", "boundary_policy": BOUNDARY_POLICY,
                        "tokenization_policy": TOKENIZATION_POLICY, "samples": 1, "records_sha256": file_hash(cache),
                        "processor_identity": _processor_identity(processor),
                        "latent_boundary_ids": record["latent_boundary_ids"],
                        "action_latents": 4, "interaction_tokens": 3, "visual_latents": 16,
                        "image_stats": {i["path"]: [Path(i["path"]).stat().st_size, Path(i["path"]).stat().st_mtime_ns]
                                        for i in record["images"]}}
            json_write(prepared / "manifest.json", manifest)
            (root / "helper.png").unlink()
            config = Stage3Config(stage2_path=str(bundle), prepared_dir=str(prepared),
                                  output_dir=str(root / "output"), max_image_pixels=6272,
                                  mode="prepare_only", attention_backend="sdpa", collapse_targets="123",
                                  collapse_margin=0).resolve_paths()
            with patch.dict(os.environ, {"SFT3_PREFLIGHT_ID": "integration-test"}):
                receipt = prepare(config)
                self.assertEqual(read_preflight(config), receipt)
                changed = copy.deepcopy(config)
                changed.collapse_weight = .7
                with self.assertRaises(ValueError):
                    read_preflight(changed)
            student, visual = load_training_components(config, torch.device("cpu"))
            self.assertTrue(all(not p.requires_grad for p in visual.parameters()))
            self.assertTrue(all(p.requires_grad for p in student.parameters()))
            torch.testing.assert_close(student.decoder.embed_tokens.weight,
                                       full.model.embed_tokens.weight.to(torch.bfloat16), rtol=0, atol=0)
            loaded_processor = load_processor(config)
            dataset = Stage3Dataset(prepared)
            collator = Stage3Collator(loaded_processor, student.model_config.to_dict(), manifest, config)
            batch = collator([dataset[0]])
            features = encode_images(visual, batch)
            self.assertEqual(len(batch["images"][0]), 1)
            # Test CPU FP32 optimization; production continues with BF16 CUDA.
            student.float()
            before = student.lm_head.weight.detach().clone()
            optimizer = torch.optim.AdamW(optimizer_groups(student, config))
            losses = student(batch, features.float())
            loss = combined_loss(losses, batch["loss_counts"], loss_weights(config), 1, 1)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
            self.assertFalse(torch.equal(before, student.lm_head.weight))
            self.assertTrue(all(p.grad is None for p in visual.parameters()))
            student.bfloat16()
            export_stage3(SimpleNamespace(module=student), visual, loaded_processor, root / "export", config,
                          receipt["stage2_fingerprint"])
            from src.interaction.model import load_full_model
            reloaded = load_full_model(str(root / "export/backbone"), "sdpa")
            torch.testing.assert_close(reloaded.model.embed_tokens.weight, student.decoder.embed_tokens.weight,
                                       rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
