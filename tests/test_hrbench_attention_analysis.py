import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import hrbench_attention_analysis as analysis


HAS_TORCH = importlib.util.find_spec("torch") is not None


class FakeOutputHead:
    def __init__(self):
        import torch

        self.weight = torch.nn.Parameter(
            torch.arange(32, dtype=torch.float32).reshape(8, 4) / 10
        )

    def parameters(self):
        yield self.weight

    def __call__(self, value):
        return value @ self.weight.T


class FakeModel:
    def __init__(self):
        self.head = FakeOutputHead()

    def get_output_embeddings(self):
        return self.head


class AttentionAnalysisTest(unittest.TestCase):
    def test_selection_is_deterministic(self):
        self.assertEqual(
            analysis.select_sample_indices(20, "random", 0, 5, 7),
            analysis.select_sample_indices(20, "random", 0, 5, 7),
        )
        self.assertEqual(
            analysis.select_sample_indices(20, "sequential", 3, 4, 99),
            [3, 4, 5, 6],
        )

    def test_interleaved_text_is_separate_from_answer_history(self):
        latent_ids = {"pad": 99, "start": 100, "end": 101}
        generated = [10, 100, 99, 101, 11, 100, 99, 101, 20, 21]
        prompt_length = 3
        latent_positions = {prompt_length + 2, prompt_length + 6}
        kinds = analysis.generated_source_kinds(
            generated,
            latent_ids,
            {99, 100, 101},
            latent_positions,
            prompt_length,
        )
        self.assertEqual(
            kinds.tolist(),
            [
                analysis.SOURCE_COT_TEXT,
                analysis.SOURCE_SPECIAL,
                analysis.SOURCE_LATENT,
                analysis.SOURCE_SPECIAL,
                analysis.SOURCE_COT_TEXT,
                analysis.SOURCE_SPECIAL,
                analysis.SOURCE_LATENT,
                analysis.SOURCE_SPECIAL,
                analysis.SOURCE_ANSWER_HISTORY,
                analysis.SOURCE_ANSWER_HISTORY,
            ],
        )

    def test_no_latent_has_no_cot_text(self):
        kinds = analysis.generated_source_kinds(
            [10, 11],
            {"pad": 99, "start": 100, "end": 101},
            {99, 100, 101},
            set(),
            2,
        )
        self.assertEqual(
            kinds.tolist(),
            [analysis.SOURCE_ANSWER_HISTORY, analysis.SOURCE_ANSWER_HISTORY],
        )

    def test_target_normalization_excludes_answer_history_and_special(self):
        kinds = np.asarray(
            [
                analysis.SOURCE_INPUT_TEXT,
                analysis.SOURCE_INPUT_VISUAL,
                analysis.SOURCE_LATENT,
                analysis.SOURCE_COT_TEXT,
                analysis.SOURCE_ANSWER_HISTORY,
                analysis.SOURCE_SPECIAL,
            ]
        )
        raw = np.asarray([[0.1, 0.2, 0.1, 0.2, 0.3, 0.1]], dtype=np.float32)
        normalized = analysis.normalize_attention_groups(
            raw, kinds, analysis.TARGET_SOURCE_CODES
        )
        np.testing.assert_allclose(
            normalized,
            [[1 / 6, 2 / 6, 1 / 6, 2 / 6, 0.0, 0.0]],
            rtol=1e-6,
        )

    def test_complete_latent_block_count_is_validated(self):
        latent_ids = {"pad": 99, "start": 100, "end": 101}
        manifest = {
            "generated_token_ids": [100, 99, 99, 101],
            "latent_records": [
                {"latent_block_index": 0},
                {"latent_block_index": 0},
            ],
        }
        analysis.validate_complete_latent_blocks(manifest, latent_ids, 2)
        with self.assertRaisesRegex(RuntimeError, "expected 3"):
            analysis.validate_complete_latent_blocks(manifest, latent_ids, 3)

    def test_cleaned_output_preserves_delimiters_and_hides_latents(self):
        raw = "a<|latent_start|>secret<|latent_end|>b"
        self.assertEqual(
            analysis.clean_latent_output(raw),
            "a<|latent_start|><latent><|latent_end|>b",
        )

    def _record_layers(self, recorder, source_count, values):
        import torch

        for layer_index in range(2):
            vector = torch.tensor([values[layer_index]], dtype=torch.float32)
            self.assertEqual(vector.shape, (1, source_count))
            recorder.record_layer_attention(layer_index, vector)

    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed in this test environment")
    def test_selective_qk_reconstruction_matches_direct_softmax(self):
        import torch

        transformers_src = Path(__file__).resolve().parents[1] / "transformers" / "src"
        sys.path.insert(0, str(transformers_src))
        try:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VLAttention,
            )
        finally:
            sys.path.pop(0)

        class Capture:
            value = None

            def wants_current_step(self):
                return True

            def record_layer_attention(self, layer_index, value):
                self.layer_index = layer_index
                self.value = value

        attention = object.__new__(Qwen2_5_VLAttention)
        torch.nn.Module.__init__(attention)
        attention.head_dim = 2
        attention.layer_idx = 3
        capture = Capture()
        attention._ilvr_attention_recorder = capture
        query = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
        keys = torch.tensor(
            [
                [
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                ]
            ]
        )
        attention._record_ilvr_analysis_attention(query, keys)
        direct = torch.softmax(
            torch.matmul(query.float(), keys.float().transpose(2, 3)) / math.sqrt(2),
            dim=-1,
        ).mean(dim=1)[:, 0, :]
        self.assertEqual(capture.layer_index, 3)
        torch.testing.assert_close(capture.value, direct)

    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed in this test environment")
    def test_recorder_and_archive_align_latent_topk_and_answer(self):
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            latent_ids = {"pad": 99, "start": 100, "end": 101}
            recorder = analysis.ILVRAttentionRecorder(
                directory,
                prompt_token_ids=[1, 2],
                image_positions=[1],
                special_token_ids={99, 100, 101},
                latent_token_ids=latent_ids,
                layer_count=2,
                top_k=2,
                storage_dtype="float32",
            )
            model = FakeModel()
            recorder.begin_generation(model, torch.tensor([[1, 2]]), latent_size=1)

            # Predict latent_start from the prompt's final query. This pending
            # answer candidate is discarded when the block begins.
            recorder.begin_step(model, 1, 0, False, None)
            self._record_layers(recorder, 2, [[0.4, 0.6], [0.3, 0.7]])
            recorder.end_step(torch.tensor([100]))

            # Consume latent_start and predict latent_pad; no query is retained.
            recorder.begin_step(model, 2, 1, False, None)
            self.assertFalse(recorder.wants_current_step())
            recorder.end_step(torch.tensor([99]))

            # The injected vector is consumed at latent_pad and predicts end.
            latent_vector = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]])
            recorder.begin_step(model, 3, 2, True, latent_vector)
            self._record_layers(
                recorder,
                4,
                [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
            )
            recorder.end_step(torch.tensor([101]))

            recorder.begin_step(model, 4, 3, False, None)
            self._record_layers(
                recorder,
                5,
                [[0.1, 0.2, 0.2, 0.2, 0.3], [0.2, 0.2, 0.2, 0.2, 0.2]],
            )
            recorder.end_step(torch.tensor([7]))
            recorder.end_generation(torch.tensor([[1, 2, 100, 99, 101, 7]]))

            manifest = recorder.manifest()
            self.assertEqual(len(manifest["latent_records"]), 1)
            self.assertEqual(len(manifest["latent_topk"]), 1)
            self.assertEqual(len(manifest["answer_records"]), 1)
            data = analysis.assemble_sample_archive(manifest, directory, latent_ids)
            self.assertEqual(
                data["query_kind_codes"].tolist(),
                [analysis.QUERY_LATENT, analysis.QUERY_ANSWER],
            )
            self.assertEqual(data["latent_topk_token_ids"].shape, (1, 2))
            np.testing.assert_allclose(
                data["category_attention_distribution"].sum(axis=-1),
                1.0,
                atol=1e-6,
            )
            # The answer query can see the previous answer source only on later
            # answer tokens; it is never folded into cot_text.
            self.assertEqual(int(data["query_latent_block_indices"][0]), 0)
            self.assertEqual(int(data["query_latent_indices"][0]), 0)


if __name__ == "__main__":
    unittest.main()
