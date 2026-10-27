"""Boundary registration, padded vocabulary migration and embedding expansion."""
import unittest
import torch

from src.sft2.tokens import ensure_boundary_tokens, initialize_student_vocabulary, student_model_config
from tests.test_interaction_qwen import tiny_qwen
from tests.test_sft1_data import real_qwen_processor


class BoundaryVocabularyTests(unittest.TestCase):
    def test_new_tokens_initialize_padded_or_expanded_rows_and_preserve_old_weights(self):
        for padded in (True, False):
            with self.subTest(padded=padded):
                processor, full = real_qwen_processor(), tiny_qwen()
                original_length = len(processor.tokenizer)
                if not padded:
                    full.resize_token_embeddings(original_length, mean_resizing=False)
                embedding = full.get_input_embeddings().weight.detach().clone()
                head = full.get_output_embeddings().weight.detach().clone()
                initialize_student_vocabulary(full, processor)
                ids = ensure_boundary_tokens(processor)
                self.assertEqual(ids, (original_length, original_length + 1))
                self.assertEqual(full.config.vocab_size, max(len(embedding), original_length + 2))
                self.assertEqual(full.config.latent_start_id, ids[0])
                self.assertEqual(full.config.latent_end_id, ids[1])
                for token_id in ids:
                    torch.testing.assert_close(full.get_input_embeddings().weight[token_id], embedding.mean(0))
                    torch.testing.assert_close(full.get_output_embeddings().weight[token_id], head.mean(0))
                unchanged = [i for i in range(len(embedding)) if i not in ids]
                torch.testing.assert_close(full.get_input_embeddings().weight[unchanged], embedding[unchanged], rtol=0, atol=0)
                torch.testing.assert_close(full.get_output_embeddings().weight[unchanged], head[unchanged], rtol=0, atol=0)
                self.assertEqual(student_model_config(processor, {"vocab_size": len(embedding)})["vocab_size"],
                                 full.config.vocab_size)

    def test_existing_boundaries_keep_trained_rows_and_registration_is_idempotent(self):
        processor, full = real_qwen_processor(), tiny_qwen()
        ids = ensure_boundary_tokens(processor)
        self.assertEqual(ensure_boundary_tokens(processor), ids)
        self.assertEqual(len(processor.tokenizer), ids[-1] + 1)
        # A processor loaded from an existing marked checkpoint has no new IDs.
        del processor._sft2_added_token_ids
        before = full.get_input_embeddings().weight.detach().clone()
        initialize_student_vocabulary(full, processor)
        self.assertEqual(processor._sft2_added_token_ids, ())
        torch.testing.assert_close(full.get_input_embeddings().weight, before, rtol=0, atol=0)

    def test_boundary_migration_cannot_hide_an_incompatible_original_vocabulary(self):
        processor, full = real_qwen_processor(), tiny_qwen()
        full.resize_token_embeddings(len(processor.tokenizer) - 1, mean_resizing=False)
        with self.assertRaisesRegex(ValueError, "original stage-I tokenizer"):
            initialize_student_vocabulary(full, processor)


if __name__ == "__main__":
    unittest.main()
