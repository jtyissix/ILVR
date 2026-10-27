"""Deterministic stage-II boundary vocabulary; the stage-I teacher is unchanged."""
from ..sft1 import data as stage1

BOUNDARY_TOKENS = ("<|latent_start|>", "<|latent_end|>")
BOUNDARY_POLICY = "start_action4_interaction3_visual16_end_v2"


def ensure_boundary_tokens(processor):
    tokenizer = processor.tokenizer
    before = tokenizer.get_vocab()
    missing_special = [token for token in BOUNDARY_TOKENS if token not in tokenizer.all_special_tokens]
    if missing_special:
        tokenizer.add_special_tokens({"additional_special_tokens": missing_special},
                                     replace_additional_special_tokens=False)
    ids = tuple(tokenizer.get_vocab()[token] for token in BOUNDARY_TOKENS)
    if (len(set(ids)) != 2 or tokenizer.pad_token_id in ids or tokenizer.eos_token_id in ids
            or any(tokenizer.encode(token, add_special_tokens=False) != [idx]
                   for token, idx in zip(BOUNDARY_TOKENS, ids))):
        raise ValueError("Latent boundaries must be distinct single special tokens, separate from PAD/EOS")
    # Retain the list on repeated preparation calls, including padded vocab rows
    # that already exist in the weight matrix but had no tokenizer entry.
    added = set(getattr(processor, "_sft2_added_token_ids", ()))
    added.update(idx for token, idx in zip(BOUNDARY_TOKENS, ids) if token not in before)
    processor._sft2_added_token_ids = tuple(sorted(added))
    return ids


def student_model_config(processor, model_config):
    start, end = ensure_boundary_tokens(processor)
    added = set(processor._sft2_added_token_ids)
    if any(idx < 0 or idx >= model_config["vocab_size"]
           for idx in processor.tokenizer.get_vocab().values() if idx not in added):
        raise ValueError("The original stage-I tokenizer exceeds model capacity; only new boundary rows may be added")
    return {**model_config, "vocab_size": max(model_config["vocab_size"], len(processor.tokenizer)),
            "latent_start_id": start, "latent_end_id": end}


def validate_tokenizer_capacity(processor, model_config):
    stage1.validate_tokenizer_capacity(processor, student_model_config(processor, model_config))


def load_processor(config):
    processor = stage1.load_processor(config)
    ensure_boundary_tokens(processor)
    return processor


def initialize_student_vocabulary(full, processor):
    """Preserve existing rows; initialize only newly registered boundary rows."""
    import torch
    start, end = ensure_boundary_tokens(processor)
    student_model_config(processor, full.config.to_dict())
    old_size = full.get_input_embeddings().weight.shape[0]
    added = getattr(processor, "_sft2_added_token_ids", ())
    # Compute deterministic means before resizing; no full FP32 matrix copy.
    with torch.no_grad():
        input_mean = full.get_input_embeddings().weight.mean(0, dtype=torch.float32)
        output_mean = full.get_output_embeddings().weight.mean(0, dtype=torch.float32)
        if len(processor.tokenizer) > old_size:
            full.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
        for idx in added:
            full.get_input_embeddings().weight[idx].copy_(input_mean)
            full.get_output_embeddings().weight[idx].copy_(output_mean)
    full.config.latent_start_id, full.config.latent_end_id = start, end
    full.model.config.latent_start_id, full.model.config.latent_end_id = start, end
    full.config.sft2_boundary_policy = BOUNDARY_POLICY
