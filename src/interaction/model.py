"""Separate trainable student, frozen visual encoder and frozen reference decoder."""
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from .execution import DecoderExecutor, causal_ce_sum, run_trajectory
from .fusion import RoundTripFusion


def input_embeddings(decoder, input_ids, image_features, image_token_id):
    values = decoder.embed_tokens(input_ids)
    if image_features is not None:
        mask = input_ids.eq(image_token_id)
        if int(mask.sum()) != image_features.shape[0]:
            raise ValueError("Input image feature count does not match the prepared token sequence")
        values = values.masked_scatter(mask[:, :, None], image_features.to(values.dtype))
    return values


@torch.no_grad()
def reference_contexts(decoder, batch, features, image_token_id, backend="flash_attention_2", execution="incremental"):
    decoder.eval()
    if not any(batch["segments"]):
        return {}
    embeds = input_embeddings(decoder, batch["input_ids"], features, image_token_id)
    executor = DecoderExecutor(decoder, backend, activation_checkpointing=False)
    # No reference state after the final latent is consumed by fusion.
    end = max(batch["latent_positions"]) + 1
    trajectory = run_trajectory(executor, embeds[:, :end], batch["attention_mask"][:, :end], batch["position_ids"][:, :, :end],
                                batch["latent_positions"], execution=execution)
    contexts = {}
    for row, segments in enumerate(batch["segments"]):
        for index, segment in enumerate(segments):
            text = trajectory.hidden[row, segment["text"]].detach()
            visual = torch.stack([trajectory.latents[row, index, step] for step in range(8)]).detach()
            contexts[row, index] = (text, visual)
    # No reference KV cache or complete hidden-state history escapes this function.
    return contexts


class InteractionStudent(nn.Module):
    def __init__(self, decoder, lm_head, model_config, config):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.model_config = model_config
        self.recipe = config
        self.fusion = RoundTripFusion(model_config.hidden_size, config.fusion_dim,
                                      config.fusion_heads, config.fusion_ffn_dim) if config.mode == "interaction_ce" else None
        self.executor = DecoderExecutor(decoder, config.attention_backend, config.activation_checkpointing)

    def forward(self, batch, image_features=None, contexts=None, events=None, padded_ce_count=None):
        embeds = input_embeddings(self.decoder, batch["input_ids"], image_features, self.model_config.image_token_id)
        injected = {}
        if self.fusion is not None:
            contexts = contexts or {}
            keys = list(contexts)
            if keys:
                # At least one padded token for empty local rationales; its mask is false.
                texts = [contexts[key][0] for key in keys]
                length = max(1, max(t.shape[0] for t in texts))
                text = pad_sequence([t if len(t) else t.new_zeros(1, t.shape[-1]) for t in texts], batch_first=True)
                text_mask = torch.arange(length, device=embeds.device)[None] < torch.tensor(
                    [len(t) for t in texts], device=embeds.device)[:, None]
                visual = torch.stack([contexts[key][1] for key in keys])
            else:
                # All ranks call every fusion submodule even for a no-latent micro-batch.
                text = embeds.new_zeros(1, 1, embeds.shape[-1])
                visual = embeds.new_zeros(1, 8, embeds.shape[-1])
                text_mask = torch.zeros(1, 1, dtype=torch.bool, device=embeds.device)
            fused = self.fusion(text.to(embeds.dtype), visual.to(embeds.dtype), text_mask)
            injected = {key: fused[i] for i, key in enumerate(keys)}
            # A no-latent rank must traverse fusion backward AFTER its decoder,
            # like ranks with injected states; important for ZeRO-3 collectives.
            embeds = embeds + fused.sum() * 0.0
        trajectory = run_trajectory(self.executor, embeds, batch["attention_mask"], batch["position_ids"],
                                    batch["latent_positions"], injected, self.recipe.execution, events)
        loss = causal_ce_sum(trajectory.hidden, batch["labels"], self.lm_head,
                             self.recipe.ce_chunk_size, self.recipe.activation_checkpointing, padded_ce_count)
        return loss


def assert_local_transformers():
    from pathlib import Path
    import transformers
    expected = Path(__file__).resolve().parents[2] / "transformers" / "src" / "transformers"
    if Path(transformers.__file__).resolve().parent != expected.resolve():
        raise RuntimeError("Use this repository's Transformers: python -m pip install -e ./transformers")


def load_full_model(path, backend, dtype=torch.bfloat16):
    assert_local_transformers()
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    config = Qwen2_5_VLConfig.from_pretrained(path, local_files_only=True)
    # Older interaction exports persisted this runtime flag, skipping propagation
    # of the requested backend into vision_config and silently selecting SDPA.
    config._attn_implementation_autoset = False
    config.vision_config._attn_implementation_autoset = False
    model, info = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path, config=config, local_files_only=True, torch_dtype=dtype, attn_implementation=backend,
        low_cpu_mem_usage=True, output_loading_info=True,
    )
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("unexpected_keys"):
        raise ValueError(f"Checkpoint does not load exactly: {info}")
    expected = {"flash_attention_2": "Qwen2_5_VLVisionFlashAttention2",
                "sdpa": "Qwen2_5_VLVisionSdpaAttention", "eager": "Qwen2_5_VLVisionAttention"}[backend]
    if any(type(block.attn).__name__ != expected for block in model.visual.blocks):
        raise RuntimeError(f"Visual attention backend does not match requested {backend}")
    return model


def load_training_components(config, device):
    full = load_full_model(config.model_path, config.attention_backend)
    model_config = full.config
    visual = full.visual.requires_grad_(False).eval().to(device)
    student = InteractionStudent(full.model, full.lm_head, model_config, config)
    del full
    # ZeRO partitions only trainable components; frozen encoders stay outside its engine.
    student.to(device=device, dtype=torch.bfloat16)
    reference = None
    if config.mode == "interaction_ce":
        ref = load_full_model(config.reference_path or config.model_path, config.attention_backend)
        reference = ref.model.requires_grad_(False).eval().to(device)
        del ref  # reference needs no vocabulary head or second copy of the vision tower
    return student, visual, reference
