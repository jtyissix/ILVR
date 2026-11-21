"""Qwen2.5-VL backbone inference, including autonomous continuous-latent decoding.

No imports from training/interaction modules. No auxiliary module is constructed.
"""
from dataclasses import dataclass
import inspect
from pathlib import Path

from .common import read_json


def resolve_backbone(directory):
    root = Path(directory).expanduser().resolve()
    if (root / "backbone/config.json").is_file():
        root = root / "backbone"
    if not (root / "config.json").is_file():
        raise ValueError(f"MODEL_DIR must be a local Qwen directory or exported bundle containing backbone/: {root}")
    if not list(root.glob("*.safetensors")) and not list(root.glob("pytorch_model*.bin")):
        raise ValueError("No HF model weights found; export with export_stage3/export_stage2/export_warmup first")
    return root


@dataclass
class LatentPolicy:
    start_id: int | None = None
    end_id: int | None = None
    min_steps: int = 1
    max_steps: int = 64
    enabled: bool = False

    def __post_init__(self):
        if not 1 <= self.min_steps <= self.max_steps:
            raise ValueError("Require 1 <= MIN_LATENT_STEPS <= MAX_LATENT_STEPS")
        if self.enabled and (self.start_id is None or self.end_id is None or self.start_id == self.end_id):
            raise ValueError("Autonomous latent inference requires two distinct trained boundary token IDs")


def decode_stream(backend, state, policy, max_steps):
    """Backend-independent state machine, tested separately from GPU execution.

    backend.next_token(state, banned) -> vocabulary token
    backend.advance(state, token=None) -> feed final hidden (None) or token embedding
    In latent mode vocabulary predictions are used ONLY to recognize latent_end.
    """
    ids, events, segment = [], [], None
    stopped, consumed = False, 0
    for step in range(max_steps):
        banned = {policy.end_id} if policy.enabled and segment is None else set()
        token = backend.next_token(state, banned - {None})
        feed_token = token
        if segment is not None:
            learned_end = token == policy.end_id and segment["steps"] >= policy.min_steps
            forced_end = segment["steps"] >= policy.max_steps
            if learned_end or forced_end:
                feed_token = policy.end_id
                segment.update(end_step=step, forced_end=forced_end and not learned_end, closed=True)
                segment = None
            else:
                feed_token = None
                segment["steps"] += 1
        elif token in backend.eos_ids:
            ids.append(token)
            stopped = True
            break
        elif policy.enabled and token == policy.start_id:
            segment = {"start_step": step, "steps": 0, "closed": False, "forced_end": False}
            events.append(segment)
        if feed_token is not None:
            ids.append(feed_token)
        state = backend.advance(state, feed_token)
        consumed += 1
    return {"token_ids": ids, "latent_segments": events, "stopped_on_eos": stopped,
            "truncated": not stopped, "decode_steps": consumed,
            "forced_latent_ends": sum(e["forced_end"] for e in events)}


class QwenBackbone:
    def __init__(self, model_dir, *, model_kind="auto", device="cuda:0", dtype="bfloat16",
                 attention="sdpa", min_pixels=None, max_pixels=None, min_latent_steps=1,
                 max_latent_steps=64, max_new_steps=4096, seed=42):
        import torch
        import transformers
        from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from transformers import set_seed
        self.torch = torch
        self.root = resolve_backbone(model_dir)
        self.max_new_steps = max_new_steps
        if max_new_steps <= 0:
            raise ValueError("MAX_NEW_STEPS must be positive")
        if model_kind not in ("auto", "base", "trained"):
            raise ValueError("MODEL_KIND must be auto/base/trained")
        config = AutoConfig.from_pretrained(self.root, local_files_only=True, trust_remote_code=False)
        if config.model_type != "qwen2_5_vl":
            raise ValueError("Only Qwen2.5-VL checkpoints are supported")
        stage = ("sft3" if getattr(config, "sft3_latent_relaxation_backbone", False) else
                 "sft2" if getattr(config, "sft2_helper_conditioned_backbone", False) or
                 getattr(config, "sft2_boundary_policy", None) else "base")
        trained = bool(getattr(config, "sft3_latent_relaxation_backbone", False) or
                       getattr(config, "sft2_helper_conditioned_backbone", False) or
                       getattr(config, "sft2_boundary_policy", None))
        if model_kind == "trained":
            trained = True
        if model_kind == "base":
            trained = False
        # The project fork has optional training modules in its constructor.
        # The exported backbone intentionally uses stage1 and no compression head.
        config.stage = "stage1"
        config.compress_strategy = None
        processor_args = {k: v for k, v in {"min_pixels": min_pixels, "max_pixels": max_pixels}.items() if v is not None}
        self.processor = AutoProcessor.from_pretrained(self.root, local_files_only=True,
                                                       trust_remote_code=False, **processor_args)
        self.tokenizer = self.processor.tokenizer
        vocab = self.tokenizer.get_vocab()
        start, end = vocab.get("<|latent_start|>"), vocab.get("<|latent_end|>")
        if trained:
            for field, token, value in (("latent_start_id", "<|latent_start|>", start),
                                        ("latent_end_id", "<|latent_end|>", end)):
                if value is None or getattr(config, field, None) != value or self.tokenizer.encode(token, add_special_tokens=False) != [value]:
                    raise ValueError(f"Checkpoint/tokenizer mismatch for {field}; use the exported backbone's own tokenizer")
        self.policy = LatentPolicy(start, end, min_latent_steps, max_latent_steps, trained)
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        set_seed(seed)
        self.model, info = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.root, config=config, torch_dtype=torch_dtype, device_map={"": device},
            attn_implementation=attention, local_files_only=True, trust_remote_code=False,
            output_loading_info=True)
        failures = {k: v for k, v in info.items() if v}
        if failures:
            raise ValueError(f"Backbone must load completely, without ignored/uninitialized weights: {failures}")
        if not hasattr(self.model.model, "embed_tokens") or not hasattr(self.model, "visual"):
            raise RuntimeError("This evaluator targets transformers 4.51.3 / the project's compatible fork; install eval/requirements.txt")
        forbidden = ("aux", "interaction", "cross_attn", "cross_attention", "latent_projection")
        names = [n for n, _ in self.model.named_modules() if any(x in n.lower() for x in forbidden)]
        if names:
            raise ValueError(f"Unexpected auxiliary modules in backbone: {names}")
        self.model.eval()
        # The ILVR fork otherwise selects its Ulysses training attention path.
        self.decoder_kwargs = ({"generate_mode": True} if "generate_mode" in
                               inspect.signature(self.model.model.forward).parameters else {})
        self.embedding = self.model.get_input_embeddings()
        self.input_device = self.embedding.weight.device
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        self.eos_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
        self.eos_ids.add(self.tokenizer.eos_token_id)
        self.eos_ids.discard(None)
        invalid = set(range(config.vocab_size)) - set(vocab.values())
        if self.tokenizer.pad_token_id not in self.eos_ids:
            invalid.add(self.tokenizer.pad_token_id)
        if not trained:
            invalid.update({start, end})
        self.invalid_ids = sorted(invalid - {None})
        self.runtime = {"model_dir": str(self.root), "transformers": transformers.__version__,
                        "training_stage": stage,
                        "decoder_inference_kwargs": self.decoder_kwargs,
                        "torch": torch.__version__, "latent_enabled": trained,
                        "latent_start_id": start, "latent_end_id": end,
                        "loaded_modules": [n for n, _ in self.model.named_children()],
                        "image_min_pixels": self.processor.image_processor.min_pixels,
                        "image_max_pixels": self.processor.image_processor.max_pixels,
                        "device": device, "dtype": dtype, "attention": attention,
                        "generation": "greedy", "max_new_steps": max_new_steps}

    def _inputs(self, messages):
        from qwen_vl_utils import process_vision_info
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  add_vision_id=True)
        # Use the SAME bounds in qwen-vl-utils and processor. Otherwise utils may
        # downsample a large HRBench image before the processor sees it.
        import copy
        bounded = copy.deepcopy(messages)
        for msg in bounded:
            for part in msg["content"]:
                if part["type"] == "image":
                    part["min_pixels"] = self.processor.image_processor.min_pixels
                    part["max_pixels"] = self.processor.image_processor.max_pixels
        images, videos = process_vision_info(bounded)
        inputs = self.processor(text=[text], images=images, videos=videos, padding=False, return_tensors="pt")
        return {k: v.to(self.input_device) for k, v in inputs.items()}

    def _embed(self, inputs):
        torch = self.torch
        ids = inputs["input_ids"]
        embeds = self.embedding(ids)
        if "pixel_values" in inputs:
            vision = self.model.visual
            parameter = next(vision.parameters())
            features = vision(inputs["pixel_values"].to(device=parameter.device, dtype=parameter.dtype),
                              grid_thw=inputs["image_grid_thw"].to(parameter.device)).to(embeds.device, embeds.dtype)
            mask = ids.eq(self.model.config.image_token_id)
            if int(mask.sum()) != len(features):
                raise ValueError("Image tokens/features count mismatch")
            embeds = embeds.masked_scatter(mask.unsqueeze(-1).expand_as(embeds), features)
        positions, _ = self.model.get_rope_index(input_ids=ids, image_grid_thw=inputs.get("image_grid_thw"),
                                                attention_mask=inputs["attention_mask"])
        return embeds, positions

    def _prefill(self, inputs):
        embeds, positions = self._embed(inputs)
        length = embeds.shape[1]
        self._check_context(length + self.max_new_steps)
        output = self.model.model(inputs_embeds=embeds, position_ids=positions,
                                  attention_mask=inputs["attention_mask"], use_cache=True, return_dict=True,
                                  cache_position=self.torch.arange(length, device=self.input_device), **self.decoder_kwargs)
        return {"hidden": output.last_hidden_state[:, -1:].clone(), "cache": output.past_key_values,
                "mask": inputs["attention_mask"], "next_position": int(positions.max()) + 1,
                "length": length}

    def _check_context(self, length):
        limit = self.model.config.max_position_embeddings
        if length > limit:
            raise ValueError(f"Requested sequence {length} exceeds model context {limit}; lower MAX_PIXELS or MAX_NEW_STEPS")

    def next_token(self, state, banned):
        logits = self.model.lm_head(state["hidden"][:, -1].to(self.model.lm_head.weight.device)).float()
        forbidden = self.invalid_ids + list(banned)
        if forbidden:
            logits[:, forbidden] = -float("inf")
        return int(logits.argmax(-1).item())

    def advance(self, state, token=None):
        torch = self.torch
        if token is None:
            # Final normalized decoder hidden -> next continuous input, exactly
            # the recurrent path in sft2/model.py; never convert it to a vocab ID.
            embeds = state["hidden"].to(self.input_device, self.embedding.weight.dtype)
        else:
            embeds = self.embedding(torch.tensor([[token]], device=self.input_device))
        mask = torch.cat([state["mask"], state["mask"].new_ones((1, 1))], dim=1)
        positions = torch.full((3, 1, 1), state["next_position"], device=self.input_device, dtype=torch.long)
        output = self.model.model(inputs_embeds=embeds, attention_mask=mask, position_ids=positions,
                                  past_key_values=state["cache"], use_cache=True, return_dict=True,
                                  cache_position=torch.tensor([state["length"]], device=self.input_device), **self.decoder_kwargs)
        return {"hidden": output.last_hidden_state[:, -1:].clone(), "cache": output.past_key_values,
                "mask": mask, "next_position": state["next_position"] + 1, "length": state["length"] + 1}

    def generate(self, messages):
        with self.torch.inference_mode():
            inputs = self._inputs(messages)
            state = self._prefill(inputs)
            result = decode_stream(self, state, self.policy, self.max_new_steps)
            result["text"] = self.tokenizer.decode(result["token_ids"], skip_special_tokens=True)
            result["raw_text"] = self.tokenizer.decode(result["token_ids"], skip_special_tokens=False)
            result["method"] = "generation"
            result["prompt_tokens"] = inputs["input_ids"].shape[1]
            result["image_grid_thw"] = inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else []
            return result

    def likelihood(self, messages, choices, *, include_eos=True):
        """Mean candidate-token NLL; prompt/image tokens never contribute to loss.

        Recompute each candidate with shared *image embeddings*, avoiding mutable
        KV-cache contamination between options. Only project answer hidden states.
        """
        torch = self.torch
        losses, lengths = {}, {}
        with torch.inference_mode():
            inputs = self._inputs(messages)
            prefix, positions = self._embed(inputs)
            prefix_len, next_position = prefix.shape[1], int(positions.max()) + 1
            for key, answer in choices.items():
                tokens = self.tokenizer.encode(str(answer), add_special_tokens=False)
                if include_eos:
                    tokens.append(self.tokenizer.eos_token_id)
                if not tokens:
                    raise ValueError("Empty candidate token sequence")
                self._check_context(prefix_len + len(tokens))
                ids = torch.tensor([tokens], device=self.input_device)
                embeds = torch.cat([prefix, self.embedding(ids)], dim=1)
                option_positions = torch.arange(next_position, next_position + len(tokens), device=self.input_device)[None, None].expand(3, 1, -1)
                output = self.model.model(inputs_embeds=embeds,
                    position_ids=torch.cat([positions, option_positions], dim=-1),
                    attention_mask=torch.ones((1, embeds.shape[1]), device=self.input_device, dtype=torch.long),
                    use_cache=False, return_dict=True, **self.decoder_kwargs)
                # Predict first answer token from last prompt state, last answer
                # token from penultimate state. Never score prompt or final state.
                hidden = output.last_hidden_state[:, prefix_len - 1:-1]
                logits = self.model.lm_head(hidden.to(self.model.lm_head.weight.device)).float()
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                         ids.to(logits.device).reshape(-1), reduction="mean")
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite candidate NLL")
                losses[key], lengths[key] = float(loss), len(tokens)
                del output, hidden, logits, loss
        return {"method": "likelihood", "selected": min(losses, key=losses.get), "nll": losses,
                "candidate_tokens": lengths, "include_eos": include_eos, "latent_segments": [],
                "prompt_tokens": prefix_len, "truncated": False,
                "image_grid_thw": inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else []}
