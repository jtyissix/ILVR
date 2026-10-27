"""One differentiable stream: text -> action latents -> interaction -> visual latents.

The current helper deliberately supplies privileged visual information through
InteractionCore. Its pooled features are also regression targets; they never
replace the student's autoregressive visual inputs. The action decoder receives
only its own four latent inputs and the teacher-forced action-word prefix.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from ..interaction.execution import DecoderExecutor, causal_ce_sum
from ..interaction.model import input_embeddings, load_full_model
from ..sft1.model import image_memories, local_text
from ..sft1.modules import IMAGE_INDEX_ENCODING, POSITION_ENCODING, InteractionCore


LOSS_NAMES = ("ce", "visual", "action_words", "action_state")


def pool_helper(features, grid, merge_size, output_size=4):
    """Pool one *post-merger, raster-ordered* still image, without padded tokens.

FP32 averaging preserves small feature differences in BF16. Rectangular and
tiny grids are legal: adaptive pooling may overlap bins but never invents new
image information, discards a tail, or combines different images.
    """
    if (len(grid) != 3 or grid[0] != 1 or min(grid[1:]) <= 0
            or grid[1] % merge_size or grid[2] % merge_size):
        raise ValueError("Pooling requires one still-image grid divisible by the merger size")
    height, width = grid[1] // merge_size, grid[2] // merge_size
    if features.ndim != 2 or len(features) != height * width:
        raise ValueError("Helper feature count must match its exact post-merger grid")
    if output_size <= 0:
        raise ValueError("Pooling output size must be positive")
    image = features.detach().float().reshape(height, width, -1).permute(2, 0, 1)[None]
    pooled = F.adaptive_avg_pool2d(image, (output_size, output_size))
    return pooled[0].permute(1, 2, 0).reshape(output_size ** 2, -1).to(features.dtype)


@dataclass
class LatentTrajectory:
    hidden: torch.Tensor
    actions: dict
    visuals: dict
    interactions: dict
    text_states: dict
    visual_targets: dict


def _continuous_positions(positions, count, row, batch, name):
    if (len(positions) != count or not positions
            or positions != list(range(positions[0], positions[0] + count))):
        raise ValueError(f"{name} must contain exactly {count} consecutive positions")
    if positions[0] <= 0 or positions[-1] >= batch["input_ids"].shape[1]:
        raise ValueError(f"{name} positions must have a valid preceding decoder state")
    if not bool(batch["attention_mask"][row, positions].all()):
        raise ValueError(f"{name} positions must be attended even when their placeholder ID is padding")
    if not bool(batch["labels"][row, positions].eq(-100).all()):
        raise ValueError(f"{name} positions cannot carry vocabulary CE labels")


class LatentModel(nn.Module):
    def __init__(self, decoder, lm_head, model_config, config, interaction=None,
                 aux_decoder=None, aux_lm_head=None):
        super().__init__()
        self.decoder, self.lm_head = decoder, lm_head
        self.model_config, self.recipe = model_config, config
        self.interaction = interaction if interaction is not None else InteractionCore(
            model_config.hidden_size, dim=config.fusion_dim, heads=config.fusion_heads,
            ffn_dim=config.fusion_ffn_dim, score_dim=config.fusion_score_dim,
            image_branch_enabled=config.image_branch_enabled, slots=config.visual_latents,
            interaction_tokens=config.interaction_tokens)
        # SIM-CoT shares the main input-embedding *path*, not the decoder stack.
        # Omit a second, unused vocabulary embedding from memory and checkpoints.
        self.aux_decoder = aux_decoder if aux_decoder is not None else copy.deepcopy(
            decoder, {id(decoder.embed_tokens): nn.Identity()})
        self.aux_decoder.embed_tokens = nn.Identity()
        self.aux_lm_head = aux_lm_head if aux_lm_head is not None else copy.deepcopy(lm_head)
        self.executor = DecoderExecutor(decoder, config.attention_backend, config.activation_checkpointing)
        self.aux_executor = DecoderExecutor(self.aux_decoder, config.attention_backend, config.activation_checkpointing)
        if config.pool_grid ** 2 != config.visual_latents:
            raise ValueError("The fixed spatial pooling grid must match visual_latents")
        if self.interaction.slots != config.visual_latents or self.interaction.interaction_tokens != config.interaction_tokens:
            raise ValueError("Interaction checkpoint slot/token counts differ from the SFT-II recipe")

    def trajectory(self, batch, image_features=None):
        ids = batch["input_ids"]
        if ids.ndim != 2 or batch["labels"].shape != ids.shape or batch["attention_mask"].shape != ids.shape:
            raise ValueError("Input IDs, labels and attention mask must have the same [B,T] shape")
        if any(len(batch[key]) != len(ids) for key in ("actions", "segments", "images")):
            raise ValueError("Actions, helpers and image ownership must match the batch rows")
        # The frozen encoder is outside the trainable engine. Also enforce its
        # gradient boundary for direct callers and tests that provide features.
        features = None if image_features is None else image_features.detach()
        memories = image_memories(features, batch["images"], self.model_config)
        questions = [[memory for item, memory in zip(items, row) if item["kind"] == "question"]
                     for items, row in zip(batch["images"], memories)]
        question_features = [memory.features for row in questions for memory in row]
        question_features = torch.cat(question_features) if question_features else None
        if int(ids.eq(self.model_config.image_token_id).sum()) != (0 if question_features is None else len(question_features)):
            raise ValueError("Only question-image tokens belong in the SFT-II student sequence")
        embeds = input_embeddings(self.decoder, ids, question_features, self.model_config.image_token_id)
        recurrences, interactions_at, occupied, targets = {}, {}, set(), {}

        def reserve(row, positions, name):
            for position in positions:
                if (row, position) in occupied:
                    raise ValueError(f"Overlapping {name}/action/visual/interaction positions")
                occupied.add((row, position))

        for row, actions in enumerate(batch["actions"]):
            previous = -1
            for index, action in enumerate(actions):
                positions = action["positions"]
                _continuous_positions(positions, self.recipe.action_latents, row, batch, "Action")
                if positions[0] <= previous:
                    raise ValueError("Actions must be in causal order")
                previous = positions[-1]
                reserve(row, positions, "action")
                for step, position in enumerate(positions):
                    recurrences.setdefault(position, []).append(("action", row, index, step))
        for row, segments in enumerate(batch["segments"]):
            previous = -1
            for index, segment in enumerate(segments):
                position, image_index = segment["position"], segment["image_index"]
                b_positions = list(range(position, position + self.recipe.interaction_tokens))
                _continuous_positions(b_positions, self.recipe.interaction_tokens, row, batch, "Interaction")
                z_positions = segment["visual_positions"]
                _continuous_positions(z_positions, self.recipe.visual_latents, row, batch, "Visual")
                if position <= previous or z_positions[0] != b_positions[-1] + 1:
                    raise ValueError("Each visual block must immediately follow its interaction in causal segment order")
                previous = z_positions[-1]
                reserve(row, b_positions, "interaction")
                reserve(row, z_positions, "visual")
                if not 0 <= image_index < len(memories[row]) or batch["images"][row][image_index]["kind"] != "helper":
                    raise ValueError("Interaction must read its own helper image")
                helper = memories[row][image_index]
                targets[row, index] = pool_helper(helper.features, batch["images"][row][image_index]["grid"],
                                                 self.model_config.vision_config.spatial_merge_size, self.recipe.pool_grid)
                interactions_at.setdefault(position, []).append((row, index, segment))
                for step, position_z in enumerate(z_positions):
                    recurrences.setdefault(position_z, []).append(("visual", row, index, step))

        self.validate_boundaries(batch, occupied)
        boundaries = sorted(set(recurrences) | set(interactions_at))
        chunks, consumed, pending = [], [], {}
        cache = None
        action_values, visual_values, b_values, text_states = {}, {}, {}, {}

        def process(value, start, end):
            nonlocal cache
            # Never save a view of a new full [B,T,H] allocation at every latent
            # step. Checkpoint would retain that entire backing storage. Only
            # materialize this chunk, including any previously constructed b.
            value = value.clone()
            keys = [key for key in pending if start <= key[1] < end]
            if keys:
                indices = torch.tensor([row * (end - start) + pos - start for row, pos in keys],
                                       device=value.device)
                replacements = torch.stack([pending.pop(key) for key in keys]).to(value.dtype)
                value = value.reshape(-1, value.shape[-1]).index_copy(0, indices, replacements).reshape_as(value)
            if self.recipe.execution == "recompute":
                prefix = torch.cat(consumed + [value], dim=1)
                hidden, _ = self.executor.chunk(prefix, batch["attention_mask"][:, :end], batch["position_ids"][:, :, :end])
                consumed.append(value)
                hidden = hidden[:, start:end]
            elif self.recipe.execution == "incremental":
                hidden, cache = self.executor.chunk(value, batch["attention_mask"][:, :end],
                                                    batch["position_ids"][:, :, start:end], cache)
            else:
                raise ValueError(f"Unknown execution mode: {self.recipe.execution}")
            chunks.append((start, hidden))

        first = boundaries[0] if boundaries else ids.shape[1]
        process(embeds[:, :first], 0, first)
        for event_index, position in enumerate(boundaries):
            entries = interactions_at.get(position, [])
            if entries:
                texts = [local_text(chunks, row, segment["text"], position) for row, _, segment in entries]
                text = pad_sequence([value if len(value) else embeds.new_zeros(1, embeds.shape[-1]) for value in texts],
                                    batch_first=True)
                text_mask = torch.arange(text.shape[1], device=embeds.device)[None] < torch.tensor(
                    [len(value) for value in texts], device=embeds.device)[:, None]
                prior = torch.stack([targets[row, index] for row, index, _ in entries])
                b = self.interaction(text, prior, text_mask, [questions[row] for row, _, _ in entries])
                for i, (row, index, _) in enumerate(entries):
                    b_values[row, index], text_states[row, index] = b[i], texts[i]
                    for token in range(self.recipe.interaction_tokens):
                        pending[row, position + token] = b[i, token]
            steps = recurrences.get(position, [])
            if steps:
                # The previous *output* is the next latent's input. No detach,
                # sampled vocabulary token, pooled-target injection, or z_17.
                previous_hidden = chunks[-1][1][:, -1]
                for kind, row, index, step in steps:
                    value = previous_hidden[row]
                    (action_values if kind == "action" else visual_values)[row, index, step] = value
                    pending[row, position] = value
            end = boundaries[event_index + 1] if event_index + 1 < len(boundaries) else ids.shape[1]
            process(embeds[:, position:end], position, end)
        return LatentTrajectory(torch.cat([hidden for _, hidden in chunks], dim=1), action_values,
                                visual_values, b_values, text_states, targets)

    def validate_boundaries(self, batch, occupied):
        """Every continuous block has supervised, attended start/end tokens."""
        boundary_ids = batch.get("latent_boundary_ids")
        if boundary_ids is None or len(boundary_ids) != 2 or boundary_ids[0] == boundary_ids[1]:
            raise ValueError("SFT-II requires latent start/end metadata; rebuild unmarked records")
        start_id, end_id = boundary_ids
        for name, value in zip(("latent_start_id", "latent_end_id"), boundary_ids):
            if getattr(self.model_config, name, value) != value:
                raise ValueError("Student/checkpoint latent boundary vocabulary mismatch")
        ids = batch["input_ids"]
        covered = set()
        active = batch.get("sample_mask")
        for row in range(len(ids)):
            starts = ids[row].eq(start_id).nonzero().flatten().tolist()
            ends = ids[row].eq(end_id).nonzero().flatten().tolist()
            if len(starts) != len(ends):
                raise ValueError("Unbalanced latent start/end tokens")
            previous = -1
            for start, end in zip(starts, ends):
                if start <= previous or end <= start:
                    raise ValueError("Nested or reversed latent boundaries")
                previous = end
                if not bool(batch["attention_mask"][row, start:end + 1].all()):
                    raise ValueError("Latent boundaries and their contents must be attended")
                if active is None or bool(active[row]):
                    if batch["labels"][row, start].item() != start_id or batch["labels"][row, end].item() != end_id:
                        raise ValueError("Latent start/end tokens must carry CE labels")
                block = {(row, p) for p in range(start + 1, end)}
                action = [a for a in batch["actions"][row] if a["positions"][0] == start + 1]
                helper_start = start + 1 + (self.recipe.action_latents if action else 0)
                helper = [s for s in batch["segments"][row] if s["position"] == helper_start]
                expected = (self.recipe.action_latents if action else 0) + (
                    self.recipe.interaction_tokens + self.recipe.visual_latents if helper else 0)
                if not expected or end - start - 1 != expected or not block.issubset(occupied):
                    raise ValueError("Latent block must contain action[4], interaction[3]+visual[16], or their ordered pair")
                covered.update(block)
        if covered != occupied:
            raise ValueError("Every action/interaction/visual slot must be inside latent start/end boundaries")

    def action_word_loss(self, trajectory, batch, active):
        """Each action is an independent sequence: [a1,a2,a3,a4,words,EOS]."""
        total = trajectory.hidden.sum() * 0.0
        # One action at a time bounds auxiliary activation/logit padding overhead
        # for ragged actions. All calls still contribute to one logical backward.
        for row, actions in enumerate(batch["actions"]):
            if not bool(active[row]):
                continue
            for index, action in enumerate(actions):
                targets = torch.tensor(action["target_ids"], device=trajectory.hidden.device, dtype=torch.long)
                if targets.ndim != 1 or not len(targets):
                    raise ValueError("Every action needs nonempty action.content tokens plus EOS")
                latents = torch.stack([trajectory.actions[row, index, k] for k in range(self.recipe.action_latents)])
                embeds = torch.cat((latents, self.decoder.embed_tokens(targets)), 0)[None]
                length = embeds.shape[1]
                mask = torch.ones(1, length, device=embeds.device, dtype=torch.bool)
                positions = torch.arange(length, device=embeds.device)[None, None].expand(3, 1, -1)
                hidden, _ = self.aux_executor.chunk(embeds, mask, positions)
                labels = torch.cat((targets.new_full((self.recipe.action_latents,), -100), targets))[None]
                total = total + causal_ce_sum(hidden, labels, self.aux_lm_head, self.recipe.ce_chunk_size,
                                             self.recipe.activation_checkpointing)
        return total

    def forward(self, batch, image_features=None):
        trajectory = self.trajectory(batch, image_features)
        active = batch.get("sample_mask")
        if active is None:
            active = torch.ones(len(batch["input_ids"]), device=trajectory.hidden.device, dtype=torch.bool)
        if active.shape != (len(batch["input_ids"]),) or active.dtype != torch.bool:
            raise ValueError("sample_mask must be boolean [batch]")
        labels = batch["labels"].masked_fill(~active[:, None], -100)
        ce = causal_ce_sum(trajectory.hidden, labels, self.lm_head, self.recipe.ce_chunk_size,
                           self.recipe.activation_checkpointing)
        zero = trajectory.hidden.sum() * 0.0
        visual, state = zero, zero
        teacher = batch.get("teacher_targets")
        has_actions = any(bool(active[r]) and actions for r, actions in enumerate(batch["actions"]))
        if self.recipe.action_state_weight and has_actions:
            expected = (len(active), max(map(len, batch["actions"])), self.model_config.hidden_size)
            if teacher is None or tuple(teacher.shape) != expected:
                raise ValueError(f"Frozen action teacher targets must have shape {expected}; build the teacher cache first")
            target_mask = batch.get("teacher_action_mask")
            if target_mask is not None:
                if tuple(target_mask.shape) != expected[:2] or target_mask.dtype != torch.bool:
                    raise ValueError("teacher_action_mask must match the action target shape")
                if any(bool(active[r]) and not bool(target_mask[r, :len(actions)].all())
                       for r, actions in enumerate(batch["actions"])):
                    raise ValueError("A supervised action is missing its frozen teacher target")
        for (row, index), target in trajectory.visual_targets.items():
            if bool(active[row]):
                predicted = torch.stack([trajectory.visuals[row, index, k] for k in range(self.recipe.visual_latents)])
                visual = visual + (1.0 - F.cosine_similarity(predicted.float(), target.detach().float(), dim=-1)).sum()
        if self.recipe.action_state_weight:
            for row, actions in enumerate(batch["actions"]):
                if not bool(active[row]):
                    continue
                for index, action in enumerate(actions):
                    # Output *after* consuming a4, not the a4 input itself.
                    predicted = trajectory.hidden[row, action["positions"][-1]].float()
                    target = teacher[row, index].detach().to(predicted.device).float()
                    if not bool(torch.isfinite(target).all()):
                        raise ValueError("Teacher action states must be finite")
                    scale = target.square().mean().clamp_min(self.recipe.state_loss_epsilon)
                    state = state + (predicted - target).square().mean() / scale
        words = self.action_word_loss(trajectory, batch, active) if self.recipe.action_words_weight else zero
        # The main decoder/head are always traversed, even with fully ignored CE.
        # Only optional branches need explicit dependencies. Avoid allocating
        # redundant dense zero gradients for every matrix of two 7B decoders.
        optional = [self.interaction]
        if not has_actions or not self.recipe.action_words_weight:
            optional.extend((self.aux_decoder, self.aux_lm_head))
        for module in optional:
            for parameter in module.parameters():
                if parameter.requires_grad:
                    ce = ce + parameter.reshape(-1)[0] * 0.0
        return {"ce": ce, "visual": visual, "action_words": words, "action_state": state}


def load_training_components(config, device):
    """Transfer the stage-I backbone and interaction, omitting its image adapter."""
    from safetensors.torch import load_file
    bundle = Path(config.warmup_path)
    metadata = json.loads((bundle / "interaction_config.json").read_text(encoding="utf-8"))
    if (metadata.get("format") != "sft1-interaction-v1"
            or metadata.get("position_encoding") != POSITION_ENCODING
            or metadata.get("image_index_encoding") != IMAGE_INDEX_ENCODING):
        raise ValueError("Expected a stage-I exported interaction component")
    full = load_full_model(str(bundle / "backbone"), config.attention_backend)
    from .tokens import initialize_student_vocabulary, load_processor
    initialize_student_vocabulary(full, load_processor(config))
    expected = dict(hidden_size=full.config.hidden_size, dim=config.fusion_dim, heads=config.fusion_heads,
                    ffn_dim=config.fusion_ffn_dim, score_dim=config.fusion_score_dim,
                    image_branch_enabled=config.image_branch_enabled, slots=config.visual_latents,
                    interaction_tokens=config.interaction_tokens)
    if metadata["kwargs"] != expected:
        raise ValueError(f"Interaction architecture differs from the warm-up export: expected {expected}, got {metadata['kwargs']}")
    interaction = InteractionCore(**metadata["kwargs"])
    interaction.load_state_dict(load_file(str(bundle / "interaction.safetensors")), strict=True)
    visual = full.visual.requires_grad_(False).eval().to(device)
    student = LatentModel(full.model, full.lm_head, full.config, config, interaction=interaction)
    del full
    student.to(device=device, dtype=torch.bfloat16)
    return student, visual


Stage2Model = LatentModel
