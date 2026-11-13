"""All three latent types are autoregressive; only the backbone is trainable."""
from dataclasses import dataclass

import torch
from torch import nn

from ..interaction.execution import DecoderExecutor, causal_ce_sum
from ..interaction.model import input_embeddings, load_full_model
from .loss import LOSS_NAMES, distance_weighted_diversity


@dataclass
class RelaxationTrajectory:
    hidden: torch.Tensor
    latents: dict


def latent_layout(batch, config, model_config):
    """Validate the same marked 4/3/16 layout as SFT-II, without helper inputs."""
    ids, labels, mask = (batch[k] for k in ("input_ids", "labels", "attention_mask"))
    if ids.ndim != 2 or ids.shape != labels.shape or ids.shape != mask.shape or not ids.shape[1]:
        raise ValueError("Input IDs, labels and attention mask must have the same nonempty [B,T] shape")
    if any(len(batch[key]) != len(ids) for key in ("actions", "segments", "images")):
        raise ValueError("Actions, segments and question images must match the batch rows")
    active = batch.get("sample_mask", torch.ones(len(ids), device=ids.device, dtype=torch.bool))
    if active.shape != (len(ids),) or active.dtype != torch.bool:
        raise ValueError("sample_mask must be boolean [batch]")
    boundaries = batch.get("latent_boundary_ids")
    if boundaries != [model_config.latent_start_id, model_config.latent_end_id] or boundaries[0] == boundaries[1]:
        raise ValueError("Latent boundary vocabulary mismatch")
    if bool(labels[~mask.bool()].ne(-100).any()):
        raise ValueError("Padding cannot carry CE labels")
    events, blocks = {}, {kind: {} for kind in ("text", "interaction", "visual")}
    for row in range(len(ids)):
        occupied = set()

        def reserve(kind, index, positions, count):
            if (len(positions) != count or positions != list(range(positions[0], positions[0] + count))
                    or positions[0] <= 0 or positions[-1] >= ids.shape[1]):
                raise ValueError(f"{kind} must contain {count} consecutive positions with a predecessor")
            if occupied.intersection(positions):
                raise ValueError("Overlapping latent positions")
            if not bool(mask[row, positions].all()) or not bool(labels[row, positions].eq(-100).all()):
                raise ValueError("Latent positions must be attended and cannot carry CE labels")
            occupied.update(positions)
            blocks[kind][row, index] = positions
            for position in positions:
                events.setdefault(position, []).append(row)

        previous = -1
        for index, action in enumerate(batch["actions"][row]):
            reserve("text", index, action["positions"], config.action_latents)
            if action["positions"][0] <= previous:
                raise ValueError("Actions must be in causal order")
            previous = action["positions"][-1]
        previous = -1
        for index, segment in enumerate(batch["segments"][row]):
            start = segment["position"]
            reserve("interaction", index, list(range(start, start + config.interaction_tokens)), config.interaction_tokens)
            reserve("visual", index, segment["visual_positions"], config.visual_latents)
            if start <= previous or segment["visual_positions"][0] != start + config.interaction_tokens:
                raise ValueError("Visual latents must immediately follow their interaction in causal order")
            previous = segment["visual_positions"][-1]
        starts = ids[row].eq(boundaries[0]).nonzero().flatten().tolist()
        ends = ids[row].eq(boundaries[1]).nonzero().flatten().tolist()
        if len(starts) != len(ends):
            raise ValueError("Unbalanced latent boundaries")
        covered, previous = set(), -1
        for start, end in zip(starts, ends):
            if start <= previous or end <= start or not bool(mask[row, start:end + 1].all()):
                raise ValueError("Invalid, nested or unattended latent boundaries")
            previous = end
            if bool(active[row]) and (labels[row, start].item() != boundaries[0] or labels[row, end].item() != boundaries[1]):
                raise ValueError("Latent start/end delimiters must carry CE labels")
            has_action = any(a["positions"][0] == start + 1 for a in batch["actions"][row])
            helper_start = start + 1 + (config.action_latents if has_action else 0)
            has_segment = any(s["position"] == helper_start for s in batch["segments"][row])
            count = (config.action_latents if has_action else 0) + (
                config.interaction_tokens + config.visual_latents if has_segment else 0)
            positions = set(range(start + 1, end))
            if not count or end - start - 1 != count or not positions.issubset(occupied):
                raise ValueError("Latent block must contain text[4], interaction[3]+visual[16], or their ordered pair")
            covered.update(positions)
        if covered != occupied:
            raise ValueError("Every latent slot must be inside start/end boundaries")
    return events, blocks, active


class RelaxationModel(nn.Module):
    def __init__(self, decoder, lm_head, model_config, config):
        super().__init__()
        self.decoder, self.lm_head = decoder, lm_head
        self.model_config, self.recipe = model_config, config
        self.executor = DecoderExecutor(decoder, config.attention_backend, config.activation_checkpointing)

    def trajectory(self, batch, image_features=None):
        events, blocks, _ = latent_layout(batch, self.recipe, self.model_config)
        if any(i["kind"] != "question" for row in batch["images"] for i in row):
            raise ValueError("SFT-III accepts question images only; remove helper images before encoding")
        features = None if image_features is None else image_features.detach()
        if int(batch["input_ids"].eq(self.model_config.image_token_id).sum()) != (0 if features is None else len(features)):
            raise ValueError("Question-image tokens and encoded question features must match")
        embeds = input_embeddings(self.decoder, batch["input_ids"], features, self.model_config.image_token_id)
        boundaries = sorted(events)
        chunks, consumed = [], []
        cache = None

        def process(value, start, end):
            nonlocal cache
            # Do not retain a full-sequence backing allocation per latent step.
            value = value.clone()
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
            chunks.append(hidden)

        process(embeds[:, :boundaries[0] if boundaries else embeds.shape[1]],
                0, boundaries[0] if boundaries else embeds.shape[1])
        for index, position in enumerate(boundaries):
            end = boundaries[index + 1] if index + 1 < len(boundaries) else embeds.shape[1]
            value = embeds[:, position:end].clone()
            rows = torch.tensor(events[position], device=value.device)
            indices = rows * (end - position)
            # e_t = h_(t-1), including interaction latents. Keep the entire graph.
            replacements = chunks[-1][:, -1].index_select(0, rows)
            value = value.flatten(0, 1).index_copy(0, indices, replacements).reshape_as(value)
            process(value, position, end)
        hidden = torch.cat(chunks, dim=1)
        latents = {kind: {key: hidden[key[0], [p - 1 for p in positions]] for key, positions in values.items()}
                   for kind, values in blocks.items()}
        return RelaxationTrajectory(hidden, latents)

    def forward(self, batch, image_features=None):
        trajectory = self.trajectory(batch, image_features)
        active = batch.get("sample_mask", torch.ones(len(batch["input_ids"]), device=trajectory.hidden.device, dtype=torch.bool))
        labels = batch["labels"].masked_fill(~active[:, None], -100)
        zero = trajectory.hidden.sum() * 0.0
        losses = {name: zero for name in LOSS_NAMES}
        losses["ce"] = causal_ce_sum(trajectory.hidden, labels, self.lm_head,
                                    self.recipe.ce_chunk_size, self.recipe.activation_checkpointing)
        for kind in self.recipe.selected_types:
            for row in range(len(active)):
                if not bool(active[row]):
                    continue
                steps = [value for (owner, _), value in trajectory.latents[kind].items() if owner == row]
                if len(steps) >= 2:
                    losses["collapse_" + kind] = losses["collapse_" + kind] + distance_weighted_diversity(
                        torch.stack(steps), self.recipe.collapse_margin, self.recipe.collapse_alpha,
                        self.recipe.collapse_epsilon)
        return losses


def load_training_components(config, device):
    from .data import load_processor
    load_processor(config)  # Validate trained boundaries without changing embeddings.
    full = load_full_model(config.model_path, config.attention_backend)
    visual = full.visual.requires_grad_(False).eval().to(device)
    student = RelaxationModel(full.model, full.lm_head, full.config, config)
    student.requires_grad_(True).to(device=device, dtype=torch.bfloat16)
    return student, visual


Stage3Model = RelaxationModel
