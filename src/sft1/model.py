"""Original-image warm-up on one differentiable, causally ordered decoder stream.

Only helper boundaries interrupt teacher forcing. Earlier interaction embeddings
remain in the actual prefix used to construct every later interaction embedding.
There is no reference decoder, reference pass, or generated visual latent here.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from src.interaction.execution import DecoderExecutor, causal_ce_sum
from src.interaction.model import input_embeddings, load_full_model
from .modules import ImageMemory, InteractionCore, WarmupImageAdapter


@dataclass
class WarmupTrajectory:
    hidden: torch.Tensor
    interactions: dict
    text_states: dict


def image_memories(features, image_rows, model_config):
    """Recover per-occurrence, per-sample image memories from merger output.

Qwen restores spatial raster order after its windowed visual encoder. Coordinates
are (x, y) on that returned merger grid, never on the pre-merger patch grid.
    """
    merge = model_config.vision_config.spatial_merge_size
    rows, expected_start = [], 0
    for descriptors in image_rows:
        memories = []
        for item in descriptors:
            start, end = item["feature_start"], item["feature_end"]
            t, h, w = item["grid"]
            if t != 1 or h <= 0 or w <= 0 or h % merge or w % merge:
                raise ValueError("SFT1 expects still-image grids divisible by the merger size")
            h, w = h // merge, w // merge
            if start != expected_start or end - start != h * w:
                raise ValueError("Image spans must follow the exact ordered merger feature layout")
            if features is None or end > len(features):
                raise ValueError("Missing or incomplete encoded image features")
            y, x = torch.meshgrid(torch.arange(h, device=features.device),
                                  torch.arange(w, device=features.device), indexing="ij")
            # Keep exact grid integers: BF16 would merge adjacent coordinates
            # beyond 256 before the FP32 sinusoidal encoding even sees them.
            coordinates = torch.stack((x.flatten(), y.flatten()), -1)
            memories.append(ImageMemory(features[start:end], coordinates, item["image_id"]))
            expected_start = end
        rows.append(memories)
    if expected_start != (0 if features is None else len(features)):
        raise ValueError("Unused encoded image features or an invalid sample/image boundary")
    return rows


def local_text(chunks, row, positions, before):
    """Gather only this segment's prior text without concatenating all history."""
    if positions != sorted(set(positions)) or any(p < 0 or p >= before for p in positions):
        raise ValueError("Local text positions must be ordered, unique, and strictly before b")
    if not chunks:
        raise ValueError("An interaction must follow a valid decoder prefix")
    pieces = []
    for start, hidden in chunks:
        selected = [p - start for p in positions if start <= p < start + hidden.shape[1]]
        if selected:
            index = torch.tensor(selected, device=hidden.device, dtype=torch.long)
            pieces.append(hidden[row].index_select(0, index))
    value = torch.cat(pieces, 0) if pieces else chunks[0][1].new_zeros((0, chunks[0][1].shape[-1]))
    if len(value) != len(positions):
        raise ValueError("Local text contains a position outside the computed causal prefix")
    return value


class WarmupModel(nn.Module):
    def __init__(self, decoder, lm_head, model_config, config):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.model_config = model_config
        self.recipe = config
        self.warmup_adapter = WarmupImageAdapter(
            model_config.hidden_size, dim=config.fusion_dim, heads=config.fusion_heads,
            ffn_dim=config.fusion_ffn_dim, layers=config.adapter_layers, slots=config.adapter_slots)
        self.interaction = InteractionCore(
            model_config.hidden_size, dim=config.fusion_dim, heads=config.fusion_heads,
            ffn_dim=config.fusion_ffn_dim, score_dim=config.fusion_score_dim,
            image_branch_enabled=config.image_branch_enabled, slots=config.adapter_slots,
            interaction_tokens=config.interaction_tokens)
        self.executor = DecoderExecutor(decoder, config.attention_backend, config.activation_checkpointing)

    def trajectory(self, batch, image_features=None):
        ids = batch["input_ids"]
        if len(batch["segments"]) != len(ids) or len(batch["images"]) != len(ids):
            raise ValueError("Segment and image ownership must match the batch rows")
        image_count = int(ids.eq(self.model_config.image_token_id).sum())
        if image_count != (0 if image_features is None else len(image_features)):
            raise ValueError("Complete question AND helper image features are required")
        embeds = input_embeddings(self.decoder, ids, image_features, self.model_config.image_token_id)
        memories = image_memories(image_features, batch["images"], self.model_config)
        question_memories = [[memory for item, memory in zip(descriptors, row) if item["kind"] == "question"]
                             for descriptors, row in zip(batch["images"], memories)]
        events = {}
        token_count = self.recipe.interaction_tokens
        for row, segments in enumerate(batch["segments"]):
            previous = -1
            for index, segment in enumerate(segments):
                pos = segment["position"]
                image_index = segment["image_index"]
                if (not previous < pos or pos <= 0 or pos + token_count > ids.shape[1]
                        or not bool(batch["attention_mask"][row, pos:pos + token_count].all())):
                    raise ValueError("Helper interaction positions must be valid and strictly ordered")
                if not bool(batch["labels"][row, pos:pos + token_count].eq(-100).all()):
                    raise ValueError("Interaction embeddings cannot carry a CE label")
                if not 0 <= image_index < len(memories[row]):
                    raise ValueError("Helper image index is outside the owning sample")
                if batch["images"][row][image_index]["kind"] != "helper":
                    raise ValueError("Warmup adapter must read the current helper, never a question image")
                if batch["images"][row][image_index]["boundary_start"] != pos + token_count:
                    raise ValueError("Interaction token count differs from the prepared helper boundary")
                events.setdefault(pos, []).append((row, index, segment))
                previous = pos + token_count - 1

        chunks, consumed = [], []
        interactions, text_states = {}, {}
        cache = None

        def process(value, start, end):
            nonlocal cache
            if self.recipe.execution == "recompute":
                # Correctness oracle: recompute THIS SAME history including earlier b.
                prefix = torch.cat(consumed + [value], dim=1)
                hidden, _ = self.executor.chunk(prefix, batch["attention_mask"][:, :end],
                                                batch["position_ids"][:, :, :end])
                consumed.append(value)
                hidden = hidden[:, start:end]
            elif self.recipe.execution == "incremental":
                hidden, cache = self.executor.chunk(value, batch["attention_mask"][:, :end],
                                                    batch["position_ids"][:, :, start:end], cache)
            else:
                raise ValueError(f"Unknown SFT1 execution mode: {self.recipe.execution}")
            chunks.append((start, hidden))

        boundaries = sorted(events)
        first = boundaries[0] if boundaries else ids.shape[1]
        process(embeds[:, :first], 0, first)
        for event_index, pos in enumerate(boundaries):
            entries = events[pos]
            texts = [local_text(chunks, row, segment["text"], pos) for row, _, segment in entries]
            helpers = [memories[row][segment["image_index"]] for row, _, segment in entries]
            # A true all-masked local text is legal for consecutive helper images.
            text = pad_sequence([t if len(t) else embeds.new_zeros(1, embeds.shape[-1]) for t in texts],
                                batch_first=True)
            text_mask = torch.arange(text.shape[1], device=embeds.device)[None] < torch.tensor(
                [len(t) for t in texts], device=embeds.device)[:, None]
            features = pad_sequence([m.features for m in helpers], batch_first=True)
            coordinates = pad_sequence([m.coordinates for m in helpers], batch_first=True)
            helper_mask = torch.arange(features.shape[1], device=embeds.device)[None] < torch.tensor(
                [len(m.features) for m in helpers], device=embeds.device)[:, None]
            slots = self.warmup_adapter(features, coordinates, helper_mask)
            b = self.interaction(text, slots, text_mask, [question_memories[row] for row, _, _ in entries])
            row_ids = torch.tensor([row for row, _, _ in entries], device=embeds.device, dtype=torch.long)
            # Other rows may start an event inside this block. Store ALL K tokens
            # now, then consume only up to the next event; future tokens survive.
            token_positions = pos + torch.arange(token_count, device=embeds.device)
            flat_positions = row_ids[:, None] * ids.shape[1] + token_positions[None]
            embeds = embeds.reshape(-1, embeds.shape[-1]).index_copy(
                0, flat_positions.flatten(), b.to(embeds.dtype).reshape(-1, embeds.shape[-1])
            ).reshape_as(embeds)
            for event_row, (row, index, _) in enumerate(entries):
                interactions[row, index] = b[event_row]
                text_states[row, index] = texts[event_row]
            end = boundaries[event_index + 1] if event_index + 1 < len(boundaries) else ids.shape[1]
            # Include b, its image and following text in ONE chunk, not one call/token.
            process(embeds[:, pos:end], pos, end)
        return WarmupTrajectory(torch.cat([h for _, h in chunks], dim=1), interactions, text_states)

    def forward(self, batch, image_features=None):
        trajectory = self.trajectory(batch, image_features)
        # Global shift includes the first supervised text token AFTER every image.
        loss = causal_ce_sum(trajectory.hidden, batch["labels"], self.lm_head,
                             self.recipe.ce_chunk_size, self.recipe.activation_checkpointing)
        # Different data-parallel ranks may see no helpers or no question images. Give
        # unused trainable branches an explicit zero gradient without fake images.
        for module in (self.warmup_adapter, self.interaction):
            for parameter in module.parameters():
                if parameter.requires_grad:
                    loss = loss + parameter.reshape(-1)[0] * 0.0
        return loss


def load_training_components(config, device):
    """Load exactly ONE backbone. The frozen vision tower stays outside ZeRO."""
    full = load_full_model(config.model_path, config.attention_backend)
    model_config = full.config
    visual = full.visual.requires_grad_(False).eval().to(device)
    model = WarmupModel(full.model, full.lm_head, model_config, config)
    del full
    model.to(device=device, dtype=torch.bfloat16)
    return model, visual
