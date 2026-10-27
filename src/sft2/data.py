"""Structured single-path latent data and full-text teacher trajectories.

The teacher is freshly rebuilt from the selected original structure. Text is
tokenized at explicit analysis/action boundaries, so an action's last output
state is unambiguous even when BPE would otherwise merge across the boundary.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from ..interaction.data import checkpoint_inventory, file_hash, json_write, position_ids_for_images, stable_hash
from ..sft1 import data as stage1
from ..sft1.data import (IMAGE_POLICY, OverlengthTrajectoryError, _field, _iter_rows, _processor_identity,
                        _source_inventory, encode_image_groups, load_training_image, processor_settings)
from .tokens import (BOUNDARY_POLICY, ensure_boundary_tokens, load_processor,
                     student_model_config, validate_tokenizer_capacity)

CACHE_VERSION = 2
TOKENIZATION_POLICY = "explicit_structured_parts_original_teacher_student_boundaries_v2"


def normalize_sample(sample, source):
    """Use stage one's strict validation, while retaining every action object."""
    normalized = stage1.normalize_sample(sample, source)
    sequence = _field(sample, source["plan_field"])
    if isinstance(sequence, str):
        sequence = json.loads(sequence)
    sequence = copy.deepcopy(sequence)
    for index, step in enumerate(sequence):
        target = normalized["sequence"][index]
        if target["type"] == "image":
            continue
        content = step["content"]
        parts = content if isinstance(content, list) else [{"sub_type": "analysis", "content": content}]
        for part in parts:
            if part["sub_type"] == "action":
                if not isinstance(part.get("content"), str) or not part["content"].strip():
                    raise ValueError("action.content must contain nonempty action words for the auxiliary decoder")
                if any(t in part["content"] for t in stage1._RESERVED):
                    raise ValueError("Action words contain reserved chat/image/latent delimiters")
        target["parts"] = parts
    # Explicitly allowed appended answer is a normal supervised analysis block.
    for step in normalized["sequence"][len(sequence):]:
        step["parts"] = [{"sub_type": "analysis", "content": step["content"]}]
    final_text = "\n".join(p["content"] for p in normalized["sequence"][-1]["parts"]
                           if p["sub_type"] == "analysis")
    if not stage1._answer_in_text(normalized["answer"], final_text):
        raise ValueError("Final answer must remain in plain/analysis text, not inside a compressed action")
    return normalized


def _prepare_sample(sample, source, processor, model_config, max_seq_length,
                    interaction_tokens=3, shrink_overlong_images=True, action_latents=4, visual_latents=16,
                    vision_budget=None):
    if (action_latents, visual_latents, interaction_tokens) != (4, 16, 3):
        raise ValueError("Expected 4 action, 16 visual and 3 interaction slots")
    normalized = normalize_sample(sample, source)
    model_config = student_model_config(processor, model_config)
    boundary_ids = ensure_boundary_tokens(processor)
    # The full teacher must fit as well; stage one's aspect-preserving resizing
    # applies the same chosen per-image cap to both teacher and student.
    base = stage1.prepare_sample(sample, source, processor, model_config, vision_budget or max_seq_length,
                                 interaction_tokens, shrink_overlong_images)
    tokenizer = processor.tokenizer
    special = set(tokenizer.all_special_ids)
    for step in normalized["sequence"]:
        for part in step.get("parts", []):
            if part["sub_type"] == "action" and any(t and t in part["content"] for t in tokenizer.all_special_tokens):
                raise ValueError("Action words contain a tokenizer special token")
    eos = tokenizer.eos_token_id
    if eos is None or not 0 <= eos < model_config["vocab_size"]:
        raise ValueError("The auxiliary decoder requires a valid existing eos_token_id")
    start = base["assistant_start"]
    ids, labels = list(base["input_ids"][:start]), [-100] * start
    tids, tlabels = list(ids), list(labels)
    images, timages = [], []
    question_count = len(normalized["question_images"])
    for item in base["images"][:question_count]:
        images.append(copy.deepcopy(item))
        timages.append(copy.deepcopy(item))
    actions, segments, tsegments = [], [], []
    memory, tmemory = [], []
    helper_index = question_count
    open_block = False

    def boundary(index):
        ids.append(boundary_ids[index])
        labels.append(boundary_ids[index])

    def text(value, target_ids, target_labels, target_memory):
        values = tokenizer.encode(value, add_special_tokens=False)
        if any(v < 0 or v >= model_config["vocab_size"] for v in values):
            raise ValueError("Encoded token ID exceeds model vocabulary capacity")
        at = len(target_ids)
        target_ids.extend(values)
        target_labels.extend(values)
        target_memory.extend(at + i for i, token in enumerate(values) if token not in special)
        return values

    for step_index, step in enumerate(normalized["sequence"]):
        if step["type"] == "text":
            for part_index, part in enumerate(step["parts"]):
                paired = (part["sub_type"] == "action" and part_index == len(step["parts"]) - 1
                          and step_index + 1 < len(normalized["sequence"])
                          and normalized["sequence"][step_index + 1]["type"] == "image")
                if part["sub_type"] == "analysis":
                    text(part["content"], ids, labels, memory)
                    text(part["content"], tids, tlabels, tmemory)
                else:
                    original = text(part["original_text"], tids, tlabels, tmemory)
                    if not original:
                        raise ValueError("action.original_text must encode to at least one token")
                    boundary(0)
                    positions = list(range(len(ids), len(ids) + action_latents))
                    ids.extend([tokenizer.pad_token_id] * action_latents)
                    labels.extend([-100] * action_latents)
                    memory.extend(positions)
                    target_ids = tokenizer.encode(part["content"], add_special_tokens=False) + [eos]
                    if any(v < 0 or v >= model_config["vocab_size"] for v in target_ids):
                        raise ValueError("Action word token ID exceeds model vocabulary capacity")
                    actions.append({"positions": positions, "teacher_position": len(tids) - 1,
                                    "target_ids": target_ids, "content": part["content"],
                                    "original_text": part["original_text"]})
                    open_block = paired
                    if not paired:
                        boundary(1)
                # A paired action/helper is ONE uninterrupted 23-vector block.
                # Keep the original newline only on the stage-I teacher path.
                if not paired:
                    text("\n", ids, labels, memory)
                text("\n", tids, tlabels, tmemory)
        else:
            item = copy.deepcopy(base["images"][helper_index])
            if not open_block:
                boundary(0)
            position = len(ids)
            visual_positions = list(range(position + interaction_tokens,
                                          position + interaction_tokens + visual_latents))
            segments.append({"position": position, "visual_positions": visual_positions,
                             "text": list(memory), "image_index": helper_index})
            ids.extend([tokenizer.pad_token_id] * (interaction_tokens + visual_latents))
            labels.extend([-100] * (interaction_tokens + visual_latents))
            boundary(1)
            open_block = False
            # Helper features remain in the vision batch, but there is no helper
            # image-token span in the student's sequence.
            for key in ("token_start", "token_end", "boundary_start", "boundary_end"):
                item.pop(key, None)
            images.append(item)
            tposition = len(tids)
            tsegments.append({"position": tposition, "text": list(tmemory), "image_index": helper_index})
            tids.extend([tokenizer.pad_token_id] * interaction_tokens)
            tlabels.extend([-100] * interaction_tokens)
            teacher_boundary = len(tids)
            original_item = base["images"][helper_index]
            block = base["input_ids"][original_item["boundary_start"]:original_item["boundary_end"]]
            tids.extend(block)
            tlabels.extend([-100] * len(block))
            timages.append({**item, "boundary_start": teacher_boundary, "boundary_end": len(tids),
                            "token_start": teacher_boundary + 1, "token_end": len(tids) - 1})
            helper_index += 1
            memory, tmemory = [], []
    user = {"role": "user", "content": [
        *({"type": "image", "image": path} for path in normalized["question_images"]),
        {"type": "text", "text": normalized["question"]}]}
    prompt = processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
    empty = processor.apply_chat_template([user, {"role": "assistant", "content": [
        {"type": "text", "text": ""}]}], tokenize=False, add_generation_prompt=False)
    if not empty.startswith(prompt):
        raise ValueError("Chat template does not have a consistent assistant prefix")
    ending = empty[len(prompt):]
    text(ending, ids, labels, memory)
    text(ending, tids, tlabels, tmemory)
    if len(ids) > max_seq_length or len(tids) > max_seq_length:
        raise OverlengthTrajectoryError(max(len(ids), len(tids)), max_seq_length)
    if not any(v != -100 for v in labels[1:]):
        raise ValueError("No supervised CE targets")
    teacher = {**base, "input_ids": tids, "labels": tlabels, "images": timages, "segments": tsegments,
               "length": len(tids), "non_visual_tokens": len(tids) - base["visual_tokens"],
               "original_length": len(tids) + base["original_visual_tokens"] - base["visual_tokens"],
               "tokenization_policy": TOKENIZATION_POLICY}
    question_visual_tokens = sum(i["visual_tokens"] for i in images if i["kind"] == "question")
    original_question_tokens = sum(i["original_visual_tokens"] for i in images if i["kind"] == "question")
    record = {**base, "input_ids": ids, "labels": labels, "images": images, "segments": segments,
              "actions": actions, "teacher_record": teacher, "length": len(ids),
              "original_length": len(ids) + original_question_tokens - question_visual_tokens,
              "non_visual_tokens": len(ids) - question_visual_tokens,
              "visual_tokens": question_visual_tokens, "original_visual_tokens": original_question_tokens,
              "visual_feature_tokens": base["visual_tokens"],
              "latent_boundary_ids": list(boundary_ids), "boundary_policy": BOUNDARY_POLICY,
              "tokenization_policy": TOKENIZATION_POLICY}
    record["record_id"] = stable_hash(record)
    return record


def prepare_sample(sample, source, processor, model_config, max_seq_length,
                   interaction_tokens=3, shrink_overlong_images=True, action_latents=4, visual_latents=16):
    """Resize until both explicit teacher and compressed student fit, never truncate."""
    budget = max_seq_length
    while True:
        try:
            return _prepare_sample(sample, source, processor, model_config, max_seq_length,
                                   interaction_tokens, shrink_overlong_images, action_latents, visual_latents,
                                   vision_budget=budget)
        except OverlengthTrajectoryError as exc:
            # A failure inside stage-one image budgeting means the image floor
            # has been reached. Only a rebuilt student's excess warrants retry.
            if not shrink_overlong_images or exc.limit != max_seq_length or exc.length <= max_seq_length:
                raise OverlengthTrajectoryError(exc.length, max_seq_length) from exc
            next_budget = budget - (exc.length - max_seq_length)
            if next_budget <= 0 or next_budget >= budget:
                raise
            budget = next_budget


def prepare_dataset(config):
    from .preprocessing import prepare_dataset as prepare
    return prepare(config)


class Stage2Dataset:
    def __init__(self, directory, verify_checksum=True):
        if hasattr(directory, "prepared_dir"):
            directory = directory.prepared_dir
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("version") != CACHE_VERSION or self.manifest.get("stage") != "sft2_latent":
            raise ValueError("Not a current SFT-II cache; rebuild from the structured sources with mode=prepare_only")
        if verify_checksum and file_hash(directory / "records.jsonl") != self.manifest["records_sha256"]:
            raise ValueError("Prepared records checksum failed")
        with (directory / "records.jsonl").open(encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if not self.records or len(self.records) != self.manifest["samples"]:
            raise ValueError("Prepared sample count differs from manifest")

    def validate_sources(self, config):
        # All reused checks are field- and policy-based, independent of stage.
        processor = load_processor(config)
        stage1.WarmupDataset.validate_sources(self, config, processor=processor)
        if (self.manifest.get("boundary_policy") != BOUNDARY_POLICY
                or self.manifest.get("latent_boundary_ids") != list(ensure_boundary_tokens(processor))):
            raise ValueError("Latent boundary vocabulary/policy changed; run prepare again")
        for key in ("action_latents", "visual_latents", "pool_grid"):
            if self.manifest.get(key) != getattr(config, key):
                raise ValueError(f"{key} changed; run prepare again")
        if self.manifest.get("tokenization_policy") != TOKENIZATION_POLICY:
            raise ValueError("Structural tokenization policy changed; run prepare again")
        if self.manifest.get("teacher_length_max", 0) > config.max_seq_length:
            raise ValueError("Prepared teacher trajectory exceeds max_seq_length")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if index < -1:
            raise IndexError("Only -1 is the distributed dummy sample sentinel")
        return {**self.records[max(index, 0)], "dummy": index == -1}


class Stage2Collator(stage1.WarmupCollator):
    def __init__(self, processor, model_config, manifest, teacher_cache_dir=None):
        super().__init__(processor, model_config, manifest)
        self.teacher_cache_dir = teacher_cache_dir
        self._teacher_reader = None

    def __call__(self, records):
        import torch
        # The inherited collator handles deduplicated vision execution and maps
        # all occurrences back to feature spans. mRoPE only sees question grids.
        views = [{**r, "image_grid_thw": [i["grid"] for i in r["images"] if i["kind"] == "question"]}
                 for r in records]
        result = super().__call__(views)
        boundary_ids = list(ensure_boundary_tokens(self.processor))
        if any(r.get("latent_boundary_ids") != boundary_ids or r.get("boundary_policy") != BOUNDARY_POLICY
               for r in records):
            raise ValueError("Unmarked or incompatible stage-II records; rebuild the prepared cache")
        result["latent_boundary_ids"] = boundary_ids
        action_count = max((len(r["actions"]) for r in records), default=0)
        hidden_size = self.config.get("hidden_size")
        targets = None
        target_mask = torch.zeros((len(records), action_count), dtype=torch.bool)
        if self.teacher_cache_dir is not None:
            from .teacher import TeacherTargetReader
            if self._teacher_reader is None:
                self._teacher_reader = TeacherTargetReader(self.teacher_cache_dir)
            loaded = [self._teacher_reader(r["record_id"]) for r in records]
            if hidden_size is None and loaded:
                hidden_size = loaded[0].shape[-1]
            targets = torch.zeros((len(records), action_count, hidden_size), dtype=loaded[0].dtype)
            for row, (record, target) in enumerate(zip(records, loaded)):
                count = len(record["actions"])
                if tuple(target.shape) != (count, hidden_size):
                    raise ValueError("Teacher target shape does not match action count and backbone hidden size")
                targets[row, :count] = target
                if not record.get("dummy"):
                    target_mask[row, :count] = True
        elif action_count:
            raise ValueError("Action state targets require teacher_cache_dir; generate the offline teacher cache first")
        counts = {"ce": result["ce_count"], "visual": 0, "action_words": 0, "action_state": 0}
        for record in records:
            if record.get("dummy"):
                continue
            counts["visual"] += sum(len(s["visual_positions"]) for s in record["segments"])
            counts["action_words"] += sum(len(a["target_ids"]) for a in record["actions"])
            counts["action_state"] += len(record["actions"])
        result.update(actions=[r["actions"] for r in records], teacher_targets=targets,
                      teacher_action_mask=target_mask, loss_counts=counts,
                      sample_mask=torch.tensor([not r.get("dummy", False) for r in records], dtype=torch.bool),
                      dummy=[bool(r.get("dummy")) for r in records], record_ids=[r["record_id"] for r in records])
        return result
