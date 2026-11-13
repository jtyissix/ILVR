"""Reuse SFT-II tokenized sequences; discard all helper images and teacher data."""
import json
from pathlib import Path

from ..interaction.data import file_hash
from ..sft1.data import WarmupCollator, _processor_identity, load_processor as _load_processor, validate_tokenizer_capacity
from ..sft2.data import CACHE_VERSION, TOKENIZATION_POLICY
from ..sft2.tokens import BOUNDARY_POLICY, BOUNDARY_TOKENS
from .loss import LOSS_NAMES


def load_processor(config):
    # Boundaries must already be trained in SFT-II; never add/reinitialize rows.
    processor = _load_processor(config)
    metadata = json.loads((Path(config.model_path) / "config.json").read_text(encoding="utf-8"))
    validate_tokenizer_capacity(processor, metadata)
    ids = [metadata.get("latent_start_id"), metadata.get("latent_end_id")]
    vocab = processor.tokenizer.get_vocab()
    if (None in ids or len(set(ids)) != 2 or any(
            token not in processor.tokenizer.all_special_tokens or vocab.get(token) != idx
            or processor.tokenizer.encode(token, add_special_tokens=False) != [idx]
            for token, idx in zip(BOUNDARY_TOKENS, ids))):
        raise ValueError("SFT-II checkpoint/tokenizer is missing its trained latent boundaries")
    return processor


def relaxation_record(record):
    """Project before collating so even missing helper files are never opened."""
    keys = ("input_ids", "labels", "length", "assistant_start", "record_id", "source", "source_line",
            "dummy", "latent_boundary_ids", "boundary_policy", "tokenization_policy")
    result = {key: record[key] for key in keys if key in record}
    result["actions"] = [{"positions": list(a["positions"])} for a in record["actions"]]
    result["segments"] = [{"position": s["position"], "visual_positions": list(s["visual_positions"])}
                          for s in record["segments"]]
    result["images"] = [dict(i) for i in record["images"] if i["kind"] == "question"]
    result["image_grid_thw"] = [i["grid"] for i in result["images"]]
    return result


class Stage3Dataset:
    def __init__(self, directory, verify_checksum=True):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (self.manifest.get("version") != CACHE_VERSION or self.manifest.get("stage") != "sft2_latent"
                or self.manifest.get("boundary_policy") != BOUNDARY_POLICY
                or self.manifest.get("tokenization_policy") != TOKENIZATION_POLICY):
            raise ValueError("Use the current SFT-II prepared student cache for SFT-III")
        if verify_checksum and file_hash(directory / "records.jsonl") != self.manifest["records_sha256"]:
            raise ValueError("Prepared records checksum failed")
        # Trim one record at a time: no full teacher trajectories in workers/RAM.
        with (directory / "records.jsonl").open(encoding="utf-8") as handle:
            self.records = [relaxation_record(json.loads(line)) for line in handle if line.strip()]
        if not self.records or len(self.records) != self.manifest["samples"]:
            raise ValueError("Prepared sample count differs from manifest")

    def validate(self, config, processor):
        if _processor_identity(processor) != self.manifest["processor_identity"]:
            raise ValueError("Processor/image settings differ from the SFT-II cache")
        ids = [processor.tokenizer.get_vocab()[token] for token in BOUNDARY_TOKENS]
        if self.manifest.get("latent_boundary_ids") != ids:
            raise ValueError("Prepared boundary IDs differ from the SFT-II checkpoint")
        for key in ("action_latents", "interaction_tokens", "visual_latents"):
            if self.manifest.get(key) != getattr(config, key):
                raise ValueError(f"Prepared {key} differs from the SFT-III recipe")
        for record in self.records:
            if (record.get("latent_boundary_ids") != ids or record.get("boundary_policy") != BOUNDARY_POLICY
                    or len(record["input_ids"]) != len(record["labels"])
                    or len(record["input_ids"]) > config.max_seq_length):
                raise ValueError("Invalid/incompatible student record; truncation is forbidden")
        # Helpers, original data sources, auxiliary weights and teacher cache
        # are intentionally absent from SFT-III's required files.
        paths = {image["path"] for record in self.records for image in record["images"]}
        for path in paths:
            stat = Path(path).stat()
            if [stat.st_size, stat.st_mtime_ns] != self.manifest["image_stats"].get(path):
                raise ValueError(f"Question image changed since preparation: {path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if index < -1:
            raise IndexError("Only -1 is the distributed dummy sample sentinel")
        return {**self.records[max(index, 0)], "dummy": index == -1}


class Stage3Collator(WarmupCollator):
    def __init__(self, processor, model_config, manifest, config):
        super().__init__(processor, model_config, manifest)
        self.recipe = config
        self.boundary_ids = [model_config["latent_start_id"], model_config["latent_end_id"]]

    def __call__(self, records):
        import torch
        records = [relaxation_record(r) for r in records]
        if any(r.get("latent_boundary_ids") != self.boundary_ids or r.get("boundary_policy") != BOUNDARY_POLICY
               for r in records):
            raise ValueError("Incompatible latent boundaries in SFT-III records")
        result = super().__call__(records)
        counts = dict.fromkeys(LOSS_NAMES, 0)
        counts["ce"] = result["ce_count"]
        for record in records:
            if record.get("dummy"):
                continue
            for kind in self.recipe.selected_types:
                k = len(record["actions"] if kind == "text" else record["segments"])
                counts["collapse_" + kind] += k * (k - 1) // 2
        result.update(actions=[r["actions"] for r in records], latent_boundary_ids=self.boundary_ids,
                      loss_counts=counts,
                      sample_mask=torch.tensor([not r.get("dummy", False) for r in records], dtype=torch.bool))
        return result
