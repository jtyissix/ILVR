"""Trust the existing prepared data; keep shape/ownership/numerical checks."""
from pathlib import Path

from src.sft2.data import Stage2Collator, Stage2Dataset
from src.sft2.teacher import CACHE_FORMAT
from .checkpoint import read_json
from .state import require


class TeacherReader:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.manifest = read_json(self.directory / "manifest.json")
        require(self.manifest.get("format") == CACHE_FORMAT, "Unsupported teacher cache format")

    def __call__(self, record_id):
        import torch
        from safetensors import safe_open
        entry = self.manifest["records"][record_id]
        path = (self.directory / entry["file"]).resolve()
        require(path.is_relative_to(self.directory), "Teacher tensor path escapes cache")
        require(entry["record_id"] == record_id, "Wrong teacher record ownership")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            require(list(handle.keys()) == ["action_states"], "Unexpected teacher tensor keys")
            metadata = handle.metadata() or {}
            require(metadata == {"identity": self.manifest["identity"], "record_id": record_id,
                                 "record_key": entry["record_key"], "format": CACHE_FORMAT},
                    "Teacher metadata mismatch")
            tensor = handle.get_tensor("action_states")
        require(list(tensor.shape) == entry["shape"] and tensor.dtype == torch.float32,
                "Teacher shape/dtype mismatch")
        require(bool(torch.isfinite(tensor).all()), "Nonfinite teacher tensor")
        return tensor.detach()


class TrustedCollator(Stage2Collator):
    def __init__(self, processor, model_config, manifest, teacher_cache_dir=None):
        # Avoid the original processor-identity hash. Inherited __call__ retains
        # latent boundary, image-grid, action ownership and tensor-shape checks.
        self.processor, self.config = processor, model_config
        self.teacher_cache_dir = teacher_cache_dir
        self._teacher_reader = None

    def __call__(self, records):
        if self._teacher_reader is None:
            self._teacher_reader = TeacherReader(self.teacher_cache_dir)
        return super().__call__(records)


def load_existing(config, info):
    dataset = Stage2Dataset(config.prepared_dir, verify_checksum=False)
    require(len(dataset) == info["dataset_samples"], "Prepared sample count changed")
    teacher = read_json(Path(config.teacher_cache_dir) / "manifest.json")
    require(teacher.get("format") == CACHE_FORMAT and teacher.get("samples") == len(dataset),
            "Teacher cache format/count mismatch")
    require(set(teacher["records"]) == {r["record_id"] for r in dataset.records},
            "Teacher records do not cover prepared data")
    for record in dataset.records:
        record.pop("teacher_record", None)
    return dataset
