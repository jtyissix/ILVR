"""CPU regression tests: python -m unittest sft_offload_4.test_migration"""
from collections import OrderedDict
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.sft2.config import Stage2Config, PROJECT_ROOT
from src.sft2.checkpoint import FORMAT
from src.interaction.distributed import GlobalBatchSampler
from .checkpoint import read_json, resume_context, signature
from .convert import convert, write_json
from .state import (MODEL_FILE, equal, load, mapping, optimizer_file, optimizer_path, partition_size,
                    reshard_rank, validate_source, verify_rank)


def fixture(root, world=8, bf16=True):
    root = Path(root)
    checkpoint = root / "old" / "checkpoints" / "global_step00000002"
    checkpoint.mkdir(parents=True)
    prepared = root / "prepared"
    prepared.mkdir()
    write_json(prepared / "manifest.json", {"samples": 80})
    ds = read_json(PROJECT_ROOT / "configs/sft2_offload/zero2.json")
    ds_path = root / "old-zero2.json"
    write_json(ds_path, ds)
    config = Stage2Config(prepared_dir=str(prepared), output_dir=str(root / "old"),
                          deepspeed=str(ds_path), gradient_accumulation_steps=2, epochs=2).resolve_paths()
    config_path = root / "old-latent.json"
    write_json(config_path, config.to_dict())
    batch = world * 2
    metadata = {"format": FORMAT, "epoch": 0, "next_window": 2, "world_size": world,
                "optimizer_steps": 2, "warmup_fingerprint": "warmup", "teacher_fingerprint": "teacher"}
    metadata["signature"] = signature(config, "dataset", "warmup", "teacher", world)
    shapes = [OrderedDict([("decoder.a", (9,)), ("decoder.b", (2, 5))]),
              OrderedDict([("interaction.one", (1,))]),
              OrderedDict([("aux_decoder.w", (7, 5))])]
    sizes = [19, 1, 35]
    weights = [torch.arange(n, dtype=torch.float32).mul(.01).add(.2) for n in sizes]
    groups = [{"params": [i], "lr": .001, "initial_lr": .001, "betas": (.9, .999),
               "eps": 1e-8, "weight_decay": .01, "amsgrad": False} for i in range(3)]
    module = OrderedDict()
    for group, flat in zip(shapes, weights):
        offset = 0
        for name, shape in group.items():
            length = torch.Size(shape).numel()
            module[name] = flat[offset:offset+length].reshape(shape).to(torch.bfloat16 if bf16 else torch.float32)
            offset += length
    effective = {**ds, "train_micro_batch_size_per_gpu": 1, "gradient_accumulation_steps": 2,
                 "train_batch_size": batch}
    model = {**metadata, "module": module, "param_shapes": shapes, "dp_world_size": world,
             "mp_world_size": 1, "ds_version": "0.18.0", "global_steps": 2,
             "global_samples": 2*batch, "skipped_steps": 0, "ds_config": effective,
             "lr_scheduler": {"last_epoch": 2, "_last_lr": [.001]*3}}
    shards = []
    for rank in range(world):
        state = {"loss_scaler": SimpleNamespace(cur_scale=1.), "dynamic_loss_scale": False,
                 "overflow": False, "clip_grad": 1., "zero_stage": 2, "ds_version": "0.18.0",
                 "partition_count": [world]*3, "group_paddings": [], "param_slice_mappings": [],
                 "single_partition_of_fp32_groups": [],
                 "base_optimizer_state": {"param_groups": groups, "state": {}}}
        for i, (numel, shape, flat) in enumerate(zip(sizes, shapes, weights)):
            size = partition_size(numel, world)
            valid = min(size, max(0, numel-rank*size))
            state["group_paddings"].append(size-valid)
            state["param_slice_mappings"].append(mapping(shape, size, rank, SimpleNamespace))
            state["single_partition_of_fp32_groups"].append(flat[rank*size:rank*size+valid].clone())
            moment = torch.zeros(size)
            moment[:valid] = flat[rank*size:rank*size+valid] * .1
            state["base_optimizer_state"]["state"][i] = {
                "step": torch.tensor(2.), "exp_avg": moment, "exp_avg_sq": moment.square()+.001}
        shard = {"optimizer_state_dict": state, "ds_config": effective, "ds_version": "0.18.0"}
        torch.save(shard, checkpoint / optimizer_file(rank, bf16))
        torch.save({"python": None, "cuda": None}, checkpoint / f"rng_rank{rank}.pt")
        shards.append(shard)
    torch.save(model, checkpoint / MODEL_FILE)
    write_json(checkpoint / "complete.json", metadata)
    write_json(checkpoint.parent.parent / "run_info.json", {
        "world_size": world, "signature": metadata["signature"], "dataset_sha256": "dataset",
        "optimizer_offload": "cpu", "effective_batch": batch, "total_optimizer_steps": 10})
    return checkpoint, config_path, config, model, shards, sizes


class MigrationTest(unittest.TestCase):
    def test_complete_conversion_preserves_states_and_only_changes_parallel_layout(self):
        with tempfile.TemporaryDirectory() as root:
            source, config_path, config, model, shards, sizes = fixture(root)
            snapshots = {p: p.read_bytes() for p in source.iterdir()}
            target_config = convert(source, config_path, Path(root) / "converted")
            config4 = Stage2Config.load(target_config)
            path, state, info = resume_context(config4)
            self.assertEqual(config4.gradient_accumulation_steps, 4)
            self.assertEqual(config4.num_gpus, 4)
            self.assertEqual(config4.prepared_dir, config.prepared_dir)
            self.assertEqual(config4.teacher_cache_dir, config.teacher_cache_dir)
            self.assertEqual(info["max_conversion_error"], 0.)
            restored = load(path / MODEL_FILE)
            equal(restored["module"], model["module"], "model")
            equal(restored["lr_scheduler"], model["lr_scheduler"], "scheduler")
            for rank in range(4):
                expected = reshard_rank(shards, model["param_shapes"], sizes, rank, 4, restored["ds_config"])
                verify_rank(optimizer_path(path, rank), shards, sizes, rank, 4, expected)
            for file, content in snapshots.items():
                self.assertEqual(file.read_bytes(), content)
            config4.gradient_accumulation_steps *= 2
            with self.assertRaisesRegex(ValueError, "settings changed"):
                resume_context(config4)

    def test_adam_next_update_matches_unsharded_reference(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, _, model, shards, sizes = fixture(root)
            full = reshard_rank(shards, model["param_shapes"], sizes, 0, 1, model["ds_config"])
            parts = [reshard_rank(shards, model["param_shapes"], sizes, r, 4, model["ds_config"]) for r in range(4)]
            def update(shard, rank, world):
                state = shard["optimizer_state_dict"]
                outputs = []
                for i, n in enumerate(sizes):
                    size = partition_size(n, world)
                    flat = torch.nn.Parameter(torch.zeros(size))
                    valid = state["single_partition_of_fp32_groups"][i].numel()
                    flat.data[:valid] = state["single_partition_of_fp32_groups"][i]
                    adam = torch.optim.AdamW([flat], lr=.001, weight_decay=.01)
                    adam.state[flat] = copy.deepcopy(state["base_optimizer_state"]["state"][i])
                    flat.grad = torch.arange(rank*size, (rank+1)*size, dtype=torch.float32).sin() * .02
                    flat.grad[valid:] = 0
                    adam.step()
                    outputs.append((flat.detach()[:valid], adam.state[flat]))
                return outputs
            expected = update(full, 0, 1)
            actual = [update(part, r, 4) for r, part in enumerate(parts)]
            for i, n in enumerate(sizes):
                equal(torch.cat([r[i][0] for r in actual]), expected[i][0], "next weight update")
                for key in ("exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(
                        torch.cat([r[i][1][key][:r[i][0].numel()] for r in actual]),
                        expected[i][1][key][:n], rtol=1e-6, atol=1e-10)

    def test_rejects_nonfinite_wrong_padding_and_negative_variance(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, _, model, shards, _ = fixture(root)
            validate_source(model, shards, 8, 2)
            for mutate in (
                lambda s: s[0]["optimizer_state_dict"]["single_partition_of_fp32_groups"][0].fill_(float("nan")),
                lambda s: s[7]["optimizer_state_dict"]["group_paddings"].__setitem__(0, 0),
                lambda s: s[0]["optimizer_state_dict"]["base_optimizer_state"]["state"][0]["exp_avg_sq"].fill_(-1.),
                lambda s: s[0]["optimizer_state_dict"]["base_optimizer_state"]["state"][0].__setitem__("step", 3),
            ):
                altered = copy.deepcopy(shards)
                mutate(altered)
                with self.assertRaises(ValueError):
                    validate_source(model, altered, 8, 2)

    def test_failed_conversion_never_publishes_complete(self):
        with tempfile.TemporaryDirectory() as root:
            source, config_path, *_ = fixture(root)
            output = Path(root) / "converted"
            with patch("sft_offload_4.convert.verify_rank", side_effect=ValueError("bad bytes")):
                with self.assertRaisesRegex(ValueError, "bad bytes"):
                    convert(source, config_path, output)
            self.assertFalse((output / source.name / "complete.json").exists())
            self.assertFalse((output / "latest").exists())

    def test_global_windows_and_denominators_match_8_and_4_ranks(self):
        records = [{"segments": [0]*(i % 5), "length": 17+i} for i in range(137)]
        for epoch in range(2):
            def windows(world, gas):
                ranks = [list(GlobalBatchSampler(records, 1, gas, world, r, 42, epoch, 1)) for r in range(world)]
                return [[ranks[r][w*gas+m][0] for m in range(gas) for r in range(world)]
                        for w in range(len(ranks[0])//gas)]
            self.assertEqual(windows(8, 8), windows(4, 16))

    def test_teacher_reader_reuses_tensor_without_any_content_hash(self):
        from safetensors.torch import save_file
        from src.sft2.teacher import CACHE_FORMAT
        from .cache import TeacherReader
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            entry = {"file": "target.safetensors", "record_id": "sample", "record_key": "key", "shape": [4, 8]}
            write_json(root / "manifest.json", {"format": CACHE_FORMAT, "identity": "existing",
                                              "records": {"sample": entry}})
            expected = torch.randn(4, 8)
            metadata = {"identity": "existing", "record_id": "sample", "record_key": "key", "format": CACHE_FORMAT}
            save_file({"action_states": expected}, root / entry["file"], metadata=metadata)
            before = (root / entry["file"]).stat().st_mtime_ns
            with patch("src.sft2.teacher._file_hash", side_effect=AssertionError("must not hash")), \
                 patch("src.sft2.teacher._hash", side_effect=AssertionError("must not hash")):
                equal(TeacherReader(root)("sample"), expected, "teacher")
            self.assertEqual((root / entry["file"]).stat().st_mtime_ns, before)
            expected[0, 0] = float("nan")
            save_file({"action_states": expected}, root / entry["file"], metadata=metadata)
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                TeacherReader(root)("sample")

    def test_prepared_reuse_does_not_hash_or_rewrite(self):
        from src.sft2.data import CACHE_VERSION
        from src.sft2.teacher import CACHE_FORMAT
        from .cache import load_existing, TrustedCollator
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            teacher = root / "teacher"
            teacher.mkdir()
            write_json(root / "manifest.json", {"version": CACHE_VERSION, "stage": "sft2_latent", "samples": 1})
            (root / "records.jsonl").write_text(json.dumps(
                {"record_id": "sample", "teacher_record": {"old": "trajectory"}}) + "\n", encoding="utf-8")
            write_json(teacher / "manifest.json", {"format": CACHE_FORMAT, "samples": 1, "records": {"sample": {}}})
            paths = [root / "manifest.json", root / "records.jsonl", teacher / "manifest.json"]
            snapshots = [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths]
            config = SimpleNamespace(prepared_dir=str(root), teacher_cache_dir=str(teacher))
            with patch("src.sft2.data.file_hash", side_effect=AssertionError("must not hash")), \
                 patch("src.sft1.data._processor_identity", side_effect=AssertionError("must not hash")):
                dataset = load_existing(config, {"dataset_samples": 1})
                TrustedCollator(object(), {}, {}, str(teacher))
            self.assertNotIn("teacher_record", dataset.records[0])
            self.assertEqual(snapshots, [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths])


if __name__ == "__main__":
    unittest.main()
