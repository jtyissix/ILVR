"""Real spawned CPU processes exercise the multi-GPU cache orchestration protocol.

Targets are deterministic fixtures; numerical Qwen/CUDA inference is covered by
test_sft2_qwen. These tests do not claim a physical multi-GPU acceptance run.
"""
import json
import multiprocessing
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import torch

from src.sft2.teacher import (_progress_entries, _shard_records, _write_records, load_teacher_targets,
                             prepare_teacher_cache, validate_teacher_cache)
from tests import test_sft2_teacher as fixtures


def cpu_cache_worker(config, records, manifest, progress_file, device_index):
    torch.set_num_threads(1)
    audit = Path(config.teacher_cache_dir) / f"test-worker-{device_index}.json"
    audit.write_text(json.dumps({"pid": os.getpid(), "device_index": device_index,
                                 "records": [r["record_id"] for r in records]}), encoding="utf-8")
    if getattr(config, "test_fail_rank", None) == device_index:
        # Let one sibling durably finish so recovery is tested against real
        # per-worker journals, not just a mocked list of completed records.
        other = Path(progress_file).with_name("progress-rank-0.jsonl")
        deadline = time.monotonic() + 20
        while not _progress_entries(other):
            if time.monotonic() > deadline:
                raise RuntimeError("Test sibling did not finish")
            time.sleep(.05)
        raise RuntimeError("intentional worker failure")
    with patch("src.sft2.teacher._load_teacher", return_value=object()), \
         patch("src.sft2.teacher._extract_record", side_effect=lambda _components, r:
               torch.full((len(r["actions"]), manifest["hidden_size"]), float(r["record_id"][-1]) + 1)):
        _write_records(config, records, manifest, progress_file, device_index)


class ParallelTeacherTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.TeacherCacheTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.config, self.dataset = fixture.config, fixture.dataset

    def parallel(self, workers=2):
        with patch("src.sft2.teacher._cache_worker", new=cpu_cache_worker), \
             patch("torch.cuda.device_count", return_value=2):
            return prepare_teacher_cache(self.config, self.dataset, workers=workers)

    def test_spawned_shards_match_serial_and_completed_cache_starts_no_workers(self):
        serial = prepare_teacher_cache(self.config, self.dataset)
        serial_directory = self.config.teacher_cache_dir
        self.config.teacher_cache_dir += "-parallel"
        parallel = self.parallel()
        self.assertEqual(serial["identity"], parallel["identity"])
        self.assertEqual(set(serial["records"]), set(parallel["records"]))
        # Safetensors header key ordering may change file bytes across writers;
        # compare the aligned hidden values, and independently verify each hash.
        for record_id in serial["records"]:
            torch.testing.assert_close(load_teacher_targets(serial_directory, record_id),
                                       load_teacher_targets(self.config.teacher_cache_dir, record_id), rtol=0, atol=0)
        validate_teacher_cache(self.config, self.dataset)
        audits = [json.loads((Path(self.config.teacher_cache_dir) / f"test-worker-{i}.json").read_text())
                  for i in range(2)]
        self.assertEqual(len({audit["pid"] for audit in audits}), 2)
        self.assertNotIn(os.getpid(), {audit["pid"] for audit in audits})
        self.assertEqual([audit["device_index"] for audit in audits], [0, 1])
        owned = [rid for audit in audits for rid in audit["records"]]
        self.assertEqual(sorted(owned), sorted(parallel["records"]))
        with patch("src.sft2.teacher._run_parallel") as spawn, patch("torch.cuda.device_count", return_value=0):
            self.assertEqual(prepare_teacher_cache(self.config, self.dataset, workers=8), parallel)
            spawn.assert_not_called()

    def test_worker_failure_publishes_no_manifest_and_resumes_with_a_different_gpu_count(self):
        self.config.test_fail_rank = 1
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            self.parallel()
        self.assertFalse((Path(self.config.teacher_cache_dir) / "manifest.json").exists())
        self.assertFalse([p for p in multiprocessing.active_children() if p.name.startswith("sft2-teacher")])
        self.fixture.mock_extract.reset_mock()
        result = prepare_teacher_cache(self.config, self.dataset, workers=1)
        self.assertEqual(self.fixture.mock_extract.call_count, 1)
        self.assertEqual(len(result["records"]), 2)
        validate_teacher_cache(self.config, self.dataset)

    def test_length_balancing_assigns_every_record_once_and_drops_student_payload(self):
        records = [{"record_id": str(i), "teacher_record": {"input_ids": [1], "length": length},
                    "actions": [{}], "input_ids": [999] * 100}
                   for i, length in enumerate([10, 7, 6, 5, 3, 2])]
        shards = _shard_records(records, 2)
        self.assertEqual(sorted(r["record_id"] for shard in shards for r in shard), list(map(str, range(6))))
        self.assertTrue(all("input_ids" not in r for shard in shards for r in shard))
        loads = [sum(r["teacher_record"]["length"] ** 2 for r in shard) for shard in shards]
        self.assertLessEqual(abs(loads[0] - loads[1]), 10)
        self.assertEqual(_shard_records([], 4), [])

    def test_repaired_journal_takes_precedence_over_an_older_complete_manifest(self):
        from src.sft2.teacher import _atomic_json, _hash
        manifest = prepare_teacher_cache(self.config, self.dataset)
        # Force an old index to disagree with repaired tensors without relying
        # on safetensors header serialization order.
        for entry in manifest["records"].values():
            entry["sha256"] = "0" * 64
            (Path(self.config.teacher_cache_dir) / entry["file"]).write_bytes(b"broken")
        manifest.pop("manifest_sha256")
        manifest["manifest_sha256"] = _hash(manifest)
        _atomic_json(Path(self.config.teacher_cache_dir) / "manifest.json", manifest)
        original = self.fixture.mock_extract.side_effect
        def interrupted(components, record):
            if record["record_id"].endswith("1"):
                raise RuntimeError("repair interrupted")
            return original(components, record)
        self.fixture.mock_extract.side_effect = interrupted
        with self.assertRaisesRegex(RuntimeError, "repair interrupted"):
            prepare_teacher_cache(self.config, self.dataset)
        self.fixture.mock_extract.side_effect = original
        self.fixture.mock_extract.reset_mock()
        prepare_teacher_cache(self.config, self.dataset)
        self.assertEqual(self.fixture.mock_extract.call_count, 1)
        validate_teacher_cache(self.config, self.dataset)


if __name__ == "__main__":
    unittest.main()
