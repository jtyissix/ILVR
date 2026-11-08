"""Train only: python -m sft_offload_4.run --config sft_offload_4/converted/latent.json"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from src.sft2_offload.config import PROJECT_ROOT, Stage2Config, configure_deepspeed, validate_local_inputs
from .checkpoint import read_json, resume_context
from .state import MODEL_FILE, optimizer_file, optimizer_path, require


def preflight(config):
    require(config.mode in {"train", "smoke"} and config.resume, "A converted/4-card resume checkpoint is required")
    require(config.num_gpus in (None, 4), "This launcher always uses exactly four GPUs")
    validate_local_inputs(config)
    configure_deepspeed(config)
    checkpoint, _, info = resume_context(config)
    require((checkpoint / MODEL_FILE).is_file(), "Missing model checkpoint")
    for rank in range(4):
        optimizer_path(checkpoint, rank)
        require((checkpoint / f"rng_rank{rank}.pt").is_file(), f"Missing RNG file rank {rank}")
    require((Path(config.prepared_dir) / "records.jsonl").is_file(), "Missing prepared data")
    manifest = read_json(Path(config.prepared_dir) / "manifest.json")
    require(manifest["samples"] == info["dataset_samples"], "Prepared sample count changed")
    require((Path(config.teacher_cache_dir) / "manifest.json").is_file(), "Missing teacher cache")
    # Check/build CPUAdam before distributed initialization; no prepare/teacher job.
    from src.sft2_offload.runtime import check_runtime
    check_runtime()
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    from src.interaction.data import json_write
    json_write(Path(config.output_dir) / "train_config.json", config.to_dict())
    print("4-card preflight passed: reusing prepared and teacher caches; no data hashing or rebuilding.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent / "converted" / "latent.json"))
    args = parser.parse_args(argv)
    filename = Path(args.config).resolve()
    config = Stage2Config.load(filename)
    if config.gpu_ids is not None:
        require(len(config.gpu_ids) == 4, "gpu_ids must contain exactly four devices")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.gpu_ids))
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(PROJECT_ROOT / "transformers/src"),
                                                          str(PROJECT_ROOT), os.environ.get("PYTHONPATH", "")]))
    import torch
    require(torch.cuda.device_count() >= 4, "Four visible GPUs are required")
    preflight(config)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               "--nproc_per_node=4", "--module", "sft_offload_4.train", "--config", str(filename)]
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
