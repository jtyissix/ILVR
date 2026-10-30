"""Edit configs/sft2_offload/latent.json, then run: python run_sft2_offload.py."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import uuid

from src.sft2_offload.config import DEFAULT_CONFIG, PROJECT_ROOT, Stage2Config, validate_local_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    filename = Path(args.config).expanduser().resolve()
    config = Stage2Config.load(filename)
    validate_local_inputs(config)
    if config.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.gpu_ids))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    local_transformers = str(PROJECT_ROOT / "transformers" / "src")
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [local_transformers, str(PROJECT_ROOT),
                                                           os.environ.get("PYTHONPATH", "")]))
    os.environ["SFT2_PREFLIGHT_ID"] = uuid.uuid4().hex
    requested = None
    if config.mode != "prepare_only":
        import torch
        visible = torch.cuda.device_count()
        requested = config.num_gpus or visible
        if not visible or requested > visible:
            raise RuntimeError(f"SFT-II needs CUDA GPUs: requested {requested}, visible {visible}. "
                               "prepare_only does not require a GPU.")
    prepared = subprocess.run([sys.executable, "-m", "src.sft2_offload.prepare", "--config", str(filename)],
                              cwd=PROJECT_ROOT, check=False)
    if prepared.returncode or config.mode == "prepare_only":
        return prepared.returncode
    # The coordinator spawns one independent teacher per selected GPU. All exit
    # and their caches are validated before the student's NCCL initialization.
    teacher = subprocess.run([sys.executable, "-m", "src.sft2.teacher", "--config", str(filename),
                              "--workers", str(requested)],
                             cwd=PROJECT_ROOT, check=False)
    if teacher.returncode:
        return teacher.returncode
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               f"--nproc_per_node={requested}", "--module", "src.sft2_offload.train", "--config", str(filename)]
    print(f"Launching optimizer-CPU-offload SFT-II ({config.mode}) on {requested} GPU(s): {filename}", flush=True)
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"SFT-II setup error: {exc}", file=sys.stderr)
        raise SystemExit(2)
