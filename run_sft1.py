"""Edit configs/sft1/warmup.json, then run: python run_sft1.py."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import uuid

from src.sft1.config import DEFAULT_CONFIG, PROJECT_ROOT, WarmupConfig, validate_local_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Optional alternative JSON file")
    args = parser.parse_args(argv)
    filename = Path(args.config).expanduser().resolve()
    config = WarmupConfig.load(filename)
    validate_local_inputs(config)
    # Set device visibility before importing torch. No shell command composition.
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
    requested = None
    if config.mode != "prepare_only":
        import torch
        visible = torch.cuda.device_count()
        requested = config.num_gpus or visible
        if not visible or requested > visible:
            raise RuntimeError(f"SFT-I needs CUDA GPUs: requested {requested}, visible {visible}. "
                               "Adjust gpu_ids/num_gpus in the JSON; prepare_only does not need a GPU.")
    os.environ["SFT1_PREFLIGHT_ID"] = uuid.uuid4().hex
    # Potentially hours of preparation must finish before distributed collective
    # timeouts start. A subprocess also releases its large in-memory record cache.
    preflight = [sys.executable, "-m", "src.sft1.prepare", "--config", str(filename)]
    prepared = subprocess.run(preflight, cwd=PROJECT_ROOT, check=False)
    if prepared.returncode or config.mode == "prepare_only":
        return prepared.returncode
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               f"--nproc_per_node={requested}", "--module", "src.sft1.train", "--config", str(filename)]
    print(f"Launching SFT-I ({config.mode}) on {requested} GPU(s), config: {filename}", flush=True)
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"SFT-I setup error: {exc}", file=sys.stderr)
        raise SystemExit(2)
