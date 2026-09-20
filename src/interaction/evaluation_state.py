"""Resumable rank-local prediction files and progress for independent evaluation."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from .data import json_write


def check_resume_config(previous, current):
    # Scheduling/diagnostics can change; inputs and decoding must remain fixed.
    keys = ("task", "model_path", "test_data_path", "image_root", "world_size", "latent_size",
            "max_new_tokens", "max_input_tokens", "attention_backend", "prompt_style", "do_sample",
            "ilvr_prompt_source_revision", "emma_prompt_source_revision", "data")
    changed = [key for key in keys if previous.get(key) != current.get(key)]
    if changed:
        raise ValueError(f"Cannot resume: evaluation settings changed: {', '.join(changed)}. Use a new output_dir.")


def load_rank_records(path, samples, task, rank_id, world_size, prompt_style, latent_size, repair_tail=False):
    """Validate completed rows, optionally preserving then trimming an interrupted write.

    EMMA correct=null is a COMPLETED generation, not a reason to regenerate.
    Repair only an unterminated final line; other corruption is an error.
    """
    path = Path(path)
    if not path.exists():
        return []
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    records, seen, offset = [], set(), 0
    trim_at = None
    for number, raw in enumerate(lines, 1):
        if not raw.strip():
            offset += len(raw)
            continue
        try:
            row = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeError) as exc:
            if repair_tail and number == len(lines) and not raw.endswith(b"\n"):
                trim_at = offset
                break
            raise ValueError(f"{path}:{number}: invalid prediction JSON") from exc
        index = row.get("index")
        if type(index) is not int or not 0 <= index < len(samples) or index % world_size != rank_id:
            raise ValueError(f"{path}:{number}: index belongs to a different sample/rank: {index}")
        if index in seen:
            raise ValueError(f"{path}:{number}: duplicate index {index}")
        sample = samples[index]
        if task == "emma" and (row.get("pid") != sample["pid"] or row.get("subject") != sample["subject"]):
            raise ValueError(f"{path}:{number}: EMMA identity mismatch")
        if task == "vsp" and str(row.get("map_id")) != str(sample["map_id"]):
            raise ValueError(f"{path}:{number}: VSP identity mismatch")
        if task == "comt" and row.get("gold") != sample["original_final_answer"]:
            raise ValueError(f"{path}:{number}: CoMT answer mismatch")
        if row.get("prompt_style") != prompt_style or any(n != latent_size for n in row.get("segment_steps", [])):
            raise ValueError(f"{path}:{number}: prompt/latent configuration mismatch")
        if not isinstance(row.get("raw_output"), str) or not isinstance(row.get("token_ids"), list):
            raise ValueError(f"{path}:{number}: incomplete generation record")
        if "correct" not in row or type(row["correct"]) is not bool and not (task == "emma" and row["correct"] is None):
            raise ValueError(f"{path}:{number}: invalid correctness field")
        records.append(row)
        seen.add(index)
        offset += len(raw)
    if trim_at is not None:
        backup = path.with_name(path.name + f".partial-{time.time_ns()}")
        backup.write_bytes(payload[trim_at:])
        with path.open("r+b") as handle:
            handle.truncate(trim_at)
        print(f"Recovered {path}: preserved interrupted tail in {backup.name}", flush=True)
        payload = payload[:trim_at]
    if repair_tail and payload and not payload.endswith(b"\n"):
        with path.open("ab") as handle:
            handle.write(b"\n")
    return records


class EvaluationProgress:
    def __init__(self, output, rank_id, local_rank, total):
        self.path = Path(output) / f"progress_rank{rank_id}.json"
        self.state = {"rank": rank_id, "local_rank": local_rank, "assigned_samples": total, "completed": 0}

    def update(self, stage, **values):
        self.state.update(stage=stage, updated_at=datetime.now(timezone.utc).isoformat(), **values)
        json_write(self.path, self.state)
        details = " ".join(f"{key}={self.state[key]}" for key in (
            "index", "pid", "completed", "input_tokens", "generated_tokens", "elapsed_seconds") if key in self.state)
        print(f"[rank {self.state['rank']}] {stage} {details}", flush=True)
