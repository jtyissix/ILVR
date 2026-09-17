"""CoMT evaluation with the existing repository's answer-matching rules."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from .data import json_write, position_ids_for_images, resolve_image, token_ids, user_message
from .distributed import rank, world
from .generation import generate_continuous
from .model import input_embeddings, load_full_model


def task_category(sample):
    paths = sample.get("image_input", [])
    paths = [paths] if isinstance(paths, str) else paths
    for path in paths:
        parts = path.replace("\\", "/").split("/")
        for category in ("creation", "deletion", "selection", "update"):
            if category in parts:
                return category
    return "unknown"


def summarize(records):
    counts = defaultdict(lambda: [0, 0])
    for r in records:
        counts[r["category"]][0] += int(r["correct"])
        counts[r["category"]][1] += 1
    total = len(records)
    return {"scoring": "repository eval.py extract_final_answer + normalize_for_match (exact match)",
            "samples": total, "correct": sum(r["correct"] for r in records),
            "accuracy": sum(r["correct"] for r in records)/max(1, total),
            "by_task": {k: {"correct": v[0], "samples": v[1], "accuracy": v[0]/v[1]} for k, v in sorted(counts.items())},
            "macro_task_accuracy": sum(v[0]/v[1] for v in counts.values())/max(1, len(counts)),
            "token_limit_count": sum(r["hit_token_limit"] for r in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_data_path", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--attention_backend", choices=["sdpa", "flash_attention_2"], default="flash_attention_2")
    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    model = load_full_model(args.model_path, args.attention_backend).eval().to(device)
    from transformers import AutoProcessor
    # Reuse established scoring, rather than changing the criterion between experiments.
    from eval import extract_final_answer, normalize_for_match
    from PIL import Image
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    ids = token_ids(processor, model.config.to_dict())
    eos = model.config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    with Path(args.test_data_path).open(encoding="utf-8") as handle:
        samples = [json.loads(line) for line in handle if line.strip()]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rank_file = output / f"predictions_rank{rank()}.jsonl"
    with rank_file.open("w", encoding="utf-8") as handle:
        for index in range(rank(), len(samples), world()):
            sample = samples[index]
            if "original_final_answer" not in sample:
                raise ValueError(f"TEST sample {index} lacks original_final_answer")
            paths = sample.get("image_input", [])
            paths = [paths] if isinstance(paths, str) else paths
            paths = [str(resolve_image(args.image_root, p)) for p in paths]
            pictures = []
            for path in paths:
                with Image.open(path) as image:
                    pictures.append(image.convert("RGB"))
            text = processor.apply_chat_template([user_message(sample, paths)], tokenize=False, add_generation_prompt=True)
            batch = processor(text=[text], images=pictures or None, return_tensors="pt")
            grids = batch["image_grid_thw"].tolist() if pictures else []
            positions = position_ids_for_images(batch["input_ids"][0].tolist(), grids, model.config.to_dict())[:, None].to(device)
            with torch.no_grad():
                features = model.visual(batch["pixel_values"].to(device=device, dtype=torch.bfloat16),
                                         grid_thw=batch["image_grid_thw"].to(device)) if pictures else None
                embeds = input_embeddings(model.model, batch["input_ids"].to(device), features, model.config.image_token_id)
                torch.cuda.synchronize()
                tick = time.perf_counter()
                result = generate_continuous(model.model, model.lm_head, embeds, positions, ids, eos,
                                             model.config.latent_size, args.max_new_tokens, args.attention_backend)
                torch.cuda.synchronize()
                seconds = time.perf_counter()-tick
            raw = processor.tokenizer.decode(result["token_ids"], skip_special_tokens=False)
            prediction = extract_final_answer(raw)
            correct = normalize_for_match(prediction) == normalize_for_match(str(sample["original_final_answer"]))
            record = {"index": index, "category": task_category(sample), "prediction": prediction,
                      "gold": sample["original_final_answer"], "correct": correct, "raw_output": raw,
                      "generation_seconds": seconds, **result}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    if dist.is_initialized():
        dist.barrier()
    if rank() == 0:
        merged = []
        for worker in range(world()):
            with (output / f"predictions_rank{worker}.jsonl").open(encoding="utf-8") as handle:
                merged.extend(json.loads(line) for line in handle)
        merged.sort(key=lambda r: r["index"])
        if [r["index"] for r in merged] != list(range(len(samples))):
            raise ValueError("Distributed evaluation lost or duplicated examples")
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for record in merged:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        metrics = summarize(merged)
        json_write(output / "metrics.json", metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
