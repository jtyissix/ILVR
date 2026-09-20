"""CoMT/VSP evaluation and EMMA response generation with continuous latents."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from .data import json_write, position_ids_for_images, resolve_image, token_ids, user_message, stable_hash
from .distributed import rank, world
from .generation import generate_continuous
from .model import input_embeddings, load_full_model
from . import vsp, emma


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
    parser.add_argument("--task", choices=["comt", "vsp", "emma"], default="comt")
    parser.add_argument("--vsp_prompt_style", choices=vsp.PROMPT_STYLES, default="ilvr_eval",
                        help="VSP prompt only: ilvr_eval matches official eval.py; mirage_marker reproduces the previous layout")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Default: EMMA 4096, CoMT/VSP 1024")
    parser.add_argument("--max_input_tokens", type=int, default=32768,
                        help="EMMA input limit: fail with pid rather than truncate a multi-image question")
    parser.add_argument("--attention_backend", choices=["sdpa", "flash_attention_2"], default="flash_attention_2")
    args = parser.parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = 4096 if args.task == "emma" else 1024
    if args.max_new_tokens <= 0 or args.max_input_tokens <= 0:
        parser.error("Token limits must be positive")
    # Validate before allocating GPU model replicas; VSP never needs gold answer text.
    if args.task == "vsp":
        samples, data_report = vsp.load_samples(args.test_data_path, args.image_root)
    elif args.task == "emma":
        samples, data_report = emma.load_samples(args.test_data_path, args.image_root)
    else:
        with Path(args.test_data_path).open(encoding="utf-8") as handle:
            samples = [json.loads(line) for line in handle if line.strip()]
        if not samples or any("original_final_answer" not in s for s in samples):
            raise ValueError("CoMT TEST must be nonempty and contain original_final_answer on every row")
        data_report = {"samples": len(samples)}
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    model = load_full_model(args.model_path, args.attention_backend).eval().to(device)
    from transformers import AutoProcessor
    # Reuse established scoring, rather than changing the criterion between experiments.
    if args.task == "comt":
        from eval import extract_final_answer, normalize_for_match
    from PIL import Image
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    ids = token_ids(processor, model.config.to_dict())
    eos = model.config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prompt_style = args.vsp_prompt_style if args.task == "vsp" else "ilvr_eval"
    if args.task == "emma":
        prompt_style = f"emma_official_{data_report['prompt_strategy']}"
    if rank() == 0:
        json_write(output / "evaluation_config.json", {**vars(args), "world_size": world(),
                   "latent_size": model.config.latent_size, "data": data_report,
                   "prompt_style": prompt_style, "do_sample": False,
                   "ilvr_prompt_source_revision": vsp.ILVR_PROMPT_REVISION if args.task != "emma" else None,
                   "emma_prompt_source_revision": emma.EMMA_REVISION if args.task == "emma" else None,
                   "scoring_status": "pending_offline_scoring" if args.task == "emma" else "inline",
                   "vision_attention": sorted({type(block.attn).__name__ for block in model.visual.blocks})})
    print(f"[rank {rank()}] evaluating {len(range(rank(), len(samples), world()))} samples; "
          f"prompt={prompt_style}, latent_size={model.config.latent_size}, decoding=greedy", flush=True)
    rank_file = output / f"predictions_rank{rank()}.jsonl"
    with rank_file.open("w", encoding="utf-8") as handle:
        for index in range(rank(), len(samples), world()):
            sample = samples[index]
            if args.task == "vsp":
                paths = vsp.image_paths(sample, args.image_root)
                message = vsp.user_message(sample, paths, args.vsp_prompt_style)
            elif args.task == "emma":
                paths = emma.image_paths(sample, args.image_root)
                message = emma.user_message(sample, paths)
            else:
                paths = sample.get("image_input", [])
                paths = [paths] if isinstance(paths, str) else paths
                paths = [str(resolve_image(args.image_root, p)) for p in paths]
                message = user_message(sample, paths)
            if args.task == "emma":
                # Official EMMA Qwen wrapper resizes images with qwen_vl_utils
                # before passing them to the checkpoint's own processor.
                pictures = emma.vision_inputs(message)
            else:
                pictures = []
                for path in paths:
                    with Image.open(path) as image:
                        pictures.append(image.convert("RGB"))
            template_options = {"add_vision_id": True} if args.task == "emma" else {}
            text = processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True, **template_options)
            batch = processor(text=[text], images=pictures or None, return_tensors="pt", padding=True)
            grids = batch["image_grid_thw"].tolist() if pictures else []
            if args.task == "emma" and batch["input_ids"].shape[-1] > args.max_input_tokens:
                raise ValueError(f"EMMA {sample['pid']}: {batch['input_ids'].shape[-1]} input tokens exceed "
                                 f"--max_input_tokens {args.max_input_tokens}; question was not truncated")
            if args.task == "emma":
                print(f"[rank {rank()}] generating index={index} pid={sample['pid']} "
                      f"images={len(pictures)} input_tokens={batch['input_ids'].shape[-1]}", flush=True)
            if args.task in ("vsp", "emma") and index == rank():
                # Capture the actual template output and expanded token IDs, not a
                # separately reconstructed prompt. One small diagnostic file/rank.
                json_write(output / f"prompt_rank{rank()}.json", {
                    "index": index, "sample_id": sample.get("pid", sample.get("map_id")), "prompt_style": prompt_style,
                    **({"map_id": sample.get("map_id"), "ilvr_prompt_source_revision": vsp.ILVR_PROMPT_REVISION}
                       if args.task == "vsp" else {"pid": sample["pid"]}),
                    "source_revision": emma.EMMA_REVISION if args.task == "emma" else vsp.ILVR_PROMPT_REVISION,
                    "image_paths": paths, "messages": [message], "rendered_prompt": text,
                    "input_ids": batch["input_ids"][0].tolist(), "image_grid_thw": grids,
                })
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
            if args.task == "vsp":
                scored = vsp.score(sample, raw)
            elif args.task == "emma":
                scored = emma.pending_record(sample, raw)
            else:
                prediction = extract_final_answer(raw)
                correct = normalize_for_match(prediction) == normalize_for_match(str(sample["original_final_answer"]))
                scored = {"category": task_category(sample), "prediction": prediction,
                          "gold": sample["original_final_answer"], "correct": correct}
            record = {"index": index, **scored, "raw_output": raw, "generation_seconds": seconds, **result,
                      "prompt_style": prompt_style,
                      "prompt_input_sha256": stable_hash({"input_ids": batch["input_ids"][0].tolist(),
                                                          "image_grid_thw": grids})}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            completed = (index - rank()) // world() + 1
            print(f"[rank {rank()}] {completed}/{len(range(rank(), len(samples), world()))} "
                  f"index={index} correct={scored['correct']} generation={seconds:.1f}s", flush=True)
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
        if args.task == "emma":
            metrics = {**emma.summarize(merged), "scoring": "pending_offline_scoring", "data": data_report,
                       "ilvr_exact_reproduction": False}
            json_write(output / "emma_responses.json", emma.official_results(samples, merged))
        else:
            metrics = vsp.summarize(merged) if args.task == "vsp" else summarize(merged)
        json_write(output / "metrics.json", metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
