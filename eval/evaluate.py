"""Edit the globals, then run python eval/evaluate.py. No CLI arguments needed."""
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ===================== 只修改此配置区 =====================
DATASET_DIR = str(Path(__file__).resolve().parent / "datasets")
MODEL_DIR = r"/path/to/Qwen2.5-VL-7B-Instruct"  # 或训练导出目录，自动定位其中的 backbone/
MODEL_KIND = "auto"              # auto / base / trained；auto 识别项目 SFT-II 导出标记
ONLY_DATASETS = ["all"]          # 全测用 ['all']；例如 ['mmvp', 'mathvista', 'hrbench']
EXCLUDE_DATASETS = []            # 两者必须恰好一个非空，例如 ONLY=[]、EXCLUDE=['seedbench']
OUTPUT_DIR = str(Path(__file__).resolve().parent / "outputs")
RUN_NAME = "qwen25vl7b"          # 区分模型；修改配置也会自动分配不同的结果子目录
MODE = "all"                    # all / infer / score / validate；score 不加载模型、不需要 GPU
RESUME = True                   # 已成功的逐题结果可恢复；失败题下次自动重试
MAX_SAMPLES = None              # 只用于调试；MMVP/HRBench 保留完整题对/循环组

DEVICE = "cuda:0"               # 每个进程单张卡；可分别选择不同数据集运行
DTYPE = "bfloat16"              # bfloat16 / float16 / float32
ATTENTION_BACKEND = "sdpa"      # sdpa / eager / flash_attention_2（需另装 flash-attn）
MIN_PIXELS = None               # None 使用保存的 processor 配置；单位为像素面积
MAX_PIXELS = None               # 基座和训练模型应保持一致；HRBench 高分辨率需要足够显存
MAX_NEW_STEPS = 4096            # 文本 token + 连续 latent 位置的共同预算
MIN_LATENT_STEPS = 1
MAX_LATENT_STEPS = 64           # 仅死循环保护；正常由模型自行预测 latent_end 退出
SEED = 42                      # greedy 生成；也固定 VisualPuzzles 官方随机兜底的种子
LIKELIHOOD_DATASETS = ["vstar", "seedbench"]  # 官方候选答案 NLL；清空则改为生成式（非原始协议）
LIKELIHOOD_INCLUDE_EOS = True   # 以 Qwen 对话结束 token 对应原模型的答案终止符

API_KEY = os.environ.get("OPENAI_API_KEY", "")  # 判分/抽取用；不参与本地模型推理
API_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
JUDGE_MODEL = "gpt-4o"          # 可换兼容接口；比较实验必须使用同一判分模型
JUDGE_MODELS = {}               # 可逐项指定，如 {'mathvista': '...', 'hrbench_4k': '...'}
JUDGE_TIMEOUT = 60
JUDGE_RETRIES = 3
RULES_ONLY = False              # 离线规则简化版，会标注非官方；MathVista 不允许静默简化
MM_MATH_PRECISION = 1e-2        # 原 MM-Math CLI 默认精度
# =========================================================

from eval.common import (record_path, read_json, read_jsonl, safe_path, select_benchmarks,
                         subset_samples, validate_samples, write_json)
from eval.judge import Judge
from eval.model import QwenBackbone, resolve_backbone
from eval.protocols import messages_for, score_one
from eval.reporting import aggregate, format_summary, write_summaries


def model_metadata(directory):
    root = resolve_backbone(directory)
    files = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix in (".json", ".safetensors", ".bin", ".txt", ".jinja"):
            stat = path.stat()
            files[path.name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return {"path": str(root), "files": files}


def load_prepared(root, name, *, limit=None, validate=False):
    prepared = Path(root) / name / "prepared"
    manifest = read_json(prepared / "manifest.json")
    samples = list(read_jsonl(prepared / "samples.jsonl"))
    if validate and len(samples) != manifest["count"]:
        raise ValueError(f"{name}: manifest/sample count mismatch")
    if validate:
        validate_samples(samples, name, prepared)
    samples = subset_samples(samples, limit)
    if validate:
        for sample in samples:
            messages_for(sample, prepared)
    return prepared, manifest, samples


def saved_result(directory, sample_id, identity, resume=True):
    path = record_path(directory, sample_id)
    if resume and path.is_file():
        result = read_json(path)
        if result.get("id") != sample_id or result.get("identity") != identity:
            return path, None
        if result.get("status") == "ok":
            return path, result
    return path, None


def run_dataset(samples, prepared, output, inference_id, score_id, *, backend_factory,
                judge, mode="all", likelihood=False, include_eos=True, resume=True,
                rules_only=False, seed=42, precision=1e-2):
    """Separate inference and scoring records so judge outages never lose predictions."""
    output = Path(output)
    scores, prediction_count, errors, truncated, latent = {}, 0, [], 0, 0
    for index, sample in enumerate(samples, 1):
        uid = sample["id"]
        prediction_identity = [inference_id, sample]
        pred_path, stored = saved_result(output / "predictions", uid, prediction_identity, resume)
        if stored is None and mode != "score":
            # Configuration/loading failures are fatal, not repeated per question.
            backend = backend_factory()
            try:
                messages = messages_for(sample, prepared, likelihood)
                start = time.monotonic()
                prediction = (backend.likelihood(messages, sample["choices"], include_eos=include_eos)
                              if likelihood else backend.generate(messages))
                stored = {"id": uid, "identity": prediction_identity, "status": "ok", "prediction": prediction,
                          "seconds": time.monotonic() - start}
                write_json(pred_path, stored)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                write_json(pred_path, {"id": uid, "identity": prediction_identity, "status": "error", "error": message})
                errors.append({"id": uid, "stage": "inference", "error": message})
        if stored is None:
            if mode == "score":
                errors.append({"id": uid, "stage": "inference", "error": "No successful saved prediction"})
            continue
        prediction = stored["prediction"]
        prediction_count += 1
        truncated += bool(prediction.get("truncated"))
        latent += len(prediction.get("latent_segments", []))
        if mode in ("all", "score"):
            # Compare the actual prediction and scoring settings on resume.
            item_score_id = [score_id, prediction, sample]
            score_path, scored = saved_result(output / "scores" / score_id, uid, item_score_id, resume)
            if scored is None:
                try:
                    result = score_one(sample, prediction, judge, seed=seed, mm_math_precision=precision, rules_only=rules_only)
                    scored = {"id": uid, "identity": item_score_id, "status": "ok", **result}
                    write_json(score_path, scored)
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    write_json(score_path, {"id": uid, "identity": item_score_id, "status": "error", "error": message})
                    errors.append({"id": uid, "stage": "scoring", "error": message})
            if scored:
                scores[uid] = scored
        if index == 1 or index % 10 == 0 or index == len(samples):
            print(f"{sample['benchmark']}: {index}/{len(samples)}, predictions={prediction_count}, scores={len(scores)}, errors={len(errors)}", flush=True)
    summary = aggregate(samples, scores)
    summary.update(predictions=prediction_count, truncated_predictions=truncated, latent_segments=latent,
                   inference_id=inference_id, score_id=score_id, rules_only=rules_only,
                   inference_method="likelihood" if likelihood else "generation", output=str(output))
    if mode == "infer":
        summary["complete"] = prediction_count == len(samples)
        summary["status"] = "inference_only"
    write_json(output / f"summary_{score_id}.json", summary)
    write_json(output / f"errors_{score_id}.json", errors)
    return summary


def main():
    selected = select_benchmarks(ONLY_DATASETS, EXCLUDE_DATASETS)
    if MODE not in ("all", "infer", "score", "validate"):
        raise ValueError("MODE must be all/infer/score/validate")
    if set(LIKELIHOOD_DATASETS) - {"seedbench", "vstar"}:
        raise ValueError("LIKELIHOOD_DATASETS supports only seedbench/vstar")
    if JUDGE_RETRIES < 1 or JUDGE_TIMEOUT <= 0:
        raise ValueError("JUDGE_RETRIES and JUDGE_TIMEOUT must be positive")
    run_root = safe_path(OUTPUT_DIR, RUN_NAME)
    run_root.mkdir(parents=True, exist_ok=True)
    prepared_data = {}
    for name in selected:
        print(f"Loading {name}...", flush=True)
        prepared_data[name] = load_prepared(DATASET_DIR, name, limit=MAX_SAMPLES, validate=MODE == "validate")
    if MODE == "validate":
        print("All selected datasets, images and prompts passed validation.", flush=True)
        return
    model_stamp = model_metadata(MODEL_DIR)
    generation_settings = {"model": model_stamp, "model_kind": MODEL_KIND, "device": DEVICE, "dtype": DTYPE,
                           "attention": ATTENTION_BACKEND, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS,
                           "max_new_steps": MAX_NEW_STEPS, "min_latent_steps": MIN_LATENT_STEPS,
                           "max_latent_steps": MAX_LATENT_STEPS, "seed": SEED}
    scoring_settings = {"base_url": API_BASE_URL, "judge_model": JUDGE_MODEL, "judge_models": JUDGE_MODELS,
                        "rules_only": RULES_ONLY, "mm_math_precision": MM_MATH_PRECISION, "seed": SEED}
    score_config = record_path(run_root / "scoring_configs", scoring_settings)
    write_json(score_config, scoring_settings)
    score_id = score_config.stem
    runtime_path = record_path(run_root / "runtimes", generation_settings)
    judge = Judge(API_KEY, API_BASE_URL, JUDGE_MODEL, run_root / "judge_cache", models=JUDGE_MODELS,
                  timeout=JUDGE_TIMEOUT, retries=JUDGE_RETRIES)
    backend = None
    def get_backend():
        nonlocal backend
        if backend is None:
            backend = QwenBackbone(MODEL_DIR, model_kind=MODEL_KIND, device=DEVICE, dtype=DTYPE,
                                   attention=ATTENTION_BACKEND, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
                                   min_latent_steps=MIN_LATENT_STEPS, max_latent_steps=MAX_LATENT_STEPS,
                                   max_new_steps=MAX_NEW_STEPS, seed=SEED)
            write_json(runtime_path, backend.runtime)
        return backend
    summaries = []
    for name, (prepared, manifest, samples) in prepared_data.items():
        likelihood = name in LIKELIHOOD_DATASETS
        identity = {**generation_settings, "dataset": manifest, "likelihood": likelihood,
                    "likelihood_include_eos": LIKELIHOOD_INCLUDE_EOS if likelihood else None}
        config_path = record_path(run_root / name / "configs", identity)
        write_json(config_path, identity)
        inference_id = config_path.stem
        destination = run_root / name / inference_id
        write_json(destination / "inference_config.json", identity)
        write_json(destination / f"score_config_{score_id}.json", scoring_settings)
        result = run_dataset(samples, prepared, destination, inference_id, score_id,
                             backend_factory=get_backend, judge=judge, mode=MODE, likelihood=likelihood,
                             include_eos=LIKELIHOOD_INCLUDE_EOS, resume=RESUME, rules_only=RULES_ONLY,
                             seed=SEED, precision=MM_MATH_PRECISION)
        result.update(diagnostic_subset=len(samples) != manifest["count"], full_dataset_count=manifest["count"],
                      source=manifest["source_url"])
        write_json(destination / f"summary_{score_id}.json", result)
        summaries.append(result)
        write_summaries(run_root, summaries)
        print(format_summary(result), flush=True)
    print(f"Results: {run_root / 'summary.json'} and summary.csv (accuracy_percent is in [0,100])", flush=True)
    if not all(s["complete"] for s in summaries):
        raise SystemExit("Incomplete run: see per-dataset errors files. Fix and rerun to resume successful predictions.")


if __name__ == "__main__":
    main()
