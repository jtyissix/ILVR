"""Replay a saved evaluation trajectory and overlay decoder attention on input images.

No gold answer enters the model. Saved token IDs and latent boundaries are forced;
continuous hidden states are recomputed. This is attention, not causal attribution.
"""
from datetime import datetime
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw

from .common import read_json, safe_path, write_json
from .protocols import messages_for


def numbered_filename(value):
    text = str(value).strip()
    if text.endswith(".json"):
        text = text[:-5]
    if not re.fullmatch(r"[0-9]+", text) or int(text) <= 0:
        raise ValueError("FILE_ID must be the positive JSON filename number, e.g. 1365 or 00001365.json")
    return f"{int(text):08d}.json"


def load_record(run, record_kind, score_id, file_id, dataset):
    run = Path(run)
    if record_kind not in ("scores", "predictions"):
        raise ValueError("RECORD_KIND must be scores or predictions")
    directory = safe_path(run / "scores", str(score_id)) if record_kind == "scores" else run / "predictions"
    path = directory / numbered_filename(file_id)
    record = read_json(path)  # Direct filename lookup; never sample_id +/- an offset.
    identity = record.get("identity")
    if not isinstance(identity, list):
        raise ValueError(f"Record has no embedded sample snapshot: {path}")
    if record_kind == "scores":
        if len(identity) < 3:
            raise ValueError("Score record must contain [score_id, prediction, sample, ...]")
        prediction, sample = identity[1:3]
    else:
        if len(identity) < 2 or record.get("status") != "ok":
            raise ValueError("Prediction record is unsuccessful or has no sample snapshot")
        prediction, sample = record.get("prediction"), identity[1]
    if not isinstance(prediction, dict) or not isinstance(sample, dict):
        raise ValueError("Record has no successful prediction/sample snapshot")
    if str(record.get("id")) != str(sample.get("id")) or sample.get("benchmark") != dataset:
        raise ValueError("Record sample ID / benchmark mismatch")
    return path, record, sample, prediction


def replay_plan(prediction, start_id, end_id, eos_ids):
    """Reconstruct advance() calls, including open/truncated latent segments.

    decode_step is 0-based as in decode_stream; segment/latent_step are 1-based.
    EOS is recorded by decode_stream but is never fed through advance().
    """
    if prediction.get("method") != "generation":
        raise ValueError("This record uses likelihood scoring, which has no latent trajectory to visualize")
    tokens = prediction["token_ids"]
    consumed = int(prediction["decode_steps"])
    if consumed < 0:
        raise ValueError("Negative decode_steps")
    latent, boundaries = {}, {}
    segments = prediction.get("latent_segments", [])
    if segments and (start_id is None or end_id is None or start_id == end_id):
        raise ValueError("Checkpoint has no distinct latent boundary IDs")
    for number, segment in enumerate(segments, 1):
        start, steps = int(segment["start_step"]), int(segment["steps"])
        stop = start + steps + 1
        if start < 0 or steps < 0 or start >= consumed or stop > consumed:
            raise ValueError("Latent segment outside saved decode_steps")
        if not segment.get("closed") and (number != len(segments) or stop != consumed):
            raise ValueError("An open latent segment must end at the truncated trajectory boundary")
        slots = {start: ("start", start_id)}
        if segment.get("closed"):
            if segment.get("end_step") != stop or stop >= consumed:
                raise ValueError("Latent end_step does not match start_step + steps + 1")
            slots[stop] = ("end", end_id)
        for step in range(start, stop + int(bool(segment.get("closed")))):
            if step in latent or step in boundaries:
                raise ValueError("Overlapping latent segments")
            if step in slots:
                boundaries[step] = slots[step]
            else:
                latent[step] = (number, step - start)
    plan, cursor = [], 0
    for step in range(consumed):
        if step in latent:
            segment, local = latent[step]
            plan.append({"decode_step": step, "token": None, "segment": segment, "latent_step": local})
        else:
            if cursor >= len(tokens):
                raise ValueError("Saved token_ids are shorter than the replay schedule")
            token = int(tokens[cursor]); cursor += 1
            if step in boundaries and token != boundaries[step][1]:
                raise ValueError(f"Boundary token mismatch at decode step {step}; check checkpoint/tokenizer")
            if step not in boundaries and token in {start_id, end_id} - {None}:
                raise ValueError("Latent boundary token missing from saved segment metadata")
            if token in eos_ids:
                raise ValueError("EOS cannot be fed during a saved advance step")
            plan.append({"decode_step": step, "token": token})
    tail = tokens[cursor:]
    if prediction.get("stopped_on_eos"):
        if len(tail) != 1 or tail[0] not in eos_ids or prediction.get("truncated"):
            raise ValueError("Saved EOS metadata is inconsistent")
    elif tail or not prediction.get("truncated"):
        raise ValueError("Saved token_ids / termination metadata are inconsistent")
    return plan


def image_layout(input_ids, grids, image_token_id, merge_size, image_paths):
    """Map contiguous prompt image tokens to post-merger row-major Qwen grids."""
    ids = np.asarray(input_ids).reshape(-1)
    positions = np.flatnonzero(ids == image_token_id)
    runs = np.split(positions, np.where(np.diff(positions) != 1)[0] + 1) if len(positions) else []
    if len(runs) != len(grids) or len(grids) != len(image_paths):
        raise ValueError("Image occurrences, image token spans and image_grid_thw disagree")
    result = []
    for index, (run, grid, path) in enumerate(zip(runs, grids, image_paths), 1):
        t, h, w = map(int, grid)
        if t != 1 or h <= 0 or w <= 0 or merge_size <= 0 or h % merge_size or w % merge_size:
            raise ValueError("Only still images with a valid merged Qwen spatial grid are supported")
        gh, gw = h // merge_size, w // merge_size
        if len(run) != gh * gw:
            raise ValueError("Image token count does not match the post-merger grid")
        result.append({"image": index, "path": str(path), "grid_h": gh, "grid_w": gw,
                       "token_start": int(run[0]), "token_end": int(run[-1]) + 1})
    return result


def select_layers(requested, count):
    indices = list(range(count)) if requested == "all" else list(requested)
    indices = [i + count if i < 0 else i for i in indices]
    if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= count for i in indices):
        raise ValueError(f"LAYERS must be distinct layer indices within {count} decoder layers")
    return indices


def extract_image_attention(attentions, layouts, layers):
    """Keep raw probabilities, mean over heads; never normalize over image tokens."""
    if attentions is None or any(attentions[i] is None for i in layers):
        raise RuntimeError("No attention returned; use SDPA with the supported Qwen 4.51.3 evaluator")
    maps = {f"image_{x['image']:03d}": [] for x in layouts}
    for layer in layers:
        weights = attentions[layer]
        if weights.ndim != 4 or weights.shape[0] != 1 or weights.shape[2] != 1:
            raise ValueError("Expected a single cached decode query [1, heads, 1, keys]")
        row = weights[0, :, 0].float().mean(dim=0).detach().cpu().numpy()
        if not np.isfinite(row).all() or not np.isclose(row.sum(), 1, atol=0.02):
            raise ValueError("Attention row is nonfinite or not a probability distribution")
        for layout in layouts:
            start, stop = layout['token_start'], layout['token_end']
            if stop > len(row):
                raise ValueError("Image span exceeds attention key length")
            maps[f"image_{layout['image']:03d}"].append(row[start:stop].reshape(layout['grid_h'], layout['grid_w']))
    return {key: np.stack(value).astype(np.float32) for key, value in maps.items()}


def overlay_image(image, weights, vmax, alpha=0.65):
    """Transparent low attention -> yellow -> red; fixed scale across steps."""
    if not 0 <= alpha <= 1 or vmax < 0 or not np.isfinite(vmax):
        raise ValueError("Invalid overlay alpha or scale")
    weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim != 2 or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Heatmap must be a finite nonnegative 2D array")
    scaled = np.clip(weights / vmax, 0, 1) if vmax > 0 else np.zeros_like(weights)
    resized = np.asarray(Image.fromarray(scaled).resize(image.size, Image.Resampling.BILINEAR))
    color = np.stack([np.ones_like(resized), 1 - resized, np.zeros_like(resized)], axis=-1) * 255
    opacity = (alpha * resized)[..., None]
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    return Image.fromarray(np.round(rgb * (1 - opacity) + color * opacity).clip(0, 255).astype(np.uint8))


def write_gallery(output, manifest):
    # Escape HTML and script terminators: model outputs are data, never executable.
    data = json.dumps(manifest, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    page = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>Latent attention</title>
<style>body{font:16px system-ui;margin:24px;background:#f5f6f8;color:#17212f}header,section{background:white;padding:20px;border-radius:12px;margin-bottom:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere}button,input{margin:8px}input{width:50%}.grid{display:flex;flex-wrap:wrap;gap:18px}.tile{flex:1;min-width:280px;max-width:800px}.tile img{width:100%;height:auto}small{color:#526174}.bar{width:240px;height:16px;background:linear-gradient(to right,white,#ffd300,#f00)}</style>
<header><h1>逐 latent step 图像注意力</h1><p id="identity"></p><pre id="question"></pre>
<p>颜色越红，注意力越高；低值透明。同一张图在所有步骤使用统一色标。显示所选 decoder 层及所有 heads 的平均注意力。</p>
<p><b>这是按保存轨迹重新计算的 attention，不是因果归因，也不等于模型一定看懂了该区域。</b></p>
<div class="bar"></div><small>0 → 本图所有步骤的最大 patch 权重</small>
<p id="audit"></p><details><summary>原回答与回放信息</summary><pre id="answer"></pre></details></header>
<section><button id="prev">上一步</button><input id="slider" type="range" min="0" value="0"><button id="next">下一步</button>
<h2 id="step"></h2><pre id="context"></pre><div class="grid" id="images"></div></section>
<script>const data=__DATA__;const frames=data.frames;const slider=document.getElementById('slider');
document.getElementById('identity').textContent=data.record_file+' | sample_id='+data.sample_id+' | layers='+data.layers.join(',');
document.getElementById('question').textContent=data.question;
document.getElementById('answer').textContent=data.saved_text+'\\n\\n'+data.replay_note;
document.getElementById('audit').textContent='重新计算与保存轨迹的策略分歧：'+data.policy_mismatches.length+' 处（不影响强制回放顺序；详见 manifest.json）';
slider.max=Math.max(0,frames.length-1);
function show(){const f=frames[Number(slider.value)];if(!f){document.getElementById('step').textContent='该记录没有连续 latent step';return;}
document.getElementById('step').textContent='Latent '+f.ordinal+' / '+frames.length+' · 段 '+f.segment+' · 段内步 '+f.latent_step+' · decode_step '+f.decode_step;
document.getElementById('context').textContent='此前文本：'+f.context;
const box=document.getElementById('images');box.replaceChildren();for(const im of f.images){const tile=document.createElement('div');tile.className='tile';
const title=document.createElement('p');title.textContent='输入图 '+im.image+' · 注意力占比 '+(100*im.attention_mass).toFixed(3)+'% · 网格 '+im.grid_h+'×'+im.grid_w;
const pic=document.createElement('img');pic.src=im.overlay;pic.alt=title.textContent;const link=document.createElement('a');link.href=im.original;link.textContent='打开原图';link.target='_blank';tile.append(title,pic,link);box.append(tile);}}
slider.oninput=show;document.getElementById('prev').onclick=()=>{slider.value=Math.max(0,Number(slider.value)-1);show()};document.getElementById('next').onclick=()=>{slider.value=Math.min(frames.length-1,Number(slider.value)+1);show()};
document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')document.getElementById('prev').click();if(e.key==='ArrowRight')document.getElementById('next').click()});show();</script></html>"""
    (output / "index.html").write_text(page.replace("__DATA__", data), encoding="utf-8")


def render_outputs(output, manifest, *, alpha, max_side):
    output = Path(output)
    (output / "overlays").mkdir()
    originals, scales = {}, {}
    for layout in manifest["images"]:
        key = f"image_{layout['image']:03d}"
        with Image.open(layout["path"]) as source:
            image = source.convert("RGB")
        # Qwen's still-image path does not EXIF-transpose or crop; follow it here.
        image.thumbnail((max_side, max_side))
        image.save(output / f"{key}_original.png")
        originals[key], scales[key] = image, 0.0
    for frame in manifest["frames"]:
        with np.load(output / frame["raw_attention"]) as arrays:
            for key in scales:
                scales[key] = max(scales[key], float(arrays[key].mean(axis=0).max()))
    for frame in manifest["frames"]:
        frame["images"] = []
        with np.load(output / frame["raw_attention"]) as arrays:
            for layout in manifest["images"]:
                key = f"image_{layout['image']:03d}"
                weights = arrays[key].mean(axis=0)
                relative = f"overlays/latent_{frame['ordinal']:05d}_{key}.png"
                overlay_image(originals[key], weights, scales[key], alpha).save(output / relative)
                frame["images"].append({"image": layout["image"], "overlay": relative,
                    "original": f"{key}_original.png", "attention_mass": float(weights.sum()),
                    "grid_h": layout["grid_h"], "grid_w": layout["grid_w"], "vmax": scales[key]})
    # Bounded, paginated contact sheets rather than a gigantic all-steps bitmap.
    for layout in manifest["images"]:
        image_index = layout["image"] - 1
        for start in range(0, len(manifest["frames"]), 24):
            frames = manifest["frames"][start:start + 24]
            sheet = Image.new("RGB", (4 * 300, ((len(frames) + 3) // 4) * 250), "white")
            draw = ImageDraw.Draw(sheet)
            for cell, frame in enumerate(frames):
                item = frame["images"][image_index]
                with Image.open(output / item["overlay"]) as image:
                    image.thumbnail((292, 216))
                    x, y = (cell % 4) * 300, (cell // 4) * 250
                    sheet.paste(image, (x + (300 - image.width) // 2, y + 25))
                draw.text((x + 5, y + 4), f"L{frame['ordinal']} S{frame['segment']}.{frame['latent_step']} mass={item['attention_mass']:.3%}", fill="black")
            sheet.save(output / f"contact_image_{layout['image']:03d}_page_{start // 24 + 1:03d}.jpg")
    manifest["color_scale"] = {"kind": "global_per_image", "maximum_patch_weights": scales, "alpha": alpha}
    write_json(output / "manifest.json", manifest)
    write_gallery(output, manifest)


def replay_policy_choice(backend, state, item, plan, policy):
    """Compare the greedy decision to the saved policy without changing replay."""
    step = item["decode_step"]
    in_latent = item.get("segment") is not None or item["token"] == policy.end_id
    choice = backend.next_token(state, set() if in_latent else {policy.end_id} - {None})
    if item["token"] is None:
        completed = item["latent_step"] - 1
        actual = policy.end_id if completed >= policy.max_steps or (choice == policy.end_id and completed >= policy.min_steps) else None
    elif item["token"] == policy.end_id:
        previous = plan[step - 1] if step else {}
        completed = previous.get("latent_step", 0)
        actual = policy.end_id if completed >= policy.max_steps or (choice == policy.end_id and completed >= policy.min_steps) else None
    else:
        actual = choice
    return actual


def visualize_record(*, eval_root, run_name, dataset, inference_id, score_id, record_kind,
                     file_id, dataset_dir, model_dir, device, layers, output_dir,
                     alpha=0.65, max_side=1024, validate_only=False):
    if not 0 <= alpha <= 1 or max_side < 64:
        raise ValueError("Require 0 <= OVERLAY_ALPHA <= 1 and DISPLAY_MAX_SIDE >= 64")
    run = safe_path(safe_path(safe_path(eval_root, run_name), dataset), str(inference_id))
    path, record, sample, prediction = load_record(run, record_kind, score_id, file_id, dataset)
    if prediction.get("method") != "generation":
        raise ValueError("Saved likelihood/NLL evaluations have no latent steps. Select a generation result.")
    config = read_json(run / "inference_config.json")
    prepared = safe_path(safe_path(dataset_dir, dataset), "prepared")
    messages = messages_for(sample, prepared)
    image_paths = [part["image"] for message in messages for part in message["content"] if part["type"] == "image"]
    if not image_paths:
        raise ValueError("This sample has no input images")
    for image_path in image_paths:
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"Input image missing; set DATASET_DIR to the matching prepared dataset: {image_path}")
    print(f"Record: {path}\nInternal sample ID: {sample['id']}\nImages: {image_paths}", flush=True)
    checkpoint = str(model_dir or config["model"]["path"])
    if validate_only:
        print(f"Checkpoint: {checkpoint}\nSaved latent steps: {sum(s['steps'] for s in prediction.get('latent_segments', []))}", flush=True)
        return {"record_file": str(path), "sample_id": sample["id"], "images": image_paths}
    from .model import QwenBackbone
    backend = QwenBackbone(checkpoint, model_kind=config.get("model_kind", "auto"), device=device,
                          dtype=config.get("dtype", "bfloat16"), attention="sdpa",
                          min_pixels=config.get("min_pixels"), max_pixels=config.get("max_pixels"),
                          min_latent_steps=config.get("min_latent_steps", 1), max_latent_steps=config.get("max_latent_steps", 64),
                          max_new_steps=config.get("max_new_steps", 4096), seed=config.get("seed", 42))
    plan = replay_plan(prediction, backend.policy.start_id, backend.policy.end_id, backend.eos_ids)
    selected = select_layers(layers, len(backend.model.model.layers))
    output = safe_path(safe_path(Path(output_dir), run_name), dataset) / f"{record_kind}_{path.stem}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "attention").mkdir()
    manifest = {"record_file": str(path), "record_kind": record_kind, "sample_id": sample['id'],
        "question": sample['question'], "gold": sample.get('answer'), "saved_text": prediction.get('text', ''),
        "saved_extracted": record.get('extracted'), "saved_score": record.get('score'),
        "layers": selected, "runtime": backend.runtime, "original_attention_backend": config.get('attention'),
        "replay_note": "Saved tokens and latent boundaries are forced; continuous hidden states are recomputed. "
                       "Same checkpoint and processor are required. SDPA prefill and one-query eager attention may differ "
                       "numerically from the original backend. Maps describe this reconstructed trajectory, not a bitwise original capture.",
        "step_definition": "Attention of the query produced by feeding the previous final hidden state as continuous input. "
                           "Boundary tokens are excluded. decode_step is zero-based; segment and latent_step are one-based.",
        "raw_attention_axes": "[selected_decoder_layers, merged_grid_height, merged_grid_width], mean over heads; no image-only renormalization",
        "policy_mismatches": [], "frames": [], "complete": False}
    try:
        with backend.torch.inference_mode():
            inputs = backend._inputs(messages)
            grids = inputs.get("image_grid_thw")
            grids = grids.tolist() if grids is not None else []
            if inputs['input_ids'].shape[1] != prediction['prompt_tokens'] or grids != prediction['image_grid_thw']:
                raise ValueError("Prompt length/image grid differs from saved evaluation; check dataset, processor and pixel limits")
            layout = image_layout(inputs['input_ids'][0].cpu().numpy(), grids, backend.model.config.image_token_id,
                                  backend.processor.image_processor.merge_size, image_paths)
            manifest["images"] = layout
            write_json(output / "manifest.json", manifest)
            state = backend._prefill(inputs)  # SDPA: do not materialize prompt x prompt attention.
            text_tokens = []
            for item in plan:
                actual = replay_policy_choice(backend, state, item, plan, backend.policy)
                if actual != item['token']:
                    manifest['policy_mismatches'].append({"decode_step": item['decode_step'], "saved": item['token'], "recomputed": actual})
                if item['token'] is None:
                    ordinal = len(manifest['frames']) + 1
                    relative = f"attention/latent_{ordinal:05d}.npz"
                    def capture(attentions):
                        maps = extract_image_attention(attentions, layout, selected)
                        np.savez_compressed(output / relative, **maps)
                    state = backend.advance(state, attention_callback=capture)
                    manifest['frames'].append({**item, "ordinal": ordinal, "raw_attention": relative,
                        "context": backend.tokenizer.decode(text_tokens[-160:], skip_special_tokens=True)})
                    print(f"latent {ordinal}: segment {item['segment']}, step {item['latent_step']}, decode {item['decode_step']}", flush=True)
                else:
                    text_tokens.append(item['token'])
                    state = backend.advance(state, item['token'])
            if prediction.get('stopped_on_eos'):
                actual = backend.next_token(state, {backend.policy.end_id} - {None})
                expected = prediction['token_ids'][-1]
                if actual != expected:
                    manifest['policy_mismatches'].append({"decode_step": len(plan), "saved": expected, "recomputed": actual})
        manifest['complete'] = True
        manifest['no_latent_steps'] = not manifest['frames']
        render_outputs(output, manifest, alpha=alpha, max_side=max_side)
    except Exception as exc:
        manifest['complete'] = False
        manifest['error'] = f"{type(exc).__name__}: {exc}"
        write_json(output / "manifest.json", manifest)
        raise
    print(f"Saved {len(manifest['frames'])} latent steps: {output / 'index.html'}", flush=True)
    return output
