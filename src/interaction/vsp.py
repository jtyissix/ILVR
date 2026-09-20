"""VSP spatial-planning data checks and deterministic path scoring (no CUDA)."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from .data import file_hash, resolve_image


MIRAGE_REVISION = "53f26de2682025e146781c5b198ec93bdfe4c4d6"
TEST_ARCHIVE_SHA256 = "0272164f29dbed8b191977f7fa71cff8964b701bb84af293bf7525ada8394956"
ILVR_PROMPT_REVISION = "13a67ed2ee975d6ca56b4e9eba18d68b352b166b"
PROMPT_STYLES = ("ilvr_eval", "mirage_marker")
ACTIONS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
WORDS = {"UP": "U", "DOWN": "D", "LEFT": "L", "RIGHT": "R"}


def validate_map(grid):
    if not isinstance(grid, list) or not grid:
        raise ValueError("map_desc must be a nonempty square grid")
    n = len(grid)
    cells = []
    for row in grid:
        if not isinstance(row, list) or len(row) != n:
            raise ValueError("map_desc must be a square, non-ragged grid")
        if any(type(value) is not int or value not in {-1, 0, 1, 2} for value in row):
            raise ValueError("map_desc cells must be integers: -1=hole, 0=safe, 1=start, 2=goal")
        cells.extend(row)
    if cells.count(1) != 1 or cells.count(2) != 1:
        raise ValueError("map_desc must contain exactly one start and one goal")
    return n


def image_paths(sample, image_root):
    values = sample.get("image_input")
    values = [values] if isinstance(values, str) else values
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError("VSP spatial planning requires exactly one initial-map image_input")
    value = values[0]
    if not isinstance(value, str) or not value:
        raise ValueError("image_input must contain a nonempty path")
    value = value.replace("\\", "/")
    if value.startswith("./"):
        value = value[2:]
    prefix = "data/vsp_spatial_planning/"
    if value.startswith(prefix):
        value = value[len(prefix):]
    return [str(resolve_image(image_root, value))]


def user_message(sample, paths, prompt_style="ilvr_eval"):
    """ILVR eval.py: image first, then UNMODIFIED text_input (including <image>).

    mirage_marker preserves the earlier evaluator's image-in-text layout for
    controlled comparisons. Neither mode exposes map_desc or helper images.
    """
    text = sample["text_input"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text_input must be a nonempty string")
    if len(paths) != 1 or text.count("<image>") > 1:
        raise ValueError("VSP expects one image and at most one <image> marker")
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(f"Unknown VSP prompt style: {prompt_style}")
    if prompt_style == "ilvr_eval":
        # Deliberately identical to official run_one_example's message content.
        # PIL images are passed separately to processor; <image> remains text.
        return {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}
    picture = {"type": "image", "image": paths[0]}
    if "<image>" in text:
        before, after = text.split("<image>")
        content = [{"type": "text", "text": before}, picture, {"type": "text", "text": after}]
    else:
        content = [picture, {"type": "text", "text": text}]
    return {"role": "user", "content": content}


def load_samples(path, image_root, expected_samples=None):
    """Read and validate all records before loading a model; never drop bad rows."""
    from PIL import Image

    records, identities = [], set()
    with Path(path).open(encoding="utf-8") as handle:
        for line, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                sample = json.loads(raw)
                n = validate_map(sample.get("map_desc"))
                if not isinstance(sample.get("map_id"), (int, str)) or not str(sample["map_id"]).strip():
                    raise ValueError("map_id must be a nonempty string or integer")
                identity = (n, str(sample["map_id"]))
                if identity in identities:
                    raise ValueError(f"Duplicate map_id within {n}x{n}: {identity[1]}")
                identities.add(identity)
                paths = image_paths(sample, image_root)
                user_message(sample, paths)
                with Image.open(paths[0]) as picture:
                    picture.verify()
                records.append(sample)
            except Exception as exc:
                raise ValueError(f"{path}: line {line}: {exc}") from exc
    if not records:
        raise ValueError("VSP test data is empty")
    counts = Counter(f"{len(s['map_desc'])}x{len(s['map_desc'])}" for s in records)
    if expected_samples is not None and len(records) != expected_samples:
        raise ValueError(f"Expected {expected_samples} test samples, found {len(records)}")
    if expected_samples == 400 and counts != Counter({f"{n}x{n}": 100 for n in range(3, 7)}):
        raise ValueError(f"Official VSP test split requires 100 maps per size 3..6; found {dict(counts)}")
    return records, {"samples": len(records), "by_grid_size": dict(sorted(counts.items())),
                     "source_sha256": file_hash(path), "image_root": str(Path(image_root).resolve())}


def answer_span(text):
    """Last boxed answer, then last final-answer line, then an action-only reply."""
    text = re.sub(r"<\|[^>]*\|>", "", text).strip()
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if matches:
        start = matches[-1].end()
        depth = 1
        for end in range(start, len(text)):
            depth += (text[end] == "{") - (text[end] == "}")
            if depth == 0:
                return text[start:end].strip()
        return ""  # A truncated boxed answer must not fall back to rationale text.
    matches = list(re.finditer(r"(?:final\s+answer\s*(?:is\s*)?[:：]?|answer\s*[:：])\s*([^\r\n]+)", text, re.I))
    return matches[-1].group(1).strip() if matches else text


def parse_actions(span):
    """Accept UDLR or direction words; reject extra prose/unknown actions."""
    span = re.sub(r"\\(?:text|mathrm|mathbf)\s*\{([^{}]*)\}", r"\1", span)
    span = span.upper().strip()
    if not span:
        raise ValueError("No complete action sequence")
    # All non-separator characters must belong to recognized action tokens.
    parts = re.split(r"[\s,，;；\[\](){}'\"`$*.]+|(?:->|→)", span)
    actions = []
    for part in filter(None, parts):
        if part in WORDS:
            actions.append(WORDS[part])
        elif re.fullmatch(r"[UDLR]+", part):
            actions.extend(part)
        else:
            raise ValueError(f"Unrecognized action text: {part!r}")
    if not actions:
        raise ValueError("Empty action sequence")
    return actions


def simulate(grid, actions):
    """Mirage rules: off-map moves do nothing; holes fail; final cell must be goal."""
    n = validate_map(grid)
    row, col = next((r, c) for r in range(n) for c in range(n) if grid[r][c] == 1)
    steps = 0
    for action in actions:
        if action not in ACTIONS:
            raise ValueError(f"Invalid action: {action}")
        steps += 1
        dr, dc = ACTIONS[action]
        nr, nc = row + dr, col + dc
        if not (0 <= nr < n and 0 <= nc < n):
            continue
        row, col = nr, nc
        if grid[row][col] == -1:
            return {"success": False, "status": "fell_into_hole", "final_position": [row, col],
                    "steps_executed": steps, "invalid": False}
    success = grid[row][col] == 2
    return {"success": success, "status": "reached_goal" if success else "did_not_reach_goal",
            "final_position": [row, col], "steps_executed": steps, "invalid": False}


def score(sample, generated_text):
    span = answer_span(generated_text)
    try:
        actions = parse_actions(span)
    except ValueError as exc:
        actions = []
        simulation = {"success": False, "status": "invalid_answer", "invalid": True, "error": str(exc)}
    else:
        simulation = simulate(sample["map_desc"], actions)
    n = len(sample["map_desc"])
    return {"map_id": sample["map_id"], "grid_size": f"{n}x{n}", "answer_span": span,
            "prediction": "".join(actions), "correct": simulation["success"], "simulation": simulation}


def summarize(records):
    counts = defaultdict(lambda: [0, 0])
    for record in records:
        counts[record["grid_size"]][0] += int(record["correct"])
        counts[record["grid_size"]][1] += 1
    total = len(records)
    correct = sum(r["correct"] for r in records)
    return {"task": "vsp_spatial_planning", "scoring": "Mirage map simulation; strict boxed/final/action-only answer parser v1",
            "samples": total, "correct": correct, "accuracy": correct/max(total, 1),
            "by_grid_size": {key: {"correct": v[0], "samples": v[1], "accuracy": v[0]/v[1]} for key, v in sorted(counts.items())},
            "macro_grid_accuracy": sum(v[0]/v[1] for v in counts.values())/max(len(counts), 1),
            "invalid_answer_count": sum(r["simulation"]["invalid"] for r in records),
            "status_counts": dict(Counter(r["simulation"]["status"] for r in records)),
            "token_limit_count": sum(r["hit_token_limit"] for r in records)}


def main():
    parser = argparse.ArgumentParser(description="Validate VSP spatial-planning TEST without loading a model")
    parser.add_argument("--test_data_path", required=True)
    parser.add_argument("--image_root", required=True, help="Directory containing imgs_test/")
    parser.add_argument("--expected_samples", type=int, default=400)
    args = parser.parse_args()
    _, report = load_samples(args.test_data_path, args.image_root, args.expected_samples)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
