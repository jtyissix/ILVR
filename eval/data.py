"""Normalize official release formats without ever giving gold answers to inference."""
import ast
import base64
import csv
import io
import json
import re
import shutil
import zipfile
from pathlib import Path

from .common import canonical_json, read_json, read_jsonl, safe_path


def extract_archive(path, destination):
    """ZIP only; reject traversal, symlinks and Windows drive/ADS names."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            if ":" in name or name.startswith("/") or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Unsafe archive entry: {name}")
            target = safe_path(destination, name)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            with archive.open(info) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            tmp.replace(target)


def parquet_rows(root, pattern):
    import pyarrow.parquet as pq
    paths = sorted(Path(root).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {root}/{pattern}")
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
            for row in batch.to_pylist():
                yield row, path.parent.name


class ImageStore:
    def __init__(self, raw, prepared):
        self.raw, self.prepared = Path(raw), Path(prepared)
        self.by_name = None
        self.by_size = {}
        self.image_count = 0

    def locate(self, relative):
        relative = str(relative).replace("\\", "/")
        candidate = safe_path(self.raw, relative)
        if candidate.is_file():
            return candidate
        candidate = safe_path(self.raw / "extracted", relative)
        if candidate.is_file():
            return candidate
        # Archives sometimes wrap all files in a named top-level directory.
        if self.by_name is None:
            self.by_name = {}
            for path in self.raw.rglob("*"):
                if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                    self.by_name.setdefault(path.name, []).append(path)
        candidates = self.by_name.get(Path(relative).name, [])
        if len(candidates) == 1:
            return candidates[0]
        if not Path(relative).suffix:
            candidates = [p for values in self.by_name.values() for p in values if p.stem == Path(relative).name]
            if len(candidates) == 1:
                return candidates[0]
        raise FileNotFoundError(f"Missing or ambiguous image: {relative} under {self.raw}")

    def save(self, value):
        from PIL import Image, ImageOps
        if value is None:
            return None
        if isinstance(value, dict):
            value = value.get("bytes") if value.get("bytes") is not None else self.locate(value["path"])
        elif isinstance(value, str):
            if value.startswith("data:") or value.startswith(("iVBOR", "/9j/", "UklGR", "R0lGOD", "Qk")):
                value = base64.b64decode(value.split(",", 1)[-1] if value.startswith("data:") else value, validate=True)
            else:
                value = self.locate(value)
        if isinstance(value, (str, Path)):
            value = Path(value).read_bytes()
        if not isinstance(value, (bytes, bytearray)):
            raise TypeError(f"Unsupported official image representation: {type(value)}")
        with Image.open(io.BytesIO(value)) as image:
            image = ImageOps.exif_transpose(image)
            # Match qwen_vl_utils alpha compositing for transparent diagrams.
            if image.mode in ("RGBA", "LA", "P"):
                rgba = image.convert("RGBA")
                rgb = Image.new("RGB", rgba.size, "white")
                rgb.paste(rgba, mask=rgba.getchannel("A"))
                image = rgb
            else:
                image = image.convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, "PNG")
        content = buffer.getvalue()
        # Compare bytes directly within equal-size candidates; retain HRBench deduplication.
        candidates = self.by_size.setdefault(len(content), [])
        for relative in candidates:
            if (self.prepared / relative).read_bytes() == content:
                return relative
        self.image_count += 1
        relative = f"images/{self.image_count:08d}.png"
        target = self.prepared / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".png.part")
        temporary.write_bytes(content)
        temporary.replace(target)
        candidates.append(relative)
        return relative


def choice_map(value):
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple, dict)):
                return choice_map(parsed)
        except (ValueError, SyntaxError):
            pass
        # MMVP '(a) ... (b) ...', MMStar 'A: ..., B: ...'.
        matches = list(re.finditer(r"(?:\(([A-Za-z])\)|\b([A-Z])\s*[:.)])\s*", value))
        if not matches:
            raise ValueError(f"Unrecognized option format: {value[:150]}")
        return {(m[1] or m[2]).upper(): value[m.end():matches[i+1].start() if i+1 < len(matches) else len(value)].strip().rstrip(",").strip()
                for i, m in enumerate(matches)}
    if isinstance(value, dict):
        return {str(k).strip("(). ").upper(): str(v) for k, v in value.items()}
    return {chr(65 + i): str(text) for i, text in enumerate(value)}


def answer_letter(value, choices, zero_based=False):
    if zero_based:
        return chr(65 + int(value))
    value = str(value).strip()
    letter = value.strip("(). ").upper()
    if letter in choices:
        return letter
    matches = [k for k, v in choices.items() if v.strip().lower() == value.lower()]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Gold answer cannot be mapped unambiguously: {value!r}")


def record(name, uid, question, answer, images, choices=None, category="all", **meta):
    choices = choices or {}
    return {"id": str(uid), "benchmark": name, "question": str(question), "answer": answer,
            "kind": "mcq" if choices else "open", "choices": choices, "images": images,
            "category": str(category), "meta": meta, "group": None}


def normalize(name, raw, prepared, emma_variant="full", scienceqa_full=False):
    raw, prepared = Path(raw), Path(prepared)
    store = ImageStore(raw, prepared)
    if name in ("mm_math", "visulogic", "seedbench"):
        for path in sorted(raw.rglob("*.zip")):
            extract_archive(path, raw / "extracted")
    out = []
    if name == "mmvp":
        with (raw / "Questions.csv").open(encoding="utf-8-sig", newline="") as f:
            for i, row in enumerate(csv.DictReader(f), 1):
                uid = int(row.get("Index", i))
                choices = choice_map(row["Options"])
                sample = record(name, uid, row["Question"], answer_letter(row["Correct Answer"], choices),
                                {"image_1": store.save(f"MMVP Images/{uid}.jpg")}, choices,
                                original_options=row["Options"])
                sample["group"] = str((uid - 1) // 2)
                out.append(sample)
    elif name == "vstar":
        for category in ("direct_attributes", "relative_position"):
            for path in sorted((raw / category).glob("*.json")):
                row = read_json(path)
                # Official files place the correct answer first in the options array.
                choices = choice_map(row["options"])
                image = next((p for p in path.parent.glob(path.stem + ".*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")), None)
                if image is None:
                    raise FileNotFoundError(f"No image paired with {path}")
                # Official vstar_bench uses 'answer' for the free-form reference;
                # multiple-choice ground truth is options[0].
                out.append(record(name, f"{category}/{path.stem}", row["question"], "A",
                                  {"image_1": store.save(image)}, choices, category))
    elif name == "mm_math":
        for i, row in enumerate(read_jsonl(raw / "MM_Math/MM_Math.jsonl")):
            image_names = row["file_name"]
            if isinstance(image_names, str):
                image_names = [image_names]
            images = {f"image_{j+1}": store.save(p) for j, p in enumerate(image_names)}
            out.append(record(name, row.get("id", i), row["question"], row["solution"], images,
                              category=row.get("difficult", "all"), year=row.get("year"),
                              knowledge=row.get("knowledge"), difficult=row.get("difficult")))
    elif name == "visulogic":
        for row in read_jsonl(raw / "data.jsonl"):
            out.append(record(name, row["id"], row["question"], row["label"].upper(),
                              {"image_1": store.save(row["image_path"])}, dict(zip("ABCD", "ABCD")), row["tag"],
                              embedded_options=True))
    elif name == "seedbench":
        for row in read_json(raw / "SEED-Bench.json")["questions"]:
            if row["data_type"] != "image":
                continue
            choices = {k: row[f"choice_{k.lower()}"] for k in "ABCD"}
            out.append(record(name, row["question_id"], row["question"], row["answer"].upper(),
                              {"image_1": store.save(row["data_id"])}, choices, row["question_type_id"]))
    else:
        patterns = {"mmstar": "mmstar.parquet", "mathvista": "data/testmini-*.parquet",
                    "scienceqa": "data/test-*.parquet", "emma": "*/test-*.parquet",
                    "blink": "*/val-*.parquet", "visualpuzzles": "data/train-*.parquet"}
        pattern = f"hr_bench_{name[-2:]}.parquet" if name.startswith("hrbench_") else patterns[name]
        image_refs, pending = {}, []
        for i, (row, subset) in enumerate(parquet_rows(raw, pattern)):
            if name == "scienceqa" and row.get("image") is None and not scienceqa_full:
                continue
            uid = row.get("pid", row.get("idx", row.get("id", row.get("index", i))))
            images = {}
            for key in ([f"image_{n}" for n in range(1, 6)] if name in ("emma", "blink") else ["decoded_image" if name == "mathvista" and "decoded_image" in row else "image"]):
                value = row.get(key)
                if value is not None:
                    if name.startswith("hrbench_") and isinstance(value, str) and len(value) < 64 and value.isdigit():
                        pending.append((images, "image_1", value))
                    else:
                        images[key if key.startswith("image_") else "image_1"] = store.save(value)
            if name.startswith("hrbench_"):
                if "image_1" in images:
                    image_refs[str(uid)] = images["image_1"]
                choices = {k: row[k] for k in "ABCD"}
                sample = record(name, uid, row["question"], row["answer"].upper(), images, choices,
                                row["category"], cycle=str(row["cycle_category"]))
            elif name == "mmstar":
                choices = {k: row[k] for k in "ABCD" if k in row}
                if not choices:
                    choices = choice_map(row["question"].split("Options:")[-1])
                sample = record(name, uid, row["question"], row["answer"].upper(), images, choices,
                                row["category"], embedded_options=True, l2_category=row.get("l2_category"))
            elif name == "mathvista":
                # Keep the original answer TYPE/content: official normalization compares content, not letters.
                sample = record(name, uid, row["question"], row["answer"], images,
                                category=row.get("metadata", {}).get("task", "all"),
                                query=row["query"], options=row["choices"], question_type=row["question_type"],
                                answer_type=row["answer_type"], precision=row["precision"], metadata=row.get("metadata", {}))
            elif name == "scienceqa":
                choices = choice_map(row["choices"])
                sample = record(name, uid, row["question"], answer_letter(row["answer"], choices, True), images,
                                choices, row["subject"], hint=row.get("hint", ""), grade=row.get("grade"),
                                full=scienceqa_full, has_image=bool(images))
            elif name == "emma":
                choices = choice_map(row["options"]) if row["type"].lower() == "multiple choice" else {}
                answer = answer_letter(row["answer"], choices) if choices else str(row["answer"])
                sample = record(name, uid, row["question"], answer, images, choices, row["subject"],
                                context=row.get("context") or "", type=row["type"], task=row.get("task"), variant=emma_variant)
            elif name == "blink":
                choices = choice_map(row["choices"])
                sample = record(name, f"{subset}/{uid}", row["question"], answer_letter(row["answer"], choices),
                                images, choices, subset, prompt=row.get("prompt"))
            elif name == "visualpuzzles":
                choices = choice_map(row["options"]) if row["options"] is not None else dict(zip("ABCD", "ABCD"))
                sample = record(name, uid, row["question"], answer_letter(row["answer"], choices), images,
                                choices, row["category"], difficulty=row["difficulty"], image_options=row["options"] is None)
            out.append(sample)
        for images, field, reference in pending:
            if reference not in image_refs:
                raise ValueError(f"Missing HRBench image reference {reference}")
            images[field] = image_refs[reference]
        if name.startswith("hrbench_"):
            for sample in out:
                sample["group"] = canonical_json([sample["question"], sorted(sample["choices"].values()), sample["images"]])
    return out
