"""Dependency-light contracts shared by downloading, inference and scoring."""
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

SCHEMA_VERSION = 2
BENCHMARKS = ("mmvp", "mmstar", "vstar", "hrbench_4k", "hrbench_8k",
              "mathvista", "mm_math", "scienceqa", "emma", "blink",
              "visualpuzzles", "visulogic", "seedbench")
ALIASES = {"hrbench": ("hrbench_4k", "hrbench_8k"), "v*": ("vstar",),
           "vstar_bench": ("vstar",), "math-vista": ("mathvista",),
           "mm-math": ("mm_math",), "visualpuzzle": ("visualpuzzles",),
           "seed-bench": ("seedbench",)}


def select_benchmarks(only, exclude):
    """Exactly one nonempty selection is required; ['all'] explicitly selects all."""
    if isinstance(only, str) or isinstance(exclude, str):
        raise ValueError("ONLY_DATASETS / EXCLUDE_DATASETS must be lists, not strings")
    if bool(only) == bool(exclude):
        raise ValueError("ONLY_DATASETS 和 EXCLUDE_DATASETS 必须恰好一个非空；全测请设置 ONLY_DATASETS=['all']")
    def expand(values):
        found = set()
        for value in values:
            key = value.strip().lower()
            names = BENCHMARKS if key == "all" else ALIASES.get(key, (key,))
            for name in names:
                if name not in BENCHMARKS:
                    raise ValueError(f"Unknown dataset {value!r}; available: {BENCHMARKS}")
                found.add(name)
        return found
    selected = expand(only) if only else set(BENCHMARKS) - expand(exclude)
    if not selected:
        raise ValueError("No datasets selected")
    return [name for name in BENCHMARKS if name in selected]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def record_path(directory, identity):
    """Allocate persistent numbered filenames by exact identity comparison.

    SQLite's ordered text index avoids scanning every request on each lookup.
    The index is part of the result/cache directory and must be kept with it.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(directory / "index.sqlite3", timeout=60)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS records (id INTEGER PRIMARY KEY, identity TEXT UNIQUE NOT NULL)")
        key = canonical_json(identity)
        db.execute("INSERT OR IGNORE INTO records(identity) VALUES (?)", (key,))
        number = db.execute("SELECT id FROM records WHERE identity = ?", (key,)).fetchone()[0]
    return directory / f"{number:08d}.json"


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes dataset directory: {relative}")
    return path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        for line, text in enumerate(f, 1):
            if text.strip():
                try:
                    yield json.loads(text)
                except ValueError as exc:
                    raise ValueError(f"Invalid JSON: {path}:{line}") from exc


def validate_samples(samples, benchmark, root, check_images=True):
    if not samples:
        raise ValueError(f"Empty dataset: {benchmark}")
    seen, groups = set(), {}
    for sample in samples:
        uid = sample["id"]
        if not isinstance(uid, str) or uid in seen:
            raise ValueError(f"Duplicate/invalid sample id: {uid}")
        seen.add(uid)
        if sample["benchmark"] != benchmark or not sample["question"].strip():
            raise ValueError(f"Invalid benchmark/question: {uid}")
        if sample.get("answer") is None or str(sample["answer"]).strip() == "":
            raise ValueError(f"Missing ground truth (hidden test is not locally scoreable): {uid}")
        choices = sample.get("choices", [])
        if sample["kind"] == "mcq" and (len(choices) < 2 or sample["answer"] not in choices):
            raise ValueError(f"Invalid MCQ choices/answer: {uid}")
        for image in sample["images"].values():
            path = safe_path(root, image)
            if check_images and not path.is_file():
                raise FileNotFoundError(path)
        if not sample["images"] and not (benchmark == "scienceqa" and sample["meta"].get("full")):
            raise ValueError(f"Visual sample has no image: {uid}")
        if sample.get("group") is not None:
            groups.setdefault(sample["group"], []).append(sample)
    expected = 2 if benchmark == "mmvp" else 4 if benchmark.startswith("hrbench_") else None
    if expected:
        if not groups or any(len(rows) != expected for rows in groups.values()):
            raise ValueError(f"{benchmark}: incomplete pair/circular group, expected {expected} rows each")
        if benchmark.startswith("hrbench_"):
            for group, rows in groups.items():
                if len({s['meta']['cycle'] for s in rows}) != 4:
                    raise ValueError(f"HRBench circular group must contain four distinct cycles: {group}")
                if len({s['choices'][s['answer']] for s in rows}) != 1:
                    raise ValueError(f"HRBench circular group has inconsistent gold content: {group}")


def subset_samples(samples, limit):
    """Diagnostic limit never cuts a MMVP pair or HRBench circular group in half."""
    if limit is None:
        return samples
    if type(limit) is not int or limit <= 0:
        raise ValueError("MAX_SAMPLES must be None or a positive integer")
    selected, groups, used = [], {}, set()
    for sample in samples:
        groups.setdefault(sample.get("group") or sample["id"], []).append(sample)
    for sample in samples:
        key = sample.get("group") or sample["id"]
        if key in used:
            continue
        if len(selected) + len(groups[key]) > limit:
            break
        selected.extend(groups[key])
        used.add(key)
    if not selected:
        raise ValueError("MAX_SAMPLES is smaller than one complete scoring group")
    return selected
