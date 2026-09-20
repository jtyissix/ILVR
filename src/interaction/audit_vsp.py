"""Rescore saved VSP replies on CPU; keep official and diagnostic scores separate."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from . import vsp


MATHRULER_SOURCE_SHA256 = "dbc8a73cf48e3a449c52125218e1400d616186549bea36fe31b2fe0b495e3eff"


def official_actions(text):
    """Mirage test.py + MathRuler boxed extraction + task.py character filter.

    MathRuler uses the LAST literal \\boxed{ marker, supports nested braces, and
    returns 'None' for absent/incomplete boxes (which produces no UDLR actions).
    Source snapshot/hash and the distinction from ILVR's fallback are in the report.
    """
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return []
    start += len(marker)
    depth = 1
    for end in range(start, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return [c for c in text[start:end].upper() if c in vsp.ACTIONS]
    return []


def extended_actions(span, max_actions=4096):
    """Diagnostic grammar, not official scoring; never consults the map.

    Accept direction lists, 'move ... then ...', 4R1U, 3Rights, and Up 2 times.
    Reject unknown words, bare numbers, and incomplete syntax. Only the chosen
    final answer is parsed; do not mine the rationale for a route that succeeds.
    """
    text = re.sub(r"\\(?:text|mathrm|mathbf)\s*\{([^{}]*)\}", r"\1", span).upper()
    # Words must be complete, so that prose such as 'GIFT' cannot supply actions.
    direction = r"(?:UP|DOWN|LEFT|RIGHT)S?|[UDLR]"
    prefix = re.compile(rf"(\d+)\s*({direction})(?![A-Z])")
    suffix = re.compile(rf"({direction})\s+(\d+)\s+TIMES\b")
    word = re.compile(r"(UP|DOWN|LEFT|RIGHT)\b")
    letters = re.compile(r"[UDLR]+(?![A-Z])")
    separator = re.compile(r"(?:->|→|[\s,，;；\[\](){}'\"`$*.])+")
    connector = re.compile(r"(?:MOVE|THEN|AND)\b")
    actions, pos, needs_action = [], 0, False
    while pos < len(text):
        match = separator.match(text, pos)
        if match:
            pos = match.end()
            continue
        match = connector.match(text, pos)
        if match:
            pos = match.end()
            needs_action = True
            continue
        match = prefix.match(text, pos)
        if match:
            count, token = int(match[1]), match[2]
        else:
            match = suffix.match(text, pos)
            if match:
                token, count = match[1], int(match[2])
            else:
                match = word.match(text, pos) or letters.match(text, pos)
                if not match:
                    raise ValueError(f"Unrecognized action text near {text[pos:pos + 50]!r}")
                token, count = match[0], 1
        token = vsp.WORDS.get(token.removesuffix("S"), token)
        if count < 1 or len(actions) + len(token) * count > max_actions:
            raise ValueError("Action count must be positive and within the audit limit")
        actions.extend(token * count)
        pos, needs_action = match.end(), False
    if not actions or needs_action:
        raise ValueError("No complete action sequence")
    return actions


def assess(sample, raw):
    current = vsp.score(sample, raw)
    result = {"strict_v1": {"prediction": current["prediction"], **current["simulation"]}}
    actions = official_actions(raw)
    result["mirage_official"] = {"prediction": "".join(actions), **vsp.simulate(sample["map_desc"], actions)}
    try:
        actions = extended_actions(current["answer_span"])
    except ValueError as exc:
        result["extended_diagnostic"] = {"prediction": "", "success": False,
                                         "status": "invalid_answer", "invalid": True, "error": str(exc)}
    else:
        result["extended_diagnostic"] = {"prediction": "".join(actions),
                                         **vsp.simulate(sample["map_desc"], actions)}
    return current["answer_span"], result


def read_jsonl(path, payload):
    for line, raw in enumerate(payload.decode("utf-8").splitlines(), 1):
        if raw.strip():
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line}: incomplete/invalid JSON; copy a completed snapshot") from exc


def audit(test_data_path, prediction_paths, allow_partial=False):
    test_payload = Path(test_data_path).read_bytes()
    samples = list(read_jsonl(test_data_path, test_payload))
    if not samples:
        raise ValueError("Empty test dataset")
    for sample in samples:
        vsp.validate_map(sample["map_desc"])
    records, seen, latent_sizes, inputs = [], set(), set(), []
    for path in prediction_paths:
        path = Path(path)
        payload = path.read_bytes()
        inputs.append({"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()})
        for row in read_jsonl(path, payload):
            index = row["index"]
            if type(index) is not int or not 0 <= index < len(samples):
                raise ValueError(f"Invalid sample index: {index!r}")
            if index in seen:
                raise ValueError(f"Duplicate index {index}; do not mix runs or merged and rank files")
            seen.add(index)
            sample = samples[index]
            grid_size = f"{len(sample['map_desc'])}x{len(sample['map_desc'])}"
            if str(sample["map_id"]) != str(row["map_id"]) or row["grid_size"] != grid_size:
                raise ValueError(f"Sample identity mismatch at index {index}; check test dataset/order")
            latent_sizes.update(row.get("segment_steps", []))
            raw = row["raw_output"]
            span, scores = assess(sample, raw)
            records.append({"index": index, "map_id": row["map_id"], "grid_size": grid_size,
                            "source": str(path), "answer_span": span, "scores": scores,
                            "stored_correct": row.get("correct"),
                            "stored_prediction": row.get("prediction"),
                            "hit_token_limit": row.get("hit_token_limit", False),
                            "numeric_only": bool(re.fullmatch(r"\d+[.]?", span)),
                            "has_literal_box": r"\boxed{" in raw,
                            "deletion_vocabulary": bool(re.search(r"\bdelet(?:e|ed|ing|ion)\b", raw, re.I))})
    if len(latent_sizes) > 1:
        raise ValueError(f"Mixed latent sizes {sorted(latent_sizes)}; audit each run separately")
    if not records:
        raise ValueError("No predictions")
    complete = seen == set(range(len(samples)))
    if not complete and not allow_partial:
        raise ValueError(f"Only {len(records)}/{len(samples)} samples; pass --allow_partial for snapshots")
    records.sort(key=lambda r: r["index"])
    profiles = {}
    for name in ("strict_v1", "mirage_official", "extended_diagnostic"):
        correct = sum(r["scores"][name]["success"] for r in records)
        profiles[name] = {"correct": correct, "accuracy": correct / len(records),
                          "status_counts": dict(Counter(r["scores"][name]["status"] for r in records)),
                          "recovered_vs_strict": [r["index"] for r in records if
                              r["scores"][name]["success"] and not r["scores"]["strict_v1"]["success"]],
                          "lost_vs_strict": [r["index"] for r in records if
                              not r["scores"][name]["success"] and r["scores"]["strict_v1"]["success"]]}
    report = {"samples": len(records), "test_samples": len(samples), "complete": complete,
              "latent_sizes_observed": sorted(latent_sizes), "inputs": inputs,
              "test_sha256": hashlib.sha256(test_payload).hexdigest(),
              "mirage_revision": vsp.MIRAGE_REVISION,
              "mathruler_source_sha256": MATHRULER_SOURCE_SHA256,
              "profiles": profiles,
              "diagnostics": {key: sum(r[key] for r in records) for key in
                              ("numeric_only", "has_literal_box", "deletion_vocabulary", "hit_token_limit")},
              "stored_score_mismatch_indices": [r["index"] for r in records if
                  r["stored_correct"] != r["scores"]["strict_v1"]["success"] or
                  r["stored_prediction"] != r["scores"]["strict_v1"]["prediction"]],
              "note": "Offline scoring audit only; extended_diagnostic is not the official metric. "
                      "Partial snapshots and observed latent sizes do not establish model identity."}
    return report, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_data_path", required=True)
    parser.add_argument("--predictions", nargs="+", required=True, help="One run: merged JSONL OR its rank files")
    parser.add_argument("--output_dir", required=True, help="New audit directory; existing reports are not overwritten")
    parser.add_argument("--allow_partial", action="store_true")
    args = parser.parse_args()
    report, records = audit(args.test_data_path, args.predictions, args.allow_partial)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "rescored.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
