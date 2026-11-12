"""Benchmark prompts, answer extraction and aggregation; gold never enters prompts."""
import random
import re
from collections import defaultdict
from statistics import mean

from .common import canonical_json, safe_path
from .official import constant, mathvista_functions, selected_source

DIRECT = "Answer with the option's letter from the given choices directly."


def options_text(sample, style="paren"):
    return "\n".join((f"({k}) {v}" if style == "paren" else f"{k}. {v}")
                     for k, v in sample["choices"].items())


def prompt_for(sample, likelihood=False):
    name, q, meta = sample["benchmark"], sample["question"], sample["meta"]
    if likelihood:
        return f"Question: {q}\nAnswer:" if name == "seedbench" else q
    if name == "mmvp":
        return q + " " + meta["original_options"]
    if name == "mmstar":
        return q + "\n" + DIRECT
    if name.startswith("hrbench_"):
        return f"Question: {q}\nOptions:\n{options_text(sample, 'dot')}\nPlease select the correct answer from the options above. \n"
    if name == "mathvista":
        return meta["query"]
    if name == "mm_math":
        return q + "\n请逐步解答，并将最终答案放在 \\boxed{} 中。"
    if name == "scienceqa":
        context = meta.get("hint") or "N/A"
        opts = " ".join(f"({k}) {v}" for k, v in sample["choices"].items())
        return f"Question: {q}\nContext: {context}\nOptions: {opts}\nAnswer:".replace("  ", " ").strip()
    if name == "emma":
        if sample["choices"]:
            opts = "".join(f"{k}: {v}\n" for k, v in sample["choices"].items())
            text = f"{meta['context']}\n\n{q}\n\n{opts}\n\nAnswer with the option's letter from the given choices and put the letter in one \"\\boxed{{}}\". "
        else:
            text = f"{meta['context']}\n\n {q}\nAnswer the question using a single word or phrase and put the answer in one \"\\boxed{{}}\". "
        return text + "Please solve the problem step by step."
    if name == "blink":
        if not meta.get("prompt"):
            raise ValueError("BLINK official prompt missing")
        prefix = (constant("blink_test.py", "disclaimer")
                  if sample["category"] in ("Forensic_Detection", "Jigsaw", "Art_Style") else "")
        return prefix + meta["prompt"]
    if name == "visualpuzzles":
        opts = "Options: Choose from (A) (B) (C) (D) in the image." if meta["image_options"] else "Options:\n" + options_text(sample)
        return "Question: " + q.strip() + "\n" + opts + "\n" + constant("visualpuzzles_utils.py", "COT_PROMPT")
    if name == "visulogic":
        return q + "\n" + constant("visulogic_prompts.py", "COT_PROMPT")
    return q + "\n" + options_text(sample) + "\n" + DIRECT


def messages_for(sample, root, likelihood=False):
    text = prompt_for(sample, likelihood)
    images = sample["images"]
    # EMMA uses inline <image_n> references; preserve repetition and exact order.
    pattern = re.compile(r"<image[_ ](\d+)>")
    content, used = [], set()
    if pattern.search(text):
        start = 0
        for match in pattern.finditer(text):
            if match.start() > start:
                content.append({"type": "text", "text": text[start:match.start()]})
            key = "image_" + match[1]
            if key not in images:
                raise ValueError(f"Unresolved image placeholder {match[0]} in {sample['id']}")
            content.append({"type": "image", "image": str(safe_path(root, images[key]))})
            used.add(key)
            start = match.end()
        if start < len(text):
            content.append({"type": "text", "text": text[start:]})
    else:
        content = [{"type": "text", "text": text}]
    # Keep all supplied question images, even when a prompt does not refer to each one.
    remaining = [{"type": "image", "image": str(safe_path(root, images[k]))}
                 for k in sorted(images, key=lambda k: int(k.split("_")[-1])) if k not in used]
    content = (content + remaining) if sample["benchmark"] == "blink" else (remaining + content)
    return [{"role": "user", "content": content}]


def extract_letter(text, choices):
    """Conservative generic parser. Never pick a random option on failure."""
    valid = set(choices)
    text = text.strip()
    for pattern in (r"\\boxed\{\s*\(?([A-Za-z])\)?\s*\}",
                    r"(?:final\s+answer|answer|答案|选项)\s*(?:is|为|是)?\s*[:：]?\s*\(?([A-Za-z])\)?(?:\b|[.。])",
                    r"^\s*\(?([A-Za-z])\)?\s*[.。]?\s*$"):
        hits = [x.upper() for x in re.findall(pattern, text, re.I) if x.upper() in valid]
        if hits:
            return hits[-1]
    # Only an unambiguous exact option text, not a substring in reasoning.
    hits = [k for k, v in choices.items() if text.casefold().strip(". ") == str(v).casefold().strip(". ")]
    return hits[0] if len(hits) == 1 else None


def score_one(sample, prediction, judge, *, seed=42, mm_math_precision=1e-2, rules_only=False):
    name, gold = sample["benchmark"], sample["answer"]
    if prediction.get("method") == "likelihood":
        extracted = prediction["selected"]
        return {"score": int(extracted == gold), "extracted": extracted, "method": "candidate_mean_nll"}
    text, choices, meta = prediction["text"], sample["choices"], sample["meta"]
    parsed = extract_letter(text, choices) if choices else None
    method = "rule"
    if name == "mmstar" and not rules_only:
        # Original MMStar evaluator deliberately inspects the answer prefix.
        value = text.lower().strip().replace("\n", " ")
        answer = str(gold).lower().strip()
        correct = (value[:1] == answer or value[:2] == "(" + answer or
                   value[:8] == "option " + answer or value[:15] == "the answer is " + answer)
        return {"score": int(correct), "method": "official_mmstar_prefix"}
    if name == "scienceqa" and not rules_only:
        # Original ScienceQA run_gpt3.py: exactly one 'The answer is X.';
        # invalid predictions receive a random choice. A per-ID seed makes this
        # fallback reproducible across resumes and different evaluation orders.
        hits = re.findall(r"The answer is ([A-Z]).", text)
        extracted = hits[0] if len(hits) == 1 and hits[0] in choices else None
        fallback = extracted is None
        if fallback:
            extracted = random.Random(int.from_bytes(canonical_json([seed, sample["id"]]).encode("utf-8"), "big")).choice(list(choices))
        return {"score": int(extracted == gold), "extracted": extracted,
                "method": "official_scienceqa", "random_fallback": fallback}
    if name == "mmvp" and not rules_only:
        result = judge.ask(name,
            f"Given the following question {prompt_for(sample)}, the correct answer is ({gold.lower()}). Does the following answer correctly answers the question, answer:{text}?",
            system="You are a helpful and precise assistant for checking the quality of the answer. Please answer in only yes or no.",
            temperature=0.2, max_tokens=128, expected_pattern=r"(?:yes|no)\.?" ).strip()
        if result.lower().strip(". ") not in ("yes", "no"):
            raise ValueError("MMVP judge returned neither yes nor no")
        return {"score": int(result.lower().strip(". ") == "yes"), "judge": result, "method": "official_mmvp_judge"}
    if name == "mm_math":
        from .vendor.mm_math_eval import AutoScoringJudge
        correct = AutoScoringJudge().judge(str(gold), text, precision=mm_math_precision)
        return {"score": int(correct), "method": "official_symbolic", "precision": mm_math_precision}
    if name == "mathvista":
        if rules_only:
            raise ValueError("MathVista requires official extraction; RULES_ONLY is not supported for this benchmark")
        extracted = text.strip()
        fast = extracted == "" or (meta["question_type"] == "multi_choice" and extracted in meta["options"])
        if not fast and meta["answer_type"] in ("integer", "float"):
            try:
                extracted = str(int(extracted)) if meta["answer_type"] == "integer" else str(float(extracted))
                fast = True
            except ValueError:
                pass
        if not fast:
            demo = constant("mathvista_prompt.py", "demo_prompt")
            extracted = judge.ask(name, f"{demo.strip()}\n\n{meta['query']}\n\n{text}\n\nExtracted answer: ")
        funcs = mathvista_functions()
        normalized = funcs["normalize_extracted_answer"](extracted, meta["options"], meta["question_type"],
                                                        meta["answer_type"], meta["precision"])
        return {"score": int(funcs["safe_equal"](normalized, gold)), "extracted": extracted,
                "normalized": normalized, "method": "official_mathvista"}
    if name == "visualpuzzles":
        rng = random.Random(int.from_bytes(canonical_json([seed, sample["id"]]).encode("utf-8"), "big"))
        class TrackedRandom:
            used = False
            def choice(self, values):
                self.used = True
                return rng.choice(values)
        tracked = TrackedRandom()
        parser = selected_source("visualpuzzles_utils.py", ["parse_response"], {"re": re, "random": tracked})["parse_response"]
        parsed = parser(text, list(choices), None if meta["image_options"] else choices).upper()
        return {"score": int(parsed == gold), "extracted": parsed, "method": "official_visualpuzzles",
                "random_fallback": tracked.used}
    if name.startswith("hrbench_") and not rules_only:
        prompt = f"{sample['question']}\nOptions:\n{options_text(sample, 'dot')}\nPlease select the correct answer from the options above. \n"
        content = (f"Given the following question: {prompt}, the correct answer is: {gold}. {choices[gold]}. "
                   f"Does the following answer correctly answers the question, answer:{text}?")
        result = judge.ask(name, "You are a helpful and precise assistant for checking the quality of the answer. Please answer in only yes or no. " + content,
                           temperature=0.2, max_tokens=128, expected_pattern=r"(?:yes|no)\b.*").strip()
        if not re.match(r"^(yes|no)\b", result, re.I):
            raise ValueError("HRBench judge returned neither yes nor no")
        return {"score": int(result.lower().startswith("yes")), "judge": result, "method": "official_hrbench_prompt"}
    if name == "emma" and not rules_only:
        demo = constant("emma_utils.py", "score_demo_prompt")
        result = judge.ask(name, f"{demo.strip()}\nResponse: {text}\nAnswer: {gold}\nCorrect_or_not:",
                           expected_pattern=r"(?:correct|incorrect)\.?").strip()
        if result.lower().strip(". ") not in ("correct", "incorrect"):
            raise ValueError("EMMA judge returned neither Correct nor Incorrect")
        return {"score": int(result.lower().strip(". ") == "correct"), "judge": result, "method": "official_emma_judge"}
    if name == "blink" and not rules_only:
        # Match the original analyze_answer ordering, including last-paragraph handling.
        all_choices = [f"({k})" for k in choices]
        intersect = set(all_choices).intersection(text.split())
        last = set(all_choices).intersection(text.split("\n\n")[-1].split())
        if text in list(choices) + all_choices:
            parsed = text.strip("()")
        elif not ((len(intersect) != 1 and len(last) != 1) or len(intersect) < 1):
            parsed = next(iter(last if len(last) == 1 else intersect)).strip("()")
        else:
            builder = selected_source("blink_mcq.py", ["build_prompt"])["build_prompt"]
            result = judge.ask(name, builder(sample["question"] + "\nSelect from the following choices", options_text(sample), text),
                               expected_pattern=r"\(?[A-EZ]\)?\.?")
            parsed = extract_letter(result, {**choices, "Z": "Failed"})
            if parsed is None:
                raise ValueError("BLINK judge returned no valid option/Z")
            method = "official_blink_match"
    if name == "visulogic" and not rules_only:
        cls = selected_source("visulogic_eval.py", ["ModelEvaluator"],
                              {"re": re, "Option_list": list("ABCD")})["ModelEvaluator"]
        extractor = cls.__new__(cls)  # Only pure extraction methods; no model/client constructor.
        boxed = extractor.extract_last_boxed_content(text).strip()
        language = extractor.extract_lang_content(text)
        parsed = boxed if boxed in choices else language if language in choices else None
    if name == "visulogic" and parsed is None and not rules_only:
        result = judge.ask(name, "Extract the option in the solution. Just answer only capital letters. If solution have multi options or has no clear option just answer None.\nSolution: " + text + "\n",
                           system="You are a helpful and precise assistant for extract the final option in the answer.",
                           expected_pattern=r"(?:[ABCD]|None)\.?")
        if result.strip().lower().strip(". ") == "none":
            parsed = None
        else:
            parsed = extract_letter(result, choices)
            if parsed is None:
                raise ValueError("VisuLogic judge returned no valid option/None")
        method = "official_visulogic_extract"
    if name == "emma" and rules_only and not choices:
        parsed = text.strip()
    return {"score": int(parsed == gold), "extracted": parsed, "method": method,
            "unparsed": parsed is None, "nonofficial_rules_only": rules_only}


def aggregate(samples, scores):
    """Do not report an incomplete subset as a complete official benchmark score."""
    name = samples[0]["benchmark"]
    valid = [(s, scores[s["id"]]) for s in samples if s["id"] in scores and scores[s["id"]].get("score") is not None]
    category, groups, cycles = defaultdict(list), defaultdict(list), defaultdict(list)
    for sample, result in valid:
        value = result["score"]
        category[sample["category"]].append(value)
        groups[sample.get("group") or sample["id"]].append(value)
        cycles[sample["meta"].get("cycle", "all")].append(value)
    by_category = {k: mean(v) for k, v in sorted(category.items())}
    primary = mean([r["score"] for _, r in valid]) if valid else None
    metric = "accuracy"
    if name == "mmvp":
        pairs = [int(all(v)) for v in groups.values() if len(v) == 2]
        primary, metric = (mean(pairs) if pairs else None), "pair_accuracy"
    elif name == "blink":
        primary, metric = (mean(by_category.values()) if by_category else None), "macro_category_accuracy"
    elif name.startswith("hrbench_"):
        primary, metric = (mean([mean(v) for v in cycles.values()]) if cycles else None), "mean_cycle_accuracy"
        by_category = {k: mean([mean([r['score'] for s, r in valid if s['category'] == k and s['meta']['cycle'] == c])
                               for c in cycles if any(s['category'] == k and s['meta']['cycle'] == c for s, _ in valid)])
                       for k in category}
    complete = len(valid) == len(samples)
    return {"benchmark": name, "metric": metric, "score": primary if complete else None,
            "partial_score": primary if not complete else None, "complete": complete,
            "expected": len(samples), "scored": len(valid), "by_category": by_category,
            "sample_accuracy": mean([r["score"] for _, r in valid]) if valid else None,
            "random_fallback_count": sum(bool(r.get("random_fallback")) for _, r in valid),
            "unparsed_count": sum(bool(r.get("unparsed")) for _, r in valid)}
