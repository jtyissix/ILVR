"""Load specific upstream functions without executing their CLI/network imports."""
import ast
import functools
import logging
import re
from pathlib import Path

VENDOR = Path(__file__).parent / "vendor"


def selected_source(filename, names, namespace=None):
    tree = ast.parse((VENDOR / filename).read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        keys = [node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else [
            target.id for target in getattr(node, "targets", []) if isinstance(target, ast.Name)]
        if any(key in names for key in keys):
            nodes.append(node)
    scope = dict(namespace or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(VENDOR / filename), "exec"), scope)
    if any(name not in scope for name in names):
        raise RuntimeError(f"Missing upstream symbols in {filename}: {names}")
    return scope


@functools.lru_cache(None)
def mathvista_functions():
    from Levenshtein import distance
    return selected_source("mathvista_score.py", ["NON_ANSWER_PATTERN", "is_non_answer",
                           "get_most_similar", "normalize_extracted_answer", "safe_equal"],
                           {"re": re, "distance": distance, "logging": logging})


@functools.lru_cache(None)
def constant(filename, name):
    return selected_source(filename, [name])[name]
