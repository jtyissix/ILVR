"""OpenAI-compatible HTTP judge with exact-request cache and explicit failures."""
import time
import re
from pathlib import Path

from .common import record_path, read_json, write_json


class Judge:
    def __init__(self, api_key, base_url, model, cache_dir, *, models=None, timeout=60, retries=3):
        self.api_key, self.base_url, self.model = api_key, base_url.rstrip("/"), model
        self.models = models or {}
        self.cache_dir, self.timeout, self.retries = Path(cache_dir), timeout, retries

    def ask(self, benchmark, prompt, *, system=None, temperature=0, max_tokens=512, expected_pattern=None):
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.models.get(benchmark, self.model), "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens}
        identity = {"url": self.base_url, "request": payload}
        path = record_path(self.cache_dir, identity)
        def valid(result):
            return isinstance(result, str) and bool(result.strip()) and (
                expected_pattern is None or re.fullmatch(expected_pattern, result.strip(), re.I | re.S) is not None)
        if path.is_file():
            cached = read_json(path)
            if all(cached.get(key) == value for key, value in identity.items()) and valid(cached.get("response")):
                return cached["response"]
        if not self.api_key:
            raise RuntimeError(f"{benchmark}: scoring requires API_KEY (or cached judge response); use MODE='infer' first")
        import requests
        error = ""
        for attempt in range(self.retries):
            try:
                response = requests.post(self.base_url + "/chat/completions", json=payload,
                                         headers={"Authorization": "Bearer " + self.api_key}, timeout=self.timeout)
                # Do not persist exception repr: providers may echo authentication details.
                if response.status_code >= 400:
                    error = f"HTTP {response.status_code}"
                    if response.status_code in (400, 401, 403, 404):
                        break
                    response.raise_for_status()
                result = response.json()["choices"][0]["message"]["content"]
                if not valid(result):
                    raise ValueError("Empty or malformed judge response")
                write_json(path, {**identity, "response": result})
                return result
            except Exception as exc:
                error = error or type(exc).__name__
                if attempt + 1 < self.retries:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"{benchmark}: judge request failed ({error}); prediction retained, score pending")
