"""
Cached LLM-judge similarity scoring over an OpenAI-compatible chat-completions
API (OpenRouter by default).

A score in [0,1] is requested for a (reference, candidate) pair under a given
judge *instruction*. Scores are persisted to a JSON cache keyed by
(model, prompt_id, reference, candidate) -- the prompt_id keeps scores for
different judge tasks/levels (e.g. "intv:state_tracking" vs "notif:video_capture")
from colliding. Re-runs and threshold sweeps never re-spend API tokens.
"""

import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# generic fallback instruction (task scripts pass their own)
DEFAULT_INSTRUCTION = (
    "Rate the semantic similarity between the two short texts on a scale from "
    "0.0 to 1.0, where 1.0 means they describe the same thing and 0.0 means "
    "completely unrelated.")


def build_prompt(instruction, ref, cand):
    return (f"{instruction}\n\n"
            f'Reference (ground truth): "{ref}"\n'
            f'Candidate (model output): "{cand}"\n\n'
            "Reply with ONLY a single number between 0.0 and 1.0.")


class SimilarityScorer:
    def __init__(self, model, cache_path, api_key=None, workers=8,
                 retries=3, timeout=60):
        self.model = model
        self.cache_path = cache_path
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.workers = workers
        self.retries = retries
        self.timeout = timeout
        self._lock = threading.Lock()
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as fh:
                self.cache = json.load(fh)

    # ---- cache keys -------------------------------------------------------
    def _key(self, prompt_id, ref, cand):
        h = hashlib.sha256()
        for part in (self.model, prompt_id, ref, cand):
            h.update(part.encode())
            h.update(b"\x00")
        return h.hexdigest()

    def cached_score(self, prompt_id, ref, cand):
        return self.cache.get(self._key(prompt_id, ref, cand))

    def missing(self, pairs, prompt_id):
        """De-duplicated (ref, cand) pairs not yet cached for this prompt_id."""
        seen, out = set(), []
        for ref, cand in pairs:
            k = self._key(prompt_id, ref, cand)
            if k in self.cache or k in seen:
                continue
            seen.add(k)
            out.append((ref, cand))
        return out

    # ---- API --------------------------------------------------------------
    def _call_api(self, instruction, ref, cand):
        import requests
        payload = {
            "model": self.model,
            "messages": [{"role": "user",
                          "content": build_prompt(instruction, ref, cand)}],
            "temperature": 0,
        }
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        last = None
        for attempt in range(self.retries):
            try:
                r = requests.post(OPENROUTER_URL, headers=headers,
                                  json=payload, timeout=self.timeout)
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
                m = re.search(r"[-+]?\d*\.?\d+", content)
                if m:
                    return max(0.0, min(1.0, float(m.group())))
                last = f"unparseable: {content!r}"
            except Exception as e:  # noqa: BLE001 - network/parse, retry
                last = repr(e)
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"OpenRouter call failed after {self.retries} tries: {last}")

    # ---- scoring ----------------------------------------------------------
    def score_pairs(self, pairs, instruction, prompt_id, progress=None):
        """Ensure every (ref, cand) pair is scored under `instruction`
        (calling the API only for cache misses) and return {(ref, cand): score}."""
        todo = self.missing(pairs, prompt_id)
        if todo:
            if not self.api_key:
                raise SystemExit(
                    f"{len(todo)} pairs need scoring but OPENROUTER_API_KEY is "
                    "not set. Export it, or run with --dry-run to see the count.")
            done = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as ex:
                futs = {ex.submit(self._call_api, instruction, r, c): (r, c)
                        for r, c in todo}
                for fut in concurrent.futures.as_completed(futs):
                    ref, cand = futs[fut]
                    score = fut.result()
                    with self._lock:
                        self.cache[self._key(prompt_id, ref, cand)] = score
                    done += 1
                    if progress and done % 50 == 0:
                        progress(done, len(todo))
            self.save()

        return {(ref, cand): self.cache[self._key(prompt_id, ref, cand)]
                for ref, cand in pairs}

    def save(self):
        if not self.cache_path:
            return
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.cache, fh)
        os.replace(tmp, self.cache_path)
