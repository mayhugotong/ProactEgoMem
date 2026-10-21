from __future__ import annotations

import gc
import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from vllm import LLM as VLLM, SamplingParams
    _VLLM_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on local installation
    VLLM = None  # type: ignore[assignment]
    SamplingParams = None  # type: ignore[assignment]
    _VLLM_IMPORT_ERROR = exc

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

from ..base import Capabilities, LLM, LLMRequest, LLMResponse
from ..utils.openai_schema import to_openai_chat_message
from proact.core.registry import MODEL_REGISTRY

logger = logging.getLogger(__name__)

PROVIDER_NAME = "vllm-local"
RESERVED_PARAM_KEYS = {
    "chat_template",
    "chat_template_path",
    "chat_template_kwargs",
    "chat_template_content_format",
    "model",
}


def _stable_kv(d: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted(d.items(), key=lambda kv: kv[0]))


def _merge_params(global_params: Dict[str, Any], req_params: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(global_params or {})
    merged.update(req_params or {})
    return merged


def _extract_chat_template_kwargs(
    config_like: Any,
    merged_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if merged_params:
        per_request_kwargs = merged_params.get("chat_template_kwargs")
        if isinstance(per_request_kwargs, dict):
            return per_request_kwargs

    extra = getattr(config_like, "extra", None)
    if extra is None and isinstance(config_like, dict):
        extra = config_like.get("extra")

    if not extra:
        return {}

    template_kwargs = extra.get("chat_template_kwargs")
    return dict(template_kwargs) if isinstance(template_kwargs, dict) else {}


def _extract_chat_template_content_format(
    config_like: Any,
    merged_params: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if merged_params:
        per_request_format = merged_params.get("chat_template_content_format")
        if per_request_format is not None:
            return str(per_request_format)

    extra = getattr(config_like, "extra", None)
    if extra is None and isinstance(config_like, dict):
        extra = config_like.get("extra")

    if not extra:
        return None

    configured_format = extra.get("chat_template_content_format")
    return str(configured_format) if configured_format is not None else None


def _extract_chat_template(
    config_like: Any,
    merged_params: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if merged_params:
        per_request_template = merged_params.get("chat_template")
        per_request_template_path = merged_params.get("chat_template_path")
        if per_request_template or per_request_template_path:
            return _load_chat_template(
                template=per_request_template,
                template_path=per_request_template_path,
                source="request params",
            )

    extra = getattr(config_like, "extra", None)
    if extra is None and isinstance(config_like, dict):
        extra = config_like.get("extra")

    if not extra:
        return None

    return _load_chat_template(
        template=extra.get("chat_template"),
        template_path=extra.get("chat_template_path"),
        source="LLM profile extra",
    )


def _load_chat_template(
    *,
    template: Any,
    template_path: Any,
    source: str,
) -> Optional[str]:
    has_template = isinstance(template, str) and bool(template.strip())
    has_template_path = isinstance(template_path, str) and bool(template_path.strip())
    if has_template and has_template_path:
        raise ValueError(
            f"Do not configure both chat_template and chat_template_path in {source}."
        )
    if has_template:
        return template
    if not has_template_path:
        return None

    path = Path(os.path.expandvars(template_path)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Configured chat_template_path from {source} does not exist or is not a file: {path}"
        )
    return path.read_text(encoding="utf-8")


def _tokenizer_has_chat_template(tokenizer: Any) -> bool:
    template = getattr(tokenizer, "chat_template", None)
    return isinstance(template, str) and bool(template.strip())


def _resolve_chat_template(*, tokenizer: Any, configured_template: Optional[str]) -> Optional[str]:
    if configured_template:
        return configured_template
    if _tokenizer_has_chat_template(tokenizer):
        return None
    raise ValueError(
        "vllm-local cannot format chat messages because the tokenizer does not define "
        "chat_template and no explicit chat_template was configured. Add "
        "`extra.chat_template_path` or `extra.chat_template` to the LLM profile in "
        "configs/llms.yaml, or pass `chat_template_path` / `chat_template` in request params."
    )


def _extract_sampling_kwargs(merged_params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in (merged_params or {}).items()
        if key not in RESERVED_PARAM_KEYS and value is not None
    }


def _guided_json_object_params():
    """Return ``(sampling_param_key, params_obj)`` constraining output to a JSON object.

    Feature-detected across vLLM versions: prefers the current structured-outputs
    API (``SamplingParams(structured_outputs=StructuredOutputsParams(...))``) and
    falls back to the legacy guided-decoding API
    (``SamplingParams(guided_decoding=GuidedDecodingParams(...))``). Raises a clear
    error only when neither is available, instead of silently ignoring json_mode."""
    # Current API (vLLM renamed guided_decoding -> structured_outputs).
    try:
        from vllm.sampling_params import StructuredOutputsParams
    except Exception:
        StructuredOutputsParams = None
    if StructuredOutputsParams is not None:
        for kwargs in ({"json_object": True}, {"json": {"type": "object"}}):
            try:
                return "structured_outputs", StructuredOutputsParams(**kwargs)
            except TypeError:
                continue
    # Legacy API.
    try:
        from vllm.sampling_params import GuidedDecodingParams
    except Exception as exc:  # pragma: no cover - depends on local installation
        raise RuntimeError(
            "json_mode was requested for the vllm-local provider but this vLLM "
            "installation exposes neither StructuredOutputsParams nor "
            "GuidedDecodingParams. Upgrade/downgrade vLLM or remove json_mode "
            "from the model request_params."
        ) from exc
    return "guided_decoding", GuidedDecodingParams(json_object=True)


def _make_sampling_params(sampling_kwargs: Dict[str, Any]):
    if SamplingParams is None:
        raise RuntimeError("vLLM is not available in this environment.")
    return SamplingParams(**sampling_kwargs)


def _build_llm_response(raw_output: Any, generation: Any) -> LLMResponse:
    text = getattr(generation, "text", "") or ""
    finish_reason = getattr(generation, "finish_reason", None)

    prompt_tokens = None
    if hasattr(raw_output, "prompt_token_ids"):
        prompt_tokens = len(getattr(raw_output, "prompt_token_ids", []) or [])

    completion_tokens = None
    if hasattr(generation, "token_ids"):
        completion_tokens = len(getattr(generation, "token_ids", []) or [])

    raw = raw_output.to_dict() if hasattr(raw_output, "to_dict") else {"outputs": [text]}

    return LLMResponse(
        text=text,
        raw=raw,
        finish_reason=finish_reason,
        usage_tokens_prompt=prompt_tokens,
        usage_tokens_completion=completion_tokens,
        ok=True,
        status_code=200,
    )


def _error_response(message: str, exc_type: str = "RuntimeError") -> LLMResponse:
    return LLMResponse(ok=False, error=message, error_type=exc_type, status_code=None)


class _Engine:
    def __init__(self, model: str, engine_kwargs: Dict[str, Any]):
        if VLLM is None:
            raise RuntimeError("vLLM is not available.") from _VLLM_IMPORT_ERROR

        kwargs = dict(engine_kwargs or {})
        kwargs["model"] = model

        self.vllm = VLLM(**kwargs)
        self.has_chat = hasattr(self.vllm, "chat")
        self.tokenizer = self.vllm.get_tokenizer()
        self.model = model

    def shutdown(self) -> None:
        try:
            if hasattr(self.vllm, "shutdown"):
                try:
                    self.vllm.shutdown()
                except Exception:
                    pass

            llm_engine = getattr(self.vllm, "llm_engine", None)
            if llm_engine is not None and hasattr(llm_engine, "shutdown"):
                try:
                    llm_engine.shutdown()
                except Exception:
                    pass
        finally:
            try:
                del self.tokenizer
            except Exception:
                pass
            try:
                del self.vllm
            except Exception:
                pass

            gc.collect()
            if torch is not None:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass


@MODEL_REGISTRY.register(PROVIDER_NAME)
class VLLMLocal(LLM):
    def __init__(self, config):
        super().__init__(config)
        self._lock = threading.RLock()
        self._busy = 0
        self._engine: Optional[_Engine] = _Engine(self.config.model, getattr(self.config, "engine", {}) or {})

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_streaming=False,
            supports_json_mode=True,
            supports_system=True,
            supports_tools=False,
            supports_vision=True,
            supports_audio=False,
            supports_thinking=True,
        )

    def unload(self, *, force: bool = False) -> None:
        with self._lock:
            if self._busy and not force:
                raise RuntimeError("Cannot unload while requests are running (set force=True to override).")
            engine = self._engine
            self._engine = None

        if engine is None:
            return

        try:
            engine.shutdown()
        except Exception:
            logger.exception("Best-effort vLLM shutdown failed.")

    def complete(self, reqs: List[LLMRequest]) -> List[LLMResponse]:
        if not reqs:
            return []

        with self._lock:
            engine = self._engine
            self._busy += 1

        if engine is None:
            with self._lock:
                self._busy -= 1
            return [_error_response("No model loaded.") for _ in reqs]

        try:
            prepared_requests = [self._prepare_request(req, engine.model) for req in reqs]
            grouped = self._group_requests(prepared_requests)
            results: List[LLMResponse] = [_error_response("Uninitialized result slot.") for _ in reqs]

            for group in grouped.values():
                try:
                    sampling_params = _make_sampling_params(group["sampling_kwargs"])
                    group_results = self._execute_group(
                        engine=engine,
                        messages_list=group["messages_list"],
                        sampling_params=sampling_params,
                        chat_template=group["chat_template"],
                        chat_template_kwargs=group["chat_template_kwargs"],
                        chat_template_content_format=group["chat_template_content_format"],
                    )
                except Exception as exc:
                    group_results = [
                        _error_response(str(exc), type(exc).__name__)
                        for _ in group["indices"]
                    ]

                for result_index, response in zip(group["indices"], group_results):
                    results[result_index] = response

            return results
        finally:
            with self._lock:
                self._busy -= 1

    def _prepare_request(self, req: LLMRequest, loaded_model: str) -> Dict[str, Any]:
        merged_params = _merge_params(self.config.params, req.params)
        requested_model = merged_params.get("model")
        if requested_model not in (None, loaded_model):
            raise ValueError(
                "Request asked for model override different from loaded engine "
                f"('{loaded_model}'). Call unload() and create a new instance."
            )

        sampling_kwargs = _extract_sampling_kwargs(merged_params)
        if req.json_mode:
            _so_key, _so_params = _guided_json_object_params()
            sampling_kwargs[_so_key] = _so_params
        chat_template_kwargs = _extract_chat_template_kwargs(self.config, merged_params)
        chat_template_content_format = _extract_chat_template_content_format(self.config, merged_params)
        chat_template = _extract_chat_template(self.config, merged_params)
        messages = [to_openai_chat_message(message) for message in req.messages]

        return {
            "messages": messages,
            "sampling_kwargs": sampling_kwargs,
            "chat_template": chat_template,
            "chat_template_kwargs": chat_template_kwargs,
            "chat_template_content_format": chat_template_content_format,
        }

    def _group_requests(self, prepared_requests: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        groups: Dict[str, Dict[str, Any]] = {}

        for index, prepared in enumerate(prepared_requests):
            key_tuple = (
                _stable_kv(prepared["sampling_kwargs"]),
                prepared.get("chat_template") or "",
                _stable_kv(prepared["chat_template_kwargs"]),
                prepared.get("chat_template_content_format") or "",
            )
            key = hashlib.md5(repr(key_tuple).encode("utf-8")).hexdigest()
            group = groups.setdefault(
                key,
                {
                    "indices": [],
                    "messages_list": [],
                    "sampling_kwargs": prepared["sampling_kwargs"],
                    "chat_template": prepared.get("chat_template"),
                    "chat_template_kwargs": prepared["chat_template_kwargs"],
                    "chat_template_content_format": prepared.get("chat_template_content_format"),
                },
            )
            group["indices"].append(index)
            group["messages_list"].append(prepared["messages"])

        return groups

    def _execute_group(
        self,
        engine: _Engine,
        messages_list: List[List[dict]],
        sampling_params: Any,
        chat_template: Optional[str],
        chat_template_kwargs: Dict[str, Any],
        chat_template_content_format: Optional[str],
    ) -> List[LLMResponse]:
        effective_chat_template = _resolve_chat_template(
            tokenizer=engine.tokenizer,
            configured_template=chat_template,
        )
        templates_to_try: list[Optional[str]] = [effective_chat_template]
        if effective_chat_template is None and chat_template:
            templates_to_try.append(chat_template)

        last_error: Exception | None = None
        for template_index, template in enumerate(templates_to_try):
            try:
                return self._execute_group_with_template(
                    engine=engine,
                    messages_list=messages_list,
                    sampling_params=sampling_params,
                    chat_template=template,
                    chat_template_kwargs=chat_template_kwargs,
                    chat_template_content_format=chat_template_content_format,
                )
            except Exception as exc:
                last_error = exc
                if template_index + 1 >= len(templates_to_try):
                    raise

        assert last_error is not None
        raise last_error

    def _execute_group_with_template(
        self,
        engine: _Engine,
        messages_list: List[List[dict]],
        sampling_params: Any,
        chat_template: Optional[str],
        chat_template_kwargs: Dict[str, Any],
        chat_template_content_format: Optional[str],
    ) -> List[LLMResponse]:
        if engine.has_chat:
            try:
                chat_kwargs: Dict[str, Any] = {
                    "messages": messages_list,
                    "sampling_params": sampling_params,
                    "chat_template_kwargs": chat_template_kwargs,
                }
                if chat_template is not None:
                    chat_kwargs["chat_template"] = chat_template
                if chat_template_content_format is not None:
                    chat_kwargs["chat_template_content_format"] = chat_template_content_format
                outputs = engine.vllm.chat(**chat_kwargs)
                return [self._response_from_output(output) for output in outputs]
            except TypeError:
                if chat_template_content_format is not None:
                    raise
                pass

        apply_chat_template_kwargs = dict(chat_template_kwargs)
        if chat_template is not None:
            apply_chat_template_kwargs["chat_template"] = chat_template
        prompts = [
            engine.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **apply_chat_template_kwargs,
            )
            for messages in messages_list
        ]
        outputs = engine.vllm.generate(prompts, sampling_params=sampling_params)
        return [self._response_from_output(output) for output in outputs]

    @staticmethod
    def _response_from_output(raw_output: Any) -> LLMResponse:
        generation = (getattr(raw_output, "outputs", None) or [None])[0]
        if generation is None:
            return _error_response("vLLM returned no generations.")
        return _build_llm_response(raw_output, generation)
