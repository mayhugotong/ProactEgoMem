"""Vertex AI (aiplatform.googleapis.com) provider for Gemini models.

Same request/response shape as the `gemini` (AI Studio) provider — only the URL
and auth differ: Vertex uses an OAuth2 Bearer token minted from a service
account (billed to the GCP project, i.e. the trial credit) instead of an API
key. google-auth is imported LAZILY so importing this module never fails on
hosts without it (registry imports every provider at startup)."""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List

from proact.core.registry import MODEL_REGISTRY
from proact.models.llm.base import Capabilities, LLM, LLMResponse
from proact.models.llm.providers.gemini import (
    _RateLimiter,
    _build_llm_response_from_gemini,
    _error_response,
    _extract_system_instructions,
    _camelize_dict,
    _llm_message_to_gemini_content,
    _merge_params,
    _post_json,
)

logger = logging.getLogger(__name__)

PROVIDER_NAME = "vertex"
_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class _TokenSource:
    """Mints and auto-refreshes an OAuth2 access token from a service account
    (or Application Default Credentials via GOOGLE_APPLICATION_CREDENTIALS)."""

    def __init__(self, service_account_file: str | None):
        self._lock = threading.Lock()
        self._sa_file = service_account_file
        self._creds = None  # lazily built on first token()

    def _ensure_creds(self):
        if self._creds is not None:
            return
        # LAZY imports: only needed on the compute node at run time.
        if self._sa_file:
            from google.oauth2 import service_account

            self._creds = service_account.Credentials.from_service_account_file(
                self._sa_file, scopes=_SCOPES
            )
        else:
            import google.auth

            self._creds, _ = google.auth.default(scopes=_SCOPES)

    def token(self) -> str:
        from google.auth.transport.requests import Request

        with self._lock:
            self._ensure_creds()
            # Refresh when missing or expired (google-auth applies a ~5min skew).
            if not getattr(self._creds, "valid", False):
                self._creds.refresh(Request())
            return self._creds.token


@MODEL_REGISTRY.register(PROVIDER_NAME)
class VertexGenerateContent(LLM):
    def __init__(self, config):
        super().__init__(config)
        self._lock = threading.RLock()
        self._rpm = getattr(self.config, "rpm_limit", None)
        self._limiter = _RateLimiter(rpm=int(self._rpm)) if self._rpm else None
        self._token_src: _TokenSource | None = None

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

    def _project_location(self, merged: Dict[str, Any]) -> tuple[str | None, str]:
        project = merged.get("project") or os.environ.get("VERTEX_PROJECT")
        location = (
            merged.get("location")
            or os.environ.get("VERTEX_LOCATION")
            or "us-central1"
        )
        return project, location

    def _token(self, merged: Dict[str, Any]) -> str:
        if self._token_src is None:
            # Only an EXPLICIT params.service_account_file forces SA-file loading.
            # Otherwise use google.auth.default(), which auto-detects BOTH a
            # service-account JSON (via $GOOGLE_APPLICATION_CREDENTIALS) and a
            # user ADC file (~/.config/gcloud, from `gcloud auth
            # application-default login`) — the latter would break
            # from_service_account_file().
            sa = merged.get("service_account_file")
            self._token_src = _TokenSource(sa)
        return self._token_src.token()

    @staticmethod
    def _endpoint(project: str, location: str, model: str) -> str:
        host = (
            "aiplatform.googleapis.com"
            if location == "global"
            else f"{location}-aiplatform.googleapis.com"
        )
        return (
            f"https://{host}/v1/projects/{project}/locations/{location}"
            f"/publishers/google/models/{model}:generateContent"
        )

    def complete(self, reqs: List["Any"]) -> List[LLMResponse]:
        if not reqs:
            return []
        with self._lock:
            pass

        results: List[LLMResponse] = [
            _error_response("Uninitialized result slot.") for _ in reqs
        ]
        # Resolve project/location/token once per batch; fail loudly & clearly.
        merged0 = _merge_params(self.config.params, {})
        project, location = self._project_location(merged0)
        if not project:
            return [
                _error_response(
                    "Vertex provider requires a GCP project (params.project or "
                    "$VERTEX_PROJECT).",
                    "ValueError",
                )
                for _ in reqs
            ]
        try:
            token = self._token(merged0)
        except Exception as exc:  # google-auth missing / bad SA / no ADC
            return [
                _error_response(
                    f"Vertex auth failed ({type(exc).__name__}): {exc}",
                    "AuthError",
                )
                for _ in reqs
            ]

        for idx, request in enumerate(reqs):
            merged = _merge_params(self.config.params, request.params)
            model = merged.get("model") or self.config.model

            generation_config: dict[str, Any] = {}
            if "temperature" in merged:
                generation_config["temperature"] = merged["temperature"]
            if "top_p" in merged:
                generation_config["topP"] = merged["top_p"]
            if "top_k" in merged:
                generation_config["topK"] = merged["top_k"]
            if "max_output_tokens" in merged:
                generation_config["maxOutputTokens"] = merged["max_output_tokens"]
            elif "max_tokens" in merged:
                generation_config["maxOutputTokens"] = merged["max_tokens"]
            if request.json_mode:
                generation_config["responseMimeType"] = "application/json"
            if isinstance(merged.get("thinking_config"), dict):
                generation_config["thinkingConfig"] = _camelize_dict(
                    merged["thinking_config"]
                )

            if self._limiter:
                self._limiter.wait()
            try:
                system_instruction, non_system = _extract_system_instructions(
                    request.messages
                )
                payload: dict[str, Any] = {
                    "contents": [
                        _llm_message_to_gemini_content(m) for m in non_system
                    ]
                }
                if system_instruction:
                    payload["system_instruction"] = {
                        "parts": [{"text": system_instruction}]
                    }
                if generation_config:
                    payload["generationConfig"] = generation_config

                # Token may have expired between batches on long runs.
                token = self._token(merged)
                response_payload = _post_json(
                    url=self._endpoint(project, location, model),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {token}",
                        # Billing/quota project for USER (ADC) credentials; harmless
                        # for service accounts. Ensures spend hits VERTEX_PROJECT.
                        "x-goog-user-project": project,
                    },
                    payload=payload,
                    timeout_s=float(self.config.request_timeout_s),
                )
                results[idx] = _build_llm_response_from_gemini(response_payload)
            except Exception as exc:
                status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                body = None
                reader = getattr(exc, "read", None)
                if callable(reader):
                    try:
                        body = reader().decode("utf-8", errors="replace")
                    except Exception:
                        body = None
                results[idx] = _error_response(
                    body or str(exc), type(exc).__name__, status_code=status
                )
        return results
