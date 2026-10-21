from __future__ import annotations

from typing import List, Dict, Any, Optional
import io, base64
from PIL import Image

from ..base import (
    LLM, LLMRequest, LLMResponse, Capabilities, LLMMessage,
    TextPart, ImageURLPart, ImageB64Part
)
from proact.core.registry import MODEL_REGISTRY
#from ..registry import register_llm

PROVIDER_NAME = "transformers_vl"


@MODEL_REGISTRY.register(PROVIDER_NAME)
class HFTransformersVision(LLM):
    """
    Minimal Vision-Language HF adapter.

    Assumes the model supports:
      - AutoProcessor.apply_chat_template(... tokenize=True, return_dict=True, return_tensors="pt")
      - AutoModelForImageTextToText.generate(...)
    """

    def __init__(self, config):
        super().__init__(config)
        self._model = None
        self._processor = None

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_streaming=False,
            supports_json_mode=False,
            supports_system=True,
            supports_tools=False,
            supports_vision=True,
            supports_audio=False,
        )

    # ---------- tiny param helpers ----------
    def _get_param(self, req: LLMRequest, name: str, default=None):
        # 1) req.<name>
        if hasattr(req, name):
            v = getattr(req, name)
            if v is not None:
                return v
        # 2) req.params[name]
        rp = getattr(req, "params", None)
        if isinstance(rp, dict) and name in rp and rp[name] is not None:
            return rp[name]
        # 3) config.params[name]
        cp = getattr(self.config, "params", None)
        if isinstance(cp, dict) and name in cp and cp[name] is not None:
            return cp[name]
        return default

    def _get_chat_template_kwargs(self, req: LLMRequest) -> Dict[str, Any]:
        # req.params.chat_template_kwargs overrides config.extra.chat_template_kwargs
        rp = getattr(req, "params", None)
        if isinstance(rp, dict):
            per = rp.get("chat_template_kwargs")
            if isinstance(per, dict):
                return per

        extra = getattr(self.config, "extra", None)
        if isinstance(extra, dict):
            c = extra.get("chat_template_kwargs")
            if isinstance(c, dict):
                return c
        return {}

    # ---------- loader ----------
    def _ensure_model(self):
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        engine = getattr(self.config, "engine", {}) or {}
        device_map = engine.get("device_map", engine.get("device", "auto"))
        dtype = engine.get("dtype", engine.get("torch_dtype", "auto"))

        # dtype parsing (keep it minimal)
        torch_dtype = "auto"
        if isinstance(dtype, str):
            d = dtype.lower()
            if d in ("bf16", "bfloat16"):
                torch_dtype = torch.bfloat16
            elif d in ("fp16", "float16", "half"):
                torch_dtype = torch.float16
            elif d in ("fp32", "float32"):
                torch_dtype = torch.float32
            else:
                torch_dtype = "auto"
        else:
            torch_dtype = dtype

        self._processor = AutoProcessor.from_pretrained(self.config.model, trust_remote_code=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.config.model,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        self._model.eval()

        # hard requirement for this minimal class:
        if not hasattr(self._processor, "apply_chat_template"):
            raise RuntimeError(
                "This VL adapter expects processor.apply_chat_template(...). "
                "Your model/processor doesn't provide it."
            )

    # ---------- message conversion ----------
    def _pil_from_b64(self, b64: str) -> Image.Image:
        s = b64.strip()
        if s.startswith("data:") and "," in s:
            s = s.split(",", 1)[1]
        data = base64.b64decode(s)
        return Image.open(io.BytesIO(data)).convert("RGB")

    def _to_hf_messages(self, messages: List[LLMMessage]) -> List[dict]:
        """
        HF multimodal chat messages:
          [{"role":"...", "content":[{"type":"text","text":"..."}, {"type":"image","image": ...}]}]
        """
        out: List[dict] = []
        for m in messages:
            if isinstance(m.content, str):
                out.append({"role": m.role, "content": [{"type": "text", "text": m.content}]})
                continue

            parts: List[dict] = []
            for p in m.content:
                if isinstance(p, TextPart):
                    parts.append({"type": "text", "text": p.text})
                elif isinstance(p, ImageURLPart):
                    # use "image" key for broader VL compatibility
                    parts.append({"type": "image", "image": str(p.url)})
                elif isinstance(p, ImageB64Part):
                    parts.append({"type": "image", "image": self._pil_from_b64(p.b64)})
                else:
                    parts.append({"type": "text", "text": "[unsupported content]"})
            if not parts:
                parts = [{"type": "text", "text": ""}]
            out.append({"role": m.role, "content": parts})
        return out

    # ---------- inference ----------
    def complete(self, reqs: List[LLMRequest]) -> List[LLMResponse]:
        if not reqs:
            return []

        self._ensure_model()

        import torch
        from transformers import set_seed

        results: List[LLMResponse] = []
        device = next(self._model.parameters()).device

        for req in reqs:
            try:
                # optional seed (per request)
                seed = self._get_param(req, "seed", None)
                if seed is not None:
                    set_seed(int(seed))

                messages = self._to_hf_messages(req.messages)
                tmpl_kwargs = self._get_chat_template_kwargs(req)

                processed = self._processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    truncation=True,
                    **(tmpl_kwargs or {}),
                ).to(device)

                max_new = int(self._get_param(req, "max_tokens", 256))
                temperature = self._get_param(req, "temperature", None)
                top_p = self._get_param(req, "top_p", None)
                top_k = self._get_param(req, "top_k", None)
                min_p = self._get_param(req, "min_p", None)
                repetition_penalty = self._get_param(req, "repetition_penalty", None)

                do_sample = temperature is not None and float(temperature) > 1e-6

                gen_kwargs = {"max_new_tokens": max_new, "do_sample": do_sample}
                if do_sample:
                    gen_kwargs["temperature"] = float(temperature)
                if top_p is not None:
                    gen_kwargs["top_p"] = float(top_p)
                if top_k is not None:
                    gen_kwargs["top_k"] = int(top_k)
                if min_p is not None:
                    gen_kwargs["min_p"] = float(min_p)
                if repetition_penalty is not None:
                    gen_kwargs["repetition_penalty"] = float(repetition_penalty)

                with torch.inference_mode():
                    out_ids = self._model.generate(**processed, **gen_kwargs)

                # decode only the completion
                prompt_len = processed["input_ids"].shape[1]
                completion_ids = out_ids[0][prompt_len:]

                tok = getattr(self._processor, "tokenizer", None)
                if tok is not None:
                    text = tok.decode(completion_ids, skip_special_tokens=True)
                else:
                    text = self._processor.decode(completion_ids, skip_special_tokens=True)

                results.append(
                    LLMResponse(
                        text=(text or "").strip(),
                        raw={"messages": messages, "gen_kwargs": gen_kwargs, "chat_template_kwargs": tmpl_kwargs},
                    )
                )

            except Exception as e:
                # per-request failure shouldn't kill the whole batch
                results.append(
                    LLMResponse(
                        ok=False,
                        error=str(e),
                        error_type=type(e).__name__,
                        status_code=None,
                    )
                )

        return results