from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from proact.evaluation.runtime.vidllms.base import (
    VideoLLMAdapter,
    VideoLLMResponse,
    VideoWindowRequest,
)


def _project_root() -> Path:
    return Path(os.environ.get("PROACT_PROJECT_ROOT", Path.cwd().parent)).expanduser()


def _setting_path(settings: dict[str, Any], key: str, default: str) -> Path:
    value = str(settings.get(key, default))
    return Path(os.path.expandvars(value)).expanduser()


def _setting_string(settings: dict[str, Any], key: str, default: str) -> str:
    return os.path.expandvars(str(settings.get(key, default))).strip()


def _insert_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


@contextmanager
def _temporary_cwd(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def _frame_to_pil(frame: Any) -> Any:
    from PIL import Image

    source = getattr(frame, "array", frame)
    if isinstance(source, Image.Image):
        return source.convert("RGB")

    import numpy as np

    array = np.asarray(source)
    if array.dtype != np.uint8:
        if array.max(initial=0) <= 1:
            array = array * 255
        array = array.clip(0, 255).astype(np.uint8)
    if array.ndim == 2:
        return Image.fromarray(array).convert("RGB")
    return Image.fromarray(array).convert("RGB")


def _frames_to_pil(frames: list[Any]) -> list[Any]:
    return [_frame_to_pil(frame) for frame in frames]


def _frames_to_numpy(frames: list[Any]) -> Any:
    import numpy as np

    return np.stack([np.asarray(image.convert("RGB")) for image in _frames_to_pil(frames)])


def _window_text(request: VideoWindowRequest) -> str:
    if request.input_text:
        return request.input_text
    frame_lines = "\n".join(
        f"[frame {index + 1}] timestamp={timestamp:.3f}s"
        for index, timestamp in enumerate(request.frame_timestamps)
    )
    frame_context = (
        "The window of video frames\n"
        "The image inputs are ordered oldest to newest and correspond to the frame labels below. "
        "The last image is the current moment (now).\n"
        f"{frame_lines}"
    )
    parts = [request.prompt_text]
    if request.state_text:
        parts.append(request.state_text)
    parts.append(frame_context)
    return "\n\n".join(parts).strip()


def _torch_dtype(torch_module: Any, value: str | None, default: Any) -> Any:
    if value is None:
        return default
    return getattr(torch_module, value)


def _call_output_to_text(output: Any) -> str:
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, list) and output:
        output = output[0]
    return "" if output is None else str(output)


def _request_messages(request: VideoWindowRequest, *, adapter: str, model_id: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": _window_text(request),
            "metadata": {
                "adapter": adapter,
                "model_id": model_id,
                "sampled_frame_count": len(request.frames),
            },
        }
    ]


class MiniCPMO45Adapter(VideoLLMAdapter):
    def __init__(self, settings: dict[str, Any] | None = None):
        self.settings = dict(settings or {})
        self.model_id = str(self.settings.get("model_id", "openbmb/MiniCPM-o-4_5"))
        self.model = None
        self._load_error: Exception | None = None

    def _ensure_model(self) -> Any:
        if self.model is not None:
            return self.model
        if self._load_error is not None:
            raise RuntimeError("MiniCPM-o model load failed previously.") from self._load_error

        import torch
        from transformers import AutoConfig, AutoModel

        dtype = _torch_dtype(torch, self.settings.get("torch_dtype"), torch.bfloat16)
        kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }
        if self.settings.get("attn_implementation"):
            kwargs["attn_implementation"] = self.settings["attn_implementation"]

        config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)

        # MiniCPM-o-4.5 remote code initializes TTS at model construction time.
        # Some configs miss these sampling fields, even for text/image-only inference.
        tts_config = getattr(config, "tts_config", None)
        if tts_config is not None:
            for key, value in {
                "top_p": 1.0,
                "top_k": 0,
                "temperature": 1.0,
                "repetition_penalty": 1.0,
            }.items():
                if not hasattr(tts_config, key):
                    setattr(tts_config, key, value)

        kwargs["config"] = config

        try:
            self.model = AutoModel.from_pretrained(self.model_id, **kwargs).eval()
            device = str(self.settings.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
            self.model = self.model.to(device=device)
        except Exception as exc:
            self._load_error = exc
            raise
        return self.model

    def complete_window(self, request: VideoWindowRequest) -> VideoLLMResponse:
        model = self._ensure_model()
        frames = _frames_to_pil(request.frames)
        text = _window_text(request)
        content_order = str(self.settings.get("content_order", "text-first"))
        if content_order == "frames-first":
            content = frames + [text]
        elif content_order == "text-first":
            content = [text] + frames
        else:
            raise ValueError(
                "MiniCPM adapter setting 'content_order' must be either "
                "'text-first' or 'frames-first'."
            )
        msgs = [{"role": "user", "content": content}]
        output = model.chat(
            msgs=msgs,
            max_new_tokens=int(self.settings.get("max_new_tokens", 256)),
            use_image_id=bool(self.settings.get("use_image_id", False)),
            max_slice_nums=int(self.settings.get("max_slice_nums", 1)),
            use_tts_template=bool(self.settings.get("use_tts_template", False)),
            enable_thinking=bool(self.settings.get("enable_thinking", False)),
        )
        return VideoLLMResponse(
            response_text=_call_output_to_text(output),
            response_raw=output,
            request_messages=_request_messages(request, adapter="minicpm-o45", model_id=self.model_id),
            raw_model_output={
                "adapter": "minicpm-o45",
                "model_id": self.model_id,
                "sampled_frame_count": len(frames),
                "content_order": content_order,
            },
        )


class LLaVAOneVision15Adapter(VideoLLMAdapter):
    def __init__(self, settings: dict[str, Any] | None = None):
        self.settings = dict(settings or {})
        self.model_id = str(self.settings.get("model_id", "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"))
        self.model = None
        self.processor = None
        self._load_error: Exception | None = None

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        if self._load_error is not None:
            raise RuntimeError("LLaVA-OneVision model load failed previously.") from self._load_error
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

            if self.settings.get("model_base"):
                raise ValueError(
                    "LLaVA-OneVision direct Transformers loading does not support "
                    "'model_base'. Use a merged/full checkpoint for this adapter."
                )

            model_kwargs: dict[str, Any] = {
                "device_map": self.settings.get("device_map", "auto"),
                "torch_dtype": self.settings.get("torch_dtype", "auto"),
                "trust_remote_code": True,
            }
            if bool(self.settings.get("low_cpu_mem_usage", True)):
                model_kwargs["low_cpu_mem_usage"] = True
            device = str(self.settings.get("device", "cuda"))
            if device != "cuda":
                model_kwargs["device_map"] = {"": device}
            if bool(self.settings.get("load_8bit", False)):
                model_kwargs["load_in_8bit"] = True
            elif bool(self.settings.get("load_4bit", False)):
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            if bool(self.settings.get("use_flash_attn", False)):
                model_kwargs["attn_implementation"] = "flash_attention_2"

            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            # transformers 5.x (PR #41541) dropped the implicit pad_token_id
            # default that LOV-1.5's remote modeling code assumes exists on the
            # (text) config. Inject it before init so model construction does not
            # crash with AttributeError. pad_token_id is unused for batch-1
            # inference, so falling back to eos (or 0) is safe.
            config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
            for cfg in (config, getattr(config, "text_config", None)):
                if cfg is None:
                    continue
                if getattr(cfg, "pad_token_id", None) is None:
                    eos = getattr(cfg, "eos_token_id", None)
                    if isinstance(eos, (list, tuple)):
                        eos = eos[0] if eos else None
                    cfg.pad_token_id = eos if eos is not None else 0
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                config=config,
                **model_kwargs,
            )
        except Exception as exc:
            self._load_error = exc
            raise
        self.model.eval()

    def complete_window(self, request: VideoWindowRequest) -> VideoLLMResponse:
        import torch

        self._ensure_model()
        video_array = _frames_to_numpy(request.frames)
        video_part = {
            "type": "video",
            "video": "proact_window",
            "fps": float(request.adapter_settings.get("fps", 1.0)),
        }
        input_text = _window_text(request)
        text_part = {"type": "text", "text": input_text}
        content_order = str(self.settings.get("content_order", "text-first"))
        if content_order == "frames-first":
            content = [video_part, text_part]
        elif content_order == "text-first":
            content = [text_part, video_part]
        else:
            raise ValueError(
                "LLaVA-OneVision adapter setting 'content_order' must be either "
                "'text-first' or 'frames-first'."
            )
        conversation = [
            {
                "role": "user",
                "content": content,
            }
        ]
        prompt = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[prompt],
            videos=[video_array],
            padding=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        inputs.pop("second_per_grid_ts", None)
        do_sample = bool(self.settings.get("do_sample", False))
        generation_kwargs = {
            "max_new_tokens": int(self.settings.get("max_new_tokens", 256)),
            "do_sample": do_sample,
            "repetition_penalty": float(self.settings.get("repetition_penalty", 1.0)),
        }
        eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
        if do_sample:
            generation_kwargs["temperature"] = float(self.settings.get("temperature", 0.2))
            generation_kwargs["top_p"] = float(self.settings.get("top_p", 0.9))
            if self.settings.get("top_k") is not None:
                generation_kwargs["top_k"] = int(self.settings["top_k"])
        else:
            generation_kwargs["top_k"] = None
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)
        input_token_count = int(inputs["input_ids"].shape[1])
        if output_ids.shape[1] > input_token_count:
            generated_ids = output_ids[:, input_token_count:]
            decode_source = "sliced_prompt_prefix"
        else:
            generated_ids = output_ids
            decode_source = "full_output_ids"
        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return VideoLLMResponse(
            response_text=output_text,
            response_raw={
                "input_ids_shape": list(inputs["input_ids"].shape),
                "output_ids_shape": list(output_ids.shape),
                "decoded_ids_shape": list(generated_ids.shape),
                "decode_source": decode_source,
            },
            request_messages=_request_messages(request, adapter="llava-onevision15", model_id=self.model_id),
            raw_model_output={
                "adapter": "llava-onevision15",
                "model_id": self.model_id,
                "sampled_frame_count": len(request.frames),
                "sampled_frame_times": [float(timestamp) for timestamp in request.frame_timestamps],
                "content_order": content_order,
                "model_input_text": input_text,
                "processor_prompt": prompt,
                "decode_source": decode_source,
            },
        )


class VideoLLaMA3Adapter(VideoLLMAdapter):
    def __init__(self, settings: dict[str, Any] | None = None):
        self.settings = dict(settings or {})
        self.model_id = _setting_string(self.settings, "model_id", "DAMO-NLP-SG/VideoLLaMA3-7B")
        self.model = None
        self.processor = None
        self._load_error: Exception | None = None

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        if self._load_error is not None:
            raise RuntimeError(
                "VideoLLaMA3 model load failed previously: "
                f"{type(self._load_error).__name__}: {self._load_error}"
            ) from self._load_error

        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

        repo = _setting_path(
            self.settings,
            "repo_path",
            str(_project_root() / "models/vidllms/VideoLLaMA3"),
        )
        try:
            _insert_sys_path(repo)
            _insert_sys_path(repo / "inference")
            # Registers the repo's custom AutoConfig/AutoModel/AutoProcessor classes
            # when running from a local checkout instead of relying on HF remote code.
            try:
                import transformers_api  # type: ignore  # noqa: F401
            except ModuleNotFoundError:
                pass

            dtype = _torch_dtype(torch, self.settings.get("torch_dtype"), torch.bfloat16)
            load_kwargs: dict[str, Any] = {
                "device_map": self.settings.get("device_map", "auto"),
                "torch_dtype": dtype,
                # Use SDPA by default so the run does not hard-fail on clusters
                # without flash-attn. Users can opt into flash_attention_2 in YAML.
                "attn_implementation": self.settings.get("attn_implementation", "sdpa"),
            }
            if self.settings.get("local_files_only") is not None:
                load_kwargs["local_files_only"] = bool(self.settings["local_files_only"])

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                **load_kwargs,
            )
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                use_fast=bool(self.settings.get("use_fast_processor", False)),
                local_files_only=bool(self.settings.get("local_files_only", False)),
            )
        except Exception as exc:
            self._load_error = exc
            raise

    def complete_window(self, request: VideoWindowRequest) -> VideoLLMResponse:
        import torch

        self._ensure_model()
        frames = _frames_to_pil(request.frames)
        text = _window_text(request)
        video_part = {
            "type": "video",
            "timestamps": [float(timestamp) for timestamp in request.frame_timestamps],
            "num_frames": len(frames),
        }
        text_part = {"type": "text", "text": text}
        content_order = str(self.settings.get("content_order", "text-first"))
        if content_order == "frames-first":
            content = [video_part, text_part]
        elif content_order == "text-first":
            content = [text_part, video_part]
        else:
            raise ValueError(
                "VideoLLaMA3 adapter setting 'content_order' must be either "
                "'text-first' or 'frames-first'."
            )

        conversation = [{"role": "user", "content": content}]
        model_conversation_for_trace = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text}
                    if item.get("type") == "text"
                    else {
                        "type": "video",
                        "num_frames": int(item["num_frames"]),
                        "timestamps": [float(timestamp) for timestamp in item["timestamps"]],
                    }
                    for item in content
                ],
            }
        ]
        processor_prompt = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_system_prompt=bool(self.settings.get("add_system_prompt", True)),
            add_generation_prompt=True,
        )
        inputs = self.processor(
            conversation=conversation,
            images=[("video", frames)],
            add_system_prompt=bool(self.settings.get("add_system_prompt", True)),
            add_generation_prompt=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        if "pixel_values" in inputs:
            vision_dtype = _torch_dtype(
                torch,
                self.settings.get("vision_dtype"),
                torch.bfloat16,
            )
            inputs["pixel_values"] = inputs["pixel_values"].to(vision_dtype)

        do_sample = bool(self.settings.get("do_sample", False))
        generation_kwargs = {
            "do_sample": do_sample,
            "max_new_tokens": int(self.settings.get("max_new_tokens", 256)),
            "use_cache": True,
        }
        eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
            generation_kwargs["pad_token_id"] = eos_token_id
        if do_sample:
            generation_kwargs["temperature"] = float(self.settings.get("temperature", 0.2))
            generation_kwargs["top_p"] = float(self.settings.get("top_p", 0.9))
            if self.settings.get("top_k") is not None:
                generation_kwargs["top_k"] = int(self.settings["top_k"])

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)
        input_token_count = int(inputs["input_ids"].shape[1])
        # VideoLLaMA3's official inference path decodes output_ids directly.
        # Some Transformers models return prompt+completion, while this path can
        # return only completion tokens. Only slice when the prompt is included.
        if output_ids.shape[1] > input_token_count:
            generated_ids = output_ids[:, input_token_count:]
            decode_source = "sliced_prompt_prefix"
        else:
            generated_ids = output_ids
            decode_source = "full_output_ids"
        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()
        return VideoLLMResponse(
            response_text=output_text,
            response_raw={
                "input_ids_shape": list(inputs["input_ids"].shape),
                "output_ids_shape": list(output_ids.shape),
                "decoded_ids_shape": list(generated_ids.shape),
                "decode_source": decode_source,
            },
            request_messages=_request_messages(request, adapter="videollama3", model_id=self.model_id),
            raw_model_output={
                "adapter": "videollama3",
                "model_id": self.model_id,
                "sampled_frame_count": len(frames),
                "sampled_frame_times": [float(timestamp) for timestamp in request.frame_timestamps],
                "content_order": content_order,
                "model_input_text": text,
                "model_conversation": model_conversation_for_trace,
                "processor_prompt": processor_prompt,
                "processor_add_system_prompt": bool(self.settings.get("add_system_prompt", True)),
                "processor_add_generation_prompt": True,
                "decode_source": decode_source,
                "merge_size": int(self.settings.get("merge_size", 2)),
            },
        )


def build_vidllm_adapter(adapter_name: str, adapter_settings: dict[str, Any] | None = None) -> VideoLLMAdapter:
    settings = dict(adapter_settings or {})
    if adapter_name == "videollama3":
        return VideoLLaMA3Adapter(settings)
    if adapter_name == "llava-onevision15":
        return LLaVAOneVision15Adapter(settings)
    if adapter_name == "minicpm-o45":
        return MiniCPMO45Adapter(settings)
    raise ValueError(f"Unsupported video-LLM adapter '{adapter_name}'.")
