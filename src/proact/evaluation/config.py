from __future__ import annotations

import os
import re
from pathlib import Path
from string import Formatter
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from proact.utils.yaml_loader import read_yaml


class EvaluationDatasetConfig(BaseModel):
    dataset: str
    root: str
    split: str = "val"
    dataset_params: dict[str, Any] = Field(default_factory=dict)

    def validate_required_fields(self) -> None:
        if self.dataset in ("ProactBenchClips", "ProactBenchScienceClips"):
            required = {"task_json"}
        else:
            required = {"version", "task_json", "ego4d_root"}
        missing = sorted(required.difference(self.dataset_params))
        if missing:
            raise ValueError(
                "Evaluation dataset config requires dataset_params keys "
                f"{missing}. Got keys={sorted(self.dataset_params.keys())}."
            )


class InputModeConfig(BaseModel):
    mode: Literal["sliding-window", "full-video", "chunked-window"]
    fps: Optional[float] = None
    window_size_frames: Optional[int] = None
    stride_frames: Optional[int] = None
    num_frames: Optional[int] = None

    def validate_mode(self) -> "InputModeConfig":
        if self.mode == "full-video":
            if self.num_frames is None or self.num_frames <= 0:
                raise ValueError("Full-video mode requires num_frames > 0.")
            return self

        if self.fps is None or self.fps <= 0:
            raise ValueError(f"{self.mode} mode requires fps > 0.")
        if self.window_size_frames is None or self.window_size_frames <= 0:
            raise ValueError(f"{self.mode} mode requires window_size_frames > 0.")

        stride = self.window_size_frames if self.mode == "chunked-window" else self.stride_frames
        if stride is None or stride <= 0:
            raise ValueError(f"{self.mode} mode requires stride_frames > 0.")
        if self.mode == "chunked-window" and self.stride_frames not in (None, self.window_size_frames):
            raise ValueError(
                "Chunked-window mode is defined as sliding-window with stride_frames == window_size_frames."
            )

        self.stride_frames = stride
        return self


class ToolStateTemplateConfig(BaseModel):
    message: str
    forwarded_args: list[str] = Field(default_factory=list)
    reserved_state_fields: set[str] = {
        "id",
        "instance_id",
        "start_time_sec",
        "start_timestamp_sec",
    }

    def validate_template(self, tool_name: str) -> None:
        placeholders = {
            field_name
            for _, field_name, _, _ in Formatter().parse(self.message)
            if field_name
        }
        allowed_fields = set(self.forwarded_args).union(self.reserved_state_fields)
        unknown = sorted(placeholders.difference(allowed_fields))
        if unknown:
            raise ValueError(
                f"Prompt state template for '{tool_name}' references placeholders {unknown} "
                f"but they are not listed in forwarded_args or reserved state fields {sorted(self.reserved_state_fields)}."
            )


class MessageHistoryConfig(BaseModel):
    enabled: bool = False
    message_key: str = "message"
    max_messages: int = 10
    include_silence: bool = False

    def validate_config(self) -> "MessageHistoryConfig":
        if self.enabled and self.max_messages <= 0:
            raise ValueError("prompt.message_history.max_messages must be > 0 when enabled.")
        if self.enabled and not self.message_key:
            raise ValueError("prompt.message_history.message_key must be non-empty when enabled.")
        return self


class HistorySelectionConfig(BaseModel):
    mode: Literal["none", "last_n", "all"] = "none"
    n: int = 10
    source: Literal["executed", "attempted"] = "executed"

    def validate_config(self, name: str) -> "HistorySelectionConfig":
        if self.mode == "last_n" and self.n <= 0:
            raise ValueError(f"{name}.n must be > 0 when mode='last_n'.")
        return self


class IntervalHistoryConfig(BaseModel):
    mode: Literal["none", "last_n", "all"] = "none"
    n: int = 5

    def validate_config(self) -> "IntervalHistoryConfig":
        if self.mode == "last_n" and self.n <= 0:
            raise ValueError(
                "prompt_context.completed_tool_intervals.n must be > 0 "
                "when mode='last_n'."
            )
        return self


class PromptContextConfig(BaseModel):
    include_active_tools: bool = True
    completed_tool_intervals: IntervalHistoryConfig = Field(default_factory=IntervalHistoryConfig)
    recent_tool_calls: HistorySelectionConfig = Field(default_factory=HistorySelectionConfig)
    recent_messages: HistorySelectionConfig = Field(default_factory=HistorySelectionConfig)

    def validate_config(self) -> "PromptContextConfig":
        self.completed_tool_intervals.validate_config()
        self.recent_tool_calls.validate_config("prompt_context.recent_tool_calls")
        self.recent_messages.validate_config("prompt_context.recent_messages")
        return self

    class Config:
        extra = "forbid"


class PromptConfig(BaseModel):
    prompt_path: str
    response_format_prompt: str = ""
    include_state: bool = True
    state_format: Literal["full", "compact", "active-only"] = "full"
    tool_state_templates: dict[str, ToolStateTemplateConfig] = Field(default_factory=dict)
    message_history: MessageHistoryConfig = Field(default_factory=MessageHistoryConfig)

    class Config:
        extra = "forbid"


class JsonValueRule(BaseModel):
    path: str
    equals: Any


class SilenceConfig(BaseModel):
    raw_text_values: list[str] = Field(default_factory=list)
    json_value_rules: list[JsonValueRule] = Field(default_factory=list)
    empty_list_means_no_action: bool = False


class ParserConfig(BaseModel):
    unified_schema: bool = False
    allow_notification_raw_text: bool = False
    # JSON is JSON, with or without a markdown code fence.
    allow_code_fence_json: bool = True
    # Unified-schema JSON recovery from noisy output: "strict" accepts only a
    # pure-JSON response, "first"/"last" take the first/last JSON value found.
    json_match: Literal["strict", "first", "last"] = "last"
    allow_first_json_match: bool = False
    allow_last_json_match: bool = False
    allow_root_list: bool = True
    allow_root_tool_object: bool = False
    single_tool_keys: list[str] = Field(default_factory=lambda: ["tool"])
    multi_tool_keys: list[str] = Field(default_factory=lambda: ["tool_calls", "tools"])
    entry_tool_keys: list[str] = Field(default_factory=list)
    tool_name_key: str = "name"
    tool_args_key: str = "args"
    timestamp_keys: list[str] = Field(default_factory=list)
    silence: SilenceConfig = Field(default_factory=SilenceConfig)

    def validate_config(self) -> "ParserConfig":
        if self.allow_first_json_match and self.allow_last_json_match:
            raise ValueError(
                "Parser config cannot enable both allow_first_json_match and allow_last_json_match at the same time."
            )
        return self


class TaskPolicyConfig(BaseModel):
    task_name: str
    expected_interval_types: list[str]
    expected_start_tools: list[str]
    expected_end_tools: list[str]
    allowed_tool_names: list[str]
    ignored_tool_names: list[str] = Field(default_factory=list)


class MetricConfig(BaseModel):
    proactive_start_tolerances_sec: list[float]
    iou_thresholds: list[float]


class FrameSamplingConfig(BaseModel):
    # GPU decoding is the default. Runs FAIL if CUDA ffmpeg is unavailable or
    # decoding errors — there is no silent CPU fallback. Machines without a
    # suitable GPU must select "pyav-cpu" explicitly.
    backend: Literal["ffmpeg-cuda", "pyav-cpu"] = "ffmpeg-cuda"
    ffmpeg_path: Optional[str] = None
    include_endpoints: bool = True
    unique_frames: bool = True
    convert_to_rgb: bool = True
    precise_seek: bool = True
    resize_frames: bool = False
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None

    def validate_config(self) -> "FrameSamplingConfig":
        if self.resize_frames:
            if self.resize_width is None or self.resize_width <= 0:
                raise ValueError("Frame resizing requires resize_width > 0.")
            if self.resize_height is None or self.resize_height <= 0:
                raise ValueError("Frame resizing requires resize_height > 0.")
        else:
            if self.resize_width is not None or self.resize_height is not None:
                raise ValueError(
                    "Set resize_frames=true before specifying resize_width/resize_height."
                )
        return self


class ModelBackendConfig(BaseModel):
    llm_config_path: str
    profile: str
    overrides: dict[str, Any] = Field(default_factory=dict)
    request_params: dict[str, Any] = Field(default_factory=dict)


class ExecutionBackendConfig(BaseModel):
    mode: Literal["stateless", "native-streaming", "video-llm"] = "stateless"
    adapter: Optional[
        Literal[
            "videollm-online",
            "mmduet",
            "llava-onevision15",
            "minicpm-o45",
            "videollama3",
        ]
    ] = None
    frame_fps: Optional[float] = None
    inject_prompt_as_initial_user_message: bool = True
    inject_tool_state_updates: bool = True
    adapter_settings: dict[str, Any] = Field(default_factory=dict)

    def validate_config(self) -> "ExecutionBackendConfig":
        streaming_adapters = {"videollm-online", "mmduet"}
        vidllm_adapters = {
            "llava-onevision15",
            "minicpm-o45",
            "videollama3",
        }
        if self.mode == "stateless":
            if self.adapter is not None:
                raise ValueError("execution_backend.adapter must be omitted when mode='stateless'.")
            if self.frame_fps is not None:
                raise ValueError("execution_backend.frame_fps must be omitted when mode='stateless'.")
            return self

        if self.mode == "video-llm":
            if self.adapter is None:
                raise ValueError("execution_backend.mode='video-llm' requires execution_backend.adapter.")
            if self.adapter not in vidllm_adapters:
                raise ValueError(
                    "execution_backend.mode='video-llm' requires a video-LLM adapter. "
                    f"Got '{self.adapter}'."
                )
            if self.frame_fps is not None:
                raise ValueError("execution_backend.frame_fps must be omitted when mode='video-llm'.")
            return self

        if self.adapter is None:
            raise ValueError("execution_backend.mode='native-streaming' requires execution_backend.adapter.")
        if self.adapter not in streaming_adapters:
            raise ValueError(
                "execution_backend.mode='native-streaming' requires a native streaming adapter. "
                f"Got '{self.adapter}'."
            )
        if self.frame_fps is None or self.frame_fps <= 0:
            raise ValueError(
                "execution_backend.mode='native-streaming' requires execution_backend.frame_fps > 0."
            )
        return self


class OutputConfig(BaseModel):
    output_dir: str
    max_clip_attempts: int = 1
    save_execution_traces: bool = True
    save_match_results: bool = False
    save_skipped_matches: bool = False

    def validate_config(self) -> "OutputConfig":
        if self.max_clip_attempts <= 0:
            raise ValueError("output.max_clip_attempts must be > 0.")
        return self


class LoggingConfig(BaseModel):
    level: str = "INFO"


class EvaluationConfig(BaseModel):
    dataset: EvaluationDatasetConfig
    context: Literal["long-video", "short-video"] = "short-video"
    proactive_level: Literal["notification", "intervention"] = "intervention"
    input_mode: Optional[InputModeConfig] = None
    execution_backend: ExecutionBackendConfig = Field(default_factory=ExecutionBackendConfig)
    task_policy: TaskPolicyConfig
    prompt: PromptConfig
    prompt_context: PromptContextConfig = Field(default_factory=PromptContextConfig)
    parser: ParserConfig = Field(default_factory=ParserConfig)
    metrics: Optional[MetricConfig] = None
    frame_sampling: FrameSamplingConfig = Field(default_factory=FrameSamplingConfig)
    model: Optional[ModelBackendConfig] = None
    # Optional dedicated model for QA post-evaluation answering. When unset, QA
    # falls back to `model`. Use this to score answers with a separate (e.g.
    # greedy-decode) profile, decoupled from the trace-generation model.
    qa_model: Optional[ModelBackendConfig] = None
    output: OutputConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    seed: int = 0

    @property
    def effective_qa_model(self) -> Optional[ModelBackendConfig]:
        """Model used for QA answering: dedicated qa_model if set, else model."""
        return self.qa_model or self.model

    def validate_config(self) -> "EvaluationConfig":
        self.dataset.validate_required_fields()
        self.execution_backend.validate_config()
        if self.execution_backend.mode in {"stateless", "video-llm"}:
            if self.input_mode is None:
                raise ValueError(f"{self.execution_backend.mode} execution requires input_mode to be configured.")
            self.input_mode.validate_mode()
            if self.execution_backend.mode == "stateless" and self.model is None:
                raise ValueError("Stateless execution requires model to be configured.")
        elif self.model is not None:
            # Allow keeping model metadata around for native-streaming runs, but do not require it.
            pass
        self.parser.validate_config()
        self.prompt.message_history.validate_config()
        self.prompt_context.validate_config()
        if (
            self.prompt_context.completed_tool_intervals.mode != "none"
            and self.prompt.state_format != "compact"
        ):
            raise ValueError(
                "prompt_context.completed_tool_intervals requires "
                "prompt.state_format='compact'."
            )
        self.frame_sampling.validate_config()
        self.output.validate_config()

        if self.dataset.dataset not in {"ProactBenchClips", "ProactBenchScienceClips", "ProactBenchGrouped"}:
            raise ValueError(
                "Evaluation requires dataset in {'ProactBenchClips', 'ProactBenchScienceClips'}. "
                f"Got '{self.dataset.dataset}'."
            )
        if self.dataset.dataset in {"ProactBenchClips", "ProactBenchScienceClips"}:
            self.parser.unified_schema = True
            if self.context != "short-video":
                raise ValueError("ProactBench trace generation requires context='short-video'.")
            if self.parser.allow_notification_raw_text and (
                self.proactive_level != "notification"
                or self.execution_backend.mode != "native-streaming"
            ):
                raise ValueError(
                    "parser.allow_notification_raw_text is only valid for "
                    "notification native-streaming clip traces."
                )

        if not self.task_policy.expected_interval_types:
            raise ValueError("Task policy must define expected_interval_types.")
        if not self.task_policy.expected_start_tools:
            raise ValueError("Task policy must define expected_start_tools.")
        if not self.task_policy.expected_end_tools:
            raise ValueError("Task policy must define expected_end_tools.")
        if not self.task_policy.allowed_tool_names:
            raise ValueError("Task policy must define allowed_tool_names.")
        if self.dataset.dataset == "ProactBenchGrouped":
            if self.metrics is None:
                raise ValueError("Legacy grouped evaluation requires metrics.")
            if not self.metrics.proactive_start_tolerances_sec:
                raise ValueError("Metric config must define proactive_start_tolerances_sec.")
            if not self.metrics.iou_thresholds:
                raise ValueError("Metric config must define iou_thresholds.")

        prompt_path = Path(self.prompt.prompt_path)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        for tool_name, template in self.prompt.tool_state_templates.items():
            template.validate_template(tool_name)

        known_policy_tools = set(self.task_policy.allowed_tool_names).union(
            self.task_policy.ignored_tool_names
        )
        unknown_templates = sorted(
            set(self.prompt.tool_state_templates.keys()).difference(known_policy_tools)
        )
        if unknown_templates:
            raise ValueError(
                "prompt.tool_state_templates contains tool names outside the active task policy: "
                f"{unknown_templates}"
            )

        if self.prompt.state_format == "active-only":
            expected_tools = set(self.task_policy.expected_start_tools).union(
                self.task_policy.expected_end_tools
            )
            missing = sorted(expected_tools.difference(self.prompt.tool_state_templates.keys()))
            if missing:
                raise ValueError(
                    "prompt.state_format='active-only' requires tool_state_templates for "
                    f"all expected start/end tools. Missing: {missing}"
                )

        return self


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[return-value]
    return model.dict()  # type: ignore[return-value]


_ENV_REFERENCE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    config_path = Path(path).expanduser()
    text = config_path.read_text(encoding="utf-8")
    referenced_variables = set(_ENV_REFERENCE_RE.findall(text))
    missing_variables = sorted(
        variable for variable in referenced_variables if variable not in os.environ
    )
    if missing_variables:
        names = ", ".join(missing_variables)
        raise ValueError(
            f"Evaluation config '{config_path}' references unset environment variables: "
            f"{names}. Export them before running or replace those placeholders with "
            "explicit paths."
        )

    cfg = EvaluationConfig(**read_yaml(config_path))
    return cfg.validate_config()
