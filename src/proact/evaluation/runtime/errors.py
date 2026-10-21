from __future__ import annotations


class ModelInferenceError(RuntimeError):
    """A model/provider/adapter failure during trace execution.

    Raised whenever an inference response is not OK (API error, rate limit,
    adapter crash, empty/blocked response classified as an error). The clip
    pipeline catches it like any other exception: the clip is retried up to
    ``max_clip_attempts`` and then recorded under ``failures/<clip_uid>.json``
    with this report. Errored output must never be parsed as model output or
    scored as silence.
    """

    def __init__(
        self,
        *,
        trace_id: str,
        step_index: int,
        timestamp_sec: float,
        error: str | None,
        error_type: str | None,
        status_code: int | None = None,
        provider: str | None = None,
    ) -> None:
        self.trace_id = trace_id
        self.step_index = step_index
        self.timestamp_sec = timestamp_sec
        self.error = error
        self.error_type = error_type
        self.status_code = status_code
        self.provider = provider
        super().__init__(
            f"Model inference failed on trace_id={trace_id} step_index={step_index} "
            f"timestamp_sec={timestamp_sec:.3f} provider={provider or 'unknown'} "
            f"error_type={error_type or 'unknown'} status_code={status_code}: {error}"
        )
