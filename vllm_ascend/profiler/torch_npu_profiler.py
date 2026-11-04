#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from contextlib import suppress
from typing import Any

import torch_npu
from vllm.config import ProfilerConfig
from vllm.profiler.wrapper import WorkerProfiler

from vllm_ascend.ascend_config import get_ascend_config


class TorchNPUProfilerWrapper(WorkerProfiler):
    """Subclass of vLLM ``WorkerProfiler`` that wires in ``torch_npu.profiler``."""

    def __init__(self, profiler_config: ProfilerConfig, trace_name: str) -> None:
        super().__init__(profiler_config)
        self._config = profiler_config
        self._trace_name = trace_name
        self._timing_only = False
        # torch_npu.profile finalizes itself on destruction even if it has not
        # started. Construct only when recording begins, so changing options or
        # cancelling a delayed session cannot finalize a different session.
        self.profiler: Any = None

    def set_profiler_kwargs(self, profiler_kwargs: dict | None) -> None:
        """Select a small timing trace without changing normal profiler defaults."""
        options = profiler_kwargs or {}
        unknown = options.keys() - {"timing_only"}
        if unknown:
            raise ValueError(f"Unsupported Ascend profiler options: {sorted(unknown)}")
        timing_only = options.get("timing_only", False)
        if type(timing_only) is not bool:
            raise ValueError("timing_only must be a bool")
        with self._lifecycle_lock:
            if self._active or self.profiler is not None:
                raise RuntimeError("Stop the active profiler before changing its options")
            self._timing_only = timing_only

    @staticmethod
    def _create_profiler(profiler_config: ProfilerConfig, trace_name: str, *, timing_only: bool = False) -> Any:
        if profiler_config.profiler != "torch":
            raise RuntimeError(f"Unrecognized profiler: {profiler_config.profiler}")
        if not profiler_config.torch_profiler_dir:
            raise RuntimeError("torch_profiler_dir cannot be empty.")
        msmonitor_use_daemon = False
        with suppress(RuntimeError):
            msmonitor_use_daemon = get_ascend_config().msmonitor_use_daemon
        if msmonitor_use_daemon:
            raise RuntimeError(
                "additional_config.msmonitor_use_daemon and torch profiler cannot be enabled at the same time."
            )

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=(
                torch_npu.profiler.ProfilerLevel.Level0 if timing_only else torch_npu.profiler.ProfilerLevel.Level1
            ),
            msprof_tx=False,
            aic_metrics=(
                torch_npu.profiler.AiCMetrics.AiCoreNone
                if timing_only
                else torch_npu.profiler.AiCMetrics.PipeUtilization
            ),
            l2_cache=False,
            op_attr=False,
            data_simplification=True,
            record_op_args=False,
            gc_detect_threshold=None,
        )

        trace_handler_kwargs: dict[str, Any] = {"worker_name": trace_name}
        if timing_only:
            # Engine workers can be daemon processes, which cannot spawn the
            # parser's subprocesses. The caller parses short captures offline.
            trace_handler_kwargs["analyse_flag"] = False

        return torch_npu.profiler.profile(
            activities=(
                [torch_npu.profiler.ProfilerActivity.NPU]
                if timing_only
                else [torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]
            ),
            record_shapes=False,
            with_stack=False,
            with_flops=False,
            profile_memory=False if timing_only else profiler_config.torch_profiler_with_memory,
            # NOTE: torch_npu.profiler.with_modules is equivalent to torch.profiler.with_stack.
            # The with_stack option in torch_npu.profiler introduces significant time overhead.
            with_modules=False if timing_only else profiler_config.torch_profiler_with_stack,
            experimental_config=experimental_config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                profiler_config.torch_profiler_dir,
                **trace_handler_kwargs,
            ),
        )

    def _start(self) -> None:
        if self.profiler is not None:
            raise RuntimeError("The previous profiler session has not stopped")
        self.profiler = self._create_profiler(self._config, self._trace_name, timing_only=self._timing_only)
        try:
            self.profiler.start()
        except Exception:
            self._stop()
            raise

    def _stop(self) -> None:
        if self.profiler is not None:
            self.profiler.stop()
            self.profiler = None

    def _profiler_step(self) -> bool:
        return True
