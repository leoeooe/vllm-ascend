# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.e2e.conftest import wait_until_npu_memory_free

MODEL = "Qwen/Qwen3-0.6B"


@pytest.mark.e2e_model(MODEL)
@pytest.mark.parametrize(("replicas", "eager", "num_batches"), [(1, False, 2), (2, False, 4), (2, True, 4)])
@wait_until_npu_memory_free(target_free_percentage=0.7)
def test_minedraft_matches_greedy_target_and_replays_graphs(tmp_path, replicas, eager, num_batches):
    """Exercise the public API with independent TP2 verifiers and TP1 draft."""
    needed_devices = 1 + 2 * replicas
    if torch.npu.device_count() < needed_devices:
        pytest.skip(f"MineDraft topology requires {needed_devices} visible NPUs")
    if importlib.util.find_spec("batch_invariant_ops") is None:
        pytest.skip("Exact cross-batch token comparison requires batch-invariant operators")
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    devices = [int(value) for value in visible.split(",")] if visible else list(range(torch.npu.device_count()))
    groups = [devices[1 + 2 * index : 3 + 2 * index] for index in range(replicas)]
    root = Path(__file__).resolve().parents[5]
    report_path = tmp_path / "report.json"
    command = [
        sys.executable,
        str(root / "examples/minedraft_npu.py"),
        "--target",
        MODEL,
        "--draft",
        MODEL,
        "--draft-devices",
        str(devices[0]),
        "--verifier-devices",
        ";".join(",".join(map(str, group)) for group in groups),
        "--num-batches",
        str(num_batches),
        "--max-tokens",
        "24",
        "--gpu-memory-utilization",
        "0.5",
        "--kv-cache-memory-bytes",
        str(1024**3),
        "--check-baseline",
        "--report",
        str(report_path),
    ]
    if eager:
        command.append("--eager")
    environment = {**os.environ, "VLLM_BATCH_INVARIANT": "1"}
    result = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=1800)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(report_path.read_text())
    assert report["baseline_equal"] is True
    assert len(report["verifiers"]) == replicas
    for verifier in report["verifiers"]:
        assert len(verifier["workers"]) == 2
        # The same model supplies drafts here, so an unused draft path must fail.
        assert (
            sum(
                metric.get("value", 0)
                for metric in verifier["metrics"]
                if metric["name"] == "vllm:spec_decode_num_accepted_tokens"
            )
            > 0
        )
