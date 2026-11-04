# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import Mock

import pytest

from vllm_ascend.worker.v2.minedraft import MineDraftWorkerExtension


class Descriptor(NamedTuple):
    num_tokens: int


def test_graph_probe_counts_replay_not_capture_and_does_not_wrap_twice():
    """Capturing a graph must not masquerade as using it for inference."""
    descriptor = Descriptor(num_tokens=4)
    manager = SimpleNamespace(graphs={descriptor: object()}, run_fullgraph=Mock(return_value="output"))
    worker = MineDraftWorkerExtension()
    worker.rank = 0
    worker.model_runner = SimpleNamespace(cudagraph_manager=manager, device="npu:0", speculator=None)

    initial = worker.minedraft_probe()
    assert initial["captured_graphs"] == 1
    assert initial["capture_token_sizes"] == [4]
    assert initial["graph_replays"] == 0
    replay = manager.run_fullgraph
    worker.minedraft_probe()
    assert manager.run_fullgraph is replay
    assert manager.run_fullgraph(descriptor) == "output"
    assert worker.minedraft_probe()["graph_replays"] == 1


def test_failed_graph_replay_is_not_reported_as_success():
    manager = SimpleNamespace(graphs={}, run_fullgraph=Mock(side_effect=RuntimeError("replay failed")))
    worker = MineDraftWorkerExtension()
    worker.rank = 0
    worker.model_runner = SimpleNamespace(cudagraph_manager=manager, device="npu:0", speculator=None)

    worker.minedraft_probe()
    with pytest.raises(RuntimeError, match="replay failed"):
        manager.run_fullgraph(None)
    assert worker.minedraft_probe()["graph_replays"] == 0
