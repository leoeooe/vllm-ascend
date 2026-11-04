# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from functools import wraps
from typing import Any


class MineDraftWorkerExtension:
    """Opt-in graph diagnostics for the independent MineDraft engines.

    Call ``minedraft_probe`` once before generation to start counting actual
    graph replays. Captured graphs alone do not prove that inference used them.
    The counter runs in Python around replay, outside the captured NPU graph.
    """

    model_runner: Any
    rank: int

    def minedraft_probe(self) -> dict[str, Any]:
        runner = self.model_runner
        manager = runner.cudagraph_manager
        if not hasattr(self, "_minedraft_graph_replays"):
            self._minedraft_graph_replays = 0
            run_fullgraph = manager.run_fullgraph

            @wraps(run_fullgraph)
            def counted_replay(*args, **kwargs):
                result = run_fullgraph(*args, **kwargs)
                self._minedraft_graph_replays += 1
                return result

            manager.run_fullgraph = counted_replay

        graphs = manager.graphs
        speculator = runner.speculator
        return {
            "rank": self.rank,
            "device": str(runner.device),
            "model_runner": type(runner).__name__,
            "speculator": type(speculator).__name__ if speculator is not None else None,
            "has_local_draft_model": hasattr(speculator, "model"),
            "captured_graphs": len(graphs),
            "capture_token_sizes": sorted({desc.num_tokens for desc in graphs}),
            "graph_replays": self._minedraft_graph_replays,
        }
