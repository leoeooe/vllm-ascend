# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def example():
    source = Path(__file__).resolve().parents[3] / "examples/minedraft_npu.py"
    spec = importlib.util.spec_from_file_location("minedraft_npu_inputs_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dataset_selection_exceeds_builtin_limit_without_answer_leakage(tmp_path, example):
    source = tmp_path / "test.jsonl"
    source.write_text("\n".join(json.dumps({"question": f"Question {i}?", "answer": f"Secret {i}"}) for i in range(40)))
    prompts, metadata, rows = example.load_inputs(source, 4, 32)
    assert prompts == [f"Question: Question {i}?\nAnswer:" for i in range(4, 36)]
    assert all("Secret" not in prompt for prompt in prompts)
    assert metadata["indices"] == list(range(4, 36))
    assert metadata["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert metadata["chat_template_applied"] is False
    assert rows[-1] == {"index": 35, "question": "Question 35?", "reference_answer": "Secret 35"}


@pytest.mark.parametrize("row", [{"question": "q"}, {"question": "", "answer": "a"}, ["q", "a"]])
def test_invalid_dataset_is_rejected_before_model_loading(tmp_path, example, row):
    source = tmp_path / "bad.jsonl"
    source.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="nonempty question and answer"):
        example.load_inputs(source, 0, 1)


def test_dataset_selection_never_repeats_records_to_fill_batch(tmp_path, example):
    source = tmp_path / "short.jsonl"
    source.write_text(json.dumps({"question": "q", "answer": "a"}))
    with pytest.raises(ValueError, match="needs 32 records.*found 1"):
        example.load_inputs(source, 0, 32)
    with pytest.raises(ValueError, match="at most 8"):
        example.load_inputs(None, 0, 32)


def test_capture_sizes_cover_full_batch_verification_and_uneven_split(example):
    assert example.graph_capture_sizes(32, 4, 5) == [1, 2, 4, 8, 16, 32, 48]
    assert 54 in example.graph_capture_sizes(33, 4, 5)
    assert example.graph_capture_sizes(4, 4, 3) == [1, 2, 4, 8, 16, 32]


def test_measured_diagnostics_exclude_warmup_without_subtracting_gauges(example):
    def engine(replays, drafts, accepted, gauge):
        return {
            "workers": [{"rank": 0, "graph_replays": replays, "captured_graphs": 3}],
            "metrics": [
                {"name": "vllm:spec_decode_num_drafts", "value": drafts},
                {"name": "vllm:spec_decode_num_accepted_tokens_per_pos", "values": accepted},
                {"name": "vllm:kv_cache_usage_perc", "value": gauge},
            ],
        }

    before = {"draft": engine(4, 0, [0, 0], 0), "verifiers": [engine(2, 3, [2, 1], 0.5)], "timeline": ["warmup"]}
    after = {
        "draft": engine(10, 0, [0, 0], 0),
        "verifiers": [engine(7, 8, [6, 3], 0.2)],
        "timeline": ["warmup", "measured"],
    }
    original_before = copy.deepcopy(before)
    result = example.measured_diagnostics(before, after)
    assert before == original_before
    assert result["timeline"] == ["measured"]
    assert result["draft"]["workers"][0]["graph_replays"] == 6
    verifier = result["verifiers"][0]
    assert verifier["workers"][0] == {"rank": 0, "graph_replays": 5, "captured_graphs": 3}
    assert verifier["metrics"] == [
        {"name": "vllm:spec_decode_num_drafts", "value": 5},
        {"name": "vllm:spec_decode_num_accepted_tokens_per_pos", "values": [4, 2]},
    ]
    assert verifier["cumulative_metrics"][-1]["value"] == 0.2


def test_measurement_snapshot_survives_live_timeline_appends(example):
    """MineDraft get_diagnostics returns its live timeline, not a frozen copy."""
    live_timeline = [{"kind": "draft", "batch_id": 0, "started": 1, "ended": 2}]

    def get_diagnostics():
        return {"timeline": live_timeline, "draft": {"workers": [], "metrics": []}, "verifiers": []}

    llm = SimpleNamespace(get_diagnostics=get_diagnostics)
    before = example.snapshot_diagnostics(llm)
    live_timeline.append({"kind": "verify", "batch_id": 0, "started": 3, "ended": 4})
    result = example.measured_diagnostics(before, llm.get_diagnostics())
    assert result["timeline"] == [{"kind": "verify", "batch_id": 0, "started": 3, "ended": 4}]
    assert len(live_timeline) == 2
    assert len(before["timeline"]) == 1
