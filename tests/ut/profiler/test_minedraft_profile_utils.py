# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def merge_traces():
    source = Path(__file__).resolve().parents[3] / "examples/minedraft_profile_utils.py"
    spec = importlib.util.spec_from_file_location("minedraft_profile_utils_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.merge_worker_traces


def test_merge_preserves_common_clock_and_isolates_worker_identifiers(tmp_path, merge_traces):
    """Repeated TP ranks must not merge lanes or connect independent flows."""
    sources = []
    originals = []
    for index, label in enumerate(("draft npu12", "verifier0 tp0 npu8")):
        start = f"1793760000000{index}.123"
        events = [
            {"ph": "M", "name": "process_name", "pid": 0, "tid": 0, "ts": 0, "args": {"name": "NPU"}},
            {
                "ph": "X",
                "name": "kernel",
                "pid": 0,
                "tid": 7,
                "ts": start,
                "dur": 0.25,
                "bind_id": 3,
                "args": {"model_id": 42},
            },
            {"ph": "s", "name": "launch", "pid": 0, "tid": 7, "ts": start, "id": 3},
            {"ph": "f", "name": "launch", "pid": 1, "tid": 7, "ts": start, "id": 3},
            {"ph": "b", "name": "async", "pid": 0, "tid": 7, "ts": start, "id2": {"global": "0x8"}, "scope": "compute"},
            {"ph": "e", "name": "async", "pid": 1, "tid": 7, "ts": start, "id2": {"global": "0x8"}, "scope": "compute"},
        ]
        path = tmp_path / f"rank{index}.json"
        path.write_text(json.dumps(events if index == 0 else {"traceEvents": events}))
        sources.append((label, path))
        originals.extend(events)
    output = tmp_path / "merged.json"
    manifest = merge_traces(sources, output)
    result = json.loads(output.read_text())["traceEvents"]
    native = [event for event in result if not (event["ph"] == "M" and "pid 1" in event["args"]["name"])]
    assert [(e.get("ts"), e.get("dur"), e["tid"]) for e in native] == [
        (e.get("ts"), e.get("dur"), e["tid"]) for e in originals
    ]
    assert len({event["pid"] for event in native}) == 4
    flows = [event for event in native if "id" in event]
    assert flows[0]["id"] == flows[1]["id"] != flows[2]["id"] == flows[3]["id"]
    assert [event["bind_id"] for event in native if "bind_id" in event] == [flows[0]["id"], flows[2]["id"]]
    asynchronous = [event for event in native if "id2" in event]
    assert asynchronous[0]["id2"] == asynchronous[1]["id2"] != asynchronous[2]["id2"] == asynchronous[3]["id2"]
    assert asynchronous[0]["scope"] != asynchronous[2]["scope"]
    assert native[0]["args"]["name"] == "draft npu12 | NPU"
    assert native[1]["args"] == originals[1]["args"]
    assert native[6]["args"]["name"] == "verifier0 tp0 npu8 | NPU"
    assert [row["span_us"] for row in manifest["sources"]] == ["0.250", "0.250"]
    assert manifest["timestamps_rebased"] is False
    assert manifest["output_bytes"] == output.stat().st_size
    assert "\n" not in output.read_text()


@pytest.mark.parametrize(
    "data",
    [
        [],
        {},
        [{"ph": "X", "ts": "invalid"}],
        {"traceEvents": [{"ph": "X", "ts": 1, "sf": "frame"}]},
        {"traceEvents": [{"ph": "X", "ts": 1}], "stackFrames": {"frame": {}}},
    ],
)
def test_invalid_or_unmergeable_trace_does_not_publish_output(tmp_path, merge_traces, data):
    source = tmp_path / "bad.json"
    source.write_text(json.dumps(data))
    output = tmp_path / "merged.json"
    with pytest.raises(ValueError):
        merge_traces([("draft", source)], output)
    assert not output.exists()


def test_device_only_preserves_kernels_and_labels_without_cpu_or_dangling_flows(tmp_path, merge_traces):
    """Hardware identity comes from metadata, even when kernels have no cat."""
    kernel = {
        "ph": "X",
        "name": "MatMul",
        "pid": 2,
        "tid": 3,
        "ts": "17937600000001.125",
        "dur": 2.5,
        "args": {"Task Type": "AI_CORE", "Stream Id": 3, "Model Id": 42},
    }
    events = [
        {"ph": "M", "name": "process_name", "pid": 1, "tid": 0, "args": {"name": "Python"}},
        {"ph": "M", "name": "process_name", "pid": 2, "tid": 0, "args": {"name": "Ascend Hardware"}},
        {"ph": "M", "name": "thread_name", "pid": 2, "tid": 3, "args": {"name": "Stream 3"}},
        {"ph": "X", "name": "cpu_launch", "pid": 1, "tid": 3, "ts": 1, "dur": 20},
        {"ph": "s", "name": "launch", "pid": 1, "tid": 3, "ts": 10, "id": 7},
        {"ph": "f", "name": "launch", "pid": 2, "tid": 3, "ts": kernel["ts"], "id": 7},
        kernel,
    ]
    path = tmp_path / "native.json"
    path.write_text(json.dumps(events))
    output = tmp_path / "device.json"
    manifest = merge_traces([("draft npu14", path)], output, device_only=True)
    result = json.loads(output.read_text())["traceEvents"]
    assert len(result) == 3
    assert [event["ph"] for event in result] == ["M", "M", "X"]
    assert {event["pid"] for event in result} == {1}
    assert result[0]["args"]["name"] == "draft npu14 | Ascend Hardware"
    assert result[1]["args"] == {"name": "Stream 3"}
    assert {key: value for key, value in result[2].items() if key != "pid"} == {
        key: value for key, value in kernel.items() if key != "pid"
    }
    assert manifest["source_event_count"] == manifest["sources"][0]["event_count"] == 7
    assert manifest["retained_event_count"] == manifest["sources"][0]["retained_event_count"] == 3
    assert manifest["filter_mode"] == "device_only"
    assert manifest["sources"][0]["start_us"] == kernel["ts"]
    assert manifest["sources"][0]["span_us"] == "2.500"


@pytest.mark.parametrize("hardware_name", ["Python", "Ascend Hardware"])
def test_device_only_rejects_a_source_without_device_kernels(tmp_path, merge_traces, hardware_name):
    path = tmp_path / "empty_device.json"
    path.write_text(
        json.dumps(
            [
                {"ph": "M", "name": "process_name", "pid": 2, "tid": 0, "args": {"name": hardware_name}},
                {"ph": "X", "name": "cpu_only", "pid": 1, "tid": 3, "ts": 1, "dur": 20},
            ]
        )
    )
    output = tmp_path / "device.json"
    with pytest.raises(ValueError, match="no Ascend Hardware device X events"):
        merge_traces([("draft", path)], output, device_only=True)
    assert not output.exists()
