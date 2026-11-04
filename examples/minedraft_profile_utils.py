# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Combine same-host Ascend worker traces without changing their time origin."""

import json
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


def _number(value: Any, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"Invalid trace {field}: {value!r}")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid trace {field}: {value!r}") from exc
    if not number.is_finite():
        raise ValueError(f"Non-finite trace {field}: {value!r}")
    return number


def _identity(value: Any) -> tuple[type, int | str]:
    if type(value) not in (int, str):
        raise ValueError(f"Unsupported trace identifier: {value!r}")
    return type(value), value


def _remap_id(mapping: dict, source_index: int, kind: str, value: Any) -> int:
    key = source_index, kind, _identity(value)
    if key not in mapping:
        mapping[key] = len(mapping) + 1
    return mapping[key]


def _read_events(path: Path) -> tuple[list[dict[str, Any]], int]:
    raw = path.read_bytes()
    data = json.loads(raw)
    if isinstance(data, dict):
        for field in ("stackFrames", "samples", "systemTraceEvents", "baseTimeNanoseconds"):
            if data.get(field):
                raise ValueError(f"Cannot merge {field}; collect native Ascend traces with stack recording disabled")
        data = data.get("traceEvents")
    if not isinstance(data, list) or not data:
        raise ValueError(f"Trace has no events: {path}")
    if any(not isinstance(event, dict) or not isinstance(event.get("ph"), str) for event in data):
        raise ValueError(f"Invalid trace event format: {path}")
    return data, len(raw)


def _device_events(events: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    hardware_pids = {
        _identity(event.get("pid"))
        for event in events
        if event["ph"] == "M"
        and event.get("name") == "process_name"
        and isinstance(event.get("args"), dict)
        and event["args"].get("name") == "Ascend Hardware"
    }
    retained = [
        event
        for event in events
        if event["ph"] in ("M", "X") and "pid" in event and _identity(event["pid"]) in hardware_pids
    ]
    if not any(event["ph"] == "X" for event in retained):
        raise ValueError(f"Trace has no Ascend Hardware device X events: {path}")
    return retained


def merge_worker_traces(
    sources: Sequence[tuple[str, Path]], output_path: Path, *, device_only: bool = False
) -> dict[str, Any]:
    """Merge native ``trace_view.json`` files for a single MindStudio timeline.

    Sources must use the same calibrated clock, as same-host Ascend exports do.
    No timestamp, duration, thread ID, or kernel argument is normalized. Worker
    process/flow identifiers are isolated because independent TP groups reuse
    ranks and profiler identifiers. Stack-reference formats are rejected.

    ``device_only`` retains complete device X events and their M metadata,
    identified by the native ``Ascend Hardware`` process name. CPU and flow
    events are omitted together, leaving no partial CPU-to-device arrows.

    Returns an audit manifest with microsecond time bounds encoded as decimal
    strings, preserving the precision of Ascend's absolute timestamps.
    """
    if not sources:
        raise ValueError("At least one worker trace is required")
    output_path = Path(output_path)
    merged: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    next_pid = 1
    ids: dict[tuple[int, str, tuple[type, int | str]], int] = {}
    for source_index, (label, source_path) in enumerate(sources):
        path = Path(source_path).resolve()
        if not label or path in seen_paths or path == output_path.resolve():
            raise ValueError("Worker labels must be nonempty and input/output trace paths must be distinct")
        seen_paths.add(path)
        events, source_bytes = _read_events(path)
        source_event_count = len(events)
        if device_only:
            events = _device_events(events, path)
        pids: dict[tuple[type, int | str], int] = {}
        named_pids: set[int] = set()
        starts: list[Decimal] = []
        ends: list[Decimal] = []

        # Isolate identifiers, preserving the source clocks and native events.
        for original in events:
            if any(field in original for field in ("sf", "esf", "stack")):
                raise ValueError(f"Stack references are unsupported: {path}")
            event = dict(original)
            if "pid" in event:
                key = _identity(event["pid"])
                if key not in pids:
                    pids[key] = next_pid
                    next_pid += 1
                event["pid"] = pids[key]
            for field in ("id", "bind_id"):
                if field in event:
                    event[field] = _remap_id(ids, source_index, "id", event[field])
            if "id2" in event:
                id2 = event["id2"]
                if not isinstance(id2, dict) or len(id2) != 1 or not set(id2) <= {"local", "global"}:
                    raise ValueError(f"Invalid trace id2: {id2!r}")
                event["id2"] = {kind: hex(_remap_id(ids, source_index, kind, value)) for kind, value in id2.items()}
            if "scope" in event:
                event["scope"] = f"source{source_index}:{event['scope']}"
            if event["ph"] == "M" and event.get("name") == "process_name":
                if "pid" not in event or not isinstance(event.get("args"), dict):
                    raise ValueError(f"Invalid process-name metadata: {path}")
                event["args"] = {**event["args"], "name": f"{label} | {event['args'].get('name', '')}"}
                named_pids.add(event["pid"])
            if "ts" in event and event["ph"] != "M":
                start = _number(event["ts"], "ts")
                duration = _number(event.get("dur", 0), "dur")
                if duration < 0:
                    raise ValueError(f"Negative trace duration: {path}")
                starts.append(start)
                ends.append(start + duration)
            merged.append(event)
        if not starts:
            raise ValueError(f"Trace has no timed events: {path}")
        for (_, original_pid), pid in pids.items():
            if pid not in named_pids:
                merged.append(
                    {
                        "ph": "M",
                        "name": "process_name",
                        "pid": pid,
                        "tid": 0,
                        "args": {"name": f"{label} | pid {original_pid}"},
                    }
                )
        manifest.append(
            {
                "label": label,
                "trace": str(path),
                "input_bytes": source_bytes,
                "event_count": source_event_count,
                "retained_event_count": len(events),
                "process_count": len(pids),
                "start_us": str(min(starts)),
                "end_us": str(max(ends)),
                "span_us": str(max(ends) - min(starts)),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {"traceEvents": merged, "displayTimeUnit": "ms"}, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ),
        encoding="utf-8",
    )
    return {
        "sources": manifest,
        "trace": str(output_path.resolve()),
        "event_count": len(merged),
        "source_event_count": sum(source["event_count"] for source in manifest),
        "retained_event_count": sum(source["retained_event_count"] for source in manifest),
        "filter_mode": "device_only" if device_only else "all_events",
        "output_bytes": output_path.stat().st_size,
        "timestamps_rebased": False,
    }
