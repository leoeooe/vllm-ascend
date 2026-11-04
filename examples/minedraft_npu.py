# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Run independent draft/verification engines, optionally checking greedy IDs."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from copy import deepcopy
from pathlib import Path

import regex as re

PROMPTS = [
    "Explain how a rainbow forms in three sentences.",
    "Write a Python function that returns the sum of a list of integers.",
    "The capital of France is",
    "The first five prime numbers are",
    "用三句话解释什么是机器学习。",
    "计算17乘以23，并简要说明计算步骤。",
    "A triangle has three sides. A square has",
    "Translate 'good morning' into French:",
]
PROBE_CLASS = "vllm_ascend.worker.v2.minedraft.MineDraftWorkerExtension"
GSM8K_PROMPT_TEMPLATE = "Question: {question}\nAnswer:"


def load_inputs(dataset_path, offset, count):
    """Select raw completion prompts without exposing reference answers."""
    if count < 1 or offset < 0:
        raise ValueError("Prompt count must be positive and dataset offset nonnegative")
    if dataset_path is None:
        if offset or count > len(PROMPTS):
            raise ValueError(f"Built-in inputs support offset 0 and at most {len(PROMPTS)} prompts")
        return PROMPTS[:count], {"source": "built_in", "indices": list(range(count))}, []
    path = Path(dataset_path).resolve()
    source = path.read_bytes()
    rows = []
    for index, line in enumerate(source.decode("utf-8").splitlines()):
        if not line.strip():
            raise ValueError(f"Empty GSM8K record at line {index + 1}")
        row = json.loads(line)
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip() for key in ("question", "answer")
        ):
            raise ValueError(f"GSM8K record {index} must contain nonempty question and answer strings")
        if offset <= index < offset + count:
            rows.append({"index": index, "question": row["question"], "reference_answer": row["answer"]})
    if len(rows) != count:
        raise ValueError(f"GSM8K selection needs {count} records at offset {offset}, found {len(rows)}")
    metadata = {
        "source": "gsm8k_jsonl",
        "path": str(path),
        "sha256": hashlib.sha256(source).hexdigest(),
        "prompt_template": GSM8K_PROMPT_TEMPLATE,
        "chat_template_applied": False,
        "indices": [row["index"] for row in rows],
    }
    return [GSM8K_PROMPT_TEMPLATE.format(question=row["question"]) for row in rows], metadata, rows


def graph_capture_sizes(num_prompts, num_batches, num_speculative_tokens):
    if min(num_prompts, num_batches, num_speculative_tokens) < 1:
        raise ValueError("Prompt count, batch count, and speculative token count must be positive")
    batch_size = (num_prompts + num_batches - 1) // num_batches
    return sorted({1, 2, 4, 8, 16, 32, batch_size, batch_size * (num_speculative_tokens + 1)})


def snapshot_diagnostics(llm):
    """Freeze counters and the live controller timeline before generation."""
    return deepcopy(llm.get_diagnostics())


def measured_diagnostics(before, after):
    """Separate measured speculative counters from cumulative engine gauges."""
    after["timeline"] = after["timeline"][len(before["timeline"]) :]
    old_engines = [before["draft"], *before["verifiers"]]
    new_engines = [after["draft"], *after["verifiers"]]
    for old, new in zip(old_engines, new_engines, strict=True):
        for previous, current in zip(old["workers"], new["workers"], strict=True):
            current["graph_replays"] -= previous["graph_replays"]
        previous_metrics = {metric["name"]: metric for metric in old["metrics"]}
        new["cumulative_metrics"] = new["metrics"]
        new["metrics"] = []
        for metric in new["cumulative_metrics"]:
            if not metric["name"].startswith("vllm:spec_decode_"):
                continue
            delta = dict(metric)
            previous = previous_metrics.get(metric["name"], {})
            if "value" in metric:
                delta["value"] -= previous.get("value", 0)
            if "values" in metric:
                old_values = previous.get("values", [0] * len(metric["values"]))
                delta["values"] = [
                    value - old_value for value, old_value in zip(metric["values"], old_values, strict=True)
                ]
            new["metrics"].append(delta)
    return after


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--draft-devices", default="0")
    parser.add_argument("--verifier-devices", default="1,2;3,4", help="Semicolon-separated TP groups")
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--num-speculative-tokens", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, help="Default: 8 with profiling, otherwise 32")
    parser.add_argument("--num-prompts", type=int, help="Default: 4 with profiling, otherwise 8")
    parser.add_argument("--dataset-path", type=Path, help="Local GSM8K JSONL containing question and answer fields")
    parser.add_argument("--dataset-offset", type=int, default=0, help="First GSM8K record index, starting at zero")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument(
        "--max-num-batched-tokens", type=int, help="Scheduling token budget; defaults to --max-model-len"
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--kv-cache-memory-bytes", type=int, help="Explicit KV cache budget per worker, in bytes")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--check-baseline", action="store_true")
    parser.add_argument("--profile-dir", type=Path, help="Empty output directory for a short, timing-only trace")
    parser.add_argument("--profile-warmup-tokens", type=int, default=8)
    parser.add_argument(
        "--warmup-tokens", type=int, help="Optional unmeasured warmup; overrides profiling warmup length"
    )
    parser.add_argument("--report", type=Path, default=Path("minedraft-report.json"))
    parser.add_argument("--baseline-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.max_tokens is None:
        args.max_tokens = 8 if args.profile_dir else 32
    if args.num_prompts is None:
        args.num_prompts = 4 if args.profile_dir else len(PROMPTS)
    if args.num_prompts < 1 or (args.dataset_path is None and args.num_prompts > len(PROMPTS)):
        parser.error(f"--num-prompts must be positive; more than {len(PROMPTS)} requires --dataset-path")
    if args.dataset_offset < 0 or (args.dataset_path is None and args.dataset_offset):
        parser.error("--dataset-offset must be nonnegative and requires --dataset-path")
    if args.warmup_tokens is not None and args.warmup_tokens < 1:
        parser.error("--warmup-tokens must be positive")
    if args.max_num_batched_tokens is not None and args.max_num_batched_tokens < 1:
        parser.error("--max-num-batched-tokens must be positive")
    if args.profile_dir and (args.profile_warmup_tokens < 1 or args.num_prompts < args.num_batches):
        parser.error("Profiling requires positive warmup tokens and at least one prompt per batch")
    if args.check_baseline and (args.temperature != 0 or os.environ.get("VLLM_BATCH_INVARIANT") != "1"):
        parser.error(
            "--check-baseline requires --temperature 0 and VLLM_BATCH_INVARIANT=1 (with its operators installed)"
        )
    return args


def counter(engine, name):
    return sum(metric.get("value", 0) for metric in engine["metrics"] if metric["name"] == name)


def validate_diagnostics(diagnostics, graph_mode):
    for role, engine in [("draft", diagnostics["draft"]), *[("verifier", v) for v in diagnostics["verifiers"]]]:
        if not engine["workers"]:
            raise AssertionError(f"Missing worker probes for {role}")
        for worker in engine["workers"]:
            if role == "verifier":
                assert worker["speculator"] == "ExternalSpeculator", worker
                assert not worker["has_local_draft_model"], worker
            if graph_mode:
                assert worker["captured_graphs"] > 0, worker
                assert worker["graph_replays"] > 0, worker
            else:
                assert worker["graph_replays"] == 0, worker
        if role == "verifier":
            assert counter(engine, "vllm:spec_decode_num_drafts") > 0, engine


def save_profile_artifacts(args, draft_devices, verifier_devices, report):
    """Keep native traces and package a compact timeline with unique worker lanes."""
    from minedraft_profile_utils import merge_worker_traces
    from torch_npu.profiler.profiler import analyse

    directory = args.profile_dir.resolve()
    # TP workers are daemon processes and cannot spawn the offline parsers.
    # All model engines have stopped before this non-daemon parent parses data.
    native_directories = list((directory / "native").glob("*_ascend_pt"))
    if any(not (path / "ASCEND_PROFILER_OUTPUT/trace_view.json").exists() for path in native_directories):
        analyse(str(directory / "native"), max_process_number=1)
    groups = {"draft": draft_devices, **{f"verifier{i}": devices for i, devices in enumerate(verifier_devices)}}
    sources = {}
    for path in sorted((directory / "native").glob("*_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json")):
        match = re.match(r"minedraft_(draft|verifier\d+)_", path.parent.parent.name)
        if match is None:
            raise RuntimeError(f"Unrecognized worker trace: {path}")
        info_paths = list(path.parent.parent.glob("profiler_info*.json"))
        if len(info_paths) != 1:
            raise RuntimeError(f"Missing or ambiguous profiler metadata: {path}")
        role = match[1]
        rank = int(json.loads(info_paths[0].read_text())["rank_id"])
        devices = groups[role]
        label = f"{role}.tp{rank}.npu{devices[rank]}"
        if label in sources:
            raise RuntimeError(f"Multiple profiling sessions for {label}; use an empty directory")
        sources[label] = path
    if len(sources) != sum(map(len, groups.values())):
        raise RuntimeError(f"Missing worker traces in {directory}; inspect profiler warnings")
    manifest = merge_worker_traces(list(sources.items()), directory / "minedraft_timeline.json", device_only=True)
    manifest.update(
        target_model=args.target,
        draft_model=args.draft,
        devices=groups,
        num_batches=args.num_batches,
        num_prompts=args.num_prompts,
        max_tokens=args.max_tokens,
        num_speculative_tokens=args.num_speculative_tokens,
        warmup_tokens=report["measurement"]["warmup_tokens"],
        dataset=report["dataset"],
        profiling=report["profiling"],
        settings={
            "activities": ["NPU"],
            "profiler_level": "Level0",
            "aic_metrics": "AiCoreNone",
            "export_type": "Text",
        },
        note="Original timestamps retained. This is a timing trace, not a throughput benchmark.",
    )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    archive = directory / "mindstudio_timing.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(directory / "minedraft_timeline.json", "minedraft_timeline.json")
        bundle.write(directory / "manifest.json", "manifest.json")
        bundle.write(args.report, "generation-report.json")
    # Keep the native, unmodified files in a separate optional download.
    with zipfile.ZipFile(directory / "native_timelines.zip", "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for label, path in sources.items():
            bundle.write(path, f"{label}/trace_view.json")
    print(f"MindStudio timing bundle: {archive} ({archive.stat().st_size / 1024**2:.2f} MiB)")


def main():
    args = parse_args()
    prompt_texts, dataset, records = load_inputs(args.dataset_path, args.dataset_offset, args.num_prompts)
    draft_devices = [int(device) for device in args.draft_devices.split(",")]
    verifier_devices = [[int(device) for device in group.split(",")] for group in args.verifier_devices.split(";")]

    if args.baseline_output:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, verifier_devices[0]))
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    # The controller never creates an NPU context. Actor imports follow device selection.
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    prompts = [tokenizer.encode(prompt) for prompt in prompt_texts]
    warmup_tokens = args.warmup_tokens or (args.profile_warmup_tokens if args.profile_dir else 0)
    if any(len(prompt) + max(args.max_tokens, warmup_tokens) > args.max_model_len for prompt in prompts):
        raise ValueError("A prompt plus generation/warmup length exceeds --max-model-len")
    params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, ignore_eos=True, seed=42)
    batch_invariant = os.environ.get("VLLM_BATCH_INVARIANT") == "1"
    engine_kwargs = {
        "max_model_len": args.max_model_len,
        "max_num_seqs": len(prompts),
        "max_num_batched_tokens": args.max_num_batched_tokens or args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_prefix_caching": not batch_invariant,
        "enable_chunked_prefill": False,
        "compilation_config": {
            "cudagraph_mode": "NONE" if args.eager else "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": graph_capture_sizes(
                args.num_prompts, args.num_batches, args.num_speculative_tokens
            ),
        },
    }
    if batch_invariant:
        engine_kwargs["block_size"] = 128
    if args.kv_cache_memory_bytes is not None:
        engine_kwargs["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes

    if args.baseline_output:
        from vllm import LLM

        llm = LLM(
            model=args.target,
            tensor_parallel_size=len(verifier_devices[0]),
            distributed_executor_backend="mp",
            enforce_eager=args.eager,
            async_scheduling=False,
            **engine_kwargs,
        )
        try:
            outputs = llm.generate([{"prompt_token_ids": ids} for ids in prompts], params, use_tqdm=False)
            args.baseline_output.write_text(json.dumps([list(output.outputs[0].token_ids) for output in outputs]))
        finally:
            llm.llm_engine.engine_core.shutdown()
        return

    baseline = None
    if args.check_baseline:
        with tempfile.TemporaryDirectory(prefix="minedraft-baseline-") as directory:
            baseline_path = Path(directory) / "baseline.json"
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--baseline-output", str(baseline_path)],
                check=True,
                timeout=args.timeout,
            )
            baseline = json.loads(baseline_path.read_text())

    from vllm.entrypoints.minedraft import MineDraftLLM

    if args.profile_dir:
        args.profile_dir = args.profile_dir.resolve()
        if args.profile_dir.exists() and any(args.profile_dir.iterdir()):
            raise ValueError("--profile-dir must be empty to avoid mixing separate capture sessions")
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        engine_kwargs["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(args.profile_dir / "native"),
            "torch_profiler_with_stack": False,
            "torch_profiler_with_memory": False,
            "torch_profiler_record_shapes": False,
            "torch_profiler_with_flops": False,
            "capture_torch_profiler": False,
            "ignore_frontend": True,
        }
    profiling = None
    with MineDraftLLM(
        draft_model=args.draft,
        target_model=args.target,
        draft_devices=draft_devices,
        verifier_devices=verifier_devices,
        num_batches=args.num_batches,
        num_speculative_tokens=args.num_speculative_tokens,
        device="npu",
        enforce_eager=args.eager,
        timeout=args.timeout,
        probe_method="minedraft_probe",
        worker_extension_cls=PROBE_CLASS,
        **engine_kwargs,
    ) as llm:
        warmup_outputs = None
        if warmup_tokens:
            # Warm all graph/shape paths before starting any worker's collector.
            warmup_params = params.clone()
            warmup_params.max_tokens = warmup_tokens
            warmup_outputs = llm.generate(prompts, warmup_params)
        before = snapshot_diagnostics(llm)
        if args.profile_dir:
            llm.start_profile("minedraft", profiler_kwargs={"timing_only": True})
        measurement = {
            "started_wall_ns": time.time_ns(),
            "started_monotonic_ns": time.monotonic_ns(),
            "warmup_tokens": warmup_tokens,
            "timeline_scope": "measured_generation_only",
            "metrics_scope": "spec_decode counters and graph_replays exclude warmup; cumulative_metrics include warmup",
        }
        outputs = llm.generate(prompts, params)
        measurement["ended_monotonic_ns"] = time.monotonic_ns()
        measurement["ended_wall_ns"] = time.time_ns()
        measurement["elapsed_seconds"] = (measurement["ended_monotonic_ns"] - measurement["started_monotonic_ns"]) / 1e9
        if args.profile_dir:
            llm.stop_profile()
        diagnostics = measured_diagnostics(before, llm.get_diagnostics())
        measurement["graph_replays_by_engine_and_rank"] = [
            [worker["graph_replays"] for worker in engine["workers"]]
            for engine in [diagnostics["draft"], *diagnostics["verifiers"]]
        ]
        if warmup_outputs is not None and args.temperature == 0 and batch_invariant:
            measurement["warmup_greedy_prefix_equal"] = all(
                list(old.outputs[0].token_ids)[: args.max_tokens] == list(new.outputs[0].token_ids)[:warmup_tokens]
                for old, new in zip(warmup_outputs, outputs, strict=True)
            )
        if args.profile_dir:
            profiling = dict(measurement)
    token_ids = [list(output.outputs[0].token_ids) for output in outputs]
    report = {
        "token_ids": token_ids,
        "baseline_equal": None if baseline is None else token_ids == baseline,
        "profiling": profiling,
        "measurement": measurement,
        "dataset": dataset,
        "configuration": {
            "target": args.target,
            "draft": args.draft,
            "draft_devices": draft_devices,
            "verifier_devices": verifier_devices,
            "num_batches": args.num_batches,
            "num_prompts": args.num_prompts,
            "num_speculative_tokens": args.num_speculative_tokens,
            "max_tokens": args.max_tokens,
            "batch_invariant": batch_invariant,
            "ignore_eos": True,
            "temperature": args.temperature,
            "seed": 42,
            "engine_kwargs": engine_kwargs,
        },
        "requests": [
            {
                **(records[index] if records else {"index": index}),
                "batch_id": index % args.num_batches,
                "prompt": text,
                "prompt_token_count": len(prompt),
                "generated_text": output.outputs[0].text,
                "generated_token_count": len(output.outputs[0].token_ids),
                "finish_reason": output.outputs[0].finish_reason,
            }
            for index, (text, prompt, output) in enumerate(zip(prompt_texts, prompts, outputs, strict=True))
        ],
        **diagnostics,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    validate_diagnostics(diagnostics, graph_mode=not args.eager)
    assert measurement.get("warmup_greedy_prefix_equal", True), "Warmup and measured greedy prefixes differ"
    if baseline is not None:
        assert token_ids == baseline, f"Greedy tokens differ; inspect {args.report}"
    if profiling is not None:
        assert profiling.get("warmup_greedy_prefix_equal", True), "Profiling changed greedy output tokens"
        if not args.eager:
            assert all(count > 0 for ranks in profiling["graph_replays_by_engine_and_rank"] for count in ranks)
        save_profile_artifacts(args, draft_devices, verifier_devices, report)
    for output in outputs:
        print(output.outputs[0].text)
    print(f"MineDraft smoke passed; diagnostics: {args.report}")


if __name__ == "__main__":
    main()
