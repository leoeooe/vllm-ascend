# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Run independent draft/verification engines, optionally checking greedy IDs."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--draft-devices", default="0")
    parser.add_argument("--verifier-devices", default="1,2;3,4", help="Semicolon-separated TP groups")
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--num-speculative-tokens", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--kv-cache-memory-bytes", type=int, help="Explicit KV cache budget per worker, in bytes")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--check-baseline", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("minedraft-report.json"))
    parser.add_argument("--baseline-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
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


def main():
    args = parse_args()
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
    prompts = [tokenizer.encode(prompt) for prompt in PROMPTS]
    params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, ignore_eos=True, seed=42)
    batch_invariant = os.environ.get("VLLM_BATCH_INVARIANT") == "1"
    engine_kwargs = {
        "max_model_len": args.max_model_len,
        "max_num_seqs": len(PROMPTS),
        "max_num_batched_tokens": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_prefix_caching": not batch_invariant,
        "enable_chunked_prefill": False,
        "compilation_config": {
            "cudagraph_mode": "NONE" if args.eager else "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32],
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
        outputs = llm.generate(prompts, params)
        diagnostics = llm.get_diagnostics()
    token_ids = [list(output.outputs[0].token_ids) for output in outputs]
    report = {
        "token_ids": token_ids,
        "baseline_equal": None if baseline is None else token_ids == baseline,
        **diagnostics,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    validate_diagnostics(diagnostics, graph_mode=not args.eager)
    if baseline is not None:
        assert token_ids == baseline, f"Greedy tokens differ; inspect {args.report}"
    for output in outputs:
        print(output.outputs[0].text)
    print(f"MineDraft smoke passed; diagnostics: {args.report}")


if __name__ == "__main__":
    main()
