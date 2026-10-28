# SPDX-License-Identifier: Apache-2.0
"""Run MineDraft on Ascend, or compare its greedy tokens with target-only vLLM.

Each mode runs in a separate process because MineDraft patches vLLM globally.
The parent enforces a timeout and terminates the process group on failure.
See docs/source/user_guide/feature_guide/minedraft.md for setup and limitations.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Target model path or ID")
    parser.add_argument("--draft-model", required=True,
                        help="Draft model with the same tokenizer/vocabulary")
    parser.add_argument("--mode", choices=("check", "parallel", "baseline"),
                        default="check")
    parser.add_argument("--target-tp", type=int, default=1,
                        help="Target ranks; parallel mode needs one extra NPU")
    parser.add_argument("--num-speculative-tokens", type=int, default=3)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--dtype", choices=("auto", "half", "bfloat16"),
                        default="half")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--timeout", type=float, default=1800,
                        help="Seconds per mode, including loading and shutdown")
    parser.add_argument("--results-dir", type=Path,
                        default=Path("minedraft-results"))
    parser.add_argument("--worker-mode", choices=("parallel", "baseline"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("target_tp", "num_speculative_tokens", "max_num_seqs",
                 "max_model_len", "max_tokens", "rounds", "timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_model_len < args.max_tokens + 256:
        parser.error("--max-model-len must be at least --max-tokens + 256")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    return args


def workloads(max_num_seqs):
    """Exercise both sub-batches, queueing, early finishes, and a single tail."""
    prompts = [
        "The capital of France is",
        "Write one sentence about a red apple:",
        "Explain why the sky appears blue in a few sentences:",
        "A library has books on mathematics, history, music and science. " * 12
        + "Summarize the library in one sentence:",
    ]
    for label, size in (("single", 1), ("queued", 2 * max_num_seqs + 1),
                        ("tail", max_num_seqs + 1)):
        yield label, [prompts[i % len(prompts)] for i in range(size)]


def run_engine(args):
    # Set before importing vLLM/torch: workers inherit these values.
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_ASCEND_ENABLE_MINEDRAFT"] = (
        "1" if args.worker_mode == "parallel" else "0")
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    plugins = ["ascend", "ascend_enhanced_model"]
    if args.worker_mode == "parallel":
        plugins.append("minedraft")
    os.environ["VLLM_PLUGINS"] = ",".join(plugins)

    from vllm import LLM, SamplingParams

    speculative_config = None
    if args.worker_mode == "parallel":
        speculative_config = {
            "model": args.draft_model,
            "draft_tensor_parallel_size": 1,
            "num_speculative_tokens": args.num_speculative_tokens,
            "is_parallel": True,
            "disable_mqa_scorer": True,
            "force_pearl": False,
        }

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.target_tp + (speculative_config is not None),
        distributed_executor_backend="mp",
        dtype=args.dtype,
        seed=0,
        enforce_eager=True,
        disable_async_output_proc=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        speculative_config=speculative_config,
    )
    records = []
    start = time.perf_counter()
    try:
        for round_id in range(args.rounds):
            for label, prompts in workloads(args.max_num_seqs):
                params = [
                    SamplingParams(
                        temperature=0,
                        max_tokens=(args.max_tokens if label == "single"
                                    else (1, min(args.num_speculative_tokens + 1,
                                                 args.max_tokens),
                                          args.max_tokens)[i % 3]),
                        ignore_eos=True,
                    ) for i in range(len(prompts))
                ]
                outputs = llm.generate(prompts, params, use_tqdm=False)
                for i, output in enumerate(outputs):
                    records.append({
                        "case": f"{round_id}/{label}/{i}",
                        "token_ids": list(output.outputs[0].token_ids),
                        "text": output.outputs[0].text,
                    })
                print(f"{args.worker_mode}: round {round_id}, {label}: "
                      f"{len(outputs)} requests finished", flush=True)
        result = {"mode": args.worker_mode, "records": records,
                  "generation_seconds": time.perf_counter() - start}
        args.results_dir.mkdir(parents=True, exist_ok=True)
        (args.results_dir / f"{args.worker_mode}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        # The outer timeout also covers shutdown, including a stuck collective.
        llm.llm_engine.model_executor.shutdown()


def stop_process_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # A child can survive its parent; terminate the entire isolated group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_isolated(args, mode):
    environment = os.environ.copy()
    visible = environment.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible:
        devices = [device.strip() for device in visible.split(",")]
        required = args.target_tp + 1
        if len(devices) < required:
            raise ValueError(f"Select at least {required} NPUs in "
                             "ASCEND_RT_VISIBLE_DEVICES; the first is draft")
        # Run baseline on exactly the same physical cards as the target scorer.
        environment["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(
            devices[1:required] if mode == "baseline" else devices[:required])
    else:
        environment["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(
            str(rank) for rank in range(1 if mode == "baseline" else 0,
                                       args.target_tp + 1))
    command = [sys.executable, str(Path(__file__).resolve()),
               *sys.argv[1:], "--worker-mode", mode]
    process = subprocess.Popen(command, env=environment, start_new_session=True)
    try:
        returncode = process.wait(timeout=args.timeout)
        if returncode:
            stop_process_group(process)
            raise RuntimeError(f"{mode} exited with status {returncode}")
    except BaseException:
        stop_process_group(process)
        raise


def compare_results(results_dir):
    results = [json.loads((results_dir / f"{mode}.json").read_text(
        encoding="utf-8"))["records"] for mode in ("baseline", "parallel")]
    if len(results[0]) != len(results[1]):
        raise AssertionError("Baseline and parallel request counts differ")
    for baseline, parallel in zip(*results):
        if (baseline["case"] != parallel["case"]
                or baseline["token_ids"] != parallel["token_ids"]):
            raise AssertionError(
                f"Greedy token mismatch at {baseline['case']}:\n"
                f"baseline={baseline['token_ids']}\n"
                f"parallel={parallel['token_ids']}")
    print(f"PASS: {len(results[0])} requests match token for token.")


def main():
    args = parse_args()
    if args.worker_mode:
        run_engine(args)
        return
    if sys.platform != "linux":
        raise SystemExit("This NPU example requires Linux and torch-npu.")
    for mode in (("baseline", "parallel") if args.mode == "check"
                 else (args.mode,)):
        run_isolated(args, mode)
    if args.mode == "check":
        compare_results(args.results_dir)


if __name__ == "__main__":
    main()
