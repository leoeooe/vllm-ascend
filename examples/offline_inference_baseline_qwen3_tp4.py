# SPDX-License-Identifier: Apache-2.0
"""Qwen3 TP=4 baselines without loading the MineDraft plugin.

Default: Standard SD, Qwen3-32B TP=4 + Qwen3-0.6B TP=1 on four shared NPUs.
Use --mode ar for target-only autoregressive decoding. Run each benchmark in
a fresh Python process. Set TARGET_MODEL/DRAFT_MODEL to local model paths.
ASCEND_RT_VISIBLE_DEVICES accepts four target cards or the MineDraft layout
of five cards; in the latter case the first (dedicated draft) card is omitted.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from statistics import mean

# Configure plugins before importing vLLM/torch, including in spawned workers.
# Do not import the MineDraft example: importing it enables its global patches.
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "1,2,3,4")
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_ASCEND_ENABLE_MINEDRAFT"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_PLUGINS"] = "ascend,ascend_enhanced_model"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sd", "ar"), default="sd",
                        help="sd: Standard SD (MineDraft's main baseline); "
                        "ar: target only, without speculative decoding.")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--num-prompts", type=int, default=8,
                        help="Requests per run; repeat the eight questions.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--num-speculative-tokens", type=int, default=5,
                        help="Fixed proposal length k, used only in sd mode.")
    parser.add_argument("--ignore-eos", action="store_true",
                        help="Generate max-tokens even after EOS.")
    parser.add_argument("--print-outputs", action="store_true",
                        help="Print answers from the last measured run.")
    parser.add_argument("--output", type=Path,
                        help="Default: baseline_<mode>_qwen3_tp4_metrics.json")
    args = parser.parse_args()
    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be nonnegative")
    for name in ("runs", "num_prompts", "max_tokens", "max_num_seqs",
                 "max_model_len", "num_speculative_tokens"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.output is None:
        args.output = Path(f"baseline_{args.mode}_qwen3_tp4_metrics.json")
    return args


def configure_devices():
    devices = [value.strip() for value in
               os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")]
    if (len(devices) not in (4, 5) or len(set(devices)) != len(devices)
            or not all(devices)):
        raise ValueError("Select four distinct target NPUs, or five distinct "
                         "NPUs in MineDraft order: dedicated draft, target x4")
    if len(devices) == 5:
        print(f"沿用 MineDraft 卡号顺序：本次不使用专用草稿卡 {devices[0]}",
              flush=True)
    target_devices = devices[-4:]
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(target_devices)
    return devices, target_devices


def latency_summary(values):
    """Unweighted request latency statistics; percentiles interpolate linearly."""
    values = sorted(value for value in values if value is not None)
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None,
                "p99": None}

    def percentile(fraction):
        index = (len(values) - 1) * fraction
        lower = int(index)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (index - lower)

    return {"count": len(values), "mean": mean(values),
            "p50": percentile(0.50), "p95": percentile(0.95),
            "p99": percentile(0.99)}


def duration_ms(start, end):
    if start is None or end is None:
        return None
    elapsed = (end - start) * 1000
    return elapsed if math.isfinite(elapsed) and elapsed >= 0 else None


def request_metrics(output):
    completion = output.outputs[0]
    output_tokens = len(completion.token_ids)
    metrics = output.metrics
    arrival = getattr(metrics, "arrival_time", None)
    first = getattr(metrics, "first_token_time", None)
    finished = getattr(metrics, "finished_time", None)
    ttft = duration_ms(arrival, first)
    decode = duration_ms(first, finished)
    return {
        "request_id": output.request_id,
        "input_tokens": len(output.prompt_token_ids),
        "output_tokens": output_tokens,
        "finish_reason": completion.finish_reason,
        "e2e_ms": duration_ms(arrival, finished),
        "ttft_ms": ttft,
        "queue_ms": duration_ms(
            arrival, getattr(metrics, "first_scheduled_time", None)),
        "tpot_ms": (decode / (output_tokens - 1)
                    if output_tokens > 1 and ttft is not None
                    and decode is not None else None),
    }


def read_spec_counters(llm, mode):
    if mode == "ar":
        return None, "AR 模式未启用投机解码"
    try:
        sampler = llm.llm_engine.model_executor.driver_worker.worker \
            .spec_decode_sampler
        return {
            "verified_draft_tokens": int(sampler.num_draft_tokens),
            "accepted_candidate_tokens": int(
                sampler.num_accepted_tokens.item()),
            "sampler_emitted_tokens": int(sampler.num_emitted_tokens.item()),
        }, None
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def spec_metrics(before, after, k):
    before_values, before_error = before
    after_values, after_error = after
    if before_values is None or after_values is None:
        return {"available": False, "reason": before_error or after_error}
    delta = {name: after_values[name] - value
             for name, value in before_values.items()}
    verified = delta["verified_draft_tokens"]
    candidates = delta["accepted_candidate_tokens"]
    emitted = delta["sampler_emitted_tokens"]
    # Native vLLM 0.9.2 Top1Proposer uses lengths 0 or k. Only nonzero
    # proposals enter this sampler. Every verification row emits its accepted
    # prefix plus exactly one recovered/bonus token, before EOS truncation.
    # Thus emitted - verified/k recovers the causal accepted prefix count.
    if verified < 0 or verified % k:
        return {"available": False, "reason": "Invalid fixed-k counter delta"}
    prefix = emitted - verified // k
    if not 0 <= prefix <= candidates <= verified:
        return {"available": False, "reason": "Invalid sampler counter delta"}
    return {
        "available": True, **delta, "accepted_prefix_tokens": prefix,
        "prefix_count_method": "vllm_0.9.2_fixed_k: emitted - draft/k",
        "effective_acceptance_rate": prefix / verified if verified else None,
        "candidate_acceptance_rate": candidates / verified if verified else None,
    }


def summarize(requests, elapsed):
    input_tokens = sum(item["input_tokens"] for item in requests)
    output_tokens = sum(item["output_tokens"] for item in requests)
    return {
        "elapsed_s": elapsed, "num_requests": len(requests),
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "requests_per_s": len(requests) / elapsed,
        "output_tokens_per_s": output_tokens / elapsed,
        "total_tokens_per_s": (input_tokens + output_tokens) / elapsed,
        "latency_ms": {
            name: latency_summary(item[name] for item in requests)
            for name in ("e2e_ms", "ttft_ms", "tpot_ms", "queue_ms")
        },
    }


def print_summary(title, summary, speculation):
    print(f"\n=== {title} ===")
    print(f"请求数: {summary['num_requests']} | "
          f"推理耗时: {summary['elapsed_s']:.3f} s")
    print(f"输入/输出 token: {summary['input_tokens']} / "
          f"{summary['output_tokens']}")
    print(f"输出吞吐量: {summary['output_tokens_per_s']:.2f} token/s | "
          f"总 token 吞吐量: {summary['total_tokens_per_s']:.2f} token/s | "
          f"请求吞吐量: {summary['requests_per_s']:.3f} req/s")
    for name, label in (("e2e_ms", "请求完成延迟 E2E"),
                        ("ttft_ms", "首 token 延迟 TTFT"),
                        ("tpot_ms", "每输出 token 平均耗时 TPOT"),
                        ("queue_ms", "首次调度排队时间")):
        stats = summary["latency_ms"][name]
        if stats["count"]:
            print(f"{label} (ms): mean={stats['mean']:.2f}, "
                  f"p50={stats['p50']:.2f}, p95={stats['p95']:.2f}, "
                  f"p99={stats['p99']:.2f} "
                  f"[有效 {stats['count']}/{summary['num_requests']}]")
        else:
            print(f"{label}: N/A（没有可用的时间戳或输出长度不足）")
    if speculation["available"]:
        for key, label in (("effective_acceptance_rate", "有效草稿接受率"),
                           ("candidate_acceptance_rate", "逐位置草稿接受率")):
            rate = speculation[key]
            rate_text = f"{rate:.2%}" if rate is not None else "N/A（无草稿验证）"
            print(f"{label}: {rate_text}")
        print(f"接受前缀/验证 token: {speculation['accepted_prefix_tokens']} / "
              f"{speculation['verified_draft_tokens']}")
    else:
        print(f"草稿接受率: N/A（{speculation['reason']}）")
    print(flush=True)


def main():
    args = parse_args()
    if any(name == "minedraft" or name.startswith("minedraft.")
           for name in sys.modules):
        raise RuntimeError("Run this baseline in a fresh Python process; "
                           "MineDraft patches cannot be unloaded safely")
    requested_devices, devices = configure_devices()

    from vllm import LLM, SamplingParams

    target_model = os.environ.get("TARGET_MODEL", "Qwen/Qwen3-32B")
    draft_model = (os.environ.get("DRAFT_MODEL", "Qwen/Qwen3-0.6B")
                   if args.mode == "sd" else None)
    mode_label = "Standard SD" if args.mode == "sd" else "AR（仅目标模型）"
    print(f"Baseline: {mode_label}; target TP=4 NPUs: {devices}", flush=True)
    if args.mode == "sd":
        print(f"Draft TP=1，与目标模型共享 NPU {devices[0]}", flush=True)
    speculative_config = None
    if args.mode == "sd":
        speculative_config = {
            "model": draft_model,
            "method": "draft_model",
            "draft_tensor_parallel_size": 1,
            "num_speculative_tokens": args.num_speculative_tokens,
            "disable_mqa_scorer": True,
            "disable_logprobs": True,
            "acceptance_method": "rejection_sampler",
            # Native vLLM has no is_parallel / force_pearl / tetris fields.
        }

    load_start = time.perf_counter()
    llm = LLM(
        model=target_model,
        tensor_parallel_size=4,
        distributed_executor_backend="mp",
        dtype="bfloat16",
        seed=0,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        disable_async_output_proc=True,
        disable_log_stats=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=0.70,
        speculative_config=speculative_config,
    )
    try:
        load_elapsed = time.perf_counter() - load_start
        print(f"模型及引擎初始化: {load_elapsed:.3f} s（不计入推理耗时）",
              flush=True)
        questions = [
            "用三句话解释什么是机器学习。",
            "解释大语言模型中 KV Cache 的作用。",
            "解释为什么投机解码可以加速推理。",
            "用一个例子解释张量并行。",
            "写一个 Python 函数判断整数是否为素数。",
            "比较 CPU 和 GPU 各自适合的工作。",
            "介绍二分查找的步骤和时间复杂度。",
            "用通俗语言解释 Transformer 的注意力机制。",
        ]
        questions = [questions[index % len(questions)]
                     for index in range(args.num_prompts)]
        tokenizer = llm.get_tokenizer()
        prompts = [
            {"prompt_token_ids": tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False,
            )}
            for question in questions
        ]
        if any(len(prompt["prompt_token_ids"]) + args.max_tokens
               > args.max_model_len for prompt in prompts):
            raise ValueError("Prompt + max-tokens exceeds max-model-len")
        sampling_params = SamplingParams(
            temperature=0, max_tokens=args.max_tokens,
            ignore_eos=args.ignore_eos)
        for index in range(args.warmup_runs):
            print(f"预热 {index + 1}/{args.warmup_runs}...", flush=True)
            llm.generate(prompts, sampling_params, use_tqdm=False)

        runs = []
        all_requests = []
        benchmark_before = read_spec_counters(llm, args.mode)
        before = benchmark_before
        for index in range(args.runs):
            print(f"正式测试 {index + 1}/{args.runs}...", flush=True)
            start = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            elapsed = time.perf_counter() - start
            # Completed CPU-visible outputs delimit timing. Read counters
            # outside it, without resetting the engine's cumulative counters.
            after = read_spec_counters(llm, args.mode)
            if len(outputs) != args.num_prompts or any(
                    not output.finished or len(output.outputs) != 1
                    for output in outputs):
                raise RuntimeError("Expected one finished output per request")
            requests = [request_metrics(output) for output in outputs]
            summary = summarize(requests, elapsed)
            speculation = spec_metrics(before, after, args.num_speculative_tokens)
            runs.append({"run": index + 1, "summary": summary,
                         "speculation": speculation, "requests": requests})
            all_requests.extend(requests)
            print_summary(f"{mode_label} 第 {index + 1} 轮", summary, speculation)
            before = after

        summary = summarize(all_requests,
                            sum(run["summary"]["elapsed_s"] for run in runs))
        speculation = spec_metrics(benchmark_before, after,
                                   args.num_speculative_tokens)
        print_summary(f"{mode_label} 汇总（不含预热）", summary, speculation)
        notes = [
            "计时仅覆盖 generate()，包含请求提交、排队、prefill、decode 和输出处理；"
            "排除初始化、外部 chat template 分词、预热、打印和指标读取。",
            "吞吐量按各轮实际返回 token 总数除以推理总耗时计算，为整批离线吞吐量。",
            "TTFT/E2E 包含引擎排队时间；TPOT=(完成时间-首 token 时间)/(输出 token 数-1)，"
            "其百分位针对每请求均值，不是逐 token 间隔。缺失指标为 null。",
            "SD 有效接受前缀数由原生固定 k 计数推导：采样器输出数-验证草稿数/k；"
            "有效接受率=接受前缀数/验证草稿数。逐位置接受率包含首次拒绝后的接受位置。",
            "接受率不含预热；验证计数可能包含最终因 EOS 或长度上限被截断的 token。"
            "AR 没有草稿接受率。",
            "本脚本禁用 MineDraft 插件；SD 为原生顺序投机，草稿与目标共享四卡。"
            "MineDraft 使用四张目标卡和一张专用草稿卡，资源数量不同。",
            "沿用相同问题、greedy 采样和请求参数进行对比；MineDraft 会将调度容量"
            "扩展为两组交错批次，相同 max-num-seqs 不表示相同的总调度容量。",
        ]
        report = {
            "schema_version": 1,
            "config": {
                **{key: value for key, value in vars(args).items()
                   if key != "output"},
                "target_model": target_model, "draft_model": draft_model,
                "requested_visible_devices": requested_devices,
                "visible_devices": devices, "target_tp": 4,
                "draft_tp": 1 if args.mode == "sd" else None,
                "executor_world_size": 4, "dtype": "bfloat16",
                "gpu_memory_utilization": 0.70, "seed": 0, "temperature": 0,
                "minedraft_enabled": False,
                "speculative_config": speculative_config,
            },
            "model_load_s": load_elapsed,
            "summary": summary, "speculation": speculation,
            "runs": runs, "notes": notes,
        }
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                          allow_nan=False) + "\n",
                               encoding="utf-8")
        for note in notes:
            print(note)
        print(f"\n完整指标已保存到: {output_path}", flush=True)
        if args.print_outputs:
            for question, output in zip(questions, outputs):
                print(f"\n问题：{question}\n回答：{output.outputs[0].text}")
    finally:
        llm.llm_engine.model_executor.shutdown()


if __name__ == "__main__":
    main()
