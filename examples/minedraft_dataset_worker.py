# SPDX-License-Identifier: Apache-2.0
"""Isolated NPU worker for the MineDraft dataset reproduction benchmark.

This module intentionally imports only the standard library at module scope.
Run one job in a fresh process; MineDraft patches cannot be unloaded between
modes. The parent owns dataset preparation, timeouts, and cross-mode checks.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from statistics import mean


def _write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def _versions():
    versions = {"python": platform.python_version(),
                "platform": platform.platform()}
    for package in ("vllm", "vllm-ascend", "minedraft", "torch", "torch-npu",
                    "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _validate_job(job, bundle):
    if job["mode"] not in ("ar", "sd", "minedraft"):
        raise ValueError("Unknown benchmark mode")
    if job["phase"] not in ("correctness", "performance"):
        raise ValueError("Unknown benchmark phase")
    for name in ("target_tp", "max_model_len", "max_num_seqs", "max_tokens",
                 "runs", "check_num_prompts", "check_max_tokens"):
        value = job[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(job["warmup_runs"], int) or job["warmup_runs"] < 0:
        raise ValueError("warmup_runs must be nonnegative")
    if job["mode"] != "ar" and (not isinstance(job["k"], int) or job["k"] < 1):
        raise ValueError("Speculative modes require positive k")
    if job["mode"] == "ar" and job["k"] != 0:
        raise ValueError("AR mode requires k=0")
    for name in ("temperature", "top_p", "gpu_memory_utilization"):
        if not math.isfinite(job[name]):
            raise ValueError(f"{name} must be finite")
    if job["temperature"] < 0 or not 0 < job["top_p"] <= 1:
        raise ValueError("Invalid sampling parameters")
    if not 0 < job["gpu_memory_utilization"] < 1:
        raise ValueError("gpu_memory_utilization must be between 0 and 1")
    devices = job["visible_devices"]
    required = job["target_tp"] + (job["mode"] == "minedraft")
    if (len(devices) != required or len(set(devices)) != required
            or any(not isinstance(value, str) or not value.strip()
                   for value in devices)):
        raise ValueError(f"Expected {required} distinct visible NPU IDs")
    datasets = bundle["datasets"]
    if not datasets or len({item["name"] for item in datasets}) != len(datasets):
        raise ValueError("Datasets must be nonempty and have unique names")
    limit = (job["check_max_tokens"] if job["phase"] == "correctness"
             else job["max_tokens"])
    for dataset in datasets:
        records = dataset["records"]
        if not records or len({row["id"] for row in records}) != len(records):
            raise ValueError(f"{dataset['name']}: empty or duplicate record IDs")
        for record in records:
            tokens = record["prompt_token_ids"]
            if (not isinstance(tokens, list) or not tokens
                    or any(not isinstance(token, int) or isinstance(token, bool)
                           or token < 0 for token in tokens)):
                raise ValueError(f"Invalid prompt tokens for {record['id']}")
            if len(tokens) + limit > job["max_model_len"]:
                raise ValueError(f"Prompt + output exceeds max_model_len: "
                                 f"{dataset['name']}/{record['id']}")


def _configure_environment(job):
    if any(name == "minedraft" or name.startswith("minedraft.")
           for name in sys.modules):
        raise RuntimeError("Run each benchmark job in a fresh Python process")
    # Workers inherit these values. No vLLM/torch import may precede this call.
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(job["visible_devices"])
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_ASCEND_ENABLE_MINEDRAFT"] = (
        "1" if job["mode"] == "minedraft" else "0")
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    plugins = ["ascend", "ascend_enhanced_model"]
    if job["mode"] == "minedraft":
        plugins.append("minedraft")
    os.environ["VLLM_PLUGINS"] = ",".join(plugins)
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"


def _latency_summary(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None,
                "p99": None}

    def percentile(fraction):
        position = (len(values) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return (values[lower]
                + (values[upper] - values[lower]) * (position - lower))

    return {"count": len(values), "mean": mean(values),
            "p50": percentile(0.50), "p95": percentile(0.95),
            "p99": percentile(0.99)}


def _duration_ms(start, end):
    if start is None or end is None:
        return None
    value = (end - start) * 1000
    return value if math.isfinite(value) and value >= 0 else None


def _request_metrics(record, output):
    completion = output.outputs[0]
    metrics = output.metrics
    arrival = getattr(metrics, "arrival_time", None)
    first = getattr(metrics, "first_token_time", None)
    finished = getattr(metrics, "finished_time", None)
    ttft = _duration_ms(arrival, first)
    decode = _duration_ms(first, finished)
    output_tokens = len(completion.token_ids)
    return {
        "id": record["id"], "vllm_request_id": output.request_id,
        "category": record.get("category"),
        "input_tokens": len(record["prompt_token_ids"]),
        "output_tokens": output_tokens,
        "token_ids": list(completion.token_ids), "text": completion.text,
        "finish_reason": completion.finish_reason,
        "stop_reason": completion.stop_reason,
        "e2e_ms": _duration_ms(arrival, finished), "ttft_ms": ttft,
        "queue_ms": _duration_ms(
            arrival, getattr(metrics, "first_scheduled_time", None)),
        # This is a per-request average, not individual token interarrival time.
        "tpot_ms": (decode / (output_tokens - 1)
                    if output_tokens > 1 and ttft is not None
                    and decode is not None else None),
    }


def _validate_outputs(outputs, records):
    if len(outputs) != len(records):
        raise RuntimeError("The engine returned an unexpected request count")
    for output, record in zip(outputs, records):
        if not output.finished or len(output.outputs) != 1:
            raise RuntimeError("Expected one completed generation per request")
        if output.prompt_token_ids != record["prompt_token_ids"]:
            raise RuntimeError(f"Output prompt/order mismatch: {record['id']}")


def _read_spec_counters(llm, mode):
    if mode == "ar":
        return None, "AR mode has no speculative decoder"
    try:
        worker = llm.llm_engine.model_executor.driver_worker.worker
        sampler = worker.spec_decode_sampler
        if mode == "minedraft":
            return {
                "verified_draft_tokens": int(
                    sampler.num_verification_tokens.item()),
                "accepted_prefix_tokens": int(
                    sampler.num_good_draft_tokens.item()),
            }, None
        return {
            "verified_draft_tokens": int(sampler.num_draft_tokens),
            "accepted_candidate_tokens": int(
                sampler.num_accepted_tokens.item()),
            "sampler_emitted_tokens": int(sampler.num_emitted_tokens.item()),
        }, None
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _spec_metrics(before, after, mode, k):
    initial, initial_error = before
    final, final_error = after
    if initial is None or final is None:
        return {"available": False, "reason": initial_error or final_error}
    delta = {name: final[name] - value for name, value in initial.items()}
    verified = delta["verified_draft_tokens"]
    if mode == "sd":
        # vLLM 0.9.2 Top1Proposer uses proposal lengths 0 or fixed k. Only
        # nonzero rows enter this sampler, each emitting prefix + one token.
        if verified < 0 or verified % k:
            return {"available": False, "reason": "Invalid fixed-k counters"}
        delta["accepted_prefix_tokens"] = (
            delta["sampler_emitted_tokens"] - verified // k)
        candidates = delta["accepted_candidate_tokens"]
        if not 0 <= delta["accepted_prefix_tokens"] <= candidates <= verified:
            return {"available": False, "reason": "Invalid sampler counters"}
        delta["candidate_acceptance_rate"] = (
            candidates / verified if verified else None)
        delta["prefix_count_method"] = "vllm_0.9.2_fixed_k: emitted - draft/k"
    prefix = delta["accepted_prefix_tokens"]
    if not 0 <= prefix <= verified:
        return {"available": False, "reason": "Invalid sampler counters"}
    return {"available": True, **delta,
            "effective_acceptance_rate": prefix / verified if verified else None}


def _summarize(requests, elapsed):
    if elapsed <= 0 or not math.isfinite(elapsed):
        raise RuntimeError("Invalid generation duration")
    input_tokens = sum(row["input_tokens"] for row in requests)
    output_tokens = sum(row["output_tokens"] for row in requests)
    return {
        "elapsed_s": elapsed, "num_requests": len(requests),
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "requests_per_s": len(requests) / elapsed,
        "output_tokens_per_s": output_tokens / elapsed,
        "total_tokens_per_s": (input_tokens + output_tokens) / elapsed,
        "latency_ms": {
            name: _latency_summary(row[name] for row in requests)
            for name in ("e2e_ms", "ttft_ms", "tpot_ms", "queue_ms")
        },
    }


def _print_summary(label, summary, speculation):
    print(f"\n{label}: {summary['num_requests']} requests, "
          f"{summary['elapsed_s']:.3f} s, "
          f"{summary['output_tokens_per_s']:.2f} output token/s, "
          f"{summary['requests_per_s']:.3f} req/s", flush=True)
    for name, values in summary["latency_ms"].items():
        if values["count"]:
            print(f"  {name}: mean={values['mean']:.2f}, "
                  f"p50={values['p50']:.2f}, p95={values['p95']:.2f}, "
                  f"p99={values['p99']:.2f} "
                  f"(n={values['count']})", flush=True)
        else:
            print(f"  {name}: N/A", flush=True)
    rate = speculation.get("effective_acceptance_rate")
    if speculation["available"] and rate is not None:
        print(f"  effective draft acceptance: {rate:.2%}", flush=True)
    else:
        print("  effective draft acceptance: N/A", flush=True)


def _run_correctness(llm, sampling_params_cls, dataset, job):
    records = dataset["records"]
    results = []
    for case, selected in (("single", records[:1]),
                           ("queued", records[:job["check_num_prompts"]])):
        lengths = (1, min(2, job["check_max_tokens"]), job["check_max_tokens"])
        params = [sampling_params_cls(
            temperature=0, top_p=1, ignore_eos=True,
            max_tokens=(job["check_max_tokens"] if case == "single"
                        else lengths[index % len(lengths)]),
        ) for index in range(len(selected))]
        prompts = [{"prompt_token_ids": row["prompt_token_ids"]}
                   for row in selected]
        print(f"[{job['mode']} k={job['k']}] correctness "
              f"{dataset['name']}/{case}: {len(selected)} requests", flush=True)
        outputs = llm.generate(prompts, params, use_tqdm=False)
        _validate_outputs(outputs, selected)
        results.append({
            "dataset": dataset["name"], "case": case,
            "num_requests": len(selected),
            "records": [
                {"id": row["id"],
                 "token_ids": list(output.outputs[0].token_ids),
                 "text": output.outputs[0].text}
                for row, output in zip(selected, outputs)
            ],
        })
    return results


def _run_performance(llm, sampling_params_cls, dataset, job):
    records = dataset["records"]
    prompts = [{"prompt_token_ids": row["prompt_token_ids"]} for row in records]
    params = sampling_params_cls(
        temperature=job["temperature"], top_p=job["top_p"],
        ignore_eos=job["ignore_eos"], max_tokens=job["max_tokens"])
    label = f"{job['mode']} k={job['k']} {dataset['name']}"
    for index in range(job["warmup_runs"]):
        print(f"[{label}] warmup {index + 1}/{job['warmup_runs']}", flush=True)
        outputs = llm.generate(prompts, params, use_tqdm=False)
        _validate_outputs(outputs, records)

    runs = []
    all_requests = []
    dataset_before = _read_spec_counters(llm, job["mode"])
    before = dataset_before
    for index in range(job["runs"]):
        print(f"[{label}] measured run {index + 1}/{job['runs']}", flush=True)
        start = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        elapsed = time.perf_counter() - start
        # generate returns completed CPU-visible outputs. Metrics reads and
        # report serialization deliberately remain outside the timed region.
        after = _read_spec_counters(llm, job["mode"])
        _validate_outputs(outputs, records)
        requests = [_request_metrics(row, output)
                    for row, output in zip(records, outputs)]
        summary = _summarize(requests, elapsed)
        speculation = _spec_metrics(before, after, job["mode"], job["k"])
        runs.append({"run": index + 1, "summary": summary,
                     "speculation": speculation, "requests": requests})
        all_requests.extend(requests)
        _print_summary(f"{label} run {index + 1}", summary, speculation)
        before = after
    summary = _summarize(all_requests,
                         sum(run["summary"]["elapsed_s"] for run in runs))
    speculation = _spec_metrics(dataset_before, after, job["mode"], job["k"])
    _print_summary(f"{label} total (warmup excluded)", summary, speculation)
    return {"name": dataset["name"], "metadata": dataset.get("metadata", {}),
            "summary": summary, "speculation": speculation, "runs": runs}


def run_worker(job_path):
    """Run a single isolated job, checkpointing after every dataset."""
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    bundle = json.loads(Path(job["bundle_path"]).read_text(encoding="utf-8"))
    stored_fingerprint = bundle.pop("fingerprint")
    encoded = json.dumps(bundle, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != stored_fingerprint:
        raise ValueError("Frozen prompt bundle changed before worker startup")
    bundle["fingerprint"] = stored_fingerprint
    _validate_job(job, bundle)
    _configure_environment(job)

    from vllm import LLM, SamplingParams

    speculative_config = None
    if job["mode"] != "ar":
        speculative_config = {
            "model": job["draft_model"], "method": "draft_model",
            "draft_tensor_parallel_size": 1, "num_speculative_tokens": job["k"],
            "disable_mqa_scorer": True, "disable_logprobs": True,
            "acceptance_method": "rejection_sampler",
        }
        if job["mode"] == "minedraft":
            speculative_config.update(is_parallel=True, force_pearl=False,
                                      tetris=False)
    report = {
        "schema_version": 1, "mode": job["mode"], "k": job["k"],
        "phase": job["phase"], "bundle_fingerprint": bundle["fingerprint"],
        "config": job, "versions": _versions(), "status": "running",
        "model_load_s": None, "datasets": [], "correctness": [],
        "notes": [
            "Timing covers generate only: request submission, queueing, "
            "prefill, decode and output processing; initialization, external "
            "tokenization, warmup, metric reads and serialization are excluded.",
            "Throughput is actual returned tokens divided by summed measured "
            "generation time. TTFT/E2E include engine queueing, not networking.",
            "TPOT=(finished-first token)/(output tokens-1) per request; its "
            "percentiles are not distributions of individual token gaps.",
            "Speculative counters are warmup-excluded deltas before stop/EOS "
            "truncation. SD prefix counts assume native vLLM 0.9.2 fixed k.",
            "Correctness is assessed across fresh mode processes by the "
            "parent. A worker finishing is not proof of correct reproduction.",
        ],
    }
    _write_report(job["output"], report)
    llm = None
    try:
        start = time.perf_counter()
        llm = LLM(
            model=job["model"],
            tensor_parallel_size=job["target_tp"] + (job["mode"] == "minedraft"),
            distributed_executor_backend="mp", dtype=job["dtype"],
            seed=job["seed"], enforce_eager=True,
            disable_custom_all_reduce=True, disable_async_output_proc=True,
            disable_log_stats=True, enable_chunked_prefill=False,
            enable_prefix_caching=False, max_model_len=job["max_model_len"],
            max_num_seqs=job["max_num_seqs"],
            gpu_memory_utilization=job["gpu_memory_utilization"],
            speculative_config=speculative_config,
        )
        report["model_load_s"] = time.perf_counter() - start
        print(f"[{job['mode']} k={job['k']}] engine initialized in "
              f"{report['model_load_s']:.3f} s", flush=True)
        for dataset in bundle["datasets"]:
            if job["phase"] == "correctness":
                report["correctness"].extend(
                    _run_correctness(llm, SamplingParams, dataset, job))
            else:
                report["datasets"].append(
                    _run_performance(llm, SamplingParams, dataset, job))
            _write_report(job["output"], report)
        # Successful shutdown is part of successful job completion, though
        # it does not count towards any generation latency or throughput.
        executor = llm.llm_engine.model_executor
        llm = None
        executor.shutdown()
        report["status"] = "ok"
        _write_report(job["output"], report)
        return report
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        _write_report(job["output"], report)
        raise
    finally:
        if llm is not None:
            try:
                llm.llm_engine.model_executor.shutdown()
            except Exception as shutdown_error:
                # Preserve the original inference failure in the traceback.
                print(f"Worker cleanup failed: {shutdown_error}", file=sys.stderr,
                      flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    run_worker(parser.parse_args().job)
