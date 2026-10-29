# SPDX-License-Identifier: Apache-2.0
"""Reproduce MineDraft/Standard SD throughput on four text datasets on NPU.

FastChat means Chatbot Arena. Defaults follow MineDraft's sequential/parallel
experiment: raw first-turn prompts, temperature=1, fixed 256-token outputs,
k=1..5, warmup=1, runs=3. GSM8K replaces the original tough dataset.
See docs/source/user_guide/feature_guide/minedraft_datasets.md for commands.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from minedraft_dataset_io import DATASET_NAMES, load_dataset_records


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get(
        "TARGET_MODEL", "Qwen/Qwen3-32B"))
    parser.add_argument("--draft-model", default=os.environ.get(
        "DRAFT_MODEL", "Qwen/Qwen3-0.6B"))
    parser.add_argument("--datasets", nargs="+", choices=DATASET_NAMES,
                        default=list(DATASET_NAMES))
    parser.add_argument("--modes", nargs="+", choices=("ar", "sd", "minedraft"),
                        default=["ar", "sd", "minedraft"])
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--target-tp", type=int, default=4)
    parser.add_argument("--devices", default=os.environ.get(
        "ASCEND_RT_VISIBLE_DEVICES", "4,0,1,2,3"),
                        help="Dedicated draft card FIRST, then target TP cards; "
                        "AR/SD skip the first card, even if run alone.")
    parser.add_argument("--dtype", choices=("bfloat16", "half"), default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--num-prompts", type=int, default=1000,
                        help="Maximum per dataset; do not duplicate short datasets.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--respect-eos", action="store_true",
                        help="Default ignores EOS to match fixed-length experiments.")
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="raw")
    parser.add_argument("--min-input-tokens", type=int,
                        help="Override dataset-specific minima: Arena16, GSM8K4, "
                        "ShareGPT256, Spec-Bench4.")
    parser.add_argument("--data-dir", type=Path, default=Path("minedraft-data"))
    for dataset in DATASET_NAMES:
        parser.add_argument(f"--{dataset.replace('_', '-')}-path", type=Path)
    parser.add_argument("--download-data", action="store_true",
                        help="Download missing dataset sources, explicitly opt-in.")
    parser.add_argument("--bundle", type=Path,
                        help="Reuse a frozen prompts.json; preparation options "
                        "must match its configuration.")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Prepare/tokenize datasets without loading an LLM.")
    parser.add_argument("--check-cases", nargs="+", choices=("single", "queued"),
                        default=["single", "queued"],
                        help="Correctness scenarios; use 'single' to skip the "
                        "queued check. Does not change performance batching.")
    parser.add_argument("--check-num-prompts", type=int,
                        help="Default 2*max-num-seqs+1; capped by dataset size.")
    parser.add_argument("--check-max-tokens", type=int, default=32)
    parser.add_argument("--skip-correctness", action="store_true",
                        help="Mark greedy equivalence NOT CHECKED; measure only.")
    parser.add_argument("--continue-on-mismatch", action="store_true",
                        help="Measure after a failed check, retaining failure status.")
    parser.add_argument("--timeout", type=float, default=7200,
                        help="Seconds per isolated job, including load and shutdown.")
    parser.add_argument("--output-dir", type=Path,
                        help="New or empty directory; existing results are preserved.")
    args = parser.parse_args()
    for field in ("target_tp", "max_model_len", "max_num_seqs", "num_prompts",
                  "max_tokens", "runs", "check_max_tokens"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.warmup_runs < 0 or args.seed < 0 or any(k <= 0 for k in args.k_values):
        parser.error("Invalid warmup, seed or k values")
    if args.min_input_tokens is not None and args.min_input_tokens < 0:
        parser.error("--min-input-tokens must be nonnegative")
    for name in ("temperature", "top_p", "gpu_memory_utilization", "timeout"):
        if not math.isfinite(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    if (args.temperature < 0 or not 0 < args.top_p <= 1
            or not 0 < args.gpu_memory_utilization < 1 or args.timeout <= 0):
        parser.error("Invalid sampling, memory utilization or timeout")
    if max(args.max_tokens, args.check_max_tokens) >= args.max_model_len:
        parser.error("Output token limits must be smaller than max-model-len")
    if args.check_num_prompts is None:
        args.check_num_prompts = 2 * args.max_num_seqs + 1
    if args.check_num_prompts <= 0:
        parser.error("--check-num-prompts must be positive")
    for field in ("datasets", "modes", "k_values", "check_cases"):
        values = getattr(args, field)
        if len(values) != len(set(values)):
            parser.error(f"Duplicate values in --{field.replace('_', '-')}")
    args.devices = [device.strip() for device in args.devices.split(",")]
    if (len(args.devices) != args.target_tp + 1
            or len(set(args.devices)) != len(args.devices)
            or not all(args.devices)):
        parser.error("--devices must list one dedicated draft card followed by "
                     "exactly target-tp target cards, all distinct")
    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output_dir = Path("minedraft-results") / f"datasets-{stamp}"
    return args


def preparation_config(args):
    return {name: getattr(args, name) for name in (
        "model", "datasets", "num_prompts", "seed", "prompt_format",
        "min_input_tokens", "max_model_len", "max_tokens", "check_max_tokens")}


def prepare_bundle(args):
    config = preparation_config(args)
    if args.bundle:
        bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
        digest = bundle.pop("fingerprint")
        if fingerprint(bundle) != digest:
            raise ValueError("Frozen bundle fingerprint mismatch")
        if bundle["preparation_config"] != config:
            raise ValueError("Bundle preparation configuration differs; use the "
                             "same preparation arguments or create a new bundle")
        bundle["fingerprint"] = digest
        return bundle

    # Tokenize ONCE, in the parent. Every worker receives identical token IDs.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    datasets = []
    minima = {"fastchat": 16, "gsm8k": 4, "sharegpt": 256, "spec_bench": 4}
    for name in args.datasets:
        records, metadata = load_dataset_records(
            name, path=getattr(args, f"{name}_path"), data_dir=args.data_dir,
            download=args.download_data, fastchat_source="arena")
        # A dataset-independent seed makes subset selection stable if other
        # datasets are removed from a later run.
        random.Random(f"{args.seed}:{name}").shuffle(records)
        selected = []
        considered = too_short = too_long = 0
        minimum = minima[name] if args.min_input_tokens is None else args.min_input_tokens
        for record in records:
            considered += 1
            if args.prompt_format == "chat":
                tokens = tokenizer.apply_chat_template(
                    [{"role": "user", "content": record["prompt"]}],
                    tokenize=True, add_generation_prompt=True,
                    enable_thinking=False)
            else:
                tokens = tokenizer(record["prompt"], add_special_tokens=True)["input_ids"]
            if len(tokens) < minimum:
                too_short += 1
                continue
            if len(tokens) + max(args.max_tokens, args.check_max_tokens) > args.max_model_len:
                too_long += 1
                continue
            selected.append({"id": record["id"], "prompt_token_ids": tokens,
                             "reference": record["reference"],
                             "category": record["category"]})
            if len(selected) == args.num_prompts:
                break
        if not selected:
            raise ValueError(f"{name}: no prompts survive token length filtering")
        metadata["selection"] = {
            "requested": args.num_prompts, "selected": len(selected),
            "considered": considered, "too_short": too_short,
            "too_long": too_long, "min_input_tokens": minimum,
            "unexamined": len(records) - considered,
            "input_tokens_min": min(len(row["prompt_token_ids"]) for row in selected),
            "input_tokens_max": max(len(row["prompt_token_ids"]) for row in selected),
        }
        datasets.append({"name": name, "metadata": metadata, "records": selected})
        print(f"{name}: selected {len(selected)}/{args.num_prompts}; "
              f"short={too_short}, long={too_long}", flush=True)
    bundle = {"schema_version": 1, "preparation_config": config,
              "tokenizer": {"class": type(tokenizer).__name__,
                            "name_or_path": tokenizer.name_or_path},
              "datasets": datasets}
    bundle["fingerprint"] = fingerprint(bundle)
    return bundle


def stop_group(process):
    # Linux-only inference: kill descendants even if their parent has exited.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_job(args, directory, bundle, mode, k, phase):
    name = f"{phase}-{mode}-k{k}"
    job_path = directory / f"{name}.job.json"
    output = directory / f"{name}.json"
    log_path = directory / f"{name}.log"
    job = {key: getattr(args, key) for key in (
        "model", "draft_model", "target_tp", "dtype", "gpu_memory_utilization",
        "max_model_len", "max_num_seqs", "seed", "temperature", "top_p",
        "max_tokens", "warmup_runs", "runs", "check_num_prompts", "check_max_tokens",
        "check_cases")}
    job.update(mode=mode, k=k, phase=phase, check=phase == "correctness",
               ignore_eos=not args.respect_eos, output=str(output),
               visible_devices=args.devices if mode == "minedraft" else args.devices[1:],
               bundle_path=str(directory / "prompts.json"))
    write_json(job_path, job)
    print(f"\nRunning {name}; log: {log_path}", flush=True)
    worker = Path(__file__).with_name("minedraft_dataset_worker.py")
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(job["visible_devices"])
    environment["VLLM_ASCEND_ENABLE_MINEDRAFT"] = "1" if mode == "minedraft" else "0"
    environment["VLLM_PLUGINS"] = "ascend,ascend_enhanced_model" + (
        ",minedraft" if mode == "minedraft" else "")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, str(worker), "--job", str(job_path)],
            stdout=log, stderr=subprocess.STDOUT, env=environment,
            start_new_session=True)
        try:
            returncode = process.wait(timeout=args.timeout)
            if returncode:
                raise RuntimeError(f"{name} exited with code {returncode}; see {log_path}")
        except BaseException:
            stop_group(process)
            raise
    result = json.loads(output.read_text(encoding="utf-8"))
    if (result.get("status") != "ok"
            or result.get("bundle_fingerprint") != bundle["fingerprint"]):
        raise RuntimeError(f"Invalid/incomplete result: {output}")
    return result


def compare_correctness(baseline, result):
    def index_cases(report):
        indexed = {}
        for case in report["correctness"]:
            for record in case["records"]:
                key = (case["dataset"], case["case"], record["id"])
                if key in indexed:
                    raise ValueError(f"Duplicate correctness record: {key}")
                indexed[key] = record["token_ids"]
        return indexed

    expected, actual = index_cases(baseline), index_cases(result)
    if not expected or expected.keys() != actual.keys():
        raise ValueError("Correctness request IDs/cases differ across modes")
    mismatches = []
    for key, tokens in expected.items():
        other = actual[key]
        if tokens != other:
            first = next((i for i, (a, b) in enumerate(zip(tokens, other)) if a != b),
                         min(len(tokens), len(other)))
            mismatches.append({"dataset": key[0], "case": key[1], "id": key[2],
                               "first_different_token_index": first,
                               "baseline_token_ids": tokens, "actual_token_ids": other})
    return {"mode": result["mode"], "k": result["k"],
            "status": "failed" if mismatches else "passed",
            "compared_requests": len(expected), "mismatches": mismatches}


def build_rows(reports, correctness_status):
    rows = []
    for report in reports:
        for dataset in report["datasets"]:
            summary = dataset["summary"]
            row = {"dataset": dataset["name"], "mode": report["mode"], "k": report["k"],
                   "greedy_check": correctness_status,
                   **{key: summary[key] for key in (
                       "elapsed_s", "num_requests", "input_tokens", "output_tokens",
                       "requests_per_s", "output_tokens_per_s", "total_tokens_per_s")},
                   "effective_acceptance_rate": dataset["speculation"].get(
                       "effective_acceptance_rate")}
            for metric, statistics in summary["latency_ms"].items():
                for statistic, value in statistics.items():
                    row[f"{metric}_{statistic}"] = value
            rows.append(row)
    for row in rows:
        choices = [candidate for candidate in rows if candidate["dataset"] == row["dataset"]]
        ar = next((candidate for candidate in choices if candidate["mode"] == "ar"), None)
        sd = next((candidate for candidate in choices if candidate["mode"] == "sd"
                   and candidate["k"] == row["k"]), None)
        best_sd = max((candidate["output_tokens_per_s"] for candidate in choices
                       if candidate["mode"] == "sd"), default=None)
        rate = row["output_tokens_per_s"]
        row["speedup_vs_ar"] = rate / ar["output_tokens_per_s"] if ar and ar["output_tokens_per_s"] > 0 else None
        row["speedup_vs_sd_same_k"] = rate / sd["output_tokens_per_s"] if sd and sd["output_tokens_per_s"] > 0 else None
        row["speedup_vs_best_measured_sd"] = rate / best_sd if best_sd and best_sd > 0 else None
    return rows


def save_summary(directory, summary, reports):
    rows = build_rows(reports, summary["correctness"]["status"])
    summary["metrics"] = rows
    write_json(directory / "summary.json", summary)
    if rows:
        with (directory / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [f"Run status: {summary['status']}",
             f"Greedy equivalence: {summary['correctness']['status']}",
             "dataset | mode | k | output token/s | TTFT mean ms | SD same-k speedup",
             "--- | --- | --- | --- | --- | ---"]
    for row in rows:
        ttft = row["ttft_ms_mean"]
        ratio = row["speedup_vs_sd_same_k"]
        ttft_text = f"{ttft:.2f}" if ttft is not None else "N/A"
        lines.append(f"{row['dataset']} | {row['mode']} | {row['k']} | "
                     f"{row['output_tokens_per_s']:.2f} | {ttft_text}")
        lines[-1] += f" | {ratio:.3f}x" if ratio is not None else " | N/A"
    lines.extend(["", *summary["notes"]])
    (directory / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if not args.prepare_only and sys.platform != "linux":
        raise SystemExit("Inference requires Linux with Ascend NPU; --prepare-only is portable")
    directory = args.output_dir.expanduser().resolve()
    if directory.exists() and any(directory.iterdir()):
        raise SystemExit(f"Use a new/empty output directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1, "status": "preparing", "performance_completed": False,
        "correctness": {"status": "not_run", "cases": args.check_cases,
                        "comparisons": []},
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        "jobs": [],
        "notes": [
            "Four requested datasets: Chatbot Arena, GSM8K main/test, ShareGPT, Spec-Bench. "
            "First user turns only; no official quality/judge scores are claimed.",
            "This is an Ascend reproduction workflow for the supported draft-model path, "
            "not a reproduction of every CUDA/Tetris/PEARL/EAGLE experiment or published number.",
            "Performance uses configured sampling; the separate check always uses greedy. "
            "Passing a finite greedy check is not a proof of sampling-distribution equivalence.",
            "Timing covers generate only; warmup, loading, data preparation and metric IO excluded. "
            "Speedup is output-token throughput ratio, not an average of per-run ratios.",
            "AR/SD use target TP cards; MineDraft uses one additional dedicated draft card. "
            "MineDraft's two interleaved batches expand total scheduler capacity.",
            "A skipped or failed check does not become passed when performance completes. "
            "Speedups from such runs are unvalidated measurements.",
        ],
    }
    reports = []
    try:
        bundle = prepare_bundle(args)
        write_json(directory / "prompts.json", bundle)
        summary["bundle_fingerprint"] = bundle["fingerprint"]
        summary["dataset_metadata"] = [{"name": item["name"], "metadata": item["metadata"]}
                                       for item in bundle["datasets"]]
        summary["code_sha256"] = {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (Path(__file__).name, "minedraft_dataset_io.py", "minedraft_dataset_worker.py")}
        if args.prepare_only:
            summary["status"] = "prepared"
            save_summary(directory, summary, reports)
            print(f"Prepared frozen prompts: {directory / 'prompts.json'}", flush=True)
            return 0
        combinations = [(mode, k) for mode in args.modes
                        for k in ([0] if mode == "ar" else args.k_values)]
        if not args.skip_correctness:
            summary["status"] = "checking"
            summary["correctness"]["status"] = "running"
            check_queued = "queued" in args.check_cases
            summary["correctness"]["coverage"] = [
                {"dataset": item["name"],
                 "queued_checked": check_queued,
                 "queued_requests": min(len(item["records"]), args.check_num_prompts)
                 if check_queued else 0,
                 "exceeds_two_batch_capacity": check_queued and
                 min(len(item["records"]), args.check_num_prompts)
                 > 2 * args.max_num_seqs} for item in bundle["datasets"]]
            if not check_queued:
                print("Queued correctness check skipped (--check-cases single). "
                      "Performance batching remains configured by num-prompts "
                      "and max-num-seqs.", flush=True)
            for item in summary["correctness"]["coverage"]:
                if check_queued and not item["exceeds_two_batch_capacity"]:
                    print(f"Limited queued coverage: {item['dataset']} has "
                          f"{item['queued_requests']} check requests, needs "
                          f"at least {2 * args.max_num_seqs + 1}.", flush=True)
            save_summary(directory, summary, reports)
            baseline = run_job(args, directory, bundle, "ar", 0, "correctness")
            summary["jobs"].append("correctness-ar-k0.json")
            comparisons = summary["correctness"]["comparisons"]
            for mode, k in combinations:
                if mode == "ar":
                    continue
                result = run_job(args, directory, bundle, mode, k, "correctness")
                summary["jobs"].append(f"correctness-{mode}-k{k}.json")
                comparison = compare_correctness(baseline, result)
                comparisons.append(comparison)
                if comparison["status"] == "failed":
                    summary["correctness"]["status"] = "failed"
                print(f"Greedy {mode} k={k}: {comparison['status']}; "
                      f"mismatches={len(comparison['mismatches'])}", flush=True)
                write_json(directory / "correctness.json", summary["correctness"])
                save_summary(directory, summary, reports)
            summary["correctness"]["status"] = (
                "failed" if any(item["status"] == "failed" for item in comparisons)
                else "passed" if comparisons else "not_applicable_ar_only")
            if (summary["correctness"]["status"] == "passed"
                    and (set(args.check_cases) != {"single", "queued"}
                         or not all(item["exceeds_two_batch_capacity"]
                                    for item in summary["correctness"]["coverage"]))):
                summary["correctness"]["status"] = "passed_limited_coverage"
            write_json(directory / "correctness.json", summary["correctness"])
            if summary["correctness"]["status"] == "failed" and not args.continue_on_mismatch:
                summary["status"] = "correctness_failed_performance_not_run"
                save_summary(directory, summary, reports)
                print(f"Greedy mismatch; inspect {directory / 'correctness.json'}. "
                      "Performance not run.", flush=True)
                return 2
        summary["status"] = "measuring"
        save_summary(directory, summary, reports)
        for mode, k in combinations:
            result = run_job(args, directory, bundle, mode, k, "performance")
            # Full per-request outputs remain on disk; do not retain every
            # dataset/run/token sequence across the entire k sweep in RAM.
            for dataset in result["datasets"]:
                dataset.pop("runs", None)
            reports.append(result)
            summary["jobs"].append(f"performance-{mode}-k{k}.json")
            save_summary(directory, summary, reports)
            for dataset in result["datasets"]:
                print(f"{dataset['name']} {mode} k={k}: "
                      f"{dataset['summary']['output_tokens_per_s']:.2f} output token/s", flush=True)
        summary["performance_completed"] = True
        check = summary["correctness"]["status"]
        summary["status"] = "completed_with_greedy_check" if check == "passed" else "completed_unvalidated"
        save_summary(directory, summary, reports)
        print((directory / "summary.txt").read_text(encoding="utf-8"), flush=True)
        print(f"Full results: {directory}", flush=True)
        return 2 if check == "failed" else 0
    except BaseException as exc:
        summary["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        save_summary(directory, summary, reports)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
