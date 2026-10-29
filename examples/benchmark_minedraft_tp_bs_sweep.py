# SPDX-License-Identifier: Apache-2.0
"""Run six TP/BS configurations sequentially without changing the benchmark.

BS means --max-num-seqs. TP=4/BS=4 occurs once. Each configuration keeps
its own normal benchmark results; this launcher does not aggregate them.
"""

import argparse
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_devices(parser, value, count, option):
    devices = [item.strip() for item in value.split(",")]
    if (len(devices) != count or len(set(devices)) != count
            or not all(item.isdecimal() for item in devices)):
        parser.error(f"{option} requires {count} distinct numeric NPU IDs")
    return devices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/home/data/Qwen3-32B")
    parser.add_argument("--draft-model", default="/home/data/Qwen3-0.6B")
    parser.add_argument("--data-dir", type=Path, default=Path("/datasets/minedraft"))
    parser.add_argument("--output-root", type=Path,
                        help="Each configuration writes to its own subdirectory")
    parser.add_argument("--tp2-devices", default="4,5,6",
                        help="Dedicated draft card first, then two target cards")
    parser.add_argument("--tp4-devices", default="4,0,1,2,3",
                        help="Dedicated draft card first, then four target cards")
    parser.add_argument("--tp8-devices", default="0,1,2,3,4,5,6,7",
                        help="Eight TARGET cards only; TP=8 runs AR and SD")
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if min(args.num_prompts, args.max_tokens, args.runs) <= 0 or args.warmup_runs < 0:
        parser.error("Request/token/run counts must be positive; warmup nonnegative")
    tp2 = parse_devices(parser, args.tp2_devices, 3, "--tp2-devices")
    tp4 = parse_devices(parser, args.tp4_devices, 5, "--tp4-devices")
    tp8 = parse_devices(parser, args.tp8_devices, 8, "--tp8-devices")

    root = Path(__file__).resolve().parent.parent
    benchmark = root / "examples" / "benchmark_minedraft_datasets.py"
    if not benchmark.is_file():
        raise SystemExit(f"Benchmark entry point not found: {benchmark}")
    if args.output_root is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output_root = root / "minedraft-results" / f"tp-bs-sweep-{stamp}"
    output_root = args.output_root.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    if output_root.exists() and not output_root.is_dir():
        raise SystemExit(f"--output-root is not a directory: {output_root}")

    # The existing entry point requires target_tp+1 distinct strings even for
    # baseline-only runs. Its run_job() passes devices[1:] to BOTH AR and SD.
    # This marker is only a CLI placeholder, never a visible device or a rank.
    # Do not add MineDraft to the TP=8 modes below: it would need a real 9th NPU.
    baseline_only_tp8 = ["baseline_only", *tp8]
    all_modes = ["ar", "sd", "minedraft"]
    matrix = [
        (2, 4, tp2, all_modes),
        (4, 4, tp4, all_modes),
        (8, 4, baseline_only_tp8, ["ar", "sd"]),
        (4, 1, tp4, all_modes),
        (4, 8, tp4, all_modes),
        (4, 16, tp4, all_modes),
    ]
    common = [
        "--model", args.model, "--draft-model", args.draft_model,
        "--dtype", "bfloat16", "--data-dir", str(data_dir),
        "--datasets", "fastchat", "gsm8k", "sharegpt", "spec_bench",
        "--k-values", "5", "--num-prompts", str(args.num_prompts),
        "--max-tokens", str(args.max_tokens),
        "--warmup-runs", str(args.warmup_runs), "--runs", str(args.runs),
        "--skip-correctness",
    ]
    commands = []
    # Check all destinations before any expensive model loading. Keep existing
    # results and leave the entry point responsible for creating result files.
    for tp, bs, devices, modes in matrix:
        output = output_root / f"tp{tp}_bs{bs}"
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise SystemExit(f"Result path is not empty: {output}; use another --output-root")
        commands.append((tp, bs, output, [
            sys.executable, str(benchmark), *common,
            "--devices", ",".join(devices), "--target-tp", str(tp),
            "--max-num-seqs", str(bs), "--modes", *modes,
            "--output-dir", str(output),
        ]))

    for index, (tp, bs, output, command) in enumerate(commands, 1):
        print(f"\n[{index}/{len(commands)}] TP={tp}, BS={bs}; results: {output}",
              flush=True)
        print(shlex.join(command), flush=True)
        # A fresh process for every configuration; the existing entry point
        # manages worker process isolation, logs, timeouts and cleanup.
        completed = subprocess.run(command, cwd=root, check=False)
        if completed.returncode:
            raise SystemExit(f"TP={tp}, BS={bs} failed ({completed.returncode}); "
                             f"see {output}. Remaining configurations not run.")
    print(f"\nAll six configurations completed. Separate results: {output_root}",
          flush=True)
    print("No cross-configuration result aggregation was performed.", flush=True)


if __name__ == "__main__":
    main()
