#!/usr/bin/env python3
"""Interleaved host-vs-device synthetic-input benchmark comparison."""

import argparse
import json
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run_one(mode, config, warmup, steps, device, precision, optimizer):
    command = [
        sys.executable,
        "benchmark.py",
        "--config", config,
        "--device", device,
        "--warmup", str(warmup),
        "--steps", str(steps),
        "--precision", precision,
        "--attention-backend", "sdpa",
        "--sdpa-backend", "flash",
        "--optimizer", optimizer,
        "--input-mode", mode,
    ]
    print("\n$ " + " ".join(command), flush=True)
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    print(result.stdout, end="", flush=True)
    match = re.search(r"^Saved: (.+)$", result.stdout, re.MULTILINE)
    if not match:
        raise RuntimeError("Benchmark completed but did not print its result JSON path")
    path = Path(match.group(1).strip())
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "input_mode": mode,
        "json_path": str(path),
        "median_step_ms": data["median_step_ms"],
        "average_step_ms": data["average_step_ms"],
        "tokens_per_second": data["tokens_per_second"],
        "final_loss": data["final_loss"],
        "step_times_ms": data["step_times_ms"],
        "git_commit": data["git_commit"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/validation_seq2048.yaml")
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--precision", default="bf16", choices=["bf16"])
    parser.add_argument("--optimizer", default="fused_adamw", choices=["fused_adamw"])
    args = parser.parse_args()

    if args.pairs < 1 or args.warmup < 0 or args.steps < 1:
        parser.error("--pairs must be >= 1, --warmup >= 0, and --steps >= 1")

    runs = []
    # Reverse the order each pair to reduce systematic order/thermal bias.
    for pair in range(args.pairs):
        order = ["device", "host"] if pair % 2 == 0 else ["host", "device"]
        print(f"\n=== Pair {pair + 1}/{args.pairs}: {' -> '.join(order)} ===", flush=True)
        for mode in order:
            runs.append({
                "pair": pair + 1,
                **run_one(
                    mode, args.config, args.warmup, args.steps,
                    args.device, args.precision, args.optimizer,
                ),
            })

    by_mode = {
        mode: [run for run in runs if run["input_mode"] == mode]
        for mode in ("device", "host")
    }
    summary = {}
    for mode, items in by_mode.items():
        summary[mode] = {
            "runs": len(items),
            "median_of_run_median_step_ms": statistics.median(
                item["median_step_ms"] for item in items
            ),
            "median_tokens_per_second": statistics.median(
                item["tokens_per_second"] for item in items
            ),
            "median_final_loss": statistics.median(
                item["final_loss"] for item in items
            ),
        }

    device_ms = summary["device"]["median_of_run_median_step_ms"]
    host_ms = summary["host"]["median_of_run_median_step_ms"]
    device_tps = summary["device"]["median_tokens_per_second"]
    host_tps = summary["host"]["median_tokens_per_second"]
    summary["comparison"] = {
        "host_step_time_change_percent_vs_device": (host_ms / device_ms - 1) * 100,
        "host_throughput_change_percent_vs_device": (host_tps / device_tps - 1) * 100,
    }

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "device": args.device,
        "precision": args.precision,
        "attention_backend": "sdpa",
        "sdpa_backend": "FLASH_ATTENTION",
        "optimizer": args.optimizer,
        "warmup_steps": args.warmup,
        "measured_steps_per_run": args.steps,
        "pairs": args.pairs,
        "runs": runs,
        "summary": summary,
        "note": (
            "Unprofiled interleaved benchmark. Input generation and transfer are "
            "included in measured step time. Host mode uses CPU torch.randint "
            "followed by .to(cuda, non_blocking=True); no transfer overlap is assumed."
        ),
    }
    out_dir = Path("results/validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"input_mode_comparison_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n=== INPUT MODE SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"Saved comparison report: {out_path}")


if __name__ == "__main__":
    main()
