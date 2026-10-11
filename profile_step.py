import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
)

from model import TransformerLM


def main():
    parser = argparse.ArgumentParser(
        description="Profile Transformer training steps."
    )
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--attention-backend",
        choices=["config", "eager", "sdpa"],
        default="config",
        help="Override the attention backend in the config.",
    )
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be at least 1")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested, but torch.cuda.is_available() is False")

    device_name = (
        "cuda" if args.device == "cuda"
        else "cpu" if args.device == "cpu"
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    device = torch.device(device_name)

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    model_cfg = config["model"]
    train_cfg = config["training"]
    batch_size = train_cfg["batch_size"]
    seq_len = train_cfg["sequence_length"]

    if seq_len < 2 or seq_len > model_cfg["max_seq_len"]:
        parser.error("sequence_length must be between 2 and max_seq_len")

    torch.manual_seed(train_cfg["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_cfg["seed"])

    configured_backend = config.get("attention", {}).get("backend", "eager")
    attention_backend = (
        configured_backend
        if args.attention_backend == "config"
        else args.attention_backend
    )
    model = TransformerLM(
        **model_cfg,
        attention_backend=attention_backend,
    ).to(device)

    attention_modules = [
        module for module in model.modules()
        if hasattr(module, "backend")
        and hasattr(module, "_cuda_attention_event_pairs")
    ]
    if device.type == "cuda":
        for module in attention_modules:
            module._profile_cuda_attention = True
            module._cuda_attention_event_pairs.clear()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    model.train()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    output_dir = Path("results/traces")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace_path = output_dir / f"{device.type}_profile_{timestamp}.json"
    metadata_path = output_dir / f"{device.type}_profile_{timestamp}.metadata.json"

    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Python: {platform.python_version()}")
    print(f"Profiling steps: {args.steps}")
    print("Profiler overhead is expected; do not use this run for throughput.")

    forward_event_pairs = []

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for step in range(args.steps):
            input_ids = torch.randint(
                0,
                model_cfg["vocab_size"],
                (batch_size, seq_len),
                device=device,
            )
            inputs = input_ids[:, :-1]
            labels = input_ids[:, 1:]

            optimizer.zero_grad(set_to_none=True)

            forward_start = forward_end = None
            if device.type == "cuda":
                forward_start = torch.cuda.Event(enable_timing=True)
                forward_end = torch.cuda.Event(enable_timing=True)
                forward_start.record()

            with record_function("training_forward_loss"):
                _, loss = model(inputs, labels)

            if device.type == "cuda":
                forward_end.record()
                forward_event_pairs.append((forward_start, forward_end))

            if loss is None or not torch.isfinite(loss).item():
                raise RuntimeError(f"Invalid loss at step {step + 1}")

            with record_function("training_backward"):
                loss.backward()

            with record_function("optimizer_step"):
                optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            prof.step()
            print(f"Step {step + 1}/{args.steps}: loss={loss.item():.4f}")

    averages = prof.key_averages()

    print("\nTop 5 operators by CPU self time:")
    print(
        averages.table(
            sort_by="self_cpu_time_total",
            row_limit=5,
        )
    )

    if device.type == "cuda":
        print("\nTop 5 operators by CUDA self time:")
        print(
            averages.table(
                sort_by="self_cuda_time_total",
                row_limit=5,
            )
        )

    # These are CPU-side profiler-range durations, not GPU kernel time.
    range_times = {
        event.key: event.cpu_time_total
        for event in averages
    }
    attention_cpu_us = range_times.get("attention_compute", 0.0)
    forward_cpu_us = range_times.get("training_forward_loss", 0.0)

    print("\nCPU-side profiler ranges (aggregated across profiled steps):")
    print(f"  attention_compute: {attention_cpu_us / 1000:.3f} ms")
    print(f"  training_forward_loss: {forward_cpu_us / 1000:.3f} ms")

    if forward_cpu_us > 0:
        share = 100.0 * attention_cpu_us / forward_cpu_us
        print(
            "  attention range / forward-loss CPU time: "
            f"{share:.1f}% (not GPU-time share)"
        )

    prof.export_chrome_trace(str(trace_path))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        event_pairs = [
            pair
            for module in attention_modules
            for pair in module._cuda_attention_event_pairs
        ]
        attention_ms = sum(
            start.elapsed_time(end)
            for start, end in event_pairs
        )
        forward_ms = sum(
            start.elapsed_time(end)
            for start, end in forward_event_pairs
        )
        print(
            "\nCUDA-event attention compute time "
            f"(summed across layers and steps): {attention_ms:.3f} ms"
        )
        print(
            "CUDA-event forward-pass time "
            f"(summed across steps): {forward_ms:.3f} ms"
        )
        if forward_ms > 0:
            print(
                "Attention compute / forward-pass time: "
                f"{100.0 * attention_ms / forward_ms:.1f}%"
            )
        print(
            "Attention timing excludes QKV projections, output projection, "
            "and backward computation."
        )

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" else platform.processor() or "CPU"
        ),
        "pytorch_version": torch.__version__,
        "python_version": platform.python_version(),
        "steps": args.steps,
        "attention_backend": attention_backend,
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trace_file": trace_path.name,
        "config": config,
        "note": "Profiler trace; timings are instrumented and not benchmark timings.",
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"\nTrace saved: {trace_path}")
    print(f"Metadata saved: {metadata_path}")
    print("Open the trace at https://ui.perfetto.dev/ to inspect the timeline.")


if __name__ == "__main__":
    main()
