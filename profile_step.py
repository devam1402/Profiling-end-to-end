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

    model = TransformerLM(**model_cfg).to(device)
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

            with record_function("training_forward_loss"):
                _, loss = model(inputs, labels)

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

    print("\nTop operators by CPU self time:")
    print(
        prof.key_averages().table(
            sort_by="self_cpu_time_total",
            row_limit=20,
        )
    )

    prof.export_chrome_trace(str(trace_path))

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
