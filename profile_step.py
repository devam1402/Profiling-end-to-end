import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2,
    )
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be at least 1")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    train_cfg = config["training"]

    torch.manual_seed(train_cfg["seed"])
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = TransformerLM(**model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    batch_size = train_cfg["batch_size"]
    seq_len = train_cfg["sequence_length"]

    model.train()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    output_dir = Path("results/traces")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
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

            prof.step()
            print(f"Step {step + 1}/{args.steps}: loss={loss.item():.4f}")

    print("\nTop operators by CPU self time:")
    print(
        prof.key_averages().table(
            sort_by="self_cpu_time_total",
            row_limit=20,
        )
    )

    trace_path = output_dir / "cpu_profile.json"
    prof.export_chrome_trace(str(trace_path))

    print(f"\nTrace saved: {trace_path}")
    print("Open the trace at https://ui.perfetto.dev/ to inspect the timeline.")


if __name__ == "__main__":
    main()
