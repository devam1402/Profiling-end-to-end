import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

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
        default=None,
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    train_cfg = config["training"]

    steps = args.steps if args.steps is not None else train_cfg["steps"]
    if steps < 1:
        raise ValueError("--steps must be at least 1")

    batch_size = train_cfg["batch_size"]
    seq_len = train_cfg["sequence_length"]
    seed = train_cfg["seed"]

    torch.manual_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    attention_backend = config.get("attention", {}).get("backend", "eager")
    model = TransformerLM(
        **model_cfg,
        attention_backend=attention_backend,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    parameters = sum(p.numel() for p in model.parameters())

    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Parameters: {parameters:,}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Training steps: {steps}")
    print("-" * 40)

    model.train()
    start = time.perf_counter()
    final_loss = None

    for step in range(1, steps + 1):
        input_ids = torch.randint(
            0,
            model_cfg["vocab_size"],
            (batch_size, seq_len),
            device=device,
        )

        inputs = input_ids[:, :-1]
        labels = input_ids[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, labels)

        if loss is None or not torch.isfinite(loss).item():
            raise RuntimeError(f"Invalid loss at step {step}: {loss}")

        loss.backward()
        optimizer.step()

        final_loss = loss.item()
        print(f"Step {step:02d}/{steps} | loss={final_loss:.4f}")

    elapsed = time.perf_counter() - start
    tokens = batch_size * (seq_len - 1) * steps
    throughput = tokens / elapsed

    print("-" * 40)
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Average step time: {elapsed / steps:.3f} seconds")
    print(f"Throughput: {throughput:.1f} tokens/sec")

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "device": str(device),
        "pytorch_version": torch.__version__,
        "parameters": parameters,
        "steps": steps,
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "tokens_processed_estimate": tokens,
        "elapsed_seconds": elapsed,
        "average_step_seconds": elapsed / steps,
        "tokens_per_second": throughput,
        "final_loss": final_loss,
        "config": config,
        "note": (
            "Initial smoke benchmark; includes data generation and "
            "startup effects. Not a final GPU performance measurement."
        ),
    }

    output_dir = Path("results/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"run_{timestamp}.json"
    output_path.write_text(
        json.dumps(result, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"Result saved: {output_path}")
    print("Training loop completed.")


if __name__ == "__main__":
    main()
