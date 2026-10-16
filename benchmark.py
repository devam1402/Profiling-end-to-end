import argparse
from contextlib import nullcontext
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import yaml

from model import TransformerLM


def git_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(
        description="Repeatable Transformer training benchmark"
    )
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--attention-backend",
        choices=["config", "eager", "sdpa"],
        default="config",
        help="Override config backend, or use the configured backend",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Compute precision; BF16 uses autocast with FP32 model parameters",
    )
    parser.add_argument(
        "--sdpa-backend",
        choices=["default", "flash"],
        default="default",
        help="Select the SDPA backend; flash forces Flash Attention",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "fused_adamw"],
        default="adamw",
        help="Optimizer implementation for controlled comparisons",
    )
    parser.add_argument(
        "--input-mode",
        choices=["device", "host"],
        default="device",
        help=(
            "Generate synthetic token batches on the selected device, or on "
            "the CPU and transfer them to the device"
        ),
    )
    args = parser.parse_args()

    if args.warmup < 0 or args.steps < 1:
        parser.error("--warmup must be >= 0 and --steps must be >= 1")

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    device_name = (
        "cuda" if args.device == "cuda"
        else "cpu" if args.device == "cpu"
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    device = torch.device(device_name)

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
    if args.sdpa_backend == "flash" and attention_backend != "sdpa":
        parser.error("--sdpa-backend flash requires --attention-backend sdpa")
    if args.precision == "bf16" and device.type != "cuda":
        parser.error("--precision bf16 currently requires --device cuda")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        fused=(args.optimizer == "fused_adamw"),
    )

    def train_step():
        if args.input_mode == "device":
            input_ids = torch.randint(
                0,
                model_cfg["vocab_size"],
                (batch_size, seq_len),
                device=device,
            )
        else:
            # Deliberately include host-side generation and H2D transfer in
            # the timed step so the input-mode comparison measures end-to-end
            # training-step cost. non_blocking=True is an optimization hint;
            # transfer overlap is not assumed.
            input_ids = torch.randint(
                0,
                model_cfg["vocab_size"],
                (batch_size, seq_len),
                device="cpu",
            ).to(device, non_blocking=True)
        inputs, labels = input_ids[:, :-1], input_ids[:, 1:]

        optimizer.zero_grad(set_to_none=True)

        backend_context = (
            sdpa_kernel(SDPBackend.FLASH_ATTENTION)
            if args.sdpa_backend == "flash"
            else nullcontext()
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=args.precision == "bf16",
        ):
            with backend_context:
                _, loss = model(inputs, labels)

        if loss is None or not torch.isfinite(loss).item():
            raise RuntimeError("Loss is missing or non-finite")

        loss.backward()
        optimizer.step()
        return loss.detach()

    model.train()

    # Warm-up steps are excluded from timed measurements.
    for _ in range(args.warmup):
        train_step()
    synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    step_times = []
    losses = []
    for _ in range(args.steps):
        synchronize(device)
        start = time.perf_counter()
        loss = train_step()
        synchronize(device)
        step_times.append(time.perf_counter() - start)
        losses.append(loss.item())

    elapsed = sum(step_times)
    tokens_per_step = batch_size * (seq_len - 1)
    median_step = statistics.median(step_times)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" else "CPU"
        ),
        "pytorch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_build_version": torch.version.cuda,
        "git_commit": git_commit(),
        "config_path": str(Path(args.config).resolve()),
        "seed": train_cfg["seed"],
        "configured_dtype": config.get("runtime", {}).get("dtype", "float32"),
        "actual_model_dtype": str(next(model.parameters()).dtype),
        "attention_backend": attention_backend,
        "precision": args.precision,
        "sdpa_backend": (
            "FLASH_ATTENTION" if args.sdpa_backend == "flash" else "default"
        ),
        "optimizer": args.optimizer,
        "input_mode": args.input_mode,
        "parameters": sum(p.numel() for p in model.parameters()),
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "tokens_per_step": tokens_per_step,
        "total_measured_tokens": tokens_per_step * args.steps,
        "total_measured_seconds": elapsed,
        "average_step_seconds": statistics.mean(step_times),
        "median_step_seconds": median_step,
        "tokens_per_second": tokens_per_step / statistics.mean(step_times),
        "step_times_seconds": step_times,
        "step_times_ms": [value * 1000 for value in step_times],
        "average_step_ms": statistics.mean(step_times) * 1000,
        "median_step_ms": median_step * 1000,
        "losses": losses,
        "final_loss": losses[-1],
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda" else None
        ),
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if device.type == "cuda" else None
        ),
        "peak_gpu_reserved_memory_bytes": (
            torch.cuda.max_memory_reserved(device)
            if device.type == "cuda" else None
        ),
        "peak_gpu_reserved_memory_mb": (
            torch.cuda.max_memory_reserved(device) / (1024 ** 2)
            if device.type == "cuda" else None
        ),
        "config": config,
        "note": (
            "Training-step timing includes input generation (on the selected "
            "device or on CPU followed by .to(device, non_blocking=True)), "
            "forward pass, backward pass, and optimizer update. "
            "Warm-up steps excluded. Profiler disabled."
        ),
    }

    output_dir = Path("results/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"benchmark_{timestamp}.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"Device: {result['device']} ({result['device_name']})")
    print(f"Parameters: {result['parameters']:,}")
    print(f"Attention backend: {result['attention_backend']}")
    print(f"Precision: {result['precision']}")
    print(f"SDPA backend: {result['sdpa_backend']}")
    print(f"Optimizer: {result['optimizer']}")
    print(f"Warm-up steps: {args.warmup}")
    print(f"Measured steps: {args.steps}")
    print(f"Average step: {result['average_step_seconds']:.3f} s")
    print(f"Average step: {result['average_step_ms']:.1f} ms")
    print(f"Median step: {result['median_step_seconds']:.3f} s")
    print(f"Median step: {result['median_step_ms']:.1f} ms")
    print(f"Throughput: {result['tokens_per_second']:.1f} tokens/sec")
    print(f"Final loss: {result['final_loss']:.4f}")
    print(f"Peak GPU allocated memory: {result['peak_gpu_memory_mb']} MB")
    print(f"Peak GPU reserved memory: {result['peak_gpu_reserved_memory_mb']} MB")
    print(f"Git commit: {result['git_commit']}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
