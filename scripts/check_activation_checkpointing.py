import argparse
import copy

import torch
import yaml
from torch.nn.attention import SDPBackend, sdpa_kernel

from model import TransformerLM


def run_backward(model, inputs, labels):
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            _, loss = model(inputs, labels)
    if not torch.isfinite(loss).item():
        raise AssertionError(f"Non-finite loss: {loss.item()}")
    loss.backward()
    grads = {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    if len(grads) == 0:
        raise AssertionError("No parameter gradients were produced")
    if not all(torch.isfinite(grad).all().item() for grad in grads.values()):
        raise AssertionError("Found non-finite parameter gradients")
    return loss.detach().float().cpu().item(), grads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/validation_seq2048.yaml")
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This check requires the L4 CUDA environment.")
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    torch.manual_seed(config["training"]["seed"])
    torch.cuda.manual_seed_all(config["training"]["seed"])
    model_cfg = config["model"]
    train_cfg = config["training"]
    common = dict(**model_cfg, attention_backend="sdpa")
    baseline = TransformerLM(**common, checkpoint_policy="none").cuda()
    tokens = torch.randint(
        0, model_cfg["vocab_size"],
        (train_cfg["batch_size"], train_cfg["sequence_length"]),
        device="cuda",
    )
    inputs, labels = tokens[:, :-1], tokens[:, 1:]

    baseline_loss, baseline_grads = run_backward(baseline, inputs, labels)
    state = {name: value.detach().cpu().clone()
             for name, value in baseline.state_dict().items()}
    del baseline
    torch.cuda.empty_cache()

    checkpointed = TransformerLM(**common, checkpoint_policy="every_block").cuda()
    checkpointed.load_state_dict(state)
    checkpoint_loss, checkpoint_grads = run_backward(checkpointed, inputs, labels)

    if baseline_grads.keys() != checkpoint_grads.keys():
        raise AssertionError("Gradient parameter sets differ")
    max_abs = 0.0
    for name in baseline_grads:
        a, b = baseline_grads[name], checkpoint_grads[name]
        if not torch.allclose(a, b, atol=args.atol, rtol=args.rtol):
            max_diff = (a - b).abs().max().item()
            raise AssertionError(f"Gradient mismatch for {name}: max_abs={max_diff}")
        max_abs = max(max_abs, (a - b).abs().max().item())

    if not torch.isclose(
        torch.tensor(baseline_loss), torch.tensor(checkpoint_loss),
        atol=args.atol, rtol=args.rtol,
    ):
        raise AssertionError(
            f"Loss mismatch: baseline={baseline_loss}, checkpoint={checkpoint_loss}"
        )

    print("PASS: activation checkpointing correctness")
    print(f"loss_none={baseline_loss:.8f}")
    print(f"loss_every_block={checkpoint_loss:.8f}")
    print(f"max_abs_gradient_difference={max_abs:.8g}")
    print(f"checked_gradients={len(baseline_grads)}")


if __name__ == "__main__":
    main()
