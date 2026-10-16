import math
from typing import Optional

import torch
from torch.profiler import record_function
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(
            dim=-1,
            keepdim=True,
        )

        x = x * torch.rsqrt(
            variance + self.eps
        )

        return self.weight * x


class EagerAttention(nn.Module):
    """
    Reference attention implementation.

    This is intentionally explicit:
        QK^T
        causal mask
        softmax
        Attention @ V

    Later we will replace this implementation with:
        1. PyTorch SDPA FlashAttention
        2. flash-attn
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        backend: str = "eager",
    ):
        super().__init__()

        if backend not in {"eager", "sdpa"}:
            raise ValueError(f"Unsupported attention backend: {backend}")

        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.backend = backend
        self._profile_cuda_attention = False
        self._cuda_attention_event_pairs = []

        self.qkv = nn.Linear(
            hidden_size,
            3 * hidden_size,
            bias=False,
        )

        self.out_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # [B, S, 3H]
        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        # [B, S, H] -> [B, heads, S, head_dim]
        q = q.view(
            batch,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        profile_cuda = (
            getattr(self, "_profile_cuda_attention", False)
            and x.is_cuda
        )
        start_event = end_event = None

        if profile_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        with record_function("attention_compute"):
            if self.backend == "sdpa":
                output = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    is_causal=True,
                )
            else:
                scale = 1.0 / math.sqrt(self.head_dim)

                # [B, heads, S, S]
                scores = torch.matmul(
                    q,
                    k.transpose(-2, -1),
                ) * scale

                # Causal mask.
                causal_mask = torch.triu(
                    torch.ones(
                        seq_len,
                        seq_len,
                        device=x.device,
                        dtype=torch.bool,
                    ),
                    diagonal=1,
                )

                scores = scores.masked_fill(
                    causal_mask,
                    float("-inf"),
                )

                attention = torch.softmax(
                    scores,
                    dim=-1,
                )

                # [B, heads, S, head_dim]
                output = torch.matmul(
                    attention,
                    v,
                )

        if profile_cuda:
            end_event.record()
            self._cuda_attention_event_pairs.append(
                (start_event, end_event)
            )

        # [B, heads, S, head_dim]
        # -> [B, S, H]
        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch,
            seq_len,
            self.hidden_size,
        )

        return self.out_proj(output)


class SwiGLU(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()

        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )

        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )

        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(
            self.gate_proj(x)
        )

        up = self.up_proj(x)

        return self.down_proj(
            gate * up
        )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        attention_backend: str = "eager",
    ):
        super().__init__()

        self.norm1 = RMSNorm(
            hidden_size
        )

        self.attention = EagerAttention(
            hidden_size,
            num_heads,
            backend=attention_backend,
        )

        self.norm2 = RMSNorm(
            hidden_size
        )

        self.mlp = SwiGLU(
            hidden_size,
            intermediate_size,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # Attention + residual
        x = x + self.attention(
            self.norm1(x)
        )

        # MLP + residual
        x = x + self.mlp(
            self.norm2(x)
        )

        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        intermediate_size: int = 2048,
        max_seq_len: int = 1024,
        attention_backend: str = "eager",
        checkpoint_policy: str = "none",
    ):
        super().__init__()

        if checkpoint_policy not in {"none", "every_other", "every_block"}:
            raise ValueError(f"Unsupported checkpoint policy: {checkpoint_policy}")

        self.checkpoint_policy = checkpoint_policy
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(
            vocab_size,
            hidden_size,
        )

        self.position_embedding = nn.Embedding(
            max_seq_len,
            hidden_size,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                    attention_backend=attention_backend,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(
            hidden_size
        )

        self.lm_head = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False,
        )

        # Weight tying.
        self.lm_head.weight = (
            self.token_embedding.weight
        )

        self.apply(self._init_weights)

    def _init_weights(
        self,
        module: nn.Module,
    ):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        batch, seq_len = input_ids.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"max_seq_len={self.max_seq_len}"
            )

        positions = torch.arange(
            seq_len,
            device=input_ids.device,
        )

        x = self.token_embedding(
            input_ids
        )

        x = x + self.position_embedding(
            positions
        )

        for index, block in enumerate(self.blocks):
            should_checkpoint = (
                self.training
                and torch.is_grad_enabled()
                and (
                    self.checkpoint_policy == "every_block"
                    or (
                        self.checkpoint_policy == "every_other"
                        and index % 2 == 0
                    )
                )
            )
            if should_checkpoint:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        loss = None

        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    self.vocab_size,
                ),
                labels.reshape(-1),
            )

        return logits, loss
