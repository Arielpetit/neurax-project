"""
ESM-2-8M — written by NEURAX's compiler from the design it priced.

Every block below is built from the values the compiler resolved for it, so
this model has the 7517125 parameters the studio reports. Regenerate it after any
change on the canvas.
"""
import math
from contextvars import ContextVar
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Graph(NamedTuple):
    """A batch of graphs: node features, edges as `[2, E]` indices, and the
    graph each node belongs to."""

    x: torch.Tensor
    edge_index: torch.Tensor
    batch: torch.Tensor
    num_graphs: int
    # Present when the data states them: edge features, edge and node types.
    edge_attr: Optional[torch.Tensor] = None
    edge_type: Optional[torch.Tensor] = None
    node_type: Optional[torch.Tensor] = None


ACTIVATIONS = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "softplus": nn.Softplus,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


def activation(name):
    """The activation a block names; `none` is the identity."""
    if name in (None, "none", "identity", "linear"):
        return nn.Identity()
    if name == "sin":
        return Sin()
    return ACTIVATIONS[name]()


def features(value):
    """The tensor a block computes on: a graph's node features, else itself."""
    return value.x if isinstance(value, Graph) else value


def with_features(value, x):
    """`value` carrying new features: a graph keeps its edges."""
    return value._replace(x=x) if isinstance(value, Graph) else x


def per_position(value, f):
    """`f` applied to each position's vector: over the last axis, or over the
    channels of each pixel of a feature map — the compiler's reading of a
    linear map or a norm on an image."""
    x = features(value)
    if x.dim() == 4:
        return with_features(value, f(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
    return with_features(value, f(x))


class Identity(nn.Module):
    """A block that passes what it receives on: the design's input and output,
    a scheduler, a position scheme applied elsewhere."""

    def forward(self, x, *rest):
        return x


class Dropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.drop = nn.Dropout(p)

    def forward(self, x):
        return with_features(x, self.drop(features(x)))


class Residual(nn.Module):
    """The sum of what arrives, dropout on the branch first. In a repeated
    layer the skip is the stream the layer started from."""

    def __init__(self, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, *others):
        if not others:
            return x
        total = features(others[0])
        for other in others[1:]:
            total = total + features(other)
        return with_features(x, total + self.drop(features(x)))


class LayerNorm(nn.Module):
    """Over the last axis, `width` wide, with or without scale and bias."""

    def __init__(self, width, eps, affine, bias):
        super().__init__()
        self.width, self.eps = width, eps
        self.weight = nn.Parameter(torch.ones(width)) if affine else None
        self.bias = nn.Parameter(torch.zeros(width)) if bias else None

    def forward(self, x):
        return per_position(x, lambda h: F.layer_norm(h, (self.width,), self.weight, self.bias, self.eps))


class RMSNorm(nn.Module):
    def __init__(self, width, eps, affine):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width)) if affine else None

    def forward(self, x):
        def norm(h):
            y = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
            return y * self.weight if self.weight is not None else y

        return per_position(x, norm)


class GlobalPool(nn.Module):
    """A sequence to one vector per sample (first token, mean or max), a
    feature map to one per image, a graph's nodes to one per graph."""

    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        if isinstance(x, Graph):
            index = x.batch.unsqueeze(-1).expand_as(x.x)
            out = x.x.new_zeros(x.num_graphs, x.x.shape[-1])
            if self.mode == "max":
                return out.scatter_reduce(0, index, x.x, reduce="amax", include_self=False)
            total = out.scatter_add(0, index, x.x)
            if self.mode in ("sum", "add"):
                return total
            count = x.x.new_zeros(x.num_graphs).scatter_add(0, x.batch, x.x.new_ones(x.x.shape[0]))
            return total / count.clamp(min=1).unsqueeze(-1)
        if x.dim() == 4:
            return x.amax(dim=(2, 3)) if self.mode == "max" else x.mean(dim=(2, 3))
        padding = _ACTIVE_TOKEN_PADDING_MASK.get()
        if padding is not None:
            if tuple(padding.shape) != tuple(x.shape[:2]):
                raise ValueError("text padding mask does not match the pooled sequence")
            if self.mode in ("cls", "first"):
                return x[:, 0]
            valid = padding.unsqueeze(-1).to(dtype=x.dtype)
            if self.mode == "max":
                pooled = x.masked_fill(~padding.unsqueeze(-1), float("-inf")).amax(dim=1)
                # A sequence of padding only pools to zeros, as a mean of nothing does.
                return torch.where(padding.any(dim=1, keepdim=True), pooled, torch.zeros_like(pooled))
            return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        if self.mode in ("cls", "first"):
            return x[:, 0]
        if self.mode == "max":
            return x.amax(dim=1)
        return x.mean(dim=1)


class Dense(nn.Module):
    """A linear map over the last axis, then its activation."""

    def __init__(self, in_features, out_features, bias, function):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.function = activation(function)

    def forward(self, x):
        return per_position(x, lambda h: self.function(self.linear(h)))


_ACTIVE_TOKEN_PADDING_MASK = ContextVar("neurax_token_padding_mask", default=None)


def token_padding_mask(*token_ids):
    """Carry non-padding token positions through one generated model call."""
    return _TokenPaddingScope(token_ids[0] if len(token_ids) == 1 else token_ids)


def active_token_padding_mask(index=0):
    masks = _ACTIVE_TOKEN_PADDING_MASK.get()
    if masks is None:
        return None
    if isinstance(masks, tuple):
        return masks[index]
    if index != 0:
        raise ValueError(f"no token padding mask for input {index}")
    return masks


class _TokenPaddingScope:
    def __init__(self, token_ids):
        self.token_ids = token_ids
        self.token = None

    def __enter__(self):
        token_ids = self.token_ids if isinstance(self.token_ids, tuple) else (self.token_ids,)
        # A sequence of padding only is allowed: a decoder's first step holds
        # only its start token, the padding id. Attention gives a query with no
        # key its own position, and pooling an empty sequence gives zeros.
        masks = tuple(ids.ne(0) for ids in token_ids)
        active = masks[0] if len(masks) == 1 else masks
        self.token = _ACTIVE_TOKEN_PADDING_MASK.set(active)
        return active

    def __exit__(self, exc_type, exc, traceback):
        _ACTIVE_TOKEN_PADDING_MASK.reset(self.token)


class TokenEmbedding(nn.Module):
    """Token ids to vectors, plus BERT's segment table (all segment 0)."""

    def __init__(self, vocab_size, width, segments, padding_idx, tied_to):
        super().__init__()
        self.padding_idx = padding_idx
        self.tied_to = [tied_to] if tied_to is not None else None
        self.tokens = None if tied_to is not None else nn.Embedding(vocab_size, width, padding_idx=padding_idx)
        self.types = nn.Embedding(segments, width) if segments else None
        # GPT's and Llama's initialisation: a unit-variance table read by a
        # tied head starts with logits in the tens.
        for table in (self.tokens, self.types):
            if table is not None:
                nn.init.normal_(table.weight, std=0.02)
                if table.padding_idx is not None:
                    with torch.no_grad():
                        table.weight[table.padding_idx].zero_()

    @property
    def weight(self):
        return self.tied_to[0].weight if self.tied_to is not None else self.tokens.weight

    def forward(self, ids):
        h = F.embedding(ids, self.weight, padding_idx=self.padding_idx)
        if self.types is not None:
            h = h + self.types(torch.zeros_like(ids))
        return h


class PositionScheme(nn.Module):
    """A position scheme applied inside attention (rotary, ALiBi): it passes
    the stream on, and the attention blocks after it are built to use it."""

    def forward(self, x):
        return x


def rotate(x, base, fraction, scale_base):
    """Rotary positions on the last axis of `x`, `[batch, heads, seq, dim]`:
    the first `fraction` of each head rotated, xPos-scaled when `scale_base`."""
    seq, dim = x.shape[-2], x.shape[-1]
    rotated = int(dim * fraction) // 2 * 2
    if rotated == 0:
        return x
    part, rest = x[..., :rotated], x[..., rotated:]
    inv = 1.0 / (base ** (torch.arange(0, rotated, 2, device=x.device, dtype=torch.float32) / rotated))
    angles = torch.outer(torch.arange(seq, device=x.device, dtype=torch.float32), inv)
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = part[..., 0::2], part[..., 1::2]
    turned = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
    if scale_base:
        powers = (torch.arange(seq, device=x.device, dtype=torch.float32) - seq // 2) / scale_base
        zeta = (torch.arange(0, rotated, 2, device=x.device, dtype=torch.float32) + 0.4 * rotated) / (1.4 * rotated)
        turned = turned * (zeta[None, :] ** powers[:, None]).repeat_interleave(2, dim=-1).to(x.dtype)
    return torch.cat((turned, rest), dim=-1)


def alibi_bias(heads, seq, device):
    """ALiBi's per-head linear penalty on distance."""
    slopes = torch.tensor([2 ** (-8.0 * (i + 1) / heads) for i in range(heads)], device=device)
    distance = torch.arange(seq, device=device)[None, :] - torch.arange(seq, device=device)[:, None]
    return slopes[:, None, None] * distance.clamp(max=0)[None]


def attention_mask(queries, keys, causal, window, dilation, block, device, padding_index=0):
    """Which keys each query sees, including batch-specific text padding."""
    allowed = None
    if causal or window or block:
        i = torch.arange(queries, device=device)[:, None]
        j = torch.arange(keys, device=device)[None, :]
        allowed = torch.ones(queries, keys, dtype=torch.bool, device=device)
        if causal:
            allowed &= j <= i
        if window:
            allowed &= (i - j < window) if causal else ((i - j).abs() <= window)
            if dilation and dilation > 1:
                allowed &= (i - j) % dilation == 0
        if block:
            same_block = (i // block) == (j // block)
            summary = (j % block) == (block - 1)
            allowed &= same_block | summary
    padding = active_token_padding_mask(padding_index)
    if padding is None:
        return allowed
    if padding.shape[-1] != keys:
        raise ValueError(f"text padding mask has {padding.shape[-1]} positions but attention has {keys} keys")
    padding = padding[:, None, None, :]
    mask = padding if allowed is None else allowed[None, None, :, :] & padding
    # A query that would see no key at all — a decoder starting on the padding
    # token, its causal row holding only that key — gives a softmax of NaN.
    # It sees itself (or the first key, across two sequences), as a published
    # decoder starting on its padding token does.
    # No branch on the data: models are also built on the meta device.
    blind = ~mask.any(dim=-1, keepdim=True)
    if queries == keys:
        own = torch.eye(queries, keys, dtype=torch.bool, device=device)
    else:
        own = torch.zeros(queries, keys, dtype=torch.bool, device=device)
        own[:, 0] = True
    return mask | (blind & own[None, None, :, :])


class Attention(nn.Module):
    """Multi-head attention: self or cross, full or patterned, softmax or
    kernelised, with the positions of the blocks before it."""

    def __init__(self, width, memory_width, heads, kv_heads, head_dim, qkv_bias, out_bias, qk_norm,
                 dropout, out_projection, causal, window, dilation, block, kernel, rotary, alibi,
                 position_bias, context_length, padding_index=0):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.causal, self.window, self.dilation, self.block = causal, window, dilation, block
        self.kernel, self.rotary, self.alibi = kernel, rotary, alibi
        self.position_bias = [position_bias] if position_bias is not None else None
        self.context_length = context_length
        self.padding_index = padding_index
        self.q = nn.Linear(width, heads * head_dim, bias=qkv_bias)
        self.k = nn.Linear(memory_width, kv_heads * head_dim, bias=qkv_bias)
        self.v = nn.Linear(memory_width, kv_heads * head_dim, bias=qkv_bias)
        norm_width = {"head": (head_dim, head_dim), "projection": (heads * head_dim, kv_heads * head_dim)}
        self.q_norm = RMSNorm(norm_width[qk_norm][0], 1e-6, True) if qk_norm else None
        self.k_norm = RMSNorm(norm_width[qk_norm][1], 1e-6, True) if qk_norm else None
        self.qk_norm = qk_norm
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(heads * head_dim, width, bias=out_bias) if out_projection else None

    def split(self, x, heads):
        return x.view(x.shape[0], x.shape[1], heads, self.head_dim).transpose(1, 2)

    def forward(self, x, memory=None, context=None):
        memory = memory if memory is not None else context
        if memory is None and self.context_length:
            raise ValueError("this attention reads a context of its own; pass it as the model's `context` input")
        memory = x if memory is None else memory
        q, k, v = self.q(x), self.k(memory), self.v(memory)
        if self.qk_norm == "projection":
            q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = self.split(q, self.heads), self.split(k, self.kv_heads), self.split(v, self.kv_heads)
        if self.qk_norm == "head":
            # Over each head's vector: flattened, so the norm reads its last axis.
            q = self.q_norm(q.reshape(-1, self.head_dim)).view(q.shape)
            k = self.k_norm(k.reshape(-1, self.head_dim)).view(k.shape)
        if self.rotary is not None:
            base, fraction, scale_base = self.rotary
            q, k = rotate(q, base, fraction, scale_base), rotate(k, base, fraction, scale_base)
        repeat = self.heads // self.kv_heads
        k, v = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)
        if self.kernel in ("elu", "relu"):
            phi = (lambda t: F.elu(t) + 1) if self.kernel == "elu" else F.relu
            q, k = phi(q), phi(k)
            padding = active_token_padding_mask(self.padding_index)
            if padding is not None:
                if padding.shape[-1] != k.shape[-2]:
                    raise ValueError(
                        f"text padding mask has {padding.shape[-1]} positions but linear attention has {k.shape[-2]} keys"
                    )
                valid = padding[:, None, :, None]
                k = k.masked_fill(~valid, 0)
                v = v.masked_fill(~valid, 0)
            state = k.transpose(-2, -1) @ v
            norm = q @ k.sum(dim=-2, keepdim=True).transpose(-2, -1)
            mixed = (q @ state) / norm.clamp(min=1e-6)
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.alibi:
                scores = scores + alibi_bias(self.heads, scores.shape[-1], x.device)[:, -scores.shape[-2]:]
            if self.position_bias is not None:
                scores = scores + self.position_bias[0].bias(scores.shape[-2], scores.shape[-1], x.device)
            mask = attention_mask(scores.shape[-2], scores.shape[-1], self.causal, self.window, self.dilation,
                                  self.block, x.device, self.padding_index)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            mixed = self.drop(scores.softmax(dim=-1)) @ v
        mixed = mixed.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.heads * self.head_dim)
        return self.out(mixed) if self.out is not None else mixed


class FeedForward(nn.Module):
    """Two linear maps around an activation, or three with a gate (SwiGLU)."""

    def __init__(self, width, inner, bias, function, gated, dropout):
        super().__init__()
        self.first = nn.Linear(width, inner, bias=bias)
        self.up = nn.Linear(width, inner, bias=bias) if gated else None
        self.function = activation(function)
        self.second = nn.Linear(inner, width, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.function(self.first(x))
        if self.up is not None:
            h = h * self.up(x)
        return self.drop(self.second(h))


def empty(value, shape, like):
    """What a block reads when the model is given nothing: the empty
    condition, zeros of its shape."""
    return value if value is not None else like.new_zeros(like.shape[0], *shape)


class LayerHfN4(nn.Module):
    """One layer of `hf-n4`."""

    def __init__(self, shared):
        super().__init__()
        self.b_hf_n5 = Attention(width=320, memory_width=320, heads=20, kv_heads=20, head_dim=16, qkv_bias=True, out_bias=True, qk_norm="projection", dropout=0.0, out_projection=True, causal=False, window=None, dilation=None, block=None, kernel="softmax", rotary=[10000.0, 1.0, None], alibi=False, position_bias=None, context_length=None, padding_index=0)
        self.b_hf_n6 = Residual(dropout=0.0)
        self.b_hf_n7 = LayerNorm(width=320, eps=1e-5, affine=True, bias=True)
        self.b_hf_n8 = FeedForward(width=320, inner=1280, bias=True, function="gelu", gated=False, dropout=0.0)
        self.b_hf_n9 = Residual(dropout=0.0)
        self.b_hf_n10 = LayerNorm(width=320, eps=1e-5, affine=True, bias=True)

    def forward(self, h):
        t_hf_n5 = self.b_hf_n5(h)
        t_hf_n6 = self.b_hf_n6(t_hf_n5, h)
        t_hf_n7 = self.b_hf_n7(t_hf_n6)
        t_hf_n8 = self.b_hf_n8(t_hf_n7)
        t_hf_n9 = self.b_hf_n9(t_hf_n8, t_hf_n7)
        t_hf_n10 = self.b_hf_n10(t_hf_n9)
        return t_hf_n10


class ESM28M(nn.Module):
    def __init__(self):
        super().__init__()
        shared = {}
        self.b_hf_n1 = Identity()
        self.b_hf_n2 = TokenEmbedding(vocab_size=33, width=320, segments=None, padding_idx=0, tied_to=None)
        shared["b_hf_n2"] = self.b_hf_n2
        self.b_hf_n3 = PositionScheme()
        self.b_hf_n4 = nn.ModuleList([LayerHfN4(shared) for _ in range(6)])
        self.b_hf_n11 = LayerNorm(width=320, eps=1e-5, affine=True, bias=True)
        self.b_hf_n12 = GlobalPool(mode="cls")
        self.b_hf_n13 = Dense(in_features=320, out_features=320, bias=True, function=None)
        self.b_hf_n14 = Dense(in_features=320, out_features=5, bias=True, function=None)
        self.b_hf_n15 = Identity()

    def forward(self, x):
        with token_padding_mask(x):
            t_hf_n1 = x
            t_hf_n2 = self.b_hf_n2(t_hf_n1)
            t_hf_n3 = self.b_hf_n3(t_hf_n2)
            h = t_hf_n3
            for n, layer in enumerate(self.b_hf_n4):
                h = layer(h)
            t_hf_n4 = h
            t_hf_n11 = self.b_hf_n11(t_hf_n4)
            t_hf_n12 = self.b_hf_n12(t_hf_n11)
            t_hf_n13 = self.b_hf_n13(t_hf_n12)
            t_hf_n14 = self.b_hf_n14(t_hf_n13)
            t_hf_n15 = self.b_hf_n15(t_hf_n14)
            return t_hf_n15


if __name__ == "__main__":
    model = ESM28M()
    built = sum(p.numel() for p in model.parameters())
    print(f"ESM28M: {built:,} parameters")
    if built != 7517125:
        print("WARNING: NEURAX's analysis counts 7517125 parameters for this design")
