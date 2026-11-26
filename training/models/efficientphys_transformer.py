"""Paper-based EfficientPhys-T reconstruction, not an official released checkpoint.

References: EfficientPhys section 3.2 and Microsoft Swin Transformer.
Uses PyTorch only. Unspecified architectural defaults are documented in README.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def partition(x, window):
    # NHWC -> (number of windows, window area, channels).
    n, h, w, c = x.shape
    return x.reshape(n, h // window, window, w // window, window, c).permute(
        0, 1, 3, 2, 4, 5).reshape(-1, window * window, c)


def unpartition(x, window, n, h, w):
    return x.reshape(n, h // window, w // window, window, window, -1).permute(
        0, 1, 3, 2, 4, 5).reshape(n, h, w, -1)


class TokenTemporalShift(nn.Module):
    """Shift channel thirds in time; no exchange between separate clips."""
    def __init__(self, segments):
        super().__init__()
        self.segments = segments

    def forward(self, x):
        n, h, w, c = x.shape
        if n % self.segments:
            raise ValueError("Temporal length must be divisible by frame_depth")
        grouped = x.reshape(-1, self.segments, h, w, c)
        shifted = torch.zeros_like(grouped)
        fold = c // 3
        shifted[:, :-1, ..., :fold] = grouped[:, 1:, ..., :fold]
        shifted[:, 1:, ..., fold:2 * fold] = grouped[:, :-1, ..., fold:2 * fold]
        shifted[..., 2 * fold:] = grouped[..., 2 * fold:]
        return shifted.reshape_as(x)


class WindowAttention(nn.Module):
    def __init__(self, dim, window, heads, dropout):
        super().__init__()
        if dim % heads:
            raise ValueError("Embedding dimension must be divisible by attention heads")
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.projection = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.relative_bias = nn.Parameter(torch.zeros((2 * window - 1) ** 2, heads))
        coordinates = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij")).flatten(1)
        relative = coordinates[:, :, None] - coordinates[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous() + window - 1
        index = relative[..., 0] * (2 * window - 1) + relative[..., 1]
        self.register_buffer("relative_index", index, persistent=False)
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def forward(self, x, mask):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, channels // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        logits = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_bias[self.relative_index.reshape(-1)].reshape(tokens, tokens, self.heads).permute(2, 0, 1)
        logits = logits + bias.unsqueeze(0)
        windows = mask.shape[0]
        logits = logits.reshape(-1, windows, self.heads, tokens, tokens) + mask[None, :, None]
        attention = self.dropout(logits.reshape(batch, self.heads, tokens, tokens).softmax(dim=-1))
        output = (attention @ v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.dropout(self.projection(output))


class SwinTemporalBlock(nn.Module):
    """TSM before each Swin block, alternating W-MSA and shifted W-MSA."""
    def __init__(self, dim, heads, window, shift, frame_depth, mlp_ratio, dropout):
        super().__init__()
        self.window, self.shift = window, shift
        self.temporal = TokenTemporalShift(frame_depth)
        self.norm1 = nn.LayerNorm(dim)
        self.attention = WindowAttention(dim, window, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, dim), nn.Dropout(dropout))

    def attention_mask(self, h, w, padded_h, padded_w, shift, device, dtype):
        window = self.window
        regions = torch.zeros((1, padded_h, padded_w, 1), device=device)
        if shift:
            ranges_h = (slice(0, -window), slice(-window, -shift), slice(-shift, None))
            ranges_w = (slice(0, -window), slice(-window, -shift), slice(-shift, None))
            index = 0
            for sh in ranges_h:
                for sw in ranges_w:
                    regions[:, sh, sw, :] = index
                    index += 1
        ids = partition(regions, window).squeeze(-1)
        mask = (ids[:, :, None] != ids[:, None, :]).to(dtype) * -100.0
        # Exclude padded keys, including after the spatial roll.
        valid = torch.zeros((1, padded_h, padded_w, 1), device=device)
        valid[:, :h, :w] = 1
        if shift:
            valid = torch.roll(valid, (-shift, -shift), (1, 2))
        valid = partition(valid, window).squeeze(-1)
        return mask + (valid[:, None, :] == 0).to(dtype) * -100.0

    def forward(self, x):
        x = self.temporal(x)
        residual = x
        n, h, w, _ = x.shape
        window = self.window
        shift = self.shift if min(h, w) > window else 0
        x = self.norm1(x)
        pad_h, pad_w = (-h) % window, (-w) % window
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        padded_h, padded_w = h + pad_h, w + pad_w
        mask = self.attention_mask(h, w, padded_h, padded_w, shift, x.device, x.dtype)
        if shift:
            x = torch.roll(x, (-shift, -shift), (1, 2))
        x = unpartition(self.attention(partition(x, window), mask), window, n, padded_h, padded_w)
        if shift:
            x = torch.roll(x, (shift, shift), (1, 2))
        x = residual + x[:, :h, :w].contiguous()
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        _, h, w, _ = x.shape
        # 72px / 4 -> 18 -> 9 -> 5 -> 3: odd grids require padding.
        x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
        merged = torch.cat((x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                            x[:, 0::2, 1::2], x[:, 1::2, 1::2]), dim=-1)
        return self.reduction(self.norm(merged))


class EfficientPhysTransformer(nn.Module):
    """Diff + BatchNorm + patch embedding + Swin/TSM stages + pulse head."""
    def __init__(self, frame_depth=10, image_size=72, variant="t1", patch_size=4,
                 embed_dim=96, num_heads=None, window_size=7, mlp_ratio=4.0,
                 dropout=0.0, gradient_checkpointing=True):
        super().__init__()
        depths = {"t1": (2, 2, 6, 2), "t2": (2, 1)}
        if variant not in depths:
            raise ValueError("Transformer variant must be t1 or t2")
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        heads = num_heads or ([3, 6, 12, 24] if variant == "t1" else [3, 6])
        if len(heads) != len(depths[variant]):
            raise ValueError("num_heads must have one entry per stage")
        self.frame_depth = frame_depth
        self.image_size = image_size
        self.gradient_checkpointing = gradient_checkpointing
        self.batch_norm = nn.BatchNorm2d(3)
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.stages, self.merges = nn.ModuleList(), nn.ModuleList()
        for stage, depth in enumerate(depths[variant]):
            dim = embed_dim * 2 ** stage
            blocks = nn.ModuleList([
                SwinTemporalBlock(dim, heads[stage], window_size,
                                  0 if i % 2 == 0 else window_size // 2,
                                  frame_depth, mlp_ratio, dropout)
                for i in range(depth)])
            self.stages.append(blocks)
            if stage < len(depths[variant]) - 1:
                self.merges.append(PatchMerging(dim))
        final_dim = embed_dim * 2 ** (len(depths[variant]) - 1)
        self.norm = nn.LayerNorm(final_dim)
        self.head = nn.Linear(final_dim, 1)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, inputs):
        # Like C, the network takes one more frame than the output sequence.
        if inputs.ndim != 4 or inputs.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError("Expected [frames+1, 3, image_size, image_size]")
        if (inputs.shape[0] - 1) % self.frame_depth:
            raise ValueError("Output frames must be divisible by frame_depth")
        x = self.batch_norm(torch.diff(inputs, dim=0))
        x = self.patch_norm(self.patch_embed(x).permute(0, 2, 3, 1))
        for stage, blocks in enumerate(self.stages):
            for block in blocks:
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                    # Non-reentrant checkpoint supports tensors without input gradients.
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
            if stage < len(self.merges):
                x = self.merges[stage](x)
        return self.head(self.norm(x).mean(dim=(1, 2)))
