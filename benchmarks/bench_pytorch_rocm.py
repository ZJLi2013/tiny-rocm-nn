#!/usr/bin/env python3
"""
PyTorch native MLP benchmark on AMD ROCm.
Compares PyTorch native training/inference throughput for the same
architecture used by tiny-rocm-nn's FullyFusedMLP.

Architecture: OneBlob(n_bins=64) encoding + MLP(128 neurons, 5 hidden, ReLU) + RelativeL2 loss + Adam
Batch size: 2^18 = 262144

Usage:
    python bench_pytorch_rocm.py [--width 128] [--hidden 5] [--batch-pow 18] [--steps 200] [--compile]
"""

import argparse
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class OneBlobEncoding(nn.Module):
    """OneBlob (Gaussian kernel) encoding matching tiny-cuda-nn's implementation."""
    def __init__(self, n_dims, n_bins=64):
        super().__init__()
        self.n_dims = n_dims
        self.n_bins = n_bins
        centers = (torch.arange(n_bins, dtype=torch.float32) + 0.5) / n_bins
        self.register_buffer("centers", centers)
        self.width = 1.0 / n_bins

    def forward(self, x):
        # x: [B, n_dims], output: [B, n_dims * n_bins]
        parts = []
        for d in range(self.n_dims):
            xd = x[:, d:d+1]  # [B, 1]
            diff = (xd - self.centers.unsqueeze(0)) / self.width  # [B, n_bins]
            parts.append(torch.exp(-0.5 * diff * diff))
        return torch.cat(parts, dim=1)

    @property
    def n_output_dims(self):
        return self.n_dims * self.n_bins


class NativeMLP(nn.Module):
    """Standard PyTorch MLP matching FullyFusedMLP architecture."""
    def __init__(self, input_dim, hidden_dim, output_dim, n_hidden_layers):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.hidden = nn.ModuleList(layers)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.output(x)


class FullModel(nn.Module):
    def __init__(self, n_input_dims, n_output_dims, hidden_dim, n_hidden_layers, n_bins):
        super().__init__()
        self.encoding = OneBlobEncoding(n_input_dims, n_bins)
        self.mlp = NativeMLP(self.encoding.n_output_dims, hidden_dim, n_output_dims, n_hidden_layers)

    def forward(self, x):
        return self.mlp(self.encoding(x))


def relative_l2_loss(pred, target):
    """RelativeL2 loss matching tiny-cuda-nn."""
    diff = pred - target
    denom = pred.detach().abs() + 1e-2
    return (diff * diff / (denom * denom)).mean()


def benchmark_training(model, optimizer, batch_size, n_warmup, n_steps, device):
    """Measure training throughput."""
    model.train()
    # Warmup
    for _ in range(n_warmup):
        x = torch.rand(batch_size, 2, device=device, dtype=torch.float32)
        t = torch.rand(batch_size, 3, device=device, dtype=torch.float32)
        pred = model(x)
        loss = relative_l2_loss(pred, t)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_steps):
        x = torch.rand(batch_size, 2, device=device, dtype=torch.float32)
        t = torch.rand(batch_size, 3, device=device, dtype=torch.float32)
        pred = model(x)
        loss = relative_l2_loss(pred, t)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed, n_steps


def benchmark_inference(model, batch_size, n_warmup, n_steps, device):
    """Measure inference throughput."""
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            x = torch.rand(batch_size, 2, device=device, dtype=torch.float32)
            _ = model(x)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_steps):
            x = torch.rand(batch_size, 2, device=device, dtype=torch.float32)
            _ = model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return elapsed, n_steps


def main():
    parser = argparse.ArgumentParser(description="PyTorch native MLP benchmark on ROCm")
    parser.add_argument("--width", type=int, default=128, help="Hidden layer width")
    parser.add_argument("--hidden", type=int, default=5, help="Number of hidden layers")
    parser.add_argument("--n-bins", type=int, default=64, help="OneBlob encoding bins")
    parser.add_argument("--batch-pow", type=int, default=18, help="log2(batch_size)")
    parser.add_argument("--steps", type=int, default=200, help="Benchmark steps")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup steps")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 (AMP)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1 << args.batch_pow

    print(f"{'='*60}")
    print(f"PyTorch Native MLP Benchmark (ROCm)")
    print(f"{'='*60}")
    print(f"Device:       {torch.cuda.get_device_name(0)}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"Network:      {args.width} neurons x {args.hidden} hidden layers")
    print(f"Encoding:     OneBlob (n_bins={args.n_bins})")
    print(f"Batch size:   {batch_size} (2^{args.batch_pow})")
    print(f"Steps:        {args.steps} (warmup={args.warmup})")
    print(f"torch.compile:{args.compile}")
    print(f"FP16 (AMP):   {args.fp16}")
    print()

    model = FullModel(2, 3, args.width, args.hidden, args.n_bins).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters:   {n_params:,}")

    if args.compile:
        model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.99))

    if args.fp16:
        scaler = torch.amp.GradScaler()
        # AMP training
        model.train()
        for _ in range(args.warmup):
            x = torch.rand(batch_size, 2, device=device)
            t = torch.rand(batch_size, 3, device=device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(x)
                loss = relative_l2_loss(pred, t)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.steps):
            x = torch.rand(batch_size, 2, device=device)
            t = torch.rand(batch_size, 3, device=device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(x)
                loss = relative_l2_loss(pred, t)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize()
        train_elapsed = time.perf_counter() - start
        train_steps = args.steps

        # AMP inference
        model.eval()
        with torch.no_grad():
            for _ in range(args.warmup):
                x = torch.rand(batch_size, 2, device=device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    _ = model(x)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.steps):
                x = torch.rand(batch_size, 2, device=device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    _ = model(x)
            torch.cuda.synchronize()
            infer_elapsed = time.perf_counter() - start
            infer_steps = args.steps
    else:
        train_elapsed, train_steps = benchmark_training(
            model, optimizer, batch_size, args.warmup, args.steps, device)
        infer_elapsed, infer_steps = benchmark_inference(
            model, batch_size, args.warmup, args.steps, device)

    train_ms = (train_elapsed / train_steps) * 1000
    train_samples_sec = (batch_size * train_steps) / train_elapsed
    infer_ms = (infer_elapsed / infer_steps) * 1000
    infer_samples_sec = (batch_size * infer_steps) / infer_elapsed

    print(f"\n{'='*60}")
    print(f"Results")
    print(f"{'='*60}")
    print(f"Training:     {train_ms:.2f} ms/step  ({train_samples_sec/1e6:.2f} M samples/sec)")
    print(f"Inference:    {infer_ms:.2f} ms/step  ({infer_samples_sec/1e6:.2f} M samples/sec)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
