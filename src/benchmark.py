import time
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn


def run_standard_benchmark(
    model: nn.Module,
    device: torch.device,
    input_size: Tuple[int, int, int, int] = (32, 3, 224, 224),
    warmup_iterations: int = 50,
    benchmark_iterations: int = 200,
    use_fp16: bool = True
) -> Dict[str, Any]:
    """
    Run standardized efficiency benchmark on GPU/CPU for any model architecture.
    
    Protocol:
      1. Warm-up iterations to stabilize GPU clocks / CUDA context
      2. Timed inference iterations with CUDA synchronization
      3. Compute average, p50, and p95 latency, throughput, and peak VRAM allocation
    """
    import copy
    bench_model = copy.deepcopy(model)
    bench_model.eval()
    bench_model.to(device)

    batch_size, c, h, w = input_size
    dummy_input = torch.randn(input_size, device=device)

    if use_fp16 and device.type == "cuda":
        bench_model = bench_model.half()
        dummy_input = dummy_input.half()

    # Reset VRAM stats if CUDA
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    # Warm-up phase
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = bench_model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Timed benchmark phase
    latencies_ms = []
    with torch.no_grad():
        for _ in range(benchmark_iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            _ = bench_model(dummy_input)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            latency_batch_ms = (end_time - start_time) * 1000.0
            latencies_ms.append(latency_batch_ms)

    latencies_ms = np.array(latencies_ms)
    per_image_latencies_ms = latencies_ms / batch_size

    avg_batch_latency = float(np.mean(latencies_ms))
    p50_batch_latency = float(np.percentile(latencies_ms, 50))
    p95_batch_latency = float(np.percentile(latencies_ms, 95))

    avg_img_latency = float(np.mean(per_image_latencies_ms))
    p50_img_latency = float(np.percentile(per_image_latencies_ms, 50))
    p95_img_latency = float(np.percentile(per_image_latencies_ms, 95))

    throughput_img_per_sec = float((batch_size * 1000.0) / avg_batch_latency)

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))

    benchmark_results = {
        "device": str(device),
        "precision": "FP16" if (use_fp16 and device.type == "cuda") else "FP32",
        "batch_size": batch_size,
        "input_resolution": f"{h}x{w}",
        "warmup_iterations": warmup_iterations,
        "benchmark_iterations": benchmark_iterations,
        "avg_batch_latency_ms": round(avg_batch_latency, 2),
        "p50_batch_latency_ms": round(p50_batch_latency, 2),
        "p95_batch_latency_ms": round(p95_batch_latency, 2),
        "avg_img_latency_ms": round(avg_img_latency, 4),
        "p50_img_latency_ms": round(p50_img_latency, 4),
        "p95_img_latency_ms": round(p95_img_latency, 4),
        "throughput_images_per_sec": round(throughput_img_per_sec, 2),
        "peak_vram_mb": round(peak_vram_mb, 2)
    }

    return benchmark_results


if __name__ == "__main__":
    import argparse
    from utils import load_config, get_device

    parser = argparse.ArgumentParser(description="Run model latency & throughput benchmark.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device, dev_info = get_device(cfg.get("device", "auto"))

    print(f"[Benchmark] Config loaded from {args.config}")
    print(f"[Benchmark] Target device: {device}")
    print(f"[Benchmark] Protocol: {cfg.get('benchmark', {})}")
    print("[Benchmark] Note: Model implementation (ResNet-50) will be attached in the next milestone.")
