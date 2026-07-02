

import time

import numpy as np
import torch

from configs.config import (
    BENCHMARK_RUNS,
    IMG_SIZE,
)

from src.metrics import save_benchmark


def benchmark_inference(
    model: torch.nn.Module,
    device: torch.device,
    n_runs: int = BENCHMARK_RUNS,
    save_results: bool = True,
) -> dict:

    model.eval()

    dummy_input = torch.randn(
        1,
        3,
        IMG_SIZE,
        IMG_SIZE,
    ).to(device)


    with torch.no_grad():

        for _ in range(10):

            _ = model(dummy_input)

    timings = []

    with torch.no_grad():

        for _ in range(n_runs):

            start = time.perf_counter()

            _ = model(dummy_input)

            if device.type == "cuda":

                torch.cuda.synchronize()

            end = time.perf_counter()

            timings.append(
                (end - start) * 1000
            )

    timings = np.array(
        timings
    )

    benchmark = {

        "device": str(device),

        "runs": n_runs,

        "mean_ms": float(
            timings.mean()
        ),

        "std_ms": float(
            timings.std()
        ),

        "min_ms": float(
            timings.min()
        ),

        "max_ms": float(
            timings.max()
        ),

        "median_ms": float(
            np.median(timings)
        ),

        "p95_ms": float(
            np.percentile(
                timings,
                95,
            )
        ),

        "fps": float(
            1000
            / timings.mean()
        ),

    }

    print()

    print("=" * 60)

    print("Inference Benchmark")

    print("=" * 60)

    print(
        f"Device          : {benchmark['device']}"
    )

    print(
        f"Runs            : {benchmark['runs']}"
    )

    print(
        f"Mean Latency    : {benchmark['mean_ms']:.2f} ms"
    )

    print(
        f"Std             : {benchmark['std_ms']:.2f} ms"
    )

    print(
        f"Minimum         : {benchmark['min_ms']:.2f} ms"
    )

    print(
        f"Median          : {benchmark['median_ms']:.2f} ms"
    )

    print(
        f"95th Percentile : {benchmark['p95_ms']:.2f} ms"
    )

    print(
        f"Maximum         : {benchmark['max_ms']:.2f} ms"
    )

    print(
        f"Throughput      : {benchmark['fps']:.2f} FPS"
    )

    print("=" * 60)

    if save_results:

        save_benchmark(
            benchmark
        )

    return benchmark