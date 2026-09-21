"""Measure fixed-batch ONNX latency, suitable for direct use on Raspberry Pi 5."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(args.model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    sample = np.random.default_rng(42).standard_normal(
        (1, 3, args.height, args.width), dtype=np.float32
    )
    for _ in range(args.warmup):
        session.run(None, {input_name: sample})

    latencies = []
    for _ in range(args.runs):
        start = time.perf_counter_ns()
        session.run(None, {input_name: sample})
        latencies.append((time.perf_counter_ns() - start) / 1e6)
    values = np.asarray(latencies)
    result = {
        "model": str(args.model),
        "model_size_mib": args.model.stat().st_size / 1024**2,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "threads": args.threads,
        "resolution": [args.height, args.width],
        "warmup": args.warmup,
        "runs": args.runs,
        "latency_mean_ms": float(values.mean()),
        "latency_p50_ms": float(np.percentile(values, 50)),
        "latency_p95_ms": float(np.percentile(values, 95)),
        "fps_from_mean": float(1000.0 / values.mean()),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
