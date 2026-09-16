"""
Session 3 · CAT 4 (second half) — ONNX -> OpenVINO IR, required alongside
plain ONNX Runtime per the handbook ("CAT 4 is mandatory for everyone").

Converts this repo's exported .onnx graphs (from scripts/sklearn_to_onnx.py)
to OpenVINO's IR format (.xml + .bin) and benchmarks against the plain
ONNX Runtime CPUExecutionProvider path in serving/onnx_service.py, so the
Module 3 report's runtime comparison table has a real OpenVINO row.

Setup:
    pip install openvino

Run:
    python serving/openvino_convert.py --model random_forest
    python serving/openvino_convert.py --model neural_network
    python serving/openvino_convert.py --benchmark random_forest
"""

import argparse
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
ONNX_DIR = BASE_DIR / "onnx_models"
OPENVINO_DIR = BASE_DIR / "openvino_models"


def convert(model_name: str) -> None:
    import openvino as ov

    OPENVINO_DIR.mkdir(exist_ok=True)
    onnx_path = ONNX_DIR / f"{model_name}.onnx"

    core = ov.Core()
    ov_model = core.read_model(onnx_path)
    ov.save_model(ov_model, OPENVINO_DIR / f"{model_name}.xml")
    print(f"[OK] {onnx_path} -> {OPENVINO_DIR / f'{model_name}.xml'} (+ .bin)")


def benchmark(model_name: str, n_requests: int = 500) -> None:
    """A rough single-process latency comparison between the ONNX Runtime
    CPUExecutionProvider path and OpenVINO's runtime, both compiled for
    CPU. For the report's real Locust numbers, benchmark through the
    actual HTTP services instead -- this is a quick sanity check of the
    raw inference call, without HTTP/serialization overhead on top."""
    import onnxruntime as ort
    import openvino as ov

    x = np.random.randn(1, 30).astype(np.float32)

    # ONNX Runtime
    ort_session = ort.InferenceSession(
        str(ONNX_DIR / f"{model_name}.onnx"), providers=["CPUExecutionProvider"]
    )
    input_name = ort_session.get_inputs()[0].name
    ort_session.run(None, {input_name: x})  # warm-up
    start = time.perf_counter()
    for _ in range(n_requests):
        ort_session.run(None, {input_name: x})
    ort_elapsed = time.perf_counter() - start

    # OpenVINO
    core = ov.Core()
    ov_model = core.read_model(OPENVINO_DIR / f"{model_name}.xml")
    compiled = core.compile_model(ov_model, "CPU")
    output_layer = compiled.output(0)
    compiled([x])  # warm-up
    start = time.perf_counter()
    for _ in range(n_requests):
        compiled([x])
    ov_elapsed = time.perf_counter() - start

    print(f"ONNX Runtime : {n_requests / ort_elapsed:,.0f} req/s ({ort_elapsed * 1000 / n_requests:.2f} ms/req)")
    print(f"OpenVINO     : {n_requests / ov_elapsed:,.0f} req/s ({ov_elapsed * 1000 / n_requests:.2f} ms/req)")
    speedup = ort_elapsed / ov_elapsed
    print(f"OpenVINO speedup vs ONNX Runtime: {speedup:.2f}x")
    print("Execution provider (ONNX Runtime): CPUExecutionProvider")
    print("Alternatives on this hardware: OpenVINOExecutionProvider (via onnxruntime-openvino), "
          "or DNNL/oneDNN-backed builds -- worth trying if this model becomes the bottleneck "
          "in the Locust results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model name to convert, e.g. random_forest")
    parser.add_argument("--benchmark", metavar="MODEL", help="benchmark ONNX Runtime vs OpenVINO for this model")
    args = parser.parse_args()

    if args.benchmark:
        benchmark(args.benchmark)
    elif args.model:
        convert(args.model)
    else:
        parser.error("pass --model to convert or --benchmark to compare runtimes")
