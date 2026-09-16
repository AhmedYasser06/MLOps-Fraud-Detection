from pathlib import Path

import onnx
import onnxruntime as ort

INPUT_PATH = Path("models/optimized/fraud_detector.onnx")
OUTPUT_PATH = Path("models/optimized/fraud_detector_optimized.onnx")


def main():
    print(f"Input : {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")

    session_options = ort.SessionOptions()

    # Enable the highest available ONNX Runtime graph optimization.
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    # Ask ONNX Runtime to save the optimized graph.
    session_options.optimized_model_filepath = str(OUTPUT_PATH)

    print("Running ONNX Runtime graph optimization...")

    session = ort.InferenceSession(
        str(INPUT_PATH),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    print("Optimization complete.")
    print(f"Optimized model: {OUTPUT_PATH}")
    print(f"Providers: {session.get_providers()}")

    # Validate the optimized graph.
    optimized = onnx.load(OUTPUT_PATH)
    onnx.checker.check_model(optimized)

    print("[PASS] Optimized ONNX model is valid.")


if __name__ == "__main__":
    main()
