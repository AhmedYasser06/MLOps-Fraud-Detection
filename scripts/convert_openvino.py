from pathlib import Path

import openvino as ov

INPUT_PATH = Path("models/optimized/fraud_detector.onnx")
OUTPUT_DIR = Path("models/optimized")
OUTPUT_PATH = OUTPUT_DIR / "fraud_detector_openvino.xml"


def main():
    print(f"Loading ONNX: {INPUT_PATH}")

    model = ov.convert_model(str(INPUT_PATH))

    print("Saving OpenVINO IR...")

    ov.save_model(
        model,
        str(OUTPUT_PATH),
    )

    print(f"OpenVINO IR: {OUTPUT_PATH}")
    print(f"BIN file:    {OUTPUT_PATH.with_suffix('.bin')}")
    print("[PASS] OpenVINO conversion complete.")


if __name__ == "__main__":
    main()
