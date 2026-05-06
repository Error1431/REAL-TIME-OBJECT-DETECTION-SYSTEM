"""
benchmark.py
Compares inference speed: PyTorch (MPS) vs ONNX Runtime (CPU).
Run AFTER export_onnx.py has been run.

Usage:
    python benchmark.py
"""

import time
import cv2
import numpy as np

from src.detector import ObjectDetector
from src.onnx_detector import ONNXDetector


REPS = 20  # Number of inference runs to average over for a stable reading


def benchmark(detector, frame, label: str) -> float:
    """
    Runs `REPS` inference passes on `frame` and returns average ms per frame.
    First 3 runs are a warm-up (not counted) — they let the runtime JIT-compile.
    """
    print(f"\n[{label}] Warming up (3 runs)...")
    for _ in range(3):
        detector.detect(frame)

    print(f"[{label}] Benchmarking {REPS} runs...")
    start = time.perf_counter()
    for _ in range(REPS):
        detector.detect(frame)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / REPS) * 1000
    fps = 1000 / avg_ms
    print(f"[{label}] Avg: {avg_ms:.1f}ms/frame  →  {fps:.1f} FPS")
    return avg_ms


def main():
    frame = cv2.imread("test_image.jpg")
    if frame is None:
        print("ERROR: test_image.jpg not found.")
        return

    print("=" * 52)
    print("  YOLOv8n Inference Speed Benchmark")
    print("=" * 52)
    print(f"  Image size: {frame.shape[1]}×{frame.shape[0]}  |  Runs: {REPS}")

    # --- Backend 1: PyTorch on MPS ---
    pt_detector = ObjectDetector(model_path="yolov8n.pt", confidence=0.4)
    pt_ms = benchmark(pt_detector, frame, "PyTorch/MPS")

    # --- Backend 2: ONNX Runtime on CPU ---
    onnx_detector = ONNXDetector(onnx_path="yolov8n.onnx", confidence=0.4)
    onnx_ms = benchmark(onnx_detector, frame, "ONNX Runtime/CPU")

    # --- Results summary ---
    speedup = pt_ms / onnx_ms
    print("\n" + "=" * 52)
    print("  RESULTS SUMMARY")
    print("=" * 52)
    print(f"  PyTorch  (MPS):           {pt_ms:>7.1f} ms  ({1000/pt_ms:>5.1f} FPS)")
    print(f"  ONNX Runtime (CPU):       {onnx_ms:>7.1f} ms  ({1000/onnx_ms:>5.1f} FPS)")
    print(f"  Speedup:                  {speedup:.2f}×")
    if speedup >= 1:
        print("  ✅ ONNX Runtime is faster on CPU for this machine.")
    else:
        print("  ℹ️  PyTorch with MPS is faster — your Apple GPU is powerful!")
        print("     ONNX Runtime's advantage is portability and edge deployment,")
        print("     not necessarily raw speed on Apple Silicon vs MPS.")
    print("=" * 52)


if __name__ == "__main__":
    main()
