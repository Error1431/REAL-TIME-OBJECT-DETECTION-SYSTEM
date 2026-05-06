"""
export_onnx.py
Exports the YOLOv8 PyTorch model (.pt) to ONNX format (.onnx).

WHY ONNX?
---------
ONNX (Open Neural Network Exchange) is an open format to represent ML models.
Think of it like a PDF for AI models — once exported:
  - Any ONNX Runtime can load it (Python, C++, Java, mobile, browser...)
  - It is decoupled from PyTorch, making deployment much simpler
  - ONNX Runtime applies graph optimizations and hardware-specific JIT compilation
  - On CPU, ONNX Runtime is typically 2–4× faster than native PyTorch inference

WHY NOT MPS/CUDA for ONNX Runtime?
-----------------------------------
ONNX Runtime on Apple Silicon only supports CPU and CoreML execution providers.
The CoreML provider requires extra setup, so we benchmark on CPU-only, which is
still very competitive thanks to ONNX's graph-level optimizations.

WHAT IS FP16?
-------------
By default, model weights are stored as 32-bit floats (FP32).
FP16 (half precision) stores each weight in 16 bits — half the memory.
  - Speeds up inference on GPUs that support Tensor Cores (NVIDIA) or ANE (Apple)
  - Slight reduction in numeric precision, but virtually no accuracy loss for detection
  - For CPU-only ONNX export, we keep FP32 as FP16 CPU support is limited.

WHAT IS INT8?
-------------
INT8 stores weights as 8-bit integers (0–255). This is "quantization."
  - Further reduces model size (~4× smaller than FP32)
  - Requires a "calibration" step where you feed sample images and the quantizer
    figures out optimal scale factors to map float → int
  - Best for deployment on edge devices (phones, Raspberry Pi, microcontrollers)
  - We skip this for now since it requires a calibration dataset.

Run this script once to export the model, then run benchmark.py to compare speeds.
"""

from ultralytics import YOLO

def main():
    print("Loading YOLOv8n PyTorch model...")
    model = YOLO("yolov8n.pt")

    print("\nExporting to ONNX format...")
    print("  - format: onnx")
    print("  - imgsz: 640  (standard YOLO input size)")
    print("  - opset: 12   (ONNX opset version — 12 has the widest compatibility)\n")

    # Export. Ultralytics handles the full export pipeline internally.
    # The resulting file will be saved alongside the .pt file as 'yolov8n.onnx'
    model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        simplify=True,  # Simplify the ONNX graph: fuses ops, removes redundancies
    )

    print("\n✅ Export complete! File saved as: yolov8n.onnx")
    print("   Run 'python benchmark.py' next to compare PyTorch vs ONNX speed.\n")

if __name__ == "__main__":
    main()
