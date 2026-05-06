import torch
from ultralytics import YOLO
import cv2
import sys

def main():
    print("=== Environment Verification ===")
    
    # 1. Check MPS (Metal Performance Shaders for Apple Silicon)
    print("\n1. Checking PyTorch setup...")
    print(f"PyTorch version: {torch.__version__}")
    if torch.backends.mps.is_available():
        print("✅ PyTorch MPS (Apple Silicon GPU) is AVAILABLE!")
    else:
        print("❌ PyTorch MPS is NOT available. Models will run on CPU, which is slower.")
        if not torch.backends.mps.is_built():
            print("   -> MPS not built in this PyTorch install.")

    # 2. Check YOLOv8 Model Download & Load
    print("\n2. Checking YOLOv8 setup...")
    try:
        # Load the smallest YOLOv8 model (Nano)
        model = YOLO("yolov8n.pt")
        print("✅ YOLOv8 loaded successfully!")
    except Exception as e:
        print(f"❌ YOLOv8 failed to load: {e}")

    # 3. Check OpenCV Webcam Access
    print("\n3. Checking OpenCV Webcam setup...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("✅ OpenCV successfully accessed the webcam!")
        ret, frame = cap.read()
        if ret:
            print(f"   Test frame captured successfully (shape: {frame.shape})")
        else:
            print("❌ OpenCV opened the webcam but couldn't capture a frame.")
        cap.release()
    else:
        print("❌ OpenCV could NOT access the webcam.")
        print("   -> On macOS, you might need to grant Terminal/VS Code camera permissions in System Settings -> Privacy & Security.")

if __name__ == "__main__":
    main()
