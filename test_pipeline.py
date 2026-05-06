"""
test_pipeline.py
Quick integration test of the Phase 3 pipeline using a static image 
instead of a webcam (so we don't need camera permissions to validate).
"""

import cv2
import time
from src.detector import ObjectDetector
from src.visualizer import draw_detections, draw_fps

def main():
    # Load detector
    detector = ObjectDetector(model_path="yolov8n.pt", confidence=0.4)
    
    # Read the test image from Phase 2
    frame = cv2.imread("test_image.jpg")
    if frame is None:
        print("ERROR: test_image.jpg not found. Run: curl -L -o test_image.jpg https://ultralytics.com/images/zidane.jpg")
        return
    
    print(f"Frame shape: {frame.shape}  (H x W x Channels)")
    
    # Simulate 5 "frames" through the pipeline and measure speed
    total_time = 0.0
    reps = 5
    for i in range(reps):
        t0 = time.perf_counter()
        detections = detector.detect(frame.copy())
        t1 = time.perf_counter()
        total_time += (t1 - t0)
    
    avg_ms = (total_time / reps) * 1000
    simulated_fps = 1000 / avg_ms
    
    print(f"\nAverage inference time over {reps} runs: {avg_ms:.1f}ms")
    print(f"Simulated pipeline FPS:                 {simulated_fps:.1f}")
    
    # Draw results on a fresh copy of the frame
    output = frame.copy()
    detections = detector.detect(output)
    draw_detections(output, detections)
    draw_fps(output, simulated_fps)
    
    # Print detection summary
    print(f"\nObjects detected: {len(detections)}")
    for i, d in enumerate(detections, 1):
        print(f"  {i}. {d['class_name']} @ {d['confidence']:.0%}  bbox={d['bbox']}")
    
    # Save annotated frame
    cv2.imwrite("pipeline_output.jpg", output)
    print("\n✅ Saved annotated image to pipeline_output.jpg")
    print("   Open it in Finder to see the Phase 3 visualization!")

if __name__ == "__main__":
    main()
