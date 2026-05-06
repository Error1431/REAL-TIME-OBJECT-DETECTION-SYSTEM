"""
test_tracking.py
Integration test for Phase 4: validates the full detect → track pipeline
using a static test image (simulating 5 consecutive frames).
"""

import cv2
from src.detector import ObjectDetector
from src.tracker import ObjectTracker
from src.visualizer import draw_tracks, draw_fps

def main():
    detector = ObjectDetector(model_path="yolov8n.pt", confidence=0.4)
    tracker = ObjectTracker(max_age=30)

    frame = cv2.imread("test_image.jpg")
    if frame is None:
        print("ERROR: test_image.jpg not found.")
        return

    print("Simulating 5 frames through the detect → track pipeline...\n")

    tracks = []
    # Feed the same frame multiple times to satisfy DeepSORT's n_init=3 confirmation threshold.
    # In real video this happens naturally across consecutive frames.
    for frame_idx in range(5):
        detections = detector.detect(frame.copy())
        tracks = tracker.update(detections, frame.copy())

        confirmed = [t for t in tracks]
        print(f"Frame {frame_idx + 1}: {len(detections)} detections → {len(confirmed)} confirmed tracks")
        for t in confirmed:
            print(f"    Track ID={t['track_id']}  class={t['class_name']}  bbox={t['bbox']}")

    # Save final annotated frame
    output = frame.copy()
    draw_tracks(output, tracks)
    draw_fps(output, 0.0)  # FPS=0 as placeholder (not measured in static test)
    cv2.imwrite("tracking_output.jpg", output)

    print("\n✅ Saved tracking result to tracking_output.jpg")
    print("   Each object now has a persistent 'ID:N' label!")

if __name__ == "__main__":
    main()
