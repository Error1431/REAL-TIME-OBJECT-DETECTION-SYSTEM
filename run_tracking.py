"""
run_tracking.py
Phase 4 entry point: YOLOv8 detection + DeepSORT tracking combined.
Run with:  python run_tracking.py
           python run_tracking.py --source path/to/video.mp4
           python run_tracking.py --source 0           (webcam)

The key difference from run_detection.py:
  - Objects now keep a stable, unique ID (ID:1, ID:2, ...) across frames.
  - The same person will always have the same ID number, even if they briefly
    walk out of frame (up to max_age frames) and return.
"""

import argparse
import time
import cv2
from src.detector import ObjectDetector
from src.visualizer import draw_tracks, draw_fps


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 + ByteTrack Real-Time Tracking")
    parser.add_argument("--source", default="0", help="Webcam index or video file path.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model weights.")
    parser.add_argument("--confidence", type=float, default=0.4, help="Detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=480, help="Input size for YOLOv8.")
    parser.add_argument("--skip-frames", type=int, default=0, help="Number of frames to skip between detections for higher FPS.")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- 1. Initialize detector ---
    detector = ObjectDetector(model_path=args.model, confidence=args.confidence)

    # --- 2. Open the video source ---
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {args.source}")
        print("  -> For webcam: grant camera access in System Settings -> Privacy -> Camera.")
        return

    print(f"[Tracking] Source opened: '{args.source}'")
    print("[Tracking] Press 'Q' to quit.\n")

    fps = 0.0
    frame_count = 0
    fps_start_time = time.time()

    # Store the last known tracks to display during skipped frames
    last_tracks = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Tracking] Stream ended.")
            break

        # --- 3. Process Frame with Skipped Logic ---
        # Only run tracking every (skip_frames + 1) frame
        if frame_count % (args.skip_frames + 1) == 0:
            tracks = detector.track(frame, tracker="bytetrack.yaml", imgsz=args.imgsz)
            last_tracks = tracks
        else:
            # We skipped inference this frame, just use the last known tracks
            tracks = last_tracks

        # --- 5. Draw tracked objects and FPS ---
        draw_tracks(frame, tracks)
        draw_fps(frame, fps)

        # Overlay track count in the top-right corner for easy monitoring
        count_text = f"Tracks: {len(tracks)}"
        cv2.putText(frame, count_text, (frame.shape[1] - 160, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, count_text, (frame.shape[1] - 162, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        # --- 6. FPS Calculation ---
        frame_count += 1
        if frame_count % 10 == 0:
            elapsed = time.time() - fps_start_time
            fps = 10 / elapsed
            fps_start_time = time.time()

        # --- 7. Display ---
        cv2.imshow("YOLOv8 + DeepSORT Tracking (Press Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[Tracking] Stopped by user.")
            break

    # --- 8. Cleanup ---
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
