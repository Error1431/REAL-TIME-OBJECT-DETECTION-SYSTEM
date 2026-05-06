"""
run_detection.py
Main entry point for the real-time object detection pipeline.
Run with:  python run_detection.py
           python run_detection.py --source path/to/video.mp4
           python run_detection.py --source 0           (webcam index 0)
"""

import argparse
import time
import cv2
from src.detector import ObjectDetector
from src.visualizer import draw_detections, draw_fps


def parse_args():
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(description="YOLOv8 Real-Time Object Detection")
    parser.add_argument(
        "--source",
        default="0",  # '0' means default webcam
        help="Video source: webcam index (0, 1...) or path to a video file."
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",  # Nano model — best speed on M1 8GB
        help="YOLOv8 model weights file (e.g., yolov8n.pt, yolov8s.pt)."
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="Minimum confidence threshold for detections (0.0 to 1.0)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- 1. Initialize the detector ---
    detector = ObjectDetector(model_path=args.model, confidence=args.confidence)

    # --- 2. Open the video source ---
    # If the source is a digit string like "0", convert it to an integer camera index.
    # Otherwise pass the string directly (it's a file path).
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {args.source}")
        print("  -> If using webcam (source=0): Check System Settings -> Privacy -> Camera.")
        print("  -> Grant camera access to Terminal or VS Code, then re-run.")
        return

    print(f"[Pipeline] Source opened: '{args.source}'")
    print("[Pipeline] Press 'Q' in the video window to quit.\n")

    # --- 3. FPS Tracking Variables ---
    # We track wall-clock time between processed frames to compute real FPS.
    fps = 0.0
    frame_count = 0
    fps_start_time = time.time()

    # --- 4. Main Loop ---
    while True:
        # Read the next frame from webcam / video file.
        # ret = True if a frame was successfully read, False at end-of-file or error.
        ret, frame = cap.read()

        if not ret:
            print("[Pipeline] Video stream ended or frame could not be read. Exiting.")
            break

        # --- 5. Run Detection ---
        detections = detector.detect(frame)

        # --- 6. Draw Results ---
        draw_detections(frame, detections)
        draw_fps(frame, fps)

        # --- 7. Calculate FPS every 10 frames ---
        # Updating every frame causes flicker; every 10 frames gives a stable reading.
        frame_count += 1_det
        if frame_count % 10 == 0:
            elapsed = time.time() - fps_start_time
            fps = 10 / elapsed  # 10 frames / elapsed seconds = frames_per_second
            fps_start_time = time.time()  # reset timer for the next 10-frame window

        # --- 8. Display the frame ---
        cv2.imshow("YOLOv8 Real-Time Detection (Press Q to quit)", frame)

        # Wait 1ms for a key press. ord('q') = ASCII code for 'q'.
        # If 'q' is pressed, break out of the loop.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[Pipeline] 'Q' pressed. Stopping.")
            break

    # --- 9. Clean up ---
    cap.release()           # Release the webcam/file handle
    cv2.destroyAllWindows() # Close the display window


if __name__ == "__main__":
    main()
