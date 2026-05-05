"""
src/visualizer.py
Handles all drawing operations on frames: bounding boxes, labels, confidence, FPS.
Keeping visualization separate from detection keeps detector.py clean and testable.
"""

import cv2


# A fixed palette of 20 distinct BGR colors — one per class group, cycling for 80 COCO classes
COLORS = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)
]


def get_color(class_id: int) -> tuple:
    """Returns a consistent BGR color for a given class ID by cycling the palette."""
    return COLORS[class_id % len(COLORS)]


def draw_detections(frame, detections: list) -> None:
    """
    Draws bounding boxes and labels directly onto the frame (in-place).

    Args:
        frame:      The OpenCV BGR frame (NumPy array). Modified in-place.
        detections: List of detection dicts from ObjectDetector.detect()
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = get_color(det["class_id"])
        label = f"{det['class_name']} {det['confidence']:.0%}"  # e.g. "person 84%"

        # Draw main bounding box rectangle (2px line thickness)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Calculate the label background size so text is always readable
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # Draw a filled rectangle as background behind the label text
        cv2.rectangle(
            frame,
            (x1, y1 - label_h - baseline - 6),  # top-left corner of label box
            (x1 + label_w, y1),                  # bottom-right corner of label box
            color,
            -1  # -1 = fill the rectangle
        )

        # Write the class name + confidence on top of the filled background
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 3),  # slight upward offset so text sits inside the box
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,    # font scale
            (255, 255, 255),  # white text
            1,      # thickness
            cv2.LINE_AA  # anti-aliased — smoother text rendering
        )


def draw_fps(frame, fps: float) -> None:
    """
    Overlays the FPS counter in the top-left corner of the frame.

    Args:
        frame: The OpenCV BGR frame.
        fps:   Current frames per second to display.
    """
    text = f"FPS: {fps:.1f}"
    # Shadow text (dark) for readability on any background
    cv2.putText(frame, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
    # White foreground text drawn on top of the shadow
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)


def draw_tracks(frame, tracks: list) -> None:
    """
    Draws tracked objects with their persistent Track IDs on the frame (in-place).
    The key difference from draw_detections(): label shows "ID:N class" instead
    of just "class confidence", making tracking visible to the viewer.

    Args:
        frame:  The OpenCV BGR frame. Modified in-place.
        tracks: List of track dicts from ObjectTracker.update()
    """
    for track in tracks:
        x1, y1, x2, y2 = track["bbox"]
        # Cast track_id to int — DeepSORT may return it as a string or other type
        color = get_color(int(track["track_id"]))
        label = f"ID:{track['track_id']} {track['class_name']}"

        # Draw bounding box (slightly thicker at 3px to indicate a confirmed track)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Label background
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(
            frame,
            (x1, y1 - label_h - baseline - 6),
            (x1 + label_w, y1),
            color,
            -1  # filled
        )

        # Label text
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),  # white
            2,
            cv2.LINE_AA
        )
