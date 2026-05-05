"""
src/detector.py
Handles loading the YOLOv8 model and running inference on a single frame.
Keeping detection logic separate means we can swap models later (e.g., to ONNX) 
without touching the pipeline code.
"""

import torch
from ultralytics import YOLO


class ObjectDetector:
    """Wraps a YOLOv8 model and provides a clean detect() interface."""

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.4):
        """
        Args:
            model_path: Path to the .pt weights file. 'yolov8n.pt' auto-downloads on first run.
            confidence: Minimum confidence score to keep a detection (0.0 - 1.0).
                        Detections below this threshold are discarded.
        """
        # Detect the best available device for Apple Silicon
        # MPS = Metal Performance Shaders (Apple GPU). Falls back to CPU if unavailable.
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"[Detector] Using device: {self.device.upper()}")

        # Load the YOLO model weights
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame):
        """
        Runs YOLOv8 inference on a single BGR frame (as returned by OpenCV).

        Args:
            frame: A NumPy array (H x W x 3) in BGR color format.

        Returns:
            A list of detection dicts, each containing:
              - 'bbox': [x1, y1, x2, y2] — bounding box corners in pixels
              - 'confidence': float — model confidence (0.0 to 1.0)
              - 'class_id': int — class index in COCO dataset
              - 'class_name': str — human-readable class label (e.g., 'person')
        """
        # Run inference. verbose=False suppresses per-frame console spam.
        results = self.model(frame, device=self.device, conf=self.confidence, verbose=False)
        
        detections = []
        # results is always a list (one item per image). We only passed one frame.
        result = results[0]

        for box in result.boxes:
            # .xyxy returns a tensor [[x1, y1, x2, y2]]; .tolist() converts to Python floats
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = self.model.names[class_id]

            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(confidence, 2),
                "class_id": class_id,
                "class_name": class_name,
            })

        return detections

    def track(self, frame, persist=True, tracker="bytetrack.yaml", imgsz=480):
        """
        Runs YOLOv8 native tracking (e.g., ByteTrack) on a single frame.
        This bypasses external trackers like DeepSORT and runs much faster.
        """
        results = self.model.track(
            frame, 
            persist=persist, 
            tracker=tracker, 
            device=self.device, 
            conf=self.confidence, 
            imgsz=imgsz, 
            verbose=False
        )

        active_tracks = []
        result = results[0]
        
        # If there are no detections or tracking IDs, result.boxes.id will be None
        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.tolist()
            confidences = result.boxes.conf.tolist()
            class_ids = result.boxes.cls.tolist()
            track_ids = result.boxes.id.tolist()

            for box, conf, cls_id, track_id in zip(boxes, confidences, class_ids, track_ids):
                x1, y1, x2, y2 = box
                class_name = self.model.names[int(cls_id)]
                
                active_tracks.append({
                    "track_id": int(track_id),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "class_name": class_name,
                    "confidence": round(conf, 2),
                })
        
        return active_tracks
