"""
src/onnx_detector.py
An alternative detector that runs inference using ONNX Runtime instead of PyTorch.
Drops in as a replacement for ObjectDetector — same detect() interface, faster on CPU.

ONNX Runtime vs PyTorch inference:
  PyTorch: Flexible, great for training. Inference has Python overhead per op.
  ONNX Runtime: Fixed graph, heavily optimized C++ execution. No Python overhead per op.
  Result: Typically 2–4× faster for inference-only workloads on CPU.
"""

import cv2
import numpy as np
import onnxruntime as ort

# COCO class names — the 80 object categories YOLOv8 was trained on.
# Index 0 = "person", index 2 = "car", etc.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


class ONNXDetector:
    """
    Runs YOLOv8 inference using ONNX Runtime (CPU).
    Drop-in replacement for ObjectDetector — same detect() interface.
    """

    # Standard YOLOv8 input: 640×640, RGB, normalized to [0.0, 1.0]
    INPUT_SIZE = 640

    def __init__(self, onnx_path: str = "yolov8n.onnx", confidence: float = 0.4):
        """
        Args:
            onnx_path:  Path to the exported .onnx model file.
            confidence: Minimum confidence threshold (0.0–1.0).
        """
        self.confidence = confidence

        # Create an ONNX Runtime InferenceSession.
        # CPUExecutionProvider = use CPU with ONNX Runtime's optimized kernels.
        # On Mac, you could also add 'CoreMLExecutionProvider' for Apple ANE, but
        # it requires extra setup (CoreML model conversion), so we keep it simple.
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"]
        )

        # Read input/output tensor names from the model (don't hardcode them)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"[ONNXDetector] Loaded: {onnx_path}")
        print(f"[ONNXDetector] Input: '{self.input_name}'  Output: '{self.output_name}'")

    def _preprocess(self, frame):
        """
        Prepares a raw OpenCV BGR frame for ONNX inference.
        Steps:
          1. BGR → RGB  (ONNX model was trained on RGB)
          2. Resize to 640×640
          3. Normalize pixel values: [0–255] → [0.0–1.0]
          4. Transpose to [C, H, W]  (channels-first, PyTorch convention)
          5. Add batch dimension → [1, C, H, W]
          6. Convert to float32

        Returns:
            blob: np.ndarray of shape (1, 3, 640, 640) dtype float32
            scale: (scale_x, scale_y) to map predictions back to original frame size
        """
        orig_h, orig_w = frame.shape[:2]
        scale_x = orig_w / self.INPUT_SIZE
        scale_y = orig_h / self.INPUT_SIZE

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.INPUT_SIZE, self.INPUT_SIZE))
        img = img.astype(np.float32) / 255.0     # Normalize
        img = np.transpose(img, (2, 0, 1))        # HWC → CHW
        img = np.expand_dims(img, axis=0)         # CHW → NCHW (batch dim)
        return img, (scale_x, scale_y)

    def _postprocess(self, output, scale_x, scale_y):
        """
        Decodes the raw ONNX output tensor into our detection dict format.

        YOLOv8 ONNX output shape: [1, 84, 8400]
          - 1    = batch size
          - 84   = 4 bbox coords + 80 class scores
          - 8400 = number of anchor predictions across all grid scales

        We transpose to [8400, 84], then filter by confidence and apply NMS.

        Returns: list of detection dicts (same format as ObjectDetector.detect())
        """
        # output shape: [1, 84, 8400] → squeeze and transpose to [8400, 84]
        preds = output[0].squeeze().T   # → (8400, 84)

        # First 4 columns: [cx, cy, w, h] (center x, center y, width, height)
        boxes_cxcywh = preds[:, :4]
        # Remaining 80 columns: per-class confidence scores
        class_scores = preds[:, 4:]

        # For each anchor, find which class has the highest score
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        # Filter by confidence threshold
        mask = confidences >= self.confidence
        boxes_cxcywh = boxes_cxcywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes_cxcywh) == 0:
            return []

        # Convert center-format [cx, cy, w, h] → corner-format [x1, y1, x2, y2]
        # and scale back to the original frame dimensions
        x1 = (boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2) * scale_x
        y1 = (boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2) * scale_y
        x2 = (boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2) * scale_x
        y2 = (boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2) * scale_y

        # Stack into [N, 4] for OpenCV's NMS function
        boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)  # [x, y, w, h] for cv2.dnn.NMSBoxes
        boxes_list = boxes.tolist()
        scores_list = confidences.tolist()

        # Apply Non-Maximum Suppression: keeps only the best box per object
        # (removes duplicate boxes around the same object)
        indices = cv2.dnn.NMSBoxes(
            boxes_list,
            scores_list,
            score_threshold=self.confidence,
            nms_threshold=0.45  # IoU threshold — boxes with >45% overlap are merged
        )

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes_list[i]
                detections.append({
                    "bbox": [int(x), int(y), int(x + w), int(y + h)],
                    "confidence": round(float(confidences[i]), 2),
                    "class_id": int(class_ids[i]),
                    "class_name": COCO_CLASSES[int(class_ids[i])],
                })
        return detections

    def detect(self, frame):
        """
        Full detect() pipeline: preprocess → ONNX run → postprocess.
        Identical interface to ObjectDetector.detect().
        """
        blob, (scale_x, scale_y) = self._preprocess(frame)
        output = self.session.run([self.output_name], {self.input_name: blob})
        return self._postprocess(output, scale_x, scale_y)
