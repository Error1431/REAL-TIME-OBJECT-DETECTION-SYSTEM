"""
src/tracker.py
Wraps the DeepSORT tracker and bridges it with our detector's output format.

How DeepSORT works (in plain English):
  Detection alone tells you WHERE objects are in a single frame.
  Tracking tells you WHICH object in frame N is the same as in frame N+1.

  DeepSORT uses two independent signals to match objects across frames:

  1. Kalman Filter (Motion Prediction):
     - A mathematical model that predicts where a moving object WILL BE next frame.
     - Think of it like: "This person was moving right at 5px/frame, so next frame
       they'll probably be ~5 pixels further right."
     - It handles small misses (e.g., a frame where detection fails briefly).

  2. Appearance Feature (Re-Identification / ReID):
     - A deep neural network extracts a 128-dimensional "appearance embedding"
       from the cropped region of each detected object.
     - Think of it like a visual fingerprint — two crops of the same person
       will have very similar embeddings even at different scales/lighting.
     - This lets the tracker recover an ID even after occlusion.

  DeepSORT combines both signals using the Hungarian Algorithm (bipartite matching)
  to find the optimal assignment of new detections to existing tracks.
"""

from deep_sort_realtime.deepsort_tracker import DeepSort


class ObjectTracker:
    """Wraps DeepSORT and converts our detection dicts into its format."""

    def __init__(self, max_age: int = 30):
        """
        Args:
            max_age: How many consecutive MISSED frames before a track is deleted.
                     Set to 30 = a track can disappear for ~1 second (at 30fps) and still be recovered.
        """
        self.tracker = DeepSort(
            max_age=max_age,         # Frames to keep a track alive with no detection hit
            n_init=3,                # Detections needed before a new track is confirmed
            nms_max_overlap=1.0,     # Allow DeepSORT's internal NMS to be permissive (YOLO handles it)
            max_cosine_distance=0.3, # How dissimilar two appearance embeddings can be and still match
            nn_budget=None,          # No limit on the appearance feature gallery
            override_track_class=None,
            embedder="mobilenet",    # MobileNet is the default appearance feature extractor
            half=True,               # Use FP16 for the embedder — faster on Apple Silicon
            bgr=True,                # OpenCV frames are BGR, not RGB — tell the embedder
            embedder_gpu=False,      # MPS is not supported by the embedder; it runs on CPU (fast enough)
        )

    def update(self, detections: list, frame) -> list:
        """
        Feed YOLO detections into DeepSORT and get back tracks with unique IDs.

        Args:
            detections: List of dicts from ObjectDetector.detect()
            frame:      The current BGR frame (needed by DeepSORT's embedder to crop objects)

        Returns:
            List of track dicts, each containing:
              - 'track_id':   int  — stable unique ID for this object across frames
              - 'bbox':       [x1, y1, x2, y2] — bounding box in pixels
              - 'class_name': str  — class label (passed through from detection)
              - 'confidence': float
        """
        # DeepSORT expects detections in a specific format:
        # [ ([left, top, width, height], confidence, class_name), ... ]
        # Note: it wants (left, top, w, h) not (x1,y1,x2,y2)!
        ds_input = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1   # width
            h = y2 - y1   # height
            ds_input.append(([x1, y1, w, h], det["confidence"], det["class_name"]))

        # Pass to DeepSORT. It extracts appearance embeddings and updates the Kalman filters.
        raw_tracks = self.tracker.update_tracks(ds_input, frame=frame)

        # Convert DeepSORT's Track objects back into our clean dict format
        active_tracks = []
        for track in raw_tracks:
            # Skip tracks that haven't been confirmed yet (need n_init=3 hits)
            if not track.is_confirmed():
                continue

            ltrb = track.to_ltrb()  # Returns [left, top, right, bottom] = [x1, y1, x2, y2]
            x1, y1, x2, y2 = [int(v) for v in ltrb]

            active_tracks.append({
                "track_id": track.track_id,
                "bbox": [x1, y1, x2, y2],
                "class_name": track.get_det_class() or "unknown",
                "confidence": track.get_det_conf() or 0.0,
            })

        return active_tracks
