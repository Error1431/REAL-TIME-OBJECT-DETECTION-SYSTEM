import threading
import time
import cv2
from src.detector import ObjectDetector
from src.visualizer import draw_tracks
import src.database as db

class VideoStreamingEngine:
    def __init__(self, source="0", model_path="yolov8n.pt"):
        self.detector = ObjectDetector(model_path=model_path, confidence=0.4)
        
        # Thread-safe states
        self.lock = threading.Lock()
        self.source = int(source) if source.isdigit() else source
        
        self.latest_frame = None
        self.fps = 0.0
        self.track_count = 0
        self.is_running = True
        
        # Line-crossing telemetry
        self.entries = 0
        self.exits = 0
        self.track_history = {} # track_id -> last centroid_x
        
        # Adjustable parameters
        self.skip_frames = 0
        self.confidence = 0.4
        self.imgsz = 480
        self.target_classes = [] # Empty means detect all
        
        # Start daemon thread
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
    def _capture_loop(self):
        print("[Streamer] Background tracking loop started.")
        
        while self.is_running:
            # Safely grab the current source
            with self.lock:
                current_source = self.source
                
            cap = cv2.VideoCapture(current_source)
            if not cap.isOpened():
                print(f"[Streamer] Could not open source: {current_source}. Retrying in 3s...")
                time.sleep(3)
                continue
                
            frame_count = 0
            fps_start_time = time.time()
            last_tracks = []
            
            while self.is_running:
                # Check for source switch request
                with self.lock:
                    if self.source != current_source:
                        print(f"[Streamer] Switching source from {current_source} to {self.source}")
                        break  # Break inner loop to reload cap
                
                ret, frame = cap.read()
                if not ret:
                    # Video stream ended (e.g. mp4 file). Wait a bit and it will restart outer loop.
                    time.sleep(0.5)
                    break
                    
                # Safely get dynamic parameters
                with self.lock:
                    current_skip = self.skip_frames
                    current_confidence = self.confidence
                    current_imgsz = self.imgsz
                    current_classes = self.target_classes
                
                self.detector.confidence = current_confidence
                
                
                # Width logic for vertical mid-line crossing
                h, w = frame.shape[:2]
                mid_x = int(w / 2)
                
                # Inference
                if frame_count % (current_skip + 1) == 0:
                    tracks = self.detector.track(frame, tracker="bytetrack.yaml", imgsz=current_imgsz)
                    
                    # Filter by class
                    if current_classes:
                        tracks = [t for t in tracks if t['class_name'] in current_classes]
                        
                    last_tracks = tracks
                else:
                    tracks = last_tracks
                    
                # Line Crossing Math
                for track in tracks:
                    tid = track['track_id']
                    x1, y1, x2, y2 = track['bbox']
                    cx = (x1 + x2) / 2
                    
                    last_cx = self.track_history.get(tid)
                    if last_cx is not None:
                        # Crossed left-to-right
                        if last_cx < mid_x and cx >= mid_x:
                            with self.lock:
                                self.entries += 1
                            # Fire and forget SQLite insert loop
                            threading.Thread(target=db.log_event, args=("ENTRY", tid, track['class_name'])).start()
                            
                        # Crossed right-to-left
                        elif last_cx >= mid_x and cx < mid_x:
                            with self.lock:
                                self.exits += 1
                            threading.Thread(target=db.log_event, args=("EXIT", tid, track['class_name'])).start()
                            
                    self.track_history[tid] = cx
                    
                draw_tracks(frame, tracks)
                
                # Draw the vertical line
                cv2.line(frame, (mid_x, 0), (mid_x, h), (0, 0, 255), 2)
                cv2.putText(frame, "ENTRY/EXIT LINE", (mid_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                
                # FPS Calculation
                frame_count += 1
                if frame_count % 10 == 0:
                    elapsed = time.time() - fps_start_time
                    current_fps = 10 / elapsed if elapsed > 0 else 0.0
                    fps_start_time = time.time()
                    with self.lock:
                        self.fps = current_fps
                
                # Store frame 
                ret_jpg, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ret_jpg:
                    with self.lock:
                        self.latest_frame = buffer.tobytes()
                        self.track_count = len(tracks)

            cap.release()

        print("[Streamer] Background tracking loop stopped.")
        
    def get_frame(self):
        with self.lock:
            return self.latest_frame
            
    def get_stats(self):
        with self.lock:
            return {
                "source": str(self.source),
                "fps": round(self.fps, 1),
                "track_count": self.track_count,
                "entries": self.entries,
                "exits": self.exits,
                "confidence": self.confidence,
                "skip_frames": self.skip_frames,
                "imgsz": self.imgsz,
                "target_classes": self.target_classes
            }
            
    def update_settings(self, confidence=None, skip_frames=None, imgsz=None, source=None, target_classes=None):
        with self.lock:
            if confidence is not None:
                self.confidence = float(confidence)
            if skip_frames is not None:
                self.skip_frames = int(skip_frames)
            if imgsz is not None:
                self.imgsz = int(imgsz)
            if source is not None and source != "":
                self.source = int(source) if source.isdigit() else source
            if target_classes is not None:
                self.target_classes = target_classes
    
    def stop(self):
        self.is_running = False
