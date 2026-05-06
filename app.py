from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.streamer import VideoStreamingEngine
import src.database as db
import uvicorn
import asyncio
import os

app = FastAPI(title="YOLOv8 Real-Time Dashboard")

# Serve static files for the frontend, but make sure directory exists
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize global tracking engine
# source "0" for webcam, will error gracefully in logs if no webcam
streamer = VideoStreamingEngine(source="0")

class SettingsRequest(BaseModel):
    confidence: float = None
    skip_frames: int = None
    imgsz: int = None
    source: str = None
    target_classes: list[str] = None

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r") as f:
        return f.read()

async def generate_frames():
    """Generator for MJPEG stream. Automatically caps at ~30 stream FPS."""
    while True:
        frame = streamer.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # Wait slightly to prevent streaming to the browser faster than needed
        await asyncio.sleep(0.033) 

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/stats")
def get_stats():
    return JSONResponse(content=streamer.get_stats())

@app.post("/api/settings")
def update_settings(settings: SettingsRequest):
    streamer.update_settings(
        confidence=settings.confidence,
        skip_frames=settings.skip_frames,
        imgsz=settings.imgsz,
        source=settings.source,
        target_classes=settings.target_classes
    )
    return {"status": "success", "new_settings": streamer.get_stats()}

@app.get("/api/history")
def get_history(minutes: int = 60):
    return JSONResponse(content=db.get_recent_history(minutes))

if __name__ == "__main__":
    print("[Server] Starting FastAPI dashboard at http://localhost:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
