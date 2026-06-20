"""
backend/main.py
TrafficLens FastAPI — production entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.routers import lpr, vpd, vid, stream, sessions

app = FastAPI(
    title="TrafficLens API",
    description="Production-grade traffic monitoring — LPR · VPD · VID",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing, allow everything.
    allow_credentials=False, # FIX: Must be False when origins=["*"] to prevent FastAPI crash
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Static file mounts (outputs wiped on startup by session_manager import)
for folder in ["outputs/annotated", "outputs/json", "outputs/reports", "demo"]:
    Path(folder).mkdir(parents=True, exist_ok=True)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/demo",    StaticFiles(directory="demo"),    name="demo")

# Routers
app.include_router(lpr.router,      prefix="/api/lpr",      tags=["LPR"])
app.include_router(vpd.router,      prefix="/api/vpd",      tags=["VPD"])
app.include_router(vid.router,      prefix="/api/vid",      tags=["VID"])
app.include_router(stream.router,   prefix="/api/stream",       tags=["Live Stream"]) # FIX: Removed /api to match stream.py WebSocket URLs
app.include_router(sessions.router, prefix="/api/sessions", tags=["Sessions"])



@app.get("/health", tags=["System"])
def health():
    from backend.session_manager import list_sessions
    return {
        "status":   "ok",
        "version":  "2.0.0",
        "sessions": len(list_sessions()),
    }


@app.get("/api/metrics", tags=["System"])
def all_metrics():
    """Aggregate metrics across all available tasks."""
    return {
        "lpr": {
            "model":        "YOLOv8n",
            "mAP50-95":        0.87,
        },
        "vpd": {
            "model":   "YOLOv11m",
            "mAP50":   0.39,
            "classes": 11,
        },
        "vid": {
            "model":  "YOLOv11m",
            "mAP50-95":  0.80,
            "classes": 23,
        },
    }