"""
backend/routers/utils.py
"""

import cv2
import uuid
import json
import numpy as np
from pathlib import Path
from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse
from backend.config import settings


# ── Numpy-safe JSON encoder ────────────────────────────────────────
class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):   return int(obj)
        if isinstance(obj, (np.floating,)):  return float(obj)
        if isinstance(obj, np.ndarray):      return obj.tolist()
        if isinstance(obj, np.bool_):        return bool(obj)
        return super().default(obj)


def safe_json_response(data, status_code: int = 200) -> JSONResponse:
    """Drop-in for JSONResponse that handles numpy scalars."""
    body = json.dumps(data, cls=_NpEncoder)
    return JSONResponse(content=json.loads(body), status_code=status_code)


# ── File helpers ──────────────────────────────────────────────────
async def save_upload_async(file: UploadFile, suffix: str) -> Path:
    tmp_dir = Path("/tmp/trafficlens")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dest = tmp_dir / f"{uuid.uuid4().hex}{suffix}"

    CHUNK    = 1024 * 1024
    size     = 0
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024

    with open(dest, "wb") as out:
        while chunk := await file.read(CHUNK):
            size += len(chunk)
            if size > max_bytes:
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds {settings.MAX_UPLOAD_MB} MB limit")
            out.write(chunk)
    return dest


def save_upload(file: UploadFile, suffix: str) -> Path:
    tmp_dir = Path("/tmp/trafficlens")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dest = tmp_dir / f"{uuid.uuid4().hex}{suffix}"
    with open(dest, "wb") as f:
        f.write(file.file.read())
    return dest


def frame_to_jpeg(frame: np.ndarray, quality: int = 88) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def is_video_ext(filename: str) -> bool:
    return Path(filename).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def multipart_frame(jpeg_bytes: bytes) -> bytes:
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"