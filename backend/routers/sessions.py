"""
backend/routers/sessions.py
Cross-task session & analytics endpoints
"""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from backend.session_manager import get_session, list_sessions

router = APIRouter()

@router.get("/", summary="All sessions across all tasks")
async def all_sessions(task: str = None):
    return safe_json_response(list_sessions(task=task or None))

@router.get("/{session_id}", summary="Detailed session record")
async def session_detail(session_id: str, request: Request):
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(404, f"Session {session_id} not found")
    base = str(request.base_url).rstrip("/")
    return safe_json_response({
        "summary":          sess.summary(),
        "records":          sess.records[-200:],
        "annotated_frames": sess.frame_urls(base),
    })
