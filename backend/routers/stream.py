"""
backend/routers/stream.py
Live camera streaming endpoints (MJPEG)

Rebuilt to reuse the exact same techniques already proven working in
lpr.py / vpd.py for uploaded-video and demo-video streaming.
"""

import cv2
import json
import time
import base64
import threading
import asyncio
import numpy as np
from collections import Counter
from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse

from backend.config import settings
from backend.inference import get_lpr_model, get_vpd_model, get_vid_model
from backend.routers.utils import safe_json_response
from backend.session_manager import (
    new_session, close_session,
    draw_plates_scaled, draw_detections_scaled,
)
from preprocess.preprocess import load_and_preprocess

router = APIRouter()

# ====================== SAME TUNING AS vpd.py / lpr.py ======================
cv2.setNumThreads(2)
YOLO_INTERVAL = 6
BOX_TTL = 2
STREAM_JPEG_QUALITY_LPR = 92
STREAM_JPEG_QUALITY_VPD = 75
INFERENCE_SIZE_VPD = (640, 640)
RECONNECT_RETRY_DELAY = 0.5
RECONNECT_MAX_ATTEMPTS = 10


# ====================== FastVideoReader ======================
class FastVideoReader:
    def __init__(self, source, is_live=False, width=1280, height=720):
        self.source = source
        self.is_live = is_live
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(source)
        if is_live:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 else 30.0
        self.frame_interval = 1.0 / self.fps

        self.ret, self.frame = self.cap.read()
        self.timestamp = 0.0
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def _reconnect(self):
        self.cap.release()
        for attempt in range(RECONNECT_MAX_ATTEMPTS):
            time.sleep(RECONNECT_RETRY_DELAY)
            self.cap = cv2.VideoCapture(self.source)
            if self.is_live:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if self.cap.isOpened():
                return True
        return False

    def update(self):
        while not self.stopped:
            t0 = time.monotonic()
            ret, frame = self.cap.read()
            if not ret:
                if self.is_live:
                    if self._reconnect():
                        continue
                self.stopped = True
                break

            timestamp = (
                time.time() if self.is_live
                else self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            )
            with self.lock:
                self.ret = ret
                self.frame = frame
                self.timestamp = timestamp

            if not self.is_live:
                elapsed = time.monotonic() - t0
                sleep_time = self.frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy(), self.timestamp
            return self.ret, None, self.timestamp

    def release(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()


# ====================== Postprocess ======================
LPR_CLASSES = [0]


def _get_class_labels(model):
    """
    Resolve class label map directly from the model.
    Tries common attributes used by ONNX/Ultralytics-exported models.
    Returns a dict {int -> str} or a list indexed by class_id.
    Falls back to reading ONNX session metadata, then to empty dict.
    """
    for attr in ("names", "class_names", "classes", "labels"):
        val = getattr(model, attr, None)
        if val is not None:
            return val
    # Fallback: check ONNX session custom metadata (Ultralytics embeds labels here)
    if hasattr(model, "session"):
        try:
            import ast
            meta = model.session.get_modelmeta().custom_metadata_map
            if "names" in meta:
                return ast.literal_eval(meta["names"])
        except Exception:
            pass
    return {}


def _postprocess_onnx_ultralytics(raw_output, meta, orig_shape, model):
    """
    Corrected post-processing: No double sigmoid, added NMS.
    """
    if isinstance(raw_output, list):
        preds = raw_output[0]
    else:
        preds = raw_output

    if hasattr(preds, 'numpy'):
        preds = preds.numpy()

    # Remove batch dim -> [4+nc, num_anchors]
    if preds.ndim == 3:
        preds = preds[0]

    # Detect layout: Ultralytics non-NMS export gives [4+nc, num_anchors].
    # Transpose to [num_anchors, 4+nc]
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T

    # Split into box coords and per-class probabilities
    boxes_raw  = preds[:, :4]    # cx, cy, w, h
    cls_probs  = preds[:, 4:]    # Probabilities (YOLO ONNX exports generally have sigmoid applied)

    conf_per_anchor   = cls_probs.max(axis=1)
    cls_id_per_anchor = cls_probs.argmax(axis=1)

    conf_threshold = 0.25
    keep = conf_per_anchor >= conf_threshold
    if not keep.any():
        return []

    boxes_raw   = boxes_raw[keep]
    confidences = conf_per_anchor[keep]
    cls_ids     = cls_id_per_anchor[keep]

    # cx, cy, w, h -> x1, y1, w, h for OpenCV NMS
    cx, cy, w, h = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2

    # Undo letterbox padding and rescale to original image coords
    scale = meta.get("scale", 1.0)
    pad   = meta.get("pad", (0, 0))
    pad_w, pad_h = pad if isinstance(pad, (list, tuple)) else (0, 0)

    x1_scaled = (x1 - pad_w) / scale
    y1_scaled = (y1 - pad_h) / scale
    w_scaled  = w / scale
    h_scaled  = h / scale

    # Prepare boxes for NMS
    nms_boxes = []
    for i in range(len(x1_scaled)):
        nms_boxes.append([int(x1_scaled[i]), int(y1_scaled[i]), int(w_scaled[i]), int(h_scaled[i])])

    # Run Non-Maximum Suppression (NMS)
    nms_threshold = 0.45
    indices = cv2.dnn.NMSBoxes(nms_boxes, confidences.tolist(), conf_threshold, nms_threshold)

    if len(indices) == 0:
        return []
        
    # Handle older/newer OpenCV API return formats for indices
    if len(indices.shape) > 1:
        indices = indices.flatten()

    orig_h, orig_w = orig_shape[:2]
    class_labels = _get_class_labels(model)
    dets = []

    for i in indices:
        cls_id = int(cls_ids[i])
        
        if isinstance(class_labels, dict):
            cls_name = class_labels.get(cls_id, str(cls_id))
        elif isinstance(class_labels, (list, tuple)) and cls_id < len(class_labels):
            cls_name = class_labels[cls_id]
        else:
            cls_name = str(cls_id)

        # Calculate final x1, y1, x2, y2 and clip to image bounds
        bx1 = int(nms_boxes[i][0])
        by1 = int(nms_boxes[i][1])
        bx2 = bx1 + int(nms_boxes[i][2])
        by2 = by1 + int(nms_boxes[i][3])

        bx1 = max(0, min(bx1, orig_w))
        by1 = max(0, min(by1, orig_h))
        bx2 = max(0, min(bx2, orig_w))
        by2 = max(0, min(by2, orig_h))

        bbox = [bx1, by1, bx2, by2]

        dets.append({
            "bbox": bbox,
            "box":  bbox,
            "confidence": round(float(confidences[i]), 4),
            "class_id":   cls_id,
            "class_name": cls_name,
        })

    return dets


# ====================== Inference ========================
def _run_lpr_inference(frame: np.ndarray, model):
    blob, original, meta = load_and_preprocess(frame)
    raw = model.session.run(None, {model.input_name: blob})[0]
    plates = model._postprocess(raw, meta, original)
    model.update_unique(plates)
    annotated = draw_plates_scaled(original, plates)
    return annotated, plates


def _run_vpd_inference(frame: np.ndarray, model):
    blob, original, meta = load_and_preprocess(frame, target_size=INFERENCE_SIZE_VPD)
    raw = model.session.run(None, {model.input_name: blob})[0]
    dets = _postprocess_onnx_ultralytics(raw, meta, frame.shape[:2], model)
    annotated = draw_detections_scaled(original, dets)
    return annotated, dets


def _run_vid_inference(frame: np.ndarray, model):
    blob, original, meta = load_and_preprocess(frame, target_size=INFERENCE_SIZE_VPD)
    raw = model.session.run(None, {model.input_name: blob})[0]
    dets = _postprocess_onnx_ultralytics(raw, meta, frame.shape[:2], model)
    annotated = draw_detections_scaled(original, dets)
    return annotated, dets


# ====================== Webcam Generators ======================
async def _lpr_camera_gen(source):
    model = get_lpr_model()
    model.unique_plates = {}
    sess = new_session("lpr", f"stream:{source}")
    video_reader = FastVideoReader(source, is_live=True)
    frame_idx = 0

    try:
        while not video_reader.stopped:
            ret, current_frame, timestamp = video_reader.read()
            if not ret or current_frame is None:
                await asyncio.sleep(0.01)
                continue

            annotated, plates = _run_lpr_inference(current_frame, model)

            if plates:
                sess.add_frame(annotated, plates, frame_idx, timestamp, save_frame=True)

            _, buf = cv2.imencode(
                ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY_LPR]
            )

            master_plates = list(model.unique_plates.values())
            headers_bytes = (
                f"X-Count: {len(master_plates)}\r\n"
                f"X-Plates: {json.dumps(master_plates, default=str)}\r\n"
                f"X-Timestamp: {round(timestamp, 3)}\r\n"
            ).encode('utf-8')

            chunk = (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n'
                + headers_bytes +
                b'\r\n'
                + buf.tobytes() + b'\r\n'
            )
            yield chunk
            frame_idx += 1
            await asyncio.sleep(0)
    finally:
        video_reader.release()
        close_session(sess.session_id)


async def _vpd_camera_gen(source, yolo_interval: int):
    model = get_vpd_model()
    sess = new_session("vpd", f"stream:{source}")
    video_reader = FastVideoReader(source, is_live=True)
    frame_idx = 0
    frames_since_inference = 0
    last_dets = []

    try:
        while not video_reader.stopped:
            ret, current_frame, timestamp = video_reader.read()
            if not ret or current_frame is None:
                await asyncio.sleep(0.01)
                continue

            if frame_idx % yolo_interval == 0:
                annotated, dets = _run_vpd_inference(current_frame, model)
                last_dets = dets
                frames_since_inference = 0
                if dets:
                    sess.add_frame(annotated, dets, frame_idx, timestamp, save_frame=True)
            else:
                frames_since_inference += 1
                if frames_since_inference < BOX_TTL:
                    annotated = draw_detections_scaled(current_frame, last_dets)
                else:
                    annotated = current_frame

            _, buf = cv2.imencode(
                ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY_VPD]
            )

            class_summary = dict(Counter(d["class_name"] for d in last_dets))
            headers_bytes = (
                f"X-Count: {len(last_dets)}\r\n"
                f"X-Classes: {json.dumps(class_summary)}\r\n"
                f"X-Detections: {json.dumps(last_dets, default=str)}\r\n"
                f"X-Timestamp: {round(timestamp, 3)}\r\n"
            ).encode('utf-8')

            chunk = (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n'
                + headers_bytes +
                b'\r\n'
                + buf.tobytes() + b'\r\n'
            )
            yield chunk
            frame_idx += 1
            await asyncio.sleep(0.001)
    finally:
        video_reader.release()
        close_session(sess.session_id)


async def _vid_camera_gen(source, yolo_interval: int):
    model = get_vid_model()
    sess = new_session("vid", f"stream:{source}")
    video_reader = FastVideoReader(source, is_live=True)
    frame_idx = 0
    frames_since_inference = 0
    last_dets = []

    try:
        while not video_reader.stopped:
            ret, current_frame, timestamp = video_reader.read()
            if not ret or current_frame is None:
                await asyncio.sleep(0.01)
                continue

            if frame_idx % yolo_interval == 0:
                annotated, dets = _run_vid_inference(current_frame, model)
                last_dets = dets
                frames_since_inference = 0
                if dets:
                    sess.add_frame(annotated, dets, frame_idx, timestamp, save_frame=True)
            else:
                frames_since_inference += 1
                if frames_since_inference < BOX_TTL:
                    annotated = draw_detections_scaled(current_frame, last_dets)
                else:
                    annotated = current_frame

            _, buf = cv2.imencode(
                ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY_VPD]
            )

            class_summary = dict(Counter(d["class_name"] for d in last_dets))
            headers_bytes = (
                f"X-Count: {len(last_dets)}\r\n"
                f"X-Classes: {json.dumps(class_summary)}\r\n"
                f"X-Detections: {json.dumps(last_dets, default=str)}\r\n"
                f"X-Timestamp: {round(timestamp, 3)}\r\n"
            ).encode('utf-8')

            chunk = (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n'
                + headers_bytes +
                b'\r\n'
                + buf.tobytes() + b'\r\n'
            )
            yield chunk
            frame_idx += 1
            await asyncio.sleep(0.001)
    finally:
        video_reader.release()
        close_session(sess.session_id)


# ====================== Mobile Camera — HTML page ===========================
MOBILE_CAM_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>Mobile Camera — {task_label}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0a0a0a; color: #e0e0e0; font-family: sans-serif; display: flex; flex-direction: column; align-items: center; min-height: 100dvh; padding: 12px; gap: 10px; }}
  h2 {{ font-size: 1rem; font-weight: 600; color: #a78bfa; text-transform: uppercase; }}
  #status {{ font-size: 0.78rem; color: #6b7280; text-align: center; }}
  #status.ok  {{ color: #34d399; }}
  #status.err {{ color: #f87171; }}
  .video-wrap {{ position: relative; width: 100%; max-width: 480px; border-radius: 12px; overflow: hidden; background: #111; border: 1px solid #1f2937; }}
  video, #annotated {{ width: 100%; display: block; }}
  #annotated {{ position: absolute; top: 0; left: 0; opacity: 0; transition: opacity 0.15s; }}
  #annotated.visible {{ opacity: 1; }}
  .controls {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; width: 100%; max-width: 480px; }}
  button {{ flex: 1; min-width: 120px; padding: 11px 16px; border-radius: 8px; border: none; font-size: 0.875rem; font-weight: 600; cursor: pointer; transition: 0.15s; }}
  button:disabled {{ opacity: 0.4; cursor: not-allowed; }}
  #btnStart  {{ background: #7c3aed; color: #fff; }}
  #btnStop   {{ background: #1f2937; color: #9ca3af; }}
  #btnFlip   {{ background: #1f2937; color: #9ca3af; }}
  #stats {{ width: 100%; max-width: 480px; background: #111827; border-radius: 10px; padding: 10px 14px; font-size: 0.8rem; line-height: 1.8; border: 1px solid #1f2937; min-height: 60px; word-break: break-word; }}
  #stats span {{ color: #a78bfa; font-weight: 600; }}
  #fps-badge {{ position: absolute; top: 8px; right: 8px; background: rgba(0,0,0,0.6); color: #34d399; font-size: 0.7rem; font-weight: 700; padding: 2px 7px; border-radius: 20px; pointer-events: none; }}
</style>
</head>
<body>
<h2>📷 {task_label} · Mobile Camera</h2>
<p id="status">Tap Start to begin</p>
<div class="video-wrap">
  <video id="video" autoplay playsinline muted></video>
  <img id="annotated" alt="annotated">
  <div id="fps-badge">-- fps</div>
</div>
<div class="controls">
  <button id="btnStart">▶ Start</button>
  <button id="btnStop" disabled>■ Stop</button>
  <button id="btnFlip">🔄 Flip</button>
</div>
<div id="stats">Waiting for detections…</div>
<canvas id="canvas" style="display:none"></canvas>
<script>
const TASK      = "{task}";
const WS_URL    = "{ws_url}";
const JPEG_Q    = 0.95;          // FIX: Increased from 0.75 for much better LPR OCR accuracy
const SEND_FPS  = 10;            
const SEND_MS   = 1000 / SEND_FPS;

let ws, stream, sendTimer;
let facingMode = "environment";
let frameCount = 0;
let fpsInterval;

const video      = document.getElementById("video");
const annotated  = document.getElementById("annotated");
const canvas     = document.getElementById("canvas");
const ctx        = canvas.getContext("2d");
const status     = document.getElementById("status");
const stats      = document.getElementById("stats");
const fpsBadge   = document.getElementById("fps-badge");
const btnStart   = document.getElementById("btnStart");
const btnStop    = document.getElementById("btnStop");
const btnFlip    = document.getElementById("btnFlip");

function setStatus(msg, cls="") {{ status.textContent = msg; status.className = cls; }}

async function startCamera() {{
  try {{
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = await navigator.mediaDevices.getUserMedia({{
      video: {{ facingMode, width: {{ ideal: 1280 }}, height: {{ ideal: 720 }} }},
      audio: false,
    }});
    video.srcObject = stream;
    await video.play();
    setStatus("Camera ready — connecting…");
    openWS();
  }} catch(e) {{
    setStatus("Camera error: " + e.message, "err");
  }}
}}

function openWS() {{
  ws = new WebSocket(WS_URL);
  ws.binaryType = "blob";

  ws.onopen = () => {{
    setStatus("Connected · streaming", "ok");
    btnStart.disabled = true;
    btnStop.disabled  = false;
    startSending();
    startFpsCounter();
  }};

  ws.onmessage = (evt) => {{
    const msg = JSON.parse(evt.data);
    if (msg.annotated_b64) {{
      annotated.src = "data:image/jpeg;base64," + msg.annotated_b64;
      annotated.classList.add("visible");
    }}
    renderStats(msg.meta || {{}});
    frameCount++;
  }};

  ws.onclose  = () => {{ setStatus("Disconnected", "err"); stopAll(); }};
  ws.onerror  = () => setStatus("WebSocket error", "err");
}}

function startSending() {{
  sendTimer = setInterval(() => {{
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!video.videoWidth) return;
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    canvas.toBlob(blob => blob && ws.send(blob), "image/jpeg", JPEG_Q);
  }}, SEND_MS);
}}

function startFpsCounter() {{
  fpsInterval = setInterval(() => {{
    fpsBadge.textContent = frameCount + " fps";
    frameCount = 0;
  }}, 1000);
}}

function stopAll() {{
  clearInterval(sendTimer);
  clearInterval(fpsInterval);
  if (ws && ws.readyState === WebSocket.OPEN) ws.close();
  if (stream) stream.getTracks().forEach(t => t.stop());
  btnStart.disabled = false;
  btnStop.disabled  = true;
  annotated.classList.remove("visible");
  fpsBadge.textContent = "-- fps";
}}

function renderStats(meta) {{
  if (TASK === "lpr") {{
    const plates = meta.plates || [];
    stats.innerHTML = plates.length
      ? "<span>Plates detected:</span> " + plates.map(p =>
          `<b style="color:#fbbf24">${{p.plate_text || p}}</b>`).join(", ")
      : "No plates in frame";
  }} else {{
    const cls = meta.class_summary || {{}};
    const entries = Object.entries(cls);
    stats.innerHTML = entries.length
      ? entries.map(([k,v]) => `<span>${{k}}</span>: ${{v}}`).join(" &nbsp;·&nbsp; ")
      : "No detections in frame";
  }}
}}

btnStart.onclick = startCamera;
btnStop.onclick  = stopAll;
btnFlip.onclick  = () => {{
  facingMode = facingMode === "environment" ? "user" : "environment";
  if (stream) startCamera();
}};
</script>
</body>
</html>
"""


@router.get("/mobile-cam", response_class=HTMLResponse,
            summary="Mobile camera capture page (open on phone browser)")
async def mobile_cam_page(request: Request, task: str = Query("lpr", enum=["lpr", "vpd", "vid"])):
    host = request.headers.get("host", "localhost:8000")
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    scheme = "wss" if proto == "https" else "ws"
    ws_path = request.url.path.replace("/mobile-cam", "/mobile-ws")

    ws_url = f"{scheme}://{host}{ws_path}?task={task}"
    task_label = "License Plate Recognition" if task == "lpr" else "Vehicle & Person Detection"
    task_label = "Violation Detection" if task == "vid" else task_label

    html = MOBILE_CAM_HTML.format(
        task=task,
        task_label=task_label,
        ws_url=ws_url,
    )
    return HTMLResponse(content=html)


@router.websocket("/mobile-ws")
async def mobile_cam_ws(
    websocket: WebSocket,
    task: str = Query("lpr", enum=["lpr", "vpd", "vid"]),
):
    await websocket.accept()

    if task == "lpr":
        model = get_lpr_model()
        model.unique_plates = {}
        sess = new_session("lpr", "mobile-camera")
        jpeg_q = STREAM_JPEG_QUALITY_LPR
    elif task == "vpd":
        model = get_vpd_model()
        sess = new_session("vpd", "mobile-camera")
        jpeg_q = STREAM_JPEG_QUALITY_VPD
    else:
        model = get_vid_model()
        sess = new_session("vid", "mobile-camera")
        jpeg_q = STREAM_JPEG_QUALITY_VPD

    frame_idx = 0
    frames_since_inference = 0
    last_dets = []

    try:
        while True:
            data = await websocket.receive_bytes()
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            timestamp = time.time()

            # FIX: Added inner try/except. A single bad frame won't break the connection anymore.
            try:
                if task == "lpr":
                    annotated, plates = _run_lpr_inference(frame, model)
                    if plates:
                        print(f"✅ DETECTED PLATES: {plates}")  # FIX: Log to terminal so you know it works
                        sess.add_frame(annotated, plates, frame_idx, timestamp, save_frame=True)
                    master_plates = list(model.unique_plates.values())
                    meta = {
                        "count": len(plates),
                        "plates": master_plates,
                    }

                elif task == "vpd":
                    if frame_idx % YOLO_INTERVAL == 0:
                        annotated, dets = _run_vpd_inference(frame, model)
                        last_dets = dets
                        frames_since_inference = 0
                        if dets:
                            sess.add_frame(annotated, dets, frame_idx, timestamp, save_frame=True)
                    else:
                        frames_since_inference += 1
                        if frames_since_inference < BOX_TTL:
                            annotated = draw_detections_scaled(frame, last_dets)
                        else:
                            annotated = frame
                    class_summary = dict(Counter(d["class_name"] for d in last_dets))
                    meta = {
                        "count": len(last_dets),
                        "class_summary": class_summary,
                        "detections": last_dets,
                    }

                else:  # vid
                    if frame_idx % YOLO_INTERVAL == 0:
                        annotated, dets = _run_vid_inference(frame, model)
                        last_dets = dets
                        frames_since_inference = 0
                        if dets:
                            sess.add_frame(annotated, dets, frame_idx, timestamp, save_frame=True)
                    else:
                        frames_since_inference += 1
                        if frames_since_inference < BOX_TTL:
                            annotated = draw_detections_scaled(frame, last_dets)
                        else:
                            annotated = frame
                    class_summary = dict(Counter(d["class_name"] for d in last_dets))
                    meta = {
                        "count": len(last_dets),
                        "class_summary": class_summary,
                        "detections": last_dets,
                    }

                _, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
                b64 = base64.b64encode(buf.tobytes()).decode("ascii")

                await websocket.send_text(json.dumps({
                    "annotated_b64": b64,
                    "meta": meta,
                    "frame_idx": frame_idx,
                    "timestamp": round(timestamp, 3),
                }))

                frame_idx += 1

            except Exception as inference_err:
                print(f"⚠️ Inference Error on frame {frame_idx}: {inference_err}")
                continue  # Skip this frame, keep connection open

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket Error: {e}")
    finally:
        close_session(sess.session_id)


# ====================== Webcam Routes ======================
@router.get("/lpr", summary="Live MJPEG stream — LPR inference on webcam or IP Cam")
async def stream_lpr(camera: str = Query("0")):
    # FIX: Allow URLs to be passed in for IP Camera apps
    source = int(camera) if camera.isdigit() else camera

    test_cap = cv2.VideoCapture(source)
    if not test_cap.isOpened():
        test_cap.release()
        raise HTTPException(503, f"Cannot open source {source}")
    test_cap.release()

    return StreamingResponse(
        _lpr_camera_gen(source),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/vpd", summary="Live MJPEG stream — VPD inference on webcam or IP Cam")
async def stream_vpd(camera: str = Query("0"), yolo_interval: int = Query(YOLO_INTERVAL, ge=1, le=30)):
    # FIX: Allow URLs to be passed in for IP Camera apps
    source = int(camera) if camera.isdigit() else camera

    test_cap = cv2.VideoCapture(source)
    if not test_cap.isOpened():
        test_cap.release()
        raise HTTPException(503, f"Cannot open source {source}")
    test_cap.release()

    return StreamingResponse(
        _vpd_camera_gen(source, yolo_interval),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/vid", summary="Live MJPEG stream — VID inference on webcam or IP Cam")
async def stream_vid(camera: str = Query("0"), yolo_interval: int = Query(YOLO_INTERVAL, ge=1, le=30)):
    # FIX: Allow URLs to be passed in for IP Camera apps
    source = int(camera) if camera.isdigit() else camera

    test_cap = cv2.VideoCapture(source)
    if not test_cap.isOpened():
        test_cap.release()
        raise HTTPException(503, f"Cannot open source {source}")
    test_cap.release()

    return StreamingResponse(
        _vid_camera_gen(source, yolo_interval),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/cameras", summary="List available camera indices")
async def list_cameras():
    available = []
    for i in range(4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            available.append({"index": i, "resolution": f"{w}x{h}"})
            cap.release()
    return safe_json_response({"cameras": available})