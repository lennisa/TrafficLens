"""
Violation Detection — BLAZING FAST VERSION
Optimized for real-time video streaming with Multiplexed Bounding Boxes & Microsecond TTL
"""

import cv2
import threading
import time
import asyncio
import json
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from backend.config import settings
from backend.inference import get_vid_model
from backend.routers.utils import save_upload_async, frame_to_jpeg, multipart_frame, safe_json_response
from backend.session_manager import (
    new_session, get_session, list_sessions, close_session,
    draw_detections_scaled,
)
from preprocess.preprocess import load_and_preprocess
from collections import Counter
import numpy as np

# ====================== PERFORMANCE TUNING ======================
cv2.setNumThreads(2)
YOLO_INTERVAL = 6
STREAM_JPEG_QUALITY = 75
INFERENCE_SIZE = (640, 640)
BOX_TTL = 2

router = APIRouter()


class FastVideoReader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 else 30.0
        self.frame_interval = 1.0 / self.fps   # seconds per frame at original speed

        self.ret, self.frame = self.cap.read()
        self.timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        self.stopped = False
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            t0 = time.monotonic()
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            ts = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            with self._lock:
                self.ret = ret
                self.frame = frame
                self.timestamp = ts
            # Sleep for the remainder of the frame interval so the reader
            # advances at exactly the video's native frame rate and doesn't
            # race ahead of the streaming loop, which was the root cause of
            # frames being skipped / playback running too fast.
            elapsed = time.monotonic() - t0
            sleep_time = self.frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def read(self):
        with self._lock:
            # Return a copy so the streaming loop holds a stable reference
            # even if the reader thread overwrites self.frame immediately.
            return self.ret, self.frame.copy() if self.frame is not None else None

    def get_timestamp(self):
        with self._lock:
            return self.timestamp

    def release(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()


# ====================== Dynamic Class Labels ======================
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


def _run_inference_on_frame(frame: np.ndarray, model):
    blob, original, meta = load_and_preprocess(frame, target_size=INFERENCE_SIZE)
    raw = model.session.run(None, {model.input_name: blob})[0]
    dets = _postprocess_onnx_ultralytics(raw, meta, frame.shape[:2], model)
    annotated = draw_detections_scaled(original, dets)
    return annotated, dets


@router.post("/image")
async def vid_image(request: Request, file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    tmp = await save_upload_async(file, suffix)
    try:
        sess = new_session("vid", file.filename)
        model = get_vid_model()
        frame = cv2.imread(str(tmp))
        annotated, dets = _run_inference_on_frame(frame, model)
        record = sess.add_frame(annotated, dets, 0, 0.0)
        close_session(sess.session_id)
        base = str(request.base_url).rstrip("/")
        class_summary = dict(Counter(d["class_name"] for d in dets))
        return safe_json_response({
            "session_id": sess.session_id,
            "task": "vid",
            "source": file.filename,
            "image_dims": {"w": annotated.shape[1], "h": annotated.shape[0]},
            "count": len(dets),
            "detections": dets,
            "class_summary": class_summary,
            "annotated_url": f"{base}/outputs/annotated/{sess.session_id}/{(record.get('annotated_image') or '').split('/')[-1]}" if record.get("annotated_image") else None,
            "summary": sess.summary(),
        })
    finally:
        tmp.unlink(missing_ok=True)


@router.post("/image/annotated")
async def vid_image_annotated(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    tmp = await save_upload_async(file, suffix)
    try:
        model = get_vid_model()
        frame = cv2.imread(str(tmp))
        annotated, dets = _run_inference_on_frame(frame, model)
        _, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return StreamingResponse(iter([buf.tobytes()]), media_type="image/jpeg", headers={"X-Count": str(len(dets))})
    finally:
        tmp.unlink(missing_ok=True)


@router.post("/video", summary="Stream VID inference on an uploaded video")
async def vid_video(
    request: Request,
    file: UploadFile = File(...),
    yolo_interval: int = Query(YOLO_INTERVAL, ge=1, le=30),
):
    tmp = await save_upload_async(file, Path(file.filename).suffix.lower())
    sess = new_session("vid", file.filename)
    model = get_vid_model()

    async def generate():
        video_reader = FastVideoReader(str(tmp))
        frame_sleep = video_reader.frame_interval
        frame_idx = 0
        frames_since_inference = 0
        last_dets = []

        try:
            while not video_reader.stopped:
                t_frame_start = asyncio.get_event_loop().time()

                ret, frame = video_reader.read()
                if not ret or frame is None:
                    await asyncio.sleep(0.005)
                    continue

                timestamp = video_reader.get_timestamp()

                if frame_idx % yolo_interval == 0:
                    annotated, dets = _run_inference_on_frame(frame, model)
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

                _, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY])

                class_summary = dict(Counter(d["class_name"] for d in last_dets))
                headers_bytes = (
                    f"X-Count: {len(last_dets)}\r\n"
                    f"X-Classes: {json.dumps(class_summary)}\r\n"
                    f"X-Detections: {json.dumps(last_dets)}\r\n"
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

                elapsed = asyncio.get_event_loop().time() - t_frame_start
                sleep_time = frame_sleep - elapsed
                await asyncio.sleep(max(0.0, sleep_time))

        finally:
            video_reader.release()
            tmp.unlink(missing_ok=True)
            close_session(sess.session_id)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"X-Session-Id": sess.session_id},
    )


@router.post("/video/analyze")
async def vid_video_analyze(
    request: Request,
    file: UploadFile = File(...),
    yolo_interval: int = Query(YOLO_INTERVAL, ge=1, le=30),
    save_frames: bool = Query(True),
    frame_save_interval: int = Query(15, ge=1),
):
    tmp = await save_upload_async(file, Path(file.filename).suffix.lower())
    sess = new_session("vid", file.filename)
    model = get_vid_model()
    base = str(request.base_url).rstrip("/")
    try:
        from preprocess.preprocess import VideoStream
        frame_results = []
        saved_count = 0
        with VideoStream(str(tmp), skip_frames=0, enhance=False) as vs:
            for idx, timestamp, frame, blob, meta in vs:
                if idx % yolo_interval != 0:
                    continue
                raw = model.session.run(None, {model.input_name: blob})[0]
                dets = _postprocess_onnx_ultralytics(raw, meta, frame.shape[:2], model)
                should_save = save_frames and (saved_count % frame_save_interval == 0)
                ann = draw_detections_scaled(frame, dets) if dets else None
                if dets:
                    sess.add_frame(ann, dets, idx, timestamp, save_frame=should_save)
                    saved_count += 1
                    frame_results.append({
                        "frame_idx": idx,
                        "timestamp": round(timestamp, 3),
                        "count": len(dets),
                        "classes": dict(Counter(d["class_name"] for d in dets)),
                        "detections": dets,
                    })
        close_session(sess.session_id)
        return safe_json_response({
            "session_id": sess.session_id,
            "summary": sess.summary(),
            "frame_results": frame_results,
            "annotated_frames": sess.frame_urls(base),
        })
    finally:
        tmp.unlink(missing_ok=True)


@router.get("/demo/video")
async def vid_demo_video():
    for ext in [".mp4", ".avi", ".mov"]:
        p = settings.DEMO_DIR / f"vid{ext}"
        if p.exists():
            sess = new_session("vid", f"demo{ext}")
            model = get_vid_model()

            async def generate():
                video_reader = FastVideoReader(str(p))
                frame_sleep = video_reader.frame_interval
                frame_idx = 0
                frames_since_inference = 0
                last_dets = []
                try:
                    while not video_reader.stopped:
                        t_frame_start = asyncio.get_event_loop().time()

                        ret, frame = video_reader.read()
                        if not ret or frame is None:
                            await asyncio.sleep(0.005)
                            continue
                        timestamp = video_reader.get_timestamp()
                        if frame_idx % YOLO_INTERVAL == 0:
                            annotated, dets = _run_inference_on_frame(frame, model)
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
                        _, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY])

                        class_summary = dict(Counter(d["class_name"] for d in last_dets))
                        headers_bytes = (
                            f"X-Count: {len(last_dets)}\r\n"
                            f"X-Classes: {json.dumps(class_summary)}\r\n"
                            f"X-Detections: {json.dumps(last_dets)}\r\n"
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

                        elapsed = asyncio.get_event_loop().time() - t_frame_start
                        await asyncio.sleep(max(0.0, frame_sleep - elapsed))

                finally:
                    video_reader.release()
                    close_session(sess.session_id)

            return StreamingResponse(
                generate(),
                media_type="multipart/x-mixed-replace; boundary=frame",
                headers={"X-Session-Id": sess.session_id},
            )
    raise HTTPException(404, "Demo video not found.")


@router.get("/demo/image")
async def vid_demo_image(request: Request):
    for ext in [".jpg", ".jpeg", ".png"]:
        p = settings.DEMO_DIR / f"vid{ext}"
        if p.exists():
            sess = new_session("vid", f"demo{ext}")
            model = get_vid_model()
            annotated, dets = _run_inference_on_frame(cv2.imread(str(p)), model)
            record = sess.add_frame(annotated, dets, 0, 0.0)
            close_session(sess.session_id)
            base = str(request.base_url).rstrip("/")
            return safe_json_response({
                "session_id": sess.session_id,
                "task": "vid",
                "source": f"demo{ext}",
                "image_dims": {"w": annotated.shape[1], "h": annotated.shape[0]},
                "count": len(dets),
                "detections": dets,
                "class_summary": dict(Counter(d["class_name"] for d in dets)),
                "annotated_url": f"{base}/outputs/annotated/{sess.session_id}/{(record.get('annotated_image') or '').split('/')[-1]}" if record.get("annotated_image") else None,
                "summary": sess.summary(),
            })
    raise HTTPException(404, "Demo image not found.")


@router.get("/demo/image/annotated", summary="Return annotated demo image directly as JPEG")
async def vid_demo_image_annotated():
    for ext in [".jpg", ".jpeg", ".png"]:
        p = settings.DEMO_DIR / f"vid{ext}"
        if p.exists():
            model = get_vid_model()
            frame = cv2.imread(str(p))
            if frame is None:
                raise HTTPException(500, "Could not read demo image")
            annotated, dets = _run_inference_on_frame(frame, model)
            _, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            return StreamingResponse(
                iter([buf.tobytes()]),
                media_type="image/jpeg",
                headers={"X-Count": str(len(dets))},
            )
    raise HTTPException(404, "Demo image not found.")


@router.get("/sessions")
async def vid_sessions():
    return safe_json_response(list_sessions(task="vid"))


@router.get("/metrics")
async def vid_metrics():
    model = get_vid_model()
    class_labels = _get_class_labels(model)
    if isinstance(class_labels, dict):
        classes = [class_labels[k] for k in sorted(class_labels.keys())]
    elif isinstance(class_labels, (list, tuple)):
        classes = list(class_labels)
    else:
        classes = []
    return safe_json_response({
        "task": "vid",
        "model": "YOLOv11 ONNX",
        "classes": classes,
        "performance": f"YOLO_INTERVAL={YOLO_INTERVAL}, STREAM_JPEG_QUALITY={STREAM_JPEG_QUALITY}",
    })