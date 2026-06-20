"""
preprocess/preprocess.py
Traffic Image Preprocessing Utilities
Handles: low light, rain, motion blur, shadows, normalization, resizing
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────── constants ───────────────────────────
YOLO_INPUT_SIZE = (640, 640)          # standard YOLOv8/v10/v11 input
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID  = (8, 8)
MEAN_NORM = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD_NORM  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────── core pipeline ───────────────────────────

def preprocess_frame(
    frame: np.ndarray,
    target_size: Tuple[int, int] = YOLO_INPUT_SIZE,
    enhance: bool = False,   # CHANGED: Default False to match ultralytics notebook
    normalize: bool = False, # CHANGED: Default False! YOLO models only need / 255.0
) -> Tuple[np.ndarray, dict]:
    """
    Full preprocessing pipeline for a single BGR frame.

    Returns
    -------
    blob  : np.ndarray  shape (1, 3, H, W) float32  – ready for ONNX inference
    meta  : dict        original shape, scale factors, padding (for post-proc)
    """
    orig_h, orig_w = frame.shape[:2]

    # 1. Enhancement
    if enhance:
        frame = auto_enhance(frame)

    # 2. Letterbox resize (preserve aspect ratio)
    resized, scale, (pad_w, pad_h) = letterbox(frame, target_size)

    # 3. BGR → RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # 4. HWC → CHW, float32 [0,1]
    blob = rgb.astype(np.float32) / 255.0

    if normalize:
        blob = (blob - MEAN_NORM) / STD_NORM

    # 5. Add batch dim: (H,W,3) → (1,3,H,W)
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

    meta = {
        "orig_shape": (orig_h, orig_w),
        "input_shape": target_size,
        "scale": scale,
        "pad": (pad_w, pad_h),
    }
    return blob, meta


def load_and_preprocess(
    source: Union[str, Path, np.ndarray],
    target_size: Tuple[int, int] = YOLO_INPUT_SIZE,
    enhance: bool = False,   # CHANGED to False
    normalize: bool = False, # CHANGED to False
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load an image file **or** accept a raw BGR array,
    run the pipeline, and return (blob, original_bgr, meta).
    """
    if isinstance(source, (str, Path)):
        frame = cv2.imread(str(source))
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {source}")
    else:
        frame = source.copy()

    original = frame.copy()
    blob, meta = preprocess_frame(frame, target_size, enhance, normalize)
    return blob, original, meta


# ─────────────────────── image enhancement ───────────────────────

def auto_enhance(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    if mean_brightness < 60:
        frame = enhance_low_light(frame)
    elif mean_brightness > 200:
        frame = reduce_overexposure(frame)

    if blur_score < 100:
        frame = deblur_frame(frame)

    frame = remove_rain_streaks(frame)
    frame = reduce_shadows(frame)
    return frame


def enhance_low_light(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    l_eq = clahe.apply(l)
    merged = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def reduce_overexposure(frame: np.ndarray) -> np.ndarray:
    gamma = 0.5
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(frame, lut)


def deblur_frame(frame: np.ndarray, strength: float = 1.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(frame, (0, 0), 3)
    return cv2.addWeighted(frame, 1 + strength, blurred, -strength, 0)


def remove_rain_streaks(frame: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    tophat = cv2.morphologyEx(frame, cv2.MORPH_TOPHAT, kernel)
    cleaned = cv2.subtract(frame, tophat)
    return cv2.bilateralFilter(cleaned, 5, 75, 75)


def reduce_shadows(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    v_eq = clahe.apply(v)
    merged = cv2.merge([h, s, v_eq])
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)


# ─────────────────────── geometry helpers ────────────────────────

def letterbox(
    image: np.ndarray,
    target: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h, w = image.shape[:2]
    th, tw = target
    scale = min(tw / w, th / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = (tw - new_w) // 2
    pad_h = (th - new_h) // 2
    padded = cv2.copyMakeBorder(
        resized, pad_h, th - new_h - pad_h,
        pad_w, tw - new_w - pad_w,
        cv2.BORDER_CONSTANT, value=color,
    )
    return padded, scale, (pad_w, pad_h)


def rescale_boxes(
    boxes: np.ndarray,
    meta: dict,
) -> np.ndarray:
    pad_w, pad_h = meta["pad"]
    scale = meta["scale"]
    boxes = boxes.copy().astype(np.float32)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale
    orig_h, orig_w = meta["orig_shape"]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
    return boxes


def crop_region(image: np.ndarray, box: np.ndarray, padding: int = 5) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box[:4].astype(int)
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return image[y1:y2, x1:x2]


# ─────────────────────── video utilities ─────────────────────────

class VideoStream:
    def __init__(
        self,
        source: Union[str, int],
        target_size: Tuple[int, int] = YOLO_INPUT_SIZE,
        enhance: bool = False, # CHANGED to False
        skip_frames: int = 0,
    ):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        self.target_size = target_size
        self.enhance = enhance
        self.skip_frames = skip_frames
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def video_info(self) -> dict:
        return {
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }

    def __iter__(self):
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if self.skip_frames and frame_idx % (self.skip_frames + 1) != 0:
                frame_idx += 1
                continue
            blob, meta = preprocess_frame(frame, self.target_size, self.enhance)
            timestamp = frame_idx / self.fps
            yield frame_idx, timestamp, frame, blob, meta
            frame_idx += 1

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()


# ─────────────────────── NMS helper ──────────────────────────────

def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
) -> np.ndarray:
    if len(boxes) == 0:
        return np.array([], dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        iou  = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return np.array(keep, dtype=int)