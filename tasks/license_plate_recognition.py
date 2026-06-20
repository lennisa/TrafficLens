"""
tasks/license_plate_recognition.py

YOLOv8 License Plate Detection
FastPlateOCR Recognition
OCR Enhancement
RapidFuzz NMS Plate Dedup
"""

from preprocess.preprocess import (
    rescale_boxes,
    non_max_suppression,
    crop_region,
    VideoStream,
)

import cv2
import re
import numpy as np
import onnxruntime as ort

from rapidfuzz.fuzz import ratio, WRatio
from typing import Generator


DEFAULT_CONF = 0.20
DEFAULT_IOU = 0.45
MIN_PLATE_AREA = 100

_FAST_OCR_ENGINE = None

INDIA_PLATE_RE = re.compile(
    r"^[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,2}\s?\d{1,4}$",
    re.IGNORECASE
)


def clean_plate(text):
    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(text).upper()
    )


def validate_plate(text):
    if not text:
        return False
    return bool(INDIA_PLATE_RE.match(text))


RAPIDFUZZ_NMS_THRESHOLD = 60


def plate_similarity(a, b):
    a = clean_plate(a)
    b = clean_plate(b)

    if not a or not b:
        return 0

    return max(
        ratio(a, b),
        WRatio(a, b)
    )


def plate_score(p):
    return (
        len(clean_plate(p.get("plate_text", ""))) * 1000
        + p.get("ocr_confidence", 0)
        + p.get("det_confidence", 0)
    )


def pure_substring_dedup(plates):
    """
    Pure O(n^2) absolute deduplication gauntlet.
    Compares every single element against every other element.
    Drops proper substrings, lower-scoring exact duplicates, and fuzzy overlaps.
    """
    if not plates:
        return []

    # Ensure everything is pre-cleaned for strict matching
    for p in plates:
        p["plate_text"] = clean_plate(p.get("plate_text", ""))

    survivors = []

    for i, p in enumerate(plates):
        text_p = p["plate_text"]
        if not text_p:
            continue

        score_p = plate_score(p)
        drop = False

        for j, q in enumerate(plates):
            if i == j:
                continue

            text_q = q["plate_text"]
            score_q = plate_score(q)

            # 1. Strict Substring Elimination: if p is inside q and q is longer, drop p
            if text_p in text_q and len(text_q) > len(text_p):
                drop = True
                break

            # 2. Exact Match Check (Keep highest score, use index as ultimate tie-breaker)
            if text_p == text_q:
                if score_p < score_q:
                    drop = True
                    break
                elif score_p == score_q and i > j:
                    drop = True
                    break

            # 3. Fuzzy Match Check
            if plate_similarity(text_p, text_q) >= RAPIDFUZZ_NMS_THRESHOLD:
                if score_p < score_q:
                    drop = True
                    break
                elif score_p == score_q and i > j:
                    drop = True
                    break

        if not drop:
            survivors.append(p)

    return survivors


# -----------------------------
# OCR enhancement
# -----------------------------

def enhance_ocr_crop(crop):
    if crop is None or crop.size == 0:
        return crop

    h, w = crop.shape[:2]

    crop = cv2.resize(
        crop,
        (w * 3, h * 3),
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(
        gray,
        (0, 0),
        1
    )

    sharp = cv2.addWeighted(
        gray,
        1.25,
        blur,
        -0.25,
        0
    )

    return cv2.cvtColor(
        sharp,
        cv2.COLOR_GRAY2BGR
    )


def run_ocr(crop):
    global _FAST_OCR_ENGINE

    try:
        if _FAST_OCR_ENGINE is None:
            from fast_plate_ocr import LicensePlateRecognizer
            _FAST_OCR_ENGINE = LicensePlateRecognizer("cct-s-v2-global-model")

        result = _FAST_OCR_ENGINE.run(crop)

        if result:
            return (
                clean_plate(result[0].plate),
                0.99
            )
    except Exception:
        pass

    return "", 0.0


class LicensePlateRecognizer:

    def __init__(
        self,
        model_path,
        conf_thres=DEFAULT_CONF,
        iou_thres=DEFAULT_IOU,
        device="cpu"
    ):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.unique_plates = {}

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4

        providers = ["CPUExecutionProvider"]

        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name


    def update_unique(self, new_plates):
        """
        GLOBAL-LEVEL DEDUP: Merges the current global history dictionary 
        with the newly parsed frame items, running a complete O(n^2) filter pass.
        """
        combined_pool = list(self.unique_plates.values()) + new_plates
        
        deduped_list = pure_substring_dedup(combined_pool)
        
        # Rebuild clean unique map
        self.unique_plates = {
            p["plate_text"]: p 
            for p in deduped_list
        }
        
        return list(self.unique_plates.values())


    def process_video_stream(
        self,
        video_path,
        yolo_interval=2
    ) -> Generator:

        last = []

        with VideoStream(
            video_path,
            skip_frames=0,
            enhance=False
        ) as vs:
            total = vs.frame_count

            for (idx, timestamp, frame, blob, meta) in vs:
                if idx % yolo_interval == 0:
                    raw = self.session.run(
                        None,
                        {self.input_name: blob}
                    )[0]

                    # 1. Postprocess frame detections
                    last = self._postprocess(raw, meta, frame)
                    
                    # 2. Push detections to global storage history for absolute O(n^2) clean up
                    self.update_unique(last)
                    
                    # 3. Intercept active frame output against global truth map.
                    # Prevent shorter partial strings from displaying if a superior complete string exists.
                    global_truth = list(self.unique_plates.values())
                    refined_display = []
                    
                    for p in last:
                        p_text = clean_plate(p.get("plate_text", ""))
                        better_match = None
                        
                        for g in global_truth:
                            g_text = g["plate_text"]
                            if p_text in g_text and len(g_text) > len(p_text):
                                better_match = g_text
                                break
                        
                        if better_match:
                            p["plate_text"] = better_match
                            p["valid_format"] = validate_plate(better_match)
                            
                        refined_display.append(p)
                        
                    last = pure_substring_dedup(refined_display)

                yield (
                    idx,
                    total,
                    draw_plates(
                        frame,
                        last,
                        False
                    )
                )


    def _postprocess(
        self,
        raw,
        meta,
        frame
    ):
        pred = raw[0] if raw.ndim == 3 else raw

        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        if pred.shape[1] == 6:
            boxes = pred[:, :4]
            conf = pred[:, 4]
        else:
            boxes = self._xywh2xyxy(pred[:, :4])
            conf = np.max(pred[:, 4:], axis=1)

        mask = conf >= self.conf_thres

        boxes = boxes[mask]
        conf = conf[mask]

        if len(boxes) == 0:
            return []

        boxes = rescale_boxes(boxes, meta)
        keep = non_max_suppression(boxes, conf, self.iou_thres)

        results = []

        for box, c in zip(boxes[keep], conf[keep]):
            x1, y1, x2, y2 = box.astype(int)

            if ((x2 - x1) * (y2 - y1) < MIN_PLATE_AREA):
                continue

            crop = crop_region(frame, box, padding=6)
            crop = enhance_ocr_crop(crop)
            text, ocr_conf = run_ocr(crop)

            results.append({
                "box": [x1, y1, x2, y2],
                "det_confidence": float(c),
                "plate_text": text,
                "ocr_confidence": ocr_conf,
                "valid_format": validate_plate(text)
            })

        return results


    @staticmethod
    def _xywh2xyxy(boxes):
        out = np.empty_like(boxes)

        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        return out


def draw_plates(frame, plates, copy=True):
    """
    Resolution-adaptive plate drawing.
    Delegates to session_manager when available (backend context);
    falls back to inline scaled drawing otherwise.
    """
    try:
        from backend.session_manager import draw_plates_scaled
        return draw_plates_scaled(frame, plates, copy)
    except ImportError:
        pass

    img = frame.copy() if copy else frame
    h, w = img.shape[:2]
    diag  = (h * h + w * w) ** 0.5
    scale = max(0.4, min(1.4, diag / 2203 * 0.7))
    thick = max(1, int(diag / 2203 * 2))
    bthick = max(1, int(diag / 2203 * 2.5))
    pad   = max(4, int(diag / 2203 * 8))

    for p in plates:
        x1, y1, x2, y2 = [int(v) for v in p["box"]]
        color = (0, 255, 0) if p.get("valid_format") else (0, 220, 255)
        text  = f"{p.get('plate_text', '??')}  {p.get('det_confidence', 0)*100:.0f}%"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, bthick)
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        ty = max(y1 - pad, th + pad)
        cv2.rectangle(img, (x1, ty - th - pad), (x1 + tw + pad, ty + bl), color, -1)
        cv2.putText(img, text, (x1 + pad // 2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
    return img