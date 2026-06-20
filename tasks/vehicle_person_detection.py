"""
tasks/vehicle_person_detection.py
YOLOv11 Vehicle & Person Detection
NOTE: drawing is intentionally delegated to session_manager.draw_detections_scaled
      so font/box sizes adapt to frame resolution. This file only does inference.
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import Generator, List, Dict
from preprocess.preprocess import rescale_boxes, non_max_suppression, VideoStream

DEFAULT_CONF = 0.25
DEFAULT_IOU  = 0.45

# Updated to IDD Classes, with "person" replaced by "pedestrian"
IDD_CLASSES = [
    'animal', 'autorickshaw', 'bicycle', 'bus', 'car', 'caravan',
    'motorcycle', 'pedestrian', 'rider', 'traffic light', 'traffic sign',
    'trailer', 'train', 'truck', 'vehicle fallback'
]

CLASS_COLORS = {
    "pedestrian":    (255,  80,  80),
    "bicycle":       ( 80, 255, 180),
    "car":           ( 80, 180, 255),
    "motorcycle":    (255, 180,  80),
    "bus":           (180,  80, 255),
    "truck":         (255,  80, 255),
    "traffic light": ( 80, 255,  80),
    "traffic sign":  ( 50,  50, 255),
    "autorickshaw":  (255, 255,  80),
    "rider":         (200, 100, 200),
}


def draw_detections(frame: np.ndarray, detections: List[Dict], copy: bool = True) -> np.ndarray:
    """
    Legacy draw function — kept for compatibility with process_video_stream.
    Uses resolution-adaptive scaling via session_manager helper.
    Imported here to avoid circular imports at module load.
    """
    try:
        from backend.session_manager import draw_detections_scaled
        return draw_detections_scaled(frame, detections, copy)
    except ImportError:
        # Fallback if used outside backend context
        img = frame.copy() if copy else frame
        h, w = img.shape[:2]
        diag = (h*h + w*w)**0.5
        sc   = max(0.4, min(1.4, diag / 2203 * 0.7))
        bk   = max(1, int(diag / 2203 * 2))
        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            class_name = d.get('class_name', "")
            color = CLASS_COLORS.get(class_name, (200, 200, 200))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, bk)
            cv2.putText(img, f"{class_name} {d.get('confidence',0)*100:.0f}%",
                        (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, sc, color, bk)
        return img


class VehiclePersonDetector:
    def __init__(self, model_path, conf_thres=DEFAULT_CONF, iou_thres=DEFAULT_IOU, device="cpu"):
        self.conf_thres = conf_thres
        self.iou_thres  = iou_thres

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4

        providers = ["CPUExecutionProvider"]
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def infer_frame(self, blob, meta, frame) -> List[Dict]:
        raw = self.session.run(None, {self.input_name: blob})[0]
        return self._postprocess(raw, meta, frame)

    def process_video_stream(self, video_path, yolo_interval=2) -> Generator:
        with VideoStream(video_path, skip_frames=0, enhance=False) as vs:
            total    = vs.frame_count
            last     = []
            for idx, timestamp, frame, blob, meta in vs:
                if idx % yolo_interval == 0:
                    last = self.infer_frame(blob, meta, frame)
                yield idx, total, draw_detections(frame, last, copy=True)

    def _postprocess(self, raw, meta, frame) -> List[Dict]:
        pred = raw[0] if raw.ndim == 3 else raw
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        boxes_raw   = pred[:, :4]
        class_scores = pred[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confs     = class_scores[np.arange(len(class_ids)), class_ids]

        # Fixed: Check if class_id is a valid index within the IDD_CLASSES list
        traffic_mask = np.array([0 <= cid < len(IDD_CLASSES) for cid in class_ids])
        conf_mask    = confs >= self.conf_thres
        mask         = traffic_mask & conf_mask

        boxes_raw = boxes_raw[mask]
        confs     = confs[mask]
        class_ids = class_ids[mask]

        if len(boxes_raw) == 0:
            return []

        # xywh → xyxy
        boxes        = np.empty_like(boxes_raw)
        boxes[:, 0]  = boxes_raw[:, 0] - boxes_raw[:, 2] / 2
        boxes[:, 1]  = boxes_raw[:, 1] - boxes_raw[:, 3] / 2
        boxes[:, 2]  = boxes_raw[:, 0] + boxes_raw[:, 2] / 2
        boxes[:, 3]  = boxes_raw[:, 1] + boxes_raw[:, 3] / 2

        boxes = rescale_boxes(boxes, meta)
        keep  = non_max_suppression(boxes, confs, self.iou_thres)

        # Fixed: Using list indexing instead of .get() which is for dictionaries
        return [
            {
                "box":        [int(v) for v in boxes[i]],
                "confidence": round(float(confs[i]), 4),
                "class_id":   int(class_ids[i]),
                "class_name": IDD_CLASSES[int(class_ids[i])]
            }
            for i in keep
        ]