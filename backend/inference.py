"""
backend/inference.py
Lazy-loaded singleton model instances
"""

from functools import lru_cache
from backend.config import settings


@lru_cache(maxsize=1)
def get_lpr_model():
    from tasks.license_plate_recognition import LicensePlateRecognizer
    if not settings.LPR_MODEL.exists():
        raise FileNotFoundError(f"LPR model not found: {settings.LPR_MODEL}")
    return LicensePlateRecognizer(
        model_path=settings.LPR_MODEL,
        conf_thres=settings.LPR_CONF,
        iou_thres=settings.LPR_IOU,
    )


@lru_cache(maxsize=1)
def get_vpd_model():
    from tasks.vehicle_person_detection import VehiclePersonDetector
    if not settings.VPD_MODEL.exists():
        raise FileNotFoundError(f"VPD model not found: {settings.VPD_MODEL}")
    return VehiclePersonDetector(
        model_path=settings.VPD_MODEL,
        conf_thres=settings.VPD_CONF,
        iou_thres=settings.VPD_IOU,
    )


@lru_cache(maxsize=1)
def get_vid_model():
    from tasks.violation_detection import ViolationDetector
    if not settings.VID_MODEL.exists():
        raise FileNotFoundError(f"VID model not found: {settings.VID_MODEL}")
    return ViolationDetector(
        model_path=settings.VID_MODEL,
        conf_thres=settings.VID_CONF,
        iou_thres=settings.VID_IOU,
    )
