"""
backend/config.py
Central configuration for all tasks and paths
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Model paths
    LPR_MODEL: Path = Path("models/lpr.onnx")
    VPD_MODEL: Path = Path("models/vpd.onnx")
    VID_MODEL: Path = Path("models/vid.onnx")

    # Task defaults
    LPR_CONF: float = 0.20
    LPR_IOU:  float = 0.45
    VPD_CONF: float = 0.25
    VPD_IOU:  float = 0.45
    VID_CONF: float = 0.30
    VID_IOU:  float = 0.45

    # Video processing
    YOLO_INTERVAL: int = 2            # run inference every N frames
    MAX_UPLOAD_MB: int = 500

    # Output paths
    OUTPUTS_ROOT: Path = Path("outputs")
    DEMO_DIR:     Path = Path("demo")

    # Camera
    DEFAULT_CAMERA_ID: str = "CAM-01"
    DEFAULT_LOCATION:  str = "Unknown Location"


settings = Settings()
