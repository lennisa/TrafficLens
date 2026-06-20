"""
backend/session_manager.py
Per-session state, record keeping, analytics, and disk persistence.
Clears all outputs on startup for a fresh slate.
"""

import cv2
import json
import uuid
import shutil
import threading
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from collections import Counter

# ── Output directories ──────────────────────────────────────────────
OUTPUTS_ROOT   = Path("outputs")
ANNOTATED_DIR  = OUTPUTS_ROOT / "annotated"
JSON_DIR       = OUTPUTS_ROOT / "json"
REPORTS_DIR    = OUTPUTS_ROOT / "reports"

def _wipe_and_recreate():
    """Delete all previous session output on every server start."""
    if OUTPUTS_ROOT.exists():
        shutil.rmtree(OUTPUTS_ROOT)
    for d in (ANNOTATED_DIR, JSON_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

_wipe_and_recreate()   # <-- runs at import time

# ── Thread-safe session store ──────────────────────────────────────
_lock    = threading.Lock()
_sessions: Dict[str, "Session"] = {}


# ── Dynamic drawing helpers ────────────────────────────────────────

def _draw_scale(frame: np.ndarray) -> tuple:
    """Return (font_scale, thickness, pad) tuned to the frame resolution."""
    h, w = frame.shape[:2]
    diag = (h * h + w * w) ** 0.5
    # base: 1920x1080 diagonal ~2203  → scale 0.7, thickness 2
    scale     = max(0.4, min(1.4, diag / 2203 * 0.7))
    thickness = max(1, int(diag / 2203 * 2))
    box_thick = max(1, int(diag / 2203 * 2.5))
    pad       = max(4, int(diag / 2203 * 8))
    return scale, thickness, box_thick, pad


def draw_plates_scaled(frame: np.ndarray, plates: list, copy: bool = True) -> np.ndarray:
    img = frame.copy() if copy else frame
    scale, thick, box_thick, pad = _draw_scale(img)

    for p in plates:
        x1, y1, x2, y2 = [int(v) for v in p["box"]]
        color = (0, 255, 0) if p.get("valid_format") else (0, 220, 255)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thick)

        label = p.get("plate_text", "??")
        conf  = p.get("det_confidence", 0.0)
        text  = f"{label}  {conf*100:.0f}%"

        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        ty = max(y1 - pad, th + pad)
        # filled label background
        cv2.rectangle(img, (x1, ty - th - pad), (x1 + tw + pad, ty + baseline), color, -1)
        cv2.putText(img, text, (x1 + pad // 2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
    return img


def draw_detections_scaled(frame: np.ndarray, detections: list, copy: bool = True) -> np.ndarray:
    CLASS_COLORS = {
        "person":        (255,  80,  80),
        "bicycle":       ( 80, 255, 180),
        "car":           ( 80, 180, 255),
        "motorcycle":    (255, 180,  80),
        "bus":           (180,  80, 255),
        "truck":         (255,  80, 255),
        "traffic light": ( 80, 255,  80),
        "stop sign":     ( 50,  50, 255),
    }
    img = frame.copy() if copy else frame
    scale, thick, box_thick, pad = _draw_scale(img)

    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        cls   = d.get("class_name", "obj")
        conf  = d.get("confidence", 0.0)
        color = CLASS_COLORS.get(cls, (200, 200, 200))
        text  = f"{cls} {conf*100:.0f}%"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thick)

        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        ty = max(y1 - pad, th + pad)
        cv2.rectangle(img, (x1, ty - th - pad), (x1 + tw + pad, ty + baseline), color, -1)
        cv2.putText(img, text, (x1 + pad // 2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
    return img


def add_hud(frame: np.ndarray, session_id: str, task: str, frame_idx: int,
            timestamp: float, count: int) -> np.ndarray:
    """Burn a minimal HUD strip onto the frame."""
    h, w = frame.shape[:2]
    strip_h = max(28, int(h * 0.03))
    strip = np.zeros((strip_h, w, 3), dtype=np.uint8)
    strip[:] = (15, 15, 15)
    out = np.vstack([strip, frame])

    ts  = f"{int(timestamp//60):02d}:{timestamp%60:05.2f}"
    txt = f"  Session:{session_id}  Task:{task.upper()}  Frame:{frame_idx}  T:{ts}  Detections:{count}"
    fsc = max(0.35, strip_h / 40 * 0.5)
    cv2.putText(out, txt, (4, strip_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, fsc, (0, 230, 120), 1, cv2.LINE_AA)
    return out


# ── Session class ──────────────────────────────────────────────────

class Session:
    def __init__(self, session_id: str, task: str, source_name: str):
        self.session_id  = session_id
        self.task        = task
        self.source_name = source_name
        self.started_at  = datetime.now(timezone.utc).isoformat()
        self.ended_at: Optional[str] = None

        self.total_frames   = 0
        self.frames_with_det = 0
        self.records: List[Dict] = []
        self.unique_plates: Dict[str, Any] = {}   # LPR only
        self.class_counter  = Counter()           # VPD and VID

        # per-session sub-dir for annotated frames
        self.frame_dir = ANNOTATED_DIR / session_id
        self.frame_dir.mkdir(parents=True, exist_ok=True)

    # ── record a processed frame ───────────────────────────────────
    def add_frame(
        self,
        frame:       np.ndarray,
        detections:  list,
        frame_idx:   int,
        timestamp:   float,
        save_frame:  bool = True,
    ) -> Dict:
        self.total_frames += 1
        if not detections:
            return {}

        self.frames_with_det += 1

        # save annotated image
        img_path = None
        if save_frame and frame is not None:
            if self.task == "lpr":
                ann = draw_plates_scaled(frame, detections)
            else:
                ann = draw_detections_scaled(frame, detections)
            ann = add_hud(ann, self.session_id, self.task,
                          frame_idx, timestamp, len(detections))
            fname = f"f{frame_idx:06d}_{int(timestamp*100):08d}.jpg"
            fpath = self.frame_dir / fname
            cv2.imwrite(str(fpath), ann, [cv2.IMWRITE_JPEG_QUALITY, 88])
            img_path = str(fpath)

        # build record
        record: Dict[str, Any] = {
            "record_id":   uuid.uuid4().hex[:8].upper(),
            "frame_idx":   frame_idx,
            "timestamp":   round(timestamp, 3),
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "annotated_image": img_path,
            "count": len(detections),
        }

        if self.task == "lpr":
            for p in detections:
                txt = p.get("plate_text", "")
                if txt:
                    existing = self.unique_plates.get(txt)
                    if not existing or p.get("ocr_confidence", 0) > existing.get("ocr_confidence", 0):
                        self.unique_plates[txt] = p
            record["plates"]       = detections
            record["plates_read"]  = [p["plate_text"] for p in detections if p.get("plate_text")]
            record["valid_plates"] = [p["plate_text"] for p in detections if p.get("valid_format")]

        elif self.task in ("vpd", "vid"):
            for d in detections:
                self.class_counter[d.get("class_name", "unknown")] += 1
            record["detections"]      = detections
            record["class_breakdown"] = dict(Counter(d.get("class_name", "unknown") for d in detections))

        self.records.append(record)
        return record

    # ── summary ────────────────────────────────────────────────────
    def summary(self) -> Dict:
        s: Dict[str, Any] = {
            "session_id":         self.session_id,
            "task":               self.task,
            "source":             self.source_name,
            "started_at":         self.started_at,
            "ended_at":           self.ended_at,
            "total_frames":       self.total_frames,
            "frames_with_events": self.frames_with_det,
            "event_rate_pct":     round(100 * self.frames_with_det / max(self.total_frames, 1), 2),
            "total_records":      len(self.records),
        }
        if self.task == "lpr":
            s["unique_plates"]           = list(self.unique_plates.keys())
            s["unique_plate_count"]      = len(self.unique_plates)
            s["total_detections"]        = sum(r.get("count", 0) for r in self.records)
            valid = [t for t, p in self.unique_plates.items() if p.get("valid_format")]
            s["valid_plate_count"]       = len(valid)
            s["valid_plates"]            = valid
        elif self.task in ("vpd", "vid"):
            s["class_totals"]    = dict(self.class_counter)
            s["total_detections"]= sum(self.class_counter.values())
            s["dominant_class"]  = self.class_counter.most_common(1)[0][0] if self.class_counter else None
            
        return s

    # ── annotated frames list ──────────────────────────────────────
    def frame_urls(self, base_url: str = "") -> List[str]:
        frames = sorted(self.frame_dir.glob("*.jpg"))
        return [f"{base_url}/outputs/annotated/{self.session_id}/{f.name}" for f in frames]

    # ── persist to JSON ────────────────────────────────────────────
    def save_json(self) -> str:
        self.ended_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "schema_version": "2.1",
            "summary":  self.summary(),
            "records":  self.records[-500:],  # cap at 500 records
        }
        path = JSON_DIR / f"{self.task}_{self.session_id}.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        return str(path)

    # ── generate Markdown Report ───────────────────────────────────
    def save_report(self) -> str:
        """Generates a detailed Markdown report mirroring the JSON summary."""
        lines = [
            f"# Session Evidence Report — {self.task.upper()}",
            f"_Session ID: `{self.session_id}`_",
            f"_Generated At: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}_\n",
            "## Session Summary",
            "| Metric | Value |",
            "|---|---|"
        ]
        
        sum_data = self.summary()
        for k, v in sum_data.items():
            if isinstance(v, list) and len(v) > 5:
                lines.append(f"| {k.replace('_', ' ').title()} | {len(v)} items (omitted for brevity) |")
            else:
                lines.append(f"| {k.replace('_', ' ').title()} | {v} |")

        lines += ["\n## Event Ledger (Latest 50)\n"]
        for rec in reversed(self.records[-50:]):
            lines.append(f"### Record `{rec['record_id']}`")
            lines.append(f"- **Frame**: {rec['frame_idx']} | **Timestamp**: {rec['timestamp']}s")
            
            if self.task == "lpr":
                for p in rec.get("plates", []):
                    lines.append(
                        f"  - **Plate**: `{p.get('plate_text', '??')}` "
                        f"(Valid: {p.get('valid_format')}, Conf: {p.get('det_confidence', 0.0):.2f})"
                    )
            elif self.task in ("vpd", "vid"):
                lines.append(f"  - **Total Detections**: {rec.get('count', 0)}")
                for d in rec.get("detections", []):
                    cls_name = d.get("class_name", "unknown")
                    conf = d.get("confidence", 0.0)
                    lines.append(f"    - `{cls_name}` (Conf: {conf:.2f})")
            lines.append("")

        report = "\n".join(lines)
        path = REPORTS_DIR / f"{self.task}_{self.session_id}.md"
        with open(path, "w") as f:
            f.write(report)
        return str(path)


# ── Public API ─────────────────────────────────────────────────────

def new_session(task: str, source_name: str) -> Session:
    sid = uuid.uuid4().hex[:8].upper()
    sess = Session(sid, task, source_name)
    with _lock:
        _sessions[sid] = sess
    return sess


def get_session(session_id: str) -> Optional[Session]:
    return _sessions.get(session_id)


def list_sessions(task: Optional[str] = None) -> List[Dict]:
    with _lock:
        sessions = list(_sessions.values())
    if task:
        sessions = [s for s in sessions if s.task == task]
    return [s.summary() for s in reversed(sessions)]


def close_session(session_id: str) -> Optional[Dict]:
    sess = get_session(session_id)
    if not sess:
        return None
    sess.save_json()
    sess.save_report()
    return sess.summary()