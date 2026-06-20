"""
tasks/evidence_generation.py
Evidence Generation — Task 4

Accepts output from any of the three upstream tasks and produces:
  • Annotated frames / images saved to disk
  • Structured JSON evidence records
  • Per-session summary report
  • Analytics aggregates

Works with three input schemas
────────────────────────────────
  TASK 1  – vpd (Vehicle & Person)  → {"frame_idx", "timestamp", "detections", "frame"}
  TASK 2  – vid (Violation)         → {"frame_idx", "timestamp", "violations", "frame"}
  TASK 3  – lpr (License Plate)     → {"frame_idx", "timestamp", "plates",     "frame"}

Each task result list is passed to `generate_evidence(task_id, results, …)`.
"""

import cv2
import json
import uuid
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict, Counter

# Ensure these match your actual import paths
from tasks.vehicle_and_road import draw_detections
from tasks.violation import draw_violations, SEV_COLORS
from tasks.license_plate_recognition import draw_plates, clean_plate, plate_score, plate_similarity

# ─────────────────────── paths ───────────────────────────────────

OUTPUTS_ROOT = Path("outputs")
ANNOTATED_DIR = OUTPUTS_ROOT / "annotated"
JSON_DIR      = OUTPUTS_ROOT / "json"
REPORTS_DIR   = OUTPUTS_ROOT / "reports"

for d in (ANNOTATED_DIR, JSON_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────── task routing ────────────────────────────

TASK_KEYS = {
    "vpd":       "detections",
    "vehicle":   "detections",  # Legacy fallback
    "vid":       "violations",
    "violation": "violations",  # Legacy fallback
    "lpr":       "plates",
}

TASK_DRAW_FN = {
    "vpd":       draw_detections,
    "vehicle":   draw_detections,
    "vid":       draw_violations,
    "violation": draw_violations,
    "lpr":       draw_plates,
}

# Strict threshold for global fuzzy clustering
GLOBAL_NMS_THRESHOLD = 60


# ─────────────────────── core function ───────────────────────────

def generate_evidence(
    task_id: str,                     # "vpd" | "vid" | "lpr"
    results: List[Dict[str, Any]],
    session_id: Optional[str] = None,
    source_name: str = "unknown",
    camera_id: str = "CAM-01",
    location: str = "Unknown Location",
    save_frames: bool = True,
    frame_stride: int = 1,            # save every Nth annotated frame
) -> Dict[str, Any]:
    """
    Main entry point – call from any task's output list.

    Returns a dict with:
      evidence_records : List[dict]   – one record per relevant frame
      summary          : dict         – session-level stats
      json_path        : str          – path of saved JSON
      report_path      : str          – path of saved markdown report
    """
    # Standardize legacy task_ids if they sneak in
    if task_id == "vehicle": task_id = "vpd"
    if task_id == "violation": task_id = "vid"

    assert task_id in TASK_KEYS, f"Unknown task_id '{task_id}'. Use: {list(TASK_KEYS)}"
    session_id = session_id or str(uuid.uuid4())[:8].upper()
    key = TASK_KEYS[task_id]
    draw_fn = TASK_DRAW_FN[task_id]
    
    # ── Unified Timestamp for File Naming ──
    session_time_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # ── GLOBAL STRING-BASED FUZZY NMS (LONGEST LENGTH WINS) ──────────
    if task_id == "lpr" and results:
        unique_texts = set()
        
        # Normalize and harvest all plate text variations across frames
        for result in results:
            for p in result.get("plates", []):
                txt = clean_plate(p.get("plate_text", ""))
                if txt:
                    p["plate_text"] = txt
                    unique_texts.add(txt)

        # Sort unique plate strings by length descending to process longest first
        sorted_texts = sorted(list(unique_texts), key=lambda x: (len(x), x), reverse=True)
        
        valid_global_strings = []
        for txt in sorted_texts:
            is_duplicate = False
            for accepted in valid_global_strings:
                # If a longer/better variant already exists above threshold, suppress the shorter one
                if plate_similarity(txt, accepted) >= GLOBAL_NMS_THRESHOLD:
                    is_duplicate = True
                    break
            if not is_duplicate:
                valid_global_strings.append(txt)

        # OPTIMIZATION: Keep ONLY ONE occurrence per unique plate to ensure an ultra-light folder.
        # We also keep ONE frame for failed reads ("UNREADABLE_OCR") to maintain accurate UI stats.
        seen_plates = set()
        for result in results:
            if "plates" in result:
                kept_plates = []
                for p in result["plates"]:
                    text = p.get("plate_text", "")
                    
                    if text:
                        if text in valid_global_strings and text not in seen_plates:
                            kept_plates.append(p)
                            seen_plates.add(text)
                    else:
                        if "UNREADABLE_OCR" not in seen_plates:
                            kept_plates.append(p)
                            seen_plates.add("UNREADABLE_OCR")
                
                # Update the frame's plate list
                result["plates"] = kept_plates

    evidence_records: List[Dict] = []
    annotated_paths: List[str] = []

    for i, result in enumerate(results):
        frame_idx = result.get("frame_idx", i)
        timestamp = result.get("timestamp", 0.0)
        frame     = result.get("frame")
        items     = result.get(key, [])

        # If empty (filtered out or no detections), skip generating evidence entirely
        if not items:
            continue

        # ── Annotate frame ────────────────────────────────────────
        img_path = None
        is_lpr = (task_id == "lpr")
        
        if frame is not None and save_frames and (is_lpr or (i % frame_stride == 0)):
            try:
                annotated = draw_fn(frame, items)
                annotated = _overlay_meta(
                    annotated, session_id, camera_id, location,
                    timestamp, frame_idx, task_id,
                )
                
                # File Naming Logic
                if is_lpr:
                    plate_nos = []
                    for p in items:
                        txt = p.get("plate_text", "").strip()
                        plate_nos.append(txt if txt else "UNREADABLE")
                    
                    if plate_nos:
                        plate_str = "_".join(plate_nos)
                        fname = f"{task_id}_{plate_str}_{session_time_stamp}.jpg"
                    else:
                        fname = f"{task_id}_{session_time_stamp}.jpg"
                else:
                    # Append frame index for non-LPR tasks to prevent image overwrite collisions
                    fname = f"{task_id}_{session_time_stamp}_f{frame_idx:06d}.jpg"
                    
                img_path = str(ANNOTATED_DIR / fname)
                cv2.imwrite(img_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
                annotated_paths.append(img_path)
            except Exception as e:
                print(f"Warning: Failed to draw or save frame {frame_idx}: {e}")

        # ── Build evidence record ─────────────────────────────────
        record = _build_record(
            task_id, session_id, camera_id, location, source_name,
            frame_idx, timestamp, items, img_path,
        )
        evidence_records.append(record)

    # ── Session summary ───────────────────────────────────────────
    summary = _build_summary(task_id, session_id, source_name, camera_id,
                              location, evidence_records, results)

    # ── Persist JSON ──────────────────────────────────────────────
    json_payload = {
        "schema_version": "1.1",
        "task_id":        task_id,
        "session_id":     session_id,
        "generated_at":   datetime.utcnow().isoformat() + "Z",
        "summary":        summary,
        "evidence":       evidence_records,
    }
    json_path = str(JSON_DIR / f"{task_id}_{session_time_stamp}.json")
    with open(json_path, "w") as f:
        json.dump(json_payload, f, indent=2, default=str)

    # ── Markdown report ───────────────────────────────────────────
    report_path = _write_report(task_id, session_id, session_time_stamp, summary, evidence_records)

    return {
        "evidence_records": evidence_records,
        "summary":          summary,
        "json_path":        json_path,
        "report_path":      report_path,
        "annotated_paths":  annotated_paths,
        "session_id":       session_id,
    }


# ─────────────────────── record builders ─────────────────────────

def _build_record(
    task_id, session_id, camera_id, location, source_name,
    frame_idx, timestamp, items, img_path,
) -> Dict:
    base = {
        "record_id":   f"{session_id}-{frame_idx:06d}",
        "task_id":     task_id,
        "session_id":  session_id,
        "camera_id":   camera_id,
        "location":    location,
        "source":      source_name,
        "frame_idx":   frame_idx,
        "timestamp":   timestamp,
        "captured_at": datetime.utcnow().isoformat() + "Z",
        "annotated_image": img_path,
    }

    if task_id == "vpd":
        base["detections"] = items
        base["detection_count"] = len(items)
        cat_counts = Counter(d.get("class_name", d.get("category", "unknown")) for d in items)
        base["class_breakdown"] = dict(cat_counts)

    elif task_id == "vid":
        base["violations"] = items
        base["violation_count"] = len(items)
        base["violation_types"] = list({v.get("class_name", v.get("violation_type", "unknown")) for v in items})
        
        # Determine pseudo-severity if model just outputs class_name
        sev_counts = Counter(v.get("severity", "HIGH") for v in items)
        base["severity_breakdown"] = dict(sev_counts)
        base["highest_severity"] = (
            "CRITICAL" if sev_counts.get("CRITICAL") else
            "HIGH"     if sev_counts.get("HIGH")     else
            "MEDIUM"   if sev_counts.get("MEDIUM")   else
            "LOW"      if sev_counts.get("LOW")      else "NONE"
        )

    elif task_id == "lpr":
        base["plates"] = [
            {k: v for k, v in p.items() if k != "crop"}   # exclude numpy array
            for p in items
        ]
        base["plate_count"]   = len(items)
        base["plates_read"]   = [p["plate_text"] for p in items if p.get("plate_text")]
        base["plates_valid"]  = [p["plate_text"] for p in items if p.get("valid_format")]

    return base


def _build_summary(
    task_id, session_id, source_name, camera_id,
    location, evidence_records, all_results,
) -> Dict:
    total_frames = len(all_results)
    frames_with_events = len(evidence_records)

    summary: Dict[str, Any] = {
        "session_id":        session_id,
        "task_id":           task_id,
        "source":            source_name,
        "camera_id":         camera_id,
        "location":          location,
        "total_frames":      total_frames,
        "frames_with_events": frames_with_events,
        "event_rate_pct":    round(100 * frames_with_events / max(total_frames, 1), 2),
    }

    if task_id == "vpd":
        all_dets = [d for r in evidence_records for d in r.get("detections", [])]
        summary["total_detections"]   = len(all_dets)
        summary["avg_per_frame"]      = round(len(all_dets) / max(frames_with_events, 1), 2)
        summary["class_totals"]       = dict(Counter(d.get("class_name", d.get("category", "unknown")) for d in all_dets))

    elif task_id == "vid":
        all_viols = [v for r in evidence_records for v in r.get("violations", [])]
        summary["total_violations"]   = len(all_viols)
        vtype_counts = Counter(v.get("class_name", v.get("violation_type", "unknown")) for v in all_viols)
        sev_counts   = Counter(v.get("severity", "HIGH") for v in all_viols)
        
        summary["violation_type_counts"] = dict(vtype_counts)
        summary["severity_counts"]       = dict(sev_counts)
        summary["most_common_violation"] = vtype_counts.most_common(1)[0][0] if vtype_counts else None

    elif task_id == "lpr":
        all_plates = [p for r in evidence_records for p in r.get("plates", [])]
        texts  = [p["plate_text"] for p in all_plates if p.get("plate_text")]
        valid  = [p["plate_text"] for p in all_plates if p.get("valid_format")]
        
        # Properly counts the unique total objects (failures and successes)
        summary["total_plates_detected"] = len(all_plates)
        summary["plates_successfully_read"] = len(texts)
        summary["valid_format_count"]    = len(valid)
        
        summary["unique_plates"]         = list(set(texts))
        summary["ocr_success_rate_pct"]  = round(100 * len(texts) / max(len(all_plates), 1), 2)

    return summary


# ─────────────────────── annotation overlay ──────────────────────

def _overlay_meta(
    frame: np.ndarray,
    session_id: str,
    camera_id: str,
    location: str,
    timestamp: float,
    frame_idx: int,
    task_id: str,
) -> np.ndarray:
    """Burn-in metadata strip at the top of the annotated frame."""
    h, w = frame.shape[:2]
    strip_h = 32
    strip = np.zeros((strip_h, w, 3), dtype=np.uint8)
    strip[:] = (30, 30, 30)
    frame = np.vstack([strip, frame])

    ts_str = f"{int(timestamp//60):02d}:{timestamp%60:06.3f}"
    info   = (f"Session:{session_id}  Cam:{camera_id}  "
              f"Loc:{location}  Frame:{frame_idx}  T:{ts_str}  Task:{task_id.upper()}")
    cv2.putText(frame, info, (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1, cv2.LINE_AA)
    return frame


# ─────────────────────── markdown report ─────────────────────────

def _write_report(
    task_id: str,
    session_id: str,
    session_time_stamp: str,
    summary: Dict,
    evidence_records: List[Dict],
) -> str:
    lines = [
        f"# Evidence Report — {task_id.upper()} / Session {session_id}",
        f"_Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}_\n",
        "## Session Summary",
        f"| Field | Value |",
        f"|---|---|",
    ]
    for k, v in summary.items():
        if isinstance(v, list) and len(v) > 5:
            lines.append(f"| {k} | {len(v)} items (omitted) |")
        else:
            lines.append(f"| {k} | {v} |")

    lines += ["\n## Evidence Records (first 20 unique extractions)\n"]
    for rec in evidence_records[:20]:
        lines.append(f"### Record `{rec['record_id']}`")
        lines.append(f"- **Frame**: {rec['frame_idx']}  **Timestamp**: {rec['timestamp']}s")
        
        if task_id == "vid":
            for v in rec.get("violations", []):
                vtype = v.get("class_name", v.get("violation_type", "unknown"))
                sev = v.get("severity", "HIGH")
                conf = v.get("confidence", 0.0)
                lines.append(
                    f"  - [{sev}] `{vtype}` conf={conf:.2f}"
                )
                
        elif task_id == "vpd":
            lines.append(f"  - Detections: {rec.get('detection_count', 0)}")
            for d in rec.get("detections", []):
                cls_name = d.get("class_name", d.get("category", "unknown"))
                conf = d.get("confidence", 0.0)
                lines.append(
                    f"    - `{cls_name}` conf={conf:.2f}"
                )

        elif task_id == "lpr":
            for p in rec.get("plates", []):
                lines.append(
                    f"  - Plate: `{p.get('plate_text','??')}` "
                    f"valid={p.get('valid_format')} conf={p.get('det_confidence','?')}"
                )
                
        lines.append("")

    report = "\n".join(lines)
    path = str(REPORTS_DIR / f"{task_id}_{session_time_stamp}.md")
    with open(path, "w") as f:
        f.write(report)
    return path


# ─────────────────────── analytics helper ────────────────────────

def compute_analytics(json_path: str) -> Dict[str, Any]:
    """
    Load a previously generated JSON evidence file and return
    enriched analytics (time-series counts, heatmap data, etc.).
    """
    with open(json_path) as f:
        data = json.load(f)

    task_id = data["task_id"]
    records = data["evidence"]

    analytics: Dict[str, Any] = {
        "session_id": data["session_id"],
        "task_id":    task_id,
        "timeline":   [],
    }

    if task_id == "vid":
        by_type: Dict[str, List] = defaultdict(list)
        for rec in records:
            t = rec["timestamp"]
            for v in rec.get("violations", []):
                vtype = v.get("class_name", v.get("violation_type", "unknown"))
                by_type[vtype].append(t)
        analytics["violation_timeline_by_type"] = {k: v for k, v in by_type.items()}
        analytics["hotspot_timestamps"] = {
            k: sorted(v)[:5] for k, v in by_type.items()
        }

    elif task_id == "lpr":
        plate_freq: Counter = Counter()
        for rec in records:
            for p in rec.get("plates", []):
                if p.get("plate_text"):
                    plate_freq[p["plate_text"]] += 1
        analytics["plate_frequency"] = dict(plate_freq.most_common(20))
        analytics["repeat_offenders"] = [p for p, c in plate_freq.items() if c > 1]

    elif task_id == "vpd":
        frame_counts = [
            {"timestamp": r["timestamp"], "count": r.get("detection_count", 0)}
            for r in records
        ]
        analytics["frame_vehicle_counts"] = frame_counts
        peak = max(frame_counts, key=lambda x: x["count"], default={})
        analytics["peak_traffic"] = peak

    return analytics