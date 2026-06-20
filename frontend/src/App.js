import React, { useState, useRef, useCallback, useEffect } from "react";
import { TASKS, INPUT_MODES, API_BASE } from "./config/api";
import "./App.css";

const PATHS = {
  Shield: <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />,
  Zap: <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />,
  Activity: <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />,
  Clock: (
    <>
      <circle cx="12" cy="12" r="10" />
      <polyline points="12 6 12 12 16 14" />
    </>
  ),
  Camera: (
    <>
      <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
      <circle cx="12" cy="13" r="4" />
    </>
  ),
  Radio: (
    <>
      <circle cx="12" cy="12" r="2" />
      <path d="M4.93 19.07a10 10 0 0 1 0-14.14m14.14 0a10 10 0 0 1 0 14.14M7.76 16.24a6 6 0 0 1 0-8.48m8.48 0a6 6 0 0 1 0 8.48" />
    </>
  ),
  ImagePlus: (
    <>
      <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7" />
      <line x1="16" x2="22" y1="5" y2="5" />
      <line x1="19" x2="19" y1="2" y2="8" />
      <circle cx="9" cy="9" r="2" />
      <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
    </>
  ),
  FileVideo: (
    <>
      <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" />
      <path d="M14 2v4a2 2 0 0 0 2 2h4" />
      <path d="m10 11 5 3-5 3v-6z" />
    </>
  ),
  Sparkles: (
    <>
      <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z" />
    </>
  ),
  PlayCircle: (
    <>
      <circle cx="12" cy="12" r="10" />
      <polygon points="10 8 16 12 10 16 10 8" />
    </>
  ),
  Upload: (
    <>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" x2="12" y1="3" y2="15" />
    </>
  ),
  Check: <polyline points="20 6 9 17 4 12" />,
  X: (
    <>
      <line x1="18" x2="6" y1="6" y2="18" />
      <line x1="6" x2="18" y1="6" y2="18" />
    </>
  ),
  Loader: <path d="M21 12a9 9 0 1 1-6.219-8.56" />,
  BarChart: (
    <>
      <line x1="12" x2="12" y1="20" y2="10" />
      <line x1="18" x2="18" y1="20" y2="4" />
      <line x1="6" x2="6" y1="20" y2="16" />
    </>
  ),
  List: (
    <>
      <line x1="8" x2="21" y1="6" y2="6" />
      <line x1="8" x2="21" y1="12" y2="12" />
      <line x1="8" x2="21" y1="18" y2="18" />
      <line x1="3" x2="3.01" y1="6" y2="6" />
      <line x1="3" x2="3.01" y1="12" y2="12" />
      <line x1="3" x2="3.01" y1="18" y2="18" />
    </>
  ),
  AlertCircle: (
    <>
      <circle cx="12" cy="12" r="10" />
      <line x1="12" x2="12" y1="8" y2="12" />
      <line x1="12" x2="12.01" y1="16" y2="16" />
    </>
  ),
  Image: (
    <>
      <rect width="18" height="18" x="3" y="3" rx="2" ry="2" />
      <circle cx="9" cy="9" r="2" />
      <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
    </>
  ),
  ChevronRight: <polyline points="9 18 15 12 9 6" />,
  ExternalLink: (
    <>
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
      <polyline points="15 3 21 3 21 9" />
      <line x1="10" x2="21" y1="14" y2="3" />
    </>
  ),
};

const Ic = ({ n, s = 16, cls = "" }) => (
  <svg
    width={s}
    height={s}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.8"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={cls}
    style={{ flexShrink: 0 }}
  >
    {PATHS[n]}
  </svg>
);

const fmtBytes = (b) =>
  b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${(b / 1e3).toFixed(0)} KB`;
const fmtTime = (iso) => (iso ? new Date(iso).toLocaleTimeString() : "—");

const StatPill = ({ val, label, color }) => (
  <div className="stat-pill">
    <span className="stat-pill__val" style={color ? { color } : {}}>
      {val ?? "—"}
    </span>
    <span className="stat-pill__lbl">{label}</span>
  </div>
);

const TaskCard = ({ task, active, onClick }) => (
  <button
    className={`task-card ${active ? "task-card--active" : ""} ${task.comingSoon ? "task-card--soon" : ""}`}
    style={{ "--task-color": task.color }}
    onClick={() => !task.comingSoon && onClick(task.id)}
    disabled={task.comingSoon}
  >
    <div className="task-card__badge">{task.short}</div>
    <div className="task-card__label">{task.label}</div>
    <div className="task-card__desc">{task.description}</div>
    {task.comingSoon && <div className="task-card__soon-pill">Coming Soon</div>}
    {active && <div className="task-card__bar" />}
  </button>
);

const ModeIcon = ({ modeId }) => {
  switch (modeId) {
    // Upload Image: red light, car stopping
    case "upload_image":
      return (
        <div className="mode-icon mode-icon--ti-stop">
          <div className="ti-pole" />
          <div className="ti-box">
            <span className="ti-lamp ti-lamp--red" />
            <span className="ti-lamp" />
            <span className="ti-lamp" />
          </div>
          <div className="ti-road" />
          <div className="ti-car ti-car--stop">
            <span />
            <span />
          </div>
        </div>
      );

    case "upload_video":
      return (
        <div className="mode-icon mode-icon--ti-go">
          <div className="ti-pole" />
          <div className="ti-box">
            <span className="ti-lamp" />
            <span className="ti-lamp" />
            <span className="ti-lamp ti-lamp--green" />
          </div>
          <div className="ti-road" />
          <div className="ti-car ti-car--go">
            <span />
            <span />
          </div>
        </div>
      );

    case "demo_image":
      return (
        <div className="mode-icon mode-icon--ti-yellow">
          <div className="ti-pole" />
          <div className="ti-box">
            <span className="ti-lamp" />
            <span className="ti-lamp ti-lamp--yellow-lit" />
            <span className="ti-lamp" />
          </div>
          <div className="ti-road" />
          <div className="ti-car ti-car--yellow-wait">
            <span />
            <span />
          </div>
        </div>
      );

    case "demo_video":
      return (
        <div className="mode-icon mode-icon--wrongway">
          <div className="ww-road" />
          <div className="ww-divider" />
          <div className="ww-car ww-car--correct">
            <span />
            <span />
          </div>
          <div className="ww-car ww-car--wrong">
            <span />
            <span />
          </div>
          <div className="ww-camera">
            <span className="ww-cam-body" />
            <span className="ww-cam-lens" />
            <span className="ww-flash" />
          </div>
        </div>
      );

    case "camera_image":
      return (
        <div className="mode-icon mode-icon--stopsign">
          <div className="ss-road" />
          <div className="ss-pole" />
          <div className="ss-sign">STOP</div>
          <div className="ss-car">
            <span />
            <span />
          </div>
        </div>
      );

    case "camera_video":
      return (
        <div className="mode-icon mode-icon--zebra">
          <span className="zb-stripe" />
          <span className="zb-stripe" />
          <span className="zb-stripe" />
          <span className="zb-stripe" />
          <span className="zb-stripe" />
          <div className="zb-person zb-person--1" />
          <div className="zb-person zb-person--2" />
        </div>
      );

    default:
      return null;
  }
};

const ModeSelector = ({ active, onChange, disabled }) => (
  <div className="mode-grid">
    {INPUT_MODES.map((m) => (
      <button
        key={m.id}
        disabled={disabled}
        className={`mode-btn ${active === m.id ? "mode-btn--active" : ""}`}
        onClick={() => onChange(m.id)}
      >
        <ModeIcon modeId={m.id} />
        <span className="mode-btn__label">{m.label}</span>
        <span className="mode-btn__desc">{m.desc}</span>
      </button>
    ))}
  </div>
);

const DropZone = ({ accept, onFile, file, label }) => {
  const [drag, setDrag] = useState(false);
  const ref = useRef(null);
  const hasFile = !!file;
  return (
    <div
      className={`dropzone ${drag ? "dropzone--active" : ""} ${hasFile ? "dropzone--hasfile" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        const f = e.dataTransfer.files[0];
        if (f) onFile(f);
      }}
      onClick={() => ref.current.click()}
    >
      <input
        ref={ref}
        type="file"
        accept={accept}
        hidden
        onChange={(e) => onFile(e.target.files[0])}
      />
      {hasFile ? (
        <>
          <Ic
            n={accept.includes("video") ? "FileVideo" : "ImagePlus"}
            s={28}
            cls="dropzone__icon"
          />
          <span className="dropzone__label">{file.name}</span>
          <span className="dropzone__hint">
            {fmtBytes(file.size)} · Click to change
          </span>
        </>
      ) : (
        <>
          <Ic n="Upload" s={28} cls="dropzone__icon" />
          <span className="dropzone__label">{label}</span>
          <span className="dropzone__hint">or click to browse</span>
        </>
      )}
    </div>
  );
};

const CameraPanel = ({ onCapture }) => {
  const vidRef = useRef(null);
  const streamRef = useRef(null);
  const [live, setLive] = useState(false);
  const [err, setErr] = useState("");

  const start = async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
      });
      streamRef.current = s;
      vidRef.current.srcObject = s;
      setLive(true);
      setErr("");
    } catch {
      setErr("Camera unavailable or permission denied.");
    }
  };
  const stop = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    setLive(false);
  };
  useEffect(() => () => stop(), []);

  const capture = () => {
    const c = document.createElement("canvas");
    c.width = vidRef.current.videoWidth;
    c.height = vidRef.current.videoHeight;
    c.getContext("2d").drawImage(vidRef.current, 0, 0);
    c.toBlob(
      (b) => onCapture(new File([b], "capture.jpg", { type: "image/jpeg" })),
      "image/jpeg",
      0.92,
    );
  };

  return (
    <div className="camera-panel">
      <video
        ref={vidRef}
        autoPlay
        playsInline
        className="camera-preview"
        style={{ display: live ? "block" : "none" }}
      />
      {!live && (
        <div className="dropzone" style={{ minHeight: 180 }} onClick={start}>
          <Ic n="Camera" s={28} cls="dropzone__icon" />
          <span className="dropzone__label">Click to start camera</span>
        </div>
      )}
      {err && (
        <div className="error-banner">
          <Ic n="AlertCircle" s={14} />
          {err}
        </div>
      )}
      {live && (
        <div className="btn-row">
          <button className="btn btn--primary" onClick={capture}>
            <Ic n="Camera" s={14} /> Capture & Run
          </button>
          <button className="btn btn--ghost" onClick={stop}>
            <Ic n="X" s={14} /> Stop
          </button>
        </div>
      )}
    </div>
  );
};

const ResultDashboard = ({ result, task }) => {
  const [tab, setTab] = useState("visual");

  if (!result) return null;
  if (result.error)
    return (
      <div className="error-banner" style={{ margin: "12px 0" }}>
        <Ic n="AlertCircle" s={15} />
        <div>
          <strong>Inference failed</strong>
          <br />
          {result.error}
        </div>
      </div>
    );

  const count = result.count ?? 0;
  const validPlates = result.valid_plates?.length ?? 0;
  const plates = result.plates ?? [];
  const detections = result.detections ?? [];
  const classSummary = result.class_summary ?? {};
  const sessionId = result.session_id;
  const frames = result.annotated_frames ?? [];

  const uniquePlateTexts = Array.from(
    new Set(plates.map((p) => p.plate_text).filter(Boolean)),
  );
  const dedupedTexts = uniquePlateTexts.filter((text) => {
    return !uniquePlateTexts.some(
      (other) => other !== text && other.includes(text),
    );
  });
  const visualChips = dedupedTexts.map((text) =>
    plates.find((p) => p.plate_text === text),
  );

  return (
    <div className="result-dashboard">
      <div className="tab-nav">
        <button
          className={`tab-btn ${tab === "visual" ? "tab-btn--active" : ""}`}
          onClick={() => setTab("visual")}
        >
          Visual
        </button>
        <button
          className={`tab-btn ${tab === "records" ? "tab-btn--active" : ""}`}
          onClick={() => setTab("records")}
        >
          Records
        </button>
        {frames.length > 0 && (
          <button
            className={`tab-btn ${tab === "frames" ? "tab-btn--active" : ""}`}
            onClick={() => setTab("frames")}
          >
            Frames{" "}
            <span style={{ opacity: 0.6, fontSize: 10, marginLeft: 4 }}>
              {frames.length}
            </span>
          </button>
        )}
        {result.summary && (
          <button
            className={`tab-btn ${tab === "session" ? "tab-btn--active" : ""}`}
            onClick={() => setTab("session")}
          >
            Session
          </button>
        )}
      </div>

      <div className="stats-row">
        <StatPill val={count} label="Detected" color={task.color} />
        {task.id === "lpr" && (
          <StatPill
            val={validPlates}
            label="Valid plates"
            color="var(--green)"
          />
        )}
        {task.id === "lpr" && (
          <StatPill
            val={visualChips.length - validPlates || 0}
            label="Partial"
            color="var(--orange)"
          />
        )}
        {(task.id === "vpd" || task.id === "vid") && (
          <StatPill val={Object.keys(classSummary).length} label="Classes" />
        )}
        {sessionId && <StatPill val={sessionId} label="Session" />}
      </div>

      {tab === "visual" && (
        <>
          {result.annotatedUrl && !result.streamUrl && (
            <div className="result-img-wrap">
              <img src={result.annotatedUrl} alt="Annotated result" />
              {count > 0 && (
                <div className="det-count-badge">{count} detected</div>
              )}
            </div>
          )}
          {result.streamUrl && (
            <img
              className="stream-img"
              src={result.streamUrl}
              alt="Live stream"
            />
          )}

          {task.id === "lpr" && visualChips.length > 0 && (
            <div>
              <div className="panel-title" style={{ marginBottom: 8 }}>
                Detected Plates
              </div>
              <div className="plates-grid">
                {visualChips.map((p, i) => (
                  <div
                    key={i}
                    className={`plate-chip ${p.valid_format ? "plate-chip--valid" : p.plate_text ? "plate-chip--partial" : "plate-chip--empty"}`}
                  >
                    <span>{p.plate_text || "UNREAD"}</span>
                    {p.det_confidence != null && (
                      <span style={{ fontSize: 10, opacity: 0.7 }}>
                        {(p.det_confidence * 100).toFixed(0)}%
                      </span>
                    )}
                    {p.valid_format && (
                      <span className="plate-valid-tag">✓</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {(task.id === "vpd" || task.id === "vid") &&
            Object.keys(classSummary).length > 0 && (
              <div>
                <div className="panel-title" style={{ marginBottom: 8 }}>
                  Class Breakdown
                </div>
                <div className="class-grid">
                  {Object.entries(classSummary)
                    .sort((a, b) => b[1] - a[1])
                    .map(([cls, cnt]) => (
                      <div key={cls} className="class-chip">
                        <span className="class-chip__name">{cls}</span>
                        <span className="class-chip__count">{cnt}</span>
                      </div>
                    ))}
                </div>
              </div>
            )}
        </>
      )}

      {tab === "records" && (
        <div>
          {task.id === "lpr" && visualChips.length > 0 && (
            <div
              style={{
                maxHeight: "400px",
                overflowY: "auto",
                borderBottom: "1px solid var(--line)",
              }}
            >
              <table
                style={{
                  width: "100%",
                  borderCollapse: "collapse",
                  fontSize: 12,
                }}
              >
                <thead
                  style={{
                    position: "sticky",
                    top: 0,
                    background: "var(--surface)",
                    zIndex: 1,
                  }}
                >
                  <tr
                    style={{
                      borderBottom: "1px solid var(--line)",
                      color: "var(--text-3)",
                    }}
                  >
                    <th style={{ padding: "6px 8px", textAlign: "left" }}>
                      Time
                    </th>
                    <th style={{ padding: "6px 8px", textAlign: "left" }}>
                      Plate
                    </th>
                    <th style={{ padding: "6px 8px", textAlign: "left" }}>
                      Det.Conf
                    </th>
                    <th style={{ padding: "6px 8px", textAlign: "left" }}>
                      OCR Conf
                    </th>
                    <th style={{ padding: "6px 8px", textAlign: "left" }}>
                      Valid
                    </th>
                    <th style={{ padding: "6px 8px", textAlign: "left" }}>
                      Box
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {visualChips.map((p, i) => (
                    <tr
                      key={i}
                      style={{ borderBottom: "1px solid var(--line)" }}
                    >
                      <td
                        style={{ padding: "6px 8px", color: "var(--text-3)" }}
                      >
                        {p.timestamp != null
                          ? `${p.timestamp.toFixed(2)}s`
                          : "—"}
                      </td>
                      <td
                        style={{
                          padding: "6px 8px",
                          fontFamily: "monospace",
                          fontWeight: 700,
                        }}
                      >
                        {p.plate_text || "—"}
                      </td>
                      <td
                        style={{ padding: "6px 8px", color: "var(--text-2)" }}
                      >
                        {p.det_confidence != null
                          ? `${(p.det_confidence * 100).toFixed(1)}%`
                          : "—"}
                      </td>
                      <td
                        style={{ padding: "6px 8px", color: "var(--text-2)" }}
                      >
                        {p.ocr_confidence != null
                          ? `${(p.ocr_confidence * 100).toFixed(1)}%`
                          : "—"}
                      </td>
                      <td style={{ padding: "6px 8px" }}>
                        <span
                          className={`tag ${p.valid_format ? "tag--green" : "tag--orange"}`}
                        >
                          {p.valid_format ? "Yes" : "No"}
                        </span>
                      </td>
                      <td
                        style={{
                          padding: "6px 8px",
                          color: "var(--text-3)",
                          fontSize: 10,
                        }}
                      >
                        {p.box ? `[${p.box.join(", ")}]` : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {(task.id === "vpd" || task.id === "vid") &&
            detections.length > 0 && (
              <div
                style={{
                  maxHeight: "400px",
                  overflowY: "auto",
                  borderBottom: "1px solid var(--line)",
                }}
              >
                <table
                  style={{
                    width: "100%",
                    borderCollapse: "collapse",
                    fontSize: 12,
                  }}
                >
                  <thead
                    style={{
                      position: "sticky",
                      top: 0,
                      background: "var(--surface)",
                      zIndex: 1,
                    }}
                  >
                    <tr
                      style={{
                        borderBottom: "1px solid var(--line)",
                        color: "var(--text-3)",
                      }}
                    >
                      <th style={{ padding: "6px 8px", textAlign: "left" }}>
                        Time
                      </th>
                      <th style={{ padding: "6px 8px", textAlign: "left" }}>
                        Class
                      </th>
                      <th style={{ padding: "6px 8px", textAlign: "left" }}>
                        Confidence
                      </th>
                      <th style={{ padding: "6px 8px", textAlign: "left" }}>
                        Location [x1,y1,x2,y2]
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {detections.map((d, i) => (
                      <tr
                        key={i}
                        style={{ borderBottom: "1px solid var(--line)" }}
                      >
                        <td
                          style={{ padding: "6px 8px", color: "var(--text-3)" }}
                        >
                          {d.timestamp != null
                            ? `${d.timestamp.toFixed(2)}s`
                            : "—"}
                        </td>
                        <td style={{ padding: "6px 8px", fontWeight: 600 }}>
                          {d.class_name}
                        </td>
                        <td
                          style={{ padding: "6px 8px", color: "var(--text-2)" }}
                        >
                          {(d.confidence * 100).toFixed(1)}%
                        </td>
                        <td
                          style={{
                            padding: "6px 8px",
                            color: "var(--text-3)",
                            fontSize: 10,
                          }}
                        >
                          [{d.box?.join(", ")}]
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
        </div>
      )}

      {tab === "frames" && frames.length > 0 && (
        <div className="frames-grid">
          {frames.map((url, i) => (
            <div
              key={i}
              className="frame-thumb"
              onClick={() => window.open(url, "_blank")}
            >
              <img src={url} alt={`Frame ${i}`} loading="lazy" />
            </div>
          ))}
        </div>
      )}

      {tab === "session" && result.summary && (
        <div>
          {Object.entries(result.summary).map(([k, v]) => (
            <div key={k} className="metric-row">
              <span className="metric-row__key">{k.replace(/_/g, " ")}</span>
              <span className="metric-row__val">
                {Array.isArray(v)
                  ? v.join(", ") || "—"
                  : typeof v === "object" && v !== null
                    ? JSON.stringify(v)
                    : String(v ?? "—")}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const MetricsPanel = ({ taskId }) => {
  const [metrics, setMetrics] = useState(null);
  const task = TASKS[taskId];

  useEffect(() => {
    if (!task?.endpoints?.metrics) return;
    fetch(task.endpoints.metrics)
      .then((r) => r.json())
      .then(setMetrics)
      .catch(() => {});
  }, [taskId]);

  if (!metrics)
    return (
      <div className="loading-state" style={{ minHeight: 200 }}>
        <Ic n="Loader" s={24} cls="spin loading-state__icon" />
        <span className="loading-state__label">Loading metrics…</span>
      </div>
    );

  const detection = metrics.detection || {};
  const perClass = metrics.per_class_ap50 || {};

  return (
    <div className="metrics-panel">
      <div className="metrics-section">
        <div className="metrics-section__title">
          <Ic n="BarChart" s={13} />
          Model Info
        </div>
        {[
          ["Model", metrics.model],
          ["Dataset", metrics.dataset],
          ["Status", metrics.status || "production"],
        ].map(
          ([k, v]) =>
            v && (
              <div key={k} className="metric-row">
                <span className="metric-row__key">{k}</span>
                <span
                  className="metric-row__val"
                  style={{ fontSize: 11, maxWidth: "60%", textAlign: "right" }}
                >
                  {v}
                </span>
              </div>
            ),
        )}
      </div>

      {Object.keys(detection).length > 0 && (
        <div className="metrics-section">
          <div className="metrics-section__title">
            <Ic n="Activity" s={13} />
            Detection Performance
          </div>
          {Object.entries(detection).map(([k, v]) => {
            const isPercent = [
              "mAP50",
              "mAP50_95",
              "precision",
              "recall",
            ].includes(k);
            const pct = isPercent ? v * 100 : null;
            return (
              <div key={k} className="perf-bar-row">
                <span className="perf-bar-row__label">
                  {k.replace(/_/g, " ")}
                </span>
                {pct != null ? (
                  <>
                    <div className="perf-bar-row__bar">
                      <div
                        className="perf-bar-row__fill"
                        style={{
                          width: `${pct}%`,
                          background:
                            pct >= 90
                              ? "var(--green)"
                              : pct >= 75
                                ? "var(--accent)"
                                : "var(--orange)",
                        }}
                      />
                    </div>
                    <span className="perf-bar-row__val">{pct.toFixed(1)}%</span>
                  </>
                ) : (
                  <span
                    className="perf-bar-row__val"
                    style={{ marginLeft: "auto", textAlign: "right" }}
                  >
                    {v}
                    {k.includes("ms") ? " ms" : ""}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      )}

      {Object.keys(perClass).length > 0 && (
        <div className="metrics-section">
          <div className="metrics-section__title">
            <Ic n="List" s={13} />
            Per-Class AP@50
          </div>
          {Object.entries(perClass)
            .sort((a, b) => b[1] - a[1])
            .map(([cls, ap]) => (
              <div key={cls} className="perf-bar-row">
                <span className="perf-bar-row__label">{cls}</span>
                <div className="perf-bar-row__bar">
                  <div
                    className="perf-bar-row__fill"
                    style={{
                      width: `${ap * 100}%`,
                      background: "var(--task-color)",
                    }}
                  />
                </div>
                <span className="perf-bar-row__val">
                  {(ap * 100).toFixed(1)}%
                </span>
              </div>
            ))}
        </div>
      )}

      {metrics.ocr && (
        <div className="metrics-section">
          <div className="metrics-section__title">
            <Ic n="List" s={13} />
            OCR Performance
          </div>
          {Object.entries(metrics.ocr).map(([k, v]) => (
            <div key={k} className="metric-row">
              <span className="metric-row__key">{k.replace(/_/g, " ")}</span>
              <span className="metric-row__val">
                {typeof v === "number"
                  ? `${(v * 100).toFixed(1)}%`
                  : Array.isArray(v)
                    ? v.join(", ")
                    : String(v)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const SessionsPanel = ({ taskId, onSelect }) => {
  const [sessions, setSessions] = useState([]);
  const task = TASKS[taskId];

  const load = useCallback(() => {
    if (!task?.endpoints?.sessions) return;
    fetch(task.endpoints.sessions)
      .then((r) => r.json())
      .then(setSessions)
      .catch(() => {});
  }, [taskId]);

  useEffect(() => {
    load();
  }, [load]);

  if (!sessions.length)
    return (
      <div className="empty-state">
        <Ic n="Clock" s={32} cls="empty-state__icon" />
        <p>
          No sessions yet.
          <br />
          Run inference to create one.
        </p>
      </div>
    );

  return (
    <div className="sessions-list">
      {sessions.map((s) => (
        <div
          key={s.session_id}
          className="session-card"
          onClick={() => onSelect(s.session_id)}
        >
          <div className="session-card__header">
            <span className="session-card__id">{s.session_id}</span>
            <span className="session-card__time">{fmtTime(s.started_at)}</span>
          </div>
          <div className="session-card__stats">
            <span className="session-card__stat">
              Source: <span>{s.source}</span>
            </span>
            <span className="session-card__stat">
              Frames: <span>{s.total_frames}</span>
            </span>
            {s.unique_plate_count != null && (
              <span className="session-card__stat">
                Plates: <span>{s.unique_plate_count}</span>
              </span>
            )}
            {s.total_detections != null && (
              <span className="session-card__stat">
                Detections: <span>{s.total_detections}</span>
              </span>
            )}
          </div>
        </div>
      ))}
      <button
        className="btn btn--ghost"
        style={{ fontSize: 11, alignSelf: "flex-start" }}
        onClick={load}
      >
        <Ic n="Loader" s={12} /> Refresh
      </button>
    </div>
  );
};

export default function App() {
  const [activeTask, setActiveTask] = useState("lpr");
  const [activeMode, setActiveMode] = useState("demo_image");
  const [rightTab, setRightTab] = useState("result");
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [file, setFile] = useState(null);
  const [serverUp, setServerUp] = useState(null);
  const [runCount, setRunCount] = useState(0);
  const [ipCamUrl, setIpCamUrl] = useState("");
  const task = TASKS[activeTask];
  const color = task.color;
  const streamAbortRef = useRef(null);

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((r) => setServerUp(r.ok))
      .catch(() => setServerUp(false));
  }, []);

  const resetAll = () => {
    setResult(null);
    setFile(null);
    setProgress(0);
  };
  const switchTask = (id) => {
    setActiveTask(id);
    resetAll();
    setActiveMode("demo_image");
    setRightTab("result");
  };
  const switchMode = (m) => {
    setActiveMode(m);
    resetAll();
  };

  const handleSessionSelect = async (sid) => {
    const ep = `${API_BASE}/api/${activeTask}/sessions/${sid}`;
    try {
      const res = await fetch(ep);
      const data = await res.json();
      setResult((prev) => ({
        ...prev,
        streamUrl: null,
        annotatedUrl: data.annotated_frames?.[0] || prev?.annotatedUrl || null,
        count:
          data.summary?.total_detections ??
          data.summary?.unique_plate_count ??
          prev?.count ??
          0,
        plates: prev?.plates || [],
        detections: prev?.detections || [],
        class_summary: Object.keys(data.summary?.class_totals || {}).length
          ? data.summary.class_totals
          : prev?.class_summary || {},
        valid_plates: data.summary?.valid_plates || prev?.valid_plates || [],
        summary: data.summary || prev?.summary,
        session_id: sid,
        annotated_frames: data.annotated_frames || prev?.annotated_frames || [],
      }));
      setRightTab("result");
    } catch {}
  };

  const processMJPEGStream = async (res, { signal } = {}) => {
    const sessId = res.headers.get("X-Session-Id");
    setResult({
      streamUrl: "",
      annotatedUrl: null,
      session_id: sessId,
      count: 0,
      class_summary: {},
      detections: [],
      plates: [],
    });
    setLoading(false);
    setStreaming(true);

    const reader = res.body.getReader();
    let buffer = new Uint8Array();
    const decoder = new TextDecoder();

    try {
      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          if (sessId) await handleSessionSelect(sessId);
          break;
        }

        const newBuffer = new Uint8Array(buffer.length + value.length);
        newBuffer.set(buffer);
        newBuffer.set(value, buffer.length);
        buffer = newBuffer;

        const soi = [0xff, 0xd8];
        const eoi = [0xff, 0xd9];

        let startIdx = -1,
          endIdx = -1;
        for (let i = 0; i < buffer.length - 1; i++) {
          if (buffer[i] === soi[0] && buffer[i + 1] === soi[1]) {
            startIdx = i;
            break;
          }
        }
        if (startIdx !== -1) {
          for (let i = startIdx; i < buffer.length - 1; i++) {
            if (buffer[i] === eoi[0] && buffer[i + 1] === eoi[1]) {
              endIdx = i + 1;
              break;
            }
          }
        }

        if (startIdx !== -1 && endIdx !== -1) {
          const headerBytes = buffer.slice(0, startIdx);
          const headerText = decoder.decode(headerBytes);

          const countMatch = headerText.match(/X-Count:\s*(.+)/);
          const classesMatch = headerText.match(/X-Classes:\s*(.+)/);
          const detsMatch = headerText.match(/X-Detections:\s*(.+)/);
          const platesMatch = headerText.match(/X-Plates:\s*(.+)/);
          const tsMatch = headerText.match(/X-Timestamp:\s*(.+)/);

          const count = countMatch ? parseInt(countMatch[1]) : 0;
          const classes = classesMatch
            ? JSON.parse(classesMatch[1].trim())
            : {};
          const detections = detsMatch ? JSON.parse(detsMatch[1].trim()) : [];
          const parsedPlates = platesMatch
            ? JSON.parse(platesMatch[1].trim())
            : [];
          const timestamp = tsMatch ? parseFloat(tsMatch[1].trim()) : 0.0;

          const stampedDets = detections.map((d) => ({ ...d, timestamp }));
          const stampedPlates = parsedPlates.map((p) => ({ ...p, timestamp }));

          const imgBytes = buffer.slice(startIdx, endIdx + 1);
          const blob = new Blob([imgBytes], { type: "image/jpeg" });
          const url = URL.createObjectURL(blob);

          setResult((prev) => {
            if (prev?.streamUrl && prev.streamUrl.startsWith("blob:")) {
              URL.revokeObjectURL(prev.streamUrl);
            }

            const mergedDets = [...stampedDets, ...(prev?.detections || [])];
            if (mergedDets.length > 2000) mergedDets.length = 2000;

            const mergedPlates =
              stampedPlates.length > 0 ? stampedPlates : prev?.plates || [];

            return {
              ...prev,
              streamUrl: url,
              annotatedUrl: null,
              count,
              class_summary: classes,
              detections: mergedDets,
              plates: mergedPlates,
            };
          });

          buffer = buffer.slice(endIdx + 1);
        }
      }
    } catch (e) {
      if (e.name !== "AbortError") console.error("MJPEG Stream Error:", e);
    } finally {
      setStreaming(false);
      streamAbortRef.current = null;
    }
  };

  const runInference = useCallback(
    async (uploadFile, opts = {}) => {
      setLoading(true);
      setProgress(0);
      setResult(null);

      const {
        isVideo = false,
        isDemo = false,
        isDemoVideo = false,
        isVideoAnalyze = false,
        isCameraLive = false,
      } = opts;

      try {
        if (isCameraLive) {
          const controller = new AbortController();
          streamAbortRef.current = controller;

          const source = encodeURIComponent(opts.cameraSource || "0");
          const res = await fetch(
            `${task.endpoints.cameraLive}?camera=${source}`,
            { signal: controller.signal },
          );
          if (!res.ok)
            throw new Error(
              (await res.text()) || "Camera stream failed to start",
            );
          await processMJPEGStream(res, { signal: controller.signal });
          setRunCount((c) => c + 1);
          return;
        }

        if (isDemoVideo) {
          const controller = new AbortController();
          streamAbortRef.current = controller;
          const res = await fetch(task.endpoints.demoVideo, {
            signal: controller.signal,
          });
          if (!res.ok) throw new Error("Demo video stream failed");
          await processMJPEGStream(res, { signal: controller.signal });
          setRunCount((c) => c + 1);
          return;
        }

        if (isVideoAnalyze) {
          const form = new FormData();
          form.append("file", uploadFile);
          const res = await fetch(
            `${task.endpoints.videoAnalyze}?save_frames=true&yolo_interval=3`,
            {
              method: "POST",
              body: form,
            },
          );
          if (!res.ok) throw new Error(await res.text());
          const data = await res.json();

          const allDets = (data.frame_results || []).flatMap((fr) =>
            (fr.detections || []).map((d) => ({
              ...d,
              timestamp: fr.timestamp,
            })),
          );
          const allPlates = (data.frame_results || []).flatMap((fr) =>
            (fr.plates || []).map((p) => ({ ...p, timestamp: fr.timestamp })),
          );

          setResult({
            count:
              data.summary?.total_detections ??
              data.summary?.unique_plate_count ??
              0,
            plates: allPlates,
            detections: allDets,
            class_summary: data.summary?.class_totals || {},
            valid_plates: data.summary?.valid_plates || [],
            annotated_frames: data.annotated_frames || [],
            summary: data.summary,
            session_id: data.session_id,
          });
          setRunCount((c) => c + 1);
          return;
        }

        if (isVideo) {
          const form = new FormData();
          form.append("file", uploadFile);
          const controller = new AbortController();
          streamAbortRef.current = controller;
          const res = await fetch(task.endpoints.video + "?yolo_interval=3", {
            method: "POST",
            body: form,
            signal: controller.signal,
          });
          if (!res.ok) throw new Error(await res.text());
          await processMJPEGStream(res, { signal: controller.signal });
          setRunCount((c) => c + 1);
          return;
        }

        if (isDemo) {
          const [annRes, jsonRes] = await Promise.all([
            fetch(task.endpoints.demoImageAnn),
            fetch(task.endpoints.demoImage),
          ]);
          if (!annRes.ok) throw new Error(await annRes.text());
          const annBlob = await annRes.blob();
          const annUrl = URL.createObjectURL(annBlob);
          const data = jsonRes.ok ? await jsonRes.json() : {};

          setResult({
            annotatedUrl: annUrl,
            count: data.count ?? 0,
            plates: data.plates ?? [],
            detections: data.detections ?? [],
            class_summary: data.class_summary ?? {},
            valid_plates: data.valid_plates ?? [],
            summary: data.summary,
            session_id: data.session_id,
            annotated_frames: data.annotated_frames || [],
          });
          setRunCount((c) => c + 1);
          return;
        }

        const form = new FormData();
        form.append("file", uploadFile);

        const [annRes, jsonRes] = await Promise.all([
          fetch(task.endpoints.imageAnnotated, {
            method: "POST",
            body: (() => {
              const f = new FormData();
              f.append("file", uploadFile);
              return f;
            })(),
          }),
          fetch(task.endpoints.image, { method: "POST", body: form }),
        ]);

        if (!jsonRes.ok) throw new Error(await jsonRes.text());
        const data = await jsonRes.json();

        let annUrl = null;
        if (annRes.ok) {
          const ab = await annRes.blob();
          annUrl = URL.createObjectURL(ab);
        } else if (data.annotated_url) {
          annUrl = data.annotated_url;
        }

        setResult({
          annotatedUrl: annUrl,
          count: data.count ?? 0,
          plates: data.plates ?? [],
          detections: data.detections ?? [],
          class_summary: data.class_summary ?? {},
          valid_plates: data.valid_plates ?? [],
          summary: data.summary,
          session_id: data.session_id,
          annotated_frames: data.annotated_frames || [],
        });
        setRunCount((c) => c + 1);
      } catch (e) {
        if (e.name !== "AbortError")
          setResult({ error: e.message || String(e) });
      } finally {
        setLoading(false);
      }
    },
    [task, activeMode],
  );

  const stopStream = () => {
    const sid = result?.session_id;
    streamAbortRef.current?.abort();
    streamAbortRef.current = null;
    setStreaming(false);
    if (sid) handleSessionSelect(sid);
  };

  const handleRun = () => {
    if (activeMode === "upload_image") return runInference(file);
    if (activeMode === "upload_video")
      return runInference(file, { isVideo: true });
    if (activeMode === "demo_image")
      return runInference(null, { isDemo: true });
    if (activeMode === "demo_video")
      return runInference(null, { isDemoVideo: true });

    if (activeMode === "camera_video")
      return runInference(null, { isCameraLive: true, cameraSource: ipCamUrl });
  };

  const handleVideoAnalyze = () => runInference(file, { isVideoAnalyze: true });

  const renderInputPanel = () => {
    if (!task.available)
      return (
        <div className="coming-soon-card">
          <div style={{ fontSize: 36 }}>🚧</div>
          <h3>Coming Soon</h3>
          <p>
            {task.description}
            <br />
            Model is currently in training. Expected Q3 2026.
          </p>
        </div>
      );

    if (activeMode === "camera_image") {
      return <CameraPanel onCapture={(f) => runInference(f)} />;
    }

    if (activeMode === "camera_video") {
      return (
        <div className="camera-panel">
          <div className="camera-info">
            <Ic n="Radio" s={13} /> Live inference on local webcam or external
            IP Camera
          </div>

          {!streaming && (
            <div style={{ marginTop: 16, marginBottom: 20, textAlign: "left" }}>
              <label
                style={{
                  display: "block",
                  fontSize: 11,
                  color: "var(--text-3)",
                  marginBottom: 6,
                  fontWeight: 500,
                }}
              >
                CAMERA SOURCE (Leave blank for default webcam)
              </label>
              <input
                type="text"
                placeholder="e.g., http://192.168.1.55:8080/video"
                value={ipCamUrl}
                onChange={(e) => setIpCamUrl(e.target.value)}
                style={{
                  width: "100%",
                  padding: "10px 12px",
                  borderRadius: 8,
                  border: "1px solid var(--line)",
                  background: "var(--bg)",
                  color: "var(--text-1)",
                  fontSize: 13,
                  fontFamily: "monospace",
                }}
              />
            </div>
          )}

          <div className="btn-row">
            {!streaming ? (
              <button
                className="btn btn--primary"
                disabled={loading}
                onClick={handleRun}
              >
                {loading ? (
                  <>
                    <Ic n="Loader" s={14} cls="spin" /> Starting…
                  </>
                ) : (
                  <>
                    <Ic n="PlayCircle" s={14} /> Start Live Stream
                  </>
                )}
              </button>
            ) : (
              <button className="btn btn--danger" onClick={stopStream}>
                <Ic n="X" s={14} /> Stop & Show Analysis
              </button>
            )}
          </div>
        </div>
      );
    }

    if (activeMode === "upload_image") {
      return (
        <>
          <DropZone
            accept="image/*"
            onFile={setFile}
            file={file}
            label="Drop an image here"
          />
          {file && (
            <div className="btn-row">
              <button
                className="btn btn--primary"
                disabled={loading}
                onClick={handleRun}
              >
                {loading ? (
                  <>
                    <Ic n="Loader" s={14} cls="spin" /> Running…
                  </>
                ) : (
                  <>
                    <Ic n="Zap" s={14} /> Run Inference
                  </>
                )}
              </button>
              <button className="btn btn--ghost" onClick={resetAll}>
                <Ic n="X" s={13} />
              </button>
            </div>
          )}
        </>
      );
    }

    if (activeMode === "upload_video") {
      return (
        <>
          <DropZone
            accept="video/*"
            onFile={setFile}
            file={file}
            label="Drop a video file here"
          />
          {file && (
            <>
              <div className="btn-row">
                {!streaming ? (
                  <button
                    className="btn btn--primary"
                    disabled={loading}
                    onClick={handleRun}
                  >
                    {loading ? (
                      <>
                        <Ic n="Loader" s={14} cls="spin" /> Streaming…
                      </>
                    ) : (
                      <>
                        <Ic n="PlayCircle" s={14} /> Live Stream
                      </>
                    )}
                  </button>
                ) : (
                  <button className="btn btn--danger" onClick={stopStream}>
                    <Ic n="X" s={14} /> Stop & Show Analysis
                  </button>
                )}
                <button
                  className="btn btn--outline"
                  disabled={loading || streaming}
                  onClick={handleVideoAnalyze}
                >
                  {loading ? (
                    <>
                      <Ic n="Loader" s={14} cls="spin" />
                    </>
                  ) : (
                    <>
                      <Ic n="BarChart" s={14} /> Full Analysis
                    </>
                  )}
                </button>
                <button className="btn btn--ghost" onClick={resetAll}>
                  <Ic n="X" s={13} />
                </button>
              </div>
              <p
                style={{ fontSize: 11, color: "var(--text-3)", marginTop: -4 }}
              >
                Live Stream: real-time MJPEG · Full Analysis: offline processing
                + saved frames + analytics
              </p>
            </>
          )}
        </>
      );
    }

    if (activeMode === "demo_image" || activeMode === "demo_video") {
      return (
        <div className="demo-panel">
          <div className="demo-panel__icon-wrap">
            <Ic
              n={activeMode === "demo_image" ? "Sparkles" : "PlayCircle"}
              s={24}
            />
          </div>
          <div className="demo-panel__title">
            {activeMode === "demo_image" ? "Demo Image" : "Demo Video"}
          </div>
          <div className="demo-panel__sub">
            Asset:{" "}
            <code style={{ color: task.color }}>
              demo/{task.id}.{activeMode === "demo_image" ? "jpg" : "mp4"}
            </code>
          </div>
          {activeMode === "demo_video" && streaming ? (
            <button
              className="btn btn--danger"
              onClick={stopStream}
              style={{ marginTop: 8 }}
            >
              <Ic n="X" s={14} /> Stop & Show Analysis
            </button>
          ) : (
            <button
              className="btn btn--primary"
              disabled={loading}
              onClick={handleRun}
              style={{ marginTop: 8 }}
            >
              {loading ? (
                <>
                  <Ic n="Loader" s={14} cls="spin" /> Running…
                </>
              ) : (
                <>
                  <Ic n="PlayCircle" s={14} /> Run Demo
                </>
              )}
            </button>
          )}
        </div>
      );
    }
  };

  const renderRightPanel = () => {
    return (
      <>
        <div className="panel-header">
          <div className="tab-nav" style={{ flex: 1 }}>
            <button
              className={`tab-btn ${rightTab === "result" ? "tab-btn--active" : ""}`}
              onClick={() => setRightTab("result")}
            >
              Result{" "}
              {runCount > 0 && (
                <span style={{ opacity: 0.5, fontSize: 10 }}>({runCount})</span>
              )}
            </button>
            <button
              className={`tab-btn ${rightTab === "metrics" ? "tab-btn--active" : ""}`}
              onClick={() => setRightTab("metrics")}
            >
              Metrics
            </button>
            <button
              className={`tab-btn ${rightTab === "sessions" ? "tab-btn--active" : ""}`}
              onClick={() => setRightTab("sessions")}
            >
              Sessions
            </button>
          </div>
        </div>

        {rightTab === "result" && (
          <>
            {loading && (
              <div className="loading-state" style={{ minHeight: 200 }}>
                <Ic n="Loader" s={32} cls="spin loading-state__icon" />
                <span className="loading-state__label">Running inference…</span>
              </div>
            )}
            {!loading && result && (
              <ResultDashboard result={result} task={task} />
            )}
            {!loading && !result && (
              <div className="empty-state">
                <Ic n="Image" s={40} cls="empty-state__icon" />
                <p>
                  Pick a mode on the left,
                  <br />
                  then run inference to see results here.
                </p>
              </div>
            )}
          </>
        )}

        {rightTab === "metrics" && (
          <div style={{ "--task-color": color }}>
            <MetricsPanel taskId={activeTask} />
          </div>
        )}

        {rightTab === "sessions" && (
          <SessionsPanel taskId={activeTask} onSelect={handleSessionSelect} />
        )}
      </>
    );
  };

  return (
    <div className="app" style={{ "--task-color": color }}>
      <header className="topbar">
        <div className="topbar__brand">
          <Ic n="Shield" s={20} cls="topbar__logo" />
          <span className="topbar__name">TrafficLens</span>
          <span className="topbar__tag">Safer Roads,Better Future</span>
        </div>
        <div className="topbar__right">
          <div className="server-indicator">
            <div
              className={`server-dot ${serverUp === true ? "server-dot--on" : serverUp === false ? "server-dot--off" : ""}`}
            />
            {serverUp === true
              ? "API Online"
              : serverUp === false
                ? "API Offline"
                : "Checking…"}
          </div>
        </div>
      </header>

      <main className="main">
        <section className="section">
          <div className="section__eyebrow">
            <Ic n="Zap" s={12} /> Select Task
          </div>
          <div className="task-grid">
            {Object.values(TASKS).map((t) => (
              <TaskCard
                key={t.id}
                task={t}
                active={activeTask === t.id}
                onClick={switchTask}
              />
            ))}
          </div>
        </section>

        <section className="section">
          <div className="section__eyebrow">
            <Ic n="Activity" s={12} /> Input Mode
          </div>
          <ModeSelector
            active={activeMode}
            onChange={switchMode}
            disabled={!task.available}
          />
        </section>

        <section className="section workspace">
          <div className="section__eyebrow">
            <Ic n="Clock" s={12} />
            Workspace — <span style={{ color }}>{task.label}</span>
          </div>
          <div className="workspace__body">
            <div className="workspace__input" style={{ "--task-color": color }}>
              {renderInputPanel()}
            </div>
            <div
              className="workspace__output"
              style={{ "--task-color": color }}
            >
              {renderRightPanel()}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
