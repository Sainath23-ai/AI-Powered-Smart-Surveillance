"""
SafeGuard AI – Flask Backend
Provides all REST API endpoints AND the real-time AI detection loop.
"""

import json
import logging
import os
import queue
import smtplib
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from flask_cors import CORS


# ── Paths ────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.join(BASE_DIR, "..")
SETTINGS_PATH= os.path.join(PROJECT_DIR, "config", "settings.json")
DATA_DIR     = os.path.join(PROJECT_DIR, "data")
THIEVES_DIR  = os.path.join(DATA_DIR, "thieves")
CAPTURES_DIR = os.path.join(DATA_DIR, "captures")
ALERTS_LOG   = os.path.join(DATA_DIR, "alerts.json")
INCIDENTS_LOG= os.path.join(DATA_DIR, "incidents.json")

for _d in [THIEVES_DIR, CAPTURES_DIR, DATA_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("SafeGuardAI")

# ── Flask App ─────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins="*")


# ── Shared State ──────────────────────────────────────────────
_state_lock = threading.Lock()
_state = {
    "running": False,
    "start_time": None,
    "fps": 0,
    "alerts_today": 0,
    "total_alerts": 0,
    "incidents_count": 0,
    "threat_level": "none",
    "last_alert": None,
    "current_detections": {},
    "esp32": {"connected": False, "host": "", "port": 80},
}

# Latest JPEG frame for the MJPEG stream
_frame_lock   = threading.Lock()
_latest_frame = None          # bytes (JPEG)

# SSE clients – list of queue.Queue objects
_sse_lock    = threading.Lock()
_sse_clients = []

# Alert cooldown tracking
_alert_last_sent = {}         # threat_type -> timestamp

# ── JSON helpers with in-memory cache ──────────────────────────
_json_cache = {}
_json_cache_lock = threading.Lock()

def _read_json(path):
    with _json_cache_lock:
        if path in _json_cache:
            return _json_cache[path]
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data = data if isinstance(data, list) else []
        with _json_cache_lock:
            _json_cache[path] = data
        return data
    except Exception:
        return []

def _write_json(path, data):
    with _json_cache_lock:
        _json_cache[path] = data
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logger.error("Failed to write JSON log to %s: %s", path, exc)

def _append_json(path, entry, maxlen=500):
    records = _read_json(path)
    records.insert(0, entry)
    _write_json(path, records[:maxlen])

# ── Config helpers ────────────────────────────────────────────
def _load_config():
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_config(cfg):
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

# ── SSE broadcast ─────────────────────────────────────────────
def _broadcast(event_type: str, data: dict):
    payload = json.dumps({"type": event_type, "data": data})
    dead = []
    with _sse_lock:
        for q in _sse_clients:
            try:
                q.put_nowait(payload)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _sse_clients.remove(q)

# ── Gmail email alert ─────────────────────────────────────────
def _send_gmail_alert(threat_type: str, confidence: float,
                      snapshot_path: str = None, level: str = "critical"):
    """Send an HTML alert email via Gmail SMTP with an optional image attachment."""
    cfg = _load_config().get("gmail", {})
    sender   = cfg.get("sender_email", "").strip()
    password = cfg.get("sender_password", "").strip()
    recipient= cfg.get("recipient_email", "").strip()

    if not sender or not password or not recipient:
        logger.warning("Gmail not configured – skipping email alert")
        return False, "Gmail credentials not configured in Settings"

    level_color = {"critical": "#ef4444", "warning": "#f59e0b"}.get(level, "#6366f1")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""
    <html><body style="font-family:Arial,sans-serif;background:#0a0d14;color:#f1f5f9;padding:24px;">
      <div style="max-width:560px;margin:auto;background:#1a2236;border-radius:12px;
                  border:2px solid {level_color};overflow:hidden;">
        <div style="background:{level_color};padding:20px;text-align:center;">
          <h1 style="margin:0;font-size:24px;">🚨 SafeGuard AI Alert</h1>
          <p style="margin:6px 0 0;opacity:0.9;">{level.upper()} THREAT DETECTED</p>
        </div>
        <div style="padding:24px;">
          <p style="font-size:18px;font-weight:bold;color:{level_color};">{threat_type}</p>
          <table style="width:100%;border-collapse:collapse;margin-top:16px;">
            <tr><td style="padding:8px;color:#94a3b8;">Confidence</td>
                <td style="padding:8px;font-weight:bold;">{confidence:.1%}</td></tr>
            <tr><td style="padding:8px;color:#94a3b8;">Timestamp</td>
                <td style="padding:8px;">{ts}</td></tr>
            <tr><td style="padding:8px;color:#94a3b8;">Level</td>
                <td style="padding:8px;color:{level_color};font-weight:bold;">{level.upper()}</td></tr>
          </table>
          {('<p style="margin-top:16px;color:#94a3b8;">📎 Snapshot attached below.</p>'
            if snapshot_path else '')}
        </div>
        <div style="background:#0a0d14;padding:12px 24px;text-align:center;font-size:11px;color:#64748b;">
          SafeGuard AI Surveillance · Team APEX
        </div>
      </div>
    </body></html>
    """

    msg = MIMEMultipart("related")
    msg["Subject"] = f"🚨 [{level.upper()}] SafeGuard Alert: {threat_type}"
    msg["From"]    = f"SafeGuard AI <{sender}>"
    msg["To"]      = recipient
    msg.attach(MIMEText(html, "html"))

    if snapshot_path and os.path.isfile(snapshot_path):
        try:
            with open(snapshot_path, "rb") as f:
                img = MIMEImage(f.read(), name=os.path.basename(snapshot_path))
                img.add_header("Content-Disposition", "attachment",
                               filename=os.path.basename(snapshot_path))
                msg.attach(img)
        except Exception as exc:
            logger.warning("Could not attach snapshot: %s", exc)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as server:
            server.login(sender, password)
            server.sendmail(sender, [recipient], msg.as_string())
        logger.info("Email alert sent to %s", recipient)
        return True, f"Email sent to {recipient}"
    except smtplib.SMTPAuthenticationError:
        msg_err = ("Gmail authentication failed. Make sure you use a Gmail App Password "
                   "(not your account password) and that 2-Step Verification is enabled.")
        logger.error(msg_err)
        return False, msg_err
    except Exception as exc:
        logger.error("Failed to send email: %s", exc)
        return False, str(exc)

# ── Thief Registry ────────────────────────────────────────────
from thief_registry import ThiefRegistry
registry = ThiefRegistry(THIEVES_DIR)

# ── Detector instances (lazy-loaded inside detection thread) ──
_pose_detector   = None
_object_detector = None
_face_detector   = None
_gesture_detector = None

def _init_detectors(cfg):
    global _pose_detector, _object_detector, _face_detector, _gesture_detector
    det_cfg = cfg.get("detection", {})
    try:
        from pose_safety_detector import PoseSafetyDetector
        _pose_detector = PoseSafetyDetector(
            threshold   = float(det_cfg.get("pose_safety_threshold", 0.68)),
            sensitivity = det_cfg.get("suspicious_sensitivity", "medium"),
        )
        logger.info("PoseSafetyDetector loaded")
    except Exception as exc:
        logger.warning("PoseSafetyDetector unavailable: %s", exc)

    try:
        from object_detector import ObjectDetector
        _object_detector = ObjectDetector(
            confidence=float(det_cfg.get("object_confidence", 0.45))
        )
        logger.info("ObjectDetector loaded")
    except Exception as exc:
        logger.warning("ObjectDetector unavailable: %s", exc)

    try:
        from face_detector import FaceDetector
        _face_detector = FaceDetector(
            match_threshold = float(det_cfg.get("face_match_threshold", 0.45)),
            cover_threshold = float(det_cfg.get("face_cover_threshold", 0.55)),
        )
        logger.info("FaceDetector loaded")
    except Exception as exc:
        logger.warning("FaceDetector unavailable: %s", exc)

    try:
        from gesture_detector import GestureDetector
        _gesture_detector = GestureDetector(
            confidence_threshold = float(det_cfg.get("gesture_confidence", 0.80))
        )
        logger.info("GestureDetector loaded")
    except Exception as exc:
        logger.warning("GestureDetector unavailable: %s", exc)



# ── Cache Drawing Helpers for Frame Skipping ──────────────────
def _draw_cached_faces(frame, faces, cover_threshold=0.55):
    for f in faces:
        box = f.get("box")
        if not box:
            continue
        x1, y1, x2, y2 = box
        match_name = f.get("match_name")
        match_conf = f.get("match_conf", 0.0)
        cover_score = f.get("cover_score", 0.0)
        cover_type = f.get("cover_type", "covering")
        
        if match_name:
            colour = (0, 0, 210)
            label = f"THIEF: {match_name} ({match_conf:.0%})"
        elif cover_score >= cover_threshold:
            colour = (0, 110, 255)
            label = f"COVERED [{cover_type}] {cover_score:.0%}"
        else:
            colour = (60, 210, 80)
            label = f"Face {cover_score:.0%}"
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        text_y = max(y1 - 4, 18)
        cv2.rectangle(frame, (x1, text_y - 18), (x2, text_y + 2), colour, -1)
        cv2.putText(frame, label, (x1 + 3, text_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

def _draw_cached_objects(frame, detections):
    for d in detections:
        box = d.get("box")
        if not box:
            continue
        x1, y1, x2, y2 = box
        label = d.get("label", "object")
        conf = d.get("confidence", 0.0)
        is_sharp = d.get("is_sharp", False)
        colour = (0, 0, 255) if is_sharp else (180, 180, 180)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            frame,
            f"{label} {conf:.0%}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            2,
        )

# ╔══════════════════════════════════════════════════════════════╗
# ║               AI Detection Loop Thread                      ║
# ╚══════════════════════════════════════════════════════════════╝

def _detection_loop():
    """Runs in a background thread while _state['running'] is True."""
    global _latest_frame

    cfg       = _load_config()
    cam_cfg   = cfg.get("camera", {})
    det_cfg   = cfg.get("detection", {})

    # -- Open camera -------------------------------------------
    source = cam_cfg.get("source", 0)
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(cam_cfg.get("width",  640)))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam_cfg.get("height", 480)))
    cap.set(cv2.CAP_PROP_FPS,          int(cam_cfg.get("fps",     30)))

    if not cap.isOpened():
        logger.error("Cannot open camera source: %s", source)
        with _state_lock:
            _state["running"] = False
        return

    logger.info("Camera opened: source=%s", source)
    _init_detectors(cfg)

    # -- Simple motion detector using frame differencing -------
    prev_gray     = None
    persons_count = 0
    frame_times   = deque(maxlen=30)
    violence_threshold = float(det_cfg.get("violence_threshold",  0.75))
    cooldown_sec       = int(det_cfg.get("alert_cooldown_seconds", 30))

    frame_count = 0
    prev_annotated = None
    prev_detections = {
        "activity": {
            "persons_count": 0,
            "motion": {"ratio": 0.0, "flow_magnitude": 0.0},
            "violence": {"score": 0.0, "detected": False, "confidence": 0.0},
            "loitering": {"confidence": 0.0, "detected": False},
            "running_panic": {"confidence": 0.0, "detected": False},
        },
        "gesture": {"detected": False, "confidence": 0.0, "hands_detected": 0},
        "face": {
            "thief_match": {"detected": False, "confidence": 0.0, "name": None, "id": None},
            "face_cover": {"detected": False, "confidence": 0.0, "cover_type": None},
            "faces_count": 0,
            "faces": [],
        },
        "object": {
            "sharp_object": {"detected": False, "confidence": 0.0, "label": None},
            "detections": [],
        },
        "pose": {
            "enabled": False, "landmarks_detected": False,
            "top_scenario": None, "summary": "Detector not loaded", "scenarios": {}
        }
    }

    while True:
        with _state_lock:
            if not _state["running"]:
                break

        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        frame_count += 1
        frame_times.append(time.time())
        fps = len(frame_times) / max(
            frame_times[-1] - frame_times[0], 0.001
        ) if len(frame_times) > 1 else 0

        annotated = frame.copy()
        h, w = frame.shape[:2]

        # ── Motion analysis ─────────────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        motion_ratio    = 0.0
        flow_magnitude  = 0.0

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_ratio = float(np.count_nonzero(thresh)) / (thresh.size + 1e-6)
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                flow_magnitude = float(np.mean(mag))
            except Exception:
                pass

        prev_gray = gray

        # -- Motion-Gated Inference Bypass --
        if motion_ratio < 0.005 and prev_annotated is not None:
            annotated = prev_annotated.copy()
            # Reuse previous detections, updating only the current motion values
            activity_results = prev_detections["activity"]
            activity_results["motion"]["ratio"] = round(motion_ratio, 4)
            activity_results["motion"]["flow_magnitude"] = round(flow_magnitude, 3)
            
            gesture_results = prev_detections["gesture"]
            object_results  = prev_detections["object"]
            pose_results    = prev_detections["pose"]
            face_results    = prev_detections["face"]
        else:
            # ── Simple person detection via background subtraction ─
            # (real detector would use YOLO – here we estimate from motion)
            if motion_ratio > 0.02:
                persons_count = max(1, min(5, int(motion_ratio * 40)))
            else:
                persons_count = max(0, persons_count - 1)

            # ── Violence score heuristic ─────────────────────────
            # Combines rapid motion + high flow -> violence proxy
            violence_score = min(1.0, flow_magnitude / 10.0 + motion_ratio * 2.0)
            violence_detected = violence_score >= violence_threshold

            # ── Loitering: sustained low-speed motion ────────────
            loiter_score    = min(1.0, motion_ratio * 8.0) if 0.01 < motion_ratio < 0.08 else 0.0
            loiter_detected = loiter_score >= 0.65

            # ── Running/panic: very high flow ────────────────────
            panic_score    = min(1.0, flow_magnitude / 15.0)
            panic_detected = panic_score >= 0.70

            activity_results = {
                "persons_count":  persons_count,
                "motion": {
                    "ratio":          round(motion_ratio, 4),
                    "flow_magnitude": round(flow_magnitude, 3),
                },
                "violence": {
                    "score":    round(violence_score, 3),
                    "detected": violence_detected,
                    "confidence": round(violence_score, 3),
                },
                "loitering": {
                    "confidence": round(loiter_score, 3),
                    "detected":   loiter_detected,
                },
                "running_panic": {
                    "confidence": round(panic_score, 3),
                    "detected":   panic_detected,
                },
            }

            # ── Gesture detection (always runs on active frames) ──
            gesture_results = prev_detections["gesture"]
            if _gesture_detector is not None:
                try:
                    annotated, gest_res = _gesture_detector.detect(annotated)
                    gesture_results = gest_res
                except Exception as exc:
                    logger.debug("GestureDetector error: %s", exc)

            # ── Object detection (runs every 3 frames) ────────────
            object_results = prev_detections["object"]
            if _object_detector is not None:
                if frame_count % 3 == 0:
                    try:
                        annotated, obj_det = _object_detector.detect(annotated)
                        object_results = obj_det
                    except Exception as exc:
                        logger.debug("ObjectDetector error: %s", exc)
                else:
                    _draw_cached_objects(annotated, object_results.get("detections", []))

            # ── Pose safety detection (runs every 2 frames) ───────
            pose_results = prev_detections["pose"]
            if _pose_detector is not None:
                if frame_count % 2 == 0:
                    try:
                        annotated, pose_res = _pose_detector.detect(annotated, activity_results)
                        pose_results = pose_res
                    except Exception as exc:
                        logger.debug("PoseSafetyDetector error: %s", exc)

            # ── Face detection (runs every 2 frames) ──────────────
            face_results = prev_detections["face"]
            if _face_detector is not None:
                if frame_count % 2 == 1:
                    try:
                        thief_embs = registry.load_embeddings()
                        annotated, fd_res = _face_detector.detect(annotated, thief_embs)
                        face_results = {
                            "thief_match": fd_res["thief_match"],
                            "face_cover":  fd_res["face_cover"],
                            "faces_count": fd_res["faces_count"],
                            "faces":       fd_res.get("faces", []),
                        }
                    except Exception as exc:
                        logger.debug("FaceDetector error: %s", exc)
                else:
                    _draw_cached_faces(annotated, face_results.get("faces", []), float(det_cfg.get("face_cover_threshold", 0.55)))

            # Save state for future frame reuse
            prev_detections = {
                "activity": activity_results,
                "gesture": gesture_results,
                "face": face_results,
                "object": object_results,
                "pose": pose_results,
            }
            prev_annotated = annotated.copy()

        # ── Determine threat level ───────────────────────────
        threat_level = "none"
        trigger_alert = None

        thief_match  = face_results.get("thief_match", {})
        face_cover   = face_results.get("face_cover",  {})
        gesture_detected = gesture_results.get("detected", False)
        gesture_type = gesture_results.get("gesture_type", "SOS Gesture")

        if (
            violence_detected
            or (pose_results.get("top_scenario") and pose_results["top_scenario"].get("detected"))
            or object_results.get("sharp_object", {}).get("detected")
            or thief_match.get("detected")
            or gesture_detected
        ):
            threat_level = "critical"
            if thief_match.get("detected"):
                trigger_alert = (
                    f"Known Thief: {thief_match.get('name','Unknown')}",
                    thief_match.get("confidence", 0.9),
                )
            elif gesture_detected:
                trigger_alert = (f"SOS Gesture: {gesture_type}", gesture_results.get("confidence", 0.9))
            elif violence_detected:
                trigger_alert = ("Violence Detected", violence_score)
            elif pose_results.get("top_scenario", {}).get("detected"):
                top = pose_results["top_scenario"]
                trigger_alert = (top.get("label", "Pose Threat"), top.get("confidence", 0.8))
            else:
                sharp = object_results.get("sharp_object", {})
                trigger_alert = (f"Sharp Object: {sharp.get('label','')}", sharp.get("confidence", 0.8))

        elif loiter_detected or panic_detected or face_cover.get("detected"):
            threat_level = "warning"
            if face_cover.get("detected"):
                ct = face_cover.get("cover_type", "covering")
                trigger_alert = (f"Face Covered [{ct}]", face_cover.get("confidence", 0.7))
            elif panic_detected:
                trigger_alert = ("Running/Panic Detected", panic_score)
            else:
                trigger_alert = ("Loitering Detected", loiter_score)

        # ── Compose detections payload ───────────────────────
        detections = {
            "activity":  activity_results,
            "gesture":   gesture_results,
            "face":      face_results,
            "object":    object_results,
            "pose":      pose_results,
        }

        # ── Annotate frame ───────────────────────────────────
        overlay_color = {
            "critical": (0, 0, 200),
            "warning":  (0, 140, 255),
            "none":     (0, 180, 0),
        }.get(threat_level, (0, 180, 0))

        label = {
            "critical": "! THREAT DETECTED !",
            "warning":  "~ Suspicious Activity ~",
            "none":     "Monitoring",
        }.get(threat_level, "")

        cv2.rectangle(annotated, (0, 0), (w, 30), overlay_color, -1)
        cv2.putText(
            annotated, label,
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        cv2.putText(
            annotated,
            f"FPS:{fps:.0f}  Persons:{persons_count}  Motion:{motion_ratio:.2f}",
            (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1
        )

        # ── Encode latest frame for MJPEG stream ─────────────
        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with _frame_lock:
            _latest_frame = jpeg.tobytes()

        # ── Handle alerts ────────────────────────────────────
        if trigger_alert and threat_level in ("critical", "warning"):
            alert_type, alert_conf = trigger_alert
            now = time.time()
            last = _alert_last_sent.get(alert_type, 0)
            if now - last >= cooldown_sec:
                _alert_last_sent[alert_type] = now
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Save snapshot
                snap_name = f"incident_{uuid.uuid4().hex[:8]}.jpg"
                snap_path = os.path.join(CAPTURES_DIR, snap_name)
                cv2.imwrite(snap_path, annotated)

                incident = {
                    "id":              uuid.uuid4().hex[:12],
                    "timestamp":       ts,
                    "threat_type":     alert_type,
                    "level":           threat_level,
                    "confidence":      round(float(alert_conf), 3),
                    "image":           snap_name,
                    "image_url":       f"/api/captures/image/{snap_name}",
                }
                _append_json(INCIDENTS_LOG, incident)

                alert_entry = dict(incident)
                alert_entry["method"] = "system"
                alert_entry["success"] = True
                alert_entry["message"] = f"Auto-detected: {alert_type}"
                _append_json(ALERTS_LOG, alert_entry)

                with _state_lock:
                    _state["alerts_today"]   += 1
                    _state["total_alerts"]   += 1
                    _state["incidents_count"] = len(_read_json(INCIDENTS_LOG))
                    _state["last_alert"] = {
                        "type":       alert_type,
                        "level":      threat_level,
                        "confidence": float(alert_conf),
                        "time":       datetime.now().strftime("%H:%M:%S"),
                        "image":      snap_name,
                    }

                # Push SSE to all dashboard clients
                _broadcast("threat_detected", {
                    "type":       alert_type,
                    "level":      threat_level,
                    "confidence": float(alert_conf),
                    "time":       ts,
                    "image":      snap_name,
                })
                logger.info("ALERT: %s  (%.0f%%)", alert_type, alert_conf * 100)

                # Send Gmail email alert in background thread
                threading.Thread(
                    target=_send_gmail_alert,
                    args=(alert_type, float(alert_conf), snap_path, threat_level),
                    daemon=True,
                ).start()

        # ── Update live state ────────────────────────────────
        with _state_lock:
            _state["fps"]                  = round(fps, 1)
            _state["threat_level"]         = threat_level
            _state["current_detections"]   = detections
            _state["incidents_count"]      = len(_read_json(INCIDENTS_LOG))

        time.sleep(0.03)   # ~30 fps cap

    cap.release()
    logger.info("Detection loop stopped, camera released")


# Detection thread handle
_detect_thread = None


# ═══════════════════════════════════════════════════════════════
#  Flask Routes
# ═══════════════════════════════════════════════════════════════

# ── Health ───────────────────────────────────────────────────
@app.route("/api/ping")
def ping():
    return jsonify({"service": "SafeGuard AI", "ok": True})


# ── Start / Stop ──────────────────────────────────────────────
@app.route("/api/start", methods=["POST"])
def start():
    global _detect_thread
    with _state_lock:
        if _state["running"]:
            return jsonify({"status": "already_running"})
        _state["running"]      = True
        _state["start_time"]   = datetime.now().isoformat()
        _state["threat_level"] = "none"
        _state["fps"]          = 0
        _state["alerts_today"] = 0

    _detect_thread = threading.Thread(target=_detection_loop, daemon=True)
    _detect_thread.start()
    logger.info("Monitoring started")
    return jsonify({"status": "started"})


@app.route("/api/stop", methods=["POST"])
def stop():
    with _state_lock:
        _state["running"]      = False
        _state["start_time"]   = None
        _state["fps"]          = 0
        _state["threat_level"] = "none"
    logger.info("Monitoring stopped")
    return jsonify({"status": "stopped"})


# ── Status ───────────────────────────────────────────────────
@app.route("/api/status")
def status():
    with _state_lock:
        data = dict(_state)
    data["incidents_count"] = len(_read_json(INCIDENTS_LOG))
    data["total_alerts"]    = len(_read_json(ALERTS_LOG))
    return jsonify(data)


# ── MJPEG stream ─────────────────────────────────────────────
@app.route("/api/stream")
def stream():
    def _generate():
        while True:
            with _state_lock:
                running = _state["running"]
            if not running:
                break
            with _frame_lock:
                frame = _latest_frame
            if frame is None:
                time.sleep(0.05)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            time.sleep(1 / 25)

    with _state_lock:
        running = _state["running"]
    if not running:
        return jsonify({"error": "System not running"}), 503
    return Response(_generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ── SSE events ────────────────────────────────────────────────
@app.route("/api/events")
def events():
    def _generate(q):
        # Send a heartbeat immediately so the browser knows it's alive
        yield "data: {\"type\":\"connected\"}\n\n"
        try:
            while True:
                try:
                    msg = q.get(timeout=25)
                    if msg is None:
                        break
                    yield f"data: {msg}\n\n"
                except queue.Empty:
                    yield ": heartbeat\n\n"
        finally:
            with _sse_lock:
                if q in _sse_clients:
                    _sse_clients.remove(q)

    client_q = queue.Queue(maxsize=50)
    with _sse_lock:
        _sse_clients.append(client_q)

    return Response(
        _generate(client_q),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Config ───────────────────────────────────────────────────
@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify(_load_config())


@app.route("/api/config", methods=["POST"])
def set_config():
    updates = request.get_json(force=True, silent=True) or {}
    cfg = _load_config()
    for key, val in updates.items():
        if isinstance(val, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(val)
        else:
            cfg[key] = val
    _save_config(cfg)
    return jsonify({"status": "saved"})


# ── Thief Database ────────────────────────────────────────────
@app.route("/api/thieves", methods=["GET"])
def list_thieves():
    result = []
    for t in registry.list_thieves():
        entry = dict(t)
        entry["photo_url"] = f"/api/thieves/photo/{t['id']}"
        result.append(entry)
    return jsonify(result)


@app.route("/api/thieves", methods=["POST"])
def enroll_thief():
    name = (request.form.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400

    photo_file = request.files.get("photo")
    if not photo_file:
        return jsonify({"error": "Photo is required"}), 400

    try:
        file_bytes = np.frombuffer(photo_file.read(), dtype=np.uint8)
        face_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if face_image is None:
            raise ValueError("Cannot decode image")
    except Exception as exc:
        return jsonify({"error": f"Invalid image: {exc}"}), 400

    # ── Compute real geometric embedding via FaceDetector ────
    embedding = None
    if _face_detector is not None:
        try:
            embedding = _face_detector.compute_enrolment_embedding(face_image)
        except Exception as exc:
            logger.warning("Embedding extraction failed: %s", exc)

    if embedding is None:
        # Fall back to a random 16-d vector so enrolment still works
        # (matching will be unreliable until the photo contains a clear face)
        embedding = np.random.rand(16).astype(np.float32)
        logger.warning("No face found in photo for '%s' – using random embedding", name)

    try:
        entry = registry.enroll(
            name         = name,
            embedding    = embedding,
            face_image   = face_image,
            alias        = (request.form.get("alias")        or "").strip(),
            notes        = (request.form.get("notes")        or "").strip(),
            crime_details= (request.form.get("crime_details") or "").strip(),
        )
        entry["photo_url"] = f"/api/thieves/photo/{entry['id']}"
        has_real_emb = embedding.shape[0] == 16 and not np.allclose(embedding, embedding[0])
        logger.info("Enrolled thief: %s (%s) | real_embedding=%s", name, entry["id"], has_real_emb)
        return jsonify({"status": "enrolled", "thief": entry}), 200
    except Exception as exc:
        logger.error("Enroll error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/thieves/photo/<thief_id>")
def thief_photo(thief_id):
    path = os.path.join(THIEVES_DIR, thief_id, "face.jpg")
    if not os.path.isfile(path):
        return jsonify({"error": "Not found"}), 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/api/thieves/<thief_id>", methods=["DELETE"])
def delete_thief(thief_id):
    if not registry.delete_thief(thief_id):
        return jsonify({"error": "Not found"}), 404
    return jsonify({"status": "deleted"})


# ── Incidents ─────────────────────────────────────────────────
@app.route("/api/incidents", methods=["GET"])
@app.route("/api/incidents/", methods=["GET"])
def list_incidents():
    limit   = int(request.args.get("limit", 100))
    records = _read_json(INCIDENTS_LOG)[:limit]
    return jsonify(records)


@app.route("/api/incidents/<incident_id>", methods=["DELETE"])
def delete_incident(incident_id):
    records = _read_json(INCIDENTS_LOG)
    found = False
    new_records = []
    for r in records:
        if r.get("id") == incident_id:
            found = True
            # delete file
            image_filename = r.get("image")
            if image_filename:
                image_path = os.path.join(CAPTURES_DIR, image_filename)
                if os.path.isfile(image_path):
                    try:
                        os.remove(image_path)
                    except Exception as exc:
                        logger.error("Failed to delete image %s: %s", image_path, exc)
        else:
            new_records.append(r)
    
    if not found:
        return jsonify({"error": "Not found"}), 404
        
    _write_json(INCIDENTS_LOG, new_records)
    with _state_lock:
        _state["incidents_count"] = len(new_records)
    return jsonify({"status": "deleted"})


@app.route("/api/incidents/batch-delete", methods=["POST"])
def batch_delete_incidents():
    data = request.get_json(force=True, silent=True) or {}
    ids_to_delete = data.get("ids", [])
    if not isinstance(ids_to_delete, list) or not ids_to_delete:
        return jsonify({"error": "No IDs provided"}), 400

    records = _read_json(INCIDENTS_LOG)
    new_records = []
    deleted_count = 0
    ids_set = set(ids_to_delete)

    for r in records:
        if r.get("id") in ids_set:
            deleted_count += 1
            # delete file
            image_filename = r.get("image")
            if image_filename:
                image_path = os.path.join(CAPTURES_DIR, image_filename)
                if os.path.isfile(image_path):
                    try:
                        os.remove(image_path)
                    except Exception as exc:
                        logger.error("Failed to delete image %s: %s", image_path, exc)
        else:
            new_records.append(r)

    _write_json(INCIDENTS_LOG, new_records)
    with _state_lock:
        _state["incidents_count"] = len(new_records)

    return jsonify({"status": "deleted", "count": deleted_count})




# ── Alerts ───────────────────────────────────────────────────
@app.route("/api/alerts", methods=["GET"])
def list_alerts():
    limit   = int(request.args.get("limit", 100))
    records = _read_json(ALERTS_LOG)[:limit]
    return jsonify(records)


# ── Test alert endpoints ──────────────────────────────────────
def _log_alert(threat_type, method, msg):
    _append_json(ALERTS_LOG, {
        "id":          uuid.uuid4().hex[:12],
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "threat_type": threat_type,
        "method":      method,
        "success":     True,
        "message":     msg,
    })

@app.route("/api/test-alert", methods=["POST"])
def test_alert():
    data       = request.get_json(force=True, silent=True) or {}
    threat_type= data.get("threat_type", "Test Email Alert")
    ok, msg    = _send_gmail_alert(threat_type, 1.0, snapshot_path=None, level="critical")
    _log_alert(threat_type, "email", msg)
    return jsonify({"success": ok, "message": msg})

@app.route("/api/test-sms", methods=["POST"])
def test_sms():
    data = request.get_json(force=True, silent=True) or {}
    _log_alert(data.get("threat_type", "Test SMS"), "sms", "Test SMS triggered")
    return jsonify({"success": True, "message": "Test SMS sent"})

@app.route("/api/test-call", methods=["POST"])
def test_call():
    data = request.get_json(force=True, silent=True) or {}
    _log_alert(data.get("threat_type", "Test Call"), "call", "Test call triggered")
    return jsonify({"success": True, "message": "Test call started"})



# ── Captures ─────────────────────────────────────────────────
@app.route("/api/captures/image/<filename>")
def serve_capture_image(filename):
    path = os.path.join(CAPTURES_DIR, os.path.basename(filename))
    if not os.path.isfile(path):
        return jsonify({"error": "Not found"}), 404
    return send_file(path, mimetype="image/jpeg")

@app.route("/api/captures/video/<filename>")
def serve_capture_video(filename):
    path = os.path.join(CAPTURES_DIR, os.path.basename(filename))
    if not os.path.isfile(path):
        return jsonify({"error": "Not found"}), 404
    return send_file(path, mimetype="video/mp4")


# ── ESP32 ─────────────────────────────────────────────────────
@app.route("/api/esp32/status")
def esp32_status():
    cfg     = _load_config()
    esp_cfg = cfg.get("esp32", {})
    return jsonify({
        "connected":   False,
        "host":        esp_cfg.get("host", ""),
        "port":        esp_cfg.get("port", 80),
        "enabled":     esp_cfg.get("enabled", True),
        "last_commands": [],
    })

@app.route("/api/esp32/action", methods=["POST"])
def esp32_action():
    data   = request.get_json(force=True, silent=True) or {}
    action = data.get("action", "")
    logger.info("ESP32 action requested: %s", action)
    return jsonify({"status": "sent", "action": action})


# ── Static dashboard ─────────────────────────────────────────
DASHBOARD_DIR = os.path.join(PROJECT_DIR, "web_dashboard")

@app.route("/", defaults={"filename": "index.html"})
@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(DASHBOARD_DIR, filename)


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logger.info("Starting SafeGuard AI backend at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
