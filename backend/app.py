"""
SafeGuard AI - Main Flask Application Server
Provides:
  - REST API for configuration and status
  - MJPEG video stream endpoint
  - WebSocket-like SSE for real-time events
  - Video upload + AI analysis endpoint
  - Web dashboard serving
"""
from hf_detector import HuggingFaceDetector
import cv2
import json
import time
import numpy as np
import threading
import logging
import os
from datetime import datetime
from flask import (Flask, Response, request, jsonify,
                   send_from_directory, stream_with_context)
from flask_cors import CORS

from config import load_config, save_config
from detector import ActivityDetector
from pose_safety_detector import PoseSafetyDetector
from gesture_detector import GestureDetector
from alert_system import AlertSystem
from esp32_comm import ESP32Controller
from capture_store import CaptureStore
from thief_registry import ThiefRegistry
from face_thief_detector import FaceThiefDetector
from object_detector import ObjectDetector

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SafeGuardAI")

# ── Flask App ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.join(BASE_DIR, '..', 'web_dashboard')

app = Flask(__name__, static_folder=DASHBOARD_DIR)
CORS(app)

# ── Global State ───────────────────────────────────────────────────────────────
config = load_config()
alert_system = AlertSystem(config)
esp32 = ESP32Controller(config)
capture_store = CaptureStore()
thief_registry = ThiefRegistry()

activity_detector = None
pose_detector = None
gesture_detector = None
face_thief_detector = None
object_detector = None
camera = None
hf_detector = None
camera_lock = threading.Lock()

system_state = {
    "running": False,
    "camera_active": False,
    "threat_level": "none",
    "current_detections": {},
    "frame_count": 0,
    "fps": 0,
    "start_time": None,
    "last_alert": None,
    "alerts_today": 0,
    "total_alerts": 0
}
state_lock = threading.Lock()

# SSE event queue
sse_events = []
sse_lock = threading.Lock()

# Background surveillance + stream frame buffer
surveillance_thread = None
stream_frame_lock = threading.Lock()
latest_jpeg_frame = None
last_threat_notify_time = {}  # threat_type -> timestamp


def push_event(event_type: str, data: dict):
    """Push event to SSE clients."""
    with sse_lock:
        sse_events.append({
            "type": event_type,
            "data": data,
            "time": time.time()
        })
        # Keep only last 100 events
        if len(sse_events) > 100:
            sse_events.pop(0)


def init_detectors():
    """Initialize AI detection models."""
    global activity_detector, pose_detector, gesture_detector
    global face_thief_detector, object_detector
    cfg = config.get("detection", {})
    activity_detector = ActivityDetector(
        violence_threshold=cfg.get("violence_threshold", 0.75),
        sensitivity=cfg.get("suspicious_sensitivity", "medium")
    )
    pose_detector = PoseSafetyDetector(
        threshold=cfg.get("pose_safety_threshold", 0.68),
        sensitivity=cfg.get("suspicious_sensitivity", "medium")
    )
    gesture_detector = GestureDetector(
        confidence_threshold=cfg.get("gesture_confidence", 0.80)
    )
    face_thief_detector = FaceThiefDetector(
        registry=thief_registry,
        match_threshold=cfg.get("face_match_threshold", 0.45),
        cover_threshold=cfg.get("face_cover_threshold", 0.55),
    )
    object_detector = ObjectDetector(
        confidence=cfg.get("object_confidence", 0.45)
    )
    logger.info("Detectors initialized")
    global hf_detector
    hf_cfg = config.get("huggingface", {})
    if hf_cfg.get("api_key") and hf_cfg.get("api_key") != "YOUR_HUGGINGFACE_API_KEY":
        hf_detector = HuggingFaceDetector(hf_cfg["api_key"], hf_cfg["model_url"])
    else:
        hf_detector = None



def open_camera():
    """Open the camera capture."""
    global camera
    cam_cfg = config.get("camera", {})
    src = cam_cfg.get("source", 0)
    # Allow numeric or string (RTSP URL)
    try:
        src = int(src)
    except (ValueError, TypeError):
        pass

    with camera_lock:
        if camera and camera.isOpened():
            camera.release()
        try:
            if isinstance(src, int) and os.name == "nt":
                camera = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            else:
                camera = cv2.VideoCapture(src)

            ok = camera.isOpened()
            if ok:
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 640))
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 480))
                camera.set(cv2.CAP_PROP_FPS, cam_cfg.get("fps", 30))
            else:
                camera.release()
                camera = None
        except Exception as exc:
            logger.exception(f"Camera startup failed (src={src}): {exc}")
            if camera:
                camera.release()
            camera = None
            ok = False

    with state_lock:
        system_state["camera_active"] = ok

    logger.info(f"Camera {'opened' if ok else 'FAILED'} (src={src})")
    return ok


def _threat_notify_allowed(threat_type: str) -> bool:
    """Respect alert cooldown for captures, SSE, and dashboard counters."""
    cooldown = config.get("detection", {}).get("alert_cooldown_seconds", 30)
    last = last_threat_notify_time.get(threat_type, 0)
    return (time.time() - last) >= cooldown


def _mark_threat_notified(threat_type: str):
    last_threat_notify_time[threat_type] = time.time()


def process_detections(frame, act_results, pose_results, gest_results, face_results, obj_results):
    """Handle detection results: update state, fire alerts, notify ESP32."""
    threat_type = None
    confidence = 0.0
    threat_level = "none"

    thief = (face_results or {}).get("thief_match", {})
    if thief.get("detected"):
        threat_type = f"Known Thief: {thief.get('name', 'Unknown')}"
        confidence = thief.get("confidence", 0.0)
        threat_level = "critical"

    cover = (face_results or {}).get("face_cover", {})
    if cover.get("detected") and threat_level != "critical":
        threat_type = "Face Covered / Masked"
        confidence = cover.get("confidence", 0.0)
        threat_level = "warning"

    sharp = (obj_results or {}).get("sharp_object", {})
    if sharp.get("detected"):
        label = sharp.get("label") or "sharp object"
        sharp_level = "critical"
        if threat_level != "critical" or sharp.get("confidence", 0) > confidence:
            threat_type = f"Sharp Object: {label.title()}"
            confidence = sharp.get("confidence", 0.0)
            threat_level = sharp_level

    if act_results["violence"]["detected"]:
        if threat_level != "critical":
            threat_type = "Violence Detected"
            confidence = act_results["violence"]["confidence"]
            threat_level = "critical"
    elif act_results["loitering"]["detected"] and threat_level == "none":
        threat_type = "Suspicious Loitering"
        confidence = act_results["loitering"]["confidence"]
        threat_level = "warning"
    elif act_results["running_panic"]["detected"] and threat_level == "none":
        threat_type = "Panic / Running"
        confidence = act_results["running_panic"]["confidence"]
        threat_level = "warning"

    pose_top = (pose_results or {}).get("top_scenario")
    if pose_top and pose_top.get("detected"):
        pose_level = (
            "critical"
            if pose_top["key"] in ("fighting", "kidnapping", "child_fall", "weapon_carry")
            else "warning"
        )
        if threat_level == "none" or (pose_level == "critical" and threat_level != "critical"):
            threat_type = f"Pose Alert: {pose_top['label']}"
            confidence = pose_top["confidence"]
            threat_level = pose_level

    if gest_results["detected"]:
        threat_type = f"SOS Gesture: {gest_results['gesture_name']}"
        confidence = gest_results["confidence"]
        threat_level = "critical"

    with state_lock:
        system_state["threat_level"] = threat_level
        system_state["current_detections"] = {
            "activity": act_results,
            "pose": pose_results,
            "gesture": gest_results,
            "face": face_results,
            "object": obj_results,
        }
        system_state["frame_count"] += 1

    if threat_type and threat_level in ("warning", "critical"):
        with state_lock:
            system_state["last_alert"] = {
                "type": threat_type,
                "confidence": confidence,
                "level": threat_level,
                "time": datetime.now().strftime("%H:%M:%S"),
            }

        if _threat_notify_allowed(threat_type):
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            snapshot = buf.tobytes()

            incident = capture_store.save_incident(
                threat_type, confidence, threat_level, frame
            )
            alert_system.trigger_alert(threat_type, confidence, snapshot)
            esp32.handle_threat(threat_type, threat_level, confidence)
            _mark_threat_notified(threat_type)

            with state_lock:
                system_state["total_alerts"] += 1
                system_state["alerts_today"] += 1

            push_event(
                "threat_detected",
                {
                    "type": threat_type,
                    "confidence": confidence,
                    "level": threat_level,
                    "timestamp": datetime.now().isoformat(),
                    "incident_id": incident.get("id"),
                    "image": incident.get("image"),
                },
            )


def surveillance_loop():
    """Background AI loop — runs whenever the system is started (not tied to stream viewers)."""
    global latest_jpeg_frame

    fps_counter = 0
    fps_time = time.time()

    while system_state.get("running"):
        with camera_lock:
            if not camera or not camera.isOpened():
                time.sleep(0.1)
                continue
            ret, frame = camera.read()

        if not ret:
            time.sleep(0.05)
            continue

        try:
            act_frame, act_results = activity_detector.detect(frame.copy())
            pose_frame, pose_results = pose_detector.detect(act_frame, act_results)
            gest_frame, gest_results = gesture_detector.detect(pose_frame)
            face_frame, face_results = face_thief_detector.detect(gest_frame) if face_thief_detector else (gest_frame, {})
            obj_frame, obj_results = object_detector.detect(face_frame) if object_detector else (face_frame, {})
            annotated = obj_frame
        except Exception as e:
            logger.error(f"Detection error: {e}")
            time.sleep(0.05)
            continue

        capture_store.push_frame(annotated)
        process_detections(
            frame, act_results, pose_results, gest_results, face_results, obj_results
        )

        fps_counter += 1
        now = time.time()
        if now - fps_time >= 1.0:
            with state_lock:
                system_state["fps"] = fps_counter
            fps_counter = 0
            fps_time = now

        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(
            annotated,
            ts,
            (10, annotated.shape[0] - 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        ret2, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ret2:
            with stream_frame_lock:
                latest_jpeg_frame = buffer.tobytes()

        time.sleep(0.001)
        
        # Hugging Face API call (every 30 frames)
        if fps_counter % 30 == 0 and hf_detector: 
            _, hf_results = hf_detector.detect(frame.copy())
            print("API Results:", hf_results)



def start_surveillance_thread():
    """Start a fresh background detection thread (safe after stop/start cycles)."""
    global surveillance_thread
    if surveillance_thread and surveillance_thread.is_alive():
        logger.info("Surveillance thread already running")
        return
    surveillance_thread = threading.Thread(target=surveillance_loop, daemon=True)
    surveillance_thread.start()
    logger.info("Surveillance background thread started")


def generate_frames():
    """MJPEG stream — serves the latest annotated frame from the background loop."""
    while system_state.get("running"):
        with stream_frame_lock:
            jpeg = latest_jpeg_frame

        if jpeg:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
        else:
            time.sleep(0.05)


# ═══════════════════════════════════════════════════════════════════════════════
#  API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return send_from_directory(DASHBOARD_DIR, 'index.html')


@app.route('/api/stream')
def video_stream():
    """MJPEG live video stream."""
    if not system_state["running"]:
        return jsonify({"error": "System not running"}), 503
    return Response(
        stream_with_context(generate_frames()),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/status')
def get_status():
    """Get full system status."""
    with state_lock:
        state = dict(system_state)
    esp_status = esp32.get_status()
    return jsonify({
        **state,
        "esp32": esp_status,
        "alert_log": alert_system.get_alert_log(20),
        "incidents_count": len(capture_store.list_incidents(200)),
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/ping')
def api_ping():
    """Health check endpoint used by the dashboard."""
    return jsonify({
        "ok": True,
        "service": "SafeGuard AI",
        "team": "APEX",
        "time": datetime.now().isoformat()
    })


@app.route('/api/start', methods=['POST'])
def start_system():
    """Start the surveillance system."""
    global system_state, latest_jpeg_frame, surveillance_thread
    if system_state["running"]:
        return jsonify({"status": "already_running"})

    init_detectors()
    if activity_detector and hasattr(activity_detector, "reset_state"):
        activity_detector.reset_state()
    cam_ok = open_camera()

    with stream_frame_lock:
        latest_jpeg_frame = None
    surveillance_thread = None

    with state_lock:
        system_state["running"] = True
        system_state["start_time"] = datetime.now().isoformat()
        system_state["camera_active"] = cam_ok
        system_state["threat_level"] = "none"
        system_state["current_detections"] = {}
        system_state["fps"] = 0
        system_state["frame_count"] = 0

    if cam_ok:
        start_surveillance_thread()

    push_event("system_started", {"camera": cam_ok})
    logger.info("Surveillance system STARTED")
    return jsonify({"status": "started", "camera": cam_ok})


@app.route('/api/stop', methods=['POST'])
def stop_system():
    """Stop the surveillance system."""
    global camera, surveillance_thread, latest_jpeg_frame

    with state_lock:
        system_state["running"] = False
        system_state["camera_active"] = False
        system_state["threat_level"] = "none"
        system_state["fps"] = 0
        system_state["current_detections"] = {}

    with camera_lock:
        if camera:
            try:
                camera.release()
            except Exception:
                pass
            camera = None

    with stream_frame_lock:
        latest_jpeg_frame = None
    surveillance_thread = None

    esp32.silence_alarm()
    esp32.set_led_status("off")
    push_event("system_stopped", {})
    logger.info("Surveillance system STOPPED")
    return jsonify({"status": "stopped"})


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration (passwords masked)."""
    cfg = load_config()
    safe_cfg = json.loads(json.dumps(cfg))
    # Mask sensitive fields
    if safe_cfg.get("gmail", {}).get("sender_password"):
        safe_cfg["gmail"]["sender_password"] = "••••••••••••••••"
    if safe_cfg.get("phone", {}).get("twilio_auth_token"):
        safe_cfg["phone"]["twilio_auth_token"] = "••••••••••••••••"
    return jsonify(safe_cfg)


@app.route('/api/config', methods=['POST'])
def update_config():
    """Update system configuration."""
    global config
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400

    current = load_config()

    # Update only provided sections (preserve existing passwords if masked)
    for section in data:
        if section not in current:
            current[section] = {}
        for key, val in data[section].items():
            if val and "••" not in str(val):  # Don't save masked passwords
                current[section][key] = val

    save_config(current)
    config = current
    alert_system.update_config(config)
    esp32.update_config(config)
    det = config.get("detection", {})
    if face_thief_detector:
        face_thief_detector.match_threshold = det.get("face_match_threshold", 0.45)
        face_thief_detector.cover_threshold = det.get("face_cover_threshold", 0.55)
    if object_detector:
        object_detector.confidence = det.get("object_confidence", 0.45)
    return jsonify({"status": "saved"})


@app.route('/api/esp32/action', methods=['POST'])
def esp32_action():
    """Send manual command to ESP32."""
    data = request.get_json()
    action = data.get("action", "")
    if action == "alarm":
        ok = esp32.trigger_alarm("warning", "Manual Test")
    elif action == "silence":
        ok = esp32.silence_alarm()
    elif action == "ping":
        ok = esp32.ping()
    elif action == "led":
        ok = esp32.set_led_status(data.get("status", "normal"))
    else:
        return jsonify({"error": "Unknown action"}), 400
    return jsonify({"success": ok, "status": esp32.get_status()})


@app.route('/api/esp32/status')
def esp32_status():
    connected = esp32.ping()
    return jsonify({**esp32.get_status(), "connected": connected})


@app.route('/api/alerts')
def get_alerts():
    """Get alert history."""
    limit = request.args.get("limit", 50, type=int)
    return jsonify(alert_system.get_alert_log(limit))


@app.route('/api/incidents')
def get_incidents():
    """Suspicious activity history with capture metadata."""
    limit = request.args.get("limit", 50, type=int)
    try:
        incidents = capture_store.list_incidents(limit)
        payload = []
        for inc in incidents:
            row = dict(inc)
            if row.get("image"):
                row["image_url"] = f"/api/captures/image/{row['image']}"
            if row.get("video"):
                row["video_url"] = f"/api/captures/video/{row['video']}"
            payload.append(row)
        return jsonify(payload)
    except Exception as exc:
        logger.exception("Failed to list incidents")
        return jsonify({"error": str(exc), "incidents": []}), 500


@app.route('/api/captures/<media_type>/<path:filename>')
def serve_capture(media_type, filename):
    """Serve saved suspicious-activity photos or videos."""
    if media_type not in ("image", "video"):
        return jsonify({"error": "Invalid media type"}), 400
    if ".." in filename or filename.startswith("/"):
        return jsonify({"error": "Invalid filename"}), 400

    path = capture_store.resolve_path(media_type, filename)
    if not os.path.isfile(path):
        return jsonify({"error": "Not found"}), 404

    folder = capture_store.images_dir if media_type == "image" else capture_store.videos_dir
    return send_from_directory(folder, filename)


@app.route('/api/events')
def sse_stream():
    """Server-Sent Events for real-time dashboard updates."""
    def event_generator():
        with sse_lock:
            client_index = len(sse_events)
        yield "data: {\"type\":\"connected\"}\n\n"
        while True:
            with sse_lock:
                current_len = len(sse_events)
                batch = list(sse_events[client_index:current_len])
                client_index = current_len
            for ev in batch:
                yield f"data: {json.dumps(ev)}\n\n"
            time.sleep(0.25)

    return Response(
        stream_with_context(event_generator()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        },
    )


@app.route('/api/thieves', methods=['GET'])
def list_thieves():
    thieves = thief_registry.list_thieves()
    for t in thieves:
        if t.get("photo"):
            t["photo_url"] = f"/api/thieves/photo/{t['id']}"
    return jsonify(thieves)


@app.route('/api/thieves', methods=['POST'])
def enroll_thief():
    """Enroll thief from uploaded photo + metadata."""
    global face_thief_detector
    if face_thief_detector is None:
        init_detectors()

    name = (request.form.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400

    file = request.files.get("photo")
    if not file:
        return jsonify({"error": "Photo is required"}), 400

    buf = np.frombuffer(file.read(), dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Invalid image file"}), 400

    try:
        entry = face_thief_detector.enroll_from_image(
            image,
            name=name,
            alias=request.form.get("alias", ""),
            notes=request.form.get("notes", ""),
            crime_details=request.form.get("crime_details", ""),
        )
        entry["photo_url"] = f"/api/thieves/photo/{entry['id']}"
        push_event("thief_enrolled", {"id": entry["id"], "name": entry["name"]})
        return jsonify({"success": True, "thief": entry})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Thief enroll failed")
        return jsonify({"error": str(exc)}), 500


@app.route('/api/thieves/<thief_id>', methods=['DELETE'])
def delete_thief(thief_id):
    ok = thief_registry.delete_thief(thief_id)
    if not ok:
        return jsonify({"error": "Thief not found"}), 404
    return jsonify({"success": True})


@app.route('/api/thieves/photo/<thief_id>')
def thief_photo(thief_id):
    entry = thief_registry.get_thief(thief_id)
    if not entry or not entry.get("photo"):
        return jsonify({"error": "Not found"}), 404
    path = thief_registry.photo_path(entry["photo"])
    if not os.path.isfile(path):
        return jsonify({"error": "Photo missing"}), 404
    return send_from_directory(os.path.dirname(path), os.path.basename(path))


@app.route('/api/test-alert', methods=['POST'])
def test_alert():
    """Send a test alert to verify configuration."""
    data = request.get_json() or {}
    threat = data.get("threat_type", "Test Alert")
    result = alert_system.send_email_alert(threat, 0.99, None, "Test Location")
    return jsonify(result)
 
 
 
 
 
 
@app.route('/api/test-sms', methods=['POST'])
def test_sms():
    """Send a Twilio SMS test alert (returns alert entry)."""
    data = request.get_json() or {}
    threat = data.get("threat_type", "Test SMS Alert")
    entry = alert_system.send_sms_alert(threat, 0.99, "Test Location")
    return jsonify(entry)


@app.route('/api/test-call', methods=['POST'])
def test_call():
    """Send a Twilio voice call test alert (returns alert entry)."""
    data = request.get_json() or {}
    threat = data.get("threat_type", "Test Call Alert")
    entry = alert_system.make_phone_call(threat, 0.99)
    return jsonify(entry)


@app.route('/css/<path:filename>')
def static_css(filename):
    return send_from_directory(os.path.join(DASHBOARD_DIR, 'css'), filename)


@app.route('/js/<path:filename>')
def static_js(filename):
    return send_from_directory(os.path.join(DASHBOARD_DIR, 'js'), filename)


@app.route('/<path:filename>')
def static_files(filename):
    """Dashboard assets only — registered last so /api/* is never shadowed."""
    if filename == 'api' or filename.startswith('api/'):
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(DASHBOARD_DIR, filename)


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("  SafeGuard AI Surveillance System")
    logger.info("  Made by Team APEX")
    logger.info("  Dashboard: http://127.0.0.1:5000")
    logger.info("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
