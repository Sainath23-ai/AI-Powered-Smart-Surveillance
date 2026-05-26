"""
Persist suspicious-activity snapshots and optional short video clips.
"""

import json
import os
import threading
import time
import uuid
from collections import deque
from datetime import datetime

import cv2

logger = None


def _log():
    global logger
    if logger is None:
        import logging
        logger = logging.getLogger("CaptureStore")
    return logger


class CaptureStore:
    """Saves images/videos and maintains an incident history index."""

    def __init__(self, base_dir: str = None):
        root = base_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data", "suspicious_captures"
        )
        self.root = os.path.abspath(root)
        self.images_dir = os.path.join(self.root, "images")
        self.videos_dir = os.path.join(self.root, "videos")
        self.index_path = os.path.join(self.root, "incidents.json")
        self._lock = threading.Lock()
        self._frame_buffer = deque(maxlen=90)  # ~3s at 30fps for clip context
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)
        if not os.path.exists(self.index_path):
            self._write_index([])

    def push_frame(self, frame):
        """Keep rolling buffer for optional video clip on incident."""
        if frame is not None:
            self._frame_buffer.append(frame.copy())

    def save_incident(
        self,
        threat_type: str,
        confidence: float,
        level: str,
        frame,
        save_video: bool = True,
    ) -> dict:
        """Save photo (and optional short video) for a suspicious event."""
        incident_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        image_name = f"{incident_id}.jpg"
        image_path = os.path.join(self.images_dir, image_name)

        cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 88])

        video_name = None
        if save_video and self._frame_buffer:
            video_name = f"{incident_id}.mp4"
            video_path = os.path.join(self.videos_dir, video_name)
            self._save_video(list(self._frame_buffer), video_path)

        entry = {
            "id": incident_id,
            "threat_type": threat_type,
            "confidence": round(float(confidence), 4),
            "level": level,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "iso_time": datetime.now().isoformat(),
            "image": image_name,
            "video": video_name,
        }

        with self._lock:
            incidents = self._read_index()
            incidents.insert(0, entry)
            incidents = incidents[:200]
            self._write_index(incidents)

        _log().info("Saved suspicious capture: %s (%s)", incident_id, threat_type)
        return entry

    def rebuild_index_from_disk(self):
        """Rebuild incidents.json from saved images (recovers after missing/corrupt index)."""
        if not os.path.isdir(self.images_dir):
            return []

        incidents = []
        names = sorted(
            (n for n in os.listdir(self.images_dir) if n.lower().endswith((".jpg", ".jpeg", ".png"))),
            reverse=True,
        )
        for name in names:
            incident_id = os.path.splitext(name)[0]
            path = os.path.join(self.images_dir, name)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = time.time()
            video_name = incident_id + ".mp4"
            video_path = os.path.join(self.videos_dir, video_name)
            incidents.append({
                "id": incident_id,
                "threat_type": "Suspicious Activity",
                "confidence": 0.0,
                "level": "warning",
                "timestamp": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "iso_time": datetime.fromtimestamp(mtime).isoformat(),
                "image": name,
                "video": video_name if os.path.isfile(video_path) else None,
            })

        with self._lock:
            self._write_index(incidents)
        _log().info("Rebuilt incidents index: %d entries", len(incidents))
        return incidents

    def list_incidents(self, limit: int = 50):
        with self._lock:
            incidents = self._read_index()

        if not incidents:
            incidents = self.rebuild_index_from_disk()
        else:
            # Drop entries whose image file was removed
            valid = []
            changed = False
            for item in incidents:
                image = item.get("image")
                if image and os.path.isfile(os.path.join(self.images_dir, image)):
                    valid.append(item)
                else:
                    changed = True
            if changed:
                with self._lock:
                    self._write_index(valid)
                incidents = valid

            if not incidents:
                incidents = self.rebuild_index_from_disk()

            # Pick up images on disk that are not in the index
            indexed = {item.get("image") for item in incidents}
            if os.path.isdir(self.images_dir):
                for name in os.listdir(self.images_dir):
                    if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    if name in indexed:
                        continue
                    incident_id = os.path.splitext(name)[0]
                    path = os.path.join(self.images_dir, name)
                    try:
                        mtime = os.path.getmtime(path)
                    except OSError:
                        mtime = time.time()
                    video_name = incident_id + ".mp4"
                    video_path = os.path.join(self.videos_dir, video_name)
                    incidents.append({
                        "id": incident_id,
                        "threat_type": "Suspicious Activity",
                        "confidence": 0.0,
                        "level": "warning",
                        "timestamp": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
                        "iso_time": datetime.fromtimestamp(mtime).isoformat(),
                        "image": name,
                        "video": video_name if os.path.isfile(video_path) else None,
                    })
                if len(incidents) > len(indexed):
                    with self._lock:
                        self._write_index(incidents)

        return incidents[:limit]

    def get_incident(self, incident_id: str):
        with self._lock:
            for item in self._read_index():
                if item.get("id") == incident_id:
                    return item
        return None

    def resolve_path(self, media_type: str, filename: str) -> str:
        if media_type == "video":
            return os.path.join(self.videos_dir, filename)
        return os.path.join(self.images_dir, filename)

    def _read_index(self):
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _write_index(self, incidents):
        os.makedirs(self.root, exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(incidents, f, indent=2)

    @staticmethod
    def _save_video(frames, path: str, fps: int = 20):
        if not frames:
            return
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
        if not writer.isOpened():
            _log().warning("Could not open video writer: %s", path)
            return
        for frame in frames:
            writer.write(frame)
        writer.release()
