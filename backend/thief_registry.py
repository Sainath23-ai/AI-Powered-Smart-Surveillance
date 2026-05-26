"""
Store and match enrolled thief face profiles.
"""

import json
import os
import threading
import uuid
from datetime import datetime

import cv2
import numpy as np

REGISTRY_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "thieves"
)


class ThiefRegistry:
    def __init__(self, base_dir: str = None):
        self.root = os.path.abspath(base_dir or REGISTRY_DIR)
        self.index_path = os.path.join(self.root, "registry.json")
        self._lock = threading.Lock()
        os.makedirs(self.root, exist_ok=True)
        if not os.path.exists(self.index_path):
            self._write([])

    def list_thieves(self):
        with self._lock:
            return self._read()

    def get_thief(self, thief_id: str):
        for entry in self.list_thieves():
            if entry.get("id") == thief_id:
                return entry
        return None

    def enroll(
        self,
        name: str,
        embedding: np.ndarray,
        face_image: np.ndarray,
        alias: str = "",
        notes: str = "",
        crime_details: str = "",
        thief_id: str = None,
    ):
        thief_id = thief_id or uuid.uuid4().hex[:12]
        person_dir = os.path.join(self.root, thief_id)
        os.makedirs(person_dir, exist_ok=True)

        face_path = os.path.join(person_dir, "face.jpg")
        emb_path = os.path.join(person_dir, "embedding.npy")
        cv2.imwrite(face_path, face_image)
        np.save(emb_path, embedding.astype(np.float32))

        entry = {
            "id": thief_id,
            "name": name.strip(),
            "alias": alias.strip(),
            "notes": notes.strip(),
            "crime_details": crime_details.strip(),
            "photo": f"{thief_id}/face.jpg",
            "enrolled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        with self._lock:
            records = self._read()
            records = [r for r in records if r.get("id") != thief_id]
            records.insert(0, entry)
            self._write(records)

        return entry

    def load_embeddings(self):
        """Return list of (entry, embedding vector)."""
        pairs = []
        for entry in self.list_thieves():
            emb_path = os.path.join(self.root, entry["id"], "embedding.npy")
            if os.path.isfile(emb_path):
                pairs.append((entry, np.load(emb_path)))
        return pairs

    def delete_thief(self, thief_id: str) -> bool:
        with self._lock:
            records = self._read()
            new_records = [r for r in records if r.get("id") != thief_id]
            if len(new_records) == len(records):
                return False
            self._write(new_records)

        person_dir = os.path.join(self.root, thief_id)
        if os.path.isdir(person_dir):
            for fn in os.listdir(person_dir):
                try:
                    os.remove(os.path.join(person_dir, fn))
                except OSError:
                    pass
            try:
                os.rmdir(person_dir)
            except OSError:
                pass
        return True

    def photo_path(self, relative: str) -> str:
        return os.path.join(self.root, relative.replace("\\", "/"))

    def _read(self):
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _write(self, records):
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
