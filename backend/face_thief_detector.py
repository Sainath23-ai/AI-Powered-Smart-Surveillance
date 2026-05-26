"""
Face detection, known-thief matching, and face-covering detection.
Uses OpenCV YuNet + SFace when available; MediaPipe Face Landmarker for cover analysis.
"""

import logging
import os
import urllib.request

import cv2
import numpy as np

from thief_registry import ThiefRegistry

logger = logging.getLogger("FaceThiefDetector")

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except Exception:
    mp = None
    mp_python = None
    mp_vision = None


class FaceThiefDetector:
    def __init__(
        self,
        registry: ThiefRegistry = None,
        match_threshold: float = 0.45,
        cover_threshold: float = 0.55,
    ):
        self.registry = registry or ThiefRegistry()
        self.match_threshold = match_threshold
        self.cover_threshold = cover_threshold
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(self.models_dir, exist_ok=True)

        self.face_detector = None
        self.face_recognizer = None
        self.landmarker = None
        self._init_opencv_models()
        self._init_landmarker()

    def _download(self, url: str, filename: str) -> str:
        path = os.path.join(self.models_dir, filename)
        if not os.path.exists(path):
            logger.info("Downloading %s ...", filename)
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as exc:
                logger.warning("Could not download %s: %s", filename, exc)
        return path

    def _init_opencv_models(self):
        yunet_path = self._download(YUNET_URL, "face_detection_yunet_2023mar.onnx")
        sface_path = self._download(SFACE_URL, "face_recognition_sface_2021dec.onnx")

        if not os.path.isfile(yunet_path) or not hasattr(cv2, "FaceDetectorYN"):
            logger.warning("YuNet face detector unavailable")
            return

        try:
            self.face_detector = cv2.FaceDetectorYN.create(
                yunet_path, "", (320, 320), 0.5, 0.3, 5000
            )
            if os.path.isfile(sface_path) and hasattr(cv2, "FaceRecognizerSF"):
                self.face_recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
        except Exception as exc:
            logger.warning("OpenCV face models failed: %s", exc)

    def _init_landmarker(self):
        if mp is None or mp_python is None or mp_vision is None:
            return
        model_path = self._download(FACE_LANDMARKER_URL, "face_landmarker.task")
        if not os.path.isfile(model_path):
            return
        try:
            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                num_faces=4,
                running_mode=mp_vision.RunningMode.IMAGE,
            )
            self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:
            logger.warning("Face landmarker unavailable: %s", exc)

    def reload_registry(self):
        """Call after enrolling new thieves."""
        pass

    def compute_embedding(self, frame: np.ndarray, face_row=None) -> np.ndarray:
        if self.face_recognizer is None:
            return None
        if face_row is None:
            faces = self._detect_faces_raw(frame)
            if faces is None or len(faces) == 0:
                return None
            face_row = faces[0]
        aligned = self.face_recognizer.alignCrop(frame, face_row)
        feature = self.face_recognizer.feature(aligned)
        return feature.flatten()

    def enroll_from_image(
        self, image: np.ndarray, name: str, alias="", notes="", crime_details=""
    ):
        faces = self._detect_faces_raw(image)
        if faces is None or len(faces) == 0:
            raise ValueError("No face found in image. Use a clear front-facing photo.")
        face_row = faces[0]
        emb = self.compute_embedding(image, face_row)
        if emb is None:
            raise ValueError("Face recognition model not available.")
        x, y, fw, fh = [int(v) for v in face_row[:4]]
        crop = image[y : y + fh, x : x + fw]
        return self.registry.enroll(name, emb, crop, alias, notes, crime_details)

    def _detect_faces_raw(self, frame):
        if self.face_detector is None:
            return None
        h, w = frame.shape[:2]
        self.face_detector.setInputSize((w, h))
        _, faces = self.face_detector.detect(frame)
        return faces

    def _detect_faces(self, frame):
        faces = self._detect_faces_raw(frame)
        if faces is None:
            return []
        return [tuple(f[:4]) for f in faces]

    def _match_thief(self, embedding: np.ndarray):
        best = None
        best_score = -1.0
        for entry, enrolled in self.registry.load_embeddings():
            if embedding.shape != enrolled.shape:
                continue
            if self.face_recognizer is not None:
                e1 = embedding.reshape(1, -1).astype(np.float32)
                e2 = enrolled.reshape(1, -1).astype(np.float32)
                score = float(
                    self.face_recognizer.match(
                        e1, e2, cv2.FaceRecognizerSF_FR_COSINE
                    )
                )
            else:
                score = float(
                    np.dot(embedding, enrolled)
                    / (np.linalg.norm(embedding) * np.linalg.norm(enrolled) + 1e-6)
                )
            if score > best_score:
                best_score = score
                best = entry
        if best and best_score >= self.match_threshold:
            return best, best_score
        return None, best_score

    def _face_covered_score(self, frame, box) -> float:
        """Higher score = more likely face is covered (mask/balaclava)."""
        x, y, fw, fh = [int(v) for v in box]
        h, w = frame.shape[:2]
        x, y = max(0, x), max(0, y)
        fw, fh = min(fw, w - x), min(fh, h - y)
        if fw < 20 or fh < 20:
            return 0.0

        roi = frame[y : y + fh, x : x + fw]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        upper = gray[: int(fh * 0.45), :]
        lower = gray[int(fh * 0.45) :, :]
        if upper.size == 0 or lower.size == 0:
            return 0.0

        upper_mean = np.mean(upper)
        lower_mean = np.mean(lower)
        lower_var = np.var(lower)
        upper_var = np.var(upper)

        darkness = max(0.0, (upper_mean - lower_mean) / 80.0)
        flatness = max(0.0, 1.0 - (lower_var / (upper_var + 1e-3)))
        score = min(1.0, darkness * 0.55 + flatness * 0.45)

        if self.landmarker is not None:
            try:
                rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.landmarker.detect(mp_image)
                if result.face_landmarks:
                    lm = result.face_landmarks[0]
                    mouth_idxs = [13, 14, 17, 61, 291, 0]
                    mouth_y = np.mean([lm[i].y for i in mouth_idxs if i < len(lm)])
                    nose_y = lm[1].y if len(lm) > 1 else 0.4
                    if mouth_y < nose_y + 0.02:
                        score = min(1.0, score + 0.15)
            except Exception:
                pass

        return float(score)

    def detect(self, frame: np.ndarray):
        raw_faces = self._detect_faces_raw(frame)
        face_rows = list(raw_faces) if raw_faces is not None else []
        faces = [tuple(f[:4]) for f in face_rows]

        thief_match = None
        thief_score = 0.0
        face_covered = False
        cover_score = 0.0
        max_cover = 0.0

        annotated = frame.copy()
        for i, box in enumerate(faces):
            face_row = face_rows[i] if i < len(face_rows) else None
            x, y, fw, fh = [int(v) for v in box]
            emb = self.compute_embedding(frame, face_row) if face_row is not None else None
            label = "Face"
            color = (0, 200, 120)

            if emb is not None:
                match, score = self._match_thief(emb)
                if match:
                    thief_match = match
                    thief_score = score
                    label = f"THIEF: {match.get('name', 'Unknown')}"
                    color = (0, 0, 255)
                else:
                    thief_score = max(thief_score, score)

            cs = self._face_covered_score(frame, box)
            max_cover = max(max_cover, cs)
            if cs >= self.cover_threshold:
                face_covered = True
                cover_score = max(cover_score, cs)
                if not thief_match:
                    label = "Face Covered"
                    color = (0, 140, 255)

            cv2.rectangle(annotated, (x, y), (x + fw, y + fh), color, 2)
            cv2.putText(
                annotated,
                label,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        detected_thief = thief_match is not None
        results = {
            "faces_count": len(faces),
            "thief_match": {
                "detected": detected_thief,
                "confidence": float(thief_score) if detected_thief else 0.0,
                "name": thief_match.get("name") if thief_match else None,
                "id": thief_match.get("id") if thief_match else None,
                "alias": thief_match.get("alias") if thief_match else None,
            },
            "face_cover": {
                "detected": face_covered,
                "confidence": float(max_cover),
            },
            "models_ready": self.face_detector is not None,
        }
        return annotated, results
