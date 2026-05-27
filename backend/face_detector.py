"""
Face Detector – SafeGuard AI
Uses the MediaPipe Tasks API (face_landmarker) for:
  • Accurate face detection with 478-point landmarks
  • Multi-signal face covering detection (mask/scarf/hood/sunglasses)
  • Geometric embedding-based thief matching
  • Real-time annotated frame output
"""

import logging
import math
import os
import urllib.request

import cv2
import numpy as np

logger = logging.getLogger("FaceDetector")

# ── MediaPipe Tasks API ───────────────────────────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MP_AVAILABLE = True
except Exception as exc:
    logger.warning("MediaPipe Tasks unavailable: %s", exc)
    _MP_AVAILABLE = False

# ── Model URL & local path ────────────────────────────────────
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_NAME = "face_landmarker.task"

# ── Landmark indices (MediaPipe 478-point face mesh) ──────────
# Lower-face zone: nose tip, mouth, chin (hidden by mask/scarf)
_LOWER_IDS  = [1, 2, 98, 327, 61, 291, 13, 14, 78, 308, 152, 148, 377, 17, 200]
# Eye zone: covered by sunglasses
_EYE_IDS    = [33, 133, 159, 145, 362, 263, 386, 374, 70, 300, 105, 334]
# Forehead zone: covered by hood/hat
_FORE_IDS   = [10, 151, 108, 337, 9]

# 16 landmark-pair distances that form the geometric embedding
_EMB_PAIRS = [
    (33,  263),   # inter-eye distance (normalisation anchor)
    (1,   152),   # nose-chin
    (61,  291),   # mouth width
    (133, 362),   # inner-eye gap
    (70,  300),   # brow width
    (10,  152),   # face height (forehead-chin)
    (234, 454),   # face width (cheekbones)
    (1,   61),    # nose to left mouth corner
    (1,   291),   # nose to right mouth corner
    (33,  1),     # left eye to nose
    (263, 1),     # right eye to nose
    (159, 386),   # inter-eye vertical
    (13,  14),    # lip gap
    (105, 334),   # brow-to-brow
    (10,  1),     # forehead to nose
    (152, 200),   # chin depth
]


def _lm_xy(face_landmarks, idx, w, h):
    lm = face_landmarks[idx]
    return lm.x * w, lm.y * h


def _dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _geometric_embedding(face_landmarks, w, h):
    """16-dim L2-normalised geometric embedding from landmark distances."""
    eye_dist = _dist(
        _lm_xy(face_landmarks, 33, w, h),
        _lm_xy(face_landmarks, 263, w, h),
    )
    if eye_dist < 2.0:
        return None
    vec = []
    for a, b in _EMB_PAIRS:
        pa = _lm_xy(face_landmarks, a, w, h)
        pb = _lm_xy(face_landmarks, b, w, h)
        vec.append(_dist(pa, pb) / eye_dist)
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return arr / (norm + 1e-8)


def _cosine_sim(a, b):
    return float(np.clip(np.dot(a, b), 0.0, 1.0))


def _lm_visibility(face_landmarks, idx):
    """Return presence score for a landmark (0-1); Tasks API uses 'presence'."""
    lm = face_landmarks[idx]
    return getattr(lm, "presence", getattr(lm, "visibility", 1.0))


# ════════════════════════════════════════════════════════════
class FaceDetector:
    def __init__(
        self,
        match_threshold: float = 0.45,
        cover_threshold: float = 0.55,
        max_faces: int = 4,
    ):
        self.match_threshold = match_threshold
        self.cover_threshold = cover_threshold
        self.max_faces       = max_faces
        self._landmarker     = None
        self._init()

    # ── Init: download model + create landmarker ──────────────
    def _init(self):
        if not _MP_AVAILABLE:
            return
        model_dir  = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_dir, _MODEL_NAME)

        if not os.path.exists(model_path):
            logger.info("Downloading face_landmarker model...")
            try:
                urllib.request.urlretrieve(_MODEL_URL, model_path)
                logger.info("Face landmarker model downloaded.")
            except Exception as exc:
                logger.error("Failed to download face model: %s", exc)
                return

        try:
            base_opts = mp_python.BaseOptions(model_asset_path=model_path)
            opts = mp_vision.FaceLandmarkerOptions(
                base_options             = base_opts,
                running_mode             = mp_vision.RunningMode.IMAGE,
                num_faces                = self.max_faces,
                min_face_detection_confidence = 0.5,
                min_face_presence_confidence  = 0.5,
                min_tracking_confidence       = 0.5,
                output_face_blendshapes       = False,
                output_facial_transformation_matrixes = False,
            )
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
            logger.info("FaceLandmarker loaded (max_faces=%d)", self.max_faces)
        except Exception as exc:
            logger.error("FaceLandmarker init failed: %s", exc)
            self._landmarker = None

    def close(self):
        if self._landmarker:
            self._landmarker.close()

    # ── Face bounding box from landmarks ─────────────────────
    def _face_box(self, face_lm, w, h, pad=20):
        xs = [lm.x * w for lm in face_lm]
        ys = [lm.y * h for lm in face_lm]
        return (
            max(0, int(min(xs)) - pad),
            max(0, int(min(ys)) - pad),
            min(w, int(max(xs)) + pad),
            min(h, int(max(ys)) + pad),
        )

    # ── Face covering analysis ────────────────────────────────
    def _analyse_covering(self, face_lm, frame, face_box):
        h, w = frame.shape[:2]

        # Signal 1: landmark presence (low = landmark occluded)
        lower_pres = float(np.mean([_lm_visibility(face_lm, i) for i in _LOWER_IDS]))
        eye_pres   = float(np.mean([_lm_visibility(face_lm, i) for i in _EYE_IDS]))
        fore_pres  = float(np.mean([_lm_visibility(face_lm, i) for i in _FORE_IDS]))

        # Signal 2: skin-colour ratio in the lower-half of the face ROI
        x1, y1, x2, y2 = face_box
        face_h   = max(y2 - y1, 1)
        roi_y1   = y1 + int(face_h * 0.5)
        roi      = frame[max(roi_y1, 0):min(y2, h), max(x1, 0):min(x2, w)]

        skin_ratio = 0.5
        if roi.size > 0 and roi.shape[0] > 6 and roi.shape[1] > 6:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            m1  = cv2.inRange(hsv, np.array([0,  15,  60]), np.array([30, 170, 255]))
            m2  = cv2.inRange(hsv, np.array([160, 15, 60]), np.array([180, 170, 255]))
            skin_pixels = float(np.count_nonzero(cv2.bitwise_or(m1, m2)))
            total_pixels = roi.shape[0] * roi.shape[1] + 1e-6
            skin_ratio = min(1.0, skin_pixels / total_pixels)

        skin_cover = max(0.0, 1.0 - skin_ratio * 1.4)

        # Signal 3: texture uniformity (cloth/mask → low std-dev)
        uniformity = 0.0
        if roi.size > 0 and roi.shape[0] > 6 and roi.shape[1] > 6:
            gray_roi   = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            uniformity = max(0.0, 1.0 - float(np.std(gray_roi)) / 35.0)

        lower_cover = (
            (1.0 - lower_pres) * 0.35 +
            skin_cover          * 0.40 +
            uniformity          * 0.25
        )
        eye_cover  = max(0.0, 1.0 - eye_pres)
        fore_cover = max(0.0, 1.0 - fore_pres)

        overall = float(np.clip(max(lower_cover, eye_cover * 0.8), 0.0, 1.0))

        cover_type = None
        if overall >= self.cover_threshold:
            if lower_cover >= eye_cover:
                cover_type = "mask/scarf"
            elif eye_cover > fore_cover:
                cover_type = "sunglasses"
            else:
                cover_type = "hood/hat"

        return round(overall, 3), cover_type

    # ── Draw annotations ──────────────────────────────────────
    def _draw(self, frame, face_lm, face_box, cover_score, cover_type,
               match_name=None, match_conf=0.0):
        x1, y1, x2, y2 = face_box
        h_f, w_f = frame.shape[:2]

        if match_name:
            colour = (0, 0, 210)
        elif cover_score >= self.cover_threshold:
            colour = (0, 110, 255)
        else:
            colour = (60, 210, 80)

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        if match_name:
            label = f"THIEF: {match_name} ({match_conf:.0%})"
        elif cover_score >= self.cover_threshold:
            label = f"COVERED [{cover_type}] {cover_score:.0%}"
        else:
            label = f"Face {cover_score:.0%}"

        text_y = max(y1 - 4, 18)
        cv2.rectangle(frame, (x1, text_y - 18), (x2, text_y + 2), colour, -1)
        cv2.putText(frame, label, (x1 + 3, text_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        # Key landmark dots
        for idx in [1, 33, 263, 61, 291, 152]:
            lm = face_lm[idx]
            cx, cy = int(lm.x * w_f), int(lm.y * h_f)
            cv2.circle(frame, (cx, cy), 3, colour, -1)

    # ── Main detect ───────────────────────────────────────────
    def detect(self, frame: np.ndarray, thief_embeddings=None):
        """
        Parameters
        ----------
        frame            : BGR ndarray
        thief_embeddings : list of (entry_dict, np.ndarray) from ThiefRegistry

        Returns
        -------
        annotated_frame, result_dict
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        empty = {
            "faces_count": 0,
            "thief_match": {"detected": False, "confidence": 0.0, "name": None, "id": None},
            "face_cover":  {"detected": False, "confidence": 0.0, "cover_type": None},
            "faces": [],
        }

        if self._landmarker is None:
            return annotated, empty

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_img)

        if not result.face_landmarks:
            return annotated, empty

        thief_embs        = thief_embeddings or []
        best_cover_score  = 0.0
        best_cover_type   = None
        best_match_name   = None
        best_match_conf   = 0.0
        best_match_id     = None
        faces_info        = []

        for face_lm in result.face_landmarks:
            face_box   = self._face_box(face_lm, w, h)
            cover_score, cover_type = self._analyse_covering(face_lm, annotated, face_box)

            # Geometric embedding
            geo_emb    = _geometric_embedding(face_lm, w, h)
            match_name = None
            match_conf = 0.0
            match_id   = None

            if geo_emb is not None:
                for entry, stored_emb in thief_embs:
                    if stored_emb.shape[0] != geo_emb.shape[0]:
                        continue
                    sim = _cosine_sim(geo_emb, stored_emb)
                    if sim >= self.match_threshold and sim > match_conf:
                        match_conf = sim
                        match_name = entry.get("name", "Unknown")
                        match_id   = entry.get("id")

            self._draw(annotated, face_lm, face_box, cover_score, cover_type,
                       match_name, match_conf)

            faces_info.append({
                "box":         face_box,
                "cover_score": cover_score,
                "cover_type":  cover_type,
                "covered":     cover_score >= self.cover_threshold,
                "thief_match": match_name is not None,
                "match_name":  match_name,
                "match_conf":  round(match_conf, 3),
                "match_id":    match_id,
            })

            if cover_score > best_cover_score:
                best_cover_score = cover_score
                best_cover_type  = cover_type
            if match_conf > best_match_conf:
                best_match_conf = match_conf
                best_match_name = match_name
                best_match_id   = match_id

        # Summary overlay banners
        if best_cover_score >= self.cover_threshold:
            cv2.rectangle(annotated, (0, 31), (w, 55), (0, 80, 200), -1)
            cv2.putText(
                annotated,
                f"FACE COVERING [{best_cover_type}]  {best_cover_score:.0%}",
                (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
            )
        if best_match_name:
            cv2.rectangle(annotated, (0, 56), (w, 80), (0, 0, 180), -1)
            cv2.putText(
                annotated,
                f"KNOWN THIEF: {best_match_name}  ({best_match_conf:.0%})",
                (8, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
            )

        return annotated, {
            "faces_count": len(faces_info),
            "thief_match": {
                "detected":   best_match_name is not None,
                "confidence": round(best_match_conf, 3),
                "name":       best_match_name,
                "id":         best_match_id,
            },
            "face_cover": {
                "detected":   best_cover_score >= self.cover_threshold,
                "confidence": round(best_cover_score, 3),
                "cover_type": best_cover_type,
            },
            "faces": faces_info,
        }

    # ── Enrolment embedding ───────────────────────────────────
    def compute_enrolment_embedding(self, image: np.ndarray):
        """
        Extract 16-dim geometric embedding from an enrolment photo.
        Returns np.ndarray(16,) or None if no face found.
        """
        if self._landmarker is None:
            return None
        rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_img)
        if not result.face_landmarks:
            return None
        h, w = image.shape[:2]
        return _geometric_embedding(result.face_landmarks[0], w, h)
