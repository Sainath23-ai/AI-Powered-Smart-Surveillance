"""
Gesture Detector – SafeGuard AI
Uses the MediaPipe Tasks API (hand_landmarker) to detect hand gestures,
specifically the "Signal for Help" (Open Palm -> Closed SOS Fist).
"""

import logging
import math
import os
import urllib.request
import cv2
import numpy as np

logger = logging.getLogger("GestureDetector")

# ── MediaPipe Tasks API ───────────────────────────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MP_AVAILABLE = True
except Exception as exc:
    logger.warning("MediaPipe Tasks unavailable: %s", exc)
    _MP_AVAILABLE = False

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
_MODEL_NAME = "hand_landmarker.task"

# ── Hand Connections for Drawing ──────────────────────────────
_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17)                                 # Palm base
]

class GestureDetector:
    def __init__(self, confidence_threshold: float = 0.80):
        self.confidence_threshold = confidence_threshold
        self._landmarker = None
        self._init()

    def _init(self):
        if not _MP_AVAILABLE:
            return
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_dir, _MODEL_NAME)

        if not os.path.exists(model_path):
            logger.info("Downloading hand_landmarker model...")
            try:
                urllib.request.urlretrieve(_MODEL_URL, model_path)
                logger.info("Hand landmarker model downloaded.")
            except Exception as exc:
                logger.error("Failed to download hand model: %s", exc)
                return

        try:
            base_opts = mp_python.BaseOptions(model_asset_path=model_path)
            opts = mp_vision.HandLandmarkerOptions(
                base_options=base_opts,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = mp_vision.HandLandmarker.create_from_options(opts)
            logger.info("HandLandmarker loaded successfully")
        except Exception as exc:
            logger.error("HandLandmarker init failed: %s", exc)
            self._landmarker = None

    def close(self):
        if self._landmarker:
            self._landmarker.close()

    @staticmethod
    def _dist(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def detect(self, frame: np.ndarray):
        annotated = frame.copy()
        h, w = frame.shape[:2]

        empty = {
            "detected": False,
            "confidence": 0.0,
            "hands_detected": 0,
            "gesture_type": None,
        }

        if self._landmarker is None:
            return annotated, empty

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_img)

        if not result.hand_landmarks:
            return annotated, empty

        hands_count = len(result.hand_landmarks)
        best_gesture_type = None
        best_confidence = 0.0
        gesture_detected = False

        for hand_lms in result.hand_landmarks:
            # Get points
            pts = [(lm.x * w, lm.y * h) for lm in hand_lms]

            # Calculate palm size
            palm_size = self._dist(pts[0], pts[9])
            if palm_size < 5.0:
                continue

            # Check if fingers are extended
            # A finger is extended if its TIP is further from Wrist than its PIP joint
            index_extended = self._dist(pts[8], pts[0]) > self._dist(pts[6], pts[0])
            middle_extended = self._dist(pts[12], pts[0]) > self._dist(pts[10], pts[0])
            ring_extended = self._dist(pts[16], pts[0]) > self._dist(pts[14], pts[0])
            pinky_extended = self._dist(pts[20], pts[0]) > self._dist(pts[18], pts[0])

            # Check if thumb is tucked
            # Thumb tip is close to MCP joints (5, 9, 13, 17) if tucked
            dist_thumb_to_index_mcp = self._dist(pts[4], pts[5])
            dist_thumb_to_middle_mcp = self._dist(pts[4], pts[9])
            dist_thumb_to_ring_mcp = self._dist(pts[4], pts[13])

            thumb_tucked = (
                dist_thumb_to_index_mcp < 0.65 * palm_size or
                dist_thumb_to_middle_mcp < 0.65 * palm_size or
                dist_thumb_to_ring_mcp < 0.65 * palm_size
            )

            # Heuristics for the "Signal for Help" gestures:
            # 1. Open Palm (initial help request): all 4 fingers extended, thumb extended outwards
            open_palm = (
                index_extended and middle_extended and ring_extended and pinky_extended
                and (not thumb_tucked)
            )

            # 2. Closed SOS Fist: fingers closed (folded) over the tucked thumb
            sos_fist = (
                (not index_extended) and (not middle_extended) and
                (not ring_extended) and (not pinky_extended) and
                thumb_tucked
            )

            current_detected = False
            current_type = None
            current_conf = 0.0

            if sos_fist:
                current_detected = True
                current_type = "SOS Fist (Help Signal)"
                current_conf = 0.95
            elif open_palm:
                current_detected = True
                current_type = "Open Palm (Help Distress)"
                current_conf = 0.85

            if current_detected:
                gesture_detected = True
                if current_conf > best_confidence:
                    best_confidence = current_conf
                    best_gesture_type = current_type

            # Draw Hand Skeleton
            colour = (0, 0, 255) if current_detected else (0, 255, 0)
            thickness = 3 if current_detected else 2

            for start, end in _CONNECTIONS:
                p_start = (int(pts[start][0]), int(pts[start][1]))
                p_end = (int(pts[end][0]), int(pts[end][1]))
                cv2.line(annotated, p_start, p_end, colour, thickness)

            for pt in pts:
                cv2.circle(annotated, (int(pt[0]), int(pt[1])), 4, (255, 255, 255), -1)

            # Draw Bounding Box around hand
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, y1 = max(0, int(min(xs)) - 15), max(0, int(min(ys)) - 15)
            x2, y2 = min(w, int(max(xs)) + 15), min(h, int(max(ys)) + 15)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

            if current_detected:
                cv2.putText(
                    annotated,
                    f"HELP: {current_type}",
                    (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

        if gesture_detected and best_confidence >= self.confidence_threshold:
            cv2.rectangle(annotated, (0, 81), (w, 105), (200, 0, 200), -1)
            cv2.putText(
                annotated,
                f"🆘 SOS GESTURE: {best_gesture_type} ({best_confidence:.0%})",
                (8, 98),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        return annotated, {
            "detected": gesture_detected and best_confidence >= self.confidence_threshold,
            "confidence": round(best_confidence, 3),
            "hands_detected": hands_count,
            "gesture_type": best_gesture_type,
        }
