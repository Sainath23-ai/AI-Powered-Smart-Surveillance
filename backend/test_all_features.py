"""
SafeGuard AI - Comprehensive Feature Test Suite
Tests every major backend module and reports PASS / FAIL per feature.
Run with: .venv/Scripts/python.exe backend/test_all_features.py
"""

import logging
import os
import shutil
import sys
import tempfile

# Suppress MediaPipe / TFLite noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.disable(logging.WARNING)

import cv2
import numpy as np

# Re-enable only our own output
logging.disable(logging.NOTSET)
logging.basicConfig(level=logging.ERROR, format="%(name)s: %(message)s")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
results = []


def check(label, ok, detail=""):
    tag = PASS if ok else FAIL
    line = f"{tag}  {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    results.append((label, ok))


def section(title):
    print(f"\n{'-'*55}")
    print(f"  {title}")
    print(f"{'-'*55}")


# ─────────────────────────────────────────────────────────────
# 1. PACKAGE IMPORTS
# ─────────────────────────────────────────────────────────────
section("1. Package Imports")

try:
    import cv2 as _cv
    check("opencv-python (cv2)", True, f"v{_cv.__version__}")
except Exception as e:
    check("opencv-python (cv2)", False, str(e))

try:
    import numpy as _np
    check("numpy", True, f"v{_np.__version__}")
except Exception as e:
    check("numpy", False, str(e))

try:
    import mediapipe as _mp
    check("mediapipe", True, f"v{_mp.__version__}")
except Exception as e:
    check("mediapipe", False, str(e))

try:
    from mediapipe.tasks import python as _mpp
    from mediapipe.tasks.python import vision as _mpv
    check("mediapipe.tasks API (HandLandmarker, FaceLandmarker)", True)
except Exception as e:
    check("mediapipe.tasks API", False, str(e))

try:
    from ultralytics import YOLO as _YOLO
    check("ultralytics (YOLOv8)", True)
except Exception as e:
    check("ultralytics (YOLOv8)", False, str(e))

try:
    from flask import Flask as _Flask
    from flask_cors import CORS as _CORS
    check("flask + flask-cors", True)
except Exception as e:
    check("flask + flask-cors", False, str(e))

# ─────────────────────────────────────────────────────────────
# 2. MODULE SYNTAX & IMPORT
# ─────────────────────────────────────────────────────────────
section("2. Backend Module Imports")

modules = [
    ("face_detector",       "face_detector"),
    ("gesture_detector",    "gesture_detector"),
    ("object_detector",     "object_detector"),
    ("pose_safety_detector","pose_safety_detector"),
    ("thief_registry",      "thief_registry"),
    ("hf_detector",         "hf_detector"),
]
imported = {}
for label, mod in modules:
    try:
        m = __import__(mod)
        imported[mod] = m
        check(f"import {label}", True)
    except Exception as e:
        imported[mod] = None
        check(f"import {label}", False, str(e))

# ─────────────────────────────────────────────────────────────
# 3. THIEF REGISTRY
# ─────────────────────────────────────────────────────────────
section("3. Thief Registry")

tmp_dir = tempfile.mkdtemp(prefix="safeguard_test_")
registry = None
try:
    from thief_registry import ThiefRegistry
    registry = ThiefRegistry(base_dir=tmp_dir)
    check("ThiefRegistry init", True)
except Exception as e:
    check("ThiefRegistry init", False, str(e))

mock_emb   = np.random.rand(16).astype(np.float32)
mock_emb  /= np.linalg.norm(mock_emb)
mock_face  = np.zeros((64, 64, 3), dtype=np.uint8)
enrolled_id = None

if registry:
    try:
        entry = registry.enroll("Test Suspect", mock_emb, mock_face, notes="test")
        enrolled_id = entry["id"]
        check("enroll() suspect", True, f"id={enrolled_id}")
    except Exception as e:
        check("enroll() suspect", False, str(e))

    try:
        pairs = registry.load_embeddings()
        check("load_embeddings() (cached)", len(pairs) == 1, f"{len(pairs)} profile(s)")
    except Exception as e:
        check("load_embeddings() (cached)", False, str(e))

    # Second call must hit cache (no disk read)
    try:
        pairs2 = registry.load_embeddings()
        check("load_embeddings() cache hit", pairs2 is pairs, "same object returned")
    except Exception as e:
        check("load_embeddings() cache hit", False, str(e))

    if enrolled_id:
        try:
            ok = registry.delete_thief(enrolled_id)
            pairs3 = registry.load_embeddings()
            check("delete_thief() + cache invalidation", ok and len(pairs3) == 0)
        except Exception as e:
            check("delete_thief() + cache invalidation", False, str(e))

# ─────────────────────────────────────────────────────────────
# 4. FACE DETECTOR
# ─────────────────────────────────────────────────────────────
section("4. Face Detector")

face_det = None
try:
    from face_detector import FaceDetector, _MP_AVAILABLE as fd_mp
    check("FaceDetector _MP_AVAILABLE", fd_mp, f"mediapipe={'yes' if fd_mp else 'no'}")
    face_det = FaceDetector(match_threshold=0.45, cover_threshold=0.55)
    check("FaceDetector instantiation", True)
    check("FaceDetector landmarker loaded", face_det._landmarker is not None)
except Exception as e:
    check("FaceDetector instantiation", False, str(e))

if face_det and face_det._landmarker:
    try:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated, res = face_det.detect(dummy, [])
        check("FaceDetector.detect() runs on blank frame", isinstance(res, dict))
        check("FaceDetector returns faces_count key", "faces_count" in res)
    except Exception as e:
        check("FaceDetector.detect()", False, str(e))

# ─────────────────────────────────────────────────────────────
# 5. GESTURE DETECTOR
# ─────────────────────────────────────────────────────────────
section("5. Gesture Detector")

gest_det = None
try:
    from gesture_detector import GestureDetector, _MP_AVAILABLE as gd_mp
    check("GestureDetector _MP_AVAILABLE", gd_mp)
    gest_det = GestureDetector(confidence_threshold=0.8)
    check("GestureDetector instantiation", True)
    check("GestureDetector landmarker loaded", gest_det._landmarker is not None)
except Exception as e:
    check("GestureDetector instantiation", False, str(e))

if gest_det and gest_det._landmarker:
    try:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated, res = gest_det.detect(dummy)
        check("GestureDetector.detect() runs on blank frame", isinstance(res, dict))
        required_keys = {"detected", "confidence", "hands_detected", "gesture_type"}
        check("GestureDetector result has all required keys",
              required_keys.issubset(res.keys()))
    except Exception as e:
        check("GestureDetector.detect()", False, str(e))

# ─────────────────────────────────────────────────────────────
# 6. BOTH DETECTORS SIMULTANEOUSLY
# ─────────────────────────────────────────────────────────────
section("6. Simultaneous Operation (Face + Gesture)")

if face_det and gest_det:
    try:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        _, face_res   = face_det.detect(dummy, [])
        _, gest_res   = gest_det.detect(dummy)
        check("Face + Gesture detect simultaneously", True,
              f"faces={face_res['faces_count']}, gesture={gest_res['detected']}")
    except Exception as e:
        check("Face + Gesture detect simultaneously", False, str(e))
else:
    check("Face + Gesture detect simultaneously", False, "one or both detectors not loaded")

# ─────────────────────────────────────────────────────────────
# 7. OBJECT DETECTOR (YOLOv8)
# ─────────────────────────────────────────────────────────────
section("7. Object Detector (YOLOv8)")

obj_det = None
try:
    from object_detector import ObjectDetector
    obj_det = ObjectDetector(confidence=0.45)
    check("ObjectDetector instantiation", True)
    check("ObjectDetector YOLO enabled", obj_det.enabled,
          "yolov8n.pt loaded" if obj_det.enabled else "model not found – disabled")
except Exception as e:
    check("ObjectDetector instantiation", False, str(e))

if obj_det:
    try:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated, res = obj_det.detect(dummy)
        check("ObjectDetector.detect() runs", isinstance(res, dict))
        check("ObjectDetector result has sharp_object key", "sharp_object" in res)
        check("ObjectDetector edge_hint computed", "edge_hint" in res.get("sharp_object", {}))
    except Exception as e:
        check("ObjectDetector.detect()", False, str(e))

# ─────────────────────────────────────────────────────────────
# 8. POSE SAFETY DETECTOR
# ─────────────────────────────────────────────────────────────
section("8. Pose Safety Detector")

pose_det = None
try:
    from pose_safety_detector import PoseSafetyDetector
    pose_det = PoseSafetyDetector()
    check("PoseSafetyDetector instantiation", True)
    check("PoseSafetyDetector enabled", getattr(pose_det, 'enabled', True))
except Exception as e:
    check("PoseSafetyDetector instantiation", False, str(e))

if pose_det:
    try:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        activity = {
            "persons_count": 0,
            "motion": {"ratio": 0.0, "flow_magnitude": 0.0},
            "violence": {"score": 0.0, "detected": False, "confidence": 0.0},
            "loitering": {"confidence": 0.0, "detected": False},
            "running_panic": {"confidence": 0.0, "detected": False},
        }
        annotated, res = pose_det.detect(dummy, activity)
        check("PoseSafetyDetector.detect() runs", isinstance(res, dict))
        check("PoseSafetyDetector result has enabled key", "enabled" in res)
    except Exception as e:
        check("PoseSafetyDetector.detect()", False, str(e))

# ─────────────────────────────────────────────────────────────
# 9. FACE EMBEDDING + REGISTRY MATCH END-TO-END
# ─────────────────────────────────────────────────────────────
section("9. Face Embedding + Registry Match (End-to-End)")

if face_det and face_det._landmarker and registry:
    try:
        from face_detector import _geometric_embedding

        # Build synthetic 478-landmark list (all at centre)
        class FakeLM:
            def __init__(self, x=0.5, y=0.5, z=0.0):
                self.x, self.y, self.z = x, y, z
                self.presence, self.visibility = 1.0, 1.0

        NUM_LM = 478
        landmarks = [FakeLM() for _ in range(NUM_LM)]
        # Place key landmarks at real positions
        positions = {
            33: (0.35, 0.45), 263: (0.65, 0.45),   # eyes
            1:  (0.50, 0.55), 152: (0.50, 0.80),    # nose-chin
            61: (0.40, 0.65), 291: (0.60, 0.65),    # mouth
            133:(0.37, 0.45), 362: (0.63, 0.45),
            70: (0.33, 0.42), 300: (0.67, 0.42),
            105:(0.36, 0.38), 334: (0.64, 0.38),
            107:(0.35, 0.36), 336: (0.65, 0.36),
            151:(0.50, 0.35),  9:  (0.50, 0.33),
            10: (0.50, 0.30), 108: (0.44, 0.34),
            337:(0.56, 0.34), 17: (0.50, 0.75),
            200:(0.50, 0.78),  2:  (0.50, 0.53),
        }
        for idx, (x, y) in positions.items():
            if idx < NUM_LM:
                landmarks[idx] = FakeLM(x, y)

        emb = _geometric_embedding(landmarks, w=640, h=480)
        check("_geometric_embedding() produces valid vector",
              emb is not None and len(emb) == 16,
              f"shape={None if emb is None else emb.shape}")

        if emb is not None:
            # Enroll and attempt match
            dummy_face = np.zeros((64, 64, 3), dtype=np.uint8)
            entry = registry.enroll("E2E Test", emb, dummy_face)
            pairs = registry.load_embeddings()

            # Compute cosine similarity manually
            from face_detector import _cosine_sim
            matched = any(_cosine_sim(emb, p_emb) >= 0.45 for _, p_emb in pairs)
            check("Enrolled embedding matches itself (cosine >= 0.45)", matched)

            registry.delete_thief(entry["id"])

    except Exception as e:
        check("Face Embedding + Registry Match", False, str(e))

# ─────────────────────────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────────────────────────
try:
    shutil.rmtree(tmp_dir, ignore_errors=True)
except Exception:
    pass

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
passed = sum(1 for _, ok in results if ok)
total  = len(results)
failed = [(lbl, ok) for lbl, ok in results if not ok]

print(f"\n{'='*55}")
print(f"  RESULTS:  {passed}/{total} tests passed")
if failed:
    print("\n  FAILED TESTS:")
    for lbl, _ in failed:
        print(f"    FAIL  {lbl}")
else:
    print("  *** All tests passed! ***")
print(f"{'='*55}\n")

sys.exit(0 if passed == total else 1)
