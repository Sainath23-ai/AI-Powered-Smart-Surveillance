"""
YOLOv8 object detection for sharp / dangerous items (knife, etc.).
"""

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger("ObjectDetector")

# COCO class ids relevant to sharp / weapon-like objects
SHARP_COCO_IDS = {
    43: "knife",
    34: "baseball bat",
}

# Labels we treat as sharp threats (name-based for YOLO output)
SHARP_LABELS = {
    "knife",
    "scissors",
    "sword",
    "machete",
    "dagger",
    "blade",
    "axe",
    "baseball bat",
    "fork",
}


class ObjectDetector:
    def __init__(self, confidence: float = 0.45):
        self.confidence = confidence
        self.model = None
        self.enabled = False
        self._init_yolo()

    def _init_yolo(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.warning("ultralytics not installed — object detection disabled")
            return

        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "yolov8n.pt")

        try:
            self.model = YOLO(model_path if os.path.isfile(model_path) else "yolov8n.pt")
            self.enabled = True
            logger.info("YOLOv8 object detector loaded")
        except Exception as exc:
            logger.warning("YOLO load failed: %s", exc)

    def _edge_sharp_hint(self, frame) -> float:
        """Heuristic boost for elongated high-contrast edges (backup cue)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 180)
        ratio = np.count_nonzero(edges) / (edges.size + 1e-6)
        return float(min(1.0, ratio * 8))

    def detect(self, frame: np.ndarray):
        annotated = frame.copy()
        detections = []
        sharp_detected = False
        top_conf = 0.0
        top_label = None

        if self.enabled and self.model is not None:
            try:
                results = self.model.predict(
                    frame,
                    conf=self.confidence,
                    verbose=False,
                    imgsz=416,
                )
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        name = r.names.get(cls_id, str(cls_id)).lower()
                        is_sharp = (
                            cls_id in SHARP_COCO_IDS
                            or name in SHARP_LABELS
                            or any(s in name for s in ("knife", "scissor", "blade", "sword"))
                        )
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                        color = (0, 0, 255) if is_sharp else (180, 180, 180)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            annotated,
                            f"{name} {conf:.0%}",
                            (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                        )
                        detections.append(
                            {
                                "label": name,
                                "confidence": conf,
                                "box": [x1, y1, x2, y2],
                                "is_sharp": is_sharp,
                            }
                        )
                        if is_sharp and conf > top_conf:
                            sharp_detected = True
                            top_conf = conf
                            top_label = name
            except Exception as exc:
                logger.error("YOLO inference error: %s", exc)

        edge_hint = self._edge_sharp_hint(frame)
        results = {
            "enabled": self.enabled,
            "objects_count": len(detections),
            "sharp_object": {
                "detected": sharp_detected,
                "confidence": top_conf,
                "label": top_label,
                "edge_hint": edge_hint,
            },
            "detections": detections[:12],
        }
        return annotated, results
