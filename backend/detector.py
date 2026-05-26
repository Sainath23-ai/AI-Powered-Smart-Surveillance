"""
AI Violence & Suspicious Activity Detector
Uses optical flow (motion analysis) + background subtraction
+ a lightweight CNN approach to detect:
  - Violence / aggressive movement
  - Suspicious loitering
  - Running/panic behavior
  - Crowd anomaly
"""

import cv2
import numpy as np
import time
from collections import deque


class ActivityDetector:
    def __init__(self, violence_threshold=0.75, sensitivity="medium"):
        self.violence_threshold = violence_threshold
        self.sensitivity = sensitivity

        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )

        # Optical flow
        self.prev_gray = None
        self.motion_history = deque(maxlen=30)
        self.flow_magnitude_history = deque(maxlen=20)

        # Loitering detection
        self.person_positions = deque(maxlen=150)  # ~5 seconds at 30fps
        self.loiter_threshold = {"low": 120, "medium": 90, "high": 60}

        # HOG person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.last_detection = {}
        self.frame_count = 0

    def reset_state(self):
        """Reset temporal state before processing a new camera/video source."""
        self.prev_gray = None
        self.motion_history.clear()
        self.flow_magnitude_history.clear()
        self.person_positions.clear()
        self.last_detection = {}
        self.frame_count = 0

    def _compute_optical_flow(self, gray):
        """Compute dense optical flow to measure motion intensity."""
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0, None

        if (
            self.prev_gray.shape != gray.shape
            or self.prev_gray.ndim != 2
            or gray.ndim != 2
        ):
            self.prev_gray = gray
            self.flow_magnitude_history.clear()
            return 0.0, None

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        self.prev_gray = gray

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_mag = np.mean(magnitude)
        self.flow_magnitude_history.append(mean_mag)
        return mean_mag, flow

    def _detect_violence(self, frame, flow_mag):
        """
        Violence score based on:
        - Sudden, large motion spikes
        - Erratic motion direction changes
        - High motion variance
        """
        if len(self.flow_magnitude_history) < 5:
            return 0.0

        mags = list(self.flow_magnitude_history)
        mean_mag = np.mean(mags)
        std_mag = np.std(mags)

        # Spike detection - sudden large motion
        score = 0.0
        sensitivity_mult = {"low": 0.7, "medium": 1.0, "high": 1.4}[self.sensitivity]

        if flow_mag > 8 * sensitivity_mult:
            score += 0.4
        if flow_mag > 15 * sensitivity_mult:
            score += 0.3
        if std_mag > 5 * sensitivity_mult:
            score += 0.2
        if flow_mag > mean_mag * 2.5 and mean_mag > 2:
            score += 0.25

        return min(score, 1.0)

    def _detect_loitering(self, persons, frame_shape):
        """Detect if a person is staying in one area too long."""
        if not persons:
            return False, 0.0

        h, w = frame_shape[:2]
        for (x, y, pw, ph) in persons:
            center = (x + pw // 2, y + ph // 2)
            self.person_positions.append(center)

        if len(self.person_positions) < 30:
            return False, 0.0

        positions = np.array(self.person_positions)
        spread = np.std(positions, axis=0)
        area_spread = np.sqrt(spread[0]**2 + spread[1]**2)

        # Low spread over long period = loitering
        threshold = self.loiter_threshold.get(self.sensitivity, 90)
        if area_spread < 40 and len(self.person_positions) >= threshold:
            confidence = max(0.0, 1.0 - (area_spread / 40))
            return True, min(confidence, 0.95)

        return False, 0.0

    def _detect_running(self, flow_mag):
        """Detect panic/running based on sustained high motion."""
        if len(self.flow_magnitude_history) < 10:
            return False, 0.0

        recent = list(self.flow_magnitude_history)[-10:]
        sustained_high = sum(1 for m in recent if m > 6)

        if sustained_high >= 7:
            return True, min(sustained_high / 10.0, 0.90)
        return False, 0.0

    def _detect_persons(self, frame):
        """HOG-based person detection (runs every 5 frames for performance)."""
        if self.frame_count % 5 != 0:
            return self.last_detection.get('persons', [])

        small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        persons, _ = self.hog.detectMultiScale(
            small, winStride=(8, 8), padding=(4, 4), scale=1.05
        )
        # Scale back
        if len(persons) > 0:
            persons = [(x*2, y*2, w*2, h*2) for (x, y, w, h) in persons]
        self.last_detection['persons'] = list(persons)
        return list(persons)

    def detect(self, frame):
        """
        Process frame and return detection results.
        Returns: (annotated_frame, results_dict)
        """
        self.frame_count += 1
        annotated = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        fg_mask = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)[1]
        motion_pixels = cv2.countNonZero(fg_mask)
        frame_area = frame.shape[0] * frame.shape[1]
        motion_ratio = motion_pixels / frame_area

        # Optical flow
        flow_mag, flow = self._compute_optical_flow(gray)

        # Person detection
        persons = self._detect_persons(frame)

        # Draw persons
        for (x, y, pw, ph) in persons:
            cv2.rectangle(annotated, (x, y), (x + pw, y + ph), (0, 255, 100), 2)

        # Detections
        violence_score = self._detect_violence(frame, flow_mag)
        loitering, loiter_conf = self._detect_loitering(persons, frame.shape)
        running, run_conf = self._detect_running(flow_mag)

        results = {
            "violence": {
                "detected": violence_score >= self.violence_threshold,
                "score": round(violence_score, 3),
                "confidence": round(violence_score, 3)
            },
            "loitering": {
                "detected": loitering,
                "confidence": round(loiter_conf, 3)
            },
            "running_panic": {
                "detected": running,
                "confidence": round(run_conf, 3)
            },
            "motion": {
                "ratio": round(motion_ratio, 3),
                "flow_magnitude": round(flow_mag, 3)
            },
            "persons_count": len(persons),
            "threat_level": "none"
        }

        # Compute overall threat level
        if violence_score >= self.violence_threshold:
            results["threat_level"] = "critical"
            h, w = annotated.shape[:2]
            cv2.rectangle(annotated, (0, 0), (w, h), (0, 0, 255), 8)
            cv2.rectangle(annotated, (0, 0), (w, 55), (0, 0, 180), -1)
            cv2.putText(annotated, f"⚠ VIOLENCE DETECTED ({violence_score:.0%})",
                        (10, 38), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        elif loitering and loiter_conf > 0.6:
            results["threat_level"] = "warning"
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 50), (0, 140, 255), -1)
            cv2.putText(annotated, f"⚠ SUSPICIOUS LOITERING ({loiter_conf:.0%})",
                        (10, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        elif running and run_conf > 0.7:
            results["threat_level"] = "warning"
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 50), (0, 165, 255), -1)
            cv2.putText(annotated, f"⚠ PANIC/RUNNING DETECTED ({run_conf:.0%})",
                        (10, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

        # HUD overlay (bottom bar)
        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (0, h - 30), (w, h), (20, 20, 20), -1)
        cv2.putText(annotated,
                    f"Persons: {len(persons)} | Motion: {motion_ratio:.1%} | Flow: {flow_mag:.1f}",
                    (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        return annotated, results
