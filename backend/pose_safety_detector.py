"""
Pose-based safety scenario estimator.

This layer uses MediaPipe Pose landmarks plus the existing motion/person
signals to estimate higher-level safety scenarios. It is intentionally
lightweight and heuristic so it can run beside the existing detectors.
"""

from collections import deque
import math
import os
import urllib.request

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except Exception:  # pragma: no cover - handled at runtime
    mp = None
    python = None
    vision = None


class PoseSafetyDetector:
    def __init__(self, threshold=0.68, sensitivity="medium"):
        self.threshold = threshold
        self.sensitivity = sensitivity
        self.pose = None
        self.center_history = deque(maxlen=24)
        self.wrist_history = deque(maxlen=16)

        if mp is not None and vision is not None:
            model_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_path = os.path.join(model_dir, "pose_landmarker_lite.task")
            if not os.path.exists(self.model_path):
                url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
                try:
                    urllib.request.urlretrieve(url, self.model_path)
                except Exception as exc:
                    print(f"Failed to download pose model: {exc}")

            if os.path.exists(self.model_path):
                try:
                    base_options = python.BaseOptions(model_asset_path=self.model_path)
                    options = vision.PoseLandmarkerOptions(
                        base_options=base_options,
                        running_mode=vision.RunningMode.IMAGE,
                        num_poses=1,
                        min_pose_detection_confidence=0.5,
                        min_pose_presence_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )
                    self.pose = vision.PoseLandmarker.create_from_options(options)
                except Exception as exc:
                    print(f"Pose detector disabled: {exc}")
                    self.pose = None

    def reset_state(self):
        """Reset temporal pose history before processing a new source."""
        self.center_history.clear()
        self.wrist_history.clear()

    def close(self):
        if self.pose is not None:
            self.pose.close()

    def _empty_result(self):
        return {
            "enabled": self.pose is not None,
            "landmarks_detected": False,
            "top_scenario": None,
            "summary": "No pose landmarks",
            "scenarios": self._scenario_map({})
        }

    def _scenario_map(self, scores):
        labels = {
            "fighting": "Fighting / Violence",
            "kidnapping": "Kidnapping Attempt",
            "harassment": "Harassment",
            "chasing": "Person Chasing",
            "child_fall": "Child Crying / Falling",
            "weapon_carry": "Weapon Carry Posture",
            "crowd_anomaly": "Unusual Crowd Behavior",
        }
        return {
            key: {
                "label": label,
                "confidence": round(float(scores.get(key, 0.0)), 3),
                "detected": float(scores.get(key, 0.0)) >= self.threshold,
            }
            for key, label in labels.items()
        }

    @staticmethod
    def _dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _mid(a, b):
        return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)

    @staticmethod
    def _clamp(value):
        return max(0.0, min(1.0, float(value)))

    def _landmark_points(self, pose_landmarks, width, height):
        points = {}
        names = {
            0: "nose",
            11: "left_shoulder",
            12: "right_shoulder",
            13: "left_elbow",
            14: "right_elbow",
            15: "left_wrist",
            16: "right_wrist",
            23: "left_hip",
            24: "right_hip",
            25: "left_knee",
            26: "right_knee",
            27: "left_ankle",
            28: "right_ankle",
        }
        for idx, name in names.items():
            if idx >= len(pose_landmarks):
                continue
            lm = pose_landmarks[idx]
            visibility = getattr(lm, "visibility", 1.0)
            presence = getattr(lm, "presence", 1.0)
            if visibility < 0.25 or presence < 0.25:
                continue
            points[name] = (lm.x * width, lm.y * height, visibility)
        return points

    @staticmethod
    def _draw_pose(image, pose_landmarks):
        h, w = image.shape[:2]
        coords = []
        for lm in pose_landmarks:
            coords.append((int(lm.x * w), int(lm.y * h)))

        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
            (24, 26), (26, 28), (0, 11), (0, 12)
        ]
        for start, end in connections:
            if start < len(coords) and end < len(coords):
                cv2.line(image, coords[start], coords[end], (255, 180, 80), 2)
        for idx in (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28):
            if idx < len(coords):
                cv2.circle(image, coords[idx], 4, (80, 220, 255), -1)

    def _pose_features(self, points, frame_shape):
        h, w = frame_shape[:2]
        needed = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        if not all(k in points for k in needed):
            return None

        ls, rs = points["left_shoulder"], points["right_shoulder"]
        lh, rh = points["left_hip"], points["right_hip"]
        shoulder_mid = self._mid(ls, rs)
        hip_mid = self._mid(lh, rh)
        torso_len = max(self._dist(shoulder_mid, hip_mid), 1.0)
        shoulder_width = max(self._dist(ls, rs), 1.0)
        center = self._mid(shoulder_mid, hip_mid)

        self.center_history.append(center)
        vertical_drop = 0.0
        if len(self.center_history) >= 8:
            old_y = np.mean([p[1] for p in list(self.center_history)[:4]])
            new_y = np.mean([p[1] for p in list(self.center_history)[-4:]])
            vertical_drop = max(0.0, (new_y - old_y) / max(h, 1))

        wrists = []
        for key in ("left_wrist", "right_wrist"):
            if key in points:
                wrists.append(points[key])
        if wrists:
            self.wrist_history.append(tuple((p[0], p[1]) for p in wrists))

        wrist_speed = 0.0
        if len(self.wrist_history) >= 2:
            prev = self.wrist_history[-2]
            curr = self.wrist_history[-1]
            pairs = zip(prev[:len(curr)], curr[:len(prev)])
            speeds = [self._dist(a, b) / max(shoulder_width, 1.0) for a, b in pairs]
            wrist_speed = min(1.0, max(speeds) if speeds else 0.0)

        torso_angle = abs(math.degrees(math.atan2(
            hip_mid[1] - shoulder_mid[1],
            hip_mid[0] - shoulder_mid[0]
        )))
        horizontal_body = 1.0 if torso_angle < 45 or torso_angle > 135 else 0.0
        low_body = 1.0 if center[1] > h * 0.63 else 0.0

        arms_raised = 0.0
        arm_extension = 0.0
        hand_near_head = 0.0
        weapon_like = 0.0
        for side in ("left", "right"):
            wrist = points.get(f"{side}_wrist")
            elbow = points.get(f"{side}_elbow")
            shoulder = points.get(f"{side}_shoulder")
            if not wrist or not shoulder:
                continue
            extension = self._dist(wrist, shoulder) / shoulder_width
            arm_extension = max(arm_extension, min(extension / 2.2, 1.0))
            if wrist[1] < shoulder[1]:
                arms_raised = max(arms_raised, 1.0)
            if wrist[1] < shoulder_mid[1] + torso_len * 0.45:
                hand_near_head = max(hand_near_head, 1.0)
            if elbow:
                straightness = self._dist(wrist, shoulder) / max(
                    self._dist(shoulder, elbow) + self._dist(elbow, wrist), 1.0
                )
                away_from_torso = abs(wrist[0] - center[0]) / max(shoulder_width, 1.0)
                mid_height = 1.0 if shoulder_mid[1] < wrist[1] < hip_mid[1] + torso_len * 0.5 else 0.0
                weapon_like = max(weapon_like, straightness * min(away_from_torso / 1.8, 1.0) * mid_height)

        return {
            "center": center,
            "torso_len": torso_len,
            "shoulder_width": shoulder_width,
            "vertical_drop": vertical_drop,
            "wrist_speed": wrist_speed,
            "horizontal_body": horizontal_body,
            "low_body": low_body,
            "arms_raised": arms_raised,
            "arm_extension": arm_extension,
            "hand_near_head": hand_near_head,
            "weapon_like": weapon_like,
        }

    def detect(self, frame, activity_results=None):
        if self.pose is None:
            return frame, self._empty_result()

        annotated = frame.copy()
        activity_results = activity_results or {}
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        pose_result = self.pose.detect(mp_image)
        persons = int(activity_results.get("persons_count", 0) or 0)
        motion = activity_results.get("motion", {})
        flow = float(motion.get("flow_magnitude", 0.0) or 0.0)
        motion_ratio = float(motion.get("ratio", 0.0) or 0.0)
        violence = float(activity_results.get("violence", {}).get("confidence", 0.0) or 0.0)
        running = float(activity_results.get("running_panic", {}).get("confidence", 0.0) or 0.0)
        loiter = float(activity_results.get("loitering", {}).get("confidence", 0.0) or 0.0)

        if not pose_result.pose_landmarks:
            scores = {
                "crowd_anomaly": self._clamp((persons - 4) / 6.0 + motion_ratio * 0.7)
            }
            result = self._empty_result()
            result["scenarios"] = self._scenario_map(scores)
            result["top_scenario"] = self._top(result["scenarios"])
            return annotated, result

        h, w = frame.shape[:2]
        landmarks = pose_result.pose_landmarks[0]
        self._draw_pose(annotated, landmarks)
        points = self._landmark_points(landmarks, w, h)
        features = self._pose_features(points, frame.shape)

        if features is None:
            return annotated, self._empty_result()

        multiple_people = 1.0 if persons >= 2 else 0.0
        crowd_pressure = self._clamp((persons - 3) / 5.0)
        high_motion = self._clamp(flow / 12.0 + motion_ratio * 0.8)

        scores = {
            "fighting": self._clamp(
                violence * 0.55 + features["wrist_speed"] * 0.25 +
                features["arm_extension"] * 0.15 + multiple_people * 0.10
            ),
            "kidnapping": self._clamp(
                multiple_people * 0.25 + high_motion * 0.25 + running * 0.25 +
                features["arms_raised"] * 0.15 + features["vertical_drop"] * 1.2
            ),
            "harassment": self._clamp(
                multiple_people * 0.30 + loiter * 0.25 +
                features["hand_near_head"] * 0.20 + features["arms_raised"] * 0.15
            ),
            "chasing": self._clamp(
                running * 0.55 + multiple_people * 0.20 + high_motion * 0.25
            ),
            "child_fall": self._clamp(
                features["horizontal_body"] * 0.35 + features["low_body"] * 0.20 +
                features["vertical_drop"] * 2.0 + high_motion * 0.15
            ),
            "weapon_carry": self._clamp(
                features["weapon_like"] * 0.60 + features["arm_extension"] * 0.20 +
                loiter * 0.10 + multiple_people * 0.10
            ),
            "crowd_anomaly": self._clamp(
                crowd_pressure * 0.45 + high_motion * 0.35 + violence * 0.20
            ),
        }

        scenarios = self._scenario_map(scores)
        top = self._top(scenarios)
        result = {
            "enabled": True,
            "landmarks_detected": True,
            "top_scenario": top,
            "summary": top["label"] if top else "Pose normal",
            "scenarios": scenarios,
        }

        if top and top["detected"]:
            cv2.rectangle(annotated, (0, 55), (w, 105), (0, 90, 180), -1)
            cv2.putText(
                annotated,
                f"POSE ALERT: {top['label']} ({top['confidence']:.0%})",
                (10, 90),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        return annotated, result

    @staticmethod
    def _top(scenarios):
        if not scenarios:
            return None
        key, scenario = max(scenarios.items(), key=lambda item: item[1]["confidence"])
        return {"key": key, **scenario}
