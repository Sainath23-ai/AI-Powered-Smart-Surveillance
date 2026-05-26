"""
Hand Gesture Detector - Detects SOS/Help Gestures
Recognizes the international "Signal for Help" gesture:
  - Thumb tucked inside closed fist
  - Also detects raised open palm (stop/help signal)
Uses MediaPipe HandLandmarker Tasks API.
"""

import cv2
import numpy as np
import time
import os
import urllib.request

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except Exception as exc:  # pragma: no cover - depends on local install
    print(f"MediaPipe gesture detector unavailable: {exc}")
    mp = None
    python = None
    vision = None


class GestureDetector:
    def __init__(self, confidence_threshold=0.80):
        self.confidence_threshold = confidence_threshold
        self.detector = None
        
        # Download the model if it's missing
        model_filename = 'hand_landmarker.task'
        model_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(model_dir, model_filename)

        if mp is None or python is None or vision is None:
            return
        
        if not os.path.exists(self.model_path):
            print("Downloading hand_landmarker.task model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            try:
                urllib.request.urlretrieve(url, self.model_path)
                print("Model downloaded successfully.")
            except Exception as e:
                print(f"Failed to download model: {e}")

        if not os.path.exists(self.model_path):
            print("Gesture detector disabled: hand_landmarker.task is missing.")
            return

        # Initialize detector
        try:
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                running_mode=vision.RunningMode.IMAGE
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Gesture detector disabled: {e}")
            self.detector = None

    def _get_finger_states(self, landmarks):
        """Returns list of booleans: [thumb, index, middle, ring, pinky] extended."""
        tips = [4, 8, 12, 16, 20]
        pip = [3, 6, 10, 14, 18]
        mcp = [2, 5, 9, 13, 17]

        fingers = []

        # Thumb - compare x coordinates (left/right hand dependent)
        wrist_x = landmarks[0].x
        thumb_tip_x = landmarks[4].x
        thumb_mcp_x = landmarks[2].x
        if wrist_x < thumb_mcp_x:
            fingers.append(thumb_tip_x > thumb_mcp_x)
        else:
            fingers.append(thumb_tip_x < thumb_mcp_x)

        # Other fingers - compare y coordinates
        for tip_id, pip_id in zip(tips[1:], pip[1:]):
            fingers.append(landmarks[tip_id].y < landmarks[pip_id].y)

        return fingers

    def _detect_sos_gesture(self, landmarks):
        """
        Detects the international Signal for Help:
        - Thumb tucked in, fingers folded over it (closed fist with thumb inside)
        Returns confidence score 0-1.
        """
        fingers = self._get_finger_states(landmarks)
        # SOS: thumb tucked (False), all other fingers down (False)
        all_fingers_down = not any(fingers)

        if all_fingers_down:
            return 0.95

        # Also detect raised open palm (help signal)
        all_fingers_up = all(fingers)
        if all_fingers_up:
            return 0.80

        # Partial help gesture - 3+ fingers down, thumb in
        fingers_down_count = sum(1 for f in fingers if not f)
        if fingers_down_count >= 4 and not fingers[0]:
            return 0.70

        return 0.0

    def _draw_landmarks(self, image, hand_landmarks):
        h, w, _ = image.shape
        coords = []
        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            coords.append((cx, cy))
            cv2.circle(image, (cx, cy), 5, (0, 255, 100), -1)

        # Connections to draw
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle
            (9, 10), (10, 11), (11, 12),
            # Ring
            (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm boundary
            (5, 9), (9, 13), (13, 17)
        ]
        for start, end in connections:
            if start < len(coords) and end < len(coords):
                cv2.line(image, coords[start], coords[end], (255, 255, 255), 2)

    def detect(self, frame):
        """
        Process frame and return:
        - annotated_frame: frame with hand landmarks drawn
        - gesture_result: dict with 'detected', 'confidence', 'gesture_name'
        """
        if self.detector is None:
            return frame, {
                "detected": False,
                "confidence": 0.0,
                "gesture_name": None,
                "hands_detected": 0,
                "enabled": False
            }

        # Convert the BGR image to RGB and create MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect hand landmarks
        results = self.detector.detect(mp_image)

        annotated = frame.copy()
        gesture_result = {
            "detected": False,
            "confidence": 0.0,
            "gesture_name": None,
            "hands_detected": 0
        }

        if results.hand_landmarks:
            gesture_result["hands_detected"] = len(results.hand_landmarks)

            for hand_landmarks in results.hand_landmarks:
                # Draw hand skeleton and points
                self._draw_landmarks(annotated, hand_landmarks)

                confidence = self._detect_sos_gesture(hand_landmarks)

                if confidence >= self.confidence_threshold:
                    gesture_result["detected"] = True
                    gesture_result["confidence"] = confidence

                    fingers = self._get_finger_states(hand_landmarks)
                    if all(fingers):
                        gesture_result["gesture_name"] = "Open Palm (Help!)"
                    else:
                        gesture_result["gesture_name"] = "SOS Fist Gesture"

                    # Draw alert overlay
                    h, w = annotated.shape[:2]
                    cv2.rectangle(annotated, (0, 0), (w, 60), (0, 0, 200), -1)
                    cv2.putText(annotated,
                                f"🆘 {gesture_result['gesture_name']} ({confidence:.0%})",
                                (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                                (255, 255, 255), 2)

        return annotated, gesture_result

    def close(self):
        if self.detector is not None:
            self.detector.close()
