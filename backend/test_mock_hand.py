import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gesture_detector import GestureDetector

# Mock landmark class
class MockLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Mock result class
class MockResult:
    def __init__(self, hand_landmarks):
        self.hand_landmarks = hand_landmarks

def main():
    detector = GestureDetector()
    lms = []
    # Wrist (0)
    lms.append(MockLandmark(0.5, 0.9))
    # Thumb (1-4)
    lms.append(MockLandmark(0.4, 0.8))
    lms.append(MockLandmark(0.3, 0.7))
    lms.append(MockLandmark(0.2, 0.6))
    lms.append(MockLandmark(0.1, 0.5)) # Thumb tip (4)
    # Index (5-8)
    lms.append(MockLandmark(0.4, 0.5)) # MCP (5)
    lms.append(MockLandmark(0.4, 0.4))
    lms.append(MockLandmark(0.4, 0.3))
    lms.append(MockLandmark(0.4, 0.2)) # Tip (8)
    # Middle (9-12)
    lms.append(MockLandmark(0.5, 0.5)) # MCP (9)
    lms.append(MockLandmark(0.5, 0.4))
    lms.append(MockLandmark(0.5, 0.3))
    lms.append(MockLandmark(0.5, 0.2)) # Tip (12)
    # Ring (13-16)
    lms.append(MockLandmark(0.6, 0.5)) # MCP (13)
    lms.append(MockLandmark(0.6, 0.4))
    lms.append(MockLandmark(0.6, 0.3))
    lms.append(MockLandmark(0.6, 0.2)) # Tip (16)
    # Pinky (17-20)
    lms.append(MockLandmark(0.7, 0.5)) # MCP (17)
    lms.append(MockLandmark(0.7, 0.45))
    lms.append(MockLandmark(0.7, 0.4))
    lms.append(MockLandmark(0.7, 0.35)) # Tip (20)

    class MockLandmarker:
        def detect(self, mp_img):
            return MockResult([lms])
    
    detector._landmarker = MockLandmarker()

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated, result = detector.detect(frame)
    print("Open Palm Test Result:", result)

    lms_fist = list(lms)
    lms_fist[8] = MockLandmark(0.4, 0.6) # Index tip
    lms_fist[12] = MockLandmark(0.5, 0.6) # Middle tip
    lms_fist[16] = MockLandmark(0.6, 0.6) # Ring tip
    lms_fist[20] = MockLandmark(0.7, 0.6) # Pinky tip
    lms_fist[4] = MockLandmark(0.45, 0.52)

    class MockLandmarkerFist:
        def detect(self, mp_img):
            return MockResult([lms_fist])

    detector._landmarker = MockLandmarkerFist()
    annotated, result = detector.detect(frame)
    print("SOS Fist Test Result:", result)

if __name__ == "__main__":
    main()
