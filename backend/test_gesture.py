import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gesture_detector import GestureDetector

def main():
    print("Initializing GestureDetector...")
    detector = GestureDetector()
    print("Creating dummy frame...")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    print("Detecting gestures...")
    annotated, result = detector.detect(frame)
    print("Result:", result)
    detector.close()
    print("Success!")

if __name__ == "__main__":
    main()
