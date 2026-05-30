import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from face_detector import FaceDetector, _geometric_embedding
from thief_registry import ThiefRegistry

# Mock landmark class
class MockLandmark:
    def __init__(self, x, y, presence=1.0):
        self.x = x
        self.y = y
        self.presence = presence
        self.visibility = presence

# Mock result class
class MockResult:
    def __init__(self, face_landmarks):
        self.face_landmarks = face_landmarks

def create_mock_face(shift_x=0.0, shift_y=0.0):
    lms = []
    # Fill with default values
    for idx in range(478):
        lms.append(MockLandmark(0.5 + shift_x, 0.5 + shift_y))
    
    # Specific landmarks for eyes, nose, mouth, chin
    lms[1] = MockLandmark(0.5 + shift_x, 0.5 + shift_y)       # nose tip
    lms[33] = MockLandmark(0.4 + shift_x, 0.4 + shift_y)      # left eye corner
    lms[263] = MockLandmark(0.6 + shift_x, 0.6 + shift_y)     # right eye corner
    lms[152] = MockLandmark(0.5 + shift_x, 0.8 + shift_y)     # chin
    lms[61] = MockLandmark(0.45 + shift_x, 0.65 + shift_y)    # mouth left corner
    lms[291] = MockLandmark(0.55 + shift_x, 0.65 + shift_y)   # mouth right corner
    lms[10] = MockLandmark(0.5 + shift_x, 0.2 + shift_y)      # forehead
    lms[105] = MockLandmark(0.42 + shift_x, 0.35 + shift_y)   # brow left
    lms[334] = MockLandmark(0.58 + shift_x, 0.35 + shift_y)   # brow right
    lms[133] = MockLandmark(0.43 + shift_x, 0.42 + shift_y)   # eye landmark
    lms[362] = MockLandmark(0.57 + shift_x, 0.42 + shift_y)   # eye landmark
    lms[70] = MockLandmark(0.4 + shift_x, 0.34 + shift_y)     # brow
    lms[300] = MockLandmark(0.6 + shift_x, 0.34 + shift_y)     # brow
    lms[234] = MockLandmark(0.3 + shift_x, 0.5 + shift_y)     # cheek left
    lms[454] = MockLandmark(0.7 + shift_x, 0.5 + shift_y)     # cheek right
    lms[159] = MockLandmark(0.4 + shift_x, 0.38 + shift_y)
    lms[386] = MockLandmark(0.6 + shift_x, 0.38 + shift_y)
    lms[13] = MockLandmark(0.48 + shift_x, 0.62 + shift_y)
    lms[14] = MockLandmark(0.52 + shift_x, 0.62 + shift_y)
    lms[200] = MockLandmark(0.5 + shift_x, 0.75 + shift_y)
    
    return lms

def main():
    print("Initializing Thief Registry and FaceDetector...")
    # Initialize Thief Registry in a temporary/mock location or standard location
    registry_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "test_thieves")
    registry = ThiefRegistry(registry_dir)
    
    detector = FaceDetector(match_threshold=0.45)
    
    print("Generating mock face landmarks...")
    face_landmarks = create_mock_face()
    w, h = 640, 480
    
    # 1. Compute embedding for enrollment
    embedding = _geometric_embedding(face_landmarks, w, h)
    if embedding is None:
        print("Error: Could not extract geometric embedding.")
        return
        
    print("Successfully extracted 16-dimensional geometric embedding:")
    print(embedding)
    
    # Create a dummy image
    dummy_image = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(dummy_image, "MOCK PHOTO", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 2. Enroll the thief in registry
    print("\nEnrolling mock thief 'John Doe (wanted for theft)' into registry...")
    entry = registry.enroll(
        name="John Doe",
        embedding=embedding,
        face_image=dummy_image,
        notes="Known shoplifter. Highly dangerous.",
        crime_details="Grand Theft Auto"
    )
    print("Enrolled successfully! ID:", entry["id"])
    
    # 3. Load embeddings to be used for matching
    print("\nLoading active thief embeddings for detector matching...")
    thief_embeddings = registry.load_embeddings()
    print(f"Loaded {len(thief_embeddings)} active profiles.")
    
    # 4. Mock the MediaPipe landmarker inside detector
    class MockFaceLandmarker:
        def detect(self, mp_img):
            return MockResult([face_landmarks])
            
    detector._landmarker = MockFaceLandmarker()
    
    # 5. Run detect on the dummy frame (which matches the enrolled face)
    print("\nRunning face detection loop with matching face...")
    annotated_frame, result = detector.detect(dummy_image, thief_embeddings)
    
    print("Detection result details:")
    print(result)
    
    if result["thief_match"]["detected"]:
        print(f"\n[SUCCESS] Face successfully matched!")
        print(f"Matched Thief Name: {result['thief_match']['name']}")
        print(f"Match Confidence: {result['thief_match']['confidence']:.1%}")
    else:
        print("\n[FAILURE] Thief face was not matched.")
        
    # Clean up mock database files
    print("\nCleaning up mock registry files...")
    registry.delete_thief(entry["id"])
    if os.path.exists(registry.index_path):
        os.remove(registry.index_path)
    if os.path.exists(registry.root):
        try:
            os.rmdir(registry.root)
        except Exception:
            pass

if __name__ == "__main__":
    main()
