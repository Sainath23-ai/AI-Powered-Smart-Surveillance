import cv2
import requests
import logging

logger = logging.getLogger("SafeGuardAI")

class HuggingFaceDetector:
    def __init__(self, api_key, model_url):
        self.api_key = api_key
        self.model_url = model_url
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def detect(self, frame):
        # 1. Convert the OpenCV frame (numpy array) to a JPEG byte string
        success, encoded_image = cv2.imencode('.jpg', frame)
        if not success:
            return frame, {"error": "Failed to encode image"}
            
        image_bytes = encoded_image.tobytes()

        # 2. Send the image to the Hugging Face API
        try:
            response = requests.post(
                self.model_url, 
                headers=self.headers, 
                data=image_bytes,
                timeout=3.0 # Don't block the camera stream for too long!
            )
            result = response.json()
            
            # The result is usually a list of dictionaries: [{'label': 'person', 'score': 0.99}]
            return frame, {"api_result": result}
            
        except Exception as e:
            logger.error(f"Hugging Face API Error: {e}")
            return frame, {"error": str(e)}
