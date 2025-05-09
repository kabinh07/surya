from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from PIL import Image

recognizer = RecognitionPredictor()
detector = DetectionPredictor()

image = Image.open("test_data/748_back.png")
langs = ['en', 'bn']

predictions = recognizer([image], [langs], detector)
print(predictions)
