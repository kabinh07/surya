from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from PIL import Image
import json

recognizer = RecognitionPredictor()
detection = DetectionPredictor()

if __name__ == "__main__": 
    image = Image.open("test_data/15998253478_77583667ac_z.jpg")
    predictions = recognizer([image], ['en,bn'.split(",")], det_predictor=detection)
    print(predictions[0].text_lines)
    results = []
    for line in predictions[0].text_lines:
        results.append(
            {
                "polygon": line.polygon, 
                "confidence": line.confidence, 
                "text": line.text
            }
        )
    with open("test_data/desco_output.json", "w", encoding = "utf-8") as f:
        json.dump(results, f)