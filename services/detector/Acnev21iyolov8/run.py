import os
import cv2
from ultralytics import YOLO
from collections import defaultdict

class YOLOImageProcessor:
    def __init__(self, model_path, confidence_threshold=0.5):

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Error: Model weights not found at {self.model_path}")
        print(f"YOLO model loaded successfully from {self.model_path}")
        return YOLO(self.model_path)

    def process_image(self, image):
        print("Here")
        results = self.model(image)[0]  
        print("Here 1")
        acne_counts = defaultdict(int) 
        print("Here 2")
        for result in results.boxes.data.tolist():
            print("Here 3")
            x1, y1, x2, y2, score, class_id = result
            if score > self.confidence_threshold:
                print("Here.4")
                class_name = self.model.names.get(int(class_id), "unkown type" ) 
                print("Here 5")
                acne_counts[class_name] += 1  
                self._draw_bounding_box(image, x1, y1, x2, y2, score, class_id)
                print("Here 6")

        return image, dict(acne_counts)  

    def _draw_bounding_box(self, image, x1, y1, x2, y2, score, class_id):
        try:

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            height, width, _ = image.shape

            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                print(f" Skipping invalid bounding box: ({x1}, {y1}, {x2}, {y2})")
                return 

            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{self.model.names[int(class_id)]} {score:.2f}"
            cv2.putText(image, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error drawing bounding box: {e}")
