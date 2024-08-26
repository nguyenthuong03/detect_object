import torch
import cv2
import numpy as np

class YOLO:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()

    def detect_image(self, image):
        image_tensor = self.preprocess(image)
        with torch.no_grad():
            detections = self.model(image_tensor)
        image = self.postprocess(detections, image)
        return image

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1)) / 255.0
        image_tensor = torch.tensor(image, dtype=torch.float).unsqueeze(0)
        return image_tensor

    def postprocess(self, detections, image):
        # Xử lý đầu ra và vẽ bounding boxes
        # ...
        return image
