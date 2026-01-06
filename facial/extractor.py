import cv2
import torch
import torchvision.transforms as transforms
from fer import FER
import numpy as np


class FacialExtractor:
    def __init__(self, use_mtcnn: bool = True):
        self.detector = FER(mtcnn=use_mtcnn)
        self._emotion_order = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

        self._legacy_mode = True

    def extract(self, image_path: str) -> torch.Tensor:
        if not self._legacy_mode:
            return self._extract_image_tensor(image_path)

        img = cv2.imread(image_path)
        if img is None:
            return torch.zeros(7, dtype=torch.float32)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            results = self.detector.detect_emotions(img_rgb)
        except Exception:
            return torch.zeros(7, dtype=torch.float32)

        if not results:
            return torch.zeros(7, dtype=torch.float32)

        emotions = results[0].get("emotions", {})
        probs = [float(emotions.get(e, 0.0)) for e in self._emotion_order]
        return torch.tensor(probs, dtype=torch.float32)

    def _extract_image_tensor(self, image_path: str) -> torch.Tensor:
        img = cv2.imread(image_path)
        if img is None:
            return torch.zeros(3, 224, 224, dtype=torch.float32)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        tensor = transform(img_rgb)
        return tensor
