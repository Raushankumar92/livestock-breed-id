import json
from dataclasses import dataclass

import numpy as np
from PIL import Image
from tensorflow import keras
from ultralytics import YOLO

from app.config import Settings
from app.schemas import Prediction

# `cow` is class 19 in COCO, which is the dataset used by the stock YOLOv8
# checkpoint. With a custom livestock detector, update these class IDs.
LIVESTOCK_CLASSES = {19}


@dataclass
class CropResult:
    image: Image.Image
    found: bool
    confidence: float | None


class BreedPredictor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.classifier: keras.Model | None = None
        self.detector: YOLO | None = None
        self.labels: list[str] = []

    def load(self) -> None:
        if not self.settings.model_path.exists():
            raise FileNotFoundError(f"Classifier model not found: {self.settings.model_path}")
        if not self.settings.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.settings.labels_path}")
        self.classifier = keras.models.load_model(self.settings.model_path)
        self.labels = json.loads(self.settings.labels_path.read_text(encoding="utf-8"))
        if self.settings.enable_detection:
            self.detector = YOLO(self.settings.yolo_model)

    def crop_livestock(self, image: Image.Image) -> CropResult:
        if self.detector is None:
            return CropResult(image, False, None)
        result = self.detector(np.asarray(image.convert("RGB")), verbose=False)[0]
        candidates: list[tuple[float, np.ndarray]] = []
        if result.boxes is not None:
            for box in result.boxes:
                if int(box.cls.item()) in LIVESTOCK_CLASSES:
                    candidates.append((float(box.conf.item()), box.xyxy[0].cpu().numpy()))
        if not candidates:
            return CropResult(image, False, None)
        confidence, coords = max(candidates, key=lambda item: item[0])
        left, top, right, bottom = (int(x) for x in coords)
        width, height = image.size
        pad_x, pad_y = int((right - left) * 0.05), int((bottom - top) * 0.05)
        box = (max(0, left - pad_x), max(0, top - pad_y), min(width, right + pad_x), min(height, bottom + pad_y))
        return CropResult(image.crop(box), True, confidence)

    def predict(self, image: Image.Image, top_k: int = 3) -> tuple[list[Prediction], CropResult]:
        if self.classifier is None:
            raise RuntimeError("Classifier has not been loaded.")
        crop = self.crop_livestock(image)
        resized = crop.image.convert("RGB").resize((224, 224))
        # The training model includes MobileNetV2 preprocessing as its first layer.
        tensor = np.asarray(resized, dtype=np.float32)
        probabilities = self.classifier.predict(np.expand_dims(tensor, 0), verbose=0)[0]
        indices = np.argsort(probabilities)[::-1][:min(top_k, len(self.labels))]
        predictions = [Prediction(breed=self.labels[int(index)], confidence=float(probabilities[index])) for index in indices]
        return predictions, crop
