from typing import List, Dict, Any, Optional
import numpy as np
from ultralytics import YOLO
from .config import TrackingConfig


class ObjectDetector:
    """Modul deteksi objek menggunakan model YOLO."""

    def __init__(self, config: TrackingConfig):
        self.config = config
        self.model = YOLO(config.model_name)
        self.class_names = self.model.names

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Menjalankan deteksi objek pada satu frame citra."""
        results = self.model.predict(
            source=frame,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            device=self.config.device,
            verbose=False
        )[0]

        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)

            for bbox, conf, cls_id in zip(boxes_xyxy, confs, clss):
                class_name = self.class_names.get(cls_id, str(cls_id))

                # Normalisasi kelas bus menjadi truck
                if class_name == "bus":
                    class_name = "truck"
                    cls_id = 7

                # Filter kelas target
                if self.config.target_classes and class_name not in self.config.target_classes:
                    continue

                detections.append({
                    "bbox": [float(coord) for coord in bbox],
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                    "class_name": class_name,
                })

        return detections
