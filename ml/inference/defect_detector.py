"""
External defect detection using YOLOv8.
Currently unavailable - returns NOT_ASSESSED until model is trained.
"""

import logging
from typing import Optional
from PIL import Image
import numpy as np

from .types import InferenceResult, ModelAvailability, DetectionBox


logger = logging.getLogger(__name__)


class DefectDetector:
    """
    Detects external surface defects on potatoes using YOLOv8.
    
    Status: Model not yet trained (training notebook ready in Milestone 3).
    """
    
    # Defect class mapping
    CLASS_NAMES = {
        0: "Abnormal Shaped Potato",
        1: "Black Rot Potato",
        2: "Cut Potato",
        3: "Greening Potato",
        4: "Healthy Potato",
        5: "Hollow heart Potato",
        6: "Rotten",
        7: "Sprouted Patato",
    }
    
    # Severity mapping
    CLASS_SEVERITY = {
        0: "medium",   # Abnormal
        1: "high",     # Black Rot
        2: "medium",   # Cut
        3: "high",     # Greening (solanine)
        4: "low",      # Healthy (not actually a defect)
        5: "medium",   # Hollow heart
        6: "critical", # Rotten
        7: "low",      # Sprouted
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        confidence_threshold: float = 0.25,
    ):
        """
        Initialize defect detector.
        
        Args:
            model_path: Path to trained YOLOv8 weights (.pt file)
            device: Device for inference ("cpu" or "cuda")
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.availability = ModelAvailability.UNAVAILABLE
        self.unavailable_reason = "External defect detector model not yet trained"
        
        # Attempt to load model if path provided
        if model_path:
            self._load_model()
        
        logger.info(
            f"DefectDetector initialized: model_path={model_path}, "
            f"device={device}, availability={self.availability.value}"
        )
    
    def _load_model(self) -> None:
        """Attempt to load YOLO model from path."""
        try:
            from ultralytics import YOLO
            
            logger.info(f"Loading YOLO model from {self.model_path}")
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            self.availability = ModelAvailability.AVAILABLE
            self.unavailable_reason = None
            logger.info("YOLO model loaded successfully")
            
        except FileNotFoundError:
            logger.warning(f"Model file not found: {self.model_path}")
            self.availability = ModelAvailability.UNAVAILABLE
            self.unavailable_reason = f"Model weights not found at {self.model_path}"
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.availability = ModelAvailability.ERROR
            self.unavailable_reason = f"Model loading error: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if model is available for inference."""
        return self.availability == ModelAvailability.AVAILABLE
    
    def get_status(self) -> dict:
        """Get model status information."""
        return {
            "id": "external_detector_v1",
            "name": "External Defect Detector",
            "type": "object_detection",
            "status": self.availability.value,
            "version": "yolov8n_baseline_v1.0" if self.is_available() else None,
            "reason": self.unavailable_reason if not self.is_available() else None,
        }
    
    def detect(self, image: Image.Image) -> InferenceResult:
        """
        Detect external defects in image.
        
        Args:
            image: PIL Image to analyze
        
        Returns:
            InferenceResult with detection boxes and metadata
        """
        if not self.is_available():
            logger.debug("Defect detection requested but model unavailable")
            return InferenceResult(
                available=False,
                reason=self.unavailable_reason,
                model_version=None,
                confidence=None,
                predictions={"detections": []},
            )
        
        try:
            # Run YOLO inference
            results = self.model.predict(
                source=image,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
            )
            
            # Extract detections
            detections = self._extract_detections(results[0])
            
            # Calculate overall confidence (max detection confidence)
            max_confidence = max([d.confidence for d in detections], default=0.0)
            
            return InferenceResult(
                available=True,
                reason=None,
                model_version="yolov8n_baseline_v1.0",
                confidence=max_confidence if detections else 1.0,  # 1.0 if no defects
                predictions={"detections": detections},
            )
        
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            return InferenceResult(
                available=False,
                reason=f"Inference error: {str(e)}",
                model_version="yolov8n_baseline_v1.0",
                confidence=None,
                predictions=None,
            )
    
    def _extract_detections(self, result) -> list[DetectionBox]:
        """
        Extract detection boxes from YOLO result.
        
        Args:
            result: YOLO prediction result
        
        Returns:
            List of DetectionBox objects
        """
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        # Extract boxes
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            
            detections.append(DetectionBox(
                class_id=int(cls_id),
                class_name=self.CLASS_NAMES.get(int(cls_id), f"Class_{cls_id}"),
                confidence=float(conf),
                x_min=int(x1),
                y_min=int(y1),
                x_max=int(x2),
                y_max=int(y2),
                severity=self.CLASS_SEVERITY.get(int(cls_id), "medium"),
            ))
        
        return detections
