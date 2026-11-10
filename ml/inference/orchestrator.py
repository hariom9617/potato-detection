"""
Inspection orchestrator - coordinates all inference services.
"""

import logging
import time
from typing import Optional
from PIL import Image
from dataclasses import dataclass

from .potato_verifier import PotatoVerifier
from .defect_detector import DefectDetector
from .types import DetectionBox


logger = logging.getLogger(__name__)


@dataclass
class InspectionResult:
    """Complete inspection result from all models."""
    
    # Object verification
    object_status: str  # "POTATO", "NON_POTATO", "UNCERTAIN", "NOT_ASSESSED"
    object_confidence: Optional[float]
    object_model_version: Optional[str]
    object_reason: Optional[str]
    
    # External assessment
    external_status: str  # "HEALTHY", "UNHEALTHY", "UNCERTAIN", "NOT_ASSESSED"
    external_confidence: Optional[float]
    external_model_version: Optional[str]
    external_reason: Optional[str]
    external_detections: list[DetectionBox]
    
    # Internal assessment
    internal_status: str  # "NOT_ASSESSED"
    internal_confidence: Optional[float]
    internal_model_version: Optional[str]
    internal_reason: Optional[str]
    
    # Overall
    overall_status: str
    grade: Optional[str]
    explanation: str
    
    # Metadata
    processing_time_ms: int


class InspectionOrchestrator:
    """
    Coordinates potato object verification, external defect detection,
    and internal spectral analysis.
    
    Determines overall inspection status based on all assessments.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        confidence_threshold: float = 0.5,
        detection_threshold: float = 0.25,
    ):
        """
        Initialize inspection orchestrator.
        
        Args:
            model_path: Path to external defect detector model
            device: Device for inference
            confidence_threshold: Confidence threshold for classification
            detection_threshold: Confidence threshold for detection
        """
        # Initialize sub-services
        self.potato_verifier = PotatoVerifier(model_path=None, device=device)
        self.defect_detector = DefectDetector(
            model_path=model_path,
            device=device,
            confidence_threshold=detection_threshold,
        )
        
        # No internal detector - spectral sensor not available
        self.internal_detector = None
        
        self.confidence_threshold = confidence_threshold
        
        logger.info("InspectionOrchestrator initialized")
    
    def get_model_statuses(self) -> list[dict]:
        """Get status of all models."""
        statuses = [
            self.potato_verifier.get_status(),
            self.defect_detector.get_status(),
            {
                "id": "internal_detector_v1",
                "name": "Internal Defect Detector (Spectral)",
                "type": "classification",
                "status": "unavailable",
                "version": None,
                "reason": "Spectral sensor not available, research phase",
            },
        ]
        return statuses
    
    def inspect(self, image: Image.Image) -> InspectionResult:
        """
        Run complete inspection pipeline.
        
        Args:
            image: PIL Image to inspect
        
        Returns:
            InspectionResult with all assessments
        """
        start_time = time.time()
        
        # 1. Object verification
        object_result = self.potato_verifier.verify(image)
        
        if object_result.available:
            object_status = "POTATO"  # Would come from predictions
            object_confidence = object_result.confidence
            object_model_version = object_result.model_version
            object_reason = None
        else:
            object_status = "NOT_ASSESSED"
            object_confidence = None
            object_model_version = None
            object_reason = object_result.reason
        
        # 2. External defect detection
        external_result = self.defect_detector.detect(image)
        
        if external_result.available:
            detections = external_result.predictions["detections"]
            
            if len(detections) == 0:
                # No defects detected
                external_status = "HEALTHY"
                external_confidence = external_result.confidence
            elif any(d.severity == "critical" for d in detections):
                # Critical defects present
                external_status = "UNHEALTHY"
                external_confidence = external_result.confidence
            else:
                # Non-critical defects
                external_status = "UNHEALTHY"
                external_confidence = external_result.confidence
            
            external_model_version = external_result.model_version
            external_reason = None
            external_detections = detections
        else:
            external_status = "NOT_ASSESSED"
            external_confidence = None
            external_model_version = None
            external_reason = external_result.reason
            external_detections = []
        
        # 3. Internal assessment (always NOT_ASSESSED)
        internal_status = "NOT_ASSESSED"
        internal_confidence = None
        internal_model_version = None
        internal_reason = "Spectral sensor not available"
        
        # 4. Determine overall status
        overall_status, grade, explanation = self._determine_overall_status(
            object_status=object_status,
            external_status=external_status,
            internal_status=internal_status,
            detections=external_detections,
        )
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return InspectionResult(
            object_status=object_status,
            object_confidence=object_confidence,
            object_model_version=object_model_version,
            object_reason=object_reason,
            external_status=external_status,
            external_confidence=external_confidence,
            external_model_version=external_model_version,
            external_reason=external_reason,
            external_detections=external_detections,
            internal_status=internal_status,
            internal_confidence=internal_confidence,
            internal_model_version=internal_model_version,
            internal_reason=internal_reason,
            overall_status=overall_status,
            grade=grade,
            explanation=explanation,
            processing_time_ms=processing_time_ms,
        )
    
    def _determine_overall_status(
        self,
        object_status: str,
        external_status: str,
        internal_status: str,
        detections: list[DetectionBox],
    ) -> tuple[str, Optional[str], str]:
        """
        Determine overall inspection status and grade.
        
        Returns:
            Tuple of (overall_status, grade, explanation)
        """
        # If all assessments are NOT_ASSESSED, status is PARTIALLY_ASSESSED
        if (
            object_status == "NOT_ASSESSED"
            and external_status == "NOT_ASSESSED"
            and internal_status == "NOT_ASSESSED"
        ):
            return (
                "PARTIALLY_ASSESSED",
                None,
                "Inspection incomplete: no models available for assessment",
            )
        
        # If external assessment shows UNHEALTHY, overall is UNHEALTHY
        if external_status == "UNHEALTHY":
            # Determine grade based on defects
            critical_defects = [d for d in detections if d.severity == "critical"]
            
            if critical_defects:
                grade = "D"
                defect_names = [d.class_name for d in critical_defects]
                explanation = (
                    f"Rejected: Critical defect detected ({', '.join(defect_names)}). "
                    "Recommend removal from processing line."
                )
            else:
                grade = "C"
                explanation = (
                    f"Quality concern: {len(detections)} defect(s) detected. "
                    "Suitable for feed grade processing only."
                )
            
            return "UNHEALTHY", grade, explanation
        
        # If external assessment is HEALTHY
        if external_status == "HEALTHY":
            # But object verification failed
            if object_status == "NON_POTATO":
                return (
                    "REJECTED",
                    None,
                    "Object verification failed: Not identified as a potato",
                )
            
            # Object is potato or not assessed, no defects
            if object_status == "NOT_ASSESSED" and internal_status == "NOT_ASSESSED":
                return (
                    "PARTIALLY_ASSESSED",
                    "A",
                    "External surface appears healthy. Internal assessment not available.",
                )
            
            return (
                "HEALTHY",
                "A",
                "No defects detected on external surface",
            )
        
        # External assessment uncertain
        if external_status == "UNCERTAIN":
            return (
                "UNCERTAIN",
                None,
                "Defect detection uncertain. Human review recommended.",
            )
        
        # External assessment not performed
        if external_status == "NOT_ASSESSED":
            return (
                "PARTIALLY_ASSESSED",
                None,
                "External assessment not available",
            )
        
        # Default fallback
        return (
            "PARTIALLY_ASSESSED",
            None,
            "Incomplete inspection",
        )
