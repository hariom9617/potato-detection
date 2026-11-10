"""
Inspection endpoints for image upload and analysis.
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status
from typing import Optional
import logging
import uuid
from datetime import datetime
import json

from ....schemas.inspection import InspectionResponse, InspectionMetadata
from ....schemas.common import (
    ImageInfo,
    ObjectVerificationResult,
    ExternalAssessmentResult,
    InternalAssessmentResult,
    Detection,
    BoundingBox,
    DefectSummary,
    DefectSeverity,
)
from ....services.image_validator import ImageValidator
from ....core.errors import ImageValidationError, AppException, InspectionNotFoundError
from ...deps import get_orchestrator


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "",
    response_model=InspectionResponse,
    summary="Create Inspection",
    description="Upload an image and run potato quality inspection",
    status_code=status.HTTP_200_OK,
)
async def create_inspection(
    file: UploadFile = File(..., description="Image file (JPEG or PNG)"),
    metadata: Optional[str] = Form(None, description="JSON metadata (optional)"),
):
    """
    Upload an image and execute complete inspection workflow.
    
    **Request**:
    - `file`: Image file (JPEG or PNG, max 5MB)
    - `metadata`: Optional JSON string with additional context
    
    **Response**:
    - Complete inspection result with all assessments
    - Bounding boxes for detected defects
    - Overall status and grade
    - Model versions and confidence scores
    
    **Errors**:
    - 400: Invalid image format or missing file
    - 413: File too large
    - 422: Image corrupt or dimensions invalid
    - 500: Internal server error
    """
    # Generate inspection ID
    inspection_id = f"insp_{uuid.uuid4().hex}"
    
    logger.info(f"Starting inspection {inspection_id}, file={file.filename}")
    
    try:
        # 1. Validate image
        validator = ImageValidator()
        
        # Read file contents
        file_data = await file.read()
        
        # Validate
        validated_image = validator.validate(file_data, filename=file.filename)
        
        logger.info(
            f"Image validated: {validated_image.format}, "
            f"{validated_image.width}x{validated_image.height}"
        )
        
        # 2. Parse metadata if provided
        parsed_metadata = None
        if metadata:
            try:
                parsed_metadata = json.loads(metadata)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse metadata: {e}")
                # Continue without metadata rather than failing
        
        # 3. Run inspection
        orchestrator = get_orchestrator()
        result = orchestrator.inspect(validated_image.image)
        
        # 4. Build response
        
        # Object verification
        object_verification = ObjectVerificationResult(
            status=result.object_status,  # type: ignore
            confidence=result.object_confidence,
            model_version=result.object_model_version,
            reason=result.object_reason,
        )
        
        # External assessment
        detections = [
            Detection(
                class_id=d.class_id,
                class_name=d.class_name,
                confidence=d.confidence,
                bbox=BoundingBox(
                    x_min=d.x_min,
                    y_min=d.y_min,
                    x_max=d.x_max,
                    y_max=d.y_max,
                ),
                severity=DefectSeverity(d.severity),
            )
            for d in result.external_detections
        ]
        
        # Calculate defect summary
        defect_summary = None
        if detections:
            by_severity = {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
            }
            for d in detections:
                by_severity[d.severity.value] += 1
            
            defect_summary = DefectSummary(
                total_count=len(detections),
                by_severity=by_severity,
            )
        
        external_assessment = ExternalAssessmentResult(
            status=result.external_status,  # type: ignore
            confidence=result.external_confidence,
            model_version=result.external_model_version,
            reason=result.external_reason,
            detections=detections,
            defect_summary=defect_summary,
        )
        
        # Internal assessment
        internal_assessment = InternalAssessmentResult(
            status=result.internal_status,  # type: ignore
            confidence=result.internal_confidence,
            model_version=result.internal_model_version,
            reason=result.internal_reason,
        )
        
        # Image info
        image_info = ImageInfo(
            width=validated_image.width,
            height=validated_image.height,
            format=validated_image.format,
            size_bytes=validated_image.size_bytes,
        )
        
        # Build final response
        response = InspectionResponse(
            inspection_id=inspection_id,
            timestamp=datetime.utcnow(),
            processing_time_ms=result.processing_time_ms,
            image=image_info,
            object_verification=object_verification,
            external_assessment=external_assessment,
            internal_assessment=internal_assessment,
            overall_status=result.overall_status,  # type: ignore
            grade=result.grade,  # type: ignore
            explanation=result.explanation,
            metadata=parsed_metadata,
        )
        
        logger.info(
            f"Inspection {inspection_id} completed: "
            f"status={result.overall_status}, grade={result.grade}, "
            f"detections={len(detections)}"
        )
        
        # TODO: Persist to database if enabled
        
        return response
    
    except (ImageValidationError, AppException):
        # Let these bubble up to the global exception handler
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error in {inspection_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred",
                    "request_id": inspection_id,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
            }
        )


@router.get(
    "/{inspection_id}",
    response_model=InspectionResponse,
    summary="Get Inspection",
    description="Retrieve a completed inspection by ID",
)
async def get_inspection(inspection_id: str):
    """
    Retrieve inspection by ID.
    
    **Note**: Database persistence not yet implemented.
    Returns 404 for all requests in current version.
    """
    # TODO: Implement database retrieval
    logger.info(f"Inspection retrieval requested: {inspection_id}")
    
    raise InspectionNotFoundError(inspection_id)
