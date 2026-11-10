"""
Format-Aware YOLO Annotation Validation

Supports both YOLO detection (bounding box) and segmentation (polygon) formats.
Provides comprehensive validation, format detection, and conversion utilities.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple


class AnnotationFormat(Enum):
    """YOLO annotation format types."""
    DETECTION = "detection"  # Bounding box: class_id x_center y_center width height
    SEGMENTATION = "segmentation"  # Polygon: class_id x1 y1 x2 y2 ... xn yn
    MIXED = "mixed"  # File contains both formats
    INVALID = "invalid"  # File format is invalid
    EMPTY = "empty"  # File is empty or contains only whitespace


@dataclass
class AnnotationLine:
    """Parsed YOLO annotation line."""
    line_number: int
    class_id: int
    values: List[float]  # For detection: [x, y, w, h]; for segmentation: [x1, y1, x2, y2, ...]
    format: AnnotationFormat
    raw_line: str
    is_valid: bool
    error_message: Optional[str] = None


@dataclass
class AnnotationValidationResult:
    """Complete validation result for an annotation file."""
    path: str
    relative_path: str
    format: AnnotationFormat
    line_count: int
    valid_line_count: int
    invalid_line_count: int
    is_valid: bool
    lines: List[AnnotationLine]
    error_message: Optional[str] = None
    class_ids: List[int] = None  # Unique class IDs in this file
    
    def __post_init__(self):
        if self.class_ids is None:
            self.class_ids = list(set(line.class_id for line in self.lines if line.is_valid))


def detect_annotation_format(annotation_path: Path) -> AnnotationFormat:
    """
    Detect the format of a YOLO annotation file.
    
    Args:
        annotation_path: Path to the annotation file
        
    Returns:
        AnnotationFormat enum value
    """
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if not lines:
            return AnnotationFormat.EMPTY
        
        formats_seen = set()
        
        for line in lines:
            parts = line.split()
            value_count = len(parts)
            
            if value_count == 5:
                formats_seen.add(AnnotationFormat.DETECTION)
            elif value_count > 5 and value_count % 2 == 1:
                # Segmentation: class_id + even number of coordinates
                formats_seen.add(AnnotationFormat.SEGMENTATION)
            else:
                return AnnotationFormat.INVALID
        
        if len(formats_seen) == 0:
            return AnnotationFormat.INVALID
        elif len(formats_seen) == 1:
            return list(formats_seen)[0]
        else:
            return AnnotationFormat.MIXED
    
    except Exception:
        return AnnotationFormat.INVALID


def validate_detection_line(
    line_number: int,
    parts: List[str],
    raw_line: str,
    max_class_id: int = 999
) -> AnnotationLine:
    """
    Validate a YOLO detection format line.
    
    Format: class_id x_center y_center width height
    
    Args:
        line_number: Line number in file (1-indexed)
        parts: Split line parts
        raw_line: Original line text
        max_class_id: Maximum allowed class ID
        
    Returns:
        AnnotationLine object with validation result
    """
    if len(parts) != 5:
        return AnnotationLine(
            line_number=line_number,
            class_id=-1,
            values=[],
            format=AnnotationFormat.INVALID,
            raw_line=raw_line,
            is_valid=False,
            error_message=f"Expected 5 values for detection format, got {len(parts)}"
        )
    
    try:
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        # Validate class ID
        if class_id < 0 or class_id > max_class_id:
            return AnnotationLine(
                line_number=line_number,
                class_id=class_id,
                values=[x_center, y_center, width, height],
                format=AnnotationFormat.DETECTION,
                raw_line=raw_line,
                is_valid=False,
                error_message=f"Class ID {class_id} out of range [0, {max_class_id}]"
            )
        
        # Validate normalized coordinates [0, 1]
        coords = {
            'x_center': x_center,
            'y_center': y_center,
            'width': width,
            'height': height
        }
        
        for name, value in coords.items():
            if value < 0 or value > 1:
                return AnnotationLine(
                    line_number=line_number,
                    class_id=class_id,
                    values=[x_center, y_center, width, height],
                    format=AnnotationFormat.DETECTION,
                    raw_line=raw_line,
                    is_valid=False,
                    error_message=f"{name}={value:.4f} not in range [0, 1]"
                )
        
        # Additional sanity checks
        if width <= 0 or height <= 0:
            return AnnotationLine(
                line_number=line_number,
                class_id=class_id,
                values=[x_center, y_center, width, height],
                format=AnnotationFormat.DETECTION,
                raw_line=raw_line,
                is_valid=False,
                error_message=f"Width and height must be positive (w={width:.4f}, h={height:.4f})"
            )
        
        return AnnotationLine(
            line_number=line_number,
            class_id=class_id,
            values=[x_center, y_center, width, height],
            format=AnnotationFormat.DETECTION,
            raw_line=raw_line,
            is_valid=True
        )
    
    except ValueError as e:
        return AnnotationLine(
            line_number=line_number,
            class_id=-1,
            values=[],
            format=AnnotationFormat.INVALID,
            raw_line=raw_line,
            is_valid=False,
            error_message=f"Cannot parse values: {str(e)}"
        )


def validate_segmentation_line(
    line_number: int,
    parts: List[str],
    raw_line: str,
    max_class_id: int = 999,
    min_points: int = 3
) -> AnnotationLine:
    """
    Validate a YOLO segmentation format line.
    
    Format: class_id x1 y1 x2 y2 x3 y3 ... xn yn
    
    Args:
        line_number: Line number in file (1-indexed)
        parts: Split line parts
        raw_line: Original line text
        max_class_id: Maximum allowed class ID
        min_points: Minimum number of polygon points
        
    Returns:
        AnnotationLine object with validation result
    """
    value_count = len(parts)
    
    # Must have class_id + even number of coordinates
    if value_count < 2 or value_count % 2 != 1:
        return AnnotationLine(
            line_number=line_number,
            class_id=-1,
            values=[],
            format=AnnotationFormat.INVALID,
            raw_line=raw_line,
            is_valid=False,
            error_message=f"Segmentation format requires class_id + even number of coordinates, got {value_count} values"
        )
    
    try:
        class_id = int(parts[0])
        coordinates = [float(x) for x in parts[1:]]
        
        # Validate class ID
        if class_id < 0 or class_id > max_class_id:
            return AnnotationLine(
                line_number=line_number,
                class_id=class_id,
                values=coordinates,
                format=AnnotationFormat.SEGMENTATION,
                raw_line=raw_line,
                is_valid=False,
                error_message=f"Class ID {class_id} out of range [0, {max_class_id}]"
            )
        
        # Validate minimum number of points
        num_points = len(coordinates) // 2
        if num_points < min_points:
            return AnnotationLine(
                line_number=line_number,
                class_id=class_id,
                values=coordinates,
                format=AnnotationFormat.SEGMENTATION,
                raw_line=raw_line,
                is_valid=False,
                error_message=f"Polygon requires at least {min_points} points, got {num_points}"
            )
        
        # Validate all coordinates are in [0, 1]
        for i, coord in enumerate(coordinates):
            if coord < 0 or coord > 1:
                coord_type = 'x' if i % 2 == 0 else 'y'
                point_num = (i // 2) + 1
                return AnnotationLine(
                    line_number=line_number,
                    class_id=class_id,
                    values=coordinates,
                    format=AnnotationFormat.SEGMENTATION,
                    raw_line=raw_line,
                    is_valid=False,
                    error_message=f"Point {point_num} {coord_type}={coord:.4f} not in range [0, 1]"
                )
        
        return AnnotationLine(
            line_number=line_number,
            class_id=class_id,
            values=coordinates,
            format=AnnotationFormat.SEGMENTATION,
            raw_line=raw_line,
            is_valid=True
        )
    
    except ValueError as e:
        return AnnotationLine(
            line_number=line_number,
            class_id=-1,
            values=[],
            format=AnnotationFormat.INVALID,
            raw_line=raw_line,
            is_valid=False,
            error_message=f"Cannot parse values: {str(e)}"
        )


def validate_annotation_file(
    annotation_path: Path,
    max_class_id: int = 999,
    allow_mixed_format: bool = True
) -> AnnotationValidationResult:
    """
    Validate a YOLO annotation file with format auto-detection.
    
    Supports detection, segmentation, and mixed format files.
    
    Args:
        annotation_path: Path to the annotation file
        max_class_id: Maximum allowed class ID
        allow_mixed_format: Whether to allow files with mixed detection/segmentation
        
    Returns:
        AnnotationValidationResult with comprehensive validation details
    """
    relative_path = annotation_path.name
    
    try:
        # Detect format first
        file_format = detect_annotation_format(annotation_path)
        
        if file_format == AnnotationFormat.EMPTY:
            return AnnotationValidationResult(
                path=str(annotation_path),
                relative_path=relative_path,
                format=AnnotationFormat.EMPTY,
                line_count=0,
                valid_line_count=0,
                invalid_line_count=0,
                is_valid=True,  # Empty files are valid (no annotations for this image)
                lines=[],
                error_message=None
            )
        
        if file_format == AnnotationFormat.INVALID:
            return AnnotationValidationResult(
                path=str(annotation_path),
                relative_path=relative_path,
                format=AnnotationFormat.INVALID,
                line_count=0,
                valid_line_count=0,
                invalid_line_count=0,
                is_valid=False,
                lines=[],
                error_message="Unable to determine annotation format"
            )
        
        if file_format == AnnotationFormat.MIXED and not allow_mixed_format:
            return AnnotationValidationResult(
                path=str(annotation_path),
                relative_path=relative_path,
                format=AnnotationFormat.MIXED,
                line_count=0,
                valid_line_count=0,
                invalid_line_count=0,
                is_valid=False,
                lines=[],
                error_message="Mixed format (detection + segmentation) not allowed"
            )
        
        # Read and validate each line
        with open(annotation_path, 'r', encoding='utf-8') as f:
            raw_lines = f.readlines()
        
        validated_lines = []
        line_number = 0
        
        for raw_line in raw_lines:
            stripped = raw_line.strip()
            if not stripped:
                continue  # Skip empty lines
            
            line_number += 1
            parts = stripped.split()
            
            # Determine line format
            if len(parts) == 5:
                line_result = validate_detection_line(line_number, parts, stripped, max_class_id)
            elif len(parts) > 5 and len(parts) % 2 == 1:
                line_result = validate_segmentation_line(line_number, parts, stripped, max_class_id)
            else:
                line_result = AnnotationLine(
                    line_number=line_number,
                    class_id=-1,
                    values=[],
                    format=AnnotationFormat.INVALID,
                    raw_line=stripped,
                    is_valid=False,
                    error_message=f"Invalid format: {len(parts)} values (expected 5 for detection or odd >5 for segmentation)"
                )
            
            validated_lines.append(line_result)
        
        # Aggregate results
        valid_lines = [line for line in validated_lines if line.is_valid]
        invalid_lines = [line for line in validated_lines if not line.is_valid]
        
        is_file_valid = len(invalid_lines) == 0 and len(validated_lines) > 0
        
        # Construct error message if there are invalid lines
        error_msg = None
        if invalid_lines:
            first_error = invalid_lines[0]
            error_msg = f"Line {first_error.line_number}: {first_error.error_message}"
            if len(invalid_lines) > 1:
                error_msg += f" (and {len(invalid_lines) - 1} more errors)"
        
        return AnnotationValidationResult(
            path=str(annotation_path),
            relative_path=relative_path,
            format=file_format,
            line_count=len(validated_lines),
            valid_line_count=len(valid_lines),
            invalid_line_count=len(invalid_lines),
            is_valid=is_file_valid,
            lines=validated_lines,
            error_message=error_msg
        )
    
    except Exception as e:
        return AnnotationValidationResult(
            path=str(annotation_path),
            relative_path=relative_path,
            format=AnnotationFormat.INVALID,
            line_count=0,
            valid_line_count=0,
            invalid_line_count=0,
            is_valid=False,
            lines=[],
            error_message=f"File read error: {str(e)}"
        )


def segmentation_to_detection(coordinates: List[float]) -> Tuple[float, float, float, float]:
    """
    Convert segmentation polygon coordinates to detection bounding box.
    
    Computes the axis-aligned bounding box from polygon points.
    
    Args:
        coordinates: List of [x1, y1, x2, y2, ..., xn, yn] values
        
    Returns:
        Tuple of (x_center, y_center, width, height) in normalized coordinates
    """
    if len(coordinates) < 6:  # Need at least 3 points
        raise ValueError(f"Need at least 3 points (6 coordinates), got {len(coordinates)}")
    
    if len(coordinates) % 2 != 0:
        raise ValueError(f"Coordinates must be in pairs, got {len(coordinates)} values")
    
    # Extract x and y coordinates
    x_coords = [coordinates[i] for i in range(0, len(coordinates), 2)]
    y_coords = [coordinates[i] for i in range(1, len(coordinates), 2)]
    
    # Compute bounding box
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    # Convert to center + width/height format
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    
    return x_center, y_center, width, height


def detection_to_segmentation(x_center: float, y_center: float, width: float, height: float) -> List[float]:
    """
    Convert detection bounding box to segmentation polygon (rectangle).
    
    Creates a 4-point polygon representing the bounding box corners.
    
    Args:
        x_center: Normalized x center coordinate [0, 1]
        y_center: Normalized y center coordinate [0, 1]
        width: Normalized width [0, 1]
        height: Normalized height [0, 1]
        
    Returns:
        List of [x1, y1, x2, y2, x3, y3, x4, y4] for rectangle corners
    """
    x_min = x_center - width / 2
    x_max = x_center + width / 2
    y_min = y_center - height / 2
    y_max = y_center + height / 2
    
    # Four corners: top-left, top-right, bottom-right, bottom-left
    return [
        x_min, y_min,  # Top-left
        x_max, y_min,  # Top-right
        x_max, y_max,  # Bottom-right
        x_min, y_max   # Bottom-left
    ]
