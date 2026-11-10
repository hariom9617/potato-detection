"""
Tests for format-aware YOLO annotation validation.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from datasets.validation import (
    AnnotationFormat,
    detect_annotation_format,
    validate_detection_line,
    validate_segmentation_line,
    validate_annotation_file,
    segmentation_to_detection,
    detection_to_segmentation
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


class TestFormatDetection:
    """Test annotation format detection."""
    
    def test_detect_empty_file(self, temp_dir):
        """Test detection of empty annotation file."""
        empty_file = temp_dir / "empty.txt"
        empty_file.write_text("")
        assert detect_annotation_format(empty_file) == AnnotationFormat.EMPTY
    
    def test_detect_detection_format(self, temp_dir):
        """Test detection of pure detection format."""
        det_file = temp_dir / "detection.txt"
        det_file.write_text("0 0.5 0.5 0.2 0.3\n1 0.3 0.4 0.1 0.15\n")
        assert detect_annotation_format(det_file) == AnnotationFormat.DETECTION
    
    def test_detect_segmentation_format(self, temp_dir):
        """Test detection of pure segmentation format."""
        seg_file = temp_dir / "segmentation.txt"
        seg_file.write_text("0 0.1 0.2 0.3 0.4 0.5 0.6\n")
        assert detect_annotation_format(seg_file) == AnnotationFormat.SEGMENTATION
    
    def test_detect_mixed_format(self, temp_dir):
        """Test detection of mixed format file."""
        mixed_file = temp_dir / "mixed.txt"
        mixed_file.write_text("0 0.5 0.5 0.2 0.3\n1 0.1 0.2 0.3 0.4 0.5 0.6\n")
        assert detect_annotation_format(mixed_file) == AnnotationFormat.MIXED
    
    def test_detect_invalid_format(self, temp_dir):
        """Test detection of invalid format."""
        invalid_file = temp_dir / "invalid.txt"
        invalid_file.write_text("0 0.5 0.6\n")  # Only 3 values
        assert detect_annotation_format(invalid_file) == AnnotationFormat.INVALID
    
    def test_detect_whitespace_lines(self, temp_dir):
        """Test format detection with whitespace lines."""
        file_path = temp_dir / "whitespace.txt"
        file_path.write_text("\n0 0.5 0.5 0.2 0.3\n\n1 0.3 0.4 0.1 0.15\n\n")
        assert detect_annotation_format(file_path) == AnnotationFormat.DETECTION


class TestDetectionValidation:
    """Test detection format validation."""
    
    def test_valid_detection_line(self):
        """Test validation of valid detection line."""
        result = validate_detection_line(1, ["0", "0.5", "0.5", "0.2", "0.3"], "0 0.5 0.5 0.2 0.3")
        assert result.is_valid
        assert result.class_id == 0
        assert result.values == [0.5, 0.5, 0.2, 0.3]
        assert result.format == AnnotationFormat.DETECTION
    
    def test_invalid_value_count(self):
        """Test detection line with wrong value count."""
        result = validate_detection_line(1, ["0", "0.5", "0.5"], "0 0.5 0.5")
        assert not result.is_valid
        assert "Expected 5 values" in result.error_message
    
    def test_invalid_class_id_negative(self):
        """Test detection line with negative class ID."""
        result = validate_detection_line(1, ["-1", "0.5", "0.5", "0.2", "0.3"], "-1 0.5 0.5 0.2 0.3")
        assert not result.is_valid
        assert "out of range" in result.error_message
    
    def test_invalid_class_id_too_large(self):
        """Test detection line with class ID exceeding maximum."""
        result = validate_detection_line(1, ["1000", "0.5", "0.5", "0.2", "0.3"], "1000 0.5 0.5 0.2 0.3", max_class_id=999)
        assert not result.is_valid
        assert "out of range" in result.error_message
    
    def test_coordinate_out_of_range_negative(self):
        """Test detection line with negative coordinate."""
        result = validate_detection_line(1, ["0", "-0.1", "0.5", "0.2", "0.3"], "0 -0.1 0.5 0.2 0.3")
        assert not result.is_valid
        assert "not in range [0, 1]" in result.error_message
    
    def test_coordinate_out_of_range_above_one(self):
        """Test detection line with coordinate > 1."""
        result = validate_detection_line(1, ["0", "0.5", "1.1", "0.2", "0.3"], "0 0.5 1.1 0.2 0.3")
        assert not result.is_valid
        assert "not in range [0, 1]" in result.error_message
    
    def test_zero_width(self):
        """Test detection line with zero width."""
        result = validate_detection_line(1, ["0", "0.5", "0.5", "0.0", "0.3"], "0 0.5 0.5 0.0 0.3")
        assert not result.is_valid
        assert "must be positive" in result.error_message
    
    def test_zero_height(self):
        """Test detection line with zero height."""
        result = validate_detection_line(1, ["0", "0.5", "0.5", "0.2", "0.0"], "0 0.5 0.5 0.2 0.0")
        assert not result.is_valid
        assert "must be positive" in result.error_message
    
    def test_invalid_number_format(self):
        """Test detection line with non-numeric values."""
        result = validate_detection_line(1, ["0", "abc", "0.5", "0.2", "0.3"], "0 abc 0.5 0.2 0.3")
        assert not result.is_valid
        assert "Cannot parse" in result.error_message


class TestSegmentationValidation:
    """Test segmentation format validation."""
    
    def test_valid_segmentation_line(self):
        """Test validation of valid segmentation line."""
        result = validate_segmentation_line(
            1,
            ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6"],
            "0 0.1 0.2 0.3 0.4 0.5 0.6"
        )
        assert result.is_valid
        assert result.class_id == 0
        assert len(result.values) == 6
        assert result.format == AnnotationFormat.SEGMENTATION
    
    def test_valid_complex_polygon(self):
        """Test validation of polygon with many points."""
        coords = ["0"] + [str(i * 0.01) for i in range(100)]  # 50 points
        line = " ".join(coords)
        result = validate_segmentation_line(1, coords, line)
        assert result.is_valid
        assert len(result.values) == 100
    
    def test_invalid_even_value_count(self):
        """Test segmentation with even number of values (missing coordinate)."""
        result = validate_segmentation_line(
            1,
            ["0", "0.1", "0.2", "0.3", "0.4", "0.5"],  # 6 values (even)
            "0 0.1 0.2 0.3 0.4 0.5"
        )
        assert not result.is_valid
        assert "even number of coordinates" in result.error_message
    
    def test_insufficient_points(self):
        """Test segmentation with less than minimum points."""
        result = validate_segmentation_line(
            1,
            ["0", "0.1", "0.2", "0.3", "0.4"],  # Only 2 points
            "0 0.1 0.2 0.3 0.4",
            min_points=3
        )
        assert not result.is_valid
        assert "at least 3 points" in result.error_message
    
    def test_invalid_class_id(self):
        """Test segmentation with invalid class ID."""
        result = validate_segmentation_line(
            1,
            ["-1", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6"],
            "-1 0.1 0.2 0.3 0.4 0.5 0.6"
        )
        assert not result.is_valid
        assert "out of range" in result.error_message
    
    def test_coordinate_out_of_range(self):
        """Test segmentation with coordinate out of range."""
        result = validate_segmentation_line(
            1,
            ["0", "1.5", "0.2", "0.3", "0.4", "0.5", "0.6"],
            "0 1.5 0.2 0.3 0.4 0.5 0.6"
        )
        assert not result.is_valid
        assert "not in range [0, 1]" in result.error_message
        assert "Point 1 x=" in result.error_message
    
    def test_invalid_number_format(self):
        """Test segmentation with non-numeric values."""
        result = validate_segmentation_line(
            1,
            ["0", "abc", "0.2", "0.3", "0.4", "0.5", "0.6"],
            "0 abc 0.2 0.3 0.4 0.5 0.6"
        )
        assert not result.is_valid
        assert "Cannot parse" in result.error_message


class TestFileValidation:
    """Test complete file validation."""
    
    def test_validate_empty_file(self, temp_dir):
        """Test validation of empty file."""
        empty_file = temp_dir / "empty.txt"
        empty_file.write_text("")
        result = validate_annotation_file(empty_file)
        assert result.is_valid  # Empty is valid (no annotations)
        assert result.format == AnnotationFormat.EMPTY
        assert result.line_count == 0
    
    def test_validate_pure_detection_file(self, temp_dir):
        """Test validation of pure detection format file."""
        det_file = temp_dir / "detection.txt"
        det_file.write_text("0 0.5 0.5 0.2 0.3\n1 0.3 0.4 0.1 0.15\n")
        result = validate_annotation_file(det_file)
        assert result.is_valid
        assert result.format == AnnotationFormat.DETECTION
        assert result.line_count == 2
        assert result.valid_line_count == 2
        assert result.invalid_line_count == 0
        assert set(result.class_ids) == {0, 1}
    
    def test_validate_pure_segmentation_file(self, temp_dir):
        """Test validation of pure segmentation format file."""
        seg_file = temp_dir / "segmentation.txt"
        seg_file.write_text("0 0.1 0.2 0.3 0.4 0.5 0.6\n2 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9\n")
        result = validate_annotation_file(seg_file)
        assert result.is_valid
        assert result.format == AnnotationFormat.SEGMENTATION
        assert result.line_count == 2
        assert result.valid_line_count == 2
        assert set(result.class_ids) == {0, 2}
    
    def test_validate_mixed_format_allowed(self, temp_dir):
        """Test validation of mixed format file when allowed."""
        mixed_file = temp_dir / "mixed.txt"
        mixed_file.write_text("0 0.5 0.5 0.2 0.3\n1 0.1 0.2 0.3 0.4 0.5 0.6\n")
        result = validate_annotation_file(mixed_file, allow_mixed_format=True)
        assert result.is_valid
        assert result.format == AnnotationFormat.MIXED
        assert result.line_count == 2
        assert result.valid_line_count == 2
    
    def test_validate_mixed_format_disallowed(self, temp_dir):
        """Test validation of mixed format file when not allowed."""
        mixed_file = temp_dir / "mixed.txt"
        mixed_file.write_text("0 0.5 0.5 0.2 0.3\n1 0.1 0.2 0.3 0.4 0.5 0.6\n")
        result = validate_annotation_file(mixed_file, allow_mixed_format=False)
        assert not result.is_valid
        assert result.format == AnnotationFormat.MIXED
        assert "not allowed" in result.error_message
    
    def test_validate_file_with_errors(self, temp_dir):
        """Test validation of file with some invalid lines."""
        bad_file = temp_dir / "bad.txt"
        # Use invalid coordinate value instead of unparseable data
        bad_file.write_text("0 0.5 0.5 0.2 0.3\n1 1.5 0.4 0.1 0.15\n2 0.3 0.4 0.1 0.15\n")
        result = validate_annotation_file(bad_file)
        assert not result.is_valid
        assert result.line_count == 3
        assert result.valid_line_count == 2
        assert result.invalid_line_count == 1
        assert "Line 2:" in result.error_message
    
    def test_validate_file_with_whitespace(self, temp_dir):
        """Test validation handles whitespace correctly."""
        file_path = temp_dir / "whitespace.txt"
        file_path.write_text("\n\n0 0.5 0.5 0.2 0.3\n\n1 0.3 0.4 0.1 0.15\n\n")
        result = validate_annotation_file(file_path)
        assert result.is_valid
        assert result.line_count == 2  # Empty lines not counted
    
    def test_validate_nonexistent_file(self, temp_dir):
        """Test validation of non-existent file."""
        nonexistent = temp_dir / "nonexistent.txt"
        result = validate_annotation_file(nonexistent)
        assert not result.is_valid
        # File read error produces "invalid" format detection
        assert result.format == AnnotationFormat.INVALID


class TestFormatConversion:
    """Test conversion between detection and segmentation formats."""
    
    def test_segmentation_to_detection_basic(self):
        """Test converting simple rectangle polygon to bounding box."""
        # Rectangle: (0.2, 0.3), (0.6, 0.3), (0.6, 0.7), (0.2, 0.7)
        coords = [0.2, 0.3, 0.6, 0.3, 0.6, 0.7, 0.2, 0.7]
        x_c, y_c, w, h = segmentation_to_detection(coords)
        
        assert abs(x_c - 0.4) < 1e-6  # Center x
        assert abs(y_c - 0.5) < 1e-6  # Center y
        assert abs(w - 0.4) < 1e-6    # Width
        assert abs(h - 0.4) < 1e-6    # Height
    
    def test_segmentation_to_detection_triangle(self):
        """Test converting triangle to bounding box."""
        # Triangle vertices
        coords = [0.1, 0.1, 0.9, 0.1, 0.5, 0.9]
        x_c, y_c, w, h = segmentation_to_detection(coords)
        
        assert abs(x_c - 0.5) < 1e-6  # Center x
        assert abs(y_c - 0.5) < 1e-6  # Center y
        assert abs(w - 0.8) < 1e-6    # Width
        assert abs(h - 0.8) < 1e-6    # Height
    
    def test_segmentation_to_detection_insufficient_points(self):
        """Test conversion with insufficient points."""
        coords = [0.1, 0.2, 0.3, 0.4]  # Only 2 points
        with pytest.raises(ValueError, match="at least 3 points"):
            segmentation_to_detection(coords)
    
    def test_segmentation_to_detection_odd_coordinates(self):
        """Test conversion with odd number of coordinates."""
        coords = [0.1, 0.2, 0.3]  # Odd count
        with pytest.raises(ValueError, match="at least 3 points|must be in pairs"):
            segmentation_to_detection(coords)
    
    def test_detection_to_segmentation_basic(self):
        """Test converting bounding box to rectangle polygon."""
        x_c, y_c, w, h = 0.5, 0.5, 0.4, 0.3
        coords = detection_to_segmentation(x_c, y_c, w, h)
        
        assert len(coords) == 8  # 4 corners
        # Check bounds
        x_coords = [coords[i] for i in range(0, 8, 2)]
        y_coords = [coords[i] for i in range(1, 8, 2)]
        
        assert abs(min(x_coords) - 0.3) < 1e-6
        assert abs(max(x_coords) - 0.7) < 1e-6
        assert abs(min(y_coords) - 0.35) < 1e-6
        assert abs(max(y_coords) - 0.65) < 1e-6
    
    def test_detection_to_segmentation_full_image(self):
        """Test converting full-image bounding box."""
        x_c, y_c, w, h = 0.5, 0.5, 1.0, 1.0
        coords = detection_to_segmentation(x_c, y_c, w, h)
        
        x_coords = [coords[i] for i in range(0, 8, 2)]
        y_coords = [coords[i] for i in range(1, 8, 2)]
        
        assert abs(min(x_coords) - 0.0) < 1e-6
        assert abs(max(x_coords) - 1.0) < 1e-6
        assert abs(min(y_coords) - 0.0) < 1e-6
        assert abs(max(y_coords) - 1.0) < 1e-6
    
    def test_roundtrip_conversion(self):
        """Test detection -> segmentation -> detection roundtrip."""
        x_c_orig, y_c_orig, w_orig, h_orig = 0.4, 0.6, 0.3, 0.2
        
        # Convert to segmentation
        seg_coords = detection_to_segmentation(x_c_orig, y_c_orig, w_orig, h_orig)
        
        # Convert back to detection
        x_c, y_c, w, h = segmentation_to_detection(seg_coords)
        
        # Should match original
        assert abs(x_c - x_c_orig) < 1e-6
        assert abs(y_c - y_c_orig) < 1e-6
        assert abs(w - w_orig) < 1e-6
        assert abs(h - h_orig) < 1e-6


class TestRealWorldScenarios:
    """Test scenarios based on actual dataset findings."""
    
    def test_real_detection_annotation(self, temp_dir):
        """Test with real detection annotation from dataset."""
        file_path = temp_dir / "real_det.txt"
        file_path.write_text("7 0.70078125 0.7453125 0.15 0.1234375\n")
        result = validate_annotation_file(file_path)
        assert result.is_valid
        assert result.format == AnnotationFormat.DETECTION
    
    def test_real_segmentation_annotation(self, temp_dir):
        """Test with real segmentation annotation from dataset."""
        # Example from Abnormal_16 file (simplified)
        seg_line = "0 0.5359375 0.425 0.5671875 0.4390625 0.5734375 0.4515625 0.5890625 0.45 0.590625 0.4640625\n"
        file_path = temp_dir / "real_seg.txt"
        file_path.write_text(seg_line)
        result = validate_annotation_file(file_path)
        assert result.is_valid
        assert result.format == AnnotationFormat.SEGMENTATION
    
    def test_real_mixed_annotation(self, temp_dir):
        """Test with real mixed annotation from dataset (Abnormal_16)."""
        content = """0 0.5359375 0.425 0.5671875 0.4390625 0.5734375 0.4515625
0 0.2046875 0.8421875 0.2171875 0.846875 0.240625 0.834375
7 0.61953125 0.6265625 0.08828125 0.12265625
2 0.28515625 0.18359375 0.0703125 0.0703125
"""
        file_path = temp_dir / "real_mixed.txt"
        file_path.write_text(content)
        result = validate_annotation_file(file_path)
        assert result.is_valid
        assert result.format == AnnotationFormat.MIXED
        assert result.line_count == 4
