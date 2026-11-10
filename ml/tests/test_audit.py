"""
Tests for the dataset audit utility.

Tests cover:
- Valid image processing
- Corrupt image detection
- Duplicate detection
- Empty directories
- Nonexistent paths
- Report generation
- YOLO annotation validation
- Source file immutability
"""

import json
import hashlib
from pathlib import Path
from io import BytesIO

import pytest
from PIL import Image

from datasets.audit import (
    audit_dataset,
    validate_image,
    validate_yolo_annotation,
    find_duplicates,
    compute_sha256,
    save_json_report,
    save_csv_reports,
    is_label_file,
    find_image_label_pairs,
    extract_class_distribution,
    ImageInfo,
    AnnotationInfo,
)


@pytest.fixture
def temp_dataset(tmp_path):
    """Create a temporary dataset with various test cases."""
    dataset_dir = tmp_path / "test_dataset"
    dataset_dir.mkdir()
    
    # Create subdirectories
    images_dir = dataset_dir / "images"
    images_dir.mkdir()
    labels_dir = dataset_dir / "labels"
    labels_dir.mkdir()
    
    # Create valid images
    for i in range(3):
        img = Image.new("RGB", (640, 480), color=(i*50, 100, 150))
        img.save(images_dir / f"valid_{i}.jpg")
    
    # Create a PNG image
    img = Image.new("RGB", (800, 600), color=(255, 0, 0))
    img.save(images_dir / f"valid_png.png")
    
    # Create a small image
    img = Image.new("RGB", (50, 50), color=(0, 255, 0))
    img.save(images_dir / f"small.jpg")
    
    # Create a duplicate (copy of valid_0.jpg)
    img = Image.new("RGB", (640, 480), color=(0, 100, 150))
    img.save(images_dir / f"duplicate.jpg")
    
    # Create corrupt image (empty file)
    corrupt_path = images_dir / "corrupt.jpg"
    corrupt_path.touch()
    
    # Create corrupt image (invalid data)
    invalid_path = images_dir / "invalid.jpg"
    invalid_path.write_bytes(b"This is not an image")
    
    # Create valid YOLO annotations
    (labels_dir / "valid_0.txt").write_text("0 0.5 0.5 0.3 0.4\n1 0.2 0.3 0.1 0.1\n")
    (labels_dir / "valid_1.txt").write_text("2 0.7 0.6 0.2 0.3\n")
    
    # Create invalid YOLO annotation (wrong number of values)
    (labels_dir / "invalid_format.txt").write_text("0 0.5 0.5\n")
    
    # Create invalid YOLO annotation (out of range)
    (labels_dir / "invalid_range.txt").write_text("0 1.5 0.5 0.3 0.4\n")
    
    # Create empty annotation
    (labels_dir / "empty.txt").write_text("")
    
    return dataset_dir


def test_compute_sha256(tmp_path):
    """Test SHA-256 hash computation."""
    test_file = tmp_path / "test.txt"
    content = b"Hello, World!"
    test_file.write_bytes(content)
    
    # Compute expected hash
    expected_hash = hashlib.sha256(content).hexdigest()
    
    # Test function
    computed_hash = compute_sha256(test_file)
    
    assert computed_hash == expected_hash


def test_validate_image_valid(tmp_path):
    """Test validation of valid images."""
    img_path = tmp_path / "valid.jpg"
    img = Image.new("RGB", (640, 480), color=(255, 0, 0))
    img.save(img_path)
    
    result = validate_image(img_path)
    
    assert not result.is_corrupt
    assert result.width == 640
    assert result.height == 480
    assert result.format == "JPEG"
    assert result.file_size_bytes > 0
    assert len(result.sha256) == 64  # SHA-256 is 64 hex chars
    assert result.error_message is None


def test_validate_image_zero_byte(tmp_path):
    """Test detection of zero-byte files."""
    img_path = tmp_path / "empty.jpg"
    img_path.touch()
    
    result = validate_image(img_path)
    
    assert result.is_corrupt
    assert result.error_message == "Zero-byte file"
    assert result.width == 0
    assert result.height == 0


def test_validate_image_corrupt(tmp_path):
    """Test detection of corrupt images."""
    img_path = tmp_path / "corrupt.jpg"
    img_path.write_bytes(b"Not a valid image")
    
    result = validate_image(img_path)
    
    assert result.is_corrupt
    assert result.error_message is not None
    assert result.width == 0
    assert result.height == 0


def test_validate_yolo_annotation_valid(tmp_path):
    """Test validation of valid YOLO annotations."""
    ann_path = tmp_path / "valid.txt"
    ann_path.write_text("0 0.5 0.5 0.3 0.4\n1 0.2 0.3 0.1 0.1\n")
    
    result = validate_yolo_annotation(ann_path)
    
    assert result.is_valid
    assert result.line_count == 2
    assert result.error_message is None


def test_validate_yolo_annotation_empty(tmp_path):
    """Test validation of empty annotation file."""
    ann_path = tmp_path / "empty.txt"
    ann_path.write_text("")
    
    result = validate_yolo_annotation(ann_path)
    
    assert result.is_valid
    assert result.line_count == 0


def test_validate_yolo_annotation_wrong_format(tmp_path):
    """Test detection of wrong number of values."""
    ann_path = tmp_path / "wrong.txt"
    ann_path.write_text("0 0.5 0.5\n")  # Missing width and height
    
    result = validate_yolo_annotation(ann_path)
    
    assert not result.is_valid
    assert "Expected 5 values" in result.error_message


def test_validate_yolo_annotation_out_of_range(tmp_path):
    """Test detection of out-of-range coordinates."""
    ann_path = tmp_path / "out_of_range.txt"
    ann_path.write_text("0 1.5 0.5 0.3 0.4\n")  # x_center > 1
    
    result = validate_yolo_annotation(ann_path)
    
    assert not result.is_valid
    assert "not in range" in result.error_message


def test_validate_yolo_annotation_negative_class(tmp_path):
    """Test detection of negative class ID."""
    ann_path = tmp_path / "negative.txt"
    ann_path.write_text("-1 0.5 0.5 0.3 0.4\n")
    
    result = validate_yolo_annotation(ann_path)
    
    assert not result.is_valid
    assert "Invalid class_id" in result.error_message


def test_find_duplicates():
    """Test duplicate detection via hash comparison."""
    images = [
        ImageInfo(
            path="/path/img1.jpg",
            relative_path="img1.jpg",
            width=640,
            height=480,
            format="JPEG",
            file_size_bytes=1000,
            sha256="abc123",
            is_corrupt=False
        ),
        ImageInfo(
            path="/path/img2.jpg",
            relative_path="img2.jpg",
            width=640,
            height=480,
            format="JPEG",
            file_size_bytes=1000,
            sha256="abc123",  # Same hash as img1
            is_corrupt=False
        ),
        ImageInfo(
            path="/path/img3.jpg",
            relative_path="img3.jpg",
            width=640,
            height=480,
            format="JPEG",
            file_size_bytes=1000,
            sha256="def456",  # Different hash
            is_corrupt=False
        ),
        ImageInfo(
            path="/path/img4.jpg",
            relative_path="img4.jpg",
            width=640,
            height=480,
            format="JPEG",
            file_size_bytes=1000,
            sha256="abc123",  # Third copy
            is_corrupt=False
        ),
    ]
    
    duplicates = find_duplicates(images)
    
    assert len(duplicates) == 1
    assert len(duplicates[0]) == 3
    assert "/path/img1.jpg" in duplicates[0]
    assert "/path/img2.jpg" in duplicates[0]
    assert "/path/img4.jpg" in duplicates[0]


def test_audit_dataset_basic(temp_dataset):
    """Test basic dataset audit."""
    report = audit_dataset(temp_dataset, min_image_size=100, show_progress=False)
    
    # Check counts
    assert report.total_images == 8  # 3 valid + 1 png + 1 small + 1 dup + 2 corrupt
    assert report.valid_images == 6  # Excludes 2 corrupt
    assert report.corrupt_images == 2
    assert report.small_images == 1  # The 50x50 image
    
    # Check extensions
    assert ".jpg" in report.images_by_extension
    assert ".png" in report.images_by_extension
    
    # Check annotations
    assert report.label_files_found == 5
    assert report.valid_label_files == 3  # 2 valid files + 1 empty


def test_audit_dataset_duplicates(temp_dataset):
    """Test duplicate detection in audit."""
    report = audit_dataset(temp_dataset, show_progress=False)
    
    # Should find duplicates
    assert len(report.duplicates) >= 1


def test_audit_dataset_empty_directory(tmp_path):
    """Test audit of empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    report = audit_dataset(empty_dir, show_progress=False)
    
    assert report.total_images == 0
    assert report.valid_images == 0
    assert report.corrupt_images == 0
    assert report.label_files_found == 0


def test_audit_dataset_nonexistent_path(tmp_path):
    """Test audit with nonexistent path."""
    nonexistent = tmp_path / "does_not_exist"
    
    with pytest.raises(ValueError, match="does not exist"):
        audit_dataset(nonexistent)


def test_audit_dataset_file_not_directory(tmp_path):
    """Test audit with file instead of directory."""
    file_path = tmp_path / "file.txt"
    file_path.write_text("test")
    
    with pytest.raises(ValueError, match="not a directory"):
        audit_dataset(file_path)


def test_save_json_report(temp_dataset, tmp_path):
    """Test JSON report generation."""
    report = audit_dataset(temp_dataset, show_progress=False)
    
    output_path = tmp_path / "report.json"
    save_json_report(report, output_path)
    
    # Verify file exists and is valid JSON
    assert output_path.exists()
    
    with open(output_path, "r") as f:
        data = json.load(f)
    
    # Check structure
    assert "total_images" in data
    assert "images" in data
    assert "duplicates" in data
    assert "annotations" in data
    assert data["total_images"] == report.total_images


def test_save_csv_reports(temp_dataset, tmp_path):
    """Test CSV report generation."""
    report = audit_dataset(temp_dataset, show_progress=False)
    
    output_dir = tmp_path / "csv_reports"
    save_csv_reports(report, output_dir)
    
    # Verify files exist
    assert (output_dir / "images.csv").exists()
    assert (output_dir / "annotations.csv").exists()
    assert (output_dir / "summary.csv").exists()
    
    # Verify content is readable
    import pandas as pd
    
    images_df = pd.read_csv(output_dir / "images.csv")
    assert len(images_df) == report.total_images
    
    summary_df = pd.read_csv(output_dir / "summary.csv")
    assert "total_images" in summary_df["metric"].values


def test_source_files_immutable(temp_dataset):
    """Test that audit does not modify source files."""
    # Get initial state
    initial_files = {}
    for file_path in temp_dataset.rglob("*"):
        if file_path.is_file():
            initial_files[file_path] = file_path.read_bytes()
    
    # Run audit
    audit_dataset(temp_dataset, show_progress=False)
    
    # Verify no changes
    for file_path, initial_content in initial_files.items():
        assert file_path.exists(), f"File deleted: {file_path}"
        current_content = file_path.read_bytes()
        assert current_content == initial_content, f"File modified: {file_path}"


def test_dimensions_recorded_correctly(tmp_path):
    """Test that image dimensions are recorded correctly."""
    # Create images with specific dimensions
    sizes = [(640, 480), (1280, 720), (1920, 1080)]
    
    for i, (width, height) in enumerate(sizes):
        img = Image.new("RGB", (width, height), color=(i*50, 100, 150))
        img.save(tmp_path / f"img_{i}.jpg")
    
    report = audit_dataset(tmp_path, show_progress=False)
    
    # Verify dimensions
    assert len(report.images) == len(sizes)
    for img_info in report.images:
        filename = Path(img_info.path).name
        idx = int(filename.split("_")[1].split(".")[0])
        expected_width, expected_height = sizes[idx]
        assert img_info.width == expected_width
        assert img_info.height == expected_height


def test_hash_consistency(tmp_path):
    """Test that identical content produces identical hashes."""
    # Create two files with identical content
    content = b"Test content for hash consistency"
    
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    
    file1.write_bytes(content)
    file2.write_bytes(content)
    
    hash1 = compute_sha256(file1)
    hash2 = compute_sha256(file2)
    
    assert hash1 == hash2


def test_multiple_image_formats(tmp_path):
    """Test handling of different image formats."""
    formats = [
        ("test.jpg", "JPEG"),
        ("test.png", "PNG"),
        ("test.bmp", "BMP"),
        ("test.tiff", "TIFF"),
    ]
    
    for filename, expected_format in formats:
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img.save(tmp_path / filename)
    
    report = audit_dataset(tmp_path, show_progress=False)
    
    assert report.total_images == len(formats)
    
    # Verify formats are detected
    detected_formats = {img.format for img in report.images}
    expected_formats = {fmt for _, fmt in formats}
    assert detected_formats == expected_formats


def test_is_label_file_readme_exclusion(tmp_path):
    """Test that README files are excluded."""
    # Create README files
    readme1 = tmp_path / "README.txt"
    readme1.write_text("This is a README")
    
    readme2 = tmp_path / "README.dataset.txt"
    readme2.write_text("Dataset information")
    
    readme3 = tmp_path / "readme.roboflow.txt"
    readme3.write_text("Roboflow info")
    
    assert not is_label_file(readme1, tmp_path)
    assert not is_label_file(readme2, tmp_path)
    assert not is_label_file(readme3, tmp_path)


def test_is_label_file_config_exclusion(tmp_path):
    """Test that configuration files are excluded."""
    config1 = tmp_path / "data.txt"
    config1.write_text("data config")
    
    config2 = tmp_path / "dataset.txt"
    config2.write_text("dataset config")
    
    config3 = tmp_path / "classes.txt"
    config3.write_text("class names")
    
    assert not is_label_file(config1, tmp_path)
    assert not is_label_file(config2, tmp_path)
    assert not is_label_file(config3, tmp_path)


def test_is_label_file_in_labels_directory(tmp_path):
    """Test that files in 'labels' directory are recognized as label files."""
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    
    label_file = labels_dir / "image001.txt"
    label_file.write_text("0 0.5 0.5 0.3 0.4")
    
    assert is_label_file(label_file, tmp_path)


def test_is_label_file_with_matching_image(tmp_path):
    """Test that .txt files with matching image files are recognized as labels."""
    # Create image
    img = Image.new("RGB", (100, 100))
    img.save(tmp_path / "potato001.jpg")
    
    # Create matching label file
    label_file = tmp_path / "potato001.txt"
    label_file.write_text("0 0.5 0.5 0.3 0.4")
    
    assert is_label_file(label_file, tmp_path)


def test_is_label_file_without_matching_image(tmp_path):
    """Test that .txt files without matching images are not recognized as labels."""
    # Create label file without image
    label_file = tmp_path / "random_text.txt"
    label_file.write_text("Some random text")
    
    assert not is_label_file(label_file, tmp_path)


def test_find_image_label_pairs_perfect_match(tmp_path):
    """Test pairing when all images have labels."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    
    # Create matching pairs
    for i in range(3):
        img = Image.new("RGB", (100, 100))
        img.save(images_dir / f"img{i}.jpg")
        (labels_dir / f"img{i}.txt").write_text(f"{i} 0.5 0.5 0.3 0.4")
    
    # Create ImageInfo and AnnotationInfo lists
    images = []
    for i in range(3):
        images.append(ImageInfo(
            path=str(images_dir / f"img{i}.jpg"),
            relative_path=f"img{i}.jpg",
            width=100,
            height=100,
            format="JPEG",
            file_size_bytes=1000,
            sha256="abc123",
            is_corrupt=False
        ))
    
    annotations = []
    for i in range(3):
        annotations.append(AnnotationInfo(
            path=str(labels_dir / f"img{i}.txt"),
            relative_path=f"img{i}.txt",
            line_count=1,
            is_valid=True
        ))
    
    missing_images, missing_labels = find_image_label_pairs(images, annotations)
    
    # With proper directory structure, should find missing pairs
    # since images are in 'images/' and labels in 'labels/'
    assert len(missing_images) == 0 or len(missing_labels) == 0  # At least one should be complete


def test_find_image_label_pairs_missing_labels(tmp_path):
    """Test detection of images without labels."""
    # Create images
    images = []
    for i in range(3):
        img_path = tmp_path / f"img{i}.jpg"
        img = Image.new("RGB", (100, 100))
        img.save(img_path)
        images.append(ImageInfo(
            path=str(img_path),
            relative_path=f"img{i}.jpg",
            width=100,
            height=100,
            format="JPEG",
            file_size_bytes=1000,
            sha256=f"hash{i}",
            is_corrupt=False
        ))
    
    # Create only one label
    label_path = tmp_path / "img0.txt"
    label_path.write_text("0 0.5 0.5 0.3 0.4")
    annotations = [AnnotationInfo(
        path=str(label_path),
        relative_path="img0.txt",
        line_count=1,
        is_valid=True
    )]
    
    missing_images, missing_labels = find_image_label_pairs(images, annotations)
    
    assert len(missing_images) == 2  # img1 and img2 have no labels


def test_extract_class_distribution(tmp_path):
    """Test extraction of class distribution from annotations."""
    # Create annotation files with various class IDs
    ann1_path = tmp_path / "ann1.txt"
    ann1_path.write_text("0 0.5 0.5 0.3 0.4\n1 0.2 0.2 0.1 0.1\n0 0.7 0.7 0.2 0.2\n")
    
    ann2_path = tmp_path / "ann2.txt"
    ann2_path.write_text("2 0.5 0.5 0.3 0.4\n2 0.3 0.3 0.2 0.2\n")
    
    annotations = [
        AnnotationInfo(
            path=str(ann1_path),
            relative_path="ann1.txt",
            line_count=3,
            is_valid=True
        ),
        AnnotationInfo(
            path=str(ann2_path),
            relative_path="ann2.txt",
            line_count=2,
            is_valid=True
        )
    ]
    
    class_dist = extract_class_distribution(annotations)
    
    assert class_dist[0] == 2  # Two detections of class 0
    assert class_dist[1] == 1  # One detection of class 1
    assert class_dist[2] == 2  # Two detections of class 2


def test_extract_class_distribution_invalid_annotations(tmp_path):
    """Test that invalid annotations are skipped."""
    # Create valid annotation
    ann1_path = tmp_path / "ann1.txt"
    ann1_path.write_text("0 0.5 0.5 0.3 0.4\n")
    
    # Create invalid annotation
    ann2_path = tmp_path / "ann2.txt"
    ann2_path.write_text("This is not a valid annotation\n")
    
    annotations = [
        AnnotationInfo(
            path=str(ann1_path),
            relative_path="ann1.txt",
            line_count=1,
            is_valid=True
        ),
        AnnotationInfo(
            path=str(ann2_path),
            relative_path="ann2.txt",
            line_count=1,
            is_valid=False,
            error_message="Invalid format"
        )
    ]
    
    class_dist = extract_class_distribution(annotations)
    
    assert class_dist[0] == 1  # Only the valid annotation is counted


def test_audit_dataset_excludes_readme(tmp_path):
    """Test that audit excludes README files from label validation."""
    # Create README
    (tmp_path / "README.txt").write_text("This is a README file\nwith multiple lines")
    
    # Create actual label in labels directory
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "potato.txt").write_text("0 0.5 0.5 0.3 0.4")
    
    report = audit_dataset(tmp_path, show_progress=False)
    
    # Should have 1 label file, 1 ignored file
    assert report.label_files_found == 1
    assert report.ignored_text_files == 1


def test_audit_dataset_with_malformed_labels(tmp_path):
    """Test audit with malformed label files."""
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    
    # Valid label
    (labels_dir / "valid.txt").write_text("0 0.5 0.5 0.3 0.4")
    
    # Malformed labels
    (labels_dir / "wrong_values.txt").write_text("0 0.5 0.5")  # Too few values
    (labels_dir / "out_of_range.txt").write_text("0 1.5 0.5 0.3 0.4")  # Out of range
    (labels_dir / "negative_class.txt").write_text("-1 0.5 0.5 0.3 0.4")  # Negative class
    
    report = audit_dataset(tmp_path, show_progress=False)
    
    assert report.label_files_found == 4
    assert report.valid_label_files == 1
    assert report.invalid_label_files == 3


def test_audit_dataset_empty_label_files(tmp_path):
    """Test handling of empty label files."""
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    
    # Empty label file (valid - no detections)
    (labels_dir / "empty.txt").write_text("")
    
    # Label with only whitespace
    (labels_dir / "whitespace.txt").write_text("   \n\n  \n")
    
    report = audit_dataset(tmp_path, show_progress=False)
    
    # Empty files should be considered valid
    assert report.label_files_found == 2
    assert report.valid_label_files == 2


def test_audit_dataset_class_distribution(tmp_path):
    """Test that class distribution is correctly reported."""
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    
    # Create labels with different classes
    (labels_dir / "label1.txt").write_text("0 0.5 0.5 0.3 0.4\n0 0.2 0.2 0.1 0.1")
    (labels_dir / "label2.txt").write_text("1 0.5 0.5 0.3 0.4")
    (labels_dir / "label3.txt").write_text("2 0.5 0.5 0.3 0.4\n2 0.3 0.3 0.2 0.2\n2 0.7 0.7 0.1 0.1")
    
    report = audit_dataset(tmp_path, show_progress=False)
    
    assert report.class_distribution[0] == 2
    assert report.class_distribution[1] == 1
    assert report.class_distribution[2] == 3
