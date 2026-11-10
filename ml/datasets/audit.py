"""
Dataset Audit Utility

Recursively scans a directory for images, validates them, and generates
comprehensive reports without modifying any source files.

Features:
- Discovers JPG, JPEG, PNG, BMP, TIFF, WEBP images
- Records dimensions, format, file size, SHA-256 hash
- Detects corrupt/unreadable images
- Identifies duplicate content via hash comparison
- Warns about small images (configurable threshold)
- Validates YOLO annotation files (syntax only)
- Generates JSON and CSV reports
- Works offline, no internet required
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
from PIL import Image
from tqdm import tqdm


# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# YOLO class ID range (adjust if needed)
YOLO_MAX_CLASS_ID = 999


@dataclass
class ImageInfo:
    """Information about a single image file."""
    
    path: str
    relative_path: str
    width: int
    height: int
    format: str
    file_size_bytes: int
    sha256: str
    is_corrupt: bool
    error_message: Optional[str] = None


@dataclass
class AnnotationInfo:
    """Information about a YOLO annotation file."""
    
    path: str
    relative_path: str
    line_count: int
    is_valid: bool
    error_message: Optional[str] = None


@dataclass
class AuditReport:
    """Complete dataset audit report."""
    
    input_directory: str
    total_images: int
    valid_images: int
    corrupt_images: int
    duplicate_images: int
    small_images: int
    total_size_bytes: int
    images_by_extension: Dict[str, int]
    images_by_directory: Dict[str, int]
    label_files_found: int
    valid_label_files: int
    invalid_label_files: int
    ignored_text_files: int
    missing_image_pairs: int
    missing_label_pairs: int
    class_distribution: Dict[int, int]
    images: List[ImageInfo]
    duplicates: List[List[str]]
    annotations: List[AnnotationInfo]
    ignored_files: List[str]


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def validate_image(image_path: Path) -> ImageInfo:
    """
    Validate an image file and extract metadata.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        ImageInfo object with validation results
    """
    relative_path = image_path.name
    
    # Check if file is readable and has content
    if image_path.stat().st_size == 0:
        return ImageInfo(
            path=str(image_path),
            relative_path=relative_path,
            width=0,
            height=0,
            format="unknown",
            file_size_bytes=0,
            sha256="",
            is_corrupt=True,
            error_message="Zero-byte file"
        )
    
    try:
        # Compute hash
        sha256 = compute_sha256(image_path)
        
        # Open and validate image
        with Image.open(image_path) as img:
            width, height = img.size
            format_name = img.format or "unknown"
            
        return ImageInfo(
            path=str(image_path),
            relative_path=relative_path,
            width=width,
            height=height,
            format=format_name,
            file_size_bytes=image_path.stat().st_size,
            sha256=sha256,
            is_corrupt=False,
            error_message=None
        )
        
    except Exception as e:
        return ImageInfo(
            path=str(image_path),
            relative_path=relative_path,
            width=0,
            height=0,
            format="unknown",
            file_size_bytes=image_path.stat().st_size,
            sha256="",
            is_corrupt=True,
            error_message=str(e)
        )


def is_label_file(txt_path: Path, input_dir: Path) -> bool:
    """
    Determine if a .txt file is a YOLO label file.
    
    A file is considered a label file if:
    1. It's in a directory named 'labels', OR
    2. It has a corresponding image file with the same stem
    
    It's NOT a label file if:
    - Filename is README (case-insensitive)
    - Filename contains 'readme' (case-insensitive)
    - Filename is 'data.yaml', 'dataset.yaml', etc.
    - Filename ends with '.roboflow.txt'
    
    Args:
        txt_path: Path to the .txt file
        input_dir: Root directory of the dataset
        
    Returns:
        True if this is likely a label file, False otherwise
    """
    filename = txt_path.name.lower()
    stem = txt_path.stem.lower()
    
    # Exclude README and documentation files
    if 'readme' in filename:
        return False
    if stem in ['data', 'dataset', 'classes', 'config', 'notes']:
        return False
    if '.roboflow' in filename:
        return False
    
    # Check if in labels directory
    if 'labels' in [p.lower() for p in txt_path.parts]:
        return True
    
    # Check for corresponding image file
    parent_dir = txt_path.parent
    image_stem = txt_path.stem
    
    for ext in IMAGE_EXTENSIONS:
        image_path = parent_dir / f"{image_stem}{ext}"
        if image_path.exists():
            return True
        # Also check upper case
        image_path_upper = parent_dir / f"{image_stem}{ext.upper()}"
        if image_path_upper.exists():
            return True
    
    return False


def validate_yolo_annotation(annotation_path: Path) -> AnnotationInfo:
    """
    Validate YOLO annotation file syntax.
    
    Note: This only checks syntax, not semantic correctness.
    It does not verify that bounding boxes are within image bounds
    or that class IDs match a specific dataset.
    
    Args:
        annotation_path: Path to the annotation file
        
    Returns:
        AnnotationInfo object with validation results
    """
    relative_path = annotation_path.name
    
    try:
        with open(annotation_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        line_count = len([line for line in lines if line.strip()])
        
        # Validate each line
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            
            # YOLO format: class_id x_center y_center width height
            if len(parts) != 5:
                return AnnotationInfo(
                    path=str(annotation_path),
                    relative_path=relative_path,
                    line_count=line_count,
                    is_valid=False,
                    error_message=f"Line {line_num}: Expected 5 values, got {len(parts)}"
                )
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Basic range checks
                if class_id < 0 or class_id > YOLO_MAX_CLASS_ID:
                    return AnnotationInfo(
                        path=str(annotation_path),
                        relative_path=relative_path,
                        line_count=line_count,
                        is_valid=False,
                        error_message=f"Line {line_num}: Invalid class_id {class_id}"
                    )
                
                # Coordinates should be normalized (0-1)
                for name, value in [("x_center", x_center), ("y_center", y_center),
                                   ("width", width), ("height", height)]:
                    if value < 0 or value > 1:
                        return AnnotationInfo(
                            path=str(annotation_path),
                            relative_path=relative_path,
                            line_count=line_count,
                            is_valid=False,
                            error_message=f"Line {line_num}: {name}={value} not in range [0, 1]"
                        )
                        
            except ValueError as e:
                return AnnotationInfo(
                    path=str(annotation_path),
                    relative_path=relative_path,
                    line_count=line_count,
                    is_valid=False,
                    error_message=f"Line {line_num}: Cannot parse values - {str(e)}"
                )
        
        return AnnotationInfo(
            path=str(annotation_path),
            relative_path=relative_path,
            line_count=line_count,
            is_valid=True,
            error_message=None
        )
        
    except Exception as e:
        return AnnotationInfo(
            path=str(annotation_path),
            relative_path=relative_path,
            line_count=0,
            is_valid=False,
            error_message=str(e)
        )


def find_duplicates(images: List[ImageInfo]) -> List[List[str]]:
    """
    Find duplicate images based on SHA-256 hash.
    
    Args:
        images: List of ImageInfo objects
        
    Returns:
        List of duplicate groups, each group contains file paths
    """
    hash_to_paths: Dict[str, List[str]] = defaultdict(list)
    
    for img in images:
        if not img.is_corrupt and img.sha256:
            hash_to_paths[img.sha256].append(img.path)
    
    # Return only groups with duplicates
    duplicates = [paths for paths in hash_to_paths.values() if len(paths) > 1]
    return duplicates


def find_image_label_pairs(images: List[ImageInfo], annotations: List[AnnotationInfo]) -> tuple:
    """
    Find missing image-label pairs.
    
    Args:
        images: List of ImageInfo objects
        annotations: List of AnnotationInfo objects (only label files)
        
    Returns:
        Tuple of (images_without_labels, labels_without_images)
    """
    # Build sets of stems (filename without extension)
    image_stems = {}
    for img in images:
        if not img.is_corrupt:
            img_path = Path(img.path)
            stem = img_path.stem
            parent = str(img_path.parent)
            key = f"{parent}|{stem}"
            image_stems[key] = img.path
    
    label_stems = {}
    for ann in annotations:
        ann_path = Path(ann.path)
        stem = ann_path.stem
        # Labels might be in 'labels' subdirectory
        parent = str(ann_path.parent)
        # Try to match with images in sibling 'images' directory
        if 'labels' in parent:
            image_parent = parent.replace('\\labels', '\\images').replace('/labels', '/images')
        else:
            image_parent = parent
        key = f"{image_parent}|{stem}"
        label_stems[key] = ann.path
    
    # Find missing pairs
    images_without_labels = []
    for key, img_path in image_stems.items():
        if key not in label_stems:
            images_without_labels.append(img_path)
    
    labels_without_images = []
    for key, label_path in label_stems.items():
        if key not in image_stems:
            labels_without_images.append(label_path)
    
    return images_without_labels, labels_without_images


def extract_class_distribution(annotations: List[AnnotationInfo]) -> Dict[int, int]:
    """
    Extract class ID distribution from valid annotations.
    
    Args:
        annotations: List of AnnotationInfo objects
        
    Returns:
        Dictionary mapping class_id to count
    """
    class_counts: Dict[int, int] = defaultdict(int)
    
    for ann in annotations:
        if not ann.is_valid:
            continue
        
        try:
            with open(ann.path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1
                        except ValueError:
                            pass
        except Exception:
            pass
    
    return dict(class_counts)


def audit_dataset(
    input_dir: Path,
    min_image_size: int = 100,
    show_progress: bool = True
) -> AuditReport:
    """
    Audit a dataset directory.
    
    Args:
        input_dir: Root directory to scan
        min_image_size: Minimum dimension to warn about small images
        show_progress: Whether to show progress bar
        
    Returns:
        AuditReport object
    """
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")
    
    # Discover all image files (deduplicate by resolving to absolute paths)
    image_files_set = set()
    for ext in IMAGE_EXTENSIONS:
        image_files_set.update(input_dir.rglob(f"*{ext}"))
        image_files_set.update(input_dir.rglob(f"*{ext.upper()}"))
    
    image_files = list(image_files_set)
    
    # Discover all .txt files
    all_txt_files = list(input_dir.rglob("*.txt"))
    
    # Process images
    images = []
    iterator = tqdm(image_files, desc="Scanning images", disable=not show_progress)
    
    for image_path in iterator:
        img_info = validate_image(image_path)
        images.append(img_info)
    
    # Separate label files from other text files
    label_files = []
    ignored_files = []
    
    for txt_path in all_txt_files:
        if is_label_file(txt_path, input_dir):
            label_files.append(txt_path)
        else:
            ignored_files.append(str(txt_path))
    
    # Process label annotations
    annotations = []
    if label_files:
        iterator = tqdm(label_files, desc="Validating labels", disable=not show_progress)
        for ann_path in iterator:
            ann_info = validate_yolo_annotation(ann_path)
            annotations.append(ann_info)
    
    # Compute statistics
    valid_images = [img for img in images if not img.is_corrupt]
    corrupt_images = [img for img in images if img.is_corrupt]
    
    small_images = [
        img for img in valid_images
        if img.width < min_image_size or img.height < min_image_size
    ]
    
    duplicates = find_duplicates(images)
    
    # Count by extension
    images_by_extension = defaultdict(int)
    for img in images:
        ext = Path(img.path).suffix.lower()
        images_by_extension[ext] += 1
    
    # Count by directory
    images_by_directory = defaultdict(int)
    for img in images:
        parent = str(Path(img.path).parent)
        images_by_directory[parent] += 1
    
    # Total size
    total_size = sum(img.file_size_bytes for img in images)
    
    # Annotation stats
    valid_annotations = [ann for ann in annotations if ann.is_valid]
    invalid_annotations = [ann for ann in annotations if not ann.is_valid]
    
    # Find missing pairs
    images_without_labels, labels_without_images = find_image_label_pairs(images, annotations)
    
    # Extract class distribution
    class_distribution = extract_class_distribution(annotations)
    
    return AuditReport(
        input_directory=str(input_dir),
        total_images=len(images),
        valid_images=len(valid_images),
        corrupt_images=len(corrupt_images),
        duplicate_images=sum(len(group) - 1 for group in duplicates),
        small_images=len(small_images),
        total_size_bytes=total_size,
        images_by_extension=dict(images_by_extension),
        images_by_directory=dict(images_by_directory),
        label_files_found=len(annotations),
        valid_label_files=len(valid_annotations),
        invalid_label_files=len(invalid_annotations),
        ignored_text_files=len(ignored_files),
        missing_image_pairs=len(labels_without_images),
        missing_label_pairs=len(images_without_labels),
        class_distribution=class_distribution,
        images=images,
        duplicates=duplicates,
        annotations=annotations,
        ignored_files=ignored_files
    )


def save_json_report(report: AuditReport, output_path: Path):
    """Save audit report as JSON."""
    # Convert to dict
    report_dict = {
        "input_directory": report.input_directory,
        "total_images": report.total_images,
        "valid_images": report.valid_images,
        "corrupt_images": report.corrupt_images,
        "duplicate_images": report.duplicate_images,
        "small_images": report.small_images,
        "total_size_bytes": report.total_size_bytes,
        "images_by_extension": report.images_by_extension,
        "images_by_directory": report.images_by_directory,
        "label_files_found": report.label_files_found,
        "valid_label_files": report.valid_label_files,
        "invalid_label_files": report.invalid_label_files,
        "ignored_text_files": report.ignored_text_files,
        "missing_image_pairs": report.missing_image_pairs,
        "missing_label_pairs": report.missing_label_pairs,
        "class_distribution": report.class_distribution,
        "images": [asdict(img) for img in report.images],
        "duplicates": report.duplicates,
        "annotations": [asdict(ann) for ann in report.annotations],
        "ignored_files": report.ignored_files
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    
    print(f"JSON report saved to: {output_path}")


def save_csv_reports(report: AuditReport, output_dir: Path):
    """Save audit report as CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Images CSV
    images_df = pd.DataFrame([asdict(img) for img in report.images])
    images_csv = output_dir / "images.csv"
    images_df.to_csv(images_csv, index=False)
    print(f"Images CSV saved to: {images_csv}")
    
    # Annotations CSV
    if report.annotations:
        annotations_df = pd.DataFrame([asdict(ann) for ann in report.annotations])
        annotations_csv = output_dir / "annotations.csv"
        annotations_df.to_csv(annotations_csv, index=False)
        print(f"Annotations CSV saved to: {annotations_csv}")
    
    # Summary CSV
    summary_data = {
        "metric": [
            "total_images",
            "valid_images",
            "corrupt_images",
            "duplicate_images",
            "small_images",
            "total_size_bytes",
            "label_files_found",
            "valid_label_files",
            "invalid_label_files",
            "ignored_text_files",
            "missing_image_pairs",
            "missing_label_pairs"
        ],
        "value": [
            report.total_images,
            report.valid_images,
            report.corrupt_images,
            report.duplicate_images,
            report.small_images,
            report.total_size_bytes,
            report.label_files_found,
            report.valid_label_files,
            report.invalid_label_files,
            report.ignored_text_files,
            report.missing_image_pairs,
            report.missing_label_pairs
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_csv = output_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary CSV saved to: {summary_csv}")


def print_summary(report: AuditReport):
    """Print audit summary to console."""
    print("\n" + "=" * 60)
    print("DATASET AUDIT SUMMARY")
    print("=" * 60)
    print(f"Input Directory: {report.input_directory}")
    print(f"Total Images: {report.total_images}")
    print(f"  Valid: {report.valid_images}")
    print(f"  Corrupt: {report.corrupt_images}")
    print(f"  Duplicates: {report.duplicate_images}")
    print(f"  Small (warning): {report.small_images}")
    print(f"Total Size: {report.total_size_bytes:,} bytes ({report.total_size_bytes / (1024**2):.2f} MB)")
    
    print(f"\nImages by Extension:")
    for ext, count in sorted(report.images_by_extension.items()):
        print(f"  {ext}: {count}")
    
    print(f"\nLabel Files Found: {report.label_files_found}")
    print(f"  Valid: {report.valid_label_files}")
    print(f"  Invalid: {report.invalid_label_files}")
    print(f"Ignored Text Files: {report.ignored_text_files} (README, docs, etc.)")
    
    print(f"\nImage-Label Pairing:")
    print(f"  Images without labels: {report.missing_label_pairs}")
    print(f"  Labels without images: {report.missing_image_pairs}")
    
    if report.class_distribution:
        print(f"\nClass Distribution (from valid labels):")
        for class_id in sorted(report.class_distribution.keys()):
            count = report.class_distribution[class_id]
            print(f"  Class {class_id}: {count} detections")
    
    if report.corrupt_images > 0:
        print(f"\nCorrupt Images:")
        corrupt = [img for img in report.images if img.is_corrupt]
        for img in corrupt[:5]:  # Show first 5
            print(f"  {img.relative_path}: {img.error_message}")
        if len(corrupt) > 5:
            print(f"  ... and {len(corrupt) - 5} more")
    
    if report.duplicates:
        print(f"\nDuplicate Groups Found: {len(report.duplicates)}")
        for i, group in enumerate(report.duplicates[:3], 1):  # Show first 3 groups
            print(f"  Group {i}: {len(group)} copies")
            for path in group[:2]:  # Show first 2 paths
                print(f"    - {Path(path).name}")
        if len(report.duplicates) > 3:
            print(f"  ... and {len(report.duplicates) - 3} more groups")
    
    print("=" * 60 + "\n")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Audit a dataset directory for images and annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic audit
  python -m datasets.audit /path/to/dataset

  # Save JSON and CSV reports
  python -m datasets.audit /path/to/dataset --output-dir ./reports

  # Custom small image threshold
  python -m datasets.audit /path/to/dataset --min-size 640

  # Quiet mode (no progress bar)
  python -m datasets.audit /path/to/dataset --quiet
        """
    )
    
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory to audit"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for reports (default: input_dir/audit_reports)"
    )
    
    parser.add_argument(
        "--min-size",
        type=int,
        default=100,
        help="Minimum image dimension for small image warning (default: 100)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress bars"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Run audit
    try:
        report = audit_dataset(
            args.input_dir,
            min_image_size=args.min_size,
            show_progress=not args.quiet
        )
        
        # Print summary
        print_summary(report)
        
        # Save reports
        if args.output_dir is None:
            args.output_dir = args.input_dir / "audit_reports"
        
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = args.output_dir / "audit_report.json"
        save_json_report(report, json_path)
        
        # Save CSVs
        save_csv_reports(report, args.output_dir)
        
        print("\nAudit complete!")
        
        # Exit with error code if issues found
        if report.corrupt_images > 0:
            sys.exit(1)
        
    except Exception as e:
        print(f"Error during audit: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
