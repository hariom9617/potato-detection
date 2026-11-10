"""
Dataset Preparation Pipeline

Generates machine-readable manifests categorizing images by annotation format,
validation status, and split membership. Prepares data for model training without
modifying source files.
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

from datasets.validation import (
    AnnotationFormat,
    validate_annotation_file,
    AnnotationValidationResult
)


@dataclass
class ImageManifestEntry:
    """Single image entry in manifest."""
    image_path: str
    relative_path: str
    split: str  # train, valid, test, or unknown
    has_label: bool
    label_path: Optional[str]
    label_format: Optional[str]  # detection, segmentation, mixed, invalid, empty
    is_valid: bool
    class_ids: List[int]
    error_message: Optional[str] = None


@dataclass
class DatasetManifest:
    """Complete dataset manifest with categorized images."""
    dataset_root: str
    total_images: int
    images_with_segmentation: List[ImageManifestEntry]
    images_with_detection: List[ImageManifestEntry]
    images_without_labels: List[ImageManifestEntry]
    images_with_invalid_labels: List[ImageManifestEntry]
    images_with_mixed_format: List[ImageManifestEntry]
    duplicate_groups: List[List[str]]
    cross_split_duplicates: List[List[str]]
    class_distribution_by_format: Dict[str, Dict[int, int]]
    split_statistics: Dict[str, Dict[str, int]]


def find_images(root_dir: Path, image_extensions: Optional[Set[str]] = None) -> List[Path]:
    """
    Find all images in directory recursively.
    
    Args:
        root_dir: Root directory to search
        image_extensions: Set of extensions to search for
        
    Returns:
        List of image paths
    """
    if image_extensions is None:
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    
    image_files = set()
    for ext in image_extensions:
        image_files.update(root_dir.rglob(f"*{ext}"))
        image_files.update(root_dir.rglob(f"*{ext.upper()}"))
    
    return sorted(image_files)


def determine_split(image_path: Path, dataset_root: Path) -> str:
    """
    Determine which split an image belongs to based on directory structure.
    
    Args:
        image_path: Path to image file
        dataset_root: Root of dataset
        
    Returns:
        Split name: 'train', 'valid', 'test', or 'unknown'
    """
    rel_parts = image_path.relative_to(dataset_root).parts
    
    for part in rel_parts:
        part_lower = part.lower()
        if part_lower == 'train' or part_lower == 'training':
            return 'train'
        elif part_lower in ['valid', 'validation', 'val']:
            return 'valid'
        elif part_lower == 'test' or part_lower == 'testing':
            return 'test'
    
    return 'unknown'


def find_label_for_image(image_path: Path, dataset_root: Path) -> Optional[Path]:
    """
    Find corresponding label file for an image.
    
    Checks:
    1. Same directory with .txt extension
    2. Sibling 'labels' directory
    
    Args:
        image_path: Path to image
        dataset_root: Root of dataset
        
    Returns:
        Path to label file or None
    """
    # Try same directory
    label_same_dir = image_path.parent / f"{image_path.stem}.txt"
    if label_same_dir.exists():
        return label_same_dir
    
    # Try labels sibling directory (common YOLO structure)
    parent_dir = image_path.parent
    if 'images' in parent_dir.name.lower():
        labels_dir = parent_dir.parent / 'labels'
        label_in_labels = labels_dir / f"{image_path.stem}.txt"
        if label_in_labels.exists():
            return label_in_labels
    
    # Try labels subdirectory
    labels_subdir = parent_dir / 'labels'
    label_in_subdir = labels_subdir / f"{image_path.stem}.txt"
    if label_in_subdir.exists():
        return label_in_subdir
    
    return None


def compute_image_hash(image_path: Path) -> str:
    """Compute SHA-256 hash of image file."""
    import hashlib
    sha256_hash = hashlib.sha256()
    with open(image_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def find_duplicate_groups(image_paths: List[Path], dataset_root: Path) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Find exact duplicate images by hash.
    
    Args:
        image_paths: List of image paths
        dataset_root: Root directory of dataset
        
    Returns:
        Tuple of (all_duplicate_groups, cross_split_duplicate_groups)
    """
    hash_to_info: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    
    print("Computing image hashes for duplicate detection...")
    for img_path in tqdm(image_paths, desc="Hashing images"):
        img_hash = compute_image_hash(img_path)
        # Store (path, split) tuple
        split = determine_split(img_path, dataset_root)
        hash_to_info[img_hash].append((str(img_path), split))
    
    # Find all duplicates
    all_duplicates = []
    cross_split_duplicates = []
    
    for img_hash, info_list in hash_to_info.items():
        if len(info_list) > 1:
            paths = [path for path, split in info_list]
            all_duplicates.append(paths)
            
            # Check if duplicates span multiple splits
            splits = set(split for path, split in info_list)
            if len(splits) > 1:
                cross_split_duplicates.append(paths)
    
    return all_duplicates, cross_split_duplicates


def prepare_dataset(
    dataset_root: Path,
    output_dir: Path,
    max_class_id: int = 999,
    show_progress: bool = True
) -> DatasetManifest:
    """
    Prepare dataset manifests categorizing images by annotation format.
    
    Args:
        dataset_root: Root directory of dataset
        output_dir: Directory to save manifests
        max_class_id: Maximum valid class ID
        show_progress: Show progress bars
        
    Returns:
        DatasetManifest object
    """
    if not dataset_root.exists():
        raise ValueError(f"Dataset root does not exist: {dataset_root}")
    
    # Find all images
    print(f"Scanning for images in {dataset_root}...")
    image_paths = find_images(dataset_root)
    print(f"Found {len(image_paths)} images")
    
    # Find duplicates
    duplicate_groups, cross_split_duplicates = find_duplicate_groups(image_paths, dataset_root)
    print(f"Found {len(duplicate_groups)} duplicate groups ({len(cross_split_duplicates)} cross-split)")
    
    # Categorize images
    images_with_segmentation = []
    images_with_detection = []
    images_without_labels = []
    images_with_invalid_labels = []
    images_with_mixed_format = []
    
    class_dist_by_format = {
        'detection': defaultdict(int),
        'segmentation': defaultdict(int),
        'mixed': defaultdict(int)
    }
    
    split_stats = defaultdict(lambda: {
        'total': 0,
        'labeled': 0,
        'unlabeled': 0,
        'detection': 0,
        'segmentation': 0,
        'mixed': 0,
        'invalid': 0
    })
    
    iterator = tqdm(image_paths, desc="Processing images", disable=not show_progress)
    
    for image_path in iterator:
        split = determine_split(image_path, dataset_root)
        label_path = find_label_for_image(image_path, dataset_root)
        
        split_stats[split]['total'] += 1
        
        entry = ImageManifestEntry(
            image_path=str(image_path),
            relative_path=str(image_path.relative_to(dataset_root)),
            split=split,
            has_label=label_path is not None,
            label_path=str(label_path) if label_path else None,
            label_format=None,
            is_valid=False,
            class_ids=[],
            error_message=None
        )
        
        if label_path is None:
            # No label file
            split_stats[split]['unlabeled'] += 1
            images_without_labels.append(entry)
        else:
            # Validate label file
            validation_result = validate_annotation_file(label_path, max_class_id=max_class_id)
            
            entry.label_format = validation_result.format.value
            entry.is_valid = validation_result.is_valid
            entry.class_ids = validation_result.class_ids
            entry.error_message = validation_result.error_message
            
            split_stats[split]['labeled'] += 1
            
            # Categorize by format
            if validation_result.format == AnnotationFormat.DETECTION and validation_result.is_valid:
                split_stats[split]['detection'] += 1
                images_with_detection.append(entry)
                # Update class distribution
                for line in validation_result.lines:
                    if line.is_valid:
                        class_dist_by_format['detection'][line.class_id] += 1
            
            elif validation_result.format == AnnotationFormat.SEGMENTATION and validation_result.is_valid:
                split_stats[split]['segmentation'] += 1
                images_with_segmentation.append(entry)
                # Update class distribution
                for line in validation_result.lines:
                    if line.is_valid:
                        class_dist_by_format['segmentation'][line.class_id] += 1
            
            elif validation_result.format == AnnotationFormat.MIXED and validation_result.is_valid:
                split_stats[split]['mixed'] += 1
                images_with_mixed_format.append(entry)
                # Update class distribution
                for line in validation_result.lines:
                    if line.is_valid:
                        class_dist_by_format['mixed'][line.class_id] += 1
            
            else:
                # Invalid or empty
                split_stats[split]['invalid'] += 1
                images_with_invalid_labels.append(entry)
    
    # Convert defaultdicts to regular dicts
    class_dist_by_format = {
        k: dict(v) for k, v in class_dist_by_format.items()
    }
    
    split_stats = {k: dict(v) for k, v in split_stats.items()}
    
    manifest = DatasetManifest(
        dataset_root=str(dataset_root),
        total_images=len(image_paths),
        images_with_segmentation=images_with_segmentation,
        images_with_detection=images_with_detection,
        images_without_labels=images_without_labels,
        images_with_invalid_labels=images_with_invalid_labels,
        images_with_mixed_format=images_with_mixed_format,
        duplicate_groups=duplicate_groups,
        cross_split_duplicates=cross_split_duplicates,
        class_distribution_by_format=class_dist_by_format,
        split_statistics=split_stats
    )
    
    return manifest


def save_manifests(manifest: DatasetManifest, output_dir: Path):
    """
    Save dataset manifests to JSON and CSV files.
    
    Args:
        manifest: DatasetManifest to save
        output_dir: Directory to save files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary JSON
    summary = {
        'dataset_root': manifest.dataset_root,
        'total_images': manifest.total_images,
        'counts': {
            'with_segmentation': len(manifest.images_with_segmentation),
            'with_detection': len(manifest.images_with_detection),
            'without_labels': len(manifest.images_without_labels),
            'with_invalid_labels': len(manifest.images_with_invalid_labels),
            'with_mixed_format': len(manifest.images_with_mixed_format)
        },
        'duplicates': {
            'total_groups': len(manifest.duplicate_groups),
            'cross_split_groups': len(manifest.cross_split_duplicates)
        },
        'class_distribution_by_format': manifest.class_distribution_by_format,
        'split_statistics': manifest.split_statistics
    }
    
    summary_path = output_dir / 'manifest_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_path}")
    
    # Save individual manifest CSVs
    for name, entries in [
        ('segmentation', manifest.images_with_segmentation),
        ('detection', manifest.images_with_detection),
        ('unlabeled', manifest.images_without_labels),
        ('invalid', manifest.images_with_invalid_labels),
        ('mixed', manifest.images_with_mixed_format)
    ]:
        if entries:
            df = pd.DataFrame([asdict(e) for e in entries])
            csv_path = output_dir / f'manifest_{name}.csv'
            df.to_csv(csv_path, index=False)
            print(f"Saved {name} manifest to: {csv_path}")
    
    # Save duplicate groups
    if manifest.duplicate_groups:
        dup_data = []
        for i, group in enumerate(manifest.duplicate_groups):
            for path in group:
                dup_data.append({'group_id': i, 'image_path': path})
        
        df_dup = pd.DataFrame(dup_data)
        dup_path = output_dir / 'duplicates.csv'
        df_dup.to_csv(dup_path, index=False)
        print(f"Saved duplicates to: {dup_path}")
    
    # Save cross-split duplicates
    if manifest.cross_split_duplicates:
        cross_dup_data = []
        for i, group in enumerate(manifest.cross_split_duplicates):
            for path in group:
                cross_dup_data.append({'group_id': i, 'image_path': path})
        
        df_cross = pd.DataFrame(cross_dup_data)
        cross_path = output_dir / 'cross_split_duplicates.csv'
        df_cross.to_csv(cross_path, index=False)
        print(f"Saved cross-split duplicates to: {cross_path}")
    
    print(f"\nAll manifests saved to: {output_dir}")


def print_manifest_summary(manifest: DatasetManifest):
    """Print summary of dataset manifest."""
    print("\n" + "=" * 70)
    print("DATASET PREPARATION SUMMARY")
    print("=" * 70)
    print(f"Dataset Root: {manifest.dataset_root}")
    print(f"Total Images: {manifest.total_images}")
    
    print(f"\nAnnotation Format Distribution:")
    print(f"  Segmentation: {len(manifest.images_with_segmentation)}")
    print(f"  Detection: {len(manifest.images_with_detection)}")
    print(f"  Mixed Format: {len(manifest.images_with_mixed_format)}")
    print(f"  No Labels: {len(manifest.images_without_labels)}")
    print(f"  Invalid Labels: {len(manifest.images_with_invalid_labels)}")
    
    print(f"\nDuplicate Analysis:")
    print(f"  Duplicate Groups: {len(manifest.duplicate_groups)}")
    print(f"  Cross-Split Duplicates: {len(manifest.cross_split_duplicates)}")
    if manifest.cross_split_duplicates:
        print(f"    ⚠️  WARNING: Cross-split duplicates found! This causes data leakage.")
    else:
        print(f"    ✅ No cross-split duplicates (excellent!)")
    
    print(f"\nSplit Statistics:")
    for split, stats in sorted(manifest.split_statistics.items()):
        print(f"  {split}:")
        print(f"    Total: {stats['total']}")
        print(f"    Labeled: {stats['labeled']}")
        print(f"    Segmentation: {stats['segmentation']}")
        print(f"    Detection: {stats['detection']}")
        print(f"    Mixed: {stats['mixed']}")
    
    print(f"\nClass Distribution by Format:")
    for format_name, dist in manifest.class_distribution_by_format.items():
        if dist:
            total = sum(dist.values())
            print(f"  {format_name} ({total} annotations):")
            for class_id in sorted(dist.keys()):
                count = dist[class_id]
                pct = 100.0 * count / total
                print(f"    Class {class_id}: {count} ({pct:.1f}%)")
    
    print("=" * 70 + "\n")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare dataset manifests for model training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Root directory of dataset"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for manifests (default: dataset_root/manifests)"
    )
    
    parser.add_argument(
        "--max-class-id",
        type=int,
        default=999,
        help="Maximum valid class ID (default: 999)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress bars"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.dataset_root.exists():
        print(f"Error: Dataset root does not exist: {args.dataset_root}", file=sys.stderr)
        sys.exit(1)
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.dataset_root / "manifests"
    
    try:
        # Prepare dataset
        manifest = prepare_dataset(
            args.dataset_root,
            args.output_dir,
            max_class_id=args.max_class_id,
            show_progress=not args.quiet
        )
        
        # Print summary
        print_manifest_summary(manifest)
        
        # Save manifests
        save_manifests(manifest, args.output_dir)
        
        print("\n✅ Dataset preparation complete!")
        
    except Exception as e:
        print(f"Error during preparation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
