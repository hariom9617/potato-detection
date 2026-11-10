"""
Training Data Eligibility Analysis for YOLOv8n Detection Baseline.

This script performs a comprehensive recheck of training-data eligibility,
reconciling manifest claims against actual file contents and validation results.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from datasets.validation import validate_annotation_file, AnnotationFormat, AnnotationLine


def analyze_mixed_format_file(label_path: Path) -> Tuple[int, int, List[int]]:
    """
    Analyze a mixed-format file to count detection vs segmentation lines.
    
    Args:
        label_path: Path to label file
        
    Returns:
        Tuple of (detection_line_count, segmentation_line_count, detection_class_ids)
    """
    detection_lines = []
    segmentation_lines = []
    
    validation = validate_annotation_file(label_path)
    
    for line in validation.lines:
        if not line.is_valid:
            continue
        
        if line.format == AnnotationFormat.DETECTION:
            detection_lines.append(line.class_id)
        elif line.format == AnnotationFormat.SEGMENTATION:
            segmentation_lines.append(line.class_id)
    
    return len(detection_lines), len(segmentation_lines), detection_lines


def recheck_training_eligibility(manifests_dir: Path, dataset_root: Path):
    """
    Comprehensive recheck of training data eligibility.
    
    Args:
        manifests_dir: Directory containing manifest CSVs
        dataset_root: Root directory of source dataset
    """
    print("="*80)
    print("TRAINING DATA ELIGIBILITY RECHECK")
    print("="*80)
    
    # Load manifests
    det_manifest = pd.read_csv(manifests_dir / 'manifest_detection.csv')
    mix_manifest = pd.read_csv(manifests_dir / 'manifest_mixed.csv')
    inv_manifest = pd.read_csv(manifests_dir / 'manifest_invalid.csv')
    seg_manifest = pd.read_csv(manifests_dir / 'manifest_segmentation.csv')
    
    print(f"\n1. MANIFEST RECONCILIATION")
    print(f"   Detection-only images: {len(det_manifest)}")
    print(f"   Mixed-format images: {len(mix_manifest)}")
    print(f"   Invalid images: {len(inv_manifest)}")
    print(f"   Claimed usable: {len(det_manifest)} + {len(mix_manifest)} = {len(det_manifest) + len(mix_manifest)}")
    
    # Analyze split distribution
    print(f"\n2. SPLIT DISTRIBUTION")
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        det_count = len(det_manifest[det_manifest['split'] == split])
        mix_count = len(mix_manifest[mix_manifest['split'] == split])
        inv_count = len(inv_manifest[inv_manifest['split'] == split])
        
        print(f"   {split:6s}: Detection={det_count:4d}, Mixed={mix_count:3d}, Invalid={inv_count:2d}, Total={det_count + mix_count:4d}")
    
    # Analyze mixed-format files
    print(f"\n3. MIXED-FORMAT FILE ANALYSIS")
    print(f"   Analyzing {len(mix_manifest)} mixed-format files...")
    
    mixed_stats = {
        'total_files': 0,
        'total_detection_boxes': 0,
        'total_segmentation_polygons': 0,
        'by_split': defaultdict(lambda: {'files': 0, 'detection': 0, 'segmentation': 0}),
        'by_class': defaultdict(lambda: {'files': 0, 'boxes': 0})
    }
    
    for _, row in mix_manifest.iterrows():
        label_path = Path(row['label_path'])
        if not label_path.exists():
            print(f"   WARNING: Label file not found: {label_path}")
            continue
        
        det_count, seg_count, det_classes = analyze_mixed_format_file(label_path)
        
        mixed_stats['total_files'] += 1
        mixed_stats['total_detection_boxes'] += det_count
        mixed_stats['total_segmentation_polygons'] += seg_count
        
        split = row['split']
        mixed_stats['by_split'][split]['files'] += 1
        mixed_stats['by_split'][split]['detection'] += det_count
        mixed_stats['by_split'][split]['segmentation'] += seg_count
        
        for class_id in det_classes:
            mixed_stats['by_class'][class_id]['boxes'] += 1
        
        for class_id in set(det_classes):
            mixed_stats['by_class'][class_id]['files'] += 1
    
    print(f"   Total files processed: {mixed_stats['total_files']}")
    print(f"   Total DETECTION boxes extractable: {mixed_stats['total_detection_boxes']}")
    print(f"   Total segmentation polygons (ignored): {mixed_stats['total_segmentation_polygons']}")
    
    print(f"\n   Mixed-format by split:")
    for split in splits:
        stats = mixed_stats['by_split'][split]
        print(f"     {split:6s}: {stats['files']:3d} files, {stats['detection']:4d} detection boxes, {stats['segmentation']:4d} polygons (ignored)")
    
    # Analyze invalid files
    print(f"\n4. INVALID ANNOTATION ANALYSIS")
    
    empty_labels = inv_manifest[inv_manifest['label_format'] == 'empty']
    coord_overflow = inv_manifest[inv_manifest['label_format'] != 'empty']
    
    print(f"   Total invalid entries: {len(inv_manifest)}")
    print(f"   Empty labels (valid background): {len(empty_labels)}")
    print(f"   Coordinate overflow: {len(coord_overflow)}")
    
    if len(coord_overflow) > 0:
        print(f"\n   Coordinate overflow details:")
        for _, row in coord_overflow.iterrows():
            print(f"     {Path(row['relative_path']).name}")
            print(f"       Error: {row['error_message']}")
    
    # Count total detection annotations
    print(f"\n5. TOTAL DETECTION ANNOTATION COUNT")
    
    # Detection-only files
    det_annotation_count = 0
    det_class_counts = defaultdict(int)
    
    for _, row in det_manifest.iterrows():
        label_path = Path(row['label_path'])
        if not label_path.exists():
            continue
        
        validation = validate_annotation_file(label_path)
        for line in validation.lines:
            if line.is_valid and line.format == AnnotationFormat.DETECTION:
                det_annotation_count += 1
                det_class_counts[line.class_id] += 1
    
    print(f"   Detection-only files: {det_annotation_count} boxes")
    
    # Mixed files (detection boxes only)
    mix_detection_count = mixed_stats['total_detection_boxes']
    print(f"   Mixed-format files (detection boxes only): {mix_detection_count} boxes")
    
    total_detection = det_annotation_count + mix_detection_count
    print(f"   TOTAL DETECTION BOXES: {total_detection}")
    
    # Per-class detection box count
    print(f"\n6. PER-CLASS DETECTION BOX COUNT")
    print(f"   Combining detection-only + mixed-format (detection boxes)...")
    
    combined_class_counts = defaultdict(int)
    for class_id, count in det_class_counts.items():
        combined_class_counts[class_id] += count
    
    for class_id, count in mixed_stats['by_class'].items():
        combined_class_counts[class_id] += count['boxes']
    
    class_names = {
        0: 'Abnormal Shaped Potato',
        1: 'Black Rot Potato',
        2: 'Cut Potato',
        3: 'Greening Potato',
        4: 'Healthy Potato',
        5: 'Hollow heart Potato',
        6: 'Rotten',
        7: 'Sprouted Patato'
    }
    
    total_boxes = sum(combined_class_counts.values())
    for class_id in sorted(combined_class_counts.keys()):
        count = combined_class_counts[class_id]
        pct = 100.0 * count / total_boxes if total_boxes > 0 else 0
        print(f"   Class {class_id} ({class_names.get(class_id, 'Unknown'):25s}): {count:5d} boxes ({pct:5.1f}%)")
    
    print(f"   {'':32s}  {'─'*10}")
    print(f"   {'Total':32s}: {total_boxes:5d} boxes")
    
    # Policy statement
    print(f"\n7. MIXED-FORMAT HANDLING POLICY")
    print(f"   ✓ Extract ONLY 5-value detection lines (bounding boxes)")
    print(f"   ✗ Ignore polygon lines (whole-potato outlines, not defects)")
    print(f"   ✓ Preserves {mixed_stats['total_detection_boxes']} valid defect boxes")
    print(f"   ✗ Excludes {mixed_stats['total_segmentation_polygons']} whole-potato masks")
    print(f"   Rationale: Detection boxes = defects, Polygons = whole potatoes")
    
    # Empty label policy
    print(f"\n8. EMPTY LABEL HANDLING POLICY")
    print(f"   Found {len(empty_labels)} empty label files")
    if len(empty_labels) > 0:
        print(f"   Files:")
        for _, row in empty_labels.iterrows():
            print(f"     - {Path(row['relative_path']).name}")
        print(f"   ✓ Include as negative samples (no objects present)")
        print(f"   ✓ Helps reduce false positives")
    
    # Coordinate overflow policy
    print(f"\n9. COORDINATE OVERFLOW POLICY")
    print(f"   Found {len(coord_overflow)} files with coordinate overflow")
    if len(coord_overflow) > 0:
        print(f"   Decision: EXCLUDE from training (strict validation)")
        print(f"   Rationale: Values outside [0, 1] indicate annotation errors")
        print(f"   Alternative: Could clip coordinates (standard YOLO practice)")
        print(f"   Impact: Excludes ~{len(coord_overflow)} of {len(det_manifest) + len(mix_manifest)} ({100.0*len(coord_overflow)/(len(det_manifest) + len(mix_manifest)):.1f}%) images")
    
    # Final usable count
    print(f"\n10. FINAL TRAINING-ELIGIBLE IMAGE COUNT")
    
    usable_detection_only = len(det_manifest)
    usable_mixed = len(mix_manifest)
    usable_empty = len(empty_labels)
    excluded_overflow = len(coord_overflow)
    
    # Note: overflow files are in invalid manifest, not detection/mixed
    final_usable = usable_detection_only + usable_mixed + usable_empty
    
    print(f"   Detection-only:  {usable_detection_only:4d} images")
    print(f"   Mixed-format:    {usable_mixed:4d} images (extract detection boxes)")
    print(f"   Empty labels:    {usable_empty:4d} images (negative samples)")
    print(f"   Excluded:        {excluded_overflow:4d} images (coordinate overflow)")
    print(f"   {'':17s}{'─'*10}")
    print(f"   TOTAL ELIGIBLE:  {final_usable:4d} images")
    
    # Split breakdown
    print(f"\n   Final eligible by split:")
    for split in splits:
        det_split = len(det_manifest[det_manifest['split'] == split])
        mix_split = len(mix_manifest[mix_manifest['split'] == split])
        empty_split = len(empty_labels[empty_labels['split'] == split])
        total_split = det_split + mix_split + empty_split
        
        print(f"     {split:6s}: {total_split:4d} images ({det_split} det + {mix_split} mix + {empty_split} empty)")
    
    print("\n" + "="*80)
    
    # Return summary
    return {
        'total_eligible': final_usable,
        'detection_only': usable_detection_only,
        'mixed_format': usable_mixed,
        'empty_labels': usable_empty,
        'excluded_overflow': excluded_overflow,
        'total_detection_boxes': total_detection,
        'class_counts': dict(combined_class_counts),
        'mixed_stats': mixed_stats
    }


def main():
    """CLI entry point."""
    manifests_dir = Path(r"D:\Important Projects\dataset-manifests")
    dataset_root = Path(r"D:\Important Projects\dataset\annoted data set labeled")
    
    summary = recheck_training_eligibility(manifests_dir, dataset_root)
    
    # Save summary
    output_path = manifests_dir / 'training_eligibility_summary.json'
    with open(output_path, 'w') as f:
        # Convert defaultdict to dict for JSON serialization
        serializable = {
            k: dict(v) if isinstance(v, defaultdict) else v 
            for k, v in summary.items()
        }
        # Handle nested defaultdict
        if 'mixed_stats' in serializable:
            ms = serializable['mixed_stats']
            if 'by_split' in ms:
                ms['by_split'] = dict(ms['by_split'])
            if 'by_class' in ms:
                ms['by_class'] = {str(k): v for k, v in ms['by_class'].items()}
        
        json.dump(serializable, f, indent=2)
    
    print(f"\nSummary saved to: {output_path}")


if __name__ == "__main__":
    main()
