"""
Analyze annotation patterns to determine what segmentation polygons represent.
"""

from pathlib import Path
from collections import defaultdict
import json

from datasets.validation import validate_annotation_file, AnnotationFormat


def analyze_annotation_patterns(dataset_root: Path, num_samples: int = 5):
    """
    Analyze annotation patterns to determine what segmentation represents.
    
    Args:
        dataset_root: Root of dataset
        num_samples: Number of samples to analyze per format/class combination
    """
    results = {
        'segmentation_by_class': defaultdict(list),
        'detection_by_class': defaultdict(list),
        'mixed_by_class': defaultdict(list),
        'analysis': {}
    }
    
    # Find all label files
    label_dir = dataset_root / 'train' / 'labels'
    label_files = list(label_dir.glob('*.txt'))
    
    print(f"Analyzing {len(label_files)} label files...")
    
    for label_path in label_files:
        validation = validate_annotation_file(label_path)
        
        if not validation.is_valid or validation.format == AnnotationFormat.EMPTY:
            continue
        
        format_name = validation.format.value
        
        # Group by class ID
        for class_id in validation.class_ids:
            if format_name == 'segmentation':
                results['segmentation_by_class'][class_id].append({
                    'path': str(label_path),
                    'line_count': validation.line_count,
                    'classes': validation.class_ids
                })
            elif format_name == 'detection':
                results['detection_by_class'][class_id].append({
                    'path': str(label_path),
                    'line_count': validation.line_count,
                    'classes': validation.class_ids
                })
            elif format_name == 'mixed':
                results['mixed_by_class'][class_id].append({
                    'path': str(label_path),
                    'line_count': validation.line_count,
                    'classes': validation.class_ids
                })
    
    # Analyze patterns
    print("\n" + "="*70)
    print("ANNOTATION PATTERN ANALYSIS")
    print("="*70)
    
    print("\nSEGMENTATION FORMAT:")
    for class_id in sorted(results['segmentation_by_class'].keys()):
        samples = results['segmentation_by_class'][class_id]
        print(f"  Class {class_id}: {len(samples)} files")
        
        # Sample analysis
        for i, sample in enumerate(samples[:num_samples]):
            print(f"    Sample {i+1}: {Path(sample['path']).name}")
            print(f"      Lines: {sample['line_count']}, Classes: {sample['classes']}")
            
            # Count annotations per class in this file
            validation = validate_annotation_file(Path(sample['path']))
            class_counts = defaultdict(int)
            for line in validation.lines:
                if line.is_valid:
                    class_counts[line.class_id] += 1
            print(f"      Annotations per class: {dict(class_counts)}")
    
    print("\nDETECTION FORMAT:")
    for class_id in sorted(results['detection_by_class'].keys()):
        samples = results['detection_by_class'][class_id]
        print(f"  Class {class_id}: {len(samples)} files")
    
    print("\nMIXED FORMAT:")
    for class_id in sorted(results['mixed_by_class'].keys()):
        samples = results['mixed_by_class'][class_id]
        print(f"  Class {class_id}: {len(samples)} files")
    
    # Key finding: Check if segmentation files have multiple polygons per image
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    seg_healthy = results['segmentation_by_class'].get(4, [])
    if seg_healthy:
        multi_polygon = [s for s in seg_healthy if s['line_count'] > 1]
        single_polygon = [s for s in seg_healthy if s['line_count'] == 1]
        
        print(f"\nClass 4 (Healthy) segmentation:")
        print(f"  Files with single polygon: {len(single_polygon)}")
        print(f"  Files with multiple polygons: {len(multi_polygon)}")
        
        if multi_polygon:
            print(f"\n  Sample multi-polygon files:")
            for sample in multi_polygon[:3]:
                print(f"    {Path(sample['path']).name}: {sample['line_count']} polygons")
    
    return results


if __name__ == "__main__":
    dataset_root = Path(r"D:\Important Projects\dataset\annoted data set labeled")
    results = analyze_annotation_patterns(dataset_root, num_samples=3)
