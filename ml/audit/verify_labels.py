"""Verify YOLO label format and detect any issues."""

import os
from pathlib import Path
import json

# Load statistics to identify special cases
stats_path = Path(__file__).parent.parent.parent / 'yolo_detection_dataset' / 'dataset_statistics.json'
with open(stats_path, 'r') as f:
    stats = json.load(f)

manifest_path = Path(__file__).parent.parent.parent / 'yolo_detection_dataset' / 'dataset_manifest.csv'

print("="*60)
print("LABEL FORMAT VERIFICATION")
print("="*60)

# Check all label files
dataset_root = Path(__file__).parent.parent.parent / 'yolo_detection_dataset'
issues = []
total_labels = 0
total_boxes = 0
empty_labels = []

for split in ['train', 'valid', 'test']:
    labels_dir = dataset_root / split / 'labels'
    print(f"\nChecking {split} labels...")
    
    split_boxes = 0
    split_files = 0
    split_empty = 0
    
    for label_file in labels_dir.glob('*.txt'):
        total_labels += 1
        split_files += 1
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            empty_labels.append(str(label_file.name))
            split_empty += 1
            continue
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            
            # Valid detection format: class_id x_center y_center width height
            if len(parts) != 5:
                issues.append({
                    'file': label_file.name,
                    'line': line_num,
                    'issue': f'Invalid format: {len(parts)} values (expected 5)',
                    'content': line[:50]
                })
                continue
            
            try:
                class_id = int(parts[0])
                coords = [float(parts[i]) for i in range(1, 5)]
                
                # Check class ID range
                if class_id < 0 or class_id > 7:
                    issues.append({
                        'file': label_file.name,
                        'line': line_num,
                        'issue': f'Invalid class ID: {class_id} (must be 0-7)',
                        'content': line[:50]
                    })
                
                # Check coordinate range
                for i, coord in enumerate(coords):
                    if coord < 0 or coord > 1:
                        issues.append({
                            'file': label_file.name,
                            'line': line_num,
                            'issue': f'Coordinate out of range [0,1]: {coord}',
                            'content': line[:50]
                        })
                        break
                
                total_boxes += 1
                split_boxes += 1
                
            except ValueError as e:
                issues.append({
                    'file': label_file.name,
                    'line': line_num,
                    'issue': f'Parse error: {e}',
                    'content': line[:50]
                })
    
    print(f"  {split}: {split_files} files, {split_boxes} boxes, {split_empty} empty")

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Total label files: {total_labels}")
print(f"Total boxes: {total_boxes}")
print(f"Empty label files: {len(empty_labels)}")
print(f"Issues found: {len(issues)}")

if empty_labels:
    print(f"\nEmpty label files ({len(empty_labels)}):")
    for fname in empty_labels[:10]:
        print(f"  - {fname}")
    if len(empty_labels) > 10:
        print(f"  ... and {len(empty_labels) - 10} more")

if issues:
    print(f"\n⚠️  ISSUES FOUND:")
    for issue in issues[:20]:
        print(f"  {issue['file']}:{issue['line']} - {issue['issue']}")
        print(f"    Content: {issue['content']}")
    if len(issues) > 20:
        print(f"  ... and {len(issues) - 20} more issues")
else:
    print("\n✓ All labels valid (5-value detection format, coordinates in [0,1])")

# Verify against statistics
print(f"\n{'='*60}")
print(f"CROSS-CHECK WITH STATISTICS")
print(f"{'='*60}")
expected_boxes = stats['overall']['total_boxes']
print(f"Expected boxes (from statistics.json): {expected_boxes}")
print(f"Actual boxes (from label files): {total_boxes}")
if expected_boxes == total_boxes:
    print("✓ Box count matches")
else:
    print(f"✗ Mismatch: {abs(expected_boxes - total_boxes)} boxes difference")

expected_images = stats['overall']['total_images']
print(f"\nExpected images: {expected_images}")
print(f"Actual label files: {total_labels}")
if expected_images == total_labels:
    print("✓ Image count matches")
else:
    print(f"✗ Mismatch: {abs(expected_images - total_labels)} files difference")
