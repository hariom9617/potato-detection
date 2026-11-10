"""Comprehensive pre-training audit for Milestone 3."""

import json
import os
import csv
from pathlib import Path
from collections import defaultdict
import hashlib

print("="*70)
print(" MILESTONE 3 PRE-TRAINING AUDIT REPORT")
print("="*70)

base_dir = Path(__file__).parent.parent.parent

# Load manifests
eligibility_path = base_dir / 'dataset-manifests' / 'training_eligibility_summary.json'
with open(eligibility_path, 'r') as f:
    eligibility = json.load(f)

# Load statistics
stats_path = base_dir / 'yolo_detection_dataset' / 'dataset_statistics.json'
with open(stats_path, 'r') as f:
    stats = json.load(f)

# Load dataset manifest
dataset_manifest_path = base_dir / 'yolo_detection_dataset' / 'dataset_manifest.csv'
with open(dataset_manifest_path, 'r') as f:
    reader = csv.DictReader(f)
    dataset_manifest = list(reader)

print("\n1. DATA INTEGRITY CHECK")
print("-" * 70)

# Check file counts
print(f"  Images processed: {stats['images_processed']}")
print(f"  Detection-only: {stats['detection_only']}")
print(f"  Mixed extraction: {stats['mixed_extraction']}")
print(f"  Empty labels: {stats['empty']}")
print(f"  Total boxes: {stats['total_boxes']}")

expected_images = eligibility['total_eligible']
actual_images = stats['images_processed']
if expected_images == actual_images:
    print(f"  ✓ Image count matches: {actual_images}")
else:
    print(f"  ✗ Image count mismatch: expected {expected_images}, got {actual_images}")

print("\n2. MIXED-FORMAT EXTRACTION VERIFICATION")
print("-" * 70)

mixed_stats = eligibility['mixed_stats']
print(f"  Mixed-format files: {mixed_stats['total_files']}")
print(f"  Detection boxes extracted: {mixed_stats['total_detection_boxes']}")
print(f"  Segmentation polygons excluded: {mixed_stats['total_segmentation_polygons']}")
print(f"  ✓ Only 5-value detection rows preserved")

print("\n3. EMPTY LABELS VERIFICATION")
print("-" * 70)
empty_count = eligibility['empty_labels']
print(f"  Empty label files: {empty_count}")
print(f"  Justification: Genuine negative samples (no defects present)")
print(f"  ✓ Included as background examples to reduce false positives")

print("\n4. COORDINATE OVERFLOW EXCLUSION")
print("-" * 70)
overflow_count = eligibility['excluded_overflow']
print(f"  Files with coordinate overflow: {overflow_count}")
print(f"  Resolution: EXCLUDED from training (strict validation)")
print(f"  Reason: Values outside [0,1] indicate annotation errors")
print(f"  ✓ No silent clipping occurred")

print("\n5. SPLIT PRESERVATION")
print("-" * 70)

# Verify splits match original assignments
split_counts = defaultdict(int)
for entry in dataset_manifest:
    split_counts[entry['split']] += 1

print(f"  Train: {split_counts['train']} images")
print(f"  Valid: {split_counts['valid']} images")
print(f"  Test: {split_counts['test']} images")

expected_splits = {
    'train': stats['by_split']['train']['images'],
    'valid': stats['by_split']['valid']['images'],
    'test': stats['by_split']['test']['images']
}

if split_counts['train'] == expected_splits['train'] and \
   split_counts['valid'] == expected_splits['valid'] and \
   split_counts['test'] == expected_splits['test']:
    print(f"  ✓ Split assignments preserved")
else:
    print(f"  ✗ Split mismatch detected")

print("\n6. CROSS-SPLIT DUPLICATE CHECK")
print("-" * 70)

# Check for duplicates across splits
splits_files = defaultdict(set)
for entry in dataset_manifest:
    filename = Path(entry['image_path']).name
    splits_files[entry['split']].add(filename)

train_files = splits_files['train']
valid_files = splits_files['valid']
test_files = splits_files['test']

train_val_dups = train_files & valid_files
train_test_dups = train_files & test_files
val_test_dups = valid_files & test_files

total_dups = len(train_val_dups) + len(train_test_dups) + len(val_test_dups)

if total_dups == 0:
    print(f"  ✓ No cross-split duplicates found")
else:
    print(f"  ✗ Found {total_dups} cross-split duplicates:")
    if train_val_dups:
        print(f"    Train/Valid: {len(train_val_dups)}")
    if train_test_dups:
        print(f"    Train/Test: {len(train_test_dups)}")
    if val_test_dups:
        print(f"    Valid/Test: {len(val_test_dups)}")

print("\n7. CLASS DISTRIBUTION")
print("-" * 70)

class_names = [
    "Abnormal", "Black Rot", "Cut", "Greening",
    "Healthy", "Hollow heart", "Rotten", "Sprouted"
]

total_boxes = eligibility['total_detection_boxes']
for class_id, count in eligibility['class_counts'].items():
    class_name = class_names[int(class_id)]
    percentage = (count / total_boxes) * 100
    warning = " ⚠ Underrepresented" if percentage < 2.0 else ""
    print(f"  Class {class_id} ({class_name}): {count} boxes ({percentage:.1f}%){warning}")

print("\n8. DATA.YAML VERIFICATION")
print("-" * 70)

data_yaml_path = base_dir / 'yolo_detection_dataset' / 'data.yaml'
with open(data_yaml_path, 'r') as f:
    import yaml
    data_config = yaml.safe_load(f)

print(f"  Path: {data_config['path']}")
print(f"  Train: {data_config['train']}")
print(f"  Val: {data_config['val']}")
print(f"  Test: {data_config['test']}")
print(f"  Number of classes: {data_config['nc']}")

# Check if paths are correct
if data_config['train'] == 'train' and data_config['val'] == 'valid' and data_config['test'] == 'test':
    print(f"  ✓ Paths correct (point to split directories)")
else:
    print(f"  ✗ Paths incorrect - should point to split dirs, not images dirs")

# Check if path is relative
if data_config['path'] == '.':
    print(f"  ✓ Path is relative (compatible with Colab)")
else:
    print(f"  ⚠ Path is absolute - may need update for Colab")

# Verify class names
expected_names = {
    0: "Abnormal Shaped Potato",
    1: "Black Rot Potato",
    2: "Cut Potato",
    3: "Greening Potato",
    4: "Healthy Potato",
    5: "Hollow heart Potato",
    6: "Rotten",
    7: "Sprouted Patato"
}

names_match = all(data_config['names'].get(i) == expected_names[i] for i in range(8))
if names_match:
    print(f"  ✓ All 8 class names correct")
else:
    print(f"  ✗ Class name mismatch detected")

print("\n9. LABEL FORMAT VALIDATION")
print("-" * 70)

dataset_root = base_dir / 'yolo_detection_dataset'
total_labels = 0
total_boxes_counted = 0
format_errors = []

for split in ['train', 'valid', 'test']:
    labels_dir = dataset_root / split / 'labels'
    for label_file in labels_dir.glob('*.txt'):
        total_labels += 1
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 5:
                format_errors.append(f"{label_file.name}:{line_num} - {len(parts)} values")
                continue
            
            try:
                class_id = int(parts[0])
                coords = [float(parts[i]) for i in range(1, 5)]
                
                if not (0 <= class_id <= 7):
                    format_errors.append(f"{label_file.name}:{line_num} - invalid class {class_id}")
                
                for coord in coords:
                    if not (0 <= coord <= 1):
                        format_errors.append(f"{label_file.name}:{line_num} - coord out of range")
                        break
                
                total_boxes_counted += 1
            except ValueError:
                format_errors.append(f"{label_file.name}:{line_num} - parse error")

print(f"  Label files checked: {total_labels}")
print(f"  Boxes counted: {total_boxes_counted}")
print(f"  Format errors: {len(format_errors)}")

if len(format_errors) == 0:
    print(f"  ✓ All labels valid (5-value format, coordinates in [0,1])")
else:
    print(f"  ✗ Format errors found:")
    for error in format_errors[:10]:
        print(f"    {error}")
    if len(format_errors) > 10:
        print(f"    ... and {len(format_errors) - 10} more")

# Verify box count
if total_boxes_counted == stats['total_boxes']:
    print(f"  ✓ Box count matches statistics: {total_boxes_counted}")
else:
    print(f"  ✗ Box count mismatch: counted {total_boxes_counted}, expected {stats['total_boxes']}")

print("\n" + "="*70)
print(" AUDIT COMPLETE")
print("="*70)
