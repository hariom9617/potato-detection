"""Verify data.yaml configuration for YOLO training."""

import yaml
import os
from pathlib import Path

# Load data.yaml
data_yaml_path = Path(__file__).parent.parent.parent / 'yolo_detection_dataset' / 'data.yaml'
with open(data_yaml_path, 'r') as f:
    config = yaml.safe_load(f)

print("="*60)
print("DATA.YAML CONFIGURATION")
print("="*60)
print(f"Path: {config.get('path', 'NOT SET')}")
print(f"Train: {config.get('train', 'NOT SET')}")
print(f"Val: {config.get('val', 'NOT SET')}")
print(f"Test: {config.get('test', 'NOT SET')}")
print(f"Number of classes: {config.get('nc', 'NOT SET')}")
print(f"\nClass names:")
for class_id, class_name in config.get('names', {}).items():
    print(f"  {class_id}: {class_name}")

# Check if paths are correct
print("\n" + "="*60)
print("PATH VERIFICATION")
print("="*60)

base_path = Path(config['path'])
print(f"Base path exists: {base_path.exists()}")

# YOLOv8 expects paths relative to 'path', pointing to directories that contain images/ and labels/
train_dir = base_path / config['train']
val_dir = base_path / config['val']
test_dir = base_path / config['test']

print(f"\nExpected structure:")
print(f"  {train_dir} -> should contain images/ and labels/")
print(f"  {val_dir} -> should contain images/ and labels/")
print(f"  {test_dir} -> should contain images/ and labels/")

# Check if the current paths point to images directories or split directories
print(f"\nCurrent path resolution:")
print(f"  train path exists: {train_dir.exists()}")
print(f"  val path exists: {val_dir.exists()}")
print(f"  test path exists: {test_dir.exists()}")

# Determine the actual structure
if 'images' in str(config['train']):
    print("\n⚠️  ISSUE: Paths point to 'images' subdirectories")
    print("  YOLOv8 expects paths to the split directory (train/val/test)")
    print("  Current: train/images")
    print("  Should be: train")
    
    # Check if correct structure exists
    correct_train = base_path / 'train'
    correct_val = base_path / 'valid'
    correct_test = base_path / 'test'
    
    if all(p.exists() for p in [correct_train, correct_val, correct_test]):
        print("\n  Correct directories exist:")
        print(f"    {correct_train} -> {correct_train.exists()}")
        print(f"    {correct_val} -> {correct_val.exists()}")
        print(f"    {correct_test} -> {correct_test.exists()}")
        
        # Verify they have images/ and labels/
        for split_name, split_dir in [('train', correct_train), ('valid', correct_val), ('test', correct_test)]:
            img_dir = split_dir / 'images'
            lbl_dir = split_dir / 'labels'
            if img_dir.exists() and lbl_dir.exists():
                img_count = len(list(img_dir.glob('*')))
                lbl_count = len(list(lbl_dir.glob('*')))
                print(f"    {split_name}: {img_count} images, {lbl_count} labels ✓")
            else:
                print(f"    {split_name}: Missing subdirectories ✗")
else:
    print("\n✓ Paths appear correct (point to split directories)")

print("\n" + "="*60)
print("CLASS VERIFICATION")
print("="*60)
expected_classes = [
    "Abnormal Shaped Potato",
    "Black Rot Potato", 
    "Cut Potato",
    "Greening Potato",
    "Healthy Potato",
    "Hollow heart Potato",
    "Rotten",
    "Sprouted Patato"
]

names_dict = config.get('names', {})
if config.get('nc') == 8 and len(names_dict) == 8:
    print("✓ Number of classes correct (8)")
    
    all_match = True
    for i, expected_name in enumerate(expected_classes):
        actual_name = names_dict.get(i, "MISSING")
        if actual_name == expected_name:
            print(f"  Class {i}: {actual_name} ✓")
        else:
            print(f"  Class {i}: Expected '{expected_name}', got '{actual_name}' ✗")
            all_match = False
    
    if all_match:
        print("\n✓ All class names match expected values")
else:
    print(f"✗ Class count mismatch: nc={config.get('nc')}, names count={len(names_dict)}")
