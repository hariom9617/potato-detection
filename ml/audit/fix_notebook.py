"""Fix issues in the Colab training notebook."""

import json
from pathlib import Path

notebook_path = Path(__file__).parent.parent.parent / 'notebooks' / 'potato_yolov8n_baseline_training.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

issues_fixed = []

# Fix 1: Incorrect image path in verify_dataset cell
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'Count files in each split' in ''.join(cell['source']):
        print(f"Found verify_dataset cell at index {i}")
        # Find and fix the lines
        new_source = []
        for line in cell['source']:
            if "'images', 'train'" in line:
                line = line.replace("'images', 'train'", "'train', 'images'")
                issues_fixed.append("Fixed train image path")
            if "'images', 'val'" in line:
                line = line.replace("'images', 'val'", "'valid', 'images'")
                issues_fixed.append("Fixed val image path")
            if "'images', 'test'" in line:
                line = line.replace("'images', 'test'", "'test', 'images'")
                issues_fixed.append("Fixed test image path")
            new_source.append(line)
        cell['source'] = new_source

# Fix 2: Remove misleading cls_weights comment (not supported)
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'cls_weights=' in ''.join(cell['source']):
        print(f"Found cls_weights at cell index {i}")
        new_source = []
        skip_next = 0
        for j, line in enumerate(cell['source']):
            if skip_next > 0:
                skip_next -= 1
                continue
            
            if '# Class weights to handle imbalance' in line:
                # Replace with warning that class weights are not supported
                new_source.append("    # NOTE: YOLOv8 does not support direct class weighting via cls_weights parameter\n")
                new_source.append("    # For class imbalance, consider:\n")
                new_source.append("    #   - Data augmentation for minority classes\n")
                new_source.append("    #   - Oversampling minority class\n")
                new_source.append("    #   - Post-training threshold tuning per class\n")
                # Skip the next 3 lines (the comment about Class 5 and the cls_weights line)
                skip_next = 3
                issues_fixed.append("Fixed cls_weights comment (not supported)")
            else:
                new_source.append(line)
        cell['source'] = new_source

# Save fixed notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print(f"\nNotebook fixed. Issues resolved:")
for issue in set(issues_fixed):
    print(f"  - {issue}")
