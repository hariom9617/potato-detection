"""
Annotation Diagnostic Tools

Tools for analyzing and diagnosing annotation format issues in datasets.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter


def analyze_invalid_annotations(annotations_csv_path: Path) -> Dict:
    """
    Analyze invalid annotation files and classify error types.
    
    Args:
        annotations_csv_path: Path to annotations CSV from audit
        
    Returns:
        Dictionary with error analysis
    """
    import pandas as pd
    
    df = pd.read_csv(annotations_csv_path)
    invalid = df[df['is_valid'] == False]
    
    # Extract value counts from error messages
    value_counts = []
    for _, row in invalid.iterrows():
        error = row['error_message']
        if 'Expected 5 values, got' in error:
            try:
                count = int(error.split('got ')[-1])
                value_counts.append(count)
            except:
                pass
    
    analysis = {
        'total_invalid': len(invalid),
        'wrong_value_count': len([e for e in invalid['error_message'] if 'Expected 5 values' in e]),
        'value_count_stats': {
            'min': min(value_counts) if value_counts else 0,
            'max': max(value_counts) if value_counts else 0,
            'mean': sum(value_counts) / len(value_counts) if value_counts else 0,
        },
        'error_types': dict(Counter(invalid['error_message'])),
        'sample_files': invalid.head(10).to_dict('records')
    }
    
    return analysis


def detect_segmentation_format(label_file_path: Path) -> Tuple[bool, str]:
    """
    Check if a label file contains segmentation (polygon) annotations.
    
    YOLO detection format: class_id x_center y_center width height (5 values)
    YOLO segmentation format: class_id x1 y1 x2 y2 x3 y3 ... (variable, always odd)
    
    Args:
        label_file_path: Path to label file
        
    Returns:
        Tuple of (is_segmentation, evidence)
    """
    try:
        with open(label_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            return False, "Empty file"
        
        first_line = lines[0].strip()
        if not first_line:
            return False, "Empty first line"
        
        parts = first_line.split()
        num_values = len(parts)
        
        # Detection format has exactly 5 values
        if num_values == 5:
            return False, "Standard YOLO detection format (5 values)"
        
        # Segmentation format has class_id + pairs of coordinates
        # So total values should be odd and > 5
        if num_values > 5 and num_values % 2 == 1:
            # Check if all values after class_id are valid floats in [0, 1]
            try:
                class_id = int(parts[0])
                coords = [float(p) for p in parts[1:]]
                
                # Check if coordinates are normalized
                in_range = all(0 <= c <= 1 for c in coords)
                
                if in_range:
                    num_points = (num_values - 1) // 2
                    return True, f"YOLO segmentation format ({num_points} polygon points)"
                else:
                    return False, f"Invalid coordinate ranges (not in [0,1])"
            except ValueError:
                return False, "Non-numeric values"
        
        return False, f"Unknown format ({num_values} values)"
        
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m datasets.diagnose <annotations.csv>")
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    analysis = analyze_invalid_annotations(csv_path)
    
    print(json.dumps(analysis, indent=2))
