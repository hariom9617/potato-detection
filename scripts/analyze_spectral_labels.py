"""
Analyze spectral dataset labels and categorize samples.
"""
import json
from pathlib import Path
from collections import Counter
import csv

def parse_label(label_str):
    """
    Parse label string to extract category and sample number.
    
    Label patterns observed:
    - gr1, gr2, ..., gr5 -> likely "greening" 
    - Healthy1, Healthy2, Healthy3 -> Healthy
    - Rott1, Rott2, ..., Rott5 -> Rotten/Rot
    - Shr1, shr2, ..., shr7 -> likely "shriveled" or "shrinkage"
    - spr1, spr2, ..., spr6 -> likely "sprouted"
    """
    label = label_str.lower()
    
    if label.startswith('healthy'):
        return 'Healthy', label
    elif label.startswith('rott'):
        return 'Rotten', label
    elif label.startswith('gr'):
        return 'Greening', label
    elif label.startswith('shr'):
        return 'Shriveled', label
    elif label.startswith('spr'):
        return 'Sprouted', label
    else:
        return 'Unknown', label

def main():
    # Load inventory
    inventory_path = Path(r"D:\Important Projects\docs\spectral_file_inventory.json")
    with open(inventory_path, 'r') as f:
        inventory = json.load(f)
    
    # Analyze labels
    categories = Counter()
    samples_by_category = {}
    readable_samples = []
    orphaned_files = []
    saturated_files = []
    
    for item in inventory:
        if item['status'] == 'error':
            orphaned_files.append(item['bil_file'])
            continue
        
        label = item['metadata']['label']
        category, sample_id = parse_label(label)
        
        categories[category] += 1
        if category not in samples_by_category:
            samples_by_category[category] = []
        
        # Check for saturation
        max_val = item['data_stats']['max']
        is_saturated = max_val >= 65535
        
        sample_info = {
            'sample_id': sample_id,
            'category': category,
            'bil_file': Path(item['bil_file']).name,
            'shape': item['data_stats']['shape'],
            'wavelength_range': item['metadata']['wavelength_range'],
            'value_range': f"[{item['data_stats']['min']:.0f}, {item['data_stats']['max']:.0f}]",
            'mean': f"{item['data_stats']['mean']:.2f}",
            'std': f"{item['data_stats']['std']:.2f}",
            'saturated': is_saturated,
            'zero_percent': f"{item['data_stats']['zero_percentage']:.2f}%"
        }
        
        samples_by_category[category].append(sample_info)
        readable_samples.append(sample_info)
        
        if is_saturated:
            saturated_files.append(sample_id)
    
    # Generate report
    print("=" * 80)
    print("IIT SPECTRAL DATASET LABEL ANALYSIS")
    print("=" * 80)
    
    print(f"\n[INVENTORY SUMMARY]")
    print(f"  Total files found: {len(inventory)}")
    print(f"  Readable files: {len(readable_samples)}")
    print(f"  Orphaned files (missing headers): {len(orphaned_files)}")
    print(f"  Files with saturation: {len(saturated_files)}")
    
    print(f"\n[LABEL CATEGORIES]")
    for category, count in sorted(categories.items()):
        print(f"  {category}: {count} samples")
    
    print(f"\n[SAMPLES BY CATEGORY]")
    for category in sorted(samples_by_category.keys()):
        samples = samples_by_category[category]
        print(f"\n  {category} ({len(samples)} samples):")
        for sample in samples:
            sat_flag = " [SATURATED]" if sample['saturated'] else ""
            print(f"    - {sample['sample_id']}: range={sample['value_range']}{sat_flag}")
    
    if orphaned_files:
        print(f"\n[ORPHANED FILES]")
        for f in orphaned_files:
            print(f"    - {Path(f).name}")
    
    if saturated_files:
        print(f"\n[SATURATED SAMPLES WARNING]")
        print(f"  The following samples contain saturated pixels (value=65535):")
        for s in saturated_files:
            print(f"    - {s}")
        print(f"  This may indicate overexposure or calibration issues.")
    
    # Save CSV
    csv_path = Path(r"D:\Important Projects\docs\spectral_samples_catalog.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['sample_id', 'category', 'bil_file', 'shape', 'wavelength_range', 
                      'value_range', 'mean', 'std', 'saturated', 'zero_percent']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in readable_samples:
            writer.writerow(sample)
    
    print(f"\n[SAVED]")
    print(f"  Catalog saved to: {csv_path}")
    
    # Ground truth assessment
    print(f"\n[GROUND TRUTH ASSESSMENT]")
    print(f"  Labels derived from: Filename patterns")
    print(f"  Label reliability: UNVERIFIED")
    print(f"  Note: These are filename-based labels, not verified ground truth.")
    print(f"        Without physical inspection records or destructive testing")
    print(f"        documentation, these labels should be treated as WEAK LABELS.")
    print(f"")
    print(f"  Interpretation of filename patterns:")
    print(f"    - 'Healthy': Assumed externally and internally healthy")
    print(f"    - 'Rott': Assumed to have internal rot defects")
    print(f"    - 'Greening' (gr): Likely external greening, unclear if internal defect")
    print(f"    - 'Shriveled' (shr): External appearance, may not indicate internal defect")
    print(f"    - 'Sprouted' (spr): External feature, may not indicate internal defect")
    
    print(f"\n[INTERNAL DEFECT RELEVANCE]")
    print(f"  For INTERNAL defect detection:")
    print(f"    - Healthy: Likely valid negative class")
    print(f"    - Rott: Likely valid positive class (internal rot)")
    print(f"    - Greening: UNCERTAIN - may be external only")
    print(f"    - Shriveled: UNCERTAIN - primarily external")
    print(f"    - Sprouted: UNCERTAIN - primarily external")
    
    print(f"\n[RECOMMENDATION]")
    if categories.get('Healthy', 0) >= 3 and categories.get('Rotten', 0) >= 3:
        print(f"  Binary classification (Healthy vs Rotten) appears FEASIBLE")
        print(f"  with {categories['Healthy']} healthy and {categories['Rotten']} rotten samples.")
        print(f"  However, labels are unverified and sample size is very small.")
    else:
        print(f"  Insufficient labeled samples for reliable model training.")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
