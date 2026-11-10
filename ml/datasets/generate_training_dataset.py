"""
Generate Training-Ready YOLO Detection Dataset.

Creates a reproducible, training-ready dataset outside the source directory,
extracting only valid detection annotations while preserving provenance.
"""

import shutil
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import json

import pandas as pd
from tqdm import tqdm

from datasets.validation import validate_annotation_file, AnnotationFormat


@dataclass
class DatasetEntry:
    """Single entry in the generated dataset."""
    image_path: str
    label_path: str
    source_image: str
    source_label: str
    split: str
    annotation_count: int
    class_ids: List[int]
    format_source: str  # 'detection_only', 'mixed_extraction', 'empty'


class TrainingDatasetGenerator:
    """Generate YOLO detection training dataset."""
    
    def __init__(
        self,
        source_root: Path,
        output_root: Path,
        manifests_dir: Path,
        class_names: Dict[int, str]
    ):
        """
        Initialize generator.
        
        Args:
            source_root: Source dataset root
            output_root: Output directory for generated dataset
            manifests_dir: Directory containing manifest CSVs
            class_names: Mapping of class IDs to names
        """
        self.source_root = source_root
        self.output_root = output_root
        self.manifests_dir = manifests_dir
        self.class_names = class_names
        
        # Load manifests
        self.det_manifest = pd.read_csv(manifests_dir / 'manifest_detection.csv')
        self.mix_manifest = pd.read_csv(manifests_dir / 'manifest_mixed.csv')
        self.inv_manifest = pd.read_csv(manifests_dir / 'manifest_invalid.csv')
        
        # Filter empty labels
        self.empty_labels = self.inv_manifest[self.inv_manifest['label_format'] == 'empty']
        
        self.entries: List[DatasetEntry] = []
        self.stats = {
            'images_processed': 0,
            'detection_only': 0,
            'mixed_extraction': 0,
            'empty': 0,  # Changed from 'empty_labels' to match format_source value
            'total_boxes': 0,
            'by_split': defaultdict(lambda: {
                'images': 0,
                'boxes': 0,
                'by_class': defaultdict(int)
            })
        }
    
    def extract_detection_lines(self, source_label: Path) -> List[str]:
        """
        Extract detection-format lines from a label file.
        
        Handles both detection-only and mixed-format files.
        
        Args:
            source_label: Path to source label file
            
        Returns:
            List of detection lines (5-value format)
        """
        validation = validate_annotation_file(source_label)
        
        detection_lines = []
        for line in validation.lines:
            if line.is_valid and line.format == AnnotationFormat.DETECTION:
                # Reconstruct YOLO detection line
                yolo_line = f"{line.class_id} {' '.join(map(str, line.values))}"
                detection_lines.append(yolo_line)
        
        return detection_lines
    
    def process_image(
        self,
        source_image: Path,
        source_label: Path,
        split: str,
        format_source: str
    ) -> Optional[DatasetEntry]:
        """
        Process a single image and generate output files.
        
        Args:
            source_image: Source image path
            source_label: Source label path
            split: train, valid, or test
            format_source: Source format type
            
        Returns:
            DatasetEntry if successful, None otherwise
        """
        # Verify source files exist
        if not source_image.exists():
            print(f"WARNING: Source image not found: {source_image}")
            return None
        
        if not source_label.exists() and format_source != 'empty':
            print(f"WARNING: Source label not found: {source_label}")
            return None
        
        # Create output directories
        output_images_dir = self.output_root / split / 'images'
        output_labels_dir = self.output_root / split / 'labels'
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filenames
        output_image = output_images_dir / source_image.name
        output_label = output_labels_dir / f"{source_image.stem}.txt"
        
        # Copy image
        shutil.copy2(source_image, output_image)
        
        # Extract and write detection lines
        if format_source == 'empty':
            # Create empty label file
            output_label.write_text('')
            detection_lines = []
        else:
            detection_lines = self.extract_detection_lines(source_label)
            output_label.write_text('\n'.join(detection_lines))
        
        # Extract class IDs
        class_ids = []
        for line in detection_lines:
            parts = line.split()
            if len(parts) >= 5:
                class_ids.append(int(parts[0]))
        
        # Create entry
        entry = DatasetEntry(
            image_path=str(output_image),
            label_path=str(output_label),
            source_image=str(source_image),
            source_label=str(source_label) if format_source != 'empty' else '',
            split=split,
            annotation_count=len(detection_lines),
            class_ids=sorted(set(class_ids)),
            format_source=format_source
        )
        
        # Update stats
        self.stats['images_processed'] += 1
        self.stats[format_source] += 1
        self.stats['total_boxes'] += len(detection_lines)
        
        split_stats = self.stats['by_split'][split]
        split_stats['images'] += 1
        split_stats['boxes'] += len(detection_lines)
        
        for class_id in class_ids:
            split_stats['by_class'][class_id] += 1
        
        return entry
    
    def generate(self):
        """Generate the complete training dataset."""
        print("="*80)
        print("GENERATING TRAINING DATASET")
        print("="*80)
        
        # Process detection-only files
        print(f"\n1. Processing {len(self.det_manifest)} detection-only files...")
        for _, row in tqdm(self.det_manifest.iterrows(), total=len(self.det_manifest), desc="Detection-only"):
            source_image = Path(row['image_path'])
            source_label = Path(row['label_path'])
            split = row['split']
            
            entry = self.process_image(source_image, source_label, split, 'detection_only')
            if entry:
                self.entries.append(entry)
        
        # Process mixed-format files
        print(f"\n2. Processing {len(self.mix_manifest)} mixed-format files (extracting detection boxes)...")
        for _, row in tqdm(self.mix_manifest.iterrows(), total=len(self.mix_manifest), desc="Mixed-format"):
            source_image = Path(row['image_path'])
            source_label = Path(row['label_path'])
            split = row['split']
            
            entry = self.process_image(source_image, source_label, split, 'mixed_extraction')
            if entry:
                self.entries.append(entry)
        
        # Process empty labels
        print(f"\n3. Processing {len(self.empty_labels)} empty-label files (negative samples)...")
        for _, row in tqdm(self.empty_labels.iterrows(), total=len(self.empty_labels), desc="Empty labels"):
            source_image = Path(row['image_path'])
            source_label = Path(row['label_path']) if row['label_path'] else None
            split = row['split']
            
            entry = self.process_image(source_image, source_label, split, 'empty')
            if entry:
                self.entries.append(entry)
        
        print(f"\n4. GENERATION COMPLETE")
        print(f"   Total images generated: {self.stats['images_processed']}")
        print(f"   Detection-only: {self.stats['detection_only']}")
        print(f"   Mixed-format extraction: {self.stats['mixed_extraction']}")
        print(f"   Empty labels: {self.stats['empty']}")
        print(f"   Total detection boxes: {self.stats['total_boxes']}")
    
    def verify_dataset(self):
        """Verify the generated dataset."""
        print(f"\n5. DATASET VERIFICATION")
        
        errors = []
        
        for entry in tqdm(self.entries, desc="Verifying"):
            # Check image exists
            if not Path(entry.image_path).exists():
                errors.append(f"Missing image: {entry.image_path}")
            
            # Check label exists
            if not Path(entry.label_path).exists():
                errors.append(f"Missing label: {entry.label_path}")
            
            # Verify label content
            label_content = Path(entry.label_path).read_text().strip()
            if entry.format_source == 'empty':
                if label_content:
                    errors.append(f"Empty label should be empty: {entry.label_path}")
            else:
                lines = [l for l in label_content.split('\n') if l.strip()]
                if len(lines) != entry.annotation_count:
                    errors.append(f"Annotation count mismatch in {entry.label_path}: expected {entry.annotation_count}, got {len(lines)}")
                
                # Verify each line is valid detection format
                for i, line in enumerate(lines, 1):
                    parts = line.split()
                    if len(parts) != 5:
                        errors.append(f"Invalid detection format in {entry.label_path} line {i}: {len(parts)} values")
                    else:
                        try:
                            class_id = int(parts[0])
                            coords = [float(x) for x in parts[1:5]]
                            
                            # Check class ID range
                            if class_id < 0 or class_id > 7:
                                errors.append(f"Invalid class ID {class_id} in {entry.label_path} line {i}")
                            
                            # Check coordinate bounds
                            for coord in coords:
                                if coord < 0 or coord > 1:
                                    errors.append(f"Coordinate out of range in {entry.label_path} line {i}: {coord}")
                        except ValueError as e:
                            errors.append(f"Parse error in {entry.label_path} line {i}: {e}")
        
        if errors:
            print(f"\n   WARNING: Found {len(errors)} verification errors:")
            for error in errors[:10]:  # Show first 10
                print(f"      - {error}")
            if len(errors) > 10:
                print(f"      ... and {len(errors) - 10} more")
        else:
            print(f"   OK: All {len(self.entries)} entries verified successfully")
    
    def save_manifest(self):
        """Save dataset manifest and statistics."""
        print(f"\n6. SAVING MANIFEST AND STATISTICS")
        
        # Save entry manifest
        manifest_path = self.output_root / 'dataset_manifest.csv'
        df = pd.DataFrame([asdict(entry) for entry in self.entries])
        df.to_csv(manifest_path, index=False)
        print(f"   Manifest saved: {manifest_path}")
        
        # Save statistics
        stats_path = self.output_root / 'dataset_statistics.json'
        
        # Convert defaultdicts for JSON serialization
        serializable_stats = {
            'images_processed': self.stats['images_processed'],
            'detection_only': self.stats['detection_only'],
            'mixed_extraction': self.stats['mixed_extraction'],
            'empty': self.stats['empty'],  # Use 'empty' not 'empty_labels'
            'total_boxes': self.stats['total_boxes'],
            'by_split': {
                split: {
                    'images': data['images'],
                    'boxes': data['boxes'],
                    'by_class': dict(data['by_class'])
                }
                for split, data in self.stats['by_split'].items()
            }
        }
        
        with open(stats_path, 'w') as f:
            json.dump(serializable_stats, f, indent=2)
        print(f"   Statistics saved: {stats_path}")
        
        # Print summary
        print(f"\n   SUMMARY BY SPLIT:")
        for split in ['train', 'valid', 'test']:
            if split in self.stats['by_split']:
                data = self.stats['by_split'][split]
                print(f"     {split:6s}: {data['images']:4d} images, {data['boxes']:5d} boxes")
        
        print(f"\n   PER-CLASS BOX COUNT:")
        # Aggregate across splits
        total_by_class = defaultdict(int)
        for split_data in self.stats['by_split'].values():
            for class_id, count in split_data['by_class'].items():
                total_by_class[class_id] += count
        
        total = sum(total_by_class.values())
        for class_id in sorted(total_by_class.keys()):
            count = total_by_class[class_id]
            pct = 100.0 * count / total if total > 0 else 0
            class_name = self.class_names.get(class_id, 'Unknown')
            print(f"     Class {class_id} ({class_name:25s}): {count:5d} ({pct:5.1f}%)")
    
    def generate_data_yaml(self):
        """Generate data.yaml configuration file."""
        print(f"\n7. GENERATING data.yaml")
        
        yaml_content = f"""# YOLOv8 Detection Dataset Configuration
# Generated from: {self.source_root}
# Date: 2026-10-18

path: {self.output_root.absolute()}
train: train/images
val: valid/images
test: test/images

nc: {len(self.class_names)}
names:
"""
        
        for class_id in sorted(self.class_names.keys()):
            yaml_content += f"  {class_id}: {self.class_names[class_id]}\n"
        
        yaml_path = self.output_root / 'data.yaml'
        yaml_path.write_text(yaml_content)
        print(f"   data.yaml saved: {yaml_path}")


def main():
    """CLI entry point."""
    # Configuration
    source_root = Path(r"D:\Important Projects\dataset\annoted data set labeled")
    output_root = Path(r"D:\Important Projects\yolo_detection_dataset")
    manifests_dir = Path(r"D:\Important Projects\dataset-manifests")
    
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
    
    # Create generator
    generator = TrainingDatasetGenerator(
        source_root=source_root,
        output_root=output_root,
        manifests_dir=manifests_dir,
        class_names=class_names
    )
    
    # Generate dataset
    generator.generate()
    
    # Verify dataset
    generator.verify_dataset()
    
    # Save manifest and stats
    generator.save_manifest()
    
    # Generate data.yaml
    generator.generate_data_yaml()
    
    print("\n" + "="*80)
    print("DATASET GENERATION COMPLETE")
    print("="*80)
    print(f"\nDataset location: {output_root}")
    print(f"Configuration: {output_root / 'data.yaml'}")
    print(f"Manifest: {output_root / 'dataset_manifest.csv'}")
    print(f"Statistics: {output_root / 'dataset_statistics.json'}")
    

if __name__ == "__main__":
    main()
