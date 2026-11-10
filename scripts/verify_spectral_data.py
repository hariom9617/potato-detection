"""
Verify IIT spectral dataset files are readable and extract metadata.
"""
import numpy as np
from pathlib import Path
import json
import sys

def parse_envi_header(hdr_path):
    """Parse ENVI header file to extract metadata."""
    metadata = {}
    with open(hdr_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Remove braces for arrays
            if value.startswith('{') and value.endswith('}'):
                value = value[1:-1]
                # Convert to list of floats if numeric
                try:
                    value = [float(v.strip()) for v in value.split(',')]
                except:
                    value = [v.strip() for v in value.split(',')]
            
            metadata[key] = value
    
    return metadata

def read_envi_bil(bil_path, metadata):
    """Read ENVI BIL format hyperspectral cube."""
    lines = int(metadata['lines'])
    samples = int(metadata['samples'])
    bands = int(metadata['bands'])
    
    # Data type mapping (ENVI codes)
    dtype_map = {
        '1': np.uint8,
        '2': np.int16,
        '3': np.int32,
        '4': np.float32,
        '5': np.float64,
        '12': np.uint16,
        '13': np.uint32,
        '14': np.int64,
        '15': np.uint64,
    }
    
    data_type = metadata.get('data type', '12')
    dtype = dtype_map.get(data_type, np.uint16)
    
    byte_order = int(metadata.get('byte order', '0'))
    # Create dtype with proper byte order
    if byte_order == 0:
        dtype = np.dtype('<u2') if dtype == np.uint16 else np.dtype('<f4')  # Little endian
    else:
        dtype = np.dtype('>u2') if dtype == np.uint16 else np.dtype('>f4')  # Big endian
    
    # Read binary data
    data = np.fromfile(bil_path, dtype=dtype)
    
    # Reshape: BIL means Band Interleaved by Line
    # Shape should be (lines, bands, samples)
    expected_size = lines * samples * bands
    if data.size != expected_size:
        raise ValueError(f"Data size mismatch: expected {expected_size}, got {data.size}")
    
    # Reshape to (lines, bands, samples)
    data = data.reshape((lines, bands, samples))
    
    return data

def verify_file(bil_path, hdr_path):
    """Verify a single spectral file pair."""
    result = {
        'bil_file': str(bil_path),
        'hdr_file': str(hdr_path),
        'status': 'unknown',
        'error': None,
        'metadata': {},
        'data_stats': {}
    }
    
    try:
        # Parse header
        metadata = parse_envi_header(hdr_path)
        result['metadata'] = {
            'lines': metadata.get('lines'),
            'samples': metadata.get('samples'),
            'bands': metadata.get('bands'),
            'data_type': metadata.get('data type'),
            'interleave': metadata.get('interleave'),
            'byte_order': metadata.get('byte order'),
            'wavelength_range': f"{metadata.get('wavelength', ['?'])[0]} - {metadata.get('wavelength', ['?'])[-1] if isinstance(metadata.get('wavelength'), list) else '?'} nm" if 'wavelength' in metadata else 'Not specified',
            'label': metadata.get('label', 'Not specified'),
            'reflectance_scale_factor': metadata.get('reflectance scale factor'),
            'imager': metadata.get('imager serial number', 'Not specified')
        }
        
        # Read data
        data = read_envi_bil(bil_path, metadata)
        
        # Calculate statistics
        result['data_stats'] = {
            'shape': data.shape,
            'dtype': str(data.dtype),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'finite_percentage': float(np.isfinite(data).sum() / data.size * 100),
            'nan_count': int(np.isnan(data).sum()),
            'inf_count': int(np.isinf(data).sum()),
            'zero_count': int((data == 0).sum()),
            'zero_percentage': float((data == 0).sum() / data.size * 100),
            'size_mb': bil_path.stat().st_size / (1024 * 1024)
        }
        
        result['status'] = 'readable'
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
    
    return result

def main():
    # Path to spectral data
    data_dir = Path(r"D:\Important Projects\dataset\IIT DATASET\potato")
    
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        sys.exit(1)
    
    # Find all .bil files
    bil_files = sorted(data_dir.glob("*.bil"))
    
    print(f"Found {len(bil_files)} .bil files")
    print("=" * 80)
    
    results = []
    
    for bil_file in bil_files:
        # Find matching header
        hdr_candidates = [
            bil_file.with_suffix('.bil.hdr'),
            bil_file.with_suffix('.hdr'),
        ]
        
        hdr_file = None
        for candidate in hdr_candidates:
            if candidate.exists():
                hdr_file = candidate
                break
        
        if hdr_file is None:
            print(f"[X] No header found for {bil_file.name}")
            results.append({
                'bil_file': str(bil_file),
                'hdr_file': 'NOT FOUND',
                'status': 'error',
                'error': 'Missing header file'
            })
            continue
        
        print(f"\nVerifying: {bil_file.name}")
        result = verify_file(bil_file, hdr_file)
        results.append(result)
        
        if result['status'] == 'readable':
            print(f"  [OK] Status: {result['status']}")
            print(f"  [DATA] Shape: {result['data_stats']['shape']}")
            print(f"  [RANGE] Range: [{result['data_stats']['min']:.2f}, {result['data_stats']['max']:.2f}]")
            print(f"  [LABEL] Label: {result['metadata']['label']}")
            print(f"  [WAVE] Wavelength: {result['metadata']['wavelength_range']}")
        else:
            print(f"  [ERROR] Status: {result['status']}")
            print(f"  [WARN] Error: {result['error']}")
    
    # Save results
    output_file = Path(r"D:\Important Projects\docs\spectral_file_inventory.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"[SAVED] Results saved to: {output_file}")
    
    # Summary
    readable = sum(1 for r in results if r['status'] == 'readable')
    errors = sum(1 for r in results if r['status'] == 'error')
    
    print(f"\n[SUMMARY]")
    print(f"  Total files: {len(results)}")
    print(f"  Readable: {readable}")
    print(f"  Errors: {errors}")

if __name__ == "__main__":
    main()
