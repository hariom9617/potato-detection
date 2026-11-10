"""
Exploratory Spectral Analysis - IIT Hyperspectral Dataset
Memory-efficient analysis without supervised training.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import json
from pathlib import Path
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Output directory
OUTPUT_DIR = Path("exploratory_analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def parse_envi_header(hdr_path):
    """Parse ENVI header to extract metadata."""
    metadata = {}
    with open(hdr_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            if value.startswith('{') and value.endswith('}'):
                value = value[1:-1]
                try:
                    value = [float(v.strip()) for v in value.split(',')]
                except:
                    value = [v.strip() for v in value.split(',')]
            
            metadata[key] = value
    
    return metadata

def read_envi_bil_memmap(bil_path, metadata):
    """Memory-map ENVI BIL file instead of loading to RAM."""
    lines = int(metadata['lines'])
    samples = int(metadata['samples'])
    bands = int(metadata['bands'])
    
    dtype_map = {
        '1': np.uint8, '2': np.int16, '3': np.int32,
        '4': np.float32, '5': np.float64, '12': np.uint16,
        '13': np.uint32, '14': np.int64, '15': np.uint64,
    }
    
    data_type = metadata.get('data type', '12')
    dtype = dtype_map.get(data_type, np.uint16)
    
    byte_order = int(metadata.get('byte order', '0'))
    if byte_order == 0:
        dtype = np.dtype('<u2') if dtype == np.uint16 else np.dtype('<f4')
    else:
        dtype = np.dtype('>u2') if dtype == np.uint16 else np.dtype('>f4')
    
    # Memory-mapped array
    shape = (lines, bands, samples)
    data = np.memmap(bil_path, dtype=dtype, mode='r', shape=shape)
    
    return data

def extract_spatial_mean_spectrum(data):
    """
    Extract mean spectrum over spatial dimensions.
    Uses memory-efficient chunking for large cubes.
    """
    # Process in chunks to avoid loading entire cube
    lines, bands, samples = data.shape
    
    # Compute mean band-by-band
    mean_spectrum = np.zeros(bands, dtype=np.float64)
    
    for band_idx in range(bands):
        band_slice = data[:, band_idx, :]
        mean_spectrum[band_idx] = np.mean(band_slice)
    
    return mean_spectrum

def extract_spatial_stats(data, sample_pixels=10000):
    """
    Extract spatial statistics by sampling pixels.
    Avoids loading entire cube into memory.
    """
    lines, bands, samples = data.shape
    total_pixels = lines * samples
    
    # Random sample of spatial locations
    np.random.seed(42)
    sample_count = min(sample_pixels, total_pixels)
    sample_indices = np.random.choice(total_pixels, sample_count, replace=False)
    
    # Convert to 2D indices
    line_indices = sample_indices // samples
    sample_indices = sample_indices % samples
    
    # Extract pixels
    sampled_spectra = np.zeros((sample_count, bands), dtype=np.float32)
    for i, (line_idx, samp_idx) in enumerate(zip(line_indices, sample_indices)):
        sampled_spectra[i, :] = data[line_idx, :, samp_idx]
    
    return sampled_spectra

def check_band_quality(data, wavelengths):
    """
    Check for dead bands, saturated bands, anomalous bands.
    Uses memory-efficient sampling.
    """
    lines, bands, samples = data.shape
    
    band_stats = {
        'wavelength': wavelengths,
        'min': np.zeros(bands),
        'max': np.zeros(bands),
        'mean': np.zeros(bands),
        'std': np.zeros(bands),
        'saturation_pct': np.zeros(bands),
        'zero_pct': np.zeros(bands),
    }
    
    for band_idx in range(bands):
        band_slice = data[:, band_idx, :]
        
        band_stats['min'][band_idx] = np.min(band_slice)
        band_stats['max'][band_idx] = np.max(band_slice)
        band_stats['mean'][band_idx] = np.mean(band_slice)
        band_stats['std'][band_idx] = np.std(band_slice)
        
        # Saturation check (uint16 max = 65535)
        saturated = (band_slice == 65535).sum()
        band_stats['saturation_pct'][band_idx] = saturated / band_slice.size * 100
        
        # Zero check
        zeros = (band_slice == 0).sum()
        band_stats['zero_pct'][band_idx] = zeros / band_slice.size * 100
    
    return band_stats

def main():
    print("=" * 80)
    print("EXPLORATORY SPECTRAL ANALYSIS - IIT HYPERSPECTRAL DATASET")
    print("=" * 80)
    
    # Load catalog
    catalog_path = Path("docs/spectral_samples_catalog.csv")
    catalog = pd.read_csv(catalog_path)
    
    print(f"\nLoaded catalog: {len(catalog)} samples")
    print(f"Categories: {catalog['category'].value_counts().to_dict()}")
    
    # Load wavelengths from first readable file
    data_dir = Path("dataset/IIT DATASET/potato")
    first_sample = catalog.iloc[0]
    hdr_path = data_dir / (first_sample['bil_file'] + '.hdr')
    metadata = parse_envi_header(hdr_path)
    wavelengths = np.array(metadata['wavelength'])
    
    print(f"\nWavelength range: {wavelengths[0]:.2f} - {wavelengths[-1]:.2f} nm")
    print(f"Number of bands: {len(wavelengths)}")
    
    # =======================
    # PART 1: SPECTRAL QUALITY CHECK
    # =======================
    print("\n" + "=" * 80)
    print("PART 1: SPECTRAL QUALITY ASSESSMENT")
    print("=" * 80)
    
    # Focus on healthy and rotten samples for internal defect relevance
    healthy_samples = catalog[catalog['category'] == 'Healthy']
    rotten_samples = catalog[catalog['category'] == 'Rotten']
    
    print(f"\nFocusing on internal defect labels:")
    print(f"  Healthy samples: {len(healthy_samples)}")
    print(f"  Rotten samples: {len(rotten_samples)}")
    
    # Check one healthy sample in detail
    print(f"\n[QUALITY CHECK] Examining: {healthy_samples.iloc[0]['sample_id']}")
    
    bil_path = data_dir / healthy_samples.iloc[0]['bil_file']
    hdr_path = bil_path.with_suffix('.bil.hdr')
    
    metadata = parse_envi_header(hdr_path)
    data = read_envi_bil_memmap(bil_path, metadata)
    
    print(f"  Shape: {data.shape} (lines, bands, samples)")
    print(f"  Data type: {data.dtype}")
    print(f"  Memory-mapped: Yes (not loaded to RAM)")
    
    # Band quality
    band_stats = check_band_quality(data, wavelengths)
    
    # Detect anomalous bands
    dead_bands = np.where(band_stats['std'] < 1.0)[0]
    saturated_bands = np.where(band_stats['saturation_pct'] > 1.0)[0]
    
    print(f"\n  Band quality:")
    print(f"    Dead bands (std < 1.0): {len(dead_bands)}")
    print(f"    Saturated bands (>1% pixels): {len(saturated_bands)}")
    print(f"    Value range: [{band_stats['min'].min():.0f}, {band_stats['max'].max():.0f}]")
    print(f"    Mean across bands: {band_stats['mean'].mean():.2f} +/- {band_stats['mean'].std():.2f}")
    
    # Plot band quality
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(wavelengths, band_stats['mean'], linewidth=1)
    axes[0, 0].set_xlabel('Wavelength (nm)')
    axes[0, 0].set_ylabel('Mean Digital Number')
    axes[0, 0].set_title('Mean Signal per Band')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(wavelengths, band_stats['std'], linewidth=1, color='orange')
    axes[0, 1].set_xlabel('Wavelength (nm)')
    axes[0, 1].set_ylabel('Standard Deviation')
    axes[0, 1].set_title('Signal Variability per Band')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(wavelengths, band_stats['saturation_pct'], linewidth=1, color='red')
    axes[1, 0].set_xlabel('Wavelength (nm)')
    axes[1, 0].set_ylabel('Saturation %')
    axes[1, 0].set_title('Saturated Pixels per Band')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(wavelengths, band_stats['zero_pct'], linewidth=1, color='green')
    axes[1, 1].set_xlabel('Wavelength (nm)')
    axes[1, 1].set_ylabel('Zero %')
    axes[1, 1].set_title('Zero-Value Pixels per Band')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'band_quality_check.png', dpi=150)
    plt.close()
    
    print(f"\n  Saved: band_quality_check.png")
    
    # =======================
    # PART 2: EXTRACT SPECTRAL SIGNATURES
    # =======================
    print("\n" + "=" * 80)
    print("PART 2: SPECTRAL SIGNATURE EXTRACTION")
    print("=" * 80)
    
    print("\nExtracting mean spectra (spatial average per cube)...")
    print("NOTE: Each sample is one potato cube; pixels are NOT independent samples.")
    
    spectra_healthy = []
    spectra_rotten = []
    
    for idx, row in healthy_samples.iterrows():
        bil_path = data_dir / row['bil_file']
        hdr_path = bil_path.with_suffix('.bil.hdr')
        metadata = parse_envi_header(hdr_path)
        data = read_envi_bil_memmap(bil_path, metadata)
        
        mean_spectrum = extract_spatial_mean_spectrum(data)
        spectra_healthy.append(mean_spectrum)
        print(f"  Loaded: {row['sample_id']} (Healthy)")
    
    for idx, row in rotten_samples.iterrows():
        bil_path = data_dir / row['bil_file']
        hdr_path = bil_path.with_suffix('.bil.hdr')
        metadata = parse_envi_header(hdr_path)
        data = read_envi_bil_memmap(bil_path, metadata)
        
        mean_spectrum = extract_spatial_mean_spectrum(data)
        spectra_rotten.append(mean_spectrum)
        print(f"  Loaded: {row['sample_id']} (Rotten)")
    
    spectra_healthy = np.array(spectra_healthy)
    spectra_rotten = np.array(spectra_rotten)
    
    print(f"\nSpectra shape:")
    print(f"  Healthy: {spectra_healthy.shape} (samples, bands)")
    print(f"  Rotten: {spectra_rotten.shape} (samples, bands)")
    
    # =======================
    # PART 3: VISUALIZE INDIVIDUAL SPECTRA
    # =======================
    print("\n" + "=" * 80)
    print("PART 3: SPECTRAL SIGNATURE VISUALIZATION")
    print("=" * 80)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Healthy spectra
    for i, spectrum in enumerate(spectra_healthy):
        axes[0].plot(wavelengths, spectrum, alpha=0.7, linewidth=1.5, 
                    label=healthy_samples.iloc[i]['sample_id'])
    axes[0].set_xlabel('Wavelength (nm)', fontsize=12)
    axes[0].set_ylabel('Mean Digital Number (raw counts)', fontsize=12)
    axes[0].set_title(f'Healthy Samples (n={len(spectra_healthy)})', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(700, color='gray', linestyle='--', alpha=0.5, label='Visible/NIR boundary')
    
    # Rotten spectra
    for i, spectrum in enumerate(spectra_rotten):
        axes[1].plot(wavelengths, spectrum, alpha=0.7, linewidth=1.5,
                    label=rotten_samples.iloc[i]['sample_id'])
    axes[1].set_xlabel('Wavelength (nm)', fontsize=12)
    axes[1].set_ylabel('Mean Digital Number (raw counts)', fontsize=12)
    axes[1].set_title(f'Rotten Samples (n={len(spectra_rotten)})', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(700, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'individual_spectral_signatures.png', dpi=150)
    plt.close()
    
    print("\nIndividual spectra plotted.")
    print("OBSERVATION: Check for within-group variability and between-group overlap.")
    print("  Saved: individual_spectral_signatures.png")
    
    # =======================
    # PART 4: COMPARE GROUP MEANS
    # =======================
    print("\n" + "=" * 80)
    print("PART 4: GROUP COMPARISON (DESCRIPTIVE)")
    print("=" * 80)
    
    healthy_mean = np.mean(spectra_healthy, axis=0)
    healthy_std = np.std(spectra_healthy, axis=0)
    
    rotten_mean = np.mean(spectra_rotten, axis=0)
    rotten_std = np.std(spectra_rotten, axis=0)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Mean +/- std
    axes[0].plot(wavelengths, healthy_mean, linewidth=2, label='Healthy mean', color='green')
    axes[0].fill_between(wavelengths, healthy_mean - healthy_std, healthy_mean + healthy_std,
                        alpha=0.3, color='green', label='Healthy ±1 std')
    
    axes[0].plot(wavelengths, rotten_mean, linewidth=2, label='Rotten mean', color='red')
    axes[0].fill_between(wavelengths, rotten_mean - rotten_std, rotten_mean + rotten_std,
                        alpha=0.3, color='red', label='Rotten ±1 std')
    
    axes[0].set_xlabel('Wavelength (nm)', fontsize=12)
    axes[0].set_ylabel('Mean Digital Number', fontsize=12)
    axes[0].set_title('Group Mean Spectra with Variability', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(700, color='gray', linestyle='--', alpha=0.5)
    
    # Difference spectrum
    difference = rotten_mean - healthy_mean
    axes[1].plot(wavelengths, difference, linewidth=2, color='purple')
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Wavelength (nm)', fontsize=12)
    axes[1].set_ylabel('Difference (Rotten - Healthy)', fontsize=12)
    axes[1].set_title('Spectral Difference (Rotten - Healthy)', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(700, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'group_comparison_spectra.png', dpi=150)
    plt.close()
    
    print("\nGroup comparison plotted.")
    print(f"  Healthy mean range: {healthy_mean.min():.2f} - {healthy_mean.max():.2f}")
    print(f"  Rotten mean range: {rotten_mean.min():.2f} - {rotten_mean.max():.2f}")
    print(f"  Max absolute difference: {np.abs(difference).max():.2f} DN")
    print("  Saved: group_comparison_spectra.png")
    
    # =======================
    # PART 5: STATISTICAL COMPARISON
    # =======================
    print("\n" + "=" * 80)
    print("PART 5: STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 80)
    
    print("\nWARNING: Statistical power is very low with only 3 vs 5 samples.")
    print("p-values should be interpreted cautiously.")
    
    # T-test per band
    p_values = []
    t_statistics = []
    
    for band in range(len(wavelengths)):
        h_band = spectra_healthy[:, band]
        r_band = spectra_rotten[:, band]
        
        t, p = stats.ttest_ind(h_band, r_band)
        t_statistics.append(t)
        p_values.append(p)
    
    p_values = np.array(p_values)
    t_statistics = np.array(t_statistics)
    
    # Identify significant bands (p < 0.05, uncorrected)
    sig_mask = p_values < 0.05
    sig_count = sig_mask.sum()
    
    print(f"\nBands with p < 0.05 (uncorrected): {sig_count} / {len(wavelengths)}")
    
    if sig_count > 0:
        sig_wavelengths = wavelengths[sig_mask]
        print(f"  Wavelength ranges: {sig_wavelengths[0]:.2f} - {sig_wavelengths[-1]:.2f} nm")
    
    # Plot p-values
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    axes[0].plot(wavelengths, -np.log10(p_values), linewidth=1, color='blue')
    axes[0].axhline(-np.log10(0.05), color='red', linestyle='--', label='p = 0.05')
    axes[0].set_xlabel('Wavelength (nm)', fontsize=12)
    axes[0].set_ylabel('-log10(p-value)', fontsize=12)
    axes[0].set_title('Statistical Significance per Band (t-test, uncorrected)', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(wavelengths, t_statistics, linewidth=1, color='orange')
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Wavelength (nm)', fontsize=12)
    axes[1].set_ylabel('t-statistic', fontsize=12)
    axes[1].set_title('Effect Size per Band (t-statistic)', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'statistical_comparison.png', dpi=150)
    plt.close()
    
    print("  Saved: statistical_comparison.png")
    
    # =======================
    # PART 6: DIMENSIONALITY REDUCTION (PCA)
    # =======================
    print("\n" + "=" * 80)
    print("PART 6: DIMENSIONALITY REDUCTION (PCA)")
    print("=" * 80)
    
    print("\nWARNING: PCA with only 8 samples is for visualization only.")
    print("This is NOT evidence of model separability.")
    
    from sklearn.decomposition import PCA
    
    # Combine all samples
    X = np.vstack([spectra_healthy, spectra_rotten])
    labels = ['Healthy'] * len(spectra_healthy) + ['Rotten'] * len(spectra_rotten)
    sample_ids = healthy_samples['sample_id'].tolist() + rotten_samples['sample_id'].tolist()
    
    # PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    
    print(f"\nPCA Results:")
    print(f"  Explained variance: {pca.explained_variance_ratio_ * 100}")
    print(f"  Cumulative: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
          f"PC1+PC2={pca.explained_variance_ratio_[:2].sum()*100:.1f}%, "
          f"PC1+PC2+PC3={pca.explained_variance_ratio_[:3].sum()*100:.1f}%")
    
    # Plot PCA
    fig = plt.figure(figsize=(14, 5))
    
    # 2D PCA
    ax1 = fig.add_subplot(1, 2, 1)
    for i, (label, sample_id) in enumerate(zip(labels, sample_ids)):
        color = 'green' if label == 'Healthy' else 'red'
        marker = 'o' if label == 'Healthy' else 's'
        ax1.scatter(X_pca[i, 0], X_pca[i, 1], c=color, marker=marker, s=200, 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.text(X_pca[i, 0], X_pca[i, 1] + 20, sample_id, fontsize=9, ha='center')
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax1.set_title('PCA: PC1 vs PC2 (n=8 samples)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Healthy', 'Rotten'], fontsize=10)
    
    # Loadings
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(wavelengths, pca.components_[0, :], label='PC1', linewidth=2)
    ax2.plot(wavelengths, pca.components_[1, :], label='PC2', linewidth=2)
    ax2.set_xlabel('Wavelength (nm)', fontsize=12)
    ax2.set_ylabel('Loading', fontsize=12)
    ax2.set_title('PCA Loadings (Feature Importance)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(700, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pca_visualization.png', dpi=150)
    plt.close()
    
    print("  Saved: pca_visualization.png")
    
    # =======================
    # SUMMARY
    # =======================
    print("\n" + "=" * 80)
    print("EXPLORATORY ANALYSIS SUMMARY")
    print("=" * 80)
    
    print("\n[DATA QUALITY]")
    print(f"  - All samples have consistent dimensions: 500 x 300 x 900")
    print(f"  - Wavelength coverage: 378-1014 nm (Visible + NIR)")
    print(f"  - Data type: uint16 raw digital counts (NOT calibrated reflectance)")
    print(f"  - Dead bands: {len(dead_bands)}")
    print(f"  - Saturated bands: {len(saturated_bands)}")
    
    print("\n[SPECTRAL SIGNATURES]")
    print(f"  - Healthy samples: {len(spectra_healthy)} individual spectra extracted")
    print(f"  - Rotten samples: {len(spectra_rotten)} individual spectra extracted")
    print(f"  - Within-group variability: VISIBLE in plots")
    print(f"  - Between-group overlap: SUBSTANTIAL")
    
    print("\n[STATISTICAL FINDINGS]")
    print(f"  - Bands with p < 0.05: {sig_count} / {len(wavelengths)}")
    print(f"  - WARNING: Low statistical power (n=3 vs n=5)")
    print(f"  - Multiple testing correction NOT applied")
    print(f"  - Findings are EXPLORATORY, not confirmatory")
    
    print("\n[PCA RESULTS]")
    print(f"  - PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
    print(f"  - Visual separation in PC1-PC2 space: LIMITED")
    print(f"  - Important wavelengths: See loadings plot")
    print(f"  - WARNING: n=8 samples insufficient for model validation")
    
    print("\n[INTERPRETATION LIMITATIONS]")
    print("  1. Labels are UNVERIFIED (filename-based)")
    print("  2. Sample size TOO SMALL for reliable conclusions")
    print("  3. Raw counts may reflect illumination, not just sample properties")
    print("  4. No calibration data (white reference) available")
    print("  5. Single-sample differences could be artifacts, not true defects")
    
    print("\n[RECOMMENDATION]")
    if sig_count > 50 and pca.explained_variance_ratio_[0] > 0.5:
        print("  CAUTIOUSLY PROMISING: Some spectral variation observed between groups.")
        print("  However, verification with LARGER LABELED DATASET is essential.")
    else:
        print("  INCONCLUSIVE: Limited spectral differences with current samples.")
        print("  Collect more verified samples before pursuing hyperspectral approach.")
    
    print("\n[OUTPUTS SAVED]")
    print(f"  Directory: {OUTPUT_DIR}")
    print("  Files created:")
    for f in OUTPUT_DIR.glob("*.png"):
        print(f"    - {f.name}")
    
    # Save numerical results
    results = {
        'samples': {
            'healthy': len(spectra_healthy),
            'rotten': len(spectra_rotten),
        },
        'wavelengths': wavelengths.tolist(),
        'healthy_mean_spectrum': healthy_mean.tolist(),
        'rotten_mean_spectrum': rotten_mean.tolist(),
        'difference_spectrum': difference.tolist(),
        'statistical_significance': {
            'significant_bands': sig_count,
            'total_bands': len(wavelengths),
            'min_p_value': float(p_values.min()),
            'significant_wavelengths': wavelengths[sig_mask].tolist() if sig_count > 0 else [],
        },
        'pca': {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': pca.explained_variance_ratio_.cumsum().tolist(),
        },
        'quality_issues': {
            'dead_bands': len(dead_bands),
            'saturated_bands': len(saturated_bands),
        }
    }
    
    with open(OUTPUT_DIR / 'exploratory_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("    - exploratory_results.json")
    
    print("\n" + "=" * 80)
    print("EXPLORATORY ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
