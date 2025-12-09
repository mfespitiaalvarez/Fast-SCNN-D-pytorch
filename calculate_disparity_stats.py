"""
Calculate statistics on RAW 16-bit disparity data (before normalization or hole filling).
This script finds the exact mean value to use for filling invalid pixels (zeros).

Key points:
- Loads raw 16-bit disparity files directly using OpenCV
- Ignores zeros (holes) when calculating statistics
- Outputs the exact integer value to use for fill=...
"""
import os
import numpy as np
import cv2
import sys

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, desc=""):
        if desc:
            print(f"{desc}...")
        return iterable


def get_disparity_paths(root_folder, split='train'):
    """
    Get all disparity file paths for the given split.
    Uses the same logic as the dataloader to find files.
    """
    def get_path_pairs(img_folder, disp_folder):
        img_paths = []
        disp_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith(".png"):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    
                    # Construct Disparity Path (same logic as dataloader)
                    dispname = filename.replace('leftImg8bit', 'disparity')
                    disppath = os.path.join(disp_folder, foldername, dispname)
                    
                    if os.path.isfile(imgpath) and os.path.isfile(disppath):
                        img_paths.append(imgpath)
                        disp_paths.append(disppath)
        print(f'Found {len(img_paths)} disparity files in {img_folder}')
        return disp_paths

    if split in ('train', 'val'):
        img_folder = os.path.join(root_folder, 'leftImg8bit/' + split)
        disp_folder = os.path.join(root_folder, 'disparity/' + split)
        return get_path_pairs(img_folder, disp_folder)
    return []


def calculate_raw_disparity_stats(disp_paths, num_samples=None):
    """
    Calculate statistics on RAW 16-bit disparity data.
    Ignores zeros (invalid pixels/holes) during calculation.
    Uses incremental batch algorithms to avoid memory issues and maximize speed.
    
    Args:
        disp_paths: List of paths to disparity files
        num_samples: Number of samples to process (None = all)
    
    Returns:
        Dictionary with statistics
    """
    # Statistics accumulators
    per_image_means = []
    per_image_stds = []
    per_image_valid_counts = []
    per_image_total_counts = []
    
    # For incremental statistics (Welford's online algorithm)
    n = 0  # Total count of valid pixels processed
    mean = 0.0  # Running mean
    M2 = 0.0  # Running sum of squared differences
    
    # For min/max (can track incrementally)
    global_min = float('inf')
    global_max = float('-inf')
    
    # For percentiles - we'll sample (or use a streaming approach)
    # For now, collect a sample for percentiles (every Nth pixel)
    sample_size = 10_000_000  # Sample 10M pixels for percentiles
    sampled_pixels = []
    sample_interval = None  # Will be calculated
    
    # Determine how many samples to process
    total_samples = len(disp_paths)
    if num_samples is None:
        num_samples = total_samples
    else:
        num_samples = min(num_samples, total_samples)
    
    print(f"\nProcessing {num_samples} raw disparity files...")
    print(f"Total available files: {total_samples}")
    
    # Process each file
    for idx, disp_path in enumerate(tqdm(disp_paths[:num_samples], desc="Loading raw disparity")):
        try:
            # Load raw 16-bit disparity (same as dataloader)
            disp_raw = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
            
            if disp_raw is None:
                print(f"Warning: Could not load {disp_path}")
                continue
            
            # Get valid pixels (non-zero)
            valid_mask = disp_raw > 0
            valid_pixels = disp_raw[valid_mask]
            
            if len(valid_pixels) == 0:
                print(f"Warning: No valid pixels in {disp_path}")
                continue
            
            # Collect per-image statistics
            img_mean = np.mean(valid_pixels)
            img_std = np.std(valid_pixels)
            per_image_means.append(img_mean)
            per_image_stds.append(img_std)
            per_image_valid_counts.append(len(valid_pixels))
            per_image_total_counts.append(disp_raw.size)
            
            # Update global min/max
            global_min = min(global_min, np.min(valid_pixels))
            global_max = max(global_max, np.max(valid_pixels))
            
            # Update running statistics using BATCH Welford's online algorithm
            # VECTORIZED VERSION - processes entire image at once (much faster!)
            batch_size = len(valid_pixels)
            batch_mean = float(img_mean)  # Use already computed mean
            batch_sum_sq_diff = float(np.sum((valid_pixels - batch_mean) ** 2))
            
            # Update global statistics using batch Welford's algorithm
            if n == 0:
                # First batch
                mean = batch_mean
                M2 = batch_sum_sq_diff
                n = batch_size
            else:
                # Merge batch statistics with global statistics
                # Formula for combining two sets of statistics
                n_new = n + batch_size
                delta = batch_mean - mean
                mean = (n * mean + batch_size * batch_mean) / n_new
                M2 = M2 + batch_sum_sq_diff + delta ** 2 * n * batch_size / n_new
                n = n_new
            
            # Sample pixels for percentile calculation (vectorized, much faster)
            if sample_interval is None:
                # Estimate total pixels and set sampling interval
                if n > 10000:  # After processing some pixels
                    estimated_total = n * (total_samples / (idx + 1))
                    sample_interval = max(1, int(estimated_total / sample_size))
            
            if sample_interval and sample_interval > 0:
                # Sample from this batch using vectorized indexing
                sample_indices = np.arange(0, batch_size, sample_interval, dtype=np.int64)
                if len(sample_indices) > 0:
                    sampled_pixels.extend(valid_pixels[sample_indices].astype(np.float32).tolist())
            
        except Exception as e:
            print(f"Error processing {disp_path}: {e}")
            continue
    
    if n == 0:
        raise ValueError("No valid disparity data found!")
    
    # Calculate global statistics
    global_mean = mean
    global_std = np.sqrt(M2 / n) if n > 1 else 0.0
    
    # Calculate percentiles from sample
    if len(sampled_pixels) > 0:
        sampled_pixels = np.array(sampled_pixels)
        percentiles = {
            'percentile_1': np.percentile(sampled_pixels, 1),
            'percentile_5': np.percentile(sampled_pixels, 5),
            'percentile_25': np.percentile(sampled_pixels, 25),
            'percentile_75': np.percentile(sampled_pixels, 75),
            'percentile_95': np.percentile(sampled_pixels, 95),
            'percentile_99': np.percentile(sampled_pixels, 99),
        }
        sample_median = np.median(sampled_pixels)
    else:
        percentiles = {
            'percentile_1': global_min,
            'percentile_5': global_min,
            'percentile_25': global_min,
            'percentile_75': global_max,
            'percentile_95': global_max,
            'percentile_99': global_max,
        }
        sample_median = global_mean
    
    total_pixels = sum(per_image_total_counts)
    total_valid_pixels = n
    
    stats = {
        'num_samples': num_samples,
        'total_valid_pixels': total_valid_pixels,
        'total_pixels': total_pixels,
        'valid_pixel_ratio': total_valid_pixels / total_pixels if total_pixels > 0 else 0,
        
        # Global statistics (across all valid pixels)
        'global_mean': global_mean,
        'global_std': global_std,
        'global_min': global_min,
        'global_max': global_max,
        'global_median': sample_median,  # Approximate from sample
        
        # The exact integer value to use for filling
        'fill_value_int': int(np.round(global_mean)),
        'fill_value_float': float(global_mean),
        
        # Percentiles (from sample)
        **percentiles,
        
        # Per-image statistics
        'per_image_mean': {
            'mean': np.mean(per_image_means),
            'std': np.std(per_image_means),
            'min': np.min(per_image_means),
            'max': np.max(per_image_means)
        },
        'per_image_std': {
            'mean': np.mean(per_image_stds),
            'std': np.std(per_image_stds),
            'min': np.min(per_image_stds),
            'max': np.max(per_image_stds)
        },
        'valid_pixel_ratio_per_image': {
            'mean': np.mean([v/t for v, t in zip(per_image_valid_counts, per_image_total_counts)]),
            'min': np.min([v/t for v, t in zip(per_image_valid_counts, per_image_total_counts)]),
            'max': np.max([v/t for v, t in zip(per_image_valid_counts, per_image_total_counts)])
        },
        'percentile_sample_size': len(sampled_pixels) if len(sampled_pixels) > 0 else 0
    }
    
    return stats



def print_statistics(stats):
    """Print statistics in a readable format."""
    print("\n" + "="*70)
    print("RAW 16-BIT DISPARITY STATISTICS (Valid Pixels Only)")
    print("="*70)
    print(f"\nDataset Info:")
    print(f"  Number of files processed: {stats['num_samples']}")
    print(f"  Total pixels: {stats['total_pixels']:,}")
    print(f"  Valid pixels (non-zero): {stats['total_valid_pixels']:,}")
    print(f"  Invalid pixels (zeros/holes): {stats['total_pixels'] - stats['total_valid_pixels']:,}")
    print(f"  Valid pixel ratio: {stats['valid_pixel_ratio']:.4%}")
    
    print(f"\n{'Global Statistics (Valid Pixels Only)':^70}")
    print("-"*70)
    print(f"  Mean:        {stats['global_mean']:>12.6f}")
    print(f"  Std:         {stats['global_std']:>12.6f}")
    print(f"  Median:      {stats['global_median']:>12.6f}")
    print(f"  Min:         {stats['global_min']:>12.0f}")
    print(f"  Max:         {stats['global_max']:>12.0f}")
    
    print(f"\n{'RECOMMENDED FILL VALUE':^70}")
    print("-"*70)
    print(f"  Integer (use this):  {stats['fill_value_int']:>12d}")
    print(f"  Float (exact):       {stats['fill_value_float']:>12.6f}")
    print(f"\n  Update your dataloader with:")
    print(f"      disp_raw[~mask_valid] = {stats['fill_value_int']}  # Fill holes with mean")
    
    print(f"\n{'Percentiles':^70}")
    print("-"*70)
    print(f"  1st:         {stats['percentile_1']:>12.0f}")
    print(f"  5th:         {stats['percentile_5']:>12.0f}")
    print(f"  25th:        {stats['percentile_25']:>12.0f}")
    print(f"  75th:        {stats['percentile_75']:>12.0f}")
    print(f"  95th:        {stats['percentile_95']:>12.0f}")
    print(f"  99th:        {stats['percentile_99']:>12.0f}")
    
    print(f"\n{'Per-Image Mean Statistics':^70}")
    print("-"*70)
    print(f"  Mean of means: {stats['per_image_mean']['mean']:>12.6f}")
    print(f"  Std of means:  {stats['per_image_mean']['std']:>12.6f}")
    print(f"  Min mean:      {stats['per_image_mean']['min']:>12.0f}")
    print(f"  Max mean:      {stats['per_image_mean']['max']:>12.0f}")
    
    print(f"\n{'Per-Image Valid Pixel Ratio':^70}")
    print("-"*70)
    print(f"  Mean ratio:    {stats['valid_pixel_ratio_per_image']['mean']:>12.4%}")
    print(f"  Min ratio:     {stats['valid_pixel_ratio_per_image']['min']:>12.4%}")
    print(f"  Max ratio:     {stats['valid_pixel_ratio_per_image']['max']:>12.4%}")
    
    print("\n" + "="*70)
    print("\nNote: Statistics computed on RAW 16-bit pixel values")
    print("      Zeros (invalid pixels/holes) are excluded from calculations")
    print("="*70 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Calculate raw disparity statistics to find exact fill value',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Process all training samples
  python3 calculate_raw_stats.py
  
  # Process a subset for quick testing
  python3 calculate_raw_stats.py --num-samples 100
  
  # Custom dataset path
  python3 calculate_raw_stats.py --root ./datasets/citys --split train
        """
    )
    parser.add_argument('--root', type=str, default='./datasets/citys',
                        help='Root directory of Cityscapes dataset (default: ./datasets/citys)')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val'],
                        help='Dataset split (default: train)')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to process (None = all, default: None)')
    
    args = parser.parse_args()
    
    # Get disparity file paths
    print(f"Scanning for disparity files in {args.root}...")
    disp_paths = get_disparity_paths(args.root, args.split)
    
    if len(disp_paths) == 0:
        print(f"ERROR: No disparity files found in {args.root}")
        print("Please check that:")
        print(f"  1. The dataset root is correct: {args.root}")
        print(f"  2. Disparity files exist in: {os.path.join(args.root, 'disparity', args.split)}")
        return
    
    # Calculate statistics
    stats = calculate_raw_disparity_stats(disp_paths, num_samples=args.num_samples)
    
    # Print results
    print_statistics(stats)


if __name__ == '__main__':
    main()