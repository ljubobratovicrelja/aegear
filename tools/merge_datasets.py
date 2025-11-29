#!/usr/bin/env python3
"""
Convert cached tracking dataset to webdataset format.

This script merges all subdatasets (and their train/val splits) into a unified
webdataset stored as multiple tar files for efficient S3 storage and access.
"""

import os
import json
import tarfile
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO


def collect_all_samples(root_dir: str) -> List[Dict]:
    """
    Traverse the dataset structure and collect all samples from all subdatasets.
    
    Args:
        root_dir: Root directory containing subdatasets (e.g., D:/PROJECTS/AEGEAR/DATA/TRAINING/CACHE/TRACKING)
    
    Returns:
        List of sample dictionaries with added 'subset' and 'split' fields
    """
    all_samples = []
    root_path = Path(root_dir)
    
    # Get list of subdataset folders
    subset_dirs = [d for d in root_path.iterdir() if d.is_dir()]
    
    # Iterate through each subdataset folder with progress bar
    with tqdm(subset_dirs, desc="Processing subdatasets", unit="subdataset") as pbar:
        for subset_dir in pbar:
            subset_name = subset_dir.name
            pbar.set_description(f"Processing {subset_name}")
            
            # Process both train and val splits
            for split in ['train', 'val']:
                split_dir = subset_dir / split
                metadata_path = split_dir / 'metadata.json'
                
                if not metadata_path.exists():
                    tqdm.write(f"  ⚠ No metadata.json in {split_dir}, skipping...")
                    continue
                
                # Load metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                samples = metadata.get('samples', [])
                tqdm.write(f"  ✓ {subset_name}/{split}: {len(samples)} samples")
                
                # Add each sample with additional context
                for sample in samples:
                    sample_copy = sample.copy()
                    sample_copy['subset'] = subset_name
                    sample_copy['split'] = split
                    sample_copy['base_dir'] = str(split_dir)
                    all_samples.append(sample_copy)
    
    print(f"\n✓ Total samples collected: {len(all_samples)}")
    return all_samples


def create_single_shard(shard_info: Dict) -> Dict:
    """
    Create a single tar shard from a list of samples.
    
    Args:
        shard_info: Dictionary containing:
            - shard_idx: Index of this shard
            - shard_path: Path where to save the tar file
            - samples: List of samples to include in this shard
            - prefix: Filename prefix
    
    Returns:
        Dictionary with shard statistics
    """
    shard_idx = shard_info['shard_idx']
    shard_path = shard_info['shard_path']
    samples = shard_info['samples']
    prefix = shard_info['prefix']
    start_idx = shard_info['start_idx']
    
    shard_name = f"{prefix}-{shard_idx:06d}.tar"
    
    with tarfile.open(shard_path, 'w') as tar:
        for local_idx, sample in enumerate(samples):
            # Global sample index for unique keys
            sample_idx = start_idx + local_idx
            key = f"{sample_idx:08d}"
            
            # Read template image
            template_full_path = os.path.join(sample['base_dir'], sample['template_path'])
            if os.path.exists(template_full_path):
                template_info = tarfile.TarInfo(name=f"{key}.template.jpg")
                with open(template_full_path, 'rb') as f:
                    template_data = f.read()
                template_info.size = len(template_data)
                tar.addfile(template_info, fileobj=BytesIO(template_data))
            
            # Read search image
            search_full_path = os.path.join(sample['base_dir'], sample['search_path'])
            if os.path.exists(search_full_path):
                search_info = tarfile.TarInfo(name=f"{key}.search.jpg")
                with open(search_full_path, 'rb') as f:
                    search_data = f.read()
                search_info.size = len(search_data)
                tar.addfile(search_info, fileobj=BytesIO(search_data))
            
            # Create metadata JSON for this sample
            sample_metadata = {
                'frame_id': sample.get('frame_id', sample_idx),
                'centroid': sample['centroid'],
                'background': sample['background'],
                'subset': sample['subset'],
                'original_split': sample['split']
            }
            
            metadata_json = json.dumps(sample_metadata).encode('utf-8')
            metadata_info = tarfile.TarInfo(name=f"{key}.json")
            metadata_info.size = len(metadata_json)
            tar.addfile(metadata_info, fileobj=BytesIO(metadata_json))
    
    return {
        'shard_idx': shard_idx,
        'shard_name': shard_name,
        'num_samples': len(samples)
    }


def create_webdataset_shards(samples: List[Dict], output_dir: str, 
                             shard_size: int = 5000, prefix: str = "tracking",
                             shuffle: bool = True, seed: int = 42,
                             num_threads: int = 4):
    """
    Create webdataset tar shards from the collected samples.
    
    Args:
        samples: List of sample dictionaries
        output_dir: Directory to write tar files
        shard_size: Maximum number of samples per shard
        prefix: Prefix for shard filenames
        shuffle: Whether to shuffle samples before packing (default: True)
        seed: Random seed for shuffling (default: 42)
        num_threads: Number of parallel threads for shard creation (default: 4)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Shuffle samples to mix data from different videos/subdatasets
    if shuffle:
        print(f"\nShuffling {len(samples)} samples with seed {seed}...")
        import random
        random.seed(seed)
        random.shuffle(samples)
        print("Shuffling complete - each shard will contain mixed data from all subdatasets")
    
    num_shards = (len(samples) + shard_size - 1) // shard_size
    print(f"\nCreating {num_shards} shards with ~{shard_size} samples each")
    print(f"Using {num_threads} threads for parallel processing")
    
    # Prepare shard information for parallel processing
    shard_tasks = []
    sample_idx = 0
    
    for shard_idx in range(num_shards):
        # Calculate samples for this shard
        start_idx = sample_idx
        end_idx = min(start_idx + shard_size, len(samples))
        shard_samples = samples[start_idx:end_idx]
        
        shard_path = output_path / f"{prefix}-{shard_idx:06d}.tar"
        
        shard_tasks.append({
            'shard_idx': shard_idx,
            'shard_path': str(shard_path),
            'samples': shard_samples,
            'prefix': prefix,
            'start_idx': start_idx
        })
        
        sample_idx = end_idx
    
    # Process shards in parallel with progress bar
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        futures = {executor.submit(create_single_shard, task): task for task in shard_tasks}
        
        # Progress bar for completed shards
        with tqdm(total=num_shards, desc="Creating shards", unit="shard") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    pbar.set_postfix_str(f"Last: {result['shard_name']}")
                    pbar.update(1)
                except Exception as e:
                    task = futures[future]
                    tqdm.write(f"  ✗ Error creating shard {task['shard_idx']}: {e}")
    
    print(f"\n✓ All {num_shards} shards created successfully in {output_dir}")
    
    # Create a manifest file
    manifest = {
        'num_shards': num_shards,
        'total_samples': len(samples),
        'shard_size': shard_size,
        'shard_pattern': f"{prefix}-{{shard_idx:06d}}.tar",
        'format': 'webdataset',
        'shuffled': shuffle,
        'shuffle_seed': seed if shuffle else None
    }
    
    manifest_path = output_path / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Manifest written to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert cached tracking dataset to webdataset format'
    )
    parser.add_argument(
        'input_dir',
        help='Root directory containing subdatasets (e.g., D:/PROJECTS/AEGEAR/DATA/TRAINING/CACHE/TRACKING)'
    )
    parser.add_argument(
        'output_dir',
        help='Output directory for webdataset tar files'
    )
    parser.add_argument(
        '--shard-size',
        type=int,
        default=5000,
        help='Number of samples per shard (default: 5000)'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='tracking',
        help='Prefix for shard filenames (default: tracking)'
    )
    parser.add_argument(
        '--shuffle',
        action='store_true',
        default=True,
        help='Shuffle samples before packing into shards (default: True)'
    )
    parser.add_argument(
        '--no-shuffle',
        dest='shuffle',
        action='store_false',
        help='Do not shuffle samples (keeps original order)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for shuffling (default: 42)'
    )
    parser.add_argument(
        '--num-threads',
        type=int,
        default=4,
        help='Number of parallel threads for shard creation (default: 4)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Converting Dataset to WebDataset Format")
    print("=" * 70)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Shard size: {args.shard_size}")
    print(f"Prefix: {args.prefix}")
    print(f"Shuffle samples: {args.shuffle}")
    if args.shuffle:
        print(f"Random seed: {args.seed}")
    print(f"Parallel threads: {args.num_threads}")
    print("=" * 70)
    
    # Step 1: Collect all samples
    samples = collect_all_samples(args.input_dir)
    
    if len(samples) == 0:
        print("Error: No samples found!")
        return
    
    # Step 2: Create webdataset shards
    create_webdataset_shards(
        samples=samples,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        prefix=args.prefix,
        shuffle=args.shuffle,
        seed=args.seed,
        num_threads=args.num_threads
    )
    
    num_shards = (len(samples) + args.shard_size - 1) // args.shard_size
    
    print("\n" + "=" * 70)
    print("Conversion Complete!")
    print("=" * 70)
    print(f"\n✓ Created {num_shards} tar files in {args.output_dir}")
    print(f"✓ Total samples: {len(samples)}")
    print(f"✓ Manifest: {os.path.join(args.output_dir, 'manifest.json')}")
    print("\nNext steps:")
    print(f"  1. Upload tar files to S3: aws s3 sync {args.output_dir} s3://your-bucket/")
    print("  2. Use webdataset_tracking.py to load data during training")
    print("=" * 70)


if __name__ == '__main__':
    main()