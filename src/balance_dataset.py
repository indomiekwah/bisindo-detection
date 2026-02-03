"""
BISINDO Dataset Balancer
=========================
Script untuk menyeimbangkan dataset dengan:
1. Undersampling kelas yang terlalu banyak
2. Augmentasi kelas yang kurang

Augmentasi yang dilakukan pada landmarks:
- Noise injection
- Time shifting
- Scale variation
- Speed variation (interpolasi)

Author: BISINDO Project
"""

import numpy as np
from pathlib import Path
import json
import shutil
import random
import argparse
from tqdm import tqdm


def add_noise(landmarks, noise_factor=0.02):
    """Add Gaussian noise to landmarks."""
    noise = np.random.normal(0, noise_factor, landmarks.shape)
    return landmarks + noise


def time_shift(landmarks, shift_max=3):
    """Shift sequence in time."""
    shift = np.random.randint(-shift_max, shift_max + 1)
    if shift == 0:
        return landmarks.copy()
    
    shifted = np.zeros_like(landmarks)
    if shift > 0:
        shifted[shift:] = landmarks[:-shift]
        shifted[:shift] = landmarks[0]  # Repeat first frame
    else:
        shifted[:shift] = landmarks[-shift:]
        shifted[shift:] = landmarks[-1]  # Repeat last frame
    
    return shifted


def scale_landmarks(landmarks, scale_range=(0.9, 1.1)):
    """Scale landmarks randomly."""
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return landmarks * scale


def speed_variation(landmarks, speed_factor=None):
    """Change speed by interpolating frames."""
    if speed_factor is None:
        speed_factor = np.random.uniform(0.8, 1.2)
    
    num_frames = landmarks.shape[0]
    new_num_frames = int(num_frames / speed_factor)
    
    if new_num_frames < 10:
        new_num_frames = 10
    if new_num_frames > num_frames * 2:
        new_num_frames = num_frames * 2
    
    old_indices = np.linspace(0, num_frames - 1, num_frames)
    new_indices = np.linspace(0, num_frames - 1, new_num_frames)
    
    new_landmarks = np.zeros((new_num_frames, landmarks.shape[1]))
    for i in range(landmarks.shape[1]):
        new_landmarks[:, i] = np.interp(new_indices, old_indices, landmarks[:, i])
    
    return new_landmarks


def augment_sample(landmarks, augmentation_type='random'):
    """
    Apply augmentation to a single sample.
    
    Args:
        landmarks: numpy array (num_frames, num_features)
        augmentation_type: 'noise', 'shift', 'scale', 'speed', or 'random'
    
    Returns:
        Augmented landmarks
    """
    if augmentation_type == 'random':
        augmentation_type = random.choice(['noise', 'shift', 'scale', 'combined'])
    
    if augmentation_type == 'noise':
        return add_noise(landmarks, noise_factor=np.random.uniform(0.01, 0.03))
    elif augmentation_type == 'shift':
        return time_shift(landmarks, shift_max=np.random.randint(2, 5))
    elif augmentation_type == 'scale':
        return scale_landmarks(landmarks, scale_range=(0.85, 1.15))
    elif augmentation_type == 'speed':
        return speed_variation(landmarks)
    elif augmentation_type == 'combined':
        # Apply multiple augmentations
        result = landmarks.copy()
        if random.random() > 0.5:
            result = add_noise(result, noise_factor=0.015)
        if random.random() > 0.5:
            result = time_shift(result, shift_max=2)
        if random.random() > 0.5:
            result = scale_landmarks(result, scale_range=(0.92, 1.08))
        return result
    
    return landmarks


def analyze_dataset(landmarks_dir):
    """Analyze current dataset distribution."""
    landmarks_path = Path(landmarks_dir)
    
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    class_counts = {}
    total_samples = 0
    
    for class_folder in sorted(landmarks_path.iterdir()):
        if class_folder.is_dir():
            npy_files = list(class_folder.glob('*.npy'))
            count = len(npy_files)
            class_counts[class_folder.name] = count
            total_samples += count
            print(f"  {class_folder.name}: {count} samples")
    
    print(f"\n  Total: {total_samples} samples")
    print(f"  Classes: {len(class_counts)}")
    
    if class_counts:
        avg = total_samples / len(class_counts)
        print(f"  Average per class: {avg:.1f}")
        print(f"  Min: {min(class_counts.values())}")
        print(f"  Max: {max(class_counts.values())}")
    
    print("="*60)
    
    return class_counts


def balance_dataset(input_dir, output_dir, target_counts=None, default_target=20, 
                    undersample_classes=None, random_seed=42):
    """
    Balance dataset with undersampling and augmentation.
    
    Args:
        input_dir: Input landmarks directory
        output_dir: Output directory for balanced dataset
        target_counts: Dict of {class_name: target_count}, or None for default
        default_target: Default target count for classes not in target_counts
        undersample_classes: Dict of {class_name: target_count} for undersampling
        random_seed: Random seed for reproducibility
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("BALANCING DATASET")
    print("="*60)
    
    # Analyze current dataset
    current_counts = analyze_dataset(input_dir)
    
    # Determine target counts
    if target_counts is None:
        target_counts = {}
    
    if undersample_classes is None:
        undersample_classes = {}
    
    # Process each class
    stats = {
        'original': {},
        'final': {},
        'augmented': {},
        'undersampled': {},
        'skipped': {}
    }
    
    for class_folder in sorted(input_path.iterdir()):
        if not class_folder.is_dir():
            continue
        
        class_name = class_folder.name
        npy_files = list(class_folder.glob('*.npy'))
        current_count = len(npy_files)
        
        # Skip empty folders
        if current_count == 0:
            print(f"\n⚠️  {class_name}: SKIPPED (no .npy files found)")
            stats['skipped'][class_name] = "No .npy files"
            continue
        
        # Determine target for this class
        if class_name in undersample_classes:
            target = undersample_classes[class_name]
        elif class_name in target_counts:
            target = target_counts[class_name]
        else:
            target = default_target
        
        # Validate target
        if target <= 0:
            print(f"\n⚠️  {class_name}: SKIPPED (target is {target})")
            stats['skipped'][class_name] = f"Invalid target: {target}"
            continue
        
        stats['original'][class_name] = current_count
        
        # Create output folder
        class_output = output_path / class_name
        class_output.mkdir(exist_ok=True)
        
        print(f"\n{class_name}: {current_count} → {target}")
        
        if current_count >= target:
            # UNDERSAMPLE: randomly select 'target' samples
            selected_files = random.sample(npy_files, target)
            stats['undersampled'][class_name] = current_count - target
            
            for f in tqdm(selected_files, desc=f"  Copying {class_name}"):
                shutil.copy(f, class_output / f.name)
            
            stats['final'][class_name] = target
            stats['augmented'][class_name] = 0
            
        else:
            # AUGMENT: copy all originals + create augmented samples
            # Copy originals
            for f in npy_files:
                shutil.copy(f, class_output / f.name)
            
            # Calculate how many augmented samples needed
            num_to_augment = target - current_count
            stats['augmented'][class_name] = num_to_augment
            stats['undersampled'][class_name] = 0
            
            # Create augmented samples
            if num_to_augment > 0 and len(npy_files) > 0:
                print(f"  Augmenting +{num_to_augment} samples...")
                
                for i in tqdm(range(num_to_augment), desc=f"  Augmenting {class_name}"):
                    # Pick a random original sample
                    source_file = random.choice(npy_files)
                    landmarks = np.load(source_file)
                    
                    # Apply augmentation
                    augmented = augment_sample(landmarks, augmentation_type='random')
                    
                    # Save with new name
                    aug_name = f"{source_file.stem}_aug{i+1}.npy"
                    np.save(class_output / aug_name, augmented)
            
            stats['final'][class_name] = target
    
    # Save metadata
    metadata = {
        'original_counts': stats['original'],
        'final_counts': stats['final'],
        'augmented_counts': stats['augmented'],
        'undersampled_counts': stats['undersampled'],
        'skipped_classes': stats['skipped'],
        'random_seed': random_seed
    }
    
    with open(output_path / 'balance_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("BALANCING COMPLETE!")
    print("="*60)
    
    total_original = sum(stats['original'].values())
    total_final = sum(stats['final'].values())
    total_augmented = sum(stats['augmented'].values())
    total_undersampled = sum(stats['undersampled'].values())
    
    print(f"\nOriginal total: {total_original}")
    print(f"Final total: {total_final}")
    print(f"  - Augmented: +{total_augmented}")
    print(f"  - Undersampled: -{total_undersampled}")
    
    if stats['skipped']:
        print(f"\n⚠️  Skipped classes: {len(stats['skipped'])}")
        for class_name, reason in stats['skipped'].items():
            print(f"    - {class_name}: {reason}")
    
    print(f"\nOutput: {output_path}")
    
    print("\nFinal distribution:")
    for class_name, count in sorted(stats['final'].items()):
        orig = stats['original'][class_name]
        aug = stats['augmented'][class_name]
        under = stats['undersampled'][class_name]
        
        change = ""
        if aug > 0:
            change = f"(+{aug} aug)"
        elif under > 0:
            change = f"(-{under} under)"
        
        print(f"  {class_name}: {count} {change}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Balance BISINDO dataset')
    
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input landmarks directory')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory for balanced dataset')
    parser.add_argument('--target', '-t', type=int, default=20,
                        help='Default target count per class (default: 20)')
    parser.add_argument('--neutral-target', '-n', type=int, default=25,
                        help='Target count for Neutral class (default: 25)')
    parser.add_argument('--analyze-only', '-a', action='store_true',
                        help='Only analyze dataset, do not balance')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        analyze_dataset(args.input)
    else:
        # Check if output is provided
        if args.output is None:
            print("Error: --output/-o is required when not using --analyze-only")
            parser.print_help()
            return
        
        # Define undersampling for Neutral
        undersample = {'Neutral': args.neutral_target}
        
        balance_dataset(
            input_dir=args.input,
            output_dir=args.output,
            default_target=args.target,
            undersample_classes=undersample,
            random_seed=args.seed
        )


if __name__ == "__main__":
    main()