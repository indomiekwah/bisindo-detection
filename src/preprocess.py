import os
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from tqdm import tqdm


def normalize_landmarks(landmarks, method='minmax'):
    """
    Normalisasi landmarks.
    
    Args:
        landmarks: np.array (num_frames, num_features)
        method: 'minmax' atau 'zscore'
        
    Returns:
        np.array: Normalized landmarks
    """
    if method == 'minmax':
        # Min-max normalization ke range [0, 1]
        min_val = landmarks.min()
        max_val = landmarks.max()
        if max_val - min_val > 0:
            return (landmarks - min_val) / (max_val - min_val)
        return landmarks
    
    elif method == 'zscore':
        # Z-score normalization
        mean = landmarks.mean()
        std = landmarks.std()
        if std > 0:
            return (landmarks - mean) / std
        return landmarks - mean
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def pad_or_truncate(sequence, target_length, pad_value=0):
    """
    Pad atau truncate sequence ke panjang tertentu.
    
    Args:
        sequence: np.array (num_frames, num_features)
        target_length: Panjang target
        pad_value: Nilai untuk padding
        
    Returns:
        np.array: (target_length, num_features)
    """
    current_length = sequence.shape[0]
    num_features = sequence.shape[1]
    
    if current_length == target_length:
        return sequence
    
    elif current_length > target_length:
        # Truncate: ambil bagian tengah
        start = (current_length - target_length) // 2
        return sequence[start:start + target_length]
    
    else:
        # Pad: tambah zeros di akhir
        padded = np.full((target_length, num_features), pad_value, dtype=sequence.dtype)
        padded[:current_length] = sequence
        return padded


def interpolate_sequence(sequence, target_length):
    """
    Interpolasi sequence ke panjang tertentu (alternatif dari padding).
    Lebih baik untuk menjaga temporal information.
    
    Args:
        sequence: np.array (num_frames, num_features)
        target_length: Panjang target
        
    Returns:
        np.array: (target_length, num_features)
    """
    current_length = sequence.shape[0]
    
    if current_length == target_length:
        return sequence
    
    # Interpolasi linear
    indices = np.linspace(0, current_length - 1, target_length)
    
    interpolated = np.zeros((target_length, sequence.shape[1]))
    for i in range(sequence.shape[1]):
        interpolated[:, i] = np.interp(indices, np.arange(current_length), sequence[:, i])
    
    return interpolated


def load_landmarks_dataset(landmarks_dir, sequence_length=60, 
                           normalization='minmax', 
                           sequence_method='interpolate'):
    """
    Load seluruh dataset landmarks dan preprocessing.
    
    Args:
        landmarks_dir: Directory berisi folder per kelas
        sequence_length: Target panjang sequence
        normalization: Metode normalisasi ('minmax', 'zscore', atau None)
        sequence_method: 'pad' atau 'interpolate'
        
    Returns:
        X: np.array (num_samples, sequence_length, num_features)
        y: np.array (num_samples,) - encoded labels
        label_encoder: LabelEncoder object
        metadata: dict dengan informasi dataset
    """
    landmarks_path = Path(landmarks_dir)
    
    # Load metadata jika ada
    metadata_file = landmarks_path / 'metadata.json'
    if metadata_file.exists():
        with open(metadata_file) as f:
            orig_metadata = json.load(f)
        classes = orig_metadata['classes']
    else:
        # Discover classes from folders
        classes = sorted([f.name for f in landmarks_path.iterdir() if f.is_dir()])
    
    print(f"Found {len(classes)} classes: {classes}")
    
    X = []
    y = []
    sample_info = []
    
    for class_name in tqdm(classes, desc="Loading classes"):
        class_folder = landmarks_path / class_name
        
        if not class_folder.exists():
            print(f"Warning: Class folder not found: {class_folder}")
            continue
        
        # Load all .npy files in this class
        npy_files = list(class_folder.glob('*.npy'))
        
        for npy_file in npy_files:
            try:
                # Load landmarks
                landmarks = np.load(npy_file)
                
                # Normalisasi
                if normalization:
                    landmarks = normalize_landmarks(landmarks, method=normalization)
                
                # Adjust sequence length
                if sequence_method == 'interpolate':
                    landmarks = interpolate_sequence(landmarks, sequence_length)
                else:
                    landmarks = pad_or_truncate(landmarks, sequence_length)
                
                X.append(landmarks)
                y.append(class_name)
                sample_info.append({
                    'class': class_name,
                    'file': npy_file.name,
                    'original_frames': np.load(npy_file).shape[0]
                })
                
            except Exception as e:
                print(f"Error loading {npy_file}: {e}")
                continue
    
    X = np.array(X)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    metadata = {
        'num_samples': len(X),
        'num_classes': len(classes),
        'classes': classes,
        'sequence_length': sequence_length,
        'num_features': X.shape[2] if len(X) > 0 else 0,
        'normalization': normalization,
        'sequence_method': sequence_method,
        'samples': sample_info
    }
    
    print(f"\nDataset loaded:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y_encoded.shape}")
    print(f"  Classes: {label_encoder.classes_}")
    
    return X, y_encoded, label_encoder, metadata


def create_train_test_split(X, y, test_size=0.2, random_state=42, stratify=True):
    """
    Split dataset menjadi train dan test.
    
    Args:
        X: Features
        y: Labels
        test_size: Proporsi test set
        random_state: Random seed
        stratify: Apakah stratified split
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify_param
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def save_processed_dataset(output_dir, X_train, X_test, y_train, y_test, 
                           label_encoder, metadata):
    """
    Simpan dataset yang sudah diproses.
    
    Args:
        output_dir: Directory output
        X_train, X_test, y_train, y_test: Data splits
        label_encoder: LabelEncoder object
        metadata: Dataset metadata
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save arrays
    np.save(output_path / 'X_train.npy', X_train)
    np.save(output_path / 'X_test.npy', X_test)
    np.save(output_path / 'y_train.npy', y_train)
    np.save(output_path / 'y_test.npy', y_test)
    
    # Save label encoder
    with open(output_path / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save metadata
    metadata['train_size'] = len(X_train)
    metadata['test_size'] = len(X_test)
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset saved to: {output_path}")
    print(f"Files created:")
    print(f"  - X_train.npy: {X_train.shape}")
    print(f"  - X_test.npy: {X_test.shape}")
    print(f"  - y_train.npy: {y_train.shape}")
    print(f"  - y_test.npy: {y_test.shape}")
    print(f"  - label_encoder.pkl")
    print(f"  - metadata.json")


def load_processed_dataset(processed_dir):
    """
    Load dataset yang sudah diproses.
    
    Args:
        processed_dir: Directory berisi processed data
        
    Returns:
        dict: {X_train, X_test, y_train, y_test, label_encoder, metadata}
    """
    processed_path = Path(processed_dir)
    
    data = {
        'X_train': np.load(processed_path / 'X_train.npy'),
        'X_test': np.load(processed_path / 'X_test.npy'),
        'y_train': np.load(processed_path / 'y_train.npy'),
        'y_test': np.load(processed_path / 'y_test.npy'),
    }
    
    with open(processed_path / 'label_encoder.pkl', 'rb') as f:
        data['label_encoder'] = pickle.load(f)
    
    with open(processed_path / 'metadata.json', 'r') as f:
        data['metadata'] = json.load(f)
    
    return data


def analyze_dataset(landmarks_dir):
    """
    Analisis statistik dataset landmarks.
    
    Args:
        landmarks_dir: Directory berisi folder per kelas
    """
    landmarks_path = Path(landmarks_dir)
    
    stats = {
        'total_samples': 0,
        'classes': {},
        'frame_lengths': [],
    }
    
    for class_folder in sorted(landmarks_path.iterdir()):
        if not class_folder.is_dir():
            continue
            
        class_name = class_folder.name
        npy_files = list(class_folder.glob('*.npy'))
        
        class_frames = []
        for npy_file in npy_files:
            data = np.load(npy_file)
            class_frames.append(data.shape[0])
            stats['frame_lengths'].append(data.shape[0])
        
        stats['classes'][class_name] = {
            'num_samples': len(npy_files),
            'avg_frames': np.mean(class_frames) if class_frames else 0,
            'min_frames': min(class_frames) if class_frames else 0,
            'max_frames': max(class_frames) if class_frames else 0,
        }
        stats['total_samples'] += len(npy_files)
    
    # Global statistics
    if stats['frame_lengths']:
        stats['global'] = {
            'avg_frames': np.mean(stats['frame_lengths']),
            'min_frames': min(stats['frame_lengths']),
            'max_frames': max(stats['frame_lengths']),
            'std_frames': np.std(stats['frame_lengths']),
            'median_frames': np.median(stats['frame_lengths']),
        }
    
    # Print report
    print("=" * 60)
    print("DATASET ANALYSIS REPORT")
    print("=" * 60)
    print(f"\nTotal samples: {stats['total_samples']}")
    print(f"Total classes: {len(stats['classes'])}")
    
    if 'global' in stats:
        print(f"\nFrame statistics:")
        print(f"  Average: {stats['global']['avg_frames']:.1f}")
        print(f"  Median: {stats['global']['median_frames']:.1f}")
        print(f"  Min: {stats['global']['min_frames']}")
        print(f"  Max: {stats['global']['max_frames']}")
        print(f"  Std: {stats['global']['std_frames']:.1f}")
    
    print(f"\nPer-class breakdown:")
    for class_name, class_stats in stats['classes'].items():
        print(f"  {class_name}: {class_stats['num_samples']} samples, "
              f"frames: {class_stats['min_frames']}-{class_stats['max_frames']} "
              f"(avg: {class_stats['avg_frames']:.1f})")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess BISINDO landmarks')
    
    parser.add_argument('--landmarks-dir', '-l', type=str, required=True,
                        help='Directory containing landmark files')
    parser.add_argument('--output-dir', '-o', type=str, required=True,
                        help='Output directory for processed data')
    parser.add_argument('--sequence-length', '-s', type=int, default=60,
                        help='Target sequence length (default: 60)')
    parser.add_argument('--normalization', '-n', type=str, default='minmax',
                        choices=['minmax', 'zscore', 'none'],
                        help='Normalization method (default: minmax)')
    parser.add_argument('--sequence-method', '-m', type=str, default='interpolate',
                        choices=['interpolate', 'pad'],
                        help='Sequence length adjustment method (default: interpolate)')
    parser.add_argument('--test-size', '-t', type=float, default=0.2,
                        help='Test set proportion (default: 0.2)')
    parser.add_argument('--analyze', '-a', action='store_true',
                        help='Only analyze dataset, no processing')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_dataset(args.landmarks_dir)
    else:
        # Load and preprocess
        norm = None if args.normalization == 'none' else args.normalization
        
        X, y, label_encoder, metadata = load_landmarks_dataset(
            landmarks_dir=args.landmarks_dir,
            sequence_length=args.sequence_length,
            normalization=norm,
            sequence_method=args.sequence_method
        )
        
        # Split
        X_train, X_test, y_train, y_test = create_train_test_split(
            X, y, test_size=args.test_size
        )
        
        # Save
        save_processed_dataset(
            output_dir=args.output_dir,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            label_encoder=label_encoder,
            metadata=metadata
        )