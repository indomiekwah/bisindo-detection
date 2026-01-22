"""
BISINDO Landmark Extraction with Palm Orientation
==================================================
Ekstraksi landmark dengan fitur tambahan untuk orientasi telapak tangan.

Features yang diekstrak:
1. Tangan kiri (21 landmarks × 3 = 63)
2. Tangan kanan (21 landmarks × 3 = 63)
3. Orientasi tangan kiri (palm normal vector × 3 + palm facing score × 1 = 4)
4. Orientasi tangan kanan (palm normal vector × 3 + palm facing score × 1 = 4)
5. Fitur tambahan: jarak antar jari, sudut jari, dll (opsional)

Total: 126 + 8 = 134 features (atau lebih dengan fitur tambahan)

Author: BISINDO Project
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import json
from tqdm import tqdm
import argparse


# MediaPipe Hand Landmark indices
class HandLandmark:
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


class OrientationLandmarkExtractor:
    """Ekstraksi landmark dengan fitur orientasi telapak tangan."""
    
    def __init__(self, 
                 include_finger_features=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Initialize extractor.
        
        Args:
            include_finger_features: Include fitur jarak & sudut jari
            min_detection_confidence: Threshold deteksi
            min_tracking_confidence: Threshold tracking
        """
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.include_finger_features = include_finger_features
        
        # Calculate total features
        self.num_hand_landmarks = 21
        self.num_hand_coords = self.num_hand_landmarks * 3  # 63 per hand
        
        # Orientation features per hand:
        # - Palm normal vector (3)
        # - Palm facing score (1) - positive = palm, negative = back
        # - Hand openness (1) - seberapa terbuka tangan
        self.num_orientation_features = 5  # per hand
        
        # Finger features per hand (optional):
        # - 5 finger extension scores
        # - 4 finger spreads (jarak antar jari)
        self.num_finger_features = 9 if include_finger_features else 0
        
        # Total per hand
        self.features_per_hand = (
            self.num_hand_coords +           # 63: x,y,z coordinates
            self.num_orientation_features +   # 5: orientation
            self.num_finger_features          # 9: finger features (optional)
        )
        
        # Total features (both hands)
        self.num_features = self.features_per_hand * 2
        
        print(f"\n{'='*50}")
        print("FEATURE CONFIGURATION")
        print(f"{'='*50}")
        print(f"Per hand:")
        print(f"  - Landmark coordinates: {self.num_hand_coords}")
        print(f"  - Orientation features: {self.num_orientation_features}")
        print(f"  - Finger features: {self.num_finger_features}")
        print(f"  - Subtotal: {self.features_per_hand}")
        print(f"\nTotal (both hands): {self.num_features}")
        print(f"{'='*50}\n")
    
    def calculate_palm_normal(self, landmarks):
        """
        Hitung normal vector dari telapak tangan.
        
        Normal vector menunjukkan arah telapak tangan menghadap.
        
        Returns:
            np.array: Normal vector (3,)
            float: Palm facing score (-1 to 1)
        """
        if landmarks is None:
            return np.zeros(3), 0.0
        
        # Get key points
        wrist = np.array([landmarks[HandLandmark.WRIST].x,
                         landmarks[HandLandmark.WRIST].y,
                         landmarks[HandLandmark.WRIST].z])
        
        index_mcp = np.array([landmarks[HandLandmark.INDEX_FINGER_MCP].x,
                              landmarks[HandLandmark.INDEX_FINGER_MCP].y,
                              landmarks[HandLandmark.INDEX_FINGER_MCP].z])
        
        pinky_mcp = np.array([landmarks[HandLandmark.PINKY_MCP].x,
                              landmarks[HandLandmark.PINKY_MCP].y,
                              landmarks[HandLandmark.PINKY_MCP].z])
        
        middle_mcp = np.array([landmarks[HandLandmark.MIDDLE_FINGER_MCP].x,
                               landmarks[HandLandmark.MIDDLE_FINGER_MCP].y,
                               landmarks[HandLandmark.MIDDLE_FINGER_MCP].z])
        
        # Create vectors on palm plane
        vec1 = middle_mcp - wrist
        vec2 = pinky_mcp - index_mcp
        
        # Calculate normal using cross product
        normal = np.cross(vec1, vec2)
        
        # Normalize
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        
        # Palm facing score: positive z = palm facing camera
        # Negative z = back of hand facing camera
        palm_facing_score = -normal[2]  # Negate because of coordinate system
        
        return normal, palm_facing_score
    
    def calculate_hand_openness(self, landmarks):
        """
        Hitung seberapa terbuka tangan (0 = mengepal, 1 = terbuka).
        """
        if landmarks is None:
            return 0.0
        
        # Calculate average distance from fingertips to wrist
        wrist = np.array([landmarks[HandLandmark.WRIST].x,
                         landmarks[HandLandmark.WRIST].y,
                         landmarks[HandLandmark.WRIST].z])
        
        fingertip_indices = [
            HandLandmark.THUMB_TIP,
            HandLandmark.INDEX_FINGER_TIP,
            HandLandmark.MIDDLE_FINGER_TIP,
            HandLandmark.RING_FINGER_TIP,
            HandLandmark.PINKY_TIP
        ]
        
        distances = []
        for idx in fingertip_indices:
            tip = np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
            dist = np.linalg.norm(tip - wrist)
            distances.append(dist)
        
        # Normalize (typical range 0.1 - 0.4)
        avg_dist = np.mean(distances)
        openness = min(1.0, max(0.0, (avg_dist - 0.1) / 0.3))
        
        return openness
    
    def calculate_finger_extensions(self, landmarks):
        """
        Hitung seberapa lurus setiap jari (0 = ditekuk, 1 = lurus).
        
        Returns:
            np.array: Extension score untuk 5 jari
        """
        if landmarks is None:
            return np.zeros(5)
        
        finger_configs = [
            # (MCP, PIP, DIP, TIP) - or (CMC, MCP, IP, TIP) for thumb
            (HandLandmark.THUMB_CMC, HandLandmark.THUMB_MCP, 
             HandLandmark.THUMB_IP, HandLandmark.THUMB_TIP),
            (HandLandmark.INDEX_FINGER_MCP, HandLandmark.INDEX_FINGER_PIP,
             HandLandmark.INDEX_FINGER_DIP, HandLandmark.INDEX_FINGER_TIP),
            (HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.MIDDLE_FINGER_PIP,
             HandLandmark.MIDDLE_FINGER_DIP, HandLandmark.MIDDLE_FINGER_TIP),
            (HandLandmark.RING_FINGER_MCP, HandLandmark.RING_FINGER_PIP,
             HandLandmark.RING_FINGER_DIP, HandLandmark.RING_FINGER_TIP),
            (HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP,
             HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP),
        ]
        
        extensions = []
        
        for base_idx, pip_idx, dip_idx, tip_idx in finger_configs:
            base = np.array([landmarks[base_idx].x, landmarks[base_idx].y, landmarks[base_idx].z])
            pip = np.array([landmarks[pip_idx].x, landmarks[pip_idx].y, landmarks[pip_idx].z])
            dip = np.array([landmarks[dip_idx].x, landmarks[dip_idx].y, landmarks[dip_idx].z])
            tip = np.array([landmarks[tip_idx].x, landmarks[tip_idx].y, landmarks[tip_idx].z])
            
            # Calculate vectors
            vec1 = pip - base
            vec2 = dip - pip
            vec3 = tip - dip
            
            # Calculate angles
            def angle_between(v1, v2):
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                cos_angle = np.clip(cos_angle, -1, 1)
                return np.arccos(cos_angle)
            
            angle1 = angle_between(vec1, vec2)
            angle2 = angle_between(vec2, vec3)
            
            # Straight finger has small angles
            # Extension score: 1 = straight, 0 = bent
            avg_angle = (angle1 + angle2) / 2
            extension = 1 - (avg_angle / np.pi)
            extensions.append(extension)
        
        return np.array(extensions)
    
    def calculate_finger_spreads(self, landmarks):
        """
        Hitung jarak antar jari yang bersebelahan.
        
        Returns:
            np.array: 4 spread values (thumb-index, index-middle, middle-ring, ring-pinky)
        """
        if landmarks is None:
            return np.zeros(4)
        
        tip_pairs = [
            (HandLandmark.THUMB_TIP, HandLandmark.INDEX_FINGER_TIP),
            (HandLandmark.INDEX_FINGER_TIP, HandLandmark.MIDDLE_FINGER_TIP),
            (HandLandmark.MIDDLE_FINGER_TIP, HandLandmark.RING_FINGER_TIP),
            (HandLandmark.RING_FINGER_TIP, HandLandmark.PINKY_TIP),
        ]
        
        spreads = []
        for idx1, idx2 in tip_pairs:
            p1 = np.array([landmarks[idx1].x, landmarks[idx1].y, landmarks[idx1].z])
            p2 = np.array([landmarks[idx2].x, landmarks[idx2].y, landmarks[idx2].z])
            dist = np.linalg.norm(p1 - p2)
            spreads.append(dist)
        
        return np.array(spreads)
    
    def extract_hand_features(self, hand_landmarks):
        """
        Extract semua fitur dari satu tangan.
        
        Returns:
            np.array: All features for one hand
        """
        if hand_landmarks is None:
            return np.zeros(self.features_per_hand)
        
        features = []
        
        # 1. Raw landmark coordinates (63 features)
        coords = np.array([
            [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
        ]).flatten()
        features.append(coords)
        
        # 2. Palm orientation (5 features)
        palm_normal, palm_facing = self.calculate_palm_normal(hand_landmarks.landmark)
        openness = self.calculate_hand_openness(hand_landmarks.landmark)
        
        orientation_features = np.concatenate([
            palm_normal,           # 3 values
            [palm_facing],         # 1 value
            [openness]             # 1 value
        ])
        features.append(orientation_features)
        
        # 3. Finger features (9 features, optional)
        if self.include_finger_features:
            extensions = self.calculate_finger_extensions(hand_landmarks.landmark)  # 5 values
            spreads = self.calculate_finger_spreads(hand_landmarks.landmark)        # 4 values
            finger_features = np.concatenate([extensions, spreads])
            features.append(finger_features)
        
        return np.concatenate(features)
    
    def extract_frame_landmarks(self, frame):
        """
        Extract landmarks dan fitur dari satu frame.
        
        Returns:
            np.array: All features (num_features,)
            results: MediaPipe results untuk visualisasi
            dict: Orientation info untuk debug
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)
        
        # Extract features for both hands
        left_features = self.extract_hand_features(results.left_hand_landmarks)
        right_features = self.extract_hand_features(results.right_hand_landmarks)
        
        all_features = np.concatenate([left_features, right_features])
        
        # Orientation info for debug/visualization
        orientation_info = {
            'left_palm_facing': None,
            'right_palm_facing': None,
        }
        
        if results.left_hand_landmarks:
            _, palm_facing = self.calculate_palm_normal(results.left_hand_landmarks.landmark)
            orientation_info['left_palm_facing'] = palm_facing
        
        if results.right_hand_landmarks:
            _, palm_facing = self.calculate_palm_normal(results.right_hand_landmarks.landmark)
            orientation_info['right_palm_facing'] = palm_facing
        
        return all_features, results, orientation_info
    
    def extract_video_landmarks(self, video_path, max_frames=None):
        """Extract landmarks dari video."""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        all_landmarks = []
        frames_with_hands = 0
        frame_count = 0
        
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks, results, _ = self.extract_frame_landmarks(frame)
            all_landmarks.append(landmarks)
            
            if results.left_hand_landmarks or results.right_hand_landmarks:
                frames_with_hands += 1
            
            frame_count += 1
        
        cap.release()
        
        return {
            'landmarks': np.array(all_landmarks),
            'metadata': {
                'video_path': str(video_path),
                'fps': fps,
                'total_frames': frame_count,
                'frames_with_hands': frames_with_hands,
                'detection_rate': frames_with_hands / frame_count if frame_count > 0 else 0,
                'num_features': self.num_features
            }
        }
    
    def close(self):
        self.holistic.close()


def process_dataset(input_dir, output_dir, include_finger_features=True, max_frames=None):
    """Process entire dataset."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    extractor = OrientationLandmarkExtractor(include_finger_features=include_finger_features)
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV']
    
    all_metadata = {
        'classes': [],
        'videos': [],
        'feature_dim': extractor.num_features,
        'include_finger_features': include_finger_features,
        'feature_breakdown': {
            'hand_coordinates': extractor.num_hand_coords * 2,
            'orientation_features': extractor.num_orientation_features * 2,
            'finger_features': extractor.num_finger_features * 2
        }
    }
    
    class_folders = [f for f in input_path.iterdir() if f.is_dir()]
    print(f"Found {len(class_folders)} classes")
    
    for class_folder in sorted(class_folders):
        class_name = class_folder.name
        all_metadata['classes'].append(class_name)
        
        class_output = output_path / class_name
        class_output.mkdir(exist_ok=True)
        
        videos = [f for f in class_folder.iterdir() if f.suffix in video_extensions]
        print(f"\nProcessing: {class_name} ({len(videos)} videos)")
        
        for video_file in tqdm(videos, desc=class_name):
            try:
                result = extractor.extract_video_landmarks(video_file, max_frames)
                
                output_file = class_output / f"{video_file.stem}.npy"
                np.save(output_file, result['landmarks'])
                
                all_metadata['videos'].append({
                    'class': class_name,
                    'video': video_file.name,
                    'output': str(output_file.relative_to(output_path)),
                    'num_frames': result['landmarks'].shape[0],
                    'detection_rate': result['metadata']['detection_rate']
                })
                
            except Exception as e:
                print(f"\nError: {video_file}: {e}")
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    extractor.close()
    
    print(f"\n{'='*50}")
    print("EXTRACTION COMPLETE!")
    print(f"{'='*50}")
    print(f"Classes: {len(all_metadata['classes'])}")
    print(f"Videos: {len(all_metadata['videos'])}")
    print(f"Features: {extractor.num_features}")
    print(f"Output: {output_path}")


def test_orientation(camera_id=0):
    """Test orientasi detection dengan webcam."""
    extractor = OrientationLandmarkExtractor(include_finger_features=True)
    
    cap = cv2.VideoCapture(camera_id)
    
    print("\n" + "="*50)
    print("PALM ORIENTATION TEST")
    print("="*50)
    print("- Putar tangan Anda untuk melihat deteksi orientasi")
    print("- PALM = telapak tangan menghadap kamera")
    print("- BACK = punggung tangan menghadap kamera")
    print("- Press 'q' to quit")
    print("="*50 + "\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # frame = cv2.flip(frame, 1)
        features, results, orientation_info = extractor.extract_frame_landmarks(frame)
        
        # Draw landmarks
        if results.left_hand_landmarks:
            extractor.mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, 
                extractor.mp_holistic.HAND_CONNECTIONS,
                extractor.mp_drawing_styles.get_default_hand_landmarks_style(),
                extractor.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        if results.right_hand_landmarks:
            extractor.mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks,
                extractor.mp_holistic.HAND_CONNECTIONS,
                extractor.mp_drawing_styles.get_default_hand_landmarks_style(),
                extractor.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Draw orientation info
        h, w = frame.shape[:2]
        
        # Background
        cv2.rectangle(frame, (0, 0), (350, 150), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.7, frame, 0.3, 0, frame)
        
        # Left hand
        if orientation_info['left_palm_facing'] is not None:
            score = orientation_info['left_palm_facing']
            orientation = "BACK" if score > 0.2 else "PALM" if score < -0.2 else "SIDE"
            color = (0,255,0) if score > 0.2 else (0, 0, 255) if score < -0.2 else (0, 255, 255)
            cv2.putText(frame, f"Left: {orientation} ({score:.2f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, "Left: Not detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        
        # Right hand
        if orientation_info['right_palm_facing'] is not None:
            score = orientation_info['right_palm_facing']
            orientation = "PALM" if score > 0.2 else "BACK" if score < -0.2 else "SIDE"
            color = (0, 255, 0) if score > 0.2 else (0, 0, 255) if score < -0.2 else (0, 255, 255)
            cv2.putText(frame, f"Right: {orientation} ({score:.2f})", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, "Right: Not detected", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        
        # Feature count
        cv2.putText(frame, f"Total features: {extractor.num_features}", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Legend
        cv2.putText(frame, "GREEN=Palm  RED=Back  YELLOW=Side", 
                   (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow('Palm Orientation Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    extractor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract landmarks with palm orientation')
    
    parser.add_argument('--input', '-i', type=str,
                        help='Input directory')
    parser.add_argument('--output', '-o', type=str,
                        help='Output directory')
    parser.add_argument('--no-finger-features', action='store_true',
                        help='Disable finger extension/spread features')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Max frames per video')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Test orientation detection')
    parser.add_argument('--camera', '-c', type=int, default=0,
                        help='Camera ID')
    
    args = parser.parse_args()
    
    if args.test:
        test_orientation(args.camera)
    elif args.input and args.output:
        process_dataset(
            args.input, 
            args.output,
            include_finger_features=not args.no_finger_features,
            max_frames=args.max_frames
        )
    else:
        parser.print_help()
        print("\n" + "="*50)
        print("CONTOH PENGGUNAAN:")
        print("="*50)
        print("\n1. Test deteksi orientasi:")
        print("   python extract_landmarks_orientation.py --test")
        print("\n2. Extract dataset dengan orientasi:")
        print("   python extract_landmarks_orientation.py -i data/raw/wl_bisindo_organized -o data/landmarks/with_orientation")
