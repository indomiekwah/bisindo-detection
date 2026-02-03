"""
BISINDO Landmark Extraction - Relative Coordinates
====================================================
Ekstraksi landmark dengan koordinat RELATIF terhadap pergelangan tangan (wrist).

Keuntungan:
- Translation Invariant: posisi tangan di layar tidak mempengaruhi hasil
- Scale Normalized: ukuran tangan dinormalisasi
- Rotation features: orientasi telapak tangan tetap dipertahankan

Features yang diekstrak per tangan (77 features):
1. Koordinat relatif: 20 landmarks × 3 = 60 (wrist jadi origin, tidak disimpan)
2. Hand scale: 1 (untuk normalisasi)
3. Palm orientation: 5 (normal vector + facing score + openness)
4. Finger features: 9 (extensions + spreads)
5. Wrist position (absolute): 2 (x, y saja, untuk referensi posisi)

Total per tangan: 77 features
Total kedua tangan: 154 features

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


class HandLandmark:
    """MediaPipe Hand Landmark indices."""
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


class RelativeLandmarkExtractor:
    """
    Ekstraksi landmark dengan koordinat relatif terhadap wrist.
    
    Ini membuat model TRANSLATION INVARIANT - tidak peduli di mana
    posisi tangan di layar, hanya peduli bentuk dan gerakan tangan.
    """
    
    def __init__(self, 
                 include_finger_features=True,
                 normalize_scale=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Initialize extractor.
        
        Args:
            include_finger_features: Include finger extension/spread features
            normalize_scale: Normalize hand size (scale invariant)
            min_detection_confidence: MediaPipe detection threshold
            min_tracking_confidence: MediaPipe tracking threshold
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
        self.normalize_scale = normalize_scale
        
        # Feature dimensions per hand:
        # - Relative coordinates: 20 landmarks × 3 = 60 (excluding wrist)
        # - Hand scale: 1
        # - Wrist absolute position: 2 (x, y only, for reference)
        # - Palm orientation: 5 (normal x,y,z + facing + openness)
        # - Finger features: 9 (5 extensions + 4 spreads) [optional]
        
        self.num_relative_coords = 20 * 3  # 60 (semua kecuali wrist)
        self.num_scale_features = 1
        self.num_wrist_position = 2  # x, y saja
        self.num_orientation_features = 5
        self.num_finger_features = 9 if include_finger_features else 0
        
        self.features_per_hand = (
            self.num_relative_coords +      # 60
            self.num_scale_features +       # 1
            self.num_wrist_position +       # 2
            self.num_orientation_features + # 5
            self.num_finger_features        # 9
        )  # Total: 77 per hand
        
        self.num_features = self.features_per_hand * 2  # 154 total
        
        print(f"\n{'='*50}")
        print("RELATIVE LANDMARK EXTRACTOR")
        print(f"{'='*50}")
        print(f"Features per hand:")
        print(f"  - Relative coordinates (20 landmarks): {self.num_relative_coords}")
        print(f"  - Hand scale: {self.num_scale_features}")
        print(f"  - Wrist position (x,y): {self.num_wrist_position}")
        print(f"  - Orientation features: {self.num_orientation_features}")
        print(f"  - Finger features: {self.num_finger_features}")
        print(f"  - Subtotal per hand: {self.features_per_hand}")
        print(f"\nTotal features: {self.num_features}")
        print(f"{'='*50}\n")
    
    def get_relative_coordinates(self, landmarks):
        """
        Convert absolute coordinates to relative (wrist = origin).
        Also normalize by hand scale if enabled.
        
        Returns:
            np.array: Relative coordinates (60,) - 20 landmarks × 3
            float: Hand scale (distance from wrist to middle finger MCP)
            tuple: Wrist absolute position (x, y)
        """
        if landmarks is None:
            return np.zeros(self.num_relative_coords), 0.0, (0.0, 0.0)
        
        # Get wrist position (origin)
        wrist = np.array([
            landmarks[HandLandmark.WRIST].x,
            landmarks[HandLandmark.WRIST].y,
            landmarks[HandLandmark.WRIST].z
        ])
        
        wrist_position = (landmarks[HandLandmark.WRIST].x, 
                         landmarks[HandLandmark.WRIST].y)
        
        # Calculate hand scale (distance from wrist to middle finger MCP)
        middle_mcp = np.array([
            landmarks[HandLandmark.MIDDLE_FINGER_MCP].x,
            landmarks[HandLandmark.MIDDLE_FINGER_MCP].y,
            landmarks[HandLandmark.MIDDLE_FINGER_MCP].z
        ])
        hand_scale = np.linalg.norm(middle_mcp - wrist)
        
        # Avoid division by zero
        if hand_scale < 0.001:
            hand_scale = 0.001
        
        # Extract relative coordinates for all landmarks except wrist
        relative_coords = []
        for i in range(1, 21):  # Skip wrist (index 0)
            lm = landmarks[i]
            
            # Relative to wrist
            rel_x = lm.x - wrist[0]
            rel_y = lm.y - wrist[1]
            rel_z = lm.z - wrist[2]
            
            # Normalize by hand scale if enabled
            if self.normalize_scale:
                rel_x /= hand_scale
                rel_y /= hand_scale
                rel_z /= hand_scale
            
            relative_coords.extend([rel_x, rel_y, rel_z])
        
        return np.array(relative_coords), hand_scale, wrist_position
    
    def calculate_palm_normal(self, landmarks):
        """Calculate palm normal vector and facing score."""
        if landmarks is None:
            return np.zeros(3), 0.0
        
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
        
        vec1 = middle_mcp - wrist
        vec2 = pinky_mcp - index_mcp
        
        normal = np.cross(vec1, vec2)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        
        palm_facing_score = -normal[2]
        
        return normal, palm_facing_score
    
    def calculate_hand_openness(self, landmarks):
        """Calculate hand openness (0=fist, 1=open)."""
        if landmarks is None:
            return 0.0
        
        wrist = np.array([landmarks[HandLandmark.WRIST].x,
                         landmarks[HandLandmark.WRIST].y,
                         landmarks[HandLandmark.WRIST].z])
        
        # Get middle MCP for scale reference
        middle_mcp = np.array([landmarks[HandLandmark.MIDDLE_FINGER_MCP].x,
                               landmarks[HandLandmark.MIDDLE_FINGER_MCP].y,
                               landmarks[HandLandmark.MIDDLE_FINGER_MCP].z])
        hand_scale = np.linalg.norm(middle_mcp - wrist)
        
        if hand_scale < 0.001:
            return 0.0
        
        fingertip_indices = [
            HandLandmark.THUMB_TIP, HandLandmark.INDEX_FINGER_TIP,
            HandLandmark.MIDDLE_FINGER_TIP, HandLandmark.RING_FINGER_TIP,
            HandLandmark.PINKY_TIP
        ]
        
        distances = []
        for idx in fingertip_indices:
            tip = np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
            # Normalize distance by hand scale
            dist = np.linalg.norm(tip - wrist) / hand_scale
            distances.append(dist)
        
        avg_dist = np.mean(distances)
        # Typical range: 1.0 (closed) to 3.0 (open)
        openness = min(1.0, max(0.0, (avg_dist - 1.0) / 2.0))
        
        return openness
    
    def calculate_finger_extensions(self, landmarks):
        """Calculate finger extension scores (0=bent, 1=straight)."""
        if landmarks is None:
            return np.zeros(5)
        
        finger_configs = [
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
            
            vec1 = pip - base
            vec2 = dip - pip
            vec3 = tip - dip
            
            def angle_between(v1, v2):
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                return np.arccos(np.clip(cos_angle, -1, 1))
            
            avg_angle = (angle_between(vec1, vec2) + angle_between(vec2, vec3)) / 2
            extension = 1 - (avg_angle / np.pi)
            extensions.append(extension)
        
        return np.array(extensions)
    
    def calculate_finger_spreads(self, landmarks):
        """Calculate normalized finger spread distances."""
        if landmarks is None:
            return np.zeros(4)
        
        # Get hand scale for normalization
        wrist = np.array([landmarks[HandLandmark.WRIST].x,
                         landmarks[HandLandmark.WRIST].y,
                         landmarks[HandLandmark.WRIST].z])
        middle_mcp = np.array([landmarks[HandLandmark.MIDDLE_FINGER_MCP].x,
                               landmarks[HandLandmark.MIDDLE_FINGER_MCP].y,
                               landmarks[HandLandmark.MIDDLE_FINGER_MCP].z])
        hand_scale = np.linalg.norm(middle_mcp - wrist)
        
        if hand_scale < 0.001:
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
            # Normalize by hand scale
            dist = np.linalg.norm(p1 - p2) / hand_scale
            spreads.append(dist)
        
        return np.array(spreads)
    
    def extract_hand_features(self, hand_landmarks):
        """
        Extract all features from one hand using relative coordinates.
        
        Returns:
            np.array: All features for one hand (77 features)
        """
        if hand_landmarks is None:
            return np.zeros(self.features_per_hand)
        
        landmarks = hand_landmarks.landmark
        features = []
        
        # 1. Relative coordinates (60 features)
        rel_coords, hand_scale, wrist_pos = self.get_relative_coordinates(landmarks)
        features.append(rel_coords)
        
        # 2. Hand scale (1 feature)
        features.append(np.array([hand_scale]))
        
        # 3. Wrist absolute position (2 features) - for reference only
        features.append(np.array([wrist_pos[0], wrist_pos[1]]))
        
        # 4. Palm orientation (5 features)
        palm_normal, palm_facing = self.calculate_palm_normal(landmarks)
        openness = self.calculate_hand_openness(landmarks)
        features.append(np.concatenate([palm_normal, [palm_facing], [openness]]))
        
        # 5. Finger features (9 features) - optional
        if self.include_finger_features:
            extensions = self.calculate_finger_extensions(landmarks)
            spreads = self.calculate_finger_spreads(landmarks)
            features.append(np.concatenate([extensions, spreads]))
        
        return np.concatenate(features)
    
    def extract_frame_landmarks(self, frame):
        """
        Extract landmarks from one frame.
        
        Returns:
            np.array: All features (154 features)
            results: MediaPipe results for visualization
            dict: Orientation info for display
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)
        
        # Extract features for both hands
        left_features = self.extract_hand_features(results.left_hand_landmarks)
        right_features = self.extract_hand_features(results.right_hand_landmarks)
        
        all_features = np.concatenate([left_features, right_features])
        
        # Orientation info for display
        orientation_info = {'left': None, 'right': None}
        
        if results.left_hand_landmarks:
            _, score = self.calculate_palm_normal(results.left_hand_landmarks.landmark)
            orientation_info['left'] = score
        
        if results.right_hand_landmarks:
            _, score = self.calculate_palm_normal(results.right_hand_landmarks.landmark)
            orientation_info['right'] = score
        
        return all_features, results, orientation_info
    
    def extract_video_landmarks(self, video_path, max_frames=None):
        """Extract landmarks from video."""
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
                'num_features': self.num_features,
                'coordinate_type': 'relative_to_wrist',
                'scale_normalized': self.normalize_scale
            }
        }
    
    def close(self):
        self.holistic.close()


def process_dataset(input_dir, output_dir, include_finger_features=True, 
                    normalize_scale=True, max_frames=None):
    """Process entire dataset."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    extractor = RelativeLandmarkExtractor(
        include_finger_features=include_finger_features,
        normalize_scale=normalize_scale
    )
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV']
    
    all_metadata = {
        'classes': [],
        'videos': [],
        'feature_dim': extractor.num_features,
        'coordinate_type': 'relative_to_wrist',
        'scale_normalized': normalize_scale,
        'include_finger_features': include_finger_features
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
    print(f"Coordinate type: relative_to_wrist")
    print(f"Scale normalized: {normalize_scale}")
    print(f"Output: {output_path}")


def test_extraction(camera_id=0):
    """Test extraction with webcam - shows BOTH hands."""
    extractor = RelativeLandmarkExtractor(include_finger_features=True)
    
    cap = cv2.VideoCapture(camera_id)
    
    print("\n" + "="*50)
    print("RELATIVE COORDINATE TEST")
    print("="*50)
    print("- Gerakkan KEDUA tangan ke berbagai posisi di layar")
    print("- Koordinat relatif akan tetap sama untuk gesture yang sama")
    print("- Press 'q' to quit")
    print("="*50 + "\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # frame = cv2.flip(frame, 1)
        features, results, orientation_info = extractor.extract_frame_landmarks(frame)
        
        # Draw landmarks for BOTH hands
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
        
        h, w = frame.shape[:2]
        
        # ==================== LEFT PANEL (Left Hand Info) ====================
        cv2.rectangle(frame, (0, 0), (350, 200), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "LEFT HAND", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if results.left_hand_landmarks:
            landmarks = results.left_hand_landmarks.landmark
            
            # Wrist position (absolute)
            wrist_x = landmarks[0].x
            wrist_y = landmarks[0].y
            cv2.putText(frame, f"Wrist (abs): ({wrist_x:.3f}, {wrist_y:.3f})", 
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Index tip relative to wrist
            rel_x = landmarks[8].x - wrist_x
            rel_y = landmarks[8].y - wrist_y
            cv2.putText(frame, f"Index tip (rel): ({rel_x:.3f}, {rel_y:.3f})", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Orientation
            if orientation_info['left'] is not None:
                score = orientation_info['left']
                # Mirror view: tangan kiri logika terbalik
                orient = "BACK" if score > 0.2 else "PALM" if score < -0.2 else "SIDE"
                color = (0, 255, 0) if orient == "PALM" else (0, 0, 255) if orient == "BACK" else (0, 255, 255)
                cv2.putText(frame, f"Orientation: {orient} ({score:.2f})", 
                           (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            cv2.putText(frame, "DETECTED", (250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Not detected", 
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            cv2.putText(frame, "---", (250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
        
        # ==================== RIGHT PANEL (Right Hand Info) ====================
        cv2.rectangle(frame, (w-350, 0), (w, 200), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "RIGHT HAND", 
                   (w-340, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if results.right_hand_landmarks:
            landmarks = results.right_hand_landmarks.landmark
            
            # Wrist position (absolute)
            wrist_x = landmarks[0].x
            wrist_y = landmarks[0].y
            cv2.putText(frame, f"Wrist (abs): ({wrist_x:.3f}, {wrist_y:.3f})", 
                       (w-340, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Index tip relative to wrist
            rel_x = landmarks[8].x - wrist_x
            rel_y = landmarks[8].y - wrist_y
            cv2.putText(frame, f"Index tip (rel): ({rel_x:.3f}, {rel_y:.3f})", 
                       (w-340, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Orientation
            if orientation_info['right'] is not None:
                score = orientation_info['right']
                orient = "PALM" if score > 0.2 else "BACK" if score < -0.2 else "SIDE"
                color = (0, 255, 0) if orient == "PALM" else (0, 0, 255) if orient == "BACK" else (0, 255, 255)
                cv2.putText(frame, f"Orientation: {orient} ({score:.2f})", 
                           (w-340, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            cv2.putText(frame, "DETECTED", (w-100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Not detected", 
                       (w-340, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            cv2.putText(frame, "---", (w-100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
        
        # ==================== BOTTOM STATUS BAR ====================
        cv2.rectangle(frame, (0, h-50), (w, h), (0, 0, 0), -1)
        
        cv2.putText(frame, f"Total features: {extractor.num_features} | Coordinate: RELATIVE (wrist=origin)", 
                   (10, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.putText(frame, "Press 'Q' to quit | Move hands around to test!", 
                   (w-350, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        cv2.imshow('Relative Coordinate Test - BOTH HANDS', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    extractor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract landmarks with relative coordinates')
    
    parser.add_argument('--input', '-i', type=str,
                        help='Input directory')
    parser.add_argument('--output', '-o', type=str,
                        help='Output directory')
    parser.add_argument('--no-finger-features', action='store_true',
                        help='Disable finger features')
    parser.add_argument('--no-scale-normalize', action='store_true',
                        help='Disable scale normalization')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Max frames per video')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Test with webcam')
    parser.add_argument('--camera', '-c', type=int, default=0,
                        help='Camera ID')
    
    args = parser.parse_args()
    
    if args.test:
        test_extraction(args.camera)
    elif args.input and args.output:
        process_dataset(
            args.input, 
            args.output,
            include_finger_features=not args.no_finger_features,
            normalize_scale=not args.no_scale_normalize,
            max_frames=args.max_frames
        )
    else:
        parser.print_help()
        print("\n" + "="*50)
        print("CONTOH PENGGUNAAN:")
        print("="*50)
        print("\n1. Test dengan webcam:")
        print("   python extract_landmarks_relative.py --test")
        print("\n2. Extract dataset:")
        print("   python extract_landmarks_relative.py -i data/raw/wl_bisindo_organized -o data/landmarks/relative")