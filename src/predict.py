"""
BISINDO Real-time Prediction (with Palm Orientation)
=====================================================
Script untuk prediksi gesture BISINDO dengan fitur orientasi telapak tangan.

Features:
- Deteksi tangan dengan MediaPipe Holistic
- Fitur orientasi telapak tangan (palm vs back)
- Prediksi gesture menggunakan trained model
- Visualisasi landmarks dan hasil prediksi

Usage:
    python predict_orientation.py --model models/bisindo_model.keras

Author: BISINDO Project
"""

import os
import cv2
import numpy as np
import pickle
import time
from pathlib import Path
from collections import deque
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import mediapipe as mp


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


class BISINDOPredictorOrientation:
    """Real-time BISINDO predictor with palm orientation features."""
    
    def __init__(self, model_path, label_encoder_path, 
                 sequence_length=60,
                 include_finger_features=True,
                 confidence_threshold=0.7):
        
        print("Loading model...")
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded: {model_path}")
        
        print("Loading label encoder...")
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        print(f"Classes: {list(self.label_encoder.classes_)}")
        
        self.sequence_length = sequence_length
        self.include_finger_features = include_finger_features
        self.confidence_threshold = confidence_threshold
        
        # MediaPipe setup
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Feature dimensions
        self.num_hand_landmarks = 21
        self.num_hand_coords = self.num_hand_landmarks * 3  # 63
        self.num_orientation_features = 5  # normal(3) + facing(1) + openness(1)
        self.num_finger_features = 9 if include_finger_features else 0  # extensions(5) + spreads(4)
        
        self.features_per_hand = (
            self.num_hand_coords +
            self.num_orientation_features +
            self.num_finger_features
        )
        self.num_features = self.features_per_hand * 2
        
        print(f"Features per frame: {self.num_features}")
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=sequence_length)
        self.prediction_history = deque(maxlen=15)
        
        # Stats
        self.fps = 0
        self.last_time = time.time()
        self.frame_count = 0
    
    def calculate_palm_normal(self, landmarks):
        """Calculate palm normal vector."""
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
        
        fingertip_indices = [
            HandLandmark.THUMB_TIP, HandLandmark.INDEX_FINGER_TIP,
            HandLandmark.MIDDLE_FINGER_TIP, HandLandmark.RING_FINGER_TIP,
            HandLandmark.PINKY_TIP
        ]
        
        distances = []
        for idx in fingertip_indices:
            tip = np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
            distances.append(np.linalg.norm(tip - wrist))
        
        avg_dist = np.mean(distances)
        openness = min(1.0, max(0.0, (avg_dist - 0.1) / 0.3))
        
        return openness
    
    def calculate_finger_extensions(self, landmarks):
        """Calculate finger extension scores."""
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
            extensions.append(1 - (avg_angle / np.pi))
        
        return np.array(extensions)
    
    def calculate_finger_spreads(self, landmarks):
        """Calculate finger spread distances."""
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
            spreads.append(np.linalg.norm(p1 - p2))
        
        return np.array(spreads)
    
    def extract_hand_features(self, hand_landmarks):
        """Extract all features from one hand."""
        if hand_landmarks is None:
            return np.zeros(self.features_per_hand)
        
        features = []
        
        # 1. Coordinates (63)
        coords = np.array([
            [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
        ]).flatten()
        features.append(coords)
        
        # 2. Orientation (5)
        palm_normal, palm_facing = self.calculate_palm_normal(hand_landmarks.landmark)
        openness = self.calculate_hand_openness(hand_landmarks.landmark)
        features.append(np.concatenate([palm_normal, [palm_facing], [openness]]))
        
        # 3. Finger features (9)
        if self.include_finger_features:
            extensions = self.calculate_finger_extensions(hand_landmarks.landmark)
            spreads = self.calculate_finger_spreads(hand_landmarks.landmark)
            features.append(np.concatenate([extensions, spreads]))
        
        return np.concatenate(features)
    
    def extract_frame_landmarks(self, frame):
        """Extract landmarks from frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)
        
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
    
    def normalize(self, sequence):
        """Normalize sequence to [0, 1]."""
        min_val = sequence.min()
        max_val = sequence.max()
        if max_val - min_val > 0:
            return (sequence - min_val) / (max_val - min_val)
        return sequence
    
    def predict(self):
        """Predict gesture from frame buffer."""
        if len(self.frame_buffer) < self.sequence_length:
            return None, 0, None
        
        sequence = np.array(list(self.frame_buffer))
        sequence = self.normalize(sequence)
        sequence = sequence.reshape(1, self.sequence_length, self.num_features)
        
        predictions = self.model.predict(sequence, verbose=0)[0]
        
        class_idx = np.argmax(predictions)
        confidence = predictions[class_idx]
        class_name = self.label_encoder.classes_[class_idx]
        
        return class_name, confidence, predictions
    
    def get_smoothed_prediction(self, class_name, confidence):
        """
        Smooth predictions using Majority Voting with Threshold.
        Hanya return hasil jika mayoritas frame (70%+) sepakat.
        """
        # 1. Jika input None, jangan masukkan ke history (atau masukkan sebagai 'Uncertain')
        if class_name is None:
            return None, 0
            
        # 2. Masukkan prediksi terbaru ke antrian
        self.prediction_history.append((class_name, confidence))
        
        # 3. Tunggu sampai history penuh dulu baru ambil keputusan
        # (Supaya di awal tidak langsung menebak sembarangan)
        if len(self.prediction_history) < self.prediction_history.maxlen:
            return None, 0
        
        # 4. Hitung Voting (Frekuensi kemunculan tiap kelas)
        class_counts = {}
        confidence_sums = {}
        
        for cn, conf in self.prediction_history:
            if cn not in class_counts:
                class_counts[cn] = 0
                confidence_sums[cn] = 0.0
            class_counts[cn] += 1
            confidence_sums[cn] += conf
        
        # 5. Cari kandidat pemenang (Most Common)
        most_common_class = max(class_counts.keys(), key=lambda x: class_counts[x])
        total_votes = len(self.prediction_history)
        winner_votes = class_counts[most_common_class]
        
        # 6. ATURAN MAYORITAS (Threshold 70%)
        # Pemenang harus mendominasi minimal 70% dari isi history
        # Contoh: Dari 15 frame, minimal 10 frame harus "SAYA".
        vote_threshold = 0.7 
        
        if winner_votes >= (total_votes * vote_threshold):
            # Hitung rata-rata confidence HANYA dari frame yang memilih pemenang
            avg_confidence = confidence_sums[most_common_class] / winner_votes
            return most_common_class, avg_confidence
        else:
            # Jika suara terpecah (misal: A=5, B=5, C=5), anggap tidak yakin
            # Ini akan menghilangkan efek "Flickering" (kedip-kedip)
            return None, 0

    def draw_landmarks(self, frame, results):
        """Draw landmarks on frame."""
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        return frame
    
    def draw_ui(self, frame, class_name, confidence, orientation_info, is_paused=False):
        """Draw UI elements."""
        h, w = frame.shape[:2]
        
        # Top bar background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h-70), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "BISINDO Recognition (with Orientation)", 
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", 
                    (w-100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Buffer status
        buffer_pct = len(self.frame_buffer) / self.sequence_length * 100
        cv2.putText(frame, f"Buffer: {buffer_pct:.0f}%", 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Orientation display
        # Left hand
        if orientation_info['left'] is not None:
            score = orientation_info['left']
            # Untuk tangan kiri di mirror view, logika terbalik
            orient = "BACK" if score > 0.2 else "PALM" if score < -0.2 else "SIDE"
            color = (0, 255, 0) if score > 0.2 else (0, 0, 255) if score < -0.2 else (0, 255, 255)
            cv2.putText(frame, f"L: {orient}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Right hand
        if orientation_info['right'] is not None:
            score = orientation_info['right']
            orient = "PALM" if score > 0.2 else "BACK" if score < -0.2 else "SIDE"
            color = (0, 255, 0) if score > 0.2 else (0, 0, 255) if score < -0.2 else (0, 255, 255)
            cv2.putText(frame, f"R: {orient}", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Paused indicator
        if is_paused:
            cv2.putText(frame, "PAUSED", (w//2-50, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Prediction result
        if class_name and confidence >= self.confidence_threshold:
            bar_width = int(confidence * 300)
            bar_color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > 0.6 else (0, 165, 255)
            
            cv2.rectangle(frame, (10, h-55), (10 + bar_width, h-35), bar_color, -1)
            cv2.rectangle(frame, (10, h-55), (310, h-35), (255, 255, 255), 2)
            
            cv2.putText(frame, f"{class_name}", 
                        (10, h-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"{confidence*100:.1f}%", 
                        (320, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Detecting...", 
                        (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        
        # Controls
        cv2.putText(frame, "Q:Quit | C:Clear | P:Pause | L:Landmarks", 
                    (w-320, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return frame
    
    def draw_top_predictions(self, frame, predictions, top_k=5):
        """Draw top predictions panel."""
        if predictions is None:
            return frame
        
        h, w = frame.shape[:2]
        top_indices = np.argsort(predictions)[::-1][:top_k]
        
        cv2.rectangle(frame, (w-200, 110), (w-10, 110 + top_k * 25 + 10), (0, 0, 0), -1)
        
        y_offset = 130
        for idx in top_indices:
            class_name = self.label_encoder.classes_[idx]
            conf = predictions[idx]
            
            display_name = class_name[:12] + ".." if len(class_name) > 14 else class_name
            color = (0, 255, 0) if conf > 0.5 else (255, 255, 255)
            
            cv2.putText(frame, f"{display_name}: {conf*100:.1f}%", 
                        (w-190, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += 22
        
        return frame
    
    def update_fps(self):
        """Update FPS counter."""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = current_time
    
    def run(self, camera_id=0, show_landmarks=True, show_top_predictions=True):
        """Run real-time prediction."""
        print(f"\nOpening camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_id}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Camera opened!")
        print("\nControls: Q=Quit, C=Clear, P=Pause, L=Landmarks, T=Top predictions\n")
        
        is_paused = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # frame = cv2.flip(frame, 1)
            
            if not is_paused:
                landmarks, results, orientation_info = self.extract_frame_landmarks(frame)
                self.frame_buffer.append(landmarks)
                
                class_name, confidence, all_predictions = self.predict()
                class_name, confidence = self.get_smoothed_prediction(class_name, confidence)
                
                if show_landmarks:
                    frame = self.draw_landmarks(frame, results)
                
                if show_top_predictions:
                    frame = self.draw_top_predictions(frame, all_predictions)
            else:
                class_name, confidence = None, 0
                orientation_info = {'left': None, 'right': None}
            
            frame = self.draw_ui(frame, class_name, confidence, orientation_info, is_paused)
            self.update_fps()
            
            cv2.imshow('BISINDO Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.frame_buffer.clear()
                self.prediction_history.clear()
                print("Buffer cleared.")
            elif key == ord('p'):
                is_paused = not is_paused
                print("Paused." if is_paused else "Resumed.")
            elif key == ord('l'):
                show_landmarks = not show_landmarks
            elif key == ord('t'):
                show_top_predictions = not show_top_predictions
        
        cap.release()
        cv2.destroyAllWindows()
        self.holistic.close()


def main():
    parser = argparse.ArgumentParser(description='BISINDO Prediction with Palm Orientation')
    
    parser.add_argument('--model', '-m', type=str, 
                        default='models/bisindo_improved_lstm_final.keras',
                        help='Path to trained model')
    parser.add_argument('--labels', '-l', type=str, 
                        default='models/label_encoder.pkl',
                        help='Path to label encoder')
    parser.add_argument('--camera', '-c', type=int, default=0,
                        help='Camera ID')
    parser.add_argument('--sequence-length', '-s', type=int, default=60,
                        help='Sequence length')
    parser.add_argument('--threshold', '-t', type=float, default=0.7,
                        help='Confidence threshold')
    parser.add_argument('--no-finger-features', action='store_true',
                        help='Disable finger features')
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        print("\nPastikan model sudah di-download dari Google Drive!")
        return
    
    if not Path(args.labels).exists():
        print(f"Error: Label encoder not found: {args.labels}")
        return
    
    predictor = BISINDOPredictorOrientation(
        model_path=args.model,
        label_encoder_path=args.labels,
        sequence_length=args.sequence_length,
        include_finger_features=not args.no_finger_features,
        confidence_threshold=args.threshold
    )
    
    predictor.run(camera_id=args.camera)


if __name__ == "__main__":
    main()