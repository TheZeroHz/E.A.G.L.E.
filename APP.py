import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from scipy.optimize import linear_sum_assignment
import timm
import torch.nn.functional as F
from collections import deque
import time
import psutil

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load YOLOv5 model for person detection
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.to(device)
yolo_model.eval()

# Configure YOLOv5 for better GPU utilization
yolo_model.conf = 0.6  # Higher confidence threshold for fewer false positives
yolo_model.iou = 0.45  # NMS IoU threshold  
yolo_model.classes = [0]  # Only person class
yolo_model.max_det = 50  # Reasonable limit for performance

# Load ViT model (feature extractor) - Use a more robust model for ReID
vit_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
vit_model.to(device)
vit_model.eval()

# Advanced preprocessing pipeline for better ReID accuracy
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # Robustness to lighting
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
])

# Simple preprocessing without augmentation for inference
preprocess_inference = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
])

class PersonTracker:
    def __init__(self, track_id, initial_feature, initial_box, initial_crop_quality=1.0):
        self.track_id = track_id
        self.features = deque([initial_feature], maxlen=10)  # Keep more features for stability
        self.boxes = deque([initial_box], maxlen=10)
        self.crop_qualities = deque([initial_crop_quality], maxlen=10)  # Track crop quality
        self.age = 0
        self.time_since_update = 0
        self.hit_streak = 1
        self.confidence = 1.0
        self.velocity = np.array([0.0, 0.0])  # Simple motion model
        self.area_history = deque([self._box_area(initial_box)], maxlen=5)
        
    def _box_area(self, box):
        """Calculate box area"""
        x1, y1, x2, y2 = box
        return (x2 - x1) * (y2 - y1)
    
    def _calculate_crop_quality(self, box):
        """Calculate quality score based on size, aspect ratio"""
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        
        # Prefer larger crops
        area = width * height
        area_score = min(1.0, area / (100 * 150))  # Normalize by reasonable person size
        
        # Prefer reasonable aspect ratios (0.3 to 0.8 for person)
        aspect_ratio = width / max(height, 1)
        aspect_score = 1.0 - abs(aspect_ratio - 0.5) * 2  # Penalize extreme ratios
        aspect_score = max(0.1, aspect_score)
        
        return area_score * aspect_score
        
    def update(self, feature, box, crop_quality):
        """Update tracker with new detection"""
        # Calculate velocity for motion prediction
        if len(self.boxes) > 0:
            prev_box = self.boxes[-1]
            dx = (box[0] + box[2])/2 - (prev_box[0] + prev_box[2])/2
            dy = (box[1] + box[3])/2 - (prev_box[1] + prev_box[3])/2
            self.velocity = np.array([dx, dy]) * 0.3 + self.velocity * 0.7  # Smooth velocity
        
        # Weight features by quality - higher quality features have more influence
        if crop_quality > 0.5:  # Only update with decent quality crops
            self.features.append(feature)
            self.crop_qualities.append(crop_quality)
        
        self.boxes.append(box)
        self.area_history.append(self._box_area(box))
        self.time_since_update = 0
        self.hit_streak += 1
        self.confidence = min(1.0, self.confidence + 0.05)
        
    def predict(self):
        """Predict next position using simple motion model"""
        if len(self.boxes) > 0:
            last_box = self.boxes[-1]
            center_x = (last_box[0] + last_box[2]) / 2
            center_y = (last_box[1] + last_box[3]) / 2
            
            # Predict new center
            pred_center_x = center_x + self.velocity[0]
            pred_center_y = center_y + self.velocity[1]
            
            # Keep same size as last box
            width = last_box[2] - last_box[0]
            height = last_box[3] - last_box[1]
            
            return (int(pred_center_x - width/2), int(pred_center_y - height/2),
                   int(pred_center_x + width/2), int(pred_center_y + height/2))
        return None
    
    def get_weighted_average_feature(self):
        """Get quality-weighted averaged feature vector"""
        if len(self.features) == 0:
            return None
        
        features_array = np.array(list(self.features))
        qualities_array = np.array(list(self.crop_qualities))
        
        # Weight by crop quality
        if len(qualities_array) == len(features_array):
            weights = qualities_array / (np.sum(qualities_array) + 1e-8)
            avg_feature = np.average(features_array, axis=0, weights=weights)
        else:
            avg_feature = np.mean(features_array, axis=0)
            
        return avg_feature / (np.linalg.norm(avg_feature) + 1e-8)  # Normalize
    
    def increment_age(self):
        """Increment age and time since update"""
        self.age += 1
        self.time_since_update += 1
        self.confidence = max(0.0, self.confidence - 0.02)

class PerformanceMonitor:
    def __init__(self):
        self.frame_times = deque(maxlen=30)
        self.detection_times = deque(maxlen=30)
        self.feature_times = deque(maxlen=30)
        self.tracking_times = deque(maxlen=30)
        self.total_frames = 0
        self.start_time = time.time()
        
    def update(self, detection_time, feature_time, tracking_time, frame_time):
        self.detection_times.append(detection_time)
        self.feature_times.append(feature_time)
        self.tracking_times.append(tracking_time)
        self.frame_times.append(frame_time)
        self.total_frames += 1
    
    def get_stats(self):
        if len(self.frame_times) == 0:
            return {}
        
        fps = 1.0 / (np.mean(self.frame_times) + 1e-8)
        avg_fps = self.total_frames / (time.time() - self.start_time + 1e-8)
        
        return {
            'fps': fps,
            'avg_fps': avg_fps,
            'detection_ms': np.mean(self.detection_times) * 1000,
            'feature_ms': np.mean(self.feature_times) * 1000,
            'tracking_ms': np.mean(self.tracking_times) * 1000,
            'total_ms': np.mean(self.frame_times) * 1000,
            'cpu_percent': psutil.cpu_percent(),
            'memory_mb': psutil.virtual_memory().used / 1024 / 1024
        }

def extract_features_batch_robust(img_crops):
    """Enhanced feature extraction with quality assessment"""
    if len(img_crops) == 0:
        return [], []
    
    batch_tensors = []
    crop_qualities = []
    
    for img_crop in img_crops:
        if img_crop.size == 0:
            continue
            
        # Calculate crop quality
        h, w = img_crop.shape[:2]
        quality = min(1.0, (w * h) / (100 * 150))  # Normalize by reasonable person size
        
        # Skip very small crops
        if w < 30 or h < 60:
            continue
            
        # Convert and preprocess
        img_pil = Image.fromarray(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess_inference(img_pil)
        batch_tensors.append(input_tensor)
        crop_qualities.append(quality)
    
    if len(batch_tensors) == 0:
        return [], []
    
    # Stack into batch and move to GPU
    batch_tensor = torch.stack(batch_tensors).to(device, non_blocking=True)
    
    # Extract features in batch
    with torch.no_grad():
        features = vit_model(batch_tensor)
        # L2 normalize features
        features = F.normalize(features, p=2, dim=1)
        features = features.cpu().numpy()
    
    return features, crop_qualities

def associate_detections_to_trackers_enhanced(detections, trackers, distance_threshold=0.4, iou_threshold=0.3):
    """Enhanced association with IoU and motion prediction"""
    if len(trackers) == 0:
        return [], list(range(len(detections))), []
    
    if len(detections) == 0:
        return [], [], list(range(len(trackers)))
    
    # Compute cost matrix combining appearance and spatial information
    cost_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    
    for d, detection in enumerate(detections):
        det_feature = detection['feature']
        det_box = detection['box']
        
        for t, tracker in enumerate(trackers):
            track_feature = tracker.get_weighted_average_feature()
            
            if track_feature is not None:
                # Appearance similarity (cosine distance)
                appearance_cost = 1 - np.dot(det_feature, track_feature)
                
                # Spatial similarity (IoU with predicted position)
                pred_box = tracker.predict()
                if pred_box is not None:
                    iou = calculate_iou(det_box, pred_box)
                    spatial_cost = 1 - iou
                else:
                    spatial_cost = 1.0
                
                # Combined cost (weighted)
                cost_matrix[d, t] = 0.7 * appearance_cost + 0.3 * spatial_cost
            else:
                cost_matrix[d, t] = 1.0
    
    # Use Hungarian algorithm for optimal assignment
    if cost_matrix.size > 0:
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_detections = []
        unmatched_trackers = []
        
        # Check which assignments are valid
        for d in range(len(detections)):
            if d in row_indices:
                t = col_indices[np.where(row_indices == d)[0][0]]
                if cost_matrix[d, t] < distance_threshold:
                    matches.append((d, t))
                else:
                    unmatched_detections.append(d)
            else:
                unmatched_detections.append(d)
        
        for t in range(len(trackers)):
            if t not in col_indices or cost_matrix[row_indices[np.where(col_indices == t)[0][0]], t] >= distance_threshold:
                unmatched_trackers.append(t)
    else:
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_trackers = list(range(len(trackers)))
    
    return matches, unmatched_detections, unmatched_trackers

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-8)

# Enhanced tracking parameters based on research findings
MAX_DISAPPEARED = 25  # Frames to keep lost tracks
MIN_HITS = 2  # Minimum hits before confirming track (reduced for faster response)
MAX_DISTANCE_THRESHOLD = 0.5  # More strict threshold for appearance matching

# Initialize
trackers = []
next_id = 1
performance_monitor = PerformanceMonitor()

cap = cv2.VideoCapture('PeopleWalk.mp4')
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video properties: {width}x{height} @ {fps} FPS")

frame_count = 0

while True:
    frame_start = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1

    # Detection phase
    detection_start = time.time()
    with torch.no_grad():
        results = yolo_model(frame)
    detections = results.xyxy[0]
    detection_time = time.time() - detection_start

    # Feature extraction phase
    feature_start = time.time()
    if len(detections) == 0:
        detection_data = []
        current_features = []
        crop_qualities = []
    else:
        detections_cpu = detections.cpu().numpy()
        current_crops = []
        current_boxes = []

        # Enhanced crop extraction with quality control
        for detection in detections_cpu:
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Dynamic padding based on crop size
            w, h = x2 - x1, y2 - y1
            padding = max(5, min(w, h) // 10)  # Adaptive padding
            
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding) 
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)
            
            person_crop = frame[y1:y2, x1:x2]
            
            # Quality filtering
            if person_crop.size > 0 and (x2-x1) >= 40 and (y2-y1) >= 80:
                current_crops.append(person_crop)
                current_boxes.append((x1, y1, x2, y2))

        current_features, crop_qualities = extract_features_batch_robust(current_crops)
        
        detection_data = []
        for i, (feature, box, quality) in enumerate(zip(current_features, current_boxes, crop_qualities)):
            detection_data.append({
                'feature': feature,
                'box': box,
                'quality': quality
            })
    
    feature_time = time.time() - feature_start

    # Tracking phase
    tracking_start = time.time()
    
    # Update all trackers' age
    for tracker in trackers:
        tracker.increment_age()

    # Enhanced association
    matches, unmatched_detections, unmatched_trackers = associate_detections_to_trackers_enhanced(
        detection_data, trackers, distance_threshold=MAX_DISTANCE_THRESHOLD
    )

    # Update matched trackers
    for detection_idx, tracker_idx in matches:
        detection = detection_data[detection_idx]
        trackers[tracker_idx].update(detection['feature'], detection['box'], detection['quality'])

    # Create new trackers for high-quality unmatched detections
    for detection_idx in unmatched_detections:
        detection = detection_data[detection_idx]
        if detection['quality'] > 0.3:  # Only create tracks for decent quality detections
            new_tracker = PersonTracker(next_id, detection['feature'], detection['box'], detection['quality'])
            trackers.append(new_tracker)
            next_id += 1

    # Remove old trackers
    trackers = [t for t in trackers if t.time_since_update < MAX_DISAPPEARED]
    
    tracking_time = time.time() - tracking_start

    # Visualization
    for tracker in trackers:
        if tracker.hit_streak >= MIN_HITS or tracker.age < MIN_HITS:
            box = tracker.boxes[-1] if len(tracker.boxes) > 0 else None
            if box is not None:
                x1, y1, x2, y2 = box
                
                # Enhanced color coding
                if tracker.confidence > 0.8:
                    color = (0, 255, 0)  # Green for high confidence
                elif tracker.confidence > 0.5:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 165, 255)  # Orange for low confidence
                
                thickness = 3 if tracker.hit_streak >= MIN_HITS else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Enhanced labeling
                label = f'ID:{tracker.track_id} C:{tracker.confidence:.2f}'
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, color, 2)

    # Performance statistics
    frame_time = time.time() - frame_start
    performance_monitor.update(detection_time, feature_time, tracking_time, frame_time)
    
    # Display performance info
    if frame_count % 10 == 0:  # Update every 10 frames for smooth display
        stats = performance_monitor.get_stats()
        
        # Performance overlay
        y_offset = 30
        info_texts = [
            f"FPS: {stats.get('fps', 0):.1f} (Avg: {stats.get('avg_fps', 0):.1f})",
            f"Detection: {stats.get('detection_ms', 0):.1f}ms",
            f"Features: {stats.get('feature_ms', 0):.1f}ms", 
            f"Tracking: {stats.get('tracking_ms', 0):.1f}ms",
            f"Total: {stats.get('total_ms', 0):.1f}ms",
            f"CPU: {stats.get('cpu_percent', 0):.1f}%",
            f"Active Tracks: {len([t for t in trackers if t.hit_streak >= MIN_HITS])}"
        ]
        
        for text in info_texts:
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            y_offset += 25
        
        # GPU info if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            cv2.putText(frame, f"GPU: {gpu_memory:.2f}GB", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Enhanced Person Re-ID Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

# Final statistics
final_stats = performance_monitor.get_stats()
print("\n=== Final Performance Report ===")
print(f"Average FPS: {final_stats.get('avg_fps', 0):.2f}")
print(f"Total unique IDs: {next_id - 1}")
print(f"Average detection time: {final_stats.get('detection_ms', 0):.2f}ms")
print(f"Average feature extraction time: {final_stats.get('feature_ms', 0):.2f}ms") 
print(f"Average tracking time: {final_stats.get('tracking_ms', 0):.2f}ms")
if torch.cuda.is_available():
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")