import os
import sys
import cv2
import re
import logging
import math
import time
import random
from pathlib import Path
import argparse
from datetime import datetime
from collections import defaultdict
from threading import Thread
from queue import Queue

import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from keras import layers
from paddleocr import PaddleOCR
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore')

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from config.appconfig import (
    DB_HOST, 
    DB_NAME, 
    DB_USER, 
    DB_PASSWORD, 
    DB_PORT
)

logs_dir = Path(__file__).parent.parent / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
log_file_path = logs_dir / "config.log"

# Set up logging
logger = logging.getLogger("ANPR")
logger.setLevel(logging.DEBUG)
if logger.hasHandlers():
    logger.handlers.clear()

# Console handler
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# File handler
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info(f"Logging to console and file: {log_file_path}")

#--------------------------------------------------------------------------------------
#  Configuration Variables
#--------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output_plates"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Define directory for saving cropped plate images
PLATE_IMAGE_DIR = OUTPUT_DIR / "plate_images"
PLATE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# VEHICLE_MODEL_PATH = "models/yolov8m-seg.pt"
VEHICLE_MODEL_PATH = "models/yolov8n.pt"
VEHICLE_COLOR_MODEL_PATH = "models/EFN-model.best.h5"
MODEL_PATH="models/license_plate_detector.pt"

# PLATE_REGEX = re.compile(r'^[A-Z0-9]{7,8}$')  # Strict 7-8 character format
PLATE_REGEX = re.compile(r'^[A-Z0-9]{8}$')  # Strict 8-character Nigerian format
TRACKING_FRAMES=30
MIN_CONFIDENCE=0.45
MIN_DETECTIONS=1

db_params = {
    "host": DB_HOST,
    "database": DB_NAME,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "port": DB_PORT
}

# Generate unique output filename with timestamp~
def get_output_path(source_path=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if source_path and (src_path := Path(source_path)).exists():
        base_name = f"{src_path.stem}_{timestamp}.mp4"
    else:
        base_name = f"anpr_output_{timestamp}.mp4"
    return OUTPUT_DIR / base_name

def parse_arguments():
    parser = argparse.ArgumentParser(description="ANPR System")
    parser.add_argument(
        "--source",
        type=str,
        default=str(PROJECT_ROOT / "Resources" / "car_vid.mp4"),
        help="Input source (video path/image path/camera index)"
    )
    return parser.parse_args()

VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    8: 'boat',
}

COLOR_CLASSES = [
    'beige', 'black', 'blue', 'brown', 'gold', 'green', 'grey',
    'orange', 'pink', 'purple', 'red', 'silver', 'tan', 'white', 'yellow'
]

class FixedDepthwiseConv2D(layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

#-------------------------------------------------------------------------------
# Video Processor Class
#-------------------------------------------------------------------------------
class VideoProcessor:
    """Handles video input/output operations"""
    def __init__(self, source):
        # Handle camera indices
        if source.isdigit():
            self.cap = cv2.VideoCapture(int(source))
            self.source_type = "camera"
        else:
            self.source_path = Path(source)
            if not self.source_path.exists():
                raise FileNotFoundError(f"Source not found: {self.source_path}")
                
            self.cap = cv2.VideoCapture(str(self.source_path))
            self.source_type = "file"

        # Auto-generate output path
        self.output_path = get_output_path(source)
            
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = 0
        
        # Initialize video writer
        self.writer = cv2.VideoWriter(
            str(self.output_path),  # Use generated path
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (self.width, self.height)
        )
        
        logger.info(f"Initialized VideoProcessor: source_type={self.source_type}, fps={self.fps}, resolution=({self.width}x{self.height}), total_frames={self.total_frames}")        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        
    def release(self):
        """Release resources"""
        self.cap.release()
        self.writer.release()
        cv2.destroyAllWindows()
        logger.info("Released video resources.")
        
    def get_frames(self):
        """Generator that yields frames"""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                logger.debug("No more frames to read.")
                break
            self.frame_count += 1
            yield frame
            
    def write_frame(self, frame):
        """Write processed frame to output"""
        self.writer.write(frame)

#-------------------------------------------------------------------------------
# ANPR System Class
#-------------------------------------------------------------------------------
class ANPRProcessor:
    """Main ANPR processing class"""
    
    #----------------------------------------------------------------------------------------------
    # Initialization
    #----------------------------------------------------------------------------------------------
    def __init__(self):
        # Initialize components
        self.vehicle_model = YOLO(VEHICLE_MODEL_PATH)
        self.model = YOLO(MODEL_PATH)
        # self.ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, lang='en', det=False, rec_only=True)
        self.ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)
        # Updated plate_tracker: Key is the *initial* OCR read (best guess)
        # Value contains list of all (text, confidence) readings for similar plates
        self.plate_tracker = defaultdict(lambda: {
            'readings': [],          # List of (text, confidence) tuples
            'first_seen': None,       # Timestamp of first sighting
            'last_seen': None,        # Timestamp of most recent sighting
            'vehicle_type': "unknown", # Associated vehicle type
            'vehicle_color': "unknown", # Associated vehicle color
            'time_details': None,     # Timestamp details from last sighting
            'detection_count': 0,     # How many frames this potential plate was seen in
            'best_image_path': None,  # Path to the best quality image saved for this plate group
            'best_image_conf': 0.0    # Confidence score associated with the best image
        })
        # Track globally saved plates for the entire processing session
        self.saved_plates = set()  # Plates already saved to DB in this run
        self.color_model = tf.keras.models.load_model(
            VEHICLE_COLOR_MODEL_PATH,
            custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D}
        )
        # self.plate_history = {}  # Track recently seen plates
        # self.cooldown_period = 30  # Seconds before a plate can be re-detected
        self.db_pool = SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            connect_timeout=5 
        )
        # Processing queue for multithreading
        self.queue = Queue(maxsize=10)
        self.running = True
        logger.info("ANPRProcessor initialized.")
    
     #----------------------------------------------------------------------------------------------
    # Detect Vehicle Type
    #----------------------------------------------------------------------------------------------  
    def detect_vehicle_type(self, frame):
        """Detect vehicle type using YOLOv8 model"""
        results = self.vehicle_model.predict(frame, conf=0.5, verbose=False)
        detected_vehicles = []
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            for box, cls, conf in zip(boxes, classes, confs):
                cls_id = int(cls)
                if cls_id in VEHICLE_CLASSES:
                    x1, y1, x2, y2 = map(int, box)
                    vehicle_type = VEHICLE_CLASSES[cls_id]
                    detected_vehicles.append({
                        'type': vehicle_type,
                        'confidence': float(conf),
                        'box': (x1, y1, x2, y2)
                    })
        
        logger.debug(f"Vehicle detection: {len(detected_vehicles)} vehicles found.")
        if not detected_vehicles:
            logger.debug("No vehicles detected in this frame.")
        return detected_vehicles
    
    #----------------------------------------------------------------------------------------------
    # Detect Vehicle Color
    #----------------------------------------------------------------------------------------------  
    def predict_vehicle_color(self, cropped_image):
        """Predict vehicle color using trained model"""
        try:
            if cropped_image.size == 0:
                logger.debug("Empty cropped image in predict_vehicle_color.")
                return "unknown"
                
            # Preprocess image for color model
            img = cv2.resize(cropped_image, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Make prediction
            predictions = self.color_model.predict(img_array, verbose=0)[0]
            logger.debug(f"Color model raw predictions: {predictions}")
            top_idx = np.argmax(predictions)
            predicted_color = COLOR_CLASSES[top_idx] if predictions[top_idx] > 0.3 else "unknown"
            logger.debug(f"Predicted vehicle color: {predicted_color} with confidence {predictions[top_idx]:.2f}")
            return predicted_color
 
        except Exception as e:
            logger.error(f"Color prediction error: {str(e)}")
            return "unknown"
    
    #----------------------------------------------------------------------------------------------
    # Time Details
    #----------------------------------------------------------------------------------------------  
    def get_time_details(self):
        """Extract detailed timestamp information"""
        now = datetime.now()
        
        # Basic time info
        hour = now.hour
        
        # Time of day classification
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"

            
        # Day of week
        day_of_week = now.strftime("%A")
        
        # # Weekend or weekday
        # is_weekend = day_of_week in ["Saturday", "Sunday"]
        
        # # Peak hours (typical traffic patterns)
        # is_peak_hour = (7 <= hour < 10) or (16 <= hour < 19)
        
        return {
            "timestamp": now.isoformat(),
            "time_of_day": time_of_day,
            "day_of_week": day_of_week,
            # "is_weekend": is_weekend,
            # "is_peak_hour": is_peak_hour
        }
    
    #----------------------------------------------------------------------------------------------
    # Preprocess Plate
    #----------------------------------------------------------------------------------------------  
    def preprocess_plate(self, image):
        """Preprocess image for better OCR results"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    
    #----------------------------------------------------------------------------------------------
    # OCR License Plate
    #----------------------------------------------------------------------------------------------
    def ocr_license_plate(self, image):
        """Perform OCR on license plate image"""
        try:
            processed = self.preprocess_plate(image)
            result = self.ocr.ocr(processed, det=False, rec=True, cls=False)
            
            if not result:
                return "", 0.0
                
            # Combine results with confidence
            texts = []
            confidences = []
            for line in result:
                if line and line[0]:
                    text, conf = line[0]
                    texts.append(text)
                    confidences.append(conf)
                    
            # Clean and validate text
            combined = "".join(texts).upper()
            cleaned = re.sub(r'[^A-Z0-9]', '', combined)
            
            if PLATE_REGEX.fullmatch(cleaned):
                avg_conf = sum(confidences) / len(confidences)
                return cleaned, avg_conf
            return "", 0.0
            
        except Exception as e:
            logger.error(f"OCR Error: {str(e)}")
            return "", 0.0

    #---------------------------------------------------------------------------------------------
    # Helper: Levenshtein Distance
    #---------------------------------------------------------------------------------------------
    def _calculate_levenshtein_distance(self, s1, s2):
        """Calculates the Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._calculate_levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    #---------------------------------------------------------------------------------------------
    # Helper: Character Level Voting
    #---------------------------------------------------------------------------------------------
    def _perform_char_level_voting(self, readings):
        """Performs confidence-weighted character-level voting.

        Args:
            readings: List of (text, confidence) tuples, all expected to be 8 chars.

        Returns:
            Tuple: (voted_plate_string, average_confidence) or (None, 0) if invalid.
        """
        if not readings:
            return None, 0.0

        plate_len = 8  # Enforce 8 characters
        char_votes = [defaultdict(float) for _ in range(plate_len)]
        total_confidence_sum = 0
        valid_readings_count = 0

        for text, confidence in readings:
            if len(text) == plate_len:
                total_confidence_sum += confidence
                valid_readings_count += 1
                for i, char in enumerate(text):
                    char_votes[i][char] += confidence  # Weight vote by confidence

        if valid_readings_count == 0:
            return None, 0.0

        voted_plate = ""
        for i in range(plate_len):
            if not char_votes[i]:
                # If somehow a position has no votes, we can't form a valid plate
                return None, 0.0 
            # Choose the character with the highest total confidence score for this position
            best_char = max(char_votes[i], key=char_votes[i].get)
            voted_plate += best_char

        average_confidence = total_confidence_sum / valid_readings_count
        return voted_plate, average_confidence

    #---------------------------------------------------------------------------------------------
    # Consolidate Plates
    #---------------------------------------------------------------------------------------------
    def consolidate_plates(self, distance_threshold=3):  # Increase threshold from 2 to 3
        """Consolidates tracked plates based on Levenshtein distance and voting.

        Groups plate tracker entries whose keys are similar, performs voting
        on all readings within each group, and returns a consolidated dictionary.

        Args:
            distance_threshold: Max Levenshtein distance to group plates.

        Returns:
            Dictionary of consolidated plates: 
            { final_plate_text: { 'first_seen', 'last_seen', 'avg_confidence', 
                                'detection_count', 'vehicle_type', 'vehicle_color', 'time_details' } }
        """
        logger.info(f"Starting plate consolidation. Initial tracker size: {len(self.plate_tracker)}")
        if not self.plate_tracker:
            return {}

        # Group plates by similarity
        plates = list(self.plate_tracker.keys())
        grouped_indices = defaultdict(list)
        visited = [False] * len(plates)
        
        # First pass: Group plates with more aggressive similarity matching
        for i in range(len(plates)):
            if visited[i]:
                continue
            visited[i] = True
            current_group_key = plates[i] # Use the first plate in group as representative
            grouped_indices[current_group_key].append(i)
            
            for j in range(i + 1, len(plates)):
                if not visited[j]:
                    # Only compare if keys are roughly similar length (optimization)
                    if abs(len(plates[i]) - len(plates[j])) <= distance_threshold:
                        distance = self._calculate_levenshtein_distance(plates[i], plates[j])
                        if distance <= distance_threshold:
                            visited[j] = True
                            grouped_indices[current_group_key].append(j)
        
        # Log the initial grouping
        for group_key, indices in grouped_indices.items():
            group_plates = [plates[idx] for idx in indices]
            logger.debug(f"Initial grouping: {group_key} -> {group_plates}")
                            
        consolidated_plates = {}
        processed_keys = set()
        
        for group_key, indices in grouped_indices.items():
            if group_key in processed_keys:
                continue

            all_readings_in_group = []
            first_seen = float('inf')
            last_seen = 0
            total_detections = 0
            # Prioritize vehicle info from entries with more readings or higher confidence? For now, take latest.
            latest_vehicle_type = "unknown"
            latest_vehicle_color = "unknown"
            latest_time_details = None
            latest_seen_time_for_group = 0

            group_keys = []
            # --- Find best image within the group --- 
            group_best_image_path = None
            group_best_image_conf = 0.0
            # --- 
            
            for index in indices:
                plate_key = plates[index]
                group_keys.append(plate_key)
                data = self.plate_tracker[plate_key]
                all_readings_in_group.extend(data['readings'])
                
                # --- Track best image path/conf within the group ---
                if data.get('best_image_path') and data.get('best_image_conf', 0) > group_best_image_conf:
                    group_best_image_path = data['best_image_path']
                    group_best_image_conf = data['best_image_conf']
                # --- 
                
                if data['first_seen'] is not None:
                    first_seen = min(first_seen, data['first_seen'])
                if data['last_seen'] is not None:
                     last_seen = max(last_seen, data['last_seen'])
                     # Track which data corresponds to the absolute last seen time
                     if data['last_seen'] > latest_seen_time_for_group:
                        latest_seen_time_for_group = data['last_seen']
                        latest_vehicle_type = data['vehicle_type']
                        latest_vehicle_color = data['vehicle_color']
                        latest_time_details = data['time_details']

                total_detections += data['detection_count']
                processed_keys.add(plate_key) # Mark original keys as processed
                
            # Perform voting on all readings for this group
            voted_plate, avg_confidence = self._perform_char_level_voting(all_readings_in_group)
            
            if voted_plate and PLATE_REGEX.fullmatch(voted_plate): # Ensure voted plate is valid 8-char
                logger.debug(f"Consolidated group {group_keys} into '{voted_plate}' (conf: {avg_confidence:.2f}, detections: {total_detections})")
                
                # Second deduplication: Check if this voted plate is similar to any already consolidated plate
                duplicate_found = False
                for existing_plate in consolidated_plates.keys():
                    if self._calculate_levenshtein_distance(voted_plate, existing_plate) <= distance_threshold:
                        logger.debug(f"Merged duplicate plate after voting: '{voted_plate}' similar to existing '{existing_plate}'")
                        # Merge with existing consolidated plate (keep the one with higher confidence)
                        if avg_confidence > consolidated_plates[existing_plate]['avg_confidence']:
                            # Update the detection count but keep the existing plate text
                            consolidated_plates[existing_plate]['detection_count'] += total_detections
                            # Update other metadata if confidence is higher
                            consolidated_plates[existing_plate]['avg_confidence'] = avg_confidence
                            consolidated_plates[existing_plate]['vehicle_type'] = latest_vehicle_type
                            consolidated_plates[existing_plate]['vehicle_color'] = latest_vehicle_color
                        else:
                            # Just update the detection count
                            consolidated_plates[existing_plate]['detection_count'] += total_detections
                        duplicate_found = True
                        break
                
                # If no duplicate found, add as new entry
                if not duplicate_found:
                    consolidated_plates[voted_plate] = {
                        'first_seen': first_seen if first_seen != float('inf') else last_seen, # Handle case where only one sighting
                        'last_seen': last_seen,
                        'avg_confidence': avg_confidence,
                        'detection_count': total_detections,
                        'vehicle_type': latest_vehicle_type, 
                        'vehicle_color': latest_vehicle_color,
                        'time_details': latest_time_details,
                        'image_filename': group_best_image_path # Add image filename here
                    }
            else:
                 logger.debug(f"Discarding group {group_keys} after voting. Voted plate: '{voted_plate}'")

        # Clear the old tracker after consolidation
        # **Important:** We do NOT clear the tracker here anymore.
        # Consolidation now returns the data, but the tracker itself might still be needed
        # if save_to_database fails or for other potential future logic.
        # Clearing should happen *after* successful DB save potentially.
        # For now, let's leave the tracker uncleared here.
        # self.plate_tracker.clear() 
        logger.info(f"Consolidation finished. Final plates: {len(consolidated_plates)}")
        return consolidated_plates
        
    #---------------------------------------------------------------------------------------------
    # Save to Database
    #---------------------------------------------------------------------------------------------
    def save_to_database(self):
        """Consolidates plates and saves valid ones to the database."""
        consolidated_plates = self.consolidate_plates()

        if not consolidated_plates:
            logger.info("No plates to save after consolidation.")
            return
            
        logger.info(f"After consolidation, found {len(consolidated_plates)} potential plates:")
        for plate, data in consolidated_plates.items():
            logger.info(f"  Plate: {plate}, Confidence: {data['avg_confidence']:.2f}, " 
                      f"Detections: {data['detection_count']}, " 
                      f"Vehicle: {data['vehicle_type']}, Color: {data['vehicle_color']}")
        
        # Track filtered plates for debugging
        color_filtered = [] # Restore color filtering
        confidence_filtered = []
        detection_filtered = []
        
        filtered_plates = {}
        for plate, data in consolidated_plates.items():
            # Check each filter condition separately for better logging
            # Restore vehicle color check
            if data.get('vehicle_color', '').lower() == 'unknown':
                color_filtered.append(plate)
                continue
                
            if data['detection_count'] < MIN_DETECTIONS:
                detection_filtered.append(plate)
                continue
                
            # Use global MIN_CONFIDENCE
            if data['avg_confidence'] < MIN_CONFIDENCE:
                confidence_filtered.append(plate)
                continue
                
            # If we get here, all filters passed
            filtered_plates[plate] = data
            
        # Log detailed filtering results
        # Restore color filtering log
        if color_filtered:
            logger.warning(f"Filtered out {len(color_filtered)} plates due to unknown vehicle color: {color_filtered}")
        if confidence_filtered:
            logger.warning(f"Filtered out {len(confidence_filtered)} plates due to low confidence (<{MIN_CONFIDENCE}): {confidence_filtered}")
        if detection_filtered:
            logger.warning(f"Filtered out {len(detection_filtered)} plates due to low detection count (<{MIN_DETECTIONS}): {detection_filtered}")
        
        if not filtered_plates:
            # Restore original warning message wording
            logger.warning("All consolidated plates filtered out due to validation (color, count, confidence).")
            return
            
        # More aggressive similarity-based deduplication with the saved plates
        new_plates = {}
        similar_to_saved_plates = []
        
        for plate, data in filtered_plates.items():
            # Check if this plate is similar to any already saved plate
            similar_to_saved = False
            for saved_plate in self.saved_plates:
                if self._calculate_levenshtein_distance(plate, saved_plate) <= 3:  # Using same threshold as consolidation
                    logger.info(f"Skipping plate '{plate}' - similar to already saved plate '{saved_plate}'")
                    similar_to_saved = True
                    similar_to_saved_plates.append(plate)
                    break
            
            if not similar_to_saved:
                new_plates[plate] = data
        
        if similar_to_saved_plates:
            logger.info(f"Filtered out {len(similar_to_saved_plates)} plates similar to already saved plates: {similar_to_saved_plates}")
            
        if not new_plates:
            logger.info("All plates have already been saved to the database in this session.")
            return
            
        logger.info(f"After session-level deduplication: {len(new_plates)}/{len(filtered_plates)} plates are new")

        conn = None
        try:
            conn = self.db_pool.getconn()
            conn.autocommit = False  # Explicit transaction control

            with conn.cursor() as cursor:
                # Prepare records including the image filename
                records_to_insert = []
                for plate, data in new_plates.items():
                    # Ensure time_details is available
                    time_details = data.get('time_details')
                    if not time_details:
                        logger.warning(f"Skipping plate {plate} due to missing time_details.")
                        continue 
                    
                    records_to_insert.append(
                        (
                            datetime.fromtimestamp(data['first_seen']).isoformat(),
                            datetime.fromtimestamp(data['last_seen']).isoformat(),
                            plate,
                            data['avg_confidence'],
                            data['detection_count'],
                            data['vehicle_type'],
                            data['vehicle_color'],
                            time_details.get('time_of_day'), # Use .get for safety
                            time_details.get('day_of_week'), # Use .get for safety
                            data.get('image_filename') # Add image filename (can be None)
                        )
                    )

                # Final deduplication based on license plate within this batch
                if records_to_insert:
                    final_unique_records = []
                    seen_plates_in_batch = set()
                    for record in records_to_insert:
                        plate_str = record[2] # License plate
                        if plate_str not in seen_plates_in_batch:
                            final_unique_records.append(record)
                            seen_plates_in_batch.add(plate_str)
                        else:
                            # Find existing record to potentially update image if current is better?
                            # For now, simpler: just log the duplicate within the batch.
                            logger.warning(f"Duplicate plate '{plate_str}' detected within save batch. Keeping first instance.")
                            
                    if not final_unique_records:
                        logger.info("No unique records left after final batch deduplication.")
                        return 

                    logger.info(f"Inserting {len(final_unique_records)} unique records into DB.")
                    
                    # Update INSERT statement to include image_filename
                    sql_insert = """
                        INSERT INTO detected_plates
                        (start_time, end_time, license_plate, confidence, detection_count, 
                         vehicle_type, vehicle_color, time_of_day, day_of_week, image_filename)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (license_plate) DO UPDATE SET -- Example: Update on conflict
                            end_time = EXCLUDED.end_time,       -- Update last seen time
                            confidence = GREATEST(detected_plates.confidence, EXCLUDED.confidence), -- Keep highest confidence
                            detection_count = detected_plates.detection_count + EXCLUDED.detection_count, -- Accumulate count
                            vehicle_type = EXCLUDED.vehicle_type,   -- Update vehicle info
                            vehicle_color = EXCLUDED.vehicle_color,
                            time_of_day = EXCLUDED.time_of_day,
                            day_of_week = EXCLUDED.day_of_week,
                            -- Update image only if new one is provided and maybe based on confidence? Simpler: update if provided.
                            image_filename = COALESCE(EXCLUDED.image_filename, detected_plates.image_filename)
                    """
                    # Note: ON CONFLICT requires a unique constraint on license_plate in your DB.
                    # If you don't have one, use ON CONFLICT DO NOTHING or handle updates differently.
                    # Assuming ON CONFLICT (license_plate) DO UPDATE for better merging over time.
                    
                    cursor.executemany(sql_insert, final_unique_records)

                    conn.commit()
                    rows_affected = cursor.rowcount # Might not be accurate for ON CONFLICT DO UPDATE in all PG versions
                    logger.info(f"Database commit successful. Rows affected/updated indication: {rows_affected}")
                    
                    # Add successfully saved/updated plates to our session-wide tracking set
                    for record in final_unique_records:
                        self.saved_plates.add(record[2])  # Add the plate to saved set

        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                self.db_pool.putconn(conn)
    
    #----------------------------------------------------------------------------------------------
    # Process Frames
    #----------------------------------------------------------------------------------------------
    def process_frame(self, frame):
        """Process a single video frame"""
        try:
            if not hasattr(self, 'window_initialized'):
                cv2.namedWindow("ANPR Processing", cv2.WINDOW_NORMAL)
                self.window_initialized = True
                logger.debug("Created OpenCV window 'ANPR Processing'.")
            
            # Get timestamp details for this frame
            time_details = self.get_time_details()
            
            # Detect vehicles and log details
            vehicles = self.detect_vehicle_type(frame)
            for vehicle in vehicles:
                logger.debug(f"Vehicle details: {vehicle}")
            
            # Detect license plates using the YOLO-based model
            results = self.model.predict(frame, conf=MIN_CONFIDENCE, verbose=False)
            
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                logger.info(f"Detected {len(boxes)} potential plates in frame")
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                for box, cls, conf in zip(boxes, classes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Crop the detected plate region and run OCR
                    plate_img = frame[y1:y2, x1:x2]
                    
                    # Check if plate_img is valid before proceeding
                    if plate_img is None or plate_img.size == 0:
                        logger.warning(f"Skipping empty plate image crop at box: {(x1, y1, x2, y2)}")
                        continue # Skip to the next detection
                        
                    plate_text, plate_conf = self.ocr_license_plate(plate_img)
                    logger.info(f"OCR result: '{plate_text}' with confidence {plate_conf:.2f}")
                    
                    if plate_text and plate_conf >= MIN_CONFIDENCE:
                        # Determine vehicle type and color from detected vehicles (if any)
                        vehicle_type = "unknown"
                        vehicle_color = "unknown"
                        if vehicles:
                            # For simplicity, choose the first detected vehicle in the frame
                            v = vehicles[0]
                            vehicle_type = v['type']
                            vx1, vy1, vx2, vy2 = v['box']
                            vehicle_roi = frame[vy1:vy2, vx1:vx2]
                            vehicle_color = self.predict_vehicle_color(vehicle_roi)
                            
                        # Update tracker with the *raw* OCR reading for potential consolidation later
                        now = time.time()
                        # Use plate_text as the initial key
                        key_plate_text = plate_text 

                        tracker_entry = self.plate_tracker[key_plate_text]
                        
                        # Append the current reading
                        tracker_entry['readings'].append((plate_text, plate_conf))
                        
                        # Update timestamps and other metadata
                        if tracker_entry['first_seen'] is None:
                            tracker_entry['first_seen'] = now
                        tracker_entry['last_seen'] = now
                        tracker_entry['vehicle_type'] = vehicle_type
                        tracker_entry['vehicle_color'] = vehicle_color
                        tracker_entry['time_details'] = time_details
                        tracker_entry['detection_count'] += 1
                        
                        # --- Save best image based on confidence --- 
                        if plate_conf > tracker_entry['best_image_conf']:
                            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] # Include milliseconds
                            # Create a relatively unique filename
                            image_filename = f"{plate_text}_{timestamp_str}.jpg"
                            save_path = PLATE_IMAGE_DIR / image_filename
                            try:
                                success = cv2.imwrite(str(save_path), plate_img)
                                if success:
                                    tracker_entry['best_image_path'] = image_filename # Store relative path
                                    tracker_entry['best_image_conf'] = plate_conf
                                    logger.info(f"Saved new best image for '{key_plate_text}' with conf {plate_conf:.2f}: {image_filename}")
                                else:
                                    logger.warning(f"Failed to save image: {save_path}")
                            except Exception as img_save_error:
                                logger.error(f"Error saving image {save_path}: {img_save_error}", exc_info=True)
                        # ------------------------------------------

                        logger.info(f"Added reading '{plate_text}' ({plate_conf:.2f}) to tracker key '{key_plate_text}'. Count: {tracker_entry['detection_count']}")
                        
                        # Draw bounding box and OCR annotation on the frame (using the current frame's OCR result)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{plate_text} ({plate_conf:.2f})", 
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.7, (0, 255, 0), 2)
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
        
        cv2.imshow("ANPR Processing", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.running = False
        return frame
    
    #----------------------------------------------------------------------------------------------
    # Cleanup Tracker
    #----------------------------------------------------------------------------------------------
    def cleanup_tracker(self):
        """Cleanup old entries from tracker"""
        now = time.time()
        expired = [plate for plate, data in self.plate_tracker.items()
                  if now - data['last_seen'] > TRACKING_FRAMES]
                  
        for plate in expired:
            del self.plate_tracker[plate]
    
    #----------------------------------------------------------------------------------------------
    # Process Video
    #----------------------------------------------------------------------------------------------
    def process_video(self):
        """Main processing loop with smart frame sampling and session-specific duplicate prevention"""
        args = parse_arguments()
        with VideoProcessor(args.source) as video:
            logger.info("Starting video processing with intelligent sampling...")
            window_start = datetime.now()
            
            # Sampling parameters
            motion_threshold = 3000
            last_processed_frame = None
            skip_count = 0
            processed_count = 0
            frames_since_last_processed = 0
            
            # Track plates we've already seen in THIS session
            # session_plates = set() # REMOVED - Consolidation handles this
            
            for i, frame in enumerate(video.get_frames()):
                if frame is None or not self.running:
                    break
                
                frames_since_last_processed += 1
                
                # Log progress periodically
                if i % 30 == 0:
                    logger.info(f"Frame {i}/{video.total_frames} (processed: {processed_count}, skipped: {skip_count})")
                
                # Processing decision logic
                process_this_frame = False
                
                # Process first frame
                if last_processed_frame is None:
                    process_this_frame = True
                # Force processing periodically
                elif frames_since_last_processed >= 20:
                    process_this_frame = True
                # Skip very close frames to reduce processing
                elif frames_since_last_processed <= 5:
                    process_this_frame = False
                # Use motion detection for frames in between
                else:
                    # Convert to grayscale and downscale for faster comparison
                    small_gray = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2GRAY)
                    last_small_gray = cv2.cvtColor(cv2.resize(last_processed_frame, (320, 240)), cv2.COLOR_BGR2GRAY)
                    
                    # Compute frame difference
                    frame_diff = cv2.absdiff(small_gray, last_small_gray)
                    blur_diff = cv2.GaussianBlur(frame_diff, (5, 5), 0)
                    _, thresh_diff = cv2.threshold(blur_diff, 20, 255, cv2.THRESH_BINARY)
                    motion_score = np.sum(thresh_diff)
                    
                    if motion_score > motion_threshold:
                        process_this_frame = True
                
                # Process the frame or skip it
                if process_this_frame:
                    # Process frame and get the before/after state of the tracker
                    # before_plates = set(self.plate_tracker.keys()) # Less relevant now
                    processed_frame = self.process_frame(frame)
                    # after_plates = set(self.plate_tracker.keys())
                    
                    # Check for new plates that were detected in this frame
                    # new_plates = after_plates - before_plates # Less relevant now
                    
                    # Remove any plates we've already seen in this session
                    # (This prevents duplicates within the same run)
                    # REMOVED - Consolidation handles this
                    # for plate in list(new_plates):
                    #     if plate in session_plates:
                    #         if plate in self.plate_tracker:
                    #             logger.info(f"Removing duplicate plate from tracker: {plate}")
                    #             del self.plate_tracker[plate]
                    #     else:
                    #         # Add to our session tracking
                    #         if len(plate) == 8 and PLATE_REGEX.fullmatch(plate):
                    #             session_plates.add(plate)
                    
                    last_processed_frame = frame.copy()
                    processed_count += 1
                    frames_since_last_processed = 0
                else:
                    # Just copy the frame without processing
                    processed_frame = frame
                    skip_count += 1
                
                # Write frame to output
                video.write_frame(processed_frame)

                # --- Final Database Save --- (Runs once after loop finishes)
                if self.plate_tracker:
                    logger.info("--- Final Database Save Triggered ---")
                    self.save_to_database()

            logger.info(f"Video processing completed. Processed: {processed_count}, skipped: {skip_count}")
            # logger.info(f"Total unique plates detected in this session: {len(session_plates)}") # Removed session plates
  
#----------------------------------------------------------------------------------------------
#                                   Main Function
#----------------------------------------------------------------------------------------------
def main():
    args = parse_arguments()
    logger.info(f"Starting ANPR system with source: {args.source}")
    processor = ANPRProcessor()
    try:
        with VideoProcessor(args.source) as video:
            processor.process_video()
    except KeyboardInterrupt:
        logger.info("Shutting down due to KeyboardInterrupt...")
    except Exception as e:
        logger.error(f"Video processing failed: {str(e)}")
    finally:
        processor.db_pool.closeall()
        logger.info("Database connections closed.")

if __name__ == "__main__":
    main()
    

