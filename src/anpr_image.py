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
        # Change extension for image output
        base_name = f"{src_path.stem}_{timestamp}.jpg"
    else:
        # Change extension for image output
        base_name = f"anpr_output_{timestamp}.jpg"
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
# Image Processor Class (Formerly VideoProcessor)
#-------------------------------------------------------------------------------
class ImageProcessor:
    """Handles image input/output operations"""
    def __init__(self, source):
        self.source_path = Path(source)
        if not self.source_path.exists():
            raise FileNotFoundError(f"Source image not found: {self.source_path}")
        if not self.source_path.is_file():
             raise ValueError(f"Source must be a file, not a directory: {self.source_path}")

        self.source_type = "image"
        # Load image immediately to get dimensions
        self.image = self.load_image()
        if self.image is None:
            raise ValueError(f"Could not load image: {self.source_path}")

        # Auto-generate output path
        self.output_path = get_output_path(source)

        self.height, self.width = self.image.shape[:2]

        logger.info(f"Initialized ImageProcessor: source_type={self.source_type}, resolution=({self.width}x{self.height})")

    def load_image(self):
        """Loads the image file"""
        img = cv2.imread(str(self.source_path))
        if img is None:
            logger.error(f"Failed to load image: {self.source_path}")
        return img

    def get_image(self):
        """Returns the loaded image"""
        # Return a copy to prevent modification of the original loaded image
        return self.image.copy() if self.image is not None else None

    def save_image(self, image, output_path=None):
        """Saves the processed image"""
        path_to_save = output_path if output_path else self.output_path
        try:
            success = cv2.imwrite(str(path_to_save), image)
            if success:
                logger.info(f"Processed image saved to: {path_to_save}")
                return True
            else:
                logger.warning(f"Failed to save image to: {path_to_save}")
                return False
        except Exception as e:
            logger.error(f"Error saving image {path_to_save}: {e}", exc_info=True)
            return False

    # No release needed for image loading
    # No __enter__ or __exit__ needed

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
        self.ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)

        # Track globally saved plates for the entire processing session
        self.saved_plates = set()  # Keep this for preventing duplicate DB entries across runs/calls

        self.color_model = tf.keras.models.load_model(
            VEHICLE_COLOR_MODEL_PATH,
            custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D}
        )
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
        logger.info("ANPRProcessor initialized for image processing.") # Updated log
    
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
        """Preprocess image for better OCR results using CLAHE and Adaptive Thresholding."""
        if image is None or image.size == 0:
            logger.warning("preprocess_plate received an empty image.")
            return None # Return None if image is invalid
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            contrast_enhanced_gray = clahe.apply(gray)
            logger.debug("Applied CLAHE for contrast enhancement.")

            # Apply Gaussian Blur slightly
            # blurred = cv2.GaussianBlur(contrast_enhanced_gray, (3, 3), 0)
            # logger.debug("Applied Gaussian Blur.")
            # Experiment: Skip blur or use median blur for salt-and-pepper noise
            blurred = cv2.medianBlur(contrast_enhanced_gray, 3)
            logger.debug("Applied Median Blur.")

            # Apply Adaptive Thresholding
            # Adjust blockSize and C for optimal results
            # blockSize must be odd
            # C is a constant subtracted from the mean or weighted sum
            adaptive_thresh = cv2.adaptiveThreshold(
                blurred, 
                255, # Max value
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Use Gaussian weighting for neighborhood
                cv2.THRESH_BINARY, # Standard binary threshold
                blockSize=15, # Size of the neighborhood area (must be odd)
                C=7 # Constant subtracted from the calculated threshold
            )
            logger.debug("Applied Adaptive Thresholding.")
            
            # Optional: Denoising (can be slow, apply if noise is significant)
            # denoised = cv2.fastNlMeansDenoising(adaptive_thresh, None, h=10, templateWindowSize=7, searchWindowSize=21)
            # logger.debug("Applied Denoising.")
            # return denoised
            
            return adaptive_thresh
        except cv2.error as cv_err:
            logger.error(f"OpenCV error during preprocessing: {cv_err}")
            return None # Return None on OpenCV errors
        except Exception as e:
            logger.error(f"Unexpected error during preprocessing: {e}", exc_info=True)
            return None # Return None on other errors
    
    #----------------------------------------------------------------------------------------------
    # OCR License Plate
    #----------------------------------------------------------------------------------------------
    def ocr_license_plate(self, image):
        """Perform OCR on license plate image"""
        try:
            processed = self.preprocess_plate(image)
            # Check if preprocessing failed
            if processed is None:
                 logger.warning("Skipping OCR because preprocessing failed.")
                 return "", 0.0
            
            # Log the preprocessed image for debugging (optional, can generate many files)
            # debug_img_path = PLATE_IMAGE_DIR / f"preprocessed_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
            # cv2.imwrite(str(debug_img_path), processed)
            # logger.debug(f"Saved preprocessed image to {debug_img_path}")
                 
            result = self.ocr.ocr(processed, det=False, rec=True, cls=False)
            logger.debug(f"Raw PaddleOCR result: {result}") # Log the raw result
            
            # Check if result is valid and has the expected structure
            if not result or not result[0] or not isinstance(result[0], list) or not result[0][0]:
                 logger.warning("PaddleOCR returned empty or invalid result.")
                 return "", 0.0
            
            # If PaddleOCR returns multiple lines/candidates, process them
            # For now, assume the most likely result is the first one as before
            # but log if there are more than one.
            if len(result[0]) > 1:
                logger.debug(f"PaddleOCR returned {len(result[0])} potential text lines/candidates. Processing the first one.")

            texts = []
            confidences = []
            # Iterate through all lines returned for the first (and likely only) detection box
            for line_info in result[0]: # result[0] contains list of [text, confidence] pairs
                if isinstance(line_info, (list, tuple)) and len(line_info) == 2:
                    text, conf = line_info
                    if isinstance(text, str) and isinstance(conf, (float, int)):
                         texts.append(text)
                         confidences.append(float(conf))
                    else:
                         logger.warning(f"Unexpected format in PaddleOCR line result: {line_info}")
                else:
                    logger.warning(f"Unexpected structure in PaddleOCR result list item: {line_info}")

            if not texts: # If no valid text/confidence pairs found
                logger.warning("No valid text/confidence pairs extracted from PaddleOCR result.")
                return "", 0.0
                    
            # Clean and validate text
            combined = "".join(texts).upper()
            # More aggressive cleaning: remove spaces and hyphens as well
            cleaned = re.sub(r'[^A-Z0-9]', '', combined) 
            logger.debug(f"Combined text: '{combined}', Cleaned text: '{cleaned}'")
            
            # Initial regex check
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            if PLATE_REGEX.fullmatch(cleaned):
                logger.info(f"Valid plate format found: '{cleaned}' with avg confidence {avg_conf:.2f}")
                return cleaned, avg_conf
            else:
                 logger.warning(f"Cleaned text '{cleaned}' ({len(cleaned)} chars) does not match PLATE_REGEX {PLATE_REGEX.pattern}. Attempting correction...")

                 # --- Post-OCR Correction Heuristic (Length 7 -> 8) ---
                 corrected_plate = None
                 if len(cleaned) == 7:
                     # Simple substitution map for potential last character errors (Digit -> Letter)
                     # Extend this map as needed based on common errors
                     correction_map = {
                         '0': ['O', 'D'],
                         '1': ['I', 'L', 'T'],
                         '2': ['Z'],
                         '5': ['S'],
                         '8': ['B'],
                         '9': ['P', 'N'] # Added N based on YAB658NP
                         # Add Letter -> Digit if needed (e.g., 'S':['5'])
                     }
                     last_char = cleaned[-1]
                     possible_corrections = correction_map.get(last_char, [])
                     
                     if possible_corrections:
                         logger.debug(f"Attempting corrections for last char '{last_char}': {possible_corrections}")
                         base = cleaned[:-1]
                         for replacement in possible_corrections:
                             potential_plate = base + replacement
                             # --- Add detailed debug log here ---
                             logger.debug(f"Testing correction: '{potential_plate}' against regex: '{PLATE_REGEX.pattern}'")
                             # -----------------------------------
                             if PLATE_REGEX.fullmatch(potential_plate):
                                 corrected_plate = potential_plate
                                 logger.info(f"Correction successful: '{cleaned}' -> '{corrected_plate}'")
                                 break # Take the first successful correction
                     else:
                         logger.debug(f"No predefined corrections found for last char '{last_char}'.")
                 # Add more heuristics here if needed (e.g., for len 6, or different positions)
                 
                 if corrected_plate:
                      # Return corrected plate with original average confidence
                      # Optionally slightly reduce confidence: return corrected_plate, avg_conf * 0.95 
                      return corrected_plate, avg_conf 
                 else:
                      logger.warning(f"Correction failed for '{cleaned}'. Discarding.")
                      return "", 0.0 # Return empty if regex doesn't match and correction fails
            
        except Exception as e:
            logger.error(f"OCR Error: {str(e)}", exc_info=True)
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
    # Save to Database
    #---------------------------------------------------------------------------------------------
    def save_to_database(self, detections):
        """Consolidates plates and saves valid ones to the database."""
        if not detections:
            logger.info("No plates to save.")
            return
            
        logger.info(f"After consolidation, found {len(detections)} potential plates:")
        for plate, data in detections.items():
            logger.info(f"  Plate: {plate}, Confidence: {data['plate_confidence']:.2f}, " 
                      f"Vehicle: {data['vehicle_type']}, Color: {data['vehicle_color']}")
        
        # Track filtered plates for debugging
        color_filtered = [] # Restore color filtering
        confidence_filtered = []
        detection_filtered = []
        
        filtered_plates = {}
        for plate, data in detections.items():
            # Check each filter condition separately for better logging
            # Restore vehicle color check
            if data.get('vehicle_color', '').lower() == 'unknown':
                color_filtered.append(plate)
                continue
                
            if data['detection_count'] < MIN_DETECTIONS:
                detection_filtered.append(plate)
                continue
                
            # Use global MIN_CONFIDENCE
            if data['plate_confidence'] < MIN_CONFIDENCE:
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
                            data['plate_confidence'],
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
    # Process Image (Formerly Process Frame)
    #----------------------------------------------------------------------------------------------
    def process_image(self, frame): # Renamed method
        """Process a single image"""
        processed_detections = [] # Collect detections for this image
        try:
            # Get timestamp details once for the image
            time_details = self.get_time_details()

            # Detect vehicles
            vehicles = self.detect_vehicle_type(frame)
            # Log vehicle details (optional, could be removed for cleaner logs in image mode)
            # for vehicle in vehicles:
            #     logger.debug(f"Vehicle details: {vehicle}")

            # Detect license plates
            results = self.model.predict(frame, conf=MIN_CONFIDENCE, verbose=False)

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                logger.info(f"Detected {len(boxes)} potential plates in image")
                # classes = result.boxes.cls.cpu().numpy() # Class not used?
                confidences = result.boxes.conf.cpu().numpy()

                for box, conf in zip(boxes, confidences): # Removed unused cls
                    x1, y1, x2, y2 = map(int, box)

                    # Crop the detected plate region and run OCR
                    plate_img = frame[y1:y2, x1:x2]

                    if plate_img is None or plate_img.size == 0:
                        logger.warning(f"Skipping empty plate image crop at box: {(x1, y1, x2, y2)}")
                        continue

                    plate_text, plate_conf = self.ocr_license_plate(plate_img)
                    logger.info(f"OCR result: '{plate_text}' with confidence {plate_conf:.2f}")

                    if plate_text and plate_conf >= MIN_CONFIDENCE:
                        # Determine vehicle type and color
                        vehicle_type = "unknown"
                        vehicle_color = "unknown"
                        # Basic association: find the vehicle bounding box that contains the plate center
                        plate_center_x = (x1 + x2) / 2
                        plate_center_y = (y1 + y2) / 2
                        associated_vehicle = None
                        min_area = float('inf')

                        for v in vehicles:
                            vx1, vy1, vx2, vy2 = v['box']
                            # Check if plate center is inside vehicle box
                            if vx1 <= plate_center_x <= vx2 and vy1 <= plate_center_y <= vy2:
                                area = (vx2 - vx1) * (vy2 - vy1)
                                # Choose the smallest enclosing vehicle box
                                if area < min_area:
                                    min_area = area
                                    associated_vehicle = v

                        if associated_vehicle:
                             v = associated_vehicle
                             vehicle_type = v['type']
                             vx1, vy1, vx2, vy2 = v['box']
                             vehicle_roi = frame[vy1:vy2, vx1:vx2]
                             if vehicle_roi.size > 0: # Ensure ROI is valid
                                 vehicle_color = self.predict_vehicle_color(vehicle_roi)
                             else:
                                 logger.warning("Could not extract valid vehicle ROI for color prediction.")

                        # --- Save the cropped plate image ---
                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                        image_filename = f"{plate_text}_{timestamp_str}.jpg"
                        save_path = PLATE_IMAGE_DIR / image_filename
                        saved_image_path = None
                        try:
                            success = cv2.imwrite(str(save_path), plate_img)
                            if success:
                                saved_image_path = image_filename # Store relative path
                                logger.info(f"Saved detected plate image: {image_filename}")
                            else:
                                logger.warning(f"Failed to save plate image: {save_path}")
                        except Exception as img_save_error:
                            logger.error(f"Error saving plate image {save_path}: {img_save_error}", exc_info=True)
                        # -----------------------------------

                        # Collect detection details instead of updating tracker
                        detection_data = {
                            'plate_text': plate_text,
                            'plate_confidence': plate_conf,
                            'vehicle_type': vehicle_type,
                            'vehicle_color': vehicle_color,
                            'time_details': time_details, # Timestamp for the whole image processing
                            'image_filename': saved_image_path, # Path to the saved *plate* image
                            'bounding_box': (x1, y1, x2, y2) # Store box for potential drawing/filtering
                        }
                        processed_detections.append(detection_data)

                        # Draw bounding box and OCR annotation on the frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{plate_text} ({plate_conf:.2f})",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 0), 2)
        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True) # Added exc_info

        # --- Display the result in a window ---
        try:
            logger.info("Displaying processed image. Press any key in the window to continue...")
            cv2.imshow("ANPR Image Result", frame)
            # Change waitKey(0) to waitKey(5000) for 5-second display
            cv2.waitKey(5000) 
            cv2.destroyAllWindows()
            logger.info("Image display window closed.")
        except Exception as display_error:
            logger.error(f"Error displaying image: {display_error}", exc_info=True)
            # Ensure windows are destroyed even if imshow/waitKey fails
            cv2.destroyAllWindows()
        # -------------------------------------

        # Return the annotated image AND the collected detections
        return frame, processed_detections

    #----------------------------------------------------------------------------------------------
    # Process Image Source (Formerly Process Video)
    #----------------------------------------------------------------------------------------------
    def process_image_source(self, source_path):
        """Loads, processes a single image, saves output, and triggers database save."""
        try:
            img_processor = ImageProcessor(source_path)
            logger.info(f"Starting image processing for: {source_path}")

            image = img_processor.get_image()
            if image is None:
                logger.error("Failed to get image from processor.")
                return # Stop if image loading failed earlier

            # Process the single image - Updated call site
            annotated_image, detections = self.process_image(image) # Get detections back

            # Save the annotated image
            img_processor.save_image(annotated_image)

            # --- Trigger Database Save ---
            if detections:
                logger.info(f"--- Triggering Database Save for {len(detections)} detections ---")
                # Pass detections directly to save_to_database (will modify this function next)
                self.save_to_database(detections)
            else:
                logger.info("No valid plates detected in the image to save.")

        except FileNotFoundError as e:
             logger.error(f"Input image file not found: {e}")
        except ValueError as e:
             logger.error(f"Image loading error: {e}")
        except Exception as e:
             logger.error(f"An unexpected error occurred during image processing: {e}", exc_info=True)

        logger.info(f"Image processing completed for: {source_path}")

#----------------------------------------------------------------------------------------------
#                                   Main Function
#----------------------------------------------------------------------------------------------
def main():
    args = parse_arguments()
    # Validate source is not a digit (camera index) for image mode
    if args.source.isdigit():
        logger.error("Camera index detected. This script is for image processing only. Use an image file path for --source.")
        sys.exit(1) # Exit if source is a camera index

    logger.info(f"Starting ANPR image processing system with source: {args.source}")
    processor = ANPRProcessor()
    try:
        # Directly process the image source
        processor.process_image_source(args.source)
    except KeyboardInterrupt:
        logger.info("Shutting down due to KeyboardInterrupt...")
    except Exception as e:
        # Log general errors during setup or processing not caught inside process_image_source
        logger.error(f"Image processing failed: {str(e)}", exc_info=True)
    finally:
        # Ensure pool is closed even if processor initialization fails
        if hasattr(processor, 'db_pool') and processor.db_pool:
            processor.db_pool.closeall()
            logger.info("Database connections closed.")
        else:
            logger.warning("Processor or DB pool not fully initialized, skipping closeall.")

if __name__ == "__main__":
    main()
    

