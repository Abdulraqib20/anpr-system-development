import os
import sys
import cv2
import re
import logging
import math
import time
from pathlib import Path
import argparse
from datetime import datetime, timedelta
from collections import defaultdict
from threading import Thread
from queue import Queue

from datetime import datetime, timedelta
import colorsys

import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from keras import layers
from paddleocr import PaddleOCR
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(str(Path(__file__).parent.parent.resolve()))
sys.path.append(os.path.abspath("src"))

from config.appconfig import (
    DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT,
)

# Configure paths
PROJECT_ROOT = Path(__file__).parent.parent  # src -> project root
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# VIDEO_SOURCE="Resources/car_vid.mp4"
# OUTPUT_PATH="output/annotated_video.mp4"

VEHICLE_MODEL_PATH = "models/yolov8m-seg.pt" 
MODEL_PATH="models/license_plate_detector.pt"
# PLATE_REGEX = re.compile(r'^[A-Z0-9]{7,10}$')  # Pre-compiled pattern
PLATE_REGEX = re.compile(r'^[A-Z0-9]{8}$')  # Strict 8-character Nigerian format
TRACKING_FRAMES=30
MIN_CONFIDENCE=0.65
PLATE_MERGE_DISTANCE=2
MIN_TRACKING_DURATION=5
MIN_DETECTIONS=1

# Generate unique output filename with timestamp
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

db_params = {
    "host": DB_HOST,
    "database": DB_NAME,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "port": DB_PORT
}

VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    8: 'boat',
}

# Color class configuration (Add this near VEHICLE_CLASSES)
COLOR_CLASSES = [
    'beige', 'black', 'blue', 'brown', 'gold', 'green', 'grey',
    'orange', 'pink', 'purple', 'red', 'silver', 'tan', 'white', 'yellow'
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ANPR")

class FixedDepthwiseConv2D(layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

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
        
        # output_dir = Path(OUTPUT_PATH).parent
        # output_dir.mkdir(parents=True, exist_ok=True)
        # self.cap = cv2.VideoCapture(source)
        # if not self.cap.isOpened():
        #     raise ValueError(f"Could not open video source {source}")
            
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = 0
        
        # Initialize video writer
        self.writer = cv2.VideoWriter(
            str(self.output_path),  # Use generated path
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (self.width, self.height)
        )

        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        
    def release(self):
        """Release resources"""
        self.cap.release()
        self.writer.release()
        cv2.destroyAllWindows()
        
    def get_frames(self):
        """Generator that yields frames"""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_count += 1
            yield frame
            
    def write_frame(self, frame):
        """Write processed frame to output"""
        self.writer.write(frame)

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
        self.plate_tracker = defaultdict(lambda: {
            'count': 0, 
            'confidence': 0, 
            'last_seen': 0,
            'vehicle_type': None,
            'vehicle_color': None,
            'time_details': {},
        })
        
        self.color_model = tf.keras.models.load_model(
            "models/EFN-model.best.h5",
            custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D}
        )
        
        self.plate_history = {}  # Track recently seen plates
        self.cooldown_period = 30  # Seconds before a plate can be re-detected
        
        self.db_pool = SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            connect_timeout=5  # Add connection timeout
        )
        
        # Processing queue for multithreading
        self.queue = Queue(maxsize=10)
        self.running = True
        # self._verify_db_schema()  # Add schema verification
    
    #----------------------------------------------------------------------------------------------
    # Merge similar plates with Levenshtein distance
    #----------------------------------------------------------------------------------------------  
    def _merge_similar_plates(self, plate_text):
        """Enhanced plate merging with length validation and OCR error handling"""
        plate_text = re.sub(r'[^A-Z0-9]', '', plate_text.strip())
        
        if not plate_text:
            return ""
            
        # Common OCR error substitutions
        ocr_replacements = {
            '8': 'B',
            '5': 'S',
            '0': 'O',
            '1': 'I',
            '2': 'Z',
            '6': 'G'
        }
        
        # Generate normalized version for comparison
        normalized = ''.join([ocr_replacements.get(c, c) for c in plate_text])
        
        # Check against existing plates
        for existing in list(self.plate_tracker.keys()):
            # Length must match exactly
            if len(existing) != len(plate_text):
                continue
                
            # Generate normalized existing plate
            existing_normalized = ''.join([ocr_replacements.get(c, c) for c in existing])
            
            # First check exact match
            if existing_normalized == normalized:
                return existing
                
            # Then check Levenshtein distance (tighter threshold)
            distance = self._levenshtein_distance(existing_normalized, normalized)
            max_allowed = 1 if len(plate_text) > 6 else 0  # Allow 1 error for longer plates
            
            if distance <= max_allowed:
                # Prefer plate with more letters (reduces 0 vs O conflicts)
                letter_count = lambda s: sum(c.isalpha() for c in s)
                if letter_count(plate_text) > letter_count(existing):
                    self.plate_tracker[plate_text] = self.plate_tracker.pop(existing)
                    return plate_text
                return existing
                
        return plate_text

    def _levenshtein_distance(self, s1, s2):
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
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
        
        return detected_vehicles
    
    #----------------------------------------------------------------------------------------------
    # Detect Vehicle Color
    #----------------------------------------------------------------------------------------------  
    
    def predict_vehicle_color(self, cropped_image):
        """Predict vehicle color using trained model"""
        try:
            if cropped_image.size == 0:
                return "unknown"
                
            # Preprocess image for color model
            img = cv2.resize(cropped_image, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Make prediction
            predictions = self.color_model.predict(img_array, verbose=0)[0]
            top_idx = np.argmax(predictions)
            
            # Only return if confidence meets threshold
            if predictions[top_idx] > 0.5:
                return COLOR_CLASSES[top_idx]
            return "unknown"
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
            
            # Nigerian plate length validation
            if len(cleaned) != 8:
                logger.info(f"Rejected plate {cleaned} - invalid length {len(cleaned)}")
                return "", 0.0
            
            if PLATE_REGEX.fullmatch(cleaned):
                avg_conf = sum(confidences) / len(confidences)
                return cleaned, avg_conf
            return "", 0.0
            
        except Exception as e:
            logger.error(f"OCR Error: {str(e)}")
            return "", 0.0

    #----------------------------------------------------------------------------------------------
    # Save to the Database
    #----------------------------------------------------------------------------------------------
    
    def save_to_database(self, plates_to_save=None):
        """Modified database saving with new attributes"""
        plates_to_save = plates_to_save or self.plate_tracker
        
        # # Strict color filtering
        # filtered_plates = {
        #     plate: data for plate, data in plates_to_save.items()
        #     if data.get('vehicle_color', '').lower() != 'unknown'
        # }
        
        # Nigerian plate validation filters
        filtered_plates = {
            plate: data for plate, data in plates_to_save.items()
            if (
                len(plate) == 8 and  # Nigerian plate length requirement
                data.get('vehicle_color', '').lower() != 'unknown' and
                re.match(r'^[A-Z0-9]{8}$', plate)  # Final regex check
            )
        }
        
        if not filtered_plates:
            logger.info("All plates filtered out due to unknown color")
            return

        # if not plates_to_save:
        #     logger.warning("No plates to save")
        #     return

        conn = None
        try:
            conn = self.db_pool.getconn()
            conn.autocommit = False  # Explicit transaction control

            with conn.cursor() as cursor:
                records = [
                    (
                        datetime.fromtimestamp(data['first_seen']).isoformat(),
                        datetime.fromtimestamp(data['last_seen']).isoformat(),
                        plate,
                        data['max_confidence'],
                        data['count'],
                        data['vehicle_type'],
                        data['vehicle_color'],
                        data['time_details']['time_of_day'],
                        data['time_details']['day_of_week'],
                        # data['time_details']['is_weekend'],
                        # data['time_details']['is_peak_hour']
                    )
                    for plate, data in filtered_plates.items()
                    if data['count'] >= MIN_DETECTIONS
                ]

                if not records:
                    logger.warning("No records passed validation for saving")
                    return

                logger.info(f"Attempting to save {len(records)} records")
                
                cursor.executemany("""
                    INSERT INTO license_plates2
                    (start_time, end_time, license_plate, confidence, detection_count, 
                    vehicle_type, vehicle_color, time_of_day, day_of_week)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (start_time, end_time, license_plate)
                    DO UPDATE SET
                        confidence = GREATEST(license_plates2.confidence, EXCLUDED.confidence),
                        detection_count = license_plates2.detection_count + EXCLUDED.detection_count,
                        vehicle_type = COALESCE(EXCLUDED.vehicle_type, license_plates2.vehicle_type),
                        vehicle_color = COALESCE(EXCLUDED.vehicle_color, license_plates2.vehicle_color)
                """, records)

                conn.commit()
                logger.info(f"Successfully saved {cursor.rowcount} plates")

                # Clear only saved plates
                for plate in [r[2] for r in records]:
                    if plate in self.plate_tracker:
                        del self.plate_tracker[plate]

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
        """Modified process_frame method to include new attributes"""
        try:
            if not hasattr(self, 'window_initialized'):
                cv2.namedWindow("ANPR Processing", cv2.WINDOW_NORMAL)
                self.window_initialized = True
            
            # Get timestamp details once per frame
            time_details = self.get_time_details()
            
            # Detect vehicles first
            vehicles = self.detect_vehicle_type(frame)
            
            # Original license plate detection
            results = self.model.predict(frame, conf=MIN_CONFIDENCE, verbose=False)
            
            current_detections = set()
            
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                logger.info(f"Detected {len(boxes)} potential plates in frame")
                
                for box in boxes:
                    try:
                        x1, y1, x2, y2 = map(int, box)
                        plate_img = frame[y1:y2, x1:x2]
                        
                        # OCR Processing with validation
                        plate_text, plate_conf = self.ocr_license_plate(plate_img)
                        logger.info(f"OCR result: '{plate_text}' with confidence {plate_conf}")
                        
                        now = time.time()
                        if plate_text and (plate_text in self.plate_history):
                            last_seen = self.plate_history[plate_text]
                            if now - last_seen < self.cooldown_period:
                                logger.info(f"Skipping plate {plate_text} - in cooldown period")
                                continue
                        self.plate_history[plate_text] = now
                        
                        if not plate_text or plate_conf < MIN_CONFIDENCE:
                            logger.info(f"Plate rejected: empty text or low confidence {plate_conf}")
                            continue
                        
                        # Plate merging and tracking
                        merged_plate = self._merge_similar_plates(plate_text)
                        current_detections.add(merged_plate)
                        
                        # Find closest vehicle to this plate
                        closest_vehicle = None
                        min_distance = float('inf')
                        
                        for vehicle in vehicles:
                            vx1, vy1, vx2, vy2 = vehicle['box']
                            # Calculate center points
                            plate_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            vehicle_center = ((vx1 + vx2) // 2, (vy1 + vy2) // 2)
                            
                            # Simple Euclidean distance
                            distance = math.sqrt((plate_center[0] - vehicle_center[0])**2 + 
                                                (plate_center[1] - vehicle_center[1])**2)
                            
                            if distance < min_distance:
                                min_distance = distance
                                closest_vehicle = vehicle
                        
                        # Get vehicle color if we found a vehicle
                        vehicle_type = "unknown"
                        vehicle_color = "unknown"
                        
                        if closest_vehicle and min_distance < 300:  # Threshold for matching
                            vehicle_type = closest_vehicle['type']
                            vx1, vy1, vx2, vy2 = closest_vehicle['box']
                            vehicle_roi = frame[vy1:vy2, vx1:vx2]
                            vehicle_color = self.predict_vehicle_color(vehicle_roi)
                        
                        # =================================================================
                        # Add this color validation check RIGHT HERE
                        # =================================================================
                        if vehicle_color.lower() == "unknown":
                            logger.info(f"Skipping plate {merged_plate} - unknown color")
                            continue  # This skips the entire plate processing
                        
                        # Update tracker with new attributes
                        now = time.time()
                        tracker_entry = {
                            'first_seen': self.plate_tracker.get(merged_plate, {}).get('first_seen', now),
                            'last_seen': now,
                            'max_confidence': max(
                                self.plate_tracker.get(merged_plate, {}).get('max_confidence', 0),
                                plate_conf
                            ),
                            'count': self.plate_tracker.get(merged_plate, {}).get('count', 0) + 1,
                            'vehicle_type': vehicle_type,
                            'vehicle_color': vehicle_color,
                            'time_details': time_details
                        }
                        
                        self.plate_tracker[merged_plate] = tracker_entry
                        
                        logger.info(f"Tracked plate {merged_plate} with count {self.plate_tracker[merged_plate]['count']}")
                        
                        # Enhanced visualization
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, 
                                f"{merged_plate} ({self.plate_tracker[merged_plate]['max_confidence']:.2f})",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Add vehicle type and color to visualization
                        cv2.putText(frame, 
                                f"{vehicle_color} {vehicle_type}",
                                (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    except Exception as e:
                        logger.error(f"Error processing box: {str(e)}")
                        continue
            
            # Display vehicles with bounding boxes
            for vehicle in vehicles:
                vx1, vy1, vx2, vy2 = vehicle['box']
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 0, 255), 2)
                cv2.putText(frame, 
                        f"{vehicle['type']} ({vehicle['confidence']:.2f})",
                        (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("ANPR Processing", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                self.running = False
                
            return frame
        
        except Exception as e:
            logger.critical(f"Frame processing failed: {str(e)}")
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
                
        if expired:
            logger.info(f"Cleaning up {len(expired)} expired plates")
            
        for plate in expired:
            # Only remove if it's been detected enough times and save it
            if self.plate_tracker[plate].get('count', 0) >= MIN_DETECTIONS:
                self.save_to_database({plate: self.plate_tracker[plate]})
            del self.plate_tracker[plate]
    
    #----------------------------------------------------------------------------------------------
    # Process Video
    #----------------------------------------------------------------------------------------------
    
    def process_video(self):
        """Main processing loop with optimized runtime controls"""
        args = parse_arguments()
        with VideoProcessor(args.source) as video:
            logger.info("Starting video processing...")
            
            # Initialize timing and control variables for our own car
            start_time = time.time()
            last_save_time = time.time()
            frame_skip = 3  # Process every nth frame
            time_limit = 40  # Seconds to process video
            max_detections_per_plate = 2  # Maximum times to detect each plate
            processed_plates = set()  # Track fully processed plates
                
            for i, frame in enumerate(video.get_frames()):
                # Check for early termination conditions
                if frame is None:
                    logger.error("Received empty frame - check video source")
                    break
                    
                # Check if we've reached the time limit
                elapsed_time = time.time() - start_time
                if elapsed_time > time_limit:
                    logger.info(f"Reached time limit of {time_limit} seconds")
                    # Ensure final save before exiting
                    self.save_to_database(self.plate_tracker)
                    break
                    
                # Skip frames to reduce processing load
                if i % frame_skip != 0:
                    continue
                    
                # Progress logging
                if i % 10 == 0:  # Log every 10 frames
                    logger.info(f"Processing frame {i} (elapsed time: {elapsed_time:.2f}s)")
                    
                # Process the current frame
                processed_frame = self.process_frame(frame)
                video.write_frame(processed_frame)
                
                # Check for plates that reached detection threshold
                for plate, data in list(self.plate_tracker.items()):
                    if data['count'] >= max_detections_per_plate and plate not in processed_plates:
                        logger.info(f"Plate {plate} reached detection threshold with {data['count']} detections")
                        # Save this plate immediately
                        self.save_to_database({plate: data})
                        processed_plates.add(plate)
                        # Option: remove from tracker to stop further processing
                        # del self.plate_tracker[plate]
                
                # Periodically save to database (for plates not yet at threshold)
                if time.time() - last_save_time > 10:
                    logger.info(f"Current tracker has {len(self.plate_tracker)} plates")
                    
                    # Filter out plates we've already fully processed
                    plates_to_save = {plate: data for plate, data in self.plate_tracker.items() 
                                    if plate not in processed_plates}
                    
                    if plates_to_save:
                        self.save_to_database(plates_to_save)
                    last_save_time = time.time()
                
                # Cleanup plates not seen recently
                self.cleanup_tracker()
                
                # Check for user quit
                if cv2.waitKey(1) == ord('q'):
                    logger.info("User requested exit")
                    break
            
            # Final save to ensure we don't miss anything
            logger.info("Video processing complete, saving final results")
            self.save_to_database(self.plate_tracker)
             
               
#----------------------------------------------------------------------------------------------
#                                   Main Function
#----------------------------------------------------------------------------------------------
def main():
    args = parse_arguments()
    processor = ANPRProcessor()
    try:
        with VideoProcessor(args.source) as video:
            processor.process_video()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Video processing failed: {str(e)}")
    finally:
        processor.db_pool.closeall()

if __name__ == "__main__":
    main()

#-----------------------------------------
# Run the script with the following command
#-----------------------------------------
# custom video file -- python src/anpr.py --source "Resources/car_vid.mp4"
# webcam input -- python src/anpr.py --source 0
