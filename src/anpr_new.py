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

logger = logging.getLogger("ANPR")
logger.setLevel(logging.DEBUG)
if logger.hasHandlers():
    logger.handlers.clear()
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

#--------------------------------------------------------------------------------------
#  Configuration Variables
#--------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent  # src -> project root
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# VEHICLE_MODEL_PATH = "models/yolov8m-seg.pt"
VEHICLE_MODEL_PATH = "models/yolov8n.pt"
VEHICLE_COLOR_MODEL_PATH = "models/EFN-model.best.h5"
MODEL_PATH="models/license_plate_detector.pt"

PLATE_REGEX = re.compile(r'^[A-Z0-9]{7,8}$')  # Strict 7-8 character format
# PLATE_REGEX = re.compile(r'^[A-Z0-9]{8}$')  # Strict 8-character Nigerian format
TRACKING_FRAMES=30
MIN_CONFIDENCE=0.40
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

# Color class configuration (Add this near VEHICLE_CLASSES)
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
        self.plate_tracker = defaultdict(lambda: {
            'count': 0, 
            'confidence': 0, 
            'last_seen': 0,
            'vehicle_type': None,
            'vehicle_color': None,
            'time_details': {},
        })
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
            connect_timeout=5  # Add connection timeout
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
    # Save to Database
    #---------------------------------------------------------------------------------------------    
    def save_to_database(self, plates_to_save=None):
        """Save to database with session-based duplicate prevention"""
        plates_to_save = plates_to_save or self.plate_tracker
        
        # Strict plate validation filters
        filtered_plates = {
            plate: data for plate, data in plates_to_save.items()
            if (
                len(plate) in (7, 8) and  # Strict length check
                data.get('vehicle_color', '').lower() != 'unknown' and
                data['count'] >= MIN_DETECTIONS and
                data['max_confidence'] >= 0.50 and
                re.match(r'^[A-Z0-9]{7,8}$', plate)
            )
        }
        
        if not filtered_plates:
            logger.info("All plates filtered out due to validation")
            return

        conn = None
        try:
            conn = self.db_pool.getconn()
            conn.autocommit = False  # Explicit transaction control

            with conn.cursor() as cursor:
                # Create records for all filtered plates
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
                    )
                    for plate, data in filtered_plates.items()
                ]

                if records:
                    # Standard insert - allows same plate across different sessions
                    cursor.executemany("""
                        INSERT INTO detected_plates
                        (start_time, end_time, license_plate, confidence, detection_count, 
                        vehicle_type, vehicle_color, time_of_day, day_of_week)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (start_time, end_time, license_plate)
                        DO UPDATE SET
                            confidence = GREATEST(detected_plates.confidence, EXCLUDED.confidence),
                            detection_count = detected_plates.detection_count + EXCLUDED.detection_count,
                            vehicle_type = COALESCE(EXCLUDED.vehicle_type, detected_plates.vehicle_type),
                            vehicle_color = COALESCE(EXCLUDED.vehicle_color, detected_plates.vehicle_color)
                    """, records)

                    conn.commit()
                    logger.info(f"Successfully saved {cursor.rowcount} plates")

                # Clear saved plates from tracker
                for plate in filtered_plates:
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
                        
                        # Update tracker with new attributes using your approach
                        now = time.time()
                        tracker_entry = {
                            'first_seen': self.plate_tracker.get(plate_text, {}).get('first_seen', now),
                            'last_seen': now,
                            'max_confidence': max(
                                self.plate_tracker.get(plate_text, {}).get('max_confidence', 0),
                                plate_conf
                            ),
                            'count': self.plate_tracker.get(plate_text, {}).get('count', 0) + 1,
                            'vehicle_type': vehicle_type,
                            'vehicle_color': vehicle_color,
                            'time_details': time_details
                        }
                        
                        self.plate_tracker[plate_text] = tracker_entry
                        
                        logger.info(f"Tracked plate {plate_text} with count {self.plate_tracker[plate_text]['count']}")
                        
                        # Draw bounding box and OCR annotation on the frame
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
            session_plates = set()
            
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
                    before_plates = set(self.plate_tracker.keys())
                    processed_frame = self.process_frame(frame)
                    after_plates = set(self.plate_tracker.keys())
                    
                    # Check for new plates that were detected in this frame
                    new_plates = after_plates - before_plates
                    
                    # Remove any plates we've already seen in this session
                    # (This prevents duplicates within the same run)
                    for plate in list(new_plates):
                        if plate in session_plates:
                            if plate in self.plate_tracker:
                                logger.info(f"Removing duplicate plate from tracker: {plate}")
                                del self.plate_tracker[plate]
                        else:
                            # Add to our session tracking
                            if len(plate) in (7, 8) and re.match(r'^[A-Z0-9]{7,8}$', plate):
                                session_plates.add(plate)
                    
                    last_processed_frame = frame.copy()
                    processed_count += 1
                    frames_since_last_processed = 0
                else:
                    # Just copy the frame without processing
                    processed_frame = frame
                    skip_count += 1
                
                # Write frame to output
                video.write_frame(processed_frame)
                
                # Save to database every 15 seconds
                elapsed = (datetime.now() - window_start).total_seconds()
                if elapsed > 15:
                    if self.plate_tracker:
                        logger.info(f"Periodic saving to database. Current tracker size: {len(self.plate_tracker)}")
                        self.save_to_database()
                    window_start = datetime.now()
            
            # Final database save
            if self.plate_tracker:
                logger.info("Final database save.")
                self.save_to_database()
                    
            logger.info(f"Video processing completed. Processed: {processed_count}, skipped: {skip_count}")
            logger.info(f"Total unique plates detected in this session: {len(session_plates)}")
  
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
    
