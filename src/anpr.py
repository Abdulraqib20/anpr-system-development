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

import numpy as np
from ultralytics import YOLO
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

MODEL_PATH="models/license_plate_detector.pt"
PLATE_REGEX = re.compile(r'^[A-Z0-9]{7,10}$')  # Pre-compiled pattern
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ANPR")

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
        self.model = YOLO(MODEL_PATH)
        self.ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)
        self.plate_tracker = defaultdict(lambda: {'count': 0, 'confidence': 0, 'last_seen': 0})
        
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
        """Smart plate merging with length validation"""
        plate_text = plate_text.strip()
        
        # Check against existing plates
        for existing in list(self.plate_tracker.keys()):
            if self._levenshtein_distance(existing, plate_text) <= 2:  # Fixed function name
                # Prefer longer plates (reduces YAB658N vs YAB658NP issue)
                if len(plate_text) > len(existing):
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

    #----------------------------------------------------------------------------------------------
    # Save to the Database
    #----------------------------------------------------------------------------------------------
    
    def save_to_database(self, valid_plates=None):
        """Robust database saving with detailed logging"""
        plates_to_save = valid_plates or self.plate_tracker

        if not plates_to_save:
            logger.warning("No plates to save")
            return

        conn = None
        try:
            conn = self.db_pool.getconn()
            conn.autocommit = False  # Explicit transaction control

            with conn.cursor() as cursor:
                # Remove current_time and use first_seen and last_seen from tracker data
                records = [
                    (
                        datetime.fromtimestamp(data['first_seen']).isoformat(),
                        datetime.fromtimestamp(data['last_seen']).isoformat(),
                        plate,
                        data['max_confidence'],
                        data['count']
                    )
                    for plate, data in plates_to_save.items()
                    if data['count'] >= MIN_DETECTIONS
                ]

                if not records:
                    logger.warning("No records passed validation for saving")
                    return

                logger.info(f"Attempting to save {len(records)} records")

                cursor.executemany("""
                    INSERT INTO license_plates 
                    (start_time, end_time, license_plate, confidence, detection_count)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (start_time, end_time, license_plate) 
                    DO UPDATE SET
                        confidence = GREATEST(license_plates.confidence, EXCLUDED.confidence),
                        detection_count = license_plates.detection_count + EXCLUDED.detection_count
                """, records)

                conn.commit()
                logger.info(f"Successfully saved {cursor.rowcount} plates")

                # Clear only saved plates
                for plate in [r[2] for r in records]:
                    if plate in self.plate_tracker:
                        del self.plate_tracker[plate]

        except psycopg2.OperationalError as e:
            logger.critical(f"Connection failed: {str(e)}")
            self.db_pool.closeall()
            self.db_pool = SimpleConnectionPool(  # Reinitialize pool
                minconn=1,
                maxconn=10,
                **db_params
            )
        except psycopg2.Error as e:
            logger.error(f"Database error [{e.pgcode}]: {e.pgerror}")
            if conn:
                conn.rollback()
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                self.db_pool.putconn(conn)
        
    #----------------------------------------------------------------------------------------------
    # Process Frames
    #----------------------------------------------------------------------------------------------
    
    def process_frame(self, frame):
        """Robust frame processing with proper error handling"""
        try:
            if not hasattr(self, 'window_initialized'):
                cv2.namedWindow("ANPR Processing", cv2.WINDOW_NORMAL)
                self.window_initialized = True
            
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
                        
                        if not plate_text or plate_conf < MIN_CONFIDENCE:
                            logger.info(f"Plate rejected: empty text or low confidence {plate_conf}")
                            continue
                        
                        # Plate merging and tracking
                        merged_plate = self._merge_similar_plates(plate_text)
                        current_detections.add(merged_plate)
                        
                        # Update tracker with fail-safe
                        now = time.time()
                        self.plate_tracker[merged_plate] = {
                            'first_seen': self.plate_tracker.get(merged_plate, {}).get('first_seen', now),
                            'last_seen': now,
                            'max_confidence': max(
                                self.plate_tracker.get(merged_plate, {}).get('max_confidence', 0),
                                plate_conf
                            ),
                            'count': self.plate_tracker.get(merged_plate, {}).get('count', 0) + 1
                        }
                        
                        logger.info(f"Tracked plate {merged_plate} with count {self.plate_tracker[merged_plate]['count']}")
                        
                        # Visualization
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, 
                                f"{merged_plate} ({self.plate_tracker[merged_plate]['max_confidence']:.2f})",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    except Exception as e:
                        logger.error(f"Error processing box: {str(e)}")
                        continue

            # Don't delete plates from tracker here - let cleanup_tracker handle it
            
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
        """Main processing loop"""
        args = parse_arguments()
        with VideoProcessor(args.source) as video:
            logger.info("Starting video processing...")
            last_save_time = time.time()
                
            for i, frame in enumerate(video.get_frames()):
                if frame is None:
                    logger.error("Received empty frame - check video source")
                    break
                
                if i % 10 == 0:  # Log every 10 frames
                    logger.info(f"Processing frame {i}")
                processed_frame = self.process_frame(frame)
                video.write_frame(processed_frame)
                
                # Periodically save to database
                if time.time() - last_save_time > 10:
                    logger.info(f"Current tracker has {len(self.plate_tracker)} plates")
                    
                    # Remove the duration filter - just save all plates
                    self.save_to_database(self.plate_tracker)
                    last_save_time = time.time()
                
                # Cleanup plates not seen recently
                self.cleanup_tracker()
  
             
               
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
