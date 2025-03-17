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
# VIDEO_SOURCE="Resources/car_vid.mp4"
# OUTPUT_PATH="output/car_vid_annotated.mp4"

PROJECT_ROOT = Path(__file__).parent.parent  # src -> project root
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VEHICLE_MODEL_PATH = "models/yolov8n.pt" 
MODEL_PATH="models/license_plate_detector.pt"
PLATE_REGEX = re.compile(r'^[A-Z0-9]{7,10}$')  # Pre-compiled pattern
TRACKING_FRAMES=30
MIN_CONFIDENCE=0.65
MIN_DETECTIONS=1

db_params = {
    "host": DB_HOST,
    "database": DB_NAME,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "port": DB_PORT
}

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
        
        # Database connection pool
        self.db_pool = SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            **db_params
        )
        
        
        # Processing queue for multithreading
        self.queue = Queue(maxsize=10)
        self.running = True
    
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
    def save_to_database(self, plates_data, start_time, end_time):
        """Save detected license plates to the database."""
        conn = None
        try:
            conn = self.db_pool.getconn()
            cursor = conn.cursor()
            for plate, confidence in plates_data:
                query = "INSERT INTO detected_plates (license_plate, confidence, start_time, end_time) VALUES (%s, %s, %s, %s);"
                cursor.execute(query, (plate, confidence, start_time, end_time))
            conn.commit()
            logger.info(f"Saved {len(plates_data)} plates to the database.")
            
        except psycopg2.Error as e:
            logger.error(f"Database error: {e}")
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
        if not hasattr(self, 'window_initialized'):
            cv2.namedWindow("ANPR Processing", cv2.WINDOW_NORMAL)
            self.window_initialized = True
        results = self.model.predict(frame, conf=MIN_CONFIDENCE, verbose=False)
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = map(int, box)
                
                # OCR processing
                plate_img = frame[y1:y2, x1:x2]
                plate_text, plate_conf = self.ocr_license_plate(plate_img)
                
                if plate_text and plate_conf >= MIN_CONFIDENCE:
                    # Update tracking information
                    self.plate_tracker[plate_text]['count'] += 1
                    self.plate_tracker[plate_text]['confidence'] = max(
                        self.plate_tracker[plate_text]['confidence'], plate_conf
                    )
                    self.plate_tracker[plate_text]['last_seen'] = time.time()
                    
                    # Draw annotations
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{plate_text} ({plate_conf:.2f})", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0), 2)
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
        """Main processing loop"""
        args = parse_arguments()
        with VideoProcessor(args.source) as video:
            logger.info("Starting video processing...")
            window_start = datetime.now()  # Initialize time window
                
            for i, frame in enumerate(video.get_frames()):
                if frame is None:
                    logger.error("Received empty frame - check video source")
                    break
                
                if i % 10 == 0:  # Log every 10 frames
                    logger.info(f"Processing frame {i}")
                processed_frame = self.process_frame(frame)
                video.write_frame(processed_frame)
                
                # Periodically save to database
                if time.time() - window_start.timestamp() > 10:
                    valid_plates = [
                        (plate, data['confidence'])
                        for plate, data in self.plate_tracker.items()
                        if data['count'] >= MIN_DETECTIONS
                    ]
                    
                    if valid_plates:
                        window_end = datetime.now()
                        self.save_to_database(
                            valid_plates,
                            window_start, 
                            window_end
                        )
                        window_start = window_end  # Reset window
                    
            # Final save
            if self.plate_tracker:
                final_plates = [
                    (plate, data['confidence']) 
                    for plate, data in self.plate_tracker.items()
                    if data['count'] >= MIN_DETECTIONS
                ]
                if final_plates:
                    final_window_end = datetime.now()
                    self.save_to_database(
                        final_plates,  # List of (plate, confidence) tuples
                        window_start,  # Starting time
                        final_window_end  # Ending time
                    )
                
            logger.info("Video processing completed")
    

    
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
    
