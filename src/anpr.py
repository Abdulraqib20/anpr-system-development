import os
import sys
import cv2
import re
import logging
import math
import time
from pathlib import Path
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
    VIDEO_SOURCE, MODEL_PATH, OUTPUT_PATH,
    PLATE_REGEX, MIN_CONFIDENCE, TRACKING_FRAMES, MIN_DETECTIONS,
)
from sqldatabase import save_to_database

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
        output_dir = Path(OUTPUT_PATH).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source {source}")
            
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = 0
        
        # Initialize video writer
        self.writer = cv2.VideoWriter(
            OUTPUT_PATH,
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
        
        test_conn = self.db_pool.getconn()
        try:
            with test_conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO license_plates 
                    (start_time, end_time, license_plate, confidence)
                    VALUES (NOW(), NOW(), 'TESTPLATE', 0.95)
                    RETURNING id
                """)
                test_id = cur.fetchone()[0]
                logger.info(f"Database test insert successful, ID: {test_id}")
                test_conn.commit()
                
                cur.execute("DELETE FROM license_plates WHERE id = %s", (test_id,))
                test_conn.commit()
                
        except Exception as e:
            logger.critical(f"Database test failed: {str(e)}")
            raise
        finally:
            self.db_pool.putconn(test_conn)
        
        
        # Processing queue for multithreading
        self.queue = Queue(maxsize=10)
        self.running = True
        
    def preprocess_plate(self, image):
        """Preprocess image for better OCR results"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
        
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

    # def save_to_database(self, plates, start_time, end_time):
    #     """Batch save plates to database with time window"""
        
    #     logger.info("==== SAVING TO DATABASE ====")
    #     logger.info(f"Start time: {start_time}, End time: {end_time}")
    #     logger.info(f"Plates to save: {plates}")
        
    #     if not plates:
    #         logger.warning("No valid plates to save")
    #         return

    #     conn = None
    #     try:
    #         # Get a connection from the pool
    #         conn = self.db_pool.getconn()
            
    #         # Create a cursor
    #         with conn.cursor() as cursor:
    #             # Convert datetimes to ISO format strings
    #             start_iso = start_time.isoformat()
    #             end_iso = end_time.isoformat()
                
    #             # Prepare records for insertion
    #             records = []
    #             for plate, conf in plates:
    #                 records.append((start_iso, end_iso, plate, float(conf)))
                
    #             logger.info(f"Attempting to save {len(records)} records")
                
    #             # Execute the insert for each record individually for better error tracking
    #             for record in records:
    #                 try:
    #                     cursor.execute(
    #                         """INSERT INTO license_plates 
    #                         (start_time, end_time, license_plate, confidence)
    #                         VALUES (%s, %s, %s, %s)
    #                         ON CONFLICT (start_time, end_time, license_plate) 
    #                         DO UPDATE SET confidence = GREATEST(license_plates.confidence, EXCLUDED.confidence)""",
    #                         record
    #                     )
    #                     logger.info(f"Successfully inserted plate: {record[2]}")
    #                 except psycopg2.Error as e:
    #                     logger.error(f"Failed to insert plate {record[2]}: {str(e)}")
                
    #             # Commit the transaction
    #             conn.commit()
                
    #             # Verify the insert
    #             cursor.execute("""
    #                 SELECT COUNT(*) 
    #                 FROM license_plates 
    #                 WHERE end_time = %s
    #             """, (end_iso,))
                
    #             count = cursor.fetchone()[0]
    #             logger.info(f"Verified {count} records in database for this time window")
                
    #     except psycopg2.Error as e:
    #         logger.error(f"Database Error: {str(e)}")
    #         if hasattr(e, 'pgcode'):
    #             logger.error(f"Postgres Error Code: {e.pgcode}")
    #         if hasattr(e, 'pgerror'):
    #             logger.error(f"Postgres Error Message: {e.pgerror}")
    #         if conn:
    #             try:
    #                 conn.rollback()
    #                 logger.info("Transaction rolled back")
    #             except Exception as rollback_error:
    #                 logger.error(f"Error during rollback: {str(rollback_error)}")
    #     except Exception as e:
    #         logger.error(f"Unexpected error in save_to_database: {str(e)}")
    #         import traceback
    #         logger.error(traceback.format_exc())
    #     finally:
    #         # Always return the connection to the pool
    #         if conn:
    #             self.db_pool.putconn(conn)
    #             logger.info("Database connection returned to pool")
    
    
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
        
    def cleanup_tracker(self):
        """Cleanup old entries from tracker"""
        now = time.time()
        expired = [plate for plate, data in self.plate_tracker.items()
                  if now - data['last_seen'] > TRACKING_FRAMES]
                  
        for plate in expired:
            del self.plate_tracker[plate]
    
    def process_video(self):
        """Main processing loop"""
        
        with VideoProcessor(VIDEO_SOURCE) as video:
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
                        save_to_database(
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
                    save_to_database(
                        final_plates,  # List of (plate, confidence) tuples
                        window_start,  # Starting time
                        final_window_end  # Ending time
                    )
                
            logger.info("Video processing completed")


def main():
    processor = ANPRProcessor()
    try:
        processor.process_video()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
    finally:
        processor.db_pool.closeall()

if __name__ == "__main__":
    main()