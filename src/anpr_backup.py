#------------------------------------------------------------------------------------------------------
#                                               first anpr.py
#------------------------------------------------------------------------------------------------------

# import cv2
# from ultralytics import YOLO
# import numpy as np
# import math
# import re
# from collections import defaultdict
# from paddleocr import PaddleOCR
# from datetime import datetime
# from src.sqldatabase import create_connection, create_table, save_to_database

# # Initialize the YOLO Model
# model = YOLO(r"weights\license_plate_detector.pt")

# # Initialize the Paddle OCR
# ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)

# # Dictionary to store license plate detections
# license_plate_tracker = defaultdict(lambda: {'detections': [], 'last_seen': 0})

# def paddle_ocr(frame, x1, y1, x2, y2):
#     frame = frame[y1:y2, x1: x2]
#     result = ocr.ocr(frame, det=False, rec=True, cls=False)
#     text = ""
#     confidence = 0
#     for r in result:
#         scores = r[0][1]
#         if np.isnan(scores):
#             scores = 0
#         else:
#             scores = int(scores * 100)
#         if scores > 60:
#             text = r[0][0]
#             confidence = scores
#     pattern = re.compile('[\W]')
#     text = pattern.sub('', text)
#     text = text.replace("???", "")
#     text = text.replace("O", "0")
#     text = text.replace("粤", "")
#     return str(text), confidence

# def process_license_plate(text, confidence, frame_number):
#     if len(text) >= 4:  # Assume a valid license plate has at least 4 characters
#         license_plate_tracker[text]['detections'].append((confidence, frame_number))
#         license_plate_tracker[text]['last_seen'] = frame_number
    
#     # Remove old detections
#     for plate in list(license_plate_tracker.keys()):
#         if frame_number - license_plate_tracker[plate]['last_seen'] > 30:  # If not seen in last 30 frames
#             del license_plate_tracker[plate]

# def get_best_license_plates():
#     best_plates = []
#     for plate, data in license_plate_tracker.items():
#         if len(data['detections']) >= 5:  # Require at least 5 detections
#             avg_confidence = sum(conf for conf, _ in data['detections']) / len(data['detections'])
#             if avg_confidence > 70:  # Require average confidence above 70%
#                 best_plates.append(plate)
#     return best_plates

# def process_frame(frame, frame_number):
#     results = model.predict(frame, conf=0.45)
#     for result in results:
#         boxes = result.boxes
#         for box in boxes:
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             label, ocr_confidence = paddle_ocr(frame, x1, y1, x2, y2)
#             process_license_plate(label, ocr_confidence, frame_number)
            
#             textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
#             c2 = x1 + textSize[0], y1 - textSize[1] - 3
#             cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
#             cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)
    
#     return frame

# def process_video(video_path, output_path, db_params):
#     cap = cv2.VideoCapture(video_path)
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
#     frame_count = 0
#     start_time = datetime.now()
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         frame_count += 1
#         processed_frame = process_frame(frame, frame_count)
#         out.write(processed_frame)
        
#         if (datetime.now() - start_time).seconds >= 20:
#             end_time = datetime.now()
#             best_plates = get_best_license_plates()
#             save_to_database(best_plates, start_time, end_time)
#             start_time = datetime.now()
#             license_plate_tracker.clear()
    
#     cap.release()
#     out.release()











#------------------------------------------------------------------------------------------------------
#                                               second anpr.py
#------------------------------------------------------------------------------------------------------
import json
import cv2
from ultralytics import YOLO
import numpy as np
import math
import re
import os
import sys
from dotenv import load_dotenv
from collections import defaultdict
from datetime import datetime
from paddleocr import PaddleOCR
import psycopg2
from psycopg2 import sql

# Add the directory two levels up from the current script to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sqldatabase import create_connection, create_table
from config.appconfig import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT

load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



# Database connection parameters
db_params = {
    "host": DB_HOST,
    "database": DB_NAME,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "port": DB_PORT
}

# Create database connection and table
conn = create_connection(db_params)
if conn:
    create_table(conn)
    conn.close()

# Create a Video Capture Object
cap = cv2.VideoCapture("Resources/car_vid.mp4")

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output/annotated_video.mp4", fourcc, fps, (width, height))

# Initialize the YOLO Model
model = YOLO("models/license_plate_detector.pt")

# Initialize the frame count 
count = 0

# Class Names
className = ["License"]

# Initialize the Paddle OCR
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)

# Dictionary to store license plate detections
license_plate_tracker = defaultdict(lambda: {'detections': [], 'last_seen': 0})

def paddle_ocr(frame, x1, y1, x2, y2):
    # Improved preprocessing
    cropped = frame[y1:y2, x1:x2]
    
    if os.getenv('OCR_PREPROCESS', 'True') == 'True':
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        cropped = cv2.medianBlur(cropped, 3)
        cropped = cv2.threshold(cropped, 0, 255, 
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    result = ocr.ocr(cropped, det=False, rec=True, cls=False)
    
    # Improved text validation
    text = "".join([r[0][0] for r in result if r[0][1] > 0.6])
    text = re.sub(os.getenv('PLATE_FORMAT_REGEX', '[^A-Z0-9]'), '', text)
    
    return text, int(max([r[0][1] for r in result], default=0) * 100)


def process_license_plate(text, confidence, frame_number):
    if len(text) >= 4:  # Assume a valid license plate has at least 4 characters
        license_plate_tracker[text]['detections'].append((confidence, frame_number))
        license_plate_tracker[text]['last_seen'] = frame_number
    
    # Remove old detections
    for plate in list(license_plate_tracker.keys()):
        if frame_number - license_plate_tracker[plate]['last_seen'] > 30:  # If not seen in last 30 frames
            del license_plate_tracker[plate]

def get_best_license_plates():
    best_plates = []
    for plate, data in license_plate_tracker.items():
        if len(data['detections']) >= 5:  # Require at least 5 detections
            avg_confidence = sum(conf for conf, _ in data['detections']) / len(data['detections'])
            if avg_confidence > 70:  # Require average confidence above 70%
                best_plates.append(plate)
    return best_plates

def save_to_database(license_plates, start_time, end_time):
    conn = create_connection(db_params)
    if conn:
        cursor = conn.cursor()
        for plate in license_plates:
            try:
                cursor.execute('''
                    INSERT INTO license_plates (start_time, end_time, license_plate)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (start_time, end_time, license_plate) DO NOTHING
                ''', (start_time.isoformat(), end_time.isoformat(), plate))
            except psycopg2.Error as e:
                print(f"Error inserting data: {e}")
        conn.commit()
        print("Data saved to the database successfully.")
        conn.close()

startTime = datetime.now()
license_plates = set()

while True:
    ret, frame = cap.read()
    if ret:
        currentTime = datetime.now()
        count += 1
        print(f"Frame Number: {count}")
        results = model.predict(frame, conf=0.45)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                classNameInt = int(box.cls[0])
                clsName = className[classNameInt]
                conf = math.ceil(box.conf[0]*100)/100
                label, ocr_confidence = paddle_ocr(frame, x1, y1, x2, y2)
                process_license_plate(label, ocr_confidence, count)
                
                textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                c2 = x1 + textSize[0], y1 - textSize[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)
        
        # Write the frame to the output video
        out.write(frame)
        
        if (currentTime - startTime).seconds >= 20:
            endTime = currentTime
            best_plates = get_best_license_plates()
            save_to_database(best_plates, startTime, endTime)
            startTime = currentTime
            license_plate_tracker.clear()
        
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()












#------------------------------------------------------------------------------------------------------
#                                               third anpr.py
#------------------------------------------------------------------------------------------------------

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
    

#----------------------------------------------------------------------------------------------
# Main Function
#----------------------------------------------------------------------------------------------
    
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












#------------------------------------------------------------------------------------------------------
#                                               fourth anpr.py
#------------------------------------------------------------------------------------------------------
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

VEHICLE_MODEL_PATH = "models/yolov8n.pt" 
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

VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    8: 'boat',
}

# Define color mapping for common vehicle colors
COLOR_RANGES = {
    'black': ([0, 0, 0], [180, 255, 30]),
    'white': ([0, 0, 200], [180, 30, 255]),
    'gray': ([0, 0, 70], [180, 30, 200]),
    'red': ([0, 100, 100], [10, 255, 255]),
    'blue': ([100, 100, 100], [140, 255, 255]),
    'green': ([40, 100, 100], [80, 255, 255]),
    'yellow': ([20, 100, 100], [35, 255, 255]),
    'orange': ([10, 100, 100], [20, 255, 255]),
    'brown': ([10, 50, 50], [20, 255, 150]),
    'silver': ([0, 0, 140], [180, 30, 200]),
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
    # Detect Venicle Type
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
    def detect_vehicle_color(self, frame, vehicle_box):
        """Detect dominant color of vehicle"""
        x1, y1, x2, y2 = vehicle_box
        
        # Extract vehicle ROI
        vehicle_roi = frame[y1:y2, x1:x2]
        if vehicle_roi.size == 0:
            return "unknown"
            
        # Convert to HSV color space
        hsv_roi = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2HSV)
        
        # Create mask to ignore background
        mask = cv2.inRange(hsv_roi, np.array([0, 30, 30]), np.array([180, 255, 255]))
        
        # Find dominant color
        if np.sum(mask) > 0:
            # Calculate histogram of masked region
            hist = cv2.calcHist([hsv_roi], [0, 1], mask, [36, 50], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            max_idx = np.argmax(hist)
            h_bin = max_idx // 50
            s_bin = max_idx % 50
            
            # Map histogram bin to color
            h_value = h_bin * 5  # 180/36 = 5
            s_value = s_bin * 5.12  # 256/50 = 5.12
            
            # Match to predefined colors
            for color_name, (lower, upper) in COLOR_RANGES.items():
                if lower[0] <= h_value <= upper[0] and lower[1] <= s_value <= upper[1]:
                    return color_name
        
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

    #----------------------------------------------------------------------------------------------
    # Save to the Database
    #----------------------------------------------------------------------------------------------
    
    def save_to_database(self, valid_plates=None):
        """Modified database saving with new attributes"""
        plates_to_save = valid_plates or self.plate_tracker

        if not plates_to_save:
            logger.warning("No plates to save")
            return

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
                    for plate, data in plates_to_save.items()
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
                        vehicle_type = EXCLUDED.vehicle_type,
                        vehicle_color = EXCLUDED.vehicle_color
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
                            vehicle_color = self.detect_vehicle_color(frame, closest_vehicle['box'])
                        
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















#-------------------------------------------------------------------------------------------------------------------------
#                                               fifth anpr.py
#-------------------------------------------------------------------------------------------------------------------------
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

VEHICLE_MODEL_PATH = "models/yolov8n.pt" 
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

VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    8: 'boat',
}

# Define color mapping for common vehicle colors
COLOR_RANGES = {
    'black': ([0, 0, 0], [180, 255, 30]),
    'white': ([0, 0, 200], [180, 30, 255]),
    'gray': ([0, 0, 70], [180, 30, 200]),
    'red': ([0, 100, 100], [10, 255, 255]),
    'blue': ([100, 100, 100], [140, 255, 255]),
    'green': ([40, 100, 100], [80, 255, 255]),
    'yellow': ([20, 100, 100], [35, 255, 255]),
    'orange': ([10, 100, 100], [20, 255, 255]),
    'brown': ([10, 50, 50], [20, 255, 150]),
    'silver': ([0, 0, 140], [180, 30, 200]),
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
    # Detect Vehicle Type
    #----------------------------------------------------------------------------------------------  
    def detect_vehicle_type(self, frame, plate_box):
        """Detect vehicle type only in the area near the license plate"""
        x1, y1, x2, y2 = plate_box
        # Expand the box upward and to sides to capture the vehicle
        height = y2 - y1
        width = x2 - x1
        
        # Expand box to likely include the vehicle
        v_x1 = max(0, x1 - width)
        v_y1 = max(0, y1 - height * 3)  # Look above the plate
        v_x2 = min(frame.shape[1], x2 + width)
        v_y2 = min(frame.shape[0], y2 + height)
        
        vehicle_roi = frame[v_y1:v_y2, v_x1:v_x2]
        
        # Run detection only on this ROI
        results = self.vehicle_model.predict(vehicle_roi, conf=0.5, verbose=False)
        
        # Process results
        if results and len(results) > 0:
            # Get highest confidence class
            result = results[0]
            if result.boxes and len(result.boxes) > 0:
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                if len(confidences) > 0:
                    best_idx = np.argmax(confidences)
                    vehicle_type = self.vehicle_classes[int(classes[best_idx])]
                    return vehicle_type
        
        return "unknown"
    
    #----------------------------------------------------------------------------------------------
    # Detect Vehicle Color
    #----------------------------------------------------------------------------------------------  
    def detect_vehicle_color(self, frame, plate_box):
        """Better color detection by analyzing a region above the license plate"""
        x1, y1, x2, y2 = plate_box
        
        # Look at the area above the license plate (likely to be part of the vehicle)
        height = y2 - y1
        color_y1 = max(0, y1 - height * 2)  # Look above the plate
        color_y2 = max(0, y1 - height//2)    # Stop before the plate
        color_x1 = x1
        color_x2 = x2
        
        # Ensure we have valid coordinates
        if color_y2 <= color_y1 or color_x2 <= color_x1:
            return "unknown"
        
        # Extract vehicle ROI
        vehicle_roi = frame[color_y1:color_y2, color_x1:color_x2]
        if vehicle_roi.size == 0:
            return "unknown"
        
        # Convert to HSV and exclude very dark/bright pixels
        hsv_roi = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, (0, 40, 40), (180, 255, 255))
        
        if np.sum(mask) == 0:
            return "unknown"
        
        # Calculate histogram with more bins for better precision
        h_bins = 36
        s_bins = 32
        hist = cv2.calcHist([hsv_roi], [0, 1], mask, [h_bins, s_bins], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Find dominant bin
        max_idx = np.argmax(hist)
        h_bin = max_idx // s_bins
        s_bin = max_idx % s_bins
        
        # Calculate actual H and S values
        h_value = h_bin * (180/h_bins)
        s_value = s_bin * (256/s_bins)
        
        # More precise color mapping
        colors = {
            'black': ([0, 0, 0], [180, 30, 80]),
            'white': ([0, 0, 200], [180, 30, 255]),
            'gray': ([0, 0, 80], [180, 30, 200]),
            'red1': ([0, 70, 70], [10, 255, 255]),
            'red2': ([170, 70, 70], [180, 255, 255]),  # Wrap-around red
            'blue': ([90, 60, 60], [130, 255, 255]),
            'green': ([40, 60, 60], [80, 255, 255]),
            'yellow': ([20, 90, 90], [40, 255, 255]),
            'orange': ([10, 90, 90], [20, 255, 255]),
            'silver': ([0, 0, 140], [180, 15, 200])
        }
        
        # Test each color range
        for color_name, (lower, upper) in colors.items():
            if (lower[0] <= h_value <= upper[0] and 
                lower[1] <= s_value <= upper[1]):
                return color_name
        
        # Special handling for red (wraps around hue 0)
        if (colors['red1'][0][0] <= h_value <= colors['red1'][1][0]) or \
        (colors['red2'][0][0] <= h_value <= colors['red2'][1][0]):
            return "red"
        
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

    #----------------------------------------------------------------------------------------------
    # Save to the Database
    #----------------------------------------------------------------------------------------------
    
    def save_to_database(self, valid_plates=None):
        """Modified database saving with new attributes"""
        plates_to_save = valid_plates or self.plate_tracker

        if not plates_to_save:
            logger.warning("No plates to save")
            return

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
                        data.get('vehicle_type', 'unknown'),
                        data.get('vehicle_color', 'unknown'),
                        # data['vehicle_type'],
                        # data['vehicle_color'],
                        data['time_details']['time_of_day'],
                        data['time_details']['day_of_week'],
                        # data['time_details']['is_weekend'],
                        # data['time_details']['is_peak_hour']
                    )
                    for plate, data in plates_to_save.items()
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
                        vehicle_type = EXCLUDED.vehicle_type,
                        vehicle_color = EXCLUDED.vehicle_color
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
            
            # Use vehicle model to detect vehicles in the entire frame
            vehicle_results = self.vehicle_model.predict(frame, conf=0.5, verbose=False)
            vehicles = []
            
            # Extract vehicle detections
            if vehicle_results:
                for result in vehicle_results:
                    boxes = result.boxes
                    if boxes and len(boxes) > 0:
                        for i, box in enumerate(boxes.xyxy.cpu().numpy()):
                            vx1, vy1, vx2, vy2 = map(int, box)
                            class_id = int(boxes.cls[i].item())
                            conf = boxes.conf[i].item()
                            
                            # Filter for vehicle classes only
                            if class_id in VEHICLE_CLASSES:
                                vehicles.append({
                                    'box': (vx1, vy1, vx2, vy2),
                                    'type': VEHICLE_CLASSES[class_id],
                                    'confidence': conf
                                })
            
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
                        
                        if not plate_text or plate_conf < MIN_CONFIDENCE:
                            logger.info(f"Plate rejected: empty text or low confidence {plate_conf}")
                            continue
                        
                        # Plate merging and tracking
                        merged_plate = self._merge_similar_plates(plate_text)
                        current_detections.add(merged_plate)
                        
                        # Detect vehicle type and color
                        vehicle_type = self.detect_vehicle_type(frame, (x1, y1, x2, y2))
                        vehicle_color = self.detect_vehicle_color(frame, (x1, y1, x2, y2))
                        
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
                            vehicle_color = self.detect_vehicle_color(frame, closest_vehicle['box'])
                        
                        # Update tracker with new attributes
                        now = time.time()
                        self.plate_tracker[merged_plate] = {
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
                        
                        # tracker_entry = {
                        #     'first_seen': self.plate_tracker.get(merged_plate, {}).get('first_seen', now),
                        #     'last_seen': now,
                        #     'max_confidence': max(
                        #         self.plate_tracker.get(merged_plate, {}).get('max_confidence', 0),
                        #         plate_conf
                        #     ),
                        #     'count': self.plate_tracker.get(merged_plate, {}).get('count', 0) + 1,
                        #     'vehicle_type': vehicle_type,
                        #     'vehicle_color': vehicle_color,
                        #     'time_details': time_details
                        # }
                        
                        # self.plate_tracker[merged_plate] = tracker_entry
                        
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
