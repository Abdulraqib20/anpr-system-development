#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------
#                                               FIRST anpr.py
#-------------------------------------------------------------------------------------------------------------------------


import cv2
from ultralytics import YOLO
import numpy as np
import math
import re
from collections import defaultdict
from paddleocr import PaddleOCR
from datetime import datetime
from src.sqldatabase import create_connection, create_table, save_to_database

# Initialize the YOLO Model
model = YOLO(r"weights\license_plate_detector.pt")

# Initialize the Paddle OCR
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)

# Dictionary to store license plate detections
license_plate_tracker = defaultdict(lambda: {'detections': [], 'last_seen': 0})

def paddle_ocr(frame, x1, y1, x2, y2):
    frame = frame[y1:y2, x1: x2]
    result = ocr.ocr(frame, det=False, rec=True, cls=False)
    text = ""
    confidence = 0
    for r in result:
        scores = r[0][1]
        if np.isnan(scores):
            scores = 0
        else:
            scores = int(scores * 100)
        if scores > 60:
            text = r[0][0]
            confidence = scores
    pattern = re.compile('[\W]')
    text = pattern.sub('', text)
    text = text.replace("???", "")
    text = text.replace("O", "0")
    text = text.replace("粤", "")
    return str(text), confidence

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

def process_frame(frame, frame_number):
    results = model.predict(frame, conf=0.45)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label, ocr_confidence = paddle_ocr(frame, x1, y1, x2, y2)
            process_license_plate(label, ocr_confidence, frame_number)

            textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
            c2 = x1 + textSize[0], y1 - textSize[1] - 3
            cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)

    return frame

def process_video(video_path, output_path, db_params):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    start_time = datetime.now()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        processed_frame = process_frame(frame, frame_count)
        out.write(processed_frame)

        if (datetime.now() - start_time).seconds >= 20:
            end_time = datetime.now()
            best_plates = get_best_license_plates()
            save_to_database(best_plates, start_time, end_time)
            start_time = datetime.now()
            license_plate_tracker.clear()

    cap.release()
    out.release()
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------
#                                               SECOND anpr.py
#-------------------------------------------------------------------------------------------------------------------------

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
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------
#                                               THIRD anpr.py
#-------------------------------------------------------------------------------------------------------------------------
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
from miscellaneous.scripts.sqldatabase import save_to_database

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
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------
#                                               FOURTH anpr.py
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

#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------
#                                               FIFTH anpr.py
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

#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------
#                                               SIXTH anpr.py
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
import tensorflow as tf
from keras import layers
from paddleocr import PaddleOCR
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv
import torch

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

VEHICLE_MODEL_PATH = "models/yolov8m-seg.pt"
MODEL_PATH="models/license_plate_detector.pt"
# PLATE_REGEX = re.compile(r'^[A-Z0-9]{7,10}$')  # Pre-compiled pattern
PLATE_REGEX = re.compile(r'^[A-Z0-9]{8}$')  # Strict 8-character Nigerian format
TRACKING_FRAMES=30
MIN_CONFIDENCE=0.65
# PLATE_MERGE_DISTANCE=2
# MIN_TRACKING_DURATION=5
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

    #---------------------------------------------------------------------------------------------
    # Enhanced helper functions
    #----------------------------------------------------------------------------------------------
    def preprocess_frame(self, frame):
        """Enhance frame quality before detection"""
        # Normalize brightness and contrast
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Add slight sharpening for edge enhancement
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        return sharpened

    def adapt_confidence_threshold(self, frame):
        """Dynamically adjust confidence threshold based on video quality"""
        # Measure image quality
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Adapt threshold based on image quality
        if fm < 100:  # Very blurry
            return 0.45  # Much lower threshold
        elif fm < 300:  # Somewhat blurry
            return 0.55  # Lower threshold
        elif fm > 1000:  # Very sharp
            return 0.75  # Higher threshold for good quality
        else:
            return 0.65  # Default threshold

    #----------------------------------------------------------------------------------------------

    #----------------------------------------------------------------------------------------------
    # Process Frames
    #----------------------------------------------------------------------------------------------
    def process_frame(self, frame):
        """Enhanced frame processing with multi-scale detection and improved OCR"""
        try:
            if not hasattr(self, 'window_initialized'):
                cv2.namedWindow("ANPR Processing", cv2.WINDOW_NORMAL)
                self.window_initialized = True

            # Get timestamp details once per frame
            time_details = self.get_time_details()

            # Apply frame preprocessing for enhanced quality
            enhanced_frame = self.preprocess_frame(frame)

            # Dynamic confidence thresholds based on frame quality
            current_confidence = self.adapt_confidence_threshold(frame)

            # Detect vehicles first using enhanced frame
            vehicles = self.detect_vehicle_type(enhanced_frame)

            # Multi-scale license plate detection
            scales = [1.0]
            results = self.model.predict(enhanced_frame, conf=current_confidence, verbose=False)

            # If no plates detected at default scale, try additional scales
            if len(results[0].boxes) == 0:
                scales = [0.8, 1.2]
                multi_results = []

                for scale in scales:
                    h, w = enhanced_frame.shape[:2]
                    resized = cv2.resize(enhanced_frame, (int(w*scale), int(h*scale)))
                    scale_results = self.model.predict(resized, conf=current_confidence*0.9, verbose=False)

                    # If we found something at this scale
                    if len(scale_results[0].boxes) > 0:
                        # Adjust coordinates back to original scale
                        adjusted_boxes = []
                        for box in scale_results[0].boxes.xyxy.cpu().numpy():
                            adjusted_box = box / scale
                            adjusted_boxes.append(adjusted_box)

                        # Replace empty results with the rescaled ones
                        if adjusted_boxes:
                            logger.info(f"Found {len(adjusted_boxes)} plates at scale {scale}")
                            results = scale_results
                            # Update the boxes with adjusted coordinates
                            results[0].boxes.xyxy = torch.tensor(adjusted_boxes).to(results[0].boxes.xyxy.device)
                            break

            current_detections = set()

            # Process all detected plates
            if results and len(results) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy() if hasattr(results[0].boxes, 'conf') else [current_confidence] * len(boxes)

                logger.info(f"Detected {len(boxes)} potential plates in frame")

                for box, conf in zip(boxes, confs):
                    try:
                        x1, y1, x2, y2 = map(int, box)

                        # Sanity check on box dimensions
                        if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                            logger.warning(f"Invalid box coordinates: {box}")
                            continue

                        # Ensure minimum plate size (too small might be false positive)
                        if (x2 - x1) < 20 or (y2 - y1) < 10:
                            logger.info(f"Plate too small: {x2-x1}x{y2-y1}")
                            continue

                        # Extract plate image with a slight margin for better OCR
                        margin_x = int((x2 - x1) * 0.05)
                        margin_y = int((y2 - y1) * 0.05)

                        # Apply margins but ensure within image bounds
                        plate_x1 = max(0, x1 - margin_x)
                        plate_y1 = max(0, y1 - margin_y)
                        plate_x2 = min(frame.shape[1], x2 + margin_x)
                        plate_y2 = min(frame.shape[0], y2 + margin_y)

                        plate_img = frame[plate_y1:plate_y2, plate_x1:plate_x2]

                        # Skip if plate image is invalid
                        if plate_img.size == 0:
                            continue

                        # Try multiple OCR approaches with validation
                        plate_texts = []
                        plate_confs = []

                        # Basic preprocessing
                        processed_plate = self.preprocess_plate(plate_img)
                        result = self.ocr.ocr(processed_plate, det=False, rec=True, cls=False)

                        if result and len(result) > 0:
                            for res in result:
                                if res and res[0]:
                                    text, confidence = res[0]
                                    if text:
                                        cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                                        if len(cleaned_text) >= 7 and len(cleaned_text) <= 9:  # Nigerian plates with slight variance
                                            plate_texts.append(cleaned_text)
                                            plate_confs.append(confidence)

                        # If we still don't have reliable results, try with inverted image
                        if not plate_texts or max(plate_confs) < 0.5:
                            inverted = cv2.bitwise_not(processed_plate)
                            result = self.ocr.ocr(inverted, det=False, rec=True, cls=False)

                            if result and len(result) > 0:
                                for res in result:
                                    if res and res[0]:
                                        text, confidence = res[0]
                                        if text:
                                            cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                                            if len(cleaned_text) >= 7 and len(cleaned_text) <= 9:
                                                plate_texts.append(cleaned_text)
                                                plate_confs.append(confidence)

                        # Select best result
                        if plate_texts and plate_confs:
                            best_idx = np.argmax(plate_confs)
                            plate_text = plate_texts[best_idx]
                            plate_conf = plate_confs[best_idx]
                        else:
                            logger.info("No valid plate text found after OCR")
                            continue

                        # Apply cooldown period logic
                        now = time.time()
                        if plate_text and (plate_text in self.plate_history):
                            last_seen = self.plate_history[plate_text]
                            if now - last_seen < self.cooldown_period:
                                logger.info(f"Skipping plate {plate_text} - in cooldown period")
                                continue
                        self.plate_history[plate_text] = now

                        # Apply confidence threshold
                        if not plate_text or plate_conf < current_confidence:
                            logger.info(f"Plate rejected: empty text or low confidence {plate_conf}")
                            continue

                        # Merge similar plates
                        merged_plate = self._merge_similar_plates(plate_text)
                        current_detections.add(merged_plate)

                        # Enhanced vehicle-plate association
                        closest_vehicle = None
                        min_distance = float('inf')
                        plate_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                        # Prioritize vehicles where the plate is near the bottom
                        for vehicle in vehicles:
                            vx1, vy1, vx2, vy2 = vehicle['box']

                            # First try bottom half matching (plates are typically at bottom)
                            if (x1 >= vx1 - 20 and x2 <= vx2 + 20 and
                                y1 >= (vy1 + vy2)/2 - 20 and y2 <= vy2 + 40):
                                closest_vehicle = vehicle
                                break

                            # Fallback to distance-based matching
                            vehicle_center = ((vx1 + vx2) // 2, (vy1 + vy2) // 2)
                            distance = math.sqrt((plate_center[0] - vehicle_center[0])**2 +
                                                (plate_center[1] - vehicle_center[1])**2)

                            if distance < min_distance:
                                min_distance = distance
                                closest_vehicle = vehicle

                        # Get vehicle color and type
                        vehicle_type = "unknown"
                        vehicle_color = "unknown"

                        # More robust vehicle association with relaxed distance threshold
                        if closest_vehicle and min_distance < 400:  # Increased from 300
                            vehicle_type = closest_vehicle['type']
                            vx1, vy1, vx2, vy2 = closest_vehicle['box']

                            # Get color from upper part of vehicle (avoid plate area)
                            color_roi_height = int((vy2 - vy1) * 0.4)  # Top 40% of vehicle
                            vehicle_roi = frame[vy1:vy1+color_roi_height, vx1:vx2]
                            vehicle_color = self.predict_vehicle_color(vehicle_roi)

                        # If color detection failed, try with whole vehicle
                        if vehicle_color.lower() == "unknown" and closest_vehicle:
                            vx1, vy1, vx2, vy2 = closest_vehicle['box']
                            vehicle_roi = frame[vy1:vy2, vx1:vx2]
                            vehicle_color = self.predict_vehicle_color(vehicle_roi)

                        # Second fallback - accept unknown color but note it
                        if vehicle_color.lower() == "unknown":
                            logger.info(f"Unable to determine color for plate {merged_plate}")
                            # We'll continue processing rather than skipping

                        # Update tracker with refined information
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
                        color = (0, 255, 0)  # Default green

                        # Change color based on confidence
                        if plate_conf < 0.6:
                            color = (0, 165, 255)  # Orange for lower confidence
                        elif plate_conf > 0.8:
                            color = (0, 255, 127)  # Light green for high confidence

                        # Draw plate box with thickness based on confidence
                        thickness = 1 + int(plate_conf * 3)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                        # Add plate text with confidence
                        cv2.putText(frame,
                                f"{merged_plate} ({plate_conf:.2f})",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        # Add vehicle type and color with detection count
                        cv2.putText(frame,
                                f"{vehicle_color} {vehicle_type} [{tracker_entry['count']}]",
                                (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                    except Exception as e:
                        logger.error(f"Error processing plate box: {str(e)}")
                        continue

                    except Exception as e:
                        logger.error(f"Error processing plate box: {str(e)}")
                        continue

            # Display vehicles with bounding boxes
            for vehicle in vehicles:
                vx1, vy1, vx2, vy2 = vehicle['box']

                # Draw vehicle boxes with different color based on type
                color_map = {
                    'car': (0, 0, 255),      # Red for cars
                    'truck': (255, 0, 0),    # Blue for trucks
                    'bus': (255, 0, 255),    # Purple for buses
                    'motorcycle': (0, 255, 255)  # Yellow for motorcycles
                }

                box_color = color_map.get(vehicle['type'], (0, 0, 255))
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), box_color, 2)

                # Add vehicle type and confidence
                label = f"{vehicle['type']} ({vehicle['confidence']:.2f})"
                cv2.putText(frame, label, (vx1, vy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

            # Add frame stats
            frame_stats = f"Quality: {self.adapt_confidence_threshold(frame):.2f}, Vehicles: {len(vehicles)}, Plates: {len(current_detections)}"
            cv2.putText(frame, frame_stats, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("ANPR Processing", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                self.running = False

            return frame

        except Exception as e:
            logger.critical(f"Frame processing failed: {str(e)}")
            # Return original frame instead of crashing
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
            time_limit = 30  # Seconds to process video
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

#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------
#                                               SEVENTH anpr.py
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
import random

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
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore')

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(str(Path(__file__).parent.parent.resolve()))
sys.path.append(os.path.abspath("src"))

from config.appconfig import (
    DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT,
)

#--------------------------------------------------------------------------------------
#  Configuration Variables
#--------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent  # src -> project root
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VEHICLE_MODEL_PATH = "models/yolov8m-seg.pt"
MODEL_PATH = "models/license_plate_detector.pt"
# PLATE_REGEX = re.compile(r'^[A-Z0-9]{7,10}$')  # Pre-compiled pattern
PLATE_REGEX = re.compile(r'^[A-Z0-9]{8}$')  # Strict 8-character Nigerian format
TRACKING_FRAMES = 30
MIN_CONFIDENCE = 0.65
MIN_DETECTIONS = 1
TIME_LIMIT = 30 # Seconds to process video
MAX_DETECTIONS_PER_PLATE = 2 # Maximum times to detect each plate
DEFAULT_FRAME_SKIP = 5  # Default value (Process every nth frame)
# HIGH_FRAME_SKIP = 5 # Skip more frames for high FPS videos
# LOW_FRAME_SKIP = 1 # Process every frame for low FPS videos

ADAPTIVE_FRAME_SKIP = True  # Enable dynamic frame skipping
FRAME_SKIP_RANGES = {
    'high_fps': (30, 3),    # For FPS > 30, skip 3 frames
    'normal': (15, 1),       # For FPS 15-30, skip 1 frame
    'low_fps': (0, 0)        # For FPS < 15, process all frames
}


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
        """Improved color detection with multiple ROIs"""
        try:
            if cropped_image.size == 0:
                return "unknown"

            # Analyze multiple regions
            h, w = cropped_image.shape[:2]
            rois = [
                cropped_image[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)],  # Center
                cropped_image[int(h*0.1):int(h*0.3), int(w*0.1):int(w*0.3)],  # Top-left
                cropped_image[int(h*0.7):int(h*0.9), int(w*0.7):int(w*0.9)]  # Bottom-right
            ]

            predictions = []
            for roi in rois:
                img = cv2.resize(roi, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Use HSV colorspace
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                pred = self.color_model.predict(img_array, verbose=0)[0]
                predictions.append(pred)

            # Use majority voting
            avg_pred = np.mean(predictions, axis=0)
            top_idx = np.argmax(avg_pred)

            return COLOR_CLASSES[top_idx] if avg_pred[top_idx] > 0.4 else "unknown"

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
        """Enhanced preprocessing pipeline"""
        if image.size == 0:
            return np.zeros((100, 100), dtype=np.uint8)

        # Resize maintaining aspect ratio
        height = 80
        ratio = image.shape[1] / image.shape[0]
        width = int(height * ratio)
        resized = cv2.resize(image, (width, height))

        # Multiple preprocessing pipelines
        processed = []

        # Pipeline 1: CLAHE + Thresholding
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(gray)
        _, th1 = cv2.threshold(cl1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed.append(th1)

        # Pipeline 2: Edge-enhanced
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        processed.append(edges)

        # Pipeline 3: Color channel selection
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        processed.append(lab[:,:,0])

        # Combine all processed versions
        return np.vstack(processed)

    # def preprocess_plate(self, image):
    #     """Enhanced preprocessing for better OCR results"""
    #     # Check if image is valid
    #     if image is None or image.size == 0:
    #         return np.zeros((100, 100), dtype=np.uint8)

    #     # Maintain aspect ratio but normalize size
    #     target_height = 80
    #     ratio = image.shape[1] / image.shape[0]
    #     target_width = int(target_height * ratio)
    #     resized = cv2.resize(image, (target_width, target_height))

    #     # Convert to grayscale
    #     gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    #     # Try multiple preprocessing approaches
    #     results = []

    #     # Approach 1: Adaptive thresholding
    #     adaptive = cv2.adaptiveThreshold(
    #         gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #         cv2.THRESH_BINARY, 11, 2
    #     )
    #     results.append(adaptive)

    #     # Approach 2: CLAHE enhancement
    #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #     enhanced = clahe.apply(gray)
    #     _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     results.append(binary)

    #     # Approach 3: Edge enhancement
    #     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #     edges = cv2.Canny(blurred, 100, 200)
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #     dilated = cv2.dilate(edges, kernel, iterations=1)
    #     results.append(255 - dilated)  # Invert for OCR

    #     # Combine the approaches (stack them for OCR to try each)
    #     return np.vstack(results)

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
    def _validate_detection(self, plate_text, plate_conf, vehicle_color):
        """Multi-stage detection validation"""
        # Stage 1: Basic format validation
        if not PLATE_REGEX.match(plate_text):
            logger.info(f"Invalid plate format: {plate_text}")
            return False

        # Stage 2: Confidence check
        if plate_conf < MIN_CONFIDENCE:
            logger.info(f"Low confidence: {plate_conf}")
            return False

        # Stage 3: Color validation
        if vehicle_color.lower() == "unknown":
            logger.info(f"Unknown vehicle color")
            return False

        # Stage 4: Temporal validation
        now = time.time()
        if plate_text in self.plate_history:
            last_seen = self.plate_history[plate_text]
            if now - last_seen < self.cooldown_period:
                logger.info(f"Plate in cooldown: {plate_text}")
                return False

        return True


    def process_frame(self, frame):
        """Modified process_frame method with enhanced validation and association"""
        try:
            if not hasattr(self, 'window_initialized'):
                cv2.namedWindow("ANPR Processing", cv2.WINDOW_NORMAL)
                self.window_initialized = True

            time_details = self.get_time_details()
            vehicles = self.detect_vehicle_type(frame)
            results = self.model.predict(frame, conf=MIN_CONFIDENCE, verbose=False)
            current_detections = set()

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                logger.info(f"Detected {len(boxes)} potential plates in frame")

                for box in boxes:
                    try:
                        x1, y1, x2, y2 = map(int, box)
                        plate_img = frame[y1:y2, x1:x2]

                        vehicle_color = "unknown"

                        # Enhanced OCR with improved preprocessing
                        plate_text, plate_conf = self.ocr_license_plate(plate_img)
                        logger.info(f"OCR result: '{plate_text}' with confidence {plate_conf}")

                        # Multi-stage validation
                        if not self._validate_detection(plate_text, plate_conf, vehicle_color):
                            continue

                        # Plate tracking and merging
                        merged_plate = self._merge_similar_plates(plate_text)
                        current_detections.add(merged_plate)

                        # Improved vehicle association using IoU
                        closest_vehicle = None
                        max_iou = 0.0
                        plate_rect = (x1, y1, x2, y2)

                        for vehicle in vehicles:
                            vx1, vy1, vx2, vy2 = vehicle['box']
                            vehicle_rect = (vx1, vy1, vx2, vx2)

                            # Calculate intersection over union
                            xi = max(plate_rect[0], vehicle_rect[0])
                            yi = max(plate_rect[1], vehicle_rect[1])
                            xu = min(plate_rect[2], vehicle_rect[2])
                            yu = min(plate_rect[3], vehicle_rect[3])

                            inter_area = max(xu - xi, 0) * max(yu - yi, 0)
                            plate_area = (x2-x1)*(y2-y1)
                            vehicle_area = (vx2-vx1)*(vy2-vy1)
                            iou = inter_area / (plate_area + vehicle_area - inter_area)

                            if iou > max_iou and iou > 0.1:  # Minimum 10% overlap
                                max_iou = iou
                                closest_vehicle = vehicle

                        # Enhanced color detection with multiple ROIs
                        vehicle_color = "unknown"
                        if closest_vehicle and max_iou > 0.1:
                            vx1, vy1, vx2, vy2 = closest_vehicle['box']
                            vehicle_roi = frame[vy1:vy2, vx1:vx2]
                            vehicle_color = self.predict_vehicle_color(vehicle_roi)

                        # Final validation checkpoint
                        if not self._validate_detection(merged_plate, plate_conf, vehicle_color):
                            continue

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
                            'vehicle_type': closest_vehicle['type'] if closest_vehicle else 'unknown',
                            'vehicle_color': vehicle_color,
                            'time_details': time_details
                        }

                        self.plate_tracker[merged_plate] = tracker_entry
                        logger.info(f"Tracked plate {merged_plate} with count {tracker_entry['count']}")

                        # Visualization
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame,
                                f"{merged_plate} ({plate_conf:.2f})",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame,
                                f"{vehicle_color} {tracker_entry['vehicle_type']}",
                                (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    except Exception as e:
                        logger.error(f"Error processing box: {str(e)}")
                        continue

            # Vehicle visualization
            for vehicle in vehicles:
                vx1, vy1, vx2, vy2 = vehicle['box']
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 0, 255), 2)
                cv2.putText(frame,
                        f"{vehicle['type']} ({vehicle['confidence']:.2f})",
                        (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("ANPR Processing", frame)
            if cv2.waitKey(1) == ord('q'):
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
    # def process_video(self):
    #     """Main processing loop with optimized runtime controls"""
    #     args = parse_arguments()
    #     with VideoProcessor(args.source) as video:
    #         logger.info("Starting video processing...")

    #         # Initialize timing and control variables
    #         start_time = time.time()
    #         last_save_time = time.time()

    #         # Dynamic frame skip initialization
    #         fps = video.fps
    #         frame_skip = DEFAULT_FRAME_SKIP
    #         if ADAPTIVE_FRAME_SKIP:
    #             if fps > FRAME_SKIP_RANGES['high_fps'][0]:
    #                 frame_skip = FRAME_SKIP_RANGES['high_fps'][1]
    #             elif fps > FRAME_SKIP_RANGES['normal'][0]:
    #                 frame_skip = FRAME_SKIP_RANGES['normal'][1]
    #             else:
    #                 frame_skip = FRAME_SKIP_RANGES['low_fps'][1]

    #         if frame_skip > 0 and i % (frame_skip + random.randint(0,2)) == 0:
    #             pass

    #         frame_skip = DEFAULT_FRAME_SKIP
    #         time_limit = TIME_LIMIT
    #         max_detections_per_plate = MAX_DETECTIONS_PER_PLATE
    #         processed_plates = set()  # Track fully processed plates

    #         for i, frame in enumerate(video.get_frames()):
    #             # Check for early termination conditions
    #             if frame is None:
    #                 logger.error("Received empty frame - check video source")
    #                 break

    #             # Check if we've reached the time limit
    #             elapsed_time = time.time() - start_time
    #             if elapsed_time > time_limit:
    #                 logger.info(f"Reached time limit of {time_limit} seconds")
    #                 # Ensure final save before exiting
    #                 self.save_to_database(self.plate_tracker)
    #                 break

    #             # Skip frames to reduce processing load
    #             if i % frame_skip != 0:
    #                 continue

    #             # Progress logging
    #             if i % 10 == 0:  # Log every 10 frames
    #                 logger.info(f"Processing frame {i} (elapsed time: {elapsed_time:.2f}s)")

    #             # Process the current frame
    #             processed_frame = self.process_frame(frame)
    #             video.write_frame(processed_frame)

    #             # Check for plates that reached detection threshold
    #             for plate, data in list(self.plate_tracker.items()):
    #                 if data['count'] >= max_detections_per_plate and plate not in processed_plates:
    #                     logger.info(f"Plate {plate} reached detection threshold with {data['count']} detections")
    #                     # Save this plate immediately
    #                     self.save_to_database({plate: data})
    #                     processed_plates.add(plate)
    #                     # Option: remove from tracker to stop further processing
    #                     # del self.plate_tracker[plate]

    #             # Periodically save to database (for plates not yet at threshold)
    #             if time.time() - last_save_time > 10:
    #                 logger.info(f"Current tracker has {len(self.plate_tracker)} plates")

    #                 # Filter out plates we've already fully processed
    #                 plates_to_save = {plate: data for plate, data in self.plate_tracker.items()
    #                                 if plate not in processed_plates}

    #                 if plates_to_save:
    #                     self.save_to_database(plates_to_save)
    #                 last_save_time = time.time()

    #             # Cleanup plates not seen recently
    #             self.cleanup_tracker()

    #             # Check for user quit
    #             if cv2.waitKey(1) == ord('q'):
    #                 logger.info("User requested exit")
    #                 break

    #         # Final save to ensure we don't miss anything
    #         logger.info("Video processing complete, saving final results")
    #         self.save_to_database(self.plate_tracker)

    def process_video(self):
        """Main processing loop with optimized runtime controls"""
        args = parse_arguments()
        with VideoProcessor(args.source) as video:
            logger.info("Starting video processing...")

            # Initialize timing and control variables
            start_time = time.time()
            last_save_time = time.time()
            frame_counter = 0  # Explicit frame counter initialization

            # Dynamic frame skip initialization
            fps = video.fps
            frame_skip = DEFAULT_FRAME_SKIP
            if ADAPTIVE_FRAME_SKIP:
                if fps > 30:
                    frame_skip = FRAME_SKIP_RANGES['high_fps'][1]
                elif fps > 15:
                    frame_skip = FRAME_SKIP_RANGES['normal'][1]
                else:
                    frame_skip = FRAME_SKIP_RANGES['low_fps'][1]

            processed_plates = set()

            for frame in video.get_frames():
                # Check for early termination conditions
                if frame is None:
                    logger.error("Received empty frame - check video source")
                    break

                # Update frame counter
                frame_counter += 1

                # Skip frames based on dynamic skip rules
                if frame_skip > 0 and (frame_counter % (frame_skip + random.randint(0, 2)) != 0):
                    continue

                # Progress logging
                if frame_counter % 10 == 0:
                    logger.info(f"Processing frame {frame_counter} (Elapsed: {time.time()-start_time:.2f}s)")

                # Process the current frame
                processed_frame = self.process_frame(frame)
                video.write_frame(processed_frame)

                # Existing plate processing logic
                for plate, data in list(self.plate_tracker.items()):
                    if data['count'] >= MAX_DETECTIONS_PER_PLATE and plate not in processed_plates:
                        logger.info(f"Plate {plate} reached detection threshold")
                        self.save_to_database({plate: data})
                        processed_plates.add(plate)

                # Periodic database save
                if time.time() - last_save_time > 10:
                    plates_to_save = {p: d for p, d in self.plate_tracker.items() if p not in processed_plates}
                    if plates_to_save:
                        self.save_to_database(plates_to_save)
                    last_save_time = time.time()

                # Cleanup and user input check
                self.cleanup_tracker()
                if cv2.waitKey(1) == ord('q'):
                    logger.info("User requested exit")
                    break

            # Final save
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
