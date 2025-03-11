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
