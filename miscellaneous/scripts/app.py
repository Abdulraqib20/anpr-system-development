import streamlit as st
import cv2
import tempfile
import os
from collections import defaultdict
from datetime import datetime
import numpy as np
import math
import re

from anpr import (
    YOLO, 
    PaddleOCR, 
    process_license_plate, 
    get_best_license_plates, 
    save_to_database, 
    db_params, 
    license_plate_tracker
    ) 

st.title("Automatic Number Plate Recognition (ANPR)")

# File uploader for video
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_file:
    # Create a temporary file to store the uploaded video
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    
    # Display video player
    st.video(uploaded_file)

    # Initialize the YOLO Model and Paddle OCR
    model = YOLO(r"weights\license_plate_detector.pt")
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

    # Process the video frame by frame
    def process_video(file_path):
        cap = cv2.VideoCapture(file_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(r"output/annotated_video.mp4", fourcc, fps, (width, height))

        count = 0
        startTime = datetime.now()

        while True:
            ret, frame = cap.read()
            if ret:
                count += 1
                results = model.predict(frame, conf=0.45)
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        label, ocr_confidence = paddle_ocr(frame, x1, y1, x2, y2)
                        process_license_plate(label, ocr_confidence, count)
                        textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                        c2 = x1 + textSize[0], y1 - textSize[1] - 3
                        cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
                        cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

                # Write the frame to the output video
                out.write(frame)

                currentTime = datetime.now()
                if (currentTime - startTime).seconds >= 20:
                    endTime = currentTime
                    best_plates = get_best_license_plates()
                    save_to_database(best_plates, startTime, endTime)
                    startTime = currentTime
                    license_plate_tracker.clear()

                # Show the processed frame on Streamlit
                st.image(frame, channels="BGR")

                if cv2.waitKey(1) & 0xFF == ord('1'):
                    break
            else:
                break

        cap.release()
        out.release()

    # Process and display the annotated video
    if st.button("Process Video"):
        st.write("Processing video, please wait...")
        process_video(temp_file.name)
        st.write("Video processing completed.")

    # Display the annotated video if processed
    if os.path.exists(r"output/annotated_video.mp4"):
        st.video(r"output/annotated_video.mp4")

