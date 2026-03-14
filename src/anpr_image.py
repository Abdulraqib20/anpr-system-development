import os
import sys
import cv2
import re
import logging
import time
import random
from pathlib import Path
import argparse
from datetime import datetime
from collections import defaultdict
from threading import Thread
from queue import Queue
import io
import base64

import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from keras import layers
from groq import Groq
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore')

try:
    from ollama import Client as OllamaClient
except Exception:
    OllamaClient = None

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from config.appconfig import (
    DB_HOST,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
    DB_PORT,
    OCR_PROVIDER,
    OCR_FALLBACK_TO_GROQ,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    GROQ_MODEL_NAME,
    GROQ_API_KEY,
)

logs_dir = Path(__file__).parent.parent / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
log_file_path = logs_dir / "config.log"

# Set up logging
logger = logging.getLogger("ANPR")
logger.setLevel(logging.DEBUG)
# Clear existing handlers to prevent duplication
if logger.hasHandlers():
    logger.handlers.clear()
# Prevent propagation to root logger to avoid duplicate logs
logger.propagate = False

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
PLATE_IMAGE_DIR = OUTPUT_DIR / "plate_images" # directory for saving cropped plate images
PLATE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# VEHICLE_MODEL_PATH = "models/yolov8m-seg.pt"
VEHICLE_MODEL_PATH = "models/yolov8n.pt"
VEHICLE_COLOR_MODEL_PATH = "models/EFN-model.best.h5"
MODEL_PATH="models/license_plate_detector.pt"

PLATE_REGEX = re.compile(r'^[A-Z0-9]{8}$')  # Strict 8-character Nigerian format
MIN_CONFIDENCE=0.45
GROQ_CONFIDENCE=0.90
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
    parser = argparse.ArgumentParser(description="UNILORIN ANPR System - Image Processor")
    parser.add_argument(
        "--source",
        type=str,
        nargs='+',  # Accept one or more source arguments
        required=True, # Make source mandatory
        help="Input source(s) (image path(s)). Provide one or more file paths."
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
# Image Processor Class
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
        """Saves the processed image and returns the filename if successful."""
        path_to_save = output_path if output_path else self.output_path
        try:
            success = cv2.imwrite(str(path_to_save), image)
            if success:
                filename = Path(path_to_save).name
                logger.info(f"Processed image saved to: {path_to_save}")
                return filename # Return filename on success
            else:
                logger.warning(f"Failed to save image to: {path_to_save}")
                return None
        except Exception as e:
            logger.error(f"Error saving image {path_to_save}: {e}", exc_info=True)
            return None


#-------------------------------------------------------------------------------
# ANPR System Class
#-------------------------------------------------------------------------------
class ANPRProcessor:
    """Main ANPR processing class"""

    #----------------------------------------------------------------------------------------------
    # Initialization
    #----------------------------------------------------------------------------------------------
    def __init__(self, socketio_instance=None):
        # Initialize components
        self.vehicle_model = YOLO(VEHICLE_MODEL_PATH)
        self.model = YOLO(MODEL_PATH)
        self.socketio = socketio_instance # Store the SocketIO instance

        # OCR provider setup
        self.ocr_provider = OCR_PROVIDER
        self.ocr_fallback_to_groq = OCR_FALLBACK_TO_GROQ
        self.ollama_host = OLLAMA_HOST
        self.ollama_model_name = OLLAMA_MODEL
        self.groq_model_name = GROQ_MODEL_NAME
        self.ollama_client = None
        self.groq_client = None
        self._initialize_ocr_clients()

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

        # --- Ensure Database Table Exists --- Start
        self._ensure_table_exists()
        # --- Ensure Database Table Exists --- End

        logger.info(
            "ANPRProcessor initialized with OCR provider=%s, ollama_model=%s, groq_model=%s, fallback_to_groq=%s",
            self.ocr_provider,
            self.ollama_model_name,
            self.groq_model_name,
            self.ocr_fallback_to_groq,
        )

    def _initialize_ocr_clients(self):
        """Initialize configured OCR provider clients."""
        errors = []

        if self.ocr_provider == 'ollama':
            if OllamaClient is None:
                errors.append("Ollama Python client is not available. Install package 'ollama'.")
            else:
                try:
                    self.ollama_client = OllamaClient(host=self.ollama_host)
                    logger.info("Ollama client initialized at %s with model %s", self.ollama_host, self.ollama_model_name)
                except Exception as e:
                    errors.append(f"Failed to initialize Ollama client: {e}")

        if self.ocr_provider == 'groq' or self.ocr_fallback_to_groq:
            if not GROQ_API_KEY:
                errors.append("GROQ_API_KEY is missing while Groq is required by provider/fallback settings.")
            else:
                try:
                    self.groq_client = Groq(api_key=GROQ_API_KEY)
                    logger.info("Groq client initialized with model: %s", self.groq_model_name)
                except Exception as e:
                    errors.append(f"Failed to initialize Groq client: {e}")

        if self.ocr_provider == 'ollama' and self.ollama_client:
            return
        if self.ocr_provider == 'groq' and self.groq_client:
            return
        if self.ocr_provider == 'ollama' and self.ocr_fallback_to_groq and self.groq_client:
            logger.warning("Ollama unavailable at startup, Groq fallback is active.")
            return

        details = "; ".join(errors) if errors else "No provider could be initialized."
        raise RuntimeError(f"OCR provider initialization failed: {details}")

    #----------------------------------------------------------------------------------------------
    # Helper: Ensure Database Table Exists
    #----------------------------------------------------------------------------------------------
    def _ensure_table_exists(self):
        """Creates the detected_plates table if it doesn't exist."""
        conn = None
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                # Define table schema for detected_plates
                create_plates_table_sql = """
                CREATE TABLE IF NOT EXISTS detected_plates (
                    id SERIAL PRIMARY KEY,
                    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
                    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
                    license_plate VARCHAR(20) NOT NULL,
                    confidence REAL, -- Using REAL for floating point
                    detection_count INTEGER,
                    vehicle_type VARCHAR(50),
                    vehicle_color VARCHAR(50),
                    car_brand VARCHAR(100), -- New column for car brand
                    time_of_day VARCHAR(20),
                    day_of_week VARCHAR(20),
                    image_filename VARCHAR(255), -- Path to the cropped plate image
                    annotated_frame_filename VARCHAR(255) -- Path to the full annotated frame
                );
                """
                cursor.execute(create_plates_table_sql)
                conn.commit()
                logger.info("Table 'detected_plates' checked/created successfully. Car_brand column ensured.")

                # --- Check if car_brand column exists, add if not (for existing tables) ---
                # This is a common pattern for making schema changes more robust
                # to already existing tables from previous versions.
                cursor.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name='detected_plates' AND column_name='car_brand';
                """)
                if not cursor.fetchone():
                    logger.info("Column 'car_brand' not found in 'detected_plates', attempting to add it.")
                    cursor.execute("ALTER TABLE detected_plates ADD COLUMN car_brand VARCHAR(100);")
                    conn.commit()
                    logger.info("Successfully ADDED 'car_brand' column to 'detected_plates'.")
                else:
                    logger.debug("Column 'car_brand' already exists in 'detected_plates'.")

                # --- Define and create usage table --- Start
                create_usage_table_sql = """
                CREATE TABLE IF NOT EXISTS groq_api_usage (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    provider VARCHAR(20) NOT NULL DEFAULT 'groq', -- e.g., 'ollama' or 'groq'
                    api_call_type VARCHAR(50) NOT NULL,       -- e.g., 'plate_ocr', 'brand_detection'
                    model_name VARCHAR(100) NOT NULL,         -- e.g., selected provider model
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    related_detection_id INTEGER DEFAULT NULL,  -- Optional: link to detected_plates.id
                    FOREIGN KEY (related_detection_id) REFERENCES detected_plates(id) ON DELETE SET NULL
                );
                """
                cursor.execute(create_usage_table_sql)
                conn.commit()
                logger.info("Table 'groq_api_usage' checked/created successfully.")

                cursor.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name='groq_api_usage' AND column_name='provider';
                """)
                if not cursor.fetchone():
                    logger.info("Column 'provider' not found in 'groq_api_usage', attempting to add it.")
                    cursor.execute("ALTER TABLE groq_api_usage ADD COLUMN provider VARCHAR(20) NOT NULL DEFAULT 'groq';")
                    conn.commit()
                    logger.info("Successfully ADDED 'provider' column to 'groq_api_usage'.")
                else:
                    logger.debug("Column 'provider' already exists in 'groq_api_usage'.")
                # --- Define and create usage table --- End

        except Exception as e:
            logger.error(f"Database error during table creation (detected_plates or groq_api_usage): {e}", exc_info=True)
            if conn:
                conn.rollback()
        finally:
            if conn:
                self.db_pool.putconn(conn)

    #----------------------------------------------------------------------------------------------
    # Helper: Log LLM API Usage to Database
    #----------------------------------------------------------------------------------------------
    def _log_api_usage_to_db(self, call_type, provider, model_name, usage_stats, related_detection_id=None):
        """Logs LLM API usage stats to the groq_api_usage table."""
        conn = None
        try:
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None

            if isinstance(usage_stats, dict):
                prompt_tokens = usage_stats.get('prompt_tokens')
                completion_tokens = usage_stats.get('completion_tokens')
                total_tokens = usage_stats.get('total_tokens')
            elif usage_stats is not None:
                prompt_tokens = getattr(usage_stats, 'prompt_tokens', None)
                completion_tokens = getattr(usage_stats, 'completion_tokens', None)
                total_tokens = getattr(usage_stats, 'total_tokens', None)

            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                sql_insert = """
                    INSERT INTO groq_api_usage
                    (provider, api_call_type, model_name, prompt_tokens, completion_tokens, total_tokens, related_detection_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                record = (
                    provider,
                    call_type,
                    model_name,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    related_detection_id
                )
                cursor.execute(sql_insert, record)
                conn.commit()
                logger.debug(
                    "Logged API usage: Provider=%s, Type=%s, Model=%s, Tokens=%s",
                    provider,
                    call_type,
                    model_name,
                    total_tokens if total_tokens is not None else 'N/A'
                )
        except Exception as e:
            logger.error(f"Failed to log API usage to DB: {e}", exc_info=True)
            if conn:
                conn.rollback()
        finally:
            if conn:
                self.db_pool.putconn(conn)

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
    # def preprocess_plate(self, image):
    #     """Preprocess image for better OCR results using CLAHE and Adaptive Thresholding."""
    #     if image is None or image.size == 0:
    #         logger.warning("preprocess_plate received an empty image.")
    #         return None # Return None if image is invalid

    #     try:
    #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #         # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    #         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #         contrast_enhanced_gray = clahe.apply(gray)
    #         logger.debug("Applied CLAHE for contrast enhancement.")

    #         # Apply Gaussian Blur slightly
    #         # blurred = cv2.GaussianBlur(contrast_enhanced_gray, (3, 3), 0)
    #         # logger.debug("Applied Gaussian Blur.")
    #         # Experiment: Skip blur or use median blur for salt-and-pepper noise
    #         blurred = cv2.medianBlur(contrast_enhanced_gray, 3)
    #         logger.debug("Applied Median Blur.")

    #         # Apply Adaptive Thresholding
    #         # Adjust blockSize and C for optimal results
    #         # blockSize must be odd
    #         # C is a constant subtracted from the mean or weighted sum
    #         adaptive_thresh = cv2.adaptiveThreshold(
    #             blurred,
    #             255, # Max value
    #             cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Gaussian weighting for neighborhood
    #             cv2.THRESH_BINARY, # Standard binary threshold
    #             blockSize=15, # Size of the neighborhood area (must be odd)
    #             C=7 # Constant subtracted from the calculated threshold
    #         )
    #         logger.debug("Applied Adaptive Thresholding.")

    #         # Optional: Denoising (can be slow, apply if noise is significant)
    #         # denoised = cv2.fastNlMeansDenoising(adaptive_thresh, None, h=10, templateWindowSize=7, searchWindowSize=21)
    #         # logger.debug("Applied Denoising.")
    #         # return denoised

    #         return adaptive_thresh
    #     except cv2.error as cv_err:
    #         logger.error(f"OpenCV error during preprocessing: {cv_err}")
    #         return None
    #     except Exception as e:
    #         logger.error(f"Unexpected error during preprocessing: {e}", exc_info=True)
    #         return None
    #
    #----------------------------------------------------------------------------------------------
    # Helper: Clean Plate Text
    #----------------------------------------------------------------------------------------------
    def _clean_plate_text(self, raw_text: str) -> str:
        """Cleans the raw OCR text to be uppercase alphanumeric only."""
        if not isinstance(raw_text, str):
            return ""
        cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
        return cleaned

    #----------------------------------------------------------------------------------------------
    # Helper: Encode image for provider calls
    #----------------------------------------------------------------------------------------------
    def _encode_image_to_base64(self, image, quality=95):
        """Encode OpenCV image to base64 JPEG string."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        is_success, buffer = cv2.imencode(".jpg", image, encode_param)
        if not is_success:
            return None
        return base64.b64encode(buffer.tobytes()).decode('utf-8')

    def _extract_ollama_usage(self, response):
        """Normalize Ollama usage values to the DB logging shape."""
        if not isinstance(response, dict):
            return None

        prompt_tokens = response.get('prompt_eval_count')
        completion_tokens = response.get('eval_count')
        total_tokens = None
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        return {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
        }

    def _get_provider_attempt_order(self):
        """Return provider attempt order, appending Groq fallback when configured."""
        providers = [self.ocr_provider]
        if self.ocr_provider == 'ollama' and self.ocr_fallback_to_groq and self.groq_client:
            providers.append('groq')
        return providers

    #----------------------------------------------------------------------------------------------
    # Helper: Process Plate with Ollama
    #----------------------------------------------------------------------------------------------
    def _process_plate_with_ollama(self, plate_image, related_detection_id=None):
        """Encodes plate image and calls Ollama model for OCR."""
        if not self.ollama_client:
            logger.warning("Ollama client not initialized; skipping Ollama plate OCR.")
            return ""

        try:
            image_base64 = self._encode_image_to_base64(plate_image, quality=95)
            if not image_base64:
                logger.error("Failed to encode plate image to JPEG for Ollama.")
                return ""

            logger.debug(
                "Sending image (approx %s base64 chars) to Ollama model for PLATE OCR: %s",
                len(image_base64),
                self.ollama_model_name,
            )

            response = self.ollama_client.chat(
                model=self.ollama_model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': "Analyze the license plate image. Extract ONLY the 8 alphanumeric characters (A-Z, 0-9) representing the plate number. Output MUST be exactly 8 characters long and in ALL UPPERCASE. Remove any hyphens, spaces, or symbols. Respond with ONLY the 8-character plate string (e.g., ABC123XY). NO other text, explanation, or formatting.",
                        'images': [image_base64],
                    }
                ],
                options={'temperature': 0},
            )

            usage_stats = self._extract_ollama_usage(response)
            self._log_api_usage_to_db('plate_ocr', 'ollama', self.ollama_model_name, usage_stats, related_detection_id)

            raw_ocr_result = response.get('message', {}).get('content', '')
            cleaned_result = self._clean_plate_text(raw_ocr_result)
            logger.info("Ollama raw result for PLATE: '%s', Cleaned result: '%s'", raw_ocr_result, cleaned_result)
            return cleaned_result

        except Exception as e:
            logger.error(f"Error processing PLATE with Ollama: {str(e)}", exc_info=True)
            return ""

    #----------------------------------------------------------------------------------------------
    # Helper: Process Plate with Groq
    #----------------------------------------------------------------------------------------------
    def _process_plate_with_groq(self, plate_image, related_detection_id=None):
        """Encodes plate image and calls Groq model for OCR."""
        if not self.groq_client:
            logger.warning("Groq client not initialized; skipping Groq plate OCR.")
            return ""

        try:
            image_base64 = self._encode_image_to_base64(plate_image, quality=95)
            if not image_base64:
                logger.error("Failed to encode plate image to JPEG for Groq.")
                return ""

            image_media_type = 'image/jpeg'

            logger.debug(
                "Sending image (approx %s base64 chars) to Groq model for PLATE OCR: %s",
                len(image_base64),
                self.groq_model_name,
            )

            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze the license plate image. Extract ONLY the 8 alphanumeric characters (A-Z, 0-9) representing the plate number. Output MUST be exactly 8 characters long and in ALL UPPERCASE. Remove any hyphens, spaces, or symbols. Respond with ONLY the 8-character plate string (e.g., ABC123XY). NO other text, explanation, or formatting."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{image_media_type};base64,{image_base64}"
                                }
                            }
                        ],
                    }
                ],
                model=self.groq_model_name,
                max_tokens=30,
            )

            if hasattr(chat_completion, 'usage') and chat_completion.usage:
                self._log_api_usage_to_db('plate_ocr', 'groq', self.groq_model_name, chat_completion.usage, related_detection_id)
            else:
                logger.warning("Groq chat_completion response did not include usage statistics for plate_ocr.")

            raw_ocr_result = chat_completion.choices[0].message.content
            cleaned_result = self._clean_plate_text(raw_ocr_result)
            logger.info("Groq raw result for PLATE: '%s', Cleaned result: '%s'", raw_ocr_result, cleaned_result)
            return cleaned_result

        except Exception as e:
            logger.error(f"Error processing PLATE with Groq: {str(e)}", exc_info=True)
            return ""

    def _process_plate_with_provider(self, plate_image, related_detection_id=None):
        """Run plate OCR with selected provider and optional fallback."""
        for provider in self._get_provider_attempt_order():
            if provider == 'ollama':
                result = self._process_plate_with_ollama(plate_image, related_detection_id)
            elif provider == 'groq':
                result = self._process_plate_with_groq(plate_image, related_detection_id)
            else:
                logger.error("Unsupported OCR provider configured: %s", provider)
                continue

            if result:
                if provider != self.ocr_provider:
                    logger.warning("OCR fallback provider used for plate OCR: %s", provider)
                return result

        return ""

    #----------------------------------------------------------------------------------------------
    # Helper: Detect Car Brand with Ollama
    #----------------------------------------------------------------------------------------------
    def _detect_car_brand_with_ollama(self, image_to_process, related_detection_id=None):
        """Encodes an image and calls Ollama model to detect car brand and model."""
        if not self.ollama_client:
            logger.warning("Ollama client not initialized; skipping Ollama car brand detection.")
            return "Unknown"

        try:
            image_base64 = self._encode_image_to_base64(image_to_process, quality=90)
            if not image_base64:
                logger.error("Failed to encode image to JPEG for Ollama car brand detection.")
                return "Unknown"

            response = self.ollama_client.chat(
                model=self.ollama_model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': "Analyze this image and determine the exact brand/make of the car (e.g., Toyota, Honda, Ford, BMW, etc.). Be as specific as possible by identifying both the make and model if visible (e.g., Toyota Camry, Honda Civic, BMW 3 Series). Provide ONLY the car brand/make and model in your response with no additional text or explanations. If unknown, respond 'Unknown'.",
                        'images': [image_base64],
                    }
                ],
                options={'temperature': 0},
            )

            usage_stats = self._extract_ollama_usage(response)
            self._log_api_usage_to_db('brand_detection', 'ollama', self.ollama_model_name, usage_stats, related_detection_id)

            brand_result = response.get('message', {}).get('content', '').strip()
            if not brand_result or len(brand_result) > 100:
                logger.warning("Ollama car brand detection returned unusual result: '%s'. Defaulting to Unknown.", brand_result)
                return "Unknown"

            logger.info("Ollama car brand detection result: '%s'", brand_result)
            return brand_result

        except Exception as e:
            logger.error(f"Error detecting car brand with Ollama: {str(e)}", exc_info=True)
            return "Unknown"

    #----------------------------------------------------------------------------------------------
    # Helper: Detect Car Brand with Groq
    #----------------------------------------------------------------------------------------------
    def _detect_car_brand_with_groq(self, image_to_process, related_detection_id=None):
        """Encodes an image and calls Groq model to detect car brand and model."""
        if not self.groq_client:
            logger.warning("Groq client not initialized; skipping Groq car brand detection.")
            return "Unknown"

        try:
            image_base64 = self._encode_image_to_base64(image_to_process, quality=90)
            if not image_base64:
                logger.error("Failed to encode image to JPEG for Groq car brand detection.")
                return "Unknown"

            image_media_type = 'image/jpeg'

            logger.debug(f"Sending image for CAR BRAND detection to Groq model: {self.groq_model_name}")

            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze this image and determine the exact brand/make of the car (e.g., Toyota, Honda, Ford, BMW, etc.).

                            Be as specific as possible by identifying both the make and model if visible (e.g., 'Toyota Camry', 'Honda Civic', 'BMW 3 Series').

                            Provide ONLY the car brand/make and model in your response with no additional text or explanations. If unknown, respond 'Unknown'.
                            """
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{image_media_type};base64,{image_base64}"}
                            }
                        ],
                    }
                ],
                model=self.groq_model_name,
                max_tokens=50
            )

            if hasattr(chat_completion, 'usage') and chat_completion.usage:
                self._log_api_usage_to_db('brand_detection', 'groq', self.groq_model_name, chat_completion.usage, related_detection_id)
            else:
                logger.warning("Groq chat_completion response did not include usage statistics for brand_detection.")

            brand_result = chat_completion.choices[0].message.content.strip()
            if not brand_result or len(brand_result) > 100:
                logger.warning(f"Groq car brand detection returned an unusual result: '{brand_result}'. Defaulting to Unknown.")
                return "Unknown"

            logger.info(f"Groq car brand detection result: '{brand_result}'")
            return brand_result

        except Exception as e:
            logger.error(f"Error detecting car brand with Groq: {str(e)}", exc_info=True)
            return "Unknown"

    def _detect_car_brand_with_provider(self, image_to_process, related_detection_id=None):
        """Run car brand detection with selected provider and optional fallback."""
        for provider in self._get_provider_attempt_order():
            if provider == 'ollama':
                result = self._detect_car_brand_with_ollama(image_to_process, related_detection_id)
            elif provider == 'groq':
                result = self._detect_car_brand_with_groq(image_to_process, related_detection_id)
            else:
                logger.error("Unsupported OCR provider configured for brand detection: %s", provider)
                continue

            if result and result != 'Unknown':
                if provider != self.ocr_provider:
                    logger.warning("OCR fallback provider used for brand detection: %s", provider)
                return result

        return "Unknown"

    #----------------------------------------------------------------------------------------------
    # OCR License Plate
    #----------------------------------------------------------------------------------------------
    def ocr_license_plate(self, image, related_detection_id=None):
        """Perform OCR on license plate image using the configured provider."""
        try:
            logger.debug("Sending original cropped plate image to OCR provider=%s.", self.ocr_provider)
            image_to_ocr = image

            cleaned_plate = self._process_plate_with_provider(image_to_ocr, related_detection_id)

            if not cleaned_plate:
                logger.warning("OCR provider did not return a valid plate string.")
                return "", 0.0

            if PLATE_REGEX.fullmatch(cleaned_plate):
                logger.info("Valid plate format found via provider OCR: '%s'", cleaned_plate)
                return cleaned_plate, GROQ_CONFIDENCE
            else:
                logger.warning(
                    "OCR result '%s' (%s chars) does not match PLATE_REGEX %s. Attempting correction...",
                    cleaned_plate,
                    len(cleaned_plate),
                    PLATE_REGEX.pattern,
                )

                corrected_plate = None
                if len(cleaned_plate) == 7:
                    correction_map = {
                        '0': ['O', 'D'], '1': ['I', 'L', 'T'], '2': ['Z'],
                        '5': ['S'], '8': ['B'], '9': ['P', 'N']
                    }
                    last_char = cleaned_plate[-1]
                    possible_corrections = correction_map.get(last_char, [])

                    if possible_corrections:
                        logger.debug("Attempting corrections for OCR result last char '%s': %s", last_char, possible_corrections)
                        base = cleaned_plate[:-1]
                        for replacement in possible_corrections:
                            potential_plate = base + replacement
                            logger.debug("Testing OCR correction: '%s' against regex '%s'", potential_plate, PLATE_REGEX.pattern)
                            if PLATE_REGEX.fullmatch(potential_plate):
                                corrected_plate = potential_plate
                                logger.info("OCR correction successful: '%s' -> '%s'", cleaned_plate, corrected_plate)
                                break
                    else:
                        logger.debug("No predefined corrections found for OCR result last char '%s'.", last_char)

                if corrected_plate:
                    return corrected_plate, GROQ_CONFIDENCE

                logger.warning("Correction failed for OCR result '%s'. Discarding.", cleaned_plate)
                return "", 0.0

        except Exception as e:
            logger.error(f"OCR processing step error: {str(e)}", exc_info=True)
            return "", 0.0

    #---------------------------------------------------------------------------------------------
    # Helper: Levenshtein Distance
    #---------------------------------------------------------------------------------------------
    # def _calculate_levenshtein_distance(self, s1, s2):
    #     """Calculates the Levenshtein distance between two strings."""
    #     if len(s1) < len(s2):
    #         return self._calculate_levenshtein_distance(s2, s1)

    #     if len(s2) == 0:
    #         return len(s1)

    #     previous_row = range(len(s2) + 1)
    #     for i, c1 in enumerate(s1):
    #         current_row = [i + 1]
    #         for j, c2 in enumerate(s2):
    #             insertions = previous_row[j + 1] + 1
    #             deletions = current_row[j] + 1
    #             substitutions = previous_row[j] + (c1 != c2)
    #             current_row.append(min(insertions, deletions, substitutions))
    #         previous_row = current_row

    #     return previous_row[-1]

    #---------------------------------------------------------------------------------------------
    # Save to Database
    #---------------------------------------------------------------------------------------------
    def save_to_database(self, detections, annotated_frame_filename=None):
        """Consolidates plates and saves valid ones to the database.

        Args:
            detections (list or dict): List or dictionary of detection data.
            annotated_frame_filename (str, optional): The filename of the saved annotated frame image.
                                                      Defaults to None.
        """
        if not detections:
            logger.info("No plates to save.")
            return False

        logger.info(f"Found {len(detections)} potential plates to process for DB:")
        for det_data in detections:
            logger.info(f"  Plate: {det_data.get('plate_text', 'N/A')}, "
                        f"Confidence: {det_data.get('plate_confidence', 0.0):.2f}, "
                        f"Vehicle: {det_data.get('vehicle_type', 'N/A')}, "
                        f"Color: {det_data.get('vehicle_color', 'N/A')}, "
                        f"Brand: {det_data.get('car_brand', 'N/A')}")

        color_filtered = []
        confidence_filtered = []
        valid_detections_for_db = []
        for det_data in detections:
            plate_text = det_data.get('plate_text', '')
            if not plate_text:
                logger.warning(f"Skipping detection with no plate text: {det_data}")
                continue

            if det_data.get('vehicle_color', '').lower() == 'unknown':
                color_filtered.append(plate_text)

            if det_data.get('plate_confidence', 0.0) < MIN_CONFIDENCE:
                confidence_filtered.append(plate_text)
                continue

            valid_detections_for_db.append(det_data)

        if color_filtered:
            logger.info(f"Note: {len(color_filtered)} plates had 'unknown' vehicle color (still processed for DB): {color_filtered}")
        if confidence_filtered:
            logger.warning(f"Filtered out {len(confidence_filtered)} plates due to low confidence (<{MIN_CONFIDENCE}): {confidence_filtered}")

        if not valid_detections_for_db:
            logger.warning("No valid detections left after filtering for DB save.")
            return False

        logger.info(f"Preparing to save {len(valid_detections_for_db)} valid detections to database")

        conn = None
        saved_successfully = False
        try:
            conn = self.db_pool.getconn()
            conn.autocommit = False

            with conn.cursor() as cursor:
                records_to_insert = []
                for data in valid_detections_for_db:
                    time_details = data.get('time_details')
                    if not time_details:
                        logger.warning(f"Skipping plate {data.get('plate_text')} due to missing time_details.")
                        continue

                    now_iso = datetime.now().isoformat()

                    records_to_insert.append(
                        (
                            now_iso,
                            now_iso,
                            data['plate_text'],
                            data['plate_confidence'],
                            1,
                            data['vehicle_type'],
                            data['vehicle_color'],
                            data.get('car_brand', 'Unknown'),
                            time_details.get('time_of_day'),
                            time_details.get('day_of_week'),
                            data.get('image_filename'),
                            annotated_frame_filename
                        )
                    )

                if not records_to_insert:
                    logger.info("No records prepared for DB insertion after final checks.")
                    return False # Early exit if no records

                logger.info(f"Inserting {len(records_to_insert)} records into DB.")

                sql_insert = """
                    INSERT INTO detected_plates
                    (start_time, end_time, license_plate, confidence, detection_count,
                     vehicle_type, vehicle_color, car_brand, time_of_day, day_of_week, image_filename,
                     annotated_frame_filename)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, license_plate; -- Return id and license_plate of inserted row
                """
                # Use executemany for batch insert, but we need to handle RETURNING for each row if possible
                # For simplicity with RETURNING and alert generation, let's insert one by one if executemany with RETURNING is complex
                # Or, fetch IDs after a batch insert if alert generation can be a separate step.
                # Let's iterate and insert one by one to get the ID immediately for alert checking.

                inserted_detection_ids_and_plates = []
                for record in records_to_insert:
                    cursor.execute(sql_insert, record)
                    inserted_row = cursor.fetchone() # Get the returned id and license_plate
                    if inserted_row:
                        inserted_detection_ids_and_plates.append({'id': inserted_row[0], 'license_plate': inserted_row[1]})

                conn.commit() # Commit after all plate detections are inserted
                rows_affected = len(inserted_detection_ids_and_plates)

                logger.info(f"Database commit for detected_plates successful. Rows affected: {rows_affected}")
                if rows_affected > 0:
                    saved_successfully = True

                # --- Check for Watchlist Matches and Create Alerts ---
                if saved_successfully and inserted_detection_ids_and_plates:
                    logger.info("Checking for watchlist matches for newly inserted detections...")
                    for det_info in inserted_detection_ids_and_plates:
                        detection_id = det_info['id']
                        detected_plate_text = det_info['license_plate']

                        # Find matching watchlist entries for the detected plate
                        cursor.execute("""
                            SELECT wl.id as watchlist_id, wle.id as watchlist_entry_id, wl.name as watchlist_name
                            FROM watchlist_entries wle
                            JOIN watchlists wl ON wle.watchlist_id = wl.id
                            WHERE wl.is_active = TRUE AND wle.license_plate = %s;
                        """, (detected_plate_text,))
                        matches = cursor.fetchall()

                        if matches:
                            for match in matches:
                                watchlist_id, watchlist_entry_id, watchlist_name = match
                                try:
                                    cursor.execute("""
                                        INSERT INTO alerts (detection_id, watchlist_id, watchlist_entry_id)
                                        VALUES (%s, %s, %s)
                                        ON CONFLICT (detection_id, watchlist_entry_id) DO NOTHING;
                                    """, (detection_id, watchlist_id, watchlist_entry_id))
                                    logger.info(f"ALERT Generated: Plate '{detected_plate_text}' (Detection ID: {detection_id}) matched watchlist '{watchlist_name}' (Entry ID: {watchlist_entry_id}).")

                                    # --- Emit SocketIO Alert --- Start ---
                                    if self.socketio:
                                        # Find the original detection data for this detection_id to get image_filename
                                        original_detection_details = next((item for item in valid_detections_for_db if item.get('temp_detection_id') == detection_id), None)
                                        if original_detection_details is None:
                                            # Fallback: search in `detections` list if `valid_detections_for_db` was filtered
                                            # This requires passing the original `detections` list to this function or a more complex lookup.
                                            # For now, we assume `image_filename` would be associated if it reached this point.
                                            # We might need to rethink how `image_filename` is retrieved for the socket event if it's not in `valid_detections_for_db`.
                                            # Let's try to get it from the record used for insertion.
                                            # The `record` variable from the loop `for record in records_to_insert:` would have been the one that resulted in `detection_id`
                                            # This is getting a bit complex due to how `detection_id` is obtained AFTER insertion.

                                            # Simplified approach: Look up the image_filename and other details for the alert payload directly from the DB record.
                                            # This ensures we have the correct context for the alert.
                                            alert_payload_cursor = conn.cursor() # New cursor for this lookup
                                            alert_payload_cursor.execute("""
                                                SELECT dp.license_plate, dp.image_filename, dp.id as detection_id,
                                                       wl.name as watchlist_name, wle.reason as watchlist_reason
                                                FROM detected_plates dp, watchlists wl, watchlist_entries wle
                                                WHERE dp.id = %s AND wl.id = %s AND wle.id = %s
                                            """, (detection_id, watchlist_id, watchlist_entry_id))
                                            payload_data = alert_payload_cursor.fetchone()
                                            alert_payload_cursor.close()

                                            if payload_data:
                                                alert_payload = {
                                                    'plate_text': payload_data[0],
                                                    'watchlist_name': payload_data[3],
                                                    'reason': payload_data[4],
                                                    'alert_time': datetime.now().isoformat(),
                                                    'detection_id': payload_data[2],
                                                    'plate_image_filename': payload_data[1], # This is the key part
                                                    # Client can construct full URLs using base paths known to it
                                                    # 'detection_url': f'/index#detection-{payload_data[2]}'
                                                }
                                                self.socketio.emit('new_alert', alert_payload, room='admins_room')
                                                logger.info(f"SocketIO event 'new_alert' emitted for plate {alert_payload['plate_text']} to admins_room.")
                                            else:
                                                logger.warning(f"Could not retrieve full payload data for alert related to detection ID {detection_id} for SocketIO emit.")
                                    # --- Emit SocketIO Alert --- End ---

                                except psycopg2.Error as alert_e:
                                    logger.error(f"Error inserting alert for detection {detection_id}, plate {detected_plate_text}, watchlist entry {watchlist_entry_id}: {alert_e}", exc_info=True)
                                    # conn.rollback() # Might not want to rollback everything for an alert error
                        else:
                            logger.debug(f"No active watchlist match for plate '{detected_plate_text}' (Detection ID: {detection_id}).")
                        conn.commit() # Commit any alerts created
                    # ----------------------------------------------------

        except Exception as e:
            logger.error(f"Database error: {str(e)}", exc_info=True)
            if conn:
                conn.rollback()
            saved_successfully = False
        finally:
            if conn:
                self.db_pool.putconn(conn)

        return saved_successfully

    #----------------------------------------------------------------------------------------------
    # Process Image
    #----------------------------------------------------------------------------------------------
    def process_image(self, frame):
        """Process a single image"""
        processed_detections = []
        car_brand_overall = "Unknown" # Initialize here
        try:
            time_details = self.get_time_details()
            vehicles = self.detect_vehicle_type(frame)

            image_for_brand_detection = frame
            if vehicles:
                largest_vehicle_roi = None
                max_area = 0
                for v_info in vehicles:
                    v_box = v_info['box']
                    v_area = (v_box[2] - v_box[0]) * (v_box[3] - v_box[1])
                    if v_area > max_area:
                        max_area = v_area
                        largest_vehicle_roi = frame[v_box[1]:v_box[3], v_box[0]:v_box[2]]
                if largest_vehicle_roi is not None and largest_vehicle_roi.size > 0:
                    image_for_brand_detection = largest_vehicle_roi
                    logger.info("Using largest vehicle ROI for car brand detection.")
                else:
                    logger.info("No suitable vehicle ROI found, using full frame for car brand detection.")
            else:
                logger.info("No vehicles detected, using full frame for car brand detection.")

            if image_for_brand_detection.size > 0:
                car_brand_overall = self._detect_car_brand_with_provider(image_for_brand_detection, None)
            else:
                logger.warning("Image for brand detection is empty.")

            results = self.model.predict(frame, conf=MIN_CONFIDENCE, verbose=False)

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                logger.info(f"Detected {len(boxes)} potential plates in image")
                confidences = result.boxes.conf.cpu().numpy()

                for box, conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    plate_img = frame[y1:y2, x1:x2]

                    if plate_img is None or plate_img.size == 0:
                        logger.warning(f"Skipping empty plate image crop at box: {(x1, y1, x2, y2)}")
                        continue

                    plate_text, plate_conf = self.ocr_license_plate(plate_img, None)
                    logger.info(f"OCR result: '{plate_text}' with confidence {plate_conf:.2f}")

                    if plate_text and plate_conf >= MIN_CONFIDENCE:
                        vehicle_type = "unknown"
                        vehicle_color = "unknown"

                        plate_center_x = (x1 + x2) / 2
                        plate_center_y = (y1 + y2) / 2
                        associated_vehicle = None
                        min_area = float('inf')

                        for v in vehicles:
                            vx1_v, vy1_v, vx2_v, vy2_v = v['box']
                            if vx1_v <= plate_center_x <= vx2_v and vy1_v <= plate_center_y <= vy2_v:
                                area = (vx2_v - vx1_v) * (vy2_v - vy1_v)
                                if area < min_area:
                                    min_area = area
                                    associated_vehicle = v

                        if associated_vehicle:
                             v_assoc = associated_vehicle
                             vehicle_type = v_assoc['type']
                             vx1_assoc, vy1_assoc, vx2_assoc, vy2_assoc = v_assoc['box']
                             vehicle_roi_color = frame[vy1_assoc:vy2_assoc, vx1_assoc:vx2_assoc]
                             if vehicle_roi_color.size > 0:
                                 vehicle_color = self.predict_vehicle_color(vehicle_roi_color)
                             else:
                                 logger.warning("Could not extract valid vehicle ROI for color prediction for an associated vehicle.")

                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                        image_filename = f"{plate_text}_{timestamp_str}.jpg"
                        save_path = PLATE_IMAGE_DIR / image_filename
                        saved_image_path = None
                        try:
                            success = cv2.imwrite(str(save_path), plate_img)
                            if success:
                                saved_image_path = image_filename
                                logger.info(f"Saved detected plate image: {image_filename}")
                            else:
                                logger.warning(f"Failed to save plate image: {save_path}")
                        except Exception as img_save_error:
                            logger.error(f"Error saving plate image {save_path}: {img_save_error}", exc_info=True)

                        detection_data = {
                            'plate_text': plate_text,
                            'plate_confidence': plate_conf,
                            'vehicle_type': vehicle_type,
                            'vehicle_color': vehicle_color,
                            'car_brand': car_brand_overall,
                            'time_details': time_details,
                            'image_filename': saved_image_path,
                            'bounding_box': (x1, y1, x2, y2)
                        }
                        processed_detections.append(detection_data)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{plate_text} ({plate_conf:.2f})",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 0), 2)

        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)

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

            # Process the single image - Get annotated frame and detections
            annotated_image, detections = self.process_image(image)

            # --- Save Annotated Frame First --- Start
            saved_annotated_filename = None # Initialize
            try:
                # Attempt to save the annotated image regardless of DB success first
                saved_annotated_filename = img_processor.save_image(annotated_image)
                if saved_annotated_filename:
                    logger.info(f"Saved annotated frame: {saved_annotated_filename}")
                else:
                    logger.warning("Failed to save annotated frame.")
            except Exception as img_save_err:
                 logger.error(f"Error saving annotated frame BEFORE DB attempt: {img_save_err}", exc_info=True)
            # --- Save Annotated Frame First --- End

            # --- Trigger Database Save (passing the filename) --- Start
            if detections:
                logger.info(f"--- Triggering Database Save for {len(detections)} potential detections (Annotated Frame: {saved_annotated_filename or 'Not Saved'}) ---")
                # Call save_to_database, passing the filename we just tried to save
                was_saved_to_db = self.save_to_database(detections, annotated_frame_filename=saved_annotated_filename)

                if was_saved_to_db:
                    logger.info("Database save reported success.")
                # No need to save image again here, it was attempted above
                # else:
                #     logger.info("No valid detections were saved to the database.")
            else:
                logger.info("No potential plates detected in the image to attempt saving.")
            # --- Trigger Database Save (passing the filename) --- End

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

    # Validate sources
    valid_sources = []
    if not args.source: # Should not happen due to required=True, but good check
        logger.error("No source image(s) provided. Use --source <path1> [<path2> ...]")
        sys.exit(1)

    for src_path_str in args.source:
        if src_path_str.isdigit():
            logger.error(f"Camera index '{src_path_str}' detected. This script is for image processing only. Please provide image file paths.")
            # Continue to check other sources, but at least one error means we might exit later
            # For now, we'll just skip this one and try to process others if any.
        elif not Path(src_path_str).is_file():
            logger.error(f"Source is not a file or does not exist: {src_path_str}")
        else:
            valid_sources.append(src_path_str)

    if not valid_sources:
        logger.error("No valid image file sources found to process after validation.")
        sys.exit(1)

    logger.info(f"Starting ANPR image processing system for {len(valid_sources)} source(s): {valid_sources}")
    processor = None # Initialize processor to None for finally block
    try:
        processor = ANPRProcessor(socketio_instance=None)
        for idx, source_image_path in enumerate(valid_sources):
            logger.info(f"--- Processing image {idx + 1} of {len(valid_sources)}: {source_image_path} ---")
            try:
                processor.process_image_source(source_image_path)
            except Exception as e_process_single:
                # Log error for this specific image but continue with the next ones
                logger.error(f"Failed to process image '{source_image_path}': {str(e_process_single)}", exc_info=True)
            logger.info(f"--- Finished processing image {idx + 1} of {len(valid_sources)}: {source_image_path} ---")

    except KeyboardInterrupt:
        logger.info("Shutting down due to KeyboardInterrupt...")
    except Exception as e_main:
        logger.error(f"Main ANPR processing loop failed: {str(e_main)}", exc_info=True)
    finally:
        if hasattr(processor, 'db_pool') and processor.db_pool:
            processor.db_pool.closeall()
            logger.info("Database connections closed.")
        else:
            logger.warning("Processor or DB pool not fully initialized, or already closed. Skipping closeall.")

if __name__ == "__main__":
    main()


