import eventlet
eventlet.monkey_patch()

import os
import sys
from pathlib import Path
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from flask import Flask, render_template, jsonify, g, send_from_directory, abort, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime as dt
import math
import traceback

# Import ANPRProcessor - Use relative import
# from .anpr_image import ANPRProcessor
from .anpr_image import ANPRProcessor

# Live camera processing
import cv2
from ultralytics import YOLO
import threading
import time
import base64
from datetime import datetime
from pathlib import Path

# Global camera variables
camera_cap = None
camera_thread = None
camera_running = False
camera_stats = {
    'start_time': None,
    'frames_processed': 0,
    'plates_detected': 0,
    'last_detection_time': None,
    'fps': 0
}

# Global auto-detection variables
auto_detection_running = False
auto_detection_thread = None
auto_detection_stats = {
    'start_time': None,
    'vehicles_detected': 0,
    'plates_processed': 0,
    'last_vehicle_time': None,
    'detection_cooldown': 10,  # seconds between vehicle detections
    'vehicle_confidence_threshold': 0.7,
    'last_processed_vehicles': {},  # track recent vehicle detections to avoid duplicates
}

class AutoDetectionManager:
    """Manages intelligent automatic vehicle detection and ANPR processing with bulletproof duplicate prevention"""

    def __init__(self, anpr_processor, socketio_instance):
        self.anpr_processor = anpr_processor
        self.socketio = socketio_instance
        self.vehicle_model = YOLO("models/yolov8n.pt")  # Lighter model for real-time detection
        self.running = False

        # Intelligent detection settings (optimized for cost efficiency)
        self.detection_cooldown = 8  # seconds between detections (reduced from 10)
        self.vehicle_confidence = 0.65  # slightly lower confidence for better coverage
        self.last_detection_time = 0

        # BULLETPROOF Enhanced duplicate prevention
        self.processed_vehicles = {}  # vehicle_fingerprint -> {timestamp, plate_results}
        self.session_plates = {}  # plate_text -> {timestamp, vehicle_fingerprint, processed_count}
        self.session_id = None  # Unique session identifier
        self.vehicle_memory_duration = 60  # Remember vehicles for 60 seconds
        self.plate_memory_duration = 600  # Remember plates for 10 minutes (increased from 5)

        # Performance optimization
        self.frame_skip_counter = 0
        self.frame_skip_interval = 3  # Process every 3rd frame for efficiency

        # Logging enhancement
        self.detection_stats = {
            'total_vehicles_seen': 0,
            'unique_vehicles_processed': 0,
            'duplicate_vehicles_blocked': 0,
            'unique_plates_found': 0,
            'duplicate_plates_blocked': 0,
            'database_saves': 0
        }

    def detect_vehicles_in_frame(self, frame):
        """Detect vehicles in frame using YOLOv8"""
        try:
            # Vehicle classes from COCO dataset that we're interested in
            vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

            results = self.vehicle_model.predict(frame, conf=self.vehicle_confidence, verbose=False)
            vehicles = []

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()

                for box, cls, conf in zip(boxes, classes, confs):
                    cls_id = int(cls)
                    if cls_id in vehicle_classes:
                        x1, y1, x2, y2 = map(int, box)
                        vehicles.append({
                            'box': (x1, y1, x2, y2),
                            'confidence': float(conf),
                            'class': cls_id,
                            'area': (x2 - x1) * (y2 - y1)
                        })

            return vehicles

        except Exception as e:
            logger.error(f"Error detecting vehicles: {e}")
            return []

    def generate_vehicle_fingerprint(self, vehicle, frame_shape):
        """Generate a unique fingerprint for a vehicle based on position, size, and characteristics"""
        x1, y1, x2, y2 = vehicle['box']
        frame_height, frame_width = frame_shape[:2]

        # Normalize position and size relative to frame
        center_x = (x1 + x2) / 2 / frame_width
        center_y = (y1 + y2) / 2 / frame_height
        width_ratio = (x2 - x1) / frame_width
        height_ratio = (y2 - y1) / frame_height

        # Create fingerprint with reduced precision for grouping similar vehicles
        fingerprint = f"{center_x:.2f}_{center_y:.2f}_{width_ratio:.2f}_{height_ratio:.2f}_{vehicle['class']}"
        return fingerprint

    def should_process_vehicle(self, vehicle, frame_shape):
        """Intelligent vehicle processing decision with enhanced duplicate prevention"""
        current_time = time.time()

        # Check global cooldown
        if current_time - self.last_detection_time < self.detection_cooldown:
            return False

        # Generate vehicle fingerprint
        vehicle_fingerprint = self.generate_vehicle_fingerprint(vehicle, frame_shape)

        # Check if we've processed this vehicle recently
        if vehicle_fingerprint in self.processed_vehicles:
            vehicle_data = self.processed_vehicles[vehicle_fingerprint]
            last_time = vehicle_data['timestamp']

            # If vehicle was seen recently, skip processing
            if current_time - last_time < self.vehicle_memory_duration:
                logger.info(f"🚫 VEHICLE DUPLICATE BLOCKED: {vehicle_fingerprint[:12]} was seen {int(current_time - last_time)}s ago")
                self.detection_stats['duplicate_vehicles_blocked'] += 1
                return False

            # If vehicle had successful plate detection, wait longer before reprocessing
            if vehicle_data.get('had_plates', False):
                if current_time - last_time < self.vehicle_memory_duration * 2:
                    logger.info(f"🚫 VEHICLE WITH PLATES BLOCKED: {vehicle_fingerprint[:12]} had plates {int(current_time - last_time)}s ago")
                    self.detection_stats['duplicate_vehicles_blocked'] += 1
                    return False

        # Vehicle passed all checks, mark for processing
        self.last_detection_time = current_time
        logger.info(f"✅ VEHICLE ACCEPTED: {vehicle_fingerprint[:12]} passed all duplicate checks - processing...")

        # Clean old entries to prevent memory growth
        self.cleanup_old_entries(current_time)

        return True

    def cleanup_old_entries(self, current_time):
        """Clean up old vehicle and plate tracking entries with enhanced logging"""
        before_vehicles = len(self.processed_vehicles)
        before_plates = len(self.session_plates)

        # Clean old vehicle entries
        self.processed_vehicles = {
            fingerprint: data for fingerprint, data in self.processed_vehicles.items()
            if current_time - data['timestamp'] < self.vehicle_memory_duration * 3
        }

        # Clean old session plates (keep for longer to prevent duplicates)
        self.session_plates = {
            plate_text: data for plate_text, data in self.session_plates.items()
            if current_time - data['timestamp'] <= self.plate_memory_duration
        }

        # Log cleanup if any entries were removed
        vehicles_cleaned = before_vehicles - len(self.processed_vehicles)
        plates_cleaned = before_plates - len(self.session_plates)

        if vehicles_cleaned > 0 or plates_cleaned > 0:
            logger.info(f"Session cleanup: Removed {vehicles_cleaned} old vehicles, {plates_cleaned} old plates. "
                       f"Active: {len(self.processed_vehicles)} vehicles, {len(self.session_plates)} plates")

    def is_plate_duplicate_in_session(self, plate_text):
        """Check if plate has already been processed in current session with detailed logging"""
        plate_text_clean = plate_text.strip().upper()

        if plate_text_clean in self.session_plates:
            plate_data = self.session_plates[plate_text_clean]
            logger.warning(f"🚫 DUPLICATE PLATE BLOCKED: '{plate_text_clean}' was already processed at "
                          f"{datetime.fromtimestamp(plate_data['timestamp']).strftime('%H:%M:%S')} "
                          f"(Session ID: {self.session_id})")
            self.detection_stats['duplicate_plates_blocked'] += 1
            return True

        return False

    def add_plate_to_session(self, plate_text, vehicle_fingerprint):
        """Add plate to session tracking with timestamp and logging"""
        plate_text_clean = plate_text.strip().upper()
        current_time = time.time()

        self.session_plates[plate_text_clean] = {
            'timestamp': current_time,
            'vehicle_fingerprint': vehicle_fingerprint,
            'processed_count': self.session_plates.get(plate_text_clean, {}).get('processed_count', 0) + 1
        }

        logger.info(f"✅ NEW PLATE ADDED TO SESSION: '{plate_text_clean}' "
                   f"(Session ID: {self.session_id}, Total unique plates: {len(self.session_plates)})")
        self.detection_stats['unique_plates_found'] += 1

    def process_detected_vehicle(self, frame, vehicle):
        """Process frame through ANPR pipeline with intelligent duplicate prevention"""
        try:
            current_time = time.time()
            vehicle_fingerprint = self.generate_vehicle_fingerprint(vehicle, frame.shape)

            logger.info(f"Auto-processing vehicle detection with confidence {vehicle['confidence']:.2f}")

            # Emit real-time update to connected clients
            if self.socketio:
                self.socketio.emit('vehicle_detected', {
                    'confidence': vehicle['confidence'],
                    'timestamp': datetime.now().isoformat(),
                    'box': vehicle['box'],
                    'class': vehicle['class']
                }, room='admins_room')

            # Update stats
            self.detection_stats['total_vehicles_seen'] += 1

            # Process through existing ANPR system
            logger.info(f"🔍 Running ANPR pipeline on vehicle {vehicle_fingerprint[:12]}...")
            annotated_frame, detections = self.anpr_processor.process_image(frame)

            # BULLETPROOF duplicate filtering with detailed logging
            unique_detections = []
            plates_processed = 0
            plates_blocked = 0

            if detections:
                logger.info(f"📋 ANPR found {len(detections)} potential plates, checking for duplicates...")

                for detection in detections:
                    plate_text = detection.get('license_plate', '').strip().upper()
                    plates_processed += 1

                    if not plate_text:
                        logger.warning(f"⚠️ Empty plate text in detection {plates_processed}")
                        continue

                    # BULLETPROOF duplicate check
                    if self.is_plate_duplicate_in_session(plate_text):
                        plates_blocked += 1
                        logger.warning(f"🚫 BLOCKED: Plate '{plate_text}' already exists in session")
                        continue

                    # This is a genuinely new plate
                    unique_detections.append(detection)
                    self.add_plate_to_session(plate_text, vehicle_fingerprint)
                    logger.info(f"✅ ACCEPTED: New unique plate '{plate_text}' (#{len(unique_detections)})")

                logger.info(f"📊 FILTERING RESULT: {len(unique_detections)} unique, {plates_blocked} duplicates blocked")
            else:
                logger.info("📋 ANPR found no license plates in vehicle")

            # Update vehicle tracking with results
            self.processed_vehicles[vehicle_fingerprint] = {
                'timestamp': current_time,
                'had_plates': len(unique_detections) > 0,
                'plates_count': len(unique_detections)
            }

            # Update internal stats
            self.detection_stats['unique_vehicles_processed'] += 1
            if len(unique_detections) > 0:
                self.detection_stats['duplicate_vehicles_blocked'] += (plates_blocked if plates_blocked > 0 else 0)

            # BULLETPROOF: Only save to database if we have genuinely new unique plates
            detection_results = {
                'success': True,
                'detections_count': len(unique_detections),
                'detections': unique_detections,
                'annotated_image': None,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
                'auto_detected': True,
                'duplicate_prevention': True,
                'session_id': self.session_id
            }

            if unique_detections:
                # ONLY save if we have new unique plates
                logger.info(f"💾 SAVING TO DATABASE: {len(unique_detections)} unique plates from session {self.session_id}")

                # Generate filename for annotated frame only if we have unique detections
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                annotated_filename = f"auto_{self.session_id}_{timestamp}.jpg"

                # Save annotated frame
                output_dir = Path("output_plates")
                output_dir.mkdir(exist_ok=True)
                annotated_path = output_dir / annotated_filename

                try:
                    success = cv2.imwrite(str(annotated_path), annotated_frame)
                    if success:
                        detection_results['annotated_image'] = annotated_filename
                        logger.info(f"💾 Saved annotated frame: {annotated_filename}")
                except Exception as e:
                    logger.error(f"❌ Error saving annotated frame: {e}")

                try:
                    # BULLETPROOF: Final check before database save
                    logger.info(f"💾 ATTEMPTING DATABASE SAVE for plates: {[d.get('license_plate', 'UNKNOWN') for d in unique_detections]}")
                    self.anpr_processor.save_to_database(unique_detections, annotated_frame_filename=annotated_filename)

                    # Update stats only on successful save
                    auto_detection_stats['plates_processed'] += len(unique_detections)
                    self.detection_stats['database_saves'] += 1

                    logger.info(f"✅ DATABASE SAVE SUCCESSFUL: {len(unique_detections)} plates saved to database "
                               f"(Session: {self.session_id}, Total saves: {self.detection_stats['database_saves']})")

                    # Emit successful detection to clients
                    if self.socketio:
                        self.socketio.emit('anpr_detection', detection_results, room='admins_room')

                except Exception as e:
                    logger.error(f"❌ DATABASE SAVE FAILED: {e}")
                    # Remove from session tracking if database save failed
                    for detection in unique_detections:
                        plate_text = detection.get('license_plate', '').strip().upper()
                        if plate_text in self.session_plates:
                            del self.session_plates[plate_text]
                            logger.warning(f"🔄 Removed '{plate_text}' from session due to DB save failure")
            else:
                if detections:
                    logger.warning(f"🚫 NO DATABASE SAVE: All {len(detections)} plates were duplicates in session {self.session_id}")
                else:
                    logger.info(f"ℹ️ NO DATABASE SAVE: No license plates found in vehicle")

            # Update global stats
            auto_detection_stats['vehicles_detected'] += 1
            auto_detection_stats['last_vehicle_time'] = datetime.now().isoformat()

            # Log session summary
            logger.info(f"📊 SESSION SUMMARY: Unique vehicles: {len(self.processed_vehicles)}, "
                       f"Unique plates: {len(self.session_plates)}, DB saves: {self.detection_stats['database_saves']}")

            return detection_results

        except Exception as e:
            logger.error(f"Error processing detected vehicle: {e}")
            return {'error': f'Auto-processing failed: {str(e)}'}

    def auto_detection_loop(self, camera_cap):
        """Intelligent auto-detection loop with frame skipping and optimized processing"""
        global auto_detection_running

        logger.info(f"🚀 STARTING BULLETPROOF AUTO-DETECTION: Session {self.session_id}")
        logger.info(f"🛡️ DUPLICATE PREVENTION: Vehicle memory={self.vehicle_memory_duration}s, Plate memory={self.plate_memory_duration}s")
        logger.info(f"⚙️ PERFORMANCE SETTINGS: Frame skip={self.frame_skip_interval}, Vehicle confidence={self.vehicle_confidence}")

        while auto_detection_running and camera_cap:
            try:
                if not camera_cap.isOpened():
                    time.sleep(0.1)
                    continue

                # Capture frame
                ret, frame = camera_cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue

                # Frame skipping for performance optimization
                self.frame_skip_counter += 1
                if self.frame_skip_counter < self.frame_skip_interval:
                    time.sleep(0.1)
                    continue

                self.frame_skip_counter = 0

                # Detect vehicles
                vehicles = self.detect_vehicles_in_frame(frame)

                # Process largest vehicle if any detected
                if vehicles:
                    # Sort by area to get largest vehicle
                    largest_vehicle = max(vehicles, key=lambda v: v['area'])

                    if self.should_process_vehicle(largest_vehicle, frame.shape):
                        self.process_detected_vehicle(frame, largest_vehicle)

                # Adaptive delay - smaller when vehicles are present
                if vehicles:
                    time.sleep(0.3)  # Faster checking when vehicles detected
                else:
                    time.sleep(0.7)  # Slower when no vehicles

            except Exception as e:
                logger.error(f"Error in auto-detection loop: {e}")
                time.sleep(1)  # Longer delay on error

        logger.info("Auto vehicle detection loop stopped")

# --- Flask-Login Imports ---
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
# -------------------------

# --- Flask-WTF Imports (for forms) ---
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect, generate_csrf
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired, Length, EqualTo, ValidationError
# --- Flask-SocketIO Imports ---
from flask_socketio import SocketIO, join_room, leave_room
# -------------------------------------

# --- functools for wraps ---
from functools import wraps
# ---------------------------

# --- Load .env file early ---
load_dotenv() # Call load_dotenv() at the top
# --------------------------

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from config.appconfig import (
    DB_HOST,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
    DB_PORT
)

# --- Path Setup ---
# Get the directory containing this script (src/)
SCRIPT_DIR = Path(__file__).parent.resolve()
# Project root is one level up from src/
PROJECT_ROOT = SCRIPT_DIR.parent
# Directory for annotated full frames (used for gallery)
OUTPUT_DIR = PROJECT_ROOT / "output_plates"
# Directory for cropped plate images (referenced in DB, served by serve_plate_image)
PLATE_IMAGE_DIR = OUTPUT_DIR / "plate_images"
# Directory for temporary uploads
UPLOADS_DIR = PROJECT_ROOT / "uploads"

# Ensure the directories exist (optional here, as anpr_image should create them)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLATE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging Setup ---
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True) # Ensure logs directory exists
LOG_FILE_PATH = LOGS_DIR / "web.log"

# Get a specific logger instance for the web app
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ANPR_WebApp") # Specific name for this logger
logger.setLevel(logging.INFO)  # Use INFO level for better performance

# Remove existing handlers if any (important for repeated runs/debugging)
if logger.hasHandlers():
    logger.handlers.clear()

# Console Handler (optional, but good for seeing logs during development)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# File Handler (rotating log file)
# Rotate log file when it reaches 1MB, keep 3 backup files
file_handler = RotatingFileHandler(LOG_FILE_PATH, maxBytes=1024*1024, backupCount=3)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# --- Configure Werkzeug Logger ---
werkzeug_logger = logging.getLogger('werkzeug')
# Clear existing handlers if any (to avoid duplicates or unwanted destinations)
werkzeug_logger.handlers.clear()
# Set level (INFO captures request/response logs)
werkzeug_logger.setLevel(logging.INFO)
# Add the same handlers used by the main app logger
werkzeug_logger.addHandler(stream_handler) # To console
werkzeug_logger.addHandler(file_handler)   # To web.log
# Prevent werkzeug logs from propagating to the root logger
# This stops them from potentially being caught by handlers configured elsewhere (e.g., in config.log)
werkzeug_logger.propagate = False
logger.info("Werkzeug logger configured to use web app handlers and disable propagation.")
# -------------------------------

logger.info("--- ANPR Web App Starting --- ")
logger.info(f"Logging configured. Log file: {LOG_FILE_PATH}")
logger.info(f"Uploads directory: {UPLOADS_DIR}")

logger.info(f"DB Config: Host={DB_HOST}, DB={DB_NAME}, User={DB_USER}, Port={DB_PORT}")

# --- Database Connection Pool ---
# Using a pool is more efficient than opening/closing connections for each request
try:
    db_pool = SimpleConnectionPool(
        minconn=1,
        maxconn=5, # Adjust max connections as needed
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT,
        connect_timeout=5
    )
    logger.info("Database connection pool created successfully.")
except Exception as e:
    logger.error(f"Error creating database connection pool: {e}", exc_info=True)
    db_pool = None # Set pool to None if connection fails

# --- Flask App Setup ---
# Since web_app.py is in src/, templates and static are expected
# to be in src/templates/ and src/static/ relative to the script's location.
TEMPLATE_DIR = SCRIPT_DIR / 'templates'
STATIC_DIR = SCRIPT_DIR / 'static'

logger.info(f"Template folder: {TEMPLATE_DIR}")
logger.info(f"Static folder: {STATIC_DIR}")

# Check if template/static folders exist (helps debugging)
if not TEMPLATE_DIR.is_dir():
    logger.warning(f"Template directory NOT found: {TEMPLATE_DIR}")
if not STATIC_DIR.is_dir():
    logger.warning(f"Static directory NOT found: {STATIC_DIR}")

# We can explicitly set the folders, or rely on Flask's default behavior
# when the app script is in src/ and templates/static are also in src/
# Explicit is clearer:
app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))

# --- CSRF Protection Setup ---
csrf = CSRFProtect(app)
# ---------------------------

# --- Flask-SocketIO Setup ---
socketio = SocketIO(
    app,
    async_mode='eventlet',
    cors_allowed_origins="*",
    ping_timeout=30,
    ping_interval=10,
    logger=False,
    engineio_logger=False,
    transports=['polling', 'websocket']
) # Simplified Socket.IO configuration for better stability
# ----------------------------

# --- Flask-Login Setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # The route name for the login page
login_manager.login_message_category = 'info' # Bootstrap class for flash messages
# -------------------------

# Configure a secret key for flashing messages (optional but good practice)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_default_secret_key_for_dev')

# --- Forms ---
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=80)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Login')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=80)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6, message='Password must be at least 6 characters long.')])
    confirm_password = PasswordField('Confirm Password',
                                   validators=[DataRequired(), EqualTo('password', message='Passwords must match.')])
    submit = SubmitField('Register')

    def validate_username(self, username):
        conn = get_db()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT id FROM users WHERE username = %s", (username.data,))
                    user_exists = cur.fetchone()
                    if user_exists:
                        raise ValidationError('That username is already taken. Please choose a different one.')
            except psycopg2.Error as e:
                logger.error(f"Registration form: Database error during username validation: {e}", exc_info=True)
                # Let the registration proceed, but log the error. Or raise a generic validation error.
                # For now, let's inform the user subtly, or rely on unique constraint of DB if not caught here.
                # raise ValidationError('Could not validate username due to a server issue. Please try again.')
            # Connection is handled by get_db and teardown
        else:
            logger.error("Registration form: Database connection not available for username validation.")
            # raise ValidationError('Username validation service temporarily unavailable.')

# --- Custom Decorators for Access Control ---
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            flash('Please log in to access this page.', 'info')
            return redirect(url_for('login', next=request.url))
        if not current_user.is_admin():
            flash('You do not have permission to access this page. Admin access required.', 'danger')
            return redirect(url_for('index')) # Or wherever you want to redirect non-admins
        return f(*args, **kwargs)
    return decorated_function
# -----------------------------------------

# --- Watchlist Forms ---
class WatchlistForm(FlaskForm):
    name = StringField('Watchlist Name', validators=[DataRequired(), Length(min=3, max=100)])
    description = StringField('Description (Optional)', validators=[Length(max=255)])
    is_active = BooleanField('Active', default=True)
    submit = SubmitField('Save Watchlist')

class WatchlistEntryForm(FlaskForm):
    license_plate = StringField('License Plate', validators=[
        DataRequired(),
        Length(min=3, max=20, message='License plate must be between 3 and 20 characters.'),
        # Basic regex for typical plate characters, can be made stricter
        # Regexp(r'^[A-Z0-9-]{3,20}$', message='Invalid characters. Use A-Z, 0-9, and hyphens.')
        # For now, let's keep it simple. Specific regex can be added later based on Nigerian plate format.
    ])
    reason = StringField('Reason (Optional)', validators=[Length(max=255)])
    submit = SubmitField('Add Plate to Watchlist')
# ----------------------

# Configure allowed extensions for upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to get a connection from the pool
def get_db():
    # Use the specific logger
    if 'db' not in g and db_pool:
        try:
            g.db = db_pool.getconn()
            # logger.debug("Acquired DB connection from pool.") # Can be noisy
        except Exception as e:
            logger.error(f"Failed to get DB connection from pool: {e}", exc_info=True)
            g.db = None
    return g.get('db', None)

# Helper function to close the connection when the request context ends
@app.teardown_appcontext
def close_db(error):
    # Use the specific logger
    db = g.pop('db', None)
    if db is not None and db_pool:
        db_pool.putconn(db)
        # logger.debug("Returned DB connection to pool.") # Can be noisy
    if error:
        # Log the error passed from Flask during teardown
        logger.error(f"App context teardown error: {error}", exc_info=True)

# --- Initialize ANPR Processor ---
anpr_processor = None
auto_detection_manager = None
try:
    # Instantiate it once when the app starts, passing the socketio instance
    anpr_processor = ANPRProcessor(socketio_instance=socketio)
    logger.info("ANPRProcessor initialized successfully with SocketIO instance.")

    # Initialize auto-detection manager
    auto_detection_manager = AutoDetectionManager(anpr_processor, socketio)
    logger.info("Auto-detection manager initialized successfully.")
except Exception as e:
    logger.error(f"CRITICAL: Failed to initialize ANPRProcessor: {e}", exc_info=True)
    # The app might still run but uploads will fail. Consider if app should exit.

# --- User Model and Database ---
class User(UserMixin):
    def __init__(self, id, username, role='user'):
        self.id = id
        self.username = username
        self.role = role
        # Password hash will be set separately when creating/fetching user

    # Flask-Login expects a get_id method
    def get_id(self):
        return str(self.id)

    # Role-based access helper
    def is_admin(self):
        return self.role == 'admin'

# Dummy user store for now - will be replaced by database interaction
# users_db = {} # Example: {1: {'username': 'admin', 'password_hash': 'hashed_pw', 'role': 'admin'}}

@login_manager.user_loader
def load_user(user_id):
    # This function will query the database for the user
    conn = get_db()
    if not conn:
        logger.error("load_user: Database connection not available.")
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, username, password_hash, role FROM users WHERE id = %s", (int(user_id),))
            user_data = cur.fetchone()
            if user_data:
                user = User(id=user_data[0], username=user_data[1], role=user_data[3])
                logger.debug(f"User loaded: {user.username} (ID: {user.id}, Role: {user.role})")
                return user
            logger.debug(f"User with ID {user_id} not found.")
            return None
    except psycopg2.Error as e:
        logger.error(f"load_user: Database error: {e}", exc_info=True)
        return None
    except Exception as e: # General exception handler
        logger.error(f"load_user: Unexpected error: {e}", exc_info=True)
        return None


def _ensure_users_table_exists():
    """Creates the users table if it doesn't exist."""
    conn = None
    pool_to_use = db_pool # Use the global pool
    if not pool_to_use:
        logger.error("Cannot ensure users table: Database pool not initialized.")
        return

    try:
        conn = pool_to_use.getconn()
        with conn.cursor() as cursor:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(80) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL, -- Increased length for modern hashes
                role VARCHAR(20) NOT NULL DEFAULT 'user', -- 'user' or 'admin'
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """
            cursor.execute(create_table_sql)
            conn.commit()
            logger.info("Table 'users' checked/created successfully.")

            # --- Create/Update default admin user ---
            default_admin_username = 'admin'
            default_admin_password_env = os.environ.get('ADMIN_PASSWORD')

            if not default_admin_password_env:
                logger.warning(f"ADMIN_PASSWORD not set in .env. Cannot create or update '{default_admin_username}'.")
            else:
                cursor.execute("SELECT id, password_hash FROM users WHERE username = %s", (default_admin_username,))
                admin_user_data = cursor.fetchone()

                if not admin_user_data:
                    # Admin user does not exist, create them
                    hashed_password = generate_password_hash(default_admin_password_env)
                    cursor.execute(
                        "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)",
                        (default_admin_username, hashed_password, 'admin')
                    )
                    conn.commit()
                    logger.info(f"Default admin user '{default_admin_username}' created. PLEASE CHANGE THE DEFAULT PASSWORD IF APPLICABLE and you haven't already.")
                else:
                    # Admin user exists, check if password needs updating
                    admin_id, stored_password_hash = admin_user_data
                    if not check_password_hash(stored_password_hash, default_admin_password_env):
                        # Password in .env is different from stored hash, update it
                        new_hashed_password = generate_password_hash(default_admin_password_env)
                        cursor.execute(
                            "UPDATE users SET password_hash = %s WHERE id = %s",
                            (new_hashed_password, admin_id)
                        )
                        conn.commit()
                        logger.info(f"Password for admin user '{default_admin_username}' has been updated from .env settings.")
                    else:
                        logger.info(f"Admin user '{default_admin_username}' already exists and password matches .env (or .env password unchanged).")
            # ----------------------------------------------------------

            # --- Create watchlists table ---
            create_watchlists_table_sql = """
            CREATE TABLE IF NOT EXISTS watchlists (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) UNIQUE NOT NULL,
                description TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            );
            """
            cursor.execute(create_watchlists_table_sql)
            conn.commit()
            logger.info("Table 'watchlists' checked/created successfully.")
            # --------------------------------

            # --- Create watchlist_entries table ---
            create_watchlist_entries_table_sql = """
            CREATE TABLE IF NOT EXISTS watchlist_entries (
                id SERIAL PRIMARY KEY,
                watchlist_id INTEGER NOT NULL REFERENCES watchlists(id) ON DELETE CASCADE,
                license_plate VARCHAR(20) NOT NULL, -- Should match format in detected_plates
                reason TEXT,
                added_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (watchlist_id, license_plate) -- Prevent duplicate plates in the same list
            );
            """
            cursor.execute(create_watchlist_entries_table_sql)
            conn.commit()
            logger.info("Table 'watchlist_entries' checked/created successfully.")
            # -------------------------------------

            # --- Create alerts table ---
            create_alerts_table_sql = """
            CREATE TABLE IF NOT EXISTS alerts (
                id SERIAL PRIMARY KEY,
                detection_id INTEGER NOT NULL REFERENCES detected_plates(id) ON DELETE CASCADE,
                watchlist_id INTEGER NOT NULL REFERENCES watchlists(id) ON DELETE CASCADE,
                watchlist_entry_id INTEGER NOT NULL REFERENCES watchlist_entries(id) ON DELETE CASCADE,
                alert_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                acknowledged_at TIMESTAMP WITH TIME ZONE,
                acknowledged_by_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                CONSTRAINT uq_alert_detection_entry UNIQUE (detection_id, watchlist_entry_id)
            );
            """
            cursor.execute(create_alerts_table_sql)
            conn.commit()
            logger.info("Table 'alerts' checked/created successfully.")
            # ---------------------------

    except Exception as e:
        logger.error(f"Database error during users table creation or admin user setup: {e}", exc_info=True)
        if conn:
            conn.rollback()
    finally:
        if conn:
            pool_to_use.putconn(conn)

# Call this function once at startup to ensure the table is there
_ensure_users_table_exists()
# -----------------------------

# --- Routes ---

@app.context_processor
def inject_datetime():
    """Make datetime module available in all templates."""
    return dict(datetime=dt)

@app.route('/about')
def about():
    """Renders the about page."""
    logger.info("Request received for about page ('/about').")
    return render_template('about.html', title='About ANPR System')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index')) # Redirect if already logged in

    form = RegistrationForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data

        hashed_password = generate_password_hash(password)

        conn = get_db()
        if conn:
            try:
                with conn.cursor() as cur:
                    # The validate_username method should catch existing usernames,
                    # but a final check or relying on DB unique constraint is also an option.
                    cur.execute("INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)",
                                (username, hashed_password, 'user'))
                    conn.commit()
                    logger.info(f"New user registered: {username}")
                    flash(f'Account created successfully for {username}! You can now log in.', 'success')
                    return redirect(url_for('login'))
            except psycopg2.IntegrityError: # Catch if username is somehow still a duplicate (unique constraint)
                conn.rollback()
                logger.warning(f"Registration failed for {username}: Username likely already exists (caught by DB constraint).")
                flash('That username is already taken. Please choose another.', 'danger')
            except psycopg2.Error as e:
                conn.rollback()
                logger.error(f"Registration: Database error for user {username}: {e}", exc_info=True)
                flash('Registration failed due to a database error. Please try again later.', 'danger')
        else:
            logger.error("Registration: Database connection not available.")
            flash('Registration service temporarily unavailable. Please try again later.', 'danger')

    return render_template('register.html', title='Register', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index')) # Redirect if already logged in

    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        remember = form.remember_me.data

        conn = get_db()
        user_object = None
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT id, username, password_hash, role FROM users WHERE username = %s", (username,))
                    user_data = cur.fetchone()
                    if user_data and check_password_hash(user_data[2], password):
                        user_object = User(id=user_data[0], username=user_data[1], role=user_data[3])
                        logger.info(f"Login successful for user: {username}")
                    else:
                        logger.warning(f"Login failed for user: {username} - Invalid credentials")
            except psycopg2.Error as e:
                logger.error(f"Login: Database error for user {username}: {e}", exc_info=True)
                flash('Login failed due to a database error. Please try again later.', 'danger')
                return render_template('login.html', title='Login', form=form)
            # No finally block to close connection here, get_db() and teardown_appcontext handle it
        else:
            logger.error("Login: Database connection not available.")
            flash('Login service temporarily unavailable. Please try again later.', 'danger')
            return render_template('login.html', title='Login', form=form)

        if user_object:
            login_user(user_object, remember=remember)
            flash(f'Welcome back, {user_object.username}!', 'success')
            # Redirect to the page the user was trying to access, or to index
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')

    return render_template('login.html', title='Login', form=form)

@app.route('/logout')
@login_required # User must be logged in to logout
def logout():
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/')
@login_required # Protect the main page
def index():
    """Renders the main page with paginated detections, stats, and image gallery."""
    # Get search query parameter
    search_query = request.args.get('search', '').strip()
    logger.info(f"Request received for index page ('/'). Search query: '{search_query}'")

    # --- Pagination Setup ---
    try:
        page = request.args.get('page', 1, type=int) # Get page number from query param, default to 1
        if page < 1: page = 1
    except ValueError:
        page = 1 # Default to page 1 if type conversion fails

    ITEMS_PER_PAGE = 5 # How many detections per page - This is server-side, JS uses its own
    offset = (page - 1) * ITEMS_PER_PAGE
    logger.info(f"Requesting page {page}, offset {offset}, items per page {ITEMS_PER_PAGE} (server-side pagination)")
    # ------------------------

    conn = get_db()
    detections = []
    total_detections = 0
    total_pages = 1
    plate_images = [] # Initialize list for image gallery data
    stats = {
        #'total_detections': 0, # Will be fetched for pagination
        'today_detections': 0
    }
    error_message = None

    # --- Fetch Detections and Stats from DB (Server-side pagination part) ---
    # This part remains for initial page load if JS is disabled or for other purposes
    # The client-side table will fetch ALL data via /api/detections
    if conn:
        try:
            with conn.cursor() as cur:
                count_query_base = "SELECT COUNT(*) FROM detected_plates"
                select_query_base = """
                    SELECT id, license_plate, start_time, end_time, confidence,
                           detection_count, vehicle_type, vehicle_color,
                           time_of_day, day_of_week, image_filename,
                           annotated_frame_filename
                    FROM detected_plates
                """
                where_clause = ""
                query_params = []

                if search_query:
                    where_clause = " WHERE license_plate ILIKE %s"
                    query_params.append(f"%{search_query}%")

                count_query = count_query_base + where_clause
                cur.execute(count_query, tuple(query_params))
                total_count_result = cur.fetchone()
                if total_count_result:
                    total_detections = total_count_result[0]
                    total_pages = math.ceil(total_detections / ITEMS_PER_PAGE) if ITEMS_PER_PAGE > 0 else 1
                else:
                    total_detections = 0
                    total_pages = 1

                if page > total_pages and total_pages > 0: page = total_pages
                elif page < 1: page = 1
                offset = (page - 1) * ITEMS_PER_PAGE

                select_query = select_query_base + where_clause + " ORDER BY end_time DESC LIMIT %s OFFSET %s"
                final_params = query_params + [ITEMS_PER_PAGE, offset]
                cur.execute(select_query, tuple(final_params))
                # No need to process these server-side detections for the template if JS handles all
                # But keeping the logic for total_detections and stats is fine.

                cur.execute("SELECT COUNT(*) FROM detected_plates WHERE DATE(end_time) = CURRENT_DATE")
                today_count_result = cur.fetchone()
                if today_count_result:
                    stats['today_detections'] = today_count_result[0]

                # --- NEW: Fetch Images for Gallery directly from DB ---
                logger.info("Fetching gallery images from the database.")
                gallery_query = """
                    SELECT annotated_frame_filename, license_plate
                    FROM detected_plates
                    WHERE annotated_frame_filename IS NOT NULL
                    ORDER BY end_time DESC
                    LIMIT 20;
                """
                cur.execute(gallery_query)
                gallery_results = cur.fetchall()
                for row in gallery_results:
                    plate_images.append({'filename': row[0], 'plate_text': row[1]})
                logger.info(f"Found {len(plate_images)} images for gallery from database.")
                # --- END NEW ---

        except psycopg2.Error as e:
            logger.error(f"Database query error on index page: {e}", exc_info=True)
            error_message = f"Database Error: Could not retrieve page data."
            # Set defaults on error
            stats = {'today_detections': 'Error'}
            total_detections = 'Error'
    else:
        error_message = "Database connection not available."
        stats = {'today_detections': 'N/A'}
        total_detections = 'N/A'

    # --- Render Template ---
    raw_csrf_token = generate_csrf()

    logger.debug("Rendering index.html template...")
    try:
        return render_template('index.html',
                               # Server-side paginated detections (can be removed if JS handles ALL displays)
                               # detections=detections,
                               # current_page=page,
                               # total_pages=total_pages,
                               total_detections_server_fallback=total_detections, # For display if JS fails to update
                               stats=stats,
                               plate_images=plate_images,
                               search_query=search_query,
                               error=error_message,
                               raw_csrf_token=raw_csrf_token) # Pass raw token to template
    except Exception as render_error:
        logger.error(f"Error rendering template 'index.html': {render_error}", exc_info=True)
        return f"Error rendering template: {render_error}", 500


@app.route('/api/detections')
def api_detections():
    """Provides ALL detection data as JSON for client-side processing."""
    logger.info("Request received for API endpoint '/api/detections' (fetching ALL data)")
    conn = get_db()
    detections_data = []
    error_message = None
    status_code = 200

    if conn:
        try:
            with conn.cursor() as cur:
                logger.debug("Executing DB query for ALL detections (for client-side table).")
                sql = """
                    SELECT
                        id,
                        license_plate,
                        start_time,
                        end_time,
                        confidence,
                        vehicle_type,
                        vehicle_color,
                        car_brand,
                        time_of_day,
                        day_of_week,
                        image_filename,
                        annotated_frame_filename
                    FROM detected_plates
                    ORDER BY end_time DESC
                """
                cur.execute(sql)
                colnames = [desc[0] for desc in cur.description]
                rows = cur.fetchall()

                for row_tuple in rows:
                    row_dict = {}
                    for i, col_name in enumerate(colnames):
                        value = row_tuple[i]
                        if isinstance(value, dt): # Check for datetime.datetime
                            row_dict[col_name] = value.isoformat() if value else None
                        else:
                            row_dict[col_name] = value
                    detections_data.append(row_dict)

                logger.info(f"Fetched {len(detections_data)} total detections for API.")

        except psycopg2.Error as e:
            logger.error(f"Database query error in API: {e}", exc_info=True)
            error_message = f"Database Error: {e}"
            status_code = 500 # Internal Server Error
    else:
        error_message = "Database connection not available."
        status_code = 503 # Service Unavailable
        logger.error("Database connection pool not available for API request.")

    if error_message:
        logger.warning(f"API request failed: {error_message}")
        return jsonify({"error": error_message, "detections": []}), status_code
    else:
        logger.debug("Returning successful API response with all detections.")
        # Return the full dataset
        return jsonify({"detections": detections_data})

# Route to serve annotated full frame images for the gallery
@app.route('/output_images/<path:filename>')
def serve_output_image(filename):
    """Serves images from the OUTPUT_DIR for the gallery."""
    logger.debug(f"Request received to serve gallery image: {filename}")
    # Use absolute path for send_from_directory
    absolute_output_dir = OUTPUT_DIR.resolve()
    logger.debug(f"Serving from directory: {absolute_output_dir}")
    try:
        # Security check
        if '..' in filename or filename.startswith('/'):
             logger.warning(f"Potential path traversal attempt blocked for gallery filename: {filename}")
             abort(404)
        # Ensure the file requested is directly in OUTPUT_DIR, not subdirs like plate_images
        requested_path = absolute_output_dir / filename
        if not requested_path.is_file() or requested_path.parent != absolute_output_dir:
             logger.warning(f"Attempt to access file outside designated gallery directory: {filename}")
             abort(404)

        return send_from_directory(absolute_output_dir, filename, as_attachment=False)
    except FileNotFoundError:
        logger.error(f"Gallery image file not found: {filename} in {absolute_output_dir}")
        abort(404)
    except Exception as e:
        logger.error(f"Error serving gallery image {filename}: {e}", exc_info=True)
        abort(500)

# Route to serve CROPPED plate images (for the table)
@app.route('/plate_images/<path:filename>')
def serve_plate_image(filename):
    """Serves CROPPED plate images from the PLATE_IMAGE_DIR."""
    logger.debug(f"Request received to serve CROPPED plate image: {filename}")
    # Use absolute path for send_from_directory for clarity and robustness
    absolute_plate_image_dir = PLATE_IMAGE_DIR.resolve()
    logger.debug(f"Serving from directory: {absolute_plate_image_dir}")
    try:
        # Security: Basic check to prevent path traversal
        if '..' in filename or filename.startswith('/'):
             logger.warning(f"Potential path traversal attempt blocked for plate filename: {filename}")
             abort(404)

        return send_from_directory(absolute_plate_image_dir, filename, as_attachment=False)
    except FileNotFoundError:
        logger.error(f"CROPPED plate image file not found: {filename} in {absolute_plate_image_dir}")
        abort(404)
    except Exception as e:
        logger.error(f"Error serving CROPPED plate image {filename}: {e}", exc_info=True)
        abort(500)

# --- Upload Route ---
@app.route('/upload', methods=['POST'])
@login_required # User must be logged in
@admin_required # ONLY ADMINS can upload
def upload_image():
    global anpr_processor
    logger.info(f"Received request for image upload endpoint (/upload) from ADMIN user: {current_user.username}")

    if not anpr_processor:
        logger.error("ANPRProcessor not initialized, cannot process upload.")
        return jsonify({"success": False, "error": "ANPR system is not ready."}), 500

    if 'imageFile' not in request.files:
        logger.warning("No 'imageFile' part in the request files.")
        return jsonify({"success": False, "error": "No file part in the request."}), 400

    uploaded_files = request.files.getlist('imageFile') # Get a list of files

    if not uploaded_files or all(f.filename == '' for f in uploaded_files):
        logger.warning("No files selected for upload.")
        return jsonify({"success": False, "error": "No selected file(s)."}), 400

    processed_count = 0
    error_count = 0
    results_summary = [] # To store summary of each file processing

    for file_storage_object in uploaded_files:
        if file_storage_object and allowed_file(file_storage_object.filename):
            original_filename = secure_filename(file_storage_object.filename)
            timestamp = dt.now().strftime("%Y%m%d%H%M%S%f") # More precision for multiple files
            unique_filename = f"{timestamp}_{original_filename}"
            temp_save_path = UPLOADS_DIR / unique_filename
            logger.info(f"Processing uploaded file: {original_filename} -> {unique_filename}")

            try:
                file_storage_object.save(temp_save_path)
                logger.info(f"Temporarily saved uploaded file to: {temp_save_path}")

                anpr_processor.process_image_source(str(temp_save_path))
                logger.info(f"ANPR processing finished for: {temp_save_path}")

                results_summary.append({"filename": original_filename, "status": "success", "message": "Processed successfully."})
                processed_count += 1

            except Exception as e:
                error_details = traceback.format_exc()
                logger.error(f"Error processing uploaded file {temp_save_path}: {e}\n{error_details}")
                results_summary.append({"filename": original_filename, "status": "error", "message": f"Processing failed: {str(e)}"})
                error_count += 1
            finally:
                if temp_save_path.exists():
                    try:
                        os.remove(temp_save_path)
                        logger.info(f"Removed temporary file: {temp_save_path}")
                    except OSError as e_remove:
                        logger.error(f"Error removing temporary file {temp_save_path}: {e_remove}")
        elif file_storage_object:
            original_filename = secure_filename(file_storage_object.filename)
            logger.warning(f"File type not allowed for file: {original_filename}")
            results_summary.append({"filename": original_filename, "status": "error", "message": "File type not allowed."})
            error_count += 1

    if processed_count == 0 and error_count == 0: # Should not happen if files were initially present
        return jsonify({"success": False, "error": "No valid files were processed."}), 400

    final_message = f"Processed {processed_count} file(s) successfully."
    if error_count > 0:
        final_message += f" Encountered errors with {error_count} file(s)."

    # The frontend currently expects a single success/error for the overall operation
    # rather than a list of results. For simplicity, we send a general status.
    # The detailed per-file status is now shown on the client-side JS as it processes.
    if error_count > 0 and processed_count == 0:
        return jsonify({"success": False, "error": final_message, "details": results_summary}), 207 # Multi-Status with error focus
    elif error_count > 0: # Mixed results
        return jsonify({"success": True, "message": final_message, "details": results_summary}), 207 # Multi-Status
    else: # All successful
        return jsonify({"success": True, "message": final_message, "details": results_summary}), 200

# --- Admin Routes ---
@app.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    # This page will be the main hub for admin-specific functions
    # For now, it can just render a simple template
    unacknowledged_alerts = []
    conn = get_db()
    if conn:
        try:
            with conn.cursor() as cur:
                # Fetch unacknowledged alerts with details
                cur.execute("""
                    SELECT
                        a.id as alert_id,
                        a.alert_time,
                        dp.license_plate,
                        dp.id as detection_id,
                        dp.vehicle_type,
                        dp.vehicle_color,
                        dp.image_filename as plate_image_filename,
                        dp.annotated_frame_filename,
                        wl.name as watchlist_name,
                        wle.reason as watchlist_reason
                    FROM alerts a
                    JOIN detected_plates dp ON a.detection_id = dp.id
                    JOIN watchlists wl ON a.watchlist_id = wl.id
                    JOIN watchlist_entries wle ON a.watchlist_entry_id = wle.id
                    WHERE a.acknowledged_at IS NULL
                    ORDER BY a.alert_time DESC
                    LIMIT 10; -- Limit for dashboard display
                """)
                alerts_data = cur.fetchall()
                if alerts_data:
                    colnames = [desc[0] for desc in cur.description]
                    unacknowledged_alerts = [dict(zip(colnames, row)) for row in alerts_data]
                logger.info(f"Admin dashboard: Fetched {len(unacknowledged_alerts)} unacknowledged alerts.")
        except psycopg2.Error as e:
            logger.error(f"Error fetching unacknowledged alerts for admin dashboard: {e}", exc_info=True)
            flash('Could not load unacknowledged alerts due to a database error.', 'danger')
    else:
        flash('Database connection not available to load alerts.', 'warning')

    csrf_form_instance = FlaskForm() # Create an instance for CSRF token
    return render_template('admin/admin_dashboard.html', title='Admin Dashboard',
                           unacknowledged_alerts=unacknowledged_alerts, csrf_form=csrf_form_instance)

@app.route('/admin/users')
@login_required
@admin_required
def admin_manage_users():
    conn = get_db()
    site_users = [] # Renamed to avoid conflict with the 'users' variable name from the table
    # Create a base FlaskForm instance for CSRF token generation in the template
    csrf_form = FlaskForm()

    if conn:
        try:
            with conn.cursor() as cur:
                # Fetch all users for display - be careful with sensitive data in larger apps
                cur.execute("SELECT id, username, role, created_at FROM users ORDER BY created_at DESC")
                colnames = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                site_users = [dict(zip(colnames, row)) for row in rows]
                logger.info(f"Admin {current_user.username} fetched {len(site_users)} users for management page.")
        except psycopg2.Error as e:
            logger.error(f"Admin manage users: Database error: {e}", exc_info=True)
            flash('Could not retrieve user list due to a database error.', 'danger')
    else:
        logger.error("Admin manage users: Database connection not available.")
        flash('User management service temporarily unavailable.', 'danger')

    return render_template('admin/admin_users.html', title='Manage Users', users_list=site_users, csrf_form=csrf_form) # Pass csrf_form

@app.route('/admin/user/change_role/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def admin_change_user_role(user_id):
    action = request.form.get('action')
    if not action in ['promote', 'demote']:
        flash('Invalid action specified.', 'danger')
        return redirect(url_for('admin_manage_users'))

    conn = get_db()
    if not conn:
        flash('Database connection unavailable.', 'danger')
        return redirect(url_for('admin_manage_users'))

    try:
        with conn.cursor() as cur:
            # Fetch user to ensure they exist and are not the current admin trying to demote self
            cur.execute("SELECT id, username, role FROM users WHERE id = %s", (user_id,))
            user_to_modify = cur.fetchone()

            if not user_to_modify:
                flash('User not found.', 'danger')
                return redirect(url_for('admin_manage_users'))

            user_to_modify_username = user_to_modify[1]
            current_role = user_to_modify[2]

            if user_to_modify_username == current_user.username:
                flash('You cannot change your own role.', 'warning')
                return redirect(url_for('admin_manage_users'))

            new_role = None
            if action == 'promote' and current_role == 'user':
                new_role = 'admin'
            elif action == 'demote' and current_role == 'admin':
                new_role = 'user'
            else:
                flash(f'Cannot {action} user {user_to_modify_username} from role {current_role}.', 'warning')
                return redirect(url_for('admin_manage_users'))

            cur.execute("UPDATE users SET role = %s WHERE id = %s", (new_role, user_id))
            conn.commit()
            logger.info(f"Admin {current_user.username} changed role of user ID {user_id} ({user_to_modify_username}) to {new_role}.")
            flash(f"User {user_to_modify_username}'s role has been updated to {new_role}.", 'success')
    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Error changing role for user ID {user_id}: {e}", exc_info=True)
        flash('Failed to change user role due to a database error.', 'danger')

    return redirect(url_for('admin_manage_users'))

@app.route('/admin/user/delete/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def admin_delete_user(user_id):
    conn = get_db()
    if not conn:
        flash('Database connection unavailable.', 'danger')
        return redirect(url_for('admin_manage_users'))

    try:
        with conn.cursor() as cur:
            # Fetch user to ensure they exist and are not the current admin trying to delete self
            cur.execute("SELECT username FROM users WHERE id = %s", (user_id,))
            user_to_delete = cur.fetchone()

            if not user_to_delete:
                flash('User not found.', 'danger')
                return redirect(url_for('admin_manage_users'))

            user_to_delete_username = user_to_delete[0]

            if user_to_delete_username == current_user.username:
                flash('You cannot delete your own account.', 'warning')
                return redirect(url_for('admin_manage_users'))

            cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
            conn.commit()
            logger.info(f"Admin {current_user.username} deleted user ID {user_id} ({user_to_delete_username}).")
            flash(f'User {user_to_delete_username} has been deleted successfully.', 'success')
    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Error deleting user ID {user_id}: {e}", exc_info=True)
        flash('Failed to delete user due to a database error.', 'danger')

    return redirect(url_for('admin_manage_users'))

# --- Detection Management Routes (Admin Only) ---
@app.route('/detection/delete/<int:detection_id>', methods=['POST'])
@login_required
@admin_required
def delete_detection(detection_id):
    logger.info(f"Admin {current_user.username} attempting to delete detection ID: {detection_id}")
    conn = get_db()
    if not conn:
        flash('Database connection unavailable. Could not delete detection.', 'danger')
        return redirect(url_for('index'))

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT image_filename, annotated_frame_filename FROM detected_plates WHERE id = %s", (detection_id,))
            image_files = cur.fetchone()

            if not image_files:
                flash(f'Detection record with ID {detection_id} not found.', 'warning')
                return redirect(url_for('index'))

            cur.execute("DELETE FROM detected_plates WHERE id = %s", (detection_id,))
            conn.commit()
            logger.info(f"Successfully deleted detection record ID: {detection_id} from database.")

            plate_image_to_delete, annotated_frame_to_delete = image_files

            if plate_image_to_delete:
                try:
                    plate_image_path = PLATE_IMAGE_DIR / plate_image_to_delete
                    if plate_image_path.exists():
                        os.remove(plate_image_path)
                        logger.info(f"Deleted cropped plate image: {plate_image_path}")
                    else:
                        logger.warning(f"Cropped plate image not found for deletion: {plate_image_path}")
                except OSError as e:
                    logger.error(f"Error deleting cropped plate image {plate_image_to_delete}: {e}")

            if annotated_frame_to_delete:
                try:
                    annotated_frame_path = OUTPUT_DIR / annotated_frame_to_delete
                    if annotated_frame_path.exists():
                        os.remove(annotated_frame_path)
                        logger.info(f"Deleted annotated frame image: {annotated_frame_path}")
                    else:
                        logger.warning(f"Annotated frame image not found for deletion: {annotated_frame_path}")
                except OSError as e:
                    logger.error(f"Error deleting annotated frame image {annotated_frame_to_delete}: {e}")

            flash(f'Detection record ID {detection_id} and associated images (if found) have been deleted.', 'success')

    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Database error deleting detection ID {detection_id}: {e}", exc_info=True)
        flash('Failed to delete detection record due to a database error.', 'danger')
    except Exception as e:
        conn.rollback()
        logger.error(f"Unexpected error deleting detection ID {detection_id}: {e}", exc_info=True)
        flash('An unexpected error occurred while trying to delete the detection.', 'danger')

    return redirect(url_for('index'))

@app.route('/admin/usage-metrics')
@login_required
@admin_required
def admin_usage_metrics():
    logger.info(f"Admin {current_user.username} accessing Groq API Usage Metrics page.")
    conn = get_db()
    usage_data = {
        'summary_stats': {},
        'recent_calls': [],
        'by_call_type': {},
        'error': None
    }

    # Define pricing rates - these can be adjusted as needed
    pricing_rates = {
        # 'per_request': 2.00,  # $2 per request
        'per_token': 0.00013      # $0.00013 per token
    }

    if not conn:
        usage_data['error'] = "Database connection not available."
        logger.error("Groq Usage Metrics: Database connection not available.")
        return render_template('admin/admin_usage_metrics.html', title='Groq API Usage Metrics', usage_data=usage_data, pricing_rates=pricing_rates)

    try:
        with conn.cursor() as cur:
            # 1. Summary Stats (Total calls, total tokens today and overall)
            cur.execute("""
                SELECT
                    COUNT(*) AS total_calls,
                    SUM(total_tokens) AS grand_total_tokens,
                    SUM(CASE WHEN DATE(timestamp) = CURRENT_DATE THEN total_tokens ELSE 0 END) AS today_total_tokens,
                    COUNT(CASE WHEN DATE(timestamp) = CURRENT_DATE THEN 1 ELSE NULL END) AS today_total_calls
                FROM groq_api_usage;
            """)
            summary = cur.fetchone()
            if summary:
                usage_data['summary_stats'] = {
                    'total_calls': summary[0] or 0,
                    'grand_total_tokens': summary[1] or 0,
                    'today_total_tokens': summary[2] or 0,
                    'today_total_calls': summary[3] or 0
                }

            # 2. Recent API Calls (e.g., last 20)
            cur.execute("""
                SELECT timestamp, api_call_type, model_name, prompt_tokens, completion_tokens, total_tokens, related_detection_id
                FROM groq_api_usage
                ORDER BY timestamp DESC
                LIMIT 20;
            """)
            colnames_recent = [desc[0] for desc in cur.description]
            usage_data['recent_calls'] = [dict(zip(colnames_recent, row)) for row in cur.fetchall()]

            # 3. Aggregated by Call Type (Overall and Today)
            cur.execute("""
                SELECT
                    api_call_type,
                    model_name,
                    COUNT(*) as num_calls,
                    SUM(total_tokens) as sum_total_tokens,
                    AVG(total_tokens)::integer as avg_total_tokens,
                    SUM(prompt_tokens) as sum_prompt_tokens,
                    SUM(completion_tokens) as sum_completion_tokens,
                    COUNT(CASE WHEN DATE(timestamp) = CURRENT_DATE THEN 1 ELSE NULL END) as today_num_calls,
                    SUM(CASE WHEN DATE(timestamp) = CURRENT_DATE THEN total_tokens ELSE 0 END) as today_sum_total_tokens
                FROM groq_api_usage
                GROUP BY api_call_type, model_name
                ORDER BY api_call_type;
            """)
            colnames_by_type = [desc[0] for desc in cur.description]
            rows_by_type = cur.fetchall()
            for row_tuple in rows_by_type:
                row_dict = dict(zip(colnames_by_type, row_tuple))
                call_type = row_dict['api_call_type']
                if call_type not in usage_data['by_call_type']:
                    usage_data['by_call_type'][call_type] = []
                usage_data['by_call_type'][call_type].append(row_dict)

    except psycopg2.Error as e:
        logger.error(f"Groq Usage Metrics: Database error: {e}", exc_info=True)
        usage_data['error'] = f"Database error occurred: {str(e)}"
    except Exception as e_general:
        logger.error(f"Groq Usage Metrics: General error: {e_general}", exc_info=True)
        usage_data['error'] = f"An unexpected error occurred: {str(e_general)}"

    # Pass both usage_data and pricing_rates to the template
    return render_template('admin/admin_usage_metrics.html', title='Groq API Usage Metrics',
                          usage_data=usage_data, pricing_rates=pricing_rates)

@app.route('/admin/detection-analytics')
@login_required
@admin_required
def admin_detection_analytics():
    logger.info(f"Admin {current_user.username} accessing Detection Analytics page.")
    conn = get_db()
    analytics_data = {
        'by_vehicle_type': [],
        'by_vehicle_color': [],
        'detections_per_day': [], # Last 30 days
        'by_hour_of_day': [],
        'by_day_of_week': [],
        'error': None
    }

    if not conn:
        analytics_data['error'] = "Database connection not available."
        logger.error("Detection Analytics: Database connection not available.")
        return render_template('admin/detection_stats.html', title='Detection Analytics', analytics_data=analytics_data)

    try:
        with conn.cursor() as cur:
            # 1. By Vehicle Type
            cur.execute("""
                SELECT vehicle_type, COUNT(*) as count
                FROM detected_plates
                WHERE vehicle_type IS NOT NULL AND vehicle_type <> 'unknown' AND vehicle_type <> ''
                GROUP BY vehicle_type
                ORDER BY count DESC;
            """)
            analytics_data['by_vehicle_type'] = [dict(zip([column[0] for column in cur.description], row)) for row in cur.fetchall()]

            # 2. By Vehicle Color
            cur.execute("""
                SELECT vehicle_color, COUNT(*) as count
                FROM detected_plates
                WHERE vehicle_color IS NOT NULL AND vehicle_color <> 'unknown' AND vehicle_color <> ''
                GROUP BY vehicle_color
                ORDER BY count DESC;
            """)
            analytics_data['by_vehicle_color'] = [dict(zip([column[0] for column in cur.description], row)) for row in cur.fetchall()]

            # 3. Detections per day (last 30 days)
            cur.execute("""
                SELECT DATE_TRUNC('day', end_time) AS detection_day, COUNT(*) AS count
                FROM detected_plates
                WHERE end_time >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY detection_day
                ORDER BY detection_day ASC;
            """)
            analytics_data['detections_per_day'] = [
                {'date': row[0].strftime('%Y-%m-%d'), 'count': row[1]}
                for row in cur.fetchall()
            ]

            # 4. By Hour of Day
            cur.execute("""
                SELECT EXTRACT(HOUR FROM end_time) AS hour_of_day, COUNT(*) AS count
                FROM detected_plates
                GROUP BY hour_of_day
                ORDER BY hour_of_day ASC;
            """)
            # Ensure all 24 hours are present, even if count is 0
            hourly_counts = {row[0]: row[1] for row in cur.fetchall()}
            analytics_data['by_hour_of_day'] = [{'hour': h, 'count': hourly_counts.get(float(h), 0)} for h in range(24)]

            # 5. By Day of Week
            cur.execute("""
                SELECT
                    EXTRACT(ISODOW FROM end_time) AS day_number,
                    TO_CHAR(end_time, 'Day') AS day_name,
                    COUNT(*) AS count
                FROM detected_plates
                GROUP BY day_number, day_name
                ORDER BY day_number ASC;
            """)
            day_mapping = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
            weekly_counts = {row[0]: {'name': row[1].strip(), 'count': row[2]} for row in cur.fetchall()}
            analytics_data['by_day_of_week'] = [
                {'day_number': i, 'day_name': day_mapping[i], 'count': weekly_counts.get(float(i), {}).get('count', 0)}
                for i in range(1, 8)
            ]

    except psycopg2.Error as e:
        logger.error(f"Detection Analytics: Database error: {e}", exc_info=True)
        analytics_data['error'] = f"Database error occurred: {str(e)}"
    except Exception as e_general:
        logger.error(f"Detection Analytics: General error: {e_general}", exc_info=True)
        analytics_data['error'] = f"An unexpected error occurred: {str(e_general)}"

    return render_template('admin/detection_stats.html', title='Detection Analytics', analytics_data=analytics_data)

# --- Watchlist Management Routes ---
@app.route('/admin/watchlists', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_watchlists():
    form = WatchlistForm()
    if form.validate_on_submit():
        conn = get_db()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO watchlists (name, description, is_active) VALUES (%s, %s, %s)",
                                (form.name.data, form.description.data, form.is_active.data))
                    conn.commit()
                    flash(f'Watchlist "{form.name.data}" created successfully!', 'success')
                    return redirect(url_for('admin_watchlists'))
            except psycopg2.IntegrityError: # Handles unique name constraint
                conn.rollback()
                flash(f'Error: A watchlist with the name "{form.name.data}" already exists.', 'danger')
            except psycopg2.Error as e:
                conn.rollback()
                logger.error(f"Error creating watchlist: {e}", exc_info=True)
                flash('Error creating watchlist. Please try again.', 'danger')
        else:
            flash('Database connection error.', 'danger')

    watchlists_data = []
    conn = get_db()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id, name, description, created_at, is_active, (SELECT COUNT(*) FROM watchlist_entries WHERE watchlist_id = watchlists.id) as entry_count FROM watchlists ORDER BY name ASC")
                watchlists_data = [dict(zip([column[0] for column in cur.description], row)) for row in cur.fetchall()]
        except psycopg2.Error as e:
            logger.error(f"Error fetching watchlists: {e}", exc_info=True)
            flash('Error fetching watchlists.', 'danger')

    return render_template('admin/admin_watchlists.html', title='Manage Watchlists',
                           form=form, watchlists=watchlists_data)

@app.route('/admin/watchlist/edit/<int:watchlist_id>', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_edit_watchlist(watchlist_id):
    conn = get_db()
    if not conn:
        flash('Database connection error.', 'danger')
        return redirect(url_for('admin_watchlists'))

    watchlist_to_edit = None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, description, is_active FROM watchlists WHERE id = %s", (watchlist_id,))
            data = cur.fetchone()
            if data:
                watchlist_to_edit = dict(zip([column[0] for column in cur.description], data))
            else:
                flash('Watchlist not found.', 'danger')
                return redirect(url_for('admin_watchlists'))
    except psycopg2.Error as e:
        logger.error(f"Error fetching watchlist ID {watchlist_id} for edit: {e}", exc_info=True)
        flash('Error fetching watchlist details.', 'danger')
        return redirect(url_for('admin_watchlists'))

    form = WatchlistForm(obj=type('obj', (object,), watchlist_to_edit)()) # Pre-populate form

    if form.validate_on_submit():
        new_name = form.name.data
        new_description = form.description.data
        new_is_active = form.is_active.data
        try:
            with conn.cursor() as cur:
                # Check if name is being changed to one that already exists (excluding itself)
                cur.execute("SELECT id FROM watchlists WHERE name = %s AND id != %s", (new_name, watchlist_id))
                if cur.fetchone():
                    flash(f'Error: A watchlist with the name "{new_name}" already exists.', 'danger')
                else:
                    cur.execute("UPDATE watchlists SET name = %s, description = %s, is_active = %s WHERE id = %s",
                                (new_name, new_description, new_is_active, watchlist_id))
                    conn.commit()
                    flash(f'Watchlist "{new_name}" updated successfully!', 'success')
                    logger.info(f"Admin {current_user.username} updated watchlist ID {watchlist_id}.")
                    return redirect(url_for('admin_watchlists'))
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Error updating watchlist ID {watchlist_id}: {e}", exc_info=True)
            flash('Error updating watchlist. Please try again.', 'danger')

    return render_template('admin/admin_edit_watchlist.html', title=f'Edit Watchlist: {watchlist_to_edit["name"]}',
                           form=form, watchlist_id=watchlist_id, current_watchlist_name=watchlist_to_edit["name"])

@app.route('/admin/watchlist/delete/<int:watchlist_id>', methods=['POST'])
@login_required
@admin_required
def admin_delete_watchlist(watchlist_id):
    conn = get_db()
    if conn:
        try:
            with conn.cursor() as cur:
                # Fetch watchlist name for flash message before deleting
                cur.execute("SELECT name FROM watchlists WHERE id = %s", (watchlist_id,))
                watchlist_name_tuple = cur.fetchone()
                watchlist_name = watchlist_name_tuple[0] if watchlist_name_tuple else f"ID {watchlist_id}"

                # ON DELETE CASCADE in DB schema should handle related entries and alerts
                cur.execute("DELETE FROM watchlists WHERE id = %s", (watchlist_id,))
                conn.commit()

                if cur.rowcount > 0:
                    flash(f'Watchlist "{watchlist_name}" and all its associated entries and alerts have been deleted.', 'success')
                    logger.info(f"Admin {current_user.username} deleted watchlist ID {watchlist_id} ('{watchlist_name}').")
                else:
                    flash(f'Watchlist ID {watchlist_id} not found or already deleted.', 'warning')
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Error deleting watchlist ID {watchlist_id}: {e}", exc_info=True)
            flash('Error deleting watchlist. Please try again.', 'danger')
    else:
        flash('Database connection error.', 'danger')
    return redirect(url_for('admin_watchlists'))

@app.route('/admin/watchlist/<int:watchlist_id>/entries', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_watchlist_entries(watchlist_id):
    form = WatchlistEntryForm()
    conn = get_db()

    # Fetch watchlist details to display its name
    watchlist_info = None
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id, name, description FROM watchlists WHERE id = %s", (watchlist_id,))
                watchlist_info = cur.fetchone()
                if watchlist_info:
                    watchlist_info = dict(zip([column[0] for column in cur.description], watchlist_info))
                else:
                    flash('Watchlist not found.', 'danger')
                    return redirect(url_for('admin_watchlists'))
        except psycopg2.Error as e:
            logger.error(f"Error fetching watchlist info (ID: {watchlist_id}): {e}", exc_info=True)
            flash('Error fetching watchlist details.', 'danger')
            return redirect(url_for('admin_watchlists'))
    else:
        flash('Database connection error.', 'danger')
        return redirect(url_for('admin_watchlists'))

    if form.validate_on_submit():
        if conn: # Re-check conn as it might have been lost or closed
            try:
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO watchlist_entries (watchlist_id, license_plate, reason) VALUES (%s, %s, %s)",
                                (watchlist_id, form.license_plate.data.upper(), form.reason.data))
                    conn.commit()
                    flash(f'Plate "{form.license_plate.data.upper()}" added to watchlist "{watchlist_info["name"]}" successfully!', 'success')
                    return redirect(url_for('admin_watchlist_entries', watchlist_id=watchlist_id))
            except psycopg2.IntegrityError: # Handles unique (watchlist_id, license_plate) constraint
                conn.rollback()
                flash(f'Error: Plate "{form.license_plate.data.upper()}" already exists in this watchlist.', 'danger')
            except psycopg2.Error as e:
                conn.rollback()
                logger.error(f"Error adding watchlist entry: {e}", exc_info=True)
                flash('Error adding plate to watchlist. Please try again.', 'danger')
        else:
            flash('Database connection error while adding plate.', 'danger')

    entries_data = []
    if conn: # Re-check conn
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id, license_plate, reason, added_at FROM watchlist_entries WHERE watchlist_id = %s ORDER BY license_plate ASC", (watchlist_id,))
                entries_data = [dict(zip([column[0] for column in cur.description], row)) for row in cur.fetchall()]
        except psycopg2.Error as e:
            logger.error(f"Error fetching watchlist entries (Watchlist ID: {watchlist_id}): {e}", exc_info=True)
            flash('Error fetching watchlist entries.', 'danger')

    return render_template('admin/admin_watchlist_entries.html', title=f'Entries for {watchlist_info["name"] if watchlist_info else "Watchlist"}',
                           form=form, entries=entries_data, watchlist=watchlist_info, csrf_form=FlaskForm()) # Pass empty FlaskForm for CSRF in delete forms

# --- Route to delete a watchlist entry ---
@app.route('/admin/watchlist_entry/delete/<int:entry_id>', methods=['POST'])
@login_required
@admin_required
def admin_delete_watchlist_entry(entry_id):
    conn = get_db()
    watchlist_id_to_redirect = None # To redirect back to the correct watchlist page

    if conn:
        try:
            with conn.cursor() as cur:
                # Optional: Get watchlist_id for redirect before deleting
                cur.execute("SELECT watchlist_id FROM watchlist_entries WHERE id = %s", (entry_id,))
                result = cur.fetchone()
                if result:
                    watchlist_id_to_redirect = result[0]

                cur.execute("DELETE FROM watchlist_entries WHERE id = %s", (entry_id,))
                conn.commit()
                if cur.rowcount > 0:
                    flash(f'Watchlist entry ID {entry_id} deleted successfully.', 'success')
                    logger.info(f"Admin {current_user.username} deleted watchlist entry ID {entry_id}.")
                else:
                    flash(f'Watchlist entry ID {entry_id} not found or already deleted.', 'warning')
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Error deleting watchlist entry ID {entry_id}: {e}", exc_info=True)
            flash('Error deleting watchlist entry. Please try again.', 'danger')
    else:
        flash('Database connection error.', 'danger')

    if watchlist_id_to_redirect:
        return redirect(url_for('admin_watchlist_entries', watchlist_id=watchlist_id_to_redirect))
    return redirect(url_for('admin_watchlists')) # Fallback redirect

# --- Alert Management Routes ---
@app.route('/admin/alert/acknowledge/<int:alert_id>', methods=['POST'])
@login_required
@admin_required
def admin_acknowledge_alert(alert_id):
    conn = get_db()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE alerts SET acknowledged_at = NOW(), acknowledged_by_user_id = %s WHERE id = %s AND acknowledged_at IS NULL",
                            (current_user.id, alert_id))
                conn.commit()
                if cur.rowcount > 0:
                    flash(f'Alert ID {alert_id} acknowledged successfully.', 'success')
                    logger.info(f"User {current_user.username} acknowledged alert ID {alert_id}.")
                else:
                    flash(f'Alert ID {alert_id} was already acknowledged or not found.', 'warning')
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Error acknowledging alert ID {alert_id}: {e}", exc_info=True)
            flash('Error acknowledging alert. Please try again.', 'danger')
    else:
        flash('Database connection error.', 'danger')
    return redirect(request.referrer or url_for('admin_dashboard'))

@app.route('/admin/alerts')
@login_required
@admin_required
def admin_all_alerts():
    conn = get_db()
    all_alerts_data = []
    # Basic form for CSRF in acknowledge buttons on this page too
    csrf_form_alerts = FlaskForm()

    if conn:
        try:
            with conn.cursor() as cur:
                query = """
                    SELECT
                        a.id as alert_id, a.alert_time, a.acknowledged_at,
                        u.username as acknowledged_by_username,
                        dp.license_plate, dp.id as detection_id,
                        wl.name as watchlist_name, wle.reason as watchlist_reason,
                        dp.image_filename as plate_image_filename
                    FROM alerts a
                    JOIN detected_plates dp ON a.detection_id = dp.id
                    JOIN watchlists wl ON a.watchlist_id = wl.id
                    JOIN watchlist_entries wle ON a.watchlist_entry_id = wle.id
                    LEFT JOIN users u ON a.acknowledged_by_user_id = u.id
                    ORDER BY a.alert_time DESC;
                """
                cur.execute(query)
                fetched_alerts = cur.fetchall()
                if fetched_alerts:
                    colnames = [desc[0] for desc in cur.description]
                    all_alerts_data = [dict(zip(colnames, row)) for row in fetched_alerts]
                logger.info(f"Admin {current_user.username} fetched {len(all_alerts_data)} alerts for the all alerts page.")
        except psycopg2.Error as e:
            logger.error(f"Error fetching all alerts: {e}", exc_info=True)
            flash('Could not retrieve full alert list due to a database error.', 'danger')
    else:
        flash('Database connection not available to load all alerts.', 'danger')

    return render_template('admin/admin_all_alerts.html', title='All System Alerts',
                           alerts=all_alerts_data, csrf_form=csrf_form_alerts)

# --- Live Camera Processing Functions ---
def get_camera_stats():
    """Get current camera statistics"""
    stats = camera_stats.copy()
    if stats['start_time']:
        runtime = (datetime.now() - datetime.fromisoformat(stats['start_time'])).total_seconds()
        stats['runtime_seconds'] = int(runtime)
    else:
        stats['runtime_seconds'] = 0
    return stats

def test_ip_camera_connection(ip_url, timeout=10):
    """Test if an IP camera URL is accessible"""
    try:
        # Basic URL validation
        if not ip_url.startswith(('http://', 'https://')):
            return False, "URL must start with http:// or https://"

        # Try to connect to the IP camera
        cap = cv2.VideoCapture(ip_url)
        if not cap.isOpened():
            cap.release()
            return False, "Cannot connect to IP camera"

        # Try to read a frame to ensure it's working
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return False, "IP camera connected but no video stream available"

        return True, "IP camera connection successful"

    except Exception as e:
        return False, f"Connection failed: {str(e)}"

def capture_and_process_frame():
    """Capture a single frame and process it through ANPR"""
    global camera_cap, anpr_processor

    if not camera_cap or not camera_cap.isOpened():
        return {'error': 'Camera not available'}

    try:
        # Capture frame
        ret, frame = camera_cap.read()
        if not ret:
            return {'error': 'Failed to capture frame'}

        logger.info("Frame captured, starting ANPR processing...")

        # Process through existing ANPR system (same as image upload)
        annotated_frame, detections = anpr_processor.process_image(frame)

        # Generate filename for annotated frame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        annotated_filename = f"live_capture_{timestamp}.jpg"

        # Save annotated frame
        output_dir = Path("output_plates")
        output_dir.mkdir(exist_ok=True)
        annotated_path = output_dir / annotated_filename

        saved_filename = None
        try:
            success = cv2.imwrite(str(annotated_path), annotated_frame)
            if success:
                saved_filename = annotated_filename
                logger.info(f"Saved capture frame: {annotated_filename}")
            else:
                logger.warning("Failed to save annotated frame")
        except Exception as e:
            logger.error(f"Error saving annotated frame: {e}")

        # Save to database if detections found
        if detections:
            try:
                anpr_processor.save_to_database(detections, annotated_frame_filename=saved_filename)
                camera_stats['plates_detected'] += len(detections)
                camera_stats['last_detection_time'] = datetime.now().isoformat()
                logger.info(f"Saved {len(detections)} detections to database")
            except Exception as e:
                logger.error(f"Error saving detections to database: {e}")

        # Return results
        return {
            'success': True,
            'detections_count': len(detections) if detections else 0,
            'detections': detections,
            'annotated_image': saved_filename,
            'timestamp': timestamp
        }

    except Exception as e:
        logger.error(f"Error in capture and process: {e}")
        return {'error': f'Processing failed: {str(e)}'}

# --- Live Camera Routes ---
@app.route('/live_camera')
@login_required
@admin_required  # Only admins can access live camera
def live_camera():
    """Live camera detection page"""
    return render_template('live_camera.html')

@app.route('/api/camera/start', methods=['POST'])
@login_required
@admin_required
@csrf.exempt
def start_camera():
    """Start camera for live preview (supports both camera indices and IP camera URLs)"""
    global camera_cap, camera_running, camera_stats

    if camera_running:
        return jsonify({'error': 'Camera is already running'}), 400

    try:
        data = request.get_json() or {}
        camera_source = data.get('camera_index', 0)
        camera_type = data.get('camera_type', 'usb')  # 'usb' or 'ip'

        logger.info(f"Attempting to start camera: type={camera_type}, source={camera_source}")

        # Handle different camera types
        if camera_type == 'ip':
            # IP Camera URL
            ip_url = str(camera_source)
            if not ip_url.startswith('http'):
                return jsonify({'error': 'IP camera URL must start with http:// or https://'}), 400

            logger.info(f"Connecting to IP camera: {ip_url}")
            camera_cap = cv2.VideoCapture(ip_url)
        else:
            # USB Camera index
            camera_index = int(camera_source)
            logger.info(f"Connecting to USB camera index: {camera_index}")
            camera_cap = cv2.VideoCapture(camera_index)

        if not camera_cap.isOpened():
            camera_cap = None
            source_name = camera_source if camera_type == 'ip' else f'camera {camera_source}'
            return jsonify({'error': f'Failed to open {source_name}'}), 500

        # Set camera properties for better performance (mainly for USB cameras)
        if camera_type == 'usb':
            camera_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera_cap.set(cv2.CAP_PROP_FPS, 15)

        camera_running = True
        camera_stats['start_time'] = datetime.now().isoformat()
        camera_stats['frames_processed'] = 0
        camera_stats['plates_detected'] = 0
        camera_stats['camera_type'] = camera_type
        camera_stats['camera_source'] = camera_source

        source_name = camera_source if camera_type == 'ip' else f'Camera {camera_source}'
        logger.info(f"{source_name} started successfully for preview")

        return jsonify({
            'success': True,
            'message': f'{source_name} started successfully',
            'camera_type': camera_type,
            'stats': get_camera_stats()
        })

    except Exception as e:
        logger.error(f"Error starting camera: {e}")
        return jsonify({'error': f'Failed to start camera: {str(e)}'}), 500

# Manual capture endpoint removed - using automatic detection only

@app.route('/api/camera/preview')
@login_required
@admin_required
@csrf.exempt
def camera_preview():
    """Get current camera preview frame"""
    global camera_cap, camera_running

    if not camera_running or not camera_cap:
        return jsonify({'error': 'Camera not available'}), 400

    try:
        ret, frame = camera_cap.read()
        if not ret:
            return jsonify({'error': 'Failed to capture preview'}), 500

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_data = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'image': frame_data,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting preview: {e}")
        return jsonify({'error': f'Preview failed: {str(e)}'}), 500

@app.route('/api/camera/stop', methods=['POST'])
@login_required
@admin_required
@csrf.exempt
def stop_camera():
    """Stop live camera"""
    global camera_cap, camera_running, camera_thread, camera_stats, auto_detection_running, auto_detection_thread

    try:
        # Stop auto-detection first if running
        if auto_detection_running:
            auto_detection_running = False
            if auto_detection_thread and auto_detection_thread.is_alive():
                auto_detection_thread.join(timeout=2)
                auto_detection_thread = None
            logger.info("Auto-detection stopped as part of camera shutdown")

        if camera_running:
            camera_running = False

        if camera_cap:
            camera_cap.release()
            camera_cap = None

        camera_stats['frames_processed'] = 0
        camera_stats['start_time'] = None

        logger.info("Camera stopped successfully")

        return jsonify({
            'success': True,
            'message': 'Camera stopped successfully',
            'stats': get_camera_stats()
        })

    except Exception as e:
        logger.error(f"Error stopping camera: {e}")
        return jsonify({'error': f'Failed to stop camera: {str(e)}'}), 500

@app.route('/api/camera/status')
@login_required
@admin_required
@csrf.exempt
def camera_status():
    """Get camera status"""
    return jsonify({
        'running': camera_running,
        'stats': get_camera_stats()
    })

@app.route('/api/camera/available')
@login_required
@admin_required
@csrf.exempt
def available_cameras():
    """Get list of available cameras (USB and IP options)"""
    cameras = []

    # Test USB cameras 0-2
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cameras.append({
                'index': i,
                'type': 'usb',
                'name': f'USB Camera {i}',
                'resolution': f"{int(width)}x{int(height)}"
            })
            cap.release()

    # Add IP Camera option
    cameras.append({
        'index': 'ip_camera',
        'type': 'ip',
        'name': 'IP Camera (Phone/Network Camera)',
        'resolution': 'Variable'
    })

    return jsonify({'cameras': cameras})

@app.route('/api/camera/test-ip', methods=['POST'])
@login_required
@admin_required
@csrf.exempt
def test_ip_camera():
    """Test IP camera connection"""
    try:
        data = request.get_json() or {}
        ip_url = data.get('ip_url', '').strip()

        if not ip_url:
            return jsonify({'error': 'IP camera URL is required'}), 400

        logger.info(f"Testing IP camera connection: {ip_url}")

        # Test the connection
        success, message = test_ip_camera_connection(ip_url)

        if success:
            return jsonify({
                'success': True,
                'message': message,
                'url': ip_url
            })
        else:
            return jsonify({
                'success': False,
                'error': message,
                'url': ip_url
            }), 400

    except Exception as e:
        logger.error(f"Error testing IP camera: {e}")
        return jsonify({'error': f'Test failed: {str(e)}'}), 500

# --- Auto-Detection API Routes ---
@app.route('/api/auto-detection/start', methods=['POST'])
@login_required
@admin_required
@csrf.exempt
def start_auto_detection():
    """Start intelligent automatic vehicle detection with duplicate prevention"""
    global auto_detection_running, auto_detection_thread, auto_detection_manager, camera_cap, camera_running

    if not camera_running or not camera_cap:
        return jsonify({'error': 'Camera must be running before starting automatic detection'}), 400

    if auto_detection_running:
        return jsonify({'error': 'Automatic detection is already running'}), 400

    if not auto_detection_manager:
        return jsonify({'error': 'Auto-detection manager not initialized'}), 500

    try:
        data = request.get_json() or {}
        intelligent_mode = data.get('intelligent_mode', True)

        # BULLETPROOF session initialization
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        auto_detection_manager.session_id = session_id
        auto_detection_manager.session_plates.clear()
        auto_detection_manager.processed_vehicles.clear()
        auto_detection_manager.frame_skip_counter = 0

        # Reset detection stats for new session
        auto_detection_manager.detection_stats = {
            'total_vehicles_seen': 0,
            'unique_vehicles_processed': 0,
            'duplicate_vehicles_blocked': 0,
            'unique_plates_found': 0,
            'duplicate_plates_blocked': 0,
            'database_saves': 0
        }

        logger.info(f"🚀 BULLETPROOF SESSION INITIALIZED: {session_id} - All tracking reset")

        auto_detection_running = True
        auto_detection_stats['start_time'] = datetime.now().isoformat()
        auto_detection_stats['vehicles_detected'] = 0
        auto_detection_stats['plates_processed'] = 0
        auto_detection_stats['session_id'] = session_id
        auto_detection_stats['intelligent_mode'] = intelligent_mode
        auto_detection_stats['bulletproof_mode'] = True

        # Start auto-detection thread
        auto_detection_thread = threading.Thread(
            target=auto_detection_manager.auto_detection_loop,
            args=(camera_cap,),
            daemon=True
        )
        auto_detection_thread.start()

        logger.info("Intelligent auto vehicle detection started successfully with duplicate prevention")

        return jsonify({
            'success': True,
            'message': 'Intelligent vehicle detection started with duplicate prevention',
            'features': {
                'duplicate_prevention': True,
                'intelligent_intervals': True,
                'session_tracking': True,
                'api_optimization': True
            },
            'stats': auto_detection_stats
        })

    except Exception as e:
        logger.error(f"Error starting auto-detection: {e}")
        return jsonify({'error': f'Failed to start auto-detection: {str(e)}'}), 500

@app.route('/api/auto-detection/stop', methods=['POST'])
@login_required
@admin_required
@csrf.exempt
def stop_auto_detection():
    """Stop automatic vehicle detection"""
    global auto_detection_running, auto_detection_thread

    try:
        auto_detection_running = False

        if auto_detection_thread and auto_detection_thread.is_alive():
            auto_detection_thread.join(timeout=2)  # Wait up to 2 seconds
            auto_detection_thread = None

        logger.info("Auto vehicle detection stopped successfully")

        return jsonify({
            'success': True,
            'message': 'Auto vehicle detection stopped',
            'stats': auto_detection_stats
        })

    except Exception as e:
        logger.error(f"Error stopping auto-detection: {e}")
        return jsonify({'error': f'Failed to stop auto-detection: {str(e)}'}), 500

@app.route('/api/auto-detection/status')
@login_required
@admin_required
@csrf.exempt
def auto_detection_status():
    """Get intelligent auto-detection status"""
    status_info = {
        'running': auto_detection_running,
        'stats': auto_detection_stats,
        'intelligent_features': {
            'duplicate_prevention': True,
            'session_tracking': True,
            'adaptive_intervals': True,
            'api_optimization': True
        }
    }

    if auto_detection_manager:
        status_info['session_info'] = {
            'session_id': auto_detection_manager.session_id,
            'unique_vehicles_processed': len(auto_detection_manager.processed_vehicles),
            'unique_plates_in_session': len(auto_detection_manager.session_plates),
            'detection_stats': auto_detection_manager.detection_stats,
            'memory_efficiency': 'optimized',
            'bulletproof_protection': 'active'
        }

        # Add list of plates in current session for debugging
        if auto_detection_manager.session_plates:
            status_info['current_session_plates'] = list(auto_detection_manager.session_plates.keys())

    return jsonify(status_info)

# Auto-detection settings endpoint removed - using intelligent optimization with fixed optimal parameters

@app.route('/api/auto-detection/test', methods=['POST'])
@login_required
@admin_required
@csrf.exempt
def test_auto_detection():
    """Test auto-detection with a static image (for development/testing)"""
    if not auto_detection_manager:
        return jsonify({'error': 'Auto-detection manager not initialized'}), 500

    try:
        # For testing, use a sample image from Resources folder
        test_image_path = PROJECT_ROOT / "Resources" / "car3.png"

        if not test_image_path.exists():
            return jsonify({'error': 'Test image not found'}), 404

        # Load test image
        test_frame = cv2.imread(str(test_image_path))
        if test_frame is None:
            return jsonify({'error': 'Could not load test image'}), 400

        # Detect vehicles in test frame
        vehicles = auto_detection_manager.detect_vehicles_in_frame(test_frame)

        result = {
            'vehicles_found': len(vehicles),
            'vehicles': vehicles,
            'test_image': test_image_path.name
        }

        # If vehicles found, process one
        if vehicles:
            largest_vehicle = max(vehicles, key=lambda v: v['area'])
            # Temporarily disable cooldown for testing
            original_cooldown = auto_detection_manager.detection_cooldown
            auto_detection_manager.detection_cooldown = 0

            try:
                detection_result = auto_detection_manager.process_detected_vehicle(test_frame, largest_vehicle)
                result['anpr_result'] = detection_result
            finally:
                # Restore original cooldown
                auto_detection_manager.detection_cooldown = original_cooldown

        logger.info(f"Auto-detection test completed: {result}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in auto-detection test: {e}")
        return jsonify({'error': f'Test failed: {str(e)}'}), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    if current_user.is_authenticated and current_user.is_admin():
        join_room('admins_room')
        logger.info(f"Admin user {current_user.username} (SID: {request.sid}) connected and joined 'admins_room'.")
    else:
        logger.warning(f"Non-admin user attempted to connect via SocketIO: {current_user.username if current_user.is_authenticated else 'Anonymous'}")

@socketio.on('disconnect')
def handle_disconnect(*args, **kwargs):
    """Handle client disconnection"""
    if current_user.is_authenticated:
        logger.info(f"Admin user {current_user.username} (SID: {request.sid}) disconnected.")

@socketio.on('join_camera_room')
def handle_join_camera_room():
    """Join camera streaming room"""
    join_room('camera_stream')
    logger.info(f"User {current_user.username} joined camera stream room")

@socketio.on('leave_camera_room')
def handle_leave_camera_room():
    """Leave camera streaming room"""
    leave_room('camera_stream')
    logger.info(f"User {current_user.username} left camera stream room")

# --- Main Execution ---
if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible on your network
    # Use debug=True only for development (provides debugger)
    # Set use_reloader=False to prevent restarts during ANPR processing
    logger.info("Starting Flask development server with SocketIO (reloader disabled).")
    # For 'production', set debug=False and use a production WSGI server like waitress
    socketio.run(app, host='127.0.0.1', port=8080, debug=True, use_reloader=False)
    # socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)

    # python -m src.web_app
