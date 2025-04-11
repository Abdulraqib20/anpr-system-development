import os
import sys
from pathlib import Path
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from flask import Flask, render_template, jsonify, g, send_from_directory, abort
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

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
# Ensure the directories exist (optional here, as anpr_image should create them)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLATE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging Setup ---
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True) # Ensure logs directory exists
LOG_FILE_PATH = LOGS_DIR / "web.log"

# Get a specific logger instance for the web app
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ANPR_WebApp") # Specific name for this logger
logger.setLevel(logging.INFO) 

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

# --- Routes ---

@app.route('/')
def index():
    """Renders the main page with the latest detections, stats, and image gallery."""
    logger.info("Request received for index page ('/')")
    conn = get_db()
    detections = []
    plate_images = [] # Initialize list for image gallery data
    stats = {
        'total_detections': 0,
        'today_detections': 0
    }
    error_message = None

    # --- Fetch Detections and Stats from DB ---
    if conn:
        try:
            with conn.cursor() as cur:
                # Fetch recent detections, including image_filename
                logger.debug("Executing DB query for recent detections.")
                cur.execute("""
                    SELECT license_plate, start_time, end_time, confidence, 
                           detection_count, vehicle_type, vehicle_color, 
                           time_of_day, day_of_week, image_filename
                    FROM detected_plates
                    ORDER BY end_time DESC
                    LIMIT 50 
                """)
                colnames = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                detections = [dict(zip(colnames, row)) for row in rows]
                logger.info(f"Fetched {len(detections)} recent detections.")

                # Fetch total detections count
                logger.debug("Executing DB query for total detection count.")
                cur.execute("SELECT COUNT(*) FROM detected_plates")
                total_count_result = cur.fetchone()
                if total_count_result:
                    stats['total_detections'] = total_count_result[0]
                    logger.info(f"Total detections count: {stats['total_detections']}")

                # Fetch detections count for today
                logger.debug("Executing DB query for today's detection count.")
                # Assuming end_time is a TIMESTAMP or TIMESTAMPTZ column
                cur.execute("SELECT COUNT(*) FROM detected_plates WHERE DATE(end_time) = CURRENT_DATE")
                today_count_result = cur.fetchone()
                if today_count_result:
                    stats['today_detections'] = today_count_result[0]
                    logger.info(f"Today's detections count: {stats['today_detections']}")

        except psycopg2.Error as e:
            logger.error(f"Database query error on index page: {e}", exc_info=True)
            error_message = f"Database Error: Could not retrieve data."
            # Reset stats if DB error occurs after fetching some data
            stats = {'total_detections': 'Error', 'today_detections': 'Error'}

    else:
        error_message = "Database connection not available."
        logger.error("Database connection pool not available for index request.")
        stats = {'total_detections': 'N/A', 'today_detections': 'N/A'}

    # --- Fetch Images for Gallery ---
    try:
        # Scan OUTPUT_DIR for full annotated frames
        logger.info(f"Scanning for annotated frame images in: {OUTPUT_DIR}")
        image_files = []
        if OUTPUT_DIR.is_dir(): # Check the correct directory
            # Iterate through files, getting Path objects
            for item in OUTPUT_DIR.iterdir(): # Iterate the correct directory
                # Filter for files ending with .jpg or .png (adjust as needed)
                if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # Exclude files from the plate_images subdirectory
                    if not item.parent.name == PLATE_IMAGE_DIR.name:
                        # Get modification time for sorting
                        mtime = item.stat().st_mtime
                        image_files.append((item, mtime))

            # Sort by modification time, newest first
            image_files.sort(key=lambda x: x[1], reverse=True)

            # Limit the number of images displayed? (e.g., latest 100)
            max_gallery_images = 100
            image_files = image_files[:max_gallery_images]

            # Process filenames
            for img_path, _ in image_files:
                filename = img_path.name
                # For full frames, maybe just use the filename or a simplified label
                plate_text = filename # Or derive a label differently if needed
                plate_images.append({'filename': filename, 'plate_text': plate_text})

            logger.info(f"Found {len(plate_images)} annotated images for the gallery (max: {max_gallery_images}).")
        else:
            logger.warning(f"Annotated image directory not found or is not a directory: {OUTPUT_DIR}")

    except Exception as e:
        logger.error(f"Error scanning annotated image directory: {e}", exc_info=True)
        # Optionally set an error message for the gallery part
        error_message = error_message + " | Error loading image gallery." if error_message else "Error loading image gallery."

    # --- Render Template ---
    template_path = TEMPLATE_DIR / 'index.html'
    if not template_path.is_file():
        logger.error(f"Template file not found at expected path: {template_path}")
        return f"Error: Template 'index.html' not found at {template_path}", 500
    
    logger.debug("Rendering index.html template with detections, stats, and image gallery data.")
    try:
        return render_template('index.html', 
                               detections=detections, 
                               stats=stats, 
                               plate_images=plate_images, # Pass gallery data
                               error=error_message)
    except Exception as render_error:
        logger.error(f"Error rendering template 'index.html': {render_error}", exc_info=True)
        return f"Error rendering template: {render_error}", 500


@app.route('/api/detections')
def api_detections():
    """Provides detection data as JSON for dynamic updates."""
    # Use the specific logger
    logger.info("Request received for API endpoint ('/api/detections')")
    conn = get_db()
    detections = []
    error_message = None
    status_code = 200

    if conn:
        try:
            with conn.cursor() as cur:
                logger.debug("Executing DB query for API.")
                cur.execute("""
                    SELECT license_plate, start_time, end_time, confidence, 
                           detection_count, vehicle_type, vehicle_color, 
                           time_of_day, day_of_week
                    FROM detected_plates
                    ORDER BY end_time DESC
                    LIMIT 50
                """)
                colnames = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                detections = []
                for row_tuple in rows:
                    row_dict = {}
                    for i, col_name in enumerate(colnames):
                        value = row_tuple[i]
                        if isinstance(value, datetime):
                            row_dict[col_name] = value.isoformat()
                        else:
                            row_dict[col_name] = value
                    detections.append(row_dict)
                logger.info(f"Fetched {len(detections)} detections for API.")

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
        logger.debug("Returning successful API response.")
        return jsonify({"detections": detections})

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

# --- Main Execution ---
if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible on your network
    # Use debug=True only for development (auto-reloads, provides debugger)
    # For 'production' on the Pi, set debug=False and use a production WSGI server like gunicorn or waitress
    logger.info("Starting Flask development server.")
    app.run(host='0.0.0.0', port=5000, debug=True) # Set debug=False for production 