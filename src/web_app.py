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

# --- Flask-Login Imports ---
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
# -------------------------

# --- Flask-WTF Imports (for forms) ---
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect, generate_csrf
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired, Length, EqualTo, ValidationError
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
try:
    # Instantiate it once when the app starts
    anpr_processor = ANPRProcessor()
    logger.info("ANPRProcessor initialized successfully.")
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

            # --- Create a default admin user if it doesn't exist ---
            cursor.execute("SELECT id FROM users WHERE username = %s", ('admin',))
            if not cursor.fetchone():
                default_admin_password = os.environ.get('ADMIN_PASSWORD', 'default_admin_pass_123') # Consider a more secure default or prompt
                hashed_password = generate_password_hash(default_admin_password)
                cursor.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)",
                    ('admin', hashed_password, 'admin')
                )
                conn.commit()
                logger.info(f"Default admin user 'admin' created. PLEASE CHANGE THE DEFAULT PASSWORD IF APPLICABLE.")
            else:
                logger.info("Admin user 'admin' already exists.")
            # ----------------------------------------------------------

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

        except psycopg2.Error as e:
            logger.error(f"Database query error on index page (server-side part): {e}", exc_info=True)
            error_message = f"Database Error: Could not retrieve initial data."
            stats = {'today_detections': 'Error'}
            total_detections = 'Error'
            total_pages = 1
            page = 1
    else:
        error_message = "Database connection not available for initial data."
        stats = {'today_detections': 'N/A'}
        total_detections = 'N/A'
        total_pages = 1
        page = 1

    # --- Fetch Images for Gallery (This part is fine as is) ---
    try:
        logger.info(f"Scanning for annotated frame images in: {OUTPUT_DIR}")
        image_files = []
        if OUTPUT_DIR.is_dir():
            for item in OUTPUT_DIR.iterdir():
                if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    if not item.parent.name == PLATE_IMAGE_DIR.name:
                        mtime = item.stat().st_mtime
                        image_files.append((item, mtime))
            image_files.sort(key=lambda x: x[1], reverse=True)
            max_gallery_images = 100
            image_files = image_files[:max_gallery_images]
            for img_path, _ in image_files:
                filename = img_path.name
                plate_text = filename
                plate_images.append({'filename': filename, 'plate_text': plate_text})
            logger.info(f"Found {len(plate_images)} annotated images for the gallery (max: {max_gallery_images}).")
        else:
            logger.warning(f"Annotated image directory not found or is not a directory: {OUTPUT_DIR}")
    except Exception as e:
        logger.error(f"Error scanning annotated image directory: {e}", exc_info=True)
        current_error = "Error loading image gallery."
        error_message = f"{error_message} | {current_error}" if error_message else current_error

    # --- Render Template ---
    raw_csrf_token = generate_csrf() # Generate the raw token value

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
    return render_template('admin/admin_dashboard.html', title='Admin Dashboard')

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

    if not conn:
        usage_data['error'] = "Database connection not available."
        logger.error("Groq Usage Metrics: Database connection not available.")
        return render_template('admin/admin_usage_metrics.html', title='Groq API Usage Metrics', usage_data=usage_data)

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

    return render_template('admin/admin_usage_metrics.html', title='Groq API Usage Metrics', usage_data=usage_data)

# ----------------------------------------------

# --- Main Execution ---
if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible on your network
    # Use debug=True only for development (provides debugger)
    # Set use_reloader=False to prevent restarts during ANPR processing
    logger.info("Starting Flask development server (reloader disabled).")
    # For 'production', set debug=False and use a production WSGI server like waitress
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
