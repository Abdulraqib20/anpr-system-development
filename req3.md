## Technical Report: UNILORIN ANPR System

**Author:** Abdulraqib Omotosho

### Abstract

This report details the design, implementation, and functionality of the UNILORIN Automatic Number Plate Recognition (ANPR) System. The system is engineered to detect vehicles in images, identify their license plates, perform Optical Character Recognition (OCR) to extract plate numbers, and determine vehicle attributes such as type and color. All extracted data, along with processed images, are persisted in a PostgreSQL database and made accessible to users through an interactive web-based interface. The system integrates various machine learning models, including YOLO for object detection and a Groq-powered Large Language Model for OCR, orchestrated by a Python backend and presented via a Flask web application. This document covers the system architecture, core processing engine, web application components, security considerations, setup procedures, and potential avenues for future development.

### 1. Introduction

The UNILORIN ANPR System addresses the need for automated vehicle identification and data logging. Its primary objective is to provide a robust platform for processing user-uploaded images to extract and store vehicle license plate information along with associated metadata. This capability serves various applications, including traffic monitoring, security, and access control. The system is designed with a modular architecture, emphasizing scalability and maintainability, and leverages modern machine learning techniques for high accuracy and efficiency.

#### 1.1. Project Goal

The core goal of this project is to develop a comprehensive ANPR solution that seamlessly integrates image processing, machine learning, data storage, and user interaction. The system aims to:
1.  Accurately detect vehicles and their license plates in static images.
2.  Reliably extract alphanumeric characters from detected license plates using advanced OCR.
3.  Identify vehicle characteristics, specifically type (e.g., car, bus, truck) and color.
4.  Store all processed information and associated images (cropped plates, annotated frames) in a structured database.
5.  Provide an intuitive web-based user interface for image uploads, data review, filtering, and visualization of ANPR results.

#### 1.2. High-Level Architecture

The system is structured into three primary layers, facilitating a clear separation of concerns and modular development:

1.  **Core ANPR Processing Engine (`anpr_image.py`):** This is the computational heart of the system. Written in Python, it employs a suite of machine learning models and image processing libraries (OpenCV, Ultralytics YOLO, TensorFlow/Keras, Groq API) to analyze images. It handles vehicle detection, license plate localization, OCR, and vehicle attribute extraction.
2.  **Web Application Backend (`web_app.py`):** Built using the Flask microframework, this layer serves as the intermediary between the user and the ANPR engine. It manages HTTP requests, provides API endpoints for data exchange, handles user authentication and authorization, interacts with the PostgreSQL database, and orchestrates calls to the processing engine upon image uploads.
3.  **Frontend User Interface (`templates/`, `static/`):** This layer, constructed with HTML, CSS (Bootstrap), and JavaScript, provides the user-facing dashboard. It allows users to upload images, view detection results in a dynamic and interactive table, inspect detailed information and images via modals, observe system statistics, and browse a gallery of processed images.

The typical data flow initiates with an image upload from the Frontend. The Backend receives the request, temporarily stores the image, and invokes the ANPR Engine. The Engine processes the image, and the results are sent back to the Backend, which then persists them in the database and stores associated image files. The Frontend can then fetch and display this data, with a polling mechanism enabling near real-time updates on the dashboard.

*(A block diagram illustrating Frontend <-> Backend <-> ANPR Engine <-> Database/File System would be beneficial here.)*

### 2. Core ANPR Processing Engine (`anpr_image.py`)

The `anpr_image.py` script encapsulates all functionalities related to image analysis and ANPR data extraction. It is designed for efficiency and accuracy, leveraging multiple specialized machine learning models.

#### 2.1. Input Handling: The `ImageProcessor` Class

The `ImageProcessor` class is specifically designed for managing single image inputs. Upon initialization with an image file path (`source`), it validates the path's existence and confirms it is a file. The image is loaded using OpenCV (`cv2.imread`), and its dimensions (height, width) are stored. This class provides methods to retrieve the loaded image (`get_image`) and to save processed (e.g., annotated) images (`save_image`). Output filenames are uniquely generated with timestamps using the `get_output_path` function, ensuring no overwrites and easy traceability. Processed images are typically saved in the `output_plates` directory.

#### 2.2. Machine Learning Models

The ANPR engine integrates several machine learning models, each serving a distinct purpose in the processing pipeline:

1.  **Vehicle Detection (YOLOv8n):**
    *   **Model:** `models/yolov8n.pt`, a YOLOv8 Nano model loaded via the `ultralytics` Python library.
    *   **Purpose:** This model is responsible for identifying and localizing common vehicle types within the input image. It recognizes classes such as 'car', 'motorcycle', 'bus', and 'truck', as defined in the `VEHICLE_CLASSES` dictionary.
    *   **Implementation:** The `ANPRProcessor.detect_vehicle_type` method utilizes `vehicle_model.predict` to perform inference. A confidence threshold (e.g., `conf=0.5`) is applied during prediction to filter out low-certainty detections.

2.  **License Plate Detection (Custom YOLO Model):**
    *   **Model:** `models/license_plate_detector.pt`, assumed to be a YOLO-based model custom-trained or fine-tuned specifically for detecting license plates. It is also loaded using the `ultralytics` library.
    *   **Purpose:** This model pinpoints the bounding boxes of license plates within the image.
    *   **Implementation:** Integrated into the `ANPRProcessor.process_image` method, using `model.predict`. The `MIN_CONFIDENCE` constant (e.g., 0.45) controls the detection threshold.

3.  **Vehicle Color Classification (EfficientNet-based Model):**
    *   **Model:** `models/EFN-model.best.h5`, a Keras/TensorFlow model, likely based on the EfficientNet architecture, saved in HDF5 format.
    *   **Purpose:** To classify the color of a detected vehicle based on a cropped image of the vehicle. It categorizes colors into predefined classes listed in `COLOR_CLASSES` (e.g., 'black', 'blue', 'red').
    *   **Implementation:** The `ANPRProcessor.predict_vehicle_color` method uses `color_model.predict`. A custom layer, `FixedDepthwiseConv2D`, is registered during model loading to handle potential architectural incompatibilities with the saved Keras model.
    *   **Preprocessing:** Input vehicle crops are resized to 224x224 pixels, converted from BGR (OpenCV default) to RGB, and pixel values are normalized before being fed to the model.
    *   **Output:** The color class name corresponding to the highest prediction score is returned if this score exceeds a predefined threshold (0.3); otherwise, "unknown" is returned.

4.  **Optical Character Recognition (Groq Llama 3.1 Vision / Scout):**
    *   **Model:** The specific model is defined by `GROQ_MODEL_NAME` (e.g., `meta-llama/llama-4-scout-17b-16e-instruct`), accessed using the `groq` Python client library. This requires a `GROQ_API_KEY` to be configured in the environment.
    *   **Purpose:** This advanced vision-language model performs OCR on cropped license plate images to extract the alphanumeric characters.
    *   **Implementation:** The `ANPRProcessor._process_plate_with_groq` method handles the interaction.
    *   **Input:** The cropped plate image is first encoded into a base64 JPEG string.
    *   **Prompt Engineering:** A carefully crafted text prompt guides the LLM to extract *only* the 8-character uppercase alphanumeric plate string, minimizing the inclusion of extraneous text or symbols. The prompt explicitly requests an 8-character output, uppercase, with no hyphens or spaces.
    *   **Confidence:** A fixed high confidence score (`GROQ_CONFIDENCE`, e.g., 0.90) is assigned if the OCR result passes subsequent validation steps. This is because LLM confidence scores are not directly comparable to object detection confidence scores and often require heuristic assignment.

#### 2.3. The Processing Pipeline: `ANPRProcessor`

The `ANPRProcessor` class orchestrates the entire detection and recognition workflow.

*   **Initialization (`__init__`):**
    *   Loads all necessary machine learning models: vehicle detection (YOLOv8n), license plate detection (custom YOLO), and vehicle color classification (EfficientNet).
    *   Initializes the Groq client with the API key and specified model name.
    *   Establishes a PostgreSQL connection pool (`psycopg2.pool.SimpleConnectionPool`) using database credentials sourced from `config/appconfig.py` (which, in turn, reads from an `.env` file). This pool enhances database interaction efficiency by reusing connections.
    *   Ensures the `detected_plates` table exists in the database by calling `_ensure_table_exists` upon instantiation.
    *   Initializes a set `saved_plates` for session-level deduplication, though its primary role appears to be overridden by logic within `save_to_database` that allows for multiple detections of the same plate at different times.

*   **Core Image Processing (`process_image` method):**
    1.  Retrieves current timestamp details (time of day, day of week) using `get_time_details`.
    2.  Detects vehicles in the input frame using `detect_vehicle_type`.
    3.  Detects license plates using the custom YOLO plate detector (`model.predict`).
    4.  Iterates through each detected license plate bounding box:
        *   Crops the plate region from the frame (`plate_img`).
        *   Performs OCR on this cropped plate using `ocr_license_plate`, which internally calls `_process_plate_with_groq`.
        *   The raw OCR output is cleaned by `_clean_plate_text` (removes non-alphanumeric characters, converts to uppercase).
        *   The cleaned plate text is validated against `PLATE_REGEX` (Nigerian 8-character format: `^[A-Z0-9]{8}$`).
        *   A heuristic attempts to correct 7-character OCR results to 8 characters by substituting the last character based on common OCR misinterpretations (e.g., '0' for 'O').
        *   If a valid plate text is obtained (passes regex and the OCR confidence, which is fixed for Groq, is deemed sufficient):
            *   The plate is associated with the smallest detected vehicle whose bounding box encloses the plate's center coordinates. This association links the plate to a specific vehicle type.
            *   The color of this associated vehicle is then predicted using `predict_vehicle_color` on the vehicle's cropped region.
            *   The cropped license plate image is saved to the `PLATE_IMAGE_DIR` (e.g., `output_plates/plate_images/`) with a unique filename incorporating the plate text and a timestamp (e.g., `ABC123XY_timestamp.jpg`).
            *   All relevant detection data (plate text, OCR confidence, vehicle type, vehicle color, time details, filenames of the cropped plate and the eventually saved annotated frame, plate bounding box coordinates) is collected into a list named `processed_detections`.
            *   For visualization, bounding boxes for the plate and annotations (plate text, confidence) are drawn onto a copy of the input frame.
    5.  The annotated frame is displayed in an OpenCV window (`cv2.imshow`) for a brief period (5 seconds) for immediate visual feedback during standalone script execution. This window then closes automatically.
    6.  The method returns two items: the fully annotated frame (image) and the list of `processed_detections` (dictionaries).

*   **Image Preprocessing Considerations:**
    *   The `anpr_image.py` script contains a `preprocess_plate` method which implements several image preprocessing techniques: grayscale conversion, Contrast Limited Adaptive Histogram Equalization (CLAHE) for contrast enhancement, median blur for noise reduction, and adaptive thresholding for binarization.
    *   However, this preprocessing pipeline is **explicitly bypassed** within the `ocr_license_plate` method when using the Groq LLM. Modern, sophisticated vision models like those accessible via Groq often perform optimally on original, unprocessed images, as they have internal mechanisms for feature extraction that can be hindered by aggressive preprocessing. This preprocessing logic is retained and could be valuable if the system were to be adapted to use a different OCR engine (e.g., Tesseract OCR) that might benefit from such steps.

*   **OCR Implementation and Validation (`ocr_license_plate`, `_process_plate_with_groq`, `_clean_plate_text`):**
    *   The raw cropped plate image is directly sent to the Groq model via `_process_plate_with_groq`. This method handles encoding the image to a base64 JPEG string, constructing the API request with the engineered prompt, making the call to the Groq API, and parsing the returned text.
    *   `_clean_plate_text` standardizes this raw text.
    *   The cleaned text undergoes validation against `PLATE_REGEX`. If it's 7 characters, the heuristic correction is applied. If the plate is valid (8 characters, alphanumeric) after these steps, it's accepted with the `GROQ_CONFIDENCE`.

*   **Vehicle Attribute Extraction:**
    *   Vehicle type is determined by the YOLOv8n model during the `detect_vehicle_type` call.
    *   Vehicle color is predicted by the EfficientNet-based classification model using the `predict_vehicle_color` method on the cropped region of the vehicle associated with a valid license plate.

*   **Data Association and Filtering during Database Save:**
    *   Plate-to-vehicle association is based on geometric inclusion: the center of the plate's bounding box must fall within a vehicle's bounding box. If multiple vehicles satisfy this, the one with the smallest area is chosen.
    *   Further filtering occurs in the `save_to_database` method *before* database insertion:
        *   The `vehicle_color` must not be 'unknown'.
        *   The `plate_confidence` must meet the `MIN_CONFIDENCE` threshold.
        *   `detection_count` (which is effectively 1 for single image processing) must meet `MIN_DETECTIONS` (typically 1).

*   **Output Generation:**
    *   The `process_image` method directly returns the OpenCV image object of the annotated frame.
    *   Cropped plate images are saved individually to `PLATE_IMAGE_DIR` by `process_image`.
    *   The full, annotated frame containing all detections is saved later by the `ImageProcessor.save_image` method within the `process_image_source` wrapper function.

#### 2.4. Data Persistence

The system ensures that all relevant extracted data and associated images are stored systematically.

*   **Database Schema (`detected_plates` table):**
    *   This table is automatically created if it doesn't exist by the `_ensure_table_exists` method, which is called during `ANPRProcessor` initialization.
    *   Key columns include:
        *   `id`: Serial primary key.
        *   `start_time`, `end_time`: Timestamps marking the processing period (effectively the same for single image processing).
        *   `license_plate`: The extracted 8-character plate string (VARCHAR(20)).
        *   `confidence`: The OCR confidence score (REAL).
        *   `detection_count`: Number of times detected (INTEGER, typically 1 for images).
        *   `vehicle_type`: Detected vehicle type (VARCHAR(50)).
        *   `vehicle_color`: Detected vehicle color (VARCHAR(50)).
        *   `time_of_day`: e.g., "morning", "afternoon" (VARCHAR(20)).
        *   `day_of_week`: e.g., "Monday" (VARCHAR(20)).
        *   `image_filename`: Filename of the saved cropped plate image (VARCHAR(255)), stored in `PLATE_IMAGE_DIR`.
        *   `annotated_frame_filename`: Filename of the saved full annotated frame (VARCHAR(255)), stored in `OUTPUT_DIR`.

*   **Database Interaction (`save_to_database` method):**
    *   Leverages the PostgreSQL connection pool (`db_pool`) for efficient database operations.
    *   Accepts a list of detection data dictionaries and the filename of the saved annotated frame.
    *   Applies the filtering rules (vehicle color not 'unknown', plate confidence >= `MIN_CONFIDENCE`).
    *   Constructs SQL `INSERT` statements to add valid detections to the `detected_plates` table.
    *   Uses `cursor.executemany` for potentially inserting multiple records in a batch, although for single image processing, this batch usually contains few items.
    *   The current implementation **allows duplicate license plate entries** in the database if they are detected at different times or in different processing sessions, as there is no `ON CONFLICT` clause in the `INSERT` statement to handle unique constraints on the plate number itself. It does, however, ensure that within a single call to `save_to_database` (i.e., for a single processed image), only the first instance of a unique plate text is kept if multiple identical plate texts were somehow generated.
    *   Database operations are performed within transactions, with `conn.commit()` on success and `conn.rollback()` on error.

*   **File Storage:**
    *   `OUTPUT_DIR` (e.g., `project_root/output_plates/`): The base directory for storing processed image outputs.
    *   `PLATE_IMAGE_DIR` (e.g., `project_root/output_plates/plate_images/`): A subdirectory within `OUTPUT_DIR` specifically for storing the small, cropped images of detected license plates (e.g., `ABC123XY_timestamp.jpg`). These are referenced by the `image_filename` column in the database.
    *   Annotated full frames (the original input image with bounding boxes and text drawn on it) are saved directly into `OUTPUT_DIR` (e.g., `original_filename_stem_timestamp.jpg`). These are referenced by the `annotated_frame_filename` column in the database.

#### 2.5. Configuration and Logging

System behavior is managed through configuration files and environment variables, and its operation is recorded through a robust logging mechanism.

*   **Configuration:**
    *   Primarily managed via `config/appconfig.py`, which sources values from environment variables.
    *   A `.env` file in the project root is used to store sensitive information like database credentials (`DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_PORT`) and the `GROQ_API_KEY`. The `python-dotenv` library (`load_dotenv()`) is used to load these variables into the environment when the application starts.
    *   Paths to machine learning models, confidence thresholds (`MIN_CONFIDENCE`, `GROQ_CONFIDENCE`), and other operational parameters are also defined, either directly in `anpr_image.py` or in `appconfig.py`.

*   **Logging:**
    *   Utilizes Python's built-in `logging` module.
    *   A logger instance named "ANPR" is configured in the global scope of `anpr_image.py`.
    *   Log messages are output to both the console (via `StreamHandler`) and a dedicated log file (`logs/config.log` via `FileHandler`).
    *   The logging level is set to `DEBUG`, capturing detailed information about the system's execution flow, decisions, and potential issues.
    *   To prevent log duplication from multiple initializations (e.g., during development or if handlers were added elsewhere), existing handlers are cleared from the "ANPR" logger before new ones are added, and `propagate` is set to `False`.

### 3. Web Application Backend (`web_app.py`)

The Flask-based web application serves as the control center for user interaction, data management, and ANPR processing requests.

#### 3.1. Framework and Structure (Flask)

*   The backend is built using Flask, a lightweight and flexible Python web framework.
*   It follows a standard Flask project structure:
    *   `templates/`: Contains Jinja2 HTML templates for rendering web pages. Located in `src/templates/`.
    *   `static/`: Contains static assets like CSS stylesheets, JavaScript files, and images. Located in `src/static/`.
*   Essential project directories such_as `OUTPUT_DIR`, `PLATE_IMAGE_DIR`, `UPLOADS_DIR` (for temporary storage of uploaded files), and `LOGS_DIR` are defined relative to the project root (`PROJECT_ROOT`), ensuring portability. These directories are created if they don't exist when the application starts.
*   The Flask application instance is initialized as `app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))`, explicitly defining the paths to template and static asset folders.
*   A secret key (`app.secret_key`), sourced from the `FLASK_SECRET_KEY` environment variable or a default development key, is configured for session management, CSRF protection, and flashing messages.
*   CSRF (Cross-Site Request Forgery) protection is enabled globally for the application using `Flask-WTF`'s `CSRFProtect`.

#### 3.2. User Authentication and Authorization

The web application implements user authentication and role-based access control using `Flask-Login`, `Flask-WTF` for forms, and `werkzeug.security` for password hashing.

*   **User Model (`User` class):** A `User` class inheriting from `Flask-Login`'s `UserMixin` is defined. It stores user `id`, `username`, and `role` (e.g., 'user', 'admin').
*   **Database Table (`users`):** A `users` table in the PostgreSQL database stores user credentials. It includes `id` (SERIAL PRIMARY KEY), `username` (UNIQUE NOT NULL), `password_hash` (NOT NULL), `role` (NOT NULL, default 'user'), and `created_at`. The `_ensure_users_table_exists` function creates this table and a default 'admin' user if they don't exist on startup. The default admin password can be set via the `ADMIN_PASSWORD` environment variable.
*   **Forms (`LoginForm`, `RegistrationForm`):** `Flask-WTF` forms are used for user login and registration, providing input fields and validation (e.g., data required, length constraints, password confirmation, username uniqueness check against the database).
*   **Login Management (`LoginManager`):**
    *   `login_manager.init_app(app)` integrates Flask-Login.
    *   `login_manager.login_view = 'login'` specifies the route name for the login page, to which users are redirected if they try to access a protected page without being authenticated.
    *   `@login_manager.user_loader`: The `load_user(user_id)` function is registered to retrieve a user object from the database given their ID.
*   **Routes for Authentication:**
    *   `/register` (GET, POST): Allows new users to create an account. Validates form data, hashes the password using `generate_password_hash`, and inserts the new user into the `users` table.
    *   `/login` (GET, POST): Allows existing users to log in. Validates credentials by comparing the provided password (after hashing) with the stored `password_hash` using `check_password_hash`. If successful, `login_user` establishes a session.
    *   `/logout`: Logs out the current user using `logout_user` and redirects to the login page.
*   **Access Control:**
    *   `@login_required`: Decorator from Flask-Login used to protect routes that require an authenticated user.
    *   `@admin_required`: A custom decorator is implemented to restrict access to certain routes (e.g., `/admin/*`, `/upload`) to users with the 'admin' role. This decorator checks if `current_user.is_authenticated` and `current_user.is_admin()`.

#### 3.3. API Endpoints and Request Handling

The backend exposes several HTTP endpoints to serve the frontend and manage operations:

*   **`GET /` (`index` route, protected by `@login_required`):**
    *   Renders the main dashboard page (`index.html`).
    *   Handles optional server-side search (`search` query parameter) and pagination (`page` query parameter) for initial data loading, although the primary, interactive display is now driven by client-side JavaScript fetching all data.
    *   Fetches initial statistics (e.g., count of detections made today) and a list of annotated frame images from the file system for the image gallery.
    *   Passes this data, along with a CSRF token (generated by `generate_csrf`), to the Jinja2 template for rendering.

*   **`GET /api/detections` (`api_detections` route):**
    *   This is a crucial JSON API endpoint designed for the frontend JavaScript.
    *   It fetches **all** detection records from the `detected_plates` table in the database, ordered by `end_time` in descending order.
    *   It selects only the columns necessary for the client-side table (e.g., `id`, `license_plate`, `start_time`, `confidence`, `vehicle_type`, `vehicle_color`, `image_filename`, `annotated_frame_filename`).
    *   `datetime` objects retrieved from the database are formatted into ISO 8601 strings to ensure JSON compatibility.
    *   Returns a JSON response structured as `{"detections": [...]}` on success, or `{"error": "message", "detections": []}` on failure.

*   **`POST /upload` (`upload_image` route, protected by `@login_required` and `@admin_required`):**
    *   Handles image file uploads from authenticated admin users via an HTTP POST request.
    *   Expects the image file to be part of the form data under the key `imageFile`.
    *   Validates file presence and ensures the file extension is within `ALLOWED_EXTENSIONS` (e.g., 'png', 'jpg', 'jpeg').
    *   Generates a unique and secure filename for the uploaded file using `secure_filename` and prepends a timestamp to prevent collisions.
    *   The uploaded file is *temporarily* saved to the `UPLOADS_DIR`.
    *   Crucially, it then calls `anpr_processor.process_image_source(str(temp_save_path))`, passing the path to the temporarily saved image. This triggers the entire core ANPR processing pipeline described earlier.
    *   Returns a JSON response: `{"success": True, "message": "File processed..."}` or `{"success": False, "error": "Processing failed..."}`.
    *   Includes comprehensive error handling to catch issues such_as the ANPR processor not being ready, file errors during upload/saving, or exceptions raised during the ANPR processing itself.
    *   A `finally` block ensures that the temporary uploaded file in `UPLOADS_DIR` is **deleted** after processing, regardless of success or failure, to conserve disk space.

*   **`GET /output_images/<path:filename>` (`serve_output_image` route):**
    *   Serves the full annotated frame images stored in `OUTPUT_DIR`. These are used by the image gallery in the frontend.
    *   Uses Flask's `send_from_directory` function for secure file serving from the specified absolute directory path.
    *   Includes basic security checks to prevent path traversal attempts (e.g., checking for '..' in filename, ensuring the file is directly in `OUTPUT_DIR` and not a subdirectory like `plate_images`).

*   **`GET /plate_images/<path:filename>` (`serve_plate_image` route):**
    *   Serves the smaller, cropped license plate images stored in `PLATE_IMAGE_DIR` (a subdirectory of `OUTPUT_DIR`). These are displayed as thumbnails in the detections table and in the detection detail modal.
    *   Also uses `send_from_directory` with similar path traversal checks.

*   **`GET /about` (`about` route):**
    *   Renders a static "About" page (`about.html`) providing information about the project.

*   **Admin Routes (e.g., `/admin`, `/admin/users`, `/admin/user/change_role/<id>`, `/admin/user/delete/<id>`):**
    *   Protected by `@login_required` and `@admin_required`.
    *   `/admin/admin_dashboard.html`: Renders an admin dashboard.
    *   `/admin/users`: Displays a list of all registered users with options to manage them.
    *   `/admin/user/change_role/<user_id>` (POST): Allows an admin to promote a 'user' to 'admin' or demote an 'admin' to 'user'. Prevents admins from changing their own role.
    *   `/admin/user/delete/<user_id>` (POST): Allows an admin to delete a user account. Prevents admins from deleting their own account.

*   **`POST /detection/delete/<int:detection_id>` (`delete_detection` route, protected by `@login_required` and `@admin_required`):**
    *   Allows an admin to delete a specific detection record from the database.
    *   It first retrieves the filenames of the associated cropped plate image and annotated frame image from the database record.
    *   Then, it deletes the record from the `detected_plates` table.
    *   Finally, it attempts to delete the corresponding image files from the filesystem (`PLATE_IMAGE_DIR` and `OUTPUT_DIR`).
    *   Uses database transactions and provides feedback to the user via flashed messages.

#### 3.4. Integration with ANPR Engine

*   A single global instance of the `ANPRProcessor` class (`anpr_processor = ANPRProcessor()`) is created when the Flask application (`web_app.py`) starts.
*   This instance is reused by the `/upload` route for every image processing request. This approach is crucial because initializing the `ANPRProcessor` (which involves loading multiple machine learning models) is a time-consuming and resource-intensive operation. Reusing the instance avoids this overhead on each upload.
*   The application includes error handling at startup to log if the `anpr_processor` fails to initialize. If it's not initialized, the `/upload` route will return an error, preventing processing attempts.

#### 3.5. Database Connectivity

*   The web application utilizes the same PostgreSQL connection pool (`db_pool`) that is initialized and used by the `ANPRProcessor` in `anpr_image.py`. This pool is configured in `web_app.py` using `psycopg2.pool.SimpleConnectionPool` with parameters from `appconfig.py`.
*   To manage database connections efficiently within the context of web requests, Flask's application context (`g`) is used:
    *   `get_db()`: This helper function attempts to retrieve a database connection from the `db_pool`. If a connection is successfully obtained, it's stored in `g.db` for the duration of the current request. If the pool is unavailable or fails to provide a connection, `g.db` will be `None`.
    *   `@app.teardown_appcontext def close_db(error):`: This function is registered to be called automatically when an application context ends (i.e., after a request has been processed). It retrieves the connection from `g.db` (if it exists) and returns it to the `db_pool` using `db_pool.putconn(db)`. This ensures that connections are properly released back to the pool, even if errors occur during request handling.

#### 3.6. Request Handling, Error Management, and Logging

*   Flask's route decorators (`@app.route(...)`) are used to map URLs to view functions.
*   Request data is accessed via Flask's `request` object (e.g., `request.args` for URL query parameters, `request.files` for uploaded files, `request.form` for form data).
*   `jsonify` is used to create well-formed JSON responses for API endpoints (like `/api/detections`) and the `/upload` route.
*   Flask's `flash()` mechanism is used to display non-critical messages to the user (e.g., success or failure of login, registration, user management actions).
*   Error handling is implemented throughout the application:
    *   Try-except blocks are used to catch potential exceptions, such as `psycopg2.Error` for database issues, `FileNotFoundError` for file operations, and general `Exception` for unexpected errors.
    *   Flask's `abort(code)` function is used to return standard HTTP error responses (e.g., `abort(404)` for "Not Found", `abort(500)` for "Internal Server Error") when serving files or in critical error conditions.
    *   Detailed error information, including tracebacks (`traceback.format_exc()`), is logged for debugging purposes, especially in the `/upload` route where complex processing occurs.
*   **Logging:**
    *   A dedicated logger instance named "ANPR_WebApp" is configured.
    *   Logs are written to `logs/web.log` using a `RotatingFileHandler` (rotates when the file reaches 1MB, keeping 3 backup copies) and also to the console via `StreamHandler`.
    *   The `werkzeug` logger (Flask's underlying WSGI development server) is also configured to use these same handlers. Its `propagate` attribute is set to `False` to prevent its logs from being duplicated by the root logger or the "ANPR" logger (from `anpr_image.py`), ensuring cleaner and non-redundant log files.

### 4. Frontend User Interface (`templates/`, `static/`)

The frontend is designed to provide a responsive and interactive experience for users to upload images, monitor detections, and analyze results.

#### 4.1. Templating (Jinja2) and Structure

*   Jinja2, Flask's default templating engine, is used for server-side rendering of HTML pages.
*   **`base.html`:** This is the master template that defines the common layout for all pages. It includes:
    *   The main HTML structure (head, body).
    *   Links to Bootstrap CSS (via Bootswatch "Brite" theme CDN by default, with a theme switcher), Bootstrap Icons, and custom CSS (`static/css/style.css`).
    *   A responsive navigation bar (navbar) with links to "Detections" (index page), "About", a theme switcher dropdown, and user authentication links (Login/Register or Username/Logout).
    *   A project banner section.
    *   A footer with project information and relevant links.
    *   Defines Jinja2 blocks (`{% raw %}{% block title %}{% endraw %}`, `{% raw %}{% block content %}{% endraw %}`, `{% raw %}{% block head_extra %}{% endraw %}`, `{% raw %}{% block scripts_extra %}{% endraw %}`) that child templates can override to inject page-specific content, titles, or additional scripts/styles.
*   **`index.html`:** This template extends `base.html` and defines the content for the main dashboard. It includes:
    *   The image upload form (only visible to admins).
    *   Statistics cards (e.g., "Total Detections," "Detections Today").
    *   The structure for the interactive detections table.
    *   A section for the image gallery of annotated frames.
    *   Bootstrap modals for displaying detection details and larger gallery images.
    *   It receives initial data from the Flask backend (e.g., initial stats, gallery image list, CSRF token) when the page is first loaded. The bulk of the dynamic content (like the detections table) is then populated by client-side JavaScript.
*   **Other templates:** `login.html`, `register.html`, `about.html`, and admin-specific templates like `admin_dashboard.html` and `admin_users.html` also extend `base.html`.

#### 4.2. Key UI Components

*   **Image Upload Interface (Admin-only):**
    *   A form (`#uploadForm`) allows admin users to select an image file for processing.
    *   Provides visual feedback during the upload process: a progress bar (`#uploadProgressBar`) and status text (`#uploadStatus`).
    *   Displays success or error messages (`#uploadResult`) returned from the backend after the upload attempt.
    *   Includes a "Cancel Upload" button (`#cancelUploadButton`) to abort an ongoing upload.
    *   The CSRF token is included as a hidden field in this form.

*   **Statistics Dashboard:**
    *   A set of cards displays key metrics. Initially, "Detections Today" is populated server-side. The "Total Detections" count and "Detections Today" are dynamically updated by JavaScript using data fetched from the `/api/detections` endpoint.

*   **Interactive Detections Table:**
    *   This is the central component for displaying ANPR results.
    *   **Client-Side Powered:** The table body (`#detections-table-body`) is populated and managed entirely by JavaScript. On page load (and periodically), the script fetches *all* detection data from the `/api/detections` endpoint. This client-side approach allows for fast filtering, sorting, and pagination without needing to reload the page or make repeated calls for partial data.
    *   **Filtering:** Users can filter the displayed detections using:
        *   Dropdown menus for `vehicle_type` (`#filterVehicleType`), `vehicle_color` (`#filterVehicleColor`), `time_of_day` (`#filterTimeOfDay`), and `day_of_week` (`#filterDayOfWeek`).
        *   A text input (`#plateSearchInput`) for searching by license plate text (case-insensitive partial match).
    *   **Sorting:** Table headers (`th` elements with class `sortable`) are clickable, enabling client-side sorting of the data by columns such as Plate Text, Detection Time, Confidence, Vehicle Type, and Color. The sorting logic handles different data types (strings, numbers, datetime strings) and indicates the current sort column and direction (ascending/descending) with icons.
    *   **Pagination:** Client-side pagination controls (`#paginationControls`) are dynamically generated based on the total number of (filtered) detections and a predefined `ITEMS_PER_PAGE` constant in the JavaScript. Users can navigate through pages of results.
    *   **Visuals:** Each row in the table can display:
        *   A thumbnail image of the cropped license plate (served by `/plate_images/...`).
        *   The recognized plate text.
        *   Vehicle type and color (with a color badge for visual emphasis).
        *   Detection timestamp.
        *   Confidence score, potentially with a visual bar.
    *   **Interactivity:** Table rows have a `clickable-row` class. Clicking a row opens the "Detection Details Modal" to show more information about that specific detection.
    *   **Loading/Error States:**
        *   A loading indicator (`#loadingIndicator`) is shown while the initial data is being fetched from the API.
        *   An error message (`#tableErrorPlaceholder`) is displayed if the API call fails.
        *   A "No Detections Found" message (`#noDetectionsMessage`) appears if the current filters result in an empty dataset or if there are no detections in the database.
    *   **Delete Button (Admin-only):** For each detection row, an admin user sees a delete button that triggers a POST request to `/detection/delete/<detection_id>` to remove the detection.

*   **Image Gallery:**
    *   Displays thumbnails of the full annotated output frames (served by `/output_images/...`).
    *   Clicking a thumbnail in the gallery opens the "Gallery View Modal" to display a larger version of that image. Images are loaded lazily as the user scrolls or interacts.

*   **Modals (Bootstrap Components):**
    *   **Detection Details Modal (`#detectionDetailModal`):**
        *   Triggered by clicking a row in the detections table.
        *   Populated dynamically with data from the clicked row (passed via `data-*` attributes on the row).
        *   Displays:
            *   A larger image of the cropped license plate (`#modalPlateImage`).
            *   A larger image of the full annotated frame (`#modalAnnotatedImage`).
            *   Detailed textual information: plate text, confidence, vehicle type, vehicle color, and timestamps.
    *   **Gallery View Modal (`#galleryImageModal`):**
        *   Triggered by clicking a thumbnail in the image gallery.
        *   Displays a large version (`#modalGalleryImage`) of the selected annotated frame.

#### 4.3. Client-Side Logic (JavaScript)

The core client-side interactivity is managed by JavaScript code embedded within `<script>` tags in `index.html`.

*   **Asynchronous Data Fetching (`fetchAndUpdateData`):**
    *   Uses the `fetch` API to retrieve all detection data from the `/api/detections` endpoint. This occurs on initial page load and is also set up to run periodically via `setInterval` (polling) to check for new detections.
    *   It compares the newly fetched data (e.g., by count or a hash of the data) with the current data to determine if an update to the UI is necessary, optimizing re-renders.

*   **Dynamic Table Rendering (`renderTablePage`):**
    *   This function is responsible for populating the detections table (`#detections-table-body`).
    *   It takes the current page of `filteredDetections` as input.
    *   It clears the existing table rows and then iterates through the detections for the current page, creating new `<tr>` elements for each.
    *   Data is formatted for display (e.g., dates are made more readable, confidence scores might be formatted as percentages, color badges are applied).
    *   Necessary CSS classes and `data-*` attributes (containing all details for the modal) are added to each row and cell.

*   **Client-Side Filtering, Sorting, and Pagination:**
    *   `applyFilterAndSort()`: This is the central function for updating the displayed detections.
        *   It reads the current values from all filter controls (dropdowns, search input).
        *   It filters the master `allDetections` array based on these criteria.
        *   It then sorts the resulting `filteredDetections` array based on `currentSortKey` and `currentSortDirection` (which are updated when table headers are clicked).
        *   After filtering and sorting, it resets `currentPage` to 1 and calls `renderTablePage()` to display the first page of the new results and `renderPaginationControls()` to update the pagination links.
    *   `renderPaginationControls()`: Dynamically generates the pagination links (Previous, Next, page numbers) based on the total number of `filteredDetections` and `ITEMS_PER_PAGE`. It handles disabling links (e.g., "Previous" on the first page) and can show an ellipsis for a large number of pages.

*   **Upload Handling:**
    *   The image upload form (`#uploadForm`) submission is handled using `XMLHttpRequest` (XHR) instead of a standard form submission. This allows for:
        *   Monitoring upload progress: The XHR `progress` event is used to update the `#uploadProgressBar`.
        *   Cancelling the upload: The `xhr.abort()` method is called if the `#cancelUploadButton` is clicked.
    *   The JavaScript updates the `#uploadStatus` text during the upload.
    *   It handles success, error, and cancellation scenarios by parsing the JSON response from the `/upload` endpoint and displaying appropriate feedback to the user using a helper function like `showUploadResult`.
    *   The upload UI elements (form, progress bar) are reset (`resetUploadUI`) after the operation completes or is cancelled.
    *   On successful processing of an uploaded image, `fetchAndUpdateData(true)` is called to force an immediate refresh of the detections table and stats, ensuring the new detection appears promptly.

*   **Modal Interaction:**
    *   Event listeners are attached to rows in the detections table (elements with `clickable-row` class). When a row is clicked, a handler function (`attachModalListener` or similar) reads the detection details from the row's `data-*` attributes and uses them to populate the content of the `#detectionDetailModal` (e.g., setting image sources for `#modalPlateImage` and `#modalAnnotatedImage`, and filling in text elements). The modal is then displayed using Bootstrap's JavaScript API.
    *   For the `#galleryImageModal`, Bootstrap's modal events (like `show.bs.modal`) are used. When a gallery thumbnail is clicked, the `href` or a `data-src` attribute of the thumbnail is used to set the `src` of the `#modalGalleryImage` just before the modal is shown.

*   **Data Polling:**
    *   `setInterval(fetchAndUpdateData, POLLING_INTERVAL_MS)` is used to periodically call `fetchAndUpdateData`. `POLLING_INTERVAL_MS` defines the refresh rate (e.g., every 10-30 seconds). This provides near real-time updates on the dashboard if new detections are added to the database by other means or by the current user's uploads, without requiring manual page reloads.

*   **Theme Switcher:**
    *   JavaScript fetches a list of available Bootswatch themes (e.g., from a predefined list or a CDN API if available).
    *   It populates the `#theme-switcher-menu` dropdown with these theme names.
    *   When a theme is selected, JavaScript dynamically changes the `href` attribute of the main Bootstrap CSS `<link>` tag (`#bootswatch-theme-link`) to the URL of the selected theme's CSS file.
    *   The selected theme is saved in `localStorage` so that the user's preference persists across sessions and page reloads. On page load, the script checks `localStorage` and applies the saved theme.

#### 4.4. Styling (Bootstrap and Custom CSS)

*   **Bootstrap 5:** The primary styling framework is Bootstrap 5, loaded via a CDN. The "Brite" theme from Bootswatch is used as the default, providing a modern and clean appearance. The theme can be changed dynamically by the user via the theme switcher.
*   **Bootstrap Icons:** Used for iconography throughout the application (e.g., in the navbar, buttons, table headers). Loaded via CDN.
*   **Custom CSS (`static/css/style.css`):** This file contains custom styles to:
    *   Override or augment Bootstrap default styles.
    *   Style specific components like the image gallery, sort indicators on table headers, modal content.
    *   Define styles for loading states, error messages, and other UI enhancements not covered by Bootstrap.
    *   Implement hover effects and other visual details.
    *   Enhance footer styling, navigation bar appearance, and overall visual consistency.

### 5. System Setup and Deployment

Setting up and running the UNILORIN ANPR system involves preparing the environment, configuring dependencies, and starting the web application.

#### 5.1. Dependencies

The system requires Python 3.x. A `requirements.txt` file should ideally list all Python package dependencies with their versions. Key libraries include:

*   **Core ANPR & ML:**
    *   `opencv-python`: For image loading, manipulation, and display.
    *   `ultralytics`: For interacting with YOLO models (vehicle and plate detection).
    *   `tensorflow` and `keras`: For running the EfficientNet-based vehicle color classification model. (Ensure version compatibility, especially with the saved `.h5` model and custom layers).
    *   `numpy`: For numerical operations, especially array manipulations for image data.
    *   `groq`: The official Python client library for interacting with the Groq API (for OCR).
*   **Web Application & Database:**
    *   `Flask`: The web framework for the backend.
    *   `psycopg2-binary`: PostgreSQL adapter for Python, enabling database communication.
    *   `python-dotenv`: For loading environment variables from a `.env` file.
    *   `Werkzeug`: WSGI utility library, used by Flask (implicitly installed).
    *   `Jinja2`: Templating engine for Flask (implicitly installed).
*   **Authentication & Forms:**
    *   `Flask-Login`: For handling user sessions and authentication.
    *   `Flask-WTF`: For creating and validating web forms, including CSRF protection.
    *   `WTForms`: The forms library used by Flask-WTF.
    *   `email_validator`: Often a dependency for WTForms email validation, if used.
*   **Utilities:**
    *   `Pillow` (PIL Fork): Often a dependency for image handling with web frameworks or ML libraries.

Installation is typically done using pip: `pip install -r requirements.txt`.

#### 5.2. Configuration Management

*   **Environment Variables (`.env` file):** A crucial aspect of configuration. A `.env` file should be created in the project root. This file is **not** to be committed to version control if it contains sensitive credentials. It should define:
    *   `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_PORT`: Credentials for connecting to the PostgreSQL database.
    *   `GROQ_API_KEY`: The API key for accessing the Groq service for OCR.
    *   `FLASK_SECRET_KEY`: A long, random string used by Flask for cryptographic signing (sessions, CSRF).
    *   `ADMIN_PASSWORD` (optional): To set the initial password for the default 'admin' user upon first setup.
*   **Application Configuration (`config/appconfig.py`):** This Python module reads the environment variables (loaded by `load_dotenv()` in `web_app.py` and `anpr_image.py`) and makes them available as Python constants. This centralizes configuration access.
*   **Model Paths and Thresholds:** Paths to the ML models (`.pt`, `.h5` files) and various confidence thresholds (e.g., `MIN_CONFIDENCE` for plate detection) are typically defined as constants within `anpr_image.py` or could also be moved to `appconfig.py` for better organization.

#### 5.3. Database Setup

*   A running PostgreSQL server instance is required.
*   The database specified by `DB_NAME` in the `.env` file must exist on the PostgreSQL server, or the `DB_USER` must have privileges to create it.
*   The `detected_plates` and `users` tables will be created automatically by the application (`_ensure_table_exists` in `anpr_image.py` and `_ensure_users_table_exists` in `web_app.py`) on its first run if they do not already exist, provided the database user has the necessary permissions (CREATE TABLE).

#### 5.4. Running the Application

*   **Development Mode:**
    1.  Ensure all dependencies are installed.
    2.  Ensure the `.env` file is correctly configured in the project root.
    3.  Ensure the PostgreSQL server is running and accessible with the credentials in `.env`.
    4.  Navigate to the `src/` directory in the terminal.
    5.  Run the Flask development server using: `python web_app.py`.
    6.  The application will typically be accessible at `http://0.0.0.0:5000` or `http://127.0.0.1:5000`.
    *   The `web_app.py` script currently runs with `debug=True` and `use_reloader=False`.
        *   `debug=True` enables Flask's interactive debugger and automatic reloading of Python modules when changes are detected.
        *   `use_reloader=False` is important because the `ANPRProcessor` (with its heavy model loading) is initialized globally. If the reloader were enabled (`True`), it might cause the ANPRProcessor to re-initialize frequently upon code changes, which is very slow. For development where backend Python code (like `web_app.py` or `anpr_image.py`) is frequently modified, one might temporarily set `use_reloader=True` and accept the re-initialization, or manually restart the server after significant changes.

*   **Production Deployment:**
    *   The Flask development server (`app.run()`) is **not suitable for production environments** due to its single-threaded nature and lack of robustness.
    *   For production, a dedicated WSGI (Web Server Gateway Interface) server such as Gunicorn (common on Linux) or Waitress (cross-platform, pure Python) should be used.
    *   This WSGI server would typically run behind a reverse proxy like Nginx or Apache. The reverse proxy handles incoming HTTP requests, serves static files efficiently, manages SSL/TLS termination, and can provide load balancing if needed.
    *   In a production setting, `debug` mode in Flask must be set to `False`.
    *   Environment variables (including secrets) should be managed securely, often through the deployment environment's configuration mechanisms rather than a plain `.env` file directly on the server.
    *   Directory permissions for uploads, outputs, and logs must be set correctly.

### 6. Security Considerations

While this system provides core ANPR functionality, several security aspects are important:

*   **Authentication and Authorization:** Implemented via Flask-Login. Passwords are hashed. Role-based access restricts sensitive operations like image uploads and user management to 'admin' users.
*   **CSRF Protection:** Flask-WTF provides CSRF protection for forms, helping prevent malicious cross-site requests.
*   **Input Validation:**
    *   File uploads: File extensions are checked (`allowed_file`), and `secure_filename` is used to sanitize filenames.
    *   Form data: WTForms validators are used for login and registration forms.
    *   API inputs: While not explicitly detailed for `/api/detections` (as it's a GET request with no body), any future POST/PUT API endpoints should rigorously validate incoming data.
*   **Path Traversal:** Basic checks (`'..'` in filename, `filename.startswith('/')`) are implemented in `serve_output_image` and `serve_plate_image` to prevent users from accessing files outside the designated directories. `send_from_directory` also offers some protection.
*   **Secret Management:** Sensitive information (API keys, database passwords, Flask secret key) is intended to be stored in an `.env` file, which should be excluded from version control and secured in the deployment environment.
*   **Database Security:** Using parameterized queries (implicitly by `psycopg2` when arguments are passed as tuples) helps prevent SQL injection vulnerabilities. Database user privileges should be limited to the minimum necessary for the application.
*   **Error Handling & Logging:** Detailed error logging can inadvertently expose sensitive system information if logs are not properly secured. The current setup logs tracebacks, which is useful for debugging but needs consideration in production.
*   **Dependencies:** Regularly update all dependencies (Python packages, OS libraries) to patch known vulnerabilities. A tool like `pip-audit` can help identify vulnerable packages.
*   **HTTPS:** In a production environment, the application must be served over HTTPS to encrypt data in transit. This is typically handled by the reverse proxy (e.g., Nginx).

### 7. Conclusion and Future Work

#### 7.1. Strengths

*   **Modular Architecture:** Clear separation between the ANPR engine, backend API, and frontend UI promotes maintainability and independent development of components.
*   **Comprehensive Feature Set:** The system integrates vehicle detection, license plate detection, advanced OCR using an LLM, vehicle color classification, robust data storage, and a feature-rich interactive web interface.
*   **Use of Modern Technologies:** Leverages popular and powerful libraries/frameworks including YOLOv8, TensorFlow/Keras, Groq LLMs, Flask, and Bootstrap 5.
*   **Interactive and Responsive Frontend:** Client-side data handling (fetching all data, then filtering/sorting/paginating locally) provides a fast and smooth user experience for browsing detections. Real-time polling keeps the dashboard updated.
*   **Rich Visual Feedback:** Users are provided with annotated full frames, cropped plate images, upload progress indicators, and clear visual cues in the UI.
*   **User Authentication and Admin Panel:** Secure access to functionalities with role-based permissions and dedicated admin views for user and detection management.
*   **Robust Error Handling and Logging:** Extensive try-except blocks and detailed logging (to console and files) are implemented throughout the backend and ANPR engine, aiding in debugging and monitoring.
*   **Efficient Database Operations:** The use of a PostgreSQL connection pool (`SimpleConnectionPool`) optimizes database interactions.

#### 7.2. Limitations and Potential Improvements

*   **Single Image Processing Focus:** The current system is optimized for processing individual images uploaded one at a time. It does not natively support continuous video stream processing or batch processing of multiple image files.
*   **OCR Dependency and Cost:** Reliance on the external Groq API for OCR means the system requires an active internet connection and a valid `GROQ_API_KEY`. This may involve costs, rate limits, and latency dependent on the API provider. The accuracy is also tied to the performance of the chosen Groq model.
*   **Basic OCR Correction Heuristic:** The 7-to-8 character plate correction is a simple rule-based heuristic and might not cover all OCR error patterns or could potentially miscorrect plates.
*   **Vehicle-Plate Association Simplicity:** The current method of associating a plate with the smallest vehicle whose bounding box contains the plate's center point might be insufficient in complex scenes with multiple, closely packed, or overlapping vehicles.
*   **Performance with Large Datasets:** While client-side rendering of the detections table is responsive for moderate datasets, fetching *all* detections via `/api/detections` could become a performance bottleneck (both for the backend generating the JSON and the frontend processing it) if the `detected_plates` table grows to contain tens or hundreds of thousands of records.
*   **Scalability of Global ANPR Processor:** The single global `ANPRProcessor` instance, while efficient for avoiding re-initialization, could become a contention point if the Flask application were deployed in a multi-process or multi-threaded WSGI environment handling many concurrent upload requests (though this is less critical for the current admin-only upload design).
*   **Deployment Complexity for Production:** Moving to a production environment requires careful setup of a WSGI server, reverse proxy, PostgreSQL, and secure management of configurations and secrets.
*   **Limited Test Coverage:** The report does not detail automated test suites (unit, integration tests), which are crucial for ensuring long-term reliability and easier refactoring.

#### 7.3. Future Work

Several enhancements could further improve the UNILORIN ANPR System:

1.  **Video Stream Processing:** Extend capabilities to process real-time video streams from cameras or recorded video files, requiring frame-by-frame analysis, object tracking, and more sophisticated temporal data aggregation.
2.  **Alternative/Hybrid OCR Solutions:**
    *   Explore open-source OCR engines like Tesseract (perhaps with fine-tuning) for an offline or lower-cost alternative.
    *   Investigate training or fine-tuning a dedicated, lightweight OCR model specifically for license plates.
    *   Implement a hybrid approach where Groq is used if available/preferred, with a fallback to a local OCR engine.
3.  **Advanced Vehicle-Plate Association:** Implement more robust association logic, possibly using Intersection over Union (IoU) between plate and vehicle boxes, or tracking algorithms if video processing is added.
4.  **Performance Optimizations:**
    *   For ML model inference: Explore techniques like model quantization, conversion to optimized runtimes (e.g., TensorRT, ONNX Runtime).
    *   For the API: Implement server-side pagination, filtering, and searching for the `/api/detections` endpoint to efficiently handle very large databases. This would involve modifying the client-side JavaScript to request data in chunks.
5.  **Enhanced User Management and Features:**
    *   Implement user profile pages, password reset functionality.
    *   Introduce more granular roles and permissions if needed.
6.  **Improved Plate Validation and Correction:** Develop more sophisticated algorithms for validating plate syntax and correcting OCR errors, potentially using dictionaries of known plate segments or machine learning-based correction models.
7.  **Batch Image Upload and Processing:** Allow users to upload and process multiple images in a single batch.
8.  **Containerization:** Package the application (ANPR engine, web app, dependencies) using Docker for simplified deployment, scalability, and environment consistency. Create a `docker-compose.yml` to orchestrate the application and database services.
9.  **Comprehensive Automated Testing:** Develop a suite of unit tests (for individual functions/classes) and integration tests (for interactions between components like API calls, database operations, and ANPR processing flow).
10. **System Monitoring and Analytics:** Integrate tools for monitoring application performance, error rates, and resource usage. Develop more advanced analytics dashboards within the UI to visualize trends in detected data.
11. **Data Export Functionality:** Allow users (especially admins) to export detection data in formats like CSV or Excel for external analysis.

This technical report provides a comprehensive overview of the UNILORIN ANPR system, detailing its current capabilities and laying the groundwork for future enhancements.
