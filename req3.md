## Technical Report: UNILORIN ANPR System

**Author:** Abdulraqib Omotosho

**Table of Contents:**

1.  **Introduction & System Overview**
    *   Project Goal
    *   High-Level Architecture
2.  **Core ANPR Processing Engine (`anpr_image.py`)**
    *   Input Handling (`ImageProcessor`)
    *   Machine Learning Models
        *   Vehicle Detection (YOLOv8n)
        *   License Plate Detection (YOLO Custom)
        *   Vehicle Color Classification (EfficientNet)
        *   Optical Character Recognition (Groq Llama 3.1 Vision)
    *   Processing Pipeline (`ANPRProcessor`)
        *   Initialization
        *   Vehicle & Plate Detection Flow
        *   Image Preprocessing (Considerations)
        *   OCR Implementation & Validation
        *   Vehicle Attribute Extraction (Type & Color)
        *   Data Association & Filtering
        *   Output Generation (Annotated Images, Cropped Plates)
    *   Data Persistence
        *   Database Schema (`detected_plates` table)
        *   Database Interaction (psycopg2, Connection Pooling)
        *   File Storage (Plate Images, Annotated Frames)
    *   Configuration & Logging
3.  **Web Application Backend (`web_app.py`)**
    *   Framework & Structure (Flask)
    *   API Endpoints
        *   `/` (Dashboard)
        *   `/api/detections` (Data API)
        *   `/upload` (Image Processing Trigger)
        *   `/output_images/<filename>` (Annotated Image Serving)
        *   `/plate_images/<filename>` (Cropped Plate Serving)
    *   Integration with ANPR Engine
    *   Database Connectivity
    *   Request Handling & Error Management
    *   Logging
4.  **Frontend User Interface (`templates/`, `static/`)**
    *   Templating (Jinja2) & Structure (`base.html`, `index.html`)
    *   Key UI Components
        *   Image Upload Interface
        *   Statistics Dashboard
        *   Interactive Detections Table
        *   Image Gallery
        *   Modals (Detection Details, Gallery View)
    *   Client-Side Logic (JavaScript)
        *   Asynchronous Data Fetching (`/api/detections`)
        *   Dynamic Table Rendering
        *   Client-Side Filtering, Sorting, Pagination
        *   Upload Handling (XHR, Progress Bar, Cancellation)
        *   Modal Interaction
        *   Data Polling for Real-time Updates
    *   Styling (Bootstrap, Custom CSS)
5.  **System Setup & Deployment**
    *   Dependencies
    *   Configuration Management (`.env`, `config/appconfig.py`)
    *   Database Setup (PostgreSQL)
    *   Running the Application
6.  **Conclusion & Future Work**
    *   Strengths
    *   Limitations & Potential Improvements

***

### 1. Introduction & System Overview

#### Project Goal

The primary objective of this project is to develop an Automatic Number Plate Recognition (ANPR) system capable of detecting vehicles, identifying their license plates, extracting the plate characters, determining vehicle attributes (type, color), and storing this information for review and analysis via a web-based user interface. This system is designed to process individual images uploaded by the user.

#### High-Level Architecture

The system employs a modular architecture consisting of three main layers:

1.  **Core ANPR Processing Engine:** Handles the heavy lifting of image analysis using various machine learning models. Written in Python using libraries like OpenCV, Ultralytics YOLO, TensorFlow/Keras, and Groq.
2.  **Web Application Backend:** A Flask-based web server that provides API endpoints, serves the user interface, manages user requests (like image uploads), interacts with the database, and orchestrates calls to the ANPR engine.
3.  **Frontend User Interface:** A web-based dashboard built with HTML, CSS (Bootstrap), and JavaScript. It allows users to upload images, view processed detections in a sortable/filterable table, see statistics, and browse a gallery of processed images.

Data flows typically from the Frontend (image upload) -> Backend (request handling) -> ANPR Engine (processing) -> Backend (database storage, response generation) -> Frontend (displaying results). A polling mechanism is also implemented for near real-time updates on the dashboard.

![High-Level Architecture Diagram Placeholder](placeholder_diagram.png)
*(Suggestion: Include a simple block diagram here showing Frontend <-> Backend <-> ANPR Engine <-> Database/File System)*

***

### 2. Core ANPR Processing Engine (`anpr_image.py`)

This Python script forms the heart of the system, responsible for analyzing input images and extracting relevant ANPR data.

#### Input Handling (`ImageProcessor`)

*   The `ImageProcessor` class is designed specifically for handling single image inputs.
*   It takes an image file path as input (`source`).
*   Validates the existence and type (must be a file) of the source path.
*   Loads the image using OpenCV (`cv2.imread`) during initialization.
*   Provides methods to retrieve the loaded image (`get_image`) and save processed images (`save_image`) with unique, timestamped filenames generated by `get_output_path`.
*   Stores image dimensions (`height`, `width`).

#### Machine Learning Models

The system leverages multiple pre-trained and custom-trained models:

1.  **Vehicle Detection (YOLOv8n):**
    *   Model: `models/yolov8n.pt` (YOLOv8 Nano) loaded via the `ultralytics` library.
    *   Purpose: Detects common vehicle types (car, motorcycle, bus, truck) within the frame. Uses class IDs defined in `VEHICLE_CLASSES`.
    *   Implementation: `ANPRProcessor.detect_vehicle_type` uses `vehicle_model.predict`.
    *   Confidence Threshold: Set within the prediction call (e.g., `conf=0.5`).
2.  **License Plate Detection (YOLO Custom):**
    *   Model: `models/license_plate_detector.pt` (Assumed to be a YOLO model fine-tuned for license plate detection). Loaded via `ultralytics`.
    *   Purpose: Detects the bounding boxes of license plates within the frame.
    *   Implementation: `ANPRProcessor.process_image` uses `model.predict`.
    *   Confidence Threshold: Controlled by `MIN_CONFIDENCE` (e.g., 0.45).
3.  **Vehicle Color Classification (EfficientNet):**
    *   Model: `models/EFN-model.best.h5` (A Keras/TensorFlow model, likely based on EfficientNet).
    *   Purpose: Classifies the color of a cropped vehicle region into predefined categories (`COLOR_CLASSES`).
    *   Implementation: `ANPRProcessor.predict_vehicle_color` uses `color_model.predict`.
    *   Custom Layer: Requires `FixedDepthwiseConv2D` during loading to handle potential compatibility issues with the saved model's architecture.
    *   Preprocessing: Resizes input to 224x224, converts BGR to RGB, normalizes pixel values.
    *   Output: Returns the color class name if the top prediction confidence exceeds a threshold (0.3), otherwise "unknown".
4.  **Optical Character Recognition (Groq Llama 3.1 Vision):**
    *   Model: Specified by `GROQ_MODEL_NAME` (e.g., `llama-3.2-90b-vision-preview`), accessed via the `groq` Python client. Requires `GROQ_API_KEY`.
    *   Purpose: Performs OCR on cropped license plate images to extract alphanumeric characters.
    *   Implementation: `ANPRProcessor._process_plate_with_groq`.
    *   Input: Base64 encoded JPEG representation of the plate image.
    *   Prompt Engineering: A specific text prompt guides the model to extract *only* the 8-character uppercase alphanumeric plate string, minimizing extraneous text.
    *   Confidence: A fixed high confidence (`GROQ_CONFIDENCE`, e.g., 0.90) is assigned if the result passes validation, as LLM confidence scores are not directly comparable to object detection scores.

#### Processing Pipeline (`ANPRProcessor`)

The `ANPRProcessor` class orchestrates the detection and recognition process:

*   **Initialization (`__init__`):**
    *   Loads all ML models (Vehicle, Plate, Color).
    *   Initializes the Groq client.
    *   Sets up a PostgreSQL connection pool (`psycopg2.pool.SimpleConnectionPool`) using configuration from `appconfig.py`/`.env`.
    *   Ensures the `detected_plates` database table exists by calling `_ensure_table_exists`.
    *   Initializes `saved_plates` set (though its session-level deduplication role seems bypassed in `save_to_database`).
*   **Vehicle & Plate Detection Flow (`process_image`):**
    1.  Gets current time details (`get_time_details`).
    2.  Detects vehicles in the frame using `detect_vehicle_type`.
    3.  Detects license plates using the custom YOLO model (`model.predict`).
    4.  Iterates through detected plate bounding boxes:
        *   Crops the plate region (`plate_img`).
        *   Performs OCR on the cropped plate using `ocr_license_plate` (which calls `_process_plate_with_groq`).
        *   Validates OCR result format and applies correction heuristic if needed.
        *   If a valid plate is found (passes regex `PLATE_REGEX` and confidence threshold):
            *   Associates the plate with the smallest enclosing detected vehicle based on the plate's center coordinates.
            *   Predicts the color of the associated vehicle (`predict_vehicle_color`).
            *   Saves the cropped plate image to `PLATE_IMAGE_DIR` with a unique filename (`{plate_text}_{timestamp}.jpg`).
            *   Collects detection data (plate text, confidence, vehicle type/color, time details, image filenames, bounding box) into `processed_detections`.
            *   Draws bounding boxes and annotations onto the input frame for visualization.
    5.  Displays the annotated frame using OpenCV (`cv2.imshow`) with a 5-second timeout.
    6.  Returns the annotated frame and the list of `processed_detections`.
*   **Image Preprocessing (Considerations):**
    *   The `preprocess_plate` method exists, implementing grayscale conversion, CLAHE (contrast enhancement), median blur, and adaptive thresholding.
    *   However, it's **explicitly skipped** in `ocr_license_plate` when using the Groq model, as advanced vision models often perform better on original, unprocessed images. This preprocessing logic might be relevant if switching to a different OCR engine (e.g., Tesseract).
*   **OCR Implementation & Validation (`ocr_license_plate`, `_process_plate_with_groq`, `_clean_plate_text`):**
    *   The original cropped plate image is sent to Groq.
    *   `_process_plate_with_groq` handles image encoding (JPEG base64), API call construction (with the specific prompt), and result parsing.
    *   `_clean_plate_text` standardizes the raw OCR output by removing non-alphanumeric characters and converting to uppercase.
    *   The cleaned text is validated against `PLATE_REGEX` (Nigerian 8-character format).
    *   A simple heuristic attempts to correct 7-character results by substituting the last character based on common OCR errors (e.g., '0' -> 'O').
*   **Vehicle Attribute Extraction:**
    *   Type is determined by the YOLOv8n vehicle detection model.
    *   Color is predicted by the EfficientNet model on the associated vehicle's cropped region.
*   **Data Association & Filtering:**
    *   Plates are associated with the vehicle bounding box containing the plate's center.
    *   Filtering occurs primarily in `save_to_database` *before* insertion:
        *   `vehicle_color` must not be 'unknown'.
        *   `detection_count` (seems fixed at 1 for image processing) must meet `MIN_DETECTIONS`.
        *   `plate_confidence` must meet `MIN_CONFIDENCE`.
*   **Output Generation:**
    *   The `process_image` method returns the annotated frame (suitable for display or saving).
    *   Cropped plate images are saved individually to `PLATE_IMAGE_DIR`.
    *   The fully annotated frame is saved later by `ImageProcessor.save_image` in `process_image_source`.

#### Data Persistence

*   **Database Schema (`detected_plates` table):**
    *   Managed by `_ensure_table_exists`.
    *   Columns: `id`, `start_time`, `end_time` (timestamps), `license_plate`, `confidence`, `detection_count`, `vehicle_type`, `vehicle_color`, `time_of_day`, `day_of_week`, `image_filename` (cropped plate), `annotated_frame_filename` (full frame).
*   **Database Interaction (`save_to_database`):**
    *   Uses a connection pool (`db_pool`) for efficiency.
    *   Takes a list/dict of detections and the filename of the saved annotated frame.
    *   Applies filtering rules (color, confidence, count).
    *   Constructs `INSERT` SQL statements.
    *   Uses `cursor.executemany` for potentially batch inserting records (though batching logic seems minimal for single-image processing).
    *   Explicitly allows duplicate license plates if detected at different times (no `ON CONFLICT` clause). Handles potential duplicates *within the same save batch* by keeping only the first instance encountered.
    *   Uses transactions (`conn.commit()`, `conn.rollback()`).
*   **File Storage:**
    *   `OUTPUT_DIR`: Base directory for outputs.
    *   `PLATE_IMAGE_DIR`: Stores cropped plate images (`<plate>_<timestamp>.jpg`). Referenced by `image_filename` in the DB.
    *   Annotated full frames are saved directly in `OUTPUT_DIR` (e.g., `<original_stem>_<timestamp>.jpg`). Referenced by `annotated_frame_filename` in the DB.

#### Configuration & Logging

*   **Configuration:** Managed via `config/appconfig.py` and environment variables loaded using `python-dotenv` (`load_dotenv()`). Key variables include DB credentials, Groq API key, model paths, confidence thresholds.
*   **Logging:** Uses Python's `logging` module.
    *   Configured in `anpr_image.py`'s global scope.
    *   Logger named "ANPR".
    *   Outputs logs to both the console (`StreamHandler`) and a file (`logs/config.log` via `FileHandler`).
    *   Level set to `DEBUG`, providing detailed operational information.
    *   Prevents duplicate logs by clearing existing handlers and setting `propagate = False`.

***

### 3. Web Application Backend (`web_app.py`)

This Flask application serves as the user interface gateway and orchestrator.

#### Framework & Structure (Flask)

*   Uses the Flask microframework.
*   Standard Flask structure: `templates` folder for HTML, `static` folder for CSS/JS/images.
*   Project directories (`OUTPUT_DIR`, `PLATE_IMAGE_DIR`, `UPLOADS_DIR`, `LOGS_DIR`) are defined relative to the project root.
*   `app = Flask(__name__, ...)` initializes the Flask app, explicitly setting template and static folders.
*   Uses a secret key for session management/flashing (`app.secret_key`).

#### API Endpoints

*   **`GET /` (`index`):**
    *   Renders the main dashboard (`index.html`).
    *   Handles optional `search` query parameter for initial server-side filtering (though the primary filtering/display is now client-side).
    *   Handles pagination parameters (`page`) for server-side pagination (also superseded by client-side JS).
    *   Fetches initial stats (today's count) and gallery images from the file system.
    *   Passes data to the Jinja2 template.
*   **`GET /api/detections` (`api_detections`):**
    *   Provides a JSON endpoint for the frontend JavaScript.
    *   Fetches **all** records from the `detected_plates` table, ordered by `end_time` descending.
    *   Selects columns needed for the client-side table.
    *   Formats `datetime` objects into ISO strings for JSON compatibility.
    *   Returns a JSON object `{"detections": [...]}` or `{"error": ..., "detections": []}`.
*   **`POST /upload` (`upload_image`):**
    *   Handles image file uploads via HTTP POST request.
    *   Expects a file in the `imageFile` form field.
    *   Validates file presence and extension (`allowed_file`, `ALLOWED_EXTENSIONS`).
    *   Generates a unique, secure filename (`secure_filename`, timestamp).
    *   Saves the uploaded file *temporarily* to the `UPLOADS_DIR`.
    *   Calls the `anpr_processor.process_image_source` method to trigger the core ANPR processing.
    *   Returns a JSON response indicating success (`{"success": True, "message": ...}`) or failure (`{"success": False, "error": ...}`).
    *   Includes error handling (ANPR processor not ready, file errors, processing exceptions).
    *   Crucially, **removes the temporary uploaded file** in a `finally` block to prevent disk space issues.
*   **`GET /output_images/<path:filename>` (`serve_output_image`):**
    *   Serves annotated full-frame images stored in `OUTPUT_DIR`. Used by the image gallery.
    *   Uses `send_from_directory` for secure file serving.
    *   Includes basic path traversal checks.
*   **`GET /plate_images/<path:filename>` (`serve_plate_image`):**
    *   Serves cropped plate images stored in `PLATE_IMAGE_DIR`. Used by the detections table and detail modal.
    *   Uses `send_from_directory`. Includes basic path traversal checks.

#### Integration with ANPR Engine

*   An instance of `ANPRProcessor` is created globally (`anpr_processor = ANPRProcessor()`) when the Flask app starts.
*   This single instance is used by the `/upload` route to process images, avoiding the overhead of re-initializing models for each request.
*   Error handling checks if `anpr_processor` was initialized successfully.

#### Database Connectivity

*   Uses the same `psycopg2.pool.SimpleConnectionPool` as the ANPR engine, initialized globally (`db_pool`).
*   Implements request-scoped database connections using Flask's application context (`g`):
    *   `get_db()`: Retrieves a connection from the pool or returns None if unavailable. Stores it in `g.db`.
    *   `@app.teardown_appcontext close_db`: Ensures the connection is returned to the pool (`db_pool.putconn()`) after each request, even if errors occur.

#### Request Handling & Error Management

*   Uses Flask's routing decorators (`@app.route`).
*   Parses request arguments (`request.args`) for search/pagination and files (`request.files`).
*   Uses `jsonify` to create JSON responses for API endpoints and the upload route.
*   Implements error handling for database connection issues, query errors (`psycopg2.Error`), file handling problems, and ANPR processing exceptions.
*   Uses `abort()` for serving file errors (404 Not Found, 500 Internal Server Error).
*   Uses `traceback.format_exc()` in the upload route for detailed error logging.

#### Logging

*   Configures a separate logger named "ANPR_WebApp".
*   Uses `RotatingFileHandler` to write logs to `logs/web.log`, rotating the file when it reaches 1MB.
*   Also logs to the console (`StreamHandler`).
*   Configures the `werkzeug` logger (Flask's development server) to use the same handlers and disables propagation to avoid duplicate logs in the core engine's log file (`config.log`).

***

### 4. Frontend User Interface (`templates/`, `static/`)

The frontend provides an interactive web dashboard for users.

#### Templating (Jinja2) & Structure

*   Uses Jinja2 for server-side HTML templating.
*   `base.html`: Defines the main page structure (navbar, project banner, footer), includes Bootstrap CSS/JS, and defines blocks (`{% block title %}`, `{% block content %}`, etc.) for content injection.
*   `index.html`: Extends `base.html` and defines the content for the main dashboard page, including the upload form, stats cards, detections table structure, gallery section, and modals. It receives initial data (stats, gallery images) from the Flask backend.

#### Key UI Components

*   **Image Upload Interface:** A form (`#uploadForm`) allowing users to select and upload image files. Includes visual feedback during upload (progress bar `#uploadProgressBar`, status text `#uploadStatus`) and displays success/error messages (`#uploadResult`). Provides a cancel button (`#cancelUploadButton`).
*   **Statistics Dashboard:** Cards displaying 'Total Detections' and 'Detections Today'. Values are updated dynamically using JavaScript based on data fetched from `/api/detections`.
*   **Interactive Detections Table:**
    *   The primary display area for ANPR results.
    *   **Client-Side Powered:** The table body (`#detections-table-body`) is populated dynamically by JavaScript after fetching *all* detections from `/api/detections`. Server-side pagination/filtering in the `index` route is largely superseded.
    *   **Filtering:** Multiple dropdowns (`#filterVehicleType`, etc.) and a text input (`#plateSearchInput`) allow users to filter the displayed data client-side.
    *   **Sorting:** Table headers (`th.sortable`) are clickable, allowing client-side sorting by various columns (plate, time, confidence, etc.). Handles different data types (string, number, datetime). Visual indicators show the current sort column and direction.
    *   **Pagination:** Client-side pagination controls (`#paginationControls`) are generated dynamically based on the number of filtered detections and `ITEMS_PER_PAGE`.
    *   **Visuals:** Includes thumbnail images of cropped plates, formatted confidence bars, and color badges.
    *   **Interactivity:** Rows are clickable (`clickable-row`) to open the Detection Details Modal.
    *   **Loading/Error States:** Displays a loading indicator (`#loadingIndicator`) during initial data fetch and an error message (`#tableErrorPlaceholder`) if the API call fails. A "No Detections" message (`#noDetectionsMessage`) is shown if filters result in an empty dataset.
*   **Image Gallery:** Displays thumbnails of the full annotated output images (`/output_images/...`). Thumbnails are loaded lazily.
*   **Modals (Bootstrap):**
    *   **Detection Details (`#detectionDetailModal`):** Triggered by clicking a table row. Displays larger versions of the cropped plate image (`#modalPlateImage`) and the full annotated frame (`#modalAnnotatedImage`), along with detailed text information (plate, confidence, vehicle type/color, timestamps).
    *   **Gallery View (`#galleryImageModal`):** Triggered by clicking a gallery thumbnail. Displays a large version (`#modalGalleryImage`) of the selected annotated frame.

#### Client-Side Logic (JavaScript)

Contained within `<script>` tags in `index.html`.

*   **Asynchronous Data Fetching:** Uses the `fetch` API to get all detection data from `/api/detections` on initial page load and periodically via polling.
*   **Dynamic Table Rendering (`renderTablePage`):** Clears and repopulates the table body based on the current page of `filteredDetections`. Formats data (dates, confidence, colors) and adds necessary CSS classes and data attributes for modals.
*   **Client-Side Filtering, Sorting, Pagination (`applyFilterAndSort`, `renderPaginationControls`):**
    *   `applyFilterAndSort`: Filters the `allDetections` array based on user selections in the filter controls and search input. Sorts the filtered results (`filteredDetections`) based on `currentSortKey` and `currentSortDirection`. Resets `currentPage` to 1 and triggers re-rendering.
    *   `renderPaginationControls`: Dynamically generates pagination links based on the total number of `filteredDetections` and `ITEMS_PER_PAGE`. Handles disabling links and showing ellipsis for large numbers of pages.
*   **Upload Handling:**
    *   Uses `XMLHttpRequest` (XHR) to allow monitoring upload progress (`progress` event) and cancellation (`abort()` method).
    *   Updates the progress bar and status text during upload.
    *   Handles success, error, and cancellation scenarios, displaying feedback to the user via `showUploadResult`.
    *   Resets the UI (`resetUploadUI`) after completion or cancellation.
    *   Triggers `fetchAndUpdateData(true)` on successful processing to refresh the table.
*   **Modal Interaction:**
    *   Attaches event listeners to table rows (`attachModalListener`) to populate and show the `#detectionDetailModal` with data stored in the row's `data-*` attributes.
    *   Uses Bootstrap's modal events (`show.bs.modal`) for the `#galleryImageModal` to load the correct image source when a thumbnail is clicked.
*   **Data Polling:** Uses `setInterval(fetchAndUpdateData, POLLING_INTERVAL_MS)` to periodically call the `/api/detections` endpoint, check for changes in the data (based on detection count), and update the UI if necessary. This provides near real-time updates without requiring page reloads.

#### Styling (Bootstrap, Custom CSS)

*   Uses Bootstrap 5 (via Bootswatch 'Brite' theme CDN) for base styling, layout, and components (navbar, cards, modals, buttons, progress bars, pagination).
*   Uses Bootstrap Icons for iconography.
*   Includes custom CSS (`static/css/style.css`) for overrides and specific styling (e.g., gallery image sizing, hover effects, sort indicators, hiding elements during load).

***

### 5. System Setup & Deployment

*   **Dependencies:** Requires Python 3.x. Key libraries include:
    *   `Flask`: Web framework.
    *   `psycopg2-binary`: PostgreSQL adapter.
    *   `ultralytics`: YOLO model interaction.
    *   `opencv-python`: Image processing.
    *   `tensorflow` / `keras`: Color model execution (ensure compatibility).
    *   `numpy`: Numerical operations.
    *   `python-dotenv`: Environment variable loading.
    *   `groq`: Groq API client.
    *   (Potentially others - a `requirements.txt` file is highly recommended).
*   **Configuration Management:**
    *   Sensitive information (DB credentials, API keys) should be stored in a `.env` file in the project root.
    *   `config/appconfig.py` reads these environment variables.
    *   Model paths and confidence thresholds are also defined here or in `anpr_image.py`.
*   **Database Setup:** Requires a running PostgreSQL server. Database connection parameters must be correctly set in the `.env` file. The `detected_plates` table will be created automatically on the first run if it doesn't exist.
*   **Running the Application:**
    *   **Development:** Run `python src/web_app.py`. This starts the Flask development server (typically on `http://0.0.0.0:5000`). `debug=True` enables auto-reloading and the debugger. `use_reloader=False` is currently set, which is important to prevent the ANPRProcessor from re-initializing constantly during development saves, but might need to be `True` if backend code changes require restarts.
    *   **Production:** The Flask development server is not suitable for production. A production-grade WSGI server (like Gunicorn or Waitress) should be used behind a reverse proxy (like Nginx or Apache). `debug` should be set to `False`.

***

### 6. Conclusion & Future Work

#### Strengths

*   **Modular Design:** Separation of concerns between the ANPR engine, backend API, and frontend UI.
*   **Comprehensive Feature Set:** Includes vehicle detection, plate detection, OCR, color classification, database storage, and an interactive web UI.
*   **Modern Technologies:** Utilizes popular libraries/frameworks like YOLOv8, TensorFlow, Groq, Flask, and Bootstrap.
*   **Interactive Frontend:** Client-side filtering, sorting, and pagination provide a responsive user experience.
*   **Visual Feedback:** Provides annotated images, cropped plates, and upload progress indicators.
*   **Robust Error Handling:** Includes try-except blocks and logging throughout the codebase.
*   **Efficient Database Access:** Uses connection pooling.

#### Limitations & Potential Improvements

*   **Single Image Processing:** The current system is designed for individual image uploads, not continuous video streams or batch processing.
*   **OCR Dependency:** Relies on the external Groq API, which requires an internet connection and API key, and may have associated costs or rate limits. Accuracy is dependent on the Groq model's performance.
*   **Heuristic Correction:** The 7-to-8 character correction is basic and might miscorrect plates.
*   **Vehicle-Plate Association:** Simple center-point-in-box association might fail in complex scenes with overlapping boxes.
*   **Performance:** Processing time per image depends heavily on model inference times and API latency. Processing large images or many simultaneous uploads could strain resources. The client-side loading of *all* detections might become slow with a very large database.
*   **Scalability:** The single global `ANPRProcessor` instance might become a bottleneck under heavy load in a multi-user scenario (though less critical for single-image uploads).
*   **Security:** Basic path traversal checks are present, but a thorough security review (input validation, dependency vulnerabilities, etc.) is recommended for production.
*   **Deployment Complexity:** Requires setup of Python environment, database, and potentially a WSGI server/reverse proxy for production.

*   **Future Work:**
    *   Implement video stream processing capabilities.
    *   Explore alternative OCR engines (e.g., Tesseract, or fine-tuning a dedicated OCR model) for offline use or different performance characteristics.
    *   Improve vehicle-plate association logic (e.g., using IoU or tracking).
    *   Optimize model inference times (e.g., using TensorRT, ONNX Runtime).
    *   Implement server-side pagination/filtering for the API endpoint (`/api/detections`) to handle very large datasets more efficiently.
    *   Add user authentication and authorization.
    *   Develop more sophisticated plate validation and correction algorithms.
    *   Containerize the application (e.g., using Docker) for easier deployment.
    *   Add comprehensive unit and integration tests.

***
