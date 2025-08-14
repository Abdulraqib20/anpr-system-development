# ANPR (Automatic Number Plate Recognition) System

An intelligent license plate recognition system that combines computer vision, machine learning, and web technologies to detect, extract, and analyze vehicle license plates from images.

## 🎯 Overview

This ANPR system provides a comprehensive solution for automatic license plate recognition with features including:

- **Image Processing**: Advanced license plate detection and recognition from static images
- **Advanced OCR**: Groq Vision API for high-accuracy text extraction
- **Vehicle Intelligence**: Car brand detection and color classification
- **Web Dashboard**: Complete admin interface with authentication
- **Database Integration**: PostgreSQL database for storing detections and analytics
- **Alert System**: Real-time notifications for watchlist matches
- **Live Camera Support**: Real-time image capture and processing

## 🏗️ Architecture

The system is built with a modular architecture:

```
📦 ANPR System
├── 🌐 Web Application (Flask + Socket.IO)
├── 🔍 ANPR Processing Engine
├── 🤖 Machine Learning Models
├── 📊 Database Layer (PostgreSQL)
├── 📷 Image Processing
└── 📱 Real-time Notifications
```

### Core Components

1. **Web Application** (`src/web_app.py`)
   - Flask-based admin dashboard with authentication
   - Live camera monitoring and control
   - Watchlist management and alerts
   - Analytics dashboard and reporting
   - Socket.IO for real-time updates

2. **ANPR Engine** (`src/anpr_image.py`)
   - Primary image processing pipeline with Groq Vision OCR
   - Vehicle detection and license plate extraction
   - Car brand and color classification
   - Database integration and logging

## 📁 Project Scripts and Components

### Main Application Scripts

- **`src/web_app.py`** - Primary Flask web application with full admin interface, authentication, live camera monitoring, Socket.IO integration, and dashboard analytics
- **`src/web_app2.py`** - Alternative Flask application variant with similar features and live camera processing capabilities
- **`src/web_app3.py`** - Third Flask application variant with enhanced auto-detection features and real-time monitoring

### Core ANPR Processing Scripts

- **`src/anpr_image.py`** - Main ANPR processing engine for single images using Groq Vision OCR, vehicle detection, and database integration

### Specialized Applications

- **`src/ocr_app.py`** - Streamlit-based OCR application for license plate text extraction with Groq Vision and Tesseract support

### Model Training and Evaluation

- **`src/evaluate.py`** - Model evaluation script for testing YOLOv8 license plate detection performance with mAP metrics and confidence thresholds

### Testing and Validation

- **`src/test_llama4_model.py`** - Testing script for Llama4 Vision model integration and validation

## 🔧 Setup Instructions

### Prerequisites

- Python 3.8+
- PostgreSQL database
- GPU support (optional, for better performance)
- Tesseract OCR engine (optional)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Abdulraqib20/anpr-system-development.git
   cd anpr-system-development
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Database Setup**
   ```bash
   # Create PostgreSQL database
   createdb anpr_database

   # Run database scripts (tables will be created automatically on first run)
   psql -d anpr_database -f db/licensePlates.sql
   psql -d anpr_database -f db/detected_plates_edit.sql
   ```

5. **Environment Configuration**

   Create a `.env` file in the root directory:
   ```env
   # Database Configuration
   DB_HOST=localhost
   DB_NAME=anpr_database
   DB_USER=your_username
   DB_PASSWORD=your_password
   DB_PORT=5432

   # API Keys
   GROQ_API_KEY=your_groq_api_key_here

   # Application Settings
   FLASK_SECRET_KEY=your_secret_key_here
   ADMIN_PASSWORD=your_admin_password
   ```

6. **Verify Models**

   Ensure the following models exist in the `models/` directory:
   - `license_plate_detector.pt` - Custom YOLOv8 model for license plate detection
   - `best.pt` - Best performing model variant
   - `EFN-model.best.h5` - EfficientNet model for car brand classification
   - `yolov8n.pt`, `yolov8s.pt`, `yolov8m-seg.pt` - YOLOv8 model variants
   - `yolov10n.pt` - YOLOv10 nano model

### Running the Application

#### Method 1: Main Flask Web Application (Recommended)
```bash
# From project root directory
python -m src.web_app
```
- Access the web dashboard at: `http://localhost:8080`
- Login with username `admin` and your `ADMIN_PASSWORD` from `.env`

#### Method 2: Alternative Web Applications
```bash
# Enhanced version with additional features
python -m src.web_app2

# Version 3 with enhanced auto-detection
python -m src.web_app3
```

#### Method 3: Streamlit OCR Application
```bash
# OCR-focused interface
streamlit run src/ocr_app.py
```
- Access at: `http://localhost:8501`

#### Method 4: Process Images via Command Line
```bash
# Process single image
python -m src.anpr_image --source path/to/image.jpg

# Process multiple images
python -m src.anpr_image --source path/to/image1.jpg path/to/image2.jpg path/to/folder/
```

### Complete Setup and Run Commands

**For a fresh installation and run:**
```bash
# 1. Clone and navigate
git clone https://github.com/Abdulraqib20/anpr-system-development.git
cd anpr-system-development

# 2. Setup environment
python -m venv venv
venv\Scripts\activate  # Windows
pip install --upgrade pip
pip install -r requirements.txt

# 3. Configure environment (create .env file with your settings)
# See Environment Configuration section above

# 4. Setup database (ensure PostgreSQL is running)
createdb anpr_database

# 5. Run the main application
python -m src.web_app
```

**Then access:**
- Web Dashboard: `http://localhost:8080`
- Login: username=`admin`, password=your `ADMIN_PASSWORD`
   git clone <repository-url>
   cd anpr-system
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Database Setup**
   ```bash
   # Create PostgreSQL database
   createdb anpr_database

   # Run database scripts
   psql -d anpr_database -f db/licensePlates.sql
   psql -d anpr_database -f db/detected_plates_edit.sql
   ```

4. **Environment Configuration**

   Create a `.env` file in the root directory:
   ```env
   # Database Configuration
   DATABASE_URL=postgresql://username:password@localhost:5432/anpr_database
   DB_HOST=localhost
   DB_NAME=anpr_database
   DB_USER=your_username
   DB_PASSWORD=your_password
   DB_PORT=5432

   # API Keys
   GROQ_API_KEY=your_groq_api_key_here

   # Application Settings
   SECRET_KEY=your_secret_key_here
   FLASK_ENV=development
   DEBUG=True

   # File Paths
   UPLOAD_FOLDER=uploads
   OUTPUT_FOLDER=output
   OUTPUT_PLATES_FOLDER=output_plates

   # Model Settings
   LICENSE_PLATE_MODEL=models/license_plate_detector.pt
   YOLO_MODEL=models/yolov8n.pt
   CAR_BRAND_MODEL=models/EFN-model.best.h5

   # OCR Settings
   TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
   PADDLEOCR_LANG=en

   # Camera Settings
   CAMERA_INDEX=0
   CAMERA_RESOLUTION_WIDTH=640
   CAMERA_RESOLUTION_HEIGHT=480
   ```

5. **Download Models**

   Place the following models in the `models/` directory:
   - `license_plate_detector.pt` - Custom YOLOv8 model for license plate detection
   - `best.pt` - Best performing model variant
   - `EFN-model.best.h5` - EfficientNet model for car brand classification
   - `yolov8n.pt`, `yolov8s.pt`, `yolov8m-seg.pt` - YOLOv8 model variants
   - `yolov10n.pt` - YOLOv10 nano model

### Running the Application

1. **Start the Main Web Application**
   ```bash
   python src/web_app.py
   ```

2. **Alternative Web Applications**
   ```bash
   # Enhanced version with auto-detection
   python src/web_app2.py

   # Version 3 with additional features
   python src/web_app3.py
   ```

3. **Streamlit Applications**
   ```bash
   # ANPR Streamlit interface
   streamlit run src/anpr_streamlit.py

   # OCR-focused interface
   streamlit run src/ocr_app.py
   ```

4. **Access the Dashboard**
   - Main Flask app: `http://localhost:5000`
   - Streamlit apps: `http://localhost:8501`

5. **Process Images/Videos**
   ```bash
   # Process single image with main engine
   python src/anpr_image.py --image path/to/image.jpg

   # Process video with PaddleOCR
   python src/anpr.py --video path/to/video.mp4

   # Enhanced processing
   python src/anpr_new.py --input path/to/media
   ```

## 📊 Database Schema

The system uses PostgreSQL with the following main tables:

- **detected_plates**: Stores license plate detections with timestamps, confidence scores, and vehicle information
- **watchlists**: Manages watchlist entries for specific license plates with alert triggers
- **alerts**: Records alert notifications when watchlist matches occur
- **groq_api_usage**: Tracks Groq Vision API usage statistics and costs
- **vehicle_detections**: Stores vehicle detection metadata including brand and color information

## 🔌 API Endpoints

### Detection Endpoints
- `POST /detect` - Process uploaded image for license plate detection
- `GET /detections` - Retrieve detection history with pagination and filtering
- `GET /detections/<id>` - Get specific detection details and metadata
- `POST /bulk_detect` - Process multiple images in batch mode

### Management Endpoints
- `GET /dashboard` - Main dashboard view with analytics and statistics
- `POST /watchlist` - Add new watchlist entry with alert configuration
- `DELETE /watchlist/<id>` - Remove watchlist entry
- `PUT /watchlist/<id>` - Update existing watchlist entry

### Real-time Endpoints
- `GET /live` - Live camera feed with real-time processing
- `POST /start_detection` - Start auto-detection mode
- `POST /stop_detection` - Stop auto-detection mode
- `GET /camera_stats` - Retrieve live camera processing statistics

### Analytics Endpoints
- `GET /analytics` - Comprehensive analytics dashboard data
- `GET /reports` - Generate detection reports with date ranges
- `GET /export` - Export detection data in various formats (JSON, CSV)

## 🤖 Machine Learning Models

The system leverages multiple AI models for comprehensive analysis:

1. **YOLOv8/YOLOv10** - Vehicle and license plate detection with high accuracy
2. **Groq Vision API** - Advanced OCR for text extraction with superior accuracy
3. **EfficientNet** - Car brand classification with 95%+ accuracy
4. **TensorFlow/Keras** - Vehicle color classification and analysis

## 📷 Image Processing

Comprehensive image analysis capabilities:
- **File Formats**: JPG, JPEG, PNG, BMP support
- **Live Camera**: USB cameras, IP cameras support
- **Real-time Processing**: Single image analysis with minimal latency
- **Output Generation**: Annotated images with detection overlays
- **Batch Processing**: Multiple image processing queues

## 📱 Real-time Features

Advanced real-time capabilities:
- **Socket.IO Integration**: Live updates and bidirectional communication
- **Auto-detection Mode**: Continuous monitoring with configurable intervals
- **Live Dashboard**: Real-time statistics, alerts, and camera feeds
- **Camera Controls**: Start/stop/pause live processing with web interface
- **Push Notifications**: Instant alerts for watchlist matches

## 🔐 Security Features

Robust security implementation:
- **User Authentication**: Secure login system with session management
- **Authorization**: Role-based access control for different user levels
- **File Validation**: Secure upload validation and virus scanning
- **Database Security**: Connection pooling and SQL injection prevention
- **API Security**: Rate limiting and authentication tokens
- **Data Encryption**: Sensitive data encryption at rest and in transit

## 📈 Analytics & Reporting

Comprehensive analytics and reporting system:
- **Detection Analytics**: Trends, patterns, and performance metrics
- **Watchlist Reports**: Match frequency and alert effectiveness
- **API Usage Analytics**: Groq API consumption and cost tracking
- **Performance Metrics**: Processing speed, accuracy, and system health
- **Export Capabilities**: PDF reports, CSV exports, JSON backups
- **Custom Dashboards**: Configurable analytics views

## 🛠️ Configuration

Highly configurable system with multiple configuration layers:

### Environment Variables (`.env` file)
- Database connection parameters
- API keys and authentication tokens
- File paths and storage locations
- Model configurations and settings

### Application Config (`config/appconfig.py`)
- Flask application settings
- OCR engine configurations
- Camera and video settings
- Logging and debugging options

### Model Parameters
- Detection confidence thresholds
- OCR accuracy settings
- Classification model parameters
- Performance optimization settings

## 🧪 Testing and Evaluation

Comprehensive testing suite:

### Model Evaluation
```bash
# Evaluate license plate detection model
python -m src.evaluate

# Test Llama4 Vision model
python -m src.test_llama4_model
```

### Performance Testing
- Model accuracy benchmarking
- Processing speed optimization
- Memory usage analysis
- Database performance testing

## 📝 Logging and Monitoring

Advanced logging and monitoring system:
- **Application Logs**: `logs/web.log` with rotating file handlers
- **Configuration Logs**: `logs/config.log` for setup and configuration
- **Error Tracking**: Detailed error logs with stack traces
- **Performance Monitoring**: Processing time and resource usage logs
- **Audit Trails**: User actions and system changes logging

## 🚀 Deployment

### Development Environment
```bash
python -m src.web_app
```

### Production Deployment

1. **Configure Production Environment**
   ```bash
   export FLASK_ENV=production
   export DEBUG=False
   export WORKERS=4
   ```

2. **Use Production WSGI Server**
   ```bash
   # Gunicorn deployment
   gunicorn -w 4 -b 0.0.0.0:8080 --timeout 300 src.web_app:app

   # uWSGI deployment
   uwsgi --http :8080 --module src.web_app:app --processes 4
   ```

3. **Setup Reverse Proxy (Nginx)**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://127.0.0.1:8080;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

4. **Docker Deployment**
   ```bash
   docker build -t anpr-system .
   docker run -p 8080:8080 anpr-system
   ```

## 📦 Project Structure

```
ANPR System/
├── config/              # Configuration files and app settings
├── db/                  # Database schemas, scripts, and SQLite files
├── diagrams/            # System architecture and workflow diagrams
├── image/               # Static images and logos
├── json/                # JSON data files and output logs
├── logs/                # Application and system logs
├── models/              # AI/ML model files (YOLO, TensorFlow, etc.)
├── output/              # Processed images and detection results
├── output_plates/       # Extracted license plate images
├── Resources/           # Additional resources and documentation
├── src/                 # Source code directory
│   ├── web_app.py       # Main Flask application
│   ├── anpr_image.py    # Primary ANPR processing engine
│   ├── ocr_app.py       # Streamlit OCR application
│   ├── evaluate.py      # Model evaluation script
│   ├── web_app2.py      # Alternative Flask application
│   ├── web_app3.py      # Enhanced Flask application
│   ├── test_llama4_model.py # Llama4 Vision testing
│   ├── templates/       # HTML templates for Flask
│   └── static/          # CSS/JS static files
├── uploads/             # User uploaded files
├── requirements.txt     # Python dependencies
├── Procfile            # Heroku deployment configuration
└── README.md           # This documentation file
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a Pull Request

### Development Guidelines
- Follow PEP 8 coding standards
- Add comprehensive tests for new features
- Update documentation for any changes
- Ensure all tests pass before submitting PR

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔧 Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Verify PostgreSQL service is running
   - Check database credentials in `.env` file
   - Ensure database exists and is accessible

2. **Model Loading Error**
   - Verify all model files exist in `models/` directory
   - Check file permissions and sizes
   - Ensure sufficient memory for model loading

3. **Camera Access Error**
   - Check camera permissions and availability
   - Verify camera device index in configuration
   - Test camera with external applications

4. **OCR Processing Error**
   - Verify Groq API key is valid and has credits
   - Check Tesseract installation and path configuration (if using Tesseract)

5. **Performance Issues**
   - Monitor system resources (CPU, RAM, GPU)
   - Adjust model confidence thresholds
   - Optimize image resolution settings

### Support and Documentation

For technical support, issues, or questions:
- Create an issue in the GitHub repository
- Check existing documentation and README files
- Review log files for detailed error information
- Join the community discussions for help and tips

### System Requirements

**Minimum Requirements:**
- Python 3.8+
- 8GB RAM
- 10GB storage space
- PostgreSQL 12+

**Recommended Requirements:**
- Python 3.9+
- 16GB RAM
- NVIDIA GPU with CUDA support
- 50GB SSD storage
- PostgreSQL 14+
