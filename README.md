# 🚘 Automatic Number Plate Recognition (ANPR) System

Production-ready ANPR for images, video, and live cameras with web dashboard, alerts, analytics, vehicle type/color, and car-brand detection.

- Python • OpenCV • Ultralytics YOLO • TensorFlow/Keras • PaddleOCR / Groq Vision OCR • Flask + Socket.IO • PostgreSQL

---

## 🌟 Overview

This project detects Nigerian vehicle license plates from images, videos, or live camera streams, recognizes plate text, infers vehicle type and color, optionally detects car brand, and stores rich detections in PostgreSQL. It ships with a modern Flask web app (with authentication and admin tools) to upload images, run live camera capture, manage watchlists, receive alerts, and view analytics.

Key highlights:
- YOLO-based detection for plates and vehicles
- OCR via Groq Vision (Meta Llama 4 Scout) or PaddleOCR
- Vehicle color classification (EfficientNet family model)
- Flask web dashboard with login/admin + Socket.IO notifications
- Watchlists and automatic alerts on matches
- Analytics (detections by hour/day, color/type breakdown)
- PostgreSQL persistence and usage tracking of Groq API

---

## 🏗️ Architecture

High-level data flow:
1) Source: image upload, video file, or live camera
2) Detection: YOLOv8 for plates; YOLOv8-nano for vehicles
3) OCR: Groq Vision (primary) or PaddleOCR (video pipeline)
4) Enrichment: vehicle color (TensorFlow) + car brand (Groq Vision)
5) Storage: PostgreSQL tables (detected_plates, watchlists, alerts, groq_api_usage)
6) Dashboard: Flask + Socket.IO for admin UI, alerts, analytics

Main components:
- Detection models: `models/license_plate_detector.pt`, `models/yolov8n.pt`, `models/yolov8m-seg.pt`
- Color classifier: `models/EFN-model.best.h5`
- Image pipeline: `src/anpr_image.py` (Groq Vision OCR + brand)
- Video pipeline: `src/anpr.py` (PaddleOCR OCR)
- Web app (admin): `src/web_app.py` (uploads, live camera, alerts, analytics)
- Config: `config/appconfig.py` (env vars, logging)

---

## ✨ Features

- License plate detection and OCR (Nigeria-centric formats)
- Vehicle detection (type) and color classification
- Car brand/model inference via Groq Vision (best-effort)
- Image uploads and batch processing
- Live camera preview and on-demand capture
- Auto-detection loop on live stream with cooldown
- Watchlists and alerting (with Socket.IO push to admins)
- Detection analytics and Groq usage metrics
- Secure login, admin role, CSRF protection
- Annotated image/video output and cropped plate images

---

## 📂 Project Structure (selected)

```
ANPR System/
├── config/
│   └── appconfig.py                 # Env loading, logging
├── models/                          # YOLO + color model weights
├── src/
│   ├── anpr.py                      # Video pipeline (YOLO + PaddleOCR)
│   ├── anpr_image.py                # Image pipeline (YOLO + Groq Vision)
│   ├── web_app.py                   # Flask app + Socket.IO + admin
│   ├── templates/                   # Jinja HTML templates
│   └── static/                      # JS/CSS/assets
├── db/                              # SQL and local DBs (if any)
├── output/                          # Annotated videos
├── output_plates/
│   └── plate_images/                # Cropped plate images
├── logs/                            # config.log, web.log, ...
├── README.md
└── requirements.txt
```

---

## 🧰 Tech Stack

- Python 3.10+ (recommended 3.11)
- Flask, Flask-Login, Flask-WTF, Flask-SocketIO (eventlet)
- OpenCV, Ultralytics YOLOv8
- TensorFlow/Keras (vehicle color)
- PaddleOCR (video OCR) and/or Groq Vision OCR (image OCR)
- PostgreSQL (psycopg2-binary)

---

## 🔐 Configuration

Create a `.env` file in the project root with:

```
DB_HOST=localhost
DB_NAME=anpr
DB_USER=postgres
DB_PASSWORD=postgres
DB_PORT=5432

# OCR/AI
GROQ_API_KEY=your_groq_api_key_here
ROBOFLOW_API_KEY=your_roboflow_key_if_used

# Web
FLASK_SECRET_KEY=replace_this_in_production
ADMIN_PASSWORD=strong_admin_password
```

Notes:
- `config/appconfig.py` enforces presence of the DB and API keys listed there.
- On first run of the web app, an admin user with username `admin` is created/updated using `ADMIN_PASSWORD`.

---

## 🛠️ Setup

Windows PowerShell examples:

```
# 1) Clone
git clone https://github.com/Abdulraqib20/anpr-system-development.git
cd anpr-system-development

# 2) Python venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3) Install deps
pip install --upgrade pip
pip install -r requirements.txt
# If using PaddleOCR (video OCR)
pip install paddleocr
```

Database:
- Ensure PostgreSQL is running and accessible with the credentials in `.env`.
- Tables are auto-created by the app (detected_plates, watchlists, watchlist_entries, alerts, groq_api_usage).

Models:
- Place model weights in `models/` (already referenced in code):
  - `license_plate_detector.pt`
  - `yolov8n.pt` (vehicles), `yolov8m-seg.pt` (optional)
  - `EFN-model.best.h5` (vehicle color)

---

## 🚀 Run the Web App

```
# From project root
python -m src.web_app
```

- URL: http://127.0.0.1:8080
- Login with `admin` and the `ADMIN_PASSWORD` you set in `.env`.

Core pages and features:
- Dashboard with detections, gallery, and stats
- Upload images (admin only): processes with image pipeline (Groq OCR)
- Live camera preview and capture (admin)
- Auto-detection on live stream (admin)
- Watchlists and alerts (admin)
- Analytics and Groq usage metrics (admin)

Files saved:
- Annotated frames: `output_plates/`
- Cropped plates: `output_plates/plate_images/`

---

## 📸 CLI Usage (Image/Video)

Image(s) with Groq OCR and brand detection:
```
# One or more image paths
python -m src.anpr_image --source Resources/example1.jpg Resources/example2.jpg
```

Video with PaddleOCR (and vehicle type/color):
```
# Custom video file
python src/anpr.py --source "Resources/car_vid.mp4"

# Webcam
python src/anpr.py --source 0
```

Output videos are saved under `output/` with timestamps.

---

## 🔎 API Endpoints (selected)

- GET `/api/detections` → All detections (JSON)
- Image/plate files:
  - GET `/output_images/<filename>` → annotated frames (gallery)
  - GET `/plate_images/<filename>` → cropped plate images
- Camera (admin):
  - POST `/api/camera/start` | `/api/camera/stop` | `/api/camera/capture` | GET `/api/camera/preview`
  - GET `/api/camera/status` | GET `/api/camera/available` | POST `/api/camera/test-ip`
- Auto-detection (admin):
  - POST `/api/auto-detection/start` | `/api/auto-detection/stop`
  - GET `/api/auto-detection/status` | POST `/api/auto-detection/settings` | POST `/api/auto-detection/test`

All admin routes require login + admin role.

---

## 🗄️ Database Schema (summary)

- `detected_plates(id, start_time, end_time, license_plate, confidence, detection_count, vehicle_type, vehicle_color, car_brand, time_of_day, day_of_week, image_filename, annotated_frame_filename)`
- `watchlists(id, name, description, created_at, is_active)`
- `watchlist_entries(id, watchlist_id, license_plate, reason, added_at)`
- `alerts(id, detection_id, watchlist_id, watchlist_entry_id, alert_time, acknowledged_at, acknowledged_by_user_id)`
- `groq_api_usage(id, timestamp, api_call_type, model_name, prompt_tokens, completion_tokens, total_tokens, related_detection_id)`

Tables are created on startup if missing.

---

## 🔧 Configuration Details

- Env + logging initialized in `config/appconfig.py` → logs under `logs/` (e.g., `config.log`, `web.log`).
- Plate format:
  - Image pipeline (`anpr_image.py`): strict 8-char regex (`^[A-Z0-9]{8}$`) with correction heuristics.
  - Video pipeline (`anpr.py`): accepts 6+ alphanumeric to accommodate noisier OCR.
- Vehicle color classes: `['beige','black','blue','brown','gold','green','grey','orange','pink','purple','red','silver','tan','white','yellow']`.

---

## 📈 Analytics & Alerts

- Alerts are generated automatically when a detected plate matches an active watchlist entry.
- Real-time admin notifications are sent via Socket.IO (`new_alert`).
- Analytics pages show breakdowns by vehicle type/color, detections per day/hour, and day-of-week.
- Groq usage metrics page summarizes token usage and recent calls.

---

## 🧪 Tips & Troubleshooting

- Missing env vars → check `.env` and console/logs.
- DB connection errors → verify PostgreSQL host/port/credentials, and that the service is running.
- PaddleOCR not installed (video OCR) → `pip install paddleocr`.
- Model files missing → place weights in `models/` per paths hardcoded in the code.
- Eventlet/WebSocket issues → ensure eventlet is installed and avoid multiple reloader processes.
- Windows camera access → try `--source 0`, ensure no other app is using the camera.

---

## 🗺️ Roadmap (suggested)

- Improve plate OCR robustness for video (multi-preprocess ensemble)
- Deployable Docker Compose for app + DB
- Role-based features beyond admin/user
- Advanced analytics dashboards
- Model training scripts and documentation

---

## 🤝 Contributing

- Fork → Branch → PR. Keep changes small and documented.
- Follow Python lint/format standards (black/isort/flake8) if available.

---

## 📜 License

No explicit license file is present. All rights reserved by the author(s). If you need a license, add a `LICENSE` file and update this section.

---

## 🙏 Acknowledgements

- Ultralytics YOLO
- PaddleOCR
- TensorFlow/Keras
- Groq (Meta Llama 4 Scout vision OCR)
- Flask + Flask-SocketIO
