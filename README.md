# 🚘 Automatic Number Plate Recognition (ANPR) System

Production-grade ANPR for images, video, and live cameras with a secure Flask admin dashboard, watchlists, real-time alerts, analytics, and PostgreSQL storage.

Tech: Python • OpenCV • Ultralytics YOLO • TensorFlow/Keras • Groq Vision OCR / PaddleOCR • Flask + Socket.IO • PostgreSQL

---

## 🌟 Project Overview

This system detects Nigerian vehicle license plates, extracts the plate text, enriches detections with vehicle type, color, and (optionally) car brand, and persists results in PostgreSQL. It includes a full-featured Flask admin web app to upload images, manage watchlists, receive real-time alerts, view analytics, and run live camera capture with auto-detection.

Core modules:
- Main ANPR: `src/anpr_image.py` (image pipeline; Groq Vision OCR + car-brand)
- Flask App: `src/web_app.py` (admin UI, uploads, live camera, auto-detection, alerts, analytics)
- Video (optional): `src/anpr.py` (video pipeline; PaddleOCR)
- Config: `config/appconfig.py` (env vars, logging bootstrap)

---

## 🚀 Key Features

Detection & OCR
- YOLOv8-based plate detection (`models/license_plate_detector.pt`)
- YOLOv8-nano vehicle detection for context (`models/yolov8n.pt`)
- OCR (Images): Groq Vision (Meta Llama 4 Scout) with 8-char Nigerian plate regex
- OCR (Video): PaddleOCR with a 6+ alphanumeric acceptance to handle noise

Enrichment
- Vehicle color classifier (`models/EFN-model.best.h5`)
- Vehicle type (car/motorcycle/bus/truck) via YOLO
- Optional car brand/model inference using Groq Vision

Admin Web App (Flask + Socket.IO)
- Secure login (user/admin), CSRF protection
- Image uploads (admin) → annotated frames + cropped plates
- Live camera preview/capture and auto-detection loop with cooldown
- Real-time alerts to admins on watchlist matches
- Analytics dashboards (by type/color, day/hour, etc.)

Storage & Ops
- PostgreSQL persistence and connection pooling
- Auto table creation for: `detected_plates`, `watchlists`, `watchlist_entries`, `alerts`, `groq_api_usage`
- Structured logs to `logs/` (config/web)

---

## 🏗️ System Architecture

Flow
1) Source: image upload, video file, or live camera
2) Detect: YOLO plates + YOLO vehicles
3) OCR: Groq Vision (image) or PaddleOCR (video)
4) Enrich: color, type, optional brand
5) Persist: PostgreSQL
6) Notify: Socket.IO alerts for watchlist matches
7) Analyze: admin dashboards

Key files
- `src/anpr_image.py` → primary ANPR pipeline for images (Groq OCR, brand)
- `src/web_app.py` → Flask app, admin UI, Socket.IO, camera, auto-detect, analytics
- `src/anpr.py` → optional video pipeline (PaddleOCR-based)
- `config/appconfig.py` → env loading + early logging

---

## 📂 Repository Structure (essential)

```
ANPR System/
├─ config/
│  └─ appconfig.py
├─ models/
│  ├─ license_plate_detector.pt
│  ├─ yolov8n.pt
│  ├─ yolov8m-seg.pt
│  └─ EFN-model.best.h5
├─ src/
│  ├─ anpr_image.py      # MAIN ANPR (images, Groq OCR + brand)
│  ├─ web_app.py         # Flask admin app
│  ├─ anpr.py            # Optional video pipeline
│  ├─ templates/         # Jinja templates
│  └─ static/            # JS/CSS
├─ output/               # Annotated videos
├─ output_plates/
│  └─ plate_images/      # Cropped plate images saved by app
├─ logs/                 # config.log, web.log, ...
└─ requirements.txt
```

---

## ✅ Prerequisites

- Python 3.10+ (3.11 recommended)
- PostgreSQL 12+
- Windows (PowerShell examples below). Linux/macOS work similarly.
- Optional GPU for faster YOLO/TF inference (CPU works too).

---

## 🔐 Configuration

Create `.env` at project root:

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

Notes
- `config/appconfig.py` validates required vars; missing ones stop the app.
- On first web run, user `admin` is created/updated with `ADMIN_PASSWORD`.

---

## 🛠️ Setup (Windows PowerShell)

```
# 1) Clone
git clone https://github.com/Abdulraqib20/anpr-system-development.git
cd anpr-system-development

# 2) Virtual env
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3) Install deps
pip install --upgrade pip
pip install -r requirements.txt
# Optional: for video OCR pipeline
pip install paddleocr
```

Models
- Ensure the following weights exist under `models/` (present in this repo or place your own):
  - `license_plate_detector.pt` (plates)
  - `yolov8n.pt` (vehicles) and optionally `yolov8m-seg.pt`
  - `EFN-model.best.h5` (vehicle color)

Database
- Start PostgreSQL and ensure credentials in `.env` are correct.
- Tables are created automatically by the app if missing.

---

## ▶️ Run the Admin Web App

```
# From project root
python -m src.web_app
```

- URL: http://127.0.0.1:8080
- Login with username `admin` and your `ADMIN_PASSWORD`.

Highlights
- Upload images (admin) → stored annotated frame + cropped plate(s)
- Live camera: preview and single-frame capture
- Auto-detection: background vehicle detection + ANPR on frames (cooldown)
- Watchlists: create lists, add plates, automatic alerting on match
- Analytics: detections by day/hour, by vehicle type/color; Groq usage metrics

Saved files
- Annotated frames: `output_plates/`
- Cropped plates: `output_plates/plate_images/`

---

## 📸 CLI: Main ANPR (Images)

`src/anpr_image.py` (Groq Vision OCR + car brand)

```
# One or more image files
python -m src.anpr_image --source Resources/example1.jpg Resources/example2.jpg
```

Behavior
- Detects plates and vehicles; runs Groq OCR (8-char strict Nigerian regex)
- Predicts vehicle color; attempts brand/model via Groq
- Saves cropped plates to `output_plates/plate_images/`
- Saves annotated frame to `output_plates/`
- Persists results to PostgreSQL and checks watchlists to raise alerts

---

## 🎞️ CLI: Optional Video Pipeline

`src/anpr.py` (PaddleOCR-based OCR tuned for noisier video)

```
# Process a video file
python src/anpr.py --source "Resources/car_vid.mp4"

# Or open a webcam
python src/anpr.py --source 0
```

Behavior
- YOLOv8 plate detection (+ vehicle type); OCR with PaddleOCR
- Adaptive frame skipping and time limits for throughput
- Saves annotated video to `output/`
- Persists plates (when valid) to PostgreSQL

---

## 🔎 HTTP API (selected)

- GET `/api/detections` → All detections (JSON)
- Images
  - GET `/output_images/<filename>` → annotated frames (gallery)
  - GET `/plate_images/<filename>` → cropped plate images
- Camera (admin)
  - POST `/api/camera/start` | `/api/camera/stop`
  - POST `/api/camera/capture` | GET `/api/camera/preview`
  - GET `/api/camera/status` | GET `/api/camera/available` | POST `/api/camera/test-ip`
- Auto-detection (admin)
  - POST `/api/auto-detection/start` | `/api/auto-detection/stop`
  - GET `/api/auto-detection/status` | POST `/api/auto-detection/settings` | POST `/api/auto-detection/test`

All admin endpoints require login (admin role).

---

## 📊 Database Schema (summary)

- `detected_plates(id, start_time, end_time, license_plate, confidence, detection_count, vehicle_type, vehicle_color, car_brand, time_of_day, day_of_week, image_filename, annotated_frame_filename)`
- `watchlists(id, name, description, created_at, is_active)`
- `watchlist_entries(id, watchlist_id, license_plate, reason, added_at)`
- `alerts(id, detection_id, watchlist_id, watchlist_entry_id, alert_time, acknowledged_at, acknowledged_by_user_id)`
- `groq_api_usage(id, timestamp, api_call_type, model_name, prompt_tokens, completion_tokens, total_tokens, related_detection_id)`

Tables are created on startup if missing.

---

## ⚙️ Configuration & Behavior Notes

- Env + logging bootstrap: `config/appconfig.py` → logs go to `logs/` (e.g., `config.log`, `web.log`).
- Plate regex
  - Images: strict 8-char `^[A-Z0-9]{8}$` with correction heuristics for common OCR confusions.
  - Video: accepts `^[A-Z0-9]{6,}$` to tolerate video noise.
- Vehicle color classes: `beige, black, blue, brown, gold, green, grey, orange, pink, purple, red, silver, tan, white, yellow`.
- Connection pooling: PostgreSQL via `psycopg2.pool.SimpleConnectionPool`.
- Real-time: Socket.IO emits alerts to `admins_room`.

---

## 🧪 Troubleshooting

- Env vars missing → verify `.env` and console/logs.
- PostgreSQL errors → check host/port/credentials; ensure DB is running and reachable.
- PaddleOCR not installed (video) → `pip install paddleocr`.
- Model files missing → put weights in `models/` with names used in code.
- Eventlet/WebSocket issues → ensure `eventlet` installed; run with `use_reloader=False` (already set).
- Windows camera in-use → close other apps; try `--source 0` for default webcam.

---

## 🗺️ Roadmap

- Video OCR ensemble and plate tracking for higher recall/precision
- Docker Compose for app + DB + optional GPU support
- Expanded RBAC, audit logging, and multi-tenant support
- Advanced analytics dashboards and exports
- Training and fine-tuning guides for models

---

## 🤝 Contributing

- Fork → branch → PR; keep changes scoped and documented.
- Follow standard Python formatting/linting (black/isort/flake8) if configured.

---

## 📜 License

No explicit license is included. Add a `LICENSE` file to specify terms if needed.

---

## 🙏 Acknowledgements

Ultralytics YOLO • PaddleOCR • TensorFlow/Keras • Groq (Meta Llama 4 Scout Vision) • Flask + Flask-Socket.IO
