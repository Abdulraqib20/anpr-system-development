import os
import re
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging before other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent.parent / 'logs/config.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    # Path configuration
    root_dir = Path(__file__).parent.parent.resolve()
    env_path = root_dir / '.env'
    
    if not env_path.exists():
        logger.critical(f"Missing .env file at {env_path}")
        sys.exit(1)

    load_dotenv(env_path)

    # Combined required variables
    REQUIRED_VARS = [
        # Database
        'DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD', 'DB_PORT',
        # ANPR
        'VIDEO_SOURCE', 'MODEL_PATH', 'OUTPUT_PATH',
        'PLATE_REGEX', 'MIN_CONFIDENCE', 'TRACKING_FRAMES', 'MIN_DETECTIONS'
    ]

    # Load and validate configuration
    config = {}
    
    # Load environment variables with fallbacks
    config.update({
        'DB_HOST': os.getenv('DB_HOST'),
        'DB_NAME': os.getenv('DB_NAME'),
        'DB_USER': os.getenv('DB_USER'),
        'DB_PASSWORD': os.getenv('DB_PASSWORD'),
        'DB_PORT': os.getenv('DB_PORT'),
        'VIDEO_SOURCE': os.getenv('VIDEO_SOURCE', 'Resources/car_vid.mp4'),
        'MODEL_PATH': os.getenv('MODEL_PATH', 'models/license_plate_detector.pt'),
        'OUTPUT_PATH': os.getenv('OUTPUT_PATH', 'output/annotated_video.mp4'),
        'PLATE_REGEX': os.getenv('PLATE_REGEX', r'^[A-Z0-9]{7,10}$'),
        'MIN_CONFIDENCE': os.getenv('MIN_CONFIDENCE', '0.65'),
        'TRACKING_FRAMES': os.getenv('TRACKING_FRAMES', '30'),
        'MIN_DETECTIONS': os.getenv('MIN_DETECTIONS', '1')
    })

    # Check missing required values (without defaults)
    missing = [var for var in REQUIRED_VARS[:5] if not config.get(var)]
    if missing:
        logger.critical(f"Missing required vars: {', '.join(missing)}")
        sys.exit(1)

    # Type conversions and validation
    try:
        # Numerical values
        config['MIN_CONFIDENCE'] = float(config['MIN_CONFIDENCE'])
        if not 0 <= config['MIN_CONFIDENCE'] <= 1:
            raise ValueError("MIN_CONFIDENCE must be between 0-1")
            
        config['TRACKING_FRAMES'] = int(config['TRACKING_FRAMES'])
        config['MIN_DETECTIONS'] = int(config['MIN_DETECTIONS'])

        # Regex validation
        config['PLATE_REGEX'] = re.compile(config['PLATE_REGEX'])
        
        # Path validation
        for path_var in ['VIDEO_SOURCE', 'MODEL_PATH']:
            path = root_dir / config[path_var]
            if not path.exists():
                logger.warning(f"Path {path_var} not found: {path}")

        # Output path creation
        output_path = root_dir / config['OUTPUT_PATH']
        output_path.parent.mkdir(parents=True, exist_ok=True)

    except (ValueError, re.error, TypeError) as e:
        logger.critical(f"Config validation failed: {str(e)}")
        sys.exit(1)

    # Export variables
    DB_HOST = config['DB_HOST']
    DB_NAME = config['DB_NAME']
    DB_USER = config['DB_USER']
    DB_PASSWORD = config['DB_PASSWORD']
    DB_PORT = config['DB_PORT']
    VIDEO_SOURCE = str(root_dir / config['VIDEO_SOURCE'])
    MODEL_PATH = str(root_dir / config['MODEL_PATH'])
    OUTPUT_PATH = str(output_path)
    PLATE_REGEX = config['PLATE_REGEX']
    MIN_CONFIDENCE = config['MIN_CONFIDENCE']
    TRACKING_FRAMES = config['TRACKING_FRAMES']
    MIN_DETECTIONS = config['MIN_DETECTIONS']

    # Secure logging
    logger.info("Configuration loaded successfully")
    sensitive_vars = ['DB_PASSWORD']
    for var in REQUIRED_VARS:
        value = locals().get(var, '')
        if var in sensitive_vars:
            logged_value = f"{value[:2]}****{value[-2:]}" if len(value) > 4 else "****"
        else:
            logged_value = str(value)
        logger.info(f"{var}: {logged_value}")

except Exception as e:
    logger.critical(f"Configuration initialization failed: {str(e)}")
    sys.exit(1)

__all__ = REQUIRED_VARS