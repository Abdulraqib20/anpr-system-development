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

    # Required environment variables
    REQUIRED_VARS = [
        'DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD', 'DB_PORT',
        'VIDEO_SOURCE', 'MODEL_PATH', 'OUTPUT_PATH',
        'PLATE_REGEX', 'MIN_CONFIDENCE', 'PLATE_MERGE_DISTANCE', 'MIN_TRACKING_DURATION'
    ]

    # Load environment variables
    config = {var: os.getenv(var) for var in REQUIRED_VARS}

    # Check for missing required values (excluding optional variables)
    missing_vars = [var for var in REQUIRED_VARS if config[var] is None]
    if missing_vars:
        logger.critical(f"Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

    # Convert numerical values
    try:
        config['MIN_CONFIDENCE'] = float(config['MIN_CONFIDENCE'])
        if not 0 <= config['MIN_CONFIDENCE'] <= 1:
            raise ValueError("MIN_CONFIDENCE must be between 0 and 1")

        config['PLATE_MERGE_DISTANCE'] = int(config['PLATE_MERGE_DISTANCE'])
        config['MIN_TRACKING_DURATION'] = int(config['MIN_TRACKING_DURATION'])

        # Compile regex
        config['PLATE_REGEX'] = re.compile(config['PLATE_REGEX'])

    except (ValueError, re.error, TypeError) as e:
        logger.critical(f"Configuration validation failed: {str(e)}")
        sys.exit(1)

    # Ensure paths exist
    for path_key in ['VIDEO_SOURCE', 'MODEL_PATH']:
        path = root_dir / config[path_key]
        if not path.exists():
            logger.warning(f"Path {path_key} not found: {path}")

    # Create output directory if missing
    output_path = root_dir / config['OUTPUT_PATH']
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
    PLATE_MERGE_DISTANCE = config['PLATE_MERGE_DISTANCE']
    MIN_TRACKING_DURATION = config['MIN_TRACKING_DURATION']

    # Secure logging (partial obfuscation of sensitive values)
    sensitive_vars = ['DB_PASSWORD']
    for var in REQUIRED_VARS:
        value = locals().get(var, '')
        logged_value = f"{value[:2]}****{value[-2:]}" if var in sensitive_vars and len(value) > 4 else str(value)
        logger.info(f"{var}: {logged_value}")

    logger.info("Configuration loaded successfully")

except Exception as e:
    logger.critical(f"Configuration initialization failed: {str(e)}")
    sys.exit(1)

__all__ = REQUIRED_VARS