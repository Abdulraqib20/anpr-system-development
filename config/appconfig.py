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
        'DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD', 'DB_PORT', 'ROBOFLOW_API_KEY'
    ]

    # Load environment variables
    config = {var: os.getenv(var) for var in REQUIRED_VARS}

    missing_vars = [var for var in REQUIRED_VARS if config[var] is None]
    if missing_vars:
        logger.critical(f"Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

    # Export variables
    DB_HOST = config['DB_HOST']
    DB_NAME = config['DB_NAME']
    DB_USER = config['DB_USER']
    DB_PASSWORD = config['DB_PASSWORD']
    DB_PORT = config['DB_PORT']
    ROBOFLOW_API_KEY = config['ROBOFLOW_API_KEY']

    sensitive_vars = ['DB_PASSWORD', 'ROBOFLOW_API_KEY']
    for var in REQUIRED_VARS:
        value = locals().get(var, '')
        logged_value = f"{value[:2]}****{value[-2:]}" if var in sensitive_vars and len(value) > 4 else str(value)
        logger.info(f"{var}: {logged_value}")

    logger.info("Configuration loaded successfully")

except Exception as e:
    logger.critical(f"Configuration initialization failed: {str(e)}")
    sys.exit(1)

__all__ = REQUIRED_VARS