import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Define log directory path
LOGS_DIR = Path(__file__).parent.parent / 'logs'
# Create logs directory if it doesn't exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging before other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'config.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def _as_bool(raw_value, default=False):
    """Parse a boolean-like environment variable value."""
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}

try:
    # Path configuration
    root_dir = Path(__file__).parent.parent.resolve()
    env_path = root_dir / '.env'

    # Check for .env existence (optional for deployment, useful locally)
    # if not env_path.exists():
    #     logger.warning(f"No .env file found at {env_path}. Relying solely on system environment variables.")
        # sys.exit(1) # DO NOT EXIT IN DEPLOYMENT IF FILE IS MISSING

    # Attempt to load .env if it exists. If not, it will do nothing.
    load_dotenv(env_path)

    # Base required environment variables
    BASE_REQUIRED_VARS = [
        'DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD', 'DB_PORT', 'ROBOFLOW_API_KEY'
    ]

    # Load base required values
    config = {var: os.getenv(var) for var in BASE_REQUIRED_VARS}

    missing_vars = [var for var in BASE_REQUIRED_VARS if config[var] is None]
    if missing_vars:
        logger.critical(f"Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

    # OCR provider configuration (Ollama-first, Groq optional fallback)
    OCR_PROVIDER = os.getenv('OCR_PROVIDER', 'ollama').strip().lower()
    if OCR_PROVIDER not in {'ollama', 'groq'}:
        logger.warning(f"Invalid OCR_PROVIDER '{OCR_PROVIDER}', defaulting to 'ollama'.")
        OCR_PROVIDER = 'ollama'

    OCR_FALLBACK_TO_GROQ = _as_bool(os.getenv('OCR_FALLBACK_TO_GROQ'), default=False)
    OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434').strip()
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen3.5:9b').strip()
    GROQ_MODEL_NAME = os.getenv('GROQ_MODEL_NAME', 'meta-llama/llama-4-scout-17b-16e-instruct').strip()
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')

    conditionally_required_vars = []
    if OCR_PROVIDER == 'groq' or OCR_FALLBACK_TO_GROQ:
        conditionally_required_vars.append('GROQ_API_KEY')

    missing_conditional = [var for var in conditionally_required_vars if not os.getenv(var)]
    if missing_conditional:
        logger.critical(
            "Missing required OCR environment variables for current provider settings: %s",
            ', '.join(missing_conditional)
        )
        sys.exit(1)

    # Export variables
    DB_HOST = config['DB_HOST']
    DB_NAME = config['DB_NAME']
    DB_USER = config['DB_USER']
    DB_PASSWORD = config['DB_PASSWORD']
    DB_PORT = config['DB_PORT']
    ROBOFLOW_API_KEY = config['ROBOFLOW_API_KEY']

    REQUIRED_VARS = BASE_REQUIRED_VARS + conditionally_required_vars

    sensitive_vars = {'DB_PASSWORD', 'ROBOFLOW_API_KEY', 'GROQ_API_KEY'}
    vars_to_log = {
        'DB_HOST': DB_HOST,
        'DB_NAME': DB_NAME,
        'DB_USER': DB_USER,
        'DB_PASSWORD': DB_PASSWORD,
        'DB_PORT': DB_PORT,
        'ROBOFLOW_API_KEY': ROBOFLOW_API_KEY,
        'OCR_PROVIDER': OCR_PROVIDER,
        'OCR_FALLBACK_TO_GROQ': OCR_FALLBACK_TO_GROQ,
        'OLLAMA_HOST': OLLAMA_HOST,
        'OLLAMA_MODEL': OLLAMA_MODEL,
        'GROQ_MODEL_NAME': GROQ_MODEL_NAME,
        'GROQ_API_KEY': GROQ_API_KEY,
    }

    for var, value in vars_to_log.items():
        text_value = '' if value is None else str(value)
        if var in sensitive_vars and len(text_value) > 4:
            logged_value = f"{text_value[:2]}****{text_value[-2:]}"
        elif var in sensitive_vars:
            logged_value = '****'
        else:
            logged_value = text_value
        logger.info(f"{var}: {logged_value}")

    logger.info("Configuration loaded successfully")

except Exception as e:
    logger.critical(f"Configuration initialization failed: {str(e)}")
    sys.exit(1)

__all__ = [
    'DB_HOST',
    'DB_NAME',
    'DB_USER',
    'DB_PASSWORD',
    'DB_PORT',
    'ROBOFLOW_API_KEY',
    'OCR_PROVIDER',
    'OCR_FALLBACK_TO_GROQ',
    'OLLAMA_HOST',
    'OLLAMA_MODEL',
    'GROQ_MODEL_NAME',
    'GROQ_API_KEY',
    'REQUIRED_VARS',
]
