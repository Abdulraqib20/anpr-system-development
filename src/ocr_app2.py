import streamlit as st
from PIL import Image
import io
from groq import Groq
import sys
import os
from pathlib import Path
import logging
import base64
import re
import numpy as np
import concurrent.futures
import pytesseract

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from config.appconfig import GROQ_API_KEY

# Set the path to Tesseract executable - update this with your actual Tesseract installation path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

logs_dir = Path(__file__).parent.parent / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
log_file_path = logs_dir / "config.log"

logger = logging.getLogger("ANPR")
logger.setLevel(logging.DEBUG)
if logger.hasHandlers():
    logger.handlers.clear()

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info(f"Logging to console and file: {log_file_path}")

# --- Page Configuration ---
st.set_page_config(
    page_title="License Plate OCR",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Persistent State Management using @st.cache_resource ---
@st.cache_resource
def get_persistent_state():
    """Initializes and returns a persistent state dictionary for the session."""
    logger.info("Initializing persistent state via @st.cache_resource.")
    return {
        "history": [] # List to store {'image_name': str, 'image_bytes': bytes, 'image_media_type': str, 'groq_result': str, 'tesseract_result': str, 'car_brand': str}
    }

# Get the persistent state object for this session
persistent_state = get_persistent_state()

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please set it in config or environment variables.")
    logger.error("GROQ_API_KEY not set.")
    st.stop()

GROQ_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct" # "llama-3.2-90b-vision-preview" # llama-3.2-11b-vision-preview

try:
    client = Groq(api_key=GROQ_API_KEY)
    logger.info("Groq client initialized successfully.")
except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}")
    logger.error(f"Failed to initialize Groq client: {e}")
    st.stop()


# --- Add a helper function for cleaning ---
def clean_plate_text(raw_text: str) -> str:
    """Cleans the raw OCR text to be uppercase alphanumeric only."""
    if not isinstance(raw_text, str):
        return ""
    cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    return cleaned

# --- Process with Groq ---
def process_with_groq(image_bytes, image_media_type):
    """Process the image with Groq Vision API to extract license plate."""
    try:
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                         {
                            "type": "text",
                            "text": """STRICTLY identify the license plate in the image. Extract ONLY the alphanumeric characters (A-Z, 0-9) of the license plate number. Ensure the output is in ALL UPPERCASE. REMOVE ALL hyphens, spaces, symbols, or any other non-alphanumeric characters. Respond with ONLY the final cleaned license plate string (e.g., ABC123XY). Do NOT include any introductory text, labels, explanations, or markdown formatting. JUST the cleaned plate string."""
                         },
                         {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_media_type};base64,{image_base64}"
                            }
                         }
                    ],
                }
            ],
            model=GROQ_MODEL_NAME,
            max_tokens=1024,
        )

        raw_ocr_result = chat_completion.choices[0].message.content
        cleaned_result = clean_plate_text(raw_ocr_result)
        logger.info(f"Groq result: '{cleaned_result}'")
        return cleaned_result
    except Exception as e:
        logger.error(f"Error processing with Groq: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

# --- Process with Groq for Car Brand Detection ---
def detect_car_brand(image_bytes, image_media_type):
    """Process the image with Groq Vision API to detect car brand."""
    try:
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                         {
                            "type": "text",
                            "text": """Analyze this image and determine the exact brand/make of the car (e.g., Toyota, Honda, Ford, BMW, etc.).

                            Be as specific as possible by identifying both the make and model if visible (e.g., 'Toyota Camry', 'Honda Civic', 'BMW 3 Series').

                            Provide ONLY the car brand/make and model in your response with no additional text or explanations."""
                         },
                         {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_media_type};base64,{image_base64}"
                            }
                         }
                    ],
                }
            ],
            model=GROQ_MODEL_NAME,
            max_tokens=1024,
        )

        brand_result = chat_completion.choices[0].message.content.strip()
        logger.info(f"Car brand detection result: '{brand_result}'")
        return brand_result
    except Exception as e:
        logger.error(f"Error detecting car brand with Groq: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

# --- Process with Tesseract OCR ---
def process_with_tesseract(image_bytes):
    """Process the image with Tesseract OCR."""
    try:
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_bytes))

        # Add some preprocessing to improve OCR quality
        # Convert to grayscale
        image = image.convert('L')

        # Increase contrast (optional)
        # from PIL import ImageEnhance
        # enhancer = ImageEnhance.Contrast(image)
        # image = enhancer.enhance(2.0)

        # Run OCR - configure for license plates
        # Using '--psm 7' (treat image as single line of text)
        # '--oem 3' (use LSTM OCR Engine)
        text = pytesseract.image_to_string(
            image,
            config='--psm 7 --oem 3'
        )

        # Clean the result
        cleaned_result = clean_plate_text(text)
        logger.info(f"Tesseract result: '{cleaned_result}'")
        return cleaned_result
    except Exception as e:
        logger.error(f"Error processing with Tesseract: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

# --- Test if tesseract is available ---
def is_tesseract_available():
    try:
        # Try a simple OCR operation
        pytesseract.get_tesseract_version()
        return True
    except Exception as e:
        logger.error(f"Tesseract not available: {e}")
        return False

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-align: center;
        padding: 0.68rem;
        color: #2c3e50; /* Darker text for header */
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        margin-bottom: 1rem;
        color: #34495e; /* Slightly lighter than main header */
    }

    .intro-card { /* New style for the introductory message */
        background-color: #e8f4fd; /* Light blue background */
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border-left: 5px solid #3498db; /* Blue accent line */
    }

    .card { /* General card style, can be used if expanders are not enough */
        padding: 1.1rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        margin-bottom: 1rem;
        background-color: #ffffff; /* White background for cards */
    }

    .result-card {
        padding: 0.8rem 1rem; /* Adjusted padding */
        border-radius: 8px;
        margin-bottom: 0.75rem; /* Spacing between result types */
        border-width: 1px;
        border-style: solid;
    }

    .groq-card { /* Defined style for Groq results */
        background-color: #f3e8fd; /* Light purple background */
        border-left: 4px solid #8e44ad; /* Purple accent */
    }

    .tesseract-card {
        background-color: #fff8f0;
        border-left: 4px solid #FFA62B;
    }

    .brand-card {
        background-color: #f0fff8;
        border-left: 4px solid #2B8A3E;
    }

    .stButton>button {
        background-color: #6C5CE7; /* Primary button color */
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.6rem 1.2rem; /* Slightly more padding */
        transition: all 0.3s ease;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #574B90; /* Darker shade on hover */
        transform: translateY(-2px); /* Slight lift on hover */
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }

    .footer {
        margin-top: 3rem; /* More space before footer */
        padding-top: 1.5rem; /* More padding in footer */
        text-align: center;
        font-size: 0.95rem;
        font-weight: 800;
        font-family: 'Source Sans Pro', sans-serif;
        color: #7f8c8d; /* Muted color for footer text */
    }

    .method-title {
        font-weight: 600;
        margin-bottom: 0.4rem; /* More space under title */
        font-size: 1.05rem; /* Slightly larger method title */
        color: #34495e;
    }

    /* Styling for st.expander to make them look like cards */
    .stExpander {
        border: none !important; /* Remove default border */
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important; /* Card-like shadow */
        border-radius: 10px !important; /* Rounded corners */
        margin-bottom: 1.5rem !important; /* Space between expanders */
        background-color: #ffffff; /* White background */
    }
    .stExpanderHeader {
        font-size: 1.15rem !important; /* Larger header text */
        font-weight: 600 !important;
        padding: 0.8rem 1rem !important; /* Adjust padding */
        border-radius: 10px 10px 0 0 !important; /* Rounded top corners */
        background-color: #f8f9fa; /* Light background for header */
        border-bottom: 1px solid #e9ecef; /* Separator line */
    }
    .stExpanderHeader:hover {
        background-color: #f1f3f5; /* Slightly darker on hover */
    }

    /* Ensure content within expander has some padding */
    .stExpander [data-testid="stVerticalBlock"] {
        padding: 1rem 1rem 0.5rem 1rem; /* Add padding to content area */
    }

    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<div class='main-header'>🚗 License Plate OCR & Car Brand Detection</div>", unsafe_allow_html=True)
st.markdown("""
<div class='intro-card'>
    <p style='text-align: center; font-size: 1.1rem;'>
        Upload one or more images to extract license plates and identify car brands.
        Results are powered by <strong>Meta's Llama 4 Scout</strong> multi-modal model via Groq and <strong>Tesseract OCR</strong>.
    </p>
</div>
""", unsafe_allow_html=True)

# Check if Tesseract is available
tesseract_available = is_tesseract_available()
if not tesseract_available:
    st.warning("⚠️ Tesseract OCR may not be properly installed or configured. Only Groq Vision will be used. To use Tesseract OCR, please install it and set the correct path.")

# --- Sidebar for Upload and Controls ---
with st.sidebar:
    st.header("Controls")
    uploaded_files = st.file_uploader( # Changed variable name
        "📂 Upload Image(s)", # Changed label
        type=['png', 'jpg', 'jpeg'],
        help="Select one or more image files for text extraction.",
        key="file_uploader",
        accept_multiple_files=True  # Allow multiple files
    )

    process_button_placeholder = st.empty() # Keep placeholder outside

    # --- Process Button Logic ---
    if uploaded_files: # Check if list is not empty
        if process_button_placeholder.button("Extract Text from All Uploaded Images 🔍", type="primary", key="process", use_container_width=True):
            processed_something = False
            with st.spinner("🤖 Processing images... This may take a moment."):
                for uploaded_file in uploaded_files: # Iterate through each file
                    try:
                        logger.info(f"Processing file: {uploaded_file.name}")
                        # Get image bytes
                        image_bytes = uploaded_file.getvalue()
                        # IMPORTANT: No need to seek(0) for multiple files if getvalue is called once per file obj in the loop.
                        # If the same uploaded_file object instance was to be read multiple times, then seek(0) would be needed.

                        # Determine media type robustly
                        file_ext = Path(uploaded_file.name).suffix.lower()
                        if file_ext == '.jpg' or file_ext == '.jpeg':
                            image_media_type = 'image/jpeg'
                        elif file_ext == '.png':
                            image_media_type = 'image/png'
                        else:
                            # Fallback, might not be universally supported by browsers for data URI
                            image_media_type = f"image/{file_ext.lstrip('.')}"
                            logger.warning(f"Using potentially less common image type: {image_media_type} for {uploaded_file.name}")

                        # Process with Groq for license plate OCR
                        groq_result = process_with_groq(image_bytes, image_media_type)

                        # Process with Groq for car brand detection - COMMENTED OUT
                        # car_brand_result = detect_car_brand(image_bytes, image_media_type)
                        car_brand_result = "Brand detection disabled" # Assign a placeholder

                        # Process with Tesseract if available
                        if tesseract_available:
                            tesseract_result = process_with_tesseract(image_bytes)
                        else:
                            tesseract_result = "Tesseract OCR not available"

                        # Add to history
                        history_entry = {
                            "image_name": uploaded_file.name, # Store name
                            "image_bytes": image_bytes,       # Store bytes
                            "image_media_type": image_media_type, # Store media type
                            "groq_result": groq_result,
                            "tesseract_result": tesseract_result,
                            "car_brand": car_brand_result
                        }

                        persistent_state["history"].insert(0, history_entry)
                        logger.info(f"Added entry for {uploaded_file.name} to history. History size: {len(persistent_state['history'])}")
                        processed_something = True

                    except Exception as e:
                        error_message = f"Error processing image {uploaded_file.name}: {str(e)}"
                        st.error(error_message)
                        logger.error(error_message, exc_info=True)

            if processed_something:
                # Clear the uploader state by rerunning if any file was processed.
                # This also helps to refresh the UI with new history items.
                st.rerun()
            else:
                st.warning("No files were processed. Please ensure files are valid.")


    # --- Clear Button Logic ---
    # Check history in persistent_state
    if persistent_state["history"]:
        # Define placeholder and render button only if history exists
        clear_button_placeholder = st.empty()
        if clear_button_placeholder.button("Clear History 🗑️", key="clear", use_container_width=True):
            persistent_state["history"].clear() # Clear the history list in persistent_state
            logger.info("History cleared by user.")
            st.rerun()

# --- Main Content Area for History Display ---
st.markdown("")
st.markdown("")
st.subheader("Processing History & Results") # Updated subheader
st.markdown("")

# Check history in persistent_state
if not persistent_state["history"]:
    st.info("Upload image(s) using the sidebar to get started. Your results will appear here.") # Updated message
else:
    # Display history items from persistent_state
    for i, entry in enumerate(persistent_state["history"]):
        expander_label = f"📜 Results for: {entry.get('image_name', f'Image {i+1}')}"
        with st.expander(expander_label, expanded=(i == 0)): # First item expanded by default
            try:
                img_bytes = entry['image_bytes']
                img_name = entry['image_name']
                groq_result = entry['groq_result']
                tesseract_result = entry.get('tesseract_result', "N/A")
                car_brand = entry.get('car_brand', "N/A")

                # Display image and results in columns
                col1, col2 = st.columns([1, 2]) # Adjust column ratio if needed

                with col1:
                    st.image(img_bytes, caption=f"Input: {img_name}", use_column_width='always')

                with col2:
                    # Groq result card
                    st.markdown('<div class="result-card groq-card">', unsafe_allow_html=True)
                    st.markdown('<p class="method-title">🤖 Groq Vision (License Plate):</p>', unsafe_allow_html=True)
                    st.code(groq_result if groq_result else "No plate detected.", language=None)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Tesseract result card
                    st.markdown('<div class="result-card tesseract-card">', unsafe_allow_html=True)
                    st.markdown('<p class="method-title">🔍 Tesseract OCR (License Plate):</p>', unsafe_allow_html=True)
                    st.code(tesseract_result if tesseract_result else "No plate detected.", language=None)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Car brand result card - COMMENTED OUT
                    # st.markdown('<div class="result-card brand-card">', unsafe_allow_html=True)
                    # st.markdown('<p class="method-title">🚙 Car Brand (Groq Vision):</p>', unsafe_allow_html=True)
                    # st.code(car_brand if car_brand else "Brand not identified.", language=None)
                    # st.markdown('</div>', unsafe_allow_html=True)

            except KeyError as ke:
                st.error(f"Missing data in history entry {i} (Image: {entry.get('image_name', 'N/A')}): {ke}. This might be an old history format.")
                logger.error(f"KeyError displaying history entry {i} (Image: {entry.get('image_name', 'N/A')}): {ke}", exc_info=True)
            except Exception as display_err:
                st.error(f"Error displaying history entry {i} (Image: {entry.get('image_name', 'N/A')}): {display_err}")
                logger.error(f"Error displaying history entry {i} (Image: {entry.get('image_name', 'N/A')}): {display_err}", exc_info=True)

# --- Footer ---
st.markdown('<div class="footer">Made with ❤️ by <a href="https://github.com/Abdulraqib20" target="_blank">raqibcodes</a></div>', unsafe_allow_html=True)

