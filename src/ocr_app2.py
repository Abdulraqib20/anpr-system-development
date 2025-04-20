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
        "history": [] # List to store {'image_file': UploadedFile, 'groq_result': str, 'tesseract_result': str}
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
    """Process the image with Groq Vision API."""
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
        padding: 0.68rem
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    
    .card {
        padding: 1.1rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        margin-bottom: 1rem;
    }
    
    .result-card {
        padding: 0.1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    
    # .groq-card {
    #     background-color: #f0f8ff;
    #     border-left: 4px solid #6C5CE7;
    # }
    
    # .tesseract-card {
    #     background-color: #fff8f0;
    #     border-left: 4px solid #FFA62B;
    # }
    
    .stButton>button {
        background-color: #6C5CE7;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #574B90;
    }
    
    .footer {
        margin-top: 2rem;
        padding-top: 1rem;
        text-align: center;
        font-size: 0.95rem;
        font-weight: 800;
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    .method-title {
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<div class='main-header'>🚗 License Plate OCR</div>", unsafe_allow_html=True)
st.markdown("""
<div class='card'>
    <p style='text-align: center;'>Extract license plates from images using Meta's Llama 4 Scout multi-modal model & Tesseract OCR!</p>
</div>
""", unsafe_allow_html=True)

# Check if Tesseract is available
tesseract_available = is_tesseract_available()
if not tesseract_available:
    st.warning("⚠️ Tesseract OCR may not be properly installed or configured. Only Groq Vision will be used. To use Tesseract OCR, please install it and set the correct path.")

# --- Sidebar for Upload and Controls ---
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader(
        "📂 Upload an Image",
        type=['png', 'jpg', 'jpeg'],
        help="Select an image file for text extraction.",
        key="file_uploader" 
    )

    # --- Process Button Logic ---
    if uploaded_file:
        # Display preview FIRST
        st.image(uploaded_file, caption="Uploaded Image Preview", use_column_width=True, output_format='JPEG')

        # Define placeholder and render button AFTER preview
        process_button_placeholder = st.empty()
        if process_button_placeholder.button("Extract Text 🔍", type="primary", key="process", use_container_width=True):
            with st.spinner("🤖 Processing with OCR engines..."):
                try:
                    # Get image bytes
                    image_bytes = uploaded_file.getvalue()
                    uploaded_file.seek(0)  # Reset pointer after getvalue()
                    
                    # Determine media type robustly
                    file_ext = Path(uploaded_file.name).suffix.lower()
                    if file_ext == '.jpg' or file_ext == '.jpeg':
                        image_media_type = 'image/jpeg'
                    elif file_ext == '.png':
                        image_media_type = 'image/png'
                    else:
                        image_media_type = f"image/{file_ext.lstrip('.')}"
                        logger.warning(f"Using potentially unsupported image type: {image_media_type}")
                    
                    # Process with Groq
                    groq_result = process_with_groq(image_bytes, image_media_type)
                    
                    # Process with Tesseract if available
                    if tesseract_available:
                        tesseract_result = process_with_tesseract(image_bytes)
                    else:
                        tesseract_result = "Tesseract OCR not available"
                    
                    # Add to history
                    history_entry = {
                        "image_file": uploaded_file,
                        "groq_result": groq_result,
                        "tesseract_result": tesseract_result
                    }
                    
                    persistent_state["history"].insert(0, history_entry)
                    logger.info(f"Added new entry to history. History size: {len(persistent_state['history'])}")
                    
                    st.rerun()
                    
                except Exception as e:
                    error_message = f"Error processing image: {str(e)}"
                    st.error(error_message)
                    logger.error(error_message, exc_info=True)
                    
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
st.subheader("📜 Processing History")
st.markdown("")

# Check history in persistent_state
if not persistent_state["history"]:
    st.info("Upload an image using the sidebar to get started. Your results will appear here.")
else:
    # Display history items from persistent_state
    for i, entry in enumerate(persistent_state["history"]):
        # Wrap each history entry in a card div
        st.markdown('<div class="card">', unsafe_allow_html=True)
        with st.container(): 
            col1, col2 = st.columns([1, 2]) 
            
            try:
                img_file = entry['image_file']
                groq_result = entry['groq_result']
                tesseract_result = entry.get('tesseract_result', "N/A")  # Backwards compatibility
                
                with col1:
                    st.image(img_file, caption=f"Input: {img_file.name}", use_column_width=True)
                
                with col2:
                    st.markdown("### Results")
                    
                    # Groq result card
                    st.markdown('<div class="result-card groq-card">', unsafe_allow_html=True)
                    st.markdown('<p class="method-title">🤖 Groq Vision:</p>', unsafe_allow_html=True)
                    st.code(groq_result, language=None)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Tesseract result card
                    st.markdown('<div class="result-card tesseract-card">', unsafe_allow_html=True)
                    st.markdown('<p class="method-title">🔍 Tesseract OCR:</p>', unsafe_allow_html=True)
                    st.code(tesseract_result, language=None)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
            except Exception as display_err:
                # Display error message within the card
                st.error(f"Error displaying history entry {i}: {display_err}")
                logger.error(f"Error displaying history entry {i}: {display_err}", exc_info=True)
            
        # Close the card div
        st.markdown('</div>', unsafe_allow_html=True) 
        st.markdown("<br>", unsafe_allow_html=True) # Add some vertical space between cards

# --- Footer ---
st.markdown('<div class="footer">Made with ❤️ by <a href="https://github.com/Abdulraqib20" target="_blank">raqibcodes</a></div>', unsafe_allow_html=True)

