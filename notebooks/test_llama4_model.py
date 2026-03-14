import sys
import os
from pathlib import Path
import base64
import time
import random
from dotenv import load_dotenv
from groq import Groq
from groq import InternalServerError, RateLimitError

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(str(Path(__file__).parent.parent.resolve()))

# GROQ_MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"
GROQ_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

# Initialize your client
client = Groq(api_key=GROQ_API_KEY)

def encode_image_to_base64(path):
    """Reads a local image file and returns a Base64 data URI."""
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def call_with_retries(model, messages, max_retries=5, base_delay=1.0):
    """
    Calls the Groq chat endpoint with retry logic for 503s (InternalServerError)
    and 429s (RateLimitError). Uses exponential back-off + jitter.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_completion_tokens=512
            )
        except RateLimitError as e:
            # 429: hit RPM/TPM. Honor retry-after header if present.
            retry_after = e.response.headers.get("retry-after")
            wait = float(retry_after) if retry_after else base_delay * (2 ** (attempt - 1))
            print(f"[429] Rate limit hit; waiting {wait:.1f}s before retry #{attempt}.")
            time.sleep(wait)
        except InternalServerError:
            # 503: transient server issue
            delay = base_delay * (2 ** (attempt - 1))
            jitter = random.uniform(0, 0.1 * delay)
            wait = delay + jitter
            print(f"[503] Service Unavailable; retrying in {wait:.1f}s (attempt #{attempt}).")
            time.sleep(wait)
    # All retries failed
    raise RuntimeError(f"API call failed after {max_retries} retries.")

def main():
    # 1) Base64-encode your local image file into a data URI
    img_b64_uri = encode_image_to_base64("Resources/Screenshot_20250510_224620_Instagram.jpg")

    # 2) Build your multimodal messages list, using image_url with a data URI
    messages = [
        {
            "role": "system",
            "content": "You are a car recognition assistant. Your only job is to name the car’s make and model."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Identify the make and model of the car in this image. "
                        "Reply with only the make and model (e.g. “Toyota Camry”). "
                        "If you can’t tell, reply “Unknown”."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_b64_uri
                    }
                }
            ]
        }
    ]

    # 3) Call with retry logic
    try:
        response = call_with_retries(
            model=GROQ_MODEL_NAME,
            messages=messages
        )
        print("Model response:", response.choices[0].message.content)
    except Exception as err:
        print("🛑 Failed to get a response:", err)

if __name__ == "__main__":
    main()
