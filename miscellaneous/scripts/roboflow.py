from inference_sdk import InferenceHTTPClient

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(str(Path(__file__).parent.parent.resolve()))
sys.path.append(os.path.abspath("src"))

from config.appconfig import ROBOFLOW_API_KEY

CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

result = CLIENT.infer("Resources/carImage1.png", model_id="vehicle-classification-v2/1")
if 'predictions' in result and len(result['predictions']) > 0:
    print(result['predictions'][0]['class'])
else:
    print("No predictions available.")
