import torch
from torchvision.ops import box_convert
from torchmetrics.detection import MeanAveragePrecision
from ultralytics import YOLO
from PIL import Image
import os
import sys
import numpy as np
from pathlib import Path
import logging # Added for logging

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout) # Log to console
    ]
)
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(str(Path(__file__).parent.parent.resolve()))

# Define a confidence threshold for predictions
CONFIDENCE_THRESHOLD = 0.25

logger.info("Starting evaluation script...")

# Load your pre-trained model
logger.info(f"Loading model from: models/license_plate_detector.pt")
model = YOLO("models/license_plate_detector.pt")  # Use YOLO class directly
model.eval()
logger.info("Model loaded successfully.")

# Define paths
test_images_dir = "data/test/images"
test_labels_dir = "data/test/labels"
logger.info(f"Test images directory: {test_images_dir}")
logger.info(f"Test labels directory: {test_labels_dir}")

def parse_label_file(label_path, img_width, img_height):
    """Parse YOLO-format label files into normalized bounding boxes."""
    # logger.debug(f"Parsing label file: {label_path} for image size ({img_width}x{img_height})")
    with open(label_path, 'r') as f:
        lines = f.readlines()
    boxes = []
    for line_idx, line in enumerate(lines):
        try:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            # Convert normalized coordinates to absolute values
            x_center_abs = x_center * img_width
            y_center_abs = y_center * img_height
            width_abs = width * img_width
            height_abs = height * img_height
            # Convert to [x_min, y_min, x_max, y_max]
            x_min = x_center_abs - width_abs / 2
            y_min = y_center_abs - height_abs / 2
            x_max = x_center_abs + width_abs / 2
            y_max = y_center_abs + height_abs / 2
            boxes.append([x_min, y_min, x_max, y_max])
        except ValueError as e:
            logger.warning(f"Skipping malformed line {line_idx+1} in {label_path}: '{line.strip()}'. Error: {e}")
            continue
    # logger.debug(f"Parsed {len(boxes)} ground truth boxes from {label_path}")
    return boxes

def get_predictions(image_path, model, confidence_threshold=CONFIDENCE_THRESHOLD):
    """Run inference using Ultralytics YOLO model and return predictions."""
    # logger.debug(f"Getting predictions for: {image_path} with conf: {confidence_threshold}")
    results = model.predict(source=image_path, conf=confidence_threshold, verbose=False)

    if not results or not hasattr(results[0].boxes, 'data') or not results[0].boxes.data.numel() > 0:
        # logger.debug(f"No predictions found for {image_path}")
        return np.array([]), np.array([]), np.array([])

    pred_boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
    pred_scores = results[0].boxes.conf.cpu().numpy()
    pred_labels = results[0].boxes.cls.cpu().numpy().astype(int)
    # logger.debug(f"Found {len(pred_boxes_xyxy)} predictions for {image_path}")
    return pred_boxes_xyxy, pred_scores, pred_labels

logger.info("Initializing MeanAveragePrecision metric.")
metric = MeanAveragePrecision(
    iou_type="bbox",
    iou_thresholds=None,
    max_detection_thresholds=[1, 10, 100],
    class_metrics=True
)

image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
logger.info(f"Found {len(image_files)} images in {test_images_dir}")
total_images = len(image_files)
processed_image_count = 0

for image_name in image_files:
    processed_image_count += 1
    logger.info(f"Processing image ({processed_image_count}/{total_images}): {image_name}")
    img_path = os.path.join(test_images_dir, image_name)
    label_file_name = Path(image_name).stem + ".txt"
    label_path = os.path.join(test_labels_dir, label_file_name)

    if not os.path.exists(label_path):
        logger.warning(f"Label file not found for {image_name} at {label_path}, skipping.")
        continue

    # Get predictions
    pred_boxes, pred_scores, pred_labels = get_predictions(img_path, model, confidence_threshold=CONFIDENCE_THRESHOLD)
    logger.info(f"Predictions for {image_name}: {len(pred_boxes)} boxes")

    # Get ground truth
    try:
        img = Image.open(img_path)
        img_width, img_height = img.size
        gt_boxes = parse_label_file(label_path, img_width, img_height)
        gt_labels = [0] * len(gt_boxes) # Assuming license plate is class 0
        logger.info(f"Ground truth for {image_name}: {len(gt_boxes)} boxes")
    except FileNotFoundError: # Should be caught by os.path.exists, but good to have
        logger.error(f"Label file {label_path} confirmed missing, skipping {image_name}.")
        continue
    except Exception as e:
        logger.error(f"Error processing ground truth for {image_name}: {e}, skipping.")
        continue

    if not gt_boxes and not pred_boxes.size > 0:
        logger.info(f"Skipping {image_name} as it has no ground truth boxes and no predictions.")
        continue
    elif not gt_boxes:
        logger.info(f"Processing {image_name} with predictions but no ground truth boxes.")
    elif not pred_boxes.size > 0:
         logger.info(f"Processing {image_name} with ground truth boxes but no predictions.")


    # Update metric
    # logger.debug(f"Updating metric for {image_name}...")
    # logger.debug(f"Pred boxes shape: {pred_boxes.shape}, type: {pred_boxes.dtype if isinstance(pred_boxes, np.ndarray) else type(pred_boxes)}")
    # logger.debug(f"Pred scores shape: {pred_scores.shape}, type: {pred_scores.dtype if isinstance(pred_scores, np.ndarray) else type(pred_scores)}")
    # logger.debug(f"Pred labels shape: {pred_labels.shape}, type: {pred_labels.dtype if isinstance(pred_labels, np.ndarray) else type(pred_labels)}")
    # logger.debug(f"GT boxes len: {len(gt_boxes)}, first item: {gt_boxes[0] if gt_boxes else 'N/A'}")
    # logger.debug(f"GT labels len: {len(gt_labels)}, first item: {gt_labels[0] if gt_labels else 'N/A'}")

    try:
        metric.update(
            [
                {
                    "boxes": torch.tensor(pred_boxes, dtype=torch.float32),
                    "scores": torch.tensor(pred_scores, dtype=torch.float32),
                    "labels": torch.tensor(pred_labels, dtype=torch.int32)
                }
            ],
            [
                {
                    "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
                    "labels": torch.tensor(gt_labels, dtype=torch.int32)
                }
            ]
        )
        # logger.debug(f"Metric updated successfully for {image_name}")
    except Exception as e:
        logger.error(f"Error updating metric for {image_name}: {e}")
        logger.error(f"Pred boxes: {pred_boxes[:5]}") # Log first 5 for brevity
        logger.error(f"GT boxes: {gt_boxes[:5]}")
        continue


logger.info("Computing final metrics...")
# Compute final metrics
results = metric.compute()

# Ensure all relevant tensors from results are on CPU before further processing or printing
results_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in results.items()}


logger.info(f"Raw results from metric.compute(): {results_cpu}")


print(f"mAP@0.5:0.95 (COCO standard): {results_cpu['map'].item():.4f}")
print(f"mAP@0.5: {results_cpu['map_50'].item():.4f}")

LICENSE_PLATE_CLASS_ID = 0
# Ensure 'classes' tensor is on CPU before numel() or any other operation
classes_tensor_cpu = results_cpu['classes'] # Already on CPU from results_cpu dictionary
map_per_class_cpu = results_cpu['map_per_class'] # Already on CPU
mar_100_per_class_cpu = results_cpu['mar_100_per_class'] # Already on CPU

found_class_metrics = False

# Case 1: Single class detected, and it is our target class (tensors are 0-dim/scalar)
if classes_tensor_cpu.ndim == 0 and classes_tensor_cpu.item() == LICENSE_PLATE_CLASS_ID:
    if map_per_class_cpu.ndim == 0 and mar_100_per_class_cpu.ndim == 0:
        ap_class_0 = map_per_class_cpu.item()
        ar_class_0_at_100_dets = mar_100_per_class_cpu.item()
        print(f"Average Precision (AP) for License Plates (class {LICENSE_PLATE_CLASS_ID}, IoU 0.5:0.95): {ap_class_0:.4f}")
        print(f"Average Recall (AR@100) for License Plates (class {LICENSE_PLATE_CLASS_ID}, IoU 0.5:0.95): {ar_class_0_at_100_dets:.4f}")
        logger.info(f"AP for class {LICENSE_PLATE_CLASS_ID}: {ap_class_0:.4f}, AR@100 for class {LICENSE_PLATE_CLASS_ID}: {ar_class_0_at_100_dets:.4f}")
        found_class_metrics = True
    else:
        logger.warning("Mismatch in tensor dimensions for single class metrics. Expected scalar map_per_class and mar_100_per_class.")

# Case 2: Multiple classes detected, or single class in a 1D tensor.
elif classes_tensor_cpu.ndim > 0 and LICENSE_PLATE_CLASS_ID in classes_tensor_cpu:
    class_idx_in_results_tensor = (classes_tensor_cpu == LICENSE_PLATE_CLASS_ID).nonzero(as_tuple=True)[0]
    if class_idx_in_results_tensor.numel() > 0:
        actual_idx = class_idx_in_results_tensor.item() # Get the scalar index
        ap_class_0 = map_per_class_cpu[actual_idx].item()
        ar_class_0_at_100_dets = mar_100_per_class_cpu[actual_idx].item()
        print(f"Average Precision (AP) for License Plates (class {LICENSE_PLATE_CLASS_ID}, IoU 0.5:0.95): {ap_class_0:.4f}")
        print(f"Average Recall (AR@100) for License Plates (class {LICENSE_PLATE_CLASS_ID}, IoU 0.5:0.95): {ar_class_0_at_100_dets:.4f}")
        logger.info(f"AP for class {LICENSE_PLATE_CLASS_ID}: {ap_class_0:.4f}, AR@100 for class {LICENSE_PLATE_CLASS_ID}: {ar_class_0_at_100_dets:.4f}")
        found_class_metrics = True
    else:
        # This sub-condition (LICENSE_PLATE_CLASS_ID in classes_tensor_cpu but not found by nonzero) should be rare.
        logger.warning(f"Metrics for class {LICENSE_PLATE_CLASS_ID} (License Plates) not found via nonzero, though class ID was present in 'classes' tensor.")

if not found_class_metrics:
    msg = f"Class {LICENSE_PLATE_CLASS_ID} (License Plates) metrics could not be determined. Available classes in results: {classes_tensor_cpu.tolist() if classes_tensor_cpu.ndim > 0 else classes_tensor_cpu.item()}."
    print(msg)
    logger.warning(msg)
    if 'map_per_class' in results_cpu: # Already on CPU
        logger.warning(f"map_per_class: {map_per_class_cpu.tolist() if map_per_class_cpu.ndim > 0 else map_per_class_cpu.item()}")
    if 'mar_100_per_class' in results_cpu: # Already on CPU
        logger.warning(f"mar_100_per_class: {mar_100_per_class_cpu.tolist() if mar_100_per_class_cpu.ndim > 0 else mar_100_per_class_cpu.item()}")

logger.info("Evaluation script finished.")


#################################### RESULTS ####################################

# 2025-05-15 07:30:46,606 [INFO] - Computing final metrics...
# 2025-05-15 07:30:48,934 [INFO] - Raw results from metric.compute(): {'map': tensor(0.6780), 'map_50': tensor(0.9691), 'map_75': tensor(0.8068), 'map_small': tensor(0.2950), 'map_medium': tensor(0.6321), 'map_large': tensor(0.7335), 'mar_1': tensor(0.7097), 'mar_10': tensor(0.7213), 'mar_100': tensor(0.7213), 'mar_small': tensor(0.3500), 'mar_medium': tensor(0.6849), 'mar_large': tensor(0.7757), 'map_per_class': tensor(0.6780), 'mar_100_per_class': tensor(0.7213), 'classes': tensor(0, dtype=torch.int32)}
# mAP@0.5:0.95 (COCO standard): 0.6780
# mAP@0.5: 0.9691
# .95): 0.7213
# 2025-05-15 07:30:48,934 [INFO] - AP for class 0: 0.6780, AR@100 for class 0: 0.7213
# 2025-05-15 07:30:48,934 [INFO] - Evaluation script finished.
