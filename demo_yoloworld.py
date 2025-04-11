import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor

IMG_PATH = "/home/lenovo/project/CQN_Improve/data/open_door2/open_door/variation0/episodes/episode0/front_rgb/0.png"
TEXT_PROMPT = "door"
OUTPUT_DIR = Path("./outputs/open_world_detection")

def set_classes(model, target_class):
    """
    This function limits the model to detect only the specified target class.
    """
    model.set_classes([target_class])

def detect_objects_with_clip(image_path, target_class, use_clip=False):
    """
    Detect objects in the image using YOLOv8. If use_clip is True, it will perform additional filtering based on CLIP model.
    """
    model = YOLO("./yolov8s-world.pt")  # Using YOLO-World model
    if use_clip:
        set_classes(model, target_class)  # Set the model to detect only the specified class
    results = model.predict(image_path)
    
    # Get detection results: boxes and labels
    boxes = results[0].boxes  # The bounding boxes
    labels = results[0].names  # The class names for each box
    #print("Detected labels:", labels)

    # Get the detected classes based on boxes and confidences
    detected_classes = []
    for idx, conf in enumerate(boxes.conf):  # Use boxes.conf to get the confidence score
        if conf > 0.25:  # Filter out boxes with low confidence (e.g., < 0.25)
            detected_classes.append(labels[boxes[idx].cls[0].item()])  # Get class name from index
    #print("Detected classes:", detected_classes)

    return boxes, labels, detected_classes, results

img = cv2.imread(IMG_PATH)
boxes, labels, detected_classes, results = detect_objects_with_clip(img, TEXT_PROMPT, False)
print(boxes)