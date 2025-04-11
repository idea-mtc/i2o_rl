# dds cloudapi for DINO-X
from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk.tasks.dinox import DinoxTask
from dds_cloudapi_sdk.tasks.detection import DetectionTask
from dds_cloudapi_sdk.tasks.types import DetectionTarget
from dds_cloudapi_sdk import TextPrompt

# using supervision for visualization
import os
import cv2
import numpy as np
import supervision as sv
from pathlib import Path

"""
Hyper Parameters
"""
API_TOKEN = "f1a20063f5388be297e707614b6ee9c1"
IMG_PATH = "/home/lenovo/project/CQN_Improve/data/open_door2/open_door/variation0/episodes/episode18/front_rgb/0.png"
TEXT_PROMPT = "handle"
OUTPUT_DIR = Path("./outputs/open_world_detection")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

"""
Prompting DINO-X with Text for Box and Mask Generation with Cloud API
"""

# Step 1: initialize the config
token = API_TOKEN
config = Config(token)

# Step 2: initialize the client
client = Client(config)

# Step 3: Run DINO-X task
# if you are processing local image file, upload them to DDS server to get the image url
image_url = client.upload_file(IMG_PATH)

from dds_cloudapi_sdk.tasks.v2_task import V2Task

task = V2Task(api_path="/v2/task/dinox/detection", api_body={
    "model": "DINO-X-1.0",
    "image": image_url,
    "prompt": {
        "type":"text",
        "text":"handle"
    },
    "targets": ["bbox"],
    "bbox_threshold": 0.25,
    "iou_threshold": 0.8
})

# task = DinoxTask(
#     image_url=image_url,
#     prompts=[TextPrompt(text=TEXT_PROMPT)],
#     bbox_threshold=0.25,
#     targets=[DetectionTarget.BBox, DetectionTarget.Mask]
# )
client.run_task(task)
predictions = task.result["objects"]
print(task.result)
"""
Visualization
"""

# decode the prediction results

classes = [x.strip().lower() for x in TEXT_PROMPT.split('.') if x]
class_name_to_id = {name: id for id, name in enumerate(classes)}
class_id_to_name = {id: name for name, id in class_name_to_id.items()}

boxes = []
# masks = []
confidences = []
class_names = []
class_ids = []

for idx, obj in enumerate(predictions):
    boxes.append(obj["bbox"])
    # masks.append(DetectionTask.rle2mask(DetectionTask.string2rle(obj.mask.counts), obj.mask.size))  # convert mask to np.array using DDS API
    confidences.append(obj["score"])
    cls_name = obj["category"].lower().strip()
    class_names.append(cls_name)
    class_ids.append(class_name_to_id[cls_name])

boxes = np.array(boxes)
img = cv2.imread(IMG_PATH)
print(boxes)
for box in boxes:
    crop_image = img[int(box[0]):int(box[2]), int(box[1]):int(box[3])]
    new_size = (15, 15)
    resized_img = cv2.resize(crop_image, new_size, interpolation=cv2.INTER_LINEAR)
    print(resized_img.shape)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "crop_image.jpg"), resized_img)
# masks = np.array(masks)
class_ids = np.array(class_ids)
labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(class_names, confidences)
]

img = cv2.imread(IMG_PATH)
detections = sv.Detections(
    xyxy = boxes,
    # mask = masks.astype(bool),
    class_id = class_ids,
)

box_annotator = sv.BoxAnnotator()
annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

label_annotator = sv.LabelAnnotator()
annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
cv2.imwrite(os.path.join(OUTPUT_DIR, "annotated_demo_image.jpg"), annotated_frame)




# mask_annotator = sv.MaskAnnotator()
# annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
# cv2.imwrite(os.path.join(OUTPUT_DIR, "annotated_demo_image_with_mask.jpg"), annotated_frame)

print(f"Annotated image has already been saved to {OUTPUT_DIR}")