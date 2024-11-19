import os
import json
from PIL import Image
import torch
import base64
import io
import numpy as np

from utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

# Load models
yolo_model = get_yolo_model(model_path='weights/icon_detect/best.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")

# Configurations
platform = 'mobile'
draw_bbox_config = {
    'text_scale': 0.8,
    'text_thickness': 2,
    'text_padding': 3,
    'thickness': 3,
}

DEVICE = torch.device('cuda')

# Define function to process and save results
def process_and_save(image_path, output_image_path, output_json_path, box_threshold=0.05, iou_threshold=0.1):
    # Load and save image for OCR
    image_input = Image.open(image_path)
    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_path, display_img=False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold': 0.9})
    text, ocr_bbox = ocr_bbox_rslt

    # Perform detection and obtain labels
    dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_path, 
        yolo_model, 
        BOX_TRESHOLD=box_threshold, 
        output_coord_in_ratio=True, 
        ocr_bbox=ocr_bbox, 
        draw_bbox_config=draw_bbox_config, 
        caption_model_processor=caption_model_processor, 
        ocr_text=text,
        iou_threshold=iou_threshold
    )
    
    # Save labeled image
    labeled_image = Image.open(io.BytesIO(base64.b64decode(dino_labeled_img)))
    labeled_image.save(output_image_path)

    # Prepare JSON data by combining label_coordinates and parsed_content_list
    json_data = []
    for label_id, bbox in label_coordinates.items():
        # Extract label type and content
        parsed_content = parsed_content_list[int(label_id)]
        label_type = "icon" if "Icon" in parsed_content else "text"
        content = parsed_content.split(":", 1)[1].strip() if ":" in parsed_content else ""

        json_data.append({
            "label_number": label_id,
            "label_type": label_type,
            "bbox": [float(coord) for coord in bbox],
            "content": content
        })

    # Save to JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=4)
    print(f'Processed and saved for {image_path}')

# Directory paths
# base_dir = 'path_to_base_directory'  # Set your base directory path here
base_dir = '/home/tangli/test_dataset'  # Set your base directory path here

# platform_dirs = ['Android', 'iOS', 'Windows']
platform_dirs = ['Android']

for platform in platform_dirs:
    platform_path = os.path.join(base_dir, platform)
    for task_dir in os.listdir(platform_path):
        task_path = os.path.join(platform_path, task_dir)
        image_raw_path = os.path.join(task_path, 'image', 'raw')
        ui_detection_result_path = os.path.join(task_path, 'image', 'ui_detection_result')
        text_detection_result_path = os.path.join(task_path, 'text', 'ui_detection_result')
        
        # Create result directories if not exist
        os.makedirs(ui_detection_result_path, exist_ok=True)
        os.makedirs(text_detection_result_path, exist_ok=True)

        # Process each raw image
        for img_file in os.listdir(image_raw_path):
            if img_file.endswith(('.jpg')):
                image_path = os.path.join(image_raw_path, img_file)
                output_image_path = os.path.join(ui_detection_result_path, img_file)
                output_json_path = os.path.join(text_detection_result_path, img_file.replace('.jpg', '.json'))
            if img_file.endswith(('.png')):
                image_path = os.path.join(image_raw_path, img_file)
                output_image_path = os.path.join(ui_detection_result_path, img_file)
                output_json_path = os.path.join(text_detection_result_path, img_file.replace('.png', '.json'))
                # Process and save results
                process_and_save(image_path, output_image_path, output_json_path)