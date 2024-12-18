import numpy as np
import cv2
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from transformers import SamProcessor, SamModel
import torch

# Suppress TensorFlow logs


# Load the model and processor once
model = SamModel.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def segment(image_path, input_points, batch_size=64):

    masks = []
    scores = []

    raw_image = Image.open(image_path)

    for i in range(0, len(input_points), batch_size):
        batch_points = input_points[i:i+batch_size]
        batch_points_nested = [[[[coord[0], coord[1]]] for coord in batch_points]]
        batch_inputs = processor(
            raw_image, 
            input_points=batch_points_nested, 
            return_tensors="pt"
            )
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

        with torch.no_grad():
            batch_outputs = model(**batch_inputs)

        batch_masks = processor.image_processor.post_process_masks(
            batch_outputs.pred_masks, 
            batch_inputs["original_sizes"], 
            batch_inputs["reshaped_input_sizes"], 
            return_tensors="pt"
        )
        batch_scores = batch_outputs.iou_scores.tolist()

        masks.extend(batch_masks[0])
        scores.extend(batch_scores[0])

        del batch_inputs, batch_outputs
        torch.cuda.empty_cache()
    return masks, scores

def composite_mask(masks):
    num_masks = len(masks)
    final_image = np.zeros(masks[0][0].shape)
    instances_list = []

    for i in range(num_masks):
        mask_np = np.array(masks[i][2], dtype=np.uint8)

        final_image += mask_np

        seg = mask_to_polygons(mask_np)
        bbox = mask_to_bboxes(mask_np)
        
        instance_dict = {}
        instance_dict['segmentation'] = seg
        instance_dict['bbox'] = bbox

        instances_list.append(instance_dict)

    final_image = np.where(final_image > 1, 1, final_image)

    return final_image, num_masks, instances_list

def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) > 2:
            poly = contour.reshape(-1).tolist()
            polygons.append(poly)
    return polygons[0]

def mask_to_bboxes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for contour in contours:
        if len(contour) > 2:
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append([x, y, x + w, y + h])
    return bboxes[0]