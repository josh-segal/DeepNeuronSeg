from transformers import TFSamModel, SamProcessor
import numpy as np
import cv2

model = TFSamModel.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

def segment(raw_image, input_points):
    inputs = processor(raw_image, input_points=input_points, return_tensors="tf")
    outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"],
        return_tensors="tf",
    )
    scores = outputs.iou_scores

    return masks, scores

def composite_mask(masks):
    num_masks = masks[0].shape[0]
    final_image = np.zeros(masks[0][0][0].shape)
    instances_list = []

    for i in range(num_masks):
        mask_np = np.array(masks[0][i][2], dtype=np.uint8)

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
            #    if len(poly) > 4: #Ensures valid polygon
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