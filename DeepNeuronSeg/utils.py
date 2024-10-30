import json
import os
from PIL import Image
import numpy as np

def get_data(file_path='image_metadata.json'):
    with open(file_path, 'r') as f:
                data = json.load(f)
    return data

def set_data(file_path='image_metadata.json', metadata=None):
    with open(file_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def save_label(labeled_data_dir='labeled_data', final_image=None, image_path=None):
    labeled_data_dir = 'labeled_data'
    if not os.path.exists(labeled_data_dir):
        os.makedirs(labeled_data_dir)
    
    final_image_path = os.path.join(labeled_data_dir, os.path.basename(image_path))
    final_image_pil = Image.fromarray((final_image * 255).astype(np.uint8))
    final_image_pil.save(final_image_path)
    return final_image_path