import json
import os
from PIL import Image
import numpy as np
import yaml

def get_data(file_path='data/image_metadata.json'):
    with open(file_path, 'r') as f:
                data = json.load(f)
    return data

def set_data(file_path='data/image_metadata.json', metadata=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(metadata, f, indent=1)

def save_label(data_labels_dir='data_labels', final_image=None, image_path=None):
    data_labels_dir = os.path.join('data', data_labels_dir)
    if not os.path.exists(data_labels_dir):
        os.makedirs(data_labels_dir)
    
    final_image_path = os.path.join(data_labels_dir, os.path.basename(image_path))
    final_image_pil = Image.fromarray((final_image * 255).astype(np.uint8))
    final_image_pil.save(final_image_path)
    return final_image_path

def yolo_dataset_preparation():
    data_dir = 'dataset'
    output_dir = 'large_dataset'

    # Define the paths for the images and labels for training and validation
    train_images_dir = os.path.join(output_dir, 'train', 'images')
    val_images_dir = os.path.join(output_dir, 'val', 'images')
    train_labels_dir = os.path.join(output_dir, 'train', 'labels')
    val_labels_dir = os.path.join(output_dir, 'val', 'labels')

    # Create the output directories if they do not exist
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # Get image and mask paths
    image_paths, mask_paths = get_image_mask_pairs(data_dir)

    # Split data into train and val
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)

    # Process and save the data in YOLO format for training and validation
    process_data(train_img_paths, train_mask_paths, train_images_dir, train_labels_dir)
    process_data(val_img_paths, val_mask_paths, val_images_dir, val_labels_dir)

    # Assume create_yaml function is defined elsewhere and set appropriate paths for the YAML file
    output_yaml_path = os.path.join(output_dir, 'data.yaml')
    train_path = os.path.join('train', 'images')
    val_path = os.path.join('val', 'images')

def create_yaml(output_yaml_path, train_images_dir, val_images_dir, nc=1):
    # Assuming all categories are the same and there is only one class, 'Cell'
    names = ['Cell']

    # Create a dictionary with the required content
    yaml_data = {
        'names': names,
        'nc': nc,  # Number of classes
        'train': train_images_dir,
        'val': val_images_dir,
        'test': ' '

    }

    # Write the dictionary to a YAML file
    with open(output_yaml_path, 'w') as file:
        yaml.dump(yaml_data, file, default_flow_style=False)

def get_image_mask_label_tuples(data_dir):
    image_paths = []
    mask_paths = []
    label_paths = []

    for root, _, files in os.walk(data_dir):
        if 'images' in root:
            for file in files:
                if file.endswith('.png'):
                    image_paths.append(os.path.join(root,file))
                    mask_paths.append(os.path.join(root.replace('images','masks'), file))
                    label_paths.append(os.path.join(root.replace('images', 'labels'), file.replace('.png', '.txt')))
    return image_paths, mask_paths, label_paths