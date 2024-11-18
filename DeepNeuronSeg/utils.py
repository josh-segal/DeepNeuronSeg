import json
import os
from PIL import Image
import numpy as np
import yaml
import pandas as pd
import xml.etree.ElementTree as ET
import cv2
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QSpinBox, QCheckBox, QPushButton

def norm_coords(coordinates, max_x=None, max_y=None):
    if max_x is not None:
        coordinates = [((x / max_x) * 512, y) for x, y in coordinates]
    if max_y is not None:
        coordinates = [(x, (y / max_y) * 512) for x, y in coordinates]
    return coordinates

def parse_png_label(label_file):
    label_array = np.array(Image.open(label_file))
    _, _, _, centroids = cv2.connectedComponentsWithStats(label_array)

    coordinates = [tuple(map(int, cent)) for cent in centroids[1:]]
    print(coordinates)
    return coordinates

def parse_txt_label(label_file):
    with open(label_file, 'r') as file:
        content = file.read()
        coordinates = []

        largest_x = 0
        largest_y = 0

        for line in content.strip().splitlines():
            x, y = map(float, line.strip().split('\t'))

            if x > largest_x:
                largest_x = x
            if y > largest_y:
                largest_y = y

            coordinates.append((x, y))
        print(coordinates)  

    if largest_x > 512 and largest_y > 512:
        coordinates = norm_coords(coordinates, max_x=largest_x, max_y=largest_y)
    elif largest_x > 512:
        coordinates = norm_coords(coordinates, max_x=largest_x)
    elif largest_y > 512:
        coordinates = norm_coords(coordinates, max_y=largest_y)

    return coordinates

def parse_csv_label(label_file):
    df = pd.read_csv(label_file)
    
    # Extract the 'X' and 'Y' columns
    x_values = df['X'].tolist()
    y_values = df['Y'].tolist()

    largest_x = max(x_values)
    largest_y = max(y_values)

    coordinates = list(zip(x_values, y_values))

    if largest_x > 512 and largest_y > 512:
        coordinates = norm_coords(coordinates, max_x=largest_x, max_y=largest_y)
    elif largest_x > 512:
        coordinates = norm_coords(coordinates, max_x=largest_x)
    elif largest_y > 512:
        coordinates = norm_coords(coordinates, max_y=largest_y)
    
    print(coordinates)
    return coordinates

def parse_xml_label(label_file):
    tree = ET.parse(label_file)
    root = tree.getroot()
    largest_x = 0
    largest_y = 0
    # Extract all MarkerX and MarkerY values
    coordinates = []
    for marker in root.findall('.//Marker'):
        x = marker.find('MarkerX').text
        y = marker.find('MarkerY').text

        if x > largest_x:
            largest_x = x
        if y > largest_y:
            largest_y = y

        coordinates.append((float(x), float(y)))

    if largest_x > 512 and largest_y > 512:
        coordinates = norm_coords(coordinates, max_x=largest_x, max_y=largest_y)
    elif largest_x > 512:
        coordinates = norm_coords(coordinates, max_x=largest_x)
    elif largest_y > 512:
        coordinates = norm_coords(coordinates, max_y=largest_y)

    print(coordinates)  
    return coordinates

def trim_underscores(file_name):
    base_name, ext = os.path.splitext(file_name)
    if base_name.endswith("_"):
        base_name = base_name[:-1]
    return base_name + ext

def check_data(data_path='data/image_metadata.json'):
    if os.path.exists(data_path):
        print("Data exists")
        existing_metadata = get_data()
    else:
        print("Data does not exist")
        existing_metadata = []
    return existing_metadata

def get_data(file_path='data/image_metadata.json'):
    """
    self.data_path = os.path.join('data', self.data_file)
        if os.path.exists(self.data_path):
    """
    if os.path.exists(file_path):
        # print("exists get_data")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except json.decoder.JSONDecodeError:
            data = []
        return data
    else:
        print(f'{file_path} does not exist')
        return []

def set_data(file_path='data/image_metadata.json', metadata=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(metadata, f, indent=4, separators=(',', ': '))
        # json.dump(metadata, f, indent=1)

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


class FrameSelectionDialog(QDialog):
    def __init__(self, max_frames):
        super().__init__()
        self.setWindowTitle("Select Frame")
        self.selected_frame = 0
        self.use_for_all = False

        layout = QVBoxLayout()
        label = QLabel(f"Select a frame (0 to {max_frames - 1}):")
        layout.addWidget(label)

        self.frame_selector = QSpinBox()
        self.frame_selector.setRange(0, max_frames - 1)
        layout.addWidget(self.frame_selector)

        self.checkbox = QCheckBox("Use this frame for all .tif files")
        layout.addWidget(self.checkbox)

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        layout.addWidget(ok_button)

        self.setLayout(layout)

    def accept(self):
        self.selected_frame = self.frame_selector.value()
        self.use_for_all = self.checkbox.isChecked()
        super().accept()