from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, 
                           QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, 
                           QPushButton, QDialog, QFileDialog, QSpinBox, QComboBox,
                        QListWidget, QDoubleSpinBox, 
                           QCheckBox, QLineEdit, QGridLayout, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRect, QPointF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.lines as mlines
import sys
from dataclasses import dataclass
from typing import List, Dict, Any
import json
import os
import random
import shutil
from PIL import Image
import numpy as np
import tempfile
from tqdm import tqdm
from tinydb import TinyDB, Query


from utils import (get_data, set_data, save_label, get_image_mask_label_tuples, create_yaml, trim_underscores, 
parse_png_label, parse_txt_label, parse_csv_label, parse_xml_label, FrameSelectionDialog)
from inference import segment, composite_mask, mask_to_bboxes, mask_to_polygons
from qa import DetectionQAMetrics
from denoise_model import DenoiseModel

class ImageLabel(QLabel):
    """Custom QLabel to handle mouse clicks on the image area only."""
    left_click_registered = pyqtSignal(QPointF)
    right_click_registered = pyqtSignal(QPointF)
    
    def __init__(self):
        super().__init__()
        self.pixmap = None

    def set_pixmap(self, pixmap):
        self.pixmap = pixmap
        self.setPixmap(pixmap)

    def mousePressEvent(self, event):
        # print("mouse click")
        if self.pixmap:
            # print("pixmap present")
            click_pos = event.pos()
            if event.button() == Qt.LeftButton:
                self.left_click_registered.emit(click_pos)
            elif event.button() == Qt.RightButton:
                self.right_click_registered.emit(click_pos)

    def adjust_pos(self, pos):
        """Adjust the position to the image coordinates."""
        adjusted_x = pos.x() - (self.width() - self.pixmap.width()) / 2
        adjusted_pos = QPointF(adjusted_x, pos.y())
        return adjusted_pos

    def _draw_points(self, labels):
        """Draw a point on the image at the given position."""
        print(f"drawing {len(labels)} points")
        painter = QPainter(self.pixmap)
        painter.setPen(QPen(Qt.red, 5))
        for pos in labels:
            painter.drawPoint(QPointF(pos[0], pos[1]))
        self.setPixmap(self.pixmap)


class ImageDisplay(QWidget):
    """Widget for displaying and interacting with images"""
    
    def __init__(self, upload_tab):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.image_label = ImageLabel()
        self.text_label = QLabel()
        self.upload_tab = upload_tab
        
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.text_label)

        self.image_label.setMinimumSize(512, 512)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.text_label.setAlignment(Qt.AlignBottom | Qt.AlignCenter)

    def _display_image(self, image_path, image_num, total_images):
        """Load and display an image from the given file path and show image number."""
        print(image_path, '<----------------------')
        self.pixmap = QPixmap(image_path)
        if not self.pixmap.isNull():
            self.image_label.set_pixmap(self.pixmap)
            self.text_label.setText(f"{image_num} / {total_images}")
        else:
            print("Failed to load image")

    def tif_check(self, item):
        if item.lower().endswith('.tif'):
            with Image.open(item) as img:
                img.seek(self.upload_tab.selected_frame)
                frame_to_display = img.copy()
                base_name, _ = os.path.splitext(item)
                temp_image_path = os.path.join(tempfile.gettempdir(), base_name + ".png")
                frame_to_display.save(temp_image_path, format='PNG')
                item = temp_image_path
        return item

    def show_item(self, mask=False, points=False, next_item=False, index=False):
        if mask:
            items = self.upload_tab.db.load_masks()
        else:
            items = self.upload_tab.db.load_images()

        if next_item:
            self.upload_tab.current_index = (self.upload_tab.current_index + 1) % len(items)

        if index:
            self.upload_tab.current_index = index

        if items:
            if self.upload_tab.current_index < len(items) and len(items) > 0:
                item = self.tif_check(items[self.upload_tab.current_index])
                self._display_image(item, self.upload_tab.current_index + 1, len(items))
                if points:
                    labels = self.upload_tab.db.load_labels()
                    self.image_label._draw_points(labels[self.upload_tab.current_index])
            else:
                print("No images uploaded")
                self.image_label.clear()
                self.text_label.setText("No mask generated" if mask else "No images uploaded")
        else:
            print("No images uploaded")
            self.image_label.clear()
            self.text_label.setText("No mask generated" if mask else "No images uploaded")
            

        

        


    # def show_image(self):
    #     """Display the next image in the list."""
    #     if len(self.upload_tab.uploaded_files) > 0:
    #         if self.upload_tab.uploaded_files[self.upload_tab.current_index].lower().endswith('.tif'):
    #             print("displaying tif")
    #             with Image.open(self.upload_tab.uploaded_files[self.upload_tab.current_index]) as img:
    #                 img.seek(self.upload_tab.selected_frame)
    #                 frame_to_display = img.copy() #TODO: image must be converted to gray and converted to rgb first
    #                 temp_image_path = os.path.join(tempfile.gettempdir(), "temp_image.png")
    #                 frame_to_display.save(temp_image_path, format='PNG')
    #                 print("temp image path", temp_image_path)
    #                 self.display_image(temp_image_path, self.upload_tab.current_index + 1, len(self.upload_tab.uploaded_files))
    #         else:
    #             self.display_image(self.upload_tab.uploaded_files[self.upload_tab.current_index], self.upload_tab.current_index + 1, len(self.upload_tab.uploaded_files))
    #     else:
    #         print("No images uploaded")

    # def show_mask(self):
    #     """Display the next mask in the list."""
    #     if len(self.upload_tab.metadata_labels) > 0:
    #         self.display_image(self.upload_tab.metadata_labels[self.upload_tab.current_index], self.upload_tab.current_index + 1, len(self.upload_tab.metadata_labels))
    #     else:
    #         print("No masks generated")

    # def show_next_image(self):
    #     """Display the next image in the list."""
    #     if len(self.upload_tab.uploaded_files) > 0:
    #         self.upload_tab.current_index = (self.upload_tab.current_index + 1) % len(self.upload_tab.uploaded_files)  # Wrap around
    #         self.show_image()
    #     else:
    #         print("No images uploaded")
    #         self.image_label.clear() 
    #         self.text_label.setText("")  


    # def show_next_mask(self):
    #     """Display the next mask in the list."""
    #     if len(self.upload_tab.metadata_labels) > 0:
    #         self.upload_tab.current_index = (self.upload_tab.current_index + 1) % len(self.upload_tab.metadata_labels)  # Wrap around
    #         self.show_mask()
    #     else:
    #         print("No masks generated")
    #         self.image_label.clear() 
    #         self.text_label.setText("")

    # def show_image_with_points(self):
    #     """Display the next image in the list."""
    #     if len(self.upload_tab.uploaded_files) > 0:
    #         self.show_image()
    #         self.image_label._draw_points(self.upload_tab.labels[self.upload_tab.current_index])
    #     else:
    #         print("No images uploaded")
    #         self.image_label.clear() 
    #         self.text_label.setText("")


    # def show_next_image_with_points(self):
    #     """Display the next image in the list."""
    #     if len(self.upload_tab.uploaded_files) > 0:
    #         self.show_next_image()
    #         self.image_label._draw_points(self.upload_tab.labels[self.upload_tab.current_index])
    #     else:
    #         print("No images uploaded")
    #         self.image_label.clear() 
    #         self.text_label.setText("")  

    # def show_image_indexed(self, index):
    #     """Display the image at the given index."""
    #     self.upload_tab.current_index = index
    #     self.show_image()

class UploadTab(QWidget):
    def __init__(self, db):
        super().__init__()
        self.db = db
        self.current_index = 0
        self.uploaded_files = []
        layout = QVBoxLayout()

         # Image preview
        self.image_display = ImageDisplay(self)
        
        # File list
        self.file_list = QListWidget()
        self.file_list.addItems([os.path.basename(file) for file in self.db.load_images()])
        
        # File selection
        self.upload_btn = QPushButton("Upload Images")
        self.upload_label_btn = QPushButton("Upload Labels")
        self.next_btn = QPushButton("Next Image")
        self.load_btn = QPushButton("Display Data")

        self.upload_btn.clicked.connect(self.upload_images)
        self.upload_label_btn.clicked.connect(self.upload_labels)
        self.load_btn.clicked.connect(self.image_display.show_item)
        self.next_btn.clicked.connect(lambda: self.image_display.show_item(next_item=True))

        self.file_list.itemClicked.connect(lambda item: self.image_display.show_item(index=self.file_list.row(item)))
        
        # Metadata input fields
        metadata_layout = QGridLayout()
        self.project = QLineEdit()
        self.cohort = QLineEdit()
        self.brain_region = QLineEdit()
        self.image_id = QLineEdit()
        metadata_layout.addWidget(QLabel("Project:"), 0, 0)
        metadata_layout.addWidget(self.project, 0, 1)
        metadata_layout.addWidget(QLabel("Cohort:"), 0, 2)
        metadata_layout.addWidget(self.cohort, 0, 3)
        metadata_layout.addWidget(QLabel("Brain Region:"), 1, 0)
        metadata_layout.addWidget(self.brain_region, 1, 1)
        metadata_layout.addWidget(QLabel("Image ID:"), 1, 2)
        metadata_layout.addWidget(self.image_id, 1, 3)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.upload_btn)
        button_layout.addWidget(self.upload_label_btn)
        button_layout.addWidget(self.next_btn)
        button_layout.addWidget(self.load_btn)
        
       
        
        layout.addLayout(button_layout)
        layout.addLayout(metadata_layout)
        layout.addWidget(self.image_display)
        layout.addWidget(self.file_list)
        self.setLayout(layout)
    
    def upload_images(self):
        """
        INTEGRATION POINT:
        1. Implement image file selection
        2. Save metadata alongside images
        3. Update UI with selected images
        """
        self.uploaded_files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.tif)")

        self.use_selected_frame_for_all = False
        self.selected_frame = 0

        image_data = Query()
        for file in self.uploaded_files[:]:

            image_name = os.path.basename(file)
            image_name = trim_underscores(image_name)
            image_name = image_name.replace(".tif", ".png")

            #TODO: store this path somewhere so not hardcoded ?
            image_path = os.path.join('data', 'data_images', image_name)

            if self.db.image_table.get(image_data.file_path == image_name):
                print(f"Image already exists in database {image_name}")
                self.uploaded_files.remove(file)
                continue
        
            # tif operations
            if file.lower().endswith('.tif'):
                with Image.open(file) as img:
                        num_frames = img.n_frames

                        if num_frames > 1 and not self.use_selected_frame_for_all:
                            dialog = FrameSelectionDialog(num_frames)
                            if dialog.exec_() == QDialog.Accepted:
                                self.selected_frame = dialog.selected_frame
                                self.use_selected_frame_for_all = dialog.use_for_all

                            img.seek(self.selected_frame)
                            frame_to_save = img.copy()
                            frame_to_save.save(image_path, format='PNG')
                        else:
                            print("Converting tif to png", image_path)
                            img.save(image_path, format='PNG')

            # png operations
            else:
                shutil.copy(file, image_path)
            
            file = image_path

            # add to db
            #TODO: add an apply to all for some metadata get rid of or automate per image ones, don't need metadata not a database.
            self.db.image_table.insert({
                "file_path": image_path, 
                "project": self.project.text(), 
                "cohort": self.cohort.text(), 
                "brain_region": self.brain_region.text(), 
                "image_id": self.image_id.text() if self.image_id.text() else len(self.db.image_table),
                "labels": []
                })

        self.file_list.addItems([os.path.basename(file["file_path"]) for file in self.db.load_images()])
        self.image_display.show_item()

    def upload_labels(self):
        self.uploaded_labels, _ = QFileDialog.getOpenFileNames(self, "Select Labels", "", "Labels (*.png *.txt *.csv *.xml)")
        labels = self.parse_labels(self.uploaded_labels)

    def parse_labels(self, labels):

        for label_file in labels:
            label_name = os.path.splitext(os.path.basename(label_file))[0]
            label_name = trim_underscores(label_name)
            label_name = "data" + "\\" + "data_images" + "\\" + label_name + ".png"

            image_data = Query()
            matched_image = self.db.image_table.get(image_data.file_path == label_name)
            if matched_image:
                if label_file.endswith(".png"):
                    label_data = parse_png_label(label_file)
                elif label_file.endswith(".txt"):
                    label_data = parse_txt_label(label_file)
                elif label_file.endswith(".csv"):
                    label_data = parse_csv_label(label_file)
                elif label_file.endswith(".xml"):
                    label_data = parse_xml_label(label_file)
                else:
                    print("Invalid label format")
                    label_data = []

                if label_data:
                    self.db.image_table.update({"labels": label_data}, image_data.file_path == label_name)
            else:
                print(f"Image not found in database {label_name}")
                continue

    def update(self):
        self.file_list.clear()
        self.file_list.addItems([os.path.basename(file) for file in self.db.load_images()])


class LabelingTab(QWidget):
    def __init__(self, db):
        super().__init__()
        self.db = db
        self.current_index = 0
        self.uploaded_files = []
        self.image_display = ImageDisplay(self)
        layout = QVBoxLayout()
        self.load_btn = QPushButton("Display Data")
        self.next_btn = QPushButton("Next Image")
        self.next_btn.clicked.connect(lambda: self.image_display.show_item(next_item=True, points=True))
        self.load_btn.clicked.connect(self.load_data)
    
        

        self.image_display.image_label.left_click_registered.connect(self.add_cell_marker)
        self.image_display.image_label.right_click_registered.connect(self.remove_cell_marker)
        
        # Controls
        controls_layout = QHBoxLayout()
        # self.undo_btn = QPushButton("Undo Last")
        # self.clear_btn = QPushButton("Clear All")
        # self.save_btn = QPushButton("Save Labels")
        # controls_layout.addWidget(self.undo_btn)
        # controls_layout.addWidget(self.clear_btn)
        # controls_layout.addWidget(self.save_btn)
        
        layout.addWidget(self.image_display)
        layout.addWidget(self.next_btn)
        layout.addWidget(self.load_btn)
        layout.addLayout(controls_layout)
        self.setLayout(layout)
       
    def load_data(self):
        # self.uploaded_files, self.labels = self.db.load_images_and_labels()
        self.image_display.show_item(points=True)
        
        

    def add_cell_marker(self, pos):
        # print("adding cell")
        adjusted_pos = self.image_display.image_label.adjust_pos(pos)
        if not (0 <= adjusted_pos.x() <= 512 and 0 <= adjusted_pos.y() <= 512):
            return

        # Get all records from the image_table
        images = self.db.image_table.all()

        # Define file_path based on self.current_index
        file_path = images[self.current_index]['file_path'] if 0 <= self.current_index < len(images) else None

        image_query = Query()
        image_data = self.db.image_table.get(image_query.file_path == file_path)
        if image_data:
            self.db.image_table.update({"labels": image_data.get("labels", []) + [(adjusted_pos.x(), adjusted_pos.y())]}, image_query.file_path == file_path)
            self.image_display.show_item(points=True)
            # self.image_display.show_image_with_points()

    def remove_cell_marker(self, pos, tolerance=5):
        adjusted_pos = self.image_display.image_label.adjust_pos(pos)
        if not (0 <= adjusted_pos.x() <= 512 and 0 <= adjusted_pos.y() <= 512):
            return

        # Get all records from the image_table
        images = self.db.image_table.all()

        # Define file_path based on self.current_index
        file_path = images[self.current_index]['file_path'] if 0 <= self.current_index < len(images) else None

        image_query = Query()
        image_data = self.db.image_table.get(image_query.file_path == file_path)
        if image_data:
            # Update labels: append the new position
            self.db.image_table.update({"labels": [label for label in image_data.get("labels", []) if not (abs(label[0] - position[0]) < tolerance and abs(label[1] - position[1]) < tolerance)]}, image_query.file_path == file_path)
            self.image_display.show_item(points=True)
            # self.image_display.show_image_with_points()

class GenerateLabelsTab(QWidget):
    def __init__(self, db):
        super().__init__()
        self.db = db
        layout = QVBoxLayout()
        image_layout = QHBoxLayout()
        config_layout = QHBoxLayout()

        self.current_index = 0
        self.uploaded_files = []
        self.metadata_labels = []
        self.data = []

        self.left_image = ImageDisplay(self)
        self.right_image = ImageDisplay(self)

        self.generate_btn = QPushButton("Generate Labels")
        self.next_btn = QPushButton("Next Image")
        self.display_btn = QPushButton("Display Labels")
        self.generate_btn.clicked.connect(self.generate_labels)
        self.next_btn.clicked.connect(lambda: self.left_image.show_item(next_item=True))
        self.next_btn.clicked.connect(lambda: self.right_image.show_item(mask=True))
        self.display_btn.clicked.connect(self.display_labels)

        
        config_layout.addWidget(self.generate_btn)
        config_layout.addWidget(self.next_btn)
        config_layout.addWidget(self.display_btn)

        
        image_layout.addWidget(self.left_image)
        image_layout.addWidget(self.right_image)

        layout.addLayout(image_layout)
        layout.addLayout(config_layout)

        self.setLayout(layout)

    def generate_labels(self):
        # print(result)
        # print(self.uploaded_files)
        # print(self.labels)
        # print(len(result))

        query = Query()

        data_to_mask = self.db.image_table.search(~query["mask_data"].exists())

        # self.uploaded_files = self.db.load_images()
        # self.labels = self.db.load_labels()

        for item in tqdm(data_to_mask, desc="Generating Labels", unit="image"):
            label = item.get("labels", [])
            file_path = item.get("file_path", "")

            if not label:
                print("No labels provided for image", file_path)
                continue

            mask_data = self.generate_label(file_path, label)
            self.db.image_table.update({"mask_data": mask_data}, query.file_path == file_path)


        self.display_labels()
        # for i, (uploaded_file, label) in enumerate(tqdm(zip(self.uploaded_files, self.labels), total=len(self.uploaded_files), desc="Generating Labels", unit="image")):
        #     # print(i)
        #     if label is None:
        #         print("No labels provided for image", uploaded_file)
        #         continue

        #     # add TQDM progress bar before images are shown
        #     label_data = self.generate_label(uploaded_file, label)
        #     for image in self.data:
        #         if image["file_path"] == uploaded_file:
        #             image["mask_data"] = label_data
        #             break

        #     # save somewhere somehow in relation to uploaded_file
        #     set_data(metadata=self.data)
        # display image label pairs with button to see next pair
        
        # allow user editing of generated labels

    def generate_label(self, image_path, labels):
        """Generate labels for the given image."""
        masks, scores = segment(image_path, labels)
        final_image, num_cells, instances_list = composite_mask(masks)

        final_image_path = save_label(final_image=final_image, image_path=image_path)
        # print(scores)
        # print(scores[0])
        # save final_image to labeled_data folder
        # print("final_image_path", final_image_path)
        # print("scores", scores_numpy)
        # print("num_cells", num_cells)
        # print("instances_list", instances_list)
        return {
            "mask_path": final_image_path,
            "scores": scores[0],
            "num_cells": num_cells,
            "instances_list": instances_list
        }

    def display_labels(self):
        self.left_image.show_item()
        self.right_image.show_item(mask=True)

    def update(self):
        pass


class DatasetTab(QWidget):
    def __init__(self, db):
        super().__init__()
        self.db = db
        layout = QVBoxLayout()
        
        # Dataset configuration
        config_layout = QGridLayout()
        self.train_split = QDoubleSpinBox()
        self.train_split.setRange(0.0, 1.0)
        self.train_split.setSingleStep(0.05)
        self.train_split.setValue(0.8)
        self.dataset_name = QLineEdit()
        
        # Augmentation options
        self.flip_horizontal = QCheckBox("Horizontal Flip")
        self.flip_vertical = QCheckBox("Vertical Flip")
        self.enable_rotation = QCheckBox("Enable Rotation")
        self.enable_crop = QCheckBox("Random Crop")
        
        config_layout.addWidget(QLabel("Train Split:"), 0, 0)
        config_layout.addWidget(self.train_split, 0, 1)
        config_layout.addWidget(self.flip_horizontal, 1, 0)
        config_layout.addWidget(self.flip_vertical, 1, 1)
        config_layout.addWidget(self.enable_rotation, 2, 0)
        config_layout.addWidget(self.enable_crop, 2, 1)
        config_layout.addWidget(QLabel("Dataset Name:"), 3, 0)
        config_layout.addWidget(self.dataset_name, 3, 1)
        
        # Image selection
        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QListWidget.MultiSelection)
        
        # Creation button
        self.create_btn = QPushButton("Create Dataset")
        self.create_btn.clicked.connect(self.create_dataset)
        
        layout.addLayout(config_layout)
        layout.addWidget(self.image_list)
        layout.addWidget(self.create_btn)
        self.setLayout(layout)
    
    def create_dataset(self):
        """
        INTEGRATION POINT:
        1. Get selected images
        2. Apply augmentation based on settings
        3. Create train/test split
        4. Save dataset configuration
        """
        uploaded_images = self.db.load_images()
        uploaded_masks = self.db.load_masks()
        uploaded_labels = self.db.load_labels()

        dataset_parent_dir = os.path.join('data', 'datasets')
        os.makedirs(dataset_parent_dir, exist_ok=True)

        # if not os.path.exists(os.path.join(dataset_parent_dir, 'dataset_metadata.json')):
        #     with open(os.path.join(dataset_parent_dir, 'dataset_metadata.json'), 'w') as f:
        #         json.dump({}, f)

        dataset_dir = 'dataset'
        counter = 0
        self.dataset_path = os.path.abspath(os.path.join(dataset_parent_dir, f"{dataset_dir}_{counter}"))
        while os.path.exists(self.dataset_path):
            counter += 1
            self.dataset_path = os.path.abspath(os.path.join(dataset_parent_dir, f"{dataset_dir}_{counter}"))

        # dataset_metadata = get_data(file_path=os.path.join(dataset_parent_dir, 'dataset_metadata.json'))

        if not self.dataset_name.text().strip():
            self.dataset_name.setText(f"{self.dataset_path}")
            print("Dataset name not provided, using default")
        if self.dataset_name.text().strip() in dataset_metadata.keys():
            print("Dataset name already exists, please choose a different name")
            return

        self.db.dataset_table.insert({
            "dataset_name": self.dataset_name.text().strip(),
            "dataset_path": self.dataset_path
        })

        # set_data(file_path=os.path.join(dataset_parent_dir, 'dataset_metadata.json'), metadata=dataset_metadata)

        os.makedirs(self.dataset_path, exist_ok=False)
        os.makedirs(os.path.join(self.dataset_path, "images"), exist_ok=False)
        os.makedirs(os.path.join(self.dataset_path, "masks"), exist_ok=False)
        os.makedirs(os.path.join(self.dataset_path, "labels"), exist_ok=False)

        for image, mask, labels in zip(uploaded_images, uploaded_masks, uploaded_labels):
            # print(labels,"\n", "\n")
            image_name = os.path.basename(image)
            mask_name = os.path.basename(mask)
            
            image_path = os.path.join(self.dataset_path, "images", image_name)
            mask_path = os.path.join(self.dataset_path, "masks", mask_name)
            label_path = os.path.join(self.dataset_path, "labels", f"{os.path.splitext(image_name)[0]}.txt")

            shutil.copy(image, image_path)
            shutil.copy(mask, mask_path)

            with open(label_path, "w") as f:
                for label in labels:
                    # print(label, "\n")
                    normalized_label = [format(coord / 512 if i % 2 == 0 else coord / 512, ".6f") for i, coord in enumerate(label)]
                    f.write(f"0 " + " ".join(normalized_label) + "\n")

        self.create_shuffle()

    def create_shuffle(self):
        image_paths, mask_paths, label_paths = get_image_mask_label_tuples(self.dataset_path)

        #TODO: would train test split be more appropriate here?
        combined = list(zip(image_paths, mask_paths, label_paths))
        random.shuffle(combined)
        shuffled_image_paths, shuffled_mask_paths, shuffled_label_paths = zip(*combined)

        split_index = int(len(shuffled_image_paths) * self.train_split.value())

        train_images = shuffled_image_paths[:split_index]
        val_images = shuffled_image_paths[split_index:]
        train_masks = shuffled_mask_paths[:split_index]
        val_masks = shuffled_mask_paths[split_index:]
        train_labels = shuffled_label_paths[:split_index]
        val_labels = shuffled_label_paths[split_index:]

        # counter = 0
        # shuffle_path = os.path.join(self.dataset_path, f"shuffle_{counter}")
        # while os.path.exists(shuffle_path):
        #     counter += 1
        #     shuffle_path = os.path.join(self.dataset_path, f"shuffle_{counter}")

        # os.makedirs(shuffle_path, exist_ok=False)

        train_images_dir = os.path.join( "train", "images")
        val_images_dir = os.path.join( "val", "images")

        create_yaml(os.path.join(self.dataset_path, "data.yaml"), train_images_dir, val_images_dir)

        os.makedirs(os.path.join(self.dataset_path, train_images_dir), exist_ok=False)
        os.makedirs(os.path.join(self.dataset_path, "train", "masks"), exist_ok=False)
        os.makedirs(os.path.join(self.dataset_path, "train", "labels"), exist_ok=False)

        os.makedirs(os.path.join(self.dataset_path, val_images_dir), exist_ok=False)
        os.makedirs(os.path.join(self.dataset_path, "val", "masks"), exist_ok=False)
        os.makedirs(os.path.join(self.dataset_path, "val", "labels"), exist_ok=False)

        for image, mask, label in zip(train_images, train_masks, train_labels):
            shutil.copy(image, os.path.join(self.dataset_path, "train", "images", os.path.basename(image)))
            shutil.copy(mask, os.path.join(self.dataset_path, "train", "masks", os.path.basename(mask)))
            shutil.copy(label, os.path.join(self.dataset_path, "train", "labels", os.path.basename(label)))

        for image, mask, label in zip(val_images, val_masks, val_labels):
            shutil.copy(image, os.path.join(self.dataset_path, "val", "images", os.path.basename(image)))
            shutil.copy(mask, os.path.join(self.dataset_path, "val", "masks", os.path.basename(mask)))
            shutil.copy(label, os.path.join(self.dataset_path, "val", "labels", os.path.basename(label)))

    def update(self):
        pass


class TrainingTab(QWidget):
    def __init__(self, db):
        super().__init__()
        self.db = db
        layout = QVBoxLayout()
        
        # Model selection
        self.model = None
        self.model_selector = QComboBox()
        self.model_selector.addItems(["YOLOv8n-seg"])
        
        # Training parameters
        params_layout = QGridLayout()
        self.dataset = QComboBox()

        dataset_dict = get_data(file_path='data/datasets/dataset_metadata.json')
        if dataset_dict:
            dataset_list = list(dataset_dict.keys())
            self.dataset.addItems(dataset_list)

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 128)
        self.model_name = QLineEdit()
        self.denoise = QCheckBox("Use Denoising Network")
        
        self.epochs_label = QLabel("Epochs:")
        self.dataset_label = QLabel("Dataset:")
        self.batch_size_label = QLabel("Batch Size:")
        self.model_name_label = QLabel("Trained Model Name:")
        self.epochs_label.setToolTip("""
        Epochs:
        -------
        Epochs refer to the number of complete passes through the entire training dataset 
        during the training process.

        Default Value:
        --------------
        The default number of epochs is set to 70.

        Notes:
        -------
        Too few epochs can lead to underfitting, while too many may result in 
        overfitting. 

        In general largers datasets require more epochs and smaller datasets require fewer epochs. 
        Watch validation metrics to determine if a model is underfitting or overfitting. 
        
        validation loss; if training loss is decreasing but validation loss is increasing or plateuing, the model is likely overfitting
        validation accurary; if training accuracy is increasing but validation accuracy is decreasing or plateuing, the model is likely overfitting

        if validation loss/accuracy and/or training loss/accuracy are not plateuing of overfitting, the model is likely underfitting.
        
        """)
        self.dataset_label.setToolTip("""
        Dataset:
        --------
        Dataset refers to the ID of the dataset you wish to train your model on.

        Default Value:
        --------------
        The default dataset ID is set to 1.

        Notes:
        -------
        Training on different datasets or shuffles of the same dataset can produce different model results.
        
        """)
        self.batch_size_label.setToolTip("""
        Batch Size:
        -----------
        Batch size refers to the number of images the model sees before updating the weights.

        Default Value:
        --------------
        The default batch size is set to 4.

        Notes:
        -------
        Larger batch sizes may speed up and stabilize training but require more memory. Smaller batch sizes update weights more frequently and may lead to better generalization.
        
        """)
        self.model_name_label.setToolTip("""
        Trained Model Name:
        --------------------
        The name of the trained model that will be saved after training.
        
        Default Value:
        --------------
        The default model name is set to 'model'.
        
        Notes:
        -------
        The model name is used to save the trained model after training. The model will be saved in the 'models' directory.
        
        """)
        self.denoise.setToolTip("""

        Denoising Network:
        -------------------
        Denoising network refers to the use of a UNet model to denoise the dataset before training the main model.

        Default Value:
        --------------
        The default value is set to False.

        Notes:
        -------
        Denoising the dataset can improve the quality of the training data and the performance of the model, but may increase training time and introduces additional preprocessing steps during training and inference.

        """)

        params_layout.addWidget(self.dataset_label, 0, 0)
        params_layout.addWidget(self.dataset, 0, 1)
        params_layout.addWidget(self.epochs_label, 1, 0)
        params_layout.addWidget(self.epochs, 1, 1)
        params_layout.addWidget(self.batch_size_label, 2, 0)
        params_layout.addWidget(self.batch_size, 2, 1)
        params_layout.addWidget(self.model_name_label, 3, 0)
        params_layout.addWidget(self.model_name, 3, 1)
        params_layout.addWidget(self.denoise, 4, 1)
        
        # Control buttons
        self.train_btn = QPushButton("Start Training")
        # self.stop_btn = QPushButton("Stop")

        self.train_btn.clicked.connect(self.trainer)
        
        label = QLabel("Base Model:")
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        label.setFixedHeight(10)
        layout.addWidget(label)
        layout.addWidget(self.model_selector)
        layout.addLayout(params_layout)
        layout.addWidget(self.train_btn)
        # layout.addWidget(self.stop_btn)
        self.setLayout(layout)

    def trainer(self):

        if not self.model_name.text().strip():
            print("Model name required")
            return

        model_name_exists = self.db.model_table.contains(Query()[field_name] == value_to_check)
        if model_name_exists:
            print("Model name already exists, please choose a different name")
            return
        else:
            self.db.model_table.insert({
                "model_name": self.model_name.text().strip(),
                "model_path": f'{dataset}/results/{self.model_name.text().strip()}'
            })

        dataset = self.db.dataset_table.get(Query().dataset_name == self.dataset.currentText())
        dataset_path = dataset.get("dataset_path", "")


        if self.denoise.isChecked():
            print("Training denoising network")

            dn_model = DenoiseModel(dataset_path=dataset_path, model_path='models/denoise_model.pth')
            dn_model.unet_trainer(num_epochs=self.epochs.value(), batch_size=self.batch_size.value())
            dn_model.create_dn_shuffle()

        if self.model_selector.currentText() == "YOLOv8n-seg":
            # offset program load times by loading model here
            from ultralytics import YOLO
            print("Training YOLOv8n-seg")

            self.model = YOLO("models/yolov8n-seg.pt")
            self.model.train(
                #TODO: if denoised use denoised data dir, recreate yaml (?)
                data = os.path.abspath(f'{dataset_path}/data.yaml'),
                project = f'{dataset_path}/results',
                name = self.model_name.text().strip(),
                epochs = self.epochs.value(),
                patience = 0,
                batch = self.batch_size.value(),
                imgsz = 1024
            )
    def update(self):
        self.dataset.clear()
        self.dataset.addItems(map(lambda dataset: dataset['dataset_name'], self.db.load_datasets()))

class EvaluationTab(QWidget):
    def __init__(self, db):
        super().__init__()
        self.db = db
        layout = QVBoxLayout()
        metrics_layout = QGridLayout()
        
        # Model selection
        self.model_selector = QComboBox()
        # model_dict = get_data(file_path='models/model_metadata.json')
        # if model_dict:
        #     model_list = list(model_dict.keys())
        #     self.model_selector.addItems(model_list)

        self.model_selector.addItems(map(lambda model: model['model_name'], self.db.load_models()))

        self.dataset_selector = QComboBox()
        # dataset_dict = get_data(file_path='data/datasets/dataset_metadata.json')
        # if dataset_dict:
        #     dataset_list = list(dataset_dict.keys())
        #     self.dataset_selector.addItems(dataset_list)

        self.dataset_selector.addItems(map(lambda dataset: dataset['dataset_name'], self.db.load_datasets()))
        
        # Visualization area (placeholder for distribution plots)
        self.canvas = FigureCanvas(Figure(figsize=(12, 5)))

        self.calculate_metrics_btn = QPushButton("Calculate Metrics")
        self.calculate_metrics_btn.clicked.connect(self.calculate_metrics)

        self.downoad_data_btn = QPushButton("Download Data")
        self.downoad_data_btn.clicked.connect(self.download_data)

         # Labels for metrics
        self.confidence_mean_mean_label = QLabel("Average Confidence Score: N/A")
        self.confidence_mean_mean_label.setToolTip("""
        Average confidence of predictions per image, averaged across all images.
        """)
        self.confidence_mean_std_label = QLabel("Confidence Score Variability: N/A")
        self.confidence_mean_std_label.setToolTip("""
        Average confidence of predictions per image, standard deviation across all images.
        """)
        self.confidence_std_mean_label = QLabel("Average Confidence Spread: N/A")
        self.confidence_std_mean_label.setToolTip("""
        Standard deviation of confidence of predictions per image, averaged across all images.
        """)
        self.confidence_std_std_label = QLabel("Confidence Spread Variability: N/A")
        self.confidence_std_std_label.setToolTip("""
        Standard deviation of confidence of predictions per image, standard deviation across all images.
        """)
        self.num_detections_mean_label = QLabel("Average Number of Detections: N/A")
        self.num_detections_mean_label.setToolTip("""
        Average number of detections per image.
        """)
        self.num_detections_std_label = QLabel("Detection Count Variability: N/A")
        self.num_detections_std_label.setToolTip("""
        Standard deviation of number of detections per image.
        """)
        self.area_mean_mean_label = QLabel("Average Detection Area: N/A")
        self.area_mean_mean_label.setToolTip("""
        Average area of detections per image, averaged across all images.
        """)
        self.area_mean_std_label = QLabel("Detection Area Variability: N/A")
        self.area_mean_std_label.setToolTip("""
        Average area of detections per image, standard deviation across all images.
        """)
        self.area_std_mean_label = QLabel("Average Area Spread: N/A")
        self.area_std_mean_label.setToolTip("""
        Standard deviation of area of detections per image, averaged across all images.
        """)
        self.area_std_std_label = QLabel("Area Spread Variability: N/A")
        self.area_std_std_label.setToolTip("""
        Standard deviation of area of detections per image, standard deviation across all images.
        """)
        self.overlap_ratio_mean_label = QLabel("Average Overlap Rati: N/A")
        self.overlap_ratio_mean_label.setToolTip("""
        Average overlap ratio of detections per image.
        """)
        self.overlap_ratio_std_label = QLabel("Overlap Ratio Variability: N/A")
        self.overlap_ratio_std_label.setToolTip("""
        Standard deviation of overlap ratio of detections per image.
        """)

        layout.addWidget(QLabel("Trained Model:"))
        layout.addWidget(self.model_selector)
        layout.addWidget(QLabel("Dataset:"))
        layout.addWidget(self.dataset_selector)
        layout.addWidget(self.canvas)
        layout.addWidget(self.calculate_metrics_btn)
        layout.addWidget(self.downoad_data_btn)

        # Adding metric labels to layout
        metrics_layout.addWidget(self.confidence_mean_mean_label, 0, 0)
        metrics_layout.addWidget(self.confidence_mean_std_label, 0, 1)
        metrics_layout.addWidget(self.confidence_std_mean_label, 0, 2)
        metrics_layout.addWidget(self.confidence_std_std_label, 0, 3)
        metrics_layout.addWidget(self.num_detections_mean_label, 0, 4)
        metrics_layout.addWidget(self.num_detections_std_label, 0, 5)
        metrics_layout.addWidget(self.area_mean_mean_label, 1, 0)
        metrics_layout.addWidget(self.area_mean_std_label, 1, 1)
        metrics_layout.addWidget(self.area_std_mean_label, 1, 2)
        metrics_layout.addWidget(self.area_std_std_label, 1, 3)
        metrics_layout.addWidget(self.overlap_ratio_mean_label, 1, 4)
        metrics_layout.addWidget(self.overlap_ratio_std_label, 1, 5)

        layout.addLayout(metrics_layout)
        self.setLayout(layout)
        
        """
        INTEGRATION POINT:
        1. Implement distribution plotting
        2. Calculate and display statistics
        3. Load and compare model predictions
        """

    def download_data(self):
        pass
    
        # check if metrics already calculated for model
        # load the dataset images
        # inference trained model on dataset images
        # calculate metrics / distributions across inferences
        # display metrics and distributions in meaningful way
        # in analyze data return quality score of inferenced image

    def calculate_metrics(self):
        # TODO: abstract
        self.model_name = self.model_selector.currentText()
        self.model_path = self.db.model_table.get(Query().model_name == model_name).get('model_path')
        print(self.model_path)
        # self.model_path = '/Users/joshua/garnercode/cellCountingModel/notebooks/yolobooks2/large_dataset/results/70_epochs_n_large_data-/weights/best.pt'
        self.dataset_name = self.dataset_selector.currentText()
        self.dataset_path = self.db.dataset_table.get(Query().dataset_name == dataset_name).get('dataset_path')
        print(self.dataset_path)
        # dataset_path = '/Users/joshua/garnercode/DeepNeuronSeg/DeepNeuronSeg/data/datasets/dataset_0/images'
        # dataset_path = 'C:/Users/joshua/garnercode/cellCountingModel/notebooks/yolobooks2/dataset/COCO_train_X'
        # dataset_path = 'C:/Users/joshua/garnercode/cellCountingModel/notebooks/yolobooks2/large_dataset/train/images'

        self.metrics = DetectionQAMetrics(self.model_path, self.dataset_path)
        print(self.metrics.dataset_metrics_mean_std)
        self.plot_metrics(self.metrics.dataset_metrics, self.metrics.dataset_metrics_mean_std)

    def plot_metrics(self, metrics, metrics_mean_std):
        # bins = len(metrics['confidence_mean'])**0.5
        ax1, ax2 = self.canvas.figure.subplots(1, 2)

        # Sort by num_detections and apply the same order to confidence_mean
        sorted_indices = sorted(range(len(metrics["num_detections"])), key=lambda i: metrics["num_detections"][i])

        sorted_num_detections = [metrics["num_detections"][i] for i in sorted_indices]
        sorted_conf_mean = [metrics["confidence_mean"][i] for i in sorted_indices]
        
        # Plotting histograms
        ax1.bar(range(len(sorted_conf_mean)), sorted_conf_mean, color='skyblue', edgecolor='black')
        ax1.set_title("Mean Confidence of Predictions Per Image")
        ax1.set_xlabel("Image")
        ax1.set_ylabel("Mean Confidence")

        ax2.bar(range(len(sorted_num_detections)), sorted_num_detections, color='salmon', edgecolor='black')
        ax2.set_title("Number of Detections Per Image")
        ax2.set_xlabel("Image")
        ax2.set_ylabel("Number of Detections")

        # Adjust layout and render
        self.canvas.figure.tight_layout()
        self.canvas.draw()

        self.update_metrics_labels(metrics_mean_std)

    def update_metrics_labels(self, metrics_mean_std):
        self.confidence_mean_mean_label.setText(f"Average Confidence Score: {metrics_mean_std['confidence_mean_mean']:.2f}")
        self.confidence_mean_std_label.setText(f"Confidence Score Variability: {metrics_mean_std['confidence_mean_std']:.2f}")
        self.confidence_std_mean_label.setText(f"Average Confidence Spread: {metrics_mean_std['confidence_std_mean']:.2f}")
        self.confidence_std_std_label.setText(f"Confidence Spread Variability: {metrics_mean_std['confidence_std_std']:.2f}")
        self.num_detections_mean_label.setText(f"Average Number of Detections: {metrics_mean_std['num_detections_mean']:.2f}")
        self.num_detections_std_label.setText(f"Detection Count Variability: {metrics_mean_std['num_detections_std']:.2f}")
        self.area_mean_mean_label.setText(f"Average Detection Area: {metrics_mean_std['area_mean_mean']:.2f}")
        self.area_mean_std_label.setText(f"Detection Area Variability: {metrics_mean_std['area_mean_std']:.2f}")
        self.area_std_mean_label.setText(f"Average Area Spread: {metrics_mean_std['area_std_mean']:.2f}")
        self.area_std_std_label.setText(f"Area Spread Variability: {metrics_mean_std['area_std_std']:.2f}")
        self.overlap_ratio_mean_label.setText(f"Average Overlap Ratio: {metrics_mean_std['overlap_ratio_mean']:.2f}")
        self.overlap_ratio_std_label.setText(f"Overlap Ratio Variability: {metrics_mean_std['overlap_ratio_std']:.2f}")

    def update(self):
        self.model_selector.clear()
        self.model_selector.addItems(map(lambda model: model['model_name'], self.db.load_models()))

        self.dataset_selector.clear()
        self.dataset_selector.addItems(map(lambda dataset: dataset['dataset_name'], self.db.load_datasets()))

       


class AnalysisTab(QWidget):
    def __init__(self, db):
        super().__init__()
        self.db = db
        layout = QVBoxLayout()
        metrics_layout = QGridLayout()
        self.uploaded_files = []
        
        # Model selection
        self.model_selector = QComboBox()
        self.model_selector.addItems(map(lambda model: model['model_name'], self.db.load_models()))
        
        # Image upload/selection
        self.select_btn = QPushButton("Select Images")
        self.inference_btn = QPushButton("Inference Images")
        self.save_btn = QPushButton("Save Inferences")  

        self.confidence_mean_mean_label = QLabel("Average Confidence Score: N/A")
        self.confidence_mean_mean_label.setToolTip("""
        Average confidence of predictions per image, averaged across all images.
        """)
        self.confidence_mean_std_label = QLabel("Confidence Score Variability: N/A")
        self.confidence_mean_std_label.setToolTip("""
        Average confidence of predictions per image, standard deviation across all images.
        """)
        self.confidence_std_mean_label = QLabel("Average Confidence Spread: N/A")
        self.confidence_std_mean_label.setToolTip("""
        Standard deviation of confidence of predictions per image, averaged across all images.
        """)
        self.confidence_std_std_label = QLabel("Confidence Spread Variability: N/A")
        self.confidence_std_std_label.setToolTip("""
        Standard deviation of confidence of predictions per image, standard deviation across all images.
        """)
        self.num_detections_mean_label = QLabel("Average Number of Detections: N/A")
        self.num_detections_mean_label.setToolTip("""
        Average number of detections per image.
        """)
        self.num_detections_std_label = QLabel("Detection Count Variability: N/A")
        self.num_detections_std_label.setToolTip("""
        Standard deviation of number of detections per image.
        """)
        self.area_mean_mean_label = QLabel("Average Detection Area: N/A")
        self.area_mean_mean_label.setToolTip("""
        Average area of detections per image, averaged across all images.
        """)
        self.area_mean_std_label = QLabel("Detection Area Variability: N/A")
        self.area_mean_std_label.setToolTip("""
        Average area of detections per image, standard deviation across all images.
        """)
        self.area_std_mean_label = QLabel("Average Area Spread: N/A")
        self.area_std_mean_label.setToolTip("""
        Standard deviation of area of detections per image, averaged across all images.
        """)
        self.area_std_std_label = QLabel("Area Spread Variability: N/A")
        self.area_std_std_label.setToolTip("""
        Standard deviation of area of detections per image, standard deviation across all images.
        """)
        self.overlap_ratio_mean_label = QLabel("Average Overlap Rati: N/A")
        self.overlap_ratio_mean_label.setToolTip("""
        Average overlap ratio of detections per image.
        """)
        self.overlap_ratio_std_label = QLabel("Overlap Ratio Variability: N/A")
        self.overlap_ratio_std_label.setToolTip("""
        Standard deviation of overlap ratio of detections per image.
        """)

        self.analysis_confidence_mean_mean_label = QLabel("Analysis Average Confidence Score: N/A")
        self.analysis_confidence_mean_mean_label.setToolTip("""
        Average confidence of predictions per image, averaged across all images.
        """)
        self.analysis_confidence_mean_std_label = QLabel("Analysis Confidence Score Variability: N/A")
        self.analysis_confidence_mean_std_label.setToolTip("""
        Average confidence of predictions per image, standard deviation across all images.
        """)
        self.analysis_confidence_std_mean_label = QLabel("Analysis Average Confidence Spread: N/A")
        self.analysis_confidence_std_mean_label.setToolTip("""
        Standard deviation of confidence of predictions per image, averaged across all images.
        """)
        self.analysis_confidence_std_std_label = QLabel("Analysis Confidence Spread Variability: N/A")
        self.analysis_confidence_std_std_label.setToolTip("""
        Standard deviation of confidence of predictions per image, standard deviation across all images.
        """)
        self.analysis_num_detections_mean_label = QLabel("Analysis Average Number of Detections: N/A")
        self.analysis_num_detections_mean_label.setToolTip("""
        Average number of detections per image.
        """)
        self.analysis_num_detections_std_label = QLabel("Analysis Detection Count Variability: N/A")
        self.analysis_num_detections_std_label.setToolTip("""
        Standard deviation of number of detections per image.
        """)
        self.analysis_area_mean_mean_label = QLabel("Analysis Average Detection Area: N/A")
        self.analysis_area_mean_mean_label.setToolTip("""
        Average area of detections per image, averaged across all images.
        """)
        self.analysis_area_mean_std_label = QLabel("Analysis Detection Area Variability: N/A")
        self.analysis_area_mean_std_label.setToolTip("""
        Average area of detections per image, standard deviation across all images.
        """)
        self.analysis_area_std_mean_label = QLabel("Analysis Average Area Spread: N/A")
        self.analysis_area_std_mean_label.setToolTip("""
        Standard deviation of area of detections per image, averaged across all images.
        """)
        self.analysis_area_std_std_label = QLabel("Analysis Area Spread Variability: N/A")
        self.analysis_area_std_std_label.setToolTip("""
        Standard deviation of area of detections per image, standard deviation across all images.
        """)
        self.analysis_overlap_ratio_mean_label = QLabel("Analysis Average Overlap Rati: N/A")
        self.analysis_overlap_ratio_mean_label.setToolTip("""
        Average overlap ratio of detections per image.
        """)
        self.analysis_overlap_ratio_std_label = QLabel("Analysis Overlap Ratio Variability: N/A")
        self.analysis_overlap_ratio_std_label.setToolTip("""
        Standard deviation of overlap ratio of detections per image.
        """)
        
        self.canvas = FigureCanvas(Figure(figsize=(12, 5)))

        self.select_btn.clicked.connect(self.select_images)
        self.inference_btn.clicked.connect(self.inference_images)
        self.save_btn.clicked.connect(self.save_inferences)
        
        layout.addWidget(QLabel("Trained Model:"))
        layout.addWidget(self.model_selector)
        layout.addWidget(self.select_btn)
        layout.addWidget(self.inference_btn)
        layout.addWidget(self.save_btn)
        layout.addWidget(self.canvas)

        metrics_layout.addWidget(self.confidence_mean_mean_label, 0, 0)
        metrics_layout.addWidget(self.confidence_mean_std_label, 0, 1)
        metrics_layout.addWidget(self.confidence_std_mean_label, 0, 2)
        metrics_layout.addWidget(self.confidence_std_std_label, 0, 3)
        metrics_layout.addWidget(self.num_detections_mean_label, 0, 4)
        metrics_layout.addWidget(self.num_detections_std_label, 0, 5)

        metrics_layout.addWidget(self.analysis_confidence_mean_mean_label, 1, 0)
        metrics_layout.addWidget(self.analysis_confidence_mean_std_label, 1, 1)
        metrics_layout.addWidget(self.analysis_confidence_std_mean_label, 1, 2)
        metrics_layout.addWidget(self.analysis_confidence_std_std_label, 1, 3)
        metrics_layout.addWidget(self.analysis_num_detections_mean_label, 1, 4)
        metrics_layout.addWidget(self.analysis_num_detections_std_label, 1, 5)

        metrics_layout.addWidget(self.area_mean_mean_label, 2, 0)
        metrics_layout.addWidget(self.area_mean_std_label, 2, 1)
        metrics_layout.addWidget(self.area_std_mean_label, 2, 2)
        metrics_layout.addWidget(self.area_std_std_label, 2, 3)
        metrics_layout.addWidget(self.overlap_ratio_mean_label, 2, 4)
        metrics_layout.addWidget(self.overlap_ratio_std_label, 2, 5)

        metrics_layout.addWidget(self.analysis_area_mean_mean_label, 3, 0)
        metrics_layout.addWidget(self.analysis_area_mean_std_label, 3, 1)
        metrics_layout.addWidget(self.analysis_area_std_mean_label, 3, 2)
        metrics_layout.addWidget(self.analysis_area_std_std_label, 3, 3)
        metrics_layout.addWidget(self.analysis_overlap_ratio_mean_label, 3, 4)
        metrics_layout.addWidget(self.analysis_overlap_ratio_std_label, 3, 5)

        layout.addLayout(metrics_layout)
        self.setLayout(layout)

    # TODO: select multiple models and compare results / ensemble ?
    def select_images(self):
        self.uploaded_files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png)")

    def inference_images(self):
        from ultralytics import YOLO
        self.model_name = self.model_selector.currentText()
        self.model_path = self.db.model_table.get(Query().model_name == model_name).get('model_path')
        self.model = YOLO(self.model_path)
        self.inference_dir = os.path.join('data/datasets/dataset_0/results/testModel', 'inference')
        os.makedirs(self.inference_dir, exist_ok=True)
        if not self.uploaded_files:
            print("No images selected")
            return  
        self.inference_result = self.model.predict(self.uploaded_files, conf=0.3, visualize=False, save=False, show_labels=False, max_det=1000, verbose=False)
        if True:
            self.save_inferences()
        if True:
            self.plot_inferences_against_dataset()

    def save_inferences(self):
        for file, result in zip(self.uploaded_files, self.inference_result):
                masks = result.masks
                mask_num = len(masks)
                print(file, '------------')
                save_path = os.path.join(self.inference_dir, f'{os.path.splitext(os.path.basename(file))[0]}_{mask_num}.png')
                mask_image = result.plot(labels=False, conf=False, boxes=False)
                mask_image = Image.fromarray(mask_image)
                print(save_path)
                mask_image.save(save_path)

    def plot_inferences_against_dataset(self):
        self.canvas.figure.clf()
        ax1, ax2 = self.canvas.figure.subplots(1, 2)

        # Sort by num_detections and apply the same order to confidence_mean
        sorted_indices = sorted(range(len(self.evaluation_tab.metrics.dataset_metrics["num_detections"])), key=lambda i: self.evaluation_tab.metrics.dataset_metrics["num_detections"][i])
        
        sorted_num_detections = [self.evaluation_tab.metrics.dataset_metrics["num_detections"][i] for i in sorted_indices]
        sorted_conf_mean = [self.evaluation_tab.metrics.dataset_metrics["confidence_mean"][i] for i in sorted_indices]

        print("sorted_num_detections", sorted_num_detections)
        print("sorted_conf_mean", sorted_conf_mean)

        additional_num_detections, additional_conf_mean = self.format_preds(self.inference_result)
        additional_sorted_indices = sorted(range(len(additional_num_detections)), key=lambda i: additional_num_detections[i], reverse=True)
        
        sorted_additional_num_detections = [additional_num_detections[i] for i in additional_sorted_indices]
        sorted_additiona_conf_mean = [additional_conf_mean[i] for i in additional_sorted_indices]

        print("sorted_additional_num_detections", sorted_additional_num_detections)
        print("sorted_additiona_conf_mean", sorted_additiona_conf_mean)

        merged_additional_indices = []
        for num in sorted_additional_num_detections:
            if num > sorted_num_detections[-1]:
                merged_additional_indices.append(len(sorted_num_detections))
                continue
            for i, sorted_num in enumerate(sorted_num_detections):
                if num <= sorted_num:
                    merged_additional_indices.append(i)
                    break

        print("merged_additional_indices", merged_additional_indices)

        for i, num in enumerate(merged_additional_indices):
            sorted_num_detections.insert(num, sorted_additional_num_detections[i])
            sorted_conf_mean.insert(num, sorted_additiona_conf_mean[i])

        print("merged_sorted_num_detections", sorted_num_detections)
        print("merged_sorted_conf_mean", sorted_conf_mean)

    
        indicies_to_color = [(len(merged_additional_indices) - (index + 1)) + value for index, value in enumerate(merged_additional_indices)]

        print("indicies_to_color", indicies_to_color)

        # Plotting histograms
        colors = ['skyblue' if i in indicies_to_color else 'salmon' for i in range(len(sorted_conf_mean))]

        print("colors", colors)

        ax1.bar(range(len(sorted_conf_mean)), sorted_conf_mean, color=colors, edgecolor='black', label='_nolegend_')
        ax1.bar(0, 0, width=0, color='salmon', edgecolor='black', label='Original Data')
        ax1.bar(0, 0, width=0, color='skyblue', edgecolor='black', label='New Data')
        ax1.set_title("Mean Confidence of Predictions Per Image")
        ax1.set_xlabel("Image")
        ax1.set_ylabel("Mean Confidence")
        ax1.legend()

        ax2.bar(range(len(sorted_num_detections)), sorted_num_detections, color=colors, edgecolor='black', label='_nolegend_')
        ax2.bar(0, 0, width=0, color='salmon', edgecolor='black', label='Original Data')
        ax2.bar(0, 0, width=0, color='skyblue', edgecolor='black', label='New Data')
        ax2.set_title("Number of Detections Per Image")
        ax2.set_xlabel("Image")
        ax2.set_ylabel("Number of Detections")
        ax2.legend()

        # Adjust layout and render
        self.canvas.figure.tight_layout()
        self.canvas.draw()
        # print("plotted")

        self.update_metrics_labels()

        self.update_analysis_metrics_labels()

        # each list is a metric where values in list are individual image values
        # restructure to be a list of list of image metrics each list contains all metrics for a single image
        analysis_list_of_list = self.analysis_metrics.get_analysis_metrics()
        reshaped_analysis_list_of_list = [dict(zip(analysis_list_of_list.keys(), values)) for values in zip(*analysis_list_of_list.values())]

        variance_list_of_list = []
        quality_score_list = []
        for i, image in enumerate(reshaped_analysis_list_of_list):
            print(f"Image {i+1} metrics: {image}")
            print('-'*50)
            variance_list = self.evaluation_tab.metrics.compute_variance(image)
            variance_list_of_list.append(variance_list)
            print(f"Image {i+1} variance: {variance_list}")
            print('-'*50)
            quality_score = self.evaluation_tab.metrics.compute_quality_score(variance_list)
            quality_score_list.append({self.uploaded_files[i]: quality_score})
            print(f"Image {i+1} quality score: {quality_score} from {self.uploaded_files[i]}")
            print('-'*50)
        

    def format_preds(self, predictions):
        print("formatting preds")
        num_detections_list = []
        mean_confidence_list = []
        for pred in predictions:
            conf = pred.boxes.conf
            cell_num = len(conf)
            num_detections_list.append(cell_num)
            mean_confidence_list.append(np.mean(conf.numpy()))

        print("num_detections_list", num_detections_list)
        print("mean_confidence_list", mean_confidence_list)

        return num_detections_list, mean_confidence_list

    def update_metrics_labels(self):
        self.metrics_mean_std = self.evaluation_tab.metrics.dataset_metrics_mean_std

        #TODO: how to know what variance is acceptable, n-fold cross variation as baseline, how to calculate?
        self.confidence_mean_mean_label.setText(f"Average Confidence Score: {self.metrics_mean_std['confidence_mean_mean']:.2f}")
        self.confidence_mean_std_label.setText(f"Confidence Score Variability: {self.metrics_mean_std['confidence_mean_std']:.2f}")
        self.confidence_std_mean_label.setText(f"Average Confidence Spread: {self.metrics_mean_std['confidence_std_mean']:.2f}")
        self.confidence_std_std_label.setText(f"Confidence Spread Variability: {self.metrics_mean_std['confidence_std_std']:.2f}")
        self.num_detections_mean_label.setText(f"Average Number of Detections: {self.metrics_mean_std['num_detections_mean']:.2f}")
        self.num_detections_std_label.setText(f"Detection Count Variability: {self.metrics_mean_std['num_detections_std']:.2f}")
        self.area_mean_mean_label.setText(f"Average Detection Area: {self.metrics_mean_std['area_mean_mean']:.2f}")
        self.area_mean_std_label.setText(f"Detection Area Variability: {self.metrics_mean_std['area_mean_std']:.2f}")
        self.area_std_mean_label.setText(f"Average Area Spread: {self.metrics_mean_std['area_std_mean']:.2f}")
        self.area_std_std_label.setText(f"Area Spread Variability: {self.metrics_mean_std['area_std_std']:.2f}")
        self.overlap_ratio_mean_label.setText(f"Average Overlap Ratio: {self.metrics_mean_std['overlap_ratio_mean']:.2f}")
        self.overlap_ratio_std_label.setText(f"Overlap Ratio Variability: {self.metrics_mean_std['overlap_ratio_std']:.2f}")
    
    def update_analysis_metrics_labels(self):
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy the selected files to the temporary directory
            for file in self.uploaded_files:
                shutil.copy(file, temp_dir)
            
            # Use the temporary directory as the dataset path
            self.analysis_metrics = DetectionQAMetrics(self.evaluation_tab.model_path, temp_dir)

        self.analysis_confidence_mean_mean_label.setText(f"Analysis Average Confidence Score: {self.analysis_metrics.dataset_metrics_mean_std['confidence_mean_mean']:.2f}")
        self.analysis_confidence_mean_std_label.setText(f"Analysis Confidence Score Variability: {self.analysis_metrics.dataset_metrics_mean_std['confidence_mean_std']:.2f}")
        self.analysis_confidence_std_mean_label.setText(f"Analysis Average Confidence Spread: {self.analysis_metrics.dataset_metrics_mean_std['confidence_std_mean']:.2f}")
        self.analysis_confidence_std_std_label.setText(f"Analysis Confidence Spread Variability: {self.analysis_metrics.dataset_metrics_mean_std['confidence_std_std']:.2f}")
        self.analysis_num_detections_mean_label.setText(f"Analysis Average Number of Detections: {self.analysis_metrics.dataset_metrics_mean_std['num_detections_mean']:.2f}")
        self.analysis_num_detections_std_label.setText(f"Analysis Detection Count Variability: {self.analysis_metrics.dataset_metrics_mean_std['num_detections_std']:.2f}")
        self.analysis_area_mean_mean_label.setText(f"Analysis Average Detection Area: {self.analysis_metrics.dataset_metrics_mean_std['area_mean_mean']:.2f}")
        self.analysis_area_mean_std_label.setText(f"Analysis Detection Area Variability: {self.analysis_metrics.dataset_metrics_mean_std['area_mean_std']:.2f}")
        self.analysis_area_std_mean_label.setText(f"Analysis Average Area Spread: {self.analysis_metrics.dataset_metrics_mean_std['area_std_mean']:.2f}")
        self.analysis_area_std_std_label.setText(f"Analysis Area Spread Variability: {self.analysis_metrics.dataset_metrics_mean_std['area_std_std']:.2f}")
        self.analysis_overlap_ratio_mean_label.setText(f"Analysis Average Overlap Ratio: {self.analysis_metrics.dataset_metrics_mean_std['overlap_ratio_mean']:.2f}")
        self.analysis_overlap_ratio_std_label.setText(f"Analysis Overlap Ratio Variability: {self.analysis_metrics.dataset_metrics_mean_std['overlap_ratio_std']:.2f}")

    def update(self):
        pass


class OutlierTab(QWidget):
    def __init__(self, db):
        super().__init__()
        self.db = db
        layout = QVBoxLayout()
        
        
        # Image display
        self.image_display = ImageDisplay(self)
        
        # Outlier controls
        controls_layout = QHBoxLayout()
        self.confirm_btn = QPushButton("Confirm")
        self.relabel_btn = QPushButton("Relabel")
        self.skip_btn = QPushButton("Skip")
        controls_layout.addWidget(self.confirm_btn)
        controls_layout.addWidget(self.relabel_btn)
        controls_layout.addWidget(self.skip_btn)
        
        # Outlier list
        self.outlier_list = QListWidget()
        
        layout.addWidget(self.image_display)
        layout.addLayout(controls_layout)
        layout.addWidget(self.outlier_list)
        self.setLayout(layout)
        
        """
        INTEGRATION POINT:
        1. Implement outlier detection
        2. Handle relabeling process
        3. Update dataset with confirmed/relabeled data
        """
        

    def update(self):
        pass


class ModelZooTab(QWidget):
    def __init__(self, db):
        super().__init__()
        self.db = db
        layout = QVBoxLayout()
        image_layout = QHBoxLayout()
        
        self.current_index = 0
        # Model selection
        self.model_selector = QComboBox()
        self.model_selector.addItems(map(lambda model: model['model_name'], self.db.load_models()))

        # image display
        self.left_image = ImageDisplay(self)
        # pred display
        self.right_image = ImageDisplay(self)
        
        # Image selection for inference
        self.select_btn = QPushButton("Select Images")
        
        # Run inference button
        self.inference_btn = QPushButton("Inference Images")
        
        # Save inferences button
        self.save_btn = QPushButton("Save Inferences")

        self.next_btn = QPushButton("Next Image")
        
        layout.addWidget(QLabel("Trained Model:"))
        layout.addWidget(self.model_selector)
        layout.addWidget(self.select_btn)
        layout.addWidget(self.inference_btn)
        layout.addWidget(self.save_btn)

        image_layout.addWidget(self.left_image)
        image_layout.addWidget(self.right_image)

        layout.addWidget(self.next_btn)
        layout.addLayout(image_layout)
        self.setLayout(layout)
        
        self.select_btn.clicked.connect(self.select_images)
        self.inference_btn.clicked.connect(self.inference_images)
        self.save_btn.clicked.connect(self.save_inferences)
        self.next_btn.clicked.connect(self.display_images)
        
    def select_images(self):
        self.uploaded_files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png)")

    def inference_images(self):
        from ultralytics import YOLO
        self.model_name = self.model_selector.currentText()
        self.model_path = self.db.model_table.get(Query().model_name == model_name).get('model_path')
        self.model = YOLO(self.model_path)
        self.inference_dir = os.path.join('data', 'data_inference')
        os.makedirs(self.inference_dir, exist_ok=True)
        if not self.uploaded_files:
            print("No images selected")
            return  
        self.inference_result = self.model.predict(self.uploaded_files, conf=0.3, visualize=False, save=False, show_labels=False, max_det=1000, verbose=False)

        self.display_images()

    def display_images(self):
        if len(self.inference_result) > 0 and len(self.uploaded_files) > 0:
            
            self.left_image._display_image(self.uploaded_files[self.current_index], self.current_index + 1, len(self.uploaded_files))
            self._show_pred()

            self.current_index = (self.current_index + 1) % len(self.uploaded_files)
            

    def save_inferences(self):
        for file, result in zip(self.uploaded_files, self.inference_result):
                masks = result.masks
                mask_num = len(masks)
                print(file, '------------')
                default_file_name = f"{os.path.splitext(os.path.basename(file))[0]}_{mask_num}.png"
                save_path, _ = QFileDialog.getSaveFileName(self, "Save Inference", default_file_name, "Images (*.png)")
                if not save_path:
                    continue
                mask_image = result.plot(labels=False, conf=False, boxes=False)
                mask_image = Image.fromarray(mask_image)
                print(save_path)
                mask_image.save(save_path)

    def _show_pred(self):
        result = self.inference_result[self.current_index]
        mask_image = result.plot(labels=False, conf=False, boxes=False)
        mask_image = Image.fromarray(mask_image)
        temp_image_path = os.path.join(tempfile.gettempdir(), "temp_mask_image.png")
        mask_image.save(temp_image_path, format='PNG')
        self.right_image._display_image(temp_image_path, self.current_index + 1, len(self.inference_result))

    def update(self):
        pass


class DataManager:
    def __init__(self, db_path='data/db.json'):
        self.db = TinyDB(db_path)
        self.image_table = self.db.table('images')
        self.dataset_table = self.db.table('datasets')
        self.model_table = self.db.table('models')

        self.model_table.insert({
            "model_name": "DeepNeuronSegBaseModel",
            "model_path": os.path.abspath("models/yolov8n-largedata-70-best.pt")
        })

    def load_images(self):
        images = self.image_table.all()

        uploaded_files = [image['file_path'] for image in images if 'labels' in image and 'file_path' in image]
        if not uploaded_files:
            print("No images found")
        else:
            return uploaded_files

    def load_labels(self):
        images = self.image_table.all()

        labels = [image['labels'] for image in images if 'labels' in image]
        if not labels:
            print("No labels found")
        else:
            return labels

    def load_masks(self):
        items = self.image_table.all()

        masks = [item['mask_data']['mask_path'] for item in items if 'mask_data' in item]
        if not masks:
            print("No masks found")
        else:
            return masks

    def load_datasets(self):
        return self.dataset_table.all()

    def load_models(self):
        return self.model_table.all()




class MainWindow(QMainWindow):
    #TODO: make shared data structure across main window instead of nesting tabs
    def __init__(self):
        super().__init__()

        self.setup_data_dir()

        self.data_manager = DataManager()

        self.setWindowTitle("DeepNeuronSeg")
        self.setMinimumSize(1024, 768)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setMovable(False)
        
        self.shared_data = {}
        
        # Create and add all self.tabs
        self.tabs.addTab(UploadTab(self.data_manager), "Upload Data")
        self.tabs.addTab(LabelingTab(self.data_manager), "Label Data")
        self.tabs.addTab(GenerateLabelsTab(self.data_manager), "Generate Labels")
        self.tabs.addTab(DatasetTab(self.data_manager), "Create Dataset")
        self.tabs.addTab(TrainingTab(self.data_manager), "Train Network")
        self.tabs.addTab(EvaluationTab(self.data_manager), "Evaluate Network")
        self.tabs.addTab(AnalysisTab(self.data_manager), "Analyze Data")
        self.tabs.addTab(OutlierTab(self.data_manager), "Extract Outliers")
        self.tabs.addTab(ModelZooTab(self.data_manager), "Model Zoo")
        
        layout.addWidget(self.tabs)
        self.tabs.currentChanged.connect(self.update_current_tab)

    def update_current_tab(self, index):
        current_tab = self.tabs.widget(index)
        if hasattr(current_tab, 'update') and callable(getattr(current_tab, 'update')):
            current_tab.update()

    def setup_data_dir(self):
        os.makedirs(os.path.join("data", "data_images"), exist_ok=True)



def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()