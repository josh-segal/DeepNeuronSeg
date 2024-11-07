from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, 
                           QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, 
                           QPushButton, QFileDialog, QSpinBox, QComboBox,
                           QProgressBar, QListWidget, QDoubleSpinBox, 
                           QCheckBox, QLineEdit, QGridLayout, QProgressBar)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRect, QPointF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys
from dataclasses import dataclass
from typing import List, Dict, Any
import json
import os
import random
import shutil
from PIL import Image


from utils import get_data, set_data, save_label, get_image_mask_label_tuples, create_yaml
from inference import segment, composite_mask, mask_to_bboxes, mask_to_polygons
from qa import DetectionQAMetrics

class ImageLabel(QLabel):
    """Custom QLabel to handle mouse clicks on the image area only."""
    click_registered = pyqtSignal(QPointF)
    
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
            self.click_registered.emit(click_pos)

    def adjust_pos(self, pos):
        """Adjust the position to the image coordinates."""
        adjusted_x = pos.x() - (self.width() - self.pixmap.width()) / 2
        adjusted_pos = QPointF(adjusted_x, pos.y())
        return adjusted_pos

    def draw_points(self, labels):
        """Draw a point on the image at the given position."""
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

        self.image_label.setMinimumSize(400, 400)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.text_label.setAlignment(Qt.AlignBottom | Qt.AlignCenter)

    def display_image(self, image_path, image_num, total_images):
        """Load and display an image from the given file path and show image number."""
        print(image_path, '<----------------------')
        self.pixmap = QPixmap(image_path)
        if not self.pixmap.isNull():
            self.image_label.set_pixmap(self.pixmap)
            # self.image_label.setPixmap(self.pixmap.scaled(self.size(), aspectRatioMode=Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation))
            self.text_label.setText(f"{image_num} / {total_images}")
        else:
            print("Failed to load image")

    def show_image(self):
        """Display the next image in the list."""
        if self.upload_tab.uploaded_files:
            self.display_image(self.upload_tab.uploaded_files[self.upload_tab.current_index], self.upload_tab.current_index + 1, len(self.upload_tab.uploaded_files))

    def show_mask(self):
        """Display the next image in the list."""
        if self.upload_tab.uploaded_labels:
            self.display_image(self.upload_tab.uploaded_labels[self.upload_tab.current_index], self.upload_tab.current_index + 1, len(self.upload_tab.uploaded_labels))

    def show_next_image(self):
        """Display the next image in the list."""
        self.upload_tab.current_index = (self.upload_tab.current_index + 1) % len(self.upload_tab.uploaded_files)  # Wrap around
        self.show_image()

    def show_next_mask(self):
        """Display the next image in the list."""
        self.upload_tab.current_index = (self.upload_tab.current_index + 1) % len(self.upload_tab.uploaded_labels)  # Wrap around
        self.show_mask()

    def show_image_with_points(self):
        """Display the next image in the list."""
        self.show_image()
        self.image_label.draw_points(self.upload_tab.labels[self.upload_tab.current_index])


    def show_next_image_with_points(self):
        """Display the next image in the list."""
        self.show_next_image()
        self.image_label.draw_points(self.upload_tab.labels[self.upload_tab.current_index])

    

class UploadTab(QWidget):
    def __init__(self):
        super().__init__()
        self.data_file = 'image_metadata.json'
        self.current_index = 0
        self.uploaded_files = []
        layout = QVBoxLayout()

         # Image preview
        self.image_display = ImageDisplay(self)
        
        # File list
        self.file_list = QListWidget()
        
        # File selection
        self.upload_btn = QPushButton("Upload Images")
        self.next_btn = QPushButton("Next Image")

        self.upload_btn.clicked.connect(self.upload_images)
        self.next_btn.clicked.connect(self.image_display.show_next_image)
        
        # Metadata input fields
        metadata_layout = QGridLayout()
        self.project = QLineEdit()
        self.cohort = QLineEdit()
        self.brain_region = QLineEdit()
        self.image_id = QLineEdit()
        metadata_layout.addWidget(QLabel("Project:"), 0, 0)
        metadata_layout.addWidget(self.project, 0, 1)
        metadata_layout.addWidget(QLabel("Cohort:"), 1, 0)
        metadata_layout.addWidget(self.cohort, 1, 1)
        metadata_layout.addWidget(QLabel("Brain Region:"), 2, 0)
        metadata_layout.addWidget(self.brain_region, 2, 1)
        metadata_layout.addWidget(QLabel("Image ID:"), 3, 0)
        metadata_layout.addWidget(self.image_id, 3, 1)
        
       
        
        layout.addWidget(self.upload_btn)
        layout.addWidget(self.next_btn)
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
        self.uploaded_files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png)")
        new_metadata = []
        for file in self.uploaded_files:
            data_subdir = 'data_images'
            data_dir = os.path.join('data', data_subdir)
            os.makedirs(data_dir, exist_ok=True)
            image_name = os.path.basename(file)
            
            image_path = os.path.join(data_dir, image_name)

            shutil.copy(file, image_path)
            # TODO: check if file is already in metadata
            new_metadata.append({
                "file_path": image_path,
                "project": self.project.text(),
                "cohort": self.cohort.text(),
                "brain_region": self.brain_region.text(),
                "image_id": self.image_id.text(),
                "labels": []
            })
        data_path = os.path.join('data', self.data_file)
        if os.path.exists(data_path):
            existing_metadata = get_data()
            existing_metadata.extend(new_metadata)
            metadata = existing_metadata
        else:
            metadata = new_metadata

        set_data(metadata=metadata)

        self.image_display.show_image()
    
    def update_file_list(self):
        self.file_list.clear()
        self.file_list.addItems(self.selected_files)


class LabelingTab(QWidget):
    def __init__(self):
        super().__init__()
        # Image display with cell marking
        self.current_index = 0
        self.image_display = ImageDisplay(self)
        self.data_file = 'image_metadata.json'
        layout = QVBoxLayout()
        self.load_btn = QPushButton("Load Data")
        self.next_btn = QPushButton("Next Image")
        self.next_btn.clicked.connect(self.image_display.show_next_image_with_points)
        self.load_btn.clicked.connect(self.load_data)
    
        

        self.image_display.image_label.click_registered.connect(self.add_cell_marker)
        
        # Controls
        controls_layout = QHBoxLayout()
        self.undo_btn = QPushButton("Undo Last")
        self.clear_btn = QPushButton("Clear All")
        self.save_btn = QPushButton("Save Labels")
        controls_layout.addWidget(self.undo_btn)
        controls_layout.addWidget(self.clear_btn)
        controls_layout.addWidget(self.save_btn)
        
        layout.addWidget(self.image_display)
        layout.addWidget(self.next_btn)
        layout.addWidget(self.load_btn)
        layout.addLayout(controls_layout)
        self.setLayout(layout)
       
    def load_data(self):
        self.data_path = os.path.join('data', self.data_file)
        if os.path.exists(self.data_path):
            self.data = get_data()
            result = [(image["file_path"], image["labels"]) for image in self.data if "file_path" in image]
            self.uploaded_files, self.labels = zip(*result)

            self.image_display.show_image_with_points()

    def add_cell_marker(self, pos):
        """
        INTEGRATION POINT:
        1. Store click coordinates
        2. Update image overlay with markers
        3. Save labeled image and coordinates
        """
        # print("adding cell")
        # TODO: change to store in files labels
        adjusted_pos = self.image_display.image_label.adjust_pos(pos)
        if adjusted_pos.x() < 0 or adjusted_pos.y() < 0 or adjusted_pos.x() > 512 or adjusted_pos.y() > 512:
            return
        self.labels[self.current_index].append((adjusted_pos.x(), adjusted_pos.y()))
        self.image_display.image_label.draw_points(self.labels[self.current_index])

        for image in self.data:
            if image["file_path"] == self.uploaded_files[self.current_index]:
                image["labels"] = [(pos[0], pos[1]) for pos in self.labels[self.current_index]]
                break
        #TODO: change to .h5 for saving??
        set_data(metadata=self.data)


class GenerateLabelsTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()
        config_layout = QGridLayout()

        self.current_index = 0

        self.progress = QProgressBar(self)
        self.progress.setAlignment(Qt.AlignCenter)

        self.left_image = ImageDisplay(self)
        self.right_image = ImageDisplay(self)

        self.generate_btn = QPushButton("Generate Labels")
        self.next_btn = QPushButton("Next Image")
        self.display_btn = QPushButton("Display Labels")
        self.generate_btn.clicked.connect(self.generate_labels)
        self.next_btn.clicked.connect(self.left_image.show_next_image)
        self.next_btn.clicked.connect(self.right_image.show_mask)
        self.display_btn.clicked.connect(self.display_labels)

        
        config_layout.addWidget(self.generate_btn)
        config_layout.addWidget(self.next_btn)
        config_layout.addWidget(self.display_btn)

        layout.addLayout(config_layout)
        layout.addWidget(self.progress)
        layout.addWidget(self.left_image)
        layout.addWidget(self.right_image)

        self.setLayout(layout)

        self.data = get_data()
        result = [(image["file_path"], image["labels"]) for image in self.data if "file_path" in image]
        self.uploaded_files, self.labels = zip(*result)

    def generate_labels(self):
        # print(result)
        # print(self.uploaded_files)
        # print(self.labels)
        # print(len(result))
        self.progress.setValue(0)
        self.progress.setMaximum(len(result))
        for i, (uploaded_file, label) in enumerate(zip(self.uploaded_files, self.labels)):
            # print(i)
            self.progress.setValue(i+1)
            if label is None:
                print("No labels provided for image", uploaded_file)
                continue

            # add TQDM progress bar before images are shown
            label_data = self.generate_label(uploaded_file, label)
            for image in self.data:
                if image["file_path"] == uploaded_file:
                    image["mask_data"] = label_data
                    break

            # save somewhere somehow in relation to uploaded_file
            set_data(metadata=self.data)
        # display image label pairs with button to see next pair
        self.display_labels()
        # allow user editing of generated labels

    def generate_label(self, image_path, labels):
        """Generate labels for the given image."""
        masks, scores = segment(image_path, labels)
        final_image, num_cells, instances_list = composite_mask(masks)

        final_image_path = save_label(final_image=final_image, image_path=image_path)
        scores_numpy = scores.detach().numpy().tolist()
        # save final_image to labeled_data folder
        # print("final_image_path", final_image_path)
        # print("scores", scores_numpy)
        # print("num_cells", num_cells)
        # print("instances_list", instances_list)
        return {
            "mask_path": final_image_path,
            "scores": scores_numpy,
            "num_cells": num_cells,
            "instances_list": instances_list
        }

    def display_labels(self):
        self.uploaded_labels = [image["mask_data"]["mask_path"] for image in self.data if "file_path" in image]
        self.left_image.display_image(self.uploaded_files[self.current_index], self.current_index + 1, len(self.uploaded_files))
        self.right_image.display_image(self.uploaded_labels[self.current_index], self.current_index + 1, len(self.uploaded_labels))



class DatasetTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        
        # Dataset configuration
        config_layout = QGridLayout()
        self.train_split = QDoubleSpinBox()
        self.train_split.setRange(0.1, 0.9)
        self.train_split.setSingleStep(0.05)
        self.train_split.setValue(0.8)
        
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
        data = get_data()
        uploaded_images, uploaded_masks, uploaded_labels = zip(*[
            (
                image["file_path"], 
                image["mask_data"]["mask_path"], 
                [instance["segmentation"] for instance in image["mask_data"]["instances_list"]]
                ) 
                for image in data if "file_path" in image and "mask_data" in image
                ])
        dataset_parent_dir = os.path.join('data', 'datasets')
        os.makedirs(dataset_parent_dir, exist_ok=True)

        dataset_dir = 'dataset'
        counter = 0
        self.dataset_path = os.path.join(dataset_parent_dir, f"{dataset_dir}_{counter}")
        while os.path.exists(self.dataset_path):
            counter += 1
            self.dataset_path = os.path.join(dataset_parent_dir, f"{dataset_dir}_{counter}")

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

        counter = 0
        shuffle_path = os.path.join(self.dataset_path, f"shuffle_{counter}")
        while os.path.exists(shuffle_path):
            counter += 1
            shuffle_path = os.path.join(self.dataset_path, f"shuffle_{counter}")

        os.makedirs(shuffle_path, exist_ok=False)

        train_images_dir = os.path.join( "train", "images")
        val_images_dir = os.path.join( "val", "images")

        create_yaml(os.path.join(shuffle_path, "data.yaml"), train_images_dir, val_images_dir)

        os.makedirs(os.path.join(shuffle_path, train_images_dir), exist_ok=False)
        os.makedirs(os.path.join(shuffle_path, "train", "masks"), exist_ok=False)
        os.makedirs(os.path.join(shuffle_path, "train", "labels"), exist_ok=False)

        os.makedirs(os.path.join(shuffle_path, val_images_dir), exist_ok=False)
        os.makedirs(os.path.join(shuffle_path, "val", "masks"), exist_ok=False)
        os.makedirs(os.path.join(shuffle_path, "val", "labels"), exist_ok=False)

        for image, mask, label in zip(train_images, train_masks, train_labels):
            shutil.copy(image, os.path.join(shuffle_path, "train", "images", os.path.basename(image)))
            shutil.copy(mask, os.path.join(shuffle_path, "train", "masks", os.path.basename(mask)))
            shutil.copy(label, os.path.join(shuffle_path, "train", "labels", os.path.basename(label)))

        for image, mask, label in zip(val_images, val_masks, val_labels):
            shutil.copy(image, os.path.join(shuffle_path, "val", "images", os.path.basename(image)))
            shutil.copy(mask, os.path.join(shuffle_path, "val", "masks", os.path.basename(mask)))
            shutil.copy(label, os.path.join(shuffle_path, "val", "labels", os.path.basename(label)))


class TrainingTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        
        # Model selection
        self.model = None
        self.model_selector = QComboBox()
        self.model_selector.addItems(["YOLOv8n-seg"])
        
        # Training parameters
        params_layout = QGridLayout()
        self.dataset = QSpinBox()
        self.dataset.setRange(1, 100)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 128)
        self.model_name = QLineEdit()
        self.denoise = QCheckBox("Use Denoising Network")
        
        params_layout.addWidget(QLabel("Dataset:"), 0, 0)
        params_layout.addWidget(self.dataset, 0, 1)
        params_layout.addWidget(QLabel("Epochs:"), 1, 0)
        params_layout.addWidget(self.epochs, 1, 1)
        params_layout.addWidget(QLabel("Batch Size:"), 2, 0)
        params_layout.addWidget(self.batch_size, 2, 1)
        params_layout.addWidget(QLabel("Model Name:"), 3, 0)
        params_layout.addWidget(self.model_name, 3, 1)
        params_layout.addWidget(self.denoise, 4, 1)
        
        # Progress tracking
        self.progress = QProgressBar()
        
        # Control buttons
        self.train_btn = QPushButton("Start Training")
        self.stop_btn = QPushButton("Stop")

        self.train_btn.clicked.connect(self.trainer)
        
        layout.addWidget(self.model_selector)
        layout.addLayout(params_layout)
        layout.addWidget(self.progress)
        layout.addWidget(self.train_btn)
        layout.addWidget(self.stop_btn)
        self.setLayout(layout)

    def trainer(self):
        """
        INTEGRATION POINT:
        1. Load selected model
        2. Train model with selected parameters
        3. Track and display training progress
        """
        if self.model_selector.currentText() == "YOLOv8n-seg":
            from ultralytics import YOLO
            print("Training YOLOv8n-seg")
            self.model = YOLO("models/yolov8n-seg.pt")
            self.model.train(
                data = 'C:/Users/joshua/garnercode/DeepNeuronSeg/DeepNeuronSeg/data/datasets/dataset_0/shuffle_0/data.yaml',
                project = 'data/datasets/dataset_0/shuffle_0/results',
                name = self.model_name.text(),
                epochs = self.epochs.value(),
                patience = 0,
                batch = self.batch_size.value(),
                imgsz = 1024
            )



class EvaluationTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        metrics_layout = QGridLayout()
        
        # Model selection
        self.model_selector = QComboBox()
        self.model_selector.addItems(["trained-models-appear-here"])

        self.dataset_selector = QComboBox()
        self.dataset_selector.addItems(["datasets-appear-here"])
        
        # Visualization area (placeholder for distribution plots)
        self.canvas = FigureCanvas(Figure(figsize=(12, 5)))

        self.calculate_metrics_btn = QPushButton("Calculate Metrics")
        self.calculate_metrics_btn.clicked.connect(self.calculate_metrics)

         # Labels for metrics
        self.confidence_mean_mean_label = QLabel("Confidence Mean Mean: N/A")
        self.confidence_mean_std_label = QLabel("Confidence Mean Std: N/A")
        self.confidence_std_mean_label = QLabel("Confidence Std Mean: N/A")
        self.confidence_std_std_label = QLabel("Confidence Std Std: N/A")
        self.num_detections_mean_label = QLabel("Num Detections Mean: N/A")
        self.num_detections_std_label = QLabel("Num Detections Std: N/A")
        self.area_mean_mean_label = QLabel("Area Mean Mean: N/A")
        self.area_mean_std_label = QLabel("Area Mean Std: N/A")
        self.area_std_mean_label = QLabel("Area Std Mean: N/A")
        self.area_std_std_label = QLabel("Area Std Std: N/A")
        self.overlap_ratio_mean_label = QLabel("Overlap Ratio Mean: N/A")
        self.overlap_ratio_std_label = QLabel("Overlap Ratio Std: N/A")

        
        layout.addWidget(self.model_selector)
        layout.addWidget(self.dataset_selector)
        layout.addWidget(self.canvas)
        layout.addWidget(self.calculate_metrics_btn)

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

        # check if metrics already calculated for model
        # load the dataset images
        # inference trained model on dataset images
        # calculate metrics / distributions across inferences
        # display metrics and distributions in meaningful way
        # in analyze data return quality score of inferenced image

    def calculate_metrics(self):
        # TODO: abstract
        # model_path = self.model_selector.currentText()
        model_path = 'C:/Users/joshua/garnercode/cellCountingModel/notebooks/yolobooks2/large_dataset/results/70_epochs_n_large_data-/weights/best.pt'
        # dataset_path = self.dataset_selector.currentText()
        # dataset_path = 'C:/Users/joshua/garnercode/DeepNeuronSeg/DeepNeuronSeg/data/datasets/dataset_0/images'
        dataset_path = 'C:/Users/joshua/garnercode/cellCountingModel/notebooks/yolobooks2/dataset/COCO_train_X'

        self.metrics = DetectionQAMetrics(model_path, dataset_path)
        print(metrics.dataset_metrics_mean_std)
        self.plot_metrics(metrics.dataset_metrics, metrics.dataset_metrics_mean_std)

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
        self.confidence_mean_mean_label.setText(f"Confidence Mean Mean: {metrics_mean_std['confidence_mean_mean']:.2f}")
        self.confidence_mean_std_label.setText(f"Confidence Mean Std: {metrics_mean_std['confidence_mean_std']:.2f}")
        self.confidence_std_mean_label.setText(f"Confidence Std Mean: {metrics_mean_std['confidence_std_mean']:.2f}")
        self.confidence_std_std_label.setText(f"Confidence Std Std: {metrics_mean_std['confidence_std_std']:.2f}")
        self.num_detections_mean_label.setText(f"Num Detections Mean: {metrics_mean_std['num_detections_mean']:.2f}")
        self.num_detections_std_label.setText(f"Num Detections Std: {metrics_mean_std['num_detections_std']:.2f}")
        self.area_mean_mean_label.setText(f"Area Mean Mean: {metrics_mean_std['area_mean_mean']:.2f}")
        self.area_mean_std_label.setText(f"Area Mean Std: {metrics_mean_std['area_mean_std']:.2f}")
        self.area_std_mean_label.setText(f"Area Std Mean: {metrics_mean_std['area_std_mean']:.2f}")
        self.area_std_std_label.setText(f"Area Std Std: {metrics_mean_std['area_std_std']:.2f}")
        self.overlap_ratio_mean_label.setText(f"Overlap Ratio Mean: {metrics_mean_std['overlap_ratio_mean']:.2f}")
        self.overlap_ratio_std_label.setText(f"Overlap Ratio Std: {metrics_mean_std['overlap_ratio_std']:.2f}")

class AnalysisTab(QWidget):
    def __init__(self, evaulation_tab):
        super().__init__()
        layout = QVBoxLayout()
        self.evaluation_tab = evaulation_tab
        
        # Model selection
        self.model_selector = QComboBox()
        self.model_selector.addItems(["YOLOv8n-seg"])
        
        # Image upload/selection
        self.select_btn = QPushButton("Select Images")
        self.inference_btn = QPushButton("Inference Images")
        self.save_btn = QPushButton("Save Inferences")  
        
        # Results display
        self.results_list = QListWidget()

        self.select_btn.clicked.connect(self.select_images)
        self.inference_btn.clicked.connect(self.inference_images)
        self.save_btn.clicked.connect(self.save_inferences)
        
        layout.addWidget(self.model_selector)
        layout.addWidget(self.select_btn)
        layout.addWidget(self.inference_btn)
        layout.addWidget(self.save_btn)
        layout.addWidget(self.results_list)
        self.setLayout(layout)

    # TODO: select multiple models and compare results / ensemble ?
    def select_images(self):
        self.uploaded_files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png)")

    def inference_images(self):
        from ultralytics import YOLO
        self.model = YOLO('C:/Users/joshua/garnercode/cellCountingModel/notebooks/yolobooks2/large_dataset/results/70_epochs_n_large_data-/weights/best.pt')
        self.inference_dir = os.path.join('data/datasets/dataset_0/shuffle_0/results/testModel', 'inference')
        os.makedirs(self.inference_dir, exist_ok=True)
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
        additional_num_detections, additional_conf_mean = self.format_preds(self.inference_result)

        ax1, ax2 = self.canvas.figure.subplots(1, 2)

        # Sort by num_detections and apply the same order to confidence_mean
        sorted_indices = sorted(range(len(evaluation_tab.metrics.dataset_metrics["num_detections"])), key=lambda i: metrics["num_detections"][i])

        sorted_num_detections = [evaluation_tab.metrics.dataset_metrics["num_detections"][i] for i in sorted_indices]
        sorted_conf_mean = [evaluation_tab.metrics.dataset_metrics["confidence_mean"][i] for i in sorted_indices]

        # Plotting histograms
        ax1.bar(range(len(sorted_conf_mean)), sorted_conf_mean, color='skyblue', edgecolor='black', label='Original')
        ax1.bar(range(len(sorted_conf_mean)), additional_conf_mean, color='gold', edgecolor='black', label='Additional')
        ax1.set_title("Mean Confidence of Predictions Per Image")
        ax1.set_xlabel("Image")
        ax1.set_ylabel("Mean Confidence")
        ax1.legend()

        ax2.bar(range(len(sorted_num_detections)), sorted_num_detections, color='salmon', edgecolor='black', label='Original')
        ax2.bar(range(len(sorted_num_detections)), additional_num_detections, color='lime', edgecolor='black', label='Additional')
        ax2.set_title("Number of Detections Per Image")
        ax2.set_xlabel("Image")
        ax2.set_ylabel("Number of Detections")
        ax2.legend()

        # Adjust layout and render
        self.canvas.figure.tight_layout()
        self.canvas.draw()


    def format_preds(self, predictions):
        num_detections_list = []
        mean_confidence_list = []
        for pred in predictions:
            conf = pred.boxes.conf
            cell_num = len(conf)
            num_detections_list.append(cell_num)
            mean_confidence_list.append(np.mean(confs.numpy()))

        return num_detections_list, mean_confidence_list


    

class OutlierTab(QWidget):
    def __init__(self):
        super().__init__()
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


class ModelZooTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        
        # Model selection
        self.model_list = QListWidget()
        
        # Model info display
        self.model_info = QLabel()
        
        # Image selection for inference
        self.select_images_btn = QPushButton("Select Images")
        
        # Run inference button
        self.run_btn = QPushButton("Run Inference")
        
        layout.addWidget(self.model_list)
        layout.addWidget(self.model_info)
        layout.addWidget(self.select_images_btn)
        layout.addWidget(self.run_btn)
        self.setLayout(layout)
        
        """
        INTEGRATION POINT:
        1. Load available models
        2. Implement inference pipeline
        3. Display and save results
        """


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepNeuronSeg")
        self.setMinimumSize(1024, 768)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create tab widget
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.North)
        tabs.setMovable(True)
        
        # Create and add all tabs
        evaluation_tab = EvaluationTab()

        tabs.addTab(UploadTab(), "Upload Data")
        tabs.addTab(LabelingTab(), "Label Data")
        tabs.addTab(GenerateLabelsTab(), "Generate Labels")
        tabs.addTab(DatasetTab(), "Create Dataset")
        tabs.addTab(TrainingTab(), "Train Network")
        tabs.addTab(evaluation_tab, "Evaluate Network")
        tabs.addTab(AnalysisTab(evaluation_tab), "Analyze Data")
        tabs.addTab(OutlierTab(), "Extract Outliers")
        tabs.addTab(ModelZooTab(), "Model Zoo")
        
        layout.addWidget(tabs)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()