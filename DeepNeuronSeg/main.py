from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, 
                           QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, 
                           QPushButton, QFileDialog, QSpinBox, QComboBox,
                           QProgressBar, QListWidget, QDoubleSpinBox, 
                           QCheckBox, QLineEdit, QGridLayout, QProgressBar)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRect, QPointF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen 
import sys
from dataclasses import dataclass
from typing import List, Dict, Any
import json
import os

from utils import get_data, set_data
from inference import segment


# @dataclass
# class ImageMetadata:
#     filepath: str
#     experiment_id: str
#     brain_region: str
#     additional_metadata: Dict[str, Any]

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
    
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.image_label = ImageLabel()
        self.text_label = QLabel()
        
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.text_label)

        self.image_label.setMinimumSize(400, 400)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.text_label.setAlignment(Qt.AlignBottom | Qt.AlignCenter)

    def display_image(self, image_path, image_num, total_images):
        """Load and display an image from the given file path and show image number."""
        self.pixmap = QPixmap(image_path)
        if not self.pixmap.isNull():
            self.image_label.set_pixmap(self.pixmap)
            # self.image_label.setPixmap(self.pixmap.scaled(self.size(), aspectRatioMode=Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation))
            self.text_label.setText(f"{image_num} / {total_images}")
        else:
            print("Failed to load image")


class UploadTab(QWidget):
    def __init__(self):
        super().__init__()
        self.data_file = 'image_metadata.json'
        self.uploaded_files = []
        layout = QVBoxLayout()
        
        # File selection
        self.upload_btn = QPushButton("Upload Images")
        self.next_btn = QPushButton("Next Image")

        self.upload_btn.clicked.connect(self.upload_images)
        self.next_btn.clicked.connect(self.show_next_image)
        
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
        
        # Image preview
        self.image_display = ImageDisplay()
        
        # File list
        self.file_list = QListWidget()
        
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
            # TODO: check if file is already in metadata
            new_metadata.append({
                "file_path": file,
                "project": self.project.text(),
                "cohort": self.cohort.text(),
                "brain_region": self.brain_region.text(),
                "image_id": self.image_id.text(),
                "labels": []
            })

        if os.path.exists(self.data_file):
            existing_metadata = get_data(self.data_file)
            existing_metadata.extend(new_metadata)
            metadata = existing_metadata
        else:
            metadata = new_metadata

        set_data(self.data_file, metadata)

        if self.uploaded_files:
            self.current_index = 0
            self.image_display.display_image(self.uploaded_files[self.current_index], self.current_index + 1, len(self.uploaded_files))
    
    def show_next_image(self):
        """Display the next image in the list."""
        if self.uploaded_files:
            self.current_index = (self.current_index + 1) % len(self.uploaded_files)  # Wrap around
            self.image_display.display_image(self.uploaded_files[self.current_index], self.current_index + 1, len(self.uploaded_files))
    
    def update_file_list(self):
        self.file_list.clear()
        self.file_list.addItems(self.selected_files)


class LabelingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.data_file = 'image_metadata.json'
        layout = QVBoxLayout()
        self.load_btn = QPushButton("Load Data")
        self.next_btn = QPushButton("Next Image")
        self.next_btn.clicked.connect(self.show_next_image)
        self.load_btn.clicked.connect(self.load_data)
    
        # Image display with cell marking
        self.image_display = ImageDisplay()

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
        if os.path.exists(self.data_file):
            self.data = get_data(self.data_file)
            result = [(image["file_path"], image["labels"]) for image in self.data if "file_path" in image]
            self.uploaded_files, self.labels = zip(*result)

            if self.uploaded_files:
                self.current_index = 0
                self.image_display.display_image(self.uploaded_files[self.current_index], self.current_index + 1, len(self.uploaded_files))
                self.image_display.image_label.draw_points(self.labels[self.current_index])
        else:
            print("No data loaded, please upload data first")


    def show_next_image(self):
        """Display the next image in the list."""
        if self.uploaded_files:
            self.current_index = (self.current_index + 1) % len(self.uploaded_files)  # Wrap around
            self.image_display.display_image(self.uploaded_files[self.current_index], self.current_index + 1, len(self.uploaded_files))
            self.image_display.image_label.draw_points(self.labels[self.current_index])

    
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
            
        set_data(self.data_file, self.data)


class GenerateLabelsTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()
        config_layout = QGridLayout()

        self.progress = QProgressBar(self)
        self.progress.setAlignment(Qt.AlignCenter)

        self.left_image = ImageDisplay()
        self.right_image = ImageDisplay()

        self.generate_btn = QPushButton("Generate Labels")
        self.next_btn = QPushButton("Next Image")
        self.generate_btn.clicked.connect(self.display_labels)
        self.next_btn.clicked.connect(self.show_next_image)

        
        config_layout.addWidget(self.generate_btn)
        config_layout.addWidget(self.next_btn)

        layout.addLayout(config_layout)
        layout.addWidget(self.progress)
        layout.addWidget(self.left_image)
        layout.addWidget(self.right_image)

        self.setLayout(layout)

    def display_labels(self):
        self.data = get_data()
        result = [(image["file_path"], image["labels"]) for image in self.data if "file_path" in image]
        self.uploaded_files, self.labels = zip(*result)
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
            generated_label = self.generate_label(uploaded_file, label)
            # save somewhere somehow in relation to uploaded_file
        
        # display image label pairs with button to see next pair
        self.current_index = 0
        self.left_image.display_image(self.uploaded_files[self.current_index], self.current_index + 1, len(self.uploaded_files))
        self.right_image.display_image(self.uploaded_files[self.current_index], self.current_index + 1, len(self.uploaded_files))
        

        # allow user editing of generated labels

    def generate_label(self, image_path, labels):
        """
        INTEGRATION POINT:
        1. Implement label generation
        2. Display generated labels
        3. Save generated labels
        """
        masks, scores = segment(image_path, labels)
        final_image, num_masks, instances_list = composite_mask(masks)
        return final_image

    def show_next_image(self):
        """Display the next image in the list."""
        if self.uploaded_files:
            self.current_index = (self.current_index + 1) % len(self.uploaded_files)  # Wrap around
            self.left_image.display_image(self.uploaded_files[self.current_index], self.current_index + 1, len(self.uploaded_files))
            self.right_image.display_image(self.uploaded_files[self.current_index], self.current_index + 1, len(self.uploaded_files))
    




class DatasetTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        
        # Dataset configuration
        config_layout = QGridLayout()
        self.train_split = QDoubleSpinBox()
        self.train_split.setRange(0.1, 0.9)
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
        


class TrainingTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        
        # Model selection
        self.model_selector = QComboBox()
        # TODO: Populate with available models
        
        # Training parameters
        params_layout = QGridLayout()
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 128)
        self.model_name = QLineEdit()
        
        params_layout.addWidget(QLabel("Epochs:"), 0, 0)
        params_layout.addWidget(self.epochs, 0, 1)
        params_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        params_layout.addWidget(self.batch_size, 1, 1)
        params_layout.addWidget(QLabel("Model Name:"), 2, 0)
        params_layout.addWidget(self.model_name, 2, 1)
        
        # Progress tracking
        self.progress = QProgressBar()
        
        # Control buttons
        self.train_btn = QPushButton("Start Training")
        self.stop_btn = QPushButton("Stop")
        
        layout.addWidget(self.model_selector)
        layout.addLayout(params_layout)
        layout.addWidget(self.progress)
        layout.addWidget(self.train_btn)
        layout.addWidget(self.stop_btn)
        self.setLayout(layout)


class EvaluationTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        
        # Model selection
        self.model_selector = QComboBox()
        
        # Visualization area (placeholder for distribution plots)
        self.plot_area = QLabel("Distribution Plot Will Appear Here")
        self.plot_area.setMinimumSize(400, 300)
        
        # Statistics display
        self.stats_display = QLabel()
        
        layout.addWidget(self.model_selector)
        layout.addWidget(self.plot_area)
        layout.addWidget(self.stats_display)
        self.setLayout(layout)
        
        """
        INTEGRATION POINT:
        1. Implement distribution plotting
        2. Calculate and display statistics
        3. Load and compare model predictions
        """


class AnalysisTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        
        # Model selection
        self.model_selector = QComboBox()
        
        # Image upload/selection
        self.select_btn = QPushButton("Select Images")
        
        # Results display
        self.results_list = QListWidget()
        
        # Save button
        self.save_btn = QPushButton("Save Results")
        
        layout.addWidget(self.model_selector)
        layout.addWidget(self.select_btn)
        layout.addWidget(self.results_list)
        layout.addWidget(self.save_btn)
        self.setLayout(layout)


class OutlierTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        
        # Image display
        self.image_display = ImageDisplay()
        
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
        tabs.addTab(UploadTab(), "Upload Data")
        tabs.addTab(LabelingTab(), "Label Data")
        tabs.addTab(GenerateLabelsTab(), "Generate Labels")
        tabs.addTab(DatasetTab(), "Create Dataset")
        tabs.addTab(TrainingTab(), "Train Network")
        tabs.addTab(EvaluationTab(), "Evaluate Network")
        tabs.addTab(AnalysisTab(), "Analyze Data")
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