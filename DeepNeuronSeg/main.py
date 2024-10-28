from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, 
                           QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, 
                           QPushButton, QFileDialog, QSpinBox, QComboBox,
                           QProgressBar, QListWidget, QDoubleSpinBox, 
                           QCheckBox, QLineEdit, QGridLayout)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
import sys
from dataclasses import dataclass
from typing import List, Dict, Any
import json
import os


@dataclass
class ImageMetadata:
    filepath: str
    experiment_id: str
    brain_region: str
    additional_metadata: Dict[str, Any]

class ImageLabel(QLabel):
    """Custom QLabel to handle mouse clicks on the image area only."""
    click_registered = pyqtSignal(QPoint)
    
    def __init__(self):
        super().__init__()
        self.displayed_pixmap = None

    def set_pixmap_with_scaling(self, pixmap, parent_size):
        # Scale and set pixmap, keeping track of scaled size
        self.displayed_pixmap = pixmap.scaled(parent_size, aspectRatioMode=Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)
        self.setPixmap(self.displayed_pixmap)

    def mousePressEvent(self, event):
        # Check if click is within displayed image bounds
        print("mouse click")
        if self.displayed_pixmap and self.geometry().contains(event.pos()):
            pixmap_rect = self.geometry().center() - QPoint(self.displayed_pixmap.width() // 2, self.displayed_pixmap.height() // 2)
            image_rect = QRect(pixmap_rect, self.displayed_pixmap.size())
            if image_rect.contains(event.pos()):
                self.click_registered.emit(event.pos() - pixmap_rect)  # Emit position relative to image


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
            self.image_label.set_pixmap_with_scaling(self.pixmap, self.size())
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
            with open(self.data_file, 'r') as f:
                existing_metadata = json.load(f)
                # print(existing_metadata)
            existing_metadata.extend(new_metadata)
            metadata = existing_metadata
        else:
            metadata = new_metadata
        
        with open(self.data_file, 'w') as f:
            json.dump(metadata, f)

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
        
        # TODO: load from json first
        self.cell_positions = []

       
    def load_data(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                    data = json.load(f)

            self.uploaded_files = [image["file_path"] for image in data if "file_path" in image]

            if self.uploaded_files:
                self.current_index = 0
                self.image_display.display_image(self.uploaded_files[self.current_index], self.current_index + 1, len(self.uploaded_files))
        else:
            print("No data loaded, please upload data first")


    def show_next_image(self):
        """Display the next image in the list."""
        if self.uploaded_files:
            self.current_index = (self.current_index + 1) % len(self.uploaded_files)  # Wrap around
            self.image_display.display_image(self.uploaded_files[self.current_index], self.current_index + 1, len(self.uploaded_files))

    
    def add_cell_marker(self, pos):
        """
        INTEGRATION POINT:
        1. Store click coordinates
        2. Update image overlay with markers
        3. Save labeled image and coordinates
        """
        print("adding cell")
        self.cell_positions.append(pos)
        self.update_display()
    
    def update_display(self):
        """Update image with cell markers"""
        # TODO: Implement overlay drawing
        pixmap = self.image_display.pixmap
        painter = QPainter(pixmap)
        painter.setPen(QColor("red"))

        for pos in self.cell_positions:
            painter.drawEllipse(pos, 1, 1)

        self.image_display.image_label.setPixmap(pixmap)

        print(self.cell_positions)


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
        pass


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