import os
import tempfile
from PIL import Image
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel, QFileDialog
from DeepNeuronSeg.views.widgets.image_display import ImageDisplay

class ModelZooView(QWidget):

    inference_images_signal = pyqtSignal(str, list)
    save_inferences_signal = pyqtSignal()
    update_index_signal = pyqtSignal(int)
    download_data_signal = pyqtSignal()
    update_signal = pyqtSignal()
    next_image_signal = pyqtSignal()
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        image_layout = QHBoxLayout()

        # Model selection
        self.model_selector = QComboBox()

        # image display
        self.left_image = ImageDisplay()
        # pred display
        self.right_image = ImageDisplay()
        
        # Image selection for inference
        self.select_btn = QPushButton("Select Images")
        
        # Run inference button
        self.inference_btn = QPushButton("Inference Images")
        
        # Save inferences button
        self.save_btn = QPushButton("Save Inferences")

        self.next_btn = QPushButton("Next Image")

        self.download_btn = QPushButton("Download Data")
        
        layout.addWidget(QLabel("Trained Model:"))
        layout.addWidget(self.model_selector)
        layout.addWidget(self.select_btn)
        layout.addWidget(self.inference_btn)
        layout.addWidget(self.save_btn)

        image_layout.addWidget(self.left_image)
        image_layout.addWidget(self.right_image)

        layout.addWidget(self.next_btn)
        layout.addLayout(image_layout)
        layout.addWidget(self.download_btn)
        layout.addStretch()
        self.setLayout(layout)
        
        self.select_btn.clicked.connect(self.select_images)
        self.inference_btn.clicked.connect(self.inference_images)
        self.save_btn.clicked.connect(self.save_inferences)
        self.next_btn.clicked.connect(self.next_image)
        self.download_btn.clicked.connect(self.download_data)

    def download_data(self):
        self.download_data_signal.emit()
        
    def select_images(self):
        self.uploaded_files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png)")

    def inference_images(self):
        self.model_name = self.model_selector.currentText() 
        self.inference_images_signal.emit(self.model_name, self.uploaded_files)

    def display_images(self, uploaded_file, inference_result, current_index, total_items):
        self.left_image.display_frame(uploaded_file, current_index + 1, total_items)
        self.right_image.display_frame(inference_result, current_index + 1, total_items, pred=True)

    def save_inferences(self):
        self.save_inferences_signal.emit()
    
    def next_image(self):
        self.next_image_signal.emit()

    def update(self):
        self.update_signal.emit()

    def update_response(self, models):
        self.model_selector.clear()
        self.model_selector.addItems(models)
