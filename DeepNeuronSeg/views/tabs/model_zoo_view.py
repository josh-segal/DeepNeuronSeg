import os
import tempfile
from PIL import Image
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel, QFileDialog
from tinydb import Query
from DeepNeuronSeg.views.widgets.image_display import ImageDisplay

class ModelZooView(QWidget):

    inference_images_signal = pyqtSignal(str, list)
    save_inferences_signal = pyqtSignal()
    update_index_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        image_layout = QHBoxLayout()

        # Model selection
        self.model_selector = QComboBox()

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
        self.model_name = self.model_selector.currentText() 
        self.inference_images_signal.emit(self.model_name, self.uploaded_files)

    def display_images(self, uploaded_files, inference_results, current_index):
        if len(inference_results) > 0 and len(uploaded_files) > 0:
            
            self.left_image._display_image(uploaded_files[current_index], current_index + 1, len(uploaded_files))
            self._show_pred()

            current_index = (current_index + 1) % len(uploaded_files)

        self.update_index_signal.emit(current_index)
            

    def save_inferences(self):
        self.save_inferences_signal.emit()

    def _show_pred(self):
        result = self.inference_result[self.current_index]
        mask_image = result.plot(labels=False, conf=False, boxes=False)
        mask_image = Image.fromarray(mask_image)
        temp_image_path = os.path.join(tempfile.gettempdir(), "temp_mask_image.png")
        mask_image.save(temp_image_path, format='PNG')
        self.right_image._display_image(temp_image_path, self.current_index + 1, len(self.inference_result))

    def update(self):
        pass
