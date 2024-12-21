from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel, QFileDialog, QDoubleSpinBox
from DeepNeuronSeg.views.widgets.image_display import ImageDisplay

class ModelZooView(QWidget):

    inference_images_signal = pyqtSignal(str, list, float)
    save_inferences_signal = pyqtSignal()
    update_index_signal = pyqtSignal(int)
    download_data_signal = pyqtSignal()
    update_signal = pyqtSignal()
    next_image_signal = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.uploaded_files = []
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
        

        self.conf_label = QLabel("Confidence Threshold:")
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.0, 1.0)
        self.conf.setSingleStep(0.05)
        self.conf.setValue(0.3)
        self.conf.setToolTip("""
                             Sets the minimum confidence threshold for detections. Detections with confidence below this threshold are discarded. Default is 0.3. Increasing this value will reduce the recall of the model and improve the precision, decreasing this value will increase the recall of the model and decrease the precision.
        """)
        # Run inference button
        self.inference_btn = QPushButton("Inference Images")
        
        # Save inferences button
        self.save_btn = QPushButton("Save Inferences")

        self.next_btn = QPushButton("Next Image")

        self.download_btn = QPushButton("Download Data")
        
        layout.addWidget(QLabel("Trained Model:"))
        layout.addWidget(self.model_selector)
        layout.addWidget(self.select_btn)
        self.conf_layout = QHBoxLayout()
        self.conf_layout.addWidget(self.conf_label)
        self.conf_layout.addWidget(self.conf)
        layout.addLayout(self.conf_layout)
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
        if self.uploaded_files:
            self.model_name = self.model_selector.currentText() 
            self.confidence = self.conf.value()     
            self.inference_images_signal.emit(self.model_name, self.uploaded_files, self.confidence)
        else:
            #QMessageBox.warning(self, "No Images", "No images selected.")
            print('No images selected.')
            something = 1

    def display_images(self, uploaded_file, inference_result, current_index, total_items):
        uploaded_file = (uploaded_file, 0)
        self.left_image.display_frame(uploaded_file, current_index, total_items)
        self.right_image.display_frame(inference_result, current_index, total_items)

    def save_inferences(self):
        self.save_inferences_signal.emit()
    
    def next_image(self):
        self.next_image_signal.emit()

    def update(self):
        self.update_signal.emit()

    def update_response(self, models):
        self.model_selector.clear()
        self.model_selector.addItems(models)
