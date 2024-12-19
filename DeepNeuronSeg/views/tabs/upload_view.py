import os
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QListWidget, QFileDialog, QCheckBox
from PyQt5.QtCore import pyqtSignal
from DeepNeuronSeg.views.widgets.image_display import ImageDisplay

class UploadView(QWidget):

    upload_images_signal = pyqtSignal(list)
    update_signal = pyqtSignal()
    load_image_signal = pyqtSignal(int)
    next_image_signal = pyqtSignal()
    blinded_signal = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

         # Image preview
        self.image_display = ImageDisplay()
        
        # File list
        self.file_list = QListWidget()
        # File selection
        self.blinded_btn = QCheckBox("Blinded")
        self.blinded_btn.toggled.connect(self.blinded_signal.emit)
        self.upload_btn = QPushButton("Upload Images")

        self.next_btn = QPushButton("Next Image")

        self.upload_btn.clicked.connect(self.upload_images)
        self.next_btn.clicked.connect(self.next_image)
        
        self.file_list.itemClicked.connect(lambda item: self.load_image(index=self.file_list.row(item)))
        
        layout.addWidget(self.blinded_btn)
        layout.addWidget(self.upload_btn)
        layout.addWidget(self.image_display)
        layout.addWidget(self.file_list)
        layout.addWidget(self.next_btn)
        layout.addStretch()
        self.setLayout(layout)

    def next_image(self):
        self.next_image_signal.emit()

    def load_image(self, index):
        self.load_image_signal.emit(index)
    
    def upload_images(self):
        self.uploaded_files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.tif)")
        self.upload_images_signal.emit(self.uploaded_files)

    def update_images(self, items):
        self.file_list.clear()
        self.file_list.addItems([os.path.basename(file) for file in items])

    def update(self):
        self.update_signal.emit()
    
    def update_response(self, items):
        self.file_list.clear()
        self.file_list.addItems([os.path.basename(file) for file in items])
        