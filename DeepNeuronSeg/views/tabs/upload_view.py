import os
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QListWidget, QFileDialog, QLineEdit, QLabel, QGridLayout, QHBoxLayout
from PyQt5.QtCore import pyqtSignal


class UploadView(QWidget):

    upload_images_signal = pyqtSignal(list, str, str, str, str)
    update_signal = pyqtSignal()

    def __init__(self, image_display):
        super().__init__()
        layout = QVBoxLayout()

         # Image preview
        self.image_display = image_display
        
        # File list
        self.file_list = QListWidget()
        # File selection
        self.upload_btn = QPushButton("Upload Images")
        self.next_btn = QPushButton("Next Image")
        self.load_btn = QPushButton("Display Data")
        self.set_text_btn = QPushButton("Update Image Metadata")

        self.upload_btn.clicked.connect(self.upload_images)
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
        button_layout.addWidget(self.next_btn)
        button_layout.addWidget(self.load_btn)
        
        layout.addLayout(button_layout)
        layout.addLayout(metadata_layout)
        layout.addWidget(self.image_display)
        layout.addWidget(self.file_list)
        self.setLayout(layout)
    
    def upload_images(self):
        self.uploaded_files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.tif)")
        self.upload_images_signal.emit(self.uploaded_files, self.project.text(), self.cohort.text(), self.brain_region.text(), self.image_id.text())
        print("uploading images")

    def update_images(self, items):
        self.file_list.clear()
        self.file_list.addItems([os.path.basename(file) for file in items])
        self.image_display.show_item()

    def update(self):
        self.image_display.show_item()
        self.update_signal.emit()
    
    def update_response(self, items):
        self.file_list.clear()
        self.file_list.addItems([os.path.basename(file) for file in items])
        