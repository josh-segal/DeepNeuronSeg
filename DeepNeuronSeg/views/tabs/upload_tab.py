import os
import shutil
from PIL import Image
from tinydb import Query

from DeepNeuronSeg.views.widgets.image_display import ImageDisplay
from DeepNeuronSeg.views.widgets.frame_selection_dialog import FrameSelectionDialog
from DeepNeuronSeg.utils.label_parsers import parse_png_label, parse_txt_label, parse_csv_label, parse_xml_label
from DeepNeuronSeg.utils.utils import trim_underscores

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QListWidget, QLineEdit, QLabel, QGridLayout, QHBoxLayout, QFileDialog, QDialog
)

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
        items = self.db.load_images()
        self.file_list.addItems([os.path.basename(file) for file in items])
        # File selection
        self.upload_btn = QPushButton("Upload Images")
        self.upload_label_btn = QPushButton("Upload Labels")
        self.next_btn = QPushButton("Next Image")
        self.load_btn = QPushButton("Display Data")
        self.set_text_btn = QPushButton("Update Image Metadata")

        self.upload_btn.clicked.connect(self.upload_images)
        self.upload_label_btn.clicked.connect(self.upload_labels)
        self.load_btn.clicked.connect(self.image_display.show_item)
        self.next_btn.clicked.connect(lambda: self.image_display.show_item(next_item=True))
        self.set_text_btn.clicked.connect(self.update_image_metadata)

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
                            # print("Converting tif to png", image_path)
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
        items = self.db.load_images()
        self.file_list.clear()
        self.file_list.addItems([os.path.basename(file) for file in items])
        self.image_display.show_item()

    def upload_labels(self):
        self.uploaded_labels, _ = QFileDialog.getOpenFileNames(self, "Select Labels", "", "Labels (*.png *.txt *.csv *.xml)")
        self.parse_labels(self.uploaded_labels)

    def parse_labels(self, labels):

        for label_file in labels:
            label_name = os.path.splitext(os.path.basename(label_file))[0]
            label_name = trim_underscores(label_name)
            label_name = label_name + ".png"
            label_name = os.path.join('data', 'data_images', label_name)

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

    def update_image_metadata(self):
        image_query = Query()
        image_data = self.db.image_table.get(image_query.file_path == file_path)

    def update(self):
        self.file_list.clear()
        items = self.db.load_images()
        self.file_list.addItems([os.path.basename(file) for file in items])