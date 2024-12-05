from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt5.QtCore import QPoint
from tinydb import Query
from DeepNeuronSeg.views.widgets.image_display import ImageDisplay



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
            self.db.image_table.update({"labels": [label for label in image_data.get("labels", []) if not (abs(label[0] - adjusted_pos.x()) < tolerance and abs(label[1] - adjusted_pos.y()) < tolerance)]}, image_query.file_path == file_path)
            self.image_display.show_item(points=True)
            # self.image_display.show_image_with_points()