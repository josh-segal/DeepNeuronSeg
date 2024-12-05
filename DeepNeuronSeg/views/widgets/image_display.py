import os
import tempfile
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image
from DeepNeuronSeg.views.widgets.image_label import ImageLabel

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

        self.image_label.setMinimumSize(512, 512)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.text_label.setAlignment(Qt.AlignBottom | Qt.AlignCenter)

    def _display_image(self, image_path, image_num, total_images):
        """Load and display an image from the given file path and show image number."""
        # print(image_path, '<----------------------')
        self.pixmap = QPixmap(image_path)
        if not self.pixmap.isNull():
            self.image_label.set_pixmap(self.pixmap)
            self.text_label.setText(f"{image_num} / {total_images}")
        else:
            print("Failed to load image")

    def tif_check(self, item):
        if item.lower().endswith('.tif'):
            with Image.open(item) as img:
                img.seek(self.upload_tab.selected_frame)
                frame_to_display = img.copy()
                base_name, _ = os.path.splitext(item)
                temp_image_path = os.path.join(tempfile.gettempdir(), base_name + ".png")
                frame_to_display.save(temp_image_path, format='PNG')
                item = temp_image_path
        return item

    def show_item(self, mask=False, points=False, next_item=False, index=None):
        if mask:
            items = self.upload_tab.db.load_masks()
        else:
            items = self.upload_tab.db.load_images()

        if next_item:
            self.upload_tab.current_index = (self.upload_tab.current_index + 1) % len(items)

        if index is not None:
            self.upload_tab.current_index = index
            # print(self.upload_tab.current_index)

        if items:
            if self.upload_tab.current_index < len(items) and len(items) > 0:
                item = self.tif_check(items[self.upload_tab.current_index])
                self._display_image(item, self.upload_tab.current_index + 1, len(items))
                if points:
                    labels = self.upload_tab.db.load_labels()
                    self.image_label._draw_points(labels[self.upload_tab.current_index])
            else:
                print("No images uploaded")
                self.image_label.clear()
                self.text_label.setText("No mask generated" if mask else "No images uploaded")
        else:
            print("No images uploaded")
            self.image_label.clear()
            self.text_label.setText("No mask generated" if mask else "No images uploaded")