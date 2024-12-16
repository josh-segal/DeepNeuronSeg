from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import os
import tempfile
from PIL import Image
from DeepNeuronSeg.views.widgets.image_label import ImageLabel

class ImageDisplay(QWidget):
    """Widget for displaying and interacting with images"""
    
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.image_label = ImageLabel()
        self.text_label = QLabel()

        self.image_label.setMinimumSize(512, 512)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.text_label.setAlignment(Qt.AlignCenter)    

        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.text_label)

    def clear(self):
        """Clear the image display and reset labels"""
        self.image_label.clear()
        self.text_label.setText("")

    def _display_image(self, image_path, image_num, total_images):
        """Load and display an image from the given file path and show image number."""
        self.pixmap = QPixmap(image_path)
        if not self.pixmap.isNull():
            self.image_label.set_pixmap(self.pixmap)
            self.text_label.setText(f"{image_num} / {total_images}")
        else:
            print("Failed to load image")

    def display_frame(self, image_path, frame_number, total_frames, points=None):
        """Display a specific frame from an image file with optional points"""
        if image_path.lower().endswith('.tif'):
            image_path = self._convert_tif_frame(image_path, frame_number)
            
        self._display_image(image_path, frame_number + 1, total_frames)
        
        if points:
            self.image_label._draw_points(points)

    def _convert_tif_frame(self, tif_path, frame_number):
        """Convert a specific frame of a TIF file to PNG and return the temporary path"""
        with Image.open(tif_path) as img:
            img.seek(frame_number)
            frame_to_display = img.copy()
            base_name, _ = os.path.splitext(tif_path)
            temp_image_path = os.path.join(tempfile.gettempdir(), base_name + ".png")
            frame_to_display.save(temp_image_path, format='PNG')
            return temp_image_path