from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtCore import pyqtSignal
from tqdm import tqdm
from tinydb import Query
from DeepNeuronSeg.views.widgets.image_display import ImageDisplay
from DeepNeuronSeg.utils.utils import save_label
from DeepNeuronSeg.models.segmentation_model import segment, composite_mask



class GenerateLabelsView(QWidget):

    generate_labels_signal = pyqtSignal()

    def __init__(self, image_display_left, image_display_right):
        super().__init__()
        layout = QVBoxLayout()
        image_layout = QHBoxLayout()
        config_layout = QHBoxLayout()

        self.left_image = image_display_left
        self.right_image = image_display_right

        self.generate_btn = QPushButton("Generate Labels")
        self.next_btn = QPushButton("Next Image")
        self.display_btn = QPushButton("Display Labels")
        self.generate_btn.clicked.connect(self.generate_labels)
        self.next_btn.clicked.connect(lambda: self.left_image.show_item(next_item=True))
        self.next_btn.clicked.connect(lambda: self.right_image.show_item(mask=True))
        self.display_btn.clicked.connect(self.display_labels)

        
        config_layout.addWidget(self.generate_btn)
        config_layout.addWidget(self.next_btn)
        config_layout.addWidget(self.display_btn)

        
        image_layout.addWidget(self.left_image)
        image_layout.addWidget(self.right_image)

        layout.addLayout(image_layout)
        layout.addLayout(config_layout)

        self.setLayout(layout)

    def generate_labels(self):
        self.generate_labels_signal.emit()

    def display_labels(self):
        self.left_image.show_item()
        self.right_image.show_item(mask=True)

    def update(self):
        pass
