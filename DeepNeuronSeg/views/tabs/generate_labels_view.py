from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget
from PyQt5.QtCore import pyqtSignal
from DeepNeuronSeg.views.widgets.image_display import ImageDisplay
import os


class GenerateLabelsView(QWidget):

    generate_labels_signal = pyqtSignal()
    next_image_signal = pyqtSignal()
    load_image_signal = pyqtSignal(int)
    update_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        image_layout = QHBoxLayout()

        self.left_image = ImageDisplay()
        self.right_image = ImageDisplay()

        self.file_list = QListWidget()

        self.generate_btn = QPushButton("Generate Labels")
        self.next_btn = QPushButton("Next Image")
        self.generate_btn.clicked.connect(self.generate_labels)
        self.next_btn.clicked.connect(self.next_image)
        self.file_list.itemClicked.connect(lambda item: self.load_image(index=self.file_list.row(item)))

        image_layout.addWidget(self.left_image)
        image_layout.addWidget(self.right_image)

        layout.addWidget(self.generate_btn)
        layout.addLayout(image_layout)
        layout.addWidget(self.file_list)
        layout.addWidget(self.next_btn)
        layout.addStretch()
        self.setLayout(layout)

    def generate_labels(self):
        self.generate_labels_signal.emit()

    def next_image(self):
        self.next_image_signal.emit()

    def load_image(self, index):
        self.load_image_signal.emit(index)

    def update(self):
        self.update_signal.emit()

    def update_response(self, items):
        self.file_list.clear()
        if items:
            if isinstance(items[0], str) and os.path.isfile(items[0]):
                self.file_list.addItems([os.path.basename(file) for file in items])
            else:
                self.file_list.addItems([str(item) for item in items])