from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QDoubleSpinBox, QListWidget, QLabel
from PyQt5.QtCore import pyqtSignal
from DeepNeuronSeg.views.widgets.image_display import ImageDisplay
import os

class OutlierView(QWidget):

    update_signal = pyqtSignal()

    update_outlier_threshold_signal = pyqtSignal(float)
    next_image_signal = pyqtSignal()
    remove_outlier_signal = pyqtSignal()
    load_image_signal = pyqtSignal(int)
    relabel_outlier_signal = pyqtSignal()
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.outlier_files = []
        self.outlier_data = {}
        
        # Image display
        self.image_display = ImageDisplay()
        
        # Outlier controls
        controls_layout = QHBoxLayout()
        self.confirm_btn = QPushButton("Confirm")
        self.relabel_btn = QPushButton("Relabel")
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_image)    
        self.confirm_btn.clicked.connect(self.confirm_outlier)
        self.relabel_btn.clicked.connect(self.relabel_outlier)
        controls_layout.addWidget(self.confirm_btn)
        controls_layout.addWidget(self.relabel_btn)

        # Outlier list
        self.outlier_threshold = QDoubleSpinBox()
        self.outlier_threshold.setSingleStep(0.5)
        self.outlier_threshold.setValue(3)
        self.outlier_threshold.valueChanged.connect(self.update_outlier_threshold)

        self.outlier_list = QListWidget()
        self.outlier_list.itemClicked.connect(self.display_outlier_image_indexed)

        threshold_layout = QHBoxLayout()

        threshold_layout.addWidget(QLabel("Outlier Threshold:"))
        threshold_layout.addWidget(self.outlier_threshold)
        
        layout.addWidget(self.image_display)
        layout.addWidget(self.outlier_list)
        layout.addWidget(self.next_btn)
        layout.addLayout(controls_layout)
        layout.addLayout(threshold_layout)
        layout.addStretch()
        self.setLayout(layout)

    def next_image(self):
        self.next_image_signal.emit()

    def confirm_outlier(self):
        self.remove_outlier_signal.emit()
        self.update()

    def relabel_outlier(self):
        self.relabel_outlier_signal.emit()

    def display_outlier_image_indexed(self, index):
        self.load_image_signal.emit(index)

    def update_outlier_threshold(self, value):
        self.update_outlier_threshold_signal.emit(value)

    def update_outliers(self, data, blinded=False):
        self.outlier_list.clear()
        self.outlier_data = data
        self.update()
                
    def update(self):
        self.update_signal.emit()

    def update_response(self, images, blinded):
        if blinded:
            outlier_files = [path[1] for path in images]
        else:
            outlier_files = [path[0] for path in images]
        self.outlier_list.clear()
        self.outlier_list.addItems(outlier_files)