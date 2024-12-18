import os
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QDoubleSpinBox, QLineEdit, QCheckBox, QLabel, QListWidget, QPushButton
from PyQt5.QtCore import pyqtSignal

class DatasetView(QWidget):

    update_signal = pyqtSignal()
    create_dataset_signal = pyqtSignal(list, str, float)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        
        # Dataset configuration
        config_layout = QGridLayout()
        self.train_split = QDoubleSpinBox()
        self.train_split.setRange(0.0, 1.0)
        self.train_split.setSingleStep(0.05)
        self.train_split.setValue(0.8)
        self.dataset_name = QLineEdit()
        
        config_layout.addWidget(QLabel("Train Split:"), 0, 0)
        config_layout.addWidget(self.train_split, 0, 1)
        config_layout.addWidget(QLabel("Dataset Name:"), 3, 0)
        config_layout.addWidget(self.dataset_name, 3, 1)
        
        # Image selection
        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QListWidget.MultiSelection)
        
        # Creation button
        self.create_btn = QPushButton("Create Dataset")
        self.create_btn.clicked.connect(self.create_dataset)
        
        layout.addLayout(config_layout)
        layout.addWidget(self.image_list)
        layout.addWidget(self.create_btn)
        layout.addStretch()
        self.setLayout(layout)
    
    def create_dataset(self):
        self.create_dataset_signal.emit(self.image_list.selectedItems(), self.dataset_name.text().strip(), self.train_split.value())
        
    def update_image_list(self, items):
        self.image_list.addItems([os.path.basename(file) for file in items])
    
    def update(self):
        self.update_signal.emit()
    
    def update_response(self, items):
        self.image_list.clear()
        self.image_list.addItems([os.path.basename(file) for file in items])