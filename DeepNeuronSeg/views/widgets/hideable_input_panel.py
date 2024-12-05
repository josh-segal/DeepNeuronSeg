from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFrame
)
from PyQt5.QtCore import Qt

class HideableInputPanel(QWidget):
    def __init__(self, augmentations):
        super().__init__()
        self.augmentations = augmentations
        self.init_ui()

    def init_ui(self):
        # Main layout
        self.main_layout = QVBoxLayout(self)
        
        # Toggle Button
        self.toggle_button = QPushButton("Advanced Settings")
        self.toggle_button.setCheckable(True)
        self.toggle_button.toggled.connect(self.toggle_inputs)
        self.main_layout.addWidget(self.toggle_button)

        # Input Panel
        self.input_panel = QFrame()
        self.input_panel_layout = QVBoxLayout(self.input_panel)

        # Dynamically create input boxes
        self.input_boxes = {}
        for key, value in self.augmentations.items():
            row_layout = QHBoxLayout()
            
            label = QLabel(key)
            row_layout.addWidget(label)
            
            input_box = QLineEdit(str(value))
            input_box.setPlaceholderText(f"Enter value for {key}")
            row_layout.addWidget(input_box)
            
            self.input_boxes[key] = input_box
            self.input_panel_layout.addLayout(row_layout)

        self.main_layout.addWidget(self.input_panel)
        self.input_panel.setVisible(False)

    def toggle_inputs(self, checked):
        self.input_panel.setVisible(checked)
        # self.toggle_button.setText("Hide Inputs" if checked else "Show Inputs")