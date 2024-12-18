from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFrame
)
from PyQt5.QtWidgets import QMessageBox


class HideableInputPanel(QWidget):
    def __init__(self, augmentations):
        super().__init__()
        self.augmentations = augmentations
        self.input_boxes = {}  # Keep references to the input widgets

        self.hyperparameters = {
                    "hsv_h": "image HSV-Hue augmentation (fraction)",
                    "hsv_s": "image HSV-Saturation augmentation (fraction)",
                    "hsv_v": "image HSV-Value augmentation (fraction)",
                    "degrees": "image rotation (+/- deg)",
                    "translate": "image translation (+/- fraction)",
                    "scale": "image scale (+/- gain)",
                    "shear": "image shear (+/- deg)",
                    "perspective": "image perspective (+/- fraction), range 0-0.001",
                    "flipud": "image flip up-down (probability)",
                    "fliplr": "image flip left-right (probability)",
                    "bgr": "image channel BGR (probability)",
                    "mosaic": "image mosaic (probability)",
                    "mixup": "image mixup (probability)",
                    "auto_augment": "auto augmentation policy for classification (randaugment, autoaugment, augmix)",
                    "erasing": "probability of random erasing during classification training (0-0.9), 0 means no erasing, must be less than 1.0",
                    "crop_fraction": "image crop fraction for classification (0.1-1), 1.0 means no crop, must be greater than 0."
                }   

        self.init_ui()
        

    def init_ui(self):
        # Main layout
        self.main_layout = QVBoxLayout(self)
        
        # Input Panel
        self.input_panel = QFrame()
        self.input_panel_layout = QVBoxLayout(self.input_panel)

        # Dynamically create input boxes for the initial set of augmentations
        for key, value in self.augmentations.items():
            self.add_input_row(key, value)

        self.main_layout.addWidget(self.input_panel)
        self.input_panel.setVisible(False)
        
        # Add stretch to push button to bottom
        self.main_layout.addStretch()
        
        # Toggle Button
        self.toggle_button = QPushButton("Advanced Augmentation Settings...")
        self.toggle_button.setCheckable(True)
        self.toggle_button.toggled.connect(self.toggle_inputs)
        
        # Add button directly to main layout with center alignment
        self.main_layout.addWidget(self.toggle_button)

    def add_input_row(self, key, value):
        """
        Add a single row with a label and input box for a given key and value.
        """
        row_layout = QHBoxLayout()

        label = QLabel(key)
        if key in self.hyperparameters:
            label.setToolTip(self.hyperparameters[key])
        row_layout.addWidget(label)

        input_box = QLineEdit(str(value))
        input_box.setPlaceholderText(f"Enter value for {key}")
        input_box.editingFinished.connect(self.create_update_handler(key))
        row_layout.addWidget(input_box)

        self.input_boxes[key] = input_box
        self.input_panel_layout.addLayout(row_layout)

    def toggle_inputs(self, checked):
        self.input_panel.setVisible(checked)
        self.toggle_button.setText("Close Advanced Augmentation Settings" if checked else "Advanced Augmentation Settings...")

    def update(self, augmentations):
        """
        Update the UI without recreating widgets.
        """
        self.augmentations = augmentations

        # Update existing widgets or add new ones
        for key, value in augmentations.items():
            if key in self.input_boxes:
                # Update existing input box
                self.input_boxes[key].setText(str(value))
            else:
                # Add a new row for the new key
                self.add_input_row(key, value)

        # Remove rows for keys that no longer exist
        keys_to_remove = set(self.input_boxes.keys()) - set(augmentations.keys())
        for key in keys_to_remove:
            self.remove_input_row(key)

    def remove_input_row(self, key):
        """
        Remove a row for a given key from the UI.
        """
        widget = self.input_boxes.pop(key)
        row_layout = widget.parentWidget().layout()
        while row_layout.count():
            item = row_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.input_panel_layout.removeItem(row_layout)

    def create_update_handler(self, key):
        def handler():
            try:
                value = float(self.input_boxes[key].text())
            except ValueError:
                value = self.input_boxes[key].text()
            self.augmentations[key] = value

        return handler
