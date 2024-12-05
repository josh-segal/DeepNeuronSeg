from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QSpinBox, QCheckBox, QPushButton

class FrameSelectionDialog(QDialog):
    def __init__(self, max_frames):
        super().__init__()
        self.setWindowTitle("Select Frame")
        self.selected_frame = 0
        self.use_for_all = False

        layout = QVBoxLayout()
        label = QLabel(f"Select a frame (0 to {max_frames - 1}):")
        layout.addWidget(label)

        self.frame_selector = QSpinBox()
        self.frame_selector.setRange(0, max_frames - 1)
        layout.addWidget(self.frame_selector)

        self.checkbox = QCheckBox("Use this frame for all .tif files")
        layout.addWidget(self.checkbox)

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        layout.addWidget(ok_button)

        self.setLayout(layout)

    def accept(self):
        self.selected_frame = self.frame_selector.value()
        self.use_for_all = self.checkbox.isChecked()
        super().accept()