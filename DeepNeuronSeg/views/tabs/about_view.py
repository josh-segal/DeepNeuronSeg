from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel, QCheckBox
from PyQt5.QtCore import pyqtSignal

class AboutView(QWidget):

    blinded_changed = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        layout = QGridLayout()
        self.setLayout(layout)

        self.blind_label = QLabel("Blinded")
        self.blind_checkbox = QCheckBox()
        self.blind_checkbox.toggled.connect(self.blind_checkbox_toggled)
        layout.addWidget(self.blind_label, 0, 0)
        layout.addWidget(self.blind_checkbox, 0, 1)
        
        layout.setColumnStretch(2, 1)
        layout.setRowStretch(1, 1)

    def blind_checkbox_toggled(self):
        self.blinded_changed.emit(self.blind_checkbox.isChecked())

