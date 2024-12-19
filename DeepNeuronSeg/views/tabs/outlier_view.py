from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QDoubleSpinBox, QListWidget, QLabel
from PyQt5.QtCore import pyqtSignal
from DeepNeuronSeg.views.widgets.image_display import ImageDisplay
import os

class OutlierView(QWidget):

    update_signal = pyqtSignal()

    update_outlier_threshold_signal = pyqtSignal(float)
    next_image_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.outlier_files = []
        
        # Image display
        self.image_display = ImageDisplay()
        
        # Outlier controls
        controls_layout = QHBoxLayout()
        self.confirm_btn = QPushButton("Confirm")
        self.relabel_btn = QPushButton("Relabel")
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_image)    
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

    def update_outlier_threshold(self, value):
        self.update_outlier_threshold_signal.emit(value)

    def display_outlier_image_indexed(self, item):
        image_path = self.outlier_files[self.outlier_list.row(item)]
        self.image_display.display_frame(image_path, self.outlier_list.row(item), self.outlier_list.count())
        # if relabel button clicked, add to db, calculate pseudo labels from masks and display labels for refining 
        # should remove prediction from pred table ? do I need pred table ?    

    def update_outliers(self, data):
        self.outlier_files = list(data.keys())
        basename_data = [f"{os.path.splitext(os.path.basename(path))[0]} (Score: {data[path]:.2f})" for path in self.outlier_files]
        self.outlier_list.addItems(basename_data)
                
    def update(self):
        self.update_signal.emit()

    def update_response(self):
        if self.outlier_files:
            self.image_display.display_frame(self.outlier_files[0], 1, len(self.outlier_files))
        else:
            self.image_display.image_label.setText("No outliers found")

    def display_outlier_image(self, item, index, total, points):
        self.image_display.display_frame(item, index, total, points)
