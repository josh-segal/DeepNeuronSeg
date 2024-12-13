from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QFileDialog
from PyQt5.QtCore import QPoint, pyqtSignal
from DeepNeuronSeg.views.widgets.image_display import ImageDisplay
class LabelingView(QWidget):

    add_cell_marker_signal = pyqtSignal(QPoint)
    remove_cell_marker_signal = pyqtSignal(QPoint, int)
    upload_labels_signal = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.image_display = ImageDisplay()
        layout = QVBoxLayout()
        self.upload_label_btn = QPushButton("Upload Labels")
        self.load_btn = QPushButton("Display Data")
        self.next_btn = QPushButton("Next Image")

        self.upload_label_btn.clicked.connect(self.upload_labels)
        self.next_btn.clicked.connect(lambda: self.image_display.show_item(next_item=True, points=True))
        self.load_btn.clicked.connect(self.load_data)
    
        

        self.image_display.image_label.left_click_registered.connect(self.add_cell_marker)
        self.image_display.image_label.right_click_registered.connect(self.remove_cell_marker)
        
        layout.addWidget(self.upload_label_btn)
        layout.addWidget(self.image_display)
        layout.addWidget(self.next_btn)
        layout.addWidget(self.load_btn)

        self.setLayout(layout)

    def upload_labels(self):
        uploaded_labels, _ = QFileDialog.getOpenFileNames(self, "Select Labels", "", "Labels (*.png *.txt *.csv *.xml)")
        self.upload_labels_signal.emit(uploaded_labels)

    def load_data(self):
        self.image_display.show_item(points=True)
        
    def add_cell_marker(self, pos):
        self.add_cell_marker_signal.emit(pos)

    def remove_cell_marker(self, pos, tolerance=5):
        self.remove_cell_marker_signal.emit(pos, tolerance)